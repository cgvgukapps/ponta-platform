#!/usr/bin/env python3
"""
pass_option_value.py — PCZ-enhanced Pass Option Value (POV) model.

For each potential receiver at a given frame, computes a POV score:

    POV(receiver) =
        xT_receiver                 ← where the receiver ends up
        × V_ctrl_receiver           ← does the receiver's team control that hex?
        × (1 − interception_risk)   ← can defenders intercept the ball in flight?
        × progressive_factor        ← is the pass going forward?
        × (1 − distance_decay)      ← pass length penalty

Both V_ctrl_receiver and interception_risk use **targeted PCZ** — kinematic
travel times computed only for corridor waypoints (~15–20 hexes) and the
receiver's hex, not the full ~7,000-hex grid.  This gives full kinematic
accuracy at ~350x less computation than full PCZ per frame.

ΔPOV (per event)
----------------
    ΔPOV(player) = POV(end_frame) − POV(start_frame)
    Positive → the player's run improved the passing option they represent.

Decision Quality (per event)
-----------------------------
Scores the carrier's action relative to the best option available at the
moment of decision (event start). Two branches:

    PASS  → POV(actual_receiver, end_frame) / max(POV teammates, start_frame)
            End-frame receiver POV rewards timing the ball into a run.

    CARRY → V(actual_end, end_frame)
            ─────────────────────────────────────────────────────────
            max over h ∈ ring_alternatives:
                  V(h, start_frame)

            V(h, f) = [W_POS · xT(h) + W_OPT · max_POV(if ball were at h, f)]
                      · (1 − press_discount(h, f))
            (W_POS = W_OPT = 0.5 by default).

            ring_alternatives = hexes on the same ring distance from the
            start hex as the chosen end hex (±1 ring tolerance, ring 0
            excluded), filtered to those the carrier could kinematically
            reach in the carry duration.  This is a pure direction-quality
            comparison at the chosen effort: "of the directions you could
            have gone this distance, was this the best one?"
            Ring-0 carries (chosen end == start hex) are scored as N/A.

    FAILED PASS (BALL LOST events with a ball trajectory)
          → POV(inferred_receiver_intent, end_frame) / max(POV, start_frame)
            The intended receiver is inferred by walking the ball line past
            the interception point and finding the first teammate whose
            kinematic PCZ would have controlled a hex along the extension.
            The intent location is then refined by intersecting the ball
            ray with the receiver's projected velocity ray (their actual
            meeting point), falling back to the receiver's xy at end_frame
            when projection isn't possible.  See §7.3 of the spec.

    ≥ 0.95 optimal · 0.80–0.95 good · 0.55–0.80 suboptimal · < 0.55 poor.
    Bands are unified across all event types — see §7.1 of the analytics spec
    for the rationale and the audit numbers behind the 0.55 / 0.80 / 0.95 cutoffs.
"""

import numpy as np
import pandas as pd
from matrix_engine.pitch_config import DEFAULT_FPS, DEFAULT_PITCH
from scipy.spatial import cKDTree

from plaiq_engine.analytics.event_stitch import (
    ball_xy_from_tracking,
    build_pass_receipt_lookup,
)
from plaiq_engine.analytics.hex_analytics import cartesian_to_hex
from plaiq_engine.analytics.pitch_control import (
    BALL_SPEED_GRAD,
    MIN_BALL_SPEED,
    player_travel_time,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CORRIDOR_WIDTH    = 1.5    # metres — half-width of passing lane
_CORRIDOR_SPACING  = 2.0    # metres — waypoint spacing along pass line
_MAX_PASS_DISTANCE = 60.0   # metres
_DISTANCE_DECAY_D  = 40.0   # metres — distance at which decay = 0.5
_PROG_FLOOR        = 0.5    # min progressive_factor (back-passes keep some value)
_CTRL_SIGMOID_K    = 2.0    # sigmoid steepness for V_ctrl_receiver

# ---- Carry decision-quality (counterfactual alternatives) ----
_CARRY_W_POS         = 0.50   # positional weight  (composite V at a hex)
_CARRY_W_OPT         = 0.50   # option-creation weight
_CARRY_RING_TOL      = 1      # ± rings around the chosen ring distance (defines the annulus)
_CARRY_INTERCEPT_R   = 2.0    # m — defender perp-dist that counts as "in the lane"
_CARRY_V_BEST_FLOOR  = 0.02   # denominator floor so dead-zone carries don't blow up
_CARRY_PRESS_MAX_DSC = 0.30   # max pressure discount on V (matches §4 PRESSURE_MAX_DISCOUNT)
_CARRY_BEST_TIE_TOL  = 0.02   # any alt within 2% of V_best is also flagged "best" (tie handling)

# ---- Failed-pass decision-quality (intercepted / lost-trajectory passes) ----
_FAILED_MIN_DIST          = 5.0    # m — trajectory length below which we skip
_FAILED_PROJECTION_MAX    = 15.0   # m — max meeting-point extension
_PCZ_WALK_STEP_M          = 1.0    # m — step size when walking the ball extension
_PCZ_WALK_MAX_EXTEND      = 40.0   # m — stop walking past this much extension
# Lateral offsets sampled at each forward step — turns the walk into a
# thin corridor rather than a 1D line.  Catches teammate PCZs whose
# controlled hex sits near (but not exactly on) the ball trajectory line.
_PCZ_CORRIDOR_OFFSETS     = (-2.0, -1.0, 0.0, 1.0, 2.0)   # m, perpendicular to walk
_RECEIVER_MIN_DISPLACEMENT = 0.5   # m — minimum start→end displacement magnitude
                                    #     required to project the receiver's intent
                                    #     line.  Below this the receiver effectively
                                    #     didn't move during the pass and we fall
                                    #     back to their interception-time xy.
_FAILED_FIELD_HALF_X      = DEFAULT_PITCH.half_length   # half pitch length (m) — boundary clamp guard
_FAILED_FIELD_HALF_Y      = DEFAULT_PITCH.half_width    # half pitch width (m)  — boundary clamp guard

# BALL LOST subtypes that should NOT be treated as failed passes for DQ.
# THEFT lacks a trajectory; CLEARANCE/HAND BALL/OFFSIDE/GOAL KICK/REFEREE
# /END HALF aren't really "decisions to pass" — they're defensive actions or
# rule events.  See §7.3 of the analytics spec for the rationale.
_FAILED_EXCLUDED_SUBTYPES = (
    "CLEARANCE", "OFFSIDE", "HAND BALL", "END HALF",
    "GOAL KICK", "REFEREE", "THEFT",
)


# ---------------------------------------------------------------------------
# Targeted ball travel time (for a single destination)
# ---------------------------------------------------------------------------

def _ball_time(ball_xy: np.ndarray, target_xy: np.ndarray) -> float:
    """Time for ball to reach target using the same speed model as pitch_control."""
    d = float(np.linalg.norm(target_xy - ball_xy))
    speed = MIN_BALL_SPEED + BALL_SPEED_GRAD * d
    return d / max(speed, 1e-6)


# ---------------------------------------------------------------------------
# Corridor waypoints
# ---------------------------------------------------------------------------

def _corridor_waypoints(
    ball_xy: np.ndarray,
    receiver_xy: np.ndarray,
    spacing: float = _CORRIDOR_SPACING,
) -> np.ndarray:
    """
    Sample points along the pass trajectory, starting 1m from ball,
    ending 1m before receiver.  Returns (M, 2) array or empty if too close.
    """
    d = np.linalg.norm(receiver_xy - ball_xy)
    if d < 2.0:
        return np.empty((0, 2))
    direction = (receiver_xy - ball_xy) / d
    n_pts = max(1, int((d - 1.0) / spacing))
    ts = np.linspace(1.0, d - 1.0, n_pts)
    return ball_xy + np.outer(ts, direction)   # (M, 2)


# ---------------------------------------------------------------------------
# Targeted PCZ — corridor interception risk + receiver control
# ---------------------------------------------------------------------------

def targeted_pcz(
    ball_xy: np.ndarray,
    receiver_xy: np.ndarray,
    defender_data: list,
) -> tuple:
    """
    Compute interception_risk and V_ctrl_receiver using kinematic travel times
    computed only for corridor waypoints + the receiver's hex.

    Parameters
    ----------
    ball_xy, receiver_xy : (2,) positions
    defender_data : list of dicts, each containing:
        {pos: (2,), vel: (2,), v_max, a_max, reaction_time}

    Returns
    -------
    interception_risk : float in [0, 1]
        Fraction of corridor waypoints where at least one defender can
        arrive before the ball.
    V_ctrl_receiver : float in [0, 1]
        Smooth sigmoid control weight at receiver's hex.
        > 0.5 → receiver's team likely controls that space.
    """
    if not defender_data:
        return 0.0, 0.5   # no defenders → open corridor, contested receiver

    waypoints = _corridor_waypoints(ball_xy, receiver_xy)  # (M, 2)

    # Compute ball travel times to waypoints and receiver
    if len(waypoints) > 0:
        t_ball_wpts = np.array([_ball_time(ball_xy, wp) for wp in waypoints])
    else:
        t_ball_wpts = np.array([])

    t_ball_recv = _ball_time(ball_xy, receiver_xy)

    # Compute defender travel times to waypoints and receiver
    n_wpts = len(waypoints)
    t_def_wpts = []   # (D, M) — defender × waypoint
    t_def_recv = []   # (D,) — defender to receiver hex

    for dd in defender_data:
        pos  = np.asarray(dd["pos"], dtype=float)
        vel  = np.asarray(dd.get("vel", [0.0, 0.0]), dtype=float)
        vmax = float(dd.get("v_max", 9.0))
        amax = float(dd.get("a_max", 5.5))
        rt   = float(dd.get("reaction_time", 0.7))

        if n_wpts > 0:
            t_w = player_travel_time(pos, vel, waypoints, vmax, amax, rt)
            t_def_wpts.append(t_w)

        t_r = player_travel_time(
            pos, vel, receiver_xy[None, :], vmax, amax, rt
        )[0]
        t_def_recv.append(t_r)

    # ---- Interception risk ----
    if n_wpts > 0 and t_def_wpts:
        t_def_wpts_arr = np.array(t_def_wpts)    # (D, M)
        t_min_def_wpts = t_def_wpts_arr.min(axis=0)  # (M,) — fastest defender per waypoint
        interceptable  = (t_min_def_wpts <= t_ball_wpts)
        interception_risk = float(interceptable.mean())
    else:
        interception_risk = 0.0

    # ---- V_ctrl_receiver ----
    if t_def_recv:
        t_fastest_def = min(t_def_recv)
        # Time gap: positive → receiver's team arrives first (ball travels faster)
        time_gap   = t_fastest_def - t_ball_recv
        sig_input  = float(np.clip(time_gap * _CTRL_SIGMOID_K, -10.0, 10.0))
        V_ctrl_receiver = float(1.0 / (1.0 + np.exp(-sig_input)))
    else:
        V_ctrl_receiver = 0.5

    return interception_risk, V_ctrl_receiver


# ---------------------------------------------------------------------------
# Progressive factor & distance decay
# ---------------------------------------------------------------------------

def _progressive_factor(ball_xT: float, receiver_xT: float) -> float:
    raw = _PROG_FLOOR + (receiver_xT - ball_xT) / 0.3
    return float(np.clip(raw, _PROG_FLOOR, 1.5))


def _distance_decay(ball_xy: np.ndarray, receiver_xy: np.ndarray) -> float:
    d = float(np.linalg.norm(receiver_xy - ball_xy))
    return float(np.clip(d / _DISTANCE_DECAY_D, 0.0, 0.5))


# ---------------------------------------------------------------------------
# Single-player POV
# ---------------------------------------------------------------------------

def player_pov(
    ball_xy: np.ndarray,
    receiver_xy: np.ndarray,
    xT_ball: float,
    xT_receiver: float,
    defender_data: list,
) -> float:
    """
    Compute Pass Option Value for one receiver using targeted PCZ.

    POV = xT_receiver × V_ctrl_receiver × (1 − interception_risk)
          × progressive_factor × (1 − distance_decay)
    """
    if np.linalg.norm(receiver_xy - ball_xy) > _MAX_PASS_DISTANCE:
        return 0.0

    interception_risk, V_ctrl_receiver = targeted_pcz(
        ball_xy, receiver_xy, defender_data
    )

    prog  = _progressive_factor(xT_ball, xT_receiver)
    decay = _distance_decay(ball_xy, receiver_xy)

    pov = (
        xT_receiver
        * V_ctrl_receiver
        * (1.0 - interception_risk)
        * prog
        * (1.0 - decay)
    )
    return float(np.clip(pov, 0.0, 2.0))


# ---------------------------------------------------------------------------
# Extract defender data from tracking rows
# ---------------------------------------------------------------------------

def _build_defender_data(
    tracking_row: pd.Series,
    team_prefix: str,
    jerseys: list,
    profiles: dict,
) -> list:
    """
    Build a list of defender dicts from a tracking row.
    Each dict: {pos, vel, v_max, a_max, reaction_time}
    """
    result = []
    for jersey in jerseys:
        px = tracking_row.get(f"{team_prefix}_{jersey}_x")
        py = tracking_row.get(f"{team_prefix}_{jersey}_y")
        if pd.isna(px) or pd.isna(py):
            continue
        vx = tracking_row.get(f"{team_prefix}_{jersey}_vx", 0.0)
        vy = tracking_row.get(f"{team_prefix}_{jersey}_vy", 0.0)
        vx = 0.0 if pd.isna(vx) else float(vx)
        vy = 0.0 if pd.isna(vy) else float(vy)
        p  = profiles.get(jersey, {})
        result.append({
            "pos":           np.array([float(px), float(py)]),
            "vel":           np.array([vx, vy]),
            "v_max":         p.get("v_max", 9.0),
            "a_max":         p.get("a_max", 5.5),
            "reaction_time": p.get("reaction_time", 0.7),
        })
    return result


# ---------------------------------------------------------------------------
# Carry decision-quality — counterfactual alternative destinations
# ---------------------------------------------------------------------------
#
# A carry is scored by comparing what the carrier actually achieved against the
# best alternative destination they could have reached in the same time window.
# This is the spatial analogue of pass-DQ (which compares the chosen receiver
# against the best teammate option).
#
# Each alternative hex h is scored with the same composite used for the actual
# end position:
#
#     V(h, frame) = W_POS · xT(h)  +  W_OPT · max_POV(if ball were at h, frame)
#
# Decision quality:
#
#     carry_dq = V(p_end, end_frame) / max( V(h, start_frame) for h in alts )
#
# Numerator at end_frame rewards carries that time-in to teammate runs;
# denominator at start_frame is the fixed counterfactual benchmark at the
# moment of decision.  Alternatives are restricted to hexes the carrier could
# have kinematically reached in the carry's duration, and always include the
# start position itself ("do nothing") so a carry that worsened the situation
# scores below the no-action baseline.
# ---------------------------------------------------------------------------


def _ring_alternatives(
    carrier_pos:    np.ndarray,
    carrier_vel:    np.ndarray,
    duration:       float,
    hex_centers:    np.ndarray,
    q_all:          np.ndarray,    # axial q for every hex centre
    r_all:          np.ndarray,    # axial r for every hex centre
    start_q:        int,
    start_r:        int,
    chosen_hex_idx: int,
    chosen_ring:    int,
    v_max:          float,
    a_max:          float,
    ring_tolerance: int = _CARRY_RING_TOL,
) -> np.ndarray:
    """
    Indices into ``hex_centers`` for the carry's counterfactual destinations:
    hexes whose **axial ring distance from the start hex** lies in
    ``[chosen_ring − tolerance, chosen_ring + tolerance]`` (and ≥ 1, so the
    do-nothing start hex itself is always excluded), filtered to those the
    carrier could kinematically reach within ``duration`` seconds.

    The actual chosen end hex is force-included even if filtered out, so the
    score always has the ground truth in the comparison set.

    Returns an empty array for ``chosen_ring == 0`` (degenerate label
    carries that didn't move) — the caller should treat such events as N/A.
    """
    if chosen_ring < 1:
        return np.array([], dtype=int)

    ring_min = max(1, chosen_ring - ring_tolerance)
    ring_max = chosen_ring + ring_tolerance

    dq = q_all - start_q
    dr = r_all - start_r
    ring_dist = (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) // 2

    ring_idx = np.where((ring_dist >= ring_min) & (ring_dist <= ring_max))[0]

    if ring_idx.size == 0:
        return np.array([int(chosen_hex_idx)], dtype=int)

    # Kinematic reachability (reaction_time=0 — the carrier is already moving)
    t = player_travel_time(
        carrier_pos, carrier_vel, hex_centers[ring_idx],
        v_max, a_max, reaction_time=0.0,
    )
    reachable = ring_idx[t <= max(duration, 0.05)]

    # Always include the actual chosen end hex (ground truth)
    if int(chosen_hex_idx) not in reachable:
        reachable = np.append(reachable, int(chosen_hex_idx))

    return reachable


def _carrier_pressure_discount(
    h_xys:        np.ndarray,         # (A, 2) hex positions
    defender_pos: np.ndarray,         # (D, 2) defender positions at the relevant frame
    max_discount: float = _CARRY_PRESS_MAX_DSC,
) -> np.ndarray:
    """
    Per-hex pressure discount on the carrier's *own* control of a destination.
    Mirrors the existing PRESSURE_MAX_DISCOUNT convention from §4 of the spec:

        pressure(h) = clip(11 - d_nearest_def(h), 1, 10)   on the 1–10 scale
        discount(h) = (pressure / 10) · max_discount

    Lets carries that end with the carrier surrounded by defenders score lower
    than equally-valued carries into open space.
    """
    A = h_xys.shape[0]
    if defender_pos.shape[0] == 0:
        return np.zeros(A, dtype=np.float64)
    d = np.linalg.norm(
        h_xys[:, None, :] - defender_pos[None, :, :], axis=2,
    ).min(axis=1)
    pressure = np.clip(11.0 - d, 1.0, 10.0)
    return (pressure / 10.0) * max_discount


def _approx_max_pov_per_alt(
    alt_xys:           np.ndarray,        # (A, 2)
    alt_xTs:           np.ndarray,        # (A,)
    teammate_pos:      np.ndarray,        # (T, 2)
    teammate_xTs:      np.ndarray,        # (T,)
    defender_pos:      np.ndarray,        # (D, 2)
    carrier_jersey_idx: int = -1,         # row in teammate_pos to mask (the carrier)
) -> np.ndarray:
    """
    Vectorised, approximate ``max(POV)`` over teammates evaluated for each
    alternative ball position.

    Approximations vs full ``player_pov``:
      * V_ctrl_receiver uses a constant-velocity proxy for defender travel time
        (REACTION + d_def_min / v_max) rather than the full kinematic model.
        Defender velocities are ignored — fine for short carry windows.
      * interception_risk is approximated as the fraction of defenders that
        sit inside the ball→receiver corridor (within ``_CARRY_INTERCEPT_R``
        perpendicular metres AND between ball and receiver along the line).
      * Same progressive_factor and distance_decay as the exact model.

    Returns
    -------
    max_pov : (A,) — best POV any teammate offers if the ball were at alt h.
    """
    A = alt_xys.shape[0]
    T = teammate_pos.shape[0]
    D = defender_pos.shape[0]

    if T == 0:
        return np.zeros(A, dtype=np.float64)

    # Build a teammate mask (drop the carrier — they can't pass to themselves)
    tm_mask = np.ones(T, dtype=bool)
    if 0 <= carrier_jersey_idx < T:
        tm_mask[carrier_jersey_idx] = False
    if not tm_mask.any():
        return np.zeros(A, dtype=np.float64)

    tms_xy  = teammate_pos[tm_mask]               # (T', 2)
    tms_xT  = teammate_xTs[tm_mask]               # (T',)
    Tp = tms_xy.shape[0]

    # ---- distances ball(alt) → teammate ----
    diff = tms_xy[None, :, :] - alt_xys[:, None, :]   # (A, T', 2)
    d_br = np.linalg.norm(diff, axis=2)               # (A, T')
    d_br_safe = np.maximum(d_br, 1e-6)

    # ---- distance from each teammate to nearest defender (T',) ----
    if D > 0:
        d_def_tm = np.linalg.norm(
            defender_pos[None, :, :] - tms_xy[:, None, :], axis=2
        ).min(axis=1)                                  # (T',)
    else:
        d_def_tm = np.full(Tp, 1e3)

    # ---- ball travel time to each teammate (A, T') ----
    ball_speed = MIN_BALL_SPEED + BALL_SPEED_GRAD * d_br
    t_ball = d_br / np.maximum(ball_speed, 1e-6)

    # ---- defender arrival proxy (T',) ----
    # Constant-velocity, ignore current vel, use 0.7s reaction baseline
    DEF_VMAX = 8.5
    DEF_RT   = 0.7
    t_def = DEF_RT + d_def_tm / DEF_VMAX               # (T',)

    # ---- V_ctrl_receiver via sigmoid on time gap ----
    time_gap = t_def[None, :] - t_ball                 # (A, T')
    V_ctrl = 1.0 / (1.0 + np.exp(-np.clip(_CTRL_SIGMOID_K * time_gap, -10.0, 10.0)))

    # ---- progressive factor (A, T') ----
    prog = np.clip(
        _PROG_FLOOR + (tms_xT[None, :] - alt_xTs[:, None]) / 0.3,
        _PROG_FLOOR, 1.5,
    )

    # ---- distance decay (A, T') ----
    decay = np.clip(d_br / _DISTANCE_DECAY_D, 0.0, 0.5)

    # ---- approximate interception risk (A, T') ----
    if D > 0:
        # unit direction alt → teammate
        u = diff / d_br_safe[:, :, None]                                # (A, T', 2)
        # defender vector from each alt
        v = defender_pos[None, None, :, :] - alt_xys[:, None, None, :]  # (A, T', D, 2)
        # along-line component
        proj = np.einsum("atd,atDd->atD", u, v)                         # (A, T', D)
        # perp distance
        v_sq = (v ** 2).sum(axis=3)                                     # (A, T', D)
        perp = np.sqrt(np.maximum(v_sq - proj ** 2, 0.0))               # (A, T', D)
        # defender "in the way" if between ball and receiver and close to line
        between = (proj > 0.0) & (proj < d_br[:, :, None])              # (A, T', D)
        in_lane = between & (perp < _CARRY_INTERCEPT_R)                 # (A, T', D)
        intercept_risk = in_lane.any(axis=2).astype(np.float64)         # (A, T')
        # any defender squarely in lane → fully discount that POV
    else:
        intercept_risk = np.zeros_like(d_br)

    # ---- POV matrix and max over teammates ----
    pov = (
        tms_xT[None, :]
        * V_ctrl
        * (1.0 - intercept_risk)
        * prog
        * (1.0 - decay)
    )
    # Mask out beyond-range receivers
    pov = np.where(d_br > _MAX_PASS_DISTANCE, 0.0, pov)
    return np.clip(pov.max(axis=1), 0.0, 2.0)          # (A,)


# ---------------------------------------------------------------------------
# Event-level ΔPOV and decision quality
# ---------------------------------------------------------------------------

def compute_event_pov(
    analytics_value_df: pd.DataFrame,
    events_df: pd.DataFrame,
    home_tracking: pd.DataFrame,
    away_tracking: pd.DataFrame,
    home_jerseys: list,
    away_jerseys: list,
    xT_home: np.ndarray,
    xT_away: np.ndarray,
    hex_centers: np.ndarray,
    R: float,
    profiles_home: dict,
    profiles_away: dict,
    fps: float = DEFAULT_FPS,
) -> pd.DataFrame:
    """
    Compute ΔPOV and decision quality for every event using targeted PCZ.

    Parameters
    ----------
    analytics_value_df  : output of load_analytics_with_value() — has xT_pos, q, r
    events_df           : standard events DataFrame
    home_tracking       : full home tracking DataFrame (Frame index)
    away_tracking       : full away tracking DataFrame (Frame index)
    home_jerseys        : list of jersey strings for home
    away_jerseys        : list of jersey strings for away
    xT_home, xT_away    : (N,) xT surfaces
    hex_centers         : (N, 2)
    R                   : hex circumradius
    profiles_home/away  : {jersey: {v_max, a_max, reaction_time}}

    Returns
    -------
    pov_df : DataFrame
        Per-player POV rows — event_idx, team, jersey, pov_start, pov_end,
        delta_pov, decision_quality.
    alt_df : DataFrame
        Carry alternatives — one row per (carry event, alternative hex).
        Columns: event_idx, hex_x, hex_y, V_alt, xT_alt, max_pov_alt,
        is_start, is_actual_end.  Used to render the alternatives heatmap
        in the event-step view.
    """
    tree = cKDTree(hex_centers)
    adf  = analytics_value_df.set_index(["Frame", "team", "jersey"])

    # Precompute axial (q, r) for every hex centre — used by the ring-based
    # carry alternatives.  Cheap one-shot; saves N×events worth of recomputation.
    _q_all, _r_all = cartesian_to_hex(
        hex_centers[:, 0], hex_centers[:, 1], R,
    )
    q_all_hex = np.asarray(_q_all, dtype=np.int32)
    r_all_hex = np.asarray(_r_all, dtype=np.int32)

    def _xT_at_pos(frame: int, team: str, jersey: str):
        key = (frame, team, str(jersey))
        if key not in adf.index:
            return np.nan
        row = adf.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        xT_val = row.get("xT_pos", np.nan)
        return float(xT_val) if pd.notna(xT_val) else np.nan

    def _player_xy_from_tracking(tracking_row: pd.Series, prefix: str, jersey: str):
        px = tracking_row.get(f"{prefix}_{jersey}_x")
        py = tracking_row.get(f"{prefix}_{jersey}_y")
        if pd.isna(px) or pd.isna(py):
            return None
        return np.array([float(px), float(py)])

    def _compute_pov_for_frame(
        frame: int,
        ball_xy: np.ndarray,
        possession: str,
    ) -> dict:
        """Return {jersey: pov} for all possessing team players at a frame."""
        if frame not in home_tracking.index or frame not in away_tracking.index:
            return {}

        row_h = home_tracking.loc[frame]
        row_a = away_tracking.loc[frame]

        poss_prefix = "Home" if possession == "Home" else "Away"
        def_prefix  = "Away" if possession == "Home" else "Home"
        poss_jerseys = home_jerseys if possession == "Home" else away_jerseys
        def_jerseys  = away_jerseys if possession == "Home" else home_jerseys
        xT_poss      = xT_home if possession == "Home" else xT_away
        poss_row     = row_h if possession == "Home" else row_a
        def_row      = row_a if possession == "Home" else row_h
        def_profiles = profiles_away if possession == "Home" else profiles_home

        # Ball xT
        _, bidx = tree.query([ball_xy])
        xT_ball = float(xT_poss[bidx[0]])

        # Build defender data once per frame
        defender_data = _build_defender_data(
            def_row, def_prefix, def_jerseys, def_profiles
        )

        pov_map = {}
        for jersey in poss_jerseys:
            recv_xy = _player_xy_from_tracking(poss_row, poss_prefix, jersey)
            if recv_xy is None:
                continue
            xT_r = _xT_at_pos(frame, possession, jersey)
            if np.isnan(xT_r):
                _, ridx = tree.query([recv_xy])
                xT_r = float(xT_poss[ridx[0]])

            pov_map[jersey] = player_pov(
                ball_xy, recv_xy, xT_ball, xT_r, defender_data
            )

        return pov_map

    rows = []
    alt_rows: list = []
    receipt_lookup = build_pass_receipt_lookup(events_df)
    for evt_idx, evt in events_df.iterrows():
        sf_raw = evt.get("Start Frame")
        if pd.isna(sf_raw):
            continue
        start_frame = int(sf_raw)
        ef_raw      = evt.get("End Frame")
        end_frame   = int(ef_raw) if pd.notna(ef_raw) else start_frame

        team_raw   = str(evt.get("Team", "Home"))
        possession = team_raw

        sx, sy = evt.get("Start X"), evt.get("Start Y")
        ex, ey = evt.get("End X"),   evt.get("End Y")
        if pd.isna(sx) or pd.isna(sy):
            continue

        ball_xy_start = np.array([float(sx), float(sy)])
        ball_xy_end   = (np.array([float(ex), float(ey)])
                         if pd.notna(ex) and pd.notna(ey)
                         else ball_xy_start)

        event_type = str(evt.get("Type", "")).upper()

        # For completed passes with a stitched receipt, score end-frame POV at
        # the next event's start (observed arrival) rather than the pass label
        # end frame (~0.7 s earlier on SkillCorner data).
        pov_end_frame = end_frame
        receipt = receipt_lookup.get(evt_idx)
        if (receipt and receipt.get("stitch_valid") and "PASS" in event_type
                and receipt.get("receipt_start_frame") is not None):
            pov_end_frame = int(receipt["receipt_start_frame"])
            tracked_end = ball_xy_from_tracking(
                home_tracking, away_tracking, pov_end_frame,
            )
            if tracked_end is not None:
                ball_xy_end = tracked_end
            elif pd.notna(receipt.get("receipt_start_x")):
                ball_xy_end = np.array([
                    float(receipt["receipt_start_x"]),
                    float(receipt["receipt_start_y"]),
                ])

        def _clean_jersey(val) -> str:
            """Normalise jersey: strip 'Player' prefix and remove .0 float suffix."""
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ""
            if isinstance(val, float) and val == int(val):
                val = int(val)
            return str(val).replace("Player", "").strip()

        carrier_jersey = _clean_jersey(evt.get("From", ""))
        receiver_raw   = _clean_jersey(evt.get("To",   ""))

        pov_start = _compute_pov_for_frame(start_frame, ball_xy_start, possession)
        pov_end   = _compute_pov_for_frame(pov_end_frame, ball_xy_end, possession)

        max_pov_start = max(pov_start.values()) if pov_start else 0.0

        # Decision quality — three branches:
        #   PASS:  POV(actual receiver at end) / max(POV all teammates at start)
        #          End-frame receiver POV rewards timing the ball into a run;
        #          start-frame max is the fixed benchmark at moment of decision.
        #
        #   CARRY: V(actual_end, end_frame)
        #          ──────────────────────────────────────────────────
        #          max( V(h, start_frame) for h ∈ ring_alternatives )
        #
        #          where  V(h, f) = composite (xT + max_POV) at h, pressure-
        #          discounted.  Alternatives are hexes on the **same ring
        #          distance** from the start hex as the chosen end hex
        #          (±1 ring tolerance, ring 0 excluded), filtered to kinematic
        #          reach.  Pure direction-quality comparison at chosen effort.
        #          Denominator stays at start_frame as the fair counterfactual
        #          baseline — an attempt to move it to end_frame inflated DQ
        #          further (end-frame teammates have converged on the actual
        #          play, leaving counterfactual alt hexes with no receivers).
        #
        #   FAILED PASS (BALL LOST with a trajectory):
        #          POV(inferred receiver intent at end) / max(POV at start)
        #          The intended receiver is inferred by walking the ball
        #          extension and finding the first teammate whose start-frame
        #          PCZ would have controlled a hex along it; the intent
        #          location is refined by intersecting the ball ray with the
        #          receiver's projected motion ray.  See §7.3 of the spec.
        dq = np.nan
        v_actual_carry    = np.nan
        v_best_carry      = np.nan
        dq_failed         = np.nan
        failed_confidence = ""
        failed_receiver   = ""
        failed_intent_xy  = None

        if "PASS" in event_type and receiver_raw and max_pov_start > 0.01:
            recv_pov_end = pov_end.get(receiver_raw, 0.0)
            dq = round(float(np.clip(recv_pov_end / max_pov_start, 0.0, 1.5)), 3)
        elif "CARRY" in event_type and carrier_jersey:
            dq, carry_alt_rows, v_actual_carry, v_best_carry = _evaluate_carry(
                evt_idx, start_frame, end_frame,
                ball_xy_start, ball_xy_end,
                possession, carrier_jersey,
                pov_end,
                home_tracking, away_tracking,
                home_jerseys, away_jerseys,
                xT_home, xT_away,
                hex_centers, tree,
                q_all_hex, r_all_hex,
                profiles_home, profiles_away,
                adf, _xT_at_pos,
                fps=fps,
            )
            alt_rows.extend(carry_alt_rows)
        elif (carrier_jersey
              and _is_failed_pass_event(
                  event_type, evt.get("Subtype"),
                  ball_xy_start, ball_xy_end,
              )):
            failed = _evaluate_failed_pass(
                start_frame, end_frame,
                ball_xy_start, ball_xy_end,
                possession, carrier_jersey,
                max_pov_start,
                home_tracking, away_tracking,
                home_jerseys, away_jerseys,
                xT_home, xT_away,
                tree,
                profiles_home, profiles_away,
            )
            dq_failed         = failed["dq"]
            failed_confidence = failed["confidence"]
            failed_receiver   = failed["receiver"]
            failed_intent_xy  = failed["intent_xy"]

        all_jerseys = set(pov_start) | set(pov_end)
        # Ensure the carrier (event actor) always has a row, even when the
        # carrier's Player-ID number does not correspond to a tracking-slot
        # number (i.e. a substitute, whose Metrica Player ID > the squad's
        # starting jerseys).  Without this, actor-only fields like
        # decision_quality are computed but never reach pov_df.
        # NB: pov_start/pov_end may not have a value for the carrier in this
        # case — the .get(..., 0.0) below handles that; the carrier's true
        # ΔPOV must be resolved downstream via the roster's slot mapping at
        # event time, not from this row.
        if carrier_jersey:
            all_jerseys = all_jerseys | {carrier_jersey}
        for jersey in all_jerseys:
            ps = pov_start.get(jersey, 0.0)
            pe = pov_end.get(jersey, 0.0)
            is_actor = jersey == carrier_jersey
            rows.append({
                "event_idx":               evt_idx,
                "team":                    possession,
                "jersey":                  jersey,
                "pov_start":               round(ps, 4),
                "pov_end":                 round(pe, 4),
                "delta_pov":               round(pe - ps, 4),
                "decision_quality":        dq if is_actor else np.nan,
                "v_actual_carry":          v_actual_carry if is_actor else np.nan,
                "v_best_carry":            v_best_carry if is_actor else np.nan,
                "decision_quality_failed": dq_failed if is_actor else np.nan,
                "inference_confidence":    failed_confidence if is_actor else "",
                "inferred_receiver":       failed_receiver if is_actor else "",
                "intent_x": (
                    float(failed_intent_xy[0])
                    if (is_actor and failed_intent_xy is not None) else np.nan
                ),
                "intent_y": (
                    float(failed_intent_xy[1])
                    if (is_actor and failed_intent_xy is not None) else np.nan
                ),
            })

    pov_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["event_idx", "team", "jersey", "pov_start", "pov_end",
                 "delta_pov", "decision_quality",
                 "v_actual_carry", "v_best_carry",
                 "decision_quality_failed", "inference_confidence",
                 "inferred_receiver", "intent_x", "intent_y"]
    )
    alt_df = pd.DataFrame(alt_rows) if alt_rows else pd.DataFrame(
        columns=["event_idx", "hex_x", "hex_y", "V_alt",
                 "xT_alt", "max_pov_alt",
                 "is_start", "is_actual_end", "is_best_alt"]
    )
    return pov_df, alt_df


# ---------------------------------------------------------------------------
# Failed-pass DQ helpers — inferred-receiver decision quality
# ---------------------------------------------------------------------------
#
# A "failed pass" is a BALL LOST event with a real trajectory (start≠end,
# distance ≥ 5 m) whose subtype isn't a non-decision (CLEARANCE, THEFT,
# OFFSIDE, etc.).  Many intercepted passes in Metrica's data aren't tagged
# INTERCEPTION but show the same signature; relying on the explicit label
# alone biases the sample heavily.
#
# Pipeline:
#   1. Walk the ball trajectory **past** the interception point in 1 m steps,
#      asking at each sample: "who would have reached this hex first?"  PCZ
#      uses kinematic travel times (player_travel_time) for every teammate
#      AND every opponent at the start-frame state (the world the passer
#      saw when deciding).  The first sample where the cheapest teammate
#      beats the cheapest opponent gives us the inferred intended receiver.
#   2. If no teammate ever wins control before we walk off the pitch or hit
#      _PCZ_WALK_MAX_EXTEND, fall back to the closest teammate by
#      perpendicular distance to the extended line ("weak" confidence).
#   3. Refine the receiver's *intended* destination by intersecting the
#      ball ray (from end_xy in the pass direction) with the receiver's
#      projected motion ray (from their end-frame xy along their velocity).
#      This captures leading through-balls — where the receiver hadn't yet
#      reached the intended meeting point when the ball was cut out.
#      Falls back to the receiver's xy at end_frame if the projection is
#      ill-defined (parallel rays, backward intersection, stationary
#      receiver, off-pitch meeting, or extrapolation > _FAILED_PROJECTION_MAX).
#   4. Compute POV at the resulting intent hex using end-frame defender
#      state — symmetric with how successful-pass DQ evaluates the
#      receiver POV at end_frame.
#
# Confidence levels surfaced to the UI:
#   "strong"  — PCZ walk hit a teammate AND meeting-point projection succeeded
#   "medium"  — PCZ walk hit a teammate but projection fell back
#   "weak"    — perpendicular-distance fallback (no PCZ control before pitch boundary)
# ---------------------------------------------------------------------------


def _is_failed_pass_event(event_type: str, subtype, ball_xy_start, ball_xy_end) -> bool:
    """
    Inclusion rule for BALL LOST events that should be scored as failed passes.
    Requires a meaningful trajectory (start_xy ≠ end_xy, dist ≥ _FAILED_MIN_DIST)
    AND a subtype that doesn't denote a non-pass action (CLEARANCE, THEFT,
    OFFSIDE, HAND BALL, END HALF, GOAL KICK, REFEREE).
    """
    if "BALL LOST" not in event_type:
        return False

    sub_norm = "" if (subtype is None or pd.isna(subtype)) else str(subtype).strip().upper()
    if any(ex in sub_norm for ex in _FAILED_EXCLUDED_SUBTYPES):
        return False

    if ball_xy_start is None or ball_xy_end is None:
        return False
    if np.array_equal(ball_xy_start, ball_xy_end):
        return False
    dist = float(np.linalg.norm(ball_xy_end - ball_xy_start))
    if not np.isfinite(dist) or dist < _FAILED_MIN_DIST:
        return False

    return True


def _project_meeting_point(
    ball_end_xy:        np.ndarray,
    ball_dir:           np.ndarray,        # unit vector
    receiver_xy:        np.ndarray,
    receiver_direction: np.ndarray,        # any 2-D direction vector (need not be unit)
    max_extend:         float = _FAILED_PROJECTION_MAX,
):
    """
    Intersect the ball's extended trajectory with the receiver's projected
    motion ray.  Returns the meeting xy if both rays extend forward into a
    sensible region, else ``None`` (caller should fall back to receiver_xy).

    `receiver_direction` is the *intent* vector along which the receiver is
    extrapolated past `receiver_xy`.  Callers typically pass the receiver's
    **start-frame → end-frame displacement** here (a more stable estimate of
    intent than instantaneous end-frame velocity, which tends to be noisy or
    decelerating exactly at the moment of interception).

    Solves the 2×2 linear system:
        ball_end + t1 · ball_dir == receiver_xy + t2 · receiver_dir_unit
    """
    rec_mag = float(np.linalg.norm(receiver_direction))
    if rec_mag < _RECEIVER_MIN_DISPLACEMENT:
        return None
    rec_dir = receiver_direction / rec_mag

    A = np.array(
        [[ball_dir[0], -rec_dir[0]],
         [ball_dir[1], -rec_dir[1]]], dtype=np.float64,
    )
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-6:
        return None                                     # parallel / anti-parallel
    b = receiver_xy - ball_end_xy
    t1 = ( A[1, 1] * b[0] - A[0, 1] * b[1]) / det
    t2 = (-A[1, 0] * b[0] + A[0, 0] * b[1]) / det
    if t1 < 0 or t2 < 0:
        return None                                     # one ray points the wrong way
    if t1 > max_extend or t2 > max_extend:
        return None                                     # extrapolation too far
    meeting = ball_end_xy + t1 * ball_dir
    if (abs(meeting[0]) > _FAILED_FIELD_HALF_X + 2.0
            or abs(meeting[1]) > _FAILED_FIELD_HALF_Y + 2.0):
        return None                                     # off-pitch
    return meeting


def _player_pos_vel(tracking_row: pd.Series, prefix: str, jersey: str):
    """Pull a single player's (pos, vel) from a tracking row. None if absent."""
    px = tracking_row.get(f"{prefix}_{jersey}_x")
    py = tracking_row.get(f"{prefix}_{jersey}_y")
    if pd.isna(px) or pd.isna(py):
        return None, None
    vx = tracking_row.get(f"{prefix}_{jersey}_vx", 0.0)
    vy = tracking_row.get(f"{prefix}_{jersey}_vy", 0.0)
    vx = 0.0 if pd.isna(vx) else float(vx)
    vy = 0.0 if pd.isna(vy) else float(vy)
    return np.array([float(px), float(py)]), np.array([vx, vy])


def _infer_intended_receiver(
    ball_start_xy: np.ndarray,
    ball_end_xy:   np.ndarray,
    poss_row:      pd.Series,         # start-frame possessing-team tracking row
    def_row:       pd.Series,         # start-frame defending-team tracking row
    poss_prefix:   str,
    def_prefix:    str,
    poss_jerseys:  list,
    def_jerseys:   list,
    poss_profiles: dict,
    def_profiles:  dict,
    passer_jersey: str = "",
) -> tuple:
    """
    Identify the teammate the pass was most likely intended for, by walking
    the ball line past the interception point and finding the first teammate
    whose kinematic PCZ would have controlled a sample hex.

    Returns
    -------
    receiver_jersey : str or None
    intent_xy       : np.ndarray (2,) or None
        Point along the extension where the teammate's PCZ first beat all
        opponents.  None when the walk failed and we used the closest-perp
        fallback (caller will use the receiver's xy in that case).
    walk_confidence : "strong" | "weak"
    """
    direction = ball_end_xy - ball_start_xy
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        return None, None, "weak"
    direction = direction / dist

    teammate_data = []   # (jersey, pos, vel, profile)
    for j in poss_jerseys:
        if j == passer_jersey:
            continue
        pos, vel = _player_pos_vel(poss_row, poss_prefix, j)
        if pos is None:
            continue
        teammate_data.append((j, pos, vel, poss_profiles.get(j, {})))
    if not teammate_data:
        return None, None, "weak"

    opponent_data = []
    for j in def_jerseys:
        pos, vel = _player_pos_vel(def_row, def_prefix, j)
        if pos is None:
            continue
        opponent_data.append((j, pos, vel, def_profiles.get(j, {})))

    # ---- 1) PCZ-walk along the extension (vectorised per player) ----
    #
    # The walk samples a thin **corridor** rather than a single line — at each
    # forward step we also evaluate a small set of lateral offsets perpendicular
    # to the ball direction (defaults to ±2 m in 1 m increments).  This catches
    # teammate PCZs whose controlled hex sits *near* but not exactly *on* the
    # pass trajectory.  An earlier centerline-only walk would skip past those
    # teammates and pick a further-out player who happened to lie on the line,
    # which mis-identified the intended receiver in cases where the closer
    # teammate was 1–2 m off the line.
    n_steps = int(_PCZ_WALK_MAX_EXTEND / _PCZ_WALK_STEP_M)
    ks = np.arange(1, n_steps + 1, dtype=np.float64)
    centerline = ball_end_xy[None, :] + direction[None, :] * (ks * _PCZ_WALK_STEP_M)[:, None]   # (S, 2)

    # Trim trailing off-pitch centerline samples (stop walking when we leave the pitch)
    on_pitch_center = (
        (np.abs(centerline[:, 0]) <= _FAILED_FIELD_HALF_X + 2.0)
        & (np.abs(centerline[:, 1]) <= _FAILED_FIELD_HALF_Y + 2.0)
    )
    off_idx = np.where(~on_pitch_center)[0]
    if off_idx.size > 0:
        centerline = centerline[: int(off_idx[0])]

    if centerline.shape[0] > 0:
        # Build the corridor: each forward step k gets N lateral offset points.
        perp_dir = np.array([-direction[1], direction[0]], dtype=np.float64)   # 90° CCW
        offsets  = np.array(_PCZ_CORRIDOR_OFFSETS, dtype=np.float64)
        N = len(offsets)
        S = centerline.shape[0]

        # (S, N, 2) → flatten to (S·N, 2) for one batched player_travel_time per player
        corridor = centerline[:, None, :] + offsets[None, :, None] * perp_dir[None, None, :]
        pts_flat = corridor.reshape(-1, 2)
        M        = pts_flat.shape[0]

        # Per-point on-pitch mask (off-pitch offsets at the edges get masked out)
        on_pitch_pts = (
            (np.abs(pts_flat[:, 0]) <= _FAILED_FIELD_HALF_X + 2.0)
            & (np.abs(pts_flat[:, 1]) <= _FAILED_FIELD_HALF_Y + 2.0)
        )

        # Stack teammates first, opponents after — winner index < T means
        # the controller is a teammate.
        T = len(teammate_data)
        O = len(opponent_data)
        all_times = np.full((T + O, M), np.inf, dtype=np.float64)
        for i, (_, pos, vel, prof) in enumerate(teammate_data):
            all_times[i] = player_travel_time(
                pos, vel, pts_flat,
                float(prof.get("v_max", 9.0)),
                float(prof.get("a_max", 5.5)),
                reaction_time=float(prof.get("reaction_time", 0.7)),
            )
        for i, (_, pos, vel, prof) in enumerate(opponent_data):
            all_times[T + i] = player_travel_time(
                pos, vel, pts_flat,
                float(prof.get("v_max", 9.0)),
                float(prof.get("a_max", 5.5)),
                reaction_time=float(prof.get("reaction_time", 0.7)),
            )
        # Off-pitch points are no-one's PCZ
        all_times[:, ~on_pitch_pts] = np.inf

        # Per-point controller = the row with the smallest t_player.
        # is_teammate[m] is True iff the controller is one of the teammates AND
        # at least one player has a finite time there (skip phantom off-pitch).
        finite_any  = np.isfinite(all_times).any(axis=0)
        winners     = all_times.argmin(axis=0)        # (M,)
        is_teammate = (winners < T) & finite_any      # (M,)

        # Walk forward step-by-step. At each step, check all N offsets for any
        # teammate-controlled point. If multiple offsets at the same step are
        # teammate-controlled, prefer the one with the tightest control time —
        # that's the teammate whose PCZ "owns" the corridor most strongly here.
        for k in range(S):
            sl = slice(k * N, (k + 1) * N)
            step_team_mask = is_teammate[sl]
            if not step_team_mask.any():
                continue
            local_pt_indices = np.arange(k * N, (k + 1) * N)[step_team_mask]
            local_t          = all_times[winners[local_pt_indices], local_pt_indices]
            best_pt          = int(local_pt_indices[int(np.argmin(local_t))])
            tm_idx           = int(winners[best_pt])
            return teammate_data[tm_idx][0], pts_flat[best_pt], "strong"

    # ---- 2) Fallback: closest teammate by perpendicular distance ----
    best_perp = np.inf
    best_j    = None
    for j, pos, _, _ in teammate_data:
        rel  = pos - ball_end_xy
        proj = float(np.dot(rel, direction))
        if proj <= 0.0 or proj > _PCZ_WALK_MAX_EXTEND:
            continue
        perp = float(np.linalg.norm(rel - proj * direction))
        if perp < best_perp:
            best_perp = perp
            best_j    = j

    if best_j is not None:
        return best_j, None, "weak"
    return None, None, "weak"


def _evaluate_failed_pass(
    start_frame:    int,
    end_frame:      int,
    ball_xy_start:  np.ndarray,
    ball_xy_end:    np.ndarray,
    possession:     str,
    passer_jersey:  str,
    max_pov_start:  float,
    home_tracking:  pd.DataFrame,
    away_tracking:  pd.DataFrame,
    home_jerseys:   list,
    away_jerseys:   list,
    xT_home:        np.ndarray,
    xT_away:        np.ndarray,
    tree:           cKDTree,
    profiles_home:  dict,
    profiles_away:  dict,
) -> dict:
    """
    Score an intercepted (or otherwise-failed) pass.  Returns a dict with:

        dq          : float in [0, 1.5] or np.nan
        confidence  : ""  (skipped)  |  "strong" | "medium" | "weak"
        receiver    : jersey of inferred receiver, or "" if skipped
        intent_xy   : np.ndarray (2,) or None — location used for POV
    """
    blank = dict(dq=np.nan, confidence="", receiver="", intent_xy=None)

    if (start_frame not in home_tracking.index
            or start_frame not in away_tracking.index
            or end_frame not in home_tracking.index
            or end_frame not in away_tracking.index):
        return blank
    if max_pov_start <= 1e-3:
        # No real passing option existed at decision time → DQ undefined
        return blank

    poss_prefix  = "Home" if possession == "Home" else "Away"
    def_prefix   = "Away" if possession == "Home" else "Home"
    poss_jerseys = home_jerseys if possession == "Home" else away_jerseys
    def_jerseys  = away_jerseys if possession == "Home" else home_jerseys
    poss_row_s   = home_tracking.loc[start_frame] if possession == "Home" else away_tracking.loc[start_frame]
    def_row_s    = away_tracking.loc[start_frame] if possession == "Home" else home_tracking.loc[start_frame]
    poss_row_e   = home_tracking.loc[end_frame]   if possession == "Home" else away_tracking.loc[end_frame]
    def_row_e    = away_tracking.loc[end_frame]   if possession == "Home" else home_tracking.loc[end_frame]
    xT_poss      = xT_home if possession == "Home" else xT_away
    poss_profile = profiles_home if possession == "Home" else profiles_away
    def_profile  = profiles_away if possession == "Home" else profiles_home

    # ---- Inferred receiver (start-frame PCZ on the extended ball line) ----
    receiver, walk_intent_xy, walk_conf = _infer_intended_receiver(
        ball_xy_start, ball_xy_end,
        poss_row_s, def_row_s,
        poss_prefix, def_prefix,
        poss_jerseys, def_jerseys,
        poss_profile, def_profile,
        passer_jersey=passer_jersey,
    )
    if receiver is None:
        return blank

    # ---- Receiver positions at start & end frames ----
    # The receiver's start-frame xy gives us their position when the pass began,
    # and end-frame xy gives us where they were at interception.  The vector
    # between them is the receiver's *average heading during the pass* — a much
    # more stable estimator of intent than instantaneous end-frame velocity,
    # which is often noisy / decelerating right at the moment of interception.
    recv_pos_start, _ = _player_pos_vel(poss_row_s, poss_prefix, receiver)
    recv_pos_end,   _ = _player_pos_vel(poss_row_e, poss_prefix, receiver)
    if recv_pos_end is None:
        # Receiver dropped out of tracking at end_frame — use start-frame pos
        if recv_pos_start is None:
            return blank
        recv_pos_end = recv_pos_start
    if recv_pos_start is None:
        # No start-frame position available — projection will fall back below
        recv_pos_start = recv_pos_end

    # ---- Meeting-point projection (ball ray ∩ receiver-intent ray) ----
    direction = ball_xy_end - ball_xy_start
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        return blank
    ball_dir = direction / dist

    # Receiver's intent vector = start→end displacement.  Extend past recv_pos_end
    # along this line, and intersect with the extended ball line.
    recv_displacement = recv_pos_end - recv_pos_start
    meeting = _project_meeting_point(
        ball_xy_end, ball_dir, recv_pos_end, recv_displacement,
    )
    if meeting is not None:
        intent_xy_final = meeting
        confidence      = walk_conf            # strong stays strong, weak stays weak
    else:
        intent_xy_final = recv_pos_end
        # PCZ walk succeeded but projection fell back → downgrade to medium
        confidence = "medium" if walk_conf == "strong" else walk_conf

    # ---- POV at the intent hex (end_frame defender state) ----
    _, b_idx = tree.query([ball_xy_end])
    xT_ball  = float(xT_poss[b_idx[0]])
    _, r_idx = tree.query([intent_xy_final])
    xT_recv  = float(xT_poss[r_idx[0]])

    defender_data_end = _build_defender_data(
        def_row_e, def_prefix, def_jerseys, def_profile,
    )
    pov_intent = player_pov(
        ball_xy_end, intent_xy_final, xT_ball, xT_recv, defender_data_end,
    )

    dq = round(float(np.clip(pov_intent / max(max_pov_start, 1e-3), 0.0, 1.5)), 3)
    return dict(
        dq=dq,
        confidence=confidence,
        receiver=str(receiver),
        intent_xy=intent_xy_final,
    )


# ---------------------------------------------------------------------------
# Carry evaluator — kinematic alternatives + composite V
# ---------------------------------------------------------------------------

def _evaluate_carry(
    evt_idx,
    start_frame:    int,
    end_frame:      int,
    ball_xy_start:  np.ndarray,
    ball_xy_end:    np.ndarray,
    possession:     str,
    carrier_jersey: str,
    pov_end:        dict,
    home_tracking:  pd.DataFrame,
    away_tracking:  pd.DataFrame,
    home_jerseys:   list,
    away_jerseys:   list,
    xT_home:        np.ndarray,
    xT_away:        np.ndarray,
    hex_centers:    np.ndarray,
    tree:           cKDTree,
    q_all_hex:      np.ndarray,
    r_all_hex:      np.ndarray,
    profiles_home:  dict,
    profiles_away:  dict,
    adf:            pd.DataFrame,
    xT_at_pos_fn,
    fps:            float = DEFAULT_FPS,
) -> tuple:
    """
    Compute carry decision quality via counterfactual alternative destinations.

    Alternatives are hexes on the **same ring** as the chosen end hex
    (±_CARRY_RING_TOL rings around it, ring 0 excluded), filtered to those
    the carrier could kinematically reach in the carry's duration.  The
    formula becomes a pure direction-quality judgement at the chosen effort:
    "given that you carried this distance, was this the best direction?"

    Returns
    -------
    dq : float
        carry_dq = V_actual_end / V_best_alt_start  (clipped to [0, 1.5]).
        Numerator at end_frame, denominator at start_frame — start_frame
        is used for the alts as the fair counterfactual baseline; see §7.2
        of the analytics spec for why an attempt to symmetrise on end_frame
        made the inflation worse.
        ``np.nan`` for ring-0 carries (no real movement → no real decision).
    alt_rows : list[dict]
        One row per alternative hex for the alt_df, used by the heatmap.
        Each row carries ``V_alt`` (start-frame) and a boolean ``is_best_alt``
        flag for the hex(es) within ``_CARRY_BEST_TIE_TOL`` of ``V_best``.
        Empty for ring-0 carries.
    v_actual : float
        V(actual_end_hex, end_frame).  Numerator of the DQ ratio.
        ``np.nan`` for ring-0 carries.
    v_best : float
        max(V(h, start_frame)) over ring alternatives.  Denominator before
        flooring; equal to ``_CARRY_V_BEST_FLOOR`` only when every ring
        alternative was sub-floor (dead-zone carry).  ``np.nan`` for ring-0.
    """
    if start_frame not in home_tracking.index or start_frame not in away_tracking.index:
        return np.nan, [], np.nan, np.nan
    if end_frame not in home_tracking.index or end_frame not in away_tracking.index:
        return np.nan, [], np.nan, np.nan

    poss_prefix  = "Home" if possession == "Home" else "Away"
    def_prefix   = "Away" if possession == "Home" else "Home"
    poss_jerseys = home_jerseys if possession == "Home" else away_jerseys
    def_jerseys  = away_jerseys if possession == "Home" else home_jerseys
    poss_row_s   = home_tracking.loc[start_frame] if possession == "Home" else away_tracking.loc[start_frame]
    xT_poss      = xT_home if possession == "Home" else xT_away
    poss_profile = profiles_home if possession == "Home" else profiles_away

    # ---- Carrier kinematic profile + state at decision moment ----
    px_s = poss_row_s.get(f"{poss_prefix}_{carrier_jersey}_x")
    py_s = poss_row_s.get(f"{poss_prefix}_{carrier_jersey}_y")
    if pd.isna(px_s) or pd.isna(py_s):
        return np.nan, [], np.nan, np.nan
    vx_s = poss_row_s.get(f"{poss_prefix}_{carrier_jersey}_vx", 0.0)
    vy_s = poss_row_s.get(f"{poss_prefix}_{carrier_jersey}_vy", 0.0)
    vx_s = 0.0 if pd.isna(vx_s) else float(vx_s)
    vy_s = 0.0 if pd.isna(vy_s) else float(vy_s)
    carrier_pos = np.array([float(px_s), float(py_s)])
    carrier_vel = np.array([vx_s, vy_s])

    prof = poss_profile.get(carrier_jersey, {})
    v_max = float(prof.get("v_max",         9.0))
    a_max = float(prof.get("a_max",         5.5))

    duration = max(0.04, (end_frame - start_frame) / fps)

    # ---- Identify start and chosen-end hex indices + axial coords ----
    start_hex_idx_pre  = int(np.argmin(np.linalg.norm(hex_centers - carrier_pos, axis=1)))
    _, _actual_q       = tree.query([ball_xy_end])
    actual_hex_idx_pre = int(_actual_q[0])
    start_q = int(q_all_hex[start_hex_idx_pre])
    start_r = int(r_all_hex[start_hex_idx_pre])
    end_q   = int(q_all_hex[actual_hex_idx_pre])
    end_r   = int(r_all_hex[actual_hex_idx_pre])

    # Chosen ring distance from start hex (axial cube distance)
    dq_se, dr_se = start_q - end_q, start_r - end_r
    chosen_ring  = (abs(dq_se) + abs(dr_se) + abs(dq_se + dr_se)) // 2

    # ---- Ring-0 carries: degenerate label events, no meaningful decision ----
    if chosen_ring == 0:
        return np.nan, [], np.nan, np.nan

    # ---- Ring-based alternative destinations (filtered by kinematic reach) ----
    alt_idx = _ring_alternatives(
        carrier_pos, carrier_vel, duration, hex_centers,
        q_all_hex, r_all_hex, start_q, start_r,
        actual_hex_idx_pre, chosen_ring,
        v_max, a_max,
    )
    if alt_idx.size == 0:
        return np.nan, [], np.nan, np.nan
    alt_xys = hex_centers[alt_idx]
    alt_xTs = xT_poss[alt_idx]

    # ---- Teammate state at start_frame (fair counterfactual baseline) ----
    # The denominator stays at start_frame deliberately.  An attempt to move
    # it to end_frame (Path A option 2 in §7.2's history) made the inflation
    # worse: at end_frame teammates have converged toward the actual play,
    # so a counterfactual alt hex has no receivers nearby and V_alt deflates,
    # *inflating* the DQ ratio.  Start-frame teammate positions are the right
    # baseline — "if you'd gone to alt instead, here's where your teammates
    # were at the moment of your decision, before they responded to the play
    # you actually made".
    tm_pos_list:  list = []
    tm_xT_list:   list = []
    carrier_idx_in_tm = -1
    for j in poss_jerseys:
        tpx = poss_row_s.get(f"{poss_prefix}_{j}_x")
        tpy = poss_row_s.get(f"{poss_prefix}_{j}_y")
        if pd.isna(tpx) or pd.isna(tpy):
            continue
        tm_pos_list.append([float(tpx), float(tpy)])
        xT_t = xT_at_pos_fn(start_frame, possession, j)
        if np.isnan(xT_t):
            _, tidx = tree.query([[float(tpx), float(tpy)]])
            xT_t = float(xT_poss[tidx[0]])
        tm_xT_list.append(float(xT_t))
        if j == carrier_jersey:
            carrier_idx_in_tm = len(tm_pos_list) - 1
    tm_pos = np.array(tm_pos_list) if tm_pos_list else np.empty((0, 2))
    tm_xTs = np.array(tm_xT_list)  if tm_xT_list  else np.empty((0,))

    # ---- Defender positions at start_frame ----
    def_row_s = (
        away_tracking.loc[start_frame] if possession == "Home"
        else home_tracking.loc[start_frame]
    )
    def_pos_list: list = []
    for j in def_jerseys:
        dpx = def_row_s.get(f"{def_prefix}_{j}_x")
        dpy = def_row_s.get(f"{def_prefix}_{j}_y")
        if pd.isna(dpx) or pd.isna(dpy):
            continue
        def_pos_list.append([float(dpx), float(dpy)])
    def_pos = np.array(def_pos_list) if def_pos_list else np.empty((0, 2))

    # ---- V_alt(h, start_frame) for every alternative ----
    max_pov_alts = _approx_max_pov_per_alt(
        alt_xys, alt_xTs, tm_pos, tm_xTs, def_pos, carrier_idx_in_tm,
    )
    # Carrier pressure discount at each alt (start-frame defender positions)
    press_disc_alts = _carrier_pressure_discount(alt_xys, def_pos)
    V_alts = (
        (_CARRY_W_POS * alt_xTs + _CARRY_W_OPT * max_pov_alts)
        * (1.0 - press_disc_alts)
    )                                                                # (A,)
    V_alts_max = float(V_alts.max())
    V_best = float(max(V_alts_max, _CARRY_V_BEST_FLOOR))

    # Flag the best alternative hex(es).  We use a small tolerance so that
    # genuine ties (multiple ring members within _CARRY_BEST_TIE_TOL of the
    # max V) all get marked — keeps the visual honest about ambiguous cases.
    # When the max sits at the floor (dead-zone carry), no hex is "best" —
    # there's no meaningful direction to highlight.
    if V_alts_max > _CARRY_V_BEST_FLOOR:
        best_mask = V_alts >= (V_alts_max - _CARRY_BEST_TIE_TOL * max(V_alts_max, 1e-6))
    else:
        best_mask = np.zeros_like(V_alts, dtype=bool)

    # ---- V_actual at end_frame (uses full POV — reuses pov_end) ----
    xT_end_val   = float(xT_poss[actual_hex_idx_pre])
    # max_POV at end excludes the carrier (can't pass to self)
    pov_end_no_carrier = {j: v for j, v in pov_end.items() if j != carrier_jersey}
    max_pov_end  = max(pov_end_no_carrier.values()) if pov_end_no_carrier else 0.0

    # Defender positions at end_frame for the actual-end pressure discount
    def_row_e    = (
        away_tracking.loc[end_frame] if possession == "Home"
        else home_tracking.loc[end_frame]
    )
    def_pos_end_list: list = []
    for j in def_jerseys:
        dpx = def_row_e.get(f"{def_prefix}_{j}_x")
        dpy = def_row_e.get(f"{def_prefix}_{j}_y")
        if pd.isna(dpx) or pd.isna(dpy):
            continue
        def_pos_end_list.append([float(dpx), float(dpy)])
    def_pos_end = np.array(def_pos_end_list) if def_pos_end_list else np.empty((0, 2))
    press_disc_actual = float(
        _carrier_pressure_discount(ball_xy_end[None, :], def_pos_end)[0]
    )
    V_actual = float(
        (_CARRY_W_POS * xT_end_val + _CARRY_W_OPT * max_pov_end)
        * (1.0 - press_disc_actual)
    )

    dq = round(float(np.clip(V_actual / V_best, 0.0, 1.5)), 3)

    # ---- Alt rows for the heatmap ----
    actual_hex_idx = actual_hex_idx_pre

    alt_rows = []
    for h_idx, x_alt, xT_a, mpov, V, is_best in zip(
        alt_idx, alt_xys, alt_xTs, max_pov_alts, V_alts, best_mask,
    ):
        alt_rows.append({
            "event_idx":     evt_idx,
            "hex_x":         float(x_alt[0]),
            "hex_y":         float(x_alt[1]),
            "V_alt":         round(float(V), 4),
            "xT_alt":        round(float(xT_a), 4),
            "max_pov_alt":   round(float(mpov), 4),
            "is_start":      False,  # start hex always excluded under ring rules
            "is_actual_end": bool(int(h_idx) == actual_hex_idx),
            "is_best_alt":   bool(is_best),
        })

    return dq, alt_rows, round(V_actual, 4), round(V_alts_max, 4)


# ---------------------------------------------------------------------------
# Corridor visualisation helpers
# ---------------------------------------------------------------------------

def corridor_traces_for_frame(
    ball_xy: np.ndarray,
    player_positions: dict,
    pov_dict: dict,
    pov_threshold: float = 0.05,
) -> list:
    """
    Plotly Scatter trace data (x, y, rgba) for visible passing corridors.
    Colour intensity scales with POV value.
    """
    traces = []
    for jersey, pos in player_positions.items():
        pov = pov_dict.get(jersey, 0.0)
        if pov < pov_threshold:
            continue
        rx, ry = float(pos[0]), float(pos[1])
        if np.isnan(rx) or np.isnan(ry):
            continue
        alpha = round(min(0.2 + pov * 0.6, 0.9), 2)
        g     = int(min(100 + pov * 155, 255))
        rgba  = f"rgba(80,{g},80,{alpha})"
        traces.append((
            [float(ball_xy[0]), rx, None],
            [float(ball_xy[1]), ry, None],
            rgba,
        ))
    return traces


def compute_frame_pov(
    ball_xy: np.ndarray,
    player_positions: dict,
    xT_ball: float,
    xT_map: np.ndarray,
    V_ctrl_attacking: np.ndarray,
    V_ctrl_defending: np.ndarray,
    hex_centers: np.ndarray,
    possession: str,
) -> dict:
    """Legacy signature kept for corridor visualisation in Frame Inspector."""
    tree = cKDTree(hex_centers)
    result: dict = {}
    for jersey, pos in player_positions.items():
        rx, ry = float(pos[0]), float(pos[1])
        if np.isnan(rx) or np.isnan(ry):
            continue
        _, idx   = tree.query([[rx, ry]])
        xT_r     = float(xT_map[idx[0]])
        v_ctrl_r = float(V_ctrl_attacking[idx[0]])
        # Approximate: use V_ctrl values as proxy without full kinematic
        prog  = _progressive_factor(xT_ball, xT_r)
        decay = _distance_decay(ball_xy, np.array([rx, ry]))
        pov   = xT_r * v_ctrl_r * prog * (1.0 - decay)
        result[jersey] = float(np.clip(pov, 0.0, 2.0))
    return result
