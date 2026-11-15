#!/usr/bin/env python3
"""
pitch_control.py

Personal Control Zone (PCZ) model for Metrica tracking data.

Core idea
---------
At every frame, each hex on the pitch is assigned to whichever player can
reach it before the ball does (given the ball is played from its current
position at realistic pass speed).

  T_ball(hex)   = time for ball to travel from ball_pos → hex
                  (speed increases with distance, modelling realistic pass pace)

  T_player(hex) = reaction_time
                + kinematic time from player position/velocity → hex
                  (accounts for current running direction and speed)

  A player controls a hex if  T_player(hex) ≤ T_ball(hex)

  When both teams have a player that can reach the hex:
      winner = player with lower T_player  (contested → coloured by winner)

Output
------
A 2-D control array on a regular (x, y) grid suitable for go.Heatmap:
    +1  = Home controls that point
    -1  = Away controls
     0  = uncontrolled (neither team reaches before ball)
"""

import numpy as np
import pandas as pd
from matrix_engine.pitch_config import FIELD_LENGTH, FIELD_WIDTH
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Default player physical profile (uniform — role-based can be added later)
DEFAULT_PROFILE = {
    "v_max":         9.0,   # m/s  — average outfield player sprint speed
    "a_max":         5.5,   # m/s² — acceleration
    "reaction_time": 0.7,   # s    — perception-action delay
}

# Ball travel speed model: speed = MIN_BALL_SPEED + BALL_SPEED_GRADIENT * distance
# Represents the AVERAGE expected ball pace at a given distance.
# Calibrated so that a player's typical 5-10m personal space is shown as controlled:
#   d=10m → speed ≈ 7 m/s, T_ball ≈ 1.43s
#   d=30m → speed ≈ 9 m/s, T_ball ≈ 3.33s
#   d=60m → speed ≈ 11 m/s, T_ball ≈ 5.45s  (roughly matches empirical pitch-control models)
MIN_BALL_SPEED    = 5.0    # m/s at d=0
BALL_SPEED_GRAD   = 0.2    # m/s per metre of distance


# ---------------------------------------------------------------------------
# Hex grid helpers
# ---------------------------------------------------------------------------

def build_pitch_hex_grid(R: float):
    """
    Return (N, 2) array of all hex centre coordinates (cx, cy) covering
    the pitch, centred at (0, 0), pointy-top orientation.
    """
    dx = np.sqrt(3) * R
    dy = 1.5 * R
    margin = R * 2

    n_cols = int(np.ceil((FIELD_LENGTH / 2 + margin) / dx)) + 1
    n_rows = int(np.ceil((FIELD_WIDTH  / 2 + margin) / dy)) + 1

    centers = []
    for row in range(-n_rows, n_rows + 1):
        cy = row * dy
        cx_offset = (dx / 2) if (row % 2 != 0) else 0.0
        for col in range(-n_cols, n_cols + 1):
            cx = col * dx + cx_offset
            if (-FIELD_LENGTH / 2 - R <= cx <= FIELD_LENGTH / 2 + R and
                    -FIELD_WIDTH  / 2 - R <= cy <= FIELD_WIDTH  / 2 + R):
                centers.append([cx, cy])

    return np.array(centers, dtype=np.float32)


def build_render_grid(resolution: float = 1.0):
    """
    Regular (x, y) grid over the pitch for go.Heatmap rendering.
    Returns x_grid (1-D), y_grid (1-D), grid_points (N, 2).
    """
    x_grid = np.arange(-FIELD_LENGTH / 2, FIELD_LENGTH / 2 + resolution, resolution)
    y_grid = np.arange(-FIELD_WIDTH  / 2, FIELD_WIDTH  / 2 + resolution, resolution)
    XX, YY = np.meshgrid(x_grid, y_grid)
    grid_points = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    return x_grid, y_grid, grid_points


def build_grid_to_hex_mapping(hex_centers: np.ndarray, grid_points: np.ndarray):
    """
    Precompute nearest-hex index for every render-grid point using a KD-tree.
    Call once and reuse across all frames.
    """
    tree = cKDTree(hex_centers)
    _, idx = tree.query(grid_points)
    return idx   # shape: (N_grid,)


# ---------------------------------------------------------------------------
# Travel time functions (fully vectorised over hex grid)
# ---------------------------------------------------------------------------

def ball_travel_time(ball_pos: np.ndarray, hex_centers: np.ndarray) -> np.ndarray:
    """
    Time for ball to travel from ball_pos to each hex centre.

    Speed model: speed = MIN_BALL_SPEED + BALL_SPEED_GRAD * distance
    This replicates empirical pass pace — short passes ~8 m/s, long ~20+ m/s.

    Parameters
    ----------
    ball_pos    : (2,) — ball x, y in metres
    hex_centers : (N, 2)

    Returns
    -------
    t_ball : (N,) in seconds
    """
    d = np.linalg.norm(hex_centers - ball_pos, axis=1)
    speed = MIN_BALL_SPEED + BALL_SPEED_GRAD * d
    return d / np.maximum(speed, 1e-6)


def player_travel_time(
    player_pos: np.ndarray,
    player_vel: np.ndarray,
    hex_centers: np.ndarray,
    v_max: float,
    a_max: float,
    reaction_time: float,
) -> np.ndarray:
    """
    Time for a player to reach each hex centre, accounting for their current
    velocity direction and speed using a constant-acceleration kinematic model.

    Phase 1 (acceleration): player accelerates from their current speed
                            (projected towards the hex) up to v_max.
    Phase 2 (cruise):       player runs at v_max for remaining distance.

    Parameters
    ----------
    player_pos  : (2,) — player x, y
    player_vel  : (2,) — player vx, vy (m/s)
    hex_centers : (N, 2)
    v_max, a_max, reaction_time : player physical profile

    Returns
    -------
    t_player : (N,) in seconds
    """
    delta = hex_centers - player_pos          # (N, 2) vectors to each hex
    d     = np.linalg.norm(delta, axis=1)     # (N,) distances

    # Unit vectors towards each hex (safe divide)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(d[:, None] > 1e-6, delta / d[:, None], 0.0)

    # Raw velocity projection — positive means moving toward hex, negative means away
    v_proj_raw  = player_vel @ u.T                   # (N,) — can be negative
    v_pos       = np.maximum(0.0, v_proj_raw)        # forward component (head-start)
    v_neg       = np.maximum(0.0, -v_proj_raw)       # adverse component (must brake first)

    # Turning correction: when moving away from the hex the player must
    # first decelerate their adverse speed before they can turn and run.
    #   t_turn      = time to brake adverse speed to zero
    #   d_overshoot = extra distance covered going the wrong way during braking
    # After braking they start from rest at effective distance (d + d_overshoot).
    t_turn      = v_neg / a_max                      # (N,)
    d_overshoot = v_neg ** 2 / (2.0 * a_max)         # (N,)

    # Effective starting conditions after any required turn
    d_eff = np.where(v_neg > 0, d + d_overshoot, d)  # (N,)
    v0    = np.where(v_neg > 0, 0.0, v_pos)           # (N,)

    # Distance covered while accelerating from v0 to v_max
    d_accel = np.maximum(0.0, (v_max ** 2 - v0 ** 2) / (2.0 * a_max))   # (N,)
    t_accel = (v_max - v0) / a_max                                         # (N,)

    # Remaining distance after acceleration phase
    d_remain = np.maximum(0.0, d_eff - d_accel)
    t_cruise  = t_accel + d_remain / v_max     # (N,) — full acceleration + cruise

    # Short case: player reaches hex before hitting v_max
    # solve d_eff = v0*t + 0.5*a*t²  →  t = (-v0 + sqrt(v0² + 2a*d_eff)) / a
    disc    = np.maximum(0.0, v0 ** 2 + 2.0 * a_max * d_eff)
    t_short = (-v0 + np.sqrt(disc)) / a_max      # (N,)

    # Choose: if d_eff < d_accel, player hasn't reached v_max yet
    t_move = np.where(d_eff <= d_accel, t_short, t_cruise)

    # Add braking time on top (only non-zero when player was moving away)
    t_move = t_move + t_turn

    # At player's current position, just reaction time
    t_move = np.where(d < 1e-6, 0.0, t_move)

    return reaction_time + t_move     # (N,)


# ---------------------------------------------------------------------------
# Frame-level control computation
# ---------------------------------------------------------------------------

def compute_frame_control(
    row_h: pd.Series,
    row_a: pd.Series,
    home_jerseys: list,
    away_jerseys: list,
    hex_centers: np.ndarray,
    profile: dict = None,
    profiles_home: dict = None,
    profiles_away: dict = None,
) -> tuple:
    """
    For a single tracking frame, compute which player/team controls each hex.

    Parameters
    ----------
    row_h, row_a    : tracking data rows for home and away (from DataFrame.loc[frame])
    home_jerseys    : list of jersey number strings for home team
    away_jerseys    : list of jersey number strings for away team
    hex_centers     : (N, 2) array of hex centre coordinates
    profile         : fallback physical profile (uses DEFAULT_PROFILE if None)
    profiles_home   : optional {jersey_str: profile_dict} — per-player overrides for home
    profiles_away   : optional {jersey_str: profile_dict} — per-player overrides for away

    Returns
    -------
    control   : (N,) float array — +1 Home, -1 Away, 0 uncontrolled
    t_ball    : (N,) ball travel times
    t_home    : (N,) min home player travel times
    t_away    : (N,) min away player travel times
    """
    fallback     = profile if profile is not None else DEFAULT_PROFILE
    profiles_home = profiles_home or {}
    profiles_away = profiles_away or {}

    N            = len(hex_centers)
    INF          = np.float32(1e6)

    # ---- Ball position and travel times ----
    bx = row_a.get("ball_x")
    by = row_a.get("ball_y")
    if pd.isna(bx) or pd.isna(by):
        # Ball not detected — return uncontrolled grid
        return np.zeros(N, dtype=np.float32), None, None, None, {}, {}

    ball_pos = np.array([bx, by], dtype=np.float32)
    t_ball   = ball_travel_time(ball_pos, hex_centers).astype(np.float32)

    # ---- Home player travel times ----
    t_home_all = np.full((len(home_jerseys), N), INF, dtype=np.float32)
    for i, jersey in enumerate(home_jerseys):
        px = row_h.get(f"Home_{jersey}_x")
        py = row_h.get(f"Home_{jersey}_y")
        if pd.isna(px) or pd.isna(py):
            continue
        vx = row_h.get(f"Home_{jersey}_vx", 0.0)
        vy = row_h.get(f"Home_{jersey}_vy", 0.0)
        vx = 0.0 if pd.isna(vx) else float(vx)
        vy = 0.0 if pd.isna(vy) else float(vy)
        p  = profiles_home.get(jersey, fallback)
        t_home_all[i] = player_travel_time(
            np.array([float(px), float(py)], dtype=np.float32),
            np.array([vx, vy], dtype=np.float32),
            hex_centers, p["v_max"], p["a_max"], p["reaction_time"],
        )

    # ---- Away player travel times ----
    t_away_all = np.full((len(away_jerseys), N), INF, dtype=np.float32)
    for i, jersey in enumerate(away_jerseys):
        px = row_a.get(f"Away_{jersey}_x")
        py = row_a.get(f"Away_{jersey}_y")
        if pd.isna(px) or pd.isna(py):
            continue
        vx = row_a.get(f"Away_{jersey}_vx", 0.0)
        vy = row_a.get(f"Away_{jersey}_vy", 0.0)
        vx = 0.0 if pd.isna(vx) else float(vx)
        vy = 0.0 if pd.isna(vy) else float(vy)
        p  = profiles_away.get(jersey, fallback)
        t_away_all[i] = player_travel_time(
            np.array([float(px), float(py)], dtype=np.float32),
            np.array([vx, vy], dtype=np.float32),
            hex_centers, p["v_max"], p["a_max"], p["reaction_time"],
        )

    # ---- Aggregate to team minimums ----
    t_home = t_home_all.min(axis=0)   # (N,)
    t_away = t_away_all.min(axis=0)   # (N,)

    # ---- Assign control ----
    home_reaches = t_home <= t_ball
    away_reaches = t_away <= t_ball

    # Both reach → winner is whoever is faster; neither → 0
    control = np.where(
        home_reaches & away_reaches,
        np.where(t_home <= t_away, 1.0, -1.0),
        np.where(home_reaches,  1.0,
        np.where(away_reaches, -1.0, 0.0)),
    ).astype(np.float32)

    # ---- Per-player attribution ----
    # For each controlled hex, which specific player gets there first?
    home_winner = t_home_all.argmin(axis=0)   # (N,) index into home_jerseys
    away_winner = t_away_all.argmin(axis=0)   # (N,) index into away_jerseys

    player_ctrl_home = {
        jersey: (control > 0.5) & (home_winner == i)
        for i, jersey in enumerate(home_jerseys)
    }
    player_ctrl_away = {
        jersey: (control < -0.5) & (away_winner == i)
        for i, jersey in enumerate(away_jerseys)
    }

    return control, t_ball, t_home, t_away, player_ctrl_home, player_ctrl_away


def control_to_heatmap(
    control: np.ndarray,
    grid_to_hex: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    """
    Map per-hex control values onto the regular render grid for go.Heatmap.

    Returns z : (len(y_grid), len(x_grid)) float array
    """
    z = control[grid_to_hex].reshape(len(y_grid), len(x_grid))
    return z


def control_to_hex_traces(
    control: np.ndarray,
    hex_centers: np.ndarray,
    R: float,
) -> tuple:
    """
    Convert per-hex control values into two sets of polygon coordinates
    (one for Home, one for Away), using the same pointy-top hex geometry
    as the rest of the application.

    Returns
    -------
    (home_x, home_y), (away_x, away_y)
        Each is a flat Python list with None separators between hexagons,
        ready to pass straight to go.Scatter(fill="toself").
    """
    angles = np.radians([30, 90, 150, 210, 270, 330])  # pointy-top
    cos_a  = np.cos(angles)
    sin_a  = np.sin(angles)

    home_x: list = []
    home_y: list = []
    away_x: list = []
    away_y: list = []

    for (cx, cy), ctrl in zip(hex_centers, control):
        if ctrl > 0.5:
            hx = (cx + R * cos_a).tolist()
            hy = (cy + R * sin_a).tolist()
            home_x += hx + [hx[0], None]
            home_y += hy + [hy[0], None]
        elif ctrl < -0.5:
            hx = (cx + R * cos_a).tolist()
            hy = (cy + R * sin_a).tolist()
            away_x += hx + [hx[0], None]
            away_y += hy + [hy[0], None]

    return (home_x, home_y), (away_x, away_y)


# Precomputed for each of the 6 axial neighbour directions (pointy-top hexes):
#   neighbour offset (dxR, dyR) in units of R, and the edge vertex-index pair
# Directions: E, NE, NW, W, SW, SE
_SQRT3 = float(np.sqrt(3))
_NEIGHBOUR_DIR = [                          # (dx/R, dy/R,  vi, vj)
    (_SQRT3,       0.0,   5, 0),           # E
    (_SQRT3 / 2,   1.5,   0, 1),           # NE
    (-_SQRT3 / 2,  1.5,   1, 2),           # NW
    (-_SQRT3,      0.0,   2, 3),           # W
    (-_SQRT3 / 2, -1.5,   3, 4),           # SW
    (_SQRT3 / 2,  -1.5,   4, 5),           # SE
]
_ANGLES = np.radians([30, 90, 150, 210, 270, 330])


def player_zone_outlines(player_ctrl: dict, hex_centers: np.ndarray, R: float) -> dict:
    """
    For each player compute the outer perimeter of their PCZ — only the edges
    that border a hex NOT in their zone (or the pitch boundary).

    Parameters
    ----------
    player_ctrl : dict  jersey → bool mask (N,)
    hex_centers : (N, 2)  array of hex centre coordinates
    R           : hex circumradius

    Returns
    -------
    dict  jersey → (flat_x, flat_y)
        Flat lists with None separators ready for go.Scatter(mode="lines").
    """
    cos_a = np.cos(_ANGLES)
    sin_a = np.sin(_ANGLES)

    result: dict = {}
    for jersey, mask in player_ctrl.items():
        if not np.any(mask):
            continue

        # Build fast membership set: rounded center tuple → True
        controlled = {
            (round(float(cx), 2), round(float(cy), 2))
            for cx, cy in hex_centers[mask]
        }

        outline_x: list = []
        outline_y: list = []

        for cx, cy in hex_centers[mask]:
            # Pre-compute 6 vertex positions for this hex
            vx = cx + R * cos_a   # shape (6,)
            vy = cy + R * sin_a

            for dxr, dyr, vi, vj in _NEIGHBOUR_DIR:
                nkey = (round(float(cx + R * dxr), 2),
                        round(float(cy + R * dyr), 2))
                if nkey not in controlled:
                    outline_x += [float(vx[vi]), float(vx[vj]), None]
                    outline_y += [float(vy[vi]), float(vy[vj]), None]
        if outline_x:
            result[jersey] = (outline_x, outline_y)

    return result


# ---------------------------------------------------------------------------
# Freeze-frame control (StatsBomb / event-moment snapshots)
# ---------------------------------------------------------------------------

def compute_freeze_frame_control(
    freeze_frame: list[dict],
    ball_pos: np.ndarray,
    hex_centers: np.ndarray,
    profile: dict = None,
) -> tuple:
    """
    Compute pitch control from a StatsBomb 360 freeze-frame snapshot.

    This is the velocity = 0 degenerate case of the kinematic PCZ model.
    Without continuous tracking we have no player velocities, so we pass a
    zero velocity vector for every player.  The acceleration / cruise phases
    of ``player_travel_time`` still apply, meaning the reaction-time + first-
    step kinematics are accurately captured — only the momentum head-start is
    absent.

    Parameters
    ----------
    freeze_frame : list of player dicts from statsbomb_io.load_sb_freeze_frames()
        Each dict: {"teammate": bool, "actor": bool, "keeper": bool,
                    "location": [x_plaiq, y_plaiq]}
        The actor is the player who performed the event; teammates are on the
        same side as the actor.
    ball_pos     : (2,) PLAIQ coordinates of the ball at the event moment
    hex_centers  : (N, 2) precomputed hex grid (from build_pitch_hex_grid)
    profile      : physical profile dict (v_max, a_max, reaction_time);
                   defaults to DEFAULT_PROFILE when None

    Returns
    -------
    control          : (N,) float32 — +1 teammate side, -1 opponent side, 0 uncontrolled
    t_ball           : (N,) ball travel times in seconds
    t_team           : (N,) min teammate travel times
    t_opp            : (N,) min opponent travel times
    player_ctrl_team : dict — freeze-frame player index → bool mask of controlled hexes
    player_ctrl_opp  : dict — freeze-frame player index → bool mask of controlled hexes

    Notes
    -----
    "Teammate" = same team as the actor (the player who performed the event).
    The output uses +1 for the actor's team and -1 for the opponent, matching
    the convention of compute_frame_control.
    """
    p = profile if profile is not None else DEFAULT_PROFILE
    N = len(hex_centers)
    INF = np.float32(1e6)
    ZERO_VEL = np.zeros(2, dtype=np.float32)

    t_ball = ball_travel_time(
        np.array(ball_pos, dtype=np.float32), hex_centers
    ).astype(np.float32)

    team_entries: list[tuple[int, np.ndarray]] = []
    opp_entries:  list[tuple[int, np.ndarray]] = []

    for idx, player in enumerate(freeze_frame):
        loc = player.get("location")
        if loc is None:
            continue
        pos = np.array([float(loc[0]), float(loc[1])], dtype=np.float32)
        t = player_travel_time(
            pos, ZERO_VEL, hex_centers,
            p["v_max"], p["a_max"], p["reaction_time"],
        ).astype(np.float32)
        if player.get("teammate", False):
            team_entries.append((idx, t))
        else:
            opp_entries.append((idx, t))

    if not team_entries:
        t_team = np.full(N, INF, dtype=np.float32)
        team_winner = np.zeros(N, dtype=np.int64)
    else:
        team_stack = np.vstack([t for _, t in team_entries])
        t_team = team_stack.min(axis=0)
        team_winner = team_stack.argmin(axis=0)

    if not opp_entries:
        t_opp = np.full(N, INF, dtype=np.float32)
        opp_winner = np.zeros(N, dtype=np.int64)
    else:
        opp_stack = np.vstack([t for _, t in opp_entries])
        t_opp = opp_stack.min(axis=0)
        opp_winner = opp_stack.argmin(axis=0)

    team_reaches = t_team <= t_ball
    opp_reaches  = t_opp  <= t_ball

    control = np.where(
        team_reaches & opp_reaches,
        np.where(t_team <= t_opp, 1.0, -1.0),
        np.where(team_reaches,  1.0,
        np.where(opp_reaches,  -1.0, 0.0)),
    ).astype(np.float32)

    player_ctrl_team = {
        pidx: (control > 0.5) & (team_winner == i)
        for i, (pidx, _) in enumerate(team_entries)
    }
    player_ctrl_opp = {
        pidx: (control < -0.5) & (opp_winner == i)
        for i, (pidx, _) in enumerate(opp_entries)
    }

    return control, t_ball, t_team, t_opp, player_ctrl_team, player_ctrl_opp


# ---------------------------------------------------------------------------
# Value surface visualisation
# ---------------------------------------------------------------------------

def value_to_hex_traces(
    V_effective: np.ndarray,
    hex_centers: np.ndarray,
    R: float,
    base_opacity: float = 0.6,
) -> tuple:
    """
    Convert a per-hex value array into polygon coordinates for a Plotly
    Scatter trace, with each hex coloured on a warm gradient keyed to its
    effective value.

    Hexes with V < 0.05 are skipped (transparent at very low value).

    Colour scale (warm — does not clash with red/blue PCZ):
        low value  → amber  rgba(255, 180, 0, ...)
        high value → gold   rgba(255, 220, 0, ...)
    The fill opacity of each hex is linearly scaled by its value,
    so high-threat hexes are visually prominent and low-threat are subtle.

    Parameters
    ----------
    V_effective  : (N,) float array, values in [0, 1]
    hex_centers  : (N, 2)
    R            : hex circumradius
    base_opacity : maximum fill opacity (0–1); sidebar slider multiplies this

    Returns
    -------
    traces_data : list of (x_list, y_list, rgba_str) tuples
        Each entry is one opacity level bucket of hexes, ready to be
        rendered as a go.Scatter(fill="toself") trace.
        Bucketing by rounded opacity keeps the Plotly trace count low.
    """
    angles = np.radians([30, 90, 150, 210, 270, 330])
    cos_a  = np.cos(angles)
    sin_a  = np.sin(angles)

    # Bucket hexes into 10 opacity bands (avoids one trace per hex)
    BANDS = 10
    buckets: dict = {}   # opacity_key → (x_list, y_list)

    for (cx, cy), v in zip(hex_centers, V_effective):
        if v < 0.05:
            continue
        band   = min(int(v * BANDS), BANDS - 1)
        band_v = (band + 0.5) / BANDS        # midpoint value for the band
        alpha  = round(band_v * base_opacity, 3)

        if band not in buckets:
            buckets[band] = ([], [], alpha)
        bx_list, by_list, _ = buckets[band]

        hx = (cx + R * cos_a).tolist()
        hy = (cy + R * sin_a).tolist()
        bx_list += hx + [hx[0], None]
        by_list += hy + [hy[0], None]

    # Build output list, sorted by band (low → high) so bright hexes render on top
    traces_data = []
    for band in sorted(buckets.keys()):
        bx_list, by_list, alpha = buckets[band]
        # Interpolate colour between amber (low) and gold (high)
        t   = band / (BANDS - 1)
        r_c = 255
        g_c = int(180 + t * 40)   # 180 → 220
        b_c = 0
        rgba = f"rgba({r_c},{g_c},{b_c},{alpha:.3f})"
        traces_data.append((bx_list, by_list, rgba))

    return traces_data

