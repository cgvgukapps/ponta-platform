#!/usr/bin/env python3
"""
hex_value.py — Three-component spatial hex value model.

Components
----------
1. V_xT   (static)  — positional threat: geometric Expected Threat surface
                       based on angle to goal and distance. Computed once per
                       game, never changes within a match.

2. V_control (dynamic) — PCZ control weight: smooth sigmoid of the time
                          advantage the possessing team has to each hex.
                          Uses output of compute_frame_control().

3. V_pressure (dynamic) — pressure discount: for each hex, find the nearest
                           player of the possessing team and discount by their
                           current pressure score (max 30% discount).

Combined formula
----------------
    V_effective(hex, frame) =
        V_xT(hex)
        × V_control_team(hex, frame)
        × (1 − V_pressure_discount(hex, frame))

Event ΔV
--------
For each event: ΔV = V_xT(end_hex) − V_xT(start_hex) from the possessing
team's perspective.  Uses only the static xT surface so values are
comparable across frames and games.

Coordinate convention
---------------------
In data space (after preprocessing normalisation), the HOME team always
attacks toward positive x (goal at x = +53 m).  The AWAY team attacks
toward x = −53 m.  V_xT_home and V_xT_away are computed accordingly.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from matrix_engine.pitch_config import FIELD_LENGTH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOAL_X      = FIELD_LENGTH / 2          # 53.0 m
HALF_POST   = 3.66                       # half goal-mouth width (m)

# xT shot model parameters
_DISTANCE_DECAY = 0.015   # per metre — calibrated so 20 m centre ≈ 0.15, 6 m ≈ 0.60
# Progression bonus parameters
_PROG_SCALE     = 0.08    # sigmoid steepness (per metre)
_PROG_SHIFT     = 20.0    # sigmoid centre offset — adds value from halfway line forward
_PROG_MAX       = 0.3     # cap on progression bonus (relative to normalised xT)

# Pressure discount: max 30 % reduction at pressure = 10
_PRESSURE_MAX_DISCOUNT = 0.3

# Smooth control sigmoid steepness (seconds^-1)
# At k=2.0, a 1-second time advantage → sigmoid ≈ 0.88 (strongly controlled)
_CTRL_SIGMOID_K = 2.0


# ---------------------------------------------------------------------------
# Component 1 — Static xT surface
# ---------------------------------------------------------------------------

def xT_surface(hex_centers: np.ndarray, attacking_right: bool = True) -> np.ndarray:
    """
    Compute the static Expected Threat value for every hex centre.

    Uses a transparent geometric model:
      1. Shot angle subtended by the goal mouth (larger = more dangerous)
      2. Exponential distance decay on the shot value
      3. Sigmoid forward-progress bonus (increases toward opponent's goal)
      4. Normalised to [0, 1] across the full pitch hex grid

    Parameters
    ----------
    hex_centers   : (N, 2) array of hex centre coordinates in metres
    attacking_right : True → goal at x = +53 (home team convention)
                      False → goal at x = −53 (away team, flip x)

    Returns
    -------
    V_xT : (N,) float32 array, values in [0, 1]
    """
    cx = hex_centers[:, 0].astype(float)
    cy = hex_centers[:, 1].astype(float)

    # Flip x for the away team (attacking toward negative x)
    if not attacking_right:
        cx = -cx

    # 1. Goal-mouth angle — angle subtended by the two posts from each hex
    #    atan2 handles all quadrants correctly (including behind-goal)
    theta1 = np.arctan2( HALF_POST - cy, GOAL_X - cx)
    theta2 = np.arctan2(-HALF_POST - cy, GOAL_X - cx)
    angle  = np.abs(theta1 - theta2)   # always positive, in radians [0, π]

    # 2. Distance to goal centre
    d_goal = np.sqrt((cx - GOAL_X) ** 2 + cy ** 2)

    # 3. Shot value: angle proportional × distance decay
    #    Zero for hexes behind or exactly on the goal line (cx >= GOAL_X)
    shot_v = (angle / np.pi) * np.exp(-_DISTANCE_DECAY * d_goal)
    shot_v = np.where(cx >= GOAL_X, 0.0, shot_v)

    # 4. Progression bonus: sigmoid that increases monotonically toward the goal
    #    Own-half hexes get a small but non-zero bonus (ball progression matters)
    prog_v = _PROG_MAX / (1.0 + np.exp(-_PROG_SCALE * (cx + _PROG_SHIFT)))

    # 5. Combine and normalise to [0, 1]
    V_raw = shot_v + prog_v
    max_v = V_raw.max()
    if max_v > 0:
        V_raw = V_raw / max_v

    return V_raw.astype(np.float32)


# ---------------------------------------------------------------------------
# Component 2 — Smooth control weight
# ---------------------------------------------------------------------------

def smooth_control(
    t_home: np.ndarray,
    t_away: np.ndarray,
    t_ball: np.ndarray,
) -> tuple:
    """
    Convert PCZ travel times into smooth control weights in [0, 1].

    Uses a sigmoid on the time gap rather than the binary +1/0/−1 control
    array, giving a graduated surface.

    Parameters
    ----------
    t_home, t_away : (N,) arrays — min player travel times per hex
    t_ball         : (N,) array  — ball travel times per hex
                     Pass None to fall back to 0.5 (contested) everywhere.

    Returns
    -------
    V_ctrl_home : (N,) float32 — home control weight per hex
    V_ctrl_away : (N,) float32 — away control weight per hex
        (V_ctrl_away = 1 − V_ctrl_home)
    """
    if t_home is None or t_away is None:
        half = np.full(1, 0.5, dtype=np.float32)
        return half, half

    t_h = np.asarray(t_home, dtype=np.float64)
    t_a = np.asarray(t_away, dtype=np.float64)

    # Time gap: positive → home is faster (home-favoured)
    time_gap = t_a - t_h

    # Clip to avoid exp overflow
    gap_clipped = np.clip(time_gap * _CTRL_SIGMOID_K, -10.0, 10.0)
    V_ctrl_home = (1.0 / (1.0 + np.exp(-gap_clipped))).astype(np.float32)
    V_ctrl_away = 1.0 - V_ctrl_home

    return V_ctrl_home, V_ctrl_away


# ---------------------------------------------------------------------------
# Component 3 — Pressure discount field
# ---------------------------------------------------------------------------

def pressure_field(
    row_h: pd.Series,
    row_a: pd.Series,
    home_jerseys: list,
    away_jerseys: list,
    hex_centers: np.ndarray,
    frame_pressure: pd.DataFrame,
    possession: str,
) -> np.ndarray:
    """
    For every hex, compute a pressure discount for the possessing team.

    For each hex we find the nearest player of the team in possession and
    use that player's current pressure score.

    Parameters
    ----------
    row_h, row_a       : tracking rows for the current frame
    home_jerseys       : list of jersey strings for home
    away_jerseys       : list of jersey strings for away
    hex_centers        : (N, 2)
    frame_pressure     : slice of analytics DataFrame for the current frame
                         columns: team, jersey, pressure
    possession         : 'Home' | 'Away'

    Returns
    -------
    discount : (N,) float32 — values in [0, _PRESSURE_MAX_DISCOUNT]
    """
    N = len(hex_centers)
    discount = np.zeros(N, dtype=np.float32)

    if frame_pressure is None or frame_pressure.empty:
        return discount

    # Collect positions + pressure of possessing team players
    poss_positions: list = []
    poss_pressures: list = []

    if possession == "Home":
        for jersey in home_jerseys:
            px = row_h.get(f"Home_{jersey}_x")
            py = row_h.get(f"Home_{jersey}_y")
            if pd.isna(px) or pd.isna(py):
                continue
            row_p = frame_pressure[
                (frame_pressure["team"] == "Home") &
                (frame_pressure["jersey"] == jersey)
            ]
            press = float(row_p["pressure"].iloc[0]) if not row_p.empty else 5.0
            poss_positions.append([float(px), float(py)])
            poss_pressures.append(press)
    else:  # Away
        for jersey in away_jerseys:
            px = row_a.get(f"Away_{jersey}_x")
            py = row_a.get(f"Away_{jersey}_y")
            if pd.isna(px) or pd.isna(py):
                continue
            row_p = frame_pressure[
                (frame_pressure["team"] == "Away") &
                (frame_pressure["jersey"] == jersey)
            ]
            press = float(row_p["pressure"].iloc[0]) if not row_p.empty else 5.0
            poss_positions.append([float(px), float(py)])
            poss_pressures.append(press)

    if not poss_positions:
        return discount

    poss_arr  = np.array(poss_positions, dtype=np.float32)
    press_arr = np.array(poss_pressures, dtype=np.float32)

    # For each hex, nearest possessing player
    tree = cKDTree(poss_arr)
    _, idx = tree.query(hex_centers)

    nearest_press = press_arr[idx]                            # (N,)
    discount = np.clip(
        nearest_press / 10.0 * _PRESSURE_MAX_DISCOUNT,
        0.0, _PRESSURE_MAX_DISCOUNT,
    ).astype(np.float32)

    return discount


# ---------------------------------------------------------------------------
# Combined value surface
# ---------------------------------------------------------------------------

def compute_value(
    xT: np.ndarray,
    v_control: np.ndarray,
    v_pressure_discount: np.ndarray,
) -> np.ndarray:
    """
    Combine the three components into a single effective value surface.

    V_effective = V_xT × V_control × (1 − V_pressure_discount)

    All inputs must be (N,) arrays aligned to the same hex grid.

    Returns
    -------
    V_effective : (N,) float32, values in [0, 1]
    """
    V = xT * v_control * (1.0 - v_pressure_discount)
    return V.astype(np.float32)


# ---------------------------------------------------------------------------
# Event ΔV tagging
# ---------------------------------------------------------------------------

def tag_events_delta_v(
    events_df: pd.DataFrame,
    xT_home: np.ndarray,
    xT_away: np.ndarray,
    hex_centers: np.ndarray,
    R: float,
) -> pd.DataFrame:
    """
    Add a `delta_v` column to the events DataFrame.

    ΔV = V_xT(end_hex) − V_xT(start_hex), from the possessing team's
    perspective.  Uses only the static xT surface so values are comparable
    across frames.

    Events with missing start/end coordinates get delta_v = NaN.

    Parameters
    ----------
    events_df   : standard events DataFrame (has Start X/Y, End X/Y, Team)
    xT_home     : (N,) xT surface for home team
    xT_away     : (N,) xT surface for away team
    hex_centers : (N, 2) pitch hex centres
    R           : hex circumradius used for cartesian_to_hex conversion

    Returns
    -------
    events_df copy with added column `delta_v` (float32)
    """
    df = events_df.copy()

    # Build a fast hex-center lookup using nearest-neighbour
    tree = cKDTree(hex_centers)

    def _lookup_xT(x_col, y_col, xT_h, xT_a, team_col):
        """Vectorised xT lookup for a coordinate column pair."""
        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        team = df[team_col].to_numpy()

        result = np.full(len(df), np.nan, dtype=np.float32)

        valid = ~(np.isnan(x) | np.isnan(y))
        if not valid.any():
            return result

        pts = np.stack([x[valid], y[valid]], axis=1)
        _, idx = tree.query(pts)

        # Select xT surface per event based on team
        is_home = team[valid] == "Home"
        xt_home_vals = xT_h[idx]
        xt_away_vals = xT_a[idx]
        result[valid] = np.where(is_home, xt_home_vals, xt_away_vals)
        return result

    xT_start = _lookup_xT("Start X", "Start Y", xT_home, xT_away, "Team")
    xT_end   = _lookup_xT("End X",   "End Y",   xT_home, xT_away, "Team")

    df["delta_v"] = (xT_end - xT_start).round(4)
    return df


# ---------------------------------------------------------------------------
# Movement Value ΔV — per-player, per-frame positional threat change
# ---------------------------------------------------------------------------

def compute_movement_delta_v(
    analytics_df: pd.DataFrame,
    xT_home: np.ndarray,
    xT_away: np.ndarray,
    hex_centers: np.ndarray,
    R: float,
) -> pd.DataFrame:
    """
    Add `xT_pos` and `movement_dv` columns to the analytics DataFrame.

    `xT_pos`      — the xT value of the hex each player is currently occupying.

    `movement_dv` — frame-over-frame change in `xT_pos` for each player:
                   `xT_pos[frame] − xT_pos[frame − 1]`

        Positive → player moved into a more threatening position.
        Negative → player dropped into lower-threat territory.
        NaN      → first frame for that player (no previous position).

    Parameters
    ----------
    analytics_df : output of `load_analytics()` — must have columns
                   Frame, team, jersey, q, r
    xT_home, xT_away : (N,) xT surfaces
    hex_centers : (N, 2)
    R           : hex circumradius

    Returns
    -------
    Copy of analytics_df with two additional columns.
    """
    from plaiq_engine.analytics.hex_analytics import hex_to_center
    from scipy.spatial import cKDTree

    df = analytics_df.copy()
    tree = cKDTree(hex_centers)

    # Vectorised xT lookup for all rows
    try:
        cx, cy = hex_to_center(
            df["q"].to_numpy(dtype=float),
            df["r"].to_numpy(dtype=float),
            R,
        )
        pts = np.stack([cx, cy], axis=1)
        _, idx = tree.query(pts)

        is_home = (df["team"] == "Home").to_numpy()
        xT_vals = np.where(is_home, xT_home[idx], xT_away[idx])
        df["xT_pos"] = xT_vals.astype(np.float32)
    except Exception:
        df["xT_pos"] = np.nan

    # Frame-over-frame diff per player — preserves time ordering
    df = df.sort_values(["team", "jersey", "Frame"])
    df["movement_dv"] = (
        df.groupby(["team", "jersey"])["xT_pos"]
        .diff()
        .round(4)
    )

    return df


# ---------------------------------------------------------------------------
# Movement Value — per-player, per-event
# ---------------------------------------------------------------------------

_TRACKING_FPS     = 25
_PRESSURE_MAX     = 0.3   # max availability discount from pressure


def compute_movement_value(
    analytics_value_df: pd.DataFrame,
    events_df: pd.DataFrame,
    xT_home: np.ndarray,
    xT_away: np.ndarray,
    hex_centers: np.ndarray,
    R: float,
) -> pd.DataFrame:
    """
    Compute movement_value for every (event, player) pair.

    Formula
    -------
    movement_value = xT_end × (1 + run_bonus)

    Where:
        xT_end    = static xT at player's position at event_end_frame
                    (or event_start_frame when no end frame exists)
        run_bonus = clip(ΔxT / (xT_end + 0.05), 0, 1.0)
        ΔxT       = xT_pos(event_end) − xT_pos(event_start)
                    movement quality within the event window

    All values are event-window scoped so they can be summed across a
    play chain without overlap or double-counting.
    """
    from plaiq_engine.analytics.hex_analytics import hex_to_center
    from scipy.spatial import cKDTree

    tree = cKDTree(hex_centers)

    # Pre-index analytics by frame for fast lookups
    adf = analytics_value_df.copy()
    adf_indexed = adf.set_index(["Frame", "team", "jersey"])

    def _get_xT(frame: int, team: str, jersey: str, xT_h, xT_a):
        """xT at a player's position for a given frame."""
        key = (frame, team, str(jersey))
        if key not in adf_indexed.index:
            return np.nan
        row = adf_indexed.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        # Use precomputed xT_pos if available
        if "xT_pos" in row.index and pd.notna(row["xT_pos"]):
            return float(row["xT_pos"])
        # Fallback: compute from hex
        try:
            cx, cy = hex_to_center(
                np.array([float(row["q"])]),
                np.array([float(row["r"])]),
                R,
            )
            _, idx = tree.query([[float(cx[0]), float(cy[0])]])
            xT_arr = xT_h if team == "Home" else xT_a
            return float(xT_arr[idx[0]])
        except Exception:
            return np.nan

    def _get_pressure(frame: int, team: str, jersey: str):
        key = (frame, team, str(jersey))
        if key not in adf_indexed.index:
            return 5.0
        row = adf_indexed.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return float(row.get("pressure", 5.0)) if pd.notna(row.get("pressure")) else 5.0

    # All unique (team, jersey) combinations across the analytics
    all_players = adf[["team", "jersey"]].drop_duplicates().values.tolist()

    rows = []
    for evt_idx, evt in events_df.iterrows():
        sf_raw = evt.get("Start Frame")
        if pd.isna(sf_raw):
            continue
        start_frame = int(sf_raw)
        ef_raw = evt.get("End Frame")
        end_frame = int(ef_raw) if pd.notna(ef_raw) else start_frame

        for team, jersey in all_players:
            jersey_str = str(jersey)

            # xT at event END (arrival quality — where they are when action completes)
            xT_end = _get_xT(end_frame, team, jersey_str, xT_home, xT_away)
            if pd.isna(xT_end):
                continue

            # xT at event START (departure quality)
            xT_start = _get_xT(start_frame, team, jersey_str, xT_home, xT_away)
            if pd.isna(xT_start):
                delta_xT = 0.0
            else:
                delta_xT = float(xT_end - xT_start)

            # Run bonus: how much of the end position was earned by movement
            run_bonus = float(np.clip(delta_xT / (xT_end + 0.05), 0.0, 1.0))

            # Pressure at start and end frame (reported separately, not applied as discount)
            pressure_start = _get_pressure(start_frame, team, jersey_str)
            pressure_end   = _get_pressure(end_frame,   team, jersey_str)

            v_movement = round(xT_end * (1.0 + run_bonus), 4)

            rows.append({
                "event_idx":      evt_idx,
                "team":           team,
                "jersey":         jersey_str,
                "xT_end":         round(xT_end, 4),
                "delta_xT_run":   round(delta_xT, 4),
                "run_bonus":      round(run_bonus, 4),
                "pressure_start": round(pressure_start, 1),
                "pressure":       round(pressure_end, 1),
                "movement_value": v_movement,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["event_idx","team","jersey","xT_end","delta_xT_run",
                 "run_bonus","pressure_start","pressure","movement_value"]
    )


