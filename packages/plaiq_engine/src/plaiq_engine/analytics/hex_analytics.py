#!/usr/bin/env python3
"""
hex_analytics.py

Hexagonal grid spatial analytics for Metrica tracking data.

Core concept
------------
The pitch is divided into a pointy-top hexagonal grid (same orientation as the
visual overlay in app.py) using axial coordinates (q, r):

  - q  : column index, increases to the right
  - r  : row index, increases upward
  - (0, 0) is the centre hex (kick-off circle)

All functions are fully vectorised with NumPy — no Python loops over frames.

Public API
----------
cartesian_to_hex(x, y, R) -> (q, r)
    Convert Cartesian pitch coords to hex axial indices.

hex_to_center(q, r, R) -> (cx, cy)
    Convert axial indices back to Cartesian hex centre.

assign_players_to_hexes(tracking_df, jerseys, team, R) -> DataFrame
    For every frame, record which hex each player occupies.
    Columns: Frame, team, jersey, q, r, cx, cy

hex_occupancy(player_hex_df) -> DataFrame
    Per-frame count of players per hex (Home / Away / total).

hex_density_map(player_hex_df, team, total_frames) -> DataFrame
    Fraction of frames each hex was occupied by at least one player of `team`.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Core coordinate conversion
# ---------------------------------------------------------------------------

_SQRT3      = np.sqrt(3.0)
_SQRT3_OVER3 = _SQRT3 / 3.0


def _hex_round(q_frac: np.ndarray, r_frac: np.ndarray):
    """Round fractional axial coordinates to the nearest hex using cube rounding."""
    s_frac = -q_frac - r_frac
    q = np.round(q_frac)
    r = np.round(r_frac)
    s = np.round(s_frac)

    dq = np.abs(q - q_frac)
    dr = np.abs(r - r_frac)
    ds = np.abs(s - s_frac)

    mask_q = (dq > dr) & (dq > ds)
    mask_r = (~mask_q) & (dr > ds)

    q[mask_q] = -r[mask_q] - s[mask_q]
    r[mask_r] = -q[mask_r] - s[mask_r]

    return q.astype(int), r.astype(int)


def cartesian_to_hex(x: np.ndarray, y: np.ndarray, R: float):
    """
    Convert Cartesian coordinates (x, y) to axial hex indices (q, r).

    Parameters
    ----------
    x, y : array-like  — pitch coordinates in metres (origin = centre circle)
    R    : float       — hex circumradius in metres

    Returns
    -------
    q, r : int arrays  — axial hex indices
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    q_frac = (_SQRT3_OVER3 * x - y / 3.0) / R
    r_frac = (2.0 / 3.0 * y) / R

    return _hex_round(q_frac, r_frac)


def hex_to_center(q: np.ndarray, r: np.ndarray, R: float):
    """
    Convert axial hex indices (q, r) to Cartesian centre coordinates.

    Parameters
    ----------
    q, r : array-like  — axial hex indices
    R    : float       — hex circumradius in metres

    Returns
    -------
    cx, cy : float arrays — Cartesian centres in metres
    """
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    cx = R * _SQRT3 * (q + r / 2.0)
    cy = R * 1.5 * r
    return cx, cy


def hex_id_str(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Return string labels 'q,r' for each (q, r) pair (useful as dict keys / groupby)."""
    q = np.asarray(q, dtype=int)
    r = np.asarray(r, dtype=int)
    return np.array([f"{qi},{ri}" for qi, ri in zip(q, r)])


# ---------------------------------------------------------------------------
# Player-level assignment
# ---------------------------------------------------------------------------

def assign_players_to_hexes(
    tracking_df: pd.DataFrame,
    jerseys: list,
    team: str,
    R: float,
) -> pd.DataFrame:
    """
    For every frame, compute which hex (q, r) each player occupies.

    Parameters
    ----------
    tracking_df : DataFrame  — processed tracking data (index = Frame)
    jerseys     : list[str]  — jersey numbers e.g. ['1','2',...]
    team        : str        — 'Home' or 'Away'
    R           : float      — hex circumradius in metres

    Returns
    -------
    DataFrame with columns:
        Frame, team, jersey, x, y, q, r, hex_id, cx, cy
    Only includes rows where the player has a valid (non-NaN) position.
    """
    parts = []

    for jersey in jerseys:
        x_col = f"{team}_{jersey}_x"
        y_col = f"{team}_{jersey}_y"
        if x_col not in tracking_df.columns:
            continue

        x = tracking_df[x_col].values
        y = tracking_df[y_col].values
        frames = tracking_df.index.values

        valid = ~(np.isnan(x) | np.isnan(y))
        if not valid.any():
            continue

        xv, yv, fv = x[valid], y[valid], frames[valid]
        q, r = cartesian_to_hex(xv, yv, R)
        cx, cy = hex_to_center(q, r, R)

        parts.append(pd.DataFrame({
            "Frame":  fv,
            "team":   team,
            "jersey": jersey,
            "x":      xv,
            "y":      yv,
            "q":      q,
            "r":      r,
            "hex_id": hex_id_str(q, r),
            "cx":     cx,
            "cy":     cy,
        }))

    if not parts:
        return pd.DataFrame(columns=["Frame","team","jersey","x","y","q","r","hex_id","cx","cy"])

    return pd.concat(parts, ignore_index=True).sort_values("Frame")


def assign_all_players(
    home_df: pd.DataFrame,
    away_df: pd.DataFrame,
    home_jerseys: list,
    away_jerseys: list,
    R: float,
) -> pd.DataFrame:
    """
    Assign both teams' players to hexes for every frame.
    Returns a single DataFrame combining home and away results.
    """
    home_hex = assign_players_to_hexes(home_df, home_jerseys, "Home", R)
    away_hex = assign_players_to_hexes(away_df, away_jerseys, "Away", R)
    return pd.concat([home_hex, away_hex], ignore_index=True).sort_values("Frame")


# ---------------------------------------------------------------------------
# Aggregate analytics
# ---------------------------------------------------------------------------

def hex_occupancy(player_hex_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-frame, per-hex count of players by team.

    Returns DataFrame with columns:
        Frame, hex_id, q, r, cx, cy, Home, Away, Total
    """
    grp = (
        player_hex_df
        .groupby(["Frame", "hex_id", "q", "r", "cx", "cy", "team"])
        .size()
        .reset_index(name="count")
    )
    pivot = grp.pivot_table(
        index=["Frame", "hex_id", "q", "r", "cx", "cy"],
        columns="team",
        values="count",
        fill_value=0,
    ).reset_index()
    pivot.columns.name = None
    for col in ["Home", "Away"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["Total"] = pivot["Home"] + pivot["Away"]
    return pivot


def hex_density_map(
    player_hex_df: pd.DataFrame,
    team: str,
    total_frames: int,
) -> pd.DataFrame:
    """
    For each hex, compute the fraction of frames it was occupied by
    at least one player of `team`.

    Returns DataFrame with columns: hex_id, q, r, cx, cy, frames_occupied, density
    """
    team_df = player_hex_df[player_hex_df["team"] == team]
    occupied = (
        team_df.groupby(["hex_id", "q", "r", "cx", "cy"])["Frame"]
        .nunique()
        .reset_index(name="frames_occupied")
    )
    occupied["density"] = occupied["frames_occupied"] / total_frames
    return occupied.sort_values("density", ascending=False)


def players_in_hex_at_frame(
    player_hex_df: pd.DataFrame,
    frame: int,
    q: int,
    r: int,
) -> pd.DataFrame:
    """
    Return all players located in hex (q, r) at the given frame.
    """
    return player_hex_df[
        (player_hex_df["Frame"] == frame) &
        (player_hex_df["q"] == q) &
        (player_hex_df["r"] == r)
    ][["team", "jersey", "x", "y"]]


# ---------------------------------------------------------------------------
# Pressure calculation
# ---------------------------------------------------------------------------

def compute_pressure(
    player_hex_df: pd.DataFrame,
    chunk_size: int = 10_000,
) -> pd.DataFrame:
    """
    For every player at every frame, compute two pressure metrics:

    individual_pressure (existing)
    --------------------------------
    Score 1-10 based on the single nearest opposing player.
        dist ≤ 1  → 10,  dist 2 → 9, … dist ≥ 10 → 1

    zonal_pressure (new)
    --------------------------------
    Mean pressure score of the 3 closest opposing players, each scored
    on the same 1-10 scale by their hex distance to this player.
    Always uses exactly 3 opponents (or however many are on-pitch if < 3).
    Output is 1-10, directly comparable to individual_pressure.

    Columns returned: Frame, team, jersey, min_hex_dist, pressure,
                      zonal_pressure
    """
    home_df = player_hex_df[player_hex_df["team"] == "Home"]
    away_df = player_hex_df[player_hex_df["team"] == "Away"]

    def _pivot(df, col):
        return df.pivot_table(index="Frame", columns="jersey",
                              values=col, aggfunc="first")

    home_q = _pivot(home_df, "q")
    home_r = _pivot(home_df, "r")
    away_q = _pivot(away_df, "q")
    away_r = _pivot(away_df, "r")

    home_jerseys = home_q.columns.tolist()
    away_jerseys = away_q.columns.tolist()

    all_frames = home_q.index.union(away_q.index)
    home_q = home_q.reindex(all_frames)
    home_r = home_r.reindex(all_frames)
    away_q = away_q.reindex(all_frames)
    away_r = away_r.reindex(all_frames)

    frames_arr = all_frames.values
    n_frames   = len(frames_arr)
    SENTINEL   = np.int32(9_999)
    N_TRIANGLE = 3   # pressure triangle — always use 3 closest opponents

    parts = []

    for start in range(0, n_frames, chunk_size):
        end   = min(start + chunk_size, n_frames)
        cf    = frames_arr[start:end]

        hq = home_q.iloc[start:end].values
        hr = home_r.iloc[start:end].values
        aq = away_q.iloc[start:end].values
        ar = away_r.iloc[start:end].values

        home_on = ~np.isnan(hq)
        away_on = ~np.isnan(aq)

        hq = np.where(home_on, hq, SENTINEL).astype(np.int32)
        hr = np.where(home_on, hr, SENTINEL).astype(np.int32)
        aq = np.where(away_on, aq, SENTINEL).astype(np.int32)
        ar = np.where(away_on, ar, SENTINEL).astype(np.int32)

        hs  = hq + hr
        as_ = aq + ar

        # Pairwise hex distances — (C, H, A)
        dq   = np.abs(hq[:, :, None] - aq[:, None, :])
        dr   = np.abs(hr[:, :, None] - ar[:, None, :])
        ds   = np.abs(hs[:, :, None] - as_[:, None, :])
        dist = (dq + dr + ds) // 2

        def _pressure_score(md):
            """1-10 score from hex distance. NaN preserved."""
            return np.where(np.isnan(md), np.nan,
                   np.maximum(1, 11 - np.maximum(1, md)))

        def _zonal(dist_masked):
            """
            dist_masked : (C, players, opponents) with off-pitch → SENTINEL
            Returns (C, players) zonal pressure — mean of N_TRIANGLE closest.
            """
            C, P, A = dist_masked.shape
            # Sort along opponent axis to get N_TRIANGLE closest
            n_use   = min(N_TRIANGLE, A)
            # partial sort — only need first n_use
            sorted_d = np.partition(dist_masked, n_use - 1, axis=2)[:, :, :n_use]  # (C,P,n_use)
            # Replace SENTINEL with NaN for mean calculation
            valid_d  = np.where(sorted_d >= SENTINEL, np.nan, sorted_d.astype(float))
            # Score each of the n_use distances
            scored   = _pressure_score(valid_d)           # (C, P, n_use)
            # Nanmean across opponents — works even if some are off-pitch
            with np.errstate(all="ignore"):
                zonal = np.nanmean(scored, axis=2)        # (C, P)
            return zonal

        # --- Home players: pressure from away opponents ---
        dist_ha    = np.where(~away_on[:, None, :], SENTINEL, dist)   # (C, H, A)
        min_dist_h = dist_ha.min(axis=2).astype(float)
        min_dist_h[~home_on] = np.nan
        press_h    = _pressure_score(min_dist_h)
        zonal_h    = _zonal(dist_ha)
        zonal_h[~home_on] = np.nan

        # --- Away players: pressure from home opponents ---
        dist_ah    = np.where(~home_on[:, :, None], SENTINEL, dist)   # (C, H, A)
        # Transpose to (C, A, H) for consistent (C, players, opponents) shape
        dist_ah_t  = np.transpose(dist_ah, (0, 2, 1))
        min_dist_a = dist_ah_t.min(axis=2).astype(float)
        min_dist_a[~away_on] = np.nan
        press_a    = _pressure_score(min_dist_a)
        zonal_a    = _zonal(dist_ah_t)
        zonal_a[~away_on] = np.nan

        # Flatten to rows
        for i, jersey in enumerate(home_jerseys):
            valid = home_on[:, i]
            if not valid.any():
                continue
            parts.append(pd.DataFrame({
                "Frame":          cf[valid],
                "team":           "Home",
                "jersey":         jersey,
                "min_hex_dist":   min_dist_h[valid, i].astype(np.int16),
                "pressure":       press_h[valid, i].astype(np.int8),
                "zonal_pressure": np.round(zonal_h[valid, i], 2),
            }))

        for i, jersey in enumerate(away_jerseys):
            valid = away_on[:, i]
            if not valid.any():
                continue
            parts.append(pd.DataFrame({
                "Frame":          cf[valid],
                "team":           "Away",
                "jersey":         jersey,
                "min_hex_dist":   min_dist_a[valid, i].astype(np.int16),
                "pressure":       press_a[valid, i].astype(np.int8),
                "zonal_pressure": np.round(zonal_a[valid, i], 2),
            }))

    if not parts:
        return pd.DataFrame(columns=["Frame", "team", "jersey",
                                     "min_hex_dist", "pressure",
                                     "zonal_pressure"])

    return (pd.concat(parts, ignore_index=True)
              .sort_values(["Frame", "team", "jersey"])
              .reset_index(drop=True))


def pressure_summary(pressure_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player summary statistics for pressure scores across all frames.

    Returns DataFrame with columns:
        team, jersey, mean_pressure, max_pressure, pct_high_pressure
        (pct_high_pressure = fraction of frames with pressure ≥ 7)
    """
    grp = pressure_df.groupby(["team", "jersey"])["pressure"]
    summary = pd.DataFrame({
        "mean_pressure":     grp.mean().round(2),
        "max_pressure":      grp.max(),
        "pct_high_pressure": (pressure_df.groupby(["team", "jersey"])["pressure"]
                              .apply(lambda s: (s >= 7).mean()).round(3)),
    }).reset_index()
    return summary.sort_values(["team", "mean_pressure"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Possession inference (Option B — event-anchored carry-forward)
# ---------------------------------------------------------------------------

# Events that definitively assign possession to the tagged team
_POSSESSION_GAIN = {"PASS", "SHOT", "SET PIECE", "RECOVERY"}

# Events that definitively end possession for the tagged team
_POSSESSION_LOSS = {"BALL LOST", "BALL OUT"}


def infer_possession(events_df: pd.DataFrame, total_frames: int) -> pd.Series:
    """
    Infer frame-by-frame ball possession using an event-anchored carry-forward model.

    Logic:
      - A PASS / SHOT / SET PIECE / RECOVERY event sets possession to that team
        from its Start Frame.
      - A BALL LOST / BALL OUT event clears possession for that team from its
        Start Frame (unknown until next gain event).
      - Between events, the last known state is carried forward (ffill).
      - Frames before the first possession event are filled backward (bfill),
        typically assigning the kick-off team.

    Parameters
    ----------
    events_df    : processed events DataFrame (output of read_event_data / preprocess)
    total_frames : total number of tracking frames in the match

    Returns
    -------
    pd.Series, index = frame number (1-based), values = 'Home' | 'Away' | None
    """
    # Build a sparse frame→possession mapping from anchor events
    anchors: dict = {}

    for _, row in events_df.sort_values("Start Frame").iterrows():
        frame = int(row["Start Frame"])
        team  = row["Team"]
        etype = row["Type"]

        if etype in _POSSESSION_GAIN:
            anchors[frame] = team
        elif etype in _POSSESSION_LOSS:
            # Team loses possession — mark as unknown
            anchors[frame] = None

    if not anchors:
        return pd.Series([None] * total_frames,
                         index=range(1, total_frames + 1))

    # Build full-length series, fill only anchor frames, then ffill + bfill
    possession = pd.Series(index=range(1, total_frames + 1), dtype=object)
    for frame, team in anchors.items():
        if 1 <= frame <= total_frames:
            possession.iloc[frame - 1] = team

    possession = possession.ffill().bfill()
    return possession


def tag_possession(
    pressure_df: pd.DataFrame,
    possession: pd.Series,
) -> pd.DataFrame:
    """
    Merge possession state into the pressure DataFrame and derive:

      - possession        : 'Home' | 'Away' — who has the ball this frame
      - in_possession     : bool — is this player's team in possession?
      - applied_pressure  : pressure score when player is OUT of possession
                            (pressing / defending — applying pressure to the ball carrier)
      - received_pressure : pressure score when player IS in possession
                            (being pressed — the ball carrier's experience)

    Parameters
    ----------
    pressure_df : output of compute_pressure()
    possession  : output of infer_possession()

    Returns
    -------
    pressure_df with additional columns:
        possession, in_possession, applied_pressure, received_pressure
    """
    df = pressure_df.copy()

    df["possession"] = df["Frame"].map(possession)
    df["in_possession"] = df["team"] == df["possession"]

    df["applied_pressure"]  = df["pressure"].where(~df["in_possession"])
    df["received_pressure"] = df["pressure"].where( df["in_possession"])

    return df


def possession_pressure_summary(tagged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player summary split by possession state.

    Returns DataFrame with columns:
        team, jersey,
        mean_applied_pressure, mean_received_pressure,
        pct_high_applied, pct_high_received
    """
    grp = tagged_df.groupby(["team", "jersey"])

    summary = pd.DataFrame({
        "mean_applied_pressure":  grp["applied_pressure"].mean().round(2),
        "mean_received_pressure": grp["received_pressure"].mean().round(2),
        "pct_high_applied":  (tagged_df.groupby(["team", "jersey"])["applied_pressure"]
                              .apply(lambda s: (s >= 7).mean()).round(3)),
        "pct_high_received": (tagged_df.groupby(["team", "jersey"])["received_pressure"]
                              .apply(lambda s: (s >= 7).mean()).round(3)),
    }).reset_index()

    return summary.sort_values(["team", "mean_received_pressure"],
                               ascending=[True, False])


