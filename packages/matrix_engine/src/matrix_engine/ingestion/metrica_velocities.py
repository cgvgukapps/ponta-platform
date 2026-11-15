#!/usr/bin/env python3
"""
metrica_velocities.py

Calculate smoothed player velocities from Metrica tracking data.
Uses a Savitzky-Golay filter by default, with a moving-average option.
Outlier speeds (above MAX_SPEED) are set to NaN before smoothing.
Each half is processed independently to avoid artefacts at half-time.
"""

import numpy as np
import pandas as pd
from scipy import signal

MAX_SPEED = 12.0  # m/s — realistic upper bound for a footballer


def calc_player_velocities(
    team: pd.DataFrame,
    smoothing: bool = True,
    filter_: str = "Savitzky-Golay",
    window: int = 7,
    polyorder: int = 1,
    maxspeed: float = MAX_SPEED,
    fps: float = 25.0,
) -> pd.DataFrame:
    """
    Add velocity columns to a tracking DataFrame.

    For each player, three columns are added:
        <player>_vx      — velocity in the x direction (m/s)
        <player>_vy      — velocity in the y direction (m/s)
        <player>_speed   — total speed (m/s)

    Parameters
    ----------
    team        : tracking DataFrame (output of metrica_io.read_tracking_data)
    smoothing   : whether to smooth velocities (default True)
    filter_     : 'Savitzky-Golay' or 'moving average'
    window      : smoothing window in frames (must be odd for Savitzky-Golay).
                  When fps != 25 and window == 7 (the unchanged default), the
                  window is auto-scaled to preserve the same ~0.28 s smoothing
                  duration: window = max(3, round(7 * fps / 25)) rounded to odd.
    polyorder   : polynomial order for Savitzky-Golay (1 = linear)
    maxspeed    : raw velocities above this are treated as tracking errors and set to NaN
    fps         : tracking frame rate in Hz (default 25). Used only to auto-scale
                  the smoothing window when fps != 25.
    """
    # Auto-scale the smoothing window for non-standard frame rates so the
    # smoothing duration (≈ 0.28 s at 25 fps, window=7) stays comparable.
    if fps != 25.0 and window == 7:
        raw_w = round(7 * fps / 25.0)
        # Savitzky-Golay requires an odd window length ≥ polyorder + 2
        window = max(polyorder + 2, raw_w)
        if window % 2 == 0:
            window += 1
    team = remove_player_velocities(team)

    # Identify unique player prefixes (e.g. "Home_1", "Away_15")
    player_ids = np.unique(
        [c.rsplit("_", 1)[0] for c in team.columns if c.endswith("_x") and "ball" not in c]
    )

    dt = team["Time [s]"].diff()
    has_second_half = (team.Period == 2).any()
    second_half_idx = (
        team.loc[team.Period == 2].index[0] if has_second_half else None
    )

    for player in player_ids:
        vx = team[f"{player}_x"].diff() / dt
        vy = team[f"{player}_y"].diff() / dt

        if maxspeed > 0:
            raw_speed = np.sqrt(vx**2 + vy**2)
            vx[raw_speed > maxspeed] = np.nan
            vy[raw_speed > maxspeed] = np.nan

        if smoothing:
            vx, vy = _smooth_velocities(
                vx, vy, second_half_idx, filter_, window, polyorder
            )

        team[f"{player}_vx"] = vx
        team[f"{player}_vy"] = vy
        team[f"{player}_speed"] = np.sqrt(vx**2 + vy**2)

    return team


def remove_player_velocities(team: pd.DataFrame) -> pd.DataFrame:
    """Drop any existing velocity/acceleration columns from the DataFrame."""
    drop_cols = [
        c for c in team.columns
        if c.split("_")[-1] in {"vx", "vy", "ax", "ay", "speed", "acceleration"}
    ]
    return team.drop(columns=drop_cols)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _smooth_velocities(
    vx: pd.Series,
    vy: pd.Series,
    second_half_idx: int | None,
    filter_: str,
    window: int,
    polyorder: int,
) -> tuple:
    """Apply smoothing to vx and vy, treating each half independently."""
    if second_half_idx is None:
        if filter_ == "Savitzky-Golay":
            vx = pd.Series(_savgol_nan_safe(vx, window, polyorder), index=vx.index)
            vy = pd.Series(_savgol_nan_safe(vy, window, polyorder), index=vy.index)
        elif filter_ == "moving average":
            kernel = np.ones(window) / window
            for series in (vx, vy):
                nan_mask = series.isna()
                filled = series.interpolate(method="linear", limit_direction="both").fillna(0)
                smoothed = np.convolve(filled.values, kernel, mode="same")
                smoothed[nan_mask.values] = np.nan
                series[:] = smoothed
        else:
            raise ValueError(
                f"Unknown filter '{filter_}'. Choose 'Savitzky-Golay' or 'moving average'."
            )
        return vx, vy

    if filter_ == "Savitzky-Golay":
        vx = _savgol_both_halves(vx, second_half_idx, window, polyorder)
        vy = _savgol_both_halves(vy, second_half_idx, window, polyorder)
    elif filter_ == "moving average":
        vx = _moving_avg_both_halves(vx, second_half_idx, window)
        vy = _moving_avg_both_halves(vy, second_half_idx, window)
    else:
        raise ValueError(f"Unknown filter '{filter_}'. Choose 'Savitzky-Golay' or 'moving average'.")
    return vx, vy


def _savgol_both_halves(v: pd.Series, split_idx: int, window: int, polyorder: int) -> pd.Series:
    v = v.copy()
    v.loc[:split_idx] = _savgol_nan_safe(v.loc[:split_idx], window, polyorder)
    v.loc[split_idx:] = _savgol_nan_safe(v.loc[split_idx:], window, polyorder)
    return v


def _savgol_nan_safe(v: pd.Series, window: int, polyorder: int) -> np.ndarray:
    """
    Apply Savitzky-Golay filter while safely handling NaN values.
    NaN positions are interpolated before filtering, then restored to NaN afterwards.
    """
    nan_mask = v.isna()
    v_filled = v.interpolate(method="linear", limit_direction="both").fillna(0)
    smoothed = signal.savgol_filter(v_filled.values, window_length=window, polyorder=polyorder)
    smoothed[nan_mask.values] = np.nan
    return smoothed


def _moving_avg_both_halves(v: pd.Series, split_idx: int, window: int) -> pd.Series:
    kernel = np.ones(window) / window
    v = v.copy()
    nan_mask_1 = v.loc[:split_idx].isna()
    nan_mask_2 = v.loc[split_idx:].isna()

    v1 = v.loc[:split_idx].interpolate(method="linear", limit_direction="both").fillna(0)
    v2 = v.loc[split_idx:].interpolate(method="linear", limit_direction="both").fillna(0)

    result1 = np.convolve(v1.values, kernel, mode="same")
    result2 = np.convolve(v2.values, kernel, mode="same")

    result1[nan_mask_1.values] = np.nan
    result2[nan_mask_2.values] = np.nan

    v.loc[:split_idx] = result1
    v.loc[split_idx:] = result2
    return v
