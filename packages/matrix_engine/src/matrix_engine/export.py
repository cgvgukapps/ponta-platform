#!/usr/bin/env python3
"""
export.py — write the reconciled dataset in Matrix's standard output format.

``Matrix.md`` specifies the deliverable as a single enriched tracking dataset
carrying, per player per frame: calibrated trajectories, unified timestamps,
velocity and acceleration, a confidence score for every frame, and source
attribution. This module produces exactly that from a fused, confidence-scored
frame.

Velocity and acceleration are differentiated with a Savitzky-Golay filter,
matching the approach already used in ``ingestion/metrica_velocities.py`` so
that Matrix output and the existing Plaiq engine describe motion the same way.
Differentiating a noisy position series amplifies the noise, which is why the
derivative comes from a fitted local polynomial rather than a raw difference.

Each player is differentiated independently and only across contiguous stretches
of the output grid — a trajectory either side of a dropout gap is two separate
runs, and pretending otherwise invents a large fake acceleration at the seam.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import signal

#: Savitzky-Golay window in seconds. Matches the ~0.28 s used by
#: ``metrica_velocities.calc_player_velocities`` at 25 Hz.
SMOOTHING_WINDOW_S: float = 0.28

#: Polynomial order for the filter. Order 2 lets one fit serve both the first
#: and second derivative.
POLYORDER: int = 2

#: Speeds above this are treated as tracking errors, as in metrica_velocities.
MAX_SPEED_M_S: float = 12.0

#: Accelerations above this are treated as tracking errors too. Elite sprinters
#: peak near 10 m/s^2; the Metrica sample data contains occasional single-frame
#: position jumps whose second derivative runs into the hundreds. Smoothed
#: velocity absorbs those glitches, but acceleration does not, so it needs its
#: own bound rather than inheriting the speed one.
MAX_ACCEL_M_S2: float = 15.0

#: Columns written, in order, by :func:`export_fused`.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "frame",
    "player_id",
    "team",
    "x",
    "y",
    "vx",
    "vy",
    "speed",
    "ax",
    "ay",
    "acceleration",
    "confidence",
    "source_attribution",
    "n_sources",
    "disagreement_m",
    "max_gap_s",
)


def add_kinematics(
    fused: pd.DataFrame,
    *,
    window_s: float = SMOOTHING_WINDOW_S,
    polyorder: int = POLYORDER,
    max_speed_m_s: float = MAX_SPEED_M_S,
    max_accel_m_s2: float = MAX_ACCEL_M_S2,
) -> pd.DataFrame:
    """
    Add velocity and acceleration columns to a fused frame.

    Parameters
    ----------
    fused : pd.DataFrame
        Fused output on a uniform time grid.
    window_s : float
        Savitzky-Golay window, in seconds. Converted to an odd sample count.
    polyorder : int
        Polynomial order of the local fit.
    max_speed_m_s : float
        Speeds above this are set to NaN, consistent with the existing engine's
        treatment of tracking errors.
    max_accel_m_s2 : float
        Accelerations above this are set to NaN as implausible.

    Returns
    -------
    pd.DataFrame
        Copy of ``fused`` with ``vx``, ``vy``, ``speed``, ``ax``, ``ay`` and
        ``acceleration``.
    """
    out = fused.sort_values(["player_id", "timestamp"], kind="stable").copy()
    for column in ("vx", "vy", "speed", "ax", "ay", "acceleration"):
        out[column] = np.nan

    if out.empty:
        return out.reset_index(drop=True)

    dt = _grid_step(out["timestamp"].to_numpy(dtype=float))
    window = max(polyorder + 1, int(round(window_s / dt)))
    if window % 2 == 0:
        window += 1

    for _, index in out.groupby("player_id", sort=False).indices.items():
        block_t = out["timestamp"].to_numpy(dtype=float)[index]
        for run in _contiguous_runs(block_t, dt):
            rows = index[run]
            if len(rows) < polyorder + 2:
                continue
            local_window = min(window, len(rows) if len(rows) % 2 == 1 else len(rows) - 1)
            if local_window < polyorder + 2:
                continue
            for axis, v_col, a_col in (("x", "vx", "ax"), ("y", "vy", "ay")):
                values = out[axis].to_numpy(dtype=float)[rows]
                out.iloc[rows, out.columns.get_loc(v_col)] = signal.savgol_filter(
                    values, window_length=local_window, polyorder=polyorder,
                    deriv=1, delta=dt,
                )
                out.iloc[rows, out.columns.get_loc(a_col)] = signal.savgol_filter(
                    values, window_length=local_window, polyorder=polyorder,
                    deriv=2, delta=dt,
                )

    speed = np.sqrt(out["vx"] ** 2 + out["vy"] ** 2)
    too_fast = speed > max_speed_m_s
    for column in ("vx", "vy", "ax", "ay"):
        out.loc[too_fast, column] = np.nan
    out["speed"] = np.where(too_fast, np.nan, speed)

    acceleration = np.sqrt(out["ax"] ** 2 + out["ay"] ** 2)
    too_hard = acceleration > max_accel_m_s2
    for column in ("ax", "ay"):
        out.loc[too_hard, column] = np.nan
    out["acceleration"] = np.where(too_hard, np.nan, acceleration)

    return out.sort_values(["timestamp", "player_id"], kind="stable").reset_index(drop=True)


def export_fused(
    scored: pd.DataFrame,
    path: str,
    *,
    with_kinematics: bool = True,
    float_format: str = "%.4f",
) -> str:
    """
    Write the standardised Matrix output to CSV.

    Parameters
    ----------
    scored : pd.DataFrame
        Fused frame, ideally already through ``confidence.score_confidence``.
        Missing optional columns are written empty rather than causing an error,
        so a partial pipeline still produces a readable file.
    path : str
        Destination CSV path. Parent directories are created.
    with_kinematics : bool
        Derive velocity/acceleration first when they are not already present.
    float_format : str
        Passed through to ``DataFrame.to_csv``.

    Returns
    -------
    str
        The path written.
    """
    out = scored
    if with_kinematics and "speed" not in out.columns:
        out = add_kinematics(out)

    for column in OUTPUT_COLUMNS:
        if column not in out.columns:
            out = out.assign(**{column: np.nan})

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    out.loc[:, list(OUTPUT_COLUMNS)].to_csv(path, index=False, float_format=float_format)
    return path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _grid_step(timestamps: np.ndarray) -> float:
    """Modal spacing of the output grid."""
    unique = np.unique(timestamps)
    if unique.size < 2:
        return 1.0
    step = float(np.median(np.diff(unique)))
    return step if step > 0 else 1.0


def _contiguous_runs(timestamps: np.ndarray, dt: float) -> list[np.ndarray]:
    """
    Split one player's sample indices into runs with no missing grid steps.

    Differentiating across a gap would read the jump either side of it as a
    single enormous acceleration.
    """
    if len(timestamps) == 0:
        return []
    breaks = np.flatnonzero(np.diff(timestamps) > dt * 1.5) + 1
    return [
        run for run in np.split(np.arange(len(timestamps)), breaks) if len(run) > 0
    ]
