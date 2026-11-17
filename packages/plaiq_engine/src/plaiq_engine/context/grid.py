"""
grid.py — the context grid every package writes onto (P0).

A match is resampled onto a fixed grid (``context_grid_rate_hz``) keyed by ``match_id`` and
``frame_ts``. ``frame_ts`` is the time in seconds since the first recorded frame, so it is
monotonic across periods even when the vendor's clock restarts (SkillCorner's second half
starts again at 45:00). The vendor clock is kept as ``clock_s``, and ``minute`` is the
football minute (second half from 46).

Each grid point maps to the nearest native frame within half a native period; that frame is
kept as ``native_frame`` so packages can refine at the source rate. Nothing is invented: a
ball or player not observed at that native frame stays unobserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matrix_engine.match import NormalisedMatch

GRID_COLUMNS: tuple[str, ...] = ("match_id", "frame_ts", "period", "native_frame", "clock_s", "minute")

#: Football minute at the start of each period (extra time as 3 and 4).
PERIOD_MINUTE_OFFSET = {1: 0, 2: 45, 3: 90, 4: 105}


def context_match_id(match: NormalisedMatch) -> str:
    """The id used across Ponta's artefacts (same form as the semantic builder's)."""
    return f"{match.provider}_{match.match_id}"


def build_grid(match: NormalisedMatch, rate_hz: float) -> pd.DataFrame:
    """Resample the match's frame clock onto a fixed grid. Deterministic."""
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    frames = match.frames.sort_values("frame", kind="stable")
    f = frames["frame"].to_numpy(dtype=np.int64)
    per = frames["period"].to_numpy()
    clock = frames["time_s"].to_numpy(dtype=float)
    if len(f) == 0:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(GRID_COLUMNS, ("str", "float64", "int64", "int64", "float64", "int64"))})
    elapsed = (f - f[0]) / match.fps
    step = 1.0 / rate_hz
    tol = 0.5 / match.fps + 1e-9
    blocks = []
    for p in sorted(pd.unique(per)):
        m = per == p
        fe, ff, fc = elapsed[m], f[m], clock[m]
        t0 = np.ceil(fe[0] / step - 1e-9) * step
        t = np.round(np.arange(t0, fe[-1] + 1e-9, step), 6)
        j = np.clip(np.searchsorted(fe, t), 0, len(fe) - 1)
        jprev = np.clip(j - 1, 0, len(fe) - 1)
        pick = np.where(np.abs(fe[jprev] - t) <= np.abs(fe[j] - t), jprev, j)
        ok = np.abs(fe[pick] - t) <= tol
        t, pick = t[ok], pick[ok]
        t_in = t - fe[0]
        minute = PERIOD_MINUTE_OFFSET.get(int(p), 0) + np.floor(t_in / 60.0).astype(np.int64) + 1
        blocks.append(pd.DataFrame({
            "match_id": context_match_id(match),
            "frame_ts": np.round(t, 3) + 0.0,
            "period": int(p),
            "native_frame": ff[pick],
            "clock_s": np.round(fc[pick], 3),
            "minute": minute,
        }))
    grid = pd.concat(blocks, ignore_index=True)
    grid["period"] = grid["period"].astype("int64")
    return grid.sort_values("frame_ts", kind="stable").reset_index(drop=True)
