"""
clock.py — one clock convention for Matrix (review DE-09, 27 November 2026).

Matrix joins and resamples on **elapsed match time**: each period starts where the previous one ended, so time never
runs backwards across a half. Metrica's files already count this way and the SkillCorner loader rebases to it. Some
feeds count from the start of each period (PFF's ``elapsed``) or put the second half at a fixed 45:00 (the German
league's reader did), which makes a first half's stoppage time overlap the second half; ``elapsed_match_time`` turns a
period-relative clock into elapsed time. The football clock (45:00 at the second half's kick-off) is a display matter,
not a join key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def elapsed_match_time(period: pd.Series, seconds_into_period: pd.Series, step_s: float) -> pd.Series:
    """
    Elapsed match time from a period and the seconds into it: each period is placed after the earlier periods' observed
    lengths (their last time plus one sample step). Periods are ordered by number.
    """
    p = pd.Series(period).astype(int)
    t = pd.Series(seconds_into_period, index=p.index).astype(float)
    start, offsets = 0.0, {}
    for k in sorted(p.unique()):
        offsets[k] = start - float(t[p == k].min())
        start = float(t[p == k].max()) + offsets[k] + step_s
    return t + p.map(offsets).astype(float)


def runs_backwards(frames: np.ndarray, times: np.ndarray) -> bool:
    """True when, in frame order, time ever decreases."""
    order = np.argsort(frames, kind="stable")
    return bool(np.any(np.diff(np.asarray(times, float)[order]) < 0))
