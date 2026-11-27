"""
ground_truth.py — a seeded synthetic match in the shape of the Metrica sample window (review SE-02, 27 November 2026).

Every Reconcile and Fuse test started from a real Metrica window, so on a clean CI runner (no sample data) 135 of 275
Matrix tests skipped and sync, fusion and identity ran at 8% to 12% line coverage. ``synthetic_ground_truth`` gives the
same long-format frame (``home_1`` to ``home_11``, ``away_15`` to ``away_25``, 25 Hz, the window's clock) from fictional
players: each holds a formation slot and moves smoothly about it at match-like speeds. The degradation and
reconciliation tests then run in CI; the Metrica window stays the local, data-marked run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from matrix_engine.schema import coerce_schema, validate_schema

#: Slots (metres, home attacking +x): a goalkeeper and a 4-4-2.
_SLOTS = [(-48, 0), (-30, -22), (-32, -8), (-32, 8), (-30, 22), (-8, -24), (-10, -8), (-10, 8), (-8, 24), (14, -8), (14, 8)]


def _smooth(noise: np.ndarray, fps: float, sigma_s: float) -> np.ndarray:
    k = np.arange(-int(4 * sigma_s * fps), int(4 * sigma_s * fps) + 1) / fps
    w = np.exp(-0.5 * (k / sigma_s) ** 2)
    return np.convolve(noise, w / w.sum(), mode="same")


def _ou(n: int, fps: float, tau_s: float, sigma: float, rng) -> np.ndarray:
    """An Ornstein-Uhlenbeck path: wanders about zero with spread ``sigma`` and memory ``tau_s``."""
    a = np.exp(-1.0 / (tau_s * fps))
    eps = rng.normal(0.0, sigma * np.sqrt(1 - a * a), n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        out[i] = a * out[i - 1] + eps[i]
    return _smooth(out, fps, 1.5)  # velocity changes over about a second: median speed about 2 m/s, 99th percentile 7.5


def synthetic_ground_truth(teams=("home",), start_time: float = 600.0, end_time: float = 780.0, fps: float = 25.0,
                           seed: int = 0) -> pd.DataFrame:
    """
    Long-format Matrix tracking for fictional players, in the Metrica window's shape. Deterministic for a seed.

    The ball wanders on a scale of seconds; each team's block shifts with it (as real blocks do), and each player
    wanders about a formation slot with a few seconds' memory. That gives the team-level signals (centroid, spread) the
    clock alignment reads something to lock on to, at match-like speeds (median about 2 m/s, 99th percentile about 7.5).
    """
    rng = np.random.default_rng(seed)
    n = int(round((end_time - start_time) * fps))
    t = start_time + np.arange(n) / fps
    frame = (np.round(t * fps)).astype(np.int64)
    ball = np.stack([30.0 * np.tanh(_ou(n, fps, 6.0, 1.2, rng)), 22.0 * np.tanh(_ou(n, fps, 5.0, 1.2, rng))], axis=1)
    blocks = []
    for team in teams:
        sign = 1.0 if team == "home" else -1.0
        first = 1 if team == "home" else 15
        shift = 0.45 * ball + np.stack([_ou(n, fps, 4.0, 2.0, rng), _ou(n, fps, 4.0, 2.0, rng)], axis=1)
        for k, (sx, sy) in enumerate(_SLOTS):
            own = 1.5 if k == 0 else 3.5
            x = sx * sign + (0.3 if k == 0 else 1.0) * shift[:, 0] + _ou(n, fps, 3.0, own, rng)
            y = sy * sign + (0.3 if k == 0 else 1.0) * shift[:, 1] + _ou(n, fps, 3.0, own, rng)
            blocks.append(pd.DataFrame({"timestamp": np.round(t, 4), "frame": frame, "player_id": f"{team}_{first + k}",
                                        "team": team, "x": np.clip(x, -52.5, 52.5), "y": np.clip(y, -34.0, 34.0),
                                        "source": "metrica", "sample_rate_hz": float(fps)}))
    out = pd.concat(blocks, ignore_index=True).sort_values(["frame", "player_id"], kind="stable").reset_index(drop=True)
    return validate_schema(coerce_schema(out), name="synthetic ground truth")
