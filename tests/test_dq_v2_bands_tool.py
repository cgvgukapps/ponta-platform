"""Candidate bands for decision quality v2 (tools/plaiq_dq_v2_bands.py), on synthetic passes."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from plaiq_dq_v2_bands import band_index, fit_cuts, rises, weighted_kappa, wilson


def test_band_edges_belong_to_the_upper_band():
    assert band_index(np.array([0.0, 0.4999, 0.5, 0.75, 0.9499, 0.95, 1.0]), (0.5, 0.75, 0.95)).tolist() == [0, 0, 1, 2, 2, 3, 3]


def test_fit_finds_the_steps_in_the_shot_rate():
    rng = np.random.default_rng(0)
    score = rng.uniform(0, 1, 40_000)
    rate = np.select([score < 0.3, score < 0.6, score < 0.85], [0.01, 0.04, 0.08], 0.12)
    d = pd.DataFrame({"score": score, "completed": True, "lead_to_shot": rng.uniform(size=score.size) < rate, "match": "m"})
    assert fit_cuts(d) == (0.3, 0.6, 0.85)
    assert fit_cuts(d, top=0.8) == (0.3, 0.6, 0.8)
    assert fit_cuts(d, top=0.95) is None  # the top band would hold under 10% of passes
    assert rises(d, (0.3, 0.6, 0.85)) and not rises(d.assign(lead_to_shot=d["lead_to_shot"].to_numpy()[::-1]), (0.3, 0.6, 0.85))


def test_interval_and_agreement():
    lo, hi = wilson(10, 100)
    assert lo < 0.1 < hi
    a = np.array([0, 1, 2, 3, 3, 2])
    assert weighted_kappa(a, a) == 1.0
