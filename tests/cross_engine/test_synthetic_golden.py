"""
The headline engine values on the licence-free synthetic match, checked in CI (review finding SE-01).

The golden (``tests/golden/synthetic/engine.json``) holds Plaiq's per-event decision values, the
alternatives, movement value, pass completion, Choice, Gain version 2, 1v1 pressure at five frames
and the control and value surfaces there. A change to any of them fails here; recapture with
``python tools/synthetic_golden.py --capture`` only when the change is intended.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from engine_snapshot import compare, snapshot
from plaiq_engine.match_analysis import MatchAnalysis
from plaiq_engine.semantic_builder import dq_band
from synthetic_match import synthetic_match

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "synthetic" / "engine.json"


@pytest.fixture(scope="module")
def analysis():
    return MatchAnalysis(synthetic_match())


def test_engine_values_match_the_synthetic_golden(analysis):
    gold = json.loads(GOLDEN.read_text())
    diffs = compare(gold, json.loads(json.dumps(snapshot(analysis))))
    assert not diffs, "engine values changed:\n" + "\n".join(diffs[:30])


def test_scales_stay_in_range(analysis):
    """Pressure 1 to 10, Gain 0 to 1.5, Choice 0 to 1, and every band agrees with its score."""
    press = analysis.frame_analytics["pressure"].dropna()
    assert len(press) and press.between(1.0, 10.0).all()
    dq = analysis.event_pov()["decision_quality"].dropna()
    assert len(dq) and dq.between(0.0, 1.5).all()
    assert all(dq_band(v) in {"low_gain", "partial_gain", "most_gain", "full_gain"} for v in dq)
    v2 = analysis.decision_quality_v2.dropna(subset=["dq_v2"])
    assert len(v2) and v2["dq_v2"].between(0.0, 1.0).all()
    bands = {"best_option": (0.95, 1.01), "near_best": (0.60, 0.95), "below_best": (0.45, 0.60), "well_below_best": (-0.01, 0.45)}
    for score, band in zip(v2["dq_v2"], v2["dq_v2_band"]):
        lo, hi = bands[band]
        assert lo <= score < hi or (band == "best_option" and score >= 0.95), (score, band)
    g2 = analysis.gain_v2["gain_v2"].dropna()
    assert g2.between(0.0, 1.5).all()
    pc = analysis.pass_completion["pass_completion_probability"].dropna()
    assert len(pc) and pc.between(0.0, 1.0).all()


def test_the_check_notices_a_changed_value(analysis):
    """The comparison is not vacuous: one nudged decision value is reported."""
    gold = json.loads(GOLDEN.read_text())
    new = json.loads(json.dumps(snapshot(analysis)))
    col = new["event_pov"]["columns"].index("decision_quality")
    row = next(i for i, r in enumerate(new["event_pov"]["rows"]) if r[col] is not None)
    new["event_pov"]["rows"][row][col] = float(np.float64(new["event_pov"]["rows"][row][col]) + 1e-3)
    assert compare(gold, new)
