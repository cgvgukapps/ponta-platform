"""Decision quality v2: expected value at the moment of the pass (analytics/decision_quality_v2.py)."""

import inspect

import numpy as np
import pytest
from plaiq_engine.analytics.decision_quality_v2 import decision_quality_v2


def _d(*pts):
    return [{"pos": np.array(p, float), "vel": np.zeros(2), "v_max": 9.0, "a_max": 5.5, "reaction_time": 0.7} for p in pts]


BALL = np.zeros(2)
MATES = [np.array([15.0, 10.0]), np.array([25.0, -5.0])]
DEF = _d((20.0, -6.0), (40.0, 0.0))


def test_choosing_the_best_option_scores_one():
    xts = [0.3, 0.5]
    best = max(range(2), key=lambda i: decision_quality_v2(BALL, MATES[i], 0.2, xts[i], [MATES[1 - i]], [xts[1 - i]], DEF)["ev_chosen"])
    r = decision_quality_v2(BALL, MATES[best], 0.2, xts[best], [MATES[1 - best]], [xts[1 - best]], DEF)
    assert r["dq"] == pytest.approx(1.0)


def test_a_risky_choice_scores_below_a_safe_better_one():
    # Into a crowded lane (defenders on the line) when an open team-mate offers similar value.
    blocked = decision_quality_v2(BALL, np.array([20.0, 0.0]), 0.2, 0.45, [np.array([18.0, 15.0])], [0.45], _d((8.0, 0.0), (12.0, 0.5)))
    assert blocked["dq"] < 0.55


def test_the_outcome_is_not_an_input():
    params = set(inspect.signature(decision_quality_v2).parameters)
    assert not params & {"completed", "outcome", "success", "end_frame"}
