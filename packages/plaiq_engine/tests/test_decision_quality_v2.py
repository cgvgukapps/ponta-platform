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


# -- 0.2.0: the narrower candidates, with the pass completion probability as the risk term ------------------

def _players(*xy, v=(0.0, 0.0)):
    return [{"pos": np.array(p, float), "vel": np.array(v, float)} for p in xy]


def test_v1_with_risk_scales_v1_by_the_completion_probability():
    from plaiq_engine.analytics.decision_quality_v2 import decision_quality_risk
    assert decision_quality_risk(1.2, 0.5) == pytest.approx(0.6)
    assert decision_quality_risk(1.4, 1.0) == pytest.approx(1.4)
    assert np.isnan(decision_quality_risk(np.nan, 0.9)) and np.isnan(decision_quality_risk(1.0, np.nan))


def test_option_probabilities_match_the_shipped_pass_completion_for_each_mate():
    from plaiq_engine.analytics.decision_quality_v2 import option_probabilities
    from plaiq_engine.analytics.pass_probability import (
        PassModel,
        pass_probability,
        projected_target,
    )
    m = PassModel()
    ball = np.array([0.0, 0.0])
    mates = _players((20.0, 0.0), (10.0, 15.0), v=(2.0, 0.0))
    defenders = _players((10.0, 1.0), (30.0, 20.0))
    targets, p = option_probabilities(ball, mates, defenders, m)
    for i, mate in enumerate(mates):
        t = projected_target(ball, mate["pos"], mate["vel"], m.flight)
        assert targets[i] == pytest.approx(t)
        assert p[i] == pytest.approx(pass_probability(ball, t, mates, defenders, m)["p_complete"], abs=1e-9)
    assert p[0] < p[1]  # the defender stands in the first lane


def test_projected_scores_the_choice_against_the_best_expected_value():
    from plaiq_engine.analytics.decision_quality_v2 import decision_quality_projected
    p, v = np.array([0.9, 0.5, 0.95]), np.array([0.02, 0.06, 0.01])  # expected values 0.018, 0.030, 0.0095
    best = decision_quality_projected(p, v, 1)
    assert best["dq"] == pytest.approx(1.0) and best["best_is_chosen"]
    safe = decision_quality_projected(p, v, 2)
    assert safe["dq"] == pytest.approx(0.0095 / 0.030)
    assert np.isnan(decision_quality_projected(p, v, 5)["dq"])
