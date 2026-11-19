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


# -- hex-cell candidates: the best ball into each run, on Plaiq's grid -------------------------------------------

def _grid():
    from plaiq_engine.analytics.hex_analytics import cartesian_to_hex
    from plaiq_engine.analytics.pitch_control import build_pitch_hex_grid
    from plaiq_engine.match_analysis import HEX_RADIUS
    h = build_pitch_hex_grid(HEX_RADIUS)
    q, r = (np.asarray(a, int) for a in cartesian_to_hex(h[:, 0], h[:, 1], HEX_RADIUS))
    return h, q, r, HEX_RADIUS


def test_run_cells_are_the_rings_around_the_run():
    from plaiq_engine.analytics.decision_quality_v2 import run_cells
    h, q, r, R = _grid()
    assert [len(run_cells((10.0, 5.0), h, q, r, R, k)) for k in (0, 1, 2)] == [1, 7, 19]
    c0 = run_cells((10.0, 5.0), h, q, r, R, 0)
    assert np.linalg.norm(h[c0[0]] - (10.0, 5.0)) <= R
    corner = h[np.argmax(h[:, 0] + h[:, 1])]
    assert 1 <= len(run_cells(corner, h, q, r, R, 2)) < 19  # the cells off the pitch drop out
    # A run ending off the pitch is centred on the nearest cell on it, whatever the distance.
    far, near = run_cells((200.0, 0.0), h, q, r, R, 2), run_cells((h[:, 0].max() + 1.0, 0.0), h, q, r, R, 2)
    assert 1 < len(far) < 19 and set(far) == set(near)
    nearest = int(np.argmin(np.linalg.norm(h - (200.0, 0.0), axis=1)))
    assert nearest in far and set(run_cells((200.0, 0.0), h, q, r, R, 0)) == {nearest}


def test_cell_probabilities_are_the_pass_completion_probability_at_each_cell():
    from plaiq_engine.analytics.decision_quality_v2 import option_cells_probabilities
    from plaiq_engine.analytics.pass_probability import PassModel, pass_probability
    h, q, r, R = _grid()
    m = PassModel()
    ball = np.array([0.0, 0.0])
    mates = _players((20.0, 0.0), (10.0, 15.0), v=(2.0, 0.0))
    defenders = _players((10.0, 1.0), (30.0, 20.0))
    cells, p = option_cells_probabilities(ball, mates, defenders, m, h, q, r, R, rings=1)
    assert [len(c) for c in cells] == [7, 7] and [len(x) for x in p] == [7, 7]
    for i in range(2):
        for c, pc in zip(cells[i], p[i], strict=True):
            assert pc == pytest.approx(pass_probability(ball, h[c], mates, defenders, m)["p_complete"], abs=1e-9)
    assert p[0].max() < p[1].max()  # the defender stands in the first lane


def test_each_option_is_its_best_cell():
    from plaiq_engine.analytics.decision_quality_v2 import decision_quality_cells
    p = [np.array([0.9, 0.5]), np.array([0.4, 0.8, 0.6])]
    v = [np.array([0.02, 0.05]), np.array([0.03, 0.04, 0.02])]  # best cells: 0.025 and 0.032
    assert decision_quality_cells(p, v, 1)["dq"] == pytest.approx(1.0)
    assert decision_quality_cells(p, v, 0)["dq"] == pytest.approx(0.025 / 0.032)
    assert np.isnan(decision_quality_cells(p, v, 3)["dq"])


def test_one_moment_arrays_equal_the_shipped_pass_arrays():
    from plaiq_engine.analytics.decision_quality_v2 import pass_arrays_one_moment
    from plaiq_engine.analytics.pass_probability import PassModel, pass_arrays
    m = PassModel()
    ball = np.array([-5.0, 3.0])
    mates = _players((20.0, 0.0), (10.0, 15.0), (-20.0, -10.0), v=(2.0, -1.0))
    defenders = _players((10.0, 1.0), (30.0, 20.0), (0.0, 0.0))
    targets = np.array([[20.0, 0.0], [12.0, 16.0], [-5.2, 3.1], [40.0, -30.0]])
    batch = [{"ball": ball, "target": t, "attackers": mates, "defenders": defenders, "high": False} for t in targets]
    a, b = pass_arrays(batch, m, max_players=3), pass_arrays_one_moment(ball, targets, mates, defenders, m)
    for field in a.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field), err_msg=field)


def test_agreed_bands_are_descriptive_and_not_version_1s():
    from plaiq_engine.analytics.decision_quality_v2 import dq_v2_band
    assert [dq_v2_band(v) for v in (0.0, 0.4499, 0.45, 0.5999, 0.60, 0.9499, 0.95, 1.0)] == [
        "well_below_best", "well_below_best", "below_best", "below_best", "near_best", "near_best", "best_option", "best_option"]
    assert dq_v2_band(None) is None and dq_v2_band(np.nan) is None
    from plaiq_engine.semantic_builder import BAND_CUTS
    assert not {b for _, b in BAND_CUTS} & {"best_option", "near_best", "below_best", "well_below_best"}


def test_the_adopted_score_is_ring_one_cells():
    from plaiq_engine.analytics.decision_quality_v2 import DQ_V2_RINGS, DQ_V2_VERSION
    assert DQ_V2_RINGS == 1 and DQ_V2_VERSION == "1.1.0"
