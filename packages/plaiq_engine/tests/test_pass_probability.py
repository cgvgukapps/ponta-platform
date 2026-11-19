"""Probabilistic pass completion (analytics/pass_probability.py): behaves as a probability and as football."""

import numpy as np
import pytest
from plaiq_engine.analytics.pass_probability import (
    MEASURED_FLIGHT,
    PassModel,
    pass_arrays,
    pass_probability,
    probability_from_arrays,
    projected_target,
)


def _p(*xy, v=(0.0, 0.0)):
    return [{"pos": np.array(p, float), "vel": np.array(v, float)} for p in xy]


BALL, TARGET = np.array([0.0, 0.0]), np.array([20.0, 0.0])
RECEIVER = _p((20.0, 0.0))


def test_an_open_pass_is_near_certain_and_a_blocked_one_is_not():
    open_ = pass_probability(BALL, TARGET, RECEIVER, _p((-20.0, 25.0)))
    lane = pass_probability(BALL, TARGET, RECEIVER, _p((8.0, 0.5), (12.0, -0.5)))
    assert open_["p_complete"] > 0.9
    assert lane["p_complete"] < open_["p_complete"] - 0.3
    for r in (open_, lane):
        assert 0.0 <= r["p_complete"] <= 1.0 and r["p_complete"] == pytest.approx(r["p_survive"] * r["share_at_target"])


def test_a_defender_closer_to_the_target_takes_the_share_at_the_target():
    far = pass_probability(BALL, TARGET, RECEIVER, _p((30.0, 10.0)))
    tight = pass_probability(BALL, TARGET, _p((26.0, 0.0)), _p((21.0, 0.5)))
    assert tight["share_at_target"] < 0.5 < far["share_at_target"]


def test_a_high_ball_clears_a_defender_in_the_middle():
    mid = _p((10.0, 0.0))
    flat = pass_probability(BALL, TARGET, RECEIVER, mid, high=False)
    high = pass_probability(BALL, TARGET, RECEIVER, mid, high=True)
    assert high["p_survive"] > flat["p_survive"]


def test_batch_equals_single_and_padding_does_not_change_results():
    passes = [{"ball": BALL, "target": TARGET, "attackers": RECEIVER, "defenders": _p((8.0, 1.0)), "high": False},
              {"ball": BALL, "target": np.array([40.0, 5.0]), "attackers": _p((40.0, 5.0), (10.0, -10.0)), "defenders": _p((25.0, 2.0), (5.0, 5.0)), "high": True}]
    m = PassModel()
    batch = probability_from_arrays(pass_arrays(passes, m), m)["p_complete"]
    single = [pass_probability(p["ball"], p["target"], p["attackers"], p["defenders"], m, p["high"])["p_complete"] for p in passes]
    assert batch == pytest.approx(single, abs=1e-9)


def test_projected_target_leads_a_running_receiver_by_the_ball_flight():
    still = projected_target(BALL, TARGET, np.zeros(2))
    run = projected_target(BALL, TARGET, np.array([6.0, 0.0]))
    assert still == pytest.approx(TARGET)
    t = float(MEASURED_FLIGHT.time_to(np.linalg.norm(run - BALL)))
    assert run == pytest.approx(TARGET + np.array([6.0, 0.0]) * t, abs=0.05)


def test_parameter_sets_by_provider():
    from plaiq_engine.analytics.pass_probability import PARAMETER_SETS, parameters_for
    assert parameters_for("SkillCorner") is PARAMETER_SETS["skillcorner"]
    assert parameters_for("metrica") is PARAMETER_SETS["default"]
    for ps in PARAMETER_SETS.values():
        r = pass_probability(BALL, TARGET, RECEIVER, _p((-20.0, 25.0)), ps.model)
        assert 0.0 <= r["p_complete"] <= 1.0
