"""Pass option value split into availability and value: the split reproduces POV exactly."""

import numpy as np
import pytest
from plaiq_engine.analytics.option_split import option_components
from plaiq_engine.analytics.pass_option_value import player_pov


def _defenders(*points):
    return [{"pos": np.array(p, float), "vel": np.zeros(2), "v_max": 9.0, "a_max": 5.5, "reaction_time": 0.7} for p in points]


@pytest.mark.parametrize("receiver, defenders, xt", [
    ((20.0, 5.0), [(30.0, 0.0), (10.0, -10.0)], (0.2, 0.4)),     # open forward pass
    ((12.0, 0.0), [(6.0, 0.5), (14.0, 1.0)], (0.2, 0.3)),         # a defender in the lane
    ((-15.0, 10.0), [(0.0, 20.0)], (0.3, 0.1)),                   # back pass
    ((70.0, 0.0), [(30.0, 0.0)], (0.1, 0.9)),                     # beyond the maximum pass distance
])
def test_availability_times_value_is_pov(receiver, defenders, xt):
    ball, rec, d = np.zeros(2), np.array(receiver, float), _defenders(*defenders)
    c = option_components(ball, rec, xt[0], xt[1], d)
    assert c["pov"] == pytest.approx(player_pov(ball, rec, xt[0], xt[1], d), abs=1e-12)
    assert 0.0 <= c["availability"] <= 1.0


def test_a_blocked_lane_lowers_availability_not_value():
    ball, rec = np.zeros(2), np.array([20.0, 0.0])
    open_ = option_components(ball, rec, 0.2, 0.4, _defenders((40.0, 20.0)))
    blocked = option_components(ball, rec, 0.2, 0.4, _defenders((8.0, 0.0), (12.0, 0.5)))
    assert blocked["availability"] < open_["availability"]
    assert blocked["value"] == open_["value"]
