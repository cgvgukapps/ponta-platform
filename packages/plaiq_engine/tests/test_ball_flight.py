"""Ball flight as explicit parameters (analytics/ball_flight.py): v1 reproduced exactly, the candidates behave."""

import numpy as np
import pytest
from plaiq_engine.analytics.ball_flight import V1, FlightModel, targeted_pcz_flight
from plaiq_engine.analytics.pass_option_value import targeted_pcz


def _defenders(rng, n):
    return [{"pos": rng.uniform([-40, -30], [40, 30]), "vel": rng.normal(0, 3, 2), "v_max": 9.0, "a_max": 5.5, "reaction_time": 0.7}
            for _ in range(n)]


def test_v1_defaults_reproduce_targeted_pcz():
    rng = np.random.default_rng(7)
    for _ in range(50):
        ball, recv = rng.uniform([-45, -30], [45, 30], 2), rng.uniform([-45, -30], [45, 30], 2)
        d = _defenders(rng, 10)
        assert targeted_pcz_flight(ball, recv, d, V1) == pytest.approx(targeted_pcz(ball, recv, d), abs=1e-12)


def test_pass_waypoints_reach_the_end_together_and_constant_speed_is_linear():
    f = FlightModel(waypoints="pass", decel=0.0)
    t = f.waypoint_times(40.0, [0.0, 10.0, 20.0, 40.0])
    assert t[-1] == pytest.approx(float(f.time_to(40.0)))
    assert t[1] == pytest.approx(t[-1] / 4)
    # A ball that slows passes the early points sooner than one at constant speed, and ends at the same time.
    g = FlightModel(waypoints="pass", decel=0.5).waypoint_times(40.0, [10.0, 40.0])
    assert g[0] < t[1] and g[1] == pytest.approx(t[-1])


def test_v1_point_timing_is_slower_early_on_long_passes():
    # The flaw the candidate addresses: v1 times the 10 m point of a 40 m pass as a 10 m pass.
    assert V1.waypoint_times(40.0, [10.0])[0] > FlightModel(waypoints="pass").waypoint_times(40.0, [10.0])[0]


def test_a_later_reaction_lowers_interception_risk():
    ball, recv = np.array([0.0, 0.0]), np.array([30.0, 0.0])
    d = [{"pos": np.array([15.0, 6.0]), "vel": np.zeros(2), "v_max": 9.0, "a_max": 5.5, "reaction_time": 0.7}]
    fast = targeted_pcz_flight(ball, recv, d, FlightModel(reaction_time=0.0))[0]
    slow = targeted_pcz_flight(ball, recv, d, FlightModel(reaction_time=1.2))[0]
    assert fast > slow
