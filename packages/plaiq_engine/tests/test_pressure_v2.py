"""Directional, count-aware pressure (analytics/pressure_v2.py)."""

import numpy as np
import pytest
from plaiq_engine.analytics.pressure_v2 import defender_pressure, pressure

PLAYER, ATTACK = (0.0, 0.0), (1.0, 0.0)   # attacking towards +x


def test_in_front_counts_more_than_behind_at_the_same_distance():
    front = pressure(PLAYER, ATTACK, [(2.0, 0.0)])
    behind = pressure(PLAYER, ATTACK, [(-2.0, 0.0)])
    assert front > behind > 0
    assert pressure(PLAYER, ATTACK, [(-4.0, 0.0)]) == 0.0     # behind, beyond the 3 m reach
    assert pressure(PLAYER, ATTACK, [(8.0, 0.0)]) > 0.0       # in front, inside the 9 m reach


def test_more_pressers_add_up_and_max_takes_the_closest():
    one = pressure(PLAYER, ATTACK, [(2.0, 0.0)])
    two = pressure(PLAYER, ATTACK, [(2.0, 0.0), (0.0, 2.0)])
    assert two > one
    assert pressure(PLAYER, ATTACK, [(2.0, 0.0), (0.0, 2.0)], aggregate="max") == pytest.approx(one)


def test_contact_is_full_pressure_and_missing_defenders_are_ignored():
    c = defender_pressure(PLAYER, ATTACK, [(0.0, 0.0), (np.nan, np.nan)])
    assert c[0] == pytest.approx(1.0) and c[1] == 0.0
