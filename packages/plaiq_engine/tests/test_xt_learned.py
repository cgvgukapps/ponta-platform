"""Expected threat learned from events (analytics/xt_learned.py), on hand-built actions."""

import numpy as np
import pytest
from plaiq_engine.analytics.xt_learned import fit_xt, lookup


def test_shots_and_moves_give_the_expected_values():
    L, W = 100.0, 60.0
    # Zone A (far end, x 95): only shots, half score. Zone B (x 55): moves, all to zone A, never shots.
    n = 100
    start_x = np.r_[np.full(n, 95.0), np.full(n, 55.0)]
    start_y = np.zeros(2 * n)
    end_x = np.r_[np.full(n, np.nan), np.full(n, 95.0)]
    end_y = np.zeros(2 * n)
    is_shot = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    success = np.r_[np.zeros(n, bool), np.r_[np.ones(n // 2, bool), np.zeros(n // 2, bool)]]  # half the moves fail
    goal = np.r_[np.r_[np.ones(n // 2, bool), np.zeros(n // 2, bool)], np.zeros(n, bool)]
    s = fit_xt(start_x, start_y, end_x, end_y, is_shot, success, goal, length=L, width=W)
    a = lookup(s, [95.0], [0.0], L, W)[0]
    b = lookup(s, [55.0], [0.0], L, W)[0]
    assert a == pytest.approx(0.5)                # every action a shot, half score
    assert b == pytest.approx(0.5 * 0.5)          # half the moves arrive in A
    assert lookup(s, [5.0], [0.0], L, W)[0] == 0.0  # no actions: no value


def test_the_surface_scales_to_another_pitch_size():
    s = fit_xt([95.0], [0.0], [np.nan], [np.nan], [True], [False], [True], length=100.0, width=60.0)
    assert lookup(s, [104.0], [0.0], 110.0, 70.0)[0] == pytest.approx(1.0)
