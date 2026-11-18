"""Corner and throw-in set-ups from positions (context/restarts.py), on synthetic positions."""

import numpy as np
from plaiq_engine.context.restarts import (
    _runs,
    corner_setups,
    median_speed,
    throw_in_setups,
)

L, W, HZ = 105.0, 68.0, 5.0


def _scene(seconds: float, taker: bool = True, crowd: int = 14, defender_at_flag: bool = False, taker_left: bool = True):
    """Attackers and defenders in their own frames: a taker at a corner flag and a crowd in the box."""
    n = int(seconds * HZ)
    att = np.full((n, 11), np.nan)
    att_w = np.full((n, 11), np.nan)
    dfd = np.full((n, 11), np.nan)
    dfd_w = np.full((n, 11), np.nan)
    att[:, 0], att_w[:, 0] = 5.0, 0.0  # goalkeeper at home
    dfd[:, 0], dfd_w[:, 0] = 1.0, 0.0  # defending goalkeeper on its line (inside its box)
    if taker:
        att[:, 1], att_w[:, 1] = L - 0.5, (W / 2 - 0.5) * (1 if taker_left else -1)
    in_box_att = max(0, min(crowd // 2, 8))
    in_box_def = max(0, min(crowd - in_box_att - 1, 9))  # the defending keeper counts too
    for j in range(in_box_att):
        att[:, 2 + j], att_w[:, 2 + j] = L - 8.0 - j, -10.0 + 2.5 * j
    for j in range(in_box_def):
        dfd[:, 1 + j], dfd_w[:, 1 + j] = 6.0 + j, 10.0 - 2.5 * j
    if defender_at_flag:
        dfd[:, 10], dfd_w[:, 10] = 1.0, W / 2 - 1.0
    ts = np.arange(n) / HZ
    return att, att_w, dfd, dfd_w, ts, np.ones(n, dtype=int)


def _find(scene, **kw):
    a, aw, d, dw, ts, per = scene
    return corner_setups(a, aw, d, dw, ts, per, L, W, **({"radius": 3.0, "crowd_min": 10, "hold_s": 2.0} | kw))


def test_a_corner_set_up_is_found_on_either_flag():
    assert len(_find(_scene(5.0))) == 1
    assert len(_find(_scene(5.0, taker_left=False))) == 1


def test_no_taker_or_no_crowd_is_not_a_corner():
    assert _find(_scene(5.0, taker=False)) == []
    assert _find(_scene(5.0, crowd=4)) == []


def test_a_brief_set_up_is_not_a_corner():
    assert _find(_scene(1.0)) == []


def test_a_defender_at_the_flag_is_not_a_taker():
    assert _find(_scene(5.0, defender_at_flag=True)) == []


def test_the_set_up_ends_about_when_the_corner_is_taken():
    s = _scene(6.0)
    a, aw, d, dw, ts, per = s
    a[20:, 1] = L - 12.0  # the taker leaves the flag after 4 s
    found = _find((a, aw, d, dw, ts, per))
    assert len(found) == 1 and abs(found[0][1] - 3.8) < 1e-9


def _throw(seconds: float = 4.0, out: float = 0.8, ball_gap: float = 1.0, walking: float = 0.5, opponent_out: bool = False):
    """A thrower beyond the left touchline at depth 40 m, the ball beside them, the rest walking slowly."""
    n = int(seconds * HZ)
    ts = np.arange(n) / HZ
    team_d, team_w = np.tile(np.linspace(10.0, 60.0, 11), (n, 1)), np.zeros((n, 11))
    other_d, other_w = np.tile(np.linspace(20.0, 70.0, 11), (n, 1)), np.zeros((n, 11))
    team_d += walking * ts[:, None]  # everyone drifts at ``walking`` m/s
    other_d += walking * ts[:, None]
    team_d[:, 3], team_w[:, 3] = 40.0, W / 2 + out
    if opponent_out:
        other_w[:, 4] = -(W / 2 + 1.0)
    ball_d, ball_w = np.full(n, 40.0), np.full(n, W / 2 + out - ball_gap)
    return team_d, team_w, other_d, other_w, ball_d, ball_w, ts, np.ones(n, dtype=int)


def _throws(scene, **kw):
    td, tw, od, ow, bd, bw, ts, per = scene
    speed = median_speed([td, od], [tw, ow], ts, per)
    return throw_in_setups(td, tw, od, ow, bd, bw, speed, ts, per, L, W,
                           **({"out_m": 0.0, "hold_s": 2.0, "still_mps": 2.0, "ball_m": 5.0} | kw))


def test_a_throw_in_set_up_is_found():
    found = _throws(_throw())
    assert len(found) == 1 and abs(found[0][1] - 3.8) < 1e-9


def test_no_throw_in_without_the_ball_near_or_when_play_runs_on():
    assert _throws(_throw(ball_gap=12.0)) == []         # the ball is elsewhere: a winger on the line
    assert _throws(_throw(walking=4.0)) == []           # everyone is running: open play
    assert _throws(_throw(seconds=1.5)) == []           # too brief
    assert _throws(_throw(out=-2.0)) == []              # inside the pitch
    assert _throws(_throw(opponent_out=True)) == []     # two players off the pitch: not one thrower


def test_runs_split_at_period_changes():
    cond = np.array([True, True, True, True])
    assert _runs(cond, np.array([1, 1, 2, 2])) == [(0, 1), (2, 3)]
