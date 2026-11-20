"""P6 movement types on the synthetic eleven-a-side match (fictional players, no sample data)."""

import dataclasses

import numpy as np
from context_fixture import eleven_a_side
from plaiq_engine.context.movement import (
    MovementPackage,
    channel_band,
    classify_attacking,
    runs_of,
)
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.context.thresholds import values

CFG = values()


def _with_runs(m):
    """Home_10 sprints in behind the away back four; away's holding midfielder (Away_6) sprints at the carrier (Home_7)."""
    tr = m.tracking.copy()
    t = tr["timestamp"].to_numpy()

    def move(pid, t0, t1, xy0, xy1):
        sel = (tr["player_id"] == pid) & (t >= t0) & (t <= t1)
        u = (t[sel] - t0) / (t1 - t0)
        tr.loc[sel, "x"] = xy0[0] + u * (xy1[0] - xy0[0])
        tr.loc[sel, "y"] = xy0[1] + u * (xy1[1] - xy0[1])
        after = (tr["player_id"] == pid) & (t > t1)
        tr.loc[after, "x"], tr.loc[after, "y"] = xy1

    move("home_10", 10.0, 13.0, (11.5, 7.0), (33.0, 7.0))     # about 7 m/s, from midfield to beyond the back four
    move("away_6", 20.0, 22.0, (10.5, 0.0), (-5.5, 7.5))      # about 8 m/s, onto the carrier (Home_7 near (-7.5, 8.8))
    return dataclasses.replace(m, tracking=tr)


def test_runs_found_typed_and_placed_against_the_units():
    res = Registry([PossessionPackage(), StructurePackage(), MovementPackage()]).run(_with_runs(eleven_a_side(seconds=30.0)))
    runs = res.outputs["p6"].segments["runs"]
    behind = runs.loc[(runs["player_id"] == "home_10") & runs["in_possession"]]
    assert len(behind) == 1
    r = behind.iloc[0]
    assert r["type"] == "behind" and r["area_end"] == "in_behind" and r["behind_from"] == "from_in_front"
    assert r["gain_m"] > 15 and r["shape"] in ("inside", "in_to_out")
    press = runs.loc[runs["player_id"] == "away_6"]
    assert len(press) == 1 and not press["in_possession"].iloc[0] and press["type"].iloc[0] == "pressing"
    f = res.frames
    assert f["p6_home_runs"].max() >= 1 and f["p6_away_runs"].max() >= 1 and f["p6_runs_in_behind"].max() == 1
    assert {e.metric: e.state for e in res.entries}["movement"] == "full"
    assert not [c for c in f.columns if c.startswith("p6_") and c.endswith(("_x", "_y"))]


def test_run_detection_and_the_type_rules():
    ts = np.arange(20) / 5.0
    speed = np.r_[np.zeros(5), np.full(5, 5.0), np.zeros(5), np.full(2, 5.0), np.zeros(3)]
    assert runs_of(speed, ts, np.ones(20), CFG["p6_run_min_speed_mps"], CFG["p6_run_min_s"]) == [(5, 9)]  # 0.4 s is too short
    base = {"carrier_d0": 40.0, "carrier_d1": 40.0, "carrier_w0": 0.0, "carrier_w1": 0.0, "ball_d0": 40.0, "ball_w0": 0.0,
            "area_end": "in_between"}
    assert classify_attacking({**base, "d0": 50.0, "d1": 50.5, "w0": 12.0, "w1": 26.0}, CFG, 105.0) == "pulling_wide"
    assert classify_attacking({**base, "d0": 50.0, "d1": 62.0, "w0": 12.0, "w1": 16.0}, CFG, 105.0) == "run_ahead_of_the_ball"
    assert classify_attacking({**base, "d0": 55.0, "d1": 47.0, "w0": 5.0, "w1": 3.0}, CFG, 105.0) == "coming_short"
    assert classify_attacking({**base, "d0": 30.0, "d1": 38.0, "w0": 6.0, "w1": 6.0}, CFG, 105.0) == "support"
    assert [channel_band(w) for w in (0.0, 9.5, -15.0, 22.0)] == [0, 1, 1, 2]


def test_the_carriers_own_run_is_a_carry_not_an_off_ball_run():
    """P6 0.2.0 (the founder's run review): Home_7 sprints with the ball; that is a carry."""
    m = eleven_a_side(seconds=30.0)
    tr, ball = m.tracking.copy(), m.ball.copy()
    t, tb = tr["timestamp"].to_numpy(), (ball["frame"].to_numpy() - 1) / m.fps
    x0, y0, x1, y1 = -7.5, 8.8, 12.0, 8.8
    sel = (tr["player_id"] == "home_7") & (t >= 5.0) & (t <= 8.0)
    u = (t[sel] - 5.0) / 3.0
    tr.loc[sel, "x"], tr.loc[sel, "y"] = x0 + u * (x1 - x0), y0
    tr.loc[(tr["player_id"] == "home_7") & (t > 8.0), "x"] = x1
    sb = (tb >= 5.0) & (tb <= 8.0)
    ball.loc[sb, "x"], ball.loc[sb, "y"] = x0 + (tb[sb] - 5.0) / 3.0 * (x1 - x0) + 0.5, y0
    ball.loc[tb > 8.0, "x"] = x1 + 0.5
    res = Registry([PossessionPackage(), StructurePackage(), MovementPackage()]).run(dataclasses.replace(m, tracking=tr, ball=ball))
    runs = res.outputs["p6"].segments["runs"]
    mine = runs.loc[runs["player_id"] == "home_7"]
    assert len(mine) == 1 and mine["type"].iloc[0] == "carry" and mine["in_possession"].iloc[0]
    assert res.outputs["p6"].measurements["carries"] == 1
