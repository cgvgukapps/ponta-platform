"""P4v, the FIFA view of phases, on the synthetic eleven-a-side match and small cases (fictional players, no sample data)."""

import numpy as np
import pandas as pd
from context_fixture import eleven_a_side
from plaiq_engine.context.fifa_view import (
    FifaViewPackage,
    chain_presses,
    held,
    runs,
    type_engagement,
)
from plaiq_engine.context.line_breaks import LineBreakPackage
from plaiq_engine.context.phase import PhasePackage
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.context.thresholds import values
from plaiq_engine.context.zones import ZonesPackage

CFG = values()


def test_runs_and_held():
    assert runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 4)]
    ts = np.arange(0, 3.0, 0.2)
    ok = np.ones(len(ts), bool)
    ok[5] = False  # one missed point in two seconds still holds (1.6 s of 2.0 s)
    h = held(ok, ts, 1.6, 2.0)
    assert not h[:8].any() or h[8]  # needs 1.6 s of history first
    assert h[-1]
    ok[9:12] = False  # 0.6 s missed: fails for the next two seconds
    assert not held(ok, ts, 1.6, 2.0)[12]


def test_engagement_types_in_priority_order():
    L = 105.0
    assert type_engagement(2.0, 1.0, 80.0, L, CFG) == "recovery"  # running back wins over everything
    assert type_engagement(0.0, 2.0, 80.0, L, CFG) == "counter_press"
    assert type_engagement(0.0, 30.0, 80.0, L, CFG) == "press"
    assert type_engagement(0.0, 30.0, 20.0, L, CFG) == "pressure"  # the ball in the defender's own third
    assert type_engagement(float("nan"), float("nan"), 80.0, L, CFG) == "press"


def test_presses_need_two_players_and_no_change_of_possession():
    e = pd.DataFrame([
        {"team": "home", "player": "home_4", "period": 1, "start_ts": 10.0, "end_ts": 11.0, "type": "counter_press"},
        {"team": "home", "player": "home_8", "period": 1, "start_ts": 12.0, "end_ts": 13.0, "type": "press"},
        {"team": "home", "player": "home_8", "period": 1, "start_ts": 30.0, "end_ts": 31.0, "type": "press"},  # alone, later
        {"team": "home", "player": "home_6", "period": 1, "start_ts": 31.5, "end_ts": 32.0, "type": "pressure"},  # not a press
    ])
    p = chain_presses(e, {"home": np.array([])}, CFG)
    assert len(p) == 1 and p.iloc[0]["type"] == "counter_press" and p.iloc[0]["players"] == 2
    split = chain_presses(e, {"home": np.array([11.5])}, CFG)  # possession changed between the two engagements
    assert split.empty


def test_package_fields_and_entries():
    res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), PhasePackage(), LineBreakPackage(),
                    FifaViewPackage()]).run(eleven_a_side(seconds=60.0))
    f = res.frames
    assert [c for c in f.columns if c.startswith("p4v_")] == ["p4v_phase", "p4v_out_phase", "p4v_engaging_defenders",
                                                              "p4v_fifa_block", "p4v_version"]
    assert f["p4v_phase"].dropna().isin(["build_up_opposed", "build_up_unopposed", "progression", "final_third", "long_ball",
                                         "counter_attack", "attacking_transition", "set_play", "in_contest"]).all()
    assert f["p4v_fifa_block"].dropna().isin(["mid_block", "low_block"]).all()
    assert {e.metric for e in res.entries if e.package == "p4v"} == {"engagements", "fifa_view", "fifa_block"}
    assert set(res.outputs["p4v"].segments) == {"engagements", "presses"}
    assert not [c for c in f.columns if c.startswith("p4v_") and c.endswith(("_x", "_y", "_cell"))]
    # P4's own labels are untouched by the view.
    assert "p4_phase" in f and f["p4_version"].iloc[0] == PhasePackage.version
