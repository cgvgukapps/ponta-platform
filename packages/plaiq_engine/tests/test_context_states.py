"""P8 episodes and citation resolution on small cases and the synthetic eleven-a-side match (fictional players)."""

import numpy as np
import pandas as pd
from context_fixture import eleven_a_side
from plaiq_engine.citations import (
    check_quoted_numbers,
    resolve_event_citation,
    resolve_state_citation,
)
from plaiq_engine.context.fifa_view import FifaViewPackage
from plaiq_engine.context.line_breaks import LineBreakPackage
from plaiq_engine.context.movement import MovementPackage
from plaiq_engine.context.phase import PhasePackage
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.space_value import SpaceValuePackage
from plaiq_engine.context.states import (
    START_REASONS,
    STATE_COLUMNS,
    StatesPackage,
    segment,
    summary_text,
)
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.context.zones import ZonesPackage


def test_segment_merges_short_runs_within_a_chain():
    ts = np.arange(0, 4.0, 0.2)  # 20 points
    per = np.ones(20, int)
    chain = np.array([0] * 12 + [1] * 8, float)
    keys = [("a",)] * 8 + [("b",)] * 2 + [("a",)] * 2 + [("c",)] * 8  # "b" lasts 0.4 s
    spans = segment(keys, chain, ts, per, 1.0, 0.2)
    assert spans == [[0, 11], [12, 19]]  # the short run and the run after it merged into chain 0's first episode
    keys[0:2] = [None, None]  # no team in possession at the start
    assert segment(keys, chain, ts, per, 1.0, 0.2)[0][0] == 2
    # A chain with a single short run keeps it.
    assert segment([("x",)] * 2 + [None] * 3, np.array([5, 5, np.nan, np.nan, np.nan]), ts[:5], per[:5], 1.0, 0.2) == [[0, 1]]


def test_summary_says_how_it_started_and_clearances():
    row = {"team": "away", "phase": "transition", "fifa_phase": "attacking_transition", "fifa_out_phase": "in_contest", "opposed_pct": None,
           "duration_s": 3.6, "start_third": "middle", "end_third": "middle", "passes": 0, "passes_completed": 0, "passes_breaking_a_line": 0,
           "carrier_pressure_max": None, "engagements": 0, "counter_press": False, "runs_in_behind": 0,
           "start_reason": "set_piece", "start_detail": "throw_in", "cleared": True, "end_reason": "cleared"}
    s = summary_text(row)
    assert "Started from a throw-in." in s and "Ended: the ball was cleared." in s and "against" not in s
    s2 = summary_text({**row, "end_reason": "possession_lost", "start_reason": "goalkeeper"})
    assert "Started: the goalkeeper had the ball." in s2 and "Away cleared the ball." in s2


def test_citations_resolve_and_numbers_are_checked():
    st = pd.DataFrame([{"match_id": "m", "state_id": "m:1:0000", "start_s": 10.0, "end_s": 24.2, "passes": 6, "duration_s": 14.2,
                        "carrier_pressure_max": 7}])
    assert resolve_state_citation(st, "m", "m:1:0000", 10.1, 24.2).ok
    assert not resolve_state_citation(st, "m", "m:1:0000", 12.0, 24.2).ok
    assert not resolve_state_citation(st, "other", "m:1:0000", 10.0, 24.2).ok
    ev = pd.DataFrame([{"match_id": "m", "event_id": "evt_m_00001", "timestamp_s": 11.4}])
    assert resolve_event_citation(ev, "m", "evt_m_00001", 11.4).ok and not resolve_event_citation(ev, "m", "evt_m_00002", 11.4).ok
    row = st.iloc[0].to_dict()
    assert check_quoted_numbers("6 passes in 14.2 s; pressure peaked at 7 on the 1 to 10 scale.", row) == []
    assert check_quoted_numbers("8 passes in 14.2 s.", row) == ["8"]


def test_package_states_and_summaries():
    res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), PhasePackage(), LineBreakPackage(), MovementPackage(),
                    SpaceValuePackage(), FifaViewPackage(), StatesPackage()]).run(eleven_a_side(seconds=60.0))
    st = res.outputs["p8"].segments["states"]
    assert list(st.columns) == list(STATE_COLUMNS) and len(st) >= 1
    assert st["state_id"].is_unique and st["label"].eq("estimated").all()
    assert st["passes"].isna().all() and st["events"].isna().all()  # no event feed: unknown, never zero
    assert st["start_reason"].isin(START_REASONS).all() and st["cleared"].isna().all()  # 0.2.0; no feed: clearances unknown
    assert not st["summary"].str.contains("against a contested spell|against a set play|against a long ball").any()
    dc = [c for c in st.columns if c.startswith("data_check_")]
    assert len(dc) == 16 and st[dc].isin(["full", "degraded", "blocked"]).all().all()  # one filterable column per metric
    assert st[[c for c in st.columns if c.startswith(("runs_", "engagements_"))]].dtypes.map(lambda d: d.kind in "iu").all()
    for r in st.to_dict("records"):
        assert check_quoted_numbers(r["summary"], r) == [], r["summary"]
        assert r["summary"] == summary_text(r)
        assert not any(w in r["summary"].lower() for w in ("caused", "because"))
    f = res.frames
    assert set(f["p8_state_id"].dropna()) == set(st["state_id"])
    assert {e.metric for e in res.entries if e.package == "p8"} == {"episodes"}
    # Deterministic.
    again = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), PhasePackage(), LineBreakPackage(), MovementPackage(),
                      SpaceValuePackage(), FifaViewPackage(), StatesPackage()]).run(eleven_a_side(seconds=60.0))
    pd.testing.assert_frame_equal(st, again.outputs["p8"].segments["states"])
