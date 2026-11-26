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


class _GappedPhase(PhasePackage):
    """P4 with no phase for 4 s in the middle of the match (the ball out of camera, say); P1's chain is unbroken."""

    def compute(self, ctx):
        out = super().compute(ctx)
        ts = ctx.grid["frame_ts"].to_numpy(float)
        out.frames.loc[(ts >= 20.0) & (ts < 24.0), "p4_phase_id"] = pd.NA
        return out


def test_an_episode_split_by_a_gap_in_one_chain_continues():
    """Same chain, a gap with no phase: the first part ends phase_unknown and the next continues (review AI-01)."""
    res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), _GappedPhase(), LineBreakPackage(), MovementPackage(),
                    SpaceValuePackage(), FifaViewPackage(), StatesPackage()]).run(eleven_a_side(seconds=60.0))
    st = res.outputs["p8"].segments["states"].sort_values("start_s").reset_index(drop=True)
    assert st["chain_id"].nunique() == 1 and len(st) >= 2
    gap = st.index[st["end_s"] <= 20.0 + 0.2][-1]
    assert st.at[gap, "end_reason"] in ("phase_unknown", "contested_kept")  # no phase there; here not a contest
    assert st.at[gap + 1, "start_reason"] == "continued"
    assert (st["end_reason"].iloc[:-1] != "end_of_period").all()  # only the chain's last episode takes the chain's end


def test_quoted_numbers_are_bound_to_their_fields():
    """The reviewers' cases (AI-02, SE-04, GC-02): a number must match the field its words name, not any field."""
    from plaiq_engine.citations import check_claims

    row = {"passes": 3, "passes_completed": 2, "carrier_pressure_max": 9.0, "duration_s": 7.24, "goal_side_defenders_median": 4.4,
           "players_observed": 22, "opposed_pct": 40, "minute_display": "34'"}
    s = "Pressure on the ball carrier peaked at 3 on the 1 to 10 scale. 9 passes, 7 completed, 4 broke a line."
    assert check_quoted_numbers(s, row) == ["3", "9", "7", "4"]
    assert check_quoted_numbers("Pressure was 22.", row) == ["22"]  # equal to another field, and sentence-final
    assert check_quoted_numbers("It lasted 3 seconds.", row) == ["3"]
    assert check_quoted_numbers("Eight passes.", row) == ["Eight"] and check_quoted_numbers("one of the best", row) == []
    assert check_quoted_numbers("In the 35th minute", row) == ["35th"] and check_quoted_numbers("In the 34th minute", row) == []
    assert check_quoted_numbers("opposed 40% of the time", row) == []
    assert check_quoted_numbers("an 81% chance", {"pass_completion_probability": 0.81}) == []
    assert check_quoted_numbers("Choice was 1.26", {"dq_score": 1.26, "dq_v2_score": 0.81}) == ["1.26"]  # Gain is not Choice
    assert check_quoted_numbers("at 88:56", {"minute_display": "88:56"}) == [] and check_quoted_numbers("at 88:57", {"minute_display": "88:56"})
    assert check_quoted_numbers("3 passes", pd.Series({"passes": np.int64(3)})) == []  # numpy integers count
    assert check_quoted_numbers("It peaked at 1 on the 1 to 10 scale.", {"carrier_pressure_max": 4.0}) == ["1"]
    assert check_claims([{"field": "dq_v2_score", "value": 0.81}], {"dq_v2_score": 0.81}) == []
    assert check_claims([{"field": "dq_v2_score", "value": 1.26}, {"field": "nope", "value": 1}], {"dq_v2_score": 0.81, "dq_score": 1.26})


def test_citations_with_no_time_never_resolve():
    """NaN times compared false with every tolerance, so they resolved (SE-04, AI-13)."""
    ev = pd.DataFrame([{"match_id": "m", "event_id": "e1", "timestamp_s": 11.4}, {"match_id": "m", "event_id": "e2", "timestamp_s": np.nan}])
    assert not resolve_event_citation(ev, "m", "e1", float("nan")).ok and not resolve_event_citation(ev, "m", "e2", 999.0).ok
    st = pd.DataFrame([{"match_id": "m", "state_id": "s", "start_s": np.nan, "end_s": np.nan}])
    assert not resolve_state_citation(st, "m", "s", 1.0, 2.0).ok


def test_the_number_check_catches_altered_summaries():
    """Mutation test on generated summaries: nudge each number by a step of its own precision; at least 95% caught."""
    import re

    res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), PhasePackage(), LineBreakPackage(), MovementPackage(),
                    SpaceValuePackage(), FifaViewPackage(), StatesPackage()]).run(eleven_a_side(seconds=120.0))
    rows = res.outputs["p8"].segments["states"].to_dict("records")
    rows += [{**r, "passes": 6, "passes_completed": 4, "passes_breaking_a_line": 2, "engagements": 3, "runs_in_behind": 1} for r in rows]
    rows += [{**r, "phase": "build_up", "opposed_pct": 35, "duration_s": 7.4, "carrier_pressure_max": 6.5, "passes": 11,
              "passes_completed": 9, "passes_breaking_a_line": 3, "engagements": 5, "counter_press": True, "runs_in_behind": 4} for r in rows[:1]]
    tried = caught = 0
    for r in rows:
        text = summary_text(r)
        assert check_quoted_numbers(text, r) == [], text
        for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?![\w:]|\.\d)", text):
            tok = m.group(0)
            if text[max(0, m.start() - 2):m.end() + 9].find(" to 1") >= 0 or tok in ("1", "10") and "scale" in text[m.end():m.end() + 12]:
                continue  # a scale's own ends are constants, not claims
            step = 10 ** -len(tok.split(".")[1]) if "." in tok else 1
            for d in (step, -step):
                new = f"{float(tok) + d:.{len(tok.split('.')[1]) if '.' in tok else 0}f}"
                if float(new) < 0:
                    continue
                tried += 1
                caught += bool(check_quoted_numbers(text[:m.start()] + new + text[m.end():], r))
    assert tried >= 20 and caught / tried >= 0.95, (caught, tried)
