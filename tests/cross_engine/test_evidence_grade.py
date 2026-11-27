"""The citation's evidence grade (review AI-03, backlog B): computed from the row, checked by the verifier."""

import pandas as pd
import pytest
from matrix_engine.match import with_observed
from matrix_engine.parity import ParityInputs, classify
from plaiq_engine.citations import (
    check_grade,
    evidence_grade,
    match_states,
    resolve_event_citation,
)
from plaiq_engine.match_analysis import MatchAnalysis
from plaiq_engine.semantic_builder import SemanticBuilder
from plaiq_engine.teams import TeamRegistry
from synthetic_match import FPS, synthetic_match


def _built(match):
    parity = classify(ParityInputs(provider="synthetic", mode="full_tracking", sample_rate_hz=FPS, teams_tracked=("away", "home"),
                                   player_dropout=0.0, has_events=True, has_shot_events=False))
    return parity, SemanticBuilder(MatchAnalysis(match), parity, teams=TeamRegistry.for_reference()).semantic_events()


@pytest.fixture(scope="module")
def plain():
    return _built(synthetic_match())


def test_the_grade_comes_from_the_row_and_the_data_check(plain):
    parity, sem = plain
    states = match_states(parity)
    row = sem.loc[sem["dq_score"].notna()].iloc[0]
    g = evidence_grade(row, "dq_score", states)
    assert g == {"field": "dq_score", "data_check": parity.metrics["decision_quality"].state, "label": "estimated",
                 "validation": "outcome_measured", "observed_near_ball": "not_flagged"}  # the synthetic feed flags no estimates
    assert evidence_grade(row, "action_type", states)["label"] == "observed"
    assert evidence_grade(row, "opposition_pressure", states)["validation"] == "reference_agrees"


def test_the_verifier_rejects_a_grade_the_row_does_not_give(plain):
    parity, sem = plain
    states = match_states(parity)
    row = sem.loc[sem["dq_score"].notna()].iloc[0]
    good = evidence_grade(row, "dq_score", states)
    assert check_grade(good, row, states) == []
    assert check_grade({**good, "validation": "analyst_validated"}, row, states)  # a model claiming more than the row
    assert check_grade({**good, "data_check": "full", "field": "dq_score"}, row, {**states, "decision_quality": "degraded"})
    assert check_grade({"data_check": "full"}, row, states)  # no field named
    assert check_grade({**good, "field": "event_id"}, row, states)  # identifiers are cited, never quoted
    ok = resolve_event_citation(sem, row["match_id"], row["event_id"], float(row["timestamp_s"]), grade=good, states=states)
    assert ok.ok and ok.data_check == good["data_check"]
    bad = resolve_event_citation(sem, row["match_id"], row["event_id"], float(row["timestamp_s"]),
                                 grade={**good, "label": "observed"}, states=states)
    assert not bad.ok and "evidence grade" in bad.reason


def test_estimated_positions_near_the_ball_lower_the_band():
    """Where the provider flags estimates, the band reads the row's own near-ball share, not a constant (the review's
    finding: ball_detected and players_observed were constant on broadcast tracking)."""
    m = synthetic_match()
    away = sorted({p for p in m.tracking["player_id"] if p.startswith("away_")})
    est = pd.DataFrame([(f, p) for f in m.frames["frame"] for p in away], columns=["frame", "player_id"])
    parity, sem = _built(with_observed(m, est))
    share = sem["observed_near_ball_share"].dropna()
    assert len(share) and share.between(0, 1).all() and (share < 0.9).all()
    row = sem.loc[sem["observed_near_ball_share"].notna()].iloc[0]
    assert evidence_grade(row, "opposition_pressure", match_states(parity))["observed_near_ball"] == "partly_estimated"
    _, clean = _built(with_observed(m, est.iloc[:0]))
    assert clean["observed_near_ball_share"].dropna().eq(1.0).all()


def test_an_unreported_state_is_never_full(plain):
    _, sem = plain
    row = sem.iloc[0]
    assert evidence_grade(row, "phase", {})["data_check"] == "degraded"
