"""The field dictionary covers every AI-facing column exactly, and its rules hold (review backlog B: AI-08, AN-10)."""

import json
from collections import Counter
from pathlib import Path

import pytest
from matrix_engine.parity import METRICS
from plaiq_engine import field_dictionary as FD
from plaiq_engine.match_analysis import MatchAnalysis
from plaiq_engine.profiles import CATALOGUE, PROFILE_COLUMNS
from plaiq_engine.semantic_builder import (
    MOVEMENT_COLUMNS,
    SEMANTIC_COLUMNS,
    STATE_EXTRA_COLUMNS,
    SemanticBuilder,
)
from plaiq_engine.teams import TeamRegistry
from synthetic_match import synthetic_match

DOCS = Path(__file__).resolve().parents[2] / "docs"
CONTEXT_CHECKS = {"p1.possession", "p1.possession_state", "p1.game_state", "p2.team_shape", "p2.formation", "p3.relative_zones",
                  "p4.phase_of_play", "p4v.engagements", "p4v.fifa_view", "p4v.fifa_block", "p5.line_breaks", "p5.relationships",
                  "p6.movement", "p7.space", "p7.adjusted_choice", "p8.episodes"}


@pytest.mark.parametrize("table, cols", [("semantic_events", SEMANTIC_COLUMNS), ("semantic_states", STATE_EXTRA_COLUMNS),
                                         ("movement_values", MOVEMENT_COLUMNS), ("profiles", PROFILE_COLUMNS),
                                         ("profile_measures", tuple(m.name for m in CATALOGUE))])
def test_every_column_has_exactly_one_entry(table, cols):
    names = FD.columns(table)
    assert not [n for n, k in Counter(names).items() if k > 1]
    assert set(names) == set(cols), (set(cols) - set(names), set(names) - set(cols))
    FD.check_columns(table, cols)


def test_the_rules_hold():
    for f in FD.fields():
        key = f"{f.table}.{f.name}"
        assert f.table in FD.TABLES and f.kind in FD.KINDS and f.validation in FD.VALIDATION, key
        assert f.definition, key
        if f.validation not in ("no_reference", "not_applicable"):
            assert f.evidence, key  # every claimed check names its figure
        if f.kind == "estimated":
            assert f.validation != "not_applicable", key
        if f.player_facing:  # only observed values reach a player until an analyst validates (Gate 2b)
            assert f.kind == "observed" or f.validation == "analyst_validated", key
        if f.validation == "reference_disagrees":
            assert not f.quotable and not f.player_facing, key
        if f.kind == "identifier":
            assert not f.quotable, key  # cited, never quoted
        if f.data_check is not None:
            assert f.data_check in METRICS or f.data_check in CONTEXT_CHECKS, key
    assert not [f for f in FD.fields() if f.validation == "analyst_validated"]  # none yet: say so if this changes


def test_the_builder_writes_only_dictionary_columns():
    b = SemanticBuilder(MatchAnalysis(synthetic_match()), _parity(), teams=TeamRegistry.for_reference())
    sem = b.semantic_events()
    FD.check_columns("semantic_events", sem.columns)
    FD.check_columns("movement_values", b.movement_values().columns)
    with pytest.raises(ValueError):
        FD.check_columns("semantic_events", [*sem.columns, "ball_cell"])  # a new column under a neutral name is caught
    view = FD.player_facing_view(sem, "semantic_events")
    assert {"action_type", "timestamp_s"} <= set(view.columns)
    assert not {"dq_score", "dq_v2_score", "phase", "opposition_pressure", "event_id"} & set(view.columns)


def test_the_generated_docs_are_current():
    """docs/ai_fields.json is written by tools/field_dictionary.py from the module; it must not drift."""
    assert json.loads((DOCS / "ai_fields.json").read_text()) == json.loads(json.dumps(FD.as_json()))


def _parity():
    from matrix_engine.parity import ParityInputs, classify
    from synthetic_match import FPS

    return classify(ParityInputs(provider="synthetic", mode="full_tracking", sample_rate_hz=FPS, teams_tracked=("away", "home"),
                                 player_dropout=0.0, has_events=True, has_shot_events=False))
