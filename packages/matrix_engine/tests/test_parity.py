"""The Analytics Parity Report rules (extraction spec, step 5)."""

import json
import os

import pytest
from matrix_engine.parity import (
    METRICS,
    ParityInputs,
    classify,
    parity_for_match,
)

#: The prototype's SkillCorner "Analytics Parity" panel (app.py, SkillCorner sidebar),
#: mapped onto metric ids. "Frame scrubber" and "Player Performance DQ timeline" in the
#: panel are interface features, not metrics: the timeline follows decision_quality.
SKILLCORNER_PANEL = {
    "full": {"hex_grid", "control_zones", "pressure_1v1", "pressure_zonal", "xt_surface", "movement_value"},
    "degraded": {"pass_option_value", "decision_quality", "event_delta_v"},
    "blocked": {"shot_quality"},
}

SKILLCORNER_OPEN_DATA = ParityInputs(
    provider="skillcorner", mode="full_tracking", sample_rate_hz=10.0, teams_tracked=("away", "home"),
    ball_detected_share=0.74, player_dropout=0.0, has_events=True, has_shot_events=False,
    events_derived_from_tracking=True,
)


def test_reproduces_the_prototype_skillcorner_panel_exactly():
    states = classify(SKILLCORNER_OPEN_DATA).states()
    for state, ids in SKILLCORNER_PANEL.items():
        for mid in ids:
            assert states[mid] == state, f"{mid}: {states[mid]} != {state}"


def test_gps_only_blocks_opposition_pressure_and_control_and_keeps_physical():
    r = classify(ParityInputs(provider="statsports", mode="gps_only", sample_rate_hz=10.0, teams_tracked=("home",)))
    s = r.states()
    for mid in ("pressure_1v1", "pressure_zonal", "control_zones", "decision_quality", "pass_option_value", "movement_value"):
        assert s[mid] == "blocked"
    assert s["physical_summary"] == "full"


def test_event_only_degrades_decision_quality_as_a_separate_class():
    r = classify(ParityInputs(provider="statsbomb", mode="event_only", has_events=True, has_shot_events=True, has_freeze_frames=True))
    dq = r.metrics["decision_quality"]
    assert dq.state == "degraded"
    assert "separate confidence class" in dq.reason
    assert r.states()["movement_value"] == "blocked"
    assert r.states()["shot_quality"] == "full"


def test_event_only_without_freeze_frames_blocks_positional_metrics():
    s = classify(ParityInputs(provider="statsbomb", mode="event_only", has_events=True, has_shot_events=True)).states()
    assert s["control_zones"] == "blocked" and s["pressure_1v1"] == "blocked"


@pytest.mark.parametrize(
    ("rate", "dropout", "expect"),
    [(25.0, 0.02, "full"), (5.0, 0.0, "degraded"), (25.0, 0.25, "degraded")],
)
def test_optical_low_rate_or_high_dropout_degrades_with_the_measured_figure(rate, dropout, expect):
    r = classify(ParityInputs(provider="metrica", mode="full_tracking", sample_rate_hz=rate, teams_tracked=("away", "home"),
                              player_dropout=dropout, has_events=True, has_shot_events=True))
    pc = r.metrics["pressure_1v1"]
    assert pc.state == expect
    assert f"{rate:g} Hz" in pc.measure
    if expect == "degraded":
        assert ("Hz" in pc.reason) or ("%" in pc.reason)


def test_no_physiological_source_blocks_load_and_fatigue_in_every_mode():
    for inputs in (
        SKILLCORNER_OPEN_DATA,
        ParityInputs(provider="statsports", mode="gps_only", teams_tracked=("home",)),
        ParityInputs(provider="statsbomb", mode="event_only", has_events=True),
    ):
        assert classify(inputs).states()["load_and_fatigue"] == "blocked"


def test_report_is_json_serialisable_and_complete():
    d = classify(SKILLCORNER_OPEN_DATA).to_dict()
    json.dumps(d)
    assert set(d["metrics"]) == set(METRICS)
    assert sum(d["summary"].values()) == len(METRICS)
    for m in d["metrics"].values():
        assert m["state"] in {"full", "degraded", "blocked"} and m["reason"]
    assert d["engine_version"].startswith("matrix-")


def test_classification_is_deterministic():
    assert classify(SKILLCORNER_OPEN_DATA).to_dict() == classify(SKILLCORNER_OPEN_DATA).to_dict()


#: Finding (15 November 2026): the SkillCorner open-data events contain shots (23 SHOT
#: events mapped by the adapter from dynamic events), so the panel's premise "no SHOT
#: events" no longer holds for this match. Measured, two metrics come out Full where the
#: panel said Degraded. Recorded as a decision for a human (EXTRACTION_REPORT.md); the
#: rules are not bent to match stale text.
MEASURED_DIFFERS_FROM_PANEL = {"pass_option_value": "full", "event_delta_v": "full"}


def test_measured_inputs_on_the_skillcorner_sample_match(data_dir):
    from matrix_engine.match import load_skillcorner_match

    match = load_skillcorner_match(os.path.join(data_dir, "skillcorner", "matches", "1886347"), 1886347)
    report = parity_for_match(match)
    inputs = report.inputs
    assert inputs["sample_rate_hz"] == 10.0
    assert inputs["has_shot_events"] is True
    assert inputs["events_derived_from_tracking"] is True
    # The ball is undetected in about a quarter of this match's frames (golden manifest).
    assert 0.70 < inputs["ball_detected_share"] < 0.78
    states = report.states()
    for state, ids in SKILLCORNER_PANEL.items():
        for mid in ids:
            expected = MEASURED_DIFFERS_FROM_PANEL.get(mid, state)
            assert states[mid] == expected, f"{mid}: {states[mid]} != {expected} (inputs {inputs})"
    # Shots derived from tracking still do not support shot quality.
    assert "derived from tracking" in report.metrics["shot_quality"].reason


def test_pass_completion_is_full_only_where_its_parameters_were_fitted():
    assert classify(SKILLCORNER_OPEN_DATA).states()["pass_completion"] == "full"
    metrica = classify(ParityInputs(provider="metrica", mode="full_tracking", sample_rate_hz=25.0, teams_tracked=("away", "home"),
                                    player_dropout=0.0, has_events=True, has_shot_events=True))
    pc = metrica.metrics["pass_completion"]
    assert pc.state == "degraded" and "another source" in pc.reason and pc.unlock
    assert classify(ParityInputs(provider="statsbomb", mode="event_only", has_events=True, has_freeze_frames=True)).states()["pass_completion"] == "blocked"
    assert classify(ParityInputs(provider="metrica", mode="full_tracking", sample_rate_hz=25.0, teams_tracked=("away", "home"))).states()["pass_completion"] == "blocked"
