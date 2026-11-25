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
    # SkillCorner's extrapolated tracking fills in players off camera (24 November 2026): about 59% of positions
    # are observed on the pitch and 83% within 15 m of the ball, so every position-based metric is Degraded,
    # where the panel (which did not know) said Full. The panel's other differences stand.
    assert 0.55 < inputs["observed_share"] < 0.65 and 0.78 < inputs["observed_near_ball_share"] < 0.88
    assert 0.55 < report.coverage["observed_position_share"] < 0.65
    states = report.states()
    estimated = {"pressure_1v1", "pressure_zonal", "control_zones", "movement_value", "pass_option_value",
                 "decision_quality", "pass_completion", "decision_quality_v2", "gain_v2"}
    for mid in estimated:
        assert states[mid] == "degraded", f"{mid}: {states[mid]} (inputs {inputs})"
    assert "estimates, not observations" in report.metrics["control_zones"].reason
    assert "within 15 m of the ball" in report.metrics["pressure_1v1"].reason
    for state, ids in SKILLCORNER_PANEL.items():
        for mid in ids - estimated:
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


def test_decision_quality_v2_follows_pass_completion_and_the_event_feed():
    sc = classify(SKILLCORNER_OPEN_DATA).metrics["decision_quality_v2"]
    assert sc.state == "degraded" and "derived from tracking" in sc.reason  # as v1, events derived from tracking
    logged = classify(ParityInputs(provider="skillcorner", mode="full_tracking", sample_rate_hz=10.0, teams_tracked=("away", "home"),
                                   player_dropout=0.0, has_events=True, has_shot_events=True))
    assert logged.states()["decision_quality_v2"] == "full"
    metrica = classify(ParityInputs(provider="metrica", mode="full_tracking", sample_rate_hz=25.0, teams_tracked=("away", "home"),
                                    player_dropout=0.0, has_events=True, has_shot_events=True)).metrics["decision_quality_v2"]
    assert metrica.state == "degraded" and "pass completion probability" in metrica.reason and metrica.unlock
    assert classify(ParityInputs(provider="statsbomb", mode="event_only", has_events=True, has_freeze_frames=True)).states()["decision_quality_v2"] == "blocked"
    assert classify(ParityInputs(provider="metrica", mode="full_tracking", sample_rate_hz=25.0, teams_tracked=("away", "home"))).states()["decision_quality_v2"] == "blocked"


def test_gain_v2_has_choices_state_everywhere():
    for inputs in (SKILLCORNER_OPEN_DATA,
                   ParityInputs(provider="metrica", mode="full_tracking", sample_rate_hz=25.0, teams_tracked=("away", "home"),
                                player_dropout=0.0, has_events=True, has_shot_events=True),
                   ParityInputs(provider="statsbomb", mode="event_only", has_events=True, has_freeze_frames=True),
                   ParityInputs(provider="statsports", mode="gps_only", sample_rate_hz=10.0, teams_tracked=("home",))):
        r = classify(inputs).metrics
        assert (r["gain_v2"].state, r["gain_v2"].reason) == (r["decision_quality_v2"].state, r["decision_quality_v2"].reason)


def _optical(**kw) -> ParityInputs:
    base = dict(provider="metrica", mode="full_tracking", sample_rate_hz=25.0, teams_tracked=("away", "home"),
                player_dropout=0.0, has_events=True, has_shot_events=True)
    return ParityInputs(**(base | kw))


def test_estimated_positions_count_as_missing_ones_near_the_ball_and_on_the_pitch():
    unflagged = classify(_optical())
    assert classify(_optical(observed_share=1.0, observed_near_ball_share=1.0)).states() == unflagged.states()
    # Mostly seen near the ball, mostly estimated elsewhere: 1v1 pressure stays, whole-pitch metrics degrade.
    r = classify(_optical(observed_share=0.6, observed_near_ball_share=0.95))
    assert r.states()["pressure_1v1"] == "full"
    for mid in ("pressure_zonal", "control_zones", "movement_value", "pass_option_value", "decision_quality"):
        m = r.metrics[mid]
        assert m.state == "degraded" and "40.0% of player positions on the pitch" in m.reason and "full-pitch" in m.unlock
    # Where another limit already degrades a metric (pass completion fitted elsewhere), its reason leads and the
    # estimated share is still in the measure.
    assert r.states()["gain_v2"] == "degraded" and "40.0% of positions on the pitch estimated" in r.metrics["gain_v2"].measure
    assert "5.0% of positions within 15 m of the ball estimated" in r.metrics["pressure_1v1"].measure
    assert "40.0% of positions are the provider's estimates" in r.metrics["physical_summary"].reason
    # Estimated near the ball too: 1v1 pressure degrades.
    assert classify(_optical(observed_share=0.6, observed_near_ball_share=0.8)).states()["pressure_1v1"] == "degraded"
    # At the bar (10% estimated) a metric stays Full, as for dropout.
    assert classify(_optical(observed_share=0.9, observed_near_ball_share=0.9)).states()["control_zones"] == "full"


def test_unflagged_reports_keep_their_coverage_and_reasons():
    r = classify(_optical())
    assert set(r.coverage) == {"ball_detected_share", "player_observation_share"}
    assert r.metrics["control_zones"].reason == "Both teams are tracked throughout."
    flagged = classify(_optical(observed_share=0.6, observed_near_ball_share=0.9))
    assert flagged.coverage["observed_position_share"] == 0.6 and flagged.coverage["observed_near_ball_share"] == 0.9
    assert flagged.to_dict()["schema_version"] == "1.1"
