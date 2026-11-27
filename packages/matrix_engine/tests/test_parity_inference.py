"""The Data check infers the mode and checks the ball at passes and carries (schema 1.2, review DE-04)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from matrix_engine.match import NormalisedMatch
from matrix_engine.parity import (
    ball_at_events_share,
    infer_mode,
    inputs_from_match,
    parity_for_match,
)
from matrix_engine.pitch_config import PitchConfig
from matrix_engine.schema import coerce_schema, validate_schema


def _match(teams=("home", "away"), ball=True, events=True, n=50) -> NormalisedMatch:
    f = np.arange(n)
    rows = [pd.DataFrame({"timestamp": f / 10.0, "frame": f, "player_id": f"{t}_{k}", "team": t, "x": float(k), "y": 0.0,
                          "source": "synthetic", "sample_rate_hz": 10.0})
            for t in teams for k in range(1, 12)]
    tracking = validate_schema(coerce_schema(pd.concat(rows, ignore_index=True)))
    ev = pd.DataFrame({"Type": ["PASS"] * 10, "Team": "Home", "Start Frame": np.arange(0, 50, 5), "Start X": 0.0, "Start Y": 0.0})
    return NormalisedMatch(provider="synthetic", match_id="t", fps=10.0, pitch=PitchConfig(fps=10.0),
                           frames=pd.DataFrame({"frame": f, "period": 1, "time_s": f / 10.0}), tracking=tracking,
                           ball=pd.DataFrame({"frame": f, "x": 0.0, "y": 0.0}) if ball else pd.DataFrame(columns=["frame", "x", "y"]),
                           events=ev if events else pd.DataFrame())


def test_a_wearable_export_is_read_as_gps_only():
    """One team (or 'unknown'), no ball: the wearable rules apply, not optical ones with wrong reasons."""
    for teams in (("unknown",), ("home",)):
        m = _match(teams=teams, ball=False, events=False)
        assert infer_mode(m) == "gps_only"
        r = parity_for_match(m)
        assert r.mode == "gps_only" and r.metrics["pressure_1v1"].state == "blocked"
        assert "wearable" in r.metrics["physical_summary"].reason.lower()


def test_a_mode_that_contradicts_the_data_is_refused():
    with pytest.raises(ValueError, match="contradicts"):
        inputs_from_match(_match(teams=("unknown",), ball=False, events=False), mode="full_tracking")
    assert infer_mode(_match()) == "full_tracking"


def test_the_ball_missing_at_passes_degrades_ball_dependent_metrics():
    m = _match()
    assert ball_at_events_share(m) == 1.0
    full = parity_for_match(m)
    lost = dataclasses.replace(m, ball=m.ball.loc[m.ball["frame"] >= 25])  # the ball missing at half the passes
    assert ball_at_events_share(lost) == 0.5
    r = parity_for_match(lost)
    assert r.metrics["decision_quality"].state == "degraded" and "not located at 50.0% of passes" in r.metrics["decision_quality"].reason
    assert r.metrics["gain_v2"].state == r.metrics["decision_quality_v2"].state == "degraded"
    assert r.metrics["pressure_1v1"].state == full.metrics["pressure_1v1"].state  # pressure reads players, not the ball


def test_the_adapter_flag_replaces_the_provider_name():
    m = _match()
    assert not inputs_from_match(m).events_derived_from_tracking
    assert inputs_from_match(dataclasses.replace(m, events_derived_from_tracking=True)).events_derived_from_tracking
    assert inputs_from_match(dataclasses.replace(m, provider="skillcorner")).events_derived_from_tracking  # old caches


def test_reasons_read_as_sentences():
    r = parity_for_match(_match(teams=("home", "away")))
    for mp in r.metrics.values():
        assert "Positions are only" not in mp.reason
