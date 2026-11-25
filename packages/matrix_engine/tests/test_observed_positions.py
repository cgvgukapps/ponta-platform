"""Positions a broadcast provider estimated rather than observed (schema column ``observed``, 24 November 2026)."""

import numpy as np
import pandas as pd
import pytest
from matrix_engine.ingestion.skillcorner_io import estimated_positions
from matrix_engine.match import NormalisedMatch, with_observed
from matrix_engine.pitch_config import PitchConfig
from matrix_engine.schema import SchemaError, coerce_schema, validate_schema


def _match(n_frames: int = 4) -> NormalisedMatch:
    f = np.arange(n_frames)
    rows = []
    for pid, team, x in (("home_9", "home", 2.0), ("away_4", "away", 40.0)):
        rows.append(pd.DataFrame({"timestamp": f / 25.0, "frame": f, "player_id": pid, "team": team, "x": x, "y": 0.0,
                                  "source": "skillcorner", "sample_rate_hz": 25.0}))
    tracking = validate_schema(coerce_schema(pd.concat(rows, ignore_index=True)))
    return NormalisedMatch(provider="skillcorner", match_id="t", fps=25.0, pitch=PitchConfig(fps=25.0),
                           frames=pd.DataFrame({"frame": f, "period": 1, "time_s": f / 25.0}), tracking=tracking,
                           ball=pd.DataFrame({"frame": f, "x": 0.0, "y": 0.0}), events=pd.DataFrame())


def test_a_source_that_does_not_flag_estimates_reports_none():
    m = _match()
    assert m.observed_share() is None and m.observed_near_ball_share(15.0) is None


def test_estimated_rows_are_marked_and_measured_on_the_pitch_and_near_the_ball():
    # The away player (40 m from the ball) is estimated in every frame, the home player (2 m) in one of four.
    est = pd.DataFrame({"frame": [0, 1, 2, 3, 0], "player_id": ["away_4"] * 4 + ["home_9"]})
    m = with_observed(_match(), est)
    assert m.tracking["observed"].dtype == bool and len(m.tracking) == 8
    assert m.observed_share() == pytest.approx(3 / 8)
    assert m.observed_near_ball_share(15.0) == pytest.approx(3 / 4)   # only the home player is within 15 m
    assert m.tracking[["x", "y"]].equals(_match().tracking[["x", "y"]])  # positions unchanged


def test_the_schema_accepts_a_boolean_observed_column_only():
    tr = _match().tracking
    validate_schema(tr.assign(observed=True))
    with pytest.raises(SchemaError, match="observed"):
        validate_schema(tr.assign(observed=1.0))


def test_skillcorner_estimated_positions_use_the_tracking_ids():
    home = [{"player_id": 11, "trackable_object": 111, "shirt": "9"}, {"player_id": 12, "trackable_object": None, "shirt": "?"}]
    away = [{"player_id": 21, "trackable_object": 221, "shirt": "4"}]
    records = [
        {"frame": 10, "player_data": [{"player_id": 11, "x": 1.0, "y": 0.0, "is_detected": True},
                                      {"player_id": 21, "x": 5.0, "y": 0.0, "is_detected": False}]},
        {"frame": 11, "player_data": [{"player_id": 111, "x": 1.0, "y": 0.0, "is_detected": False},   # by trackable object
                                      {"player_id": 12, "x": 2.0, "y": 0.0, "is_detected": False},    # no shirt: not tracked
                                      {"player_id": 21, "x": None, "y": None, "is_detected": False}]},  # no position
    ]
    est = estimated_positions(records, home, away)
    assert list(zip(est["frame"], est["player_id"], strict=True)) == [(10, "away_4"), (11, "home_9")]
