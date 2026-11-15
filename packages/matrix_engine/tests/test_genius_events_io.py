"""Tests for the Genius event-spec v1 -> PLAIQ events-CSV adapter."""

import os

import numpy as np
import pandas as pd
import pytest

from matrix_engine.ingestion.genius_events_io import genius_events_to_dataframe, load_genius_events_game
from matrix_engine.ingestion.second_spectrum_io import EVENT_COLUMNS, load_secondspectrum_game
from genius_events_fixture import build_events, write_genius_events_fixture
from ss_fixture import write_ss_fixture


# Period frame bounds matching the tracking fixture (8 frames/half, step 25).
PERIODS = {1: (0, 175), 2: (1000, 1175)}
FPS = 25.0


@pytest.fixture
def events_df():
    return genius_events_to_dataframe(build_events(), periods=PERIODS, fps=FPS)


def test_schema_and_skips(events_df):
    # Columns match the legacy events contract exactly.
    assert list(events_df.columns) == EVENT_COLUMNS
    # 5 positional events kept; admin event + frameless event skipped.
    assert len(events_df) == 5
    assert "DANGER_STATE_CHANGED" not in set(events_df["Type"])


def test_completed_pass_maps_to_pass(events_df):
    row = events_df[events_df["Start Frame"] == 25].iloc[0]
    assert row["Type"] == "PASS"
    assert row["From"] == "h3"
    assert row["To"] == "h4"
    assert row["Team"] == "Home"
    # centre-origin metres passed through unchanged
    assert row["Start X"] == -5.0 and row["Start Y"] == 0.0
    assert row["End X"] == 12.0 and row["End Y"] == 8.0


def test_incomplete_pass_maps_to_ball_lost(events_df):
    row = events_df[events_df["Start Frame"] == 1025].iloc[0]
    assert row["Type"] == "BALL LOST"
    assert row["Team"] == "Away"
    assert row["End X"] == -8.0 and row["End Y"] == 12.0


def test_carry_and_shot_coordinates(events_df):
    carry = events_df[events_df["Type"] == "CARRY"].iloc[0]
    assert carry["Start X"] == 12.0 and carry["End X"] == 20.0

    shot = events_df[events_df["Type"] == "SHOT"].iloc[0]
    # SHOT end uses goalmouthLocation.
    assert shot["End X"] == 52.0


def test_clearance_subtype_excluded_from_failed_pass(events_df):
    clr = events_df[events_df["Start Frame"] == 1075].iloc[0]
    assert clr["Type"] == "BALL LOST"
    assert clr["Subtype"] == "CLEARANCE"


def test_frame_to_time_matches_tracking_convention(events_df):
    # Frame 25 in period 1 at 25 fps -> 1.0s (same formula the tracking loader uses).
    row = events_df[events_df["Start Frame"] == 25].iloc[0]
    assert row["Start Time [s]"] == pytest.approx(1.0)
    # Period 2 frame 1025: 1s into the second half + full first-half duration.
    p2 = events_df[events_df["Start Frame"] == 1025].iloc[0]
    p1_duration = (175 - 0 + 1) / FPS
    assert p2["Start Time [s]"] == pytest.approx(1.0 + p1_duration)


def test_second_half_sign_flip_option():
    flipped = genius_events_to_dataframe(
        build_events(), periods=PERIODS, fps=FPS, second_half_sign_flip=True
    )
    row = flipped[flipped["Start Frame"] == 1025].iloc[0]
    # Period-2 coords negated relative to the raw feed (10.0, -5.0).
    assert row["Start X"] == -10.0 and row["Start Y"] == 5.0


def test_participant_id_bridge():
    df = genius_events_to_dataframe(
        build_events(), periods=PERIODS, fps=FPS,
        participant_id_map={"h3": "TRACK_9"},
    )
    row = df[df["Start Frame"] == 25].iloc[0]
    assert row["From"] == "TRACK_9"


def test_end_to_end_with_tracking_fixture(tmp_path):
    # Tracking + events from coherent fixtures load and align.
    write_ss_fixture(str(tmp_path))
    home, away, _empty, meta = load_secondspectrum_game(str(tmp_path))

    ev_path = write_genius_events_fixture(str(tmp_path / "events.jsonl"))
    events = load_genius_events_game(ev_path, periods=meta["periods"], fps=meta["frame_rate"])

    assert not events.empty
    # Every event frame falls inside a tracking period span.
    p1_lo, p1_hi = meta["periods"][1]
    p2_lo, p2_hi = meta["periods"][2]
    for _, r in events.iterrows():
        f = r["Start Frame"]
        assert (p1_lo <= f <= p1_hi) or (p2_lo <= f <= p2_hi)
    # From-ids are tracking player_ids present in the roster metadata.
    track_ids = {p["player_id"] for p in meta["team_a_players"] + meta["team_b_players"]}
    assert set(events["From"].dropna()) <= track_ids
