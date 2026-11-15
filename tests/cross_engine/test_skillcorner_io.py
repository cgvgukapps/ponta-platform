"""
test_skillcorner_io.py — Unit tests for ingestion/skillcorner_io.py.

Tests cover:
  - Metadata parsing (load_skillcorner_meta)
  - Tracking-to-wide conversion (tracking_to_wide)
  - Playing direction detection and normalisation
  - Dynamic events mapping (load_skillcorner_events)
  - Event coordinate scaling and frame alignment
  - SkillCornerAdapter end-to-end load()
  - Roster building (build_skillcorner_roster)
  - calc_player_velocities fps parameter (auto-scaled window)
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from matrix_engine.ingestion.skillcorner_io import (
    SkillCornerAdapter,
    load_skillcorner_meta,
    load_skillcorner_events,
    tracking_to_wide,
    SC_FPS,
    SC_NOMINAL_PITCH_LENGTH,
    SC_NOMINAL_PITCH_WIDTH,
)
from skillcorner_fixture import (
    MATCH_ID,
    HOME_ID,
    AWAY_ID,
    HOME_GK_ID,
    HOME_DF_ID,
    HOME_FW_ID,
    AWAY_GK_ID,
    AWAY_DF_ID,
    P1_START,
    P1_END,
    P2_START,
    P2_END,
    make_match_json,
    make_tracking_records,
    make_dynamic_events_csv,
    write_fixture_dir,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _tmp_match_json(tmp_path, **kwargs) -> str:
    data = make_match_json(**kwargs)
    p = tmp_path / f"{MATCH_ID}_match.json"
    p.write_text(json.dumps(data))
    return str(p)


def _tmp_events_csv(tmp_path) -> str:
    p = tmp_path / f"{MATCH_ID}_dynamic_events.csv"
    p.write_text(make_dynamic_events_csv())
    return str(p)


def _periods() -> dict[int, tuple[int, int]]:
    return {1: (P1_START, P1_END), 2: (P2_START, P2_END)}


# ===========================================================================
# Phase 0 — load_skillcorner_meta
# ===========================================================================

class TestLoadSkillcornerMeta:
    def test_parses_home_away_players(self, tmp_path):
        path = _tmp_match_json(tmp_path)
        meta = load_skillcorner_meta(path)
        assert len(meta["home_players"]) == 3, "Expected 3 home players"
        assert len(meta["away_players"]) == 2, "Expected 2 away players"

    def test_pitch_dimensions(self, tmp_path):
        path = _tmp_match_json(tmp_path)
        meta = load_skillcorner_meta(path)
        assert meta["pitch_length"] == 104.0
        assert meta["pitch_width"]  == 68.0

    def test_h1_home_sign_right_to_left(self, tmp_path):
        """home_team_side[0]='right_to_left' → home attacks -x in H1 → sign=-1."""
        path = _tmp_match_json(tmp_path, home_attacks_right_in_h1=False)
        meta = load_skillcorner_meta(path)
        assert meta["h1_home_sign"] == -1

    def test_h1_home_sign_left_to_right(self, tmp_path):
        """home_team_side[0]='left_to_right' → home attacks +x in H1 → sign=+1."""
        path = _tmp_match_json(tmp_path, home_attacks_right_in_h1=True)
        meta = load_skillcorner_meta(path)
        assert meta["h1_home_sign"] == +1

    def test_periods_present(self, tmp_path):
        path = _tmp_match_json(tmp_path)
        meta = load_skillcorner_meta(path)
        assert 1 in meta["periods"] and 2 in meta["periods"]
        assert meta["periods"][1] == (P1_START, P1_END)
        assert meta["periods"][2] == (P2_START, P2_END)

    def test_player_jersey_numbers(self, tmp_path):
        path = _tmp_match_json(tmp_path)
        meta = load_skillcorner_meta(path)
        home_jerseys = {p["shirt"] for p in meta["home_players"]}
        assert {"1", "5", "9"} == home_jerseys

    def test_gk_position_normalised(self, tmp_path):
        path = _tmp_match_json(tmp_path)
        meta = load_skillcorner_meta(path)
        gk = next(p for p in meta["home_players"] if p["shirt"] == "1")
        assert gk["position_type"] == "GK"

    def test_downstream_compat_keys(self, tmp_path):
        """Keys used by second_spectrum_io / build_roster must be present."""
        path = _tmp_match_json(tmp_path)
        meta = load_skillcorner_meta(path)
        for key in ("team_a_id", "team_b_id", "team_a_name", "team_b_name",
                    "team_a_players", "team_b_players", "frame_rate"):
            assert key in meta, f"Missing key: {key}"


# ===========================================================================
# Phase 1c — tracking_to_wide shape and values
# ===========================================================================

class TestTrackingToWide:
    def _load(self, **kwargs):
        path = None  # not needed — use in-memory records
        records = make_tracking_records(**kwargs)
        meta = make_match_json(home_attacks_right_in_h1=False)  # h1_sign=-1
        home_players = [p for p in meta["players"] if p["team_id"] == HOME_ID]
        away_players = [p for p in meta["players"] if p["team_id"] == AWAY_ID]

        from matrix_engine.ingestion.skillcorner_io import _parse_players
        hp = _parse_players(home_players)
        ap = _parse_players(away_players)
        return tracking_to_wide(records, hp, ap, h1_home_sign=-1)

    def test_dataframe_shape(self):
        home, away = self._load(n_p1_frames=5, n_p2_frames=5)
        assert len(home) == 10
        assert len(away) == 10

    def test_index_name(self):
        home, away = self._load()
        assert home.index.name == "Frame"
        assert away.index.name == "Frame"

    def test_home_columns_present(self):
        home, _ = self._load()
        for jersey in ("1", "5", "9"):
            assert f"Home_{jersey}_x" in home.columns
            assert f"Home_{jersey}_y" in home.columns

    def test_away_columns_present(self):
        _, away = self._load()
        for jersey in ("1", "5"):
            assert f"Away_{jersey}_x" in away.columns

    def test_ball_on_away_df(self):
        _, away = self._load()
        assert "ball_x" in away.columns
        assert "ball_y" in away.columns

    def test_period_column(self):
        home, _ = self._load(n_p1_frames=5, n_p2_frames=5)
        assert set(home["Period"].unique()) == {1, 2}

    def test_time_column_present(self):
        home, _ = self._load()
        assert "Time [s]" in home.columns

    def test_second_half_x_negated(self):
        """
        Period 2 coords should be negated relative to period 1 when h1_sign=-1.
        With gk_x_p1=-40 and h1_sign=-1:
          P1 normalised: gk_x = -40 * (-1) = +40
          P2 raw fixture uses gk_x_sign=-1 → raw_x = (-40 * -1) = +40
          P2 normalised: sign=-(-1)=+1 → 40 * 1 = +40  (same side = correct)
        """
        records = make_tracking_records(gk_x_p1=-40.0, n_p1_frames=3, n_p2_frames=3)
        from matrix_engine.ingestion.skillcorner_io import _parse_players
        match = make_match_json(home_attacks_right_in_h1=False)
        hp = _parse_players([p for p in match["players"] if p["team_id"] == HOME_ID])
        ap = _parse_players([p for p in match["players"] if p["team_id"] == AWAY_ID])

        home_n, _ = tracking_to_wide(records, hp, ap, h1_home_sign=-1, normalize_direction=True)
        home_raw, _ = tracking_to_wide(records, hp, ap, h1_home_sign=-1, normalize_direction=False)

        p1_frames = home_n[home_n["Period"] == 1]
        p2_frames = home_n[home_n["Period"] == 2]

        # After normalisation, GK should be at +40 in both halves
        assert (p1_frames["Home_1_x"] > 0).all(), "GK should be at +x in P1 after flip"
        assert (p2_frames["Home_1_x"] > 0).all(), "GK should be at +x in P2 after flip"

    def test_no_normalisation(self):
        records = make_tracking_records(gk_x_p1=-40.0, n_p1_frames=3, n_p2_frames=3)
        from matrix_engine.ingestion.skillcorner_io import _parse_players
        match = make_match_json()
        hp = _parse_players([p for p in match["players"] if p["team_id"] == HOME_ID])
        ap = _parse_players([p for p in match["players"] if p["team_id"] == AWAY_ID])

        home, _ = tracking_to_wide(records, hp, ap, h1_home_sign=-1, normalize_direction=False)
        p1 = home[home["Period"] == 1]
        # Raw coords preserved: gk_x_p1 = -40
        assert (p1["Home_1_x"] < 0).all()


# ===========================================================================
# Phase 1e — load_skillcorner_events
# ===========================================================================

class TestLoadSkillcornerEvents:
    def _load(self, tmp_path) -> pd.DataFrame:
        csv_path = _tmp_events_csv(tmp_path)
        return load_skillcorner_events(
            csv_path,
            home_team_id=HOME_ID,
            away_team_id=AWAY_ID,
            periods=_periods(),
            fps=SC_FPS,
            pitch_length=104.0,
            pitch_width=68.0,
            h1_home_sign=-1,
        )

    def test_passing_option_skipped(self, tmp_path):
        """passing_option events must not appear in PLAIQ events."""
        df = self._load(tmp_path)
        # passing_option player is AWAY_GK_ID (player_id=2001)
        assert str(AWAY_GK_ID) not in df["From"].astype(str).values

    def test_successful_pass_mapped(self, tmp_path):
        df = self._load(tmp_path)
        passes = df[df["Type"] == "PASS"]
        assert len(passes) >= 1

    def test_failed_pass_mapped_to_ball_lost(self, tmp_path):
        df = self._load(tmp_path)
        ball_lost = df[df["Type"] == "BALL LOST"]
        assert len(ball_lost) >= 1

    def test_carry_mapped(self, tmp_path):
        df = self._load(tmp_path)
        carries = df[df["Type"] == "CARRY"]
        assert len(carries) >= 1

    def test_event_columns_present(self, tmp_path):
        from matrix_engine.ingestion.second_spectrum_io import EVENT_COLUMNS
        df = self._load(tmp_path)
        for col in EVENT_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_team_side_assigned(self, tmp_path):
        df = self._load(tmp_path)
        home_events = df[df["Team"] == "Home"]
        away_events = df[df["Team"] == "Away"]
        assert len(home_events) >= 1
        assert len(away_events) >= 1

    def test_start_frame_in_range(self, tmp_path):
        df = self._load(tmp_path)
        all_frames = set(range(P1_START, P1_END + 1)) | set(range(P2_START, P2_END + 1))
        out_of_range = df[~df["Start Frame"].isin(all_frames)]
        assert out_of_range.empty, f"Frames outside expected range: {out_of_range['Start Frame'].values}"

    def test_home_event_coords_not_flipped(self, tmp_path):
        """
        SkillCorner dynamic_events use a team-relative coordinate frame where
        each team's attacking direction is +x.  Home events are therefore
        already in the PLAIQ-normalised frame (home attacks +x) and must NOT
        be negated.  Only Away events require a 180° rotation.
        ev_1 (Home forward) has x_start=30.0, period=1, pitch_length=104.
        Expected Start X = +30.0 * (104/105).
        """
        scale_x = 104.0 / SC_NOMINAL_PITCH_LENGTH
        expected = 30.0 * scale_x          # positive — no sign flip for Home
        df = self._load(tmp_path)
        ev1 = df[(df["From"] == str(HOME_FW_ID)) & (df["Type"] == "CARRY")]
        if not ev1.empty:
            assert ev1.iloc[0]["Start X"] == pytest.approx(expected, abs=1e-3)

    def test_missing_events_file_returns_empty(self, tmp_path):
        df = load_skillcorner_events(
            str(tmp_path / "nonexistent.csv"),
            home_team_id=HOME_ID,
            away_team_id=AWAY_ID,
            periods=_periods(),
        )
        assert df.empty

    def test_frame_time_alignment(self, tmp_path):
        """
        Start Time [s] must equal _period_time_seconds(frame, period, periods, fps).
        """
        from matrix_engine.ingestion.second_spectrum_io import _period_time_seconds
        df = self._load(tmp_path)
        for _, row in df.iterrows():
            expected_t = _period_time_seconds(
                int(row["Start Frame"]), int(row["Period"]), _periods(), SC_FPS
            )
            assert row["Start Time [s]"] == pytest.approx(expected_t, abs=1e-4)


# ===========================================================================
# Phase 1f — SkillCornerAdapter end-to-end
# ===========================================================================

class TestSkillCornerAdapter:
    def test_load_returns_four_items(self, tmp_path):
        match_dir = write_fixture_dir(str(tmp_path))
        adapter = SkillCornerAdapter(match_dir, MATCH_ID)
        result = adapter.load()
        assert len(result) == 4

    def test_home_df_has_frame_index(self, tmp_path):
        match_dir = write_fixture_dir(str(tmp_path))
        home, _, _, _ = SkillCornerAdapter(match_dir, MATCH_ID).load()
        assert home.index.name == "Frame"

    def test_meta_has_provider_key(self, tmp_path):
        match_dir = write_fixture_dir(str(tmp_path))
        _, _, _, meta = SkillCornerAdapter(match_dir, MATCH_ID).load()
        assert meta.get("provider") == "skillcorner"

    def test_meta_has_pitch_config(self, tmp_path):
        from matrix_engine.pitch_config import PitchConfig
        match_dir = write_fixture_dir(str(tmp_path))
        _, _, _, meta = SkillCornerAdapter(match_dir, MATCH_ID).load()
        assert isinstance(meta.get("pitch"), PitchConfig)

    def test_events_not_empty(self, tmp_path):
        match_dir = write_fixture_dir(str(tmp_path))
        _, _, events, _ = SkillCornerAdapter(match_dir, MATCH_ID).load()
        assert not events.empty

    def test_no_normalisation_flag(self, tmp_path):
        match_dir = write_fixture_dir(str(tmp_path))
        adapter = SkillCornerAdapter(match_dir, MATCH_ID, normalize_playing_direction=False)
        home, _, _, _ = adapter.load()
        # Without normalisation, P1 GK at -40 stays at -40
        p1 = home[home["Period"] == 1]
        gk_col = "Home_1_x"
        assert gk_col in p1.columns
        # Raw value: gk_x_p1 = -40 (fixture default)
        assert (p1[gk_col] < 0).all()


# ===========================================================================
# Phase 2 — build_skillcorner_roster
# ===========================================================================

class TestBuildSkillcornerRoster:
    def _build(self, tmp_path):
        match_dir = write_fixture_dir(str(tmp_path))
        home, away, _, meta = SkillCornerAdapter(match_dir, MATCH_ID).load()
        from plaiq_engine.roster.build_roster import build_skillcorner_roster
        return build_skillcorner_roster(MATCH_ID, meta, home, away)

    def test_returns_dict_with_teams(self, tmp_path):
        roster = self._build(tmp_path)
        assert "teams" in roster
        assert "Home" in roster["teams"]
        assert "Away" in roster["teams"]

    def test_home_player_count(self, tmp_path):
        roster = self._build(tmp_path)
        # 3 home players in fixture; only those with data in tracking counted
        home_players = roster["teams"]["Home"]["players"]
        assert len(home_players) == 3

    def test_away_player_count(self, tmp_path):
        roster = self._build(tmp_path)
        away_players = roster["teams"]["Away"]["players"]
        assert len(away_players) == 2

    def test_player_has_required_keys(self, tmp_path):
        roster = self._build(tmp_path)
        p = roster["teams"]["Home"]["players"][0]
        for key in ("player_id", "display_name", "position", "primary_slot",
                    "v_max", "a_max", "reaction_time", "spells"):
            assert key in p, f"Missing key: {key}"

    def test_provider_field(self, tmp_path):
        roster = self._build(tmp_path)
        assert roster.get("provider") == "skillcorner"

    def test_player_id_is_string(self, tmp_path):
        roster = self._build(tmp_path)
        for p in roster["teams"]["Home"]["players"]:
            assert isinstance(p["player_id"], str), "player_id must be string for event join"


# ===========================================================================
# Phase 3a — calc_player_velocities fps auto-scaling
# ===========================================================================

class TestCalcPlayerVelocitiesFps:
    def _make_tracking(self, fps):
        """Small synthetic wide-format tracking DF at given fps."""
        n = 40
        t = np.arange(n) / fps
        frames = np.arange(n)
        data = {
            "Period":   np.where(frames < n // 2, 1, 2),
            "Time [s]": t,
            "Home_1_x": np.sin(2 * np.pi * t / 5),
            "Home_1_y": np.cos(2 * np.pi * t / 5),
        }
        df = pd.DataFrame(data, index=frames)
        df.index.name = "Frame"
        return df

    def test_velocity_columns_added_at_25fps(self):
        from matrix_engine.ingestion.metrica_velocities import calc_player_velocities
        df = self._make_tracking(25)
        out = calc_player_velocities(df, fps=25.0)
        assert "Home_1_speed" in out.columns

    def test_velocity_columns_added_at_10fps(self):
        from matrix_engine.ingestion.metrica_velocities import calc_player_velocities
        df = self._make_tracking(10)
        out = calc_player_velocities(df, fps=10.0)
        assert "Home_1_speed" in out.columns

    def test_speeds_finite(self):
        from matrix_engine.ingestion.metrica_velocities import calc_player_velocities
        df = self._make_tracking(10)
        out = calc_player_velocities(df, fps=10.0)
        # Allow a few NaN at period boundary but most should be finite
        finite_ratio = out["Home_1_speed"].notna().mean()
        assert finite_ratio > 0.8

    def test_window_auto_scaled(self):
        """At 10 fps, window should be scaled below 7 (auto-computed to 3)."""
        from matrix_engine.ingestion.metrica_velocities import calc_player_velocities
        import scipy.signal as signal

        # Monkey-patch to capture the actual window used
        windows_used = []
        _orig = signal.savgol_filter
        def _capture(x, window_length, polyorder):
            windows_used.append(window_length)
            return _orig(x, window_length, polyorder)
        signal.savgol_filter = _capture

        try:
            df = self._make_tracking(10)
            calc_player_velocities(df, fps=10.0)
        finally:
            signal.savgol_filter = _orig

        # At 10 fps: round(7 * 10/25) = round(2.8) = 3 (then made odd = 3)
        assert all(w <= 5 for w in windows_used), f"Window too large at 10fps: {windows_used}"


# ===========================================================================
# Phase 1a — pitch_config factory
# ===========================================================================

class TestPitchConfigFactory:
    def test_from_skillcorner_meta(self, tmp_path):
        from matrix_engine.pitch_config import PitchConfig
        path = _tmp_match_json(tmp_path)
        meta = load_skillcorner_meta(path)
        pc = PitchConfig.from_skillcorner_meta(meta)
        assert pc.length_m == pytest.approx(104.0)
        assert pc.width_m  == pytest.approx(68.0)
        assert pc.fps      == pytest.approx(10.0)
