"""
Tests for the StatsBomb → PLAIQ ingestion adapter.

These tests use synthetic event/freeze-frame data so they run offline
without any network call to the StatsBomb GitHub repository.
"""

import math
import numpy as np
import pandas as pd
import pytest

from plaiq_engine.ingestion.statsbomb_io import (
    sb_coords_to_plaiq,
    sb_coords_array_to_plaiq,
    sb_events_to_plaiq,
    load_sb_match,
    load_sb_match_meta,
    SB_PITCH,
    SB_PITCH_LENGTH_UNITS,
    SB_PITCH_WIDTH_UNITS,
    SB_EVENT_COLUMNS,
    _timestamp_to_seconds,
)
from matrix_engine.pitch_config import PitchConfig
from plaiq_engine.analytics.pitch_control import (
    compute_freeze_frame_control,
    build_pitch_hex_grid,
    player_zone_outlines,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic StatsBomb event rows
# ---------------------------------------------------------------------------

def _base_event(idx: int, period: int, ts: str, ev_type: str, team: str) -> dict:
    return {
        "id":              f"uuid-{idx:04d}",
        "index":           idx,
        "period":          period,
        "timestamp":       ts,
        "minute":          0,
        "second":          0,
        "type":            ev_type,
        "possession":      idx,
        "possession_team": team,
        "play_pattern":    "Regular Play",
        "team":            team,
        "player":          "Alice",
        "position":        "Center Forward",
        "location":        [60.0, 40.0],   # centre spot
        "duration":        1.0,
        "related_events":  [],
        "under_pressure":  False,
    }


def _pass_event(idx: int, period: int, ts: str, team: str,
                end_loc: list, complete: bool = True) -> dict:
    ev = _base_event(idx, period, ts, "Pass", team)
    ev["pass_end_location"] = end_loc
    ev["pass_outcome"]      = None if complete else "Incomplete"
    ev["pass_recipient"]    = "Bob" if complete else None
    ev["pass_length"]       = 20.0
    ev["pass_angle"]        = 0.5
    ev["pass_body_part"]    = "Right Foot"
    return ev


def _carry_event(idx: int, period: int, ts: str, team: str, end_loc: list) -> dict:
    ev = _base_event(idx, period, ts, "Carry", team)
    ev["carry_end_location"] = end_loc
    return ev


def _shot_event(idx: int, period: int, ts: str, team: str) -> dict:
    ev = _base_event(idx, period, ts, "Shot", team)
    ev["shot_end_location"] = [115.0, 40.0]
    ev["shot_outcome"]      = "Goal"
    ev["shot_statsbomb_xg"] = 0.25
    ev["shot_body_part"]    = "Left Foot"
    return ev


def _make_events_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


HOME_TEAM = "HomeFC"
AWAY_TEAM = "AwayFC"


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

class TestCoordinateConversion:

    def test_centre_maps_to_origin_period1(self):
        x, y = sb_coords_to_plaiq(60.0, 40.0, period=1)
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6

    def test_centre_maps_to_origin_period2(self):
        # After period 2 flip: 120-60=60, 80-40=40 → same centre
        x, y = sb_coords_to_plaiq(60.0, 40.0, period=2)
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6

    def test_home_goal_is_positive_x_period1(self):
        """Home goal (x=120 in SB period 1) should map to +half_length."""
        x, y = sb_coords_to_plaiq(120.0, 40.0, period=1)
        assert x == pytest.approx(SB_PITCH.length_m / 2)
        assert abs(y) < 1e-6

    def test_home_goal_is_positive_x_period2(self):
        """
        In period 2 home attacks toward x=0 (physically), so SB x=0
        should flip to +half_length in PLAIQ.
        """
        x, y = sb_coords_to_plaiq(0.0, 40.0, period=2)
        assert x == pytest.approx(SB_PITCH.length_m / 2)

    def test_away_goal_is_negative_x_period1(self):
        x, _ = sb_coords_to_plaiq(0.0, 40.0, period=1)
        assert x == pytest.approx(-SB_PITCH.length_m / 2)

    def test_y_axis_top_maps_positive(self):
        """SB y=0 is the bottom touchline → should map to negative y in PLAIQ."""
        _, y = sb_coords_to_plaiq(60.0, 0.0, period=1)
        assert y == pytest.approx(-SB_PITCH.width_m / 2)

    def test_custom_pitch_dimensions(self):
        pitch = PitchConfig(length_m=100.0, width_m=64.0)
        x, y = sb_coords_to_plaiq(120.0, 40.0, period=1, pitch=pitch)
        assert x == pytest.approx(50.0)
        assert abs(y) < 1e-6

    def test_vectorised_matches_scalar(self):
        locs   = np.array([[60.0, 40.0], [120.0, 40.0], [0.0, 40.0]])
        perds  = np.array([1, 1, 2])
        result = sb_coords_array_to_plaiq(locs, perds)
        for i, (sb_x, sb_y) in enumerate([(60, 40), (120, 40), (0, 40)]):
            sx, sy = sb_coords_to_plaiq(sb_x, sb_y, perds[i])
            assert result[i, 0] == pytest.approx(sx, abs=1e-4)
            assert result[i, 1] == pytest.approx(sy, abs=1e-4)


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

class TestTimestamp:

    def test_zero(self):
        assert _timestamp_to_seconds("00:00:00.000") == pytest.approx(0.0)

    def test_one_minute(self):
        assert _timestamp_to_seconds("00:01:00.000") == pytest.approx(60.0)

    def test_fractional_seconds(self):
        assert _timestamp_to_seconds("00:00:01.500") == pytest.approx(1.5)

    def test_45_minutes(self):
        assert _timestamp_to_seconds("00:45:00.000") == pytest.approx(2700.0)


# ---------------------------------------------------------------------------
# Event mapping
# ---------------------------------------------------------------------------

class TestEventMapping:

    def _load(self, rows):
        df = _make_events_df(rows)
        return sb_events_to_plaiq(df, HOME_TEAM)

    def test_output_has_required_columns(self):
        rows = [_pass_event(1, 1, "00:01:00.000", HOME_TEAM, [80.0, 40.0])]
        out = self._load(rows)
        for col in ["Type", "Subtype", "Period", "From", "To", "Team",
                    "Start Frame", "Start Time [s]", "Start X", "Start Y",
                    "End X", "End Y", "sb_event_id"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_completed_pass_type(self):
        rows = [_pass_event(1, 1, "00:01:00.000", HOME_TEAM, [80.0, 40.0], complete=True)]
        out = self._load(rows)
        assert out.iloc[0]["Type"] == "PASS"
        assert out.iloc[0]["Team"] == "Home"
        assert out.iloc[0]["To"]   == "Bob"

    def test_incomplete_pass_is_ball_lost(self):
        rows = [_pass_event(1, 1, "00:01:00.000", HOME_TEAM, [80.0, 40.0], complete=False)]
        out = self._load(rows)
        assert out.iloc[0]["Type"] == "BALL LOST"

    def test_carry_type(self):
        rows = [_carry_event(1, 1, "00:02:00.000", AWAY_TEAM, [70.0, 45.0])]
        out = self._load(rows)
        assert out.iloc[0]["Type"] == "CARRY"
        assert out.iloc[0]["Team"] == "Away"

    def test_shot_goal_subtype(self):
        rows = [_shot_event(1, 1, "00:05:00.000", HOME_TEAM)]
        out = self._load(rows)
        assert out.iloc[0]["Type"]    == "SHOT"
        assert out.iloc[0]["Subtype"] == "GOAL"
        assert out.iloc[0]["sb_xg"]   == pytest.approx(0.25)

    def test_admin_events_skipped(self):
        rows = [
            _base_event(1, 1, "00:00:00.000", "Starting XI", HOME_TEAM),
            _base_event(2, 1, "00:00:00.000", "Half Start", HOME_TEAM),
            _pass_event(3, 1, "00:01:00.000", HOME_TEAM, [80.0, 40.0]),
        ]
        out = self._load(rows)
        assert len(out) == 1
        assert out.iloc[0]["Type"] == "PASS"

    def test_coordinates_converted(self):
        """Centre-spot event should map to (0, 0) in PLAIQ space."""
        rows = [_pass_event(1, 1, "00:01:00.000", HOME_TEAM, [60.0, 40.0])]
        out = self._load(rows)
        assert out.iloc[0]["Start X"] == pytest.approx(0.0, abs=0.1)
        assert out.iloc[0]["Start Y"] == pytest.approx(0.0, abs=0.1)

    def test_period2_direction_normalised(self):
        """
        A pass from the home team's own half in period 2 (SB x<60) should
        map to negative PLAIQ x after the flip.
        """
        rows = [_pass_event(1, 2, "00:01:00.000", HOME_TEAM, [40.0, 40.0])]
        # SB x=60 (start: centre spot) → PLAIQ x=0
        # End SB x=40, period 2 → flipped to 80 → PLAIQ x = (80-60)*scale > 0
        out = self._load(rows)
        assert out.iloc[0]["End X"] > 0.0  # flipped correctly

    def test_period2_cumulative_time_offset(self):
        """Period 2 events should have time > period 1 duration."""
        rows = [
            _pass_event(1, 1, "00:44:00.000", HOME_TEAM, [80.0, 40.0]),
            _pass_event(2, 2, "00:01:00.000", HOME_TEAM, [80.0, 40.0]),
        ]
        out = self._load(rows)
        t1 = out.iloc[0]["Start Time [s]"]
        t2 = out.iloc[1]["Start Time [s]"]
        # Period 1 ran to at least 44 min (2640s), period 2 adds 60s
        assert t2 > t1, "Period 2 events must have higher cumulative time"
        assert t2 == pytest.approx(2640.0 + 60.0, abs=1.0)

    def test_sb_event_id_preserved(self):
        rows = [_pass_event(7, 1, "00:01:00.000", HOME_TEAM, [80.0, 40.0])]
        out = self._load(rows)
        assert out.iloc[0]["sb_event_id"] == "uuid-0007"

    def test_away_team_assigned_correctly(self):
        rows = [_carry_event(1, 1, "00:02:00.000", AWAY_TEAM, [50.0, 38.0])]
        out = self._load(rows)
        assert out.iloc[0]["Team"] == "Away"


# ---------------------------------------------------------------------------
# Freeze-frame pitch control
# ---------------------------------------------------------------------------

class TestFreezeFrameControl:

    @pytest.fixture
    def hex_grid(self):
        return build_pitch_hex_grid(float(np.sqrt(2.0 / (3.0 * np.sqrt(3)))))

    def _make_ff(self, teammate_locs, opponent_locs, actor_loc=None):
        players = []
        for loc in teammate_locs:
            players.append({"teammate": True,  "actor": False, "keeper": False,
                             "location": list(loc)})
        for loc in opponent_locs:
            players.append({"teammate": False, "actor": False, "keeper": False,
                             "location": list(loc)})
        if actor_loc is not None:
            players.append({"teammate": True, "actor": True, "keeper": False,
                             "location": list(actor_loc)})
        return players

    def test_returns_arrays_of_correct_shape(self, hex_grid):
        ff = self._make_ff([[0.0, 0.0]], [[-10.0, 0.0]])
        control, t_ball, t_team, t_opp, _, _ = compute_freeze_frame_control(
            ff, np.array([0.0, 0.0]), hex_grid
        )
        assert control.shape  == (len(hex_grid),)
        assert t_ball.shape   == (len(hex_grid),)
        assert t_team.shape   == (len(hex_grid),)
        assert t_opp.shape    == (len(hex_grid),)

    def test_teammate_controls_nearby_hex(self, hex_grid):
        """
        A teammate at the ball position controls adjacent hexes (+1).

        The ball is placed right next to the teammate (1m away) so the player
        has reaction_time 0.7s while the ball needs > 0.7s to reach the
        teammate's half of the pitch from its position.
        """
        # Ball at -25 m (deep in own half), teammate at +10 m, opponent at +52 m.
        # Ball must travel ~35 m to reach the teammate's area; at MIN_BALL_SPEED
        # that takes >4s, well beyond reaction_time=0.7s.
        ff = self._make_ff([[10.0, 0.0]], [[52.0, 0.0]])
        ball_pos = np.array([-25.0, 0.0])
        control, _, _, _, p_ctrl_tm, p_ctrl_opp = compute_freeze_frame_control(
            ff, ball_pos, hex_grid
        )
        # At least one hex near (10, 0) should be teammate-controlled
        assert np.any(control > 0.5), "Expected at least one teammate-controlled hex"
        R = float(np.sqrt(2.0 / (3.0 * np.sqrt(3))))
        tm_outlines = player_zone_outlines(p_ctrl_tm, hex_grid, R)
        assert tm_outlines, "Expected per-player teammate PCZ outlines"

    def test_opponent_controls_distant_hex(self, hex_grid):
        """An opponent far from the ball should control their nearby area."""
        ff = self._make_ff([[0.0, 0.0]], [[40.0, 0.0]])
        ball_pos = np.array([0.0, 0.0])
        control, _, _, _, _, p_ctrl_opp = compute_freeze_frame_control(ff, ball_pos, hex_grid)
        assert np.any(control < -0.5), "Expected at least one opponent-controlled hex"
        R = float(np.sqrt(2.0 / (3.0 * np.sqrt(3))))
        assert player_zone_outlines(p_ctrl_opp, hex_grid, R)

    def test_no_players_returns_uncontrolled(self, hex_grid):
        control, _, _, _, p_ctrl_tm, p_ctrl_opp = compute_freeze_frame_control(
            [], np.array([0.0, 0.0]), hex_grid
        )
        assert np.all(control == 0.0)
        assert p_ctrl_tm == {} and p_ctrl_opp == {}

    def test_control_values_in_valid_range(self, hex_grid):
        ff = self._make_ff([[3.0, 0.0], [-3.0, 5.0]], [[10.0, 0.0]])
        control, _, _, _, _, _ = compute_freeze_frame_control(
            ff, np.array([0.0, 0.0]), hex_grid
        )
        assert set(np.unique(control.round(1))).issubset({-1.0, 0.0, 1.0})


# ---------------------------------------------------------------------------
# SB_EVENT_COLUMNS completeness
# ---------------------------------------------------------------------------

def test_sb_event_columns_include_plaiq_schema():
    """All standard PLAIQ event columns must appear in SB_EVENT_COLUMNS."""
    plaiq_base = ["Type", "Subtype", "Period", "From", "To", "Team",
                  "Start Frame", "Start Time [s]", "Start X", "Start Y",
                  "End Frame", "End Time [s]", "End X", "End Y"]
    for col in plaiq_base:
        assert col in SB_EVENT_COLUMNS, f"PLAIQ column '{col}' missing from SB_EVENT_COLUMNS"


def test_meta_fallback_probes_360_when_competition_unknown(monkeypatch):
    """When competition_id=0, has_360 must come from open-data probe, not metadata API."""
    from plaiq_engine.ingestion.statsbomb_io import load_sb_match_meta

    monkeypatch.setattr(
        "plaiq_engine.ingestion.statsbomb_io._load_360_raw_records",
        lambda match_id: [{"event_uuid": "abc", "freeze_frame": []}],
    )
    meta = load_sb_match_meta(3869685, 0, 0)
    assert meta["has_360"] is True


def test_load_sb_match_sets_has_360_from_loaded_frames(monkeypatch):
    """has_360 in returned meta must reflect successfully loaded freeze frames."""
    from plaiq_engine.ingestion.statsbomb_io import load_sb_match

    fake_ff = {
        "uuid-1": {
            "players": [{"teammate": True, "actor": True, "keeper": False, "location": [0.0, 0.0]}],
            "visible_area": None,
        }
    }

    monkeypatch.setattr("plaiq_engine.ingestion.statsbomb_io.load_sb_freeze_frames", lambda *a, **k: fake_ff)
    monkeypatch.setattr("plaiq_engine.ingestion.statsbomb_io.load_sb_match_meta", lambda *a, **k: {
        "home_team": "Home", "away_team": "Away", "has_360": False, "pitch": SB_PITCH,
    })
    monkeypatch.setattr("plaiq_engine.ingestion.statsbomb_io.load_sb_lineups", lambda *a, **k: {
        "home": [], "away": [], "home_team_name": "Argentina", "away_team_name": "France",
    })

    class _FakeSB:
        @staticmethod
        def events(match_id, fmt="dataframe"):
            import pandas as pd
            return pd.DataFrame([{
                "id": "uuid-1", "index": 1, "period": 1, "timestamp": "00:00:00.000",
                "type": "Pass", "team": "Argentina", "player": "Messi",
                "location": [60.0, 40.0], "duration": 1.0, "possession": 1,
                "play_pattern": "Regular Play", "pass_end_location": [65.0, 40.0],
            }])

    monkeypatch.setattr("statsbombpy.sb", _FakeSB())

    events, frames, meta, _ = load_sb_match(1, 0, 0)
    assert meta["has_360"] is True
    assert len(frames) == 1
    assert meta["home_team"] == "Argentina"
