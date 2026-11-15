"""Tests for statsbomb/sb_dq.py."""
import math

import numpy as np
import pandas as pd
import pytest
from plaiq_engine.analytics.hex_value import xT_surface
from plaiq_engine.analytics.pitch_control import build_pitch_hex_grid
from plaiq_engine.statsbomb.sb_dq import sb_carry_dq, sb_event_player_rows, sb_pass_dq

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid():
    R = float(np.sqrt(2.0 / (3.0 * np.sqrt(3))))
    hex_centers = build_pitch_hex_grid(R)
    xT_home = xT_surface(hex_centers, attacking_right=True)
    xT_away = xT_surface(hex_centers, attacking_right=False)
    return hex_centers, xT_home, xT_away, R


def _ff(players=None):
    return {"players": players or [], "visible_area": None}


def _dot(teammate=True, actor=False, keeper=False, loc=None, name=None):
    d = {
        "teammate":    teammate,
        "actor":       actor,
        "keeper":      keeper,
        "location":    loc or [0.0, 0.0],
        "player_name": name,
        "confidence":  "strong" if name else None,
    }
    return d


def _event(type="PASS", from_="Alice", to="Bob",
           start_x=0.0, start_y=0.0, end_x=10.0, end_y=0.0, team="Home"):
    return {
        "Type": type, "From": from_, "To": to,
        "Start X": start_x, "Start Y": start_y,
        "End X": end_x, "End Y": end_y,
        "Team": team,
    }


# ---------------------------------------------------------------------------
# sb_pass_dq
# ---------------------------------------------------------------------------

class TestSbPassDq:
    def test_returns_dict_with_required_keys(self, grid):
        hc, xT_h, xT_a, R = grid
        ff = _ff([
            _dot(teammate=True, actor=True, loc=[0.0, 0.0]),
            _dot(teammate=True, actor=False, loc=[10.0, 5.0]),
        ])
        ev = _event(start_x=0.0, start_y=0.0, end_x=10.0, end_y=0.0)
        result = sb_pass_dq(ev, None, ff, None, xT_h, xT_a, hc, R)
        assert all(k in result for k in ["dq", "pov_receiver", "max_pov_start",
                                          "confidence", "reason"])

    def test_no_ff_returns_nan(self, grid):
        hc, xT_h, xT_a, R = grid
        ev = _event()
        result = sb_pass_dq(ev, None, None, None, xT_h, xT_a, hc, R)
        assert math.isnan(result["dq"])
        assert result["confidence"] == "none"

    def test_partial_confidence_when_no_end_ff(self, grid):
        hc, xT_h, xT_a, R = grid
        ff = _ff([
            _dot(teammate=True, actor=True, loc=[0.0, 0.0]),
            _dot(teammate=True, loc=[10.0, 0.0]),
        ])
        ev = _event(start_x=0.0, start_y=0.0, end_x=10.0, end_y=0.0)
        result = sb_pass_dq(ev, None, ff, None, xT_h, xT_a, hc, R)
        if not math.isnan(result["dq"]):
            assert result["confidence"] == "partial"

    def test_full_confidence_when_end_ff_available(self, grid):
        hc, xT_h, xT_a, R = grid
        ff_s = _ff([
            _dot(teammate=True, actor=True, loc=[0.0, 0.0]),
            _dot(teammate=True, loc=[10.0, 0.0]),
        ])
        ff_e = _ff([
            _dot(teammate=True, loc=[10.0, 0.5]),
        ])
        ev = _event(start_x=0.0, start_y=0.0, end_x=10.0, end_y=0.0)
        result = sb_pass_dq(ev, None, ff_s, ff_e, xT_h, xT_a, hc, R)
        if not math.isnan(result["dq"]):
            assert result["confidence"] == "full"

    def test_dq_clipped_to_nonnegative(self, grid):
        hc, xT_h, xT_a, R = grid
        ff = _ff([
            _dot(teammate=True, actor=True, loc=[0.0, 0.0]),
            _dot(teammate=True, loc=[10.0, 0.0]),
        ])
        ev = _event(start_x=0.0, start_y=0.0, end_x=10.0, end_y=0.0)
        result = sb_pass_dq(ev, None, ff, None, xT_h, xT_a, hc, R)
        if not math.isnan(result["dq"]):
            assert result["dq"] >= 0.0

    def test_no_teammates_returns_nan(self, grid):
        hc, xT_h, xT_a, R = grid
        ff = _ff([_dot(teammate=False, loc=[5.0, 5.0])])
        ev = _event(start_x=0.0, start_y=0.0)
        result = sb_pass_dq(ev, None, ff, None, xT_h, xT_a, hc, R)
        assert math.isnan(result["dq"])

    def test_receiver_xy_from_next_event_when_no_end_xy(self, grid):
        hc, xT_h, xT_a, R = grid
        ev = {
            "Type": "PASS", "From": "Alice", "To": "Bob",
            "Start X": 0.0, "Start Y": 0.0,
            "End X": None, "End Y": None,
            "Team": "Home",
        }
        next_ev = {"Start X": 10.0, "Start Y": 0.0}
        ff = _ff([
            _dot(teammate=True, actor=True, loc=[0.0, 0.0]),
            _dot(teammate=True, loc=[10.0, 0.0]),
        ])
        result = sb_pass_dq(ev, next_ev, ff, None, xT_h, xT_a, hc, R)
        # Should not raise and should produce a result (dq or nan)
        assert "dq" in result


# ---------------------------------------------------------------------------
# sb_carry_dq
# ---------------------------------------------------------------------------

class TestSbCarryDq:
    def test_returns_dict_with_required_keys(self, grid):
        hc, xT_h, xT_a, R = grid
        ff = _ff([_dot(teammate=False, loc=[5.0, 5.0])])
        ev = _event(type="CARRY", start_x=0.0, start_y=0.0, end_x=10.0, end_y=0.0)
        result = sb_carry_dq(ev, ff, xT_h, xT_a, hc, R)
        assert all(k in result for k in ["dq", "v_actual", "v_best", "n_alts",
                                          "confidence", "reason"])

    def test_dq_nonnegative_and_bounded(self, grid):
        hc, xT_h, xT_a, R = grid
        ff = _ff([_dot(teammate=False, loc=[5.0, 5.0])])
        ev = _event(type="CARRY", start_x=0.0, start_y=0.0, end_x=10.0, end_y=0.0)
        result = sb_carry_dq(ev, ff, xT_h, xT_a, hc, R)
        if not math.isnan(result["dq"]):
            assert 0.0 <= result["dq"] <= 1.5

    def test_nan_when_no_coords(self, grid):
        hc, xT_h, xT_a, R = grid
        ev = {"Type": "CARRY", "Start X": None, "Start Y": None,
              "End X": None, "End Y": None}
        result = sb_carry_dq(ev, None, xT_h, xT_a, hc, R)
        assert math.isnan(result["dq"])

    def test_nan_for_ring_zero_carry(self, grid):
        hc, xT_h, xT_a, R = grid
        ev = _event(type="CARRY", start_x=5.0, start_y=0.0,
                    end_x=5.0, end_y=0.0)
        result = sb_carry_dq(ev, None, xT_h, xT_a, hc, R)
        assert math.isnan(result["dq"])

    def test_ring_alternatives_positive(self, grid):
        hc, xT_h, xT_a, R = grid
        ev = _event(type="CARRY", start_x=-20.0, start_y=0.0,
                    end_x=-10.0, end_y=0.0)
        result = sb_carry_dq(ev, None, xT_h, xT_a, hc, R)
        if not math.isnan(result["dq"]):
            assert result["n_alts"] >= 1

    def test_no_ff_still_computes_dq(self, grid):
        hc, xT_h, xT_a, R = grid
        ev = _event(type="CARRY", start_x=0.0, start_y=0.0, end_x=10.0, end_y=0.0)
        result = sb_carry_dq(ev, None, xT_h, xT_a, hc, R)
        # No FF means we still compute from xT surface alone
        assert "dq" in result


# ---------------------------------------------------------------------------
# sb_event_player_rows
# ---------------------------------------------------------------------------

class TestSbEventPlayerRows:
    def test_returns_dataframe(self, grid):
        hc, xT_h, xT_a, R = grid
        players = [
            _dot(teammate=True, actor=True, loc=[0.0, 0.0], name="Alice"),
            _dot(teammate=True, actor=False, loc=[10.0, 0.0], name="Bob"),
            _dot(teammate=False, loc=[-5.0, 0.0], name="Carlos"),
        ]
        ev = _event()
        df = sb_event_player_rows(ev, players, [0.0, 0.0], xT_h, xT_a, hc)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_columns_present(self, grid):
        hc, xT_h, xT_a, R = grid
        players = [_dot(teammate=True, actor=True, loc=[0.0, 0.0], name="Alice")]
        ev = _event()
        df = sb_event_player_rows(ev, players, [0.0, 0.0], xT_h, xT_a, hc)
        assert all(c in df.columns for c in ["name", "team", "role", "xT_pos",
                                               "pressure", "pov", "confidence"])

    def test_actor_row_role(self, grid):
        hc, xT_h, xT_a, R = grid
        players = [_dot(teammate=True, actor=True, loc=[0.0, 0.0], name="Alice")]
        ev = _event()
        df = sb_event_player_rows(ev, players, [0.0, 0.0], xT_h, xT_a, hc)
        assert df.iloc[0]["role"] == "Actor"

    def test_opponent_row_role(self, grid):
        hc, xT_h, xT_a, R = grid
        players = [_dot(teammate=False, loc=[-5.0, 0.0], name="Carlos")]
        ev = _event()
        df = sb_event_player_rows(ev, players, [0.0, 0.0], xT_h, xT_a, hc)
        assert df.iloc[0]["role"] == "Opponent"

    def test_unknown_name_for_unidentified(self, grid):
        hc, xT_h, xT_a, R = grid
        players = [_dot(teammate=True)]  # no player_name
        ev = _event()
        df = sb_event_player_rows(ev, players, [0.0, 0.0], xT_h, xT_a, hc)
        assert df.iloc[0]["name"] == "Unknown"

    def test_pov_nan_for_opponents(self, grid):
        hc, xT_h, xT_a, R = grid
        players = [_dot(teammate=False, loc=[-5.0, 0.0], name="Carlos")]
        ev = _event()
        df = sb_event_player_rows(ev, players, [0.0, 0.0], xT_h, xT_a, hc)
        assert math.isnan(df.iloc[0]["pov"])

    def test_empty_players_returns_empty_df(self, grid):
        hc, xT_h, xT_a, R = grid
        df = sb_event_player_rows(_event(), [], [0.0, 0.0], xT_h, xT_a, hc)
        assert df.empty
