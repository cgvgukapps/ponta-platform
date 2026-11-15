"""Tests for statsbomb/stitch.py event stitching."""
import math
import numpy as np
import pandas as pd
import pytest

from plaiq_engine.statsbomb.stitch import (
    build_stitch_graph,
    get_possession_chains,
    end_proxy_ff,
    stitch_coverage_summary,
)


def _make_events(*rows) -> pd.DataFrame:
    """
    Build a minimal events DataFrame from row dicts.
    Defaults: Period=1, sb_possession=1, Type=PASS, no coords.
    """
    base = {
        "sb_event_index": 0,
        "Period": 1,
        "sb_possession": 1,
        "Type": "PASS",
        "From": None,
        "To": None,
        "Start X": None, "Start Y": None,
        "End X": None, "End Y": None,
        "Start Time [s]": 0.0,
        "End Time [s]": 1.0,
        "sb_event_id": None,
    }
    data = []
    for i, r in enumerate(rows):
        row = dict(base)
        row["sb_event_index"] = i + 1
        row.update(r)
        data.append(row)
    df = pd.DataFrame(data)
    df.index = pd.RangeIndex(len(df))
    return df


class TestBuildStitchGraph:
    def test_empty_returns_empty(self):
        ev = _make_events()
        g = build_stitch_graph(ev)
        assert g.empty

    def test_single_event_returns_empty(self):
        ev = _make_events({"Period": 1})
        g = build_stitch_graph(ev)
        assert g.empty

    def test_basic_shape(self):
        ev = _make_events({}, {})
        g = build_stitch_graph(ev)
        assert len(g) == 1
        assert set(g.columns) >= {
            "ev_a_pos", "ev_b_pos", "ev_a_idx", "ev_b_idx",
            "ball_gap_m", "dt_s", "same_period", "same_poss",
            "identity_link", "stitch_valid",
        }

    def test_stitch_valid_same_poss_small_gap(self):
        ev = _make_events(
            {"End X": 10.0, "End Y": 0.0},
            {"Start X": 10.5, "Start Y": 0.0},
        )
        g = build_stitch_graph(ev)
        assert g.iloc[0]["stitch_valid"] == True
        assert math.isclose(g.iloc[0]["ball_gap_m"], 0.5, abs_tol=1e-6)

    def test_stitch_invalid_large_gap(self):
        ev = _make_events(
            {"End X": 0.0, "End Y": 0.0},
            {"Start X": 30.0, "Start Y": 0.0},
        )
        g = build_stitch_graph(ev)
        assert g.iloc[0]["stitch_valid"] == False

    def test_stitch_invalid_different_period(self):
        ev = _make_events(
            {"Period": 1, "End X": 0.0, "End Y": 0.0},
            {"Period": 2, "Start X": 0.0, "Start Y": 0.0},
        )
        g = build_stitch_graph(ev)
        assert g.iloc[0]["stitch_valid"] == False
        assert g.iloc[0]["same_period"] == False

    def test_stitch_invalid_different_possession(self):
        ev = _make_events(
            {"sb_possession": 1, "End X": 0.0, "End Y": 0.0},
            {"sb_possession": 2, "Start X": 0.0, "Start Y": 0.0},
        )
        g = build_stitch_graph(ev)
        assert g.iloc[0]["stitch_valid"] == False
        assert g.iloc[0]["same_poss"] == False

    def test_identity_link_pass_receipt(self):
        ev = _make_events(
            {"Type": "PASS", "From": "Alice", "To": "Bob",
             "End X": 5.0, "End Y": 0.0},
            {"Type": "CARRY", "From": "Bob",
             "Start X": 5.0, "Start Y": 0.0},
        )
        g = build_stitch_graph(ev)
        assert g.iloc[0]["identity_link"] == "pass_receipt"

    def test_identity_link_carry_cont(self):
        ev = _make_events(
            {"Type": "CARRY", "From": "Alice",
             "End X": 5.0, "End Y": 0.0},
            {"Type": "PASS", "From": "Alice",
             "Start X": 5.0, "Start Y": 0.0},
        )
        g = build_stitch_graph(ev)
        assert g.iloc[0]["identity_link"] == "carry_cont"

    def test_identity_link_none_when_different_players(self):
        ev = _make_events(
            {"Type": "PASS", "From": "Alice", "To": "Bob",
             "End X": 5.0, "End Y": 0.0},
            {"Type": "CARRY", "From": "Charlie",
             "Start X": 5.0, "Start Y": 0.0},
        )
        g = build_stitch_graph(ev)
        assert g.iloc[0]["identity_link"] == "none"

    def test_missing_coords_give_nan_gap(self):
        ev = _make_events({"End X": None}, {"Start X": 5.0, "Start Y": 0.0})
        g = build_stitch_graph(ev)
        assert math.isnan(g.iloc[0]["ball_gap_m"])
        assert g.iloc[0]["stitch_valid"] == False

    def test_multiple_rows_count(self):
        ev = _make_events({}, {}, {}, {})
        g = build_stitch_graph(ev)
        assert len(g) == 3


class TestGetPossessionChains:
    def test_all_valid_one_chain(self):
        ev = _make_events(
            {"End X": 0.0, "End Y": 0.0},
            {"Start X": 0.0, "Start Y": 0.0, "End X": 1.0, "End Y": 0.0},
            {"Start X": 1.0, "Start Y": 0.0},
        )
        g = build_stitch_graph(ev)
        chains = get_possession_chains(ev, g)
        assert len(chains) == 1
        assert len(chains[0]) == 3

    def test_possession_break_makes_two_chains(self):
        ev = _make_events(
            {"sb_possession": 1, "End X": 0.0, "End Y": 0.0},
            {"sb_possession": 2, "Start X": 0.0, "Start Y": 0.0,
             "End X": 1.0, "End Y": 0.0},
            {"sb_possession": 2, "Start X": 1.0, "Start Y": 0.0},
        )
        g = build_stitch_graph(ev)
        chains = get_possession_chains(ev, g)
        assert len(chains) == 2
        assert len(chains[0]) == 1
        assert len(chains[1]) == 2

    def test_empty_events(self):
        ev = _make_events()
        g = build_stitch_graph(ev)
        chains = get_possession_chains(ev, g)
        assert chains == []


class TestEndProxyFF:
    def _make_ev_with_ids(self):
        ev = _make_events(
            {"sb_event_id": "uuid-A", "End X": 0.0, "End Y": 0.0},
            {"sb_event_id": "uuid-B", "Start X": 0.0, "Start Y": 0.0},
            {"sb_event_id": "uuid-C", "sb_possession": 2},
        )
        return ev

    def test_returns_ff_of_next_stitched_event(self):
        ev = self._make_ev_with_ids()
        g = build_stitch_graph(ev)
        ff = {"uuid-B": {"players": [{"location": [0, 0]}], "visible_area": None}}
        result = end_proxy_ff(0, ev, g, ff)
        assert result is not None
        assert result["players"][0]["location"] == [0, 0]

    def test_returns_none_when_no_valid_stitch(self):
        ev = self._make_ev_with_ids()
        g = build_stitch_graph(ev)
        ff = {"uuid-C": {"players": [], "visible_area": None}}
        # event index 1 (uuid-B) → stitch to index 2 (uuid-C) is invalid
        # (different possession)
        result = end_proxy_ff(1, ev, g, ff)
        assert result is None

    def test_returns_none_when_next_has_no_ff(self):
        ev = self._make_ev_with_ids()
        g = build_stitch_graph(ev)
        result = end_proxy_ff(0, ev, g, {})
        assert result is None


class TestStitchCoverageSummary:
    def test_summary_keys_present(self):
        ev = _make_events(
            {"sb_event_id": "a", "End X": 0.0, "End Y": 0.0},
            {"sb_event_id": "b", "Start X": 0.0, "Start Y": 0.0},
        )
        g = build_stitch_graph(ev)
        ff = {
            "a": {"players": [], "visible_area": None},
            "b": {"players": [], "visible_area": None},
        }
        s = stitch_coverage_summary(ev, g, ff)
        assert all(k in s for k in [
            "total_events", "ff_events", "valid_stitches",
            "pct_valid_stitches", "dual_ff_pairs",
            "pct_dual_ff_of_ff", "pass_receipt_links", "carry_cont_links",
        ])
        assert s["total_events"] == 2
        assert s["ff_events"] == 2
        assert s["dual_ff_pairs"] == 1
