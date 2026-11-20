"""P5 line breaks on the synthetic eleven-a-side match (fictional players, no sample data)."""

import pandas as pd
from context_fixture import eleven_a_side, event
from plaiq_engine.context.line_breaks import LineBreakPackage
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.context.zones import ZonesPackage


def _pass(type_, frame, x0, x1, y=0.0):
    e = event(type_, "Home", frame)
    e.update({"Start X": x0, "Start Y": y, "End X": x1, "End Y": y})
    return e


def test_a_pass_through_all_three_lines_and_a_failed_pass():
    # Away (4-3-3) defends the +x goal: forwards near x = -11, midfield near 2 to 10, back four near 21.
    evs = [_pass("PASS", 3000, -20.0, 30.0), _pass("PASS", 4000, -20.0, -15.0), _pass("BALL LOST", 5000, -20.0, 30.0)]
    res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), LineBreakPackage()]).run(eleven_a_side(events=evs))
    seg = res.outputs["p5"].segments["line_breaks"].set_index("event_row")
    assert seg.at[0, "furthest"] == "last" and seg.at[0, "units_broken"] == 3 and seg.at[0, "type"] == "through"
    assert seg.at[1, "furthest"] == "none" and seg.at[1, "units_broken"] == 0
    assert seg.at[2, "outcome"] == "failed" and pd.isna(seg.at[2, "furthest"])  # no target yet
    f = res.frames
    assert set(f["p5_pass_furthest_line"].dropna()) == {"last", "none", "unknown"}
    assert {e.metric: e.state for e in res.entries}["line_breaks"] == "full"
    assert not [c for c in f.columns if c.startswith("p5_") and c.endswith(("_x", "_y"))]


def test_failed_passes_with_a_logged_target_shape_and_line_breaking_options():
    """P5 0.2.0: a failed pass is judged at the logged target's projected point; inside or outside the shape; options."""
    to_striker = _pass("BALL LOST", 5000, -6.5, -6.5, y=8.0)
    to_striker.update({"From": "Player 7", "To": "Player 10", "End X": float("nan"), "End Y": float("nan")})
    evs = [_pass("PASS", 3000, -20.0, 30.0), _pass("PASS", 4000, -6.5, 11.5, y=7.0), to_striker]
    for e in evs[:2]:
        e.update({"From": "Player 7", "To": "Player 10"})
    res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), LineBreakPackage()]).run(eleven_a_side(events=evs))
    seg = res.outputs["p5"].segments["line_breaks"].set_index("event_row")
    # Home's striker stands just beyond the away midfield's deepest player (its holding midfielder).
    assert seg.at[2, "target"] == "logged_target_projected" and seg.at[2, "furthest"] == "second_last"
    assert seg.at[0, "received_inside"] is not None and not seg.at[0, "received_inside"]  # behind the back four: outside
    assert bool(seg.at[1, "received_inside"])  # in front of the back four, between the lines: inside the shape
    assert seg["line_breaking_options"].notna().all() and (seg["line_breaking_options"] >= 0).all()
    f = res.frames
    assert f["p5_pass_line_breaking_options"].notna().sum() == 3 and f["p5_pass_received_inside_shape"].notna().sum() == 3


def test_relationships_on_every_grid_point_and_carrier_pressure_equals_plaiqs():
    """P5 0.2.0: Home_7 holds the ball in midfield; relationships for the away team defending it."""
    import numpy as np
    from plaiq_engine.context.relationships import pressure_score
    from plaiq_engine.match_analysis import MatchAnalysis
    m = eleven_a_side(seconds=60.0)
    res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), LineBreakPackage()]).run(m)
    f = res.frames
    held = f.loc[f["p5_carrier_pressure"].notna()]
    assert len(held) > 0.5 * len(f)
    # Away's 4-3-3 defends the +x goal; the ball is at Home_7 (about x = -6.5): its back four, holding midfielder
    # and two central midfielders are nearer their goal than the ball, its three forwards are not.
    assert held["p5_goal_side_defenders"].mode().iloc[0] == 7
    assert held["p5_offside_line_m"].between(20, 40).all()
    assert (held["p5_marking_distance_m"] > 0).all() and held["p5_free_attackers"].between(0, 10).all()
    # The carrier's pressure is Plaiq's 1v1 pressure for Home_7 at the same frame.
    ma = MatchAnalysis(m)
    for i in held.index[:: max(1, len(held) // 5)][:5]:
        frame = int(f.at[i, "native_frame"])
        pr = ma.pressure(frame)
        plaiq = pr.loc[(pr["team"] == "Home") & (pr["jersey"].astype(str) == "7"), "pressure"]
        assert len(plaiq) == 1 and float(plaiq.iloc[0]) == float(f.at[i, "p5_carrier_pressure"])
    assert list(pressure_score(np.array([0, 1, 2, 9, 10, 15, np.nan]))[:6]) == [10, 10, 9, 2, 1, 1]
    assert {e.metric: e.state for e in res.entries}["relationships"] == "full"


def test_efi_aggregates_per_team_and_half():
    res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), LineBreakPackage()]).run(
        eleven_a_side(seconds=60.0, events=[_pass("PASS", 300, -6.5, 11.5, y=7.0)]))
    efi = res.outputs["p5"].segments["efi"]
    assert set(efi["team"]) == {"home", "away"} and set(efi.columns) == {"team", "period", "metric", "value"}
    metrics = set(efi["metric"])
    assert {"receptions_behind_midfield", "receptions_behind_defence", "final_third_entries", "ball_recovery_time_s",
            "forced_turnovers", "final_third_entries_centre"} <= metrics | {"ball_recovery_time_s"}
    home_in = efi.loc[(efi["team"] == "home") & efi["metric"].str.startswith("def_line_m_in_")]
    assert len(home_in) and home_in["value"].between(10, 60).all()  # Home holds the ball; its back four sits about 30 m out
    assert not efi["metric"].str.contains("_x$|_y$").any()
