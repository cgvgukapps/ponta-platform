"""P5 line breaks on the synthetic eleven-a-side match (fictional players, no sample data)."""

import pandas as pd
from context_fixture import eleven_a_side, event
from plaiq_engine.context.line_breaks import LineBreakPackage
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.structure import StructurePackage


def _pass(type_, frame, x0, x1, y=0.0):
    e = event(type_, "Home", frame)
    e.update({"Start X": x0, "Start Y": y, "End X": x1, "End Y": y})
    return e


def test_a_pass_through_all_three_lines_and_a_failed_pass():
    # Away (4-3-3) defends the +x goal: forwards near x = -11, midfield near 2 to 10, back four near 21.
    evs = [_pass("PASS", 3000, -20.0, 30.0), _pass("PASS", 4000, -20.0, -15.0), _pass("BALL LOST", 5000, -20.0, 30.0)]
    res = Registry([PossessionPackage(), StructurePackage(), LineBreakPackage()]).run(eleven_a_side(events=evs))
    seg = res.outputs["p5"].segments["line_breaks"].set_index("event_row")
    assert seg.at[0, "furthest"] == "last" and seg.at[0, "units_broken"] == 3 and seg.at[0, "type"] == "through"
    assert seg.at[1, "furthest"] == "none" and seg.at[1, "units_broken"] == 0
    assert seg.at[2, "outcome"] == "failed" and pd.isna(seg.at[2, "furthest"])  # no target yet
    f = res.frames
    assert set(f["p5_pass_furthest_line"].dropna()) == {"last", "none", "unknown"}
    assert {e.metric: e.state for e in res.entries}["line_breaks"] == "full"
    assert not [c for c in f.columns if c.startswith("p5_") and c.endswith(("_x", "_y"))]
