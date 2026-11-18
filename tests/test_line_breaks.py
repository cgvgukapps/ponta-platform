"""P5 line breaks (plaiq_engine/context/line_breaks.py) on hand-built units."""

import numpy as np
from plaiq_engine.context.line_breaks import breaks, furthest, unit_names, units_at

# Defending units in the attacking team's frame (depth towards the opponent's goal, width).
UNITS = {
    2: np.array([[45.0, 5.0], [46.0, -5.0]]),                               # forwards
    1: np.array([[60.0, 15.0], [62.0, 5.0], [61.0, -5.0], [60.0, -15.0]]),  # midfield
    0: np.array([[80.0, 20.0], [82.0, 7.0], [82.0, -7.0], [80.0, -20.0]]),  # defence
}


def test_a_pass_beyond_the_midfield_line_breaks_it_through_the_middle():
    b = breaks((50.0, 0.0), (70.0, 0.0), UNITS)
    assert b == {2: False, 1: "through", 0: False}
    assert furthest(b) == "second_last"


def test_around_the_widest_players_and_beyond_two_lines():
    b = breaks((40.0, 25.0), (85.0, 30.0), UNITS)
    assert b == {2: "around", 1: "around", 0: "around"}
    assert furthest(b) == "last"


def test_a_backward_pass_breaks_nothing():
    assert furthest(breaks((70.0, 0.0), (50.0, 0.0), UNITS)) == "none"


def test_a_unit_needs_two_players_and_names_follow_the_count():
    units = {2: np.array([[45.0, 0.0]]), 1: UNITS[1], 0: UNITS[0]}  # a lone striker is not a unit
    b = breaks((30.0, 0.0), (70.0, 0.0), units)
    assert b[2] is None and unit_names(b) == {1: "first", 0: "last"}
    assert furthest(b) == "first"


def test_units_at_leaves_out_the_keeper_and_groups_by_depth():
    # Defending team's own frame: keeper at 3 m, back four ~20 m, midfield ~40 m, two forwards ~55 m.
    depth = np.array([3.0, 20.0, 21.0, 20.5, 19.5, 40.0, 41.0, 39.0, 40.5, 55.0, 56.0])
    width = np.array([0.0, 20.0, 7.0, -7.0, -20.0, 15.0, 5.0, -5.0, -15.0, 5.0, -5.0])
    u = units_at(depth, width, 105.0, keeper=0)
    assert [len(u[c]) for c in (2, 1, 0)] == [2, 4, 4]
    assert u[0][:, 0].max() == 105.0 - 19.5      # the back line, in the attacking team's frame
