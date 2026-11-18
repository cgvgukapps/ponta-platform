"""The line-break rule (tools/context_line_breaks.py) on hand-built units."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from context_line_breaks import breaks, furthest, unit_names

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
