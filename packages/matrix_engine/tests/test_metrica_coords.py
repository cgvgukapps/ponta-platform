"""Regression lock for Metrica coordinate math (must be unchanged by the pivot)."""

import numpy as np
import pandas as pd
from matrix_engine.ingestion.metrica_io import (
    home_attack_sign,
    orient_home_attacking_positive,
    to_metric_coordinates,
    to_single_playing_direction,
)


def test_normalized_to_metric_corners():
    # Metrica 0-1, top-left origin -> metres, centre origin, +y up.
    df = pd.DataFrame(
        {
            "Home_1_x": [0.5, 1.0, 0.0, 0.5, 0.5],
            "Home_1_y": [0.5, 0.5, 0.5, 0.0, 1.0],
        }
    )
    out = to_metric_coordinates(df.copy())
    # centre, right edge, left edge stay on y=0
    np.testing.assert_allclose(out["Home_1_x"].tolist(), [0.0, 53.0, -53.0, 0.0, 0.0])
    # top edge -> +34, bottom edge -> -34 (y flipped)
    np.testing.assert_allclose(out["Home_1_y"].tolist(), [0.0, 0.0, 0.0, 34.0, -34.0])


def test_single_playing_direction_flips_only_second_half():
    home = pd.DataFrame(
        {"Period": [1, 1, 2, 2], "Home_1_x": [10.0, 11.0, 12.0, 13.0], "Home_1_y": [1.0, 2.0, 3.0, 4.0]},
        index=[0, 1, 2, 3],
    )
    away = pd.DataFrame(
        {"Period": [1, 1, 2, 2], "Away_1_x": [-10.0, -11.0, -12.0, -13.0], "Away_1_y": [1.0, 2.0, 3.0, 4.0]},
        index=[0, 1, 2, 3],
    )
    events = pd.DataFrame(columns=["Start X", "Start Y", "End X", "End Y"])
    h, a, _ = to_single_playing_direction(home.copy(), away.copy(), events.copy())
    # First half unchanged, second half negated.
    assert h["Home_1_x"].tolist() == [10.0, 11.0, -12.0, -13.0]
    assert a["Away_1_x"].tolist() == [-10.0, -11.0, 12.0, 13.0]


def test_single_playing_direction_flips_second_half_events_by_their_own_period():
    # Tracking frames are labelled by frame number; events by row. The half-time split must
    # come from each table's own Period (the prototype sliced events by a tracking frame label).
    home = pd.DataFrame({"Period": [1, 2], "Home_1_x": [10.0, 12.0], "Home_1_y": [1.0, 3.0]}, index=[5000, 70000])
    away = pd.DataFrame({"Period": [1, 2], "Away_1_x": [-10.0, -12.0], "Away_1_y": [1.0, 3.0]}, index=[5000, 70000])
    events = pd.DataFrame({"Period": [1, 1, 2], "Start X": [5.0, 6.0, 7.0], "Start Y": [1.0, 2.0, 3.0],
                           "End X": [8.0, 9.0, 10.0], "End Y": [-1.0, -2.0, -3.0]})
    h, _, e = to_single_playing_direction(home.copy(), away.copy(), events.copy())
    assert h["Home_1_x"].tolist() == [10.0, -12.0]
    assert e["Start X"].tolist() == [5.0, 6.0, -7.0] and e["End Y"].tolist() == [-1.0, -2.0, 3.0]


def _teams(home_gk_x: float):
    """A home goalkeeper at ``home_gk_x`` and an outfielder near the centre, after the half-time flip."""
    home = pd.DataFrame(
        {
            "Period": [1, 1, 2, 2],
            "Home_1_x": [home_gk_x, home_gk_x, home_gk_x, home_gk_x],
            "Home_1_y": [1.0, 1.0, 1.0, 1.0],
            "Home_5_x": [-2.0, 3.0, -1.0, 2.0],
            "Home_5_y": [5.0, 6.0, 7.0, 8.0],
        }
    )
    away = pd.DataFrame(
        {"Period": [1, 1, 2, 2], "Away_1_x": [-home_gk_x] * 4, "Away_1_y": [-1.0, -1.0, -1.0, -1.0]}
    )
    events = pd.DataFrame({"Start X": [10.0], "Start Y": [2.0], "End X": [20.0], "End Y": [-3.0]})
    return home, away, events


def test_home_attack_sign_reads_the_goalkeeper():
    assert home_attack_sign(_teams(-48.0)[0]) == 1   # keeper at -x: home attacks +x
    assert home_attack_sign(_teams(48.0)[0]) == -1   # keeper at +x: home attacks -x (sample game 2)


def test_orientation_is_a_no_op_when_home_attacks_positive():
    home, away, events = _teams(-48.0)
    h, a, e = orient_home_attacking_positive(home.copy(), away.copy(), events.copy())
    pd.testing.assert_frame_equal(h, home)
    pd.testing.assert_frame_equal(a, away)
    pd.testing.assert_frame_equal(e, events)


def test_orientation_rotates_the_whole_match_when_home_attacks_negative():
    home, away, events = _teams(48.0)
    h, a, e = orient_home_attacking_positive(home.copy(), away.copy(), events.copy())
    # Both halves, both axes, tracking and events: a 180-degree rotation.
    assert h["Home_1_x"].tolist() == [-48.0] * 4
    assert h["Home_5_y"].tolist() == [-5.0, -6.0, -7.0, -8.0]
    assert a["Away_1_x"].tolist() == [48.0] * 4
    assert e.iloc[0].tolist() == [-10.0, -2.0, -20.0, 3.0]
    assert h["Period"].tolist() == [1, 1, 2, 2]
    assert home_attack_sign(h) == 1
