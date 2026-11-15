"""Regression lock for Metrica coordinate math (must be unchanged by the pivot)."""

import numpy as np
import pandas as pd

from matrix_engine.ingestion.metrica_io import to_metric_coordinates, to_single_playing_direction


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
