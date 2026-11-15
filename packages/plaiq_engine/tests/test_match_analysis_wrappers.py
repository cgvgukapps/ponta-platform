"""The public wrappers in match_analysis return exactly what the private engine helpers return."""

import numpy as np
from plaiq_engine.analytics.pass_option_value import _is_failed_pass_event
from plaiq_engine.analytics.pitch_control import DEFAULT_PROFILE, build_pitch_hex_grid
from plaiq_engine.match_analysis import (
    HEX_RADIUS,
    default_profiles,
    is_failed_pass_event,
    nearest_hex,
)
from plaiq_engine.statsbomb.sb_pov_adapter import _nearest_hex


def test_failed_pass_wrapper_matches_engine():
    cases = [
        ("BALL LOST", "INTERCEPTION", np.array([0.0, 0.0]), np.array([20.0, 5.0])),
        ("BALL LOST", "CLEARANCE", np.array([0.0, 0.0]), np.array([20.0, 5.0])),
        ("BALL LOST", None, np.array([0.0, 0.0]), np.array([0.5, 0.0])),
        ("PASS", None, np.array([0.0, 0.0]), np.array([20.0, 5.0])),
    ]
    for args in cases:
        assert is_failed_pass_event(*args) == _is_failed_pass_event(*args)


def test_nearest_hex_wrapper_matches_engine():
    centres = build_pitch_hex_grid(HEX_RADIUS)
    for xy in ([0.0, 0.0], [51.3, -20.2], [-10.0, 33.9]):
        assert nearest_hex(np.array(xy), centres) == _nearest_hex(np.array(xy), centres)
    assert nearest_hex(np.array([0.0, 0.0]), np.empty((0, 2))) is None


def test_default_profiles():
    assert default_profiles(["1", "7"]) == {"1": DEFAULT_PROFILE, "7": DEFAULT_PROFILE}


def test_hex_radius_is_one_square_metre_per_hexagon():
    # Area of a regular hexagon with circumradius R is (3 * sqrt(3) / 2) * R^2.
    assert abs(3 * np.sqrt(3) / 2 * HEX_RADIUS**2 - 1.0) < 1e-12
