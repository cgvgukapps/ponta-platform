"""
Track A — two genuinely different real vendors, one schema.

Metrica sample data and the SkillCorner open-data match are different fixtures,
different formats, different frame rates, different pitch dimensions and
different identifier schemes. Nothing here fuses them; the claim under test is
narrower and is the one Matrix's normalisation layer actually has to support:
after each vendor's adapter, the two feeds are structurally interchangeable and
measured in the same units.
"""

import numpy as np
import pandas as pd
import pytest
from matrix_engine.providers import (
    load_metrica_source,
    load_skillcorner_source,
    metrica_pitch,
    skillcorner_pitch,
)
from matrix_engine.schema import (
    REQUIRED_COLUMNS,
    validate_pitch_bounds,
    validate_schema,
)


@pytest.fixture(scope="module")
def skillcorner(skillcorner_dir, skillcorner_match_id, skillcorner_max_frames):
    return load_skillcorner_source(
        skillcorner_dir, skillcorner_match_id, max_frames=skillcorner_max_frames
    )


@pytest.fixture(scope="module")
def both(metrica_both, skillcorner):
    return {"metrica": metrica_both, "skillcorner": skillcorner}


# ---------------------------------------------------------------------------
# Each provider on its own
# ---------------------------------------------------------------------------

def test_both_providers_return_non_empty_data(both):
    for name, df in both.items():
        assert len(df) > 0, f"{name} produced no rows"


def test_both_providers_pass_schema_validation(both):
    for name, df in both.items():
        validate_schema(df, name=name)


def test_both_providers_cover_both_teams(both):
    for name, df in both.items():
        assert set(df["team"].unique()) == {"home", "away"}, name


def test_both_providers_report_their_native_rate(both):
    assert both["metrica"]["sample_rate_hz"].eq(25.0).all()
    assert both["skillcorner"]["sample_rate_hz"].eq(10.0).all()


def test_source_labels_are_distinct(both):
    assert both["metrica"]["source"].unique().tolist() == ["metrica"]
    assert both["skillcorner"]["source"].unique().tolist() == ["skillcorner"]


def test_timestamps_are_monotonic_per_player(both):
    """SkillCorner's clock restarts each half; the adapter has to rebase it."""
    for name, df in both.items():
        for player_id, block in df.groupby("player_id", sort=False):
            t = block["timestamp"].to_numpy(dtype=float)
            assert np.all(np.diff(t) > 0), f"{name}/{player_id} timestamps not increasing"


# ---------------------------------------------------------------------------
# Structural comparability across vendors
# ---------------------------------------------------------------------------

def test_outputs_are_column_and_dtype_identical(both):
    metrica, skillcorner = both["metrica"], both["skillcorner"]
    required = list(REQUIRED_COLUMNS)
    assert list(metrica.columns)[: len(required)] == required
    assert list(skillcorner.columns)[: len(required)] == required
    for column in required:
        assert metrica[column].dtype == skillcorner[column].dtype, column


def test_coordinates_are_metres_in_each_vendors_own_pitch(
    both, skillcorner_dir, skillcorner_match_id
):
    """
    The units check that matters: one vendor in metres and another in
    normalised 0-1 or yards would sail through a columns-and-dtypes comparison.
    Each is validated against its own real pitch — 106x68 for the Metrica
    sample, 104x68 read from the SkillCorner match metadata.
    """
    validate_pitch_bounds(both["metrica"], metrica_pitch(), name="metrica")
    validate_pitch_bounds(
        both["skillcorner"],
        skillcorner_pitch(skillcorner_dir, skillcorner_match_id),
        name="skillcorner",
    )


def test_coordinate_ranges_are_the_same_order_of_magnitude(both):
    metrica, skillcorner = both["metrica"], both["skillcorner"]
    for axis in ("x", "y"):
        m_span = float(metrica[axis].max() - metrica[axis].min())
        s_span = float(skillcorner[axis].max() - skillcorner[axis].min())
        assert 0.5 < m_span / s_span < 2.0, (
            f"{axis} spans differ by more than 2x: metrica {m_span:.1f} m, "
            f"skillcorner {s_span:.1f} m"
        )


def test_both_vendors_are_centre_origin(both):
    """Each team's positions straddle x=0 rather than sitting in [0, length]."""
    for name, df in both.items():
        assert df["x"].min() < -5.0, name
        assert df["x"].max() > 5.0, name
        assert abs(float(df["y"].mean())) < 10.0, name


def test_a_single_concatenated_frame_is_still_valid(both):
    """
    The practical form of "structurally comparable": stack two vendors into one
    frame and the result is still a legal Matrix dataset.
    """
    combined = pd.concat(both.values(), ignore_index=True)
    validate_schema(combined, name="metrica+skillcorner")
    assert set(combined["source"].unique()) == {"metrica", "skillcorner"}


# ---------------------------------------------------------------------------
# Vendor-specific behaviour worth pinning down
# ---------------------------------------------------------------------------

def test_the_two_vendors_sample_at_genuinely_different_rates(both):
    """
    The structural difference Matrix's sync layer exists for: 10 Hz against
    25 Hz, measured from the data rather than taken from the declared rate.
    """
    intervals = {}
    for name, df in both.items():
        per_player = [
            float(np.median(np.diff(np.sort(block["timestamp"].to_numpy(dtype=float)))))
            for _, block in df.groupby("player_id", sort=False)
            if len(block) > 2
        ]
        intervals[name] = float(np.median(per_player))
    assert intervals["metrica"] == pytest.approx(0.04, abs=0.005)
    assert intervals["skillcorner"] == pytest.approx(0.10, abs=0.005)


def test_both_windows_are_densely_covered(both):
    """
    Measured over each player's own observed span, both feeds report nearly
    every sample. SkillCorner is broadcast tracking, but this is its
    *extrapolated* file, which fills in players who are off camera — so the
    sparsity a raw broadcast feed would show is not present here.
    """
    for name, df in both.items():
        rate = float(df["sample_rate_hz"].iloc[0])
        ratios = []
        for _, block in df.groupby("player_id", sort=False):
            t = block["timestamp"].to_numpy(dtype=float)
            span = float(t.max() - t.min())
            if span <= 0:
                continue
            ratios.append(len(t) / (span * rate))
        assert np.median(ratios) > 0.9, f"{name} coverage {np.median(ratios):.2f}"


def test_metrica_can_emit_native_velocities(data_dir):
    """The adapter wraps metrica_velocities as well as metrica_io."""
    df = load_metrica_source(
        data_dir, 1, teams=("home",), start_time=600.0, end_time=660.0,
        with_velocities=True,
    )
    assert {"vx", "vy", "speed"} <= set(df.columns)
    speed = df["speed"].dropna()
    assert len(speed) > 0
    assert speed.max() < 12.0


def test_max_frames_bounds_the_skillcorner_read(skillcorner_dir, skillcorner_match_id):
    small = load_skillcorner_source(
        skillcorner_dir, skillcorner_match_id, teams=("home",), max_frames=500
    )
    larger = load_skillcorner_source(
        skillcorner_dir, skillcorner_match_id, teams=("home",), max_frames=1500
    )
    assert 0 < len(small) < len(larger)
