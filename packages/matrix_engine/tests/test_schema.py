"""Schema validation accepts conforming frames and rejects each way of breaking one."""

import numpy as np
import pandas as pd
import pytest
from matrix_engine.pitch_config import PitchConfig
from matrix_engine.schema import (
    REQUIRED_COLUMNS,
    SchemaError,
    coerce_schema,
    empty_frame,
    validate_pitch_bounds,
    validate_schema,
)


def _good(n=4):
    return pd.DataFrame(
        {
            "timestamp": np.arange(n, dtype="float64") * 0.04,
            "frame": np.arange(n, dtype="int64"),
            "player_id": ["home_7"] * n,
            "team": ["home"] * n,
            "x": np.linspace(-10.0, 10.0, n),
            "y": np.linspace(5.0, -5.0, n),
            "source": ["metrica"] * n,
            "sample_rate_hz": np.full(n, 25.0),
        }
    )


def test_valid_frame_passes_and_is_returned_unchanged():
    df = _good()
    assert validate_schema(df) is df


def test_empty_frame_helper_is_structurally_valid():
    validate_schema(empty_frame())


def test_non_dataframe_is_rejected():
    with pytest.raises(SchemaError, match="expected a pandas DataFrame"):
        validate_schema({"timestamp": [0.0]})


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_missing_required_column_is_rejected(column):
    df = _good().drop(columns=[column])
    with pytest.raises(SchemaError, match="missing required column"):
        validate_schema(df)


def test_wrong_dtype_is_rejected():
    df = _good()
    df["timestamp"] = df["timestamp"].astype(str)
    with pytest.raises(SchemaError, match="column 'timestamp' has dtype"):
        validate_schema(df)


def test_frame_must_be_integral():
    df = _good()
    df["frame"] = df["frame"].astype("float64")
    with pytest.raises(SchemaError, match="column 'frame' has dtype"):
        validate_schema(df)


@pytest.mark.parametrize("column", ["timestamp", "x", "y"])
def test_non_finite_coordinates_are_rejected(column):
    df = _good()
    df.loc[1, column] = np.nan
    with pytest.raises(SchemaError, match="non-finite"):
        validate_schema(df)


def test_infinity_is_rejected_too():
    df = _good()
    df.loc[2, "x"] = np.inf
    with pytest.raises(SchemaError, match="non-finite"):
        validate_schema(df)


def test_unknown_team_is_rejected():
    df = _good()
    df.loc[0, "team"] = "neutral"
    with pytest.raises(SchemaError, match="invalid value"):
        validate_schema(df)


def test_non_positive_sample_rate_is_rejected():
    df = _good()
    df.loc[0, "sample_rate_hz"] = 0.0
    with pytest.raises(SchemaError, match="strictly positive"):
        validate_schema(df)


def test_empty_player_id_is_rejected():
    df = _good()
    df.loc[0, "player_id"] = ""
    with pytest.raises(SchemaError, match="player_id"):
        validate_schema(df)


def test_duplicate_observation_is_rejected():
    df = pd.concat([_good(2), _good(2)], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        validate_schema(df)


def test_same_frame_from_two_sources_is_allowed():
    """Two sources reporting one player at one frame is the whole point."""
    other = _good(2).assign(source="skillcorner")
    validate_schema(pd.concat([_good(2), other], ignore_index=True))


def test_coerce_schema_casts_orders_and_drops_bad_rows():
    df = _good()
    df["timestamp"] = df["timestamp"].astype("object")
    df["x"] = df["x"].astype("float32")
    df.loc[1, "y"] = np.nan
    df = df.loc[:, list(reversed(REQUIRED_COLUMNS))]

    out = coerce_schema(df)

    assert list(out.columns) == list(REQUIRED_COLUMNS)
    assert len(out) == 3  # the NaN row is dropped
    validate_schema(out)


def test_coerce_schema_preserves_extra_columns():
    df = _good().assign(vx=1.0, gap_s=0.0)
    out = coerce_schema(df)
    assert list(out.columns)[: len(REQUIRED_COLUMNS)] == list(REQUIRED_COLUMNS)
    assert {"vx", "gap_s"} <= set(out.columns)


def test_pitch_bounds_accepts_players_just_off_the_pitch():
    pitch = PitchConfig()
    df = _good()
    df.loc[0, "x"] = pitch.half_length + 2.0  # keeper behind the goal line
    validate_pitch_bounds(df, pitch)


def test_pitch_bounds_rejects_normalised_coordinates():
    """The unit error this check exists to catch: 0-1 coords read as metres."""
    pitch = PitchConfig()
    df = _good()
    df["x"] = df["x"] * 1000.0
    with pytest.raises(SchemaError, match="max \\|x\\|"):
        validate_pitch_bounds(df, pitch)


def test_pitch_bounds_rejects_out_of_range_y():
    pitch = PitchConfig()
    df = _good()
    df.loc[3, "y"] = pitch.half_width + 50.0
    with pytest.raises(SchemaError, match="max \\|y\\|"):
        validate_pitch_bounds(df, pitch)
