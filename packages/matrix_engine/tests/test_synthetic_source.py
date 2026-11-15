"""
Check the degradation generator before trusting it as a measuring instrument.

Every Track B number is stated relative to what this generator injected. If it
silently applied 0.2 m of noise when asked for 0.5, the fusion result would look
better than it is. So each degradation is verified against the ground truth it
was derived from, and each is verified to be independently switchable — a test
that isolates dropout has to be able to turn everything else off.
"""

import numpy as np
import pytest
from matrix_engine.schema import validate_schema
from matrix_engine.testing import DegradationParams, degrade


@pytest.fixture(scope="module")
def quiet(metrica_home):
    """Resampling only — every other degradation disabled."""
    return degrade(
        metrica_home,
        DegradationParams(
            target_hz=10.0, position_noise_m=0.0, dropout_rate=0.0,
            n_dropout_bursts=0, shuffle_identities=False,
            time_offset_s=0.0, time_drift_rate=0.0, seed=3,
        ),
    )


def _paired(synthetic):
    """Join degraded rows back to their clean counterparts on (player, frame)."""
    clean = synthetic.clean.rename(columns={"x": "clean_x", "y": "clean_y"})
    degraded = synthetic.data.assign(
        true_id=synthetic.data["player_id"].map(synthetic.identity_map)
    )
    return degraded.merge(
        clean[["player_id", "frame", "clean_x", "clean_y"]],
        left_on=["true_id", "frame"],
        right_on=["player_id", "frame"],
        suffixes=("", "_clean"),
    )


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_output_is_schema_valid(drifted):
    validate_schema(drifted["synthetic"].data, name="synthetic_gps")


def test_output_is_labelled_as_its_own_source(drifted):
    assert drifted["synthetic"].data["source"].unique().tolist() == ["synthetic_gps"]


def test_empty_input_is_refused(metrica_home):
    with pytest.raises(ValueError, match="empty"):
        degrade(metrica_home.iloc[0:0])


def test_invalid_parameters_are_refused(metrica_home):
    with pytest.raises(ValueError, match="dropout_rate"):
        degrade(metrica_home, DegradationParams(dropout_rate=1.5))
    with pytest.raises(ValueError, match="target_hz"):
        degrade(metrica_home, DegradationParams(target_hz=0.0))


def test_generator_is_reproducible(metrica_home):
    params = DegradationParams(seed=42)
    first = degrade(metrica_home, params).data
    second = degrade(metrica_home, params).data
    assert first.equals(second)


def test_different_seeds_differ(metrica_home):
    first = degrade(metrica_home, DegradationParams(seed=1)).data
    second = degrade(metrica_home, DegradationParams(seed=2)).data
    assert not first.equals(second)


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------

def test_resamples_to_the_requested_rate(quiet):
    assert quiet.data["sample_rate_hz"].eq(10.0).all()
    times = np.unique(quiet.data["timestamp"].to_numpy(dtype=float))
    assert np.allclose(np.diff(times), 0.1, atol=1e-6)


def test_resampling_alone_barely_moves_positions(quiet, metrica_home, position_rmse):
    """With everything else off, the only error is interpolation onto a coarser grid."""
    rmse, n = position_rmse(quiet.data, metrica_home)
    assert n > 0
    assert rmse < 0.01, f"resampling alone introduced {rmse:.4f} m"


# ---------------------------------------------------------------------------
# Position noise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.3, 0.5, 1.0])
def test_injected_noise_matches_the_requested_sigma(metrica_home, sigma):
    synthetic = degrade(
        metrica_home,
        DegradationParams(
            position_noise_m=sigma, dropout_rate=0.0, n_dropout_bursts=0,
            time_offset_s=0.0, seed=5,
        ),
    )
    paired = _paired(synthetic)
    for axis in ("x", "y"):
        measured = float((paired[axis] - paired[f"clean_{axis}"]).std())
        assert measured == pytest.approx(sigma, rel=0.05), axis


def test_noise_is_independent_between_axes(metrica_home):
    synthetic = degrade(
        metrica_home,
        DegradationParams(position_noise_m=0.5, dropout_rate=0.0,
                          n_dropout_bursts=0, seed=5),
    )
    paired = _paired(synthetic)
    correlation = np.corrcoef(
        paired["x"] - paired["clean_x"], paired["y"] - paired["clean_y"]
    )[0, 1]
    assert abs(correlation) < 0.05


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

def test_random_dropout_removes_about_the_requested_fraction(metrica_home):
    synthetic = degrade(
        metrica_home,
        DegradationParams(dropout_rate=0.25, n_dropout_bursts=0, seed=9),
    )
    kept = len(synthetic.data) / len(synthetic.clean)
    assert kept == pytest.approx(0.75, abs=0.02)


def test_bursts_create_gaps_of_the_requested_length(metrica_home):
    burst_s = 5.0
    synthetic = degrade(
        metrica_home,
        DegradationParams(dropout_rate=0.0, n_dropout_bursts=1,
                          dropout_burst_s=burst_s, seed=4),
    )
    longest = []
    for _, block in synthetic.data.groupby("player_id", sort=False):
        t = np.sort(block["timestamp"].to_numpy(dtype=float))
        longest.append(float(np.max(np.diff(t))))
    # A 5 s burst removes samples inclusively, so the surviving gap spans
    # roughly the burst plus one sampling interval either side.
    assert np.median(longest) == pytest.approx(burst_s, abs=0.3)


def test_injected_gaps_are_reported(metrica_home):
    synthetic = degrade(
        metrica_home,
        DegradationParams(dropout_rate=0.0, n_dropout_bursts=2,
                          dropout_burst_s=4.0, seed=4),
    )
    gaps = synthetic.injected_gaps
    players = synthetic.data["player_id"].nunique()
    assert len(gaps) == 2 * players
    assert (gaps["end"] > gaps["start"]).all()
    assert np.allclose(gaps["end"] - gaps["start"], 4.0, atol=0.05)


def test_no_dropout_keeps_every_sample(quiet):
    assert len(quiet.data) == len(quiet.clean)


# ---------------------------------------------------------------------------
# Identity shuffling
# ---------------------------------------------------------------------------

def test_shuffled_ids_share_nothing_with_the_originals(drifted):
    synthetic = drifted["synthetic"]
    original = set(drifted["truth"]["player_id"].unique())
    shuffled = set(synthetic.data["player_id"].unique())
    assert not (original & shuffled)
    assert all(pid.startswith("gps_") for pid in shuffled)


def test_shuffle_is_a_bijection_with_no_fixed_points(drifted):
    synthetic = drifted["synthetic"]
    mapping = synthetic.identity_map
    assert len(set(mapping.values())) == len(mapping)
    # A player must not keep its own positional slot, or matching could be
    # "solved" by ordering alone.
    ordered_true = sorted(mapping.values())
    for index, synthetic_id in enumerate(sorted(mapping)):
        assert mapping[synthetic_id] != ordered_true[index]


def test_shuffle_preserves_team_membership(drifted):
    synthetic = drifted["synthetic"]
    truth_teams = (
        drifted["truth"].drop_duplicates("player_id").set_index("player_id")["team"]
    )
    synthetic_teams = (
        synthetic.data.drop_duplicates("player_id").set_index("player_id")["team"]
    )
    for synthetic_id, true_id in synthetic.identity_map.items():
        assert synthetic_teams[synthetic_id] == truth_teams[true_id]


def test_shuffling_can_be_disabled(quiet, metrica_home):
    assert set(quiet.data["player_id"].unique()) == set(metrica_home["player_id"].unique())


def test_inverse_mapping_round_trips(drifted):
    synthetic = drifted["synthetic"]
    for synthetic_id, true_id in synthetic.identity_map.items():
        assert synthetic.true_to_synthetic[true_id] == synthetic_id


# ---------------------------------------------------------------------------
# Clock error
# ---------------------------------------------------------------------------

def test_constant_offset_shifts_the_whole_clock(metrica_home):
    offset = 3.25
    synthetic = degrade(
        metrica_home,
        DegradationParams(position_noise_m=0.0, dropout_rate=0.0, n_dropout_bursts=0,
                          shuffle_identities=False, time_offset_s=offset,
                          time_drift_rate=0.0, seed=6),
    )
    shift = float(synthetic.data["timestamp"].min() - synthetic.clean["timestamp"].min())
    assert shift == pytest.approx(offset, abs=1e-6)


def test_drift_accumulates_linearly(metrica_home):
    offset, rate = 1.0, 2e-3
    synthetic = degrade(
        metrica_home,
        DegradationParams(position_noise_m=0.0, dropout_rate=0.0, n_dropout_bursts=0,
                          shuffle_identities=False, time_offset_s=offset,
                          time_drift_rate=rate, seed=6),
    )
    clean_t = synthetic.clean["timestamp"].to_numpy(dtype=float)
    t0 = clean_t.min()
    span = clean_t.max() - t0
    end_shift = float(synthetic.data["timestamp"].max() - clean_t.max())
    assert end_shift == pytest.approx(offset + rate * span, abs=1e-3)


def test_no_drift_leaves_a_constant_offset(metrica_home):
    offset = 2.0
    synthetic = degrade(
        metrica_home,
        DegradationParams(position_noise_m=0.0, dropout_rate=0.0, n_dropout_bursts=0,
                          shuffle_identities=False, time_offset_s=offset,
                          time_drift_rate=0.0, seed=6),
    )
    start = float(synthetic.data["timestamp"].min() - synthetic.clean["timestamp"].min())
    end = float(synthetic.data["timestamp"].max() - synthetic.clean["timestamp"].max())
    assert start == pytest.approx(end, abs=1e-6)
