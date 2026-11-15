"""
Two sources on different rates and different clocks end up on one timeline.

The offset is recovered twice — once from permutation-invariant team signals
before identity is known, once from matched players afterwards — so both passes
are tested, along with the claim that the second is materially sharper than the
first. That claim is the justification for running sync twice at all.
"""

import numpy as np
import pytest
from matrix_engine.identity import match_identities, split_sources
from matrix_engine.schema import validate_schema
from matrix_engine.sync import estimate_clock_alignment, refine_alignment, synchronize
from matrix_engine.testing import DegradationParams, degrade

#: Accuracy the centroid-based first pass is expected to reach on this window.
COARSE_TOLERANCE_S = 0.30

#: Accuracy expected once matched players are available.
REFINED_TOLERANCE_S = 0.05


def _true_offset(synthetic, at_time):
    """Offset the generator injected, evaluated on the ground-truth clock."""
    params = synthetic.params
    t0 = float(synthetic.clean["timestamp"].min())
    return params.time_offset_s + params.time_drift_rate * (at_time - t0)


# ---------------------------------------------------------------------------
# Pass 1 — coarse, identity-free
# ---------------------------------------------------------------------------

def test_coarse_pass_recovers_a_constant_offset(clean):
    alignment = estimate_clock_alignment(clean["truth"], clean["synthetic"].data)
    expected = _true_offset(clean["synthetic"], alignment.t_ref)
    assert alignment.offset_s == pytest.approx(expected, abs=COARSE_TOLERANCE_S)


def test_coarse_pass_works_without_any_shared_identifiers(clean):
    """
    The point of using centroid and spread: the two sources share no player IDs
    at all at this stage, and alignment still has to work.
    """
    truth_ids = set(clean["truth"]["player_id"].unique())
    synthetic_ids = set(clean["synthetic"].data["player_id"].unique())
    assert not (truth_ids & synthetic_ids)
    alignment = estimate_clock_alignment(clean["truth"], clean["synthetic"].data)
    assert np.isfinite(alignment.offset_s)


def test_coarse_pass_reports_one_estimate_per_window(clean):
    alignment = estimate_clock_alignment(
        clean["truth"], clean["synthetic"].data, n_windows=5
    )
    assert len(alignment.window_offsets) == 5
    assert alignment.stage == "coarse"


def test_offset_outside_the_search_range_is_not_silently_wrong(metrica_home):
    """A search that cannot reach the true offset must not return it by luck."""
    synthetic = degrade(
        metrica_home,
        DegradationParams(time_offset_s=12.0, position_noise_m=0.3, seed=2),
    )
    alignment = estimate_clock_alignment(metrica_home, synthetic.data, max_offset_s=3.0)
    assert abs(alignment.offset_s) <= 3.0 + 1e-6


def test_non_overlapping_sources_are_rejected(metrica_home):
    shifted = metrica_home.copy()
    shifted["timestamp"] = shifted["timestamp"] + 10_000.0
    shifted["source"] = "elsewhere"
    with pytest.raises(ValueError, match="do not overlap"):
        estimate_clock_alignment(metrica_home, shifted)


def test_multi_source_frame_is_rejected_as_an_input(drifted):
    combined = drifted["result"].sync.data
    with pytest.raises(ValueError, match="exactly one source"):
        estimate_clock_alignment(combined, drifted["synthetic"].data)


# ---------------------------------------------------------------------------
# Pass 2 — refined, identity-aware
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def refined(drifted):
    truth, synthetic = drifted["truth"], drifted["synthetic"]
    coarse = estimate_clock_alignment(truth, synthetic.data)
    synced = synchronize({"metrica": truth, "synthetic_gps": synthetic.data},
                         reference="metrica")
    ref_slice, other_slice = split_sources(synced.data, "metrica", "synthetic_gps")
    mapping = match_identities(ref_slice, other_slice).mapping
    return coarse, refine_alignment(truth, synthetic.data, mapping, coarse)


def test_refined_pass_recovers_the_offset_far_more_precisely(refined, drifted):
    coarse, fine = refined
    expected = _true_offset(drifted["synthetic"], fine.t_ref)
    assert fine.offset_s == pytest.approx(expected, abs=REFINED_TOLERANCE_S)
    assert fine.stage == "refined"


def test_refinement_reduces_window_to_window_scatter(refined):
    coarse, fine = refined
    assert fine.residual_s < coarse.residual_s / 2.0, (
        f"refinement did not sharpen the estimate: coarse {coarse.residual_s:.4f} s, "
        f"refined {fine.residual_s:.4f} s"
    )


def test_refined_pass_recovers_the_drift_rate(refined, drifted):
    _, fine = refined
    true_rate = drifted["synthetic"].params.time_drift_rate
    assert fine.drift_rate == pytest.approx(true_rate, rel=0.25)


def test_refinement_falls_back_when_no_identities_are_known(refined, drifted):
    coarse, _ = refined
    unchanged = refine_alignment(
        drifted["truth"], drifted["synthetic"].data, {}, coarse
    )
    assert unchanged is coarse


# ---------------------------------------------------------------------------
# Applying an alignment
# ---------------------------------------------------------------------------

def test_to_reference_clock_removes_the_offset(refined, drifted):
    _, fine = refined
    synthetic = drifted["synthetic"].data
    reported = synthetic["timestamp"].to_numpy(dtype=float)
    corrected = fine.to_reference_clock(reported)
    residual = (reported - corrected) - fine.offset_at(reported)
    assert np.allclose(residual, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Resampling onto a common grid
# ---------------------------------------------------------------------------

def test_synchronized_output_is_schema_valid(drifted):
    validate_schema(drifted["result"].sync.data, name="synced")


def test_all_sources_land_on_one_uniform_grid(drifted):
    synced = drifted["result"].sync
    times = np.unique(synced.data["timestamp"].to_numpy(dtype=float))
    step = 1.0 / synced.target_hz
    assert np.allclose(np.diff(times), step, atol=1e-6)


def test_every_source_shares_the_same_timestamps(drifted):
    synced = drifted["result"].sync.data
    per_source = [
        set(np.round(block["timestamp"].to_numpy(dtype=float), 6))
        for _, block in synced.groupby("source")
    ]
    shared = set.intersection(*per_source)
    # The 10 Hz source is upsampled onto the 25 Hz grid, so it should cover
    # essentially all of it rather than only every third sample.
    assert len(shared) > 0.9 * max(len(s) for s in per_source)


def test_target_rate_defaults_to_the_fastest_source(drifted):
    assert drifted["result"].sync.target_hz == 25.0


def test_explicit_target_rate_is_honoured(clean):
    synced = synchronize(
        {"metrica": clean["truth"], "synthetic_gps": clean["synthetic"].data},
        reference="metrica", target_hz=5.0,
    )
    times = np.unique(synced.data["timestamp"].to_numpy(dtype=float))
    assert np.allclose(np.diff(times), 0.2, atol=1e-6)


def test_alignment_actually_improves_position_agreement(drifted, position_rmse):
    """
    Synchronising without correcting the clock should be measurably worse. This
    is the end-to-end check that alignment is doing something, not just
    resampling tidily.
    """
    truth, synthetic = drifted["truth"], drifted["synthetic"].data
    sources = {"metrica": truth, "synthetic_gps": synthetic}

    aligned = synchronize(sources, reference="metrica")
    unaligned = synchronize(sources, reference="metrica", align_clocks=False)

    mapping = drifted["result"].identity["synthetic_gps"].mapping
    inverse = {v: k for k, v in mapping.items()}
    renamed_truth = truth.assign(player_id=truth["player_id"].map(inverse))

    aligned_rmse, _ = position_rmse(
        aligned.data[aligned.data["source"] == "synthetic_gps"], renamed_truth
    )
    unaligned_rmse, _ = position_rmse(
        unaligned.data[unaligned.data["source"] == "synthetic_gps"], renamed_truth
    )
    assert aligned_rmse < unaligned_rmse / 2.0


def test_gap_distance_is_recorded(drifted):
    synced = drifted["result"].sync.data
    assert "gap_s" in synced.columns
    synthetic = synced[synced["source"] == "synthetic_gps"]
    assert synthetic["gap_s"].max() > 1.0, "injected bursts should show as large gaps"
    reference = synced[synced["source"] == "metrica"]
    assert reference["gap_s"].max() < 0.1, "the dense source should have no real gaps"


def test_empty_source_is_rejected(metrica_home):
    with pytest.raises(ValueError, match="empty"):
        synchronize({"metrica": metrica_home, "nothing": metrica_home.iloc[0:0]})


def test_unknown_reference_is_rejected(metrica_home):
    with pytest.raises(KeyError):
        synchronize({"metrica": metrica_home}, reference="nope")
