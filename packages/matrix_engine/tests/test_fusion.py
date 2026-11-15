"""
The core claim: fused positions are closer to the truth than the degraded feed.

Two scenarios, because they prove different things.

**Metrica against a degraded copy** is the scenario the task specifies. It is
also the easy one: one input *is* the ground truth, so a fusion that simply
identified the better source would pass. It is worth testing — recognising which
feed to trust is a real capability, and getting it wrong would be a real failure
— but on its own it would overstate the result.

**Two independently degraded sources**, neither of which is the truth, is the
harder claim. Beating both inputs there requires genuinely combining
information, and the error should approach the 1/sqrt(2) that inverse-variance
weighting predicts for two equally noisy feeds.
"""

import numpy as np
import pytest

from matrix_engine.fusion import estimate_source_noise, fuse
from matrix_engine.schema import validate_schema
from matrix_engine.testing import DegradationParams, degrade


def _degraded_alone(run):
    """The synthetic source after sync and relabelling — the baseline to beat."""
    from matrix_engine.identity import apply_mapping

    synced = run["result"].sync.data
    synthetic = synced[synced["source"] == "synthetic_gps"]
    return apply_mapping(synthetic, run["result"].identity["synthetic_gps"].mapping)


# ---------------------------------------------------------------------------
# Scenario 1 — ground truth plus a degraded copy
# ---------------------------------------------------------------------------

def test_fused_output_is_schema_valid(drifted):
    validate_schema(drifted["result"].fused, name="fused")


def test_fused_beats_the_degraded_source_alone(drifted, position_rmse):
    truth = drifted["truth"]
    degraded_rmse, n_degraded = position_rmse(_degraded_alone(drifted), truth)
    fused_rmse, n_fused = position_rmse(drifted["result"].fused, truth)

    assert n_degraded > 0 and n_fused > 0
    assert fused_rmse < degraded_rmse, (
        f"fusion did not improve on the degraded source: "
        f"degraded {degraded_rmse:.4f} m, fused {fused_rmse:.4f} m"
    )


def test_the_improvement_is_large_not_marginal(drifted, position_rmse):
    degraded_rmse, _ = position_rmse(_degraded_alone(drifted), drifted["truth"])
    fused_rmse, _ = position_rmse(drifted["result"].fused, drifted["truth"])
    assert fused_rmse < degraded_rmse / 5.0


def test_fusion_weights_the_precise_source_far_higher(drifted):
    """
    Where one feed is much better, inverse-variance weighting should say so.
    This is the mechanism behind the scenario-1 result, asserted directly so a
    passing RMSE cannot hide a broken weighting.
    """
    weights = drifted["result"].fusion.weights.set_index("source")["mean_weight"]
    assert weights["metrica"] > 100 * weights["synthetic_gps"]


# ---------------------------------------------------------------------------
# Scenario 2 — two degraded sources, neither of them the truth
# ---------------------------------------------------------------------------

def test_fused_beats_both_degraded_sources(dual_degraded, position_rmse):
    truth = dual_degraded["truth"]
    shift = dual_degraded["truth_clock_shift"]
    result = dual_degraded["result"]

    from matrix_engine.identity import apply_mapping

    synced = result.sync.data
    a_alone = synced[synced["source"] == "gps_a"]
    b_alone = apply_mapping(
        synced[synced["source"] == "gps_b"], result.identity["gps_b"].mapping
    )

    a_rmse, _ = position_rmse(a_alone, truth, clock_shift=shift)
    b_rmse, _ = position_rmse(b_alone, truth, clock_shift=shift)
    fused_rmse, n = position_rmse(result.fused, truth, clock_shift=shift)

    assert n > 0
    assert fused_rmse < min(a_rmse, b_rmse), (
        f"fused {fused_rmse:.4f} m did not beat both inputs "
        f"(gps_a {a_rmse:.4f} m, gps_b {b_rmse:.4f} m)"
    )


def test_error_reduction_approaches_the_theoretical_limit(dual_degraded, position_rmse):
    """
    Averaging two independent estimates of equal precision cuts the error by
    1/sqrt(2), about 29%. Real dropout, interpolation and residual clock error
    mean the achieved reduction is smaller, but it should be in the same
    territory rather than a rounding error.
    """
    truth = dual_degraded["truth"]
    shift = dual_degraded["truth_clock_shift"]
    result = dual_degraded["result"]

    from matrix_engine.identity import apply_mapping

    synced = result.sync.data
    best_input = min(
        position_rmse(synced[synced["source"] == "gps_a"], truth, clock_shift=shift)[0],
        position_rmse(
            apply_mapping(synced[synced["source"] == "gps_b"],
                          result.identity["gps_b"].mapping),
            truth, clock_shift=shift,
        )[0],
    )
    fused_rmse, _ = position_rmse(result.fused, truth, clock_shift=shift)
    reduction = 1.0 - fused_rmse / best_input
    assert 0.10 < reduction < 0.45, f"error reduction {reduction:.1%} outside expectation"


def test_equally_noisy_sources_are_weighted_equally(dual_degraded):
    weights = dual_degraded["result"].fusion.weights.set_index("source")["mean_weight"]
    ratio = weights["gps_a"] / weights["gps_b"]
    assert 0.5 < ratio < 2.0


# ---------------------------------------------------------------------------
# Noise estimation, which drives everything above
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.3, 0.5, 1.0])
def test_noise_is_estimated_from_the_data_not_declared(metrica_home, sigma):
    synthetic = degrade(
        metrica_home,
        DegradationParams(position_noise_m=sigma, dropout_rate=0.1,
                          n_dropout_bursts=2, seed=21),
    )
    estimated = estimate_source_noise(synthetic.data)
    assert estimated == pytest.approx(sigma, rel=0.15), (
        f"estimated {estimated:.3f} m for an injected {sigma:.3f} m"
    )


def test_a_clean_source_estimates_near_the_noise_floor(metrica_home):
    from matrix_engine.fusion import NOISE_FLOOR_M

    assert estimate_source_noise(metrica_home) < 5 * NOISE_FLOOR_M


def test_dropout_gaps_do_not_inflate_the_noise_estimate(metrica_home):
    """Only genuinely consecutive samples are differenced, so gaps must not count."""
    without = degrade(
        metrica_home,
        DegradationParams(position_noise_m=0.5, dropout_rate=0.0,
                          n_dropout_bursts=0, seed=23),
    )
    with_gaps = degrade(
        metrica_home,
        DegradationParams(position_noise_m=0.5, dropout_rate=0.2,
                          n_dropout_bursts=3, dropout_burst_s=5.0, seed=23),
    )
    assert estimate_source_noise(with_gaps.data) == pytest.approx(
        estimate_source_noise(without.data), rel=0.20
    )


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_source_attribution_names_the_contributing_feeds(drifted):
    fused = drifted["result"].fused
    attributions = set(fused["source_attribution"].unique())
    assert "metrica+synthetic_gps" in attributions
    assert all(
        set(a.split("+")) <= {"metrica", "synthetic_gps"} for a in attributions
    )


def test_attribution_and_source_count_agree(drifted):
    fused = drifted["result"].fused
    counted = fused["source_attribution"].str.count(r"\+") + 1
    assert (counted == fused["n_sources"]).all()


def test_most_frames_are_covered_by_both_sources(drifted):
    fused = drifted["result"].fused
    assert (fused["n_sources"] == 2).mean() > 0.95


def test_disagreement_is_reported_and_plausible(drifted):
    fused = drifted["result"].fused
    both = fused[fused["n_sources"] == 2]
    assert (both["disagreement_m"] >= 0).all()
    # RMS distance from the fused point, which for two sources is half the
    # distance between them — so of order the injected per-axis noise.
    sigma = drifted["synthetic"].params.position_noise_m
    assert both["disagreement_m"].median() < 3 * sigma


def test_fusion_needs_a_noise_estimate_for_every_source(drifted):
    combined = drifted["result"].sync.data
    with pytest.raises(KeyError, match="noise estimate"):
        fuse(combined, noise={"metrica": 0.02})


def test_fusing_an_empty_frame_is_refused(drifted):
    with pytest.raises(ValueError, match="empty"):
        fuse(drifted["result"].sync.data.iloc[0:0])


def test_interpolated_samples_are_downweighted(metrica_home):
    """
    A position interpolated across a long gap should pull the fused result less
    than a directly observed one, even from the same source.
    """
    import pandas as pd

    base = dict(team="home", player_id="home_1", frame=0, sample_rate_hz=10.0)
    rows = pd.DataFrame(
        [
            {**base, "timestamp": 0.0, "x": 0.0, "y": 0.0, "source": "a", "gap_s": 0.0},
            {**base, "timestamp": 0.0, "x": 10.0, "y": 0.0, "source": "b", "gap_s": 5.0},
        ]
    )
    fused = fuse(rows, noise={"a": 0.5, "b": 0.5}).data
    assert fused["x"].iloc[0] < 2.0, "the interpolated sample dominated the fusion"
