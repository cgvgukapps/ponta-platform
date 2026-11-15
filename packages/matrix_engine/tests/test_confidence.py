"""
Confidence has to be lower exactly where the data is worse.

A score that is merely well-formed is useless; the property that makes it worth
shipping is that filtering on it removes the bad rows. So the tests compare
confidence inside injected dropout gaps against clean stretches, and under high
inter-source disagreement against agreement, on the same run.
"""

import numpy as np
import pandas as pd
import pytest
from matrix_engine.confidence import (
    DEFAULT_DISAGREEMENT_SCALE_M,
    confidence_summary,
    expected_disagreement,
    score_confidence,
)


@pytest.fixture(scope="module")
def scored(drifted):
    return drifted["result"].fused


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_every_row_is_scored_within_range(scored):
    assert scored["confidence"].notna().all()
    assert scored["confidence"].between(0.0, 1.0).all()


def test_the_three_factors_are_reported_alongside_the_score(scored):
    for column in ("coverage_factor", "gap_factor", "agreement_factor"):
        assert column in scored.columns
        assert scored[column].between(0.0, 1.0).all()


def test_score_is_the_product_of_its_factors(scored):
    product = (
        scored["coverage_factor"] * scored["gap_factor"] * scored["agreement_factor"]
    )
    assert np.allclose(product, scored["confidence"], atol=1e-9)


def test_scoring_does_not_disturb_the_fused_columns(drifted):
    fused = drifted["result"].fusion.data
    rescored = score_confidence(fused, noise=drifted["result"].fusion.noise)
    pd.testing.assert_frame_equal(rescored[fused.columns], fused)


# ---------------------------------------------------------------------------
# The property that matters: low where the data is bad
# ---------------------------------------------------------------------------

def test_confidence_is_lower_inside_injected_dropout_gaps(scored, drifted):
    """
    Compare rows the generator deliberately blanked against everything else.
    Gap windows are on the synthetic clock, so they are mapped back onto the
    reference timeline using the alignment the pipeline actually used.
    """
    alignment = drifted["result"].sync.alignments["synthetic_gps"]
    gaps = drifted["synthetic"].injected_gaps
    mapping = drifted["result"].identity["synthetic_gps"].mapping

    in_gap = pd.Series(False, index=scored.index)
    for row in gaps.itertuples():
        reference_id = mapping.get(row.player_id)
        if reference_id is None:
            continue
        start = float(alignment.to_reference_clock(np.array([row.start]))[0])
        end = float(alignment.to_reference_clock(np.array([row.end]))[0])
        in_gap |= (
            (scored["player_id"] == reference_id)
            & (scored["timestamp"] > start)
            & (scored["timestamp"] < end)
        )

    assert in_gap.sum() > 100, "no rows fell inside the injected gaps"
    inside = scored.loc[in_gap, "confidence"].mean()
    outside = scored.loc[~in_gap, "confidence"].mean()
    assert inside < outside, f"gap confidence {inside:.3f} not below clean {outside:.3f}"


def test_confidence_falls_monotonically_with_gap_length(scored):
    both = scored[scored["n_sources"] == 2]
    buckets = pd.cut(both["max_gap_s"], [-0.01, 0.05, 0.5, 1.5, 100.0])
    means = both.groupby(buckets, observed=True)["gap_factor"].mean()
    assert len(means) >= 3
    assert means.is_monotonic_decreasing


def test_confidence_is_lower_where_the_sources_disagree_most(scored):
    both = scored[scored["n_sources"] == 2]
    high = both["disagreement_m"] > both["disagreement_m"].quantile(0.9)
    low = both["disagreement_m"] < both["disagreement_m"].quantile(0.1)
    assert both.loc[high, "confidence"].mean() < both.loc[low, "confidence"].mean()


def test_single_source_rows_are_discounted():
    rows = pd.DataFrame(
        {
            "n_sources": [1, 2],
            "max_gap_s": [0.0, 0.0],
            "disagreement_m": [0.0, 0.0],
        }
    )
    scored = score_confidence(rows)
    assert scored["confidence"].iloc[0] < scored["confidence"].iloc[1]


def test_perfect_agreement_and_no_gap_scores_full_confidence():
    rows = pd.DataFrame(
        {"n_sources": [2], "max_gap_s": [0.0], "disagreement_m": [0.0]}
    )
    assert score_confidence(rows)["confidence"].iloc[0] == pytest.approx(1.0)


def test_wild_disagreement_collapses_the_score():
    rows = pd.DataFrame(
        {"n_sources": [2], "max_gap_s": [0.0], "disagreement_m": [25.0]}
    )
    assert score_confidence(rows, noise={"a": 0.5, "b": 0.5})["confidence"].iloc[0] < 0.01


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_disagreement_scale_follows_the_sources_own_noise():
    assert expected_disagreement({"a": 0.5, "b": 0.5}) == pytest.approx(0.5)
    assert expected_disagreement({"a": 0.1, "b": 0.1}) == pytest.approx(0.1)
    assert expected_disagreement({}) == DEFAULT_DISAGREEMENT_SCALE_M


def test_the_same_disagreement_is_judged_against_the_source_noise():
    """
    A metre apart is unremarkable for two GPS feeds and alarming for two
    optical ones. The score has to reflect that rather than using a fixed
    distance.
    """
    rows = pd.DataFrame(
        {"n_sources": [2], "max_gap_s": [0.0], "disagreement_m": [1.0]}
    )
    noisy = score_confidence(rows, noise={"a": 1.0, "b": 1.0})["confidence"].iloc[0]
    precise = score_confidence(rows, noise={"a": 0.05, "b": 0.05})["confidence"].iloc[0]
    assert precise < noisy


def test_sources_disagreeing_as_much_as_their_noise_predicts_stay_confident():
    rows = pd.DataFrame(
        {"n_sources": [2], "max_gap_s": [0.0], "disagreement_m": [0.5]}
    )
    scored = score_confidence(rows, noise={"a": 0.5, "b": 0.5})
    assert scored["confidence"].iloc[0] > 0.85


def test_missing_optional_columns_are_treated_neutrally():
    scored = score_confidence(pd.DataFrame({"x": [1.0, 2.0]}))
    assert scored["confidence"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def test_summary_breaks_confidence_down_by_attribution(scored):
    summary = confidence_summary(scored)
    assert set(summary["source_attribution"]) <= {"metrica", "synthetic_gps",
                                                  "metrica+synthetic_gps"}
    assert summary["rows"].sum() == len(scored)
    assert summary["mean_confidence"].between(0.0, 1.0).all()
