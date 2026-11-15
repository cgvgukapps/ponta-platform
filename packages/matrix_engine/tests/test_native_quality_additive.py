#!/usr/bin/env python3
"""
Proof that the ``native_quality`` extension changed nothing for the sources that
do not report it.

The claim being defended is narrow and worth stating precisely: adding a fourth
confidence factor, and threading an optional column through ``sync`` and
``fusion``, must leave Track A and Track B numerically identical. Not "close" —
identical, because a source with no vendor diagnostics has no reason to be scored
differently than it was before the column existed.

Three independent angles on that, because "the old tests still pass" only shows
nobody wrote an assertion tight enough to catch a small drift:

1. The original three-factor formula is reimplemented here from its definition
   and asserted bit-identical against what ``score_confidence`` now returns for
   the real Track B fixtures.
2. The optional column must be *absent*, not present-and-neutral, so no consumer
   can accidentally start depending on it.
3. A perfect-quality column must be exactly neutral, which is what makes the
   fourth factor safe to introduce at all.
"""

import numpy as np
import pandas as pd
import pytest
from matrix_engine import fuse, synchronize
from matrix_engine.confidence import (
    DISAGREEMENT_TOLERANCE,
    GAP_HALF_LIFE_S,
    NATIVE_QUALITY_FLOOR,
    SINGLE_SOURCE_COVERAGE,
    expected_disagreement,
    score_confidence,
)
from matrix_engine.quality import assess_quality, native_quality_summary

#: Columns fused output carried before this change. Anything new appearing here
#: for a source without diagnostics would be a silent contract change.
ORIGINAL_FUSED_COLUMNS = {
    "timestamp", "frame", "player_id", "team", "x", "y", "source",
    "sample_rate_hz", "n_sources", "source_attribution", "disagreement_m",
    "max_gap_s",
}

ORIGINAL_CONFIDENCE_COLUMNS = ORIGINAL_FUSED_COLUMNS | {
    "coverage_factor", "gap_factor", "agreement_factor", "confidence",
}


def _confidence_before_the_change(
    fused: pd.DataFrame, noise: dict[str, float] | None
) -> np.ndarray:
    """
    The original three-factor score, written out from its definition.

    Deliberately not a call into ``confidence.py``: an independent restatement is
    what makes this a regression test rather than a tautology.
    """
    n_rows = len(fused)
    n_sources = fused["n_sources"].to_numpy(dtype=float) if "n_sources" in fused else np.ones(n_rows)
    gap = fused["max_gap_s"].to_numpy(dtype=float) if "max_gap_s" in fused else np.zeros(n_rows)
    disagreement = (
        fused["disagreement_m"].to_numpy(dtype=float)
        if "disagreement_m" in fused
        else np.zeros(n_rows)
    )

    scale = max(expected_disagreement(noise) if noise else 0.5, 1e-6)
    coverage = np.where(n_sources >= 2, 1.0, SINGLE_SOURCE_COVERAGE)
    gap_factor = 1.0 / (1.0 + (np.maximum(gap, 0.0) / GAP_HALF_LIFE_S) ** 2)
    ratio = np.maximum(disagreement, 0.0) / (scale * DISAGREEMENT_TOLERANCE)
    agreement = np.where(n_sources >= 2, np.exp(-0.5 * ratio**2), 1.0)
    return np.clip(coverage * gap_factor * agreement, 0.0, 1.0)


@pytest.mark.parametrize("scenario", ["drifted", "clean", "dual_degraded"])
def test_confidence_is_bit_identical_to_the_pre_change_formula(scenario, request):
    """Every original Track B scenario, scored by both implementations."""
    bundle = request.getfixturevalue(scenario)
    result = bundle["result"]

    expected = _confidence_before_the_change(result.fusion.data, result.fusion.noise)
    actual = result.fused["confidence"].to_numpy(dtype=float)

    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("scenario", ["drifted", "clean", "dual_degraded"])
def test_no_native_quality_column_appears_for_sources_without_diagnostics(scenario, request):
    bundle = request.getfixturevalue(scenario)
    result = bundle["result"]

    assert set(result.fusion.data.columns) == ORIGINAL_FUSED_COLUMNS
    assert set(result.fused.columns) == ORIGINAL_CONFIDENCE_COLUMNS
    assert "native_quality" not in result.sync.data.columns
    assert "native_quality_factor" not in result.fused.columns


def test_the_quality_report_reports_no_diagnostics_rather_than_perfect_ones(drifted):
    """
    An empty table is the correct answer for Metrica and a synthetic GPS copy:
    neither ships vendor diagnostics, and a row of 1.0s would claim they did.
    """
    report = drifted["result"].quality
    assert report.native_quality.empty
    assert report.poor_quality_sources == []
    # The checks that existed before still populate.
    assert report.drift
    assert not report.coverage.empty


def test_a_perfect_quality_column_is_exactly_neutral(dual_degraded):
    """
    The property that makes the fourth factor safe: at quality 1.0 it multiplies
    by 1.0, so the only thing that can move a score is a genuine vendor flag.
    """
    fused = dual_degraded["result"].fusion.data
    noise = dual_degraded["result"].fusion.noise

    without = score_confidence(fused, noise=noise)
    with_perfect = score_confidence(fused.assign(native_quality=1.0), noise=noise)

    np.testing.assert_array_equal(
        without["confidence"].to_numpy(), with_perfect["confidence"].to_numpy()
    )
    assert with_perfect["native_quality_factor"].eq(1.0).all()


def test_a_missing_quality_reading_is_neutral_not_a_penalty(dual_degraded):
    """
    NaN means "no source had an opinion here", which happens as soon as a feed
    with diagnostics is fused with one without. Treating it as bad would punish
    every row the optical source covered alone.
    """
    fused = dual_degraded["result"].fusion.data
    quality = np.full(len(fused), np.nan)
    quality[: len(fused) // 2] = 1.0

    scored = score_confidence(fused.assign(native_quality=quality))
    assert scored["native_quality_factor"].eq(1.0).all()


def test_a_poor_reading_scales_confidence_by_the_documented_floor():
    """The factor is a stated linear map, not an opaque adjustment."""
    fused = pd.DataFrame(
        {
            "timestamp": [0.0, 0.1, 0.2],
            "frame": [0, 1, 2],
            "player_id": ["a", "a", "a"],
            "team": ["unknown"] * 3,
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "source": ["fused"] * 3,
            "sample_rate_hz": [10.0] * 3,
            "n_sources": [2, 2, 2],
            "max_gap_s": [0.0, 0.0, 0.0],
            "disagreement_m": [0.0, 0.0, 0.0],
            "native_quality": [1.0, 0.5, 0.0],
        }
    )
    scored = score_confidence(fused)

    assert scored["confidence"].iloc[0] == pytest.approx(1.0)
    assert scored["confidence"].iloc[2] == pytest.approx(NATIVE_QUALITY_FLOOR)
    assert scored["confidence"].iloc[1] == pytest.approx(
        NATIVE_QUALITY_FLOOR + (1.0 - NATIVE_QUALITY_FLOOR) * 0.5
    )


def test_sync_and_fusion_only_carry_quality_when_a_source_supplies_it(metrica_home):
    """
    The plumbing has to be conditional at every hop, or Track A sources would
    start emitting a column they know nothing about.
    """
    plain = synchronize({"metrica": metrica_home}, reference="metrica")
    assert "native_quality" not in plain.data.columns
    assert "native_quality" not in fuse(plain.data).data.columns

    with_quality = metrica_home.assign(native_quality=0.5)
    carried = synchronize({"metrica": with_quality}, reference="metrica")
    assert "native_quality" in carried.data.columns
    assert carried.data["native_quality"].eq(0.5).all()
    assert "native_quality" in fuse(carried.data).data.columns


def test_fusion_keeps_the_worst_contributing_quality_reading(metrica_home):
    """
    Pessimistic on purpose: a fused position built partly on a flagged sample
    should not inherit the clean source's verdict.
    """
    window = metrica_home.loc[metrica_home["timestamp"] < metrica_home["timestamp"].min() + 2.0]
    good = window.assign(native_quality=1.0, source="good")
    bad = window.assign(native_quality=0.2, source="bad")

    fused = fuse(pd.concat([good, bad], ignore_index=True), noise={"good": 0.1, "bad": 0.1}).data
    assert fused["native_quality"].eq(0.2).all()


def test_assess_quality_still_reports_everything_it_used_to(drifted):
    """The new table is an addition to the report, not a replacement of it."""
    sources = {
        "metrica": drifted["truth"],
        "synthetic_gps": drifted["synthetic"].data,
    }
    alignments = drifted["result"].sync.alignments
    report = assess_quality(sources, alignments)

    assert set(report.drift) == set(alignments)
    assert set(report.coverage["source"]) == set(sources)
    assert report.native_quality.empty
    assert native_quality_summary(sources).empty
