#!/usr/bin/env python3
"""
confidence.py — say how much to trust each fused position.

A fused dataset that looks uniformly authoritative is worse than the raw feeds
it replaced, because it hides where the answer was interpolated across a
four-second dropout or where two sensors disagreed by five metres. Every fused
row therefore carries a score in [0, 1], and a downstream consumer can filter
on it before, say, computing a sprint count.

The score is the product of three independent factors, kept separate rather
than blended into an opaque weight so a low score can be explained:

**Coverage** — was more than one source watching?
    A position only one feed reported cannot be cross-checked at all. It is not
    necessarily wrong, so the penalty is a discount rather than a rejection.

**Gap** — how far is this sample from a real observation?
    ``sync.py`` records ``gap_s``. Confidence falls off quadratically with it,
    so a position interpolated across a long signal loss is visibly weaker than
    one interpolated between adjacent samples.

**Agreement** — did the sources that saw this player agree?
    Disagreement is judged against what the sources' own measured noise
    predicts, not against a fixed distance. Two GPS-grade feeds a metre apart
    is ordinary; two optical feeds a metre apart means something is wrong, and
    the same score should not describe both.

**Native quality** — did the sensor itself report a problem? *(optional)*
    The three factors above share a blind spot: all of them infer trouble from
    the *relationship* between sources or samples. A lone source that quietly
    degrades while still emitting plausible positions at a steady rate looks
    perfect to every one of them. Where a vendor ships per-sample diagnostics —
    satellite geometry, reported horizontal accuracy — its provider condenses
    them into ``native_quality``, and this factor is the only way that
    information reaches the score.

    This factor applies only when the column is present. Sources without it are
    scored by exactly the three original factors, and no extra column is added
    to their output, so their confidence values are unchanged bit for bit.

Multiplying rather than averaging is deliberate: any one of these being bad is
enough to make the position untrustworthy, and averaging would let two good
factors mask a fatal third.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Multiplier applied when only one source saw the player.
SINGLE_SOURCE_COVERAGE: float = 0.7

#: Gap, in seconds, at which the gap factor falls to 0.5.
GAP_HALF_LIFE_S: float = 1.0

#: Fallback expected inter-source disagreement, in metres, when no per-source
#: noise estimate is available to derive one from.
DEFAULT_DISAGREEMENT_SCALE_M: float = 0.5

#: How many multiples of the *expected* disagreement are tolerated before the
#: agreement factor falls to 0.6. Sources disagreeing by exactly what their
#: noise predicts are behaving normally and should not be marked down for it;
#: the factor is meant to catch disagreement that the noise cannot explain.
DISAGREEMENT_TOLERANCE: float = 2.0

#: Multiplier applied where a source reports its worst possible native quality.
#: A floor rather than zero, matching how :data:`SINGLE_SOURCE_COVERAGE` treats
#: an uncorroborated position: the sensor is saying "do not lean on this", not
#: "this is fabricated", and a flagged sample is still usually within a few
#: metres. Interpolating between the floor and 1 keeps the factor linear in the
#: reported quality, which is as much structure as an uncalibrated vendor
#: diagnostic supports.
NATIVE_QUALITY_FLOOR: float = 0.25


def expected_disagreement(noise: dict[str, float]) -> float:
    """
    Inter-source spread that the sources' own noise levels predict.

    ``fusion.py`` reports ``disagreement_m`` as the RMS distance of the
    contributing sources from the fused position. For two sources with per-axis
    noise ``s1`` and ``s2`` that quantity has expectation ``sqrt((s1^2+s2^2)/2)``,
    which generalises to the root-mean-square of the per-source noise levels.
    """
    values = [v for v in noise.values() if np.isfinite(v) and v > 0]
    if not values:
        return DEFAULT_DISAGREEMENT_SCALE_M
    return float(np.sqrt(np.mean(np.square(values))))


def score_confidence(
    fused: pd.DataFrame,
    *,
    noise: dict[str, float] | None = None,
    disagreement_scale_m: float | None = None,
    gap_half_life_s: float = GAP_HALF_LIFE_S,
    single_source_coverage: float = SINGLE_SOURCE_COVERAGE,
    disagreement_tolerance: float = DISAGREEMENT_TOLERANCE,
    native_quality_floor: float = NATIVE_QUALITY_FLOOR,
) -> pd.DataFrame:
    """
    Attach a per-row confidence score, and the factors behind it, to fused output.

    Parameters
    ----------
    fused : pd.DataFrame
        ``fusion.fuse`` output. Uses ``n_sources``, ``max_gap_s`` and
        ``disagreement_m``; any that are absent are treated as the neutral case.
        An optional ``native_quality`` column adds a fourth factor.
    noise : dict[str, float], optional
        Per-source noise from fusion, used to set the disagreement scale.
    disagreement_scale_m : float, optional
        Overrides the scale derived from ``noise``.
    gap_half_life_s : float
        Gap at which the gap factor reaches 0.5.
    single_source_coverage : float
        Coverage factor for rows only one source contributed to.
    disagreement_tolerance : float
        Multiples of the expected disagreement tolerated before the agreement
        factor drops sharply.
    native_quality_floor : float
        Factor applied at a reported quality of 0. Only consulted when the
        input carries ``native_quality``.

    Returns
    -------
    pd.DataFrame
        A copy of ``fused`` with ``confidence`` plus the ``coverage_factor``,
        ``gap_factor`` and ``agreement_factor`` it was built from, and
        ``native_quality_factor`` when the input reported native quality. That
        column is omitted rather than filled with ones when it does not apply,
        so output for a source without vendor diagnostics is identical to what
        this function produced before the factor existed.
    """
    out = fused.copy()
    n_rows = len(out)

    n_sources = (
        out["n_sources"].to_numpy(dtype=float)
        if "n_sources" in out.columns
        else np.ones(n_rows)
    )
    gap = (
        out["max_gap_s"].to_numpy(dtype=float)
        if "max_gap_s" in out.columns
        else np.zeros(n_rows)
    )
    disagreement = (
        out["disagreement_m"].to_numpy(dtype=float)
        if "disagreement_m" in out.columns
        else np.zeros(n_rows)
    )

    if disagreement_scale_m is None:
        disagreement_scale_m = (
            expected_disagreement(noise) if noise else DEFAULT_DISAGREEMENT_SCALE_M
        )
    disagreement_scale_m = max(float(disagreement_scale_m), 1e-6)
    gap_half_life_s = max(float(gap_half_life_s), 1e-6)

    coverage = np.where(n_sources >= 2, 1.0, single_source_coverage)
    gap_factor = 1.0 / (1.0 + (np.maximum(gap, 0.0) / gap_half_life_s) ** 2)

    # Only meaningful where at least two sources could be compared; a lone
    # source reports zero disagreement by construction, which is not evidence
    # of agreement.
    tolerance = max(float(disagreement_tolerance), 1e-6)
    ratio = np.maximum(disagreement, 0.0) / (disagreement_scale_m * tolerance)
    agreement = np.where(n_sources >= 2, np.exp(-0.5 * ratio**2), 1.0)

    out["coverage_factor"] = coverage
    out["gap_factor"] = gap_factor
    out["agreement_factor"] = agreement

    score = coverage * gap_factor * agreement
    if "native_quality" in out.columns:
        quality = out["native_quality"].to_numpy(dtype=float)
        floor = float(np.clip(native_quality_floor, 0.0, 1.0))
        native_factor = floor + (1.0 - floor) * np.clip(quality, 0.0, 1.0)
        # A row no source had an opinion about is not evidence of a problem.
        native_factor = np.where(np.isnan(quality), 1.0, native_factor)
        out["native_quality_factor"] = native_factor
        score = score * native_factor

    out["confidence"] = np.clip(score, 0.0, 1.0)
    return out


def confidence_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate confidence by source attribution — a quick read on which
    combinations of feeds are producing trustworthy positions.
    """
    group_col = "source_attribution" if "source_attribution" in scored.columns else "source"
    return (
        scored.groupby(group_col, sort=True)
        .agg(
            rows=("confidence", "size"),
            mean_confidence=("confidence", "mean"),
            p10_confidence=("confidence", lambda s: float(np.percentile(s, 10))),
            mean_gap_s=("max_gap_s", "mean") if "max_gap_s" in scored.columns
            else ("confidence", "size"),
        )
        .reset_index()
    )
