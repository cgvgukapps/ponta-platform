#!/usr/bin/env python3
"""
quality.py — catch the failures that would otherwise be silently fused in.

Fusion always produces an answer. That is the danger: a sensor whose clock is
sliding, or which stopped reporting for ten seconds, still yields plausible
numbers. This module looks for those two failure modes explicitly and reports
them, so a bad feed is flagged rather than quietly averaged into the output.

**Sensor drift** (:func:`detect_clock_drift`).
    ``sync.py`` estimates the clock offset separately in each window and fits a
    line through them. The slope is the drift rate. Flagging it needs care in
    both directions: a slope is only meaningful if it is large compared with the
    scatter of the individual window estimates, and if the error it accumulates
    over the recording is actually big enough to matter. Requiring both keeps a
    clean sensor from being condemned by estimator noise, which is the failure
    that would make the check useless in practice.

**Dropout gaps** (:func:`detect_gaps`).
    Measured on each source's native timeline, before resampling papers over
    them. Reported per player, since GPS loss and optical occlusion are usually
    individual rather than match-wide.

**Self-reported quality** (:func:`native_quality_summary`).
    Both checks above are inferential: they catch a bad feed by its behaviour.
    A source that carries the vendor's own per-sample diagnostics can be asked
    directly instead, which is the only route to spotting a source degrading
    without either drifting or dropping out. Reported per source, and absent
    for the sources that ship no diagnostics at all.

The drift check has a real sensitivity limit, and it is stated rather than
hidden: drift is only detectable once the error it accumulates across the
recording exceeds roughly ``n_sigma`` times the per-window offset noise. On a
short window a slow drift is genuinely below the noise floor, and
:attr:`DriftReport.detectable_drift_rate` reports the threshold that applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from matrix_engine.sync import ClockAlignment

#: Drift must clear this many standard errors of the fitted slope to be flagged.
DEFAULT_N_SIGMA: float = 3.0

#: ...and must also accumulate at least this many seconds of error across the
#: recording. Guards against flagging a statistically clean but operationally
#: irrelevant slope on a very long, very quiet recording.
DEFAULT_MIN_ACCUMULATED_S: float = 0.20

#: A missing stretch is a gap once it exceeds this many nominal sample
#: intervals — long enough to be signal loss rather than one dropped frame.
DEFAULT_GAP_INTERVALS: float = 3.0

#: ``native_quality`` at or below which a sample counts as vendor-flagged.
#: 0.7 sits below anything the observed STATSports session produces from an
#: unflagged sample, so it selects rows the unit itself was unhappy with.
DEFAULT_POOR_QUALITY: float = 0.7

#: Fraction of vendor-flagged samples above which a source is worth raising.
#: A GPS session with a tenth of its samples flagged is not a normal session.
DEFAULT_POOR_QUALITY_FRACTION: float = 0.10


@dataclass
class DriftReport:
    """
    Whether a source's clock is drifting against the reference.

    Attributes
    ----------
    source, reference : str
    drift_rate : float
        Fitted slope, in seconds of offset per second elapsed.
    accumulated_s : float
        Offset change the slope implies across the analysed span.
    span_s : float
        Duration the estimate was fitted over.
    slope_stderr : float
        Standard error of the slope, from the scatter of window estimates.
    n_sigma : float
        ``|drift_rate| / slope_stderr`` — how far from zero the slope sits.
    detectable_drift_rate : float
        Smallest drift rate this recording could have flagged. The honest
        sensitivity limit for a "no drift" result.
    flagged : bool
        True when the slope is both statistically and operationally significant.
    reason : str
        Human-readable explanation of the verdict.
    """

    source: str
    reference: str
    drift_rate: float
    accumulated_s: float
    span_s: float
    slope_stderr: float
    n_sigma: float
    detectable_drift_rate: float
    flagged: bool
    reason: str


@dataclass
class QualityReport:
    """Everything :func:`assess_quality` found."""

    drift: dict[str, DriftReport]
    gaps: pd.DataFrame
    coverage: pd.DataFrame = field(repr=False)
    #: Per-source vendor diagnostics, empty when no source reports any. Defaulted
    #: so that callers constructing a report by hand keep working.
    native_quality: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    @property
    def drifting_sources(self) -> list[str]:
        return sorted(name for name, report in self.drift.items() if report.flagged)

    @property
    def poor_quality_sources(self) -> list[str]:
        """Sources whose own diagnostics flag an unusual share of their samples."""
        if self.native_quality.empty or "flagged" not in self.native_quality.columns:
            return []
        flagged = self.native_quality.loc[self.native_quality["flagged"], "source"]
        return sorted(flagged.astype(str).tolist())

    def gap_summary(self) -> pd.DataFrame:
        """Per-source totals: how many gaps, how long, worst case."""
        if self.gaps.empty:
            return pd.DataFrame(
                columns=["source", "n_gaps", "total_gap_s", "max_gap_s", "players_affected"]
            )
        return (
            self.gaps.groupby("source", sort=True)
            .agg(
                n_gaps=("duration_s", "size"),
                total_gap_s=("duration_s", "sum"),
                max_gap_s=("duration_s", "max"),
                players_affected=("player_id", "nunique"),
            )
            .reset_index()
        )


def detect_clock_drift(
    alignment: ClockAlignment,
    *,
    n_sigma: float = DEFAULT_N_SIGMA,
    min_accumulated_s: float = DEFAULT_MIN_ACCUMULATED_S,
) -> DriftReport:
    """
    Decide whether a fitted clock alignment shows genuine drift.

    Parameters
    ----------
    alignment : ClockAlignment
        From ``sync.estimate_clock_alignment`` or, preferably,
        ``sync.refine_alignment`` — the refined pass has far less per-window
        scatter and so can resolve much slower drift.
    n_sigma : float
        Significance the slope must reach against its own standard error.
    min_accumulated_s : float
        Minimum offset change across the span for the drift to be worth raising.

    Returns
    -------
    DriftReport
    """
    windows = alignment.window_offsets
    times = windows["time"].to_numpy(dtype=float)
    span = float(times.max() - times.min()) if len(times) > 1 else 0.0
    accumulated = abs(alignment.drift_rate) * span

    centred = times - times.mean()
    sum_sq = float(np.sum(centred**2))
    n = len(times)

    if n < 3 or sum_sq <= 0:
        return DriftReport(
            source=alignment.source,
            reference=alignment.reference,
            drift_rate=alignment.drift_rate,
            accumulated_s=accumulated,
            span_s=span,
            slope_stderr=float("inf"),
            n_sigma=0.0,
            detectable_drift_rate=float("inf"),
            flagged=False,
            reason=(
                f"only {n} window estimate(s) — need at least 3 to separate a "
                "slope from noise"
            ),
        )

    # Residual scatter of the window estimates about the fitted line, converted
    # into a standard error on the slope. Two degrees of freedom are spent on
    # the intercept and slope themselves.
    residual_std = alignment.residual_s * np.sqrt(n / max(n - 2, 1))
    slope_stderr = float(residual_std / np.sqrt(sum_sq)) if sum_sq > 0 else float("inf")
    observed_sigma = abs(alignment.drift_rate) / slope_stderr if slope_stderr > 0 else 0.0

    detectable = max(
        n_sigma * slope_stderr,
        min_accumulated_s / span if span > 0 else float("inf"),
    )

    significant = observed_sigma >= n_sigma
    material = accumulated >= min_accumulated_s
    flagged = bool(significant and material)

    if flagged:
        reason = (
            f"drift {alignment.drift_rate:+.2e} s/s accumulates "
            f"{accumulated:.3f} s over {span:.0f} s ({observed_sigma:.1f} sigma)"
        )
    elif not significant:
        reason = (
            f"slope {alignment.drift_rate:+.2e} s/s is only {observed_sigma:.1f} sigma "
            f"from zero; this recording could detect {detectable:.2e} s/s"
        )
    else:
        reason = (
            f"drift {alignment.drift_rate:+.2e} s/s accumulates only "
            f"{accumulated:.3f} s over {span:.0f} s, below the "
            f"{min_accumulated_s:.2f} s materiality threshold"
        )

    return DriftReport(
        source=alignment.source,
        reference=alignment.reference,
        drift_rate=float(alignment.drift_rate),
        accumulated_s=float(accumulated),
        span_s=span,
        slope_stderr=slope_stderr,
        n_sigma=float(observed_sigma),
        detectable_drift_rate=float(detectable),
        flagged=flagged,
        reason=reason,
    )


def detect_gaps(
    sources: dict[str, pd.DataFrame],
    *,
    gap_intervals: float = DEFAULT_GAP_INTERVALS,
    min_gap_s: float | None = None,
) -> pd.DataFrame:
    """
    Find stretches where a source stopped reporting a player.

    Run on native-rate source data, not on resampled output — resampling fills
    gaps in, which is exactly what this needs to see past.

    Parameters
    ----------
    sources : dict[str, pd.DataFrame]
        Matrix-schema tracking keyed by source name.
    gap_intervals : float
        A break counts as a gap once it exceeds this many nominal sample
        intervals for that source.
    min_gap_s : float, optional
        Absolute threshold overriding ``gap_intervals``.

    Returns
    -------
    pd.DataFrame
        Columns ``source`` / ``player_id`` / ``team`` / ``start`` / ``end`` /
        ``duration_s`` / ``missed_samples``, sorted longest first.
    """
    rows: list[dict] = []
    for name, df in sources.items():
        if df.empty:
            continue
        nominal_dt = 1.0 / float(df["sample_rate_hz"].iloc[0])
        threshold = min_gap_s if min_gap_s is not None else gap_intervals * nominal_dt

        for (player_id, team), block in df.groupby(["player_id", "team"], sort=True):
            t = np.sort(block["timestamp"].to_numpy(dtype=float))
            if len(t) < 2:
                continue
            dt = np.diff(t)
            for i in np.flatnonzero(dt > threshold):
                rows.append(
                    {
                        "source": name,
                        "player_id": player_id,
                        "team": team,
                        "start": float(t[i]),
                        "end": float(t[i + 1]),
                        "duration_s": float(dt[i]),
                        "missed_samples": int(round(dt[i] / nominal_dt)) - 1,
                    }
                )

    columns = ["source", "player_id", "team", "start", "end", "duration_s", "missed_samples"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("duration_s", ascending=False)
        .reset_index(drop=True)
    )


def coverage_summary(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-source shape of the feed: players, span, observed vs expected rows."""
    rows: list[dict] = []
    for name, df in sources.items():
        if df.empty:
            rows.append({"source": name, "players": 0, "rows": 0, "span_s": 0.0,
                         "sample_rate_hz": np.nan, "coverage": 0.0})
            continue
        rate = float(df["sample_rate_hz"].iloc[0])
        span = float(df["timestamp"].max() - df["timestamp"].min())
        players = int(df["player_id"].nunique())
        expected = players * span * rate
        rows.append(
            {
                "source": name,
                "players": players,
                "rows": int(len(df)),
                "span_s": span,
                "sample_rate_hz": rate,
                "coverage": float(len(df) / expected) if expected > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def native_quality_summary(
    sources: dict[str, pd.DataFrame],
    *,
    poor_quality: float = DEFAULT_POOR_QUALITY,
    poor_quality_fraction: float = DEFAULT_POOR_QUALITY_FRACTION,
) -> pd.DataFrame:
    """
    Summarise what each source says about its own samples.

    Only sources carrying a ``native_quality`` column appear. That is the
    honest shape of the result: a source without vendor diagnostics has not
    reported good quality, it has reported nothing, and a row of zeros or ones
    would misrepresent both.

    Parameters
    ----------
    sources : dict[str, pd.DataFrame]
        Matrix-schema tracking keyed by source name, at native rate.
    poor_quality : float
        Score at or below which a sample counts as vendor-flagged.
    poor_quality_fraction : float
        Flagged fraction above which the source is marked ``flagged``.

    Returns
    -------
    pd.DataFrame
        Columns ``source`` / ``rows`` / ``reported`` / ``mean_quality`` /
        ``p10_quality`` / ``min_quality`` / ``flagged_samples`` /
        ``flagged_fraction`` / ``flagged`` / ``reason``. Empty with those
        columns when no source reports quality.
    """
    columns = [
        "source", "rows", "reported", "mean_quality", "p10_quality", "min_quality",
        "flagged_samples", "flagged_fraction", "flagged", "reason",
    ]
    rows: list[dict] = []
    for name, df in sources.items():
        if df.empty or "native_quality" not in df.columns:
            continue
        values = df["native_quality"].to_numpy(dtype=float)
        known = values[np.isfinite(values)]
        if known.size == 0:
            continue
        n_flagged = int((known <= poor_quality).sum())
        fraction = n_flagged / known.size
        is_flagged = fraction > poor_quality_fraction
        rows.append(
            {
                "source": name,
                "rows": int(len(df)),
                "reported": int(known.size),
                "mean_quality": float(known.mean()),
                "p10_quality": float(np.percentile(known, 10)),
                "min_quality": float(known.min()),
                "flagged_samples": n_flagged,
                "flagged_fraction": float(fraction),
                "flagged": bool(is_flagged),
                "reason": (
                    f"{fraction:.1%} of samples at or below {poor_quality:.2f} "
                    f"native quality, over the {poor_quality_fraction:.0%} threshold"
                    if is_flagged
                    else f"{fraction:.1%} of samples flagged by the source itself, "
                    f"within the {poor_quality_fraction:.0%} threshold"
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("source").reset_index(drop=True)


def assess_quality(
    sources: dict[str, pd.DataFrame],
    alignments: dict[str, ClockAlignment],
    *,
    n_sigma: float = DEFAULT_N_SIGMA,
    min_accumulated_s: float = DEFAULT_MIN_ACCUMULATED_S,
    gap_intervals: float = DEFAULT_GAP_INTERVALS,
    poor_quality: float = DEFAULT_POOR_QUALITY,
) -> QualityReport:
    """Run every quality check and collect the results."""
    drift = {
        name: detect_clock_drift(
            alignment, n_sigma=n_sigma, min_accumulated_s=min_accumulated_s
        )
        for name, alignment in alignments.items()
    }
    return QualityReport(
        drift=drift,
        gaps=detect_gaps(sources, gap_intervals=gap_intervals),
        coverage=coverage_summary(sources),
        native_quality=native_quality_summary(sources, poor_quality=poor_quality),
    )
