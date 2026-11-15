"""
Drift is flagged when it is there and not when it isn't.

Both halves matter equally. A detector that flags everything catches every
drifting sensor and is worthless, so the negative control — the same pipeline on
a source with a constant offset and no drift — is tested as hard as the positive
case. The two runs differ only in ``time_drift_rate``.
"""

import numpy as np
import pandas as pd
import pytest

from matrix_engine.quality import (
    assess_quality,
    coverage_summary,
    detect_clock_drift,
    detect_gaps,
)
from matrix_engine.sync import ClockAlignment


def _alignment(offsets, times, *, drift_rate, residual, source="s"):
    """Build a ClockAlignment directly, to test the verdict logic in isolation."""
    return ClockAlignment(
        source=source,
        reference="ref",
        offset_s=float(np.mean(offsets)),
        drift_rate=drift_rate,
        t_ref=float(np.mean(times)),
        window_offsets=pd.DataFrame(
            {"time": times, "offset": offsets, "score": np.ones(len(times))}
        ),
        residual_s=residual,
        stage="refined",
    )


# ---------------------------------------------------------------------------
# Drift detection on the real pipeline
# ---------------------------------------------------------------------------

def test_injected_drift_is_flagged(drifted):
    report = drifted["result"].quality.drift["synthetic_gps"]
    assert report.flagged, report.reason


def test_flagged_drift_rate_matches_what_was_injected(drifted):
    report = drifted["result"].quality.drift["synthetic_gps"]
    injected = drifted["synthetic"].params.time_drift_rate
    assert report.drift_rate == pytest.approx(injected, rel=0.30)


def test_a_clean_clock_is_not_flagged(clean):
    """The negative control: a constant offset is not drift."""
    report = clean["result"].quality.drift["synthetic_gps"]
    assert not report.flagged, report.reason


def test_the_clean_run_still_has_a_constant_offset_to_correct(clean):
    """Confirms the negative control is not passing merely by having no offset."""
    alignment = clean["result"].sync.alignments["synthetic_gps"]
    assert abs(alignment.offset_s) > 1.0


def test_drifting_sources_are_listed(drifted, clean):
    assert drifted["result"].quality.drifting_sources == ["synthetic_gps"]
    assert clean["result"].quality.drifting_sources == []


def test_report_states_its_own_sensitivity_limit(clean):
    """
    A "no drift" verdict is only meaningful alongside what could have been
    detected, so the threshold is reported rather than implied.
    """
    report = clean["result"].quality.drift["synthetic_gps"]
    assert np.isfinite(report.detectable_drift_rate)
    assert report.detectable_drift_rate > 0
    assert abs(report.drift_rate) < report.detectable_drift_rate


def test_sensitivity_improves_with_a_longer_recording():
    """
    Detection depends on accumulated error, not drift rate, so the same clock
    becomes detectable given a longer recording. This is why a rate that is
    invisible over three minutes is obvious over ninety.
    """
    rate, residual = 1e-4, 0.005
    short_times = np.linspace(0.0, 200.0, 6)
    long_times = np.linspace(0.0, 5400.0, 6)

    short = detect_clock_drift(
        _alignment(2.0 + rate * short_times, short_times,
                   drift_rate=rate, residual=residual)
    )
    long = detect_clock_drift(
        _alignment(2.0 + rate * long_times, long_times,
                   drift_rate=rate, residual=residual)
    )
    assert not short.flagged
    assert long.flagged
    assert long.detectable_drift_rate < short.detectable_drift_rate


def test_reason_is_human_readable(drifted, clean):
    assert "accumulates" in drifted["result"].quality.drift["synthetic_gps"].reason
    assert clean["result"].quality.drift["synthetic_gps"].reason


# ---------------------------------------------------------------------------
# Drift verdict logic in isolation
# ---------------------------------------------------------------------------

def test_a_large_clean_slope_is_flagged():
    times = np.linspace(0.0, 300.0, 6)
    report = detect_clock_drift(
        _alignment(2.0 + 2e-3 * times, times, drift_rate=2e-3, residual=0.005)
    )
    assert report.flagged
    assert report.n_sigma > 3.0


def test_a_slope_buried_in_noise_is_not_flagged():
    """Same slope, scatter twenty times larger — it is no longer evidence."""
    times = np.linspace(0.0, 300.0, 6)
    report = detect_clock_drift(
        _alignment(2.0 + 2e-3 * times, times, drift_rate=2e-3, residual=0.5)
    )
    assert not report.flagged
    assert "sigma from zero" in report.reason


def test_a_statistically_clean_but_trivial_slope_is_not_flagged():
    """Precise enough to measure, too small to matter over the recording."""
    times = np.linspace(0.0, 300.0, 6)
    report = detect_clock_drift(
        _alignment(2.0 + 1e-5 * times, times, drift_rate=1e-5, residual=1e-9)
    )
    assert not report.flagged
    assert "materiality" in report.reason


def test_too_few_windows_yields_no_verdict():
    times = np.array([0.0, 100.0])
    report = detect_clock_drift(
        _alignment([2.0, 2.2], times, drift_rate=2e-3, residual=0.001)
    )
    assert not report.flagged
    assert "at least 3" in report.reason


def test_negative_drift_is_flagged_too():
    times = np.linspace(0.0, 300.0, 6)
    report = detect_clock_drift(
        _alignment(2.0 - 2e-3 * times, times, drift_rate=-2e-3, residual=0.005)
    )
    assert report.flagged


def test_accumulated_drift_is_reported_over_the_analysed_span():
    times = np.linspace(0.0, 300.0, 6)
    report = detect_clock_drift(
        _alignment(2.0 + 2e-3 * times, times, drift_rate=2e-3, residual=0.005)
    )
    assert report.span_s == pytest.approx(300.0)
    assert report.accumulated_s == pytest.approx(0.6, rel=1e-6)


# ---------------------------------------------------------------------------
# Dropout gap detection
# ---------------------------------------------------------------------------

def test_injected_bursts_are_detected(drifted):
    gaps = drifted["result"].quality.gaps
    synthetic = gaps[gaps["source"] == "synthetic_gps"]
    assert len(synthetic) > 0
    burst_s = drifted["synthetic"].params.dropout_burst_s
    assert synthetic["duration_s"].max() >= burst_s * 0.9


def test_every_player_with_an_injected_burst_shows_a_gap(drifted):
    gaps = drifted["result"].quality.gaps
    burst_s = drifted["synthetic"].params.dropout_burst_s
    long_gaps = gaps[gaps["duration_s"] > burst_s * 0.75]
    affected = set(long_gaps["player_id"].unique())
    expected = set(drifted["synthetic"].injected_gaps["player_id"].unique())
    assert expected <= affected


def test_the_dense_source_reports_no_gaps(drifted):
    gaps = drifted["result"].quality.gaps
    assert gaps[gaps["source"] == "metrica"].empty


def test_gap_summary_aggregates_per_source(drifted):
    summary = drifted["result"].quality.gap_summary()
    assert set(summary["source"]) == {"synthetic_gps"}
    row = summary.iloc[0]
    assert row["n_gaps"] > 0
    assert row["max_gap_s"] >= row["total_gap_s"] / row["n_gaps"]


def test_a_source_with_no_gaps_yields_an_empty_frame(metrica_home):
    assert detect_gaps({"metrica": metrica_home}).empty


def test_the_gap_threshold_is_adjustable(drifted):
    sources = {"synthetic_gps": drifted["synthetic"].data}
    loose = detect_gaps(sources, min_gap_s=3.0)
    tight = detect_gaps(sources, min_gap_s=0.15)
    assert len(tight) > len(loose)


def test_gaps_are_measured_on_native_data_not_resampled_output(drifted):
    """
    Resampling interpolates gaps away. Detecting them afterwards would find
    nothing, so this pins the input the detector is given.
    """
    resampled = drifted["result"].sync.data
    synthetic_resampled = resampled[resampled["source"] == "synthetic_gps"].copy()
    assert detect_gaps({"synthetic_gps": synthetic_resampled}).empty
    assert not drifted["result"].quality.gaps.empty


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_coverage_reflects_dropout(drifted):
    summary = coverage_summary(
        {"metrica": drifted["truth"], "synthetic_gps": drifted["synthetic"].data}
    ).set_index("source")
    assert summary.loc["metrica", "coverage"] > 0.95
    assert 0.5 < summary.loc["synthetic_gps", "coverage"] < 0.95


def test_assess_quality_collects_everything(drifted):
    report = assess_quality(
        {"metrica": drifted["truth"], "synthetic_gps": drifted["synthetic"].data},
        drifted["result"].sync.alignments,
    )
    assert "synthetic_gps" in report.drift
    assert not report.gaps.empty
    assert len(report.coverage) == 2
