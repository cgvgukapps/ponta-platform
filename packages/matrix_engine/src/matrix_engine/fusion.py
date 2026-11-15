#!/usr/bin/env python3
"""
fusion.py — combine several aligned sources into one trajectory per player.

Once sources share a clock and a player ID space, the remaining question is
what to believe when they disagree. Matrix answers it with inverse-variance
weighting: each source contributes in proportion to how precise it has been
shown to be, which is the minimum-variance combination for independent
estimates and needs no tuning beyond measuring the noise.

Two things set a source's weight at a given sample:

**Its measurement noise**, estimated from the data rather than declared.
    Player motion is smooth over a few frames, so the frame-to-frame jitter a
    source shows is almost entirely its own error. :func:`estimate_source_noise`
    reads that jitter off the second difference of each trajectory. A source
    that says it is accurate but jitters by a metre gets weighted as a
    metre-accurate source.

**How far the sample sits from a real observation.**
    ``sync.py`` records ``gap_s`` for every row. A position interpolated across
    a four-second dropout is a guess whose error grows with the square of the
    gap, and it should not carry the same weight as a measurement. The variance
    is inflated accordingly before weighting.

Honest note on what this proves. When one source is far more precise than the
other — the prototype's Metrica-versus-synthetic-GPS case — inverse-variance
weighting correctly collapses onto the better source, and the resulting error
reduction mostly demonstrates that Matrix identifies which source to trust. The
harder claim, that fusing two comparably noisy sources beats either, is tested
separately by degrading the ground truth into two independent sources; see
``matrix_engine/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from matrix_engine.schema import validate_schema

#: Smallest per-axis noise a source may be credited with, in metres. Without a
#: floor, a very smooth source produces a near-zero variance and a weight that
#: swamps every other source on arithmetic alone.
NOISE_FLOOR_M: float = 0.02

#: Typical player acceleration used to price interpolation error, in m/s^2.
#: Linear interpolation across a gap of g seconds under acceleration a is off
#: by at most a*g^2/8, which is the model applied to ``gap_s``.
INTERP_ACCEL_M_S2: float = 3.0

#: Fraction of the nominal sampling interval two timestamps may differ by and
#: still count as consecutive when estimating jitter.
_INTERVAL_TOLERANCE: float = 0.1


@dataclass
class FusionResult:
    """
    Output of :func:`fuse`.

    Attributes
    ----------
    data : pd.DataFrame
        One row per (timestamp, player). Carries the fused ``x``/``y`` plus
        ``n_sources``, ``source_attribution``, ``disagreement_m`` and
        ``max_gap_s``, which ``confidence.py`` turns into a score. When any
        input source reported ``native_quality``, the worst contributing value
        is carried through too.
    noise : dict[str, float]
        Per-source position noise, in metres per axis, as estimated or supplied.
    weights : pd.DataFrame
        Mean effective weight each source ended up carrying — the quickest way
        to see which feed the fusion actually believed.
    """

    data: pd.DataFrame
    noise: dict[str, float]
    weights: pd.DataFrame = field(repr=False)


def estimate_source_noise(
    df: pd.DataFrame, *, floor_m: float = NOISE_FLOOR_M
) -> float:
    """
    Estimate a source's per-axis position noise from trajectory jitter.

    Uses the second difference of each player's path. For white measurement
    noise of standard deviation sigma the second difference has variance
    6*sigma^2, while genuine motion contributes only a*dt^2 — 5 mm at 25 Hz for
    a 3 m/s^2 acceleration. The scale is read with a median-absolute-deviation
    estimator so that a handful of tracking glitches cannot inflate it.

    Only triples of genuinely consecutive samples at the source's nominal
    interval are used, so dropout gaps do not masquerade as noise.

    Parameters
    ----------
    df : pd.DataFrame
        Matrix-schema tracking for exactly one source, on its own clock and at
        its native rate. Estimating on resampled data would measure the
        interpolator instead of the sensor.
    floor_m : float
        Lower bound on the returned estimate.

    Returns
    -------
    float
        Estimated per-axis noise standard deviation, in metres.
    """
    if df.empty:
        return floor_m
    nominal_dt = 1.0 / float(df["sample_rate_hz"].iloc[0])
    tolerance = nominal_dt * _INTERVAL_TOLERANCE

    scales: list[float] = []
    for _, block in df.groupby("player_id", sort=False):
        block = block.sort_values("timestamp")
        t = block["timestamp"].to_numpy(dtype=float)
        if len(t) < 3:
            continue
        dt = np.diff(t)
        consecutive = np.abs(dt - nominal_dt) <= tolerance
        # A usable triple needs both of its intervals to be nominal.
        usable = consecutive[:-1] & consecutive[1:]
        if usable.sum() < 8:
            continue
        for axis in ("x", "y"):
            v = block[axis].to_numpy(dtype=float)
            second_diff = (v[2:] - 2 * v[1:-1] + v[:-2])[usable]
            scales.append(_mad_std(second_diff) / np.sqrt(6.0))

    if not scales:
        return floor_m
    return float(max(np.median(scales), floor_m))


def fuse(
    synced: pd.DataFrame,
    *,
    noise: dict[str, float] | None = None,
    interp_accel_m_s2: float = INTERP_ACCEL_M_S2,
) -> FusionResult:
    """
    Combine aligned, identity-matched sources into one trajectory per player.

    Parameters
    ----------
    synced : pd.DataFrame
        Matrix-schema tracking from two or more sources on a shared clock and a
        shared player ID space — i.e. ``sync.synchronize`` output with
        ``identity.apply_mapping`` already applied. An optional ``gap_s``
        column is used to price interpolated samples; absent, every sample is
        treated as directly observed.
    noise : dict[str, float], optional
        Per-source per-axis noise in metres. Estimated from ``synced`` when
        omitted, though estimating on native-rate data and passing the result
        in is more accurate — see :func:`estimate_source_noise`.
    interp_accel_m_s2 : float
        Acceleration used to price interpolation error across gaps.

    Returns
    -------
    FusionResult
    """
    validate_schema(synced, name="fusion input")
    if synced.empty:
        raise ValueError("cannot fuse an empty frame")

    sources = sorted(synced["source"].unique())
    if noise is None:
        noise = {
            name: estimate_source_noise(synced.loc[synced["source"] == name])
            for name in sources
        }
    missing = [s for s in sources if s not in noise]
    if missing:
        raise KeyError(f"no noise estimate supplied for source(s) {missing}")

    work = synced.loc[:, ["timestamp", "frame", "player_id", "team", "x", "y", "source"]].copy()
    gap = (
        synced["gap_s"].to_numpy(dtype=float)
        if "gap_s" in synced.columns
        else np.zeros(len(synced))
    )
    carry_quality = "native_quality" in synced.columns
    if carry_quality:
        work["native_quality"] = synced["native_quality"].to_numpy(dtype=float)

    sigma = work["source"].map(noise).to_numpy(dtype=float)
    interp_sigma = interp_accel_m_s2 * gap**2 / 8.0
    variance = sigma**2 + interp_sigma**2
    weight = 1.0 / np.maximum(variance, 1e-12)

    work["w"] = weight
    work["gap_s"] = gap
    work["wx"] = weight * work["x"]
    work["wy"] = weight * work["y"]
    work["x2"] = work["x"] ** 2
    work["y2"] = work["y"] ** 2
    # A bit per source lets the contributing set be summed like a number and
    # decoded afterwards, which is far cheaper than aggregating strings over
    # hundreds of thousands of groups.
    bit = {name: 1 << i for i, name in enumerate(sources)}
    work["bits"] = work["source"].map(bit).astype("int64")

    grouped = work.groupby(["timestamp", "player_id"], sort=False)
    aggregations = dict(
        frame=("frame", "first"),
        team=("team", "first"),
        w_sum=("w", "sum"),
        wx=("wx", "sum"),
        wy=("wy", "sum"),
        x_sum=("x", "sum"),
        y_sum=("y", "sum"),
        x2_sum=("x2", "sum"),
        y2_sum=("y2", "sum"),
        n_sources=("w", "size"),
        bits=("bits", "sum"),
        max_gap_s=("gap_s", "max"),
    )
    if carry_quality:
        # The worst reading among the sources that contributed. Pessimistic on
        # purpose: a fused position built partly on a sample its own sensor
        # flagged should not inherit the better sensor's clean bill of health.
        # NaN only survives where no contributing source reported anything.
        aggregations["native_quality"] = ("native_quality", "min")
    agg = grouped.agg(**aggregations).reset_index()

    x_fused = agg["wx"] / agg["w_sum"]
    y_fused = agg["wy"] / agg["w_sum"]
    n = agg["n_sources"].to_numpy(dtype=float)

    # RMS distance of the contributing sources from the fused position,
    # expanded from the group sums. For the usual two-source case this is half
    # the distance between the two feeds.
    spread_sq = (
        (agg["x2_sum"] - 2 * x_fused * agg["x_sum"] + n * x_fused**2)
        + (agg["y2_sum"] - 2 * y_fused * agg["y_sum"] + n * y_fused**2)
    ) / n
    disagreement = np.sqrt(np.clip(spread_sq.to_numpy(dtype=float), 0.0, None))

    fused = pd.DataFrame(
        {
            "timestamp": agg["timestamp"].astype("float64"),
            "frame": agg["frame"].astype("int64"),
            "player_id": agg["player_id"].astype("object"),
            "team": agg["team"].astype("object"),
            "x": x_fused.astype("float64"),
            "y": y_fused.astype("float64"),
            "source": "fused",
            "sample_rate_hz": _infer_rate(agg["timestamp"].to_numpy(dtype=float)),
            "n_sources": agg["n_sources"].astype("int64"),
            "source_attribution": _decode_bits(agg["bits"].to_numpy(dtype=np.int64), sources),
            "disagreement_m": disagreement,
            "max_gap_s": agg["max_gap_s"].astype("float64"),
        }
    )
    if carry_quality:
        fused["native_quality"] = agg["native_quality"].astype("float64").to_numpy()
    fused = fused.sort_values(["timestamp", "player_id"], kind="stable").reset_index(drop=True)
    validate_schema(fused, name="fused")

    weights = (
        work.groupby("source", sort=True)["w"]
        .mean()
        .rename("mean_weight")
        .reset_index()
        .assign(noise_m=lambda d: d["source"].map(noise))
    )
    return FusionResult(data=fused, noise=dict(noise), weights=weights)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mad_std(values: np.ndarray) -> float:
    """Standard deviation estimated from the median absolute deviation."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    mad = float(np.median(np.abs(finite - np.median(finite))))
    return mad / 0.6744897501960817


def _infer_rate(timestamps: np.ndarray) -> float:
    """Sampling rate of the fused grid, from the modal timestamp spacing."""
    unique = np.unique(timestamps)
    if unique.size < 2:
        return 1.0
    dt = np.median(np.diff(unique))
    return float(1.0 / dt) if dt > 0 else 1.0


def _decode_bits(bits: np.ndarray, sources: list[str]) -> np.ndarray:
    """Turn per-group source bitmasks back into '+'-joined source names."""
    labels: dict[int, str] = {}
    for mask in np.unique(bits):
        names = [name for i, name in enumerate(sources) if mask & (1 << i)]
        labels[int(mask)] = "+".join(names) if names else ""
    return np.array([labels[int(b)] for b in bits], dtype=object)
