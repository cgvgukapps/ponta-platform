#!/usr/bin/env python3
"""
sync.py — put independently-clocked sources on one timeline.

Two feeds of the same match rarely agree on time. They start counting at
different moments, sample at different rates, and their clocks can drift apart
over 90 minutes. Everything after this module — identity matching, fusion,
confidence — assumes a shared clock, so this is the stage that has to earn it.

Alignment happens in two passes, because of a chicken-and-egg problem: the
sharpest way to compare two feeds is player against player, but Matrix cannot
assume the feeds label players the same way, and identity matching itself needs
an aligned clock to work from.

**Pass 1 — coarse, identity-free** (:func:`estimate_clock_alignment`).
    Correlates per-team **centroid** and **spread**, which are invariant to
    relabelling players and degrade gracefully when a source misses some of
    them. On the prototype's sample window this recovers the offset to roughly
    0.1 s: enough for identity matching, not enough for fusion.

**Pass 2 — fine, identity-aware** (:func:`refine_alignment`).
    Once ``identity.py`` has matched players, the offset is re-estimated by
    minimising position disagreement between matched pairs. Every player
    becomes an independent estimate of the same quantity, which tightens the
    result by roughly an order of magnitude and makes the per-window estimates
    precise enough to fit a meaningful drift slope.

Both passes estimate the offset separately in several windows and fit a line
through the per-window estimates: a constant offset is the intercept, drift is
the slope. ``quality.py`` reads the same fit to decide whether to flag a
drifting sensor.

Resampling then interpolates every player onto one uniform grid. Interpolation
across a dropout gap is allowed but recorded: each output row carries ``gap_s``,
the time to the nearest genuine observation, so downstream stages can tell a
measured position from an invented one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from matrix_engine.schema import coerce_schema, validate_schema

#: Rate of the internal grid used for correlation, in Hz.
_FEATURE_GRID_HZ: float = 10.0

#: Moving-average applied to the centroid/spread channels, in seconds. Smooths
#: out the jitter caused by the set of visible players changing frame to frame.
_FEATURE_SMOOTH_S: float = 0.4

#: Minimum finite samples a window needs before its estimate is trusted.
_MIN_WINDOW_SAMPLES: int = 20

#: Resolution of the sub-grid-step refinement, in seconds.
_FINE_STEP_S: float = 0.005


@dataclass
class ClockAlignment:
    """
    How one source's clock relates to the reference's.

    ``offset_at(t)`` gives the seconds the source's reported clock runs ahead of
    the reference clock at time ``t``::

        reference_time ≈ reported_time - offset_at(reported_time)

    Attributes
    ----------
    source, reference : str
        The source this alignment describes, and the source whose clock is
        treated as the timeline.
    offset_s : float
        Estimated offset at ``t_ref``, in seconds.
    drift_rate : float
        Estimated change in offset per second elapsed. Dimensionless.
    t_ref : float
        Reference time the intercept is anchored to.
    window_offsets : pd.DataFrame
        Per-window estimates (``time`` / ``offset`` / ``score``) the line was
        fitted through. Consumed by drift detection and useful for inspection
        when an alignment looks wrong.
    residual_s : float
        RMS of the per-window estimates about the fitted line — the usable
        measure of how much to trust this alignment, and the noise floor any
        drift claim has to clear.
    stage : str
        ``"coarse"`` (centroid-based) or ``"refined"`` (identity-based).
    """

    source: str
    reference: str
    offset_s: float
    drift_rate: float
    t_ref: float
    window_offsets: pd.DataFrame = field(repr=False)
    residual_s: float = 0.0
    stage: str = "coarse"

    def offset_at(self, t: np.ndarray | float) -> np.ndarray | float:
        return self.offset_s + self.drift_rate * (np.asarray(t, dtype=float) - self.t_ref)

    def to_reference_clock(self, reported: np.ndarray) -> np.ndarray:
        """
        Map reported timestamps onto the reference timeline.

        The drift term is evaluated at the reported time rather than solving
        exactly for the true time; the difference is second-order in
        ``drift_rate`` (sub-millisecond for any realistic sensor).
        """
        reported = np.asarray(reported, dtype=float)
        return reported - self.offset_at(reported)


@dataclass
class SyncResult:
    """Output of :func:`synchronize`."""

    data: pd.DataFrame
    reference: str
    target_hz: float
    alignments: dict[str, ClockAlignment]
    grid: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# Pass 1 — coarse, identity-free alignment
# ---------------------------------------------------------------------------

def estimate_clock_alignment(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    *,
    max_offset_s: float = 15.0,
    n_windows: int = 6,
    grid_hz: float = _FEATURE_GRID_HZ,
) -> ClockAlignment:
    """
    Estimate ``other``'s clock offset and drift against ``reference``'s, without
    knowing which player is which.

    Parameters
    ----------
    reference, other : pd.DataFrame
        Matrix-schema tracking from two sources.
    max_offset_s : float
        Half-width of the lag search. The true offset must fall inside it or
        the peak is outside the search and the estimate is meaningless.
    n_windows : int
        Windows the overlap is split into. More windows give more points to fit
        drift through, at the cost of a noisier estimate in each.
    grid_hz : float
        Resolution of the internal correlation grid.

    Returns
    -------
    ClockAlignment
        With ``stage="coarse"``.
    """
    ref_name = _single_source_name(reference, "reference")
    other_name = _single_source_name(other, "other")

    step = 1.0 / grid_hz
    lo = min(float(reference["timestamp"].min()), float(other["timestamp"].min()))
    hi = max(float(reference["timestamp"].max()), float(other["timestamp"].max()))
    grid = np.arange(lo, hi + step / 2.0, step)

    ref_feats = _team_features(reference, grid, grid_hz)
    other_feats = _team_features(other, grid, grid_hz)
    channels = sorted(set(ref_feats) & set(other_feats))
    if not channels:
        raise ValueError(
            f"no shared teams with enough coverage between '{ref_name}' and "
            f"'{other_name}' — cannot align clocks"
        )

    edges = _window_edges(reference, other, n_windows, ref_name, other_name)
    max_lag = int(round(max_offset_s * grid_hz))

    rows: list[dict] = []
    for i in range(n_windows):
        idx = np.flatnonzero((grid >= edges[i]) & (grid < edges[i + 1]))
        if len(idx) < _MIN_WINDOW_SAMPLES:
            continue
        lag, score = _coarse_lag(ref_feats, other_feats, channels, idx, max_lag)
        if lag is None:
            continue
        lag_s, score = _refine_lag(
            ref_feats, other_feats, channels, grid, idx, lag * step, step
        )
        rows.append({"time": float(0.5 * (edges[i] + edges[i + 1])), "offset": lag_s,
                     "score": score})

    return _fit_alignment(
        rows, other_name, ref_name, stage="coarse",
        failure_hint="no window produced a usable correlation peak",
    )


# ---------------------------------------------------------------------------
# Pass 2 — fine, identity-aware alignment
# ---------------------------------------------------------------------------

def refine_alignment(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    identity_map: dict[str, str],
    base: ClockAlignment,
    *,
    search_half_width_s: float = 1.0,
    coarse_step_s: float = 0.05,
    n_windows: int = 6,
) -> ClockAlignment:
    """
    Sharpen a coarse alignment using players that have already been matched.

    For each window and each candidate lag, matched players' positions are
    compared directly and the lag minimising mean squared disagreement wins.
    Unlike the centroid pass this uses every player as an independent estimate,
    so the per-window estimates get precise enough for the drift slope to mean
    something.

    Parameters
    ----------
    reference, other : pd.DataFrame
        The *original* sources, each on its own reported clock.
    identity_map : dict[str, str]
        ``other``'s player_id -> ``reference``'s player_id, from ``identity.py``.
    base : ClockAlignment
        The coarse alignment to search around.
    search_half_width_s : float
        How far either side of ``base`` to search. Should comfortably exceed the
        coarse pass's residual.
    coarse_step_s : float
        Step of the initial scan before sub-step refinement.
    n_windows : int
        Windows to estimate the offset in.

    Returns
    -------
    ClockAlignment
        With ``stage="refined"``. Falls back to ``base`` if too few windows
        could be scored.
    """
    ref_name = _single_source_name(reference, "reference")
    other_name = _single_source_name(other, "other")
    if not identity_map:
        return base

    pairs = _matched_series(reference, other, identity_map)
    if not pairs:
        return base

    edges = _window_edges(reference, other, n_windows, ref_name, other_name)
    rows: list[dict] = []
    for i in range(n_windows):
        w_lo, w_hi = edges[i], edges[i + 1]
        base_offset = float(base.offset_at(0.5 * (w_lo + w_hi)))
        lag, cost, n_used = _best_lag_by_disagreement(
            pairs, w_lo, w_hi, base_offset, search_half_width_s, coarse_step_s
        )
        if lag is None:
            continue
        rows.append(
            {
                "time": float(0.5 * (w_lo + w_hi)),
                "offset": lag,
                # Report agreement, not cost, so higher is better in both passes.
                "score": float(1.0 / (1.0 + cost)),
                "n_samples": n_used,
            }
        )

    if len(rows) < 2:
        return base
    return _fit_alignment(
        rows, other_name, ref_name, stage="refined",
        failure_hint="matched players did not overlap in any window",
    )


# ---------------------------------------------------------------------------
# Resampling onto a common grid
# ---------------------------------------------------------------------------

def synchronize(
    sources: dict[str, pd.DataFrame],
    *,
    reference: str | None = None,
    target_hz: float | None = None,
    max_offset_s: float = 15.0,
    n_windows: int = 6,
    max_interp_gap_s: float = 15.0,
    align_clocks: bool = True,
    alignments: dict[str, ClockAlignment] | None = None,
) -> SyncResult:
    """
    Align every source's clock to the reference and resample onto one grid.

    Parameters
    ----------
    sources : dict[str, pd.DataFrame]
        Matrix-schema tracking keyed by source name.
    reference : str, optional
        Source whose clock defines the timeline. Defaults to the
        highest-sample-rate source, usually the most trustworthy clock.
    target_hz : float, optional
        Output grid rate. Defaults to the highest native rate present, so
        fusion does not discard the best source's temporal resolution.
    max_offset_s, n_windows :
        See :func:`estimate_clock_alignment`.
    max_interp_gap_s : float
        Grid points further than this from a genuine observation are dropped
        rather than invented.
    align_clocks : bool
        Set False when the sources are known to share a clock already.
    alignments : dict[str, ClockAlignment], optional
        Pre-computed alignments to apply instead of estimating fresh ones. This
        is how a refined second pass is fed back in.

    Returns
    -------
    SyncResult
        ``.data`` concatenates every source on the common grid, with an extra
        ``gap_s`` column giving each row's distance to the nearest genuine
        observation (0.0 where a sample landed on one).
    """
    if not sources:
        raise ValueError("synchronize needs at least one source")
    for name, df in sources.items():
        validate_schema(df, name=name)
        if df.empty:
            raise ValueError(f"source '{name}' is empty")

    if reference is None:
        reference = max(sources, key=lambda n: float(sources[n]["sample_rate_hz"].max()))
    if reference not in sources:
        raise KeyError(f"reference source '{reference}' not among {sorted(sources)}")
    if target_hz is None:
        target_hz = float(max(df["sample_rate_hz"].max() for df in sources.values()))

    resolved: dict[str, ClockAlignment] = {}
    on_ref_clock: dict[str, pd.DataFrame] = {}
    for name, df in sources.items():
        if name == reference or not align_clocks:
            on_ref_clock[name] = df
            continue
        alignment = (alignments or {}).get(name)
        if alignment is None:
            alignment = estimate_clock_alignment(
                sources[reference], df, max_offset_s=max_offset_s, n_windows=n_windows
            )
        resolved[name] = alignment
        corrected = df.copy()
        corrected["timestamp"] = alignment.to_reference_clock(
            corrected["timestamp"].to_numpy(dtype=float)
        )
        on_ref_clock[name] = corrected

    grid_lo = max(float(df["timestamp"].min()) for df in on_ref_clock.values())
    grid_hi = min(float(df["timestamp"].max()) for df in on_ref_clock.values())
    if grid_hi <= grid_lo:
        raise ValueError(
            "sources do not overlap after clock alignment — check max_offset_s"
        )
    step = 1.0 / target_hz
    grid = np.arange(grid_lo, grid_hi + step / 2.0, step)

    data = pd.concat(
        [
            _resample_source(df, grid, target_hz, max_interp_gap_s)
            for df in on_ref_clock.values()
        ],
        ignore_index=True,
    )
    data = data.sort_values(["timestamp", "source", "player_id"], kind="stable")
    data = coerce_schema(data).reset_index(drop=True)
    validate_schema(data, name="synchronized")

    return SyncResult(
        data=data,
        reference=reference,
        target_hz=float(target_hz),
        alignments=resolved,
        grid=grid,
    )


# ---------------------------------------------------------------------------
# Internal helpers — shared
# ---------------------------------------------------------------------------

def _single_source_name(df: pd.DataFrame, role: str) -> str:
    names = df["source"].unique()
    if len(names) != 1:
        raise ValueError(f"{role} must contain exactly one source, found {list(names)}")
    return str(names[0])


def _window_edges(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    n_windows: int,
    ref_name: str,
    other_name: str,
) -> np.ndarray:
    lo = max(float(reference["timestamp"].min()), float(other["timestamp"].min()))
    hi = min(float(reference["timestamp"].max()), float(other["timestamp"].max()))
    if hi <= lo:
        raise ValueError(
            f"'{ref_name}' and '{other_name}' do not overlap in time even before "
            "offset correction"
        )
    return np.linspace(lo, hi, n_windows + 1)


def _fit_alignment(
    rows: list[dict],
    source: str,
    reference: str,
    *,
    stage: str,
    failure_hint: str,
) -> ClockAlignment:
    """Fit offset + drift through per-window estimates."""
    window_offsets = pd.DataFrame(rows)
    if window_offsets.empty:
        raise ValueError(
            f"could not estimate a clock offset between '{reference}' and "
            f"'{source}' — {failure_hint}"
        )

    t_ref = float(window_offsets["time"].mean())
    centred = window_offsets["time"].to_numpy(dtype=float) - t_ref
    offsets = window_offsets["offset"].to_numpy(dtype=float)

    if len(window_offsets) >= 3:
        drift_rate, offset_s = np.polyfit(centred, offsets, 1)
        residual = float(
            np.sqrt(np.mean((offsets - (offset_s + drift_rate * centred)) ** 2))
        )
    else:
        drift_rate, offset_s = 0.0, float(np.median(offsets))
        residual = float(np.std(offsets))

    return ClockAlignment(
        source=source,
        reference=reference,
        offset_s=float(offset_s),
        drift_rate=float(drift_rate),
        t_ref=t_ref,
        window_offsets=window_offsets,
        residual_s=residual,
        stage=stage,
    )


# ---------------------------------------------------------------------------
# Internal helpers — coarse pass
# ---------------------------------------------------------------------------

def _team_features(
    df: pd.DataFrame, grid: np.ndarray, grid_hz: float
) -> dict[str, np.ndarray]:
    """
    Summarise each team per timestamp in a way relabelling cannot change.

    Three channels per team — centroid x, centroid y, and radial spread —
    interpolated onto ``grid`` and lightly smoothed. Smoothing matters because
    which players a source sees changes frame to frame, and that alone moves an
    11-player centroid by a metre or two.
    """
    features: dict[str, np.ndarray] = {}
    window = max(1, int(round(_FEATURE_SMOOTH_S * grid_hz)))

    for team, block in df.groupby("team", sort=True):
        agg = _centroid_agg(block)
        agg = agg.loc[agg["n"] >= 2]
        if len(agg) < _MIN_WINDOW_SAMPLES:
            continue

        variance = (agg["mx2"] - agg["cx"] ** 2) + (agg["my2"] - agg["cy"] ** 2)
        agg = agg.assign(spread=np.sqrt(np.clip(variance.to_numpy(dtype=float), 0.0, None)))

        t = agg.index.to_numpy(dtype=float)
        in_span = (grid >= t[0]) & (grid <= t[-1])
        for channel in ("cx", "cy", "spread"):
            values = np.full(len(grid), np.nan)
            values[in_span] = np.interp(
                grid[in_span], t, agg[channel].to_numpy(dtype=float)
            )
            features[f"{team}.{channel}"] = _smooth(values, window)
    return features


def _centroid_agg(block: pd.DataFrame) -> pd.DataFrame:
    """Per-timestamp centroid and mean-square terms, computed without apply()."""
    work = block.loc[:, ["timestamp", "x", "y"]].copy()
    work["mx2"] = work["x"] ** 2
    work["my2"] = work["y"] ** 2
    return work.groupby("timestamp").agg(
        cx=("x", "mean"), cy=("y", "mean"),
        mx2=("mx2", "mean"), my2=("my2", "mean"),
        n=("x", "size"),
    )


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average that leaves out-of-span NaNs as NaN."""
    if window <= 1:
        return values
    nan = np.isnan(values)
    if nan.all():
        return values
    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    smoothed = np.convolve(filled, np.ones(window) / window, mode="same")
    smoothed[nan] = np.nan
    return smoothed


def _coarse_lag(
    ref_feats: dict[str, np.ndarray],
    other_feats: dict[str, np.ndarray],
    channels: list[str],
    idx: np.ndarray,
    max_lag: int,
) -> tuple[int | None, float]:
    """Scan integer grid lags for the best mean channel correlation."""
    lags = np.arange(-max_lag, max_lag + 1)
    total = np.zeros(len(lags))
    counts = np.zeros(len(lags))
    n_grid = len(next(iter(ref_feats.values())))

    for channel in channels:
        ref_seg = ref_feats[channel][idx]
        other = other_feats[channel]
        for k, lag in enumerate(lags):
            shifted = idx + lag
            inside = (shifted >= 0) & (shifted < n_grid)
            if inside.sum() < _MIN_WINDOW_SAMPLES:
                continue
            a = ref_seg[inside]
            b = other[shifted[inside]]
            both = np.isfinite(a) & np.isfinite(b)
            if both.sum() < _MIN_WINDOW_SAMPLES:
                continue
            corr = _pearson(a[both], b[both])
            if np.isfinite(corr):
                total[k] += corr
                counts[k] += 1

    usable = counts > 0
    if not usable.any():
        return None, 0.0
    mean_scores = np.where(usable, total / np.maximum(counts, 1), -np.inf)
    peak = int(np.argmax(mean_scores))
    return int(lags[peak]), float(mean_scores[peak])


def _refine_lag(
    ref_feats: dict[str, np.ndarray],
    other_feats: dict[str, np.ndarray],
    channels: list[str],
    grid: np.ndarray,
    idx: np.ndarray,
    lag0_s: float,
    grid_step_s: float,
) -> tuple[float, float]:
    """Search sub-grid-step lags around ``lag0_s`` by interpolating the other source."""
    candidates = np.arange(
        lag0_s - grid_step_s, lag0_s + grid_step_s + _FINE_STEP_S / 2, _FINE_STEP_S
    )
    scores = np.array(
        [
            _corr_at_lag(ref_feats, other_feats, channels, grid, idx, lag)
            for lag in candidates
        ]
    )
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def _corr_at_lag(
    ref_feats: dict[str, np.ndarray],
    other_feats: dict[str, np.ndarray],
    channels: list[str],
    grid: np.ndarray,
    idx: np.ndarray,
    lag: float,
) -> float:
    shifted_times = grid[idx] + lag
    total, used = 0.0, 0
    for channel in channels:
        a = ref_feats[channel][idx]
        other = other_feats[channel]
        finite = np.isfinite(other)
        if finite.sum() < _MIN_WINDOW_SAMPLES:
            continue
        b = np.interp(
            shifted_times, grid[finite], other[finite], left=np.nan, right=np.nan
        )
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() < _MIN_WINDOW_SAMPLES:
            continue
        corr = _pearson(a[both], b[both])
        if np.isfinite(corr):
            total += corr
            used += 1
    return total / used if used else -np.inf


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return np.nan
    return float((a * b).sum() / denom)


# ---------------------------------------------------------------------------
# Internal helpers — refined pass
# ---------------------------------------------------------------------------

def _matched_series(
    reference: pd.DataFrame, other: pd.DataFrame, identity_map: dict[str, str]
) -> list[dict]:
    """Pre-extract per-pair arrays so the lag scan does no pandas work."""
    ref_by_player = {
        pid: block.sort_values("timestamp")
        for pid, block in reference.groupby("player_id", sort=False)
    }
    pairs: list[dict] = []
    for other_id, ref_id in identity_map.items():
        ref_block = ref_by_player.get(ref_id)
        if ref_block is None:
            continue
        other_block = other.loc[other["player_id"] == other_id].sort_values("timestamp")
        if len(other_block) < _MIN_WINDOW_SAMPLES or len(ref_block) < 2:
            continue
        pairs.append(
            {
                "ref_t": ref_block["timestamp"].to_numpy(dtype=float),
                "ref_x": ref_block["x"].to_numpy(dtype=float),
                "ref_y": ref_block["y"].to_numpy(dtype=float),
                "other_t": other_block["timestamp"].to_numpy(dtype=float),
                "other_x": other_block["x"].to_numpy(dtype=float),
                "other_y": other_block["y"].to_numpy(dtype=float),
            }
        )
    return pairs


def _disagreement_at_lag(pairs: list[dict], w_lo: float, w_hi: float, lag: float) -> tuple[float, int]:
    """Mean squared distance between matched players when ``other`` is shifted by ``lag``."""
    total, count = 0.0, 0
    for pair in pairs:
        t = pair["other_t"] - lag
        inside = (t >= w_lo) & (t < w_hi)
        if not inside.any():
            continue
        t = t[inside]
        ref_t = pair["ref_t"]
        covered = (t >= ref_t[0]) & (t <= ref_t[-1])
        if not covered.any():
            continue
        t = t[covered]
        dx = np.interp(t, ref_t, pair["ref_x"]) - pair["other_x"][inside][covered]
        dy = np.interp(t, ref_t, pair["ref_y"]) - pair["other_y"][inside][covered]
        total += float(np.sum(dx * dx + dy * dy))
        count += len(t)
    if count == 0:
        return np.inf, 0
    return total / count, count


def _best_lag_by_disagreement(
    pairs: list[dict],
    w_lo: float,
    w_hi: float,
    base_offset: float,
    half_width: float,
    coarse_step: float,
) -> tuple[float | None, float, int]:
    """Scan, then refine, the lag that minimises matched-player disagreement."""
    coarse = np.arange(
        base_offset - half_width, base_offset + half_width + coarse_step / 2, coarse_step
    )
    costs = np.array([_disagreement_at_lag(pairs, w_lo, w_hi, lag)[0] for lag in coarse])
    if not np.isfinite(costs).any():
        return None, np.inf, 0
    peak = int(np.argmin(np.where(np.isfinite(costs), costs, np.inf)))

    fine = np.arange(
        coarse[peak] - coarse_step, coarse[peak] + coarse_step + _FINE_STEP_S / 2,
        _FINE_STEP_S,
    )
    results = [_disagreement_at_lag(pairs, w_lo, w_hi, lag) for lag in fine]
    fine_costs = np.array([r[0] for r in results])
    best = int(np.argmin(np.where(np.isfinite(fine_costs), fine_costs, np.inf)))
    return float(fine[best]), float(fine_costs[best]), int(results[best][1])


# ---------------------------------------------------------------------------
# Internal helpers — resampling
# ---------------------------------------------------------------------------

def _resample_source(
    df: pd.DataFrame, grid: np.ndarray, target_hz: float, max_interp_gap_s: float
) -> pd.DataFrame:
    """Interpolate one source's players onto ``grid``, recording gap distance."""
    blocks: list[pd.DataFrame] = []
    source_name = str(df["source"].iloc[0])
    native_hz = float(df["sample_rate_hz"].iloc[0])
    t0 = float(grid[0])
    # A source's own quality reading is only useful downstream if it survives
    # resampling, so it rides along on the same interpolation as the position.
    carry_quality = "native_quality" in df.columns

    for (player_id, team), block in df.groupby(["player_id", "team"], sort=True):
        block = block.sort_values("timestamp")
        t = block["timestamp"].to_numpy(dtype=float)
        if len(t) < 2:
            continue
        in_span = (grid >= t[0]) & (grid <= t[-1])
        if not in_span.any():
            continue
        g = grid[in_span]

        gap = _distance_to_nearest(g, t)
        keep = gap <= max_interp_gap_s
        if not keep.any():
            continue

        resampled = pd.DataFrame(
            {
                "timestamp": g[keep],
                "frame": np.rint((g[keep] - t0) * target_hz).astype("int64"),
                "player_id": player_id,
                "team": team,
                "x": np.interp(g, t, block["x"].to_numpy(dtype=float))[keep],
                "y": np.interp(g, t, block["y"].to_numpy(dtype=float))[keep],
                "source": source_name,
                "sample_rate_hz": native_hz,
                "gap_s": gap[keep],
            }
        )
        if carry_quality:
            quality = block["native_quality"].to_numpy(dtype=float)
            observed = np.isfinite(quality)
            if observed.any():
                resampled["native_quality"] = np.interp(
                    g, t[observed], quality[observed]
                )[keep]
            else:
                resampled["native_quality"] = np.nan
        blocks.append(resampled)

    if not blocks:
        raise ValueError(f"source '{source_name}' had no player overlapping the grid")
    return pd.concat(blocks, ignore_index=True)


def _distance_to_nearest(grid: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Time from each grid point to the closest genuine observation."""
    pos = np.searchsorted(observed, grid)
    left = np.clip(pos - 1, 0, len(observed) - 1)
    right = np.clip(pos, 0, len(observed) - 1)
    return np.minimum(np.abs(grid - observed[left]), np.abs(grid - observed[right]))
