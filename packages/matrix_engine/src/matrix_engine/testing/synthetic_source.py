#!/usr/bin/env python3
"""
synthetic_source.py — build a degraded "second source" from a known ground truth.

There is no real GPS/wearable data in this repo, and no two recordings of the
same match from different vendors. Validating fusion still requires a second
stream whose true answer is known, so this module derives one: take a real
Metrica source and corrupt it the way an independent sensor would differ from
an optical one.

Five independent degradations, each separately configurable:

- **Rate**       resample onto the second source's own, lower sampling grid.
- **Noise**      additive Gaussian position error (GPS-grade, not optical-grade).
- **Dropout**    random and/or bursty missing observations (signal loss).
- **Identity**   relabel players into a disjoint ID space with no shared scheme.
- **Clock**      constant offset plus optional slow linear drift.

The returned :class:`SyntheticSource` carries the degraded data *and* the
answer key — the true identity mapping, the exact parameters, the injected gap
windows, and the pre-noise clean signal. Only tests and the demo's scoring step
may read the answer key. The reconciliation pipeline is given
``SyntheticSource.data`` and nothing else, exactly as if it were a real feed.

This proves the reconciliation math is correct. It does not prove any specific
real vendor pair reconciles cleanly — see ``matrix_engine/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from matrix_engine.schema import coerce_schema, validate_schema

#: Default ``source`` label for a generated second source.
SOURCE_NAME: str = "synthetic_gps"

#: Minimum observations a player needs before it is worth degrading.
_MIN_PLAYER_SAMPLES: int = 4


@dataclass(frozen=True)
class DegradationParams:
    """
    Knobs describing how far the synthetic source departs from ground truth.

    Attributes
    ----------
    target_hz : float
        Sampling rate of the synthetic source. Positions are interpolated onto
        a uniform grid at this rate rather than decimated, since an independent
        sensor's samples do not land on the reference sensor's frame times.
    position_noise_m : float
        Standard deviation of independent Gaussian noise added to x and y.
        0.3–1.0 m emulates GPS-grade rather than optical-grade accuracy.
    dropout_rate : float
        Fraction of observations dropped independently at random.
    n_dropout_bursts : int
        Number of contiguous signal-loss gaps injected per player, on top of
        ``dropout_rate``. Bursts are what make dropout detectable as a gap
        rather than as thinning.
    dropout_burst_s : float
        Duration of each burst, in seconds.
    shuffle_identities : bool
        Relabel players into a disjoint ID space. The permutation is a
        derangement per team, so no player keeps a recoverable label.
    time_offset_s : float
        Constant clock offset added to every reported timestamp.
    time_drift_rate : float
        Linear clock drift, in seconds of error per second elapsed. 1e-3 means
        the synthetic clock gains 1 ms every second (3.6 s over a full match).
    seed : int
        Seed for all randomness, so a run is reproducible.
    """

    target_hz: float = 10.0
    position_noise_m: float = 0.5
    dropout_rate: float = 0.10
    n_dropout_bursts: int = 2
    dropout_burst_s: float = 4.0
    shuffle_identities: bool = True
    time_offset_s: float = 2.5
    time_drift_rate: float = 0.0
    seed: int = 0


@dataclass
class SyntheticSource:
    """
    A degraded source plus the answer key needed to score reconciliation.

    Attributes
    ----------
    data : pd.DataFrame
        Matrix-schema tracking. This is the *only* field the pipeline sees.
    identity_map : dict[str, str]
        Synthetic ``player_id`` -> true ``player_id``. Test/scoring use only.
    params : DegradationParams
        The exact parameters used.
    injected_gaps : pd.DataFrame
        Columns ``player_id`` / ``start`` / ``end`` describing every burst gap,
        with times on the synthetic source's own (offset/drifted) clock.
    clean : pd.DataFrame
        The resampled signal before noise, dropout and clock error — the
        noise-free reference for checking the generator itself.
    """

    data: pd.DataFrame
    identity_map: dict[str, str]
    params: DegradationParams
    injected_gaps: pd.DataFrame
    clean: pd.DataFrame = field(repr=False)

    @property
    def true_to_synthetic(self) -> dict[str, str]:
        """Inverse of :attr:`identity_map`."""
        return {v: k for k, v in self.identity_map.items()}


def degrade(
    ground_truth: pd.DataFrame,
    params: DegradationParams | None = None,
    *,
    source_name: str = SOURCE_NAME,
) -> SyntheticSource:
    """
    Derive a degraded second source from a Matrix-schema ground-truth frame.

    Parameters
    ----------
    ground_truth : pd.DataFrame
        Validated Matrix-schema tracking, treated as the true positions.
    params : DegradationParams, optional
        Degradation configuration; defaults are a moderate GPS-like source.
    source_name : str
        Value written to the ``source`` column of the result.

    Returns
    -------
    SyntheticSource
    """
    params = params or DegradationParams()
    validate_schema(ground_truth, name="ground_truth")
    if ground_truth.empty:
        raise ValueError("cannot degrade an empty ground-truth frame")
    if params.target_hz <= 0:
        raise ValueError("target_hz must be positive")
    if not 0.0 <= params.dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")

    rng = np.random.default_rng(params.seed)

    clean = _resample(ground_truth, params.target_hz, source_name)
    identity_map = (
        _derangement_map(clean, rng) if params.shuffle_identities
        else {p: p for p in clean["player_id"].unique()}
    )

    degraded = clean.copy()
    degraded["x"] += rng.normal(0.0, params.position_noise_m, len(degraded))
    degraded["y"] += rng.normal(0.0, params.position_noise_m, len(degraded))

    true_to_syn = {v: k for k, v in identity_map.items()}
    degraded["player_id"] = degraded["player_id"].map(true_to_syn)

    degraded, gaps = _apply_dropout(degraded, params, rng)

    # The sensor reports its own faulty clock: a fixed offset plus, if
    # configured, an error that accumulates linearly with elapsed time.
    t0 = float(clean["timestamp"].min())
    elapsed = degraded["timestamp"] - t0
    degraded["timestamp"] = (
        degraded["timestamp"] + params.time_offset_s + params.time_drift_rate * elapsed
    )
    if not gaps.empty:
        gap_elapsed_start = gaps["start"] - t0
        gap_elapsed_end = gaps["end"] - t0
        gaps["start"] = (
            gaps["start"] + params.time_offset_s + params.time_drift_rate * gap_elapsed_start
        )
        gaps["end"] = (
            gaps["end"] + params.time_offset_s + params.time_drift_rate * gap_elapsed_end
        )

    degraded["source"] = source_name
    degraded = degraded.sort_values(["timestamp", "player_id"], kind="stable")
    degraded = validate_schema(coerce_schema(degraded), name=source_name)

    return SyntheticSource(
        data=degraded,
        identity_map=identity_map,
        params=params,
        injected_gaps=gaps.reset_index(drop=True),
        clean=clean,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample(source: pd.DataFrame, target_hz: float, source_name: str) -> pd.DataFrame:
    """
    Interpolate every player onto a uniform grid at ``target_hz``.

    Grid times outside a player's observed span are not emitted, so a
    substitute does not acquire invented positions. Gaps already present in the
    input are interpolated across, which is what a sensor sampling continuously
    would report.
    """
    t_start = float(source["timestamp"].min())
    t_end = float(source["timestamp"].max())
    step = 1.0 / target_hz
    grid = np.arange(t_start, t_end + step / 2.0, step)

    blocks: list[pd.DataFrame] = []
    for (player_id, team), block in source.groupby(["player_id", "team"], sort=True):
        block = block.sort_values("timestamp")
        t = block["timestamp"].to_numpy(dtype=float)
        if len(t) < _MIN_PLAYER_SAMPLES:
            continue
        in_span = (grid >= t[0]) & (grid <= t[-1])
        if not in_span.any():
            continue
        g = grid[in_span]
        blocks.append(
            pd.DataFrame(
                {
                    "timestamp": g,
                    "frame": np.rint((g - t_start) * target_hz).astype("int64"),
                    "player_id": player_id,
                    "team": team,
                    "x": np.interp(g, t, block["x"].to_numpy(dtype=float)),
                    "y": np.interp(g, t, block["y"].to_numpy(dtype=float)),
                    "source": source_name,
                    "sample_rate_hz": float(target_hz),
                }
            )
        )

    if not blocks:
        raise ValueError(
            "no player had enough observations to resample; check the input window"
        )
    return pd.concat(blocks, ignore_index=True)


def _derangement_map(clean: pd.DataFrame, rng: np.random.Generator) -> dict[str, str]:
    """
    Build a synthetic -> true player mapping with no fixed points, per team.

    The synthetic label space (``gps_000``, ``gps_001``, ...) shares nothing
    with the source's, so identity matching cannot succeed by string similarity
    — it has to use the trajectories.
    """
    mapping: dict[str, str] = {}
    counter = 0
    for team, block in clean.groupby("team", sort=True):
        players = sorted(block["player_id"].unique())
        n = len(players)
        if n == 1:
            order = [0]
        else:
            order = list(rng.permutation(n))
            # Reject fixed points so no player keeps its own slot.
            while any(i == j for i, j in enumerate(order)):
                order = list(rng.permutation(n))
        for slot in order:
            mapping[f"gps_{counter:03d}"] = players[slot]
            counter += 1
    return mapping


def _apply_dropout(
    df: pd.DataFrame, params: DegradationParams, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop observations randomly and in contiguous bursts.

    Returns the surviving rows and a frame describing each injected burst
    window, on the pre-clock-error timeline (the caller shifts them).
    """
    keep = np.ones(len(df), dtype=bool)
    if params.dropout_rate > 0:
        keep &= rng.random(len(df)) >= params.dropout_rate

    gap_rows: list[dict] = []
    if params.n_dropout_bursts > 0 and params.dropout_burst_s > 0:
        timestamps = df["timestamp"].to_numpy(dtype=float)
        player_ids = df["player_id"].to_numpy()
        for player_id in np.unique(player_ids):
            in_player = player_ids == player_id
            p_times = timestamps[in_player]
            span_start, span_end = p_times.min(), p_times.max()
            usable = span_end - span_start - params.dropout_burst_s
            if usable <= 0:
                continue
            for _ in range(params.n_dropout_bursts):
                start = span_start + float(rng.random()) * usable
                end = start + params.dropout_burst_s
                keep &= ~(in_player & (timestamps >= start) & (timestamps <= end))
                gap_rows.append({"player_id": player_id, "start": start, "end": end})

    gaps = pd.DataFrame(gap_rows, columns=["player_id", "start", "end"])
    return df.loc[keep].copy(), gaps
