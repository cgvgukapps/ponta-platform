#!/usr/bin/env python3
"""
skillcorner.py — Matrix provider adapter for SkillCorner broadcast tracking.

Thin wrapper around the existing, tested ``ingestion/skillcorner_io.py``. As
with the Metrica adapter, no parsing is re-implemented here: this module calls
``load_skillcorner_meta`` / ``load_skillcorner_tracking`` / ``tracking_to_wide``
and reshapes their output into the long Matrix schema.

SkillCorner specifics the adapter is responsible for hiding:

- It is *broadcast* tracking at 10 Hz, not full optical at 25 Hz. Only players
  in frame are tracked, so roughly half the player-frames are absent. The long
  schema represents that natively by simply not emitting a row.
- Pitch dimensions are real venue metres from ``match.json`` (104 x 68 for the
  sample match), not a nominal 105 x 68 — so the PitchConfig is built from the
  match metadata rather than assumed.
- Playing-direction normalisation is delegated to ``tracking_to_wide`` via the
  ``h1_home_sign`` parsed from ``home_team_side``.
- The vendor clock restarts each period (H2 begins again at 45:00) and so is
  not monotonic across half time. ``load_skillcorner_source`` rebases period 2
  onto a single continuous match clock, because every downstream Matrix stage
  assumes one increasing timeline per source.

This is the second *real* vendor in Track A. It is a different fixture from the
Metrica sample games, so it is used to prove normalisation across genuinely
different vendor formats — never fused with Metrica as though it were the same
match.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from matrix_engine.ingestion.skillcorner_io import (
    SC_FPS,
    load_skillcorner_meta,
    load_skillcorner_tracking,
    pitch_config_from_sc_meta,
    tracking_to_wide,
)
from matrix_engine.schema import coerce_schema, empty_frame, validate_schema
from matrix_engine.pitch_config import PitchConfig

#: Default ``source`` label written into the schema's source column.
SOURCE_NAME: str = "skillcorner"


def skillcorner_pitch(match_dir: str, match_id: int) -> PitchConfig:
    """PitchConfig built from this match's real venue dimensions."""
    return pitch_config_from_sc_meta(_meta(match_dir, match_id))


def load_skillcorner_source(
    match_dir: str,
    match_id: int,
    *,
    teams: tuple[str, ...] = ("home", "away"),
    start_time: float | None = None,
    end_time: float | None = None,
    max_frames: int | None = None,
    source_name: str = SOURCE_NAME,
) -> pd.DataFrame:
    """
    Load a SkillCorner open-data match as a Matrix-schema DataFrame.

    Parameters
    ----------
    match_dir : str
        Directory holding ``{id}_match.json`` and
        ``{id}_tracking_extrapolated.jsonl``.
    match_id : int
        SkillCorner match ID.
    teams : tuple of str
        Which teams to include.
    start_time, end_time : float, optional
        Bounds in seconds on the rebased continuous match clock.
    max_frames : int, optional
        Read at most this many tracking records. The sample match's JSONL is
        ~89 MB / 59k frames and takes ~10 s to parse in full; bounding it keeps
        the test suite fast. ``None`` reads the whole file.
    source_name : str
        Value written to the ``source`` column.

    Returns
    -------
    pd.DataFrame
        Validated Matrix-schema long-format tracking.
    """
    meta = _meta(match_dir, match_id)
    home, away = _wide_frames(match_dir, match_id, max_frames)

    parts: list[pd.DataFrame] = []
    for team in teams:
        wide = {"home": home, "away": away}[team]
        parts.append(_wide_to_long(wide, team=team, source_name=source_name))

    if not parts:
        raise ValueError("teams must name at least one of 'home' / 'away'")

    long_df = pd.concat(parts, ignore_index=True)
    if long_df.empty:
        return validate_schema(empty_frame(), name=source_name)

    long_df = _rebase_period_clock(long_df)

    if start_time is not None:
        long_df = long_df.loc[long_df["timestamp"] >= start_time]
    if end_time is not None:
        long_df = long_df.loc[long_df["timestamp"] < end_time]

    long_df = long_df.sort_values(["timestamp", "player_id"], kind="stable")
    return validate_schema(coerce_schema(long_df), name=source_name)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _meta(match_dir: str, match_id: int) -> dict:
    return load_skillcorner_meta(os.path.join(match_dir, f"{match_id}_match.json"))


def _read_records(tracking_path: str, max_frames: int | None) -> list[dict]:
    """
    Parse the tracking JSONL, optionally stopping after ``max_frames`` records.

    Delegates to ``load_skillcorner_tracking`` for the unbounded case. The
    bounded fast path streams lines directly, and hands a Git LFS pointer stub
    back to the shared loader so its own informative error is the one raised.
    """
    if max_frames is None:
        return load_skillcorner_tracking(tracking_path)

    records: list[dict] = []
    with open(tracking_path, "r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()
        if first_line.startswith("version https://git-lfs"):
            load_skillcorner_tracking(tracking_path)  # raises with LFS guidance
        if first_line:
            records.append(json.loads(first_line))
        for line in fh:
            if len(records) >= max_frames:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@lru_cache(maxsize=4)
def _wide_frames(
    match_dir: str, match_id: int, max_frames: int | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return direction-normalised (home, away) wide frames for a match.

    Cached: parsing the JSONL dominates load time and both the tests and the
    demo ask for the same match more than once. Callers must not mutate.
    """
    meta = _meta(match_dir, match_id)
    records = _read_records(
        os.path.join(match_dir, f"{match_id}_tracking_extrapolated.jsonl"), max_frames
    )
    # Frames before kick-off carry period=None and no player data; dropping
    # them keeps the period-rebasing below well defined.
    records = [r for r in records if r.get("period") is not None]
    return tracking_to_wide(
        records,
        meta["home_players"],
        meta["away_players"],
        h1_home_sign=meta["h1_home_sign"],
    )


def _wide_to_long(wide: pd.DataFrame, *, team: str, source_name: str) -> pd.DataFrame:
    """Reshape one team's wide tracking frame into long Matrix rows."""
    if wide.empty:
        return empty_frame().assign(period=pd.Series(dtype="int64"))

    prefix = team.capitalize()  # "Home" / "Away", matching skillcorner_io naming
    jerseys = [
        c[len(prefix) + 1 : -2]
        for c in wide.columns
        if c.startswith(f"{prefix}_") and c.endswith("_x")
    ]

    timestamps = wide["Time [s]"].to_numpy(dtype=float)
    periods = wide["Period"].to_numpy(dtype=int)
    frames = wide.index.to_numpy()

    blocks: list[pd.DataFrame] = []
    for jersey in jerseys:
        x = wide[f"{prefix}_{jersey}_x"].to_numpy(dtype=float)
        y = wide[f"{prefix}_{jersey}_y"].to_numpy(dtype=float)
        present = np.isfinite(x) & np.isfinite(y)
        if not present.any():
            continue
        blocks.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps[present],
                    "frame": frames[present],
                    "player_id": f"{team}_{jersey}",
                    "team": team,
                    "x": x[present],
                    "y": y[present],
                    "source": source_name,
                    "sample_rate_hz": SC_FPS,
                    "period": periods[present],
                }
            )
        )

    if not blocks:
        return empty_frame().assign(period=pd.Series(dtype="int64"))
    return pd.concat(blocks, ignore_index=True)


def _rebase_period_clock(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn SkillCorner's per-period clock into one continuous match clock.

    The vendor restarts the clock at 45:00 for the second half, which overlaps
    the end of a first half that ran past 45:00. Each period after the first is
    shifted to start one sample after the previous period's last observation,
    preserving within-period timing exactly while making the source's timeline
    monotonic. The original ``period`` is dropped once it has been applied.
    """
    out = long_df.sort_values(["period", "timestamp"], kind="stable")
    offset = 0.0
    prev_end: float | None = None
    shifted: list[pd.DataFrame] = []

    for period in sorted(out["period"].unique()):
        block = out.loc[out["period"] == period].copy()
        if prev_end is not None:
            offset = prev_end + 1.0 / SC_FPS - float(block["timestamp"].min())
            block["timestamp"] = block["timestamp"] + offset
        prev_end = float(block["timestamp"].max())
        shifted.append(block)

    return pd.concat(shifted, ignore_index=True).drop(columns=["period"])
