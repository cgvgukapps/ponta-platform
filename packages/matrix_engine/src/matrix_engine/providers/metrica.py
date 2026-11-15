#!/usr/bin/env python3
"""
metrica.py — Matrix provider adapter for Metrica Sports sample tracking.

Thin wrapper around the existing, tested ``ingestion/metrica_io.py`` and
``ingestion/metrica_velocities.py``. This module does not re-implement any
parsing: it calls those loaders, then reshapes their wide, one-column-per-player
output into the long Matrix schema defined in ``matrix_engine/schema.py``.

Metrica specifics the adapter is responsible for hiding from the rest of Matrix:

- Raw coordinates are normalised 0–1 with a top-left origin, so
  ``to_metric_coordinates`` converts to centred metres and flips y.
- Teams swap ends at half time, so ``to_single_playing_direction`` is applied
  before reshaping — downstream Matrix code can then assume home attacks +x
  for the whole match.
- Columns are named ``Home_11_x`` / ``Away_3_y``; player IDs are namespaced to
  ``home_11`` / ``away_3`` so a merged multi-source frame stays unambiguous.

Sample games 1 and 2 are the CSV format read here. Game 3 is an EPTS/XML export
handled by ``ingestion/epts_io.py`` and is not wired into this adapter — the
prototype only needs one Metrica fixture as ground truth.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from matrix_engine.ingestion.metrica_io import (
    read_match_data,
    to_metric_coordinates,
    to_single_playing_direction,
)
from matrix_engine.ingestion.metrica_velocities import calc_player_velocities
from matrix_engine.schema import coerce_schema, validate_schema
from matrix_engine.pitch_config import PitchConfig

#: Metrica sample data is recorded at 25 Hz.
METRICA_FPS: float = 25.0

#: Default ``source`` label written into the schema's source column.
SOURCE_NAME: str = "metrica"


def metrica_pitch() -> PitchConfig:
    """PitchConfig for the Metrica sample games (the repo-wide 106 x 68 default)."""
    return PitchConfig(fps=METRICA_FPS)


@lru_cache(maxsize=4)
def _read_direction_normalised(data_dir: str, game_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read a sample game and return (home, away) in centred metres, single
    playing direction.

    Cached because the raw CSVs are ~145k rows each and both the test suite and
    the demo script load the same game repeatedly. Callers must treat the
    returned frames as read-only and copy before mutating.
    """
    home, away, events = read_match_data(data_dir, game_id)
    home = to_metric_coordinates(home)
    away = to_metric_coordinates(away)
    home, away, _ = to_single_playing_direction(home, away, events)
    return home, away


def load_metrica_source(
    data_dir: str,
    game_id: int,
    *,
    teams: tuple[str, ...] = ("home", "away"),
    start_time: float | None = None,
    end_time: float | None = None,
    with_velocities: bool = False,
    source_name: str = SOURCE_NAME,
) -> pd.DataFrame:
    """
    Load one Metrica sample game as a Matrix-schema DataFrame.

    Parameters
    ----------
    data_dir : str
        Directory containing ``Sample_Game_{id}/`` (e.g. ``sample-data/data``).
    game_id : int
        Metrica sample game number (1 or 2 for the CSV format).
    teams : tuple of str
        Which teams to include — ``("home",)`` is a useful, much smaller slice
        for tests.
    start_time, end_time : float, optional
        Match-relative second bounds, applied on ``Time [s]``. A window keeps
        the long-format output tractable: the full match is ~2 million rows.
    with_velocities : bool
        Also emit provider-native ``vx``/``vy``/``speed`` columns via
        ``metrica_velocities.calc_player_velocities``. These are extra columns
        beyond the required schema. They are computed on the requested window,
        so a short window carries the usual filter edge effects.
    source_name : str
        Value written to the ``source`` column.

    Returns
    -------
    pd.DataFrame
        Validated Matrix-schema long-format tracking.
    """
    home, away = _read_direction_normalised(data_dir, game_id)

    parts: list[pd.DataFrame] = []
    for team in teams:
        wide = {"home": home, "away": away}[team]
        wide = _window(wide, start_time, end_time)
        if with_velocities:
            # calc_player_velocities mutates and returns the frame it is given.
            wide = calc_player_velocities(wide.copy(), fps=METRICA_FPS)
        parts.append(_wide_to_long(wide, team=team, source_name=source_name))

    if not parts:
        raise ValueError("teams must name at least one of 'home' / 'away'")

    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.sort_values(["timestamp", "player_id"], kind="stable")
    return validate_schema(coerce_schema(long_df), name=source_name)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _window(
    wide: pd.DataFrame, start_time: float | None, end_time: float | None
) -> pd.DataFrame:
    """Slice a wide tracking frame to a match-relative time window."""
    if start_time is None and end_time is None:
        return wide
    t = wide["Time [s]"]
    mask = pd.Series(True, index=wide.index)
    if start_time is not None:
        mask &= t >= start_time
    if end_time is not None:
        mask &= t < end_time
    return wide.loc[mask]


def _wide_to_long(wide: pd.DataFrame, *, team: str, source_name: str) -> pd.DataFrame:
    """
    Reshape one team's wide tracking frame into long Matrix rows.

    Ball columns are skipped: the Matrix schema tracks players, and the ball is
    not a reconciliation target for this prototype.
    """
    prefix = team.capitalize()  # "Home" / "Away", matching metrica_io naming
    jerseys = [
        c[len(prefix) + 1 : -2]
        for c in wide.columns
        if c.startswith(f"{prefix}_") and c.endswith("_x")
    ]

    timestamps = wide["Time [s]"].to_numpy(dtype=float)
    frames = wide.index.to_numpy()

    blocks: list[pd.DataFrame] = []
    for jersey in jerseys:
        x = wide[f"{prefix}_{jersey}_x"].to_numpy(dtype=float)
        y = wide[f"{prefix}_{jersey}_y"].to_numpy(dtype=float)
        present = np.isfinite(x) & np.isfinite(y)
        if not present.any():
            continue

        block = pd.DataFrame(
            {
                "timestamp": timestamps[present],
                "frame": frames[present],
                "player_id": f"{team}_{jersey}",
                "team": team,
                "x": x[present],
                "y": y[present],
                "source": source_name,
                "sample_rate_hz": METRICA_FPS,
            }
        )
        for extra, suffix in (("vx", "_vx"), ("vy", "_vy"), ("speed", "_speed")):
            col = f"{prefix}_{jersey}{suffix}"
            if col in wide.columns:
                block[extra] = wide[col].to_numpy(dtype=float)[present]
        blocks.append(block)

    if not blocks:
        from matrix_engine.schema import empty_frame

        return empty_frame()
    return pd.concat(blocks, ignore_index=True)
