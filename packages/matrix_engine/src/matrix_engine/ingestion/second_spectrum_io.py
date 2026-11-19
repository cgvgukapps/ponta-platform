#!/usr/bin/env python3
"""
second_spectrum_io.py — Genius Sports / Second Spectrum tracking adapter (via Kloppy).

Loads raw Second Spectrum metadata (XML/JSON) and tracking (JSONL) in the
provider's NATIVE coordinate system (metres, origin at the centre spot, y-axis
bottom-to-top) and emits the same wide-format DataFrames used by metrica_io /
epts_io so the analytical engine (hex_analytics, pitch_control, app.py) works
unchanged.

Why native metres (not Metrica 0-1):
    Second Spectrum already ships real-world metres with the centre spot at
    (0, 0) and +y pointing up — which is exactly PLAIQ's internal convention.
    Loading natively avoids a lossy round-trip through Metrica's 0-1 grid and
    keeps real per-match pitch dimensions (e.g. 104.85 x 67.97).

Playing direction:
    The whole analytics stack assumes the HOME team attacks +x in BOTH halves
    (see xT_surface(attacking_right=True/False) in hex_value.py). We guarantee
    this invariant with Kloppy's Orientation.STATIC_HOME_AWAY transform, which
    is more robust than the Metrica per-half flip (it does not rely on which
    side home happened to kick off from).

Expected raw layout (filenames are configurable):
    {data_dir}/metadata.xml          — match metadata
    {data_dir}/tracking.jsonl        — frame-by-frame tracking
    {data_dir}/metadata.json         — optional additional metadata

Output conventions match epts_io / metrica_io:
    home_df  index=Frame, cols: Period, Time [s], Home_{shirt}_x/y
    away_df  same for Away_{shirt}_* plus ball_x, ball_y
    events_df — empty scaffold when no event feed is supplied
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from kloppy import secondspectrum
from kloppy.domain import Orientation, TrackingDataset
from kloppy.io import FileLike

from matrix_engine.pitch_config import PitchConfig

# Second Spectrum native coordinate system in Kloppy: metres, centre origin,
# y bottom-to-top. Request it explicitly so load() does not normalise to 0-1.
SECONDSPECTRUM_COORDINATES = "secondspectrum"

EVENT_COLUMNS = [
    "Type",
    "Subtype",
    "Period",
    "From",
    "To",
    "Team",
    "Start Frame",
    "Start Time [s]",
    "Start X",
    "Start Y",
    "End Frame",
    "End Time [s]",
    "End X",
    "End Y",
]

#: Optional event column: whether a pass was played high, as the provider logs it (nullable boolean).
#: ``True`` a high pass, ``False`` not high, ``<NA>`` unknown or not a pass. Absent is valid, as for
#: ``native_quality`` in the tracking schema: a provider that logs no height (Metrica) leaves it out, and
#: a consumer treats a missing column as unknown, never as "not high". Each provider's own definition is
#: kept, mapped to this one flag by its adapter:
#:
#: - SkillCorner: ``high_pass`` in the dynamic events. **Logged on completed passes only** (open data, 20
#:   matches: 986 high and 11,821 not high, all successful; no unsuccessful pass carries it), so on
#:   SkillCorner the flag is known only once the outcome is: it describes a completed pass and must never
#:   be an input to a model that predicts the outcome;
#: - StatsBomb: ``pass_height`` "High Pass" is ``True``; "Ground Pass" and "Low Pass" are ``False``
#:   (logged on every pass).
#:
#: Added 19 November 2026. The pass completion probability (``plaiq_engine.analytics.pass_probability``)
#: does not read it: SkillCorner's flag gives away the outcome, and on the league's reference tracking,
#: where height is logged on every pass, it did not improve the model (``PLAIQ_REFERENCES_REPORT.md``).
HIGH_PASS = "High Pass"
OPTIONAL_EVENT_COLUMNS: tuple[str, ...] = (HIGH_PASS,)


def load_secondspectrum_tracking(
    meta_data: FileLike,
    raw_data: FileLike,
    additional_meta_data: FileLike | None = None,
    *,
    sample_rate: float | None = None,
    limit: int | None = None,
    only_alive: bool = False,
    coordinates: str = SECONDSPECTRUM_COORDINATES,
) -> TrackingDataset:
    """
    Load raw Second Spectrum tracking via Kloppy in native metre coordinates.

    Note: Kloppy exposes this as ``kloppy.secondspectrum.load``; this wrapper
    keeps a provider-specific name in our ingestion layer and pins the
    coordinate system to Second Spectrum's native metres.
    """
    return secondspectrum.load(
        meta_data=meta_data,
        raw_data=raw_data,
        additional_meta_data=additional_meta_data,
        sample_rate=sample_rate,
        limit=limit,
        only_alive=only_alive,
        coordinates=coordinates,
    )


def _normalize_orientation(dataset: TrackingDataset) -> TrackingDataset:
    """Force HOME to attack +x in both halves (analytics invariant)."""
    return dataset.transform(to_orientation=Orientation.STATIC_HOME_AWAY)


def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _tracked_players(dataset: TrackingDataset, team) -> list:
    """Return unique on-pitch players for a team, sorted by jersey number."""
    seen: dict[str, object] = {}
    for record in dataset.records:
        for player in record.players_data.keys():
            if player.team == team:
                seen[player.player_id] = player
    return sorted(
        seen.values(),
        key=lambda p: (p.jersey_no is None, p.jersey_no or 0, p.player_id),
    )


def _player_position(player) -> str | None:
    pos = getattr(player, "starting_position", None)
    if pos is None:
        return None
    name = getattr(pos, "name", None) or str(pos)
    return None if name in (None, "Unknown") else name


def _player_dict(player) -> dict:
    return {
        "shirt": str(player.jersey_no) if player.jersey_no is not None else "?",
        "name": player.name or f"Player {player.jersey_no}",
        "position_type": _player_position(player),
        # Second Spectrum persistent identifier — primary key for roster/identity.
        "player_id": player.player_id,
    }


def _build_meta(dataset: TrackingDataset) -> dict:
    """Build metadata dict compatible with epts_io consumers."""
    teams = list(dataset.metadata.teams)
    home_team = next((t for t in teams if t.ground.name == "HOME"), teams[0])
    away_team = next((t for t in teams if t.ground.name == "AWAY"), teams[-1])

    periods: dict[int, tuple[int, int]] = {}
    for period in dataset.metadata.periods:
        period_records = [r for r in dataset.records if r.period.id == period.id]
        if not period_records:
            continue
        frames = [r.frame_id for r in period_records]
        periods[int(period.id)] = (min(frames), max(frames))

    pitch = PitchConfig.from_kloppy_metadata(dataset.metadata)
    return {
        "team_a_id": home_team.team_id,
        "team_b_id": away_team.team_id,
        "team_a_name": home_team.name,
        "team_b_name": away_team.name,
        "team_a_players": [_player_dict(p) for p in _tracked_players(dataset, home_team)],
        "team_b_players": [_player_dict(p) for p in _tracked_players(dataset, away_team)],
        "periods": periods,
        # After STATIC_HOME_AWAY normalisation home always attacks +x.
        "team_a_attack_h1": "left_to_right",
        "frame_rate": pitch.fps,
        "pitch": pitch,
        "provider": "secondspectrum",
    }


def _period_time_seconds(
    frame_id: int,
    period_id: int,
    periods: dict[int, tuple[int, int]],
    fps: float,
) -> float:
    p1_start, p1_end = periods.get(1, (0, 0))
    p2_start, _ = periods.get(2, (p1_end + 1, p1_end + 1))

    if period_id == 1:
        return round((frame_id - p1_start) / fps, 4)
    p1_duration = (p1_end - p1_start + 1) / fps
    return round((frame_id - p2_start) / fps + p1_duration, 4)


def tracking_dataset_to_dataframes(
    dataset: TrackingDataset,
    *,
    normalize_playing_direction: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Convert a native-metre Kloppy TrackingDataset to PLAIQ wide-format DataFrames.

    Coordinates are already metres centred at (0, 0) with +y up. When
    ``normalize_playing_direction`` is True the dataset is re-oriented so the
    home team attacks +x in both halves before extraction.
    """
    if normalize_playing_direction:
        dataset = _normalize_orientation(dataset)

    meta = _build_meta(dataset)
    fps = meta["frame_rate"]
    periods = meta["periods"]

    home_players = meta["team_a_players"]
    away_players = meta["team_b_players"]

    home_jerseys = [p["shirt"] for p in home_players]
    away_jerseys = [p["shirt"] for p in away_players]

    home_cols = {j: f"Home_{j}_x" for j in home_jerseys}
    home_cols_y = {j: f"Home_{j}_y" for j in home_jerseys}
    away_cols = {j: f"Away_{j}_x" for j in away_jerseys}
    away_cols_y = {j: f"Away_{j}_y" for j in away_jerseys}

    home_rows: list[dict] = []
    away_rows: list[dict] = []
    frames: list[int] = []

    for record in dataset.records:
        frame_id = int(record.frame_id)
        period_id = int(record.period.id)
        time_s = _period_time_seconds(frame_id, period_id, periods, fps)
        frames.append(frame_id)

        home_row = {"Period": period_id, "Time [s]": time_s}
        away_row = {"Period": period_id, "Time [s]": time_s}

        for jersey in home_jerseys:
            home_row[home_cols[jersey]] = np.nan
            home_row[home_cols_y[jersey]] = np.nan
        for jersey in away_jerseys:
            away_row[away_cols[jersey]] = np.nan
            away_row[away_cols_y[jersey]] = np.nan

        ball_x, ball_y = np.nan, np.nan
        if record.ball_coordinates is not None:
            ball_x = float(record.ball_coordinates.x)
            ball_y = float(record.ball_coordinates.y)

        for player, pdata in record.players_data.items():
            if pdata.coordinates is None:
                continue
            jersey = str(player.jersey_no) if player.jersey_no is not None else "?"
            px = float(pdata.coordinates.x)
            py = float(pdata.coordinates.y)
            if player.team.ground.name == "HOME":
                if jersey in home_cols:
                    home_row[home_cols[jersey]] = px
                    home_row[home_cols_y[jersey]] = py
            else:
                if jersey in away_cols:
                    away_row[away_cols[jersey]] = px
                    away_row[away_cols_y[jersey]] = py

        away_row["ball_x"] = ball_x
        away_row["ball_y"] = ball_y
        home_rows.append(home_row)
        away_rows.append(away_row)

    home_df = pd.DataFrame(home_rows, index=frames)
    away_df = pd.DataFrame(away_rows, index=frames)
    home_df.index.name = "Frame"
    away_df.index.name = "Frame"

    events_df = _empty_events_df()

    return home_df, away_df, events_df, meta


class SecondSpectrumAdapter:
    """
    Ingest Genius Sports / Second Spectrum tracking feeds via Kloppy.

    Parameters
    ----------
    data_dir : directory containing metadata + tracking files
    meta_file, tracking_file, additional_meta_file :
        override default filenames inside ``data_dir``
    """

    DEFAULT_META_FILE = "metadata.xml"
    DEFAULT_TRACKING_FILE = "tracking.jsonl"
    DEFAULT_ADDITIONAL_META_FILE = "metadata.json"

    def __init__(
        self,
        data_dir: str,
        *,
        meta_file: str = DEFAULT_META_FILE,
        tracking_file: str = DEFAULT_TRACKING_FILE,
        additional_meta_file: str | None = DEFAULT_ADDITIONAL_META_FILE,
        sample_rate: float | None = None,
        limit: int | None = None,
        only_alive: bool = False,
        normalize_playing_direction: bool = True,
    ):
        self.data_dir = data_dir
        self.meta_path = os.path.join(data_dir, meta_file)
        self.tracking_path = os.path.join(data_dir, tracking_file)
        self.additional_meta_path = (
            os.path.join(data_dir, additional_meta_file)
            if additional_meta_file
            else None
        )
        self.sample_rate = sample_rate
        self.limit = limit
        self.only_alive = only_alive
        self.normalize_playing_direction = normalize_playing_direction

    def load_raw(self) -> TrackingDataset:
        additional = self.additional_meta_path
        if additional and not os.path.isfile(additional):
            additional = None
        return load_secondspectrum_tracking(
            meta_data=self.meta_path,
            raw_data=self.tracking_path,
            additional_meta_data=additional,
            sample_rate=self.sample_rate,
            limit=self.limit,
            only_alive=self.only_alive,
        )

    def load(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """
        Load and normalise Second Spectrum data.

        Returns
        -------
        home_df, away_df, events_df, meta
        """
        dataset = self.load_raw()
        return tracking_dataset_to_dataframes(
            dataset,
            normalize_playing_direction=self.normalize_playing_direction,
        )


def load_secondspectrum_game(
    data_dir: str,
    *,
    meta_file: str = SecondSpectrumAdapter.DEFAULT_META_FILE,
    tracking_file: str = SecondSpectrumAdapter.DEFAULT_TRACKING_FILE,
    additional_meta_file: str | None = SecondSpectrumAdapter.DEFAULT_ADDITIONAL_META_FILE,
    sample_rate: float | None = None,
    limit: int | None = None,
    only_alive: bool = False,
    normalize_playing_direction: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Convenience wrapper around :class:`SecondSpectrumAdapter`."""
    adapter = SecondSpectrumAdapter(
        data_dir,
        meta_file=meta_file,
        tracking_file=tracking_file,
        additional_meta_file=additional_meta_file,
        sample_rate=sample_rate,
        limit=limit,
        only_alive=only_alive,
        normalize_playing_direction=normalize_playing_direction,
    )
    return adapter.load()
