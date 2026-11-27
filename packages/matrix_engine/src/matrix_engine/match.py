"""
match.py — one match, normalised by Matrix, in the form Plaiq consumes.

The Matrix schema (``schema.py``) describes player tracking only: one row per observed
player per frame. Analysing a match also needs the ball, the event feed and the
vendor's frame clock, which the reconciliation pipeline does not carry. A
``NormalisedMatch`` bundles all of them, already converted to Matrix's convention
(metres, centred at (0, 0), home attacking +x for the whole match).

The loaders below call the same ingestion functions the prototype's offline
``preprocess.py`` called, in the same order, so a match loaded here carries the
same positions, ball and events the prototype analysed. Velocities are not
computed here: they are derived by the consumer from positions, exactly as the
prototype did after preprocessing.

``to_wide_frames`` turns a match back into the vendor-style wide frames
(``Home_7_x``, ``ball_x`` …, indexed by ``Frame``) that the Plaiq engine reads.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from matrix_engine.ingestion.epts_io import load_game3
from matrix_engine.ingestion.metrica_io import (
    orient_home_attacking_positive,
    read_match_data,
    to_metric_coordinates,
    to_single_playing_direction,
)
from matrix_engine.ingestion.skillcorner_io import SC_FPS, SkillCornerAdapter
from matrix_engine.pitch_config import DEFAULT_FPS, FIELD_DIMENSIONS, PitchConfig
from matrix_engine.schema import coerce_schema, empty_frame, validate_schema

#: Columns of the frame clock table.
FRAME_COLUMNS: tuple[str, ...] = ("frame", "period", "time_s")
#: Columns of the ball table (rows only where the ball was detected).
BALL_COLUMNS: tuple[str, ...] = ("frame", "x", "y")

TEAMS: tuple[str, ...] = ("home", "away")


@dataclass(frozen=True)
class NormalisedMatch:
    """
    One match after Matrix normalisation.

    Attributes
    ----------
    provider : str
        Vendor the data came from (``metrica``, ``skillcorner``).
    match_id : str
        Vendor match identifier.
    fps : float
        Native sampling rate of the tracking.
    pitch : PitchConfig
        Pitch geometry for this match.
    frames : pd.DataFrame
        The vendor's frame clock, one row per frame, columns ``frame``, ``period``,
        ``time_s``. Includes frames where nothing was observed, so the frame grid
        is the vendor's, not just the frames that carry data. ``time_s`` is the
        vendor's own clock.
    tracking : pd.DataFrame
        Player positions in the Matrix long schema (validated). Rows exist only
        where a player was observed.
    ball : pd.DataFrame
        Ball positions, columns ``frame``, ``x``, ``y``; rows only where detected.
    events : pd.DataFrame
        The event feed in the same coordinates (columns as the vendor ingest
        produces them: ``Type``, ``From``, ``Start Frame``, ``Start X`` …).
    slots : dict
        Per team, the vendor's jersey-slot order (strings). Kept so a round trip
        to wide frames reproduces the vendor's column order exactly.
    teams : dict or None
        The provider's own team id per side (``{"home": …, "away": …}``), never a
        name; None when the provider does not give one (Metrica's sample games).
    competition, season : str or None
        The provider's competition id (``"<provider>:<id>"``) and season label
        (``"2024/2025"``), when the provider gives them. Stage E (team identity).
    """

    provider: str
    match_id: str
    fps: float
    pitch: PitchConfig
    frames: pd.DataFrame
    tracking: pd.DataFrame
    ball: pd.DataFrame
    events: pd.DataFrame
    slots: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Class-level defaults, so matches pickled before Stage E still load with these unset.
    teams: dict[str, str] | None = None
    competition: str | None = None
    season: str | None = None
    #: Set by the adapter (review DE-04, 27 November 2026): the events were inferred from tracking, not logged. None on
    #: matches built before the flag existed; the Data check then falls back to the provider (SkillCorner's are).
    events_derived_from_tracking: bool | None = None
    #: What the adapter read, dropped and assumed (``matrix_engine.ingest.IngestReport``; review DE-12). None where the
    #: adapter does not report yet.
    ingest: object | None = None

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def ball_detected_share(self) -> float:
        """Share of frames in which the ball was detected."""
        return 0.0 if self.frames.empty else len(self.ball) / len(self.frames)

    def observed_share(self) -> float | None:
        """
        Share of player positions the source observed rather than estimated, or None when the
        source does not say (no ``observed`` column: every row is an observation).
        """
        if "observed" not in self.tracking.columns or self.tracking.empty:
            return None
        return float(self.tracking["observed"].mean())

    def observed_near_ball_share(self, radius_m: float) -> float | None:
        """
        The same share over positions within ``radius_m`` of the ball, at frames with a ball
        position; None when the source does not say or no position is near a located ball.
        """
        if "observed" not in self.tracking.columns or self.tracking.empty or self.ball.empty:
            return None
        ball = self.ball.drop_duplicates("frame").set_index("frame")
        tr = self.tracking.loc[self.tracking["frame"].isin(ball.index), ["frame", "x", "y", "observed"]]
        if tr.empty:
            return None
        bx = ball["x"].reindex(tr["frame"]).to_numpy(dtype=float)
        by = ball["y"].reindex(tr["frame"]).to_numpy(dtype=float)
        near = np.hypot(tr["x"].to_numpy(dtype=float) - bx, tr["y"].to_numpy(dtype=float) - by) <= radius_m
        return float(tr["observed"].to_numpy()[near].mean()) if near.any() else None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_metrica_csv_match(data_dir: str, game_id: int) -> NormalisedMatch:
    """
    Load a Metrica CSV sample game (games 1 and 2).

    Same steps as the prototype's ``preprocess._preprocess_metrica_csv``: read,
    convert tracking and event coordinates to centred metres, then normalise the
    playing direction. One step the prototype lacked follows: if the home team
    attacks -x (game 2), the whole match is rotated so it attacks +x as documented.
    """
    home, away, events = read_match_data(data_dir, game_id)
    home = to_metric_coordinates(home)
    away = to_metric_coordinates(away)
    events = events.copy()
    for col in ("Start X", "Start Y", "End X", "End Y"):
        if col in events.columns:
            axis_y = col.endswith("Y")
            size = FIELD_DIMENSIONS[1] if axis_y else FIELD_DIMENSIONS[0]
            sign = -1.0 if axis_y else 1.0
            events[col] = sign * (events[col] - 0.5) * size
    home, away, events = to_single_playing_direction(home, away, events)
    # Not in the prototype: game 2's home team starts attacking -x (see metrica_io).
    home, away, events = orient_home_attacking_positive(home, away, events)
    return from_wide_frames(
        home, away, events,
        provider="metrica", match_id=str(game_id), fps=DEFAULT_FPS, pitch=PitchConfig(fps=DEFAULT_FPS),
    )


def load_metrica_epts_match(data_dir: str, game_id: int = 3) -> NormalisedMatch:
    """
    Load a Metrica EPTS sample game (game 3).

    ``data_dir`` is the directory holding ``Sample_Game_3/``. Same steps as the
    prototype's ``preprocess._preprocess_epts``: ``epts_io.load_game3`` already
    returns metres, single playing direction, and events in the same frame.
    """
    home, away, events, _meta = load_game3(os.path.join(data_dir, f"Sample_Game_{game_id}"))
    return from_wide_frames(
        home, away, events,
        provider="metrica", match_id=str(game_id), fps=DEFAULT_FPS, pitch=PitchConfig(fps=DEFAULT_FPS),
    )


def load_skillcorner_match(match_dir: str, match_id: int, *, prototype_inputs: bool = False) -> NormalisedMatch:
    """
    Load a SkillCorner open-data match.

    Same step as the prototype's ``preprocess._preprocess_skillcorner``:
    ``SkillCornerAdapter.load``, which normalises the playing direction and maps
    the dynamic events onto tracking frames. ``time_s`` is elapsed match time (review DE-09, 27 November 2026: the
    vendor's clock restarts the second half at 45:00, which put every second-half event 78.1 s after its frame on
    match 1886347); ``match.ingest`` says what the load dropped and assumed (review DE-12).

    Two corrections depart from the prototype's inputs (review backlog A, 25 November 2026): records with no period
    (pre-match, half-time, post-match) are dropped, and SkillCorner's recorded stoppages become SET PIECE rows.
    ``prototype_inputs=True`` reproduces the prototype's inputs, for the engine-equivalence test only.
    """
    home, away, events, meta = SkillCornerAdapter(match_dir, match_id, prototype_inputs=prototype_inputs).load()
    pitch = meta.get("pitch") or PitchConfig(fps=SC_FPS)
    m = from_wide_frames(
        home, away, events, provider="skillcorner", match_id=str(match_id), fps=SC_FPS, pitch=pitch,
    )
    m = dataclasses.replace(m, ingest=meta.get("ingest"))
    m = with_observed(m, meta.get("estimated_positions"))
    return with_skillcorner_metadata(m, match_dir, match_id)


def with_observed(m: NormalisedMatch, estimated: pd.DataFrame | None) -> NormalisedMatch:
    """
    Mark each tracking row ``observed`` (schema optional column): False where the provider
    estimated the position (``estimated``: ``frame``, ``player_id``). Positions are unchanged.
    """
    import dataclasses

    if estimated is None:
        return m
    key = pd.MultiIndex.from_frame(estimated[["frame", "player_id"]])
    rows = pd.MultiIndex.from_arrays([m.tracking["frame"].to_numpy(), m.tracking["player_id"].to_numpy()])
    tracking = m.tracking.assign(observed=~rows.isin(key))
    return dataclasses.replace(m, tracking=validate_schema(coerce_schema(tracking), name=m.provider))


def with_skillcorner_metadata(m: NormalisedMatch, match_dir: str, match_id: int) -> NormalisedMatch:
    """Team ids, competition and season from SkillCorner's match file (ids only, never names)."""
    import dataclasses
    import json

    path = os.path.join(match_dir, f"{match_id}_match.json")
    if not os.path.exists(path):
        return m
    with open(path) as fh:
        meta = json.load(fh)
    edition = meta.get("competition_edition") or {}
    comp, season = edition.get("competition") or {}, edition.get("season") or {}
    return dataclasses.replace(
        m,
        teams={"home": str(meta["home_team"]["id"]), "away": str(meta["away_team"]["id"])},
        competition=f"skillcorner:{comp['id']}" if comp.get("id") is not None else None,
        season=season.get("name"),
        events_derived_from_tracking=True,  # SkillCorner's dynamic events are inferred from its tracking
    )


# ---------------------------------------------------------------------------
# Wide <-> normalised
# ---------------------------------------------------------------------------

def _slot_order(wide: pd.DataFrame, prefix: str) -> tuple[str, ...]:
    return tuple(c[len(prefix) + 1 : -2] for c in wide.columns if c.startswith(f"{prefix}_") and c.endswith("_x"))


def from_wide_frames(
    home: pd.DataFrame,
    away: pd.DataFrame,
    events: pd.DataFrame,
    *,
    provider: str,
    match_id: str,
    fps: float,
    pitch: PitchConfig,
) -> NormalisedMatch:
    """
    Build a ``NormalisedMatch`` from vendor-style wide frames (already in Matrix
    coordinates). Velocity columns, if present, are ignored.
    """
    clock = home if "Period" in home.columns else away
    frames = pd.DataFrame(
        {
            "frame": clock.index.to_numpy(),
            "period": clock["Period"].to_numpy(),
            "time_s": clock["Time [s]"].to_numpy(dtype=float),
        }
    )

    slots: dict[str, tuple[str, ...]] = {}
    blocks: list[pd.DataFrame] = []
    for team, wide in (("home", home), ("away", away)):
        prefix = team.capitalize()
        slots[team] = _slot_order(wide, prefix)
        frame_idx = wide.index.to_numpy()
        timestamps = wide["Time [s]"].to_numpy(dtype=float)
        for jersey in slots[team]:
            x = wide[f"{prefix}_{jersey}_x"].to_numpy(dtype=float)
            y = wide[f"{prefix}_{jersey}_y"].to_numpy(dtype=float)
            present = np.isfinite(x) & np.isfinite(y)
            if not present.any():
                continue
            blocks.append(
                pd.DataFrame(
                    {
                        "timestamp": timestamps[present],
                        "frame": frame_idx[present],
                        "player_id": f"{team}_{jersey}",
                        "team": team,
                        "x": x[present],
                        "y": y[present],
                        "source": provider,
                        "sample_rate_hz": float(fps),
                    }
                )
            )
    tracking = pd.concat(blocks, ignore_index=True) if blocks else empty_frame()
    tracking = validate_schema(coerce_schema(tracking), name=provider)

    ball_src = away if "ball_x" in away.columns else home
    if "ball_x" in ball_src.columns:
        bx = ball_src["ball_x"].to_numpy(dtype=float)
        by = ball_src["ball_y"].to_numpy(dtype=float)
        seen = np.isfinite(bx) & np.isfinite(by)
        ball = pd.DataFrame({"frame": ball_src.index.to_numpy()[seen], "x": bx[seen], "y": by[seen]})
    else:
        ball = pd.DataFrame({c: pd.Series(dtype="float64") for c in BALL_COLUMNS})

    return NormalisedMatch(
        provider=provider,
        match_id=str(match_id),
        fps=float(fps),
        pitch=pitch,
        frames=frames,
        tracking=tracking,
        ball=ball,
        events=events.reset_index(drop=True).copy(),
        slots=slots,
    )


def to_wide_frames(match: NormalisedMatch) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rebuild vendor-style wide frames (home, away) from a ``NormalisedMatch``.

    Both frames are indexed by ``Frame`` over the full vendor frame grid and carry
    ``Period``, ``Time [s]``, ``{Team}_{jersey}_x/_y`` in the vendor's slot order,
    then ``ball_x``/``ball_y``. A player or ball not observed at a frame is NaN.
    No velocity columns: the consumer derives them.
    """
    index = pd.Index(match.frames["frame"].to_numpy(), name="Frame")
    out: dict[str, pd.DataFrame] = {}
    for team in TEAMS:
        prefix = team.capitalize()
        cols: dict[str, np.ndarray] = {
            "Period": match.frames["period"].to_numpy(),
            "Time [s]": match.frames["time_s"].to_numpy(dtype=float),
        }
        team_rows = match.tracking.loc[match.tracking["team"] == team]
        by_player = {pid: g for pid, g in team_rows.groupby("player_id", sort=False)}
        for jersey in match.slots.get(team, ()):
            xs = np.full(len(index), np.nan)
            ys = np.full(len(index), np.nan)
            g = by_player.get(f"{team}_{jersey}")
            if g is not None:
                pos = index.get_indexer(g["frame"].to_numpy())
                ok = pos >= 0
                xs[pos[ok]] = g["x"].to_numpy(dtype=float)[ok]
                ys[pos[ok]] = g["y"].to_numpy(dtype=float)[ok]
            cols[f"{prefix}_{jersey}_x"] = xs
            cols[f"{prefix}_{jersey}_y"] = ys
        bx = np.full(len(index), np.nan)
        by = np.full(len(index), np.nan)
        if len(match.ball):
            pos = index.get_indexer(match.ball["frame"].to_numpy())
            ok = pos >= 0
            bx[pos[ok]] = match.ball["x"].to_numpy(dtype=float)[ok]
            by[pos[ok]] = match.ball["y"].to_numpy(dtype=float)[ok]
        cols["ball_x"] = bx
        cols["ball_y"] = by
        out[team] = pd.DataFrame(cols, index=index)
    return out["home"], out["away"]
