#!/usr/bin/env python3
"""
statsbomb_io.py — Hudl StatsBomb open data adapter.

Loads StatsBomb events, lineups, and 360 freeze-frame data via statsbombpy
and maps them onto the same internal contracts used by metrica_io / epts_io /
second_spectrum_io, so the analytics engine (hex_analytics, pitch_control,
hex_value, pass_option_value, app.py) works unchanged.

Coordinate system
-----------------
StatsBomb pitch: [0, 120] × [0, 80] (StatsBomb units, centre = [60, 40]).
Home team attacks toward x = 120 in period 1; coordinates are physically
reversed in period 2 (teams switch ends).

PLAIQ convention: centre at (0, 0), home attacks +x in BOTH halves, pitch
105 × 68 m (standard stadium default — actual dimensions unavailable in the
open data).

Conversion (applied per event, using period to determine flip):
    Period 2 flip first: sb_x ← 120 − sb_x, sb_y ← 80 − sb_y
    Then: x_plaiq = (sb_x − 60) × (length_m / 120)
          y_plaiq = (sb_y − 40) × (width_m  / 80)

StatsBomb event → PLAIQ event schema
-------------------------------------
    Type        : mapped from StatsBomb type name (see _SB_TYPE_MAP)
    Subtype     : pass outcome, set-piece type, clearance, etc.
    Period      : 1 or 2
    From        : player name (carrier / actor)
    To          : pass recipient name
    Team        : "Home" or "Away"
    Start Frame : StatsBomb event index (proxy; there are no tracking frames)
    Start Time [s]: seconds since kick-off
    Start X / Y : event location after coordinate conversion
    End X / Y   : end location (pass/carry destination) after conversion

360 freeze frames
-----------------
StatsBomb open data 360 files provide per-event position snapshots with
teammate/opponent/actor/keeper flags but WITHOUT player identities (paid
subscription required for player IDs in 360 data). Stored as a dict keyed
by event UUID and accessed via load_sb_freeze_frames().

The freeze frame list for each event contains dicts:
    {"teammate": bool, "actor": bool, "keeper": bool,
     "location": [x_plaiq, y_plaiq]}   ← converted to PLAIQ space

Reference:
  https://github.com/statsbomb/open-data
  https://github.com/statsbomb/statsbombpy
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from matrix_engine.ingestion.second_spectrum_io import HIGH_PASS
from matrix_engine.pitch_config import PitchConfig

# ---------------------------------------------------------------------------
# StatsBomb pitch constants
# ---------------------------------------------------------------------------

# StatsBomb normalises every pitch to 120 × 80 "units" (approximately yards).
# We convert to PLAIQ metres using standard stadium dimensions as the scale.
SB_PITCH_LENGTH_UNITS = 120.0   # StatsBomb x-axis span
SB_PITCH_WIDTH_UNITS  = 80.0    # StatsBomb y-axis span

# Default real-world mapping (FIFA standard pitch: 105 × 68 m)
SB_DEFAULT_LENGTH_M = 105.0
SB_DEFAULT_WIDTH_M  = 68.0

SB_PITCH = PitchConfig(
    length_m=SB_DEFAULT_LENGTH_M,
    width_m=SB_DEFAULT_WIDTH_M,
    fps=None,   # StatsBomb is event-based, no fixed frame rate
)

# PLAIQ event column schema (mirrors second_spectrum_io.EVENT_COLUMNS)
SB_EVENT_COLUMNS = [
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
    HIGH_PASS,            # Matrix's optional pass-height flag, from pass_height (nullable boolean)
    # StatsBomb-native extras preserved for enriched analytics
    "sb_event_id",        # UUID — primary key for 360 join
    "sb_event_index",     # sequential index within match
    "sb_possession",      # StatsBomb possession number
    "sb_play_pattern",    # "Regular Play", "From Goal Kick", etc.
    "sb_outcome",         # raw outcome name (e.g. "Complete", "Incomplete")
    "sb_body_part",       # "Right Foot", "Head", etc.
    "sb_under_pressure",  # bool — opponent pressing at moment of action
    "sb_xg",              # StatsBomb xG (shots only)
    "sb_pass_length",     # metres (StatsBomb units)
    "sb_pass_angle",      # radians
    "sb_duration",        # event duration in seconds
]

# ---------------------------------------------------------------------------
# Event type mapping: StatsBomb type → PLAIQ type
# ---------------------------------------------------------------------------
#
# PLAIQ analytics special-cases three Type strings:
#   "PASS"      — pass option value + decision quality (successful pass)
#   "CARRY"     — carry decision quality (ring alternatives)
#   "BALL LOST" — failed-pass decision quality + dispossession
#
# Everything else maps to descriptive labels that mirror Metrica/EPTS vocab.

_SB_TYPE_MAP: dict[str, str] = {
    "Pass":              "PASS",         # split by outcome below
    "Carry":             "CARRY",
    "Shot":              "SHOT",
    "Dribble":           "CARRY",        # dribble = ball-at-feet carry
    "Clearance":         "BALL LOST",
    "Interception":      "BALL LOST",
    "Miscontrol":        "BALL LOST",
    "Ball Recovery":     "RECOVERY",
    "Ball Receipt*":     "RECOVERY",
    "Foul Won":          "FAULT RECEIVED",
    "Foul Committed":    "FAULT RECEIVED",
    "Goal Keeper":       "BALL LOST",
    "Duel":              "CHALLENGE",
    "50/50":             "CHALLENGE",
    "Pressure":          "CHALLENGE",
    "Block":             "BALL LOST",
    "Offside":           "BALL LOST",
    "Error":             "BALL LOST",
    "Substitution":      "SUBSTITUTION",
    "Injury Stoppage":   "INJURY",
    "Referee Ball-Drop": "SET PIECE",
    "Kick Off":          "SET PIECE",
    "Corner Received":   "SET PIECE",
    "Free Kick":         "SET PIECE",
    "Throw-in":          "SET PIECE",
    "Goal Kick":         "SET PIECE",
    "Bad Behaviour":     "CARD",
    "Starting XI":       None,           # administrative — skip
    "Half Start":        None,
    "Half End":          None,
    "Tactical Shift":    None,
    "Player On":         None,
    "Player Off":        None,
}

# Pass outcome names that map to BALL LOST (incomplete / intercepted)
_PASS_LOST_OUTCOMES = {
    "Incomplete", "Out", "Pass Offside", "Unknown",
    "Injury Clearance",
}

# Pass subtypes mapped to PLAIQ Subtype
_PASS_TYPE_TO_SUBTYPE: dict[str, str] = {
    "Free Kick":   "FREE KICK",
    "Corner":      "CORNER",
    "Goal Kick":   "GOAL KICK",
    "Throw-in":    "THROW IN",
    "Kick Off":    "KICK OFF",
}

# Shot outcome names that are goals (enrich Subtype)
_SHOT_GOAL_OUTCOMES = {"Goal"}


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def sb_coords_to_plaiq(
    sb_x: float,
    sb_y: float,
    period: int,
    pitch: PitchConfig = SB_PITCH,
) -> tuple[float, float]:
    """
    Convert a single StatsBomb coordinate pair to PLAIQ internal metres.

    StatsBomb coordinate space: [0, 120] × [0, 80], origin bottom-left.
    Home attacks toward x = 120 in period 1; sides swap in period 2.

    PLAIQ convention: origin at centre spot; home attacks +x in both halves.
    """
    if period == 2:
        sb_x = SB_PITCH_LENGTH_UNITS - sb_x
        sb_y = SB_PITCH_WIDTH_UNITS  - sb_y
    x = (sb_x - SB_PITCH_LENGTH_UNITS / 2) * (pitch.length_m / SB_PITCH_LENGTH_UNITS)
    y = (sb_y - SB_PITCH_WIDTH_UNITS  / 2) * (pitch.width_m  / SB_PITCH_WIDTH_UNITS)
    return float(x), float(y)


def sb_coords_array_to_plaiq(
    xy: np.ndarray,
    periods: np.ndarray,
    pitch: PitchConfig = SB_PITCH,
) -> np.ndarray:
    """
    Vectorised coordinate conversion for a (N, 2) array of [sb_x, sb_y] pairs.

    Parameters
    ----------
    xy      : (N, 2) float array — StatsBomb [x, y]
    periods : (N,) int array — period number per row
    pitch   : PitchConfig to use for scale factors

    Returns
    -------
    (N, 2) PLAIQ [x_m, y_m]
    """
    xy = np.array(xy, dtype=float)
    periods = np.array(periods, dtype=int)
    mask_p2 = periods == 2
    xy[mask_p2, 0] = SB_PITCH_LENGTH_UNITS - xy[mask_p2, 0]
    xy[mask_p2, 1] = SB_PITCH_WIDTH_UNITS  - xy[mask_p2, 1]
    xy[:, 0] = (xy[:, 0] - SB_PITCH_LENGTH_UNITS / 2) * (pitch.length_m / SB_PITCH_LENGTH_UNITS)
    xy[:, 1] = (xy[:, 1] - SB_PITCH_WIDTH_UNITS  / 2) * (pitch.width_m  / SB_PITCH_WIDTH_UNITS)
    return xy


# ---------------------------------------------------------------------------
# Timestamp → match time (seconds)
# ---------------------------------------------------------------------------

def _timestamp_to_seconds(ts: str) -> float:
    """Convert StatsBomb timestamp 'HH:MM:SS.mmm' to seconds since half start."""
    try:
        parts = ts.split(":")
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s
    except Exception:
        return float("nan")


def _compute_match_time(events_df: pd.DataFrame) -> pd.Series:
    """
    Compute cumulative match clock (seconds) respecting period breaks.

    Period 1 starts at 0 s. Period 2 starts after period 1's last event
    plus a standard half-time gap. Extra-time periods handled similarly.
    """
    times = events_df["timestamp"].apply(_timestamp_to_seconds)
    periods = events_df["period"]

    cumulative = times.copy()
    # For each period beyond 1, add the end of the previous period as offset.
    offsets: dict[int, float] = {1: 0.0}
    for p in sorted(events_df["period"].unique()):
        if p == 1:
            continue
        prev_end = times[periods == p - 1].max()
        offsets[p] = offsets.get(p - 1, 0.0) + prev_end
    cumulative = periods.map(offsets) + times
    return cumulative.round(4)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_location(loc) -> tuple[float | None, float | None]:
    """Safely unpack a StatsBomb [x, y] location (may be NaN/None/list)."""
    if loc is None or (isinstance(loc, float) and np.isnan(loc)):
        return None, None
    try:
        return float(loc[0]), float(loc[1])
    except (TypeError, IndexError):
        return None, None


def _safe_str(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return s if s not in ("nan", "None", "") else None


def _resolve_team_side(team_name: str, home_team: str) -> str:
    """Map StatsBomb team name to 'Home' or 'Away'."""
    return "Home" if team_name == home_team else "Away"


# ---------------------------------------------------------------------------
# Event DataFrame builder
# ---------------------------------------------------------------------------

def sb_events_to_plaiq(
    events_df: pd.DataFrame,
    home_team: str,
    pitch: PitchConfig = SB_PITCH,
) -> pd.DataFrame:
    """
    Map a statsbombpy events DataFrame to the PLAIQ events schema.

    Parameters
    ----------
    events_df  : flat DataFrame from ``sb.events(match_id=...)``
    home_team  : home team name (used to assign "Home" / "Away" side)
    pitch      : PitchConfig for coordinate scaling (default: 105 × 68 m)

    Returns
    -------
    DataFrame with SB_EVENT_COLUMNS (rows where _SB_TYPE_MAP returns None
    are dropped — administrative events like Starting XI, Half Start, etc.)
    """
    rows: list[dict] = []

    for _, ev in events_df.iterrows():
        sb_type = _safe_str(ev.get("type"))
        if sb_type is None:
            continue

        plaiq_type = _SB_TYPE_MAP.get(sb_type)
        if plaiq_type is None:
            continue  # administrative event — skip

        period = int(ev.get("period", 1))

        # --- Coordinates ---
        sx, sy = _extract_location(ev.get("location"))
        if sx is not None:
            sx, sy = sb_coords_to_plaiq(sx, sy, period, pitch)

        # End location depends on event type
        end_loc = None
        if sb_type == "Pass":
            end_loc = ev.get("pass_end_location")
        elif sb_type in ("Carry", "Dribble"):
            end_loc = ev.get("carry_end_location")
        elif sb_type == "Shot":
            end_loc = ev.get("shot_end_location")
        elif sb_type == "Goal Keeper":
            end_loc = ev.get("goalkeeper_end_location")

        ex, ey = _extract_location(end_loc)
        if ex is not None:
            ex, ey = sb_coords_to_plaiq(ex, ey, period, pitch)

        # --- Pass-specific: refine Type and derive Subtype ---
        subtype: str | None = None
        outcome = _safe_str(ev.get("pass_outcome"))

        if sb_type == "Pass":
            # Incomplete pass → BALL LOST so failed-pass DQ path can score it
            if outcome in _PASS_LOST_OUTCOMES:
                plaiq_type = "BALL LOST"
                subtype = "INTERCEPTION"
            # Set-piece sub-type from pass_type
            pass_type = _safe_str(ev.get("pass_type"))
            if pass_type in _PASS_TYPE_TO_SUBTYPE:
                subtype = _PASS_TYPE_TO_SUBTYPE[pass_type]
        elif sb_type == "Shot":
            shot_outcome = _safe_str(ev.get("shot_outcome"))
            subtype = "GOAL" if shot_outcome in _SHOT_GOAL_OUTCOMES else shot_outcome
        elif sb_type == "Dribble":
            subtype = "DRIBBLE"
        elif sb_type == "Clearance":
            subtype = "CLEARANCE"
        elif sb_type == "Interception":
            subtype = "INTERCEPTION"
        elif sb_type in ("Foul Won", "Foul Committed"):
            foul_card = _safe_str(ev.get("foul_committed_card"))
            subtype = foul_card if foul_card else None

        # --- Player identity ---
        player_name = _safe_str(ev.get("player"))
        recipient   = _safe_str(ev.get("pass_recipient"))
        team_name   = _safe_str(ev.get("team")) or ""
        team_side   = _resolve_team_side(team_name, home_team)

        # --- Time ---
        ts = _safe_str(ev.get("timestamp")) or "00:00:00.000"
        half_time_s = _timestamp_to_seconds(ts)

        # --- Duration (event spans start to end) ---
        duration = float(ev.get("duration") or 0.0)
        end_time_offset = half_time_s + duration

        # --- Misc StatsBomb fields ---
        sb_xg = float(ev.get("shot_statsbomb_xg") or float("nan"))
        sb_pass_len = float(ev.get("pass_length") or float("nan"))
        sb_pass_ang = float(ev.get("pass_angle") or float("nan"))
        sb_under_pr = bool(ev.get("under_pressure") or False)

        event_idx = int(ev.get("index", 0))

        rows.append({
            "Type":              plaiq_type,
            "Subtype":           subtype,
            "Period":            period,
            "From":              player_name,
            "To":                recipient,
            "Team":              team_side,
            "Start Frame":       event_idx,
            "Start Time [s]":    half_time_s,
            "Start X":           sx,
            "Start Y":           sy,
            "End Frame":         event_idx,     # no tracking frames; same index
            "End Time [s]":      end_time_offset,
            "End X":             ex,
            "End Y":             ey,
            HIGH_PASS:           _high_pass(ev.get("pass_height")) if sb_type == "Pass" else pd.NA,
            "sb_event_id":       _safe_str(ev.get("id")),
            "sb_event_index":    event_idx,
            "sb_possession":     int(ev.get("possession") or 0),
            "sb_play_pattern":   _safe_str(ev.get("play_pattern")),
            "sb_outcome":        outcome,
            "sb_body_part":      _safe_str(ev.get("pass_body_part")) or _safe_str(ev.get("shot_body_part")),
            "sb_under_pressure": sb_under_pr,
            "sb_xg":             sb_xg,
            "sb_pass_length":    sb_pass_len,
            "sb_pass_angle":     sb_pass_ang,
            "sb_duration":       duration,
        })

    df = pd.DataFrame(rows, columns=SB_EVENT_COLUMNS)
    df[HIGH_PASS] = df[HIGH_PASS].astype("boolean")
    df = df.reset_index(drop=True)

    # Compute cumulative match time properly (period 2 continues from period 1)
    _add_cumulative_match_time(df, events_df, home_team)
    return df


def _high_pass(pass_height):
    """StatsBomb's ``pass_height`` as Matrix's flag: "High Pass" is high; "Ground Pass" and "Low Pass" are not."""
    name = _safe_str(pass_height.get("name") if isinstance(pass_height, dict) else pass_height)
    return pd.NA if not name else name == "High Pass"


def _add_cumulative_match_time(
    plaiq_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    _home_team: str,
) -> None:
    """
    Replace half-relative Start/End Time [s] with cumulative match time.

    Period 1 events: unchanged (half time = match time).
    Period 2 events: offset by the duration of period 1.
    """
    p1_events = raw_df[raw_df["period"] == 1]
    if p1_events.empty:
        return

    p1_times = p1_events["timestamp"].apply(_timestamp_to_seconds)
    p1_duration = float(p1_times.max()) if not p1_times.empty else 0.0

    mask_p2 = plaiq_df["Period"] == 2
    plaiq_df.loc[mask_p2, "Start Time [s]"] = (
        plaiq_df.loc[mask_p2, "Start Time [s]"] + p1_duration
    ).round(4)
    plaiq_df.loc[mask_p2, "End Time [s]"] = (
        plaiq_df.loc[mask_p2, "End Time [s]"] + p1_duration
    ).round(4)


# ---------------------------------------------------------------------------
# 360 freeze frame loader
# ---------------------------------------------------------------------------

_SB_OPEN_DATA_360_URL = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{match_id}.json"
)


def _load_360_raw_records(match_id: int) -> list[dict]:
    """
    Load raw 360 JSON records for a match.

    statsbombpy's ``sb.frames()`` currently fails on many matches (pandas
    concat error), so we fetch the open-data file directly from GitHub.
    """
    import urllib.error
    import urllib.request

    url = _SB_OPEN_DATA_360_URL.format(match_id=match_id)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data if isinstance(data, list) else []
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise


def load_sb_freeze_frames(
    match_id: int,
    pitch: PitchConfig = SB_PITCH,
) -> dict[str, dict]:
    """
    Load StatsBomb 360 freeze-frame data for a match.

    Returns a dict keyed by event UUID (``sb_event_id``). Each value is::

        {"players": [{"teammate": bool, "actor": bool, "keeper": bool,
                      "location": [x_plaiq, y_plaiq]}, ...],
         "visible_area": [...]}

    The event's period is needed for coordinate normalisation; this function
    resolves it from the events feed. Returns an empty dict when 360 data is
    unavailable for the match.
    """
    try:
        raw_records = _load_360_raw_records(match_id)
    except Exception:
        raw_records = []

    if not raw_records:
        return {}

    # Load events to get period per event_uuid
    try:
        from statsbombpy import sb as _sb
        ev = _sb.events(match_id=match_id, fmt="dataframe")[["id", "period"]]
        uuid_to_period: dict[str, int] = dict(
            zip(ev["id"].astype(str), ev["period"].astype(int))
        )
    except Exception:
        uuid_to_period = {}

    result: dict[str, dict] = {}
    for record in raw_records:
        event_uuid = str(record.get("event_uuid") or record.get("id") or "")
        if not event_uuid:
            continue
        period = uuid_to_period.get(event_uuid, 1)
        visible_area = record.get("visible_area")
        raw_ff = record.get("freeze_frame")

        if raw_ff is None or (isinstance(raw_ff, float) and np.isnan(raw_ff)):
            continue

        if isinstance(raw_ff, str):
            try:
                raw_ff = json.loads(raw_ff)
            except json.JSONDecodeError:
                continue

        converted: list[dict] = []
        for player in raw_ff:
            loc = player.get("location")
            if loc is None:
                continue
            px, py = sb_coords_to_plaiq(float(loc[0]), float(loc[1]), period, pitch)
            converted.append({
                "teammate": bool(player.get("teammate", False)),
                "actor":    bool(player.get("actor",    False)),
                "keeper":   bool(player.get("keeper",   False)),
                "location": [px, py],
            })

        if converted:
            result[event_uuid] = {
                "players": converted,
                "visible_area": visible_area,
            }

    return result


# ---------------------------------------------------------------------------
# Lineup / roster builder
# ---------------------------------------------------------------------------

def load_sb_lineups(match_id: int) -> dict[str, list[dict]]:
    """
    Load StatsBomb lineups for both teams.

    Returns
    -------
    dict with keys ``"home"`` and ``"away"``, each containing a list of
    player dicts::

        {"player_id": int, "name": str, "nickname": str | None,
         "jersey_number": int, "position": str | None,
         "height_cm": float | None, "weight_kg": float | None}
    """
    try:
        from statsbombpy import sb as _sb
        lineups = _sb.lineups(match_id=match_id)
    except Exception:
        return {"home": [], "away": []}

    team_names = list(lineups.keys())
    if len(team_names) < 2:
        return {"home": [], "away": []}

    # statsbombpy returns lineups keyed by team name, first is home
    home_name, away_name = team_names[0], team_names[1]

    def _parse_lineup(df: pd.DataFrame) -> list[dict]:
        players = []
        for _, row in df.iterrows():
            pos = None
            positions = row.get("positions")
            if isinstance(positions, list) and positions:
                pos = positions[0].get("position") if isinstance(positions[0], dict) else None
            players.append({
                "player_id":    int(row.get("player_id") or 0),
                "name":         str(row.get("player_name") or ""),
                "nickname":     _safe_str(row.get("player_nickname")),
                "jersey_number": int(row.get("jersey_number") or 0),
                "position":     pos,
                "height_cm":    float(row["player_height"]) if pd.notna(row.get("player_height")) else None,
                "weight_kg":    float(row["player_weight"])  if pd.notna(row.get("player_weight"))  else None,
            })
        return players

    return {
        "home": _parse_lineup(lineups[home_name]),
        "away": _parse_lineup(lineups[away_name]),
        "home_team_name": home_name,
        "away_team_name": away_name,
    }


# ---------------------------------------------------------------------------
# Match metadata helper
# ---------------------------------------------------------------------------

def load_sb_match_meta(match_id: int, competition_id: int, season_id: int) -> dict:
    """
    Load match-level metadata (teams, score, stadium, competition info).
    Falls back to minimal dict when the API is unavailable.
    """
    try:
        from statsbombpy import sb as _sb
        matches = _sb.matches(competition_id=competition_id, season_id=season_id)
        row = matches[matches["match_id"] == match_id]
        if row.empty:
            raise ValueError(f"match_id {match_id} not found in competition {competition_id}/{season_id}")
        r = row.iloc[0]
        return {
            "match_id":       match_id,
            "competition_id": competition_id,
            "season_id":      season_id,
            "home_team":      str(r.get("home_team") or "Home"),
            "away_team":      str(r.get("away_team") or "Away"),
            "home_score":     int(r.get("home_score") or 0),
            "away_score":     int(r.get("away_score") or 0),
            "match_date":     str(r.get("match_date") or ""),
            "competition":    str(r.get("competition") or ""),
            "season":         str(r.get("season") or ""),
            "stadium":        str(r.get("stadium") or ""),
            "has_360":        str(r.get("match_status_360") or "") == "available",
            "pitch":          SB_PITCH,
            "provider":       "statsbomb",
        }
    except Exception:
        # Competition/season may be unknown (e.g. stale session state). Probe 360
        # directly from open-data — do not infer availability from metadata alone.
        has_360 = bool(_load_360_raw_records(match_id))
        return {
            "match_id": match_id,
            "competition_id": competition_id,
            "season_id": season_id,
            "home_team": "Home",
            "away_team": "Away",
            "has_360": has_360,
            "pitch": SB_PITCH,
            "provider": "statsbomb",
        }


# ---------------------------------------------------------------------------
# Top-level convenience loader
# ---------------------------------------------------------------------------

def load_sb_match(
    match_id: int,
    competition_id: int = 0,
    season_id: int = 0,
    *,
    load_360: bool = True,
    pitch: PitchConfig = SB_PITCH,
) -> tuple[pd.DataFrame, dict, dict, dict]:
    """
    Load a complete StatsBomb match.

    Parameters
    ----------
    match_id        : StatsBomb match identifier
    competition_id  : required for metadata lookup (can be 0 to skip)
    season_id       : required for metadata lookup (can be 0 to skip)
    load_360        : whether to attempt loading 360 freeze-frame data
    pitch           : PitchConfig for coordinate scaling

    Returns
    -------
    events_df   : PLAIQ-schema events DataFrame (SB_EVENT_COLUMNS)
    freeze_frames : dict keyed by sb_event_id; empty if 360 unavailable
    meta          : match metadata dict (teams, score, pitch, has_360, …)
    lineups       : {"home": [...], "away": [...]} player dicts
    """
    from statsbombpy import sb as _sb

    raw_events = _sb.events(match_id=match_id, fmt="dataframe")

    meta = load_sb_match_meta(match_id, competition_id, season_id)
    home_team = meta["home_team"]

    events_df = sb_events_to_plaiq(raw_events, home_team, pitch)

    freeze_frames: dict = {}
    if load_360:
        freeze_frames = load_sb_freeze_frames(match_id, pitch)

    lineups = load_sb_lineups(match_id)

    # Authoritative 360 flag: open-data file loaded successfully.
    meta["has_360"] = bool(freeze_frames)
    meta["freeze_frame_events"] = len(freeze_frames)

    # Enrich generic fallback team names from lineups when metadata lookup failed.
    if meta.get("home_team") == "Home" and lineups.get("home_team_name"):
        meta["home_team"] = lineups["home_team_name"]
    if meta.get("away_team") == "Away" and lineups.get("away_team_name"):
        meta["away_team"] = lineups["away_team_name"]

    return events_df, freeze_frames, meta, lineups


# ---------------------------------------------------------------------------
# Competition / season / match catalogue helpers (for app.py match browser)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def list_sb_competitions() -> pd.DataFrame:
    """
    Return all available competitions as a DataFrame.
    Cached after first call — competitions list doesn't change at runtime.
    """
    from statsbombpy import sb as _sb
    return _sb.competitions()


def list_sb_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return matches for a competition/season via statsbombpy."""
    from statsbombpy import sb as _sb
    return _sb.matches(competition_id=competition_id, season_id=season_id)
