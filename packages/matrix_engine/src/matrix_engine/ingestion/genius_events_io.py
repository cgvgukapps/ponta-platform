#!/usr/bin/env python3
"""
genius_events_io.py — Genius Sports football event-spec v1 -> PLAIQ events CSV.

Maps Genius Sports' documented football event feed
(``com.geniussports.dataplatform.football.eventspec.v1``) onto the same
wide-format events DataFrame the analytics engine already consumes from
Metrica / EPTS (Type, Subtype, Period, From, To, Team, Start/End Frame,
Start/End Time [s], Start/End X/Y).

Why this is buildable without a real sample
-------------------------------------------
The event-spec is public and aligns with our internal model on the two things
that previously blocked us:

  * Coordinates: ``FootballCoordinate`` is {x, y, z} in metres with the origin
    at the centre of the pitch — identical to our internal convention and to
    Second Spectrum native tracking. No remapping needed.
  * Synchronisation: every event carries ``metadata.startFrameIdx`` /
    ``endFrameIdx``, so events align to tracking frames directly. We convert
    those frame indices to ``Time [s]`` using the SAME period-bounds + fps
    logic as the tracking loader so the two timelines match exactly.

Reference:
  https://sportsdata.api.geniussports.com/docs/ingestion/apis/football/event-spec-v1/avro-schema
  https://sportsdata.api.geniussports.com/docs/match-events/football/json-schema

Open items to confirm against a real payload (documented, not guessed):
  * Attacking-direction convention. The schema says "centre origin, metres" but
    does not state whether x is a fixed stadium frame (home->away constant) or
    normalised per attacking direction. Our analytics require HOME attacking +x
    in BOTH halves (we enforce STATIC_HOME_AWAY on tracking). If a real feed
    turns out to mirror the second half physically, set
    ``second_half_sign_flip=True`` to match tracking. Default is pass-through.
  * Identity bridge. ``participant.id`` / ``team.id`` come from the Fixtures API
    and may live in a different id-space than the tracking player_ids. Pass
    ``participant_id_map`` / ``teamid_to_side`` to bridge when needed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import numpy as np
import pandas as pd

from matrix_engine.ingestion.second_spectrum_io import (
    EVENT_COLUMNS,
    _period_time_seconds,
)

# ---------------------------------------------------------------------------
# Event-type mapping (Genius EventType -> PLAIQ Type)
# ---------------------------------------------------------------------------
#
# The analytics engine special-cases three Type strings (see pass_option_value
# and app.py): "PASS" (POV), "CARRY" (carry DQ) and "BALL LOST" (failed-pass /
# dispossession DQ, with Subtype exclusions). PASS is split on `complete`:
# a completed pass -> "PASS"; an incomplete pass -> "BALL LOST" (so the
# failed-pass DQ path can score it). Everything else maps to descriptive Types
# that mirror Metrica's vocabulary; administrative events are skipped.

_TYPE_MAP = {
    "CARRY": "CARRY",
    "SHOT": "SHOT",
    "GOAL": "SHOT",
    "RECEPTION": "RECOVERY",
    "CLEARANCE": "BALL LOST",
    "OFFSIDE": "BALL LOST",
    "DEFLECTION": "BALL LOST",
    "GOALKEEPER_ACTION": "BALL LOST",
    "OUT": "BALL OUT",
    "FOUL": "FAULT RECEIVED",
    "DUEL": "CHALLENGE",
    "AERIAL_DUEL": "CHALLENGE",
    "CORNER": "SET PIECE",
    "THROW_IN": "SET PIECE",
    "KICK_OFF": "SET PIECE",
    "GOAL_KICK": "SET PIECE",
    "CARD": "CARD",
}

# Non-positional / administrative events we do not emit as on-ball events.
_SKIP_TYPES = {
    "UNKNOWN", "COMMENT", "CLOCK", "PERIOD_CHANGED", "MATCH_STATUS_CHANGED",
    "DANGER_STATE_CHANGED", "STATISTICS", "RELIABILITY", "STOPPAGE_TIME",
    "VAR", "SHIRT_COLOR_MAJOR", "SHIRT_COLOR_MINOR", "LINEUP", "RISK",
    "PENALTY", "SUBSTITUTION",
}

# Restart-type subtype labels for set pieces.
_SETPIECE_SUBTYPE = {
    "CORNER": "CORNER KICK",
    "THROW_IN": "THROW IN",
    "KICK_OFF": "KICK OFF",
    "GOAL_KICK": "GOAL KICK",
}

# Participant types that identify the actor ("From"), in priority order.
_ACTOR_TYPES = (
    "PASSER", "SHOOTER", "SCORER", "SCORER_NO_ASSIST", "KICKER", "TAKER",
    "POSSESSOR", "CLEARER", "FOULER", "WINNER", "LOSER", "ATTACKER",
    "GOALKEEPER", "LAST_TOUCH", "PARTICIPANT",
)
# Participant types that identify the target ("To"), in priority order.
_TARGET_TYPES = (
    "RECEIVER", "ASSISTER", "FOULED", "DEFENDER", "CONTESTOR", "SHIELDED",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_genius_events_file(path: str) -> list[dict]:
    """Read a Genius event feed. Accepts JSONL (one event/line) or a JSON
    array (optionally wrapped in {"events": [...]})."""
    with open(path, "r", encoding="utf-8-sig") as fh:
        text = fh.read().strip()
    if not text:
        return []

    # Try a whole-document parse first (JSON array, single object, or
    # {"events": [...]}); fall back to JSONL (one object per line).
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        out = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "events" in obj and isinstance(obj["events"], list):
            return list(obj["events"])
        return [obj]
    return []


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def _unwrap_attributes(attributes) -> dict:
    """Avro-union JSON may wrap attrs as {"PassAttributes": {...}}. Unwrap to
    the inner dict; otherwise return as-is."""
    if not isinstance(attributes, dict):
        return {}
    if len(attributes) == 1:
        (key, value), = attributes.items()
        if key.endswith("Attributes") and isinstance(value, dict):
            return value
    return attributes


def _xy(coord) -> tuple[float, float]:
    if not isinstance(coord, dict):
        return np.nan, np.nan
    x = coord.get("x")
    y = coord.get("y")
    return (float(x) if x is not None else np.nan,
            float(y) if y is not None else np.nan)


def _start_end_coords(genius_type: str, attrs: dict) -> tuple[float, float, float, float]:
    """Return (start_x, start_y, end_x, end_y) for an event's attributes."""
    if genius_type == "CARRY":
        sx, sy = _xy(attrs.get("startLocation"))
        ex, ey = _xy(attrs.get("endLocation"))
    elif genius_type in ("SHOT", "GOAL"):
        sx, sy = _xy(attrs.get("location"))
        ex, ey = _xy(attrs.get("goalmouthLocation"))
    elif genius_type == "PASS":
        sx, sy = _xy(attrs.get("location"))
        ex, ey = _xy(attrs.get("endLocation"))
    else:
        sx, sy = _xy(attrs.get("location"))
        ex, ey = _xy(attrs.get("endLocation"))

    # Point events: collapse end onto start when no explicit end was given.
    if np.isnan(ex) and np.isnan(ey):
        ex, ey = sx, sy
    return sx, sy, ex, ey


def _resolve_participant(participants: Iterable[dict], wanted: tuple[str, ...]) -> dict | None:
    if not participants:
        return None
    by_type = {}
    for p in participants:
        ptype = str(p.get("type", "")).upper()
        by_type.setdefault(ptype, p)
    for t in wanted:
        if t in by_type:
            return by_type[t]
    return None


def _map_participant_id(pid: str | None, participant_id_map: dict | None) -> str | None:
    if pid is None:
        return None
    if participant_id_map and pid in participant_id_map:
        return participant_id_map[pid]
    return pid


def _resolve_team(event: dict, actor: dict | None,
                  teamid_to_side: dict | None) -> str | None:
    # Prefer an explicit HOME/AWAY team entry on the event.
    for team in event.get("teams", []) or []:
        ttype = str(team.get("type", "")).upper()
        if ttype == "HOME":
            return "Home"
        if ttype == "AWAY":
            return "Away"
    # Fall back to the actor's teamId via a provided mapping.
    if actor is not None and teamid_to_side:
        side = teamid_to_side.get(actor.get("teamId"))
        if side:
            return side
    # Last resort: map any team entry's id through the provided mapping.
    if teamid_to_side:
        for team in event.get("teams", []) or []:
            side = teamid_to_side.get(team.get("id"))
            if side:
                return side
    return None


def _subtype(genius_type: str, attrs: dict) -> str | None:
    if genius_type in _SETPIECE_SUBTYPE:
        return _SETPIECE_SUBTYPE[genius_type]
    if genius_type == "CLEARANCE":
        return "CLEARANCE"
    if genius_type == "OFFSIDE":
        return "OFFSIDE"
    if genius_type == "PASS":
        if attrs.get("crossed"):
            return "CROSS"
        rt = attrs.get("restartType")
        if rt and rt != "UNKNOWN":
            return str(rt).replace("_", " ")
    return None


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def genius_events_to_dataframe(
    events: list[dict],
    *,
    periods: dict[int, tuple[int, int]],
    fps: float,
    teamid_to_side: dict | None = None,
    participant_id_map: dict | None = None,
    second_half_sign_flip: bool = False,
) -> pd.DataFrame:
    """
    Convert Genius event-spec v1 records to a PLAIQ events DataFrame.

    Parameters
    ----------
    events : parsed Genius FootballEvent dicts
    periods : {period_id: (min_frame, max_frame)} from the tracking loader meta
    fps : frames per second from the tracking loader meta
    teamid_to_side : optional {team_id: "Home"|"Away"} bridge
    participant_id_map : optional {event_participant_id: tracking_player_id} bridge
    second_half_sign_flip : if True, negate x/y for period 2 to match a tracking
        feed that was normalised with a halftime side-swap. Default False
        (assumes the event feed shares the tracking attacking-direction frame).
    """
    rows: list[dict] = []

    for event in events:
        genius_type = str(event.get("type", "")).upper()
        if genius_type in _SKIP_TYPES:
            continue

        meta = event.get("metadata") or {}
        start_frame = meta.get("startFrameIdx")
        if start_frame is None:
            # Cannot place on the tracking timeline without a frame index.
            continue
        end_frame = meta.get("endFrameIdx")
        if end_frame is None:
            end_frame = start_frame

        period_obj = event.get("period") or {}
        period_id = int(period_obj.get("number", 1) or 1)

        attrs = _unwrap_attributes(event.get("attributes"))

        # Type / Subtype
        if genius_type == "PASS":
            completed = attrs.get("complete")
            plaiq_type = "PASS" if completed is not False else "BALL LOST"
        else:
            plaiq_type = _TYPE_MAP.get(genius_type, genius_type)
        subtype = _subtype(genius_type, attrs)

        # Participants -> From / To
        actor = _resolve_participant(event.get("participants", []), _ACTOR_TYPES)
        target = _resolve_participant(event.get("participants", []), _TARGET_TYPES)
        from_id = _map_participant_id(actor.get("id") if actor else None, participant_id_map)
        to_id = _map_participant_id(target.get("id") if target else None, participant_id_map)

        team = _resolve_team(event, actor, teamid_to_side)

        # Coordinates (centre-origin metres -> our convention as-is)
        sx, sy, ex, ey = _start_end_coords(genius_type, attrs)
        if second_half_sign_flip and period_id == 2:
            sx, sy, ex, ey = -sx, -sy, -ex, -ey

        start_t = _period_time_seconds(int(start_frame), period_id, periods, fps)
        end_t = _period_time_seconds(int(end_frame), period_id, periods, fps)

        rows.append({
            "Type": plaiq_type,
            "Subtype": subtype if subtype is not None else np.nan,
            "Period": period_id,
            "From": from_id if from_id is not None else np.nan,
            "To": to_id if to_id is not None else np.nan,
            "Team": team if team is not None else np.nan,
            "Start Frame": int(start_frame),
            "Start Time [s]": start_t,
            "Start X": sx,
            "Start Y": sy,
            "End Frame": int(end_frame),
            "End Time [s]": end_t,
            "End X": ex,
            "End Y": ey,
        })

    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    if not df.empty:
        df = df.sort_values(["Start Frame", "Period"], kind="stable").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class GeniusEventAdapter:
    """Load a Genius football event-spec feed and emit a PLAIQ events DataFrame."""

    def __init__(
        self,
        events_path: str,
        *,
        teamid_to_side: dict | None = None,
        participant_id_map: dict | None = None,
        second_half_sign_flip: bool = False,
    ):
        self.events_path = events_path
        self.teamid_to_side = teamid_to_side
        self.participant_id_map = participant_id_map
        self.second_half_sign_flip = second_half_sign_flip

    def load(self, *, periods: dict[int, tuple[int, int]], fps: float) -> pd.DataFrame:
        raw = load_genius_events_file(self.events_path)
        return genius_events_to_dataframe(
            raw,
            periods=periods,
            fps=fps,
            teamid_to_side=self.teamid_to_side,
            participant_id_map=self.participant_id_map,
            second_half_sign_flip=self.second_half_sign_flip,
        )


def load_genius_events_game(
    events_path: str,
    *,
    periods: dict[int, tuple[int, int]],
    fps: float,
    teamid_to_side: dict | None = None,
    participant_id_map: dict | None = None,
    second_half_sign_flip: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper around :class:`GeniusEventAdapter`."""
    adapter = GeniusEventAdapter(
        events_path,
        teamid_to_side=teamid_to_side,
        participant_id_map=participant_id_map,
        second_half_sign_flip=second_half_sign_flip,
    )
    return adapter.load(periods=periods, fps=fps)
