"""
genius_events_fixture.py — synthetic Genius football event-spec v1 feed.

Produces a small JSONL event feed shaped like
``com.geniussports.dataplatform.football.eventspec.v1`` FootballEvent records,
frame-consistent with the tracking fixture in ss_fixture.py (period 1 frames
0..175, period 2 frames 1000..1175). Participant/team ids match the tracking
fixture's optaIds (h1..h4 / a1..a4, teams HOMEFC/AWAYFC) so the whole pipeline
is coherent and testable end to end.

Coordinates use FootballCoordinate (centre-origin metres), matching PLAIQ's
internal convention.
"""

from __future__ import annotations

import json
import os


def _coord(x, y, z=0.0):
    return {"x": x, "y": y, "z": z}


def _event(event_type, period, start_frame, *, end_frame=None, participants=None,
           teams=None, attributes=None):
    return {
        "id": f"evt-{event_type}-{start_frame}",
        "source": "GeniusInPlay",
        "fixture": {"id": "fixture-001", "externalIds": {}},
        "sport": {"id": "10", "name": "Football"},
        "period": {"number": period, "type": "REGULAR", "status": "IN_PROGRESS"},
        "type": event_type,
        "startFixtureClock": start_frame * 40,  # ms (illustrative)
        "participants": participants or [],
        "teams": teams or [],
        "metadata": {
            "startFrameIdx": start_frame,
            "endFrameIdx": end_frame if end_frame is not None else start_frame,
            "version": "1.0",
            "externalIds": {},
        },
        "tags": {},
        "attributes": attributes,
    }


def _home_team():
    return [{"id": "111", "type": "HOME", "externalIds": {}}]


def _away_team():
    return [{"id": "222", "type": "AWAY", "externalIds": {}}]


def _participant(pid, ptype, team_id):
    return {"id": pid, "teamId": team_id, "type": ptype, "externalIds": {}}


def build_events() -> list[dict]:
    events = []

    # Completed home pass h3 -> h4 (period 1).
    events.append(_event(
        "PASS", 1, 25, end_frame=50,
        participants=[_participant("h3", "PASSER", "111"),
                      _participant("h4", "RECEIVER", "111")],
        teams=_home_team(),
        attributes={"PassAttributes": {
            "complete": True,
            "location": _coord(-5.0, 0.0),
            "endLocation": _coord(12.0, 8.0),
            "isProgressive": True,
        }},
    ))

    # Home carry h4 (period 1).
    events.append(_event(
        "CARRY", 1, 50, end_frame=75,
        participants=[_participant("h4", "POSSESSOR", "111")],
        teams=_home_team(),
        attributes={"CarryAttributes": {
            "startLocation": _coord(12.0, 8.0),
            "endLocation": _coord(20.0, 5.0),
            "distance": 8.5,
        }},
    ))

    # Home shot h4 (period 1).
    events.append(_event(
        "SHOT", 1, 75,
        participants=[_participant("h4", "SHOOTER", "111")],
        teams=_home_team(),
        attributes={"ShotAttributes": {
            "location": _coord(20.0, 5.0),
            "goalmouthLocation": _coord(52.0, 1.0),
            "xG": 0.12,
            "onTarget": True,
        }},
    ))

    # Incomplete away pass a2 -> a3 (period 2) -> should map to BALL LOST.
    events.append(_event(
        "PASS", 2, 1025, end_frame=1050,
        participants=[_participant("a2", "PASSER", "222"),
                      _participant("a3", "RECEIVER", "222")],
        teams=_away_team(),
        attributes={"PassAttributes": {
            "complete": False,
            "location": _coord(10.0, -5.0),
            "endLocation": _coord(-8.0, 12.0),
        }},
    ))

    # Away clearance (period 2) -> BALL LOST / CLEARANCE subtype (DQ-excluded).
    events.append(_event(
        "CLEARANCE", 2, 1075,
        participants=[_participant("a1", "CLEARER", "222")],
        teams=_away_team(),
        attributes={"ClearanceAttributes": {"location": _coord(-40.0, 2.0)}},
    ))

    # Administrative event that must be skipped.
    events.append(_event(
        "DANGER_STATE_CHANGED", 1, 60,
        attributes={"DangerStateChangedAttributes": {"state": "ATTACK"}},
    ))

    # Event missing a frame index -> must be skipped (cannot sync).
    no_frame = _event("PASS", 1, 30,
                      participants=[_participant("h2", "PASSER", "111")],
                      teams=_home_team(),
                      attributes={"PassAttributes": {"complete": True,
                                                     "location": _coord(0.0, 0.0)}})
    no_frame["metadata"]["startFrameIdx"] = None
    no_frame["metadata"]["endFrameIdx"] = None
    events.append(no_frame)

    return events


def write_genius_events_fixture(path: str) -> str:
    """Write the synthetic event feed as JSONL to ``path``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ev in build_events():
            fh.write(json.dumps(ev) + "\n")
    return path
