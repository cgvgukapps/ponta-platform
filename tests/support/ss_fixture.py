"""
ss_fixture.py — deterministic synthetic Second Spectrum fixture generator.

Writes a minimal but schema-valid Second Spectrum feed (metadata XML, additional
metadata JSON, tracking JSONL) into a directory so tests can exercise the real
Kloppy deserializer + our adapter without depending on Kloppy's private test
files or any real (licensed) match data.

Geometry is built to be physically coherent with a halftime side-swap:
    Period 1 — home defends -x (GK at -x), attacks +x.
    Period 2 — teams switch ends: home defends +x natively.
After Kloppy's STATIC_HOME_AWAY normalisation (applied by second_spectrum_io)
the home team must attack +x in BOTH periods, i.e. the home keeper ends up on
the -x side in every frame. Tests assert exactly that invariant.
"""

from __future__ import annotations

import json
import os

# Jersey 1 is the goalkeeper for each side; the rest are outfielders.
_HOME_PLAYERS = [
    {"name": "Home GK", "number": 1, "position": "GK", "optaId": "h1"},
    {"name": "Home RCB", "number": 2, "position": "RCB", "optaId": "h2"},
    {"name": "Home CM", "number": 3, "position": "CM", "optaId": "h3"},
    {"name": "Home ST", "number": 4, "position": "ST", "optaId": "h4"},
]
_AWAY_PLAYERS = [
    {"name": "Away GK", "number": 1, "position": "GK", "optaId": "a1"},
    {"name": "Away LCB", "number": 2, "position": "LCB", "optaId": "a2"},
    {"name": "Away CM", "number": 3, "position": "CM", "optaId": "a3"},
    {"name": "Away ST", "number": 4, "position": "ST", "optaId": "a4"},
]

# Period 1 base x positions (home defends -x). y is fixed per player.
_HOME_BASE = {1: (-50.0, 0.0), 2: (-20.0, -10.0), 3: (-5.0, 0.0), 4: (10.0, 8.0)}
_AWAY_BASE = {1: (50.0, 0.0), 2: (20.0, 10.0), 3: (5.0, 0.0), 4: (-10.0, -8.0)}

# Each half needs more frames than the velocity smoothing window (7).
_FRAMES_PER_PERIOD = 8
_DX_PER_FRAME = 1.0  # metres/frame drift toward each team's attacking goal


def _player_frame(number: int, base_x: float, base_y: float, step: int, team_dir: int, mirror: bool, speed: float):
    # In period 1 home attacks +x (team_dir=+1), away attacks -x (team_dir=-1).
    x = base_x + _DX_PER_FRAME * step * team_dir
    y = base_y
    if mirror:  # period 2: mirror to model the halftime side-swap
        x = -x
        y = -y
    return {"playerId": f"raw{number}", "number": number, "xyz": [x, y, 0.0], "speed": speed}


def _build_frames():
    lines = []
    for period, mirror, idx0, gc0 in [(1, False, 0, 0.0), (2, True, 1000, 40.0)]:
        for step in range(_FRAMES_PER_PERIOD):
            home = [
                _player_frame(p["number"], *_HOME_BASE[p["number"]], step, +1, mirror, 3.0 + 0.2 * step)
                for p in _HOME_PLAYERS
            ]
            away = [
                _player_frame(p["number"], *_AWAY_BASE[p["number"]], step, -1, mirror, 2.5 + 0.2 * step)
                for p in _AWAY_PLAYERS
            ]
            ball_x = (-30.0 + 4.0 * step) * (-1.0 if mirror else 1.0)
            frame = {
                "period": period,
                "frameIdx": idx0 + 25 * step,
                "gameClock": gc0 + float(step),
                "homePlayers": home,
                "awayPlayers": away,
                "ball": {"xyz": [ball_x, 1.0, 1.5], "speed": [12.0]},
                "live": True,
                "lastTouch": "home",
            }
            lines.append(json.dumps(frame))
    return "\n".join(lines) + "\n"


def _metadata_xml(pitch_length: float, pitch_width: float, fps: int) -> str:
    last = 25 * (_FRAMES_PER_PERIOD - 1)
    return (
        '<MetaData sVersion="1.0">\n'
        f'   <match dtDate="2026-01-01 00:00:00" fPitchXSizeMeters="{pitch_length}" '
        f'fPitchYSizeMeters="{pitch_width}" iFrameRateFps="{fps}" iId="0">\n'
        f'      <period iEndFrame="{last}" iId="1" iStartFrame="0"/>\n'
        f'      <period iEndFrame="{1000 + last}" iId="2" iStartFrame="1000"/>\n'
        '      <period iEndFrame="0" iId="3" iStartFrame="0"/>\n'
        '      <period iEndFrame="0" iId="4" iStartFrame="0"/>\n'
        '   </match>\n'
        '</MetaData>\n'
    )


def _metadata_json(pitch_length: float, pitch_width: float, fps: int) -> dict:
    def _players(players):
        out = []
        for p in players:
            out.append(
                {
                    "name": p["name"],
                    "number": p["number"],
                    "position": p["position"],
                    "ssiId": f"ssi_{p['optaId']}",
                    "optaId": p["optaId"],
                    "optaUuid": f"uuid_{p['optaId']}",
                }
            )
        return out

    return {
        "description": "HOMEFC - AWAYFC : 2026-1-1",
        "startTime": 0,
        "year": 2026,
        "month": 1,
        "day": 1,
        "pitchLength": pitch_length,
        "pitchWidth": pitch_width,
        "fps": float(fps),
        "periods": [
            {"number": 1, "startFrameClock": 0, "endFrameClock": 25 * (_FRAMES_PER_PERIOD - 1)},
            {"number": 2, "startFrameClock": 1000, "endFrameClock": 1000 + 25 * (_FRAMES_PER_PERIOD - 1)},
        ],
        "homePlayers": _players(_HOME_PLAYERS),
        "awayPlayers": _players(_AWAY_PLAYERS),
        "homeScore": 1,
        "awayScore": 0,
        "ssiId": "fixture-001",
        "homeOptaId": 111,
        "awayOptaId": 222,
    }


def write_ss_fixture(
    directory: str,
    *,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    fps: int = 25,
    meta_file: str = "metadata.xml",
    tracking_file: str = "tracking.jsonl",
    additional_meta_file: str = "metadata.json",
) -> dict:
    """Write a synthetic SS fixture into ``directory``; return the file paths."""
    os.makedirs(directory, exist_ok=True)
    meta_path = os.path.join(directory, meta_file)
    tracking_path = os.path.join(directory, tracking_file)
    additional_path = os.path.join(directory, additional_meta_file)

    with open(meta_path, "w", encoding="utf-8") as fh:
        fh.write(_metadata_xml(pitch_length, pitch_width, fps))
    with open(tracking_path, "w", encoding="utf-8") as fh:
        fh.write(_build_frames())
    with open(additional_path, "w", encoding="utf-8") as fh:
        json.dump(_metadata_json(pitch_length, pitch_width, fps), fh, indent=2)

    return {
        "data_dir": directory,
        "meta_file": meta_file,
        "tracking_file": tracking_file,
        "additional_meta_file": additional_meta_file,
    }
