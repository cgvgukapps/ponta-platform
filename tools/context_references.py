"""
context_references.py — reference labels for validating context packages (Gate 2a).

SkillCorner's open data carries its own labels: a role for every player and a phase of play
on every dynamic event. Under founder decision 2 (``ponta-blueprint/context-packages.md``)
these are **comparison data only**: they measure agreement, they never become Ponta's
output. Only ids, shirt numbers, roles, frames and phase columns are read; player names
never are.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import sample_data_dir

SC_ID = 1886347
_BASE = Path(sample_data_dir()) / "skillcorner" / "matches" / str(SC_ID)


def _files(match_dir: Path | None) -> tuple[Path, Path]:
    """The match file and dynamic-events file of a SkillCorner match folder (default: the regression match)."""
    base = Path(match_dir) if match_dir is not None else _BASE
    mid = base.name
    return base / f"{mid}_match.json", base / f"{mid}_dynamic_events.csv"


def _hms(s: str | None) -> float | None:
    if not s:
        return None
    h, m, sec = (float(x) for x in s.split(":"))
    return h * 3600 + m * 60 + sec


def sc_roles(match_dir: Path | None = None) -> pd.DataFrame:
    """One row per player who played: ``player_id`` (``home_10``), role acronym, on and off match time (s)."""
    match_file, events_file = _files(match_dir)
    meta = json.loads(match_file.read_text())
    home = meta["home_team"]["id"]
    rows = []
    for p in meta["players"]:
        role = p["player_role"]["acronym"] if isinstance(p["player_role"], dict) else p["player_role"]
        if role == "SUB" or p.get("start_time") is None:
            continue
        team = "home" if p["team_id"] == home else "away"
        rows.append({"player_id": f"{team}_{p['number']}", "team": team, "role": role,
                     "on_s": _hms(p["start_time"]), "off_s": _hms(p["end_time"])})
    return pd.DataFrame(rows).sort_values("player_id").reset_index(drop=True)


def sc_phases(grid: pd.DataFrame, match_dir: Path | None = None) -> pd.DataFrame:
    """
    SkillCorner's phase of play at each grid point, from its player-possession events: the team
    in possession and both teams' phase types. Null between possessions.
    """
    match_file, events_file = _files(match_dir)
    meta = json.loads(match_file.read_text())
    home = meta["home_team"]["id"]
    ev = pd.read_csv(events_file,
                     usecols=["event_type", "frame_start", "frame_end", "team_id",
                              "team_in_possession_phase_type", "team_out_of_possession_phase_type"])
    pp = ev.loc[ev["event_type"] == "player_possession"].dropna(subset=["frame_start", "frame_end"]).sort_values("frame_start")
    nf = grid["native_frame"].to_numpy()
    team = np.full(len(grid), None, dtype=object)
    ip = np.full(len(grid), None, dtype=object)
    op = np.full(len(grid), None, dtype=object)
    for _, r in pp.iterrows():
        m = (nf >= r["frame_start"]) & (nf <= r["frame_end"])
        team[m] = "home" if r["team_id"] == home else "away"
        ip[m] = r["team_in_possession_phase_type"]
        op[m] = r["team_out_of_possession_phase_type"]
    return pd.DataFrame({"sc_team_in_possession": team, "sc_in_possession_phase": ip, "sc_out_of_possession_phase": op})


def sc_phase_spans(grid: pd.DataFrame, match_dir: Path | None = None) -> pd.DataFrame:
    """
    SkillCorner's phases as spans: each ``phase_index`` runs from its first player possession's
    start to its last one's end. Gives the team in possession and the in-possession phase at
    every grid point a phase covers (about 40% of match time). Comparison only.
    """
    match_file, events_file = _files(match_dir)
    meta = json.loads(match_file.read_text())
    home = meta["home_team"]["id"]
    ev = pd.read_csv(events_file,
                     usecols=["event_type", "frame_start", "frame_end", "team_id", "phase_index", "team_in_possession_phase_type"])
    pp = ev.loc[ev["event_type"] == "player_possession"].dropna(subset=["frame_start", "frame_end", "phase_index"])
    spans = pp.groupby("phase_index").agg(s=("frame_start", "min"), e=("frame_end", "max"),
                                          phase=("team_in_possession_phase_type", lambda x: x.mode().iloc[0]),
                                          team=("team_id", lambda x: x.mode().iloc[0]), teams=("team_id", "nunique")).sort_values("s")
    nf = grid["native_frame"].to_numpy()
    team = np.full(len(grid), None, dtype=object)
    phase = np.full(len(grid), None, dtype=object)
    for r in spans.itertuples():
        m = (nf >= r.s) & (nf <= r.e)
        team[m] = None if r.teams > 1 else ("home" if r.team == home else "away")
        phase[m] = r.phase
    return pd.DataFrame({"sc_team": team, "sc_phase": phase})


def sc_phase_file(grid: pd.DataFrame, match_dir: Path) -> pd.DataFrame:
    """
    SkillCorner's own phases-of-play file (newer open-data releases): one row per phase with its
    frames and the team in possession. Gives ``sc_team`` and ``sc_phase`` at every grid point a
    phase covers; null when the ball is out of play. Comparison only; no names are read.
    """
    base = Path(match_dir)
    meta = json.loads((base / f"{base.name}_match.json").read_text())
    home = meta["home_team"]["id"]
    ph = pd.read_csv(base / f"{base.name}_phases_of_play.csv",
                     usecols=["frame_start", "frame_end", "team_in_possession_id", "team_in_possession_phase_type"])
    nf = grid["native_frame"].to_numpy()
    team = np.full(len(grid), None, dtype=object)
    phase = np.full(len(grid), None, dtype=object)
    for r in ph.itertuples():
        m = (nf >= r.frame_start) & (nf <= r.frame_end)
        team[m] = "home" if r.team_in_possession_id == home else "away"
        phase[m] = r.team_in_possession_phase_type
    return pd.DataFrame({"sc_team": team, "sc_phase": phase})
