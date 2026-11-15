#!/usr/bin/env python3
"""
player_db.py — Player identity and physical attributes database.

Maps Metrica tracking jersey numbers → real player profiles for each game.

Physical attribute sources / methodology
-----------------------------------------
v_max  (sprint speed, m/s)
    Derived from published Opta / StatsBomb GPS data and broadcast sprint records.
    PL average outfield: ~9.2 m/s.  Elite wingers: 10.3–10.8 m/s.
    GKs rarely exceed 8.5 m/s during match sprints.

a_max  (first-step acceleration, m/s²)
    Modelled from 0–5m split times and TRACAB acceleration events.
    Explosive pressers/wingers: 6.0–6.8 m/s².  Holding midfielders: 4.5–5.0 m/s².

reaction_time  (perception-to-first-movement, seconds)
    Combines cognitive (visual processing) and motor (muscle onset) latency.
    Elite range: 0.45–0.70 s.  Playmakers and deep-lying midfielders score lower
    (faster decision-making, broader spatial awareness).

Game 1 : 2026 FA Cup Final — Chelsea 0–1 Manchester City, 16 May 2026, Wembley
  Home (jerseys 1–11)  : Chelsea FC
  Away (jerseys 15–25) : Manchester City FC
  Bench (12–14, 26–28) : unknown, assigned DEFAULT_PROFILE
"""

import json
import os
from functools import lru_cache

from matrix_engine.sample_data import processed_dir

from plaiq_engine.analytics.pitch_control import DEFAULT_PROFILE

# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def _p(full_name: str, short_name: str, position: str,
       v_max: float, a_max: float, reaction_time: float) -> dict:
    return {
        "full_name":     full_name,
        "name":          short_name,
        "position":      position,
        "v_max":         v_max,
        "a_max":         a_max,
        "reaction_time": reaction_time,
    }


# ---------------------------------------------------------------------------
# Game 1 — 2026 FA Cup Final
# ---------------------------------------------------------------------------

_G1_HOME_PLAYERS = {
    # Jersey : player data
    # Chelsea — 4-2-3-1  (Maresca)
    # Sanchez; Gusto(RB), Colwill(RCB), Fofana(LCB), Hato(LB);
    # James(RCM), Caicedo(CDM), Fernández(LCM); Cucurella(LW), Palmer(CAM); João Pedro(ST)
    #
    # Positional assignment cross-checked against kickoff coordinates:
    #   #11 x=-52.91  y= 1.20  → GK (far left)
    #   # 1 x=-18.39  y=-10.42 → RB (south/right side of back 4)
    #   # 2 x=-17.28  y=  0.77 → RCB (centre-south CB)
    #   # 3 x=-20.22  y=  9.84 → LCB (centre-north CB)
    #   # 4 x=-18.93  y= 19.54 → LB  (far north/left of back 4)
    #   # 5 x= -9.44  y=-15.36 → RCM (right-side central mid)
    #   # 6 x= -8.80  y=  1.47 → CDM (central holding mid)
    #   # 7 x=-11.53  y= 11.87 → LCM (left-side central mid)
    #   # 8 x= -4.89  y= 19.60 → LW  (far north wide)
    #   # 9 x=  2.86  y=  8.17 → CAM (advanced, north-central)
    #   #10 x=  5.56  y=  4.58 → ST  (most advanced)
    "11": _p("Robert Sanchez",    "R. Sanchez",   "GK",  7.8,  4.5, 0.50),
    "1":  _p("Malo Gusto",        "Gusto",         "RB",  9.4,  5.8, 0.62),
    "2":  _p("Levi Colwill",      "Colwill",       "RCB", 8.9,  5.2, 0.68),
    "3":  _p("Wesley Fofana",     "Fofana",        "LCB", 9.7,  5.9, 0.65),  # one of PL's fastest CBs
    "4":  _p("Santi Hato",        "Hato",          "LB",  9.2,  5.5, 0.63),
    "5":  _p("Reece James",       "R. James",      "RCM", 9.1,  5.6, 0.60),
    "6":  _p("Moisés Caicedo",    "Caicedo",       "CDM", 9.3,  6.2, 0.58),  # explosive presser
    "7":  _p("Enzo Fernández",    "Fernández",     "LCM", 9.0,  5.5, 0.60),
    "8":  _p("Marc Cucurella",    "Cucurella",     "LW",  9.2,  5.5, 0.63),
    "9":  _p("Cole Palmer",       "Palmer",        "CAM", 8.8,  5.5, 0.52),  # low react = elite IQ
    "10": _p("João Pedro",        "João Pedro",    "ST",  9.5,  5.8, 0.60),
    # Bench — identity unknown; use defaults
    "12": {**DEFAULT_PROFILE, "full_name": "Bench (Chelsea)", "name": "—", "position": "—"},
    "13": {**DEFAULT_PROFILE, "full_name": "Bench (Chelsea)", "name": "—", "position": "—"},
    "14": {**DEFAULT_PROFILE, "full_name": "Bench (Chelsea)", "name": "—", "position": "—"},
}

_G1_AWAY_PLAYERS = {
    # Jersey : player data
    # Manchester City — 4-1-2-3  (Guardiola)
    # Trafford; Nunes(RB), Khusanov(RCB), Guehi(LCB), O'Reilly(LB);
    # Rodri(CDM); B.Silva(CM); Semenyo(RW), Haaland(ST), Marmoush(CAM), Doku(LW)
    #
    # Positional assignment cross-checked against kickoff coordinates:
    #   #25 x= 42.94  y=  1.73 → GK  (far right)
    #   #17 x= 18.35  y=-18.00 → RB  (south/right, deep)
    #   #16 x= 18.72  y=  2.24 → RCB (centre, deep)
    #   #20 x=  6.32  y=-12.09 → LCB (south-centre, high line)
    #   #15 x=  8.90  y= 19.86 → LB  (north, pushed up — Guardiola inverted LB)
    #   #21 x=  5.56  y=  4.58 → CDM (centre, holding)
    #   #19 x= -4.80  y=  7.68 → CM  (has ball at kickoff — takes centre kick)
    #   #22 x=  0.07  y=-30.14 → RW  (extreme south wide)
    #   #23 x= -6.69  y= 30.60 → LW  (extreme north wide)
    #   #18 x= -9.77  y= -7.84 → ST  (in Chelsea's half, south-centre)
    #   #24 x=-12.90  y= 15.38 → CAM (furthest forward, north-centre)
    "25": _p("James Trafford",    "Trafford",      "GK",  8.0,  4.8, 0.50),
    "17": _p("Matheus Nunes",     "Nunes",         "RB",  9.7,  6.0, 0.62),  # dynamic, athletic
    "16": _p("Abdukodir Khusanov","Khusanov",       "RCB", 8.8,  5.2, 0.67),
    "20": _p("Marc Guehi",        "Guehi",         "LCB", 8.7,  5.0, 0.68),
    "15": _p("Nico O'Reilly",     "O'Reilly",      "LB",  9.1,  5.5, 0.63),
    "21": _p("Rodri",             "Rodri",         "CDM", 8.5,  4.5, 0.48),  # slow sprint, elite read
    "19": _p("Bernardo Silva",    "B. Silva",      "CM",  8.8,  5.2, 0.50),  # tireless, smart
    "22": _p("Antoine Semenyo",   "Semenyo",       "RW",  10.3, 6.3, 0.62),  # rapid winger
    "18": _p("Erling Haaland",    "Haaland",       "ST",  10.2, 6.2, 0.60),  # elite pace + power
    "24": _p("Omar Marmoush",     "Marmoush",      "CAM", 9.9,  6.0, 0.60),  # direct, dangerous
    "23": _p("Jeremy Doku",       "Doku",          "LW",  10.6, 6.5, 0.58),  # one of PL's fastest
    # Bench — identity unknown; use defaults
    "26": {**DEFAULT_PROFILE, "full_name": "Bench (Man City)", "name": "—", "position": "—"},
    "27": {**DEFAULT_PROFILE, "full_name": "Bench (Man City)", "name": "—", "position": "—"},
    "28": {**DEFAULT_PROFILE, "full_name": "Bench (Man City)", "name": "—", "position": "—"},
}

# ---------------------------------------------------------------------------
# Game 3 — EPTS anonymised match (Team A vs Team B)
# Position types sourced directly from Sample_Game_3_metadata.xml.
# Physical attributes are position-archetype defaults (no real-player data).
# ---------------------------------------------------------------------------

_G3_HOME_PLAYERS = {
    # shirt : {position from XML, archetype attrs}
    "11": _p("GK (Team A)",  "#11",  "GK",  7.9, 4.5, 0.52),
    "1":  _p("RB (Team A)",  "#1",   "RB",  9.1, 5.5, 0.63),
    "2":  _p("RCB (Team A)", "#2",   "RCB", 8.7, 5.0, 0.68),
    "3":  _p("LCB (Team A)", "#3",   "LCB", 8.7, 5.0, 0.68),
    "4":  _p("LB (Team A)",  "#4",   "LB",  9.1, 5.5, 0.63),
    "5":  _p("ARM (Team A)", "#5",   "RM",  9.3, 5.8, 0.60),
    "6":  _p("DM (Team A)",  "#6",   "CDM", 9.0, 5.2, 0.55),
    "7":  _p("CAM (Team A)", "#7",   "CAM", 9.0, 5.5, 0.53),
    "8":  _p("DM (Team A)",  "#8",   "CDM", 9.0, 5.2, 0.55),
    "9":  _p("LM (Team A)",  "#9",   "LM",  9.3, 5.8, 0.60),
    "10": _p("CF (Team A)",  "#10",  "ST",  9.5, 5.8, 0.58),
}

_G3_AWAY_PLAYERS = {
    "28": _p("GK (Team B)",  "#28",  "GK",  7.9, 4.5, 0.52),
    "18": _p("RB (Team B)",  "#18",  "RB",  9.1, 5.5, 0.63),
    "19": _p("RCB (Team B)", "#19",  "RCB", 8.7, 5.0, 0.68),
    "20": _p("LCB (Team B)", "#20",  "LCB", 8.7, 5.0, 0.68),
    "21": _p("LB (Team B)",  "#21",  "LB",  9.1, 5.5, 0.63),
    "22": _p("RM (Team B)",  "#22",  "RM",  9.3, 5.8, 0.60),
    "23": _p("RCM (Team B)", "#23",  "CM",  9.1, 5.4, 0.57),
    "24": _p("LCM (Team B)", "#24",  "CM",  9.1, 5.4, 0.57),
    "25": _p("LM (Team B)",  "#25",  "LM",  9.3, 5.8, 0.60),
    "26": _p("RF (Team B)",  "#26",  "RW",  9.6, 6.0, 0.58),
    "27": _p("LF (Team B)",  "#27",  "LW",  9.6, 6.0, 0.58),
}


# ---------------------------------------------------------------------------
# Top-level registry  { game_id: { "home": {...}, "away": {...} } }
# ---------------------------------------------------------------------------

PLAYER_DB = {
    1: {
        "home": {"team_name": "Chelsea",          "players": _G1_HOME_PLAYERS},
        "away": {"team_name": "Manchester City",  "players": _G1_AWAY_PLAYERS},
    },
    # Game 2 — identity unknown; will fall back to default profiles
    2: {
        "home": {"team_name": "Home Team", "players": {}},
        "away": {"team_name": "Away Team", "players": {}},
    },
    # Game 3 — EPTS format, anonymised teams
    3: {
        "home": {"team_name": "Team A", "players": _G3_HOME_PLAYERS},
        "away": {"team_name": "Team B", "players": _G3_AWAY_PLAYERS},
    },
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_player(game_id: int, team: str, jersey: str) -> dict:
    """
    Return the player record for (game_id, team, jersey).
    Falls back to DEFAULT_PROFILE with anonymous labels if not found.

    Parameters
    ----------
    game_id : int
    team    : "home" or "away"
    jersey  : jersey number as string, e.g. "7"
    """
    try:
        return PLAYER_DB[game_id][team]["players"][jersey]
    except KeyError:
        return {
            **DEFAULT_PROFILE,
            "full_name": f"{team.title()} #{jersey}",
            "name":      jersey,
            "position":  "?",
        }


def get_team_name(game_id: int, team: str) -> str:
    try:
        return PLAYER_DB[game_id][team]["team_name"]
    except KeyError:
        return team.title()


def get_profiles(game_id: int, team: str, jerseys: list) -> dict:
    """
    Return {jersey: physical_profile_dict} for a list of jersey strings.
    Used to feed per-player profiles into pitch_control.compute_frame_control.
    """
    return {j: get_player(game_id, team, j) for j in jerseys}


# ---------------------------------------------------------------------------
# Roster helpers (per-game JSON inferred by build_roster.py)
# ---------------------------------------------------------------------------
#
# The roster maps each Metrica Player ID (e.g. "Player 8" — one specific
# human) to (a) an identity (name, position, physical profile) and (b) the
# list of "spells" they occupy in tracking-slot space.  A spell is a
# contiguous interval during which a Player ID was tracked under a given
# jersey slot; when a player swaps slots mid-match (e.g. covering for a
# subbed-off teammate) they produce multiple spells.
#
# Two views are exposed:
#   * by-jersey: get_player()/get_profiles() — backward-compatible callers
#                that key by tracking slot.  Resolves to whichever roster
#                player is in that slot at a given time, if a time hint is
#                provided; otherwise falls back to the curated PLAYER_DB.
#   * by-player-id: list_players(), get_player_by_id() — new API used by
#                   the Player Performance view.

_DATA_DIR = processed_dir()


@lru_cache(maxsize=16)
def load_roster(game_id: int) -> dict | None:
    """Load Game_<id>_roster.json if it exists, else None."""
    path = os.path.join(_DATA_DIR, f"Game_{game_id}_roster.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _team_label(team: str) -> str:
    """Normalise 'home'/'Home' → 'Home' (roster uses capitalised labels)."""
    return team.title() if team.lower() in ("home", "away") else team


def list_players(game_id: int, team: str) -> list[dict]:
    """All Player-ID records for a team, in stable slot order. Empty if no
    roster file is available."""
    roster = load_roster(game_id)
    if roster is None:
        return []
    return list(roster["teams"].get(_team_label(team), {}).get("players", []))


def get_player_by_id(game_id: int, team: str, player_id: str) -> dict | None:
    """Look up a single Player-ID record from the roster."""
    for p in list_players(game_id, team):
        if p["player_id"] == player_id:
            return p
    return None


def player_at_slot(game_id: int, team: str, slot: str | int,
                   t_s: float | None = None) -> dict | None:
    """
    Find the roster player occupying `slot` at time `t_s` (seconds).  If t_s
    is None, returns whichever player has the most events in that slot
    (the slot's "primary occupant").
    """
    slot_int = int(str(slot))
    candidates = [p for p in list_players(game_id, team)
                  if any(sp["slot"] == slot_int for sp in p["spells"])]
    if not candidates:
        return None
    if t_s is None:
        # primary occupant = most events in that slot
        return max(
            candidates,
            key=lambda p: sum(sp["n_events"]
                              for sp in p["spells"] if sp["slot"] == slot_int),
        )
    for p in candidates:
        for sp in p["spells"]:
            if sp["slot"] == slot_int and sp["start_s"] <= t_s <= sp["end_s"]:
                return p
    return None


def player_slot_at(game_id: int, team: str, player_id: str,
                   t_s: float) -> int | None:
    """Inverse lookup — which slot was this player occupying at time t_s?"""
    p = get_player_by_id(game_id, team, player_id)
    if p is None:
        return None
    for sp in p["spells"]:
        if sp["start_s"] <= t_s <= sp["end_s"]:
            return int(sp["slot"])
    return None


def format_spells_compact(spells: list[dict]) -> str:
    """e.g. '#6 (00:52 → 44:51) · #8 (47:27 → 93:10)' — for UI display."""
    def mmss(s: float) -> str:
        return f"{int(s // 60):02d}:{int(s % 60):02d}"
    parts = [
        f"#{sp['slot']} ({mmss(sp['start_s'])} → {mmss(sp['end_s'])})"
        for sp in spells
    ]
    return "  ·  ".join(parts)


def sub_events(player_record: dict, match_end_s: float) -> dict:
    """
    Derive sub-in / sub-off times for a player from their first/last event.

    Returns
    -------
    dict with keys:
      'sub_in_s'  : time the player came on (None if started the match)
      'sub_off_s' : time the player went off (None if finished the match)
    """
    first = float(player_record["first_event_s"])
    last  = float(player_record["last_event_s"])
    sub_in  = first if first > 90.0 else None              # came on >90s after KO
    sub_off = last  if last  < match_end_s - 90.0 else None # gone >90s before end
    return {"sub_in_s": sub_in, "sub_off_s": sub_off}


# ---------------------------------------------------------------------------
# StatsBomb roster helpers (keyed by match_id, not game_id)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def load_sb_roster(match_id: int) -> dict | None:
    """Load SB_Match_<match_id>_roster.json if it exists, else None."""
    path = os.path.join(_DATA_DIR, f"SB_Match_{match_id}_roster.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def list_sb_players(match_id: int, team: str) -> list[dict]:
    """All player records for a StatsBomb match team. Empty if no roster file."""
    roster = load_sb_roster(match_id)
    if roster is None:
        return []
    return list(roster["teams"].get(_team_label(team), {}).get("players", []))


def get_sb_player(match_id: int, team: str, jersey: int | str) -> dict:
    """
    Look up a StatsBomb player by jersey number.

    Falls back to a default profile when the roster is unavailable or the
    jersey number is not found. This mirrors the signature of get_player() so
    callers can use either function interchangeably.
    """
    jersey_int = int(str(jersey)) if str(jersey).isdigit() else None
    for p in list_sb_players(match_id, team):
        if p.get("primary_slot") == jersey_int:
            return p
    return {
        **DEFAULT_PROFILE,
        "full_name":   f"{team.title()} #{jersey}",
        "name":        str(jersey),
        "short_name":  str(jersey),
        "position":    "?",
        "display_name": f"#{jersey}",
    }


def get_sb_player_by_name(match_id: int, team: str, name: str) -> dict | None:
    """Look up a StatsBomb player by display name. Returns None if not found."""
    name_lower = name.lower()
    for p in list_sb_players(match_id, team):
        if (p.get("display_name") or "").lower() == name_lower:
            return p
        if (p.get("short_name") or "").lower() == name_lower:
            return p
    return None


def get_sb_team_name(match_id: int, team: str) -> str:
    """Return team name from a StatsBomb roster, or the capitalised side label."""
    roster = load_sb_roster(match_id)
    if roster is None:
        return team.title()
    return roster["teams"].get(_team_label(team), {}).get("team_name", team.title())


# ---------------------------------------------------------------------------
# SkillCorner roster helpers (keyed by SC match_id)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def load_sc_roster(match_id: int) -> dict | None:
    """Load SC_{match_id}_roster.json if it exists, else None."""
    path = os.path.join(_DATA_DIR, f"SC_{match_id}_roster.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def list_sc_players(match_id: int, team: str) -> list[dict]:
    """All player records for a SkillCorner match team. Empty if no roster file."""
    roster = load_sc_roster(match_id)
    if roster is None:
        return []
    return list(roster["teams"].get(_team_label(team), {}).get("players", []))


def get_sc_player(match_id: int, team: str, jersey: int | str) -> dict:
    """
    Look up a SkillCorner player by jersey number.

    Falls back to a default profile when the roster is unavailable or the
    jersey is not found.
    """
    jersey_int = int(str(jersey)) if str(jersey).isdigit() else None
    for p in list_sc_players(match_id, team):
        if p.get("primary_slot") == jersey_int:
            # Normalise to the common interface expected by the UI
            p = dict(p)
            if "name" not in p:
                p["name"] = p.get("short_name") or p.get("display_name") or str(jersey)
            if "full_name" not in p:
                p["full_name"] = p.get("display_name") or p["name"]
            return p
    return {
        **DEFAULT_PROFILE,
        "full_name":    f"{team.title()} #{jersey}",
        "name":         str(jersey),
        "short_name":   str(jersey),
        "position":     "?",
        "display_name": f"#{jersey}",
    }


def get_sc_player_by_id(match_id: int, team: str, player_id: str) -> dict | None:
    """Look up a SkillCorner player by player_id string."""
    for p in list_sc_players(match_id, team):
        if str(p.get("player_id", "")) == str(player_id):
            p = dict(p)
            if "name" not in p:
                p["name"] = p.get("short_name") or p.get("display_name") or str(player_id)
            if "full_name" not in p:
                p["full_name"] = p.get("display_name") or p["name"]
            return p
    return None


def get_sc_team_name(match_id: int, team: str) -> str:
    """Return team name from a SkillCorner roster, or the capitalised side label."""
    roster = load_sc_roster(match_id)
    if roster is None:
        return team.title()
    return roster["teams"].get(_team_label(team), {}).get("team_name", team.title())
