#!/usr/bin/env python3
"""
build_roster.py — Infer a per-game player roster from tracking + events data.

Why a roster file is necessary
------------------------------
The raw Metrica data uses two distinct keys for "who is on the pitch":

  * Tracking CSVs use **jersey slots** (e.g. column `Home_8_x` is "whoever is
    currently in the home #8 slot").  A jersey slot is reused when a player
    is subbed off — the substitute's positions silently take over the slot.

  * Event CSVs use **Player IDs** (e.g. `From == "Player 8"` is one specific
    human).  A Player ID never reappears after that player has been subbed
    off.

The app needs a stable mapping between these two views so that, for any
selected human player, we can:
  1) filter their event-based metrics (DQ, ΔPOV) by Player ID, and
  2) read their tracking-based metrics (MV) from the correct slot at any
     given frame, even if they swap slots mid-match (e.g. Game 3 Player 6
     moves from slot 6 → slot 8 at HT when Player 8 goes off).

This script infers that mapping by, for each Player ID, looking up the
event's recorded (X, Y) at its `Start Frame` and choosing the jersey slot
whose tracking position is closest.  Inferred per-half to capture mid-match
slot swaps.  Spells are then assembled, names are generated for any
players not in player_db.PLAYER_DB, and the result is written to
`sample-data/data/processed/Game_{N}_roster.json`.

Run:
    python build_roster.py            # builds rosters for Games 1, 2, 3
    python build_roster.py --game 3   # single game
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd
from matrix_engine.sample_data import processed_dir

from plaiq_engine.analytics.pitch_control import DEFAULT_PROFILE
from plaiq_engine.roster.player_db import PLAYER_DB, get_player

# Processed-data directory, resolved outside the repo (see matrix_engine.sample_data).
# Defined locally so the Second Spectrum roster path stays free of any demo-script import.
DATA_DIR = processed_dir()


# ---------------------------------------------------------------------------
# Name generation pools (used only when no real identity is known)
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "Lucas", "Mateo", "Noah", "Diego", "Tomás", "Adrián", "Felix", "Marco",
    "Aitor", "Oliver", "Hugo", "Eli", "Jonas", "Niko", "Pablo", "Theo",
    "Andrés", "Kai", "Dani", "Raúl", "Sergio", "Luca", "Iván", "Joel",
    "Bruno", "Carlos", "Javi", "Mikel", "Rubén", "Saúl", "Víctor", "Xavi",
    "Cristian", "Ezequiel", "Gonzalo", "Joaquín", "Leo", "Manu", "Nacho",
    "Pedro", "Rodri", "Samuel", "Unai", "Yago",
]

_LAST_NAMES = [
    "Vidal",    "Reyes",     "Navarro",  "Aguilar",  "Castro",  "Ortega",
    "Romero",   "Soler",     "Hidalgo",  "Lozano",   "Mendoza", "Iglesias",
    "Vargas",   "Cabrera",   "Espinosa", "Fuentes",  "Guerrero","Heredia",
    "Ibáñez",   "Jiménez",   "Lara",     "Marín",    "Núñez",   "Olmedo",
    "Paredes",  "Quintero",  "Ramos",    "Salazar",  "Toro",    "Ureña",
    "Valle",    "Yáñez",     "Zabala",   "Bermejo",  "Carrasco","Delgado",
    "Esteban",  "Fonseca",   "Galindo",  "Herrera",  "Izquierdo","Jaén",
    "Kross",    "Linares",   "Molina",   "Negrete",  "Oviedo",  "Palomar",
]


def _gen_name(seed_str: str) -> tuple[str, str]:
    """Deterministic plausible name from a string seed."""
    rng = random.Random(seed_str)
    first = rng.choice(_FIRST_NAMES)
    last  = rng.choice(_LAST_NAMES)
    return first, last


# ---------------------------------------------------------------------------
# Slot inference
# ---------------------------------------------------------------------------

def _tracking_slot_numbers(tracking: pd.DataFrame, side: str) -> list[int]:
    cols = [c for c in tracking.columns
            if c.startswith(f"{side}_") and c.endswith("_x") and "ball" not in c]
    return sorted({int(c.split("_")[1]) for c in cols
                   if c.split("_")[1].isdigit()})


def _slot_at(side: str, tracking: pd.DataFrame, frame: int,
             x: float, y: float, slots: Iterable[int]) -> int | None:
    if frame not in tracking.index:
        return None
    best, best_d2 = None, np.inf
    for s in slots:
        col_x, col_y = f"{side}_{s}_x", f"{side}_{s}_y"
        if col_x not in tracking.columns:
            continue
        tx = tracking.at[frame, col_x]
        ty = tracking.at[frame, col_y]
        if pd.isna(tx) or pd.isna(ty):
            continue
        d2 = (tx - x) ** 2 + (ty - y) ** 2
        if d2 < best_d2:
            best_d2, best = d2, s
    return best


def _build_spells(player_events: pd.DataFrame,
                  side: str,
                  tracking: pd.DataFrame,
                  slots: list[int]) -> list[dict]:
    """
    Infer (slot, start_s, end_s) spells for one Player ID's events.

    Strategy
    --------
    For each event the player produces, identify which jersey slot in tracking
    is geographically closest at the event's `Start Frame`.  Bucket events by
    Period and find the dominant slot per half; if it changes between H1 and
    H2 (e.g. Game 3 Player 6: slot 6 → slot 8), emit two spells.
    """
    # Per-event slot observation
    obs = []
    for _, ev in player_events.iterrows():
        fr = ev.get("Start Frame")
        ex, ey = ev.get("Start X"), ev.get("Start Y")
        t = ev.get("Start Time [s]")
        if pd.isna(fr) or pd.isna(ex) or pd.isna(ey) or pd.isna(t):
            continue
        slot = _slot_at(side, tracking, int(fr), float(ex), float(ey), slots)
        if slot is None:
            continue
        obs.append({"period": int(ev["Period"]),
                    "slot": int(slot),
                    "t": float(t)})

    if not obs:
        return []

    obs_df = pd.DataFrame(obs)

    spells: list[dict] = []
    for period, g in obs_df.groupby("period", sort=True):
        dominant_slot, n_dom = Counter(g["slot"]).most_common(1)[0]
        # Restrict to events on the dominant slot for the spell time bounds
        g_dom = g[g["slot"] == dominant_slot]
        spells.append({
            "slot":           int(dominant_slot),
            "period":         int(period),
            "start_s":        float(g_dom["t"].min()),
            "end_s":          float(g_dom["t"].max()),
            "n_events":       len(g_dom),
            "slot_agreement": float(n_dom / len(g)),
        })

    # Merge consecutive spells if they're on the same slot (e.g. full-match
    # player whose slot was inferred identically in H1 and H2)
    merged: list[dict] = []
    for sp in spells:
        if merged and merged[-1]["slot"] == sp["slot"]:
            merged[-1]["end_s"]    = sp["end_s"]
            merged[-1]["n_events"] += sp["n_events"]
            merged[-1]["period_span_end"] = sp["period"]
        else:
            merged.append({**sp, "period_span_start": sp["period"],
                                  "period_span_end":   sp["period"]})
    return merged


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------

def _profile_for_slot(game_id: int, team_key: str, slot: int) -> dict:
    """Position + physical profile inherited from whoever the slot belongs to
    in the hand-curated PLAYER_DB.  If unknown, returns DEFAULT_PROFILE."""
    rec = get_player(game_id, team_key, str(slot))
    # `get_player` always returns *something*; we strip identifying fields
    return {
        "position":      rec.get("position", "?"),
        "v_max":         float(rec.get("v_max", DEFAULT_PROFILE["v_max"])),
        "a_max":         float(rec.get("a_max", DEFAULT_PROFILE["a_max"])),
        "reaction_time": float(rec.get("reaction_time",
                                       DEFAULT_PROFILE["reaction_time"])),
    }


def _existing_identity(game_id: int, team_key: str, slot: int) -> dict | None:
    """If PLAYER_DB has a real identity for this slot, return its name fields
    so we don't overwrite real player names with generated ones."""
    rec = get_player(game_id, team_key, str(slot))
    full = rec.get("full_name", "")
    if (full.startswith("Bench")
            or full.startswith(team_key.title() + " #")
            or "(Team " in full):
        return None  # placeholder, not a real identity
    return {
        "display_name": rec.get("full_name", ""),
        "short_name":   rec.get("name", ""),
    }


# ---------------------------------------------------------------------------
# Roster assembly
# ---------------------------------------------------------------------------

def build_team_roster(game_id: int, side_label: str, team_key: str,
                      tracking: pd.DataFrame, events: pd.DataFrame) -> dict:
    """
    side_label : "Home" | "Away"  (matches events["Team"])
    team_key   : "home" | "away"  (matches PLAYER_DB)
    """
    side_events = events[events["Team"] == side_label]
    slots = _tracking_slot_numbers(tracking, side_label)
    team_name = PLAYER_DB.get(game_id, {}).get(team_key, {}).get(
        "team_name", f"{side_label} Team")

    players: list[dict] = []
    for pid, evs in side_events.groupby("From"):
        spells = _build_spells(evs, side_label, tracking, slots)
        if not spells:
            continue

        first_evt_s = float(evs["Start Time [s]"].min())
        last_evt_s  = float(evs["Start Time [s]"].max())

        # The "primary" slot is the one with the most events — used for
        # inheriting the position/physical profile for this player.
        primary_slot = max(spells, key=lambda s: s["n_events"])["slot"]

        # Preserve real identities (e.g. Chelsea / Man City for Game 1)
        identity = _existing_identity(game_id, team_key, primary_slot)
        is_starter = any(sp["start_s"] < 60.0 for sp in spells)

        if identity is not None and is_starter:
            display_name = identity["display_name"]
            short_name   = identity["short_name"]
            generated    = False
        else:
            first, last = _gen_name(f"{game_id}:{team_key}:{pid}")
            display_name = f"{first} {last}"
            short_name   = last
            generated    = True

        profile = _profile_for_slot(game_id, team_key, primary_slot)

        players.append({
            "player_id":     pid,                     # "Player 8"
            "display_name":  display_name,
            "short_name":    short_name,
            "name_source":   "generated" if generated else "curated",
            "position":      profile["position"],
            "v_max":         profile["v_max"],
            "a_max":         profile["a_max"],
            "reaction_time": profile["reaction_time"],
            "primary_slot":  primary_slot,
            "first_event_s": first_evt_s,
            "last_event_s":  last_evt_s,
            "n_events":      len(evs),
            "spells":        spells,
        })

    # Sort by primary slot for stable presentation
    players.sort(key=lambda p: (p["primary_slot"], p["first_event_s"]))

    return {"team_name": team_name, "players": players}


def _load_game(game_id: int):
    """Plain CSV read of a processed Metrica game. Local equivalent of the prototype's
    ``replay.load_game`` (a matplotlib demo script that is not ported)."""
    home = pd.read_csv(f"{DATA_DIR}/Game_{game_id}_tracking_home.csv", index_col="Frame")
    away = pd.read_csv(f"{DATA_DIR}/Game_{game_id}_tracking_away.csv", index_col="Frame")
    events = pd.read_csv(f"{DATA_DIR}/Game_{game_id}_events.csv")
    return home, away, events


def build_game_roster(game_id: int) -> dict:
    print(f"Loading Game {game_id} …")
    home, away, events = _load_game(game_id)
    p1_end = float(events[events["Period"] == 1]["Start Time [s]"].max())
    p2_start_series = events[events["Period"] == 2]["Start Time [s]"]
    p2_start = float(p2_start_series.min()) if not p2_start_series.empty else None
    match_end = float(events["Start Time [s]"].max())

    home_roster = build_team_roster(game_id, "Home", "home", home, events)
    away_roster = build_team_roster(game_id, "Away", "away", away, events)

    return {
        "game_id":           game_id,
        "period_1_end_s":    p1_end,
        "period_2_start_s":  p2_start,
        "match_end_s":       match_end,
        "teams": {
            "Home": home_roster,
            "Away": away_roster,
        },
    }


# ---------------------------------------------------------------------------
# Second Spectrum roster (persistent player_id — no slot inference needed)
# ---------------------------------------------------------------------------
#
# Metrica's roster has to *infer* the slot<->human mapping because tracking
# slots are reused after substitutions and events key off a separate Player ID.
# Second Spectrum ships a persistent `player_id` (Opta id) AND a stable jersey
# per human, so we can build the roster directly from metadata + tracking
# presence, skipping the fragile spatial slot matching entirely.

def _ss_team_roster(side_label: str, players: list[dict], tracking: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    for p in players:
        shirt = p.get("shirt")
        if not shirt or shirt == "?":
            continue
        col_x = f"{side_label}_{shirt}_x"
        if col_x not in tracking.columns:
            continue
        present = tracking[col_x].notna()
        if not present.any():
            continue

        present_rows = tracking.loc[present]
        start_s = float(present_rows["Time [s]"].iloc[0])
        end_s = float(present_rows["Time [s]"].iloc[-1])
        periods = sorted(int(x) for x in present_rows["Period"].unique())
        slot = int(shirt) if str(shirt).isdigit() else shirt

        name = p.get("name") or f"#{shirt}"
        spell = {
            "slot": slot,
            "period": periods[0],
            "start_s": start_s,
            "end_s": end_s,
            "n_events": 0,
            "slot_agreement": 1.0,  # persistent id => no ambiguity
            "period_span_start": periods[0],
            "period_span_end": periods[-1],
        }
        out.append({
            "player_id": p.get("player_id") or f"{side_label}-{shirt}",
            "display_name": name,
            "short_name": name.split()[-1] if name else str(shirt),
            "name_source": "secondspectrum",
            "position": p.get("position_type") or "?",
            "v_max": DEFAULT_PROFILE["v_max"],
            "a_max": DEFAULT_PROFILE["a_max"],
            "reaction_time": DEFAULT_PROFILE["reaction_time"],
            "primary_slot": slot,
            "first_event_s": start_s,
            "last_event_s": end_s,
            "n_events": 0,
            "spells": [spell],
        })

    out.sort(key=lambda r: (
        r["primary_slot"] if isinstance(r["primary_slot"], int) else 1_000,
        r["first_event_s"],
    ))
    return out


def _period_bounds(tracking: pd.DataFrame) -> tuple[float, float | None, float]:
    p1 = tracking.loc[tracking["Period"] == 1, "Time [s]"]
    p2 = tracking.loc[tracking["Period"] == 2, "Time [s]"]
    p1_end = float(p1.max()) if not p1.empty else 0.0
    p2_start = float(p2.min()) if not p2.empty else None
    match_end = float(tracking["Time [s]"].max())
    return p1_end, p2_start, match_end


def build_secondspectrum_roster(game_id: int, meta: dict,
                                home: pd.DataFrame, away: pd.DataFrame) -> dict:
    """Build a roster directly from Second Spectrum metadata + tracking presence."""
    p1_end, p2_start, match_end = _period_bounds(home)
    return {
        "game_id": game_id,
        "provider": "secondspectrum",
        "period_1_end_s": p1_end,
        "period_2_start_s": p2_start,
        "match_end_s": match_end,
        "teams": {
            "Home": {
                "team_name": meta.get("team_a_name", "Home Team"),
                "players": _ss_team_roster("Home", meta.get("team_a_players", []), home),
            },
            "Away": {
                "team_name": meta.get("team_b_name", "Away Team"),
                "players": _ss_team_roster("Away", meta.get("team_b_players", []), away),
            },
        },
    }


def write_secondspectrum_roster(game_id: int, meta: dict,
                                home: pd.DataFrame, away: pd.DataFrame) -> str:
    roster = build_secondspectrum_roster(game_id, meta, home, away)
    path = roster_path(game_id)
    with open(path, "w") as f:
        json.dump(roster, f, indent=2)
    return path


def roster_path(game_id: int) -> str:
    return os.path.join(DATA_DIR, f"Game_{game_id}_roster.json")


# ---------------------------------------------------------------------------
# SkillCorner roster (persistent player_id — same pattern as Second Spectrum)
# ---------------------------------------------------------------------------
#
# SkillCorner ships a persistent `player_id` (SC integer ID) AND a stable
# jersey number per human from the lineup in match.json. Roster building
# follows the Second Spectrum path: presence in tracking data + lineup
# metadata, no slot inference required.

def _sc_team_roster(
    side_label: str,
    sc_players: list[dict],
    tracking: pd.DataFrame,
) -> list[dict]:
    """
    Build a roster list for one team from SkillCorner players + tracking.

    Parameters
    ----------
    side_label  : "Home" or "Away"
    sc_players  : player list from load_skillcorner_meta() for this team
    tracking    : wide tracking DataFrame (home_df or away_df)
    """
    out: list[dict] = []
    for p in sc_players:
        shirt = p.get("shirt")
        if not shirt or shirt == "?":
            continue
        col_x = f"{side_label}_{shirt}_x"
        if col_x not in tracking.columns:
            continue
        present = tracking[col_x].notna()
        if not present.any():
            continue

        present_rows = tracking.loc[present]
        start_s = float(present_rows["Time [s]"].iloc[0])
        end_s   = float(present_rows["Time [s]"].iloc[-1])
        periods = sorted(int(x) for x in present_rows["Period"].unique())
        slot    = int(shirt) if str(shirt).isdigit() else shirt

        name       = p.get("name") or f"#{shirt}"
        short_name = p.get("short_name") or name.split()[-1]
        position   = p.get("position_type") or "?"

        spell = {
            "slot":              slot,
            "period":            periods[0],
            "start_s":           start_s,
            "end_s":             end_s,
            "n_events":          0,
            "slot_agreement":    1.0,
            "period_span_start": periods[0],
            "period_span_end":   periods[-1],
        }
        out.append({
            "player_id":    str(p["player_id"]),
            "display_name": name,
            "short_name":   short_name,
            "name_source":  "skillcorner",
            "position":     position,
            "v_max":        DEFAULT_PROFILE["v_max"],
            "a_max":        DEFAULT_PROFILE["a_max"],
            "reaction_time": DEFAULT_PROFILE["reaction_time"],
            "primary_slot": slot,
            "first_event_s": start_s,
            "last_event_s":  end_s,
            "n_events":      0,
            "spells":        [spell],
        })

    out.sort(key=lambda r: (
        r["primary_slot"] if isinstance(r["primary_slot"], int) else 1_000,
        r["first_event_s"],
    ))
    return out


def build_skillcorner_roster(
    match_id: int,
    meta: dict,
    home: pd.DataFrame,
    away: pd.DataFrame,
) -> dict:
    """
    Build a PLAIQ roster from SkillCorner lineup + tracking presence.

    Parameters
    ----------
    match_id : SkillCorner match ID
    meta     : output of load_skillcorner_meta()
    home     : wide home tracking DataFrame (output of SkillCornerAdapter.load)
    away     : wide away tracking DataFrame

    Returns
    -------
    Roster dict compatible with player_db.load_sc_roster() / list_sc_players()
    """
    p1_end, p2_start, match_end = _period_bounds(home)
    return {
        "match_id":          match_id,
        "provider":          "skillcorner",
        "period_1_end_s":    p1_end,
        "period_2_start_s":  p2_start,
        "match_end_s":       match_end,
        "teams": {
            "Home": {
                "team_name": meta.get("home_team_name", "Home Team"),
                "players":   _sc_team_roster("Home", meta.get("home_players", []), home),
            },
            "Away": {
                "team_name": meta.get("away_team_name", "Away Team"),
                "players":   _sc_team_roster("Away", meta.get("away_players", []), away),
            },
        },
    }


def write_skillcorner_roster(
    match_id: int,
    meta: dict,
    home: pd.DataFrame,
    away: pd.DataFrame,
) -> str:
    """Write SC_{match_id}_roster.json and return the path."""
    roster = build_skillcorner_roster(match_id, meta, home, away)
    path = sc_roster_path(match_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(roster, f, indent=2)
    return path


def sc_roster_path(match_id: int) -> str:
    return os.path.join(DATA_DIR, f"SC_{match_id}_roster.json")


def sb_roster_path(match_id: int) -> str:
    return os.path.join(DATA_DIR, f"SB_Match_{match_id}_roster.json")


# ---------------------------------------------------------------------------
# StatsBomb roster (from lineups — real names, positions, jersey numbers)
# ---------------------------------------------------------------------------
#
# StatsBomb lineups provide persistent player_id (Opta / StatsBomb ID), real
# names, jersey numbers, height, and weight. Physical profile defaults
# (v_max, a_max, reaction_time) fall back to DEFAULT_PROFILE until calibrated.

def build_statsbomb_roster(match_id: int, lineups: dict, meta: dict) -> dict:
    """
    Build a PLAIQ roster from StatsBomb lineups.

    Parameters
    ----------
    match_id  : StatsBomb match ID
    lineups   : dict from statsbomb_io.load_sb_lineups() — keys "home", "away",
                "home_team_name", "away_team_name"
    meta      : dict from statsbomb_io.load_sb_match_meta()

    Returns
    -------
    roster dict compatible with player_db.load_roster() / list_players()
    """
    def _make_player(p: dict, side_label: str, jersey: int) -> dict:
        name = p.get("nickname") or p.get("name") or f"#{jersey}"
        short = name.split()[-1] if name else str(jersey)
        return {
            "player_id":    str(p.get("player_id") or f"{side_label}-{jersey}"),
            "display_name": name,
            "short_name":   short,
            "name_source":  "statsbomb",
            "position":     p.get("position") or "?",
            "v_max":        DEFAULT_PROFILE["v_max"],
            "a_max":        DEFAULT_PROFILE["a_max"],
            "reaction_time": DEFAULT_PROFILE["reaction_time"],
            "height_cm":    p.get("height_cm"),
            "weight_kg":    p.get("weight_kg"),
            "primary_slot": jersey,
            "first_event_s": 0.0,
            "last_event_s":  5400.0,  # full match placeholder
            "n_events":      0,
            "spells": [{
                "slot":              jersey,
                "period":            1,
                "start_s":           0.0,
                "end_s":             5400.0,
                "n_events":          0,
                "slot_agreement":    1.0,
                "period_span_start": 1,
                "period_span_end":   2,
            }],
        }

    home_players = [
        _make_player(p, "Home", p.get("jersey_number", i + 1))
        for i, p in enumerate(lineups.get("home", []))
    ]
    away_players = [
        _make_player(p, "Away", p.get("jersey_number", i + 1))
        for i, p in enumerate(lineups.get("away", []))
    ]

    return {
        "match_id":          match_id,
        "provider":          "statsbomb",
        "period_1_end_s":    2700.0,   # placeholder — no tracking to derive exact time
        "period_2_start_s":  2700.0,
        "match_end_s":       5400.0,
        "teams": {
            "Home": {
                "team_name": lineups.get("home_team_name") or meta.get("home_team", "Home"),
                "players":   home_players,
            },
            "Away": {
                "team_name": lineups.get("away_team_name") or meta.get("away_team", "Away"),
                "players":   away_players,
            },
        },
    }


def write_statsbomb_roster(match_id: int, lineups: dict, meta: dict) -> str:
    roster = build_statsbomb_roster(match_id, lineups, meta)
    path = sb_roster_path(match_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(roster, f, indent=2)
    return path


def write_roster(game_id: int) -> str:
    roster = build_game_roster(game_id)
    path = roster_path(game_id)
    with open(path, "w") as f:
        json.dump(roster, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build per-game player rosters.")
    ap.add_argument("--game", type=int, default=None,
                    help="Build for a single game; default = all available.")
    args = ap.parse_args()

    if args.game is not None:
        games = [args.game]
    else:
        games = []
        for gid in range(1, 50):
            evts_path = os.path.join(DATA_DIR, f"Game_{gid}_events.csv")
            if os.path.exists(evts_path):
                games.append(gid)

    for gid in games:
        path = write_roster(gid)
        print(f"  → wrote {path}")

    print("Done.")


if __name__ == "__main__":
    main()
