"""
statsbomb/identity.py — Identity assignment for StatsBomb 360 freeze-frame dots.

StatsBomb open data provides positional snapshots (freeze frames) with only
teammate/opponent/actor/keeper flags — no player IDs.  This module assigns
lineup names to freeze-frame dots using a five-tier confidence cascade:

    Tier 1  Actor flag      → event["From"]                           STRONG
    Tier 2  Pass target     → "To" + nearest dot at pass End X/Y      STRONG
    Tier 3  Keepers         → keeper flag + lineup GKs by team        STRONG
    Tier 4  Cross-event     → Hungarian nearest-neighbour tracking    MEDIUM
              matching        between consecutive FF within possession
    Tier 5  Lineup fill     → remaining dots matched to unassigned    WEAK
              (side only)    lineup players on the correct team side

Each enriched player dict gains:
    player_name : str | None
    jersey      : int | None
    confidence  : "strong" | "medium" | "weak" | None

Notes
-----
- ~17 players visible per snapshot on average; typically 5 remain None.
- Off-screen players (not in FF) can never be assigned.
- This is a best-effort heuristic — downstream consumers (sb_dq.py) must
  handle None identities gracefully.
"""
from __future__ import annotations

import math
import copy
from typing import Optional

import pandas as pd


# Maximum distance (metres) to accept a spatial anchor match (Tier 2)
_TIER2_MAX_M = 3.0

# Maximum distance (metres) for cross-event Hungarian matching (Tier 4)
_TIER4_MAX_M = 10.0


def _dist(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _closest_dot(
    target_xy: list[float],
    dots: list[dict],
    max_dist: float = float("inf"),
) -> Optional[int]:
    """Return index of closest unassigned dot within max_dist, or None."""
    best_i, best_d = None, float("inf")
    for i, d in enumerate(dots):
        if d.get("player_name") is not None:
            continue
        loc = d.get("location", [0.0, 0.0])
        dd = _dist(target_xy, loc)
        if dd < best_d and dd <= max_dist:
            best_i, best_d = i, dd
    return best_i


def _hungarian_assign(
    sources: list[dict],
    targets: list[dict],
    max_dist: float,
) -> list[tuple[int, int]]:
    """
    Greedy nearest-neighbour assignment from source dots to target (unassigned)
    dots.  Returns list of (source_idx, target_idx) pairs.

    A full Hungarian algorithm would be optimal but greedy is sufficient here
    (~17 dots, large spatial separation between players).
    """
    used_targets: set[int] = set()
    assignments: list[tuple[int, int]] = []

    dist_matrix = []
    for si, s in enumerate(sources):
        row = []
        for ti, t in enumerate(targets):
            if t.get("player_name") is not None:
                row.append(float("inf"))
            else:
                row.append(_dist(s["location"], t["location"]))
        dist_matrix.append(row)

    n_s = len(sources)
    n_t = len(targets)

    for _ in range(min(n_s, n_t)):
        best_si, best_ti, best_d = None, None, float("inf")
        for si in range(n_s):
            if any(si == a[0] for a in assignments):
                continue
            for ti in range(n_t):
                if ti in used_targets:
                    continue
                d = dist_matrix[si][ti]
                if d < best_d and d <= max_dist:
                    best_si, best_ti, best_d = si, ti, d
        if best_si is None:
            break
        assignments.append((best_si, best_ti))
        used_targets.add(best_ti)

    return assignments


def resolve_identities(
    ff_players: list[dict],
    event: dict | pd.Series,
    lineup: dict,
    prev_event: Optional[dict | pd.Series] = None,
    prev_ff_identified: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Assign player names to freeze-frame dots using the 5-tier cascade.

    Parameters
    ----------
    ff_players          : list of freeze-frame player dicts (from statsbomb_io)
                          Each: {"teammate": bool, "actor": bool, "keeper": bool,
                                 "location": [x_plaiq, y_plaiq]}
    event               : current event dict or pd.Series with at minimum
                          "Type", "From", "To", "End X", "End Y", "Team"
    lineup              : {"home": [...], "away": [...]}
                          Each player: {"name": str, "jersey_number": int,
                                        "position": str | None}
    prev_event          : previous event dict (optional, for Tier 4)
    prev_ff_identified  : previous event's identified FF (optional, for Tier 4)

    Returns
    -------
    List of enriched dicts (deep copy) with added keys:
        player_name : str | None
        jersey      : int | None
        confidence  : "strong" | "medium" | "weak" | None
    """
    dots = copy.deepcopy(ff_players)
    for d in dots:
        d.setdefault("player_name", None)
        d.setdefault("jersey", None)
        d.setdefault("confidence", None)

    ev_from = _safe_str(event.get("From") if hasattr(event, "get") else event["From"])
    ev_to   = _safe_str(event.get("To")   if hasattr(event, "get") else event.get("To"))
    ev_type = _safe_str(event.get("Type") if hasattr(event, "get") else event["Type"])
    ev_team = _safe_str(event.get("Team") if hasattr(event, "get") else event.get("Team"))
    end_x   = event.get("End X")   if hasattr(event, "get") else event.get("End X")
    end_y   = event.get("End Y")   if hasattr(event, "get") else event.get("End Y")

    home_players = lineup.get("home", [])
    away_players = lineup.get("away", [])

    def _is_home_team() -> bool:
        return ev_team == "Home"

    teammate_players = home_players if _is_home_team() else away_players
    opponent_players = away_players if _is_home_team() else home_players

    # ---- Tier 1: Actor flag → event["From"] --------------------------------
    for d in dots:
        if d.get("actor") and ev_from:
            p = _find_player_by_name(ev_from, teammate_players)
            d["player_name"] = ev_from
            d["jersey"]      = p["jersey_number"] if p else None
            d["confidence"]  = "strong"
            break  # only one actor per event

    # ---- Tier 2: Pass target → nearest dot at pass End X/Y -----------------
    if ev_type == "PASS" and ev_to:
        if pd.notna(end_x) and pd.notna(end_y):
            end_xy = [float(end_x), float(end_y)]
            # Receiver is an opponent from the passer's view OR a teammate
            # (StatsBomb marks the receiver as "teammate" to the passer's team)
            # Find closest dot near pass destination that is unassigned
            best_i = None
            best_d = float("inf")
            for i, d in enumerate(dots):
                if d["player_name"] is not None:
                    continue
                dd = _dist(end_xy, d["location"])
                if dd < best_d and dd <= _TIER2_MAX_M:
                    best_i, best_d = i, dd
            if best_i is not None:
                # Determine which lineup the receiver is from
                p = _find_player_by_name(ev_to, home_players) or \
                    _find_player_by_name(ev_to, away_players)
                dots[best_i]["player_name"] = ev_to
                dots[best_i]["jersey"]      = p["jersey_number"] if p else None
                dots[best_i]["confidence"]  = "strong"

    # ---- Tier 3: Keeper flag → lineup GKs by team side ---------------------
    home_gks = [p for p in home_players if _is_gk(p)]
    away_gks = [p for p in away_players if _is_gk(p)]

    for d in dots:
        if d.get("keeper") and d["player_name"] is None:
            on_home = d.get("teammate") if _is_home_team() else not d.get("teammate")
            gk_pool = home_gks if on_home else away_gks
            # Assign closest GK by position to this dot
            if gk_pool:
                gk = gk_pool[0]
                d["player_name"] = gk["name"]
                d["jersey"]      = gk.get("jersey_number")
                d["confidence"]  = "strong"

    # ---- Tier 4: Cross-event Hungarian matching ----------------------------
    if prev_ff_identified:
        prev_named_tm = [d for d in prev_ff_identified
                         if d.get("player_name") and d.get("teammate")]
        prev_named_op = [d for d in prev_ff_identified
                         if d.get("player_name") and not d.get("teammate")
                         and not d.get("actor")]

        curr_unnamed_tm = [d for d in dots
                            if d["player_name"] is None and d.get("teammate")]
        curr_unnamed_op = [d for d in dots
                            if d["player_name"] is None and not d.get("teammate")
                            and not d.get("actor")]

        for prev_pool, curr_pool in [
            (prev_named_tm, curr_unnamed_tm),
            (prev_named_op, curr_unnamed_op),
        ]:
            pairs = _hungarian_assign(prev_pool, curr_pool, _TIER4_MAX_M)
            for si, ti in pairs:
                curr_pool[ti]["player_name"] = prev_pool[si]["player_name"]
                curr_pool[ti]["jersey"]      = prev_pool[si].get("jersey")
                curr_pool[ti]["confidence"]  = "medium"

    # ---- Tier 5: Lineup fill by side (remaining unassigned) ----------------
    assigned_names: set[str] = {d["player_name"] for d in dots
                                 if d["player_name"] is not None}

    unassigned_tm_lineup = [p for p in teammate_players
                             if p["name"] not in assigned_names]
    unassigned_op_lineup = [p for p in opponent_players
                             if p["name"] not in assigned_names]

    for d in dots:
        if d["player_name"] is not None:
            continue
        if d.get("teammate") or d.get("actor"):
            pool = unassigned_tm_lineup
        else:
            pool = unassigned_op_lineup
        if pool:
            p = pool.pop(0)
            d["player_name"] = p["name"]
            d["jersey"]      = p.get("jersey_number")
            d["confidence"]  = "weak"

    return dots


def identified_by_confidence(
    dots: list[dict],
) -> dict[str, list[dict]]:
    """
    Partition identified dots by confidence tier.

    Returns
    -------
    {"strong": [...], "medium": [...], "weak": [...], "none": [...]}
    """
    result: dict[str, list] = {"strong": [], "medium": [], "weak": [], "none": []}
    for d in dots:
        key = d.get("confidence") or "none"
        result[key].append(d)
    return result


def identity_summary(dots: list[dict]) -> dict:
    """Return count of dots at each confidence tier."""
    parts = identified_by_confidence(dots)
    return {k: len(v) for k, v in parts.items()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_str(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def _find_player_by_name(name: str, pool: list[dict]) -> Optional[dict]:
    name_l = name.lower()
    for p in pool:
        if p.get("name", "").lower() == name_l:
            return p
    return None


def _is_gk(player: dict) -> bool:
    pos = str(player.get("position") or "").upper()
    return "GOALKEEPER" in pos or pos == "GK"
