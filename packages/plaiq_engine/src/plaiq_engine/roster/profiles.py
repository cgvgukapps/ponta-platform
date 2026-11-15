"""
profiles.py — per-player physical profiles from the processed roster files.

The same lookup the prototype app used (``app.load_pov_scores`` for Metrica via
``player_db.get_profiles``, ``app.load_sc_pov_scores_cache`` for SkillCorner), so
decision quality is computed with the same profiles as the golden outputs. Players
without a roster entry get the engine's default profile.
"""

from __future__ import annotations

from plaiq_engine.analytics.pitch_control import DEFAULT_PROFILE
from plaiq_engine.roster.player_db import get_profiles, load_sc_roster

_KEYS = ("v_max", "a_max", "reaction_time")


def roster_profiles(provider: str, match_id: int | str, home_jerseys: list[str], away_jerseys: list[str]) -> tuple[dict, dict]:
    """Return ``(profiles_home, profiles_away)`` for a sample match."""
    if provider == "metrica":
        return get_profiles(int(match_id), "home", home_jerseys), get_profiles(int(match_id), "away", away_jerseys)

    roster = load_sc_roster(int(match_id))

    def side(team_label: str, jerseys: list[str]) -> dict:
        if roster is None:
            return {j: DEFAULT_PROFILE for j in jerseys}
        players = {str(p.get("primary_slot", "")): p for p in roster["teams"].get(team_label, {}).get("players", [])}
        return {j: ({k: players[j].get(k, DEFAULT_PROFILE[k]) for k in _KEYS} if j in players else DEFAULT_PROFILE) for j in jerseys}

    return side("Home", home_jerseys), side("Away", away_jerseys)
