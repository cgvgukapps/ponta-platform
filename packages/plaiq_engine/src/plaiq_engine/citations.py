"""
citations.py — resolving the AI layer's citations against the semantic tables.

Stage D (``ponta-blueprint/context-states-task.md``, deliverable 4; ``ponta-blueprint/ai-layer.md``, "every claim
cites"). Pure functions the AI layer's verifier calls; nothing here reads a model or a file.

- An **event citation** is ``[event_id, timestamp_s, confidence_score]`` plus ``match_id``; a **state citation** (an
  episode) is ``[state_id, start_s, end_s]`` plus ``match_id``. Each resolves only when the row exists in that match and
  its times agree within ``TIME_TOLERANCE_S``. What fills the confidence slot for single-source matches is still an
  open decision, so it is not checked here.
- **Quoted numbers:** every number in a sentence about a cited row must equal one of that row's numeric fields, as
  the sentence rounds it, or be a stated scale constant (``SCALE_CONSTANTS``: the 1 to 10 pressure scale, Gain's 0 to
  1.5 and Choice's 0 to 1).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import pandas as pd

#: How far a cited time may be from the row's time (s): one context grid step and a little rounding.
TIME_TOLERANCE_S = 0.25
#: Numbers a sentence may quote without a field: the ends of the published scales.
SCALE_CONSTANTS = (0.0, 1.0, 1.5, 10.0)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


@dataclass(frozen=True)
class Resolution:
    ok: bool
    reason: str | None = None


def resolve_event_citation(events: pd.DataFrame, match_id: str, event_id: str, timestamp_s: float) -> Resolution:
    """An event citation resolves when the event exists in the match and its time agrees."""
    hit = events.loc[(events["match_id"] == match_id) & (events["event_id"] == event_id)]
    if hit.empty:
        return Resolution(False, "no such event in this match")
    if abs(float(hit["timestamp_s"].iloc[0]) - float(timestamp_s)) > TIME_TOLERANCE_S:
        return Resolution(False, "the cited time does not match the event")
    return Resolution(True)


def resolve_state_citation(states: pd.DataFrame, match_id: str, state_id: str, start_s: float, end_s: float) -> Resolution:
    """A state citation resolves when the episode exists in the match and both its times agree."""
    hit = states.loc[(states["match_id"] == match_id) & (states["state_id"] == state_id)]
    if hit.empty:
        return Resolution(False, "no such episode in this match")
    r = hit.iloc[0]
    if abs(float(r["start_s"]) - float(start_s)) > TIME_TOLERANCE_S or abs(float(r["end_s"]) - float(end_s)) > TIME_TOLERANCE_S:
        return Resolution(False, "the cited times do not match the episode")
    return Resolution(True)


def _row_numbers(row: dict) -> list[float]:
    out: list[float] = []
    for v in row.values():
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, int | float):
            if not math.isnan(v):
                out.append(float(v))
        elif isinstance(v, str) and v.startswith("{"):
            try:
                out.extend(float(x) for x in json.loads(v).values() if isinstance(x, int | float) and not isinstance(x, bool))
            except (ValueError, AttributeError):
                pass
    return out


def check_quoted_numbers(text: str, row: dict | pd.Series) -> list[str]:
    """The numbers in ``text`` that are neither a field of ``row`` (as the text rounds it) nor a scale constant."""
    row = dict(row)
    values = _row_numbers(row) + list(SCALE_CONSTANTS)
    bad = []
    for tok in _NUMBER.findall(text):
        q = float(tok)
        digits = len(tok.split(".")[1]) if "." in tok else 0
        if not any(round(v, digits) == q for v in values):
            bad.append(tok)
    return bad


def resolve_profile_citation(profiles: pd.DataFrame, team_key: str, measure: str, match_ids: list[str]) -> Resolution:
    """
    A claim about a team's tendency cites ``[team_key, measure, match_ids]`` (plus an example episode, resolved apart).
    It resolves when the measure is in the catalogue, every cited match has a usable profile row for the team, the
    profiles share one set of versions, and the matches meet the measure's measured minimum (Stage E).
    """
    from plaiq_engine.profiles import MEASURES, MIN_MATCHES

    if measure not in MEASURES:
        return Resolution(False, "not a measure in the profile catalogue")
    rows = profiles.loc[(profiles["team_key"] == team_key) & (profiles["measure"] == measure) & profiles["match_id"].isin(match_ids)]
    usable = rows.loc[rows["value"].notna() & (rows["data_check"] != "blocked")]
    if set(usable["match_id"]) != set(match_ids):
        return Resolution(False, "a cited match has no usable profile for this team and measure")
    if usable["versions"].nunique() > 1:
        return Resolution(False, "the cited profiles were built from different versions")
    need = MIN_MATCHES.get(measure)
    if need is None:
        return Resolution(False, "the minimum number of matches for this measure is not measured yet")
    if len(set(match_ids)) < need:
        return Resolution(False, f"a pattern needs at least {need} matches for this measure")
    return Resolution(True)
