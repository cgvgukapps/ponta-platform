"""
citations.py — resolving the AI layer's citations against the semantic tables.

Stage D (``ponta-blueprint/context-states-task.md``, deliverable 4; ``ponta-blueprint/ai-layer.md``, "every claim
cites"). Pure functions the AI layer's verifier calls; nothing here reads a model or a file.

- An **event citation** is ``[event_id, timestamp_s, confidence_score]`` plus ``match_id``; a **state citation** (an
  episode) is ``[state_id, start_s, end_s]`` plus ``match_id``. Each resolves only when the row exists in that match and
  its times agree within ``TIME_TOLERANCE_S``. What fills the confidence slot for single-source matches is still an
  open decision, so it is not checked here.
- **Quoted numbers** (field-bound since 25 November 2026, review AI-02, SE-04, GC-02): every number in a sentence about
  a cited row must equal, as the sentence rounds it, the field its own words name ("6 passes" the pass count, "3.8 s" a
  time in seconds, "40%" a percentage or a share, a number after "Choice" a Choice field), never just any field of the
  row. Scale ends (0, 1, 1.5, 10) pass only inside the scale's own phrase ("on the 1 to 10 scale"). Ordinals, glued
  units and clock times are read; numbers written as words, and numbers no word ties to a field, are rejected.
  ``check_claims`` is the stricter form the AI layer should use: structured claims, each naming its field.
- Times that are not finite never resolve (review SE-04, AI-13).
"""

from __future__ import annotations

import json
import math
import numbers
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: How far a cited time may be from the row's time (s): one context grid step and a little rounding.
TIME_TOLERANCE_S = 0.25
#: The ends of the published scales: quotable only inside the scale's own phrase ("on the 1 to 10 scale").
SCALE_CONSTANTS = (0.0, 1.0, 1.5, 10.0)


@dataclass(frozen=True)
class Resolution:
    ok: bool
    reason: str | None = None
    #: The worst Data check state behind a resolved profile citation, which the AI must disclose (review AI-05).
    data_check: str | None = None


def _finite(*xs) -> bool:
    try:
        return all(x is not None and math.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


def resolve_event_citation(events: pd.DataFrame, match_id: str, event_id: str, timestamp_s: float) -> Resolution:
    """An event citation resolves when the event exists in the match and its time agrees (both times finite)."""
    hit = events.loc[(events["match_id"] == match_id) & (events["event_id"] == event_id)]
    if hit.empty:
        return Resolution(False, "no such event in this match")
    row_t = hit["timestamp_s"].iloc[0]
    if not _finite(row_t, timestamp_s):
        return Resolution(False, "the event or the cited time has no time")
    if abs(float(row_t) - float(timestamp_s)) > TIME_TOLERANCE_S:
        return Resolution(False, "the cited time does not match the event")
    return Resolution(True)


def resolve_state_citation(states: pd.DataFrame, match_id: str, state_id: str, start_s: float, end_s: float) -> Resolution:
    """A state citation resolves when the episode exists in the match and both its times agree (all finite)."""
    hit = states.loc[(states["match_id"] == match_id) & (states["state_id"] == state_id)]
    if hit.empty:
        return Resolution(False, "no such episode in this match")
    r = hit.iloc[0]
    if not _finite(r["start_s"], r["end_s"], start_s, end_s):
        return Resolution(False, "the episode or the cited times have no time")
    if abs(float(r["start_s"]) - float(start_s)) > TIME_TOLERANCE_S or abs(float(r["end_s"]) - float(end_s)) > TIME_TOLERANCE_S:
        return Resolution(False, "the cited times do not match the episode")
    return Resolution(True)


def _is_number(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool | np.bool_) and math.isfinite(float(v))


def _row_fields(row: dict) -> dict[str, float]:
    """The row's numeric fields by name, numpy types included; a JSON object's numbers as ``field.key``."""
    out: dict[str, float] = {}
    for k, v in row.items():
        if _is_number(v):
            out[str(k)] = float(v)
        elif isinstance(v, str) and v.startswith("{"):
            try:
                for kk, x in json.loads(v).items():
                    if _is_number(x):
                        out[f"{k}.{kk}"] = float(x)
            except (ValueError, AttributeError):
                pass
    return out


#: Words that name a family of fields (lower case): the field-name tokens a number next to them may match.
_SYNONYMS = {"peaked": "max", "peak": "max", "highest": "max", "maximum": "max", "median": "median", "typical": "median",
             "broke": "breaking", "break": "breaking", "breaks": "breaking", "engaged": "engagements", "engaging": "engaging",
             "engagement": "engagements", "run": "runs", "pass": "passes", "completed": "completed", "defenders": "defenders",
             "units": "units", "lines": "line", "players": "players", "observed": "observed", "minute": "minute"}
#: Named scores: only their own fields.
_NAMED = {"gain": ("dq_score", "gain_v2_score", "gain_median"),
          "choice": ("dq_v2_score", "choice_median", "adjusted_choice", "adjusted_choice_median")}
_STOP = frozenset(["a", "an", "the", "of", "at", "on", "to", "and", "or", "was", "were", "is", "are", "with", "for", "by", "from", "it", "its", "this", "that", "their", "there", "than", "then", "about", "around", "only", "just", "over", "under", "after", "before", "during", "ball", "carrier", "time", "times", "home", "away"])
_SECONDS = frozenset({"s", "sec", "secs", "second", "seconds"})
_METRES = frozenset({"m", "metre", "metres", "meter", "meters"})
_NUMBER_WORDS = frozenset(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred"])
# A number ends before a letter, a colon or a decimal part; a full stop after it ends the sentence ("was 22.").
_TOKEN = re.compile(r"(?<![\w.:])(-?\d+(?:\.\d+)?)(%|st|nd|rd|th|m|s)?(?![\w:]|\.\d)")
_CLOCK = re.compile(r"(?<![\w.:])(\d{1,3}):(\d{2})(?![\w:]|\.\d)")
_SCALE = re.compile(r"\b(?:0|1) to (?:1\.5|10|1)\b")
_WORD = re.compile(r"[A-Za-z%]+")
_SENTENCE = re.compile(r"[.;!?](?!\d)")


def _word_matches(w: str, field: str) -> bool:
    stem = _SYNONYMS.get(w, w)
    for t in re.split(r"[_.]", field.lower()):
        if t == stem or (len(stem) >= 4 and len(t) >= 4 and (t.startswith(stem) or stem.startswith(t))):
            return True
    return False


def _candidates(words: list[str], fields: dict[str, float], unit: str | None = None) -> set[str]:
    """
    The fields the words name. A named score (Gain, Choice) or a unit (s, m, %) restricts the set; within it the field
    matching most of the words wins, and among those the one with the fewest other name parts ("6 passes" is
    ``passes``, not ``passes_completed``). Empty when no word names a field.
    """
    ws = [w.lower() for w in words]
    pool: set[str] | None = None
    # The number's own unit (glued, or the word right after it): s, m or %.
    if unit in _SECONDS:
        pool = {f for f in fields if f.endswith("_s")}
    elif unit in _METRES:
        pool = {f for f in fields if f.endswith("_m")}
    elif unit == "%":
        pool = {f for f in fields if f.endswith("_pct") or "share" in f or "probability" in f}
    for w in ws:
        if w in _NAMED:
            named = {f for f in _NAMED[w] if f in fields}
            pool = named if pool is None else (pool & named or pool)
    plain = [w for w in ws if w not in _STOP and w not in _NAMED and w not in _SECONDS and w not in _METRES and w != "%"]
    scope = pool if pool is not None else set(fields)
    scored = {f: sum(_word_matches(w, f) for w in plain) for f in scope}
    top = max(scored.values(), default=0)
    if top == 0:
        return set(pool) if pool is not None else set()
    best = [f for f, v in scored.items() if v == top]
    extra = {f: len([t for t in re.split(r"[_.]", f.lower()) if not any(_word_matches(w, t) for w in plain)]) for f in best}
    least = min(extra.values())
    return {f for f in best if extra[f] == least}


def _matches(q: str, digits: int, field: str, fields: dict[str, float], percent: bool) -> bool:
    v = fields[field]
    if percent and not field.endswith("_pct"):
        v = v * 100.0  # a share or probability quoted as a percentage
    return round(v, digits) == float(q)


def check_quoted_numbers(text: str, row: dict | pd.Series) -> list[str]:
    """
    The numbers in ``text`` that do not match the field their words name in ``row`` (as the text rounds them), with
    numbers written as words and numbers no word ties to a field. Empty when every number checks out.
    """
    row = dict(row)
    fields = _row_fields(row)
    minute = str(row.get("minute_display") or "")
    if (lead := re.match(r"\d+", minute)) and "minute_display" not in fields:
        fields["minute_display"] = float(lead.group())  # "34'" or "33:51": the minute
    bad: list[str] = []
    words = _WORD.findall(text)
    for i, w in enumerate(words):
        # A count written as a word next to what it counts ("seven passes") cannot be checked: write the numeral.
        if w.lower() in _NUMBER_WORDS and _candidates(words[i + 1:i + 3], fields):
            bad.append(w)
    scale_spans = [m.span() for m in _SCALE.finditer(text)]
    clock_spans = []
    for m in _CLOCK.finditer(text):
        clock_spans.append(m.span())
        if m.group(0) != minute.strip("'"):
            bad.append(m.group(0))
    bounds = [0] + [m.end() for m in _SENTENCE.finditer(text)] + [len(text)]
    for m in _TOKEN.finditer(text):
        a, b = m.span()
        if any(x <= a and b <= y for x, y in clock_spans):
            continue
        if any(x <= a and b <= y for x, y in scale_spans):
            continue  # a scale's own end, inside "1 to 10" and the like
        q, unit = m.group(1), (m.group(2) or "")
        digits = len(q.split(".")[1]) if "." in q else 0
        s0 = max(x for x in bounds if x <= a)
        s1 = min(x for x in bounds if x >= b)
        # The words of the number's own clause first ("6 passes, 4 completed": 4 is "completed"), then its sentence.
        c1 = text.find(",", b, s1)
        c0 = text.rfind(",", s0, a)
        after = _WORD.findall(text[b:c1 if c1 >= 0 else s1])[:3]
        before = _WORD.findall(text[(c0 + 1) if c0 >= 0 else s0:a])[-8:]
        if not after and not before:
            before = _WORD.findall(text[s0:a])[-8:]
        percent = unit == "%"
        if not unit and after and after[0].lower() in _SECONDS | _METRES and text[b:b + 1] == " ":
            unit, after = after[0].lower(), after[1:]  # "3.8 s"
        if unit in ("st", "nd", "rd", "th"):
            cand = {f for f in fields if "minute" in f}  # "the 34th minute"
        elif unit:
            cand = _candidates(after, fields, unit) or _candidates(before, fields, unit)
        else:
            cand = _candidates(after, fields) or _candidates(before, fields)
        if not cand or not any(_matches(q, digits, f, fields, percent) for f in cand):
            bad.append(m.group(0))
    return bad


def check_claims(claims: list[dict], row: dict | pd.Series) -> list[str]:
    """
    Structured claims, the form the AI layer should emit: each ``{"field": name, "value": number}`` (optionally
    ``"percent": True`` for a share quoted as a percentage). Returns a reason per claim whose field is not a numeric
    field of ``row`` or whose value does not equal it as the claim rounds it.
    """
    fields = _row_fields(dict(row))
    out = []
    for c in claims:
        f, val = c.get("field"), c.get("value")
        if f not in fields:
            out.append(f"{f}: not a numeric field of the cited row")
            continue
        if not _is_number(val):
            out.append(f"{f}: the claimed value is not a number")
            continue
        tok = repr(float(val)) if not float(val).is_integer() else str(int(val))
        digits = len(tok.split(".")[1]) if "." in tok else 0
        if not _matches(tok, digits, f, fields, bool(c.get("percent"))):
            out.append(f"{f}: claimed {val}, the row has {fields[f]}")
    return out


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
    from plaiq_engine.profiles import worst_state

    return Resolution(True, data_check=worst_state(usable["data_check"]))
