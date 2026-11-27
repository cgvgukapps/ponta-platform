"""
Build the payload for a shareable match page: the data check, coverage and notable
moments for one match, as one JSON file (ponta-blueprint/frontend-design-brief-v2.md, 5.3).

This is the concierge deliverable before any AI layer exists: everything on the page
comes from the deterministic engines. Notable moments are selected by rules over the
semantic events (ai-layer.md, "deterministic extraction"); no language model is involved.
Optional founder notes can be added from a text file and are labelled as such.

    python tools/match_page_payload.py skillcorner --out /tmp/ponta-page/skillcorner.json
    python tools/match_page_payload.py metrica --notes notes.txt --out /tmp/ponta-page/metrica.json

The output is derived from open sample data whose redistribution terms are unverified,
so it is written outside the repository (the script refuses otherwise). Field names
follow the semantic event schema. Players are pseudonymous references only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.match import (
    NormalisedMatch,
    load_metrica_epts_match,
    load_skillcorner_match,
)
from matrix_engine.parity import parity_for_match
from matrix_engine.sample_data import repo_root, sample_data_dir
from plaiq_engine.match_analysis import MatchAnalysis
from plaiq_engine.roster.profiles import roster_profiles
from plaiq_engine.semantic_builder import SCHEMA_VERSION, SemanticBuilder
from plaiq_engine.teams import TeamRegistry

#: 1.1 (19 November 2026): version 1's descriptive band names (semantic events schema 1.5); group ids follow them.
PAYLOAD_VERSION = "1.2"  # 1.2 (27 November 2026): semantic schema 2.1; movement moments from movement_values; every lost pass
#: Moments per group on the page.
PER_GROUP = 4
#: 1v1 pressure (engine scale 1 to 10) treated as "under pressure" for selection only.
#: Stated on the page; not a product threshold.
SELECTION_PRESSURE = 7.0


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


MOMENT_FIELDS = (
    "event_id", "event_idx", "timestamp_s", "minute_display", "period", "actor_player_ref", "actor_team",
    "action_type", "action_outcome", "target_player_ref", "inferred_intended_ref", "inference_confidence",
    "dq_score", "dq_band", "chosen_option_pov", "best_option_player_ref", "best_option_pov", "xt_delta",
    "movement_value", "opposition_pressure", "pitch_zone", "ball_detected", "players_observed",
)


def _snapshot(match: NormalisedMatch, analysis: MatchAnalysis, event_idx: int) -> dict | None:
    """Positions of every observed player and the ball at the event's start frame, plus the event's end point."""
    ev = analysis.events.loc[event_idx]
    if pd.isna(ev.get("Start Frame")):
        return None
    frame = int(ev["Start Frame"])
    players = []
    for team, wide in (("home", analysis.home), ("away", analysis.away)):
        if frame not in wide.index:
            continue
        row = wide.loc[frame]
        prefix = team.capitalize()
        for j in match.slots.get(team, ()):
            x, y = row.get(f"{prefix}_{j}_x"), row.get(f"{prefix}_{j}_y")
            if pd.notna(x) and pd.notna(y):
                players.append({"ref": f"{team}_{j}", "team": team, "x": round(float(x), 2), "y": round(float(y), 2)})
    bx = analysis.away.loc[frame, "ball_x"] if frame in analysis.away.index else np.nan
    by = analysis.away.loc[frame, "ball_y"] if frame in analysis.away.index else np.nan
    return {
        "frame": frame,
        "players": players,
        "ball": None if pd.isna(bx) or pd.isna(by) else {"x": round(float(bx), 2), "y": round(float(by), 2)},
        "start": None if pd.isna(ev.get("Start X")) else {"x": round(float(ev["Start X"]), 2), "y": round(float(ev["Start Y"]), 2)},
        "end": None if pd.isna(ev.get("End X")) else {"x": round(float(ev["End X"]), 2), "y": round(float(ev["End Y"]), 2)},
    }


def _movement_rows(builder: SemanticBuilder, sem: pd.DataFrame) -> pd.DataFrame:
    """Schema 2.1: movement values are their own table; each moment is the source event seen from the moving player."""
    mv = builder.movement_values().sort_values(["movement_value", "timestamp_s"], ascending=[False, True], kind="stable").head(PER_GROUP)
    src = sem.drop_duplicates("event_id").set_index("event_id")
    rows = []
    for r in mv.itertuples():
        e = src.loc[r.source_event_id].to_dict() if r.source_event_id in src.index else {}
        rows.append({**{k: e.get(k) for k in MOMENT_FIELDS}, "event_id": r.event_id, "event_idx": e.get("event_idx"),
                     "actor_player_ref": r.player_ref, "actor_team": r.team, "action_type": "movement_value", "action_outcome": None,
                     "target_player_ref": None, "inferred_intended_ref": None, "inference_confidence": None, "dq_score": None,
                     "dq_band": None, "chosen_option_pov": None, "best_option_player_ref": None, "best_option_pov": None,
                     "xt_delta": r.xt_delta, "movement_value": r.movement_value, "opposition_pressure": r.opposition_pressure,
                     "pitch_zone": None})
    return pd.DataFrame(rows, columns=list(MOMENT_FIELDS))


def build_payload(match: NormalisedMatch, *, notes: list[str] | None = None, sample: bool = True) -> dict:
    hj, aj = list(match.slots["home"]), list(match.slots["away"])
    ph, pa = roster_profiles(match.provider, match.match_id, hj, aj)
    analysis = MatchAnalysis(match, profiles_home=ph, profiles_away=pa)
    parity = parity_for_match(match)
    builder = SemanticBuilder(analysis, parity, teams=TeamRegistry.for_reference())
    sem = builder.semantic_events()

    decisions = sem.loc[sem["action_type"].isin(["pass", "carry"])]
    scored = decisions["dq_score"].notna()
    causes = {
        "goalkeeper": int(decisions["not_scored_reason"].fillna("").str.startswith("goalkeeper").sum()),
        "set_piece": int(decisions["not_scored_reason"].fillna("").str.startswith("set piece").sum()),
        "lost_pass": int(decisions["not_scored_reason"].fillna("").str.startswith("lost pass").sum()),
        "no_score_from_engine": int((~scored & ~decisions["not_scored"].astype(bool)).sum()),
    }
    coverage = {
        "decisions_total": len(decisions),
        "decisions_scored": int(scored.sum()),
        "scored_share": None if not len(decisions) else round(float(scored.mean()), 4),
        "not_scored_causes": causes,
        "ball_detected_share": parity.coverage["ball_detected_share"],
        "player_observation_share": parity.coverage["player_observation_share"],
    }

    def pick(df: pd.DataFrame, by: list[str], ascending: list[bool]) -> pd.DataFrame:
        return df.sort_values(by + ["event_idx"], ascending=ascending + [True], kind="stable").head(PER_GROUP)

    s = sem.loc[sem["dq_score"].notna()]
    groups = [
        {"id": "low_gain_under_pressure", "title": "Lowest-gain decisions under pressure",
         "rule": f"Passes and carries with Low gain or Partial gain under 1v1 pressure of {SELECTION_PRESSURE:g} or more, lowest gain first.",
         "rows": pick(s.loc[(s["opposition_pressure"].fillna(0) >= SELECTION_PRESSURE) & s["dq_band"].isin(["low_gain", "partial_gain"])], ["dq_score"], [True])},
        # Every lost pass, scored or not: since schema 1.9 a lost pass has no Gain score, so filtering on Gain left this
        # group empty (found 27 November 2026, review backlog B).
        {"id": "losses", "title": "Passes that lost the ball",
         "rule": "Failed passes, highest pressure first.",
         "rows": pick(sem.loc[(sem["action_type"] == "pass") & (sem["action_outcome"] == "lost")], ["opposition_pressure"], [False])},
        {"id": "best_progressive", "title": "Best decisions that moved the ball forward in value",
         "rule": "Completed passes and carries with Full gain, largest value change first.",
         "rows": pick(s.loc[(s["dq_band"] == "full_gain") & (s["action_outcome"] == "completed") & (s["xt_delta"].fillna(0) > 0)], ["xt_delta"], [False])},
        {"id": "off_ball_runs", "title": "Most valuable off-ball movement",
         "rule": "Team-mates' movements the engine rewarded at an event, highest movement value first.",
         "rows": _movement_rows(builder, sem)},
    ]
    moment_groups = []
    for g in groups:
        moments = []
        for _, r in g["rows"].iterrows():
            m = {k: _clean(r[k]) for k in MOMENT_FIELDS}
            m["snapshot"] = _snapshot(match, analysis, int(r["event_idx"]))
            moments.append(m)
        moment_groups.append({"id": g["id"], "title": g["title"], "rule": g["rule"], "moments": moments})

    return {
        "payload_version": PAYLOAD_VERSION,
        "schema_version": SCHEMA_VERSION,
        "engine_version": builder.engine_version,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample_data": sample,
        "match": {
            "match_id": builder.match_id, "provider": match.provider, "home": "Home", "away": "Away",
            "duration_min": round(float(match.frames["time_s"].max()) / 60.0, 1), "sample_rate_hz": match.fps,
            "pitch": {"length_m": match.pitch.length_m, "width_m": match.pitch.width_m},
        },
        "data_check": parity.to_dict(),
        "coverage": coverage,
        "moment_groups": moment_groups,
        "notes": notes or [],
        "statement": "Every figure on this page comes from Ponta's deterministic engines. Moments were selected by the rules shown. No language model wrote any part of it.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("match", choices=["metrica", "skillcorner"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--notes", type=Path, help="text file of founder notes, one paragraph per blank-line-separated block")
    args = ap.parse_args()
    out = args.out.resolve()
    if str(out).startswith(str(repo_root())):
        print("ERROR: refusing to write sample-derived output inside the repository", file=sys.stderr)
        return 2
    match = (load_metrica_epts_match(sample_data_dir(), 3) if args.match == "metrica"
             else load_skillcorner_match(os.path.join(sample_data_dir(), "skillcorner", "matches", "1886347"), 1886347))
    notes = [p.strip() for p in args.notes.read_text().split("\n\n") if p.strip()] if args.notes else []
    payload = build_payload(match, notes=notes)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    n = sum(len(g["moments"]) for g in payload["moment_groups"])
    print(f"wrote {out}: {n} moments, coverage {payload['coverage']['scored_share']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
