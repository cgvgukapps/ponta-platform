"""
context_states.py — P8 0.1.0 (episodes) against SkillCorner's phases of play, Metrica and the German league (Stage D).

``ponta-blueprint/context-states-task.md``, deliverable 5. Every context package runs as shipped with one shared
``MatchAnalysis``; P8 is then rerun on the same upstream outputs for each variant of the episode rule.

1. **SkillCorner's 20 open-data matches** (comparison only, founder decision 2): its ``phases_of_play`` file is a
   published episode segmentation. Per variant: boundaries found within 1 s and 2 s both ways (each side's starts and
   ends), episodes per match, durations, the share of each side's time the other covers, and phase agreement (κ) on
   grid points where both give the same team.
2. **Metrica's three sample matches** (logged events) and **the German league's seven** (official possession):
   episodes per match, durations, the share of possession time in an episode, end reasons, and the share of logged
   or official possession changes with an episode boundary within 1 s and 2 s.
3. **Checks:** every summary against its own row (``check_quoted_numbers``), no coordinate, cell or name column, P8's
   runtime and the episode table's storage per match.

Fit (even) and held-out (odd) SkillCorner matches are reported apart. Reads data outside the repository and writes
JSON outside it; no names are read.

    python tools/context_states.py --sc-data ~/ponta-reference-data/skillcorner-opendata --sc-cache /tmp/ponta-context/sc-cache \
        --metrica-cache /tmp/ponta-context/cache --idsse ~/ponta-reference-data/idsse --idsse-cache /tmp/ponta-context/idsse-cache \
        --out /tmp/ponta-context/states.json
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.citations import check_quoted_numbers
from plaiq_engine.context.grid import GRID_COLUMNS
from plaiq_engine.context.registry import ContextInput
from plaiq_engine.context.states import StatesPackage
from plaiq_engine.context.thresholds import values
from plaiq_engine.match_analysis import MatchAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_golden import registry
from context_measure import load as load_sample
from context_reference_set import _load as load_sc
from idsse import MATCH_IDS, load_idsse_match

VARIANTS = {
    "base": {},
    "no_formation_split": {"p8_split_formation": 0.0},
    "fifa_phase_split": {"p8_split_fifa_phase": 1.0},
    "min_1s": {"p8_min_state_s": 1.0},
    "min_3s": {"p8_min_state_s": 3.0},
    "min_5s": {"p8_min_state_s": 5.0},
    "drop_short_chains": {"p8_keep_short_chains": 0.0},
}
TOLS = (1.0, 2.0)
SC_PHASE = {"build_up": "build_up", "create": "create", "finish": "finish", "direct": "direct", "quick_break": "quick_break",
            "transition": "transition", "set_play": "set_play", "chaotic": "chaotic"}


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def kappa(a, b) -> float | None:
    a, b = np.asarray(a, object), np.asarray(b, object)
    if not len(a):
        return None
    labels = sorted(set(a) | set(b))
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == x)) * float(np.mean(b == x)) for x in labels)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def near(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    """For each time in ``a``, whether ``b`` has a time within ``tol``."""
    if not len(a) or not len(b):
        return np.zeros(len(a), bool)
    b = np.sort(b)
    k = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), int)
    d = np.minimum(np.abs(b[k] - a), np.abs(b[np.maximum(k - 1, 0)] - a))
    return d <= tol


def run_p8(m, res, cfg: dict):
    pkg = StatesPackage()
    g = res.frames[list(GRID_COLUMNS)]
    t0 = time.perf_counter()
    out = pkg.compute(ContextInput(match=m, grid=g, config=cfg, upstream={k: res.outputs[k] for k in pkg.depends_on}))
    return out, time.perf_counter() - t0


def checks(st: pd.DataFrame) -> dict:
    buf = io.BytesIO()
    st.to_parquet(buf, index=False)
    bad = sum(1 for r in st.to_dict("records") if check_quoted_numbers(r["summary"], r))
    forbidden = [c for c in st.columns if c.endswith(("_x", "_y", "_cell")) or "name" in c]
    return {"summaries_failing_number_check": bad, "forbidden_columns": forbidden, "storage_kb": round(buf.tell() / 1024, 1)}


def sc_scores(name: str, res, st: pd.DataFrame, match_dir: Path) -> dict:
    g = res.frames
    nf, ts = g["native_frame"].to_numpy(float), g["frame_ts"].to_numpy(float)
    meta = json.loads((match_dir / f"{name}_match.json").read_text())
    home = meta["home_team"]["id"]
    ph = pd.read_csv(match_dir / f"{name}_phases_of_play.csv", usecols=["frame_start", "frame_end", "team_in_possession_id", "team_in_possession_phase_type"])
    ph = ph.assign(t0=np.interp(ph["frame_start"], nf, ts), t1=np.interp(ph["frame_end"], nf, ts),
                   team=np.where(ph["team_in_possession_id"] == home, "home", "away"))
    sc_b = np.r_[ph["t0"].to_numpy(), ph["t1"].to_numpy()]
    ours_b = np.r_[st["start_s"].to_numpy(float), st["end_s"].to_numpy(float)] if len(st) else np.array([])
    # Grid labels for time cover and phase agreement.
    n = len(ts)
    sc_team = np.full(n, None, dtype=object)
    sc_phase = np.full(n, None, dtype=object)
    for r in ph.itertuples():
        sel = (ts >= r.t0) & (ts <= r.t1)
        sc_team[sel], sc_phase[sel] = r.team, r.team_in_possession_phase_type
    our_team = np.full(n, None, dtype=object)
    our_phase = np.full(n, None, dtype=object)
    for r in st.itertuples():
        sel = (ts >= r.start_s - 1e-9) & (ts < r.end_s - 1e-9)
        our_team[sel], our_phase[sel] = r.team, r.phase
    both = pd.notna(sc_team) & pd.notna(our_team) & (sc_team == our_team)
    out = {"sc_phases": int(len(ph)), "episodes": int(len(st)),
           "sc_time_covered": float(np.mean(pd.notna(our_team)[pd.notna(sc_team)])) if pd.notna(sc_team).any() else None,
           "our_time_in_sc": float(np.mean(pd.notna(sc_team)[pd.notna(our_team)])) if pd.notna(our_team).any() else None,
           "phase_pairs": (sc_phase[both], our_phase[both]), "durations": st["duration_s"].to_numpy(float),
           "sc_durations": (ph["t1"] - ph["t0"]).to_numpy(float)}
    for tol in TOLS:
        out[f"recall_{tol:g}s"] = (int(near(sc_b, ours_b, tol).sum()), len(sc_b))
        out[f"precision_{tol:g}s"] = (int(near(ours_b, sc_b, tol).sum()), len(ours_b))
    return out


def pool_sc(rows: list[dict]) -> dict:
    if not rows:
        return {}
    a = np.concatenate([r["phase_pairs"][0] for r in rows])
    b = np.concatenate([r["phase_pairs"][1] for r in rows])
    d = np.concatenate([r["durations"] for r in rows])
    sd = np.concatenate([r["sc_durations"] for r in rows])
    out = {"matches": len(rows), "episodes_per_match": round(float(np.mean([r["episodes"] for r in rows])), 1),
           "sc_phases_per_match": round(float(np.mean([r["sc_phases"] for r in rows])), 1),
           "duration_s_p10_p50_p90": [round(float(np.percentile(d, q)), 1) for q in (10, 50, 90)],
           "sc_duration_s_p10_p50_p90": [round(float(np.percentile(sd, q)), 1) for q in (10, 50, 90)],
           "sc_time_covered": round(float(np.mean([r["sc_time_covered"] for r in rows if r["sc_time_covered"] is not None])), 3),
           "our_time_in_sc": round(float(np.mean([r["our_time_in_sc"] for r in rows if r["our_time_in_sc"] is not None])), 3),
           "phase_kappa_same_team": kappa(a, b), "phase_agreement_same_team": round(float(np.mean(a == b)), 3) if len(a) else None}
    for tol in TOLS:
        for k in ("recall", "precision"):
            hit = sum(r[f"{k}_{tol:g}s"][0] for r in rows)
            tot = sum(r[f"{k}_{tol:g}s"][1] for r in rows)
            out[f"boundary_{k}_{tol:g}s"] = round(hit / tot, 3) if tot else None
    return out


def possession_changes_idsse(off, fps: float) -> np.ndarray:
    """Official changes of the team in possession (each side holding the ball in play for at least 1 s), as frames."""
    b = off.ball.sort_values("frame")
    live = b["ball_status"].eq(1).to_numpy() & b["ball_team"].isin(["home", "away"]).to_numpy()
    team = np.where(live, b["ball_team"].astype(object).to_numpy(), None)
    fr = b["frame"].to_numpy(np.int64)
    ch = np.flatnonzero(np.r_[True, team[1:] != team[:-1]])
    spans = [(s, e, team[s]) for s, e in zip(ch, np.r_[ch[1:], len(team)] - 1, strict=True) if team[s] is not None and e - s + 1 >= fps]
    return np.array([fr[c[0]] for a, c in pairwise(spans) if a[2] != c[2]], float)


def other_scores(key: str, m, res, st: pd.DataFrame, change_frames: np.ndarray) -> dict:
    g = res.frames
    nf, ts = g["native_frame"].to_numpy(float), g["frame_ts"].to_numpy(float)
    poss = g["p1_possession"].astype(object).isin(["home", "away"]).to_numpy()
    covered = g["p8_state_id"].notna().to_numpy() if "p8_state_id" in g else np.zeros(len(g), bool)
    ch_t = np.interp(change_frames, nf, ts) if len(change_frames) else np.array([])
    starts = st["start_s"].to_numpy(float) if len(st) else np.array([])
    out = {"episodes": int(len(st)), "duration_s_p10_p50_p90": [round(float(np.percentile(st["duration_s"], q)), 1) for q in (10, 50, 90)] if len(st) else None,
           "possession_time_covered": round(float(covered[poss].mean()), 3) if poss.any() else None,
           "end_reasons": {k: int(v) for k, v in st["end_reason"].value_counts().items()},
           "possession_changes": int(len(ch_t))}
    for tol in TOLS:
        out[f"changes_with_episode_start_{tol:g}s"] = round(float(near(ch_t, starts, tol).mean()), 3) if len(ch_t) else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sc-data", type=Path, required=True)
    ap.add_argument("--sc-cache", type=Path, required=True)
    ap.add_argument("--metrica-cache", type=Path)
    ap.add_argument("--idsse", type=Path)
    ap.add_argument("--idsse-cache", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--skip-sc", action="store_true")
    args = ap.parse_args()
    for p in (args.sc_data, args.sc_cache, args.out, args.metrica_cache, args.idsse, args.idsse_cache):
        if p is not None and not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    cfg = values()
    result: dict = {"variants": list(VARIANTS)}
    runtimes, check_rows = [], []
    if not args.skip_sc:
        dirs = [d for d in sorted((args.sc_data / "matches").iterdir()) if (d / f"{d.name}_phases_of_play.csv").exists()]
        if args.matches:
            dirs = [d for d in dirs if d.name in args.matches]
        per: dict = {v: {"fit": [], "test": []} for v in VARIANTS}
        for n, d in enumerate(dirs):
            m = load_sc(d, args.sc_cache)
            res = registry().run(m, extras={"analysis": MatchAnalysis(m)})
            for v, over in VARIANTS.items():
                out, rt = run_p8(m, res, {**cfg, **over})
                st = out.segments["states"]
                per[v]["fit" if n % 2 == 0 else "test"].append(sc_scores(d.name, res, st, d))
                if v == "base":
                    runtimes.append(rt)
                    check_rows.append(checks(st))
            print(d.name, len(res.outputs["p8"].segments["states"]), "episodes", flush=True)
        result["skillcorner"] = {v: {h: pool_sc(rows) for h, rows in halves.items()} for v, halves in per.items()}
    others = {}
    if args.metrica_cache is not None:
        for key in ("metrica_1", "metrica_2", "metrica_3"):
            m = load_sample(key, args.metrica_cache)
            res = registry().run(m, extras={"analysis": MatchAnalysis(m)})
            st = res.outputs["p8"].segments["states"]
            ev = m.events.dropna(subset=["Start Frame"])
            ev = ev.loc[ev["Team"].astype(str).str.lower().isin(["home", "away"])].sort_values("Start Frame")
            tm = ev["Team"].astype(str).str.lower().to_numpy()
            changes = ev["Start Frame"].to_numpy(float)[np.r_[False, tm[1:] != tm[:-1]]]
            others[key] = other_scores(key, m, res, st, changes)
            check_rows.append(checks(st))
            print(key, len(st), "episodes", flush=True)
    if args.idsse is not None and args.idsse_cache is not None:
        for mid in MATCH_IDS:
            try:
                m, off = load_idsse_match(args.idsse, mid, args.idsse_cache)
                if "End X" not in m.events:  # the league's event table has no end points; Plaiq's event tagging expects the columns
                    m = dataclasses.replace(m, events=m.events.assign(**{"End X": np.nan, "End Y": np.nan}))
                res = registry().run(m, extras={"analysis": MatchAnalysis(m)})
            except Exception as exc:  # report and continue: one league match failing should not stop the others
                others[f"idsse_{mid}"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                print(mid, "failed", exc, flush=True)
                continue
            st = res.outputs["p8"].segments["states"]
            others[f"idsse_{mid}"] = other_scores(mid, m, res, st, possession_changes_idsse(off, float(m.fps)))
            check_rows.append(checks(st))
            print(mid, len(st), "episodes", flush=True)
    result["other_sources"] = others
    result["checks"] = {"summaries_failing_number_check": sum(c["summaries_failing_number_check"] for c in check_rows),
                        "forbidden_columns": sorted({x for c in check_rows for x in c["forbidden_columns"]}),
                        "storage_kb_median": float(np.median([c["storage_kb"] for c in check_rows])) if check_rows else None}
    if runtimes:
        result["p8_runtime_s"] = {"median": round(float(np.median(runtimes)), 2), "max": round(float(np.max(runtimes)), 2)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
