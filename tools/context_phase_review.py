"""
context_phase_review.py — score P4 rule variants against the founder's replay review and SkillCorner.

The founder judged Ponta's phase on 50 replayed moments (``tools/data/phase_review_2026-11-17.json``,
from ``tools/context_spotcheck.py --replay``), with the re-check of 18 November 2026 added. Rules tuned to 50 moments judged by one person
can overfit, so every variant is also scored against SkillCorner's phases on its 20 open-data
matches (pooled Cohen's kappa by half, against event spans and against the phases-of-play file).
A variant is worth adopting only if it agrees better with the review **and** does not lose
agreement with SkillCorner.

P1 and P2 run once per match; P4 runs once per variant. SkillCorner's labels are comparison data
only (founder decision 2). Reads sample data outside the repository and writes JSON outside it.

    python tools/context_phase_review.py --cache /tmp/ponta-context/cache \
        --sc-data ~/ponta-reference-data/skillcorner-opendata --sc-cache /tmp/ponta-context/sc-cache \
        --out /tmp/ponta-context/phase_review.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.phase import PhasePackage
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import ContextInput, Registry
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.context.thresholds import values

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_measure import load
from context_reference_set import _kappa, _load, _pairs
from context_references import sc_phase_file, sc_phase_spans

REVIEW = Path(__file__).resolve().parent / "data" / "phase_review_2026-11-17.json"
#: Later re-checks of single cases: their accepted labels are added to the review's (never removed).
RECHECKS = (Path(__file__).resolve().parent / "data" / "phase_review_2026-11-18_recheck.json",)


def load_cases() -> list[dict]:
    """The review's cases, with labels accepted on a later re-check added to each case's 'accept'."""
    cases = json.loads(REVIEW.read_text())["cases"]
    extra: dict[str, list[str]] = {}
    for path in RECHECKS:
        for c in json.loads(path.read_text())["cases"]:
            if c["judged_right"]:
                extra.setdefault(c["id"], []).extend(c["accept"])
    for c in cases:
        if c["id"] in extra and c.get("accept"):
            c["accept"] = sorted(set(c["accept"]) | set(extra[c["id"]]))
    return cases
GRID_COLS = ["match_id", "frame_ts", "period", "native_frame", "clock_s", "minute"]

#: The rules before the review (P4 0.1.0) and each refinement on its own, against the adopted
#: set (P4 0.2.0, the thresholds' defaults). Two more refinements were tried and dropped on
#: 17 November 2026 (direct following the ball's flight, set play ending in the own half);
#: their scores are in CONTEXT_STRUCTURE_REPORT.md. P4 0.3.0 adds throw-ins from positions
#: (no regain straight after one); ``adopted_without_throw_ins`` is P4 0.2.0.
BEFORE = {"p4_finish_min_depth_share": 0.0, "p4_build_up_max_depth_share": 0.0, "p4_break_end_depth_share": 0.0,
          "p4_corner_setup_crowd_min": 0.0, "p4_throw_in_hold_s": 0.0}
VARIANTS: dict[str, dict] = {
    "before_review": BEFORE,
    "finish_in_attacking_half_only": BEFORE | {"p4_finish_min_depth_share": 0.5},
    "build_up_in_own_third_only": BEFORE | {"p4_build_up_max_depth_share": 0.333},
    "break_ends_in_final_third_only": BEFORE | {"p4_break_end_depth_share": 0.667},
    "finish_in_final_third_only": BEFORE | {"p4_finish_min_depth_share": 2 / 3},
    "adopted_without_corner_setups": {"p4_corner_setup_crowd_min": 0.0, "p4_throw_in_hold_s": 0.0},
    "adopted_without_throw_ins": {"p4_throw_in_hold_s": 0.0},
    "adopted": {},
}

def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def _p4(m, res, cfg: dict):
    ctx = ContextInput(match=m, grid=res.frames[GRID_COLS], config=cfg, upstream={"p1": res.outputs["p1"], "p2": res.outputs["p2"]})
    return PhasePackage().compute(ctx).frames


def score_review(cases: list[dict], labels: dict[str, str]) -> dict:
    right, changed, rows = 0, 0, []
    for c in cases:
        lab = labels[c["id"]]
        ok = (lab in c["accept"]) if c.get("accept") else (lab not in c.get("reject", []))
        right += ok
        changed += lab != c["ponta_v1"]
        if lab != c["ponta_v1"]:
            rows.append({"id": c["id"], "was": c["ponta_v1"], "now": lab, "judged_right_before": c["judged_right"], "right_now": bool(ok)})
    return {"right": right, "of": len(cases), "changed": changed, "changes": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True, help="pickled regression matches (as context_measure.py)")
    ap.add_argument("--sc-data", type=Path, required=True, help="local copy of SkillCorner's open data")
    ap.add_argument("--sc-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="score only these variants")
    args = ap.parse_args()
    if args.only:
        for name in list(VARIANTS):
            if name not in args.only:
                del VARIANTS[name]
    for p in (args.cache, args.sc_data, args.sc_cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    cases = load_cases()
    cfgs = {name: values(over) for name, over in VARIANTS.items()}
    review_labels: dict[str, dict[str, str]] = {name: {} for name in VARIANTS}
    by_match: dict[str, list[int]] = {}
    for c in cases:
        key, i = c["id"].rsplit("-", 1)
        by_match.setdefault(key, []).append(int(i))
    for key, ids in by_match.items():
        m = load(key, args.cache)
        res = Registry([PossessionPackage(), StructurePackage()]).run(m)
        for name, cfg in cfgs.items():
            ph = _p4(m, res, cfg)["p4_phase"].astype(object).to_numpy()
            for i in ids:
                review_labels[name][f"{key}-{i}"] = str(ph[i])
        print("review", key, flush=True)
    review = {name: score_review(cases, review_labels[name]) for name in VARIANTS}

    pooled: dict[str, dict[str, list]] = {name: {"event_spans": [], "phase_file": []} for name in VARIANTS}
    for d in sorted((args.sc_data / "matches").iterdir()):
        if not (d / f"{d.name}_phases_of_play.csv").exists():
            continue
        m = _load(d, args.sc_cache)
        res = Registry([PossessionPackage(), StructurePackage()]).run(m)
        g = res.frames
        per = g["period"].to_numpy()
        refs = {"event_spans": sc_phase_spans(g, d), "phase_file": sc_phase_file(g, d)}
        for name, cfg in cfgs.items():
            f = _p4(m, res, cfg)
            for r, ref in refs.items():
                pooled[name][r].append(_pairs(ref, f, per))
        print("skillcorner", d.name, flush=True)

    out = {"review_file": REVIEW.name, "variants": {}}
    for name, over in VARIANTS.items():
        sc = {}
        for r, parts in pooled[name].items():
            a = np.concatenate([p[0] for p in parts])
            b = np.concatenate([p[1] for p in parts])
            p = np.concatenate([p[2] for p in parts])
            sc[r] = {"all": _kappa(a, b), "half_1": _kappa(a[p == 1], b[p == 1]), "half_2": _kappa(a[p == 2], b[p == 2])}
        out["variants"][name] = {"overrides": over, "review": review[name], "skillcorner": sc}
        print(f"{name:34s} review {review[name]['right']}/{review[name]['of']} (changed {review[name]['changed']}) | "
              f"SC spans kappa {sc['event_spans']['all']['kappa']} ({sc['event_spans']['half_1']['kappa']}/{sc['event_spans']['half_2']['kappa']}) | "
              f"SC file kappa {sc['phase_file']['all']['kappa']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
