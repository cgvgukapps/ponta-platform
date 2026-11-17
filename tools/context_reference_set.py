"""
context_reference_set.py — Stage B against SkillCorner's open data (Gate 2a, does it generalise).

The Gate 2a bars were measured on one SkillCorner match. This runs P1 to P4 on every match in a
local copy of SkillCorner's open data (MIT licence, kept **outside the repository**) and reports,
per match and across the set:

- P2 roles against SkillCorner's player roles (position groups, weighted by playing time);
- P2 formation changes per team, state and half;
- P4 in-possession phases against SkillCorner's phases, as Cohen's kappa per half, with two
  references: spans built from the dynamic events (the method behind the Gate 2a figure) and
  SkillCorner's own phases-of-play file;
- the Data check state of each Stage B entry, and runtime.

SkillCorner's labels are comparison data only (founder decision 2). No player names are read.
Loaded matches are cached as pickles in ``--cache``; the JSON report is written to ``--out``.

    python tools/context_reference_set.py --data ~/ponta-reference-data/skillcorner-opendata \
        --cache /tmp/ponta-context/sc-cache --out /tmp/ponta-context/reference_set.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.match import load_skillcorner_match
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.phase import PhasePackage
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.context.zones import ZonesPackage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_phase_sweep import agreement
from context_references import sc_phase_file, sc_phase_spans
from context_structure_measure import formations_by_half, role_agreement

#: The Gate 2a bars (agreed by the founder on 17 November 2026).
BARS = {"roles_grouped_tw": 0.75, "roles_cm_merged_tw": 0.85, "formation_changes_max": 2, "phase_kappa_each_half": 0.55}


def _inside(path: Path) -> bool:
    root = repo_root()
    return path.resolve() == root or root in path.resolve().parents


def _load(match_dir: Path, cache: Path):
    p = cache / f"skillcorner_{match_dir.name}.pkl"
    if p.exists():
        with open(p, "rb") as fh:
            return pickle.load(fh)
    m = load_skillcorner_match(str(match_dir), int(match_dir.name))
    cache.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as fh:
        pickle.dump(m, fh)
    return m


def _pairs(ref: pd.DataFrame, frames: pd.DataFrame, period: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference and Ponta phase where both give the same team in possession, with the period."""
    team = frames["p4_team"].astype(object).to_numpy()
    both = ref["sc_team"].notna().to_numpy() & (team == ref["sc_team"].to_numpy()) & frames["p4_phase"].notna().to_numpy()
    return (ref["sc_phase"].to_numpy()[both].astype(str), frames["p4_phase"].astype(object).to_numpy()[both].astype(str),
            period[both])


def _kappa(a: np.ndarray, b: np.ndarray) -> dict:
    if not len(a):
        return {"points": 0}
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in set(a) | set(b))
    return {"points": int(len(a)), "accuracy": round(po, 4), "kappa": round((po - pe) / (1 - pe), 4) if pe < 1 else None}


def one(match_dir: Path, cache: Path) -> tuple[dict, dict]:
    m = _load(match_dir, cache)
    t0 = time.perf_counter()
    res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), PhasePackage()]).run(m)
    seconds = time.perf_counter() - t0
    g, p2, p4 = res.frames, res.outputs["p2"], res.outputs["p4"]
    per = g["period"].to_numpy()
    ra = role_agreement(p2.segments["roles"], p2.segments["formations"], match_dir)
    roles = {**ra["all"], "pairs": ra["pairs"]}
    forms = formations_by_half(p2.segments["formations"])
    refs = {"event_spans": sc_phase_spans(g, match_dir), "phase_file": sc_phase_file(g, match_dir)}
    phases, raw = {}, {}
    for name, ref in refs.items():
        phases[name] = {"all": agreement(ref, p4.frames), "half_1": agreement(ref, p4.frames, per == 1),
                        "half_2": agreement(ref, p4.frames, per == 2), "reference_coverage": round(float(ref["sc_phase"].notna().mean()), 4)}
        raw[name] = _pairs(ref, p4.frames, per)
    k = phases["event_spans"]
    entry = {
        "runtime_s": round(seconds, 1), "grid_points": int(len(g)),
        "orientation_consistent": p2.measurements["orientation_consistent"],
        "shape_coverage_in_play": p2.measurements["shape_coverage_in_play"],
        "roles": roles, "formation_changes_max": max(v["changes"] for v in forms.values()), "formations": forms,
        "phases": phases,
        "entries": {f"{e.package}:{e.metric}": e.state for e in res.entries if e.package in ("p2", "p3", "p4")},
    }
    entry["bars"] = {
        "roles_grouped_tw": roles["grouped_time_weighted"] >= BARS["roles_grouped_tw"],
        "roles_cm_merged_tw": roles["central_mid_merged_time_weighted"] >= BARS["roles_cm_merged_tw"],
        "formation_changes_max": entry["formation_changes_max"] <= BARS["formation_changes_max"],
        "phase_kappa_each_half": all((k[h].get("kappa") or 0) >= BARS["phase_kappa_each_half"] for h in ("half_1", "half_2")),
    }
    return entry, raw


def summarise(report: dict, raw: dict) -> dict:
    ms = list(report)

    def dist(values: list[float]) -> dict:
        v = np.array(values, dtype=float)
        return {"median": round(float(np.median(v)), 4), "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}

    out = {"matches": len(ms),
           "roles_grouped_tw": dist([report[k]["roles"]["grouped_time_weighted"] for k in ms]),
           "roles_cm_merged_tw": dist([report[k]["roles"]["central_mid_merged_time_weighted"] for k in ms]),
           "formation_changes_max": dist([report[k]["formation_changes_max"] for k in ms]),
           "runtime_s": dist([report[k]["runtime_s"] for k in ms]),
           "shape_coverage_in_play": dist([report[k]["shape_coverage_in_play"] for k in ms]),
           "bars_met": {b: sum(report[k]["bars"][b] for k in ms) for b in BARS},
           "all_bars_met": sum(all(report[k]["bars"].values()) for k in ms)}
    for name in ("event_spans", "phase_file"):
        a = np.concatenate([raw[k][name][0] for k in ms])
        b = np.concatenate([raw[k][name][1] for k in ms])
        p = np.concatenate([raw[k][name][2] for k in ms])
        out[f"phases_{name}"] = {
            "pooled": _kappa(a, b), "pooled_half_1": _kappa(a[p == 1], b[p == 1]), "pooled_half_2": _kappa(a[p == 2], b[p == 2]),
            "kappa_lower_half_per_match": dist([min(report[k]["phases"][name][h]["kappa"] for h in ("half_1", "half_2")) for k in ms]),
            "confusion": {r: {c: int(v) for c, v in row.items() if v}
                          for r, row in pd.crosstab(pd.Series(a, name="skillcorner"), pd.Series(b, name="ponta")).iterrows()},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="local copy of SkillCorner's open data (has matches/<id>/)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.data, args.cache, args.out):
        if _inside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    report, raw = {}, {}
    for d in sorted((args.data / "matches").iterdir()):
        if not (d / f"{d.name}_phases_of_play.csv").exists() or not (d / f"{d.name}_tracking_extrapolated.jsonl").exists():
            continue
        report[d.name], raw[d.name] = one(d, args.cache)
        e = report[d.name]
        print(d.name, json.dumps({"roles": e["roles"]["grouped_time_weighted"], "forms": e["formation_changes_max"],
                                  "kappa_h1_h2": [e["phases"]["event_spans"][h].get("kappa") for h in ("half_1", "half_2")],
                                  "s": e["runtime_s"]}), flush=True)
    out = {"bars": BARS, "summary": summarise(report, raw), "matches": report}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["summary"], indent=1, default=str))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
