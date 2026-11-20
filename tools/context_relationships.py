"""
context_relationships.py — P5 0.2.0's relationships against SkillCorner's labels (Stage C, deliverable 2).

On every SkillCorner open-data match (comparison only, founder decision 2), with P1, P2 and P5 run as shipped:

- **carrier pressure** at the start of each SkillCorner player possession against its five pressure levels
  (no, low, medium, high, very high), cut at the Plaiq review's 1v1 cut points (3, 6, 8, 10; measured there, not
  agreed) and scored as a quadratic-weighted kappa, on the even (fit) and odd (held-out) matches;
- **goal-side:** at each SkillCorner on-ball engagement, whether the engaging defender is nearer its own goal than
  the ball (P5's goal-side test for that player) against SkillCorner's ``goal_side_start``;
- **distributions** of the fields with no reference (free attackers, overload at the ball, marking distance,
  offside line), for the founder's review of the proposed radii;
- P1 to P5's runtime.

Reads the data outside the repository and writes JSON outside it; no names are read.

    python tools/context_relationships.py --data ~/ponta-reference-data/skillcorner-opendata \
        --cache /tmp/ponta-context/sc-cache --out /tmp/ponta-context/relationships.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.line_breaks import LineBreakPackage
from plaiq_engine.context.possession import PossessionPackage, ball_on_grid
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.structure import StructurePackage, team_frames
from plaiq_engine.context.thresholds import values
from plaiq_engine.context.zones import ZonesPackage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_reference_set import _kappa, _load

LEVELS = ("no_pressure", "low_pressure", "medium_pressure", "high_pressure", "very_high_pressure")
CUTS = (3, 6, 8, 10)  # the Plaiq review's 1v1 cut points to SkillCorner's five levels (measured, not agreed)


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def weighted_kappa(a: np.ndarray, b: np.ndarray, k: int = 5) -> float | None:
    if not len(a):
        return None
    o = np.zeros((k, k))
    np.add.at(o, (a, b), 1)
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return round(float(1 - (w * o).sum() / (w * e).sum()), 4)


def one(match_dir: Path, cache: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    m = _load(match_dir, cache)
    t0 = time.perf_counter()
    res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), LineBreakPackage()]).run(m, config=values())
    runtime = round(time.perf_counter() - t0, 1)
    g = res.frames
    nf = g["native_frame"].to_numpy(np.int64)
    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    home = meta["home_team"]["id"]
    pid = {p["id"]: f"{'home' if p['team_id'] == home else 'away'}_{p['number']}" for p in meta["players"]}
    ev = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", low_memory=False,
                     usecols=["event_type", "frame_start", "team_id", "player_id", "overall_pressure_start", "goal_side_start"])
    # Pressure at each player possession's start.
    pp = ev.loc[(ev["event_type"] == "player_possession") & ev["overall_pressure_start"].isin(LEVELS)]
    k = np.clip(np.searchsorted(nf, pp["frame_start"].to_numpy(np.int64)), 0, len(nf) - 1)
    press = pd.DataFrame({"match": match_dir.name, "sc": pp["overall_pressure_start"].map(LEVELS.index).to_numpy(),
                          "ponta": g["p5_carrier_pressure"].to_numpy(dtype=float, na_value=np.nan)[k]})
    # Goal-side of the engaging defender at each on-ball engagement.
    p2 = res.outputs["p2"]
    tf = team_frames(m, g, p2.segments["orientation"])
    sign = {(int(r.period), r.team): r.attack_sign for r in p2.segments["orientation"].itertuples()}
    ball = ball_on_grid(m, g)[["ball_x", "ball_y"]].to_numpy(float)
    per = g["period"].to_numpy()
    L = m.pitch.length_m
    ob = ev.loc[(ev["event_type"] == "on_ball_engagement") & ev["goal_side_start"].notna()]
    rows = []
    for r in ob.itertuples():
        who = pid.get(r.player_id)
        if who is None:
            continue
        t = who.split("_")[0]
        if who not in tf[t].ids:
            continue
        i = int(np.clip(np.searchsorted(nf, int(r.frame_start)), 0, len(nf) - 1))
        dep = tf[t].depth[i, tf[t].ids.index(who)]
        b_d = sign.get((int(per[i]), t), 1) * ball[i, 0] + L / 2
        if not (np.isfinite(dep) and np.isfinite(b_d)):
            continue
        rows.append({"match": match_dir.name, "sc": str(r.goal_side_start).lower() == "true", "ponta": bool(dep < b_d)})
    gs = pd.DataFrame(rows)
    held = g.loc[g["p5_carrier_pressure"].notna()]
    dist = {c: {q: round(float(held[c].astype(float).quantile(q)), 2) for q in (0.1, 0.5, 0.9)}
            for c in ("p5_free_attackers", "p5_overload_at_ball", "p5_marking_distance_m", "p5_offside_line_m", "p5_goal_side_defenders")}
    return press, gs, {"runtime_s": runtime, "distributions": dist, "share_with_relationships": round(float(len(held) / len(g)), 4)}


def score_pressure(d: pd.DataFrame) -> dict:
    d = d.dropna(subset=["ponta"])
    lvl = np.searchsorted(np.array(CUTS), d["ponta"].to_numpy(), side="left")  # below 3: no ... 10: very high
    lvl = np.clip(lvl, 0, 4)
    return {"possessions": int(len(d)), "weighted_kappa": weighted_kappa(d["sc"].to_numpy(int), lvl),
            "spearman": round(float(pd.Series(d["ponta"].to_numpy()).corr(pd.Series(d["sc"].to_numpy()), method="spearman")), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--matches", nargs="*")
    args = ap.parse_args()
    for p in (args.data, args.cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    dirs = [d for d in sorted((args.data / "matches").iterdir()) if (d / f"{d.name}_dynamic_events.csv").exists()]
    if args.matches:
        dirs = [d for d in dirs if d.name in args.matches]
    presses, sides, per_match = [], [], {}
    for n, d in enumerate(dirs):
        p, s, info = one(d, args.cache)
        p["set"] = s["set"] = "fit" if n % 2 == 0 else "test"
        presses.append(p)
        sides.append(s)
        per_match[d.name] = info
        print(d.name, info["runtime_s"], "s", flush=True)
    P, S = pd.concat(presses, ignore_index=True), pd.concat(sides, ignore_index=True)
    out = {"pressure": {h: score_pressure(P.loc[P["set"] == h]) for h in ("fit", "test")},
           "goal_side": {h: {"engagements": int((S["set"] == h).sum()),
                             **_kappa(S.loc[S["set"] == h, "sc"].astype(str).to_numpy(), S.loc[S["set"] == h, "ponta"].astype(str).to_numpy())}
                         for h in ("fit", "test")},
           "runtime_s": {"median": float(np.median([v["runtime_s"] for v in per_match.values()])),
                         "max": float(max(v["runtime_s"] for v in per_match.values()))},
           "per_match": per_match}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("pressure", "goal_side", "runtime_s")}, indent=1))


if __name__ == "__main__":
    main()
