"""
context_corner_sweep.py — measure the position-based corner detector (``context/restarts.py``).

References (comparison only, founder decision 2):
- **SkillCorner:** its dynamic events mark the restart before each possession
  (``game_interruption_before == "corner_for"``). These are used on all 20 open-data matches.
- **Metrica:** logged corner kicks, read from the event feed with the team as logged, on Sample
  Games 1 to 3. (P1 chains are not used: before P1 0.2.0 a corner won at the first touch by the
  defending team was credited to the defenders' chain.)

The detector's settings are swept on half the SkillCorner matches (even positions in id order)
and read on the other half and on Metrica, so they are not tuned to the matches they are judged
on. The current ball-based rule (``phase.corner_at_start`` with the agreed thresholds) is the
baseline. Reads sample data outside the repository and writes JSON outside it; no names read.

    python tools/context_corner_sweep.py --cache /tmp/ponta-context/cache \
        --sc-data ~/ponta-reference-data/skillcorner-opendata --sc-cache /tmp/ponta-context/sc-cache \
        --out /tmp/ponta-context/corner_sweep.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.phase import corner_at_start
from plaiq_engine.context.possession import PossessionPackage, ball_on_grid
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.restarts import _runs, corner_counts
from plaiq_engine.context.structure import OTHER, StructurePackage, team_frames
from plaiq_engine.context.thresholds import values

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_measure import load
from context_reference_set import _load

RADII = (2.0, 3.0, 4.0)
MARGINS = (0.0, 3.0)
CROWDS = (6, 8, 10, 12, 14)
HOLDS = (1.0, 2.0, 3.0, 4.0)
#: A detected set-up matches a reference corner of the same team when it ends this close to it.
EARLY_S, LATE_S = 6.0, 3.0


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def sc_corners(g: pd.DataFrame, match_dir: Path) -> list[tuple[str, float]]:
    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    home = meta["home_team"]["id"]
    ev = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", usecols=["frame_start", "team_id", "game_interruption_before"], low_memory=False)
    # "corner_for" sits on the taking team's first event after the corner, "corner_against" on the
    # defending team's (a corner cleared at the first touch); either marks one corner.
    ev = ev.loc[ev["game_interruption_before"].isin(["corner_for", "corner_against"])].sort_values("frame_start")
    nf, ts = g["native_frame"].to_numpy(), g["frame_ts"].to_numpy(float)
    out: list[tuple[str, float]] = []
    for r in ev.itertuples():
        k = int(np.clip(np.searchsorted(nf, r.frame_start), 0, len(nf) - 1))
        own = "home" if r.team_id == home else "away"
        team = own if r.game_interruption_before == "corner_for" else OTHER[own]
        if not any(t == team and abs(ts[k] - x) < 10 for t, x in out):
            out.append((team, float(ts[k])))
    return out


def logged_corners(m) -> list[tuple[str, float]]:
    """Metrica's logged corner kicks, read from the events (team as logged; time on the grid clock)."""
    ev = m.events
    c = ev.loc[(ev["Type"].astype(str) == "SET PIECE") & ev["Subtype"].astype(str).str.contains("CORNER")]
    f0 = int(m.frames["frame"].min())
    return [(str(t).lower(), float((f - f0) / m.fps)) for t, f in zip(c["Team"], c["Start Frame"], strict=True)]


def match_events(found: list[tuple[str, float]], ref: list[tuple[str, float]]) -> dict:
    used, tp = set(), 0
    for team, e in found:
        for j, (rt, r) in enumerate(ref):
            if j not in used and rt == team and r - EARLY_S <= e <= r + LATE_S:
                used.add(j)
                tp += 1
                break
    return {"reference": len(ref), "found": len(found), "true": tp, "missed": len(ref) - tp, "false_alarms": len(found) - tp}


def prepare(m) -> dict:
    """Per match: counts per (radius, margin) and team, the references, and the baseline detections."""
    res = Registry([PossessionPackage(), StructurePackage()]).run(m)
    g = res.frames
    tf = team_frames(m, g, res.outputs["p2"].segments["orientation"])
    L, W = m.pitch.length_m, m.pitch.width_m
    counts = {(r, mg, t): corner_counts(tf[t].depth, tf[t].width, tf[OTHER[t]].depth, tf[OTHER[t]].width, L, W, r, mg)
              for r in RADII for mg in MARGINS for t in ("home", "away")}
    chains = res.outputs["p1"].segments["chains"]
    cid = res.outputs["p1"].frames["p1_chain_id"].to_numpy(dtype=float, na_value=np.nan)
    poss = res.outputs["p1"].frames["p1_possession"].to_numpy(object)
    ball = ball_on_grid(m, g)
    bx, by = ball["ball_x"].to_numpy(float), ball["ball_y"].to_numpy(float)
    cfg = values()
    base = []
    for c in chains.itertuples():
        idx = np.where((cid == c.chain_id) & (poss == c.team))[0]
        if len(idx) and corner_at_start(bx, by, idx[: max(1, int(round(cfg["p4_corner_window_s"] * 5)))], L, W, cfg["p4_corner_radius_m"]):
            base.append((c.team, float(c.start_ts)))
    return {"ts": g["frame_ts"].to_numpy(float), "period": g["period"].to_numpy(), "counts": counts, "baseline": base, "g": g, "chains": chains}


def detect(p: dict, radius: float, margin: float, crowd: int, hold: float) -> list[tuple[str, float]]:
    out = []
    for t in ("home", "away"):
        a_flag, d_flag, n = p["counts"][(radius, margin, t)]
        for s, e in _runs((a_flag == 1) & (d_flag == 0) & (n >= crowd), p["period"]):
            if p["ts"][e] - p["ts"][s] >= hold:
                out.append((t, float(p["ts"][e])))
    return out


def total(rows: list[dict]) -> dict:
    t = {k: sum(r[k] for r in rows) for k in ("reference", "found", "true", "missed", "false_alarms")}
    t["recall"] = round(t["true"] / t["reference"], 3) if t["reference"] else None
    t["precision"] = round(t["true"] / t["found"], 3) if t["found"] else None
    t["f1"] = round(2 * t["true"] / (t["reference"] + t["found"]), 3) if t["reference"] + t["found"] else None
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--sc-data", type=Path, required=True)
    ap.add_argument("--sc-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.cache, args.sc_data, args.sc_cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    sets: dict[str, dict[str, dict]] = {"fit": {}, "test": {}, "metrica": {}}
    dirs = [d for d in sorted((args.sc_data / "matches").iterdir()) if (d / f"{d.name}_dynamic_events.csv").exists()]
    for n, d in enumerate(dirs):
        p = prepare(_load(d, args.sc_cache))
        p["ref"] = sc_corners(p["g"], d)
        sets["fit" if n % 2 == 0 else "test"][d.name] = p
        print("skillcorner", d.name, len(p["ref"]), "corners", flush=True)
    for key in ("metrica_1", "metrica_2", "metrica_3"):
        m = load(key, args.cache)
        p = prepare(m)
        p["ref"] = logged_corners(m)
        sets["metrica"][key] = p
        print(key, len(p["ref"]), "corners", flush=True)

    grid = list(itertools.product(RADII, MARGINS, CROWDS, HOLDS))
    scores = {}
    for combo in grid:
        scores[combo] = {name: total([match_events(detect(p, *combo), p["ref"]) for p in s.values()]) for name, s in sets.items()}
    best = max(grid, key=lambda c: (scores[c]["fit"]["f1"] or 0, scores[c]["fit"]["precision"] or 0))
    baseline = {name: total([match_events(p["baseline"], p["ref"]) for p in s.values()]) for name, s in sets.items()}
    out = {
        "match_window_s": [-EARLY_S, LATE_S],
        "baseline_ball_rule": baseline,
        "best_on_fit": {"radius_m": best[0], "box_margin_m": best[1], "crowd_min": best[2], "hold_s": best[3], **{k: v for k, v in scores[best].items()}},
        "per_match_best": {name: {k: match_events(detect(p, *best), p["ref"]) for k, p in s.items()} for name, s in sets.items()},
        "top_on_fit": [{"radius_m": c[0], "box_margin_m": c[1], "crowd_min": c[2], "hold_s": c[3], **scores[c]}
                       for c in sorted(grid, key=lambda c: -(scores[c]["fit"]["f1"] or 0))[:12]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"baseline": baseline, "best": out["best_on_fit"]}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
