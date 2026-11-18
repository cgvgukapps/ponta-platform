"""
context_restart_sweep.py — measure the position-based throw-in detector (``context/restarts.py``).

References (comparison only, founder decision 2):
- **SkillCorner:** the restart marker on the first possession after each restart
  (``game_interruption_before``: ``throw_in_for`` / ``_against``, and the same for goal kicks,
  corners and free kicks). All 20 open-data matches.
- **Metrica:** logged throw-ins (``SET PIECE`` / ``THROW IN``), team as logged, Sample Games 1 to 3.
  Metrica does not log goal kicks.

The throw-in settings are swept on half the SkillCorner matches (even positions in id order) and
read on the other half and on Metrica, so they are not tuned to the matches they are judged on. A
detection matches a reference of the same team when the set-up ends between ``SETUP_BEFORE_S``
before it and ``SETUP_AFTER_S`` after it.

It also reports why goal kicks are not detected: the share of each restart type with a tracking
gap (at least 2 s with fewer than 4 players tracked, or a jump in the clock) in the 15 s before.
Reads sample data outside the repository and writes JSON outside it; no names are read.

    python tools/context_restart_sweep.py --cache /tmp/ponta-context/cache \
        --sc-data ~/ponta-reference-data/skillcorner-opendata --sc-cache /tmp/ponta-context/sc-cache \
        --out /tmp/ponta-context/restart_sweep.json
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
from plaiq_engine.context.grid import build_grid
from plaiq_engine.context.possession import ball_on_grid
from plaiq_engine.context.restarts import (
    SETUP_AFTER_S,
    SETUP_BEFORE_S,
    median_speed,
    throw_in_setups,
)
from plaiq_engine.context.structure import OTHER, TEAMS, attack_signs, team_frames
from plaiq_engine.context.thresholds import values

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_measure import load
from context_reference_set import _load

OUTS = (-0.5, 0.0, 0.5, 1.0)
HOLDS = (0.5, 1.0, 2.0, 3.0, 4.0)
STILLS = (float("inf"), 2.0, 2.5)
BALLS = (float("inf"), 5.0)
METRICA_KINDS = {"THROW IN": "throw_in", "CORNER KICK": "corner", "FREE KICK": "free_kick"}


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def prepare(m) -> dict:
    """Team frames, the ball in each team's frame and the median player speed, on the 5 Hz grid."""
    g = build_grid(m, 5.0)
    signs = attack_signs(m, g)
    tf = team_frames(m, g, signs)
    sign_of = {(int(r.period), r.team): r.attack_sign for r in signs.itertuples()}
    per = g["period"].to_numpy()
    ts = g["frame_ts"].to_numpy(float)
    b = ball_on_grid(m, g)
    bx, by = b["ball_x"].to_numpy(float), b["ball_y"].to_numpy(float)
    ball = {}
    for t in TEAMS:
        s = np.array([sign_of.get((int(p), t), 1 if t == "home" else -1) for p in per], float)
        ball[t] = (s * bx + m.pitch.length_m / 2, s * by)
    speed = median_speed([tf[t].depth for t in TEAMS], [tf[t].width for t in TEAMS], ts, per)
    return {"g": g, "ts": ts, "period": per, "tf": tf, "ball": ball, "speed": speed, "L": m.pitch.length_m, "W": m.pitch.width_m}


def sc_restarts(p: dict, match_dir: Path) -> list[tuple[str, str, float]]:
    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    home = meta["home_team"]["id"]
    ev = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", usecols=["frame_start", "team_id", "game_interruption_before"],
                     low_memory=False).dropna(subset=["game_interruption_before"]).sort_values("frame_start")
    nf, ts = p["g"]["native_frame"].to_numpy(), p["ts"]
    out: list[tuple[str, str, float]] = []
    for r in ev.itertuples():
        kind, _, side = r.game_interruption_before.rpartition("_")
        if kind not in ("throw_in", "goal_kick", "corner", "free_kick"):
            continue
        k = int(np.clip(np.searchsorted(nf, r.frame_start), 0, len(nf) - 1))
        own = "home" if r.team_id == home else "away"
        team = own if side == "for" else OTHER[own]
        if not any(x[0] == kind and x[1] == team and abs(ts[k] - x[2]) < 10 for x in out):
            out.append((kind, team, float(ts[k])))
    return out


def metrica_restarts(m) -> list[tuple[str, str, float]]:
    ev = m.events
    sp = ev.loc[ev["Type"].astype(str) == "SET PIECE"]
    f0 = int(m.frames["frame"].min())
    out = []
    for t, sub, f in zip(sp["Team"], sp["Subtype"].astype(str), sp["Start Frame"], strict=True):
        kind = next((v for k, v in METRICA_KINDS.items() if k in sub), None)
        if kind:
            out.append((kind, str(t).lower(), float((f - f0) / m.fps)))
    return out


def detect(p: dict, out_m: float, hold: float, still: float, ball_m: float) -> list[tuple[str, float]]:
    found = []
    for t in TEAMS:
        f, o = p["tf"][t], p["tf"][OTHER[t]]
        for _, e in throw_in_setups(f.depth, f.width, o.depth, o.width, *p["ball"][t], p["speed"], p["ts"], p["period"], p["L"], p["W"],
                                    out_m=out_m, hold_s=hold, still_mps=still, ball_m=ball_m):
            found.append((t, e))
    return found


def match_events(found: list[tuple[str, float]], refs: list[tuple[str, str, float]], kind: str = "throw_in") -> dict:
    ref = [(t, x) for k, t, x in refs if k == kind]
    used, tp = set(), 0
    for team, e in found:
        for j, (rt, r) in enumerate(ref):
            if j not in used and rt == team and r - SETUP_BEFORE_S <= e <= r + SETUP_AFTER_S:
                used.add(j)
                tp += 1
                break
    return {"reference": len(ref), "found": len(found), "true": tp}


def total(rows: list[dict]) -> dict:
    t = {k: sum(r[k] for r in rows) for k in ("reference", "found", "true")}
    t["recall"] = round(t["true"] / t["reference"], 3) if t["reference"] else None
    t["precision"] = round(t["true"] / t["found"], 3) if t["found"] else None
    t["f1"] = round(2 * t["true"] / (t["reference"] + t["found"]), 3) if t["reference"] + t["found"] else None
    return t


def tracking_gaps(p: dict, refs: list[tuple[str, str, float]]) -> dict[str, list[bool]]:
    """Per restart type: whether the 15 s before each restart hold a tracking gap."""
    ts = p["ts"]
    seen = sum(np.isfinite(p["tf"][t].depth).sum(axis=1) for t in TEAMS)
    out: dict[str, list[bool]] = {}
    for kind, _, x in refs:
        k = int(np.searchsorted(ts, x))
        sl = slice(max(0, k - 75), k)
        low = (seen[sl] < 4).sum() / 5 >= 2
        jump = bool(np.diff(ts[sl]).max() > 1.0) if k - max(0, k - 75) > 1 else False
        out.setdefault(kind, []).append(bool(low or jump))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--sc-data", type=Path, required=True)
    ap.add_argument("--sc-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for path in (args.cache, args.sc_data, args.sc_cache, args.out):
        if not _outside(path):
            sys.exit(f"refusing to read or write sample data inside the repository: {path}")
    sets: dict[str, dict[str, dict]] = {"fit": {}, "test": {}, "metrica": {}}
    gaps: dict[str, list[bool]] = {}
    dirs = [d for d in sorted((args.sc_data / "matches").iterdir()) if (d / f"{d.name}_dynamic_events.csv").exists()]
    for n, d in enumerate(dirs):
        p = prepare(_load(d, args.sc_cache))
        p["ref"] = sc_restarts(p, d)
        for kind, v in tracking_gaps(p, p["ref"]).items():
            gaps.setdefault(kind, []).extend(v)
        sets["fit" if n % 2 == 0 else "test"][d.name] = p
        print("skillcorner", d.name, flush=True)
    for key in ("metrica_1", "metrica_2", "metrica_3"):
        m = load(key, args.cache)
        p = prepare(m)
        p["ref"] = metrica_restarts(m)
        sets["metrica"][key] = p
        print(key, flush=True)

    grid = list(itertools.product(OUTS, HOLDS, STILLS, BALLS))
    scores = {c: {name: total([match_events(detect(p, *c), p["ref"]) for p in s.values()]) for name, s in sets.items()} for c in grid}
    best = max(grid, key=lambda c: (scores[c]["fit"]["f1"] or 0, scores[c]["fit"]["precision"] or 0))
    cfg = values()
    current = (cfg["p4_throw_in_out_m"], cfg["p4_throw_in_hold_s"], cfg["p4_throw_in_still_mps"], cfg["p4_throw_in_ball_m"])

    def row(c):
        return {"out_m": c[0], "hold_s": c[1], "still_mps": None if np.isinf(c[2]) else c[2], "ball_m": None if np.isinf(c[3]) else c[3], **scores[c]}

    out = {
        "match_window_s": [-SETUP_BEFORE_S, SETUP_AFTER_S],
        "best_on_fit": row(best),
        "thresholds": row(current) if current in scores else None,
        "top_on_fit": [row(c) for c in sorted(grid, key=lambda c: -(scores[c]["fit"]["f1"] or 0))[:12]],
        "per_match_thresholds": {name: {k: match_events(detect(p, *current), p["ref"]) for k, p in s.items()} for name, s in sets.items()},
        "skillcorner_tracking_gap_before": {k: {"restarts": len(v), "share_with_gap": round(float(np.mean(v)), 3)} for k, v in gaps.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("best_on_fit", "thresholds", "skillcorner_tracking_gap_before")}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
