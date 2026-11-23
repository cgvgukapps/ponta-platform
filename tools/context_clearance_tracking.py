"""
context_clearance_tracking.py — can clearances be read from tracking? Against the German league's logged clearances.

P8 0.2.0 marks an episode ``cleared`` from logged clearances only, and logged clearances are rare. This measures a
tracking-only candidate against the league's own event coding (the professional reference, founder's decision order of
23 November 2026), before anything is built into a package:

- a **kick** is the end of a spell in which one team controls the ball (P1's ``p1_ball_state``), with the ball then
  travelling at ``speed`` m/s or more within 0.4 s;
- it is a **clearance candidate** when the ball was within ``own`` metres of the kicking team's own goal line, gains at
  least ``gain`` metres towards the other goal within 2 s, and the next team to control the ball (within 6 s) is not the
  kicking team (an opponent, a contest, or nobody before the ball goes dead): an unaimed ball out of danger.

Logged clearances (``CLEARANCE``) by the same team within 1 s of a candidate are matches. Reports recall (logged
clearances found) and precision (candidates that are logged clearances) for a small grid of the three values, fitted
on the first four matches and read on the other three. Reads data outside the repository; prints JSON; no names read.

    python tools/context_clearance_tracking.py --idsse ~/ponta-reference-data/idsse --cache /tmp/ponta-context/idsse-cache
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.possession import PossessionPackage, ball_on_grid
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.structure import StructurePackage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idsse import MATCH_IDS, load_idsse_match

GRID = {"speed": (12.0, 15.0, 18.0), "own": (30.0, 35.0, 40.0), "gain": (20.0, 25.0, 30.0)}


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def kicks(m) -> dict:
    """Per kick: time, team, ball depth in the kicking team's frame, peak speed, forward gain, next controlling team."""
    res = Registry([PossessionPackage(), StructurePackage()]).run(m)
    g, f1 = res.frames, res.outputs["p1"].frames
    ts, per = g["frame_ts"].to_numpy(float), g["period"].to_numpy()
    n, L = len(g), m.pitch.length_m
    step = float(np.median(np.diff(ts)))
    bs = f1["p1_ball_state"].astype(object).to_numpy()
    spd = f1["p1_ball_speed_mps"].to_numpy(dtype=float, na_value=np.nan)
    bx = ball_on_grid(m, g)["ball_x"].to_numpy(float)
    sign = {(int(r.period), r.team): r.attack_sign for r in res.outputs["p2"].segments["orientation"].itertuples()}
    ctl = np.array([s[len("controlled_"):] if isinstance(s, str) and s.startswith("controlled_") else None for s in bs], dtype=object)
    w04, w2, w6 = int(round(0.4 / step)), int(round(2.0 / step)), int(round(6.0 / step))
    out = {"t": [], "team": [], "depth": [], "speed": [], "gain": [], "next_same": []}
    for i in range(n - 1):
        t = ctl[i]
        if t is None or ctl[i + 1] == t or per[i + 1] != per[i]:
            continue
        s = sign.get((int(per[i]), t), 1)
        d0 = s * bx[i] + L / 2
        seg = slice(i + 1, min(n, i + 1 + w2))
        ahead = s * bx[seg] + L / 2
        nxt = next((ctl[k] for k in range(i + 1, min(n, i + 1 + w6)) if ctl[k] is not None and per[k] == per[i]), None)
        out["t"].append(ts[i])
        out["team"].append(t)
        out["depth"].append(d0)
        out["speed"].append(np.nanmax(spd[i:min(n, i + 1 + w04)]) if np.isfinite(spd[i:min(n, i + 1 + w04)]).any() else np.nan)
        out["gain"].append((np.nanmax(ahead) - d0) if np.isfinite(ahead).any() and np.isfinite(d0) else np.nan)
        out["next_same"].append(nxt == t)
    nf = g["native_frame"].to_numpy(float)
    ev = m.events.dropna(subset=["Start Frame"])
    clr = ev.loc[ev["Type"].astype(str).str.upper().eq("CLEARANCE")]
    return {k: np.asarray(v) for k, v in out.items()} | {
        "logged_t": np.interp(clr["Start Frame"].to_numpy(float), nf, ts), "logged_team": clr["Team"].astype(str).str.lower().to_numpy()}


def score(K: list[dict], speed: float, own: float, gain: float) -> dict:
    found = logged = hits = cands = 0
    for k in K:
        c = (k["speed"] >= speed) & (k["depth"] <= own) & (k["gain"] >= gain) & ~k["next_same"]
        ct, cteam = k["t"][c], k["team"][c]
        cands += int(c.sum())
        logged += len(k["logged_t"])
        found += sum(bool(((np.abs(ct - lt) <= 1.0) & (cteam == lteam)).any()) for lt, lteam in zip(k["logged_t"], k["logged_team"], strict=True))
        hits += sum(bool(((np.abs(k["logged_t"] - t) <= 1.0) & (k["logged_team"] == tm)).any()) for t, tm in zip(ct, cteam, strict=True))
    return {"recall": round(found / logged, 3) if logged else None, "precision": round(hits / cands, 3) if cands else None,
            "candidates": cands, "logged": logged}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idsse", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.idsse, args.cache):
        if not _outside(p):
            sys.exit(f"refusing to read sample data inside the repository: {p}")
    K = []
    for mid in MATCH_IDS:
        m, _ = load_idsse_match(args.idsse, mid, args.cache)
        K.append(kicks(m))
        print(mid, len(K[-1]["t"]), "kicks", len(K[-1]["logged_t"]), "logged clearances", file=sys.stderr, flush=True)
    fit, test = K[:4], K[4:]
    rows = []
    for sp, ow, ga in itertools.product(GRID["speed"], GRID["own"], GRID["gain"]):
        f = score(fit, sp, ow, ga)
        rows.append({"speed": sp, "own": ow, "gain": ga, "fit": f})
    best = max(rows, key=lambda r: (2 * r["fit"]["recall"] * r["fit"]["precision"] / (r["fit"]["recall"] + r["fit"]["precision"])
                                    if r["fit"]["recall"] and r["fit"]["precision"] else 0))
    best["held_out"] = score(test, best["speed"], best["own"], best["gain"])
    print(json.dumps({"best_on_fit_by_f1": best, "grid_fit": rows}, indent=1))


if __name__ == "__main__":
    main()
