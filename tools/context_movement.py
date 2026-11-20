"""
context_movement.py — P6 0.1.0's movement types against SkillCorner's off-ball runs (Stage C, deliverable 3).

On every SkillCorner open-data match (comparison only, founder decision 2), with P1, P2 and P6 run as shipped:

- **detection:** each SkillCorner off-ball run is matched to a P6 run of the same player overlapping it for at least
  half of the shorter of the two; recall (SkillCorner's runs found) and precision (P6's off-ball runs that
  SkillCorner also labels);
- **type**, on matched runs: P6's type against SkillCorner's subtype (the same ten names), confusion and kappa;
- **FIFA's shape categories:** P6's ``in_to_out`` / ``out_to_in`` against SkillCorner's ``in_to_out`` and
  ``out_to_in`` flags; P6's ``in_behind`` area against SkillCorner's ``behind`` subtype;
- **pressing:** each SkillCorner on-ball engagement (pressure, pressing, counter-press, recovery press) is matched to
  a P6 pressing run by the same player ending within 1 s of the engagement's start; recall by subtype, and the share
  of P6 pressing runs near an engagement;
- every measure on the even (fit) and odd (held-out) matches. Nothing is tuned here: the rules were fixed first.

Reads the data outside the repository and writes JSON outside it; no names are read.

    python tools/context_movement.py --data ~/ponta-reference-data/skillcorner-opendata \
        --cache /tmp/ponta-context/sc-cache --out /tmp/ponta-context/movement.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.movement import MovementPackage
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.context.thresholds import values

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_reference_set import _kappa, _load

ENGAGEMENTS = ("pressure", "pressing", "counter_press", "recovery_press")


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def one(match_dir: Path, cache: Path, overrides: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m = _load(match_dir, cache)
    res = Registry([PossessionPackage(), StructurePackage(), MovementPackage()]).run(m, config=values(overrides or {}))
    g, runs = res.frames, res.outputs["p6"].segments["runs"]
    nf, ts = g["native_frame"].to_numpy(np.int64), g["frame_ts"].to_numpy(float)

    def t_of(frame: float) -> float:
        return float(ts[int(np.clip(np.searchsorted(nf, int(frame)), 0, len(nf) - 1))])

    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    home = meta["home_team"]["id"]
    pid = {p["id"]: f"{'home' if p['team_id'] == home else 'away'}_{p['number']}" for p in meta["players"]}
    ev = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", low_memory=False,
                     usecols=["event_type", "event_subtype", "frame_start", "frame_end", "player_id", "in_to_out", "out_to_in"])
    sc_runs = ev.loc[ev["event_type"] == "off_ball_run"]
    att = runs.loc[runs["in_possession"]].copy()
    att["matched"] = False
    rows = []
    for r in sc_runs.itertuples():
        who = pid.get(r.player_id)
        if who is None:
            continue
        s0, s1 = t_of(r.frame_start), t_of(r.frame_end)
        c = att.loc[att["player_id"] == who]
        ov = np.minimum(c["end_ts"], s1) - np.maximum(c["start_ts"], s0)
        need = 0.5 * np.minimum(c["end_ts"] - c["start_ts"], s1 - s0)
        hit = c.loc[(ov > 0) & (ov >= need)]
        row = {"match": match_dir.name, "sc_type": r.event_subtype, "sc_in_to_out": str(r.in_to_out) == "True",
               "sc_out_to_in": str(r.out_to_in) == "True", "found": len(hit) > 0, "ponta_type": None, "ponta_shape": None,
               "ponta_area": None, "ponta_channel": None}
        if len(hit):
            best = hit.loc[ov[hit.index].idxmax()]
            att.loc[best.name, "matched"] = True
            row.update({"ponta_type": best["type"], "ponta_shape": best["shape"], "ponta_area": best["area_end"],
                        "ponta_channel": best["channel_move"]})
        rows.append(row)
    sc = pd.DataFrame(rows)
    ours = att[["type", "matched"]].assign(match=match_dir.name)
    press = runs.loc[runs["type"] == "pressing"]
    eng = ev.loc[(ev["event_type"] == "on_ball_engagement") & ev["event_subtype"].isin(ENGAGEMENTS)]
    erows, near = [], np.zeros(len(press), bool)
    for r in eng.itertuples():
        who = pid.get(r.player_id)
        if who is None:
            continue
        t = t_of(r.frame_start)
        k = (press["player_id"] == who).to_numpy() & (np.abs(press["end_ts"].to_numpy() - t) <= 1.0)
        near |= k
        erows.append({"match": match_dir.name, "subtype": r.event_subtype, "found": bool(k.any())})
    pe = pd.DataFrame(erows)
    pe.attrs["pressing_runs"], pe.attrs["near"] = int(len(press)), int(near.sum())
    return sc, ours, pe


def score(sc: pd.DataFrame, ours: pd.DataFrame, pe: pd.DataFrame, press_counts: tuple[int, int]) -> dict:
    m = sc.loc[sc["found"]]
    out = {"sc_runs": int(len(sc)), "recall": round(float(sc["found"].mean()), 4),
           "ponta_off_ball_runs": int(len(ours)), "precision": round(float(ours["matched"].mean()), 4),
           "type": _kappa(m["sc_type"].astype(str).to_numpy(), m["ponta_type"].astype(str).to_numpy()),
           "type_recall_by_sc_type": {t: round(float((g["ponta_type"] == t).mean()), 3) for t, g in m.groupby("sc_type")},
           "in_to_out": _kappa(m["sc_in_to_out"].astype(str).to_numpy(), (m["ponta_shape"] == "in_to_out").astype(str).to_numpy()),
           "out_to_in": _kappa(m["sc_out_to_in"].astype(str).to_numpy(), (m["ponta_shape"] == "out_to_in").astype(str).to_numpy()),
           "channel_outward_vs_sc_in_to_out": _kappa(m["sc_in_to_out"].astype(str).to_numpy(), (m["ponta_channel"] == "outward").astype(str).to_numpy()),
           "channel_inward_vs_sc_out_to_in": _kappa(m["sc_out_to_in"].astype(str).to_numpy(), (m["ponta_channel"] == "inward").astype(str).to_numpy()),
           "behind": _kappa((m["sc_type"] == "behind").astype(str).to_numpy(), (m["ponta_area"] == "in_behind").astype(str).to_numpy()),
           "pressing_recall_by_subtype": {s: round(float(g["found"].mean()), 3) for s, g in pe.groupby("subtype")},
           "pressing_runs_near_an_engagement": round(press_counts[1] / max(press_counts[0], 1), 4)}
    out["confusion"] = {r: {c: int(v) for c, v in row.items() if v}
                        for r, row in pd.crosstab(m["sc_type"], m["ponta_type"].fillna("none")).iterrows()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--set", action="append", help="override a threshold, e.g. p6_press_min_speed_mps=2.5 (repeatable)")
    args = ap.parse_args()
    overrides = {k.strip(): float(v) for k, _, v in (x.partition("=") for x in args.set or [])}
    for p in (args.data, args.cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    dirs = [d for d in sorted((args.data / "matches").iterdir()) if (d / f"{d.name}_dynamic_events.csv").exists()]
    if args.matches:
        dirs = [d for d in dirs if d.name in args.matches]
    parts = {"fit": ([], [], [], [0, 0]), "test": ([], [], [], [0, 0])}
    for n, d in enumerate(dirs):
        sc, ours, pe = one(d, args.cache, overrides)
        h = "fit" if n % 2 == 0 else "test"
        parts[h][0].append(sc)
        parts[h][1].append(ours)
        parts[h][2].append(pe)
        parts[h][3][0] += pe.attrs["pressing_runs"]
        parts[h][3][1] += pe.attrs["near"]
        print(d.name, len(sc), "SkillCorner runs", flush=True)
    out = {h: score(pd.concat(v[0], ignore_index=True), pd.concat(v[1], ignore_index=True), pd.concat(v[2], ignore_index=True), tuple(v[3]))
           for h, v in parts.items() if v[0]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({h: {k: v for k, v in o.items() if k != "confusion"} for h, o in out.items()}, indent=1))


if __name__ == "__main__":
    main()
