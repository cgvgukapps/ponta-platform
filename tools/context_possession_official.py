"""
context_possession_official.py — P1's control thresholds against the official frame-level possession.

The German league's data (IDSSE, ``tools/idsse.py``) carries the official team in possession and
ball status on every frame: an independent reference for P1, which Gate 1 measured only against
possession inferred from events. This sweeps the three control thresholds (control radius, ball-speed
limit, minimum switch time; the axes of ``context_measure.py``):

- **chosen on four IDSSE matches** (even positions in id order), read on the other three;
- **objective, fixed in advance:** accuracy over every in-play grid point where the league names a
  team, so contested, unknown or dead-ball labels count as misses and a setting cannot gain by
  labelling less;
- **guards, which a candidate must not break:** Gate 1 on Metrica Sample Games 1 to 3 (at least 93%
  agreement with event-anchored possession, team switches within 15% of the event feed's) and P1's
  agreement with SkillCorner's own player possessions on match 1886347 (no more than ``SC_TOLERANCE``
  lower than today).

Reads sample data outside the repository and writes JSON outside it; no names are read.

    python tools/context_possession_official.py --cache /tmp/ponta-context/cache \
        --idsse ~/ponta-reference-data/idsse --idsse-cache /tmp/ponta-context/idsse-cache --out /tmp/ponta-context/p1_official.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.grid import build_grid
from plaiq_engine.context.possession import (
    TEAMS,
    ball_on_grid,
    dead_ball_on_grid,
    event_possession_on_grid,
    label_possession,
    nearest_on_grid,
)
from plaiq_engine.context.thresholds import values

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_measure import (
    RADII,
    SPEEDS,
    SWITCHES,
    compare,
    load,
    provider_possession,
    switches,
)
from idsse import MATCH_IDS, load_idsse_match

GATE1_AGREEMENT, GATE1_SWITCH_TOLERANCE = 0.93, 0.15
#: A candidate may not lose more than this against SkillCorner's player possessions (rounding noise).
SC_TOLERANCE = 0.005


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def inputs(m, cfg: dict) -> dict:
    grid = build_grid(m, cfg["context_grid_rate_hz"])
    ball = ball_on_grid(m, grid)
    dead, _ = dead_ball_on_grid(m, grid, cfg)
    return {"grid": grid, "ball": ball, "near": nearest_on_grid(m, grid, ball), "ev": event_possession_on_grid(m, grid), "dead": dead}


def label(p: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    lab, src, _ = label_possession(p["grid"], p["ball"], p["near"], p["ev"], p["dead"], cfg)
    return lab, src


def official_scores(lab: np.ndarray, live: np.ndarray, team: np.ndarray) -> dict:
    named = live & np.isin(team, TEAMS)
    both = named & np.isin(lab, TEAMS)
    return {"points": int(named.sum()), "right": int((lab[named] == team[named]).sum()),
            "agreement_where_labelled": round(float((lab[both] == team[both]).mean()), 4),
            "accuracy": round(float((lab[named] == team[named]).mean()), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--idsse", type=Path, required=True)
    ap.add_argument("--idsse-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.cache, args.idsse, args.idsse_cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    base = values()
    grid_axes = list(itertools.product(RADII, SPEEDS, SWITCHES))
    cfgs = {c: values({"p1_control_radius_m": c[0], "p1_max_control_ball_speed_mps": c[1], "p1_min_switch_s": c[2]}) for c in grid_axes}

    official: dict[str, dict] = {}
    for n, mid in enumerate(MATCH_IDS):
        m, off = load_idsse_match(args.idsse, mid, args.idsse_cache)
        p = inputs(m, base)
        ob = p["grid"][["native_frame"]].merge(off.ball.drop(columns=["period"]).rename(columns={"frame": "native_frame"}), on="native_frame", how="left")
        live, team = (ob["ball_status"] == 1).to_numpy(), ob["ball_team"].astype(object).to_numpy()
        scores = {}
        for c in grid_axes:
            lab = label(p, cfgs[c])[0]
            scores[c] = official_scores(lab, live, team) | {"switches": switches(lab)}
        official[mid] = {"set": "fit" if n % 2 == 0 else "test", "switches_official": switches(np.where(live, team, None)), "scores": scores}
        print("idsse", mid, flush=True)

    guards: dict[str, dict] = {}
    for key in ("metrica_1", "metrica_2", "metrica_3", "skillcorner_1886347"):
        m = load(key, args.cache)
        p = inputs(m, base)
        ref = provider_possession(p["grid"]) if key.startswith("skillcorner") else p["ev"]
        rs = switches(ref)
        guards[key] = {"reference_switches": rs, "scores": {}}
        for c in grid_axes:
            lab, src = label(p, cfgs[c])
            sw = switches(lab)
            guards[key]["scores"][c] = compare(lab, src, ref) | {"switches": sw, "switch_ratio": round(sw / rs - 1, 3)}
        print("guard", key, flush=True)

    def pooled(c, which):
        rows = [o["scores"][c] for o in official.values() if o["set"] == which]
        return round(sum(r["right"] for r in rows) / sum(r["points"] for r in rows), 4)

    def gate1(c):
        return all(guards[k]["scores"][c]["agreement"] >= GATE1_AGREEMENT and abs(guards[k]["scores"][c]["switch_ratio"]) <= GATE1_SWITCH_TOLERANCE
                   for k in ("metrica_1", "metrica_2", "metrica_3"))

    current = (base["p1_control_radius_m"], base["p1_max_control_ball_speed_mps"], base["p1_min_switch_s"])
    sc_now = guards["skillcorner_1886347"]["scores"][current]["agreement"]
    rows = []
    for c in grid_axes:
        rows.append({"radius_m": c[0], "max_speed_mps": None if c[1] > 1e8 else c[1], "min_switch_s": c[2],
                     "fit_accuracy": pooled(c, "fit"), "test_accuracy": pooled(c, "test"), "gate1": gate1(c),
                     "skillcorner_agreement": guards["skillcorner_1886347"]["scores"][c]["agreement"],
                     "metrica_agreement": [guards[k]["scores"][c]["agreement"] for k in ("metrica_1", "metrica_2", "metrica_3")],
                     "metrica_switch_ratio": [guards[k]["scores"][c]["switch_ratio"] for k in ("metrica_1", "metrica_2", "metrica_3")],
                     "idsse_switches_vs_official": round(sum(o["scores"][c]["switches"] for o in official.values())
                                                         / sum(o["switches_official"] for o in official.values()), 3)})
    ok = [r for r in rows if r["gate1"] and r["skillcorner_agreement"] >= sc_now - SC_TOLERANCE]
    best_any = max(rows, key=lambda r: r["fit_accuracy"])
    best_ok = max(ok, key=lambda r: r["fit_accuracy"]) if ok else None
    now = next(r for r in rows if (r["radius_m"], r["max_speed_mps"] or 1e9, r["min_switch_s"]) == current)
    out = {"current": now, "best_on_fit_with_guards": best_ok, "best_on_fit_any": best_any,
           "top_on_fit": sorted(rows, key=lambda r: -r["fit_accuracy"])[:15], "all": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("current", "best_on_fit_with_guards", "best_on_fit_any")}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
