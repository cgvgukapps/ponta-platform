"""
context_measure.py — the measurements behind context package P1's thresholds.

Runs on the regression set (Metrica Sample Games 1 to 3, SkillCorner 1886347) and writes a
JSON file of figures to an output directory **outside the repository** (the script refuses to
write inside it). Nothing here reaches the product; it is how the P1 thresholds are proposed.

    python tools/context_measure.py --out /tmp/ponta-context --cache /tmp/ponta-context/cache

Measured per match: ball detection and gap lengths; for a sweep of control radius, ball-speed
limit and minimum switch time, agreement of proximity possession with the engine's existing
event-anchored possession (Metrica) and with the provider's player possessions (SkillCorner,
comparison only: decision 2 in ponta-blueprint/context-packages.md). Player names in the
SkillCorner file are never read.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.match import (
    load_metrica_csv_match,
    load_metrica_epts_match,
    load_skillcorner_match,
)
from matrix_engine.sample_data import repo_root, sample_data_dir
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
from cache_tag import tag

SC_ID = 1886347
RADII = (1.0, 1.5, 2.0, 2.5, 3.0)
SPEEDS = (8.0, 12.0, 16.0, 1e9)
SWITCHES = (0.0, 0.4, 0.8, 1.2)


def load(key: str, cache: Path):
    p = cache / f"{key}.{tag()}.pkl"
    if p.exists():
        with open(p, "rb") as fh:
            return pickle.load(fh)
    d = sample_data_dir()
    m = {
        "metrica_1": lambda: load_metrica_csv_match(d, 1),
        "metrica_2": lambda: load_metrica_csv_match(d, 2),
        "metrica_3": lambda: load_metrica_epts_match(d, 3),
        "skillcorner_1886347": lambda: load_skillcorner_match(os.path.join(d, "skillcorner", "matches", str(SC_ID)), SC_ID),
    }[key]()
    cache.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as fh:
        pickle.dump(m, fh)
    return m


def provider_possession(grid: pd.DataFrame) -> np.ndarray:
    """SkillCorner's own player possessions on the grid (comparison only). Names are not read."""
    base = Path(sample_data_dir()) / "skillcorner" / "matches" / str(SC_ID)
    meta = json.loads((base / f"{SC_ID}_match.json").read_text())
    home_id = meta["home_team"]["id"]
    ev = pd.read_csv(base / f"{SC_ID}_dynamic_events.csv", usecols=["event_type", "frame_start", "frame_end", "team_id"])
    pp = ev.loc[ev["event_type"] == "player_possession"].dropna(subset=["frame_start", "frame_end"])
    out = np.full(len(grid), None, dtype=object)
    nf = grid["native_frame"].to_numpy()
    for _, r in pp.iterrows():
        m = (nf >= r["frame_start"]) & (nf <= r["frame_end"])
        out[m] = "home" if r["team_id"] == home_id else "away"
    return out


def compare(lab: np.ndarray, src: np.ndarray, ref: np.ndarray) -> dict:
    both = np.isin(lab, TEAMS) & np.isin(ref, TEAMS)
    obs = both & (src == "observed")
    return {
        "compared_share": round(float(both.mean()), 4),
        "agreement": round(float((lab[both] == ref[both]).mean()), 4) if both.any() else None,
        "agreement_observed_only": round(float((lab[obs] == ref[obs]).mean()), 4) if obs.any() else None,
    }


def switches(lab: np.ndarray) -> int:
    seq = [x for x in lab if x in TEAMS]
    return int(sum(a != b for a, b in itertools.pairwise(seq)))


def longest_disagreements(grid: pd.DataFrame, lab: np.ndarray, ref: np.ndarray, top: int = 5) -> list[dict]:
    dis = np.isin(lab, TEAMS) & np.isin(ref, TEAMS) & (lab != ref)
    ts, per, minute = grid["frame_ts"].to_numpy(), grid["period"].to_numpy(), grid["minute"].to_numpy()
    runs, start = [], None
    for i in range(len(dis) + 1):
        on = i < len(dis) and dis[i] and (start is None or per[i] == per[start])
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((ts[i - 1] - ts[start], start, i - 1))
            start = i if i < len(dis) and dis[i] else None
    runs.sort(reverse=True)
    return [{"period": int(per[s]), "minute": int(minute[s]), "duration_s": round(float(d), 1),
             "ponta": str(lab[s]), "reference": str(ref[s])} for d, s, e in runs[:top]]


def confusion(lab: np.ndarray, ref: np.ndarray) -> dict:
    refk = np.array([r if r in TEAMS else "none" for r in ref], dtype=object)
    out = {}
    for a in ("home", "away", "contested", "dead_ball", "unknown"):
        out[a] = {b: int(((lab == a) & (refk == b)).sum()) for b in ("home", "away", "none")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--matches", nargs="*", default=["metrica_1", "metrica_2", "metrica_3", "skillcorner_1886347"])
    args = ap.parse_args()
    out, cache = Path(args.out).resolve(), Path(args.cache).resolve()
    root = repo_root().resolve()
    for p in (out, cache):
        if p == root or root in p.parents:
            print(f"refusing to write inside the repository: {p}", file=sys.stderr)
            return 2
    out.mkdir(parents=True, exist_ok=True)

    base = values()
    report: dict = {"thresholds_base": base, "sweep_axes": {"radius_m": RADII, "max_speed_mps": SPEEDS, "min_switch_s": SWITCHES}, "matches": {}}
    for key in args.matches:
        t0 = time.time()
        m = load(key, cache)
        grid = build_grid(m, base["context_grid_rate_hz"])
        ball = ball_on_grid(m, grid)
        near = nearest_on_grid(m, grid, ball)
        ev_poss = event_possession_on_grid(m, grid)
        dead, _ = dead_ball_on_grid(m, grid, base)
        ref_name = "provider_player_possession" if key.startswith("skillcorner") else "event_anchored"
        ref = provider_possession(grid) if key.startswith("skillcorner") else ev_poss
        rows = []
        for r, v, w in itertools.product(RADII, SPEEDS, SWITCHES):
            cfg = values({"p1_control_radius_m": r, "p1_max_control_ball_speed_mps": v, "p1_min_switch_s": w})
            lab, src, _ = label_possession(grid, ball, near, ev_poss, dead, cfg)
            in_play = lab != "dead_ball"
            rows.append({"radius_m": r, "max_speed_mps": v if v < 1e8 else None, "min_switch_s": w,
                         **compare(lab, src, ref), "team_switches": switches(lab),
                         "observed_share_in_play": round(float(((src == "observed") & in_play).sum() / max(in_play.sum(), 1)), 4)})
        lab, src, _ = label_possession(grid, ball, near, ev_poss, dead, base)
        report["matches"][key] = {
            "reference": ref_name,
            "reference_team_switches": switches(ref),
            "grid_points": int(len(grid)),
            "base": {**compare(lab, src, ref), "team_switches": switches(lab), "confusion": confusion(lab, ref),
                     "longest_disagreements": longest_disagreements(grid, lab, ref)},
            "sweep": rows,
            "seconds": round(time.time() - t0, 1),
        }
        print(key, "done in", report["matches"][key]["seconds"], "s", flush=True)
    (out / "context_measurements.json").write_text(json.dumps(report, indent=1, default=str))
    print("wrote", out / "context_measurements.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
