"""
plaiq_pressure.py — pressure v2 (``analytics/pressure_v2.py``) against SkillCorner and the German league.

Step 4 of ``ponta-blueprint/plaiq-review.md``. Pressure on the player in possession is compared with:

- **SkillCorner's pressure level** at the start and end of every player possession on its 20 open-data
  matches (no, low, medium, high, very high pressure; comparison only, founder decision 2). The
  settings are chosen on the fit matches (even positions) and read on the others;
- **the league's pressure on each shot** (IDSSE, 164 shots), an independent check nothing was fitted to.

Beside pressure v2 (published defaults, and the setting chosen on fit): Plaiq's 1v1 and zonal pressure
(``hex_analytics.compute_pressure``). Measures: rank correlation, and for SkillCorner a quadratic-weighted
kappa on its five levels with cut points chosen on fit. Writes JSON outside the repository; no names read.

    python tools/plaiq_pressure.py --sc-data ~/ponta-reference-data/skillcorner-opendata --sc-cache /tmp/ponta-context/sc-cache \
        --idsse ~/ponta-reference-data/idsse --idsse-cache /tmp/ponta-context/idsse-cache --out /tmp/ponta-context/pressure.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.analytics.hex_analytics import assign_all_players, compute_pressure
from plaiq_engine.analytics.pressure_v2 import D_BACK_M, D_FRONT_M, EXPONENT, pressure
from plaiq_engine.context.grid import build_grid
from plaiq_engine.context.structure import attack_signs
from plaiq_engine.match_analysis import HEX_RADIUS, MatchAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_reference_set import _load
from idsse import MATCH_IDS, _files, load_idsse_match
from plaiq_references import spearman

GRID = list(itertools.product((6.0, 9.0, 12.0), (2.0, 3.0, 5.0), (1.0, 1.75, 2.5), ("sum", "max")))
DEFAULT = (D_FRONT_M, D_BACK_M, EXPONENT, "sum")
LEVELS = ("no_pressure", "low_pressure", "medium_pressure", "high_pressure", "very_high_pressure")


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def moments(m, ma: MatchAnalysis, frames_carriers: list[tuple[int, str, str]]) -> pd.DataFrame:
    """Pressure v2 (every setting) and Plaiq's 1v1 and zonal pressure on a carrier at each (frame, team, jersey)."""
    g = build_grid(m, 5.0)
    signs = {(int(r.period), r.team): r.attack_sign for r in attack_signs(m, g).itertuples()}
    period_of = dict(zip(m.frames["frame"], m.frames["period"], strict=True))
    frames = sorted({f for f, _, _ in frames_carriers if f in ma.home.index})
    if not frames:
        return pd.DataFrame()
    hx = compute_pressure(assign_all_players(ma.home.loc[frames], ma.away.loc[frames], ma.home_jerseys, ma.away_jerseys, HEX_RADIUS))
    hx = hx.set_index(["Frame", "team", "jersey"])
    rows = []
    for f, team, jersey in frames_carriers:
        if f not in ma.home.index:
            continue
        own, opp = (ma.home, ma.away) if team == "home" else (ma.away, ma.home)
        pre, opre = team.capitalize(), ("Away" if team == "home" else "Home")
        x, y = own.at[f, f"{pre}_{jersey}_x"] if f"{pre}_{jersey}_x" in own else np.nan, own.at[f, f"{pre}_{jersey}_y"] if f"{pre}_{jersey}_y" in own else np.nan
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ojer = ma.away_jerseys if team == "home" else ma.home_jerseys
        dxy = np.array([[opp.at[f, f"{opre}_{j}_x"], opp.at[f, f"{opre}_{j}_y"]] for j in ojer], float)
        s = signs.get((int(period_of.get(f, 1)), team), 1)
        row = {"frame": f, "team": team, "jersey": jersey,
               "plaiq_1v1": hx["pressure"].get((f, pre, jersey), np.nan), "plaiq_zonal": hx["zonal_pressure"].get((f, pre, jersey), np.nan)}
        for c in GRID:
            row[c] = pressure((x, y), (s, 0.0), dxy, d_front=c[0], d_back=c[1], q=c[2], aggregate=c[3])
        rows.append(row)
    return pd.DataFrame(rows)


def skillcorner(match_dir: Path, cache: Path) -> pd.DataFrame:
    m = _load(match_dir, cache)
    ma = MatchAnalysis(m)
    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    home = meta["home_team"]["id"]
    number = {p["id"]: str(p["number"]) for p in meta["players"]}
    ev = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", low_memory=False,
                     usecols=["event_type", "player_id", "team_id", "frame_start", "frame_end", "overall_pressure_start", "overall_pressure_end"])
    pp = ev.loc[ev["event_type"] == "player_possession"]
    want, labels = [], {}
    for r in pp.itertuples():
        team, jersey = ("home" if r.team_id == home else "away"), number.get(r.player_id)
        for f, lab in ((r.frame_start, r.overall_pressure_start), (r.frame_end, r.overall_pressure_end)):
            if jersey and isinstance(lab, str) and np.isfinite(f):
                want.append((int(f), team, jersey))
                labels[(int(f), team, jersey)] = LEVELS.index(lab)
    df = moments(m, ma, want)
    if len(df):
        df["sc_level"] = [labels[(f, t, j)] for f, t, j in zip(df["frame"], df["team"], df["jersey"], strict=True)]
        df["match"] = match_dir.name
    return df


def idsse(data: Path, cache: Path) -> pd.DataFrame:
    parts = []
    for mid in MATCH_IDS:
        m, off = load_idsse_match(data, mid, cache)
        xml = ET.parse(_files(data, mid)["events"]).getroot()
        shots_xml = [s for e in xml.iter("Event") for s in [e.find("ShotAtGoal")] if s is not None]
        shots = off.events.loc[off.events["kind"] == "ShotAtGoal"].reset_index(drop=True)
        if len(shots) != len(shots_xml):
            continue
        want, lab = [], {}
        for s, x in zip(shots.itertuples(), shots_xml, strict=True):
            if isinstance(s.player, str):
                team, jersey = s.player.split("_")
                want.append((int(s.frame), team, jersey))
                lab[(int(s.frame), team, jersey)] = float(x.get("Pressure", "nan"))
        df = moments(m, MatchAnalysis(m), want)
        df["league_pressure"] = [lab[(f, t, j)] for f, t, j in zip(df["frame"], df["team"], df["jersey"], strict=True)]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def weighted_kappa(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    o = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        o[x, y] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return round(float(1 - (w * o).sum() / (w * e).sum()), 4)


def cuts_for(scores: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Cut points that give the same share of each level as the reference (fit set)."""
    shares = np.cumsum(np.bincount(levels, minlength=5))[:-1] / len(levels)
    return np.quantile(scores, shares)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sc-data", type=Path, required=True)
    ap.add_argument("--sc-cache", type=Path, required=True)
    ap.add_argument("--idsse", type=Path, required=True)
    ap.add_argument("--idsse-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.sc_data, args.sc_cache, args.idsse, args.idsse_cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    parts = []
    dirs = [d for d in sorted((args.sc_data / "matches").iterdir()) if (d / f"{d.name}_dynamic_events.csv").exists()]
    for n, d in enumerate(dirs):
        df = skillcorner(d, args.sc_cache)
        df["set"] = "fit" if n % 2 == 0 else "test"
        parts.append(df)
        print(d.name, len(df), "moments", flush=True)
    sc = pd.concat(parts, ignore_index=True)
    sc[["match", "set", "frame", "team", "sc_level", "plaiq_1v1", "plaiq_zonal"]].to_pickle(args.out.parent / "pressure_rows.pkl")
    fit, test = sc.loc[sc["set"] == "fit"], sc.loc[sc["set"] == "test"]
    rho_fit = {c: spearman(fit[c], fit["sc_level"]) for c in GRID}
    best = max(GRID, key=lambda c: rho_fit[c] or -1)
    sh = idsse(args.idsse, args.idsse_cache)

    def report(col) -> dict:
        cuts = cuts_for(fit[col].to_numpy(float), fit["sc_level"].to_numpy(int))
        lv = np.searchsorted(cuts, test[col].to_numpy(float), side="right")
        return {"cuts_on_fit": [round(float(c), 3) for c in cuts],
                "sc_spearman_fit": spearman(fit[col], fit["sc_level"]), "sc_spearman_test": spearman(test[col], test["sc_level"]),
                "sc_weighted_kappa_test": weighted_kappa(test["sc_level"].to_numpy(int), np.clip(lv, 0, 4)),
                "league_shot_spearman": spearman(sh[col], sh["league_pressure"])}

    out = {"moments": {"fit": int(len(fit)), "test": int(len(test))}, "league_shots": int(len(sh)),
           "best_on_fit": {"d_front": best[0], "d_back": best[1], "q": best[2], "aggregate": best[3]},
           "pressure_v2_best": report(best), "pressure_v2_defaults": report(DEFAULT),
           "plaiq_1v1": report("plaiq_1v1"), "plaiq_zonal": report("plaiq_zonal"),
           "top_on_fit": sorted(({"setting": list(c), "rho_fit": rho_fit[c]} for c in GRID), key=lambda r: -(r["rho_fit"] or -1))[:8]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
