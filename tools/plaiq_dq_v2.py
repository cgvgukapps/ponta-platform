"""
plaiq_dq_v2.py — decision quality v2 (``analytics/decision_quality_v2.py``) against v1, on the groundings.

The same three groundings as ``plaiq_dq_grounding.py`` (independent maths, outcomes, context), on every
SkillCorner pass of its 20 open-data matches, for v1 (cached ``MatchAnalysis.event_pov``), v2 (expected
value at the pass, POV's value part) and v2 with the threat alone as value, plus:

- **consistency:** the band shares of completed and failed passes (judging the intention, failed passes
  should not look better than completed ones);
- **halves:** every measure on the even and odd matches separately (nothing is fitted; it shows stability).

On SkillCorner's data a failed pass's target is SkillCorner's targeted player (the reader's end point).
Comparison data only (founder decision 2). Writes JSON outside the repository; no names are read.

    python tools/plaiq_dq_v2.py --data ~/ponta-reference-data/skillcorner-opendata --cache /tmp/ponta-context/sc-cache \
        --out /tmp/ponta-context/dq_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.analytics.decision_quality_v2 import decision_quality_v2
from plaiq_engine.analytics.pass_option_value import _build_defender_data
from plaiq_engine.match_analysis import MatchAnalysis
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_reference_set import _load
from plaiq_dq_replay import BANDS, band_of
from plaiq_references import auc, spearman

SCORES = ("v1", "v2", "v2_xt")


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def _xt_at(surface: np.ndarray, tree: cKDTree, points: list) -> list[float]:
    """Plaiq's threat at each point (nearest hex)."""
    return [float(surface[tree.query(x)[1]]) for x in points]


def one(match_dir: Path, cache: Path) -> pd.DataFrame:
    p = cache / f"pov_skillcorner_{match_dir.name}.pkl"
    if not p.exists():
        return pd.DataFrame()
    pov = pd.read_pickle(p)
    actor = pov.loc[pov["decision_quality"].notna() | pov["decision_quality_failed"].notna()].drop_duplicates("event_idx").set_index("event_idx")
    m = _load(match_dir, cache)
    ma = MatchAnalysis(m)
    tree = cKDTree(ma.hex_centers)
    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    home = meta["home_team"]["id"]
    ev = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", low_memory=False,
                     usecols=["event_id", "event_type", "end_type", "frame_end", "team_id", "pass_outcome", "lead_to_shot", "lead_to_goal",
                              "overall_pressure_end", "associated_player_possession_event_id", "xpass_completion", "xthreat", "targeted"])
    opts = ev.loc[ev["event_type"] == "passing_option"].assign(evx=lambda o: o["xpass_completion"] * o["xthreat"])
    by_pass = {k: g for k, g in opts.groupby("associated_player_possession_event_id")}
    passes = ev.loc[(ev["event_type"] == "player_possession") & (ev["end_type"] == "pass") & ev["pass_outcome"].isin(["successful", "unsuccessful"])]
    mev = m.events.loc[m.events["Type"].isin(["PASS", "BALL LOST"])]
    rows = []
    for r in passes.itertuples():
        team = "Home" if r.team_id == home else "Away"
        cand = mev.loc[(mev["Team"] == team) & ((mev["Start Frame"] - r.frame_end).abs() <= 2)]
        if cand.empty:
            continue
        e = cand.iloc[0]
        f = int(e["Start Frame"])
        own, opp = (ma.home, ma.away) if team == "Home" else (ma.away, ma.home)
        if f not in own.index or not (np.isfinite(e["Start X"]) and np.isfinite(e.get("End X", np.nan))):
            continue
        jer, ojer = (ma.home_jerseys, ma.away_jerseys) if team == "Home" else (ma.away_jerseys, ma.home_jerseys)
        oprof = ma.profiles_away if team == "Home" else ma.profiles_home
        xt = ma.xt_home if team == "Home" else ma.xt_away
        defenders = _build_defender_data(opp.loc[f], "Away" if team == "Home" else "Home", ojer, oprof)
        ball, target = np.array([e["Start X"], e["Start Y"]], float), np.array([e["End X"], e["End Y"]], float)
        mates = [np.array([own.at[f, f"{team}_{j}_x"], own.at[f, f"{team}_{j}_y"]], float) for j in jer if str(j) != str(e["From"])]
        mates = [x for x in mates if np.isfinite(x).all()]
        xb, xtg = _xt_at(xt, tree, [ball])[0], _xt_at(xt, tree, [target])[0]
        xm = _xt_at(xt, tree, mates)
        v2 = decision_quality_v2(ball, target, xb, xtg, mates, xm, defenders)
        v2x = decision_quality_v2(ball, target, xb, xtg, mates, xm, defenders, value_key="value_xt")
        v1 = np.nan
        if e.name in actor.index:
            a = actor.loc[e.name]
            v1 = a["decision_quality"] if e["Type"] == "PASS" else a["decision_quality_failed"]
        o = by_pass.get(r.event_id)
        sc_dq = chose = np.nan
        if o is not None and o["targeted"].any() and o["evx"].max() > 0:
            t = o.loc[o["targeted"], "evx"].max()
            sc_dq, chose = float(t / o["evx"].max()), bool(t >= o["evx"].max() - 1e-12)
        rows.append({"match": match_dir.name, "completed": r.pass_outcome == "successful", "v1": float(v1), "v2": v2["dq"], "v2_xt": v2x["dq"],
                     "availability": v2["availability_chosen"], "sc_dq": sc_dq, "sc_chose_best": chose, "lead_to_shot": bool(r.lead_to_shot),
                     "lead_to_goal": bool(r.lead_to_goal), "pressure": r.overall_pressure_end if isinstance(r.overall_pressure_end, str) else None})
    return pd.DataFrame(rows)


def measures(df: pd.DataFrame, s: str) -> dict:
    d = df.dropna(subset=[s])
    band = d[s].map(band_of)
    comp, fail = d.loc[d["completed"]], d.loc[~d["completed"]]
    k = comp.dropna(subset=["sc_dq"])
    return {
        "optimal_share_completed": round(float((band[d["completed"]] == "optimal").mean()), 3),
        "optimal_share_failed": round(float((band[~d["completed"]] == "optimal").mean()), 3),
        "median_completed": round(float(comp[s].median()), 3), "median_failed": round(float(fail[s].median()), 3),
        "auc_shot_completed": auc(comp[s], comp["lead_to_shot"]), "auc_goal_completed": auc(comp[s], comp["lead_to_goal"]),
        "auc_shot_all": auc(d[s], d["lead_to_shot"]), "auc_completion_all": auc(d[s], d["completed"]),
        "shot_rate_by_band_completed": {b: round(float(comp.loc[band[d["completed"]] == b, "lead_to_shot"].mean()), 4) for b in BANDS
                                        if (band[d["completed"]] == b).any()},
        "spearman_vs_sc_dq": spearman(k[s], k["sc_dq"]), "auc_sc_chose_best": auc(k[s], k["sc_chose_best"].astype(bool)),
        "median_by_pressure_completed": {pl: round(float(comp.loc[comp["pressure"] == pl, s].median()), 3)
                                         for pl in ("no_pressure", "low_pressure", "medium_pressure", "high_pressure", "very_high_pressure")
                                         if (comp["pressure"] == pl).any()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.data, args.cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    parts = []
    dirs = [d for d in sorted((args.data / "matches").iterdir()) if (d / f"{d.name}_dynamic_events.csv").exists()]
    for n, d in enumerate(dirs):
        df = one(d, args.cache)
        df["half"] = "even" if n % 2 == 0 else "odd"
        parts.append(df)
        print(d.name, len(df), "passes", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_pickle(args.out.parent / "dq_v2_rows.pkl")
    out = {"passes": int(len(df)), "completed": int(df["completed"].sum()),
           "all": {s: measures(df, s) for s in SCORES},
           "halves": {h: {s: {k: v for k, v in measures(df.loc[df["half"] == h], s).items() if not isinstance(v, dict)} for s in SCORES}
                      for h in ("even", "odd")}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["all"], indent=1, default=str))


if __name__ == "__main__":
    main()
