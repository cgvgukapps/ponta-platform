"""
plaiq_option_split.py — availability and value (``analytics/option_split.py``) against SkillCorner.

Step 3 of ``ponta-blueprint/plaiq-review.md``. For every SkillCorner pass on its 20 open-data matches,
every team-mate of the passer (not the passer) is scored as a passing option at the pass:
availability, value, value gain, and whether a pass to them would break a line (P5's rule, units at
the pass moment). Compared with SkillCorner (comparison only, founder decision 2):

- **availability** against whether the pass to the targeted player was completed, and against
  SkillCorner's own list of passing options for that possession (is this team-mate on it?);
- **an availability threshold**, chosen on the fit matches (even positions) to best reproduce
  SkillCorner's option lists, then used to count **line-breaking options** per pass against
  SkillCorner's ``n_passing_options_line_break`` on the held-out matches;
- **value** candidates for the targeted player against SkillCorner's threat (rank correlation) and
  its "dangerous" flag (AUC): POV's value part, the threat gain, and the threat at the receiver.

Writes JSON outside the repository; no names are read.

    python tools/plaiq_option_split.py --data ~/ponta-reference-data/skillcorner-opendata --cache /tmp/ponta-context/sc-cache \
        --out /tmp/ponta-context/option_split.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.analytics.option_split import option_components
from plaiq_engine.analytics.pass_option_value import _build_defender_data
from plaiq_engine.context.line_breaks import breaks, units_at
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.structure import (
    OTHER,
    StructurePackage,
    team_frames,
    window_index,
    windows_from_segments,
)
from plaiq_engine.context.thresholds import values
from plaiq_engine.match_analysis import MatchAnalysis
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_reference_set import _kappa, _load
from plaiq_references import auc, spearman

THETAS = tuple(round(x, 2) for x in np.arange(0.05, 0.96, 0.05))


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def one(match_dir: Path, cache: Path) -> pd.DataFrame:
    m = _load(match_dir, cache)
    cfg = values()
    ma = MatchAnalysis(m)
    tree = cKDTree(ma.hex_centers)
    res = Registry([PossessionPackage(), StructurePackage()]).run(m, config=cfg)
    g, p2 = res.frames, res.outputs["p2"]
    tf = team_frames(m, g, p2.segments["orientation"])
    sign = {(int(r.period), r.team): r.attack_sign for r in p2.segments["orientation"].itertuples()}
    keepers, block = windows_from_segments(p2.segments).gk, window_index(g, cfg)
    nf, per = g["native_frame"].to_numpy(np.int64), g["period"].to_numpy()
    L = m.pitch.length_m
    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    home = meta["home_team"]["id"]
    number = {p["id"]: str(p["number"]) for p in meta["players"]}
    ev_all = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", low_memory=False,
                         usecols=["event_id", "event_type", "end_type", "frame_end", "team_id", "pass_outcome", "player_id", "player_targeted_id",
                                  "player_targeted_xthreat", "player_targeted_dangerous", "n_passing_options_line_break",
                                  "associated_player_possession_event_id"])
    opts = ev_all.loc[ev_all["event_type"] == "passing_option"].groupby("associated_player_possession_event_id")["player_id"] \
        .apply(lambda s: {number.get(p) for p in s}).to_dict()
    ref = ev_all.loc[(ev_all["event_type"] == "player_possession") & (ev_all["end_type"] == "pass")
                     & ev_all["pass_outcome"].isin(["successful", "unsuccessful"])]
    ev = m.events.loc[m.events["Type"].isin(["PASS", "BALL LOST"])]
    rows = []
    for r in ref.itertuples():
        team = "Home" if r.team_id == home else "Away"
        cand = ev.loc[(ev["Team"] == team) & ((ev["Start Frame"] - r.frame_end).abs() <= 2)]
        if cand.empty:
            continue
        e = cand.iloc[0]
        f = int(e["Start Frame"])
        own, opp = (ma.home, ma.away) if team == "Home" else (ma.away, ma.home)
        if f not in own.index or not np.isfinite(e["Start X"]):
            continue
        jerseys, ojer = (ma.home_jerseys, ma.away_jerseys) if team == "Home" else (ma.away_jerseys, ma.home_jerseys)
        oprof = ma.profiles_away if team == "Home" else ma.profiles_home
        xt = ma.xt_home if team == "Home" else ma.xt_away
        defenders = _build_defender_data(opp.loc[f], "Away" if team == "Home" else "Home", ojer, oprof)
        ball = np.array([e["Start X"], e["Start Y"]], float)
        xt_ball = float(xt[tree.query(ball)[1]])
        t = team.lower()
        k = int(np.clip(np.searchsorted(nf, f), 0, len(nf) - 1))
        s = sign.get((int(per[k]), t), 1)
        d = OTHER[t]
        gk = keepers.get((d, int(per[k]), int(block[k])))
        units = units_at(tf[d].depth[k], tf[d].width[k], L, tf[d].ids.index(gk) if gk in tf[d].ids else None)
        start_att = (s * ball[0] + L / 2, s * ball[1])
        sc_opts = opts.get(r.event_id, set())
        target = number.get(r.player_targeted_id)
        for j in jerseys:
            if str(j) == str(e["From"]):
                continue
            x, y = own.at[f, f"{team}_{j}_x"], own.at[f, f"{team}_{j}_y"]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            rec = np.array([x, y], float)
            c = option_components(ball, rec, xt_ball, float(xt[tree.query(rec)[1]]), defenders)
            b = breaks(start_att, (s * x + L / 2, s * y), units)
            rows.append({"match": match_dir.name, "pass": int(r.event_id), "jersey": str(j), "target": str(j) == target,
                         "sc_option": str(j) in sc_opts, "completed": r.pass_outcome == "successful",
                         "sc_xthreat": r.player_targeted_xthreat, "sc_dangerous": r.player_targeted_dangerous,
                         "sc_lb_options": r.n_passing_options_line_break, "breaks": int(sum(bool(v) for v in b.values())),
                         "xt_receiver": float(xt[tree.query(rec)[1]]), **c})
    return pd.DataFrame(rows)


def score(df: pd.DataFrame, theta: float) -> dict:
    tg = df.loc[df["target"]]
    # Line-breaking options per pass: available team-mates (availability at least theta) beyond at least one line.
    lb = df.assign(lbo=(df["availability"] >= theta) & (df["breaks"] > 0)).groupby("pass").agg(sc=("sc_lb_options", "first"), ponta=("lbo", "sum"))
    lb = lb.dropna()
    ref, n = lb["sc"].astype(int).to_numpy(), lb["ponta"].astype(int).to_numpy()
    return {
        "availability": {"auc_completion_of_targeted_pass": auc(tg["availability"], tg["completed"]),
                         "auc_completion_pov": auc(tg["pov"], tg["completed"]),
                         "auc_is_skillcorner_option": auc(df["availability"], df["sc_option"])},
        "line_breaking_options": {"theta": theta, "passes": int(len(lb)), "sc_mean": round(float(ref.mean()), 2), "ponta_mean": round(float(n.mean()), 2),
                                  "any_option": _kappa((ref > 0).astype(str), (n > 0).astype(str)),
                                  "exact": round(float((n == ref).mean()), 4), "within_1": round(float((abs(n - ref) <= 1).mean()), 4),
                                  "spearman": spearman(n, ref)},
        "value_of_targeted_player": {
            name: {"spearman_sc_xthreat": spearman(tg[col], tg["sc_xthreat"]),
                   "auc_sc_dangerous": auc(tg[col], tg["sc_dangerous"].fillna(False).astype(bool))}
            for name, col in (("pov", "pov"), ("value_pov_part", "value"), ("value_gain", "value_gain"), ("xt_receiver", "xt_receiver"))},
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
        df["set"] = "fit" if n % 2 == 0 else "test"
        parts.append(df)
        print(d.name, df["pass"].nunique(), "passes", len(df), "options", flush=True)
    allo = pd.concat(parts, ignore_index=True)
    allo.to_pickle(args.out.parent / "option_split_rows.pkl")
    fit, test = allo.loc[allo["set"] == "fit"], allo.loc[allo["set"] == "test"]
    # The threshold that best reproduces SkillCorner's option lists on the fit matches (F1).
    f1 = {}
    for th in THETAS:
        pred = fit["availability"] >= th
        tp = int((pred & fit["sc_option"]).sum())
        f1[th] = 2 * tp / max(int(pred.sum()) + int(fit["sc_option"].sum()), 1)
    theta = max(f1, key=f1.get)
    out = {"theta_chosen_on_fit": theta, "option_list_f1_by_theta_fit": {str(k): round(v, 4) for k, v in f1.items()},
           "fit": score(fit, theta), "test": score(test, theta)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("theta_chosen_on_fit", "fit", "test")}, indent=1, default=str))


if __name__ == "__main__":
    main()
