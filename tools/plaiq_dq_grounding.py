"""
plaiq_dq_grounding.py — decision quality grounded in independent maths, outcomes and context.

Engine governance asks whether decision quality measures a good decision. A judge's rating would make
one person's opinion the truth, so this grounds Plaiq's decision quality three other ways, on every
pass of SkillCorner's 20 open-data matches (comparison data only, founder decision 2):

1. **Independent maths.** SkillCorner values every passing option at the moment of the pass (expected
   completion and threat) and marks the option the passer targeted. Its expected value per option is
   ``xpass_completion × xthreat``; SkillCorner's analogue of decision quality is the targeted option's
   value over the best option's (``sc_dq``), and ``sc_chose_best`` whether the passer picked the best.
   Compared with Plaiq's decision quality (rank correlation, AUC for picking the best).
2. **What happened next.** Completion, and SkillCorner's flags that the possession led to a shot or a
   goal, by Plaiq's band (a good decision should, on average, go better) and, for comparison, by
   SkillCorner's own analogue.
3. **Context.** Plaiq's decision quality by the pressure on the passer at the pass (SkillCorner's level)
   and by how many options SkillCorner listed: does the score behave sensibly under pressure?

Plaiq's scores come from the cached ``MatchAnalysis.event_pov`` (``context_line_breaks.py``); completed
passes use ``decision_quality``, failed passes ``decision_quality_failed``. Writes JSON outside the
repository; no names are read.

    python tools/plaiq_dq_grounding.py --data ~/ponta-reference-data/skillcorner-opendata --cache /tmp/ponta-context/sc-cache \
        --out /tmp/ponta-context/dq_grounding.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_reference_set import _load
from plaiq_dq_replay import BANDS, band_of
from plaiq_references import auc, spearman

PRESSURE = ("no_pressure", "low_pressure", "medium_pressure", "high_pressure", "very_high_pressure")


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def one(match_dir: Path, cache: Path) -> pd.DataFrame:
    p = cache / f"pov_skillcorner_{match_dir.name}.pkl"
    if not p.exists():
        return pd.DataFrame()
    pov = pd.read_pickle(p)
    m = _load(match_dir, cache)
    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    home = meta["home_team"]["id"]
    ev = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", low_memory=False,
                     usecols=["event_id", "event_type", "end_type", "frame_end", "team_id", "pass_outcome", "lead_to_shot", "lead_to_goal",
                              "overall_pressure_end", "associated_player_possession_event_id", "xpass_completion", "xthreat", "targeted"])
    opts = ev.loc[ev["event_type"] == "passing_option"].assign(evx=lambda o: o["xpass_completion"] * o["xthreat"])
    by_pass = {k: g for k, g in opts.groupby("associated_player_possession_event_id")}
    passes = ev.loc[(ev["event_type"] == "player_possession") & (ev["end_type"] == "pass") & ev["pass_outcome"].isin(["successful", "unsuccessful"])]
    actor = pov.loc[pov["decision_quality"].notna() | pov["decision_quality_failed"].notna()].drop_duplicates("event_idx").set_index("event_idx")
    mev = m.events.loc[m.events["Type"].isin(["PASS", "BALL LOST"])]
    rows = []
    for r in passes.itertuples():
        team = "Home" if r.team_id == home else "Away"
        cand = mev.loc[(mev["Team"] == team) & ((mev["Start Frame"] - r.frame_end).abs() <= 2)]
        if cand.empty or cand.index[0] not in actor.index:
            continue
        e, a = cand.iloc[0], actor.loc[cand.index[0]]
        dq = a["decision_quality"] if e["Type"] == "PASS" else a["decision_quality_failed"]
        if not np.isfinite(dq):
            continue
        o = by_pass.get(r.event_id)
        sc_dq = chose_best = np.nan
        n_opts = 0
        if o is not None and o["targeted"].any() and o["evx"].max() > 0:
            n_opts = len(o)
            t = o.loc[o["targeted"], "evx"].max()
            sc_dq = float(t / o["evx"].max())
            chose_best = bool(t >= o["evx"].max() - 1e-12)
        rows.append({"match": match_dir.name, "completed": r.pass_outcome == "successful", "plaiq_dq": float(dq), "plaiq_band": band_of(float(dq)),
                     "sc_dq": sc_dq, "sc_chose_best": chose_best, "n_options": n_opts, "lead_to_shot": bool(r.lead_to_shot),
                     "lead_to_goal": bool(r.lead_to_goal), "pressure": r.overall_pressure_end if isinstance(r.overall_pressure_end, str) else None})
    return pd.DataFrame(rows)


def by_group(df: pd.DataFrame, key: str, order) -> dict:
    out = {}
    for k in order:
        g = df.loc[df[key] == k]
        if len(g):
            out[str(k)] = {"n": int(len(g)), "completion": round(float(g["completed"].mean()), 3), "lead_to_shot": round(float(g["lead_to_shot"].mean()), 4),
                           "lead_to_goal": round(float(g["lead_to_goal"].mean()), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.data, args.cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    df = pd.concat([one(d, args.cache) for d in sorted((args.data / "matches").iterdir())
                    if (d / f"{d.name}_dynamic_events.csv").exists()], ignore_index=True)
    df.to_pickle(args.out.parent / "dq_grounding_rows.pkl")
    comp = df.loc[df["completed"]]
    k = comp.dropna(subset=["sc_dq"])
    k = k.assign(sc_band=k["sc_dq"].map(band_of))
    out = {
        "passes": int(len(df)), "completed": int(len(comp)), "with_skillcorner_options": int(len(k)),
        "plaiq_band_shares_completed": {b: round(float((comp["plaiq_band"] == b).mean()), 3) for b in BANDS},
        "independent_maths": {
            "spearman_plaiq_dq_vs_sc_dq": spearman(k["plaiq_dq"], k["sc_dq"]),
            "auc_plaiq_dq_for_sc_chose_best": auc(k["plaiq_dq"], k["sc_chose_best"].astype(bool)),
            "sc_chose_best_share": round(float(k["sc_chose_best"].mean()), 3),
            "plaiq_optimal_share": round(float((k["plaiq_band"] == "optimal").mean()), 3),
            "band_agreement_exact": round(float((k["plaiq_band"] == k["sc_band"]).mean()), 3),
            "crosstab_plaiq_band_by_sc_band": {b: {bb: int(((k["plaiq_band"] == b) & (k["sc_band"] == bb)).sum()) for bb in BANDS} for b in BANDS},
        },
        "outcomes_by_plaiq_band": by_group(df, "plaiq_band", BANDS),
        "outcomes_by_sc_band": by_group(df.dropna(subset=["sc_dq"]).assign(sc_band=lambda x: x["sc_dq"].map(band_of)), "sc_band", BANDS),
        "outcome_auc": {"plaiq_dq_completion": auc(df["plaiq_dq"], df["completed"]), "plaiq_dq_lead_to_shot": auc(df["plaiq_dq"], df["lead_to_shot"]),
                        "sc_dq_completion": auc(df["sc_dq"], df["completed"]), "sc_dq_lead_to_shot": auc(df["sc_dq"], df["lead_to_shot"])},
        "context_pressure": {pl: {"n": int((comp["pressure"] == pl).sum()), "plaiq_dq_median": round(float(comp.loc[comp["pressure"] == pl, "plaiq_dq"].median()), 3),
                                  "plaiq_poor_or_suboptimal": round(float(comp.loc[comp["pressure"] == pl, "plaiq_band"].isin(["poor", "suboptimal"]).mean()), 3),
                                  "sc_chose_best": round(float(k.loc[k["pressure"] == pl, "sc_chose_best"].mean()), 3) if (k["pressure"] == pl).any() else None}
                             for pl in PRESSURE},
    }
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
