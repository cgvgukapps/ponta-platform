"""
plaiq_pass_completion_check.py — the adopted pass completion probability, end to end, on SkillCorner's open data.

``plaiq_pass_probability.py`` fitted the model on extracted passes. This checks what ships:
``MatchAnalysis.pass_completion`` with the adopted parameter set, on every pass of SkillCorner's 20
open-data matches that has a logged target (completed passes, and failed passes SkillCorner logs a
target for). Its parameters were fitted on the even matches, so the odd matches are held out. Reports
log-loss, Brier score, AUC and observed completion by predicted probability (compare SkillCorner's own
expected completion. Also reports Metrica's sample games by half (completed
passes only: Metrica logs no target for a failed pass).

Comparison data only (founder decision 2); data and output stay outside the repository; no names are read.

    python tools/plaiq_pass_completion_check.py --sc-data ~/ponta-reference-data/skillcorner-opendata \
        --cache /tmp/ponta-context/sc-cache --out /tmp/ponta-context/pass_completion_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from matrix_engine.match import load_metrica_csv_match
from matrix_engine.sample_data import repo_root, sample_data_dir
from plaiq_engine.match_analysis import MatchAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_reference_set import _load
from plaiq_pass_probability import scores


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def one(match_dir: Path, cache: Path) -> pd.DataFrame:
    ma = MatchAnalysis(_load(match_dir, cache))
    pc = ma.pass_completion.set_index("event_idx")
    ev = ma.events.loc[pc.index]
    return pd.DataFrame({"match": match_dir.name, "completed": (ev["Type"] == "PASS").to_numpy(),
                         "p": pc["pass_completion_probability"].to_numpy(), "parameter_set": pc["parameter_set"].to_numpy()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sc-data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.sc_data, args.cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    dirs = [d for d in sorted((args.sc_data / "matches").iterdir()) if (d / f"{d.name}_dynamic_events.csv").exists()]
    df = pd.concat([one(d, args.cache) for d in dirs], ignore_index=True)
    order = {m: i for i, m in enumerate(sorted(df["match"].unique()))}
    df["half"] = df["match"].map(lambda m: "fit" if order[m] % 2 == 0 else "held_out")
    ok = df.loc[df["p"].notna()]
    out = {"skillcorner": {"parameter_set": sorted(df["parameter_set"].unique()), "passes": int(len(df)), "with_probability": int(len(ok)),
                           **{h: scores(g["p"].to_numpy(), g["completed"].to_numpy()) for h, g in ok.groupby("half")}}}
    metrica = {}
    for game in (1, 2):
        ma = MatchAnalysis(load_metrica_csv_match(sample_data_dir(), game))
        pc = ma.pass_completion.set_index("event_idx").join(ma.events[["Type", "Period"]])
        done = pc.loc[pc["Type"] == "PASS"]
        metrica[f"game_{game}"] = {"parameter_set": str(pc["parameter_set"].iloc[0]), "failed_passes_without_target": int((pc["Type"] == "BALL LOST").sum()),
                                   "completed_mean_by_period": {int(k): round(float(v), 3) for k, v in done.groupby("Period")["pass_completion_probability"].mean().items()}}
    out["metrica_completed_passes"] = metrica
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
