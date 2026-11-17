"""
context_phase_sweep.py — measure P4 phase of play against SkillCorner's phases (Gate 2a).

Runs P1 and P2 once on SkillCorner 1886347, then P4 for each threshold setting, and reports
agreement (accuracy and Cohen's kappa) on grid points where both give the same team in
possession, for the whole match and for each half. Fitting on the first half and reading the
second guards against tuning to one match's noise. SkillCorner's phases are comparison data
only (founder decision 2); they never become output.

    python tools/context_phase_sweep.py --cache /tmp/ponta-context/cache --out /tmp/ponta-context/phase_sweep.json
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
from plaiq_engine.context.phase import PhasePackage
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import ContextInput, Registry
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.context.thresholds import values

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_measure import load
from context_references import sc_phase_spans

GRIDS = {
    "blocks": {"p4_low_block_max_m": [22.0, 24.0, 26.0, 28.0, 30.0], "p4_high_block_min_m": [38.0, 41.0, 44.0, 47.0, 50.0]},
    "chaotic": {"p4_chaotic_max_s": [0.0, 2.0, 3.0, 5.0]},
    "breaks": {"p4_break_min_gain_m": [5.0, 10.0, 15.0], "p4_break_max_s": [8.0, 15.0]},
    "smoothing": {"p4_min_phase_s": [0.0, 1.0, 2.0]},
    "direct": {"p4_direct_window_s": [2.0, 3.0, 4.0]},
}


def agreement(ref: pd.DataFrame, frames: pd.DataFrame, mask: np.ndarray | None = None) -> dict:
    team = frames["p4_team"].astype(object).to_numpy()
    both = ref["sc_team"].notna().to_numpy() & (team == ref["sc_team"].to_numpy()) & frames["p4_phase"].notna().to_numpy()
    if mask is not None:
        both &= mask
    a, b = ref["sc_phase"].to_numpy()[both].astype(str), frames["p4_phase"].astype(object).to_numpy()[both].astype(str)
    if not len(a):
        return {"points": 0}
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in set(a) | set(b))
    return {"points": int(len(a)), "accuracy": round(po, 4), "kappa": round((po - pe) / (1 - pe), 4) if pe < 1 else None}


def confusion(ref: pd.DataFrame, frames: pd.DataFrame) -> dict:
    team = frames["p4_team"].astype(object).to_numpy()
    both = ref["sc_team"].notna().to_numpy() & (team == ref["sc_team"].to_numpy()) & frames["p4_phase"].notna().to_numpy()
    ct = pd.crosstab(pd.Series(ref["sc_phase"].to_numpy()[both], name="skillcorner"), pd.Series(frames["p4_phase"].astype(object).to_numpy()[both], name="ponta"))
    return {r: {c: int(v) for c, v in row.items() if v} for r, row in ct.iterrows()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if repo_root() in args.out.resolve().parents:
        sys.exit("refusing to write measurements inside the repository")
    m = load("skillcorner_1886347", args.cache)
    base_cfg = values()
    res = Registry([PossessionPackage(), StructurePackage()]).run(m, config=base_cfg)
    ref = sc_phase_spans(res.frames)
    per = res.frames["period"].to_numpy()
    pkg = PhasePackage()

    def run(cfg: dict) -> dict:
        ctx = ContextInput(match=m, grid=res.frames[["match_id", "frame_ts", "period", "native_frame", "clock_s", "minute"]], config=cfg,
                           upstream={"p1": res.outputs["p1"], "p2": res.outputs["p2"]})
        f = pkg.compute(ctx).frames
        f["p4_team"] = f["p4_team"]
        return {"all": agreement(ref, f), "half_1": agreement(ref, f, per == 1), "half_2": agreement(ref, f, per == 2)}, f

    out = {"reference_coverage": round(float(ref["sc_phase"].notna().mean()), 4), "baseline": None, "sweeps": {}}
    base, f0 = run(base_cfg)
    out["baseline"] = {"config": {k: v for k, v in base_cfg.items() if k.startswith("p4_")}, **base, "confusion": confusion(ref, f0)}
    for name, grid in GRIDS.items():
        rows = []
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            cfg = values({**dict(zip(keys, combo, strict=True))})
            r, _ = run(cfg)
            rows.append({**dict(zip(keys, combo, strict=True)), **{f"{k}_{m_}": v[m_] for k, v in r.items() for m_ in ("accuracy", "kappa")}})
            print(name, dict(zip(keys, combo, strict=True)), r["all"], flush=True)
        out["sweeps"][name] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
