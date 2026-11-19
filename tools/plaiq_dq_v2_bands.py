"""
plaiq_dq_v2_bands.py — candidate bands for decision quality v2 (projected, 1.0.0), measured, not agreed.

Decision quality v2 runs 0 to 1 and shipped with no bands (founder decision, 19 November 2026). Version 1's
bands are cut on its 0 to 1.5 scale. This measures candidate cut-off sets on the rows
``plaiq_dq_v2.py`` writes (``dq_v2_rows.pkl``: every SkillCorner pass of its 20 open-data matches, with
the shipped v2 score, the outcome of the possession, and SkillCorner's own analogue). Candidates:

- **v1_cuts:** version 1's cut-offs (0.55, 0.80, 0.95) applied as they are, as a baseline;
- **meaning:** cut-offs that say what the score means: under half the best option's expected value is poor,
  under three-quarters suboptimal, the best option (within 5%) optimal (0.50, 0.75, 0.95);
- **fitted:** three cut-offs on a 0.05 grid chosen on the even matches only, maximising the likelihood of the
  possession leading to a shot (completed passes), with every band holding at least 10% of passes and the
  shot rate rising band by band; read on the odd matches;
- **fitted_top_anchored:** as ``fitted`` with the top cut held at 0.95, so optimal keeps its meaning.

Version 1's bands on version 1's scores are reported beside them as the reference. Measures per half (even,
odd) and in all: band shares (all, completed and failed passes); the shot rate per band on completed passes
with a Wilson interval, and on all passes; the goal rate; whether the shot rate rises band by band, and in
what share of 1,000 resamples of the matches it does; the bands' AUC for a shot; SkillCorner's
chose-the-best rate per band and a quadratic-weighted kappa against SkillCorner's analogue banded at the same
cut-offs. Intervals treat passes as independent, so they overstate precision (passes in one possession
share an outcome); the match resamples do not.

Comparison data only (founder decision 2); reads and writes outside the repository; no names are read.

    python tools/plaiq_dq_v2.py ... --out /tmp/ponta-context/dq_v2.json      # writes dq_v2_rows.pkl beside it
    python tools/plaiq_dq_v2_bands.py --rows /tmp/ponta-context/dq_v2_rows.pkl --out /tmp/ponta-context/dq_v2_bands.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plaiq_references import auc

BANDS = ("poor", "suboptimal", "good", "optimal")
V1_CUTS = (0.55, 0.80, 0.95)
CANDIDATES_FIXED = {"v1_cuts": V1_CUTS, "meaning": (0.50, 0.75, 0.95)}
GRID = tuple(round(x, 2) for x in np.arange(0.10, 0.96, 0.05))
MIN_SHARE = 0.10
RESAMPLES = 1000


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def band_index(score: np.ndarray, cuts) -> np.ndarray:
    return np.searchsorted(np.asarray(cuts, float), np.asarray(score, float), side="right")


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def weighted_kappa(a: np.ndarray, b: np.ndarray, k: int = 4) -> float:
    o = np.zeros((k, k))
    np.add.at(o, (a, b), 1)
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return round(float(1 - (w * o).sum() / (w * e).sum()), 4)


def _loglik(band: np.ndarray, y: np.ndarray) -> float:
    ll = 0.0
    for b in range(4):
        n, k = int((band == b).sum()), int(y[band == b].sum())
        if n and 0 < k < n:
            ll += k * np.log(k / n) + (n - k) * np.log(1 - k / n)
    return ll


def fit_cuts(d: pd.DataFrame, top: float | None = None) -> tuple[float, float, float] | None:
    """Cut-offs on the grid maximising the shot likelihood on completed passes (fit half only); None if none qualify."""
    s_all = d["score"].to_numpy()
    c = d.loc[d["completed"]]
    s, y = c["score"].to_numpy(), c["lead_to_shot"].to_numpy(bool)
    best, best_ll = None, -np.inf
    triples = (itertools.combinations(GRID, 3) if top is None
               else ((a, b, top) for a, b in itertools.combinations([g for g in GRID if g < top], 2)))
    for cuts in triples:
        shares = np.bincount(band_index(s_all, cuts), minlength=4) / len(s_all)
        if shares.min() < MIN_SHARE:
            continue
        band = band_index(s, cuts)
        rates = [y[band == b].mean() for b in range(4)]
        if not all(rates[i] < rates[i + 1] for i in range(3)):
            continue
        ll = _loglik(band, y)
        if ll > best_ll:
            best, best_ll = cuts, ll
    return None if best is None else tuple(float(x) for x in best)


def rises(d: pd.DataFrame, cuts) -> bool:
    c = d.loc[d["completed"]]
    band = band_index(c["score"], cuts)
    rates = [c["lead_to_shot"].to_numpy(bool)[band == b].mean() if (band == b).any() else np.nan for b in range(4)]
    return bool(all(rates[i] < rates[i + 1] for i in range(3)))


def measures(d: pd.DataFrame, cuts, rng: np.random.Generator) -> dict:
    band = band_index(d["score"], cuts)
    comp = d["completed"].to_numpy(bool)
    shot, goal = d["lead_to_shot"].to_numpy(bool), d["lead_to_goal"].to_numpy(bool)
    out = {"passes": int(len(d)), "per_band": {}}
    for b, name in enumerate(BANDS):
        m, mc = band == b, (band == b) & comp
        k = int(shot[mc].sum())
        out["per_band"][name] = {
            "share_all": round(float(m.mean()), 3), "share_completed": round(float(mc.sum() / comp.sum()), 3),
            "share_failed": round(float((m & ~comp).sum() / (~comp).sum()), 3),
            "shot_rate_completed": round(float(k / mc.sum()), 4) if mc.any() else None, "shot_ci": wilson(k, int(mc.sum())),
            "shot_rate_all": round(float(shot[m].mean()), 4) if m.any() else None,
            "goal_rate_completed": round(float(goal[mc].mean()), 4) if mc.any() else None, "goals": int(goal[mc].sum()),
        }
    c = d.loc[comp].assign(band=band[comp])
    out["shot_rate_rises"] = rises(d, cuts)
    matches = d["match"].unique()
    hits = 0
    for _ in range(RESAMPLES):
        pick = rng.choice(matches, size=len(matches), replace=True)
        hits += rises(pd.concat([d.loc[d["match"] == m] for m in pick], ignore_index=True), cuts)
    out["rises_in_match_resamples"] = round(hits / RESAMPLES, 3)
    out["auc_shot_completed_by_band"] = auc(c["band"], c["lead_to_shot"])
    out["auc_shot_all_by_band"] = auc(pd.Series(band), d["lead_to_shot"])
    k = c.dropna(subset=["sc_dq"])
    out["sc_chose_best_by_band"] = {BANDS[b]: round(float(g["sc_chose_best"].astype(bool).mean()), 3) for b, g in k.groupby("band")}
    out["kappa_vs_sc_analogue_same_cuts"] = weighted_kappa(k["band"].to_numpy(int), band_index(k["sc_dq"].clip(0, 1), cuts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.rows, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    rows = pd.read_pickle(args.rows)
    v2 = rows.dropna(subset=["projected"]).assign(score=lambda x: x["projected"].astype(float))
    v1 = rows.dropna(subset=["v1"]).assign(score=lambda x: x["v1"].astype(float))
    halves = {"even": v2.loc[v2["half"] == "even"], "odd": v2.loc[v2["half"] == "odd"], "all": v2}
    cands = dict(CANDIDATES_FIXED)
    cands["fitted"] = fit_cuts(halves["even"])
    cands["fitted_top_anchored"] = fit_cuts(halves["even"], top=0.95)
    cands = {k: v for k, v in cands.items() if v is not None}
    rng = np.random.default_rng(20261119)
    out = {"scored_passes": int(len(v2)), "completed": int(v2["completed"].sum()), "fit_half": "even", "held_out_half": "odd",
           "candidates": {name: {"cuts": list(cuts), **{h: measures(d, cuts, rng) for h, d in halves.items()}}
                          for name, cuts in cands.items()},
           "reference_v1_bands_on_v1": {"cuts": list(V1_CUTS), **{h: measures(v1.loc[v1["half"] == h] if h != "all" else v1, V1_CUTS, rng)
                                                                   for h in ("even", "odd", "all")}}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({n: c["cuts"] for n, c in out["candidates"].items()}, indent=1))


if __name__ == "__main__":
    main()
