"""
plaiq_pass_probability.py — the probabilistic, height-aware pass model fitted and tested on tracking.

Follows step 6 of ``ponta-blueprint/plaiq-review.md``: targeted pitch control's constants were measured
and did not close the gap to a provider's completion model, so ``analytics/pass_probability.py``
(after Spearman 2018) rebuilds its structure. This fits the model and tests it:

- **targets:** where the recipient stands at the kick, or projected along their run to the ball's
  arrival (``projected_target``, positions and velocities only);
- **passes:** the same as step 6 (``plaiq_calibration.py``). IDSSE's open-play passes with the
  league's recipient, the kick found in the tracking, the target where the recipient stands at the
  kick, the league's flat or high flag; SkillCorner's passes to the targeted player at the pass, with
  its ``high_pass`` flag;
- **fit:** maximum likelihood (log-loss) on IDSSE's even matches, for the arrival spread, the two
  control rates, the two reaction times and the reachable share of a high ball; the ball flight is the
  measured one, not fitted again;
- **test:** IDSSE's odd matches, then all of SkillCorner's 20 matches (another league and provider);
  and, as a ceiling for that provider, a refit on SkillCorner's even matches tested on its odd ones;
- **measures:** log-loss, Brier score, AUC and observed completion by predicted probability, against
  the base rate, v1's availability (recalibrated to a probability on the fit matches, so it is judged
  fairly) and, on SkillCorner, SkillCorner's own expected completion.

Comparison data only (founder decision 2); data, caches and output stay outside the repository; no
names are read.

    python tools/plaiq_pass_probability.py --idsse ~/ponta-reference-data/idsse \
        --sc-data ~/ponta-reference-data/skillcorner-opendata --cache /tmp/ponta-context --out /tmp/ponta-context/pass_probability.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.analytics.ball_flight import V1
from plaiq_engine.analytics.pass_probability import (
    PassModel,
    pass_arrays,
    probability_from_arrays,
    projected_target,
)
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idsse import MATCH_IDS
from plaiq_calibration import availability, idsse_passes, skillcorner_rows
from plaiq_references import auc

FREE = ("sigma", "lam", "lam_flight", "rt_att", "rt_def", "high_reach")
BOUNDS = {"sigma": (0.05, 2.0), "lam": (0.1, 30.0), "lam_flight": (0.01, 30.0), "rt_att": (0.0, 1.5), "rt_def": (0.0, 1.5), "high_reach": (0.0, 0.5)}
KINEMATICS = {"v1": (9.0, 5.5), "measured": (8.35, 4.69)}  # measured: step 6, median of IDSSE's fit matches
BINS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001)


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    ok = np.isfinite(p)
    p, y = p[ok], y[ok].astype(float)
    table = [{"predicted": f"{lo:g}-{min(hi, 1):g}", "passes": int(((p >= lo) & (p < hi)).sum()),
              "mean_predicted": round(float(p[(p >= lo) & (p < hi)].mean()), 3) if ((p >= lo) & (p < hi)).any() else None,
              "completed": round(float(y[(p >= lo) & (p < hi)].mean()), 3) if ((p >= lo) & (p < hi)).any() else None}
             for lo, hi in pairwise(BINS)]
    return {"passes": int(len(p)), "log_loss": round(log_loss(p, y), 4), "brier": round(float(np.mean((p - y) ** 2)), 4),
            "auc": auc(p, y.astype(bool)), "mean_predicted": round(float(p.mean()), 3), "completed": round(float(y.mean()), 3), "by_predicted": table}


def recalibrate(x_fit: np.ndarray, y_fit: np.ndarray):
    """A logistic map of a score to a probability, fitted on the fit matches (to judge v1's availability fairly)."""
    ok = np.isfinite(x_fit)
    x, y = x_fit[ok], y_fit[ok].astype(float)
    res = minimize(lambda w: log_loss(1 / (1 + np.exp(-(w[0] + w[1] * x))), y), x0=[0.0, 1.0], method="Nelder-Mead")
    return lambda z: 1 / (1 + np.exp(-(res.x[0] + res.x[1] * z)))


def fit(arrays, y: np.ndarray, base: PassModel, free: tuple[str, ...]) -> PassModel:
    def model_of(theta):
        return replace(base, **{k: float(np.clip(v, *BOUNDS[k])) for k, v in zip(free, theta, strict=True)})

    def loss(theta):
        return log_loss(probability_from_arrays(arrays, model_of(theta))["p_complete"], y)

    res = minimize(loss, x0=[getattr(base, k) for k in free], method="Powell", bounds=[BOUNDS[k] for k in free],
                   options={"xtol": 1e-3, "ftol": 1e-5, "maxfev": 3000})
    return model_of(res.x)


def _target(r, projected: bool) -> tuple:
    """The recipient at the kick, or projected along their run to the ball's arrival (their velocity from the tracking)."""
    if not projected:
        return r.recipient_at_kick
    rec = np.asarray(r.recipient_at_kick, float)
    me = min(r.attackers, key=lambda p: float(np.hypot(*(np.asarray(p["pos"], float) - rec))))
    return tuple(projected_target(r.ball, rec, me["vel"]))


def as_passes(df: pd.DataFrame, use_height: bool, projected: bool = False) -> list[dict]:
    return [{"ball": r.ball, "target": _target(r, projected), "attackers": r.attackers, "defenders": r.defenders,
             "high": bool(r.high) and use_height} for r in df.itertuples()]


def usable(df: pd.DataFrame) -> pd.DataFrame:
    ok = df["defenders"].map(len).gt(0) & df["attackers"].map(len).gt(0) \
        & df["recipient_at_kick"].map(lambda t: bool(np.isfinite(np.asarray(t, float)).all()))
    return df.loc[ok].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idsse", type=Path, required=True)
    ap.add_argument("--sc-data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.idsse, args.sc_data, args.cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    ipk, spk = args.cache / "pass_probability_idsse.pkl", args.cache / "pass_probability_skillcorner.pkl"
    if ipk.exists():
        idsse = pd.read_pickle(ipk)
    else:
        idsse = pd.concat([idsse_passes(args.idsse, args.cache, mid)[0] for mid in MATCH_IDS], ignore_index=True)
        idsse.to_pickle(ipk)
    if spk.exists():
        sc = pd.read_pickle(spk)
    else:
        sc = pd.concat([skillcorner_rows(d, args.cache / "sc-cache") for d in sorted((args.sc_data / "matches").iterdir())
                        if (d / f"{d.name}_dynamic_events.csv").exists()], ignore_index=True)
        sc.to_pickle(spk)
    print("passes loaded", flush=True)
    half = {mid: ("fit" if i % 2 == 0 else "test") for i, mid in enumerate(MATCH_IDS)}
    idsse = usable(idsse.loc[idsse["kick"]].assign(high=lambda d: d["height"] == "high", half=lambda d: d["match"].map(half)))
    sc = usable(sc)
    sc_order = {m: i for i, m in enumerate(sorted(sc["match"].unique()))}
    sc["half"] = sc["match"].map(lambda m: "fit" if sc_order[m] % 2 == 0 else "test")
    y_i, y_s = idsse["completed"].to_numpy(), sc["completed"].to_numpy()
    fit_i, fit_s = (idsse["half"] == "fit").to_numpy(), (sc["half"] == "fit").to_numpy()

    out: dict = {"passes": {"idsse_fit": int(fit_i.sum()), "idsse_test": int((~fit_i).sum()), "skillcorner": int(len(sc)),
                            "idsse_high_share": round(float(idsse["high"].mean()), 3), "skillcorner_high_share": round(float(sc["high"].mean()), 3)}}

    # Baselines.
    base_rate = float(y_i[fit_i].mean())
    a_i, a_s = availability(idsse, V1), availability(sc, V1)
    to_p = recalibrate(a_i[fit_i], y_i[fit_i])
    out["baselines"] = {
        "idsse_test": {"base_rate": scores(np.full((~fit_i).sum(), base_rate), y_i[~fit_i]), "v1_availability_recalibrated": scores(to_p(a_i[~fit_i]), y_i[~fit_i])},
        "skillcorner": {"base_rate": scores(np.full(len(sc), base_rate), y_s), "v1_availability_recalibrated": scores(to_p(a_s), y_s),
                        "skillcorner_xpass": scores(sc["sc_xpass"].to_numpy(float), y_s)},
    }
    print("baselines done", flush=True)

    # Candidates.
    variants = {"published_no_height": ("v1", False, False, False), "published_height": ("v1", True, False, False),
                "fitted_no_height": ("v1", False, True, False), "fitted_height": ("v1", True, True, False),
                "fitted_height_measured_players": ("measured", True, True, False),
                "fitted_no_height_projected": ("v1", False, True, True), "fitted_height_projected": ("v1", True, True, True),
                "fitted_height_projected_measured_players": ("measured", True, True, True)}
    arrays_cache: dict = {}

    def arrays(df, kin, height, key, projected=False):
        k = (key, kin, height, projected)
        if k not in arrays_cache:
            vmax, amax = KINEMATICS[kin]
            m = replace(PassModel(), v_max=vmax, a_max=amax)
            arrays_cache[k] = pass_arrays(as_passes(df, height, projected), m)
        return arrays_cache[k]

    def subset(a, mask):
        return type(a)(**{k: v[mask] for k, v in a.__dict__.items()})

    results, models = {}, {}
    for name, (kin, height, fitted, projected) in variants.items():
        vmax, amax = KINEMATICS[kin]
        base = replace(PassModel(), v_max=vmax, a_max=amax)
        ai, asc = arrays(idsse, kin, height, "idsse", projected), arrays(sc, kin, height, "sc", projected)
        model = fit(subset(ai, fit_i), y_i[fit_i], base, FREE if height else tuple(k for k in FREE if k != "high_reach")) if fitted else base
        models[name] = model
        p_i, p_s = probability_from_arrays(ai, model)["p_complete"], probability_from_arrays(asc, model)["p_complete"]
        results[name] = {"parameters": {k: round(getattr(model, k), 3) for k in FREE} | {"v_max": model.v_max, "a_max": model.a_max},
                         "idsse_fit": scores(p_i[fit_i], y_i[fit_i]), "idsse_test": scores(p_i[~fit_i], y_i[~fit_i]), "skillcorner": scores(p_s, y_s)}
        print(name, json.dumps({k: results[name][k]["log_loss"] for k in ("idsse_fit", "idsse_test", "skillcorner")}),
              "auc", results[name]["idsse_test"]["auc"], results[name]["skillcorner"]["auc"], flush=True)
    out["candidates"] = results

    # A ceiling for the provider: the same structure refitted on SkillCorner's even matches, tested on its odd ones.
    out["skillcorner_refit"] = {}
    for name, projected, height, start_name in (("target_at_pass", False, True, "fitted_height"),
                                                ("projected_target", True, True, "fitted_height_projected"),
                                                ("projected_target_no_height", True, False, "fitted_no_height_projected")):
        asc = arrays(sc, "v1", height, "sc", projected)
        start = models[start_name]
        sc_model = fit(subset(asc, fit_s), y_s[fit_s], start, FREE if height else tuple(k for k in FREE if k != "high_reach"))
        p = probability_from_arrays(asc, sc_model)["p_complete"]
        out["skillcorner_refit"][name] = {
            "parameters": {k: round(getattr(sc_model, k), 3) for k in FREE},
            "fit": scores(p[fit_s], y_s[fit_s]), "test": scores(p[~fit_s], y_s[~fit_s]),
            "skillcorner_xpass_test": scores(sc["sc_xpass"].to_numpy(float)[~fit_s], y_s[~fit_s]),
            "idsse_model_on_same_test": scores(probability_from_arrays(subset(asc, ~fit_s), start)["p_complete"], y_s[~fit_s])}
        print("refit", name, out["skillcorner_refit"][name]["test"]["log_loss"], out["skillcorner_refit"][name]["test"]["auc"], flush=True)
    out["flight"] = asdict(PassModel().flight)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
