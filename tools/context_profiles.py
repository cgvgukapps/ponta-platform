"""
context_profiles.py — Stage E measurements (``ponta-blueprint/context-profiles-task.md``, deliverable 6).

Two steps, so the slow one runs once:

- ``build``: for every match of a source, run every context package (one shared ``MatchAnalysis``), the semantic builder
  and ``profiles.match_profiles``, and cache the profiles (and a few reference figures per team and match) as parquet in
  ``--cache`` (outside the repository; no positions, no names).
- ``analyse``: read the cache and report, per measure:
  - **stability**: the intraclass correlation ICC(1) across teams (one-way random effects, unequal group sizes) on the
    teams that recur, with a 90% interval from resampling teams; the reliability of a team's mean over n matches by
    Spearman-Brown, and the matches needed to reach ``pf_reliability_target`` (0.7) and 0.8;
  - **agreement with references** (comparison only): phase shares against SkillCorner's own phases-of-play shares, and
    press counts against its pressing chains, per team and match; possession share against the German league's official
    possession time;
  - **does a baseline help?**: leaving one match out, the shrunk team baseline's error against the set mean's, on teams
    with three or more matches;
  - checks: every profile row's Data check, label and basis present; versions identical across the cache.

    python tools/context_profiles.py build --source skillcorner --data ~/ponta-reference-data/skillcorner-opendata \
        --match-cache /tmp/ponta-context/sc-cache --cache ~/ponta-reference-data/cache/stage-e
    python tools/context_profiles.py build --source idsse --data ~/ponta-reference-data/idsse \
        --match-cache /tmp/ponta-context/idsse-cache --cache ~/ponta-reference-data/cache/stage-e
    python tools/context_profiles.py analyse --cache ~/ponta-reference-data/cache/stage-e --out /tmp/ponta-context/profiles.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.parity import parity_for_match
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.thresholds import values
from plaiq_engine.match_analysis import MatchAnalysis
from plaiq_engine.profiles import CATALOGUE, PHASES, baseline
from plaiq_engine.semantic_builder import SemanticBuilder
from plaiq_engine.teams import TeamRegistry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_golden import registry
from context_reference_set import _load as load_sc
from idsse import MATCH_IDS, load_idsse_match


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def _profiles_for(m, cfg) -> tuple[pd.DataFrame, object]:
    from plaiq_engine.profiles import match_profiles

    ma = MatchAnalysis(m)
    res = registry().run(m, config=cfg, extras={"analysis": ma})
    sem = SemanticBuilder(ma, parity_for_match(m), context=res, teams=TeamRegistry()).semantic_events()
    return match_profiles(m, res, TeamRegistry().keys_for(m), sem), res


def build(args) -> None:
    cfg = values()
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.source == "skillcorner":
        items = [d for d in sorted((args.data / "matches").iterdir()) if (d / f"{d.name}_phases_of_play.csv").exists()]
    else:
        items = list(MATCH_IDS)
    i, n = (int(x) for x in args.shard.split("/"))
    for it in items[i::n]:
        name = it.name if args.source == "skillcorner" else it
        out = args.cache / f"{args.source}_{name}.parquet"
        if out.exists() and not args.force:
            print(name, "cached", flush=True)
            continue
        if args.source == "skillcorner":
            m = load_sc(it, args.match_cache)
        else:
            m, off = load_idsse_match(args.data, it, args.match_cache)
            if "End X" not in m.events:
                m = dataclasses.replace(m, events=m.events.assign(**{"End X": np.nan, "End Y": np.nan}))
            if "Start Time [s]" not in m.events:  # the league's event table has frames only; the builder reads times
                clock = m.frames.set_index("frame")["time_s"]
                m = dataclasses.replace(m, events=m.events.assign(**{"Start Time [s]": m.events["Start Frame"].map(clock)}))
        pr, res = _profiles_for(m, cfg)
        ref = reference_figures(args.source, it, m, res, off if args.source == "idsse" else None)
        pr = pr.merge(ref, on="team", how="left") if len(ref) else pr
        pr.to_parquet(out, index=False)
        print(name, len(pr), "profile rows", flush=True)


def reference_figures(source: str, it, m, res, off) -> pd.DataFrame:
    """Per team: the reference's own phase shares and pressing chains (SkillCorner) or official possession (league)."""
    g = res.frames
    nf, ts = g["native_frame"].to_numpy(float), g["frame_ts"].to_numpy(float)
    rows = []
    if source == "skillcorner":
        meta = json.loads((it / f"{it.name}_match.json").read_text())
        home = meta["home_team"]["id"]
        ph = pd.read_csv(it / f"{it.name}_phases_of_play.csv", usecols=["frame_start", "frame_end", "team_in_possession_id", "team_in_possession_phase_type"])
        ph = ph.assign(team=np.where(ph["team_in_possession_id"] == home, "home", "away"), dur=(ph["frame_end"] - ph["frame_start"]) / 10.0)
        ev = pd.read_csv(it / f"{it.name}_dynamic_events.csv", low_memory=False, usecols=["event_type", "team_id", "pressing_chain", "pressing_chain_index"])
        ch = ev.loc[(ev["event_type"] == "on_ball_engagement") & (ev["pressing_chain"] == True)]
        for t in ("home", "away"):
            mine = ph.loc[ph["team"] == t]
            tot = mine.loc[mine["team_in_possession_phase_type"].isin(PHASES), "dur"].sum()
            row = {"team": t, "ref_pressing_chains": int(ch.loc[(ch["team_id"] == home) == (t == "home"), "pressing_chain_index"].nunique())}
            for p in PHASES:
                row[f"ref_phase_share_{p}"] = float(mine.loc[mine["team_in_possession_phase_type"] == p, "dur"].sum() / tot) if tot else np.nan
            rows.append(row)
    else:
        b = off.ball
        live = b["ball_status"].eq(1)
        for t in ("home", "away"):
            rows.append({"team": t, "ref_possession_share": float((b.loc[live, "ball_team"] == t).mean())})
    return pd.DataFrame(rows)


def icc1(groups: list[np.ndarray]) -> float:
    """One-way random-effects ICC(1) with unequal group sizes."""
    groups = [g for g in groups if len(g) >= 1]
    k = len(groups)
    n_i = np.array([len(g) for g in groups], float)
    N = n_i.sum()
    if k < 2 or N - k < 1:
        return np.nan
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else np.nan


def matches_needed(icc: float, target: float) -> float | None:
    if not np.isfinite(icc) or icc <= 0:
        return None
    return float(np.ceil(target * (1 - icc) / (icc * (1 - target))))


def analyse(args) -> None:
    cfg = values()
    files = sorted(args.cache.glob("*.parquet"))
    pr = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out: dict = {"matches": int(pr["match_id"].nunique()), "team_matches": int(pr[["match_id", "team"]].drop_duplicates().shape[0]),
                 "versions_identical": bool(pr["versions"].nunique() == 1)}
    rng = np.random.default_rng(1)
    sc = pr.loc[pr["match_id"].str.startswith("skillcorner")]
    stab = {}
    for meas in CATALOGUE:
        d = sc.loc[(sc["measure"] == meas.name) & sc["value"].notna() & (sc["data_check"] != "blocked")]
        g = {k: v["value"].to_numpy(float) for k, v in d.groupby("team_key") if len(v) >= 2}
        if len(g) < 3:
            stab[meas.name] = {"teams": len(g), "icc": None}
            continue
        icc = icc1(list(g.values()))
        boots = []
        keys = list(g)
        for _ in range(500):
            pick = rng.choice(len(keys), len(keys), replace=True)
            boots.append(icc1([g[keys[i]] for i in pick]))
        boots = np.array([b for b in boots if np.isfinite(b)])
        stab[meas.name] = {"teams": len(g), "team_matches": int(sum(len(v) for v in g.values())), "icc": round(icc, 3) if np.isfinite(icc) else None,
                           "icc_90pct": [round(float(np.percentile(boots, 5)), 3), round(float(np.percentile(boots, 95)), 3)] if len(boots) else None,
                           "matches_for_0.7": matches_needed(icc, cfg["pf_reliability_target"]), "matches_for_0.8": matches_needed(icc, 0.8),
                           "basis": meas.basis}
    out["stability_skillcorner"] = stab
    # Agreement with the references.
    agree = {}
    for p in PHASES:
        d = sc.loc[sc["measure"] == f"phase_share_{p}", ["value", f"ref_phase_share_{p}"]].dropna()
        agree[f"phase_share_{p}"] = {"n": int(len(d)), "spearman": round(float(d["value"].rank().corr(d[f"ref_phase_share_{p}"].rank())), 3) if len(d) > 5 else None}
    d = sc.loc[sc["measure"] == "presses_per_min_out"].assign(count=lambda x: x["value"] * x["sample_minutes"])[["count", "ref_pressing_chains"]].dropna()
    agree["presses_vs_pressing_chains"] = {"n": int(len(d)), "spearman": round(float(d["count"].rank().corr(d["ref_pressing_chains"].rank())), 3) if len(d) > 5 else None}
    ids = pr.loc[pr["match_id"].str.startswith("idsse") & (pr["measure"] == "possession_share"), ["value", "ref_possession_share"]].dropna()
    agree["possession_share_vs_official"] = {"n": int(len(ids)), "mean_abs_diff": round(float((ids["value"] - ids["ref_possession_share"]).abs().mean()), 4) if len(ids) else None,
                                             "pearson": round(float(ids["value"].corr(ids["ref_possession_share"])), 3) if len(ids) > 3 else None}
    out["agreement"] = agree
    # Does a baseline help? Leave one match out on teams with three or more matches.
    help_ = {}
    for meas in CATALOGUE:
        d = sc.loc[(sc["measure"] == meas.name) & sc["value"].notna() & (sc["data_check"] != "blocked")]
        teams = [k for k, v in d.groupby("team_key") if len(v) >= 3]
        err_b, err_s = [], []
        for t in teams:
            for mid in d.loc[d["team_key"] == t, "match_id"]:
                b = baseline(d, t, meas.name, exclude_match=mid)
                if "refused" in b:
                    continue
                v = float(d.loc[(d["team_key"] == t) & (d["match_id"] == mid), "value"].iloc[0])
                err_b.append(abs(v - b["shrunk"]))
                err_s.append(abs(v - b["set_mean"]))
        if err_b:
            help_[meas.name] = {"n": len(err_b), "mae_baseline": round(float(np.mean(err_b)), 4), "mae_set_mean": round(float(np.mean(err_s)), 4),
                                "error_reduction": round(1 - float(np.mean(err_b)) / float(np.mean(err_s)), 3) if np.mean(err_s) > 0 else None}
    out["baseline_vs_set_mean"] = help_
    out["checks"] = {"rows": int(len(pr)), "missing_label_or_basis": int(pr[["label", "basis", "data_check"]].isna().any(axis=1).sum()),
                     "blocked_share": round(float((pr["data_check"] == "blocked").mean()), 3)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("stability_skillcorner", "baseline_vs_set_mean")}, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", choices=("skillcorner", "idsse"), required=True)
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--match-cache", type=Path, required=True)
    b.add_argument("--cache", type=Path, required=True)
    b.add_argument("--force", action="store_true")
    b.add_argument("--shard", default="0/1", help="i/n: build every n-th match from the i-th, so shards can run in parallel")
    a = sub.add_parser("analyse")
    a.add_argument("--cache", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for p in [x for x in (getattr(args, "data", None), getattr(args, "match_cache", None), args.cache, getattr(args, "out", None)) if x]:
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    build(args) if args.cmd == "build" else analyse(args)


if __name__ == "__main__":
    main()
