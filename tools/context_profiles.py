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
    python tools/context_profiles.py build --source pff --data ~/ponta-reference-data/pff-worldcup2022 \
        --match-cache ~/ponta-reference-data/cache/pff-matches --cache ~/ponta-reference-data/cache/stage-e --step 3
    python tools/context_profiles.py build --source pff ... --step 3 --refs-only   # PFF's own figures only, from cache
    python tools/context_profiles.py analyse --cache ~/ponta-reference-data/cache/stage-e --out /tmp/ponta-context/profiles.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import zlib
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
from pff import load_pff_match
from pff import match_ids as pff_match_ids


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
    elif args.source == "pff":
        items = pff_match_ids(args.data)
    else:
        items = list(MATCH_IDS)
    if args.matches:
        wanted = set(args.matches.split(","))
        items = [it for it in items if (it.name if args.source == "skillcorner" else it) in wanted]
    i, n = (int(x) for x in args.shard.split("/"))
    for it in items[i::n]:
        name = it.name if args.source == "skillcorner" else it
        off = None
        out = args.cache / f"{args.source}_{name}.parquet"
        if args.refs_only:
            # Recompute only the reference's own figures on a cached profile (the context measures are untouched).
            if args.source != "pff" or not out.exists():
                sys.exit("--refs-only is for cached PFF profiles")
            m, off = load_pff_match(args.data, it, args.match_cache, ball=args.ball, step=args.step)
            pr = pd.read_parquet(out)
            pr = pr[[c for c in pr.columns if not c.startswith("ref_")]].merge(reference_figures(args.source, it, m, None, off), on="team", how="left")
            pr.to_parquet(out, index=False)
            print(name, "reference figures refreshed", flush=True)
            continue
        if out.exists() and not args.force:
            print(name, "cached", flush=True)
            continue
        if args.source == "skillcorner":
            m = load_sc(it, args.match_cache)
        elif args.source == "pff":
            m, off = load_pff_match(args.data, it, args.match_cache, ball=args.ball, step=args.step)
        else:
            m, off = load_idsse_match(args.data, it, args.match_cache)
            if "End X" not in m.events:
                m = dataclasses.replace(m, events=m.events.assign(**{"End X": np.nan, "End Y": np.nan}))
            if "Start Time [s]" not in m.events:  # the league's event table has frames only; the builder reads times
                clock = m.frames.set_index("frame")["time_s"]
                m = dataclasses.replace(m, events=m.events.assign(**{"Start Time [s]": m.events["Start Frame"].map(clock)}))
        pr, res = _profiles_for(m, cfg)
        ref = reference_figures(args.source, it, m, res, off)
        pr = pr.merge(ref, on="team", how="left") if len(ref) else pr
        pr.to_parquet(out, index=False)
        print(name, len(pr), "profile rows", flush=True)


def reference_figures(source: str, it, m, res, off) -> pd.DataFrame:
    """Per team: the reference's own phase shares and pressing chains (SkillCorner) or official possession (league)."""
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
    elif source == "pff":
        # PFF's own event data (comparison only): on-ball time, pressure applied, lines broken. Source: PFF FC.
        ev = off.events
        minutes = len(m.frames) / m.fps / 60.0
        on_ball = ev.loc[ev["gev"] == "OTB"].drop_duplicates("game_event_id")
        total = float(on_ball["duration"].fillna(0).sum())
        pressed = ev.loc[ev["pressure"].isin(["P", "A"]) | ev["initial_pressure"].isin(["P", "A"])].drop_duplicates("possession_event_id")
        lines = ev.loc[ev["lines_broken"].notna() & ev["pev"].isin(["PA", "CR", "BC", "CH"])]
        for t in ("home", "away"):
            mine = lines.loc[lines["team"] == t, "lines_broken"].astype(str)
            # Pressure is applied while the opponents have the ball: a rate per minute of their on-ball time, so a team
            # that defends for longer does not score higher just for that (the per-90 count is kept for reference).
            opp_on_ball_min = float(on_ball.loc[on_ball["team"].notna() & (on_ball["team"] != t), "duration"].fillna(0).sum()) / 60.0
            rows.append({"team": t,
                         "ref_possession_share": float(on_ball.loc[on_ball["team"] == t, "duration"].fillna(0).sum() / total) if total else np.nan,
                         "ref_pressures_p90": float((pressed["pressure_team"] == t).sum() / minutes * 90) if minutes else np.nan,
                         "ref_pressures_per_opp_on_ball_min": float((pressed["pressure_team"] == t).sum() / opp_on_ball_min) if opp_on_ball_min else np.nan,
                         "ref_dropped_periods": ",".join(map(str, off.dropped_periods)), "ref_events_without_frames": off.events_without_frames,
                         "ref_lines_broken_midfield_p90": float(mine.str.contains("M").sum() / minutes * 90) if minutes else np.nan,
                         "ref_lines_broken_defence_p90": float(mine.str.contains("D").sum() / minutes * 90) if minutes else np.nan,
                         "ref_visible_share": off.visible_share, "ref_ball_share": off.ball_share, "ref_orientation_agrees": off.orientation_agrees})
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


def icc1_interval(groups: list[np.ndarray], level: float = 0.90) -> tuple[float, float] | None:
    """
    The analytic interval for ICC(1) from the F distribution (Shrout and Fleiss 1979; McGraw and Wong 1996), with the
    unequal-size n0: deterministic, no resampling seed (review ST-04, 25 November 2026).
    """
    from scipy.stats import f as f_dist

    groups = [g for g in groups if len(g) >= 1]
    k = len(groups)
    n_i = np.array([len(g) for g in groups], float)
    N = n_i.sum()
    if k < 2 or N - k < 1:
        return None
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    if msw <= 0:
        return None
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    F, a = msb / msw, (1 - level) / 2
    fl = F / f_dist.ppf(1 - a, k - 1, N - k)
    fu = F * f_dist.ppf(1 - a, N - k, k - 1)
    return float((fl - 1) / (fl + n0 - 1)), float((fu - 1) / (fu + n0 - 1))


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
    for src in ("skillcorner", "pff"):
        part = pr.loc[pr["match_id"].str.startswith(src)]
        if part["match_id"].nunique():
            out[f"stability_{src}"] = stability(part, cfg, rng)
            out[f"baseline_vs_set_mean_{src}"] = baseline_help(part)
    _agreement(pr, sc, out)
    # Raw values beside rank agreement for every published and common measure (review AN-01): a level error (PPDA at 71)
    # is invisible to a rank correlation.
    out["value_ranges"] = {}
    for src, part in pr.groupby(pr["match_id"].str.split("_").str[0]):
        rows = {}
        for meas in CATALOGUE:
            if meas.basis not in ("published", "common"):
                continue
            v = part.loc[(part["measure"] == meas.name) & part["value"].notna(), "value"].astype(float)
            rows[meas.name] = ({"n": int(len(v)), "p10": round(float(v.quantile(0.1)), 3), "median": round(float(v.median()), 3),
                                "p90": round(float(v.quantile(0.9)), 3)} if len(v) else {"n": 0})
        out["value_ranges"][src] = rows
    out["checks"] = {"rows": int(len(pr)), "missing_label_or_basis": int(pr[["label", "basis", "data_check"]].isna().any(axis=1).sum()),
                     "blocked_share": round(float((pr["data_check"] == "blocked").mean()), 3),
                     "blocked_share_by_source": {src: round(float((g["data_check"] == "blocked").mean()), 3)
                                                 for src, g in pr.groupby(pr["match_id"].str.split("_").str[0])}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith(("stability_", "baseline_vs_set_mean"))}, indent=1, default=str))


#: Team resamples per measure for the bootstrap interval (review ST-04).
RESAMPLES = 5000


def stability(sc: pd.DataFrame, cfg: dict, rng=None) -> dict:
    """ICC(1) per measure on the teams with two or more usable matches, with a 90% interval from resampling teams."""
    stab = {}
    for meas in CATALOGUE:
        d = sc.loc[(sc["measure"] == meas.name) & sc["value"].notna() & (sc["data_check"] != "blocked")]
        g = {k: v["value"].to_numpy(float) for k, v in d.groupby("team_key") if len(v) >= 2}
        if len(g) < 3:
            stab[meas.name] = {"teams": len(g), "icc": None}
            continue
        icc = icc1(list(g.values()))
        # 5,000 resamples of teams with a fixed seed per measure (review ST-04: 500 left the selection seed-dependent),
        # beside the analytic F interval, which needs no seed.
        boots = []
        keys = list(g)
        r = np.random.default_rng(zlib.crc32(meas.name.encode()))
        for _ in range(RESAMPLES):
            pick = r.choice(len(keys), len(keys), replace=True)
            boots.append(icc1([g[keys[i]] for i in pick]))
        boots = np.array([b for b in boots if np.isfinite(b)])
        f_int = icc1_interval(list(g.values()))
        target = cfg["pf_reliability_target"]
        stab[meas.name] = {"teams": len(g), "team_matches": int(sum(len(v) for v in g.values())), "icc": round(icc, 3) if np.isfinite(icc) else None,
                           "icc_90pct": [round(float(np.percentile(boots, 5)), 3), round(float(np.percentile(boots, 95)), 3)] if len(boots) else None,
                           "icc_90pct_f": [round(f_int[0], 3), round(f_int[1], 3)] if f_int else None,
                           "matches_for_0.7": matches_needed(icc, target), "matches_for_0.8": matches_needed(icc, 0.8),
                           # The minimum as a range: matches needed at the interval's upper and lower ends.
                           "matches_for_0.7_range": [matches_needed(f_int[1], target), matches_needed(f_int[0], target)] if f_int else None,
                           "basis": meas.basis}
    return stab


def _spearman(d: pd.DataFrame, a: str, b: str) -> dict:
    d = d[[a, b]].dropna()
    return {"n": int(len(d)), "spearman": round(float(d[a].rank().corr(d[b].rank())), 3) if len(d) > 5 else None}


def _agreement(pr: pd.DataFrame, sc: pd.DataFrame, out: dict) -> None:
    """Agreement with each reference's own figures (comparison only)."""
    agree = {}
    for p in PHASES:
        d = sc.loc[sc["measure"] == f"phase_share_{p}", ["value", f"ref_phase_share_{p}"]].dropna()
        agree[f"phase_share_{p}"] = {"n": int(len(d)), "spearman": round(float(d["value"].rank().corr(d[f"ref_phase_share_{p}"].rank())), 3) if len(d) > 5 else None}
    d = sc.loc[sc["measure"] == "presses_per_min_out"].assign(count=lambda x: x["value"] * x["sample_minutes"])[["count", "ref_pressing_chains"]].dropna()
    agree["presses_vs_pressing_chains"] = {"n": int(len(d)), "spearman": round(float(d["count"].rank().corr(d["ref_pressing_chains"].rank())), 3) if len(d) > 5 else None}
    ids = pr.loc[pr["match_id"].str.startswith("idsse") & (pr["measure"] == "possession_share"), ["value", "ref_possession_share"]].dropna()
    agree["possession_share_vs_official"] = {"n": int(len(ids)), "mean_abs_diff": round(float((ids["value"] - ids["ref_possession_share"]).abs().mean()), 4) if len(ids) else None,
                                             "pearson": round(float(ids["value"].corr(ids["ref_possession_share"])), 3) if len(ids) > 3 else None}
    pf = pr.loc[pr["match_id"].str.startswith("pff")]
    if len(pf):
        def wide(measure: str, ref: str) -> pd.DataFrame:
            return pf.loc[pf["measure"] == measure, ["value", ref]].rename(columns={"value": measure})
        d = wide("possession_share", "ref_possession_share").dropna()
        agree["pff_possession_share_vs_on_ball_time"] = {
            "n": int(len(d)), "pearson": round(float(d["possession_share"].corr(d["ref_possession_share"])), 3) if len(d) > 3 else None,
            "mean_abs_diff": round(float((d["possession_share"] - d["ref_possession_share"]).abs().mean()), 4) if len(d) else None}
        # PFF's pressure rate per minute of the opponents' on-ball time (the per-90 count mixes in time out of possession).
        eng = [c for c in pf["measure"].unique() if c.startswith("engagements_") and c.endswith("_per_min_out")]
        tot = pf.loc[pf["measure"].isin(eng)].groupby(["match_id", "team"], as_index=False).agg(
            engagements_per_min_out=("value", "sum"), ref=("ref_pressures_per_opp_on_ball_min", "first"))
        for meas in ("engagements_pressure_per_min_out", "engagements_counter_press_per_min_out", "presses_per_min_out",
                     "carrier_pressure_median_out", "ppda"):
            agree[f"pff_{meas}_vs_pressure_rate"] = _spearman(wide(meas, "ref_pressures_per_opp_on_ball_min"), meas, "ref_pressures_per_opp_on_ball_min")
        agree["pff_engagements_total_per_min_out_vs_pressure_rate"] = _spearman(
            tot.rename(columns={"ref": "ref_pressures_per_opp_on_ball_min"}), "engagements_per_min_out", "ref_pressures_per_opp_on_ball_min")
        agree["pff_pressure_rate_vs_own_possession"] = _spearman(
            wide("possession_share", "ref_pressures_per_opp_on_ball_min"), "possession_share", "ref_pressures_per_opp_on_ball_min")
        agree["pff_receptions_behind_midfield_vs_lines_broken"] = _spearman(
            wide("efi_receptions_behind_midfield_p90", "ref_lines_broken_midfield_p90"), "efi_receptions_behind_midfield_p90", "ref_lines_broken_midfield_p90")
        agree["pff_receptions_behind_defence_vs_lines_broken"] = _spearman(
            wide("efi_receptions_behind_defence_p90", "ref_lines_broken_defence_p90"), "efi_receptions_behind_defence_p90", "ref_lines_broken_defence_p90")
        q = pf.drop_duplicates(["match_id", "team"])
        agree["pff_tracking"] = {"visible_share_median": round(float(q["ref_visible_share"].median()), 3),
                                 "ball_share_median": round(float(q["ref_ball_share"].median()), 3),
                                 "orientation_agrees": int(q.drop_duplicates("match_id")["ref_orientation_agrees"].sum()),
                                 "periods_dropped": {k: v for k, v in q.drop_duplicates("match_id").set_index("match_id")["ref_dropped_periods"].items() if v},
                                 "events_without_frames": {k: int(v) for k, v in q.drop_duplicates("match_id").set_index("match_id")["ref_events_without_frames"].items() if v},
                                 "matches": int(q["match_id"].nunique())}
    out["agreement"] = agree


def baseline_help(sc: pd.DataFrame) -> dict:
    """Leaving one match out, the shrunk team baseline's error against the set mean's, on teams with three or more matches."""
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
    return help_


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", choices=("skillcorner", "idsse", "pff"), required=True)
    b.add_argument("--ball", choices=("tracked", "smoothed"), default="tracked", help="PFF only: the tracked ball, or PFF's event-placed ball")
    b.add_argument("--step", type=int, default=1, help="PFF only: keep every n-th video frame (3 gives 9.99 Hz, SkillCorner's rate)")
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--match-cache", type=Path, required=True)
    b.add_argument("--cache", type=Path, required=True)
    b.add_argument("--force", action="store_true")
    b.add_argument("--shard", default="0/1", help="i/n: build every n-th match from the i-th, so shards can run in parallel")
    b.add_argument("--matches", default="", help="comma-separated match ids: build only these")
    b.add_argument("--refs-only", action="store_true", help="PFF only: recompute the reference columns of cached profiles")
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
