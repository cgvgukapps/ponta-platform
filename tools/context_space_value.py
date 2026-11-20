"""
context_space_value.py — P7 0.1.0 against SkillCorner's open data (Stage C, deliverable 4).

On every SkillCorner open-data match (comparison only, founder decision 2), with P1, P2 and P7 run as shipped and one
shared ``MatchAnalysis`` (so Choice is computed once):

- **Context-adjusted Choice against Choice**, on the groundings Choice was adopted on (``tools/plaiq_dq_v2.py``):
  the possession leading to a shot (AUC, completed passes), rank correlation with SkillCorner's decision analogue (the
  targeted option's ``xpass_completion × xthreat`` over the best option's), and completion (AUC: below 0.5 leans to
  risk, above to safety). SkillCorner's matches have no logged score, so the game-state weight is neutral here: the
  comparison isolates the cost-of-losing-the-ball term.
- **Does the game-state weight point the right way?** SkillCorner's own game state and phases (comparison only): the
  share of a team's possessions that are ``direct``, by winning / drawing / losing and before or after minute 75.
- **Space sanity:** the open share of the team in possession's high-value space by SkillCorner's phase of play.
- Every measure on the even (fit) and odd (held-out) matches. Runtime of P7 per match.

Reads the data outside the repository and writes JSON outside it; no names are read.

    python tools/context_space_value.py --data ~/ponta-reference-data/skillcorner-opendata \
        --cache /tmp/ponta-context/sc-cache --out /tmp/ponta-context/space_value.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.space_value import SpaceValuePackage
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.match_analysis import MatchAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_reference_set import _load
from plaiq_references import auc, spearman


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def one(match_dir: Path, cache: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    m = _load(match_dir, cache)
    ma = MatchAnalysis(m)
    _ = ma.decision_quality_v2  # Choice first, as the semantic builder would; P7 then reuses its options
    t0 = time.perf_counter()
    res = Registry([PossessionPackage(), StructurePackage(), SpaceValuePackage()]).run(m, extras={"analysis": ma})
    runtime = time.perf_counter() - t0
    g, seg = res.frames, res.outputs["p7"].segments["adjusted_choice"]
    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    home = meta["home_team"]["id"]
    ev = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", low_memory=False,
                     usecols=["event_id", "event_type", "end_type", "frame_start", "frame_end", "team_id", "pass_outcome", "lead_to_shot",
                              "associated_player_possession_event_id", "xpass_completion", "xthreat", "targeted", "game_state",
                              "team_in_possession_phase_type", "minute_start"])
    # Adjusted against base Choice, joined to SkillCorner's passes (as plaiq_dq_v2.py matches them).
    opts = ev.loc[ev["event_type"] == "passing_option"].assign(evx=lambda o: o["xpass_completion"] * o["xthreat"])
    by_pass = {k: grp for k, grp in opts.groupby("associated_player_possession_event_id")}
    passes = ev.loc[(ev["event_type"] == "player_possession") & (ev["end_type"] == "pass") & ev["pass_outcome"].isin(["successful", "unsuccessful"])]
    starts = m.events.loc[seg["event_idx"], "Start Frame"].to_numpy() if len(seg) else np.array([])
    seg = seg.assign(start_frame=starts)
    rows = []
    for r in passes.itertuples():
        team = "home" if r.team_id == home else "away"
        c = seg.loc[(seg["team"] == team) & ((seg["start_frame"] - r.frame_end).abs() <= 2)]
        if c.empty:
            continue
        x = c.iloc[0]
        o = by_pass.get(r.event_id)
        sc_dq = np.nan
        if o is not None and o["targeted"].any() and o["evx"].max() > 0:
            sc_dq = float(o.loc[o["targeted"], "evx"].max() / o["evx"].max())
        rows.append({"match": match_dir.name, "completed": r.pass_outcome == "successful", "shot": bool(r.lead_to_shot),
                     "choice": x["choice"], "adjusted": x["adjusted"], "sc_dq": sc_dq})
    # Game state and direct play (SkillCorner's labels).
    pp = ev.loc[ev["event_type"] == "player_possession"].assign(late=lambda d: d["minute_start"] >= 75,
                                                               direct=lambda d: d["team_in_possession_phase_type"] == "direct")
    gs = pp[["game_state", "late", "direct"]].assign(match=match_dir.name)
    # Open space by SkillCorner's phase at its possessions' starts.
    nf = g["native_frame"].to_numpy(np.int64)
    k = np.clip(np.searchsorted(nf, pp["frame_start"].to_numpy(np.int64)), 0, len(nf) - 1)
    team = np.where(pp["team_id"].to_numpy() == home, "home", "away")
    # The open share is sampled every few seconds: take the latest sample at or before each start.
    open_ = {t: pd.Series(g[f"p7_{t}_high_value_open"].to_numpy(dtype=float, na_value=np.nan)).ffill().to_numpy() for t in ("home", "away")}
    sp = pd.DataFrame({"match": match_dir.name, "phase": pp["team_in_possession_phase_type"].to_numpy(),
                       "open": [open_[t][i] for t, i in zip(team, k, strict=True)]})
    return pd.DataFrame(rows), gs, sp, runtime


def score(d: pd.DataFrame) -> dict:
    out = {"passes": int(len(d))}
    c = d.loc[d["completed"]]
    for s in ("choice", "adjusted"):
        k = c.dropna(subset=[s, "sc_dq"])
        out[s] = {"shot_auc_completed": auc(c[s].astype(float), c["shot"]),
                  "spearman_sc_analogue": spearman(k[s].astype(float), k["sc_dq"]),
                  "completion_auc": auc(d[s].astype(float), d["completed"]),
                  "median": round(float(d[s].astype(float).median()), 3)}
    both = d.dropna(subset=["choice", "adjusted"])
    out["rank_correlation_adjusted_vs_choice"] = spearman(both["adjusted"].astype(float), both["choice"].astype(float))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--matches", nargs="*")
    args = ap.parse_args()
    for p in (args.data, args.cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    dirs = [d for d in sorted((args.data / "matches").iterdir()) if (d / f"{d.name}_dynamic_events.csv").exists()]
    if args.matches:
        dirs = [d for d in dirs if d.name in args.matches]
    parts: dict = {"fit": [], "test": []}
    gss, sps, runtimes = [], [], []
    for n, d in enumerate(dirs):
        p, gs, sp, rt = one(d, args.cache)
        parts["fit" if n % 2 == 0 else "test"].append(p)
        gss.append(gs)
        sps.append(sp)
        runtimes.append(rt)
        print(d.name, len(p), "passes", round(rt, 1), "s", flush=True)
    GS, SP = pd.concat(gss, ignore_index=True), pd.concat(sps, ignore_index=True)
    out = {"adjusted_vs_choice": {h: score(pd.concat(v, ignore_index=True)) for h, v in parts.items() if v},
           "direct_share_by_game_state": {f"{s}_{'late' if late else 'before_75'}": round(float(grp["direct"].mean()), 4)
                                          for (s, late), grp in GS.groupby(["game_state", "late"])},
           "direct_possessions_by_game_state": {f"{s}_{'late' if late else 'before_75'}": int(len(grp)) for (s, late), grp in GS.groupby(["game_state", "late"])},
           "open_high_value_share_by_phase": {ph: round(float(grp["open"].median()), 3) for ph, grp in SP.dropna().groupby("phase")},
           "p7_runtime_s": {"median": round(float(np.median(runtimes)), 1), "max": round(float(np.max(runtimes)), 1)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
