"""
context_risk_check.py — the risk-side check on P7's cost term (Stage C, deliverable 4).

Context-adjusted Choice charges a pass for what losing the ball would give away: the other team's threat at the cell
(Plaiq's ``xT_surface``). The value-only references (shots, SkillCorner's decision analogue) cannot test that term, so
this asks the question it rests on: **do losses where the cost is high lead to the opponent's shots more often?**

On every loss of the ball in play (the other team takes the ball within ``--gap-s`` of the loser's last touch), the
cost is the regaining team's threat at the ball's position when it regains (the nearest detected ball within 1 s), and
the outcome is a shot by the regaining team within ``--window-s`` (10, 15 and 20 s are reported) before the losing
team has the ball again. Measured:

- **AUC of the cost** for the opponent's shot, against the AUC of plain depth (how near its own goal the loser lost
  it), so the threat surface is compared with the simplest alternative.
- **Shot rate by cost quintile**, and the top quintile's rate over the bottom one's.
- **The term as Choice's options apply it** (SkillCorner only), at the chosen option's best cell (the cell that
  maximises ``U`` at ``p7_cost_weight``): on failed passes, the cost there against the opponent's shot after the loss;
  on every pass Choice scored, the charge ``(1 − p) × cost`` (the expected cost, known before the outcome) against
  "this pass is lost and the opponent shoots", beside ``1 − p`` and the cost alone.

Two sources, comparison only (founder decision 2):
- **SkillCorner's 20 open-data matches:** losses from its player possessions (the next possession is the other team's,
  within ``--gap-s``; not after the loser's shot), shots from its possessions ending in a shot.
- **The German league's seven matches (IDSSE):** losses from the official team in possession (the ball in play, each
  side holding it for at least 1 s; not within 3 s of the loser's shot), shots from the official ``ShotAtGoal`` events.

Reads the data outside the repository and writes JSON outside it; no names are read.

    python tools/context_risk_check.py --sc-data ~/ponta-reference-data/skillcorner-opendata \
        --sc-cache /tmp/ponta-context/sc-cache --idsse ~/ponta-reference-data/idsse \
        --idsse-cache /tmp/ponta-context/idsse-cache --out /tmp/ponta-context/risk_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.thresholds import values
from plaiq_engine.match_analysis import MatchAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_reference_set import _load
from idsse import MATCH_IDS, load_idsse_match
from plaiq_references import auc

WINDOWS = (10.0, 15.0, 20.0)


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def _ball_at(ball: pd.DataFrame, frames: np.ndarray, fps: float) -> np.ndarray:
    """The nearest detected ball position within 1 s of each frame; NaN where there is none."""
    bf = ball["frame"].to_numpy(np.int64)
    xy = ball[["x", "y"]].to_numpy(float)
    out = np.full((len(frames), 2), np.nan)
    if not len(bf):
        return out
    k = np.clip(np.searchsorted(bf, frames), 1, len(bf) - 1)
    near = np.where(np.abs(bf[k - 1] - frames) <= np.abs(bf[k] - frames), k - 1, k)
    ok = np.abs(bf[near] - frames) <= fps
    out[ok] = xy[near[ok]]
    return out


def _cost_and_depth(ma: MatchAnalysis, losses: pd.DataFrame, xy: np.ndarray) -> pd.DataFrame:
    """The regaining team's threat at the ball (the cost) and how near the loser's own goal it is (home attacks +x)."""
    ok = np.isfinite(xy).all(axis=1)
    cost = np.full(len(losses), np.nan)
    for t, label in (("home", "Home"), ("away", "Away")):
        sel = ok & (losses["regainer"].to_numpy() == t)
        if sel.any():
            cost[sel] = ma._xt_at(label, xy[sel])
    # Closeness to the loser's own goal: the plain alternative to the threat surface (higher is nearer its goal).
    depth = np.where(losses["loser"].to_numpy() == "home", -xy[:, 0], xy[:, 0])
    return losses.assign(cost=cost, depth=depth)


def _shots_after(losses: pd.DataFrame, shots: pd.DataFrame, back: pd.DataFrame, fps: float) -> pd.DataFrame:
    """For each loss and window: a shot by the regaining team before the loser has the ball again."""
    out = {}
    for w in WINDOWS:
        hit = []
        for r in losses.itertuples():
            end = r.frame + w * fps
            nb = back.loc[(back["team"] == r.loser) & (back["frame"] > r.frame), "frame"]
            if len(nb):
                end = min(end, float(nb.min()))
            s = shots.loc[(shots["team"] == r.regainer) & (shots["frame"] >= r.frame) & (shots["frame"] <= end)]
            hit.append(not s.empty)
        out[f"shot_{int(w)}s"] = hit
    return losses.assign(**out)


def skillcorner(match_dir: Path, cache: Path, gap_s: float, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = _load(match_dir, cache)
    ma = MatchAnalysis(m)
    _ = ma.decision_quality_v2  # Choice's options, kept by MatchAnalysis
    fps = float(m.fps)
    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    home = meta["home_team"]["id"]
    ev = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", low_memory=False,
                     usecols=["event_type", "end_type", "frame_start", "frame_end", "team_id", "period", "pass_outcome"])
    pp = ev.loc[ev["event_type"] == "player_possession"].sort_values("frame_start").reset_index(drop=True)
    pp = pp.assign(team=np.where(pp["team_id"] == home, "home", "away"))
    prev = pp.shift(1)
    turn = (pp["team"] != prev["team"]) & (pp["period"] == prev["period"]) & ((pp["frame_start"] - prev["frame_end"]) <= gap_s * fps)
    turn &= prev["team"].notna() & (prev["end_type"] != "shot")  # a save or a goal kick after a shot is not a loss
    L = pd.DataFrame({"match": match_dir.name, "frame": pp.loc[turn, "frame_start"].to_numpy(np.int64),
                      "loser": prev.loc[turn, "team"].to_numpy(), "regainer": pp.loc[turn, "team"].to_numpy(),
                      "after_pass": (prev.loc[turn, "end_type"] == "pass").to_numpy()})
    L = _cost_and_depth(ma, L, _ball_at(m.ball, L["frame"].to_numpy(np.int64), fps))
    shots = pd.DataFrame({"team": pp.loc[pp["end_type"] == "shot", "team"], "frame": pp.loc[pp["end_type"] == "shot", "frame_end"]})
    back = pp[["team", "frame_start"]].rename(columns={"frame_start": "frame"})
    L = _shots_after(L, shots, back, fps)

    # The term as Choice applies it: failed passes that Choice scored, joined to the loss that follows them.
    xt = {"Home": ma.xt_home, "Away": ma.xt_away}
    rows = []
    starts = ma.events["Start Frame"]
    failed = ma.events["Type"].astype(str).str.upper().eq("BALL LOST")
    shot_cols = [c for c in L.columns if c.startswith("shot_")]
    for idx, o in ma.choice_options.items():
        k = o["chosen"]
        cells, p, v = np.asarray(o["cells"][k], int), np.asarray(o["p"][k], float), np.asarray(o["values"][k], float)
        if not len(cells):
            continue
        cost = xt["Away" if o["team"] == "Home" else "Home"][cells]
        j = int(np.argmax(p * v - cfg["p7_cost_weight"] * (1 - p) * cost))
        row = {"match": match_dir.name, "failed": bool(failed.loc[idx]), "option_cost": float(cost[j]),
               "one_minus_p": float(1 - p[j]), "charge": float((1 - p[j]) * cost[j]), "loss_cost": np.nan,
               **{c: False for c in shot_cols}}
        if row["failed"]:
            f0 = int(starts.loc[idx])
            nxt = L.loc[(L["frame"] >= f0) & (L["frame"] <= f0 + 5 * fps) & (L["loser"] == o["team"].lower())]
            if nxt.empty:
                continue  # no loss by the passer's team within 5 s (the ball went out, or the feeds disagree)
            r = nxt.iloc[0]
            row.update({"loss_cost": r["cost"], **{c: bool(r[c]) for c in shot_cols}})
        rows.append(row)
    return L, pd.DataFrame(rows)


def idsse(data: Path, mid: str, cache: Path) -> pd.DataFrame:
    m, off = load_idsse_match(data, mid, cache)
    ma = MatchAnalysis(m)
    fps = float(m.fps)
    b = off.ball.sort_values("frame").reset_index(drop=True)
    live = b["ball_status"].eq(1).to_numpy() & b["ball_team"].isin(["home", "away"]).to_numpy()
    team = np.where(live, b["ball_team"].astype(object).to_numpy(), None)
    fr = b["frame"].to_numpy(np.int64)
    # Spells: runs of one team with the ball in play; a loss is a spell of at least 1 s followed directly by the
    # other team's spell of at least 1 s.
    change = np.flatnonzero(np.r_[True, team[1:] != team[:-1]])
    spells = pd.DataFrame({"start": change, "end": np.r_[change[1:], len(team)] - 1})
    spells["team"] = team[spells["start"]]
    spells["n"] = spells["end"] - spells["start"] + 1
    spells = spells.loc[spells["team"].notna() & (spells["n"] >= fps)].reset_index(drop=True)
    rows = []
    for a, c in zip(spells.iloc[:-1].itertuples(), spells.iloc[1:].itertuples(), strict=True):
        if a.team != c.team and fr[c.start] - fr[a.end] <= fps and live[a.end:c.start + 1].all():
            rows.append({"match": mid, "frame": int(fr[c.start]), "loser": a.team, "regainer": c.team})
    L = pd.DataFrame(rows, columns=["match", "frame", "loser", "regainer"])
    ev = off.events
    shots = ev.loc[ev["kind"] == "ShotAtGoal", ["team", "frame"]].dropna()
    # A save or a goal kick after a shot is not a loss: drop regains within 3 s of the loser's own shot.
    own = [((shots["team"] == r.loser) & (shots["frame"] <= r.frame) & (shots["frame"] >= r.frame - 3 * fps)).any() for r in L.itertuples()]
    L = L.loc[~np.array(own, dtype=bool)].reset_index(drop=True)
    L = _cost_and_depth(ma, L, _ball_at(m.ball, L["frame"].to_numpy(np.int64), fps))
    back = pd.DataFrame({"team": spells["team"], "frame": fr[spells["start"]]})
    return _shots_after(L, shots, back, fps)


def summary(d: pd.DataFrame, score_cols: tuple[str, ...]) -> dict:
    out = {"losses": int(len(d))}
    for w in WINDOWS:
        y = d[f"shot_{int(w)}s"].astype(bool)
        res = {"shot_rate": round(float(y.mean()), 4), "shots": int(y.sum())}
        for s in score_cols:
            k = d[s].notna()
            res[f"auc_{s}"] = auc(d.loc[k, s].astype(float), y[k])
        if "cost" in d:
            k = d["cost"].notna()
            q = pd.qcut(d.loc[k, "cost"].rank(method="first"), 5, labels=False)
            rate = y[k].groupby(q).mean()
            res["shot_rate_by_cost_quintile"] = [round(float(x), 4) for x in rate]
            res["top_over_bottom_quintile"] = round(float(rate.iloc[-1] / rate.iloc[0]), 2) if rate.iloc[0] > 0 else None
        out[f"{int(w)}s"] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sc-data", type=Path, required=True)
    ap.add_argument("--sc-cache", type=Path, required=True)
    ap.add_argument("--idsse", type=Path, required=True)
    ap.add_argument("--idsse-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gap-s", type=float, default=3.0, help="the most time between the loser's last touch and the regain")
    args = ap.parse_args()
    for p in (args.sc_data, args.sc_cache, args.idsse, args.idsse_cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    cfg = values()
    sc_l, sc_p = [], []
    for d in sorted((args.sc_data / "matches").iterdir()):
        if (d / f"{d.name}_dynamic_events.csv").exists():
            a, b = skillcorner(d, args.sc_cache, args.gap_s, cfg)
            sc_l.append(a)
            sc_p.append(b)
            print(d.name, len(a), "losses", len(b), "failed passes", flush=True)
    de = []
    for mid in MATCH_IDS:
        de.append(idsse(args.idsse, mid, args.idsse_cache))
        print(mid, len(de[-1]), "losses", flush=True)
    SL, SP, DE = (pd.concat(x, ignore_index=True) for x in (sc_l, sc_p, de))
    out = {
        "p7_cost_weight": cfg["p7_cost_weight"],
        "skillcorner": {"all_losses": summary(SL, ("cost", "depth")),
                        "losses_after_a_pass": summary(SL.loc[SL["after_pass"]], ("cost", "depth")),
                        "ball_located_share": round(float(SL["cost"].notna().mean()), 3),
                        "choice_failed_passes": summary(SP.loc[SP["failed"]], ("option_cost", "charge", "loss_cost")),
                        "choice_all_passes": summary(SP, ("charge", "one_minus_p", "option_cost"))},
        "german_league": {"all_losses": summary(DE, ("cost", "depth")),
                          "ball_located_share": round(float(DE["cost"].notna().mean()), 3)},
        "per_match_auc_cost_15s_above_half": {
            "skillcorner": f"{sum(1 for _, g in SL.groupby('match') if (auc(g['cost'].dropna(), g.loc[g['cost'].notna(), 'shot_15s']) or 0) > 0.5)} of {SL['match'].nunique()}",
            "german_league": f"{sum(1 for _, g in DE.groupby('match') if (auc(g['cost'].dropna(), g.loc[g['cost'].notna(), 'shot_15s']) or 0) > 0.5)} of {DE['match'].nunique()}"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
