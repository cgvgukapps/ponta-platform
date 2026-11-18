"""
plaiq_references.py — Plaiq's models against independent references (measurement only).

Step 1 of ``ponta-blueprint/plaiq-review.md``. Nothing here changes the engine: Plaiq's own
functions are called and their outputs compared with references that are comparison data only
(founder decision 2):

1. **SkillCorner's open data (20 matches)**, on every pass SkillCorner records:
   - **failed-pass target:** Plaiq's inferred intended receiver (``MatchAnalysis.event_pov``) against
     SkillCorner's targeted player, beside a simple baseline (the team-mate nearest the ball's end);
   - **completion:** Plaiq's targeted pitch control for the pass to the targeted player
     (``targeted_pcz``: control at the receiver and interception risk on the way) against whether the
     pass was completed (ranking, AUC) and SkillCorner's expected completion (rank correlation);
   - **value:** Plaiq's pass option value of the targeted player at the pass against SkillCorner's threat
     for that player (rank correlation) and its "dangerous" flag (AUC).
2. **The German league's data (IDSSE, 7 matches)**, on every shot: Plaiq's 1v1 and zonal pressure on
   the shooter (``hex_analytics.compute_pressure``) against the league's shot pressure, and Plaiq's
   threat surface at the shot (``hex_value.xT_surface``) against the league's expected goals.
3. **Wyscout's open events (1,941 matches)**: Plaiq's threat surface at each zone of a 12 x 8 grid
   against the share of events after which the same team scores within 10 s, and against the goal
   rate of shots from the zone.

Reads data outside the repository and writes JSON outside it; no names are read.

    python tools/plaiq_references.py --sc-data ~/ponta-reference-data/skillcorner-opendata --sc-cache /tmp/ponta-context/sc-cache \
        --idsse ~/ponta-reference-data/idsse --idsse-cache /tmp/ponta-context/idsse-cache \
        --wyscout ~/ponta-reference-data/wyscout-opendata --out /tmp/ponta-context/plaiq_references.json
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.analytics.hex_analytics import assign_all_players, compute_pressure
from plaiq_engine.analytics.hex_value import xT_surface
from plaiq_engine.analytics.pass_option_value import _build_defender_data, targeted_pcz
from plaiq_engine.analytics.pitch_control import build_pitch_hex_grid
from plaiq_engine.match_analysis import HEX_RADIUS, MatchAnalysis
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_line_breaks import plaiq_pov
from context_reference_set import _load
from idsse import MATCH_IDS, _files, load_idsse_match


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def auc(score: np.ndarray, label: np.ndarray) -> float | None:
    """Probability that a positive scores above a negative (ties count half)."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.all() or not y.any():
        return None
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return round(float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)


def spearman(a, b) -> float | None:
    x = pd.DataFrame({"a": a, "b": b}).dropna()
    return round(float(x["a"].corr(x["b"], method="spearman")), 4) if len(x) > 2 else None


# ---------------------------------------------------------------------------
# 1. SkillCorner
# ---------------------------------------------------------------------------

def skillcorner(match_dir: Path, cache: Path) -> pd.DataFrame:
    m = _load(match_dir, cache)
    ma = MatchAnalysis(m)
    pov = plaiq_pov(m, cache)
    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    home = meta["home_team"]["id"]
    number = {p["id"]: str(p["number"]) for p in meta["players"]}
    cols = ["event_type", "end_type", "frame_end", "team_id", "pass_outcome", "player_targeted_id", "player_targeted_xpass_completion",
            "player_targeted_xthreat", "player_targeted_dangerous"]
    ref = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", usecols=cols, low_memory=False)
    ref = ref.loc[(ref["event_type"] == "player_possession") & (ref["end_type"] == "pass") & ref["pass_outcome"].isin(["successful", "unsuccessful"])]
    ev = m.events.loc[m.events["Type"].isin(["PASS", "BALL LOST"])]
    actor = pov.loc[pov["inferred_receiver"].astype(str) != ""].drop_duplicates("event_idx").set_index("event_idx")["inferred_receiver"].astype(str)
    rows = []
    for r in ref.itertuples():
        team = "Home" if r.team_id == home else "Away"
        cand = ev.loc[(ev["Team"] == team) & ((ev["Start Frame"] - r.frame_end).abs() <= 2)]
        if cand.empty:
            continue
        e = cand.iloc[0]
        target = number.get(r.player_targeted_id)
        f = int(e["Start Frame"])
        own, opp = (ma.home, ma.away) if team == "Home" else (ma.away, ma.home)
        oprefix = "Away" if team == "Home" else "Home"
        ojer = ma.away_jerseys if team == "Home" else ma.home_jerseys
        oprof = ma.profiles_away if team == "Home" else ma.profiles_home
        row = {"match": match_dir.name, "completed": r.pass_outcome == "successful", "sc_xpass": r.player_targeted_xpass_completion,
               "sc_xthreat": r.player_targeted_xthreat, "sc_dangerous": r.player_targeted_dangerous, "risk": np.nan, "ctrl": np.nan,
               "pov_target": np.nan, "plaiq_target_right": None, "nearest_target_right": None}
        if target is not None and f in own.index and f in opp.index:
            rx, ry = own.at[f, f"{team}_{target}_x"] if f"{team}_{target}_x" in own.columns else np.nan, \
                own.at[f, f"{team}_{target}_y"] if f"{team}_{target}_y" in own.columns else np.nan
            if np.isfinite(rx) and np.isfinite(e["Start X"]):
                risk, ctrl = targeted_pcz(np.array([e["Start X"], e["Start Y"]], float), np.array([rx, ry], float),
                                          _build_defender_data(opp.loc[f], oprefix, ojer, oprof))
                row["risk"], row["ctrl"] = risk, ctrl
            pv = pov.loc[(pov["event_idx"] == e.name) & (pov["jersey"].astype(str) == target), "pov_start"]
            row["pov_target"] = float(pv.iloc[0]) if len(pv) else np.nan
        if e["Type"] == "BALL LOST" and target is not None:
            if e.name in actor.index:
                row["plaiq_target_right"] = actor[e.name] == target
            # baseline: the team-mate (not the passer) nearest the ball's end point at the pass
            if np.isfinite(e.get("End X", np.nan)) and f in own.index:
                best, best_d = None, np.inf
                for j in (ma.home_jerseys if team == "Home" else ma.away_jerseys):
                    if str(j) == str(e["From"]):
                        continue
                    x, y = own.at[f, f"{team}_{j}_x"], own.at[f, f"{team}_{j}_y"]
                    if np.isfinite(x) and np.hypot(x - e["End X"], y - e["End Y"]) < best_d:
                        best, best_d = str(j), np.hypot(x - e["End X"], y - e["End Y"])
                row["nearest_target_right"] = best == target
        rows.append(row)
    return pd.DataFrame(rows)


def score_skillcorner(d: pd.DataFrame) -> dict:
    p = d.dropna(subset=["ctrl"])
    pcomp = p["ctrl"] * (1 - p["risk"])
    f = d.loc[d["plaiq_target_right"].notna()]
    b = d.loc[d["nearest_target_right"].notna()]
    return {
        "passes": int(len(d)), "with_plaiq_control": int(len(p)),
        "completion": {"completed_share": round(float(p["completed"].mean()), 4),
                       "auc_plaiq_ctrl_x_no_interception": auc(pcomp, p["completed"]), "auc_plaiq_ctrl": auc(p["ctrl"], p["completed"]),
                       "auc_plaiq_no_interception": auc(1 - p["risk"], p["completed"]), "auc_sc_xpass": auc(p["sc_xpass"], p["completed"]),
                       "spearman_plaiq_vs_sc_xpass": spearman(pcomp, p["sc_xpass"])},
        "value": {"spearman_pov_vs_sc_xthreat": spearman(d["pov_target"], d["sc_xthreat"]),
                  "auc_pov_for_sc_dangerous": auc(d["pov_target"], d["sc_dangerous"].fillna(False).astype(bool))},
        "failed_pass_target": {"failed_passes": int(len(f)), "plaiq_right": round(float(f["plaiq_target_right"].mean()), 4) if len(f) else None,
                               "nearest_to_ball_end_right": round(float(b["nearest_target_right"].mean()), 4) if len(b) else None},
    }


# ---------------------------------------------------------------------------
# 2. IDSSE shots
# ---------------------------------------------------------------------------

def idsse_shots(data: Path, cache: Path) -> pd.DataFrame:
    rows = []
    hexes = build_pitch_hex_grid(HEX_RADIUS)
    tree = cKDTree(hexes)
    xt = {True: xT_surface(hexes, attacking_right=True), False: xT_surface(hexes, attacking_right=False)}
    for mid in MATCH_IDS:
        m, off = load_idsse_match(data, mid, cache)
        xml = ET.parse(_files(data, mid)["events"]).getroot()
        shots_xml = [s for e in xml.iter("Event") for s in [e.find("ShotAtGoal")] if s is not None]
        shots = off.events.loc[off.events["kind"] == "ShotAtGoal"].reset_index(drop=True)
        if len(shots) != len(shots_xml):
            print(mid, "shot counts differ; skipped", flush=True)
            continue
        ma = MatchAnalysis(m)
        frames = [f for f in shots["frame"] if f in ma.home.index]
        hx = assign_all_players(ma.home.loc[frames], ma.away.loc[frames], ma.home_jerseys, ma.away_jerseys, HEX_RADIUS)
        pr = compute_pressure(hx).set_index(["Frame", "team", "jersey"])
        for s, x in zip(shots.itertuples(), shots_xml, strict=True):
            if s.frame not in ma.home.index or not isinstance(s.player, str):
                continue
            team, jersey = s.player.split("_")
            key = (s.frame, team.capitalize(), jersey)
            ind = pr["pressure"].get(key, np.nan)
            zon = pr["zonal_pressure"].get(key, np.nan)
            _, h = tree.query([s.x, s.y])
            rows.append({"match": mid, "league_pressure": float(x.get("Pressure", "nan")), "league_xg": float(x.get("xG", "nan")),
                         "plaiq_1v1": ind, "plaiq_zonal": zon, "plaiq_xt": float(xt[team == "home"][h]),
                         "goal": next((c.tag for c in x), None) == "SuccessfulShot"})
        print("idsse", mid, len(shots), "shots", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Wyscout zones
# ---------------------------------------------------------------------------

def wyscout_zones(wyscout: Path, nx: int = 12, ny: int = 8, horizon_s: float = 10.0) -> pd.DataFrame:
    counts = np.zeros((nx, ny)); scored = np.zeros((nx, ny)); shots = np.zeros((nx, ny)); goals = np.zeros((nx, ny))
    with zipfile.ZipFile(wyscout / "events.zip") as z:
        for name in sorted(z.namelist()):
            for _, ev in pd.DataFrame(json.loads(z.read(name))).groupby("matchId"):
                ev = ev.loc[ev["matchPeriod"].isin(["1H", "2H"])]
                x = ev["positions"].map(lambda p: p[0]["x"] if p else np.nan).to_numpy(float)
                y = ev["positions"].map(lambda p: p[0]["y"] if p else np.nan).to_numpy(float)
                goal = ev["tags"].map(lambda t: any(g["id"] == 101 for g in t)).to_numpy()
                own = ev["tags"].map(lambda t: any(g["id"] == 102 for g in t)).to_numpy()
                team, per, t = ev["teamId"].to_numpy(), ev["matchPeriod"].to_numpy(), ev["eventSec"].to_numpy(float)
                is_shot = (ev["eventName"] == "Shot").to_numpy()
                goals_at = [(per[i], t[i], team[i]) for i in np.where(goal & ~own)[0]]
                ix = np.clip((x / 100 * nx).astype(int), 0, nx - 1)
                iy = np.clip((y / 100 * ny).astype(int), 0, ny - 1)
                ok = np.isfinite(x) & np.isfinite(y)
                for i in np.where(ok)[0]:
                    counts[ix[i], iy[i]] += 1
                    if any(p == per[i] and tm == team[i] and 0 <= tg - t[i] <= horizon_s for p, tg, tm in goals_at):
                        scored[ix[i], iy[i]] += 1
                    if is_shot[i]:
                        shots[ix[i], iy[i]] += 1
                        goals[ix[i], iy[i]] += goal[i] and not own[i]
            print("wyscout", name, flush=True)
    hexes = build_pitch_hex_grid(HEX_RADIUS)
    xt = xT_surface(hexes, attacking_right=True)
    tree = cKDTree(hexes)
    rows = []
    for i in range(nx):
        for j in range(ny):
            cx, cy = ((i + 0.5) / nx) * 105 - 52.5, (0.5 - (j + 0.5) / ny) * 68
            _, h = tree.query([cx, cy])
            rows.append({"zone_x": i, "zone_y": j, "events": counts[i, j], "scored_within": scored[i, j] / counts[i, j] if counts[i, j] else np.nan,
                         "shots": shots[i, j], "shot_goal_rate": goals[i, j] / shots[i, j] if shots[i, j] >= 30 else np.nan, "plaiq_xt": float(xt[h])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sc-data", type=Path, required=True)
    ap.add_argument("--sc-cache", type=Path, required=True)
    ap.add_argument("--idsse", type=Path, required=True)
    ap.add_argument("--idsse-cache", type=Path, required=True)
    ap.add_argument("--wyscout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parts", nargs="*", default=["skillcorner", "idsse", "wyscout"])
    args = ap.parse_args()
    for p in (args.sc_data, args.sc_cache, args.idsse, args.idsse_cache, args.wyscout, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    out = {}
    if "skillcorner" in args.parts:
        parts = []
        for d in sorted((args.sc_data / "matches").iterdir()):
            if (d / f"{d.name}_dynamic_events.csv").exists():
                parts.append(skillcorner(d, args.sc_cache))
                print("skillcorner", d.name, len(parts[-1]), "passes", flush=True)
        sc = pd.concat(parts, ignore_index=True)
        sc.to_pickle(args.out.parent / "plaiq_references_sc.pkl")
        out["skillcorner"] = score_skillcorner(sc)
    if "idsse" in args.parts:
        sh = idsse_shots(args.idsse, args.idsse_cache)
        sh.to_pickle(args.out.parent / "plaiq_references_shots.pkl")
        out["idsse_shots"] = {"shots": int(len(sh)), "goals": int(sh["goal"].sum()),
                              "spearman_plaiq_1v1_vs_league_pressure": spearman(sh["plaiq_1v1"], sh["league_pressure"]),
                              "spearman_plaiq_zonal_vs_league_pressure": spearman(sh["plaiq_zonal"], sh["league_pressure"]),
                              "spearman_plaiq_xt_vs_league_xg": spearman(sh["plaiq_xt"], sh["league_xg"]),
                              "auc_goal_league_xg": auc(sh["league_xg"], sh["goal"]), "auc_goal_plaiq_xt": auc(sh["plaiq_xt"], sh["goal"])}
    if "wyscout" in args.parts:
        zt = wyscout_zones(args.wyscout)
        zt.to_pickle(args.out.parent / "plaiq_references_zones.pkl")
        out["wyscout_zones"] = {"zones": int(len(zt)), "events": int(zt["events"].sum()),
                                "spearman_plaiq_xt_vs_scored_within_10s": spearman(zt["plaiq_xt"], zt["scored_within"]),
                                "spearman_plaiq_xt_vs_shot_goal_rate": spearman(zt["plaiq_xt"], zt["shot_goal_rate"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
