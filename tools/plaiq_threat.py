"""
plaiq_threat.py — a learned expected-threat surface (``analytics/xt_learned.py``) against the geometric one.

Step 5 of ``ponta-blueprint/plaiq-review.md``. The surface is learned from Wyscout's open events
(Pappalardo et al. 2019, CC BY 4.0) on half of its 1,941 matches (even positions in match id order):
passes and accelerations (carries) as moves, accurate when tagged so; shots (penalties left out) with
goals. It is kept outside the repository. Both surfaces, learned and Plaiq's geometric ``V_xT``
(``hex_value.xT_surface``), are then measured on data the learned one never saw:

- **Wyscout's held-out matches:** whether the team scores within 10 s of each event (event level, AUC;
  and by zone, rank correlation), and the goal rate of shots by zone;
- **SkillCorner's threat** for the targeted player of every pass on its 20 open-data matches (rank
  correlation) and its "dangerous" flag (AUC);
- **the league's expected goals and goals** on 164 IDSSE shots (data: Deutsche Fußball Liga).

Comparison data only (founder decision 2). Writes JSON and the learned surface outside the repository.

    python tools/plaiq_threat.py --wyscout ~/ponta-reference-data/wyscout-opendata --sc-data ~/ponta-reference-data/skillcorner-opendata \
        --sc-cache /tmp/ponta-context/sc-cache --idsse ~/ponta-reference-data/idsse --idsse-cache /tmp/ponta-context/idsse-cache \
        --out /tmp/ponta-context/threat.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.analytics.hex_value import xT_surface
from plaiq_engine.analytics.pitch_control import build_pitch_hex_grid
from plaiq_engine.analytics.xt_learned import NX, NY, fit_xt, lookup, zone
from plaiq_engine.context.grid import build_grid
from plaiq_engine.context.structure import attack_signs
from plaiq_engine.match_analysis import HEX_RADIUS, MatchAnalysis
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_reference_set import _load
from idsse import MATCH_IDS, _files, load_idsse_match
from plaiq_references import auc, spearman

WL, WW = 105.0, 68.0  # Wyscout's coordinates are shares of the pitch; read on a 105 x 68 m pitch
HORIZON_S = 10.0
_HEX = build_pitch_hex_grid(HEX_RADIUS)
_TREE = cKDTree(_HEX)
_GEO = xT_surface(_HEX, attacking_right=True)


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def geometric(x_att, y, length: float) -> np.ndarray:
    """Plaiq's geometric threat at attacking-frame positions (its surface has the goal at +x)."""
    pts = np.column_stack([np.asarray(x_att, float) - length / 2, np.asarray(y, float)])
    return _GEO[_TREE.query(pts)[1]]


def wyscout(path: Path) -> pd.DataFrame:
    """Every first- and second-half event with a position, its match, and whether its team scores within 10 s."""
    parts = []
    with zipfile.ZipFile(path / "events.zip") as z:
        for name in sorted(z.namelist()):
            ev = pd.DataFrame(json.loads(z.read(name)))
            ev = ev.loc[ev["matchPeriod"].isin(["1H", "2H"])]
            tags = ev["tags"].map(lambda t: {g["id"] for g in t})
            p0 = ev["positions"].map(lambda p: p[0] if p else {"x": np.nan, "y": np.nan})
            p1 = ev["positions"].map(lambda p: p[1] if len(p) > 1 else {"x": np.nan, "y": np.nan})
            df = pd.DataFrame({
                "match": ev["matchId"].to_numpy(), "team": ev["teamId"].to_numpy(), "period": ev["matchPeriod"].to_numpy(),
                "t": ev["eventSec"].to_numpy(float), "event": ev["eventName"].to_numpy(), "sub": ev["subEventName"].to_numpy(),
                "x": p0.map(lambda p: p["x"]).to_numpy(float) / 100 * WL, "y": (50 - p0.map(lambda p: p["y"]).to_numpy(float)) / 100 * WW,
                "ex": p1.map(lambda p: p["x"]).to_numpy(float) / 100 * WL, "ey": (50 - p1.map(lambda p: p["y"]).to_numpy(float)) / 100 * WW,
                "accurate": tags.map(lambda t: 1801 in t).to_numpy(), "goal": tags.map(lambda t: 101 in t and 102 not in t).to_numpy()})
            parts.append(df)
            print("wyscout", name, len(df), flush=True)
    d = pd.concat(parts, ignore_index=True)
    d["scores_within"] = False
    for (m, per), g in d.groupby(["match", "period"]):
        goals = g.loc[g["goal"], ["t", "team"]].to_numpy()
        if not len(goals):
            continue
        t, tm = g["t"].to_numpy(), g["team"].to_numpy()
        hit = np.zeros(len(g), bool)
        for gt, gteam in goals:
            hit |= (tm == gteam) & (gt - t >= 0) & (gt - t <= HORIZON_S)
        d.loc[g.index, "scores_within"] = hit
    return d


def skillcorner(match_dir: Path, cache: Path, surface: dict) -> pd.DataFrame:
    m = _load(match_dir, cache)
    ma = MatchAnalysis(m)
    L = m.pitch.length_m
    g = build_grid(m, 5.0)
    signs = {(int(r.period), r.team): r.attack_sign for r in attack_signs(m, g).itertuples()}
    period_of = dict(zip(m.frames["frame"], m.frames["period"], strict=True))
    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    home = meta["home_team"]["id"]
    number = {p["id"]: str(p["number"]) for p in meta["players"]}
    ref = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", low_memory=False,
                      usecols=["event_type", "end_type", "frame_end", "team_id", "player_targeted_id", "player_targeted_xthreat", "player_targeted_dangerous"])
    ref = ref.loc[(ref["event_type"] == "player_possession") & (ref["end_type"] == "pass") & ref["player_targeted_xthreat"].notna()]
    rows = []
    for r in ref.itertuples():
        team = "home" if r.team_id == home else "away"
        j = number.get(r.player_targeted_id)
        f = int(r.frame_end)
        own = ma.home if team == "home" else ma.away
        col = f"{team.capitalize()}_{j}_x"
        if j is None or f not in own.index or col not in own:
            continue
        x, y = own.at[f, col], own.at[f, f"{team.capitalize()}_{j}_y"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        s = signs.get((int(period_of.get(f, 1)), team), 1)
        xa, ya = s * x + L / 2, s * y
        rows.append({"match": match_dir.name, "sc_xthreat": r.player_targeted_xthreat, "sc_dangerous": bool(r.player_targeted_dangerous),
                     "learned": float(lookup(surface, [xa], [ya], L, m.pitch.width_m)[0]), "geometric": float(geometric([xa], [ya], L)[0])})
    return pd.DataFrame(rows)


def idsse(data: Path, cache: Path, surface: dict) -> pd.DataFrame:
    import xml.etree.ElementTree as ET
    rows = []
    for mid in MATCH_IDS:
        m, off = load_idsse_match(data, mid, cache)
        xml = ET.parse(_files(data, mid)["events"]).getroot()
        sx = [s for e in xml.iter("Event") for s in [e.find("ShotAtGoal")] if s is not None]
        shots = off.events.loc[off.events["kind"] == "ShotAtGoal"].reset_index(drop=True)
        if len(shots) != len(sx):
            continue
        L, W = m.pitch.length_m, m.pitch.width_m
        for s, x in zip(shots.itertuples(), sx, strict=True):
            sign = 1 if s.team == "home" else -1  # the reader rotates halves so home attacks +x
            xa, ya = sign * s.x + L / 2, sign * s.y
            rows.append({"league_xg": float(x.get("xG", "nan")), "goal": next((c.tag for c in x), None) == "SuccessfulShot",
                         "learned": float(lookup(surface, [xa], [ya], L, W)[0]), "geometric": float(geometric([xa], [ya], L)[0])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--wyscout", "--sc-data", "--sc-cache", "--idsse", "--idsse-cache", "--out"):
        ap.add_argument(a, type=Path, required=True)
    args = ap.parse_args()
    for p in (args.wyscout, args.sc_data, args.sc_cache, args.idsse, args.idsse_cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    d = wyscout(args.wyscout)
    ids = np.sort(d["match"].unique())
    fit_ids = set(ids[::2])
    d["set"] = np.where(d["match"].isin(fit_ids), "fit", "test")
    fit = d.loc[d["set"] == "fit"]
    moves = fit["event"].eq("Pass") | (fit["event"].eq("Others on the ball") & fit["sub"].eq("Acceleration"))
    shots = fit["event"].eq("Shot")
    act = fit.loc[moves | shots]
    surface = fit_xt(act["x"], act["y"], act["ex"], act["ey"], act["event"].eq("Shot"), act["accurate"], act["goal"], length=WL, width=WW)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out.parent / "xt_learned_surface.pkl", "wb") as fh:
        pickle.dump(surface, fh)

    test = d.loc[(d["set"] == "test") & np.isfinite(d["x"]) & np.isfinite(d["y"])].copy()
    test["learned"] = lookup(surface, test["x"], test["y"], WL, WW)
    test["geometric"] = geometric(test["x"], test["y"], WL)
    ix, iy = zone(test["x"], test["y"], WL, WW, NX, NY)
    test["zone"] = ix * NY + iy
    zt = test.groupby("zone").agg(scored=("scores_within", "mean"), learned=("learned", "mean"), geometric=("geometric", "mean"))
    ts = test.loc[test["event"].eq("Shot")]
    zs = ts.groupby("zone").agg(rate=("goal", "mean"), n=("goal", "size"), learned=("learned", "mean"), geometric=("geometric", "mean"))
    zs = zs.loc[zs["n"] >= 30]
    sc = pd.concat([skillcorner(dd, args.sc_cache, surface) for dd in sorted((args.sc_data / "matches").iterdir())
                    if (dd / f"{dd.name}_dynamic_events.csv").exists()], ignore_index=True)
    sh = idsse(args.idsse, args.idsse_cache, surface)
    out = {"fit_actions": int(len(act)), "test_events": int(len(test)), "test_matches": int(test["match"].nunique()),
           "surface": {"nx": NX, "ny": NY, "max": round(float(surface["xt"].max()), 4)}}
    for name in ("learned", "geometric"):
        out[name] = {
            "wyscout_event_auc_scores_within_10s": auc(test[name], test["scores_within"]),
            "wyscout_zone_spearman_scores_within_10s": spearman(zt[name], zt["scored"]),
            "wyscout_zone_spearman_shot_goal_rate": spearman(zs[name], zs["rate"]),
            "skillcorner_spearman_threat": spearman(sc[name], sc["sc_xthreat"]),
            "skillcorner_auc_dangerous": auc(sc[name], sc["sc_dangerous"]),
            "league_spearman_xg": spearman(sh[name], sh["league_xg"]), "league_auc_goal": auc(sh[name], sh["goal"])}
    out["counts"] = {"skillcorner_passes": int(len(sc)), "league_shots": int(len(sh)), "zones_with_30_shots": int(len(zs))}
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
