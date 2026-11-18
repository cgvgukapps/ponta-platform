"""
context_idsse.py — the context packages against the official German league references (IDSSE).

Seven matches with TRACAB tracking (25 Hz) and the league's official data (``tools/idsse.py``).
The league's labels are comparison data only (founder decision 2). Per match, and pooled:

1. **P1 possession and dead ball** against the official ball status and team in possession on
   every grid point, run twice: with the event feed, and with the events removed (tracking only).
2. **P2 roles:** each player's main role against the official playing position, by position group.
   The league's codes are mapped to groups in ``POSITION_GROUP``; codes whose meaning depends on the
   line-up are settled from where those players stand (``--positions`` prints that table).
3. **P4 restarts without a feed:** corners and throw-ins found from positions against every
   official corner and throw-in (the window of ``tools/context_restart_sweep.py``). The league logs a
   restart 1 to 3 s before the ball is played, so each one is timed when the official ball status
   next turns live (within 8 s; the logged time otherwise).
4. **Formation plausibility:** P2's out-of-possession formation against the official line-up.

Reads the dataset outside the repository and writes JSON outside it; no names are read.

    python tools/context_idsse.py --data ~/ponta-reference-data/idsse --cache /tmp/ponta-context/idsse-cache \
        --out /tmp/ponta-context/idsse.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.phase import PhasePackage
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.restarts import SETUP_AFTER_S, SETUP_BEFORE_S
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.context.zones import ZonesPackage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_structure_measure import COARSE, GROUP
from idsse import MATCH_IDS, load_idsse_match

#: The league's playing positions as Ponta's comparison groups (``context_structure_measure.GROUP``).
#: OLM / ORM are the wide players of a 4-2-3-1 or a 4-4-2, LA / RA wingers. Codes marked
#: "measured" were settled from the players' mean positions on these seven matches.
POSITION_GROUP = {
    "TW": "GK", "IVL": "CB", "IVR": "CB", "IVZ": "CB", "LV": "LB", "RV": "RB",
    "DML": "DM", "DMR": "DM", "DMZ": "DM", "ZO": "AM", "OLM": "W_L", "ORM": "W_R", "LA": "W_L", "RA": "W_R",
    "STZ": "CF", "STL": "CF", "STR": "CF", "HL": "CM", "HR": "CM",
    # Measured (--positions): in a back three LM / RM and DLM / DRM stand 46 to 48 m from their own goal and
    # 16 to 18 m wide, wider than full-backs (15 m) at midfield depth: wing-backs, grouped with full-backs as
    # Ponta's LWB / RWB are. HL / HR stand centrally at 53 to 55 m: central midfield.
    "LM": "LB", "RM": "RB", "DLM": "LB", "DRM": "RB",
}
RESTART_KINDS = {"CornerKick": "corner_kick", "ThrowIn": "throw_in"}


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def possession_agreement(frames: pd.DataFrame, ball: pd.DataFrame) -> dict:
    """P1 against the official ball status and team at each grid point."""
    j = frames[["native_frame", "p1_possession", "p1_dead_ball"]].merge(
        ball.rename(columns={"frame": "native_frame"}), on="native_frame", how="left")
    lab = j["p1_possession"].astype(object).to_numpy()
    live = (j["ball_status"] == 1).to_numpy()
    team = j["ball_team"].astype(object).to_numpy()
    both = live & np.isin(lab, ["home", "away"]) & np.isin(team, ["home", "away"])
    dead_ref = (j["ball_status"] == 0).to_numpy()
    dead_p1 = j["p1_dead_ball"].astype(object).to_numpy()
    known_dead = np.array([v is not None and v is not pd.NA and not (isinstance(v, float) and np.isnan(v)) for v in dead_p1])
    out = {
        "points_in_play": int(live.sum()),
        "team_agreement_in_play": round(float((lab[both] == team[both]).mean()), 4) if both.any() else None,
        "labelled_share_in_play": round(float(np.isin(lab[live], ["home", "away"]).mean()), 4) if live.any() else None,
        "official_dead_share": round(float(dead_ref.mean()), 4),
        "p1_dead_ball_known": bool(known_dead.any()),
    }
    if known_dead.any():
        p1d = np.array([bool(v) if k else False for v, k in zip(dead_p1, known_dead, strict=True)])
        out["dead_ball"] = {"official_dead_found": round(float(p1d[dead_ref].mean()), 4) if dead_ref.any() else None,
                            "p1_dead_right": round(float(dead_ref[p1d].mean()), 4) if p1d.any() else None}
    return out


def role_agreement(main: pd.DataFrame, roles: pd.DataFrame) -> dict:
    j = roles.loc[roles["position"].notna() & (roles["position"] != "TW")].merge(main, on="player_id", how="inner", suffixes=("", "_p2"))
    g_ref, g_us = j["position"].map(POSITION_GROUP), j["role"].map(GROUP)
    grouped = g_ref == g_us
    coarse = g_ref.map(lambda x: COARSE.get(x, x)) == g_us.map(lambda x: COARSE.get(x, x))
    w = j["role_s"].fillna(0.0)
    return {"players": int(len(j)), "grouped_time_weighted": round(float((grouped * w).sum() / w.sum()), 4),
            "central_mid_merged_time_weighted": round(float((coarse * w).sum() / w.sum()), 4),
            "misses": sorted(f"{a}->{b}" for a, b, ok in zip(j["position"], j["role"], grouped, strict=True) if not ok)}


def kick_frame(frame: int, ball: pd.DataFrame) -> int:
    """When the official ball status next turns live after a logged restart (within 8 s), else the logged frame."""
    s = ball.loc[(ball["frame"] >= frame - 25) & (ball["frame"] <= frame + 200)]
    v = s["ball_status"].to_numpy()
    go = s["frame"].to_numpy()[1:][(v[1:] == 1) & (v[:-1] == 0)]
    return int(go[0]) if len(go) else int(frame)


def restart_agreement(found: pd.DataFrame, ev: pd.DataFrame, ts_of: dict[int, float]) -> dict:
    out = {}
    for kind, name in RESTART_KINDS.items():
        ref = [(r.team, ts_of[r.frame]) for r in ev.loc[ev["kind"] == kind].itertuples() if r.frame in ts_of]
        det = [(r.team, r.ts) for r in found.loc[found["type"] == name].itertuples()]
        used, tp = set(), 0
        for team, e in det:
            for k, (rt, r) in enumerate(ref):
                if k not in used and rt == team and r - SETUP_BEFORE_S <= e <= r + SETUP_AFTER_S:
                    used.add(k)
                    tp += 1
                    break
        out[name] = {"reference": len(ref), "found": len(det), "true": tp}
    return out


def _total(rows: list[dict]) -> dict:
    t = {k: sum(r[k] for r in rows) for k in ("reference", "found", "true")}
    t["recall"] = round(t["true"] / t["reference"], 3) if t["reference"] else None
    t["precision"] = round(t["true"] / t["found"], 3) if t["found"] else None
    return t


def positions_table(data: Path, cache: Path) -> pd.DataFrame:
    """Mean position (team frame: depth from own goal, width with left positive) per league code."""
    from plaiq_engine.context.grid import build_grid
    from plaiq_engine.context.structure import attack_signs, team_frames
    rows = []
    for mid in MATCH_IDS:
        m, off = load_idsse_match(data, mid, cache)
        g = build_grid(m, 5.0)
        tf = team_frames(m, g, attack_signs(m, g))
        for r in off.roles.dropna(subset=["position"]).itertuples():
            f = tf[r.team]
            if r.player_id in f.ids:
                i = f.ids.index(r.player_id)
                seen = np.isfinite(f.depth[:, i]).sum() / 5.0
                if seen > 600:
                    rows.append({"match": mid, "lineup": off.lineups[r.team], "position": r.position,
                                 "depth": float(np.nanmean(f.depth[:, i])), "width": float(np.nanmean(f.width[:, i]))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--positions", action="store_true", help="print mean positions per league code and stop")
    args = ap.parse_args()
    for p in [args.data, args.cache] + ([args.out] if args.out else []):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    if args.positions:
        t = positions_table(args.data, args.cache)
        print(t.groupby(["position"])[["depth", "width"]].agg(["mean", "count"]).round(1).to_string())
        print(t.groupby(["lineup", "position"])[["depth", "width"]].mean().round(0).to_string())
        return
    report: dict = {}
    pooled = {"feed": [], "tracking": [], "restarts": []}
    for mid in MATCH_IDS:
        m, off = load_idsse_match(args.data, mid, args.cache)
        packages = [PossessionPackage(), StructurePackage(), ZonesPackage(), PhasePackage()]
        res = Registry(packages).run(m)
        bare = dataclasses.replace(m, events=m.events.iloc[0:0])
        res_t = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), PhasePackage()]).run(bare)
        ts_of = dict(zip(res_t.frames["native_frame"], res_t.frames["frame_ts"], strict=True))
        # Official restart frames snap to the nearest grid point.
        nf = res_t.frames["native_frame"].to_numpy()
        ev = off.events.copy()
        ev["logged_frame"] = ev["frame"]
        ev["frame"] = [int(nf[np.clip(np.searchsorted(nf, f), 0, len(nf) - 1)]) for f in (kick_frame(f, off.ball) for f in ev["frame"])]
        entry = {
            "lineups": off.lineups,
            "p1_feed": possession_agreement(res.frames, off.ball),
            "p1_tracking_only": possession_agreement(res_t.frames, off.ball),
            "p2_roles": role_agreement(res.outputs["p2"].segments["main_roles"], off.roles),
            "p2_formation_out": {t: res.outputs["p2"].measurements["formations"][f"{t}_out"] for t in ("home", "away")},
            "p4_restarts_tracking_only": restart_agreement(res_t.outputs["p4"].segments["restarts"], ev, ts_of),
            "entries_feed": {f"{e.package}:{e.metric}": e.state for e in res.entries},
        }
        report[mid] = entry
        pooled["restarts"].append(entry["p4_restarts_tracking_only"])
        print(mid, json.dumps({k: entry[k] for k in ("p1_feed", "p1_tracking_only")}), json.dumps(entry["p2_roles"]["grouped_time_weighted"]),
              json.dumps(entry["p4_restarts_tracking_only"]), flush=True)
    summary = {
        "p1_feed_team_agreement": [report[k]["p1_feed"]["team_agreement_in_play"] for k in report],
        "p1_tracking_team_agreement": [report[k]["p1_tracking_only"]["team_agreement_in_play"] for k in report],
        "p2_roles_grouped_tw": [report[k]["p2_roles"]["grouped_time_weighted"] for k in report],
        "p2_roles_merged_tw": [report[k]["p2_roles"]["central_mid_merged_time_weighted"] for k in report],
        "p4_restarts": {name: _total([r[name] for r in pooled["restarts"]]) for name in RESTART_KINDS.values()},
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "matches": report}, indent=1, default=str))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
