"""
context_structure_measure.py — Stage B measurements for ``CONTEXT_STRUCTURE_REPORT.md``.

Runs P1 to P4 on the regression set (Metrica Sample Games 1 to 3, SkillCorner 1886347) and
reports, per match: runtime and storage; P2 shape ranges, formations per team and half,
formation stability and fit clarity; P3 zone shares; P4 phase shares. Against references
(comparison only, founder decision 2): P2 roles against SkillCorner's player roles; P4
phases against SkillCorner's phases; the tracking-only corner rule against Metrica's logged
corners. Writes JSON **outside the repository**; reads no player names.

    python tools/context_structure_measure.py --cache /tmp/ponta-context/cache --out /tmp/ponta-context/structure.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.phase import PhasePackage, corner_at_start
from plaiq_engine.context.possession import PossessionPackage, ball_on_grid
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.context.zones import ZonesPackage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_measure import load
from context_phase_sweep import agreement, confusion
from context_references import sc_phase_spans, sc_roles

MATCHES = ("metrica_1", "metrica_2", "metrica_3", "skillcorner_1886347")
GROUP = {"LB": "LB", "LWB": "LB", "RB": "RB", "RWB": "RB", "LCB": "CB", "CB": "CB", "RCB": "CB", "DM": "DM", "LDM": "DM", "RDM": "DM",
         "LCM": "CM", "CM": "CM", "RCM": "CM", "AM": "AM", "LAM": "AM", "RAM": "AM", "LW": "W_L", "LM": "W_L", "RW": "W_R", "RM": "W_R",
         "CF": "CF", "LCF": "CF", "RCF": "CF"}
COARSE = {"DM": "C", "CM": "C", "AM": "C"}


def role_agreement(roles: pd.DataFrame, formations: pd.DataFrame) -> dict:
    ref = sc_roles()
    r = roles.merge(formations[["team", "period", "block", "state", "data_s"]], on=["team", "period", "block", "state"])
    out = {}
    for state in ("all", "out", "in"):
        d = (r.loc[r["state"] == state].groupby(["player_id", "role"])["data_s"].sum().reset_index()
             .sort_values(["data_s", "role"], ascending=[False, True]).drop_duplicates("player_id"))
        j = ref.merge(d, on="player_id", how="left")
        j = j.loc[j["role_x"] != "GK"]
        w = j["data_s"].fillna(0.0)
        g_ref, g_us = j["role_x"].map(GROUP), j["role_y"].map(GROUP)
        exact, grouped = j["role_x"] == j["role_y"], g_ref == g_us
        coarse = g_ref.map(lambda x: COARSE.get(x, x)) == g_us.map(lambda x: COARSE.get(x, x))
        out[state] = {"players": int(len(j)), "exact": round(float(exact.mean()), 4), "grouped": round(float(grouped.mean()), 4),
                      "grouped_time_weighted": round(float((grouped * w).sum() / w.sum()), 4),
                      "central_mid_merged": round(float(coarse.mean()), 4),
                      "central_mid_merged_time_weighted": round(float((coarse * w).sum() / w.sum()), 4)}
        if state == "all":
            out["pairs"] = sorted(f"{a}->{b}" for a, b in zip(j["role_x"], j["role_y"].fillna("none"), strict=True) if GROUP.get(a) != GROUP.get(b))
    gk = set(sc_roles().query("role == 'GK'")["player_id"])
    out["goalkeepers_excluded_from_roles"] = not (set(roles["player_id"]) & gk)
    return out


def formations_by_half(formations: pd.DataFrame) -> dict:
    out = {}
    for (team, state, period), g in formations.loc[formations["state"] != "all"].groupby(["team", "state", "period"]):
        out[f"{team}_{state}_h{period}"] = {"current": g["current"].value_counts().idxmax(),
                                           "changes": int((g.sort_values("block")["current"] != g.sort_values("block")["current"].shift()).sum() - 1),
                                           "clear_share": round(float((g["formation"] != "unclear").mean()), 3)}
    return out


def corner_rule(m, res, radius: float, window_s: float) -> dict:
    """The tracking-only corner rule against the logged corners (Metrica)."""
    ch = res.outputs["p1"].segments["chains"]
    g = res.frames
    ball = ball_on_grid(m, g)
    bx, by = ball["ball_x"].to_numpy(float), ball["ball_y"].to_numpy(float)
    cid = g["p1_chain_id"].to_numpy(dtype=float, na_value=np.nan)
    tp = fp = fn = 0
    for c in ch.itertuples():
        idx = np.where((cid == c.chain_id) & (g["p1_possession"].to_numpy(object) == c.team))[0]
        hit = corner_at_start(bx, by, idx[: max(1, int(round(window_s * 5)))], m.pitch.length_m, m.pitch.width_m, radius) if len(idx) else False
        logged = str(c.start_type).startswith("set_piece:corner")
        tp += hit and logged
        fp += hit and not logged
        fn += logged and not hit
    return {"logged_corners": tp + fn, "found": tp, "false_alarms": fp, "missed": fn}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if repo_root() in args.out.resolve().parents:
        sys.exit("refusing to write measurements inside the repository")
    report = {}
    for key in MATCHES:
        m = load(key, args.cache)
        t0 = time.perf_counter()
        res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), PhasePackage()]).run(m)
        seconds = time.perf_counter() - t0
        with tempfile.TemporaryDirectory() as d:
            paths = res.write(d)
            size = sum(p.stat().st_size for p in paths.values())
        p2, p3, p4 = (res.outputs[k] for k in ("p2", "p3", "p4"))
        entry = {
            "runtime_s": round(seconds, 1), "storage_mb": round(size / 1e6, 2), "grid_points": int(len(res.frames)),
            "p2": {k: p2.measurements[k] for k in ("orientation_consistent", "shape_coverage_in_play", "shape", "windows")},
            "p2_formations_by_half": formations_by_half(p2.segments["formations"]),
            "p3": p3.measurements, "p4": p4.measurements,
            "entries": [{"metric": e.metric, "state": e.state, "coverage": e.coverage} for e in res.entries if e.package != "p1"],
        }
        entry["p2"]["attack_signs"] = p2.measurements["attack_signs"]
        if key.startswith("skillcorner"):
            entry["p2_roles_vs_skillcorner"] = role_agreement(p2.segments["roles"], p2.segments["formations"])
            ref = sc_phase_spans(res.frames)
            per = res.frames["period"].to_numpy()
            entry["p4_vs_skillcorner"] = {"all": agreement(ref, p4.frames), "half_1": agreement(ref, p4.frames, per == 1),
                                          "half_2": agreement(ref, p4.frames, per == 2), "confusion": confusion(ref, p4.frames),
                                          "reference_coverage": round(float(ref["sc_phase"].notna().mean()), 4)}
        else:
            entry["corner_rule_vs_logged"] = {f"r{r:g}_w{w:g}": corner_rule(m, res, r, w) for r in (3.0, 4.0, 5.0, 6.0) for w in (1.0, 2.0, 3.0)}
        report[key] = entry
        print(key, json.dumps({k: entry[k] for k in ("runtime_s", "storage_mb")}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
