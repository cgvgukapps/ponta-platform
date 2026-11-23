"""
context_fifa_view.py — P4v 0.1.0 (the FIFA view of phases) against SkillCorner's open data (Stage C, deliverable 5).

On every SkillCorner open-data match (comparison only, founder decision 2), with P1 to P5 and P4v run as shipped:

1. **Engagements** against SkillCorner's on-ball engagements: one is found when a P4v engagement by the same player
   overlaps it (1 s either side); precision the other way. **Type** on found pairs (SkillCorner's ``pressing``,
   ``pressure``, ``counter_press`` and ``recovery_press`` against P4v's press, pressure, counter-press and recovery;
   its ``other`` left out): agreement, κ and recall per type.
2. **Presses** against SkillCorner's pressing chains (each chain from its first engagement's start to its last one's
   end): the share of chain time inside a P4v press of the same team, and the share of P4v press time inside a chain
   (1 s either side). Counter-press chains (first engagement a counter-press) found by a P4v counter-press.
3. **Build-up opposed or unopposed:** SkillCorner's player possessions in its build-up phase, where P4 also says
   build-up; opposed in the reference when one of its engagements falls in the possession. Ponta's call: opposed at
   any grid point of the possession. Also the two parts of the rule apart (engagement only; carrier pressure at
   6 to 9 or more only).
4. **FIFA's block rule** against P4's block and SkillCorner's out-of-possession phase (grid points where the team
   is out of possession), and the defending team's width and length during FIFA's mid-block (P2), beside FIFA's
   World Cup 2022 average (40.1 m wide, 27.2 m long; internal comparison only).
5. **Sweeps** (fit matches, P4v rerun on the same upstream outputs): engagement radius × closing speed; counter-press
   window; the depth that splits press from pressure; press gap.

Every measure on the even (fit) and odd (held-out) matches. Runtime of P4v per match. Reads the data outside the
repository and writes JSON outside it; no names are read.

    python tools/context_fifa_view.py --data ~/ponta-reference-data/skillcorner-opendata \
        --cache /tmp/ponta-context/sc-cache --out /tmp/ponta-context/fifa_view.json
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
from plaiq_engine.context.fifa_view import FifaViewPackage
from plaiq_engine.context.grid import GRID_COLUMNS
from plaiq_engine.context.line_breaks import LineBreakPackage
from plaiq_engine.context.phase import PhasePackage
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import ContextInput, Registry
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.context.thresholds import values
from plaiq_engine.context.zones import ZonesPackage
from plaiq_engine.match_analysis import MatchAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_reference_set import _load

TYPE_MAP = {"pressing": "press", "pressure": "pressure", "counter_press": "counter_press", "recovery_press": "recovery"}
SWEEP_RADIUS = (6.0, 8.0, 10.0, 12.0)
SWEEP_CLOSING = (0.5, 1.0, 1.5, 2.0)
SWEEP_WINDOW = (3.0, 4.0, 5.0, 6.0)
SWEEP_GAP = (1.5, 2.0, 3.0, 4.0)
SWEEP_PRESSURE_DEPTH = (0.2, 0.25, 1 / 3, 0.4, 0.5)


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def kappa(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, object), np.asarray(b, object)
    if not len(a):
        return None
    labels = sorted(set(a) | set(b))
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == lab)) * float(np.mean(b == lab)) for lab in labels)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


class Match:
    """One SkillCorner match: the upstream context outputs, the grid clock and SkillCorner's references."""

    def __init__(self, match_dir: Path, cache: Path):
        self.name = match_dir.name
        m = _load(match_dir, cache)
        self.m = m
        ma = MatchAnalysis(m)
        cfg = values()
        res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), PhasePackage(), LineBreakPackage()]).run(
            m, config=cfg, extras={"analysis": ma})
        self.up, self.grid = res.outputs, res.frames[list(GRID_COLUMNS)]
        self.frames = res.frames
        self.ts = self.grid["frame_ts"].to_numpy(float)
        self.nf = self.grid["native_frame"].to_numpy(np.int64)
        meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
        home = meta["home_team"]["id"]
        self.pid = {p["id"]: f"{'home' if p['team_id'] == home else 'away'}_{p['number']}" for p in meta["players"]}
        ev = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", low_memory=False,
                         usecols=["event_type", "event_subtype", "frame_start", "frame_end", "player_id", "team_id",
                                  "pressing_chain", "pressing_chain_index", "team_in_possession_phase_type"])
        ev = ev.assign(team=np.where(ev["team_id"] == home, "home", "away"),
                       t0=self.t_of(ev["frame_start"].to_numpy()), t1=self.t_of(ev["frame_end"].to_numpy()))
        self.eng = ev.loc[ev["event_type"] == "on_ball_engagement"].assign(player=lambda d: d["player_id"].map(self.pid))
        self.poss = ev.loc[ev["event_type"] == "player_possession"]
        ph = pd.read_csv(match_dir / f"{match_dir.name}_phases_of_play.csv",
                         usecols=["frame_start", "frame_end", "team_in_possession_id", "team_out_of_possession_phase_type"])
        self.sc_out = np.full(len(self.grid), None, dtype=object)
        self.sc_out_team = np.full(len(self.grid), None, dtype=object)
        for r in ph.itertuples():
            sel = (self.nf >= r.frame_start) & (self.nf <= r.frame_end)
            self.sc_out[sel] = r.team_out_of_possession_phase_type
            self.sc_out_team[sel] = "away" if r.team_in_possession_id == home else "home"

    def t_of(self, frames: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(frames, float), self.nf.astype(float), self.ts)

    def run(self, cfg: dict):
        pkg = FifaViewPackage()
        ctx = ContextInput(match=self.m, grid=self.grid, config=cfg, upstream={k: self.up[k] for k in pkg.depends_on})
        t0 = time.perf_counter()
        out = pkg.compute(ctx)
        return out, time.perf_counter() - t0


def engagement_scores(M: Match, out) -> dict:
    ours = out.segments["engagements"]
    sc = M.eng.dropna(subset=["player"])
    found, types = 0, []
    by_player = {p: g for p, g in ours.groupby("player")}
    for r in sc.itertuples():
        g = by_player.get(r.player)
        if g is None:
            continue
        hit = g.loc[(g["start_ts"] <= r.t1 + 1.0) & (g["end_ts"] >= r.t0 - 1.0)]
        if len(hit):
            found += 1
            if r.event_subtype in TYPE_MAP:
                types.append((TYPE_MAP[r.event_subtype], hit.iloc[0]["type"]))
    sc_by = {p: g for p, g in sc.groupby("player")}
    matched = sum(1 for r in ours.itertuples() if (g := sc_by.get(r.player)) is not None
                  and ((g["t0"] <= r.end_ts + 1.0) & (g["t1"] >= r.start_ts - 1.0)).any())
    return {"sc": len(sc), "found": found, "ours": len(ours), "matched": matched, "types": types}


def press_scores(M: Match, out) -> dict:
    presses = out.segments["presses"]
    ts = M.ts
    chains = M.eng.loc[M.eng["pressing_chain"] == True]
    res = {"chain_pts": 0, "chain_pts_in_press": 0, "press_pts": 0, "press_pts_in_chain": 0, "cp_chains": 0, "cp_found": 0}
    for t in ("home", "away"):
        in_chain = np.zeros(len(ts), bool)
        in_chain_wide = np.zeros(len(ts), bool)
        for _, c in chains.loc[chains["team"] == t].groupby("pressing_chain_index"):
            a, b = c["t0"].min(), c["t1"].max()
            in_chain |= (ts >= a) & (ts <= b)
            in_chain_wide |= (ts >= a - 1) & (ts <= b + 1)
            first = c.sort_values("t0").iloc[0]["event_subtype"]
            if first == "counter_press":
                res["cp_chains"] += 1
                p = presses.loc[(presses["team"] == t) & (presses["type"] == "counter_press")]
                res["cp_found"] += int(((p["start_ts"] <= b + 1) & (p["end_ts"] >= a - 1)).any())
        in_press = np.zeros(len(ts), bool)
        for r in presses.loc[presses["team"] == t].itertuples():
            in_press |= (ts >= r.start_ts) & (ts <= r.end_ts)
        res["chain_pts"] += int(in_chain.sum())
        res["chain_pts_in_press"] += int((in_chain & in_press).sum())
        res["press_pts"] += int(in_press.sum())
        res["press_pts_in_chain"] += int((in_press & in_chain_wide).sum())
    return res


def build_up_rows(M: Match, out) -> pd.DataFrame:
    """SkillCorner's build-up possessions that P4 also calls build-up, with the reference and Ponta's calls."""
    ph4 = M.up["p4"].frames["p4_phase"].astype(object).to_numpy()
    view = out.frames["p4v_phase"].astype(object).to_numpy()
    engaged = out.frames["p4v_engaging_defenders"].to_numpy(dtype=float, na_value=np.nan) > 0
    pressure = M.up["p5"].frames["p5_carrier_pressure"].to_numpy(dtype=float, na_value=np.nan)
    rows = []
    for r in M.poss.loc[M.poss["team_in_possession_phase_type"] == "build_up"].itertuples():
        sel = (M.ts >= r.t0) & (M.ts <= r.t1)
        if not sel.any() or not (ph4[sel] == "build_up").any():
            continue
        e = M.eng.loc[(M.eng["team"] != r.team) & (M.eng["t0"] <= r.t1) & (M.eng["t1"] >= r.t0)]
        rows.append({"match": M.name, "reference": bool(len(e)), "ponta": bool((view[sel] == "build_up_opposed").any()),
                     "engaged": bool(engaged[sel].any()), **{f"pressure_{k}": bool((pressure[sel] >= k).any()) for k in (6, 7, 8, 9)}})
    return pd.DataFrame(rows)


def block_rows(M: Match, out) -> pd.DataFrame:
    f4 = M.up["p4"].frames
    team4 = f4["p4_team"].astype(object).to_numpy()
    out_team = np.array([{"home": "away", "away": "home"}.get(t) for t in team4], dtype=object)
    sel = pd.notna(out_team)
    fr = M.frames
    width = np.where(out_team == "home", fr["p2_home_width_m"].to_numpy(float, na_value=np.nan), fr["p2_away_width_m"].to_numpy(float, na_value=np.nan))
    length = np.where(out_team == "home", fr["p2_home_length_m"].to_numpy(float, na_value=np.nan), fr["p2_away_length_m"].to_numpy(float, na_value=np.nan))
    sc = np.where(M.sc_out_team == out_team, M.sc_out, None)
    return pd.DataFrame({"match": M.name, "p4_block": f4["p4_block"].astype(object).to_numpy(), "fifa": out.frames["p4v_fifa_block"].astype(object).to_numpy(),
                         "sc_out": sc, "width": width, "length": length})[sel]


def summarise(E: list[dict], P: list[dict], B: pd.DataFrame, K: pd.DataFrame) -> dict:
    sc, found, ours, matched = (sum(e[k] for e in E) for k in ("sc", "found", "ours", "matched"))
    types = [t for e in E for t in e["types"]]
    a = np.array([x for x, _ in types], object)
    b = np.array([y for _, y in types], object)
    pr = {k: sum(p[k] for p in P) for k in P[0]} if P else {}
    out = {"engagements": {"sc": sc, "found_share": round(found / sc, 3) if sc else None, "ponta": ours,
                           "precision": round(matched / ours, 3) if ours else None,
                           "type_pairs": len(types), "type_agreement": round(float(np.mean(a == b)), 3) if len(a) else None,
                           "type_kappa": kappa(a, b),
                           "type_recall": {t: round(float(np.mean(b[a == t] == t)), 3) for t in TYPE_MAP.values() if (a == t).any()},
                           "confusion": pd.crosstab(pd.Series(a, name="sc"), pd.Series(b, name="ponta")).to_dict()},
           "presses": {"chain_time_in_press": round(pr["chain_pts_in_press"] / pr["chain_pts"], 3) if pr.get("chain_pts") else None,
                       "press_time_in_chain": round(pr["press_pts_in_chain"] / pr["press_pts"], 3) if pr.get("press_pts") else None,
                       "counter_press_chains": pr.get("cp_chains"), "counter_press_chains_found": pr.get("cp_found")}}
    if len(B):
        out["build_up"] = {"possessions": int(len(B)), "reference_opposed_share": round(float(B["reference"].mean()), 3),
                           **{c: {"agreement": round(float((B[c] == B["reference"]).mean()), 3), "kappa": kappa(B[c].to_numpy(), B["reference"].to_numpy()),
                                  "opposed_share": round(float(B[c].mean()), 3)}
                              for c in ["ponta", "engaged"] + [f"pressure_{k}" for k in (6, 7, 8, 9)]}}
    if len(K):
        out["fifa_block"] = {
            "share_of_out_of_possession_points": {k: round(float((K["fifa"] == k).mean()), 3) for k in ("mid_block", "low_block")},
            "by_p4_block": pd.crosstab(K["p4_block"].fillna("none"), K["fifa"].fillna("none"), normalize="index").round(3).to_dict(),
            "by_skillcorner_out_phase": pd.crosstab(K["sc_out"].fillna("none"), K["fifa"].fillna("none"), normalize="index").round(3).to_dict(),
            "mid_block_width_length_median_m": [round(float(K.loc[K["fifa"] == "mid_block", c].median()), 1) for c in ("width", "length")],
            "p4_medium_block_width_length_median_m": [round(float(K.loc[K["p4_block"] == "medium", c].median()), 1) for c in ("width", "length")],
            "fifa_world_cup_2022_mid_block_m": [40.1, 27.2]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()
    for p in (args.data, args.cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    dirs = [d for d in sorted((args.data / "matches").iterdir()) if (d / f"{d.name}_phases_of_play.csv").exists()]
    if args.matches:
        dirs = [d for d in dirs if d.name in args.matches]
    cfg = values()
    halves: dict = {"fit": {"E": [], "P": [], "B": [], "K": []}, "test": {"E": [], "P": [], "B": [], "K": []}}
    sweeps: dict = {}
    runtimes = []
    for n, d in enumerate(dirs):
        M = Match(d, args.cache)
        out, rt = M.run(cfg)
        runtimes.append(rt)
        h = halves["fit" if n % 2 == 0 else "test"]
        h["E"].append(engagement_scores(M, out))
        h["P"].append(press_scores(M, out))
        h["B"].append(build_up_rows(M, out))
        h["K"].append(block_rows(M, out))
        if n % 2 == 0 and not args.no_sweep:
            for r in SWEEP_RADIUS:
                for c in SWEEP_CLOSING:
                    o, _ = M.run({**cfg, "p4v_engage_radius_m": r, "p4v_engage_closing_mps": c})
                    sweeps.setdefault(f"radius_{r:g}_closing_{c:g}", {"E": [], "P": []})["E"].append(engagement_scores(M, o))
            for w in SWEEP_WINDOW:
                o, _ = M.run({**cfg, "p4v_counter_press_s": w})
                sweeps.setdefault(f"counter_press_{w:g}s", {"E": [], "P": []})
                sweeps[f"counter_press_{w:g}s"]["E"].append(engagement_scores(M, o))
                sweeps[f"counter_press_{w:g}s"]["P"].append(press_scores(M, o))
            for sh in SWEEP_PRESSURE_DEPTH:
                o, _ = M.run({**cfg, "p4v_pressure_max_depth_share": sh})
                sweeps.setdefault(f"pressure_depth_{sh:.2f}", {"E": [], "P": []})["E"].append(engagement_scores(M, o))
            for gp in SWEEP_GAP:
                o, _ = M.run({**cfg, "p4v_press_gap_s": gp})
                sweeps.setdefault(f"press_gap_{gp:g}s", {"E": [], "P": []})["P"].append(press_scores(M, o))
        print(d.name, round(rt, 1), "s", len(out.segments["engagements"]), "engagements", len(out.segments["presses"]), "presses", flush=True)
    result = {h: summarise(v["E"], v["P"], pd.concat(v["B"], ignore_index=True) if v["B"] else pd.DataFrame(),
                           pd.concat(v["K"], ignore_index=True) if v["K"] else pd.DataFrame())
              for h, v in halves.items() if v["E"]}
    result["sweeps_fit"] = {}
    for k, v in sweeps.items():
        s = summarise(v["E"], v["P"], pd.DataFrame(), pd.DataFrame()) if v["E"] else {}
        if v["P"] and not v["E"]:
            s = summarise([{"sc": 0, "found": 0, "ours": 0, "matched": 0, "types": []}], v["P"], pd.DataFrame(), pd.DataFrame())
        e = s.get("engagements", {})
        result["sweeps_fit"][k] = {"found_share": e.get("found_share"), "precision": e.get("precision"), "type_kappa": e.get("type_kappa"),
                                   "counter_press_recall": e.get("type_recall", {}).get("counter_press"), **s.get("presses", {})}
    result["p4v_runtime_s"] = {"median": round(float(np.median(runtimes)), 2), "max": round(float(np.max(runtimes)), 2)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "sweeps_fit"}, indent=1, default=str))


if __name__ == "__main__":
    main()
