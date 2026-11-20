"""
context_line_breaks.py — a line-break rule on the FIFA definition, measured against SkillCorner's labels.

The measurement behind context package P5's line breaks (``plaiq_engine/context/line_breaks.py``,
which holds the rule; ``--package`` scores the package's own output). The rule follows the FIFA Football
Language and EFI documents (``ponta-blueprint/fifa-alignment.md``):

- **units** are the defending team's lines (P2's per-window line membership: defence, midfield,
  attack; goalkeeper excluded) at the moment of the pass; a unit counts only with at least two
  observed players ("a unit line must consist of at least two players");
- a unit's **line** is its deepest player (nearest its own goal); a pass **breaks** the unit when
  the ball starts in front of that line and is received beyond it ("the attacking team play the
  ball beyond the deepest player in that line");
- **through** when the ball crosses the line's depth between the unit's two widest players,
  **around** otherwise (FIFA's third type, over, needs the ball's height, which SkillCorner lacks).

Passes are the match's event rows (``PASS`` completed, ``BALL LOST`` failed) with start and end
positions. Against SkillCorner's dynamic events on its 20 open-data matches (comparison only,
founder decision 2): the furthest line broken (none, first, second-last, last) per pass where
SkillCorner knows the number of defensive lines, each line's break, and through or around.
Writes JSON outside the repository; no names are read.

    python tools/context_line_breaks.py --data ~/ponta-reference-data/skillcorner-opendata \
        --cache /tmp/ponta-context/sc-cache --out /tmp/ponta-context/line_breaks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.line_breaks import breaks, furthest, unit_names
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.structure import (
    OTHER,
    StructurePackage,
    assign_lines,
    line_members,
    team_frames,
    team_states,
    window_index,
    windows_from_segments,
)
from plaiq_engine.context.thresholds import values
from plaiq_engine.context.zones import ZonesPackage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_reference_set import _kappa, _load

MIN_UNIT_PLAYERS = 2


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def sc_passes(match_dir: Path, home_id: int) -> pd.DataFrame:
    cols = ["event_type", "end_type", "frame_end", "team_id", "pass_outcome", "n_defensive_lines", "first_line_break",
            "second_last_line_break", "last_line_break", "furthest_line_break", "furthest_line_break_type", "n_passing_options_line_break"]
    ev = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", usecols=cols, low_memory=False)
    p = ev.loc[(ev["event_type"] == "player_possession") & (ev["end_type"] == "pass")].copy()
    p["team"] = np.where(p["team_id"] == home_id, "home", "away")
    return p


def plaiq_pov(m, cache: Path) -> pd.DataFrame:
    """Plaiq's pass option value per event and teammate (``MatchAnalysis.event_pov``). Cached: it takes minutes."""
    p = cache / f"pov_{m.provider}_{m.match_id}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    from plaiq_engine.match_analysis import MatchAnalysis
    pov = MatchAnalysis(m).event_pov()
    pov.to_pickle(p)
    return pov


def intents(pov: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """
    Plaiq's intended meeting point for each failed pass (event row index to x, y): the ball's line
    walked past the interception to the first teammate who would have controlled it, refined by
    that teammate's run (``pass_option_value._evaluate_failed_pass``).
    """
    rows = pov.loc[pov["intent_x"].notna(), ["event_idx", "intent_x", "intent_y"]].drop_duplicates("event_idx")
    return {int(r.event_idx): (float(r.intent_x), float(r.intent_y)) for r in rows.itertuples()}


def one_package(match_dir: Path, cache: Path, thetas: tuple[float, ...] = ()) -> pd.DataFrame:
    """
    P5's own output (``context.line_breaks``) joined to SkillCorner's passes: completed and failed (0.2.0: a failed
    pass by its logged target's projected point), with the count of line-breaking options at each ``theta`` (one
    run of the package per value; the first run's labels are kept) and the package's runtime.
    """
    import time

    from plaiq_engine.context.line_breaks import LineBreakPackage
    m = _load(match_dir, cache)
    seg, counts, runtime = None, {}, None
    for th in thetas or (values()["p5_lb_option_min_p"],):
        t0 = time.perf_counter()
        res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), LineBreakPackage()]).run(m, config=values({"p5_lb_option_min_p": th}))
        runtime = runtime or round(time.perf_counter() - t0, 1)
        s_ = res.outputs["p5"].segments["line_breaks"]
        seg = s_ if seg is None else seg
        counts[th] = s_.set_index("event_row")["line_breaking_options"]
    seg = seg.assign(start_frame=lambda d: m.events.loc[d["event_row"], "Start Frame"].to_numpy())
    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    ref = sc_passes(match_dir, meta["home_team"]["id"])
    rows = []
    for r in ref.itertuples():
        c = seg.loc[(seg["team"] == r.team) & ((seg["start_frame"] - r.frame_end).abs() <= 2)]
        if c.empty:
            continue
        x = c.iloc[0]
        rows.append({"match": match_dir.name, "outcome": x["outcome"], "target": x["target"], "runtime_s": runtime,
                     "sc_lb_options": r.n_passing_options_line_break,
                     **{f"lb_options_{th:g}": counts[th].get(x["event_row"]) for th in counts},
                     "sc_lines": r.n_defensive_lines,
                     "sc_furthest": r.furthest_line_break if isinstance(r.furthest_line_break, str) else "none", "sc_type": r.furthest_line_break_type,
                     "sc_first": r.first_line_break, "sc_second_last": r.second_last_line_break, "sc_last": r.last_line_break,
                     "ponta_furthest": x["furthest"] if isinstance(x["furthest"], str) else "none", "ponta_type": x["type"],
                     "ponta_first": x["first"], "ponta_second_last": x["second_last"], "ponta_last": x["last"]})
    return pd.DataFrame(rows)


def one(match_dir: Path, cache: Path, unit_mode: str = "window", use_intent: bool = False, options: bool = False) -> pd.DataFrame:
    m = _load(match_dir, cache)
    pov = plaiq_pov(m, cache) if (use_intent or options) else None
    intent = intents(pov) if use_intent else {}
    pov_by_event = {k: g for k, g in pov.groupby("event_idx")} if options else {}
    cfg = values()
    res = Registry([PossessionPackage(), StructurePackage()]).run(m, config=cfg)
    g, p1, p2 = res.frames, res.outputs["p1"], res.outputs["p2"]
    tf = team_frames(m, g, p2.segments["orientation"])
    lines = line_members(g, tf, team_states(g, p1.frames, p1.segments["chains"], cfg), window_index(g, cfg), windows_from_segments(p2.segments))
    sign = {(int(r.period), r.team): r.attack_sign for r in p2.segments["orientation"].itertuples()}
    nf, per = g["native_frame"].to_numpy(), g["period"].to_numpy()
    L = m.pitch.length_m
    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    ref = sc_passes(match_dir, meta["home_team"]["id"])
    ev = m.events.loc[m.events["Type"].isin(["PASS", "BALL LOST"])].dropna(subset=["Start X", "End X"])
    rows = []
    for r in ref.itertuples():
        cand = ev.loc[(ev["Team"].str.lower() == r.team) & ((ev["Start Frame"] - r.frame_end).abs() <= 2)]
        if cand.empty:
            continue
        e = cand.iloc[0]
        k = int(np.clip(np.searchsorted(nf, int(e["Start Frame"])), 0, len(nf) - 1))
        t, d = r.team, OTHER[r.team]
        s = sign.get((int(per[k]), t), 1)
        start = (s * float(e["Start X"]) + L / 2, s * float(e["Start Y"]))
        end = (s * float(e["End X"]) + L / 2, s * float(e["End Y"]))
        used_intent = e["Type"] == "BALL LOST" and int(e.name) in intent
        if used_intent:
            # FIFA judges a failed pass by where the intended team-mate was: Plaiq's intended meeting point.
            ix, iy = intent[int(e.name)]
            end = (s * ix + L / 2, s * iy)
        D, Wd, Ld = tf[d].depth[k], tf[d].width[k], lines[d][k]
        if unit_mode == "moment":
            # FIFA: units are "groups of players that act in similar roles at the moment of the pass":
            # the defending outfield players grouped by depth at this moment (exact 3-means).
            out = np.isfinite(D) & (Ld != 3)
            Ld = np.full(len(D), -1)
            if out.sum() >= 3:
                Ld[out] = assign_lines(D[out])
        units = {}
        for code in (0, 1, 2):
            sel = np.isfinite(D) & (Ld == code)
            units[code] = np.column_stack([L - D[sel], -Wd[sel]])  # the defenders in the attacking team's frame
        b = breaks(start, end, units)
        # Line-breaking options: teammates (not the passer) beyond at least one defending line from
        # the ball, with Plaiq's option value at the pass (kept per option, so a threshold can be swept).
        opt_values = []
        if options:
            A = tf[t]
            col = {pid: i for i, pid in enumerate(A.ids)}
            for o in pov_by_event.get(int(e.name), pd.DataFrame()).itertuples():
                pid = f"{t}_{o.jersey}"
                if str(o.jersey) == str(e["From"]) or pid not in col or not np.isfinite(A.depth[k, col[pid]]):
                    continue
                ob = breaks(start, (A.depth[k, col[pid]], A.width[k, col[pid]]), units)
                if any(ob.values()):
                    opt_values.append(round(float(o.pov_start), 4))
        rows.append({"match": match_dir.name, "outcome": "completed" if e["Type"] == "PASS" else "failed", "intent": bool(used_intent),
                     "sc_lines": r.n_defensive_lines, "sc_furthest": r.furthest_line_break if isinstance(r.furthest_line_break, str) else "none",
                     "sc_type": r.furthest_line_break_type, "sc_first": r.first_line_break, "sc_second_last": r.second_last_line_break,
                     "sc_last": r.last_line_break, "ponta_furthest": furthest(b),
                     "ponta_type": next((b[c] for c in (0, 1, 2) if b.get(c)), None),
                     **{f"ponta_{name}": b.get(code) for name in ("first", "second_last", "last")
                        for code in [next((c for c, n in unit_names(b).items() if n == name), None)]},
                     "units_known": sum(v is not None for v in b.values()),
                     "sc_lb_options": r.n_passing_options_line_break, "lb_option_values": opt_values})
    return pd.DataFrame(rows)


THETAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)


def score_options(df: pd.DataFrame) -> dict:
    """Ponta's line-breaking options (option value at least theta) against SkillCorner's count, per pass."""
    k = df.loc[df["sc_lb_options"].notna() & df["sc_lines"].notna()]
    ref = k["sc_lb_options"].astype(int).to_numpy()
    out = {"passes": int(len(k)), "sc_mean": round(float(ref.mean()), 2) if len(k) else None}
    for th in THETAS:
        n = k["lb_option_values"].map(lambda v, th=th: sum(x >= th for x in v)).to_numpy()
        out[f"theta_{th:g}"] = {"any_option": _kappa((ref > 0).astype(str), (n > 0).astype(str)),
                                "exact": round(float((n == ref).mean()), 4), "within_1": round(float((abs(n - ref) <= 1).mean()), 4),
                                "ponta_mean": round(float(n.mean()), 2),
                                "spearman": round(float(pd.Series(n).corr(pd.Series(ref), method="spearman")), 4)}
    return out


def score_package_options(df: pd.DataFrame) -> dict:
    """P5's count of line-breaking options at each theta against SkillCorner's count, per pass."""
    k = df.loc[df["sc_lb_options"].notna()]
    ref = k["sc_lb_options"].astype(int).to_numpy()
    out = {"passes": int(len(k)), "sc_mean": round(float(ref.mean()), 2) if len(k) else None}
    for col in [c for c in df.columns if c.startswith("lb_options_")]:
        v = k[col]
        ok = v.notna().to_numpy()
        n, r = v[ok].astype(int).to_numpy(), ref[ok]
        out[col] = {"passes": int(ok.sum()), "any_option": _kappa((r > 0).astype(str), (n > 0).astype(str)),
                    "exact": round(float((n == r).mean()), 4), "within_1": round(float((abs(n - r) <= 1).mean()), 4),
                    "ponta_mean": round(float(n.mean()), 2), "spearman": round(float(pd.Series(n).corr(pd.Series(r), method="spearman")), 4)}
    return out


def score(df: pd.DataFrame) -> dict:
    known = df.loc[df["sc_lines"].notna()]
    a, b = known["sc_furthest"].to_numpy(str), known["ponta_furthest"].to_numpy(str)
    out = {"passes_matched": int(len(df)), "passes_with_sc_lines": int(len(known)), "furthest_line_broken": _kappa(a, b),
           "confusion": {r: {c: int(v) for c, v in row.items() if v} for r, row in pd.crosstab(pd.Series(a, name="sc"), pd.Series(b, name="ponta")).iterrows()},
           "any_break": _kappa((known["sc_furthest"] != "none").to_numpy(str), (known["ponta_furthest"] != "none").to_numpy(str))}
    for line in ("first", "second_last", "last"):
        k = known.loc[known[f"sc_{line}"].notna()]
        out[f"{line}_line"] = _kappa(k[f"sc_{line}"].astype(bool).to_numpy(str), k[f"ponta_{line}"].map(bool).to_numpy(str))
    both = known.loc[(known["sc_furthest"] != "none") & (known["sc_furthest"] == known["ponta_furthest"])]
    out["type_where_both_agree_on_the_line"] = _kappa(both["sc_type"].astype(str).to_numpy(), both["ponta_type"].astype(str).to_numpy())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--units", choices=("window", "moment"), default="window",
                    help="window: P2's per-window lines; moment: lines grouped at the pass moment (FIFA)")
    ap.add_argument("--intent", action="store_true", help="judge a failed pass by Plaiq's intended meeting point")
    ap.add_argument("--options", action="store_true", help="count line-breaking passing options (Plaiq option values)")
    ap.add_argument("--package", action="store_true", help="score P5's own output (completed and failed passes) instead of the prototype")
    ap.add_argument("--thetas", nargs="*", type=float, default=[], help="with --package: line-breaking option thresholds to sweep")
    args = ap.parse_args()
    for p in (args.data, args.cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    dirs = [d for d in sorted((args.data / "matches").iterdir()) if (d / f"{d.name}_dynamic_events.csv").exists()]
    if args.matches:
        dirs = [d for d in dirs if d.name in args.matches]
    parts = []
    for n, d in enumerate(dirs):
        df = one_package(d, args.cache, tuple(args.thetas)) if args.package else one(d, args.cache, args.units, args.intent, args.options)
        df.to_pickle(args.out.parent / f"line_breaks_{d.name}.pkl")
        df["set"] = "fit" if n % 2 == 0 else "test"
        parts.append(df)
        print(d.name, len(df), "passes", flush=True)
    allp = pd.concat(parts, ignore_index=True)
    out = {"all": score(allp), "fit": score(allp.loc[allp["set"] == "fit"]), "test": score(allp.loc[allp["set"] == "test"]),
           "by_outcome": {o: score(allp.loc[allp["outcome"] == o]) for o in ("completed", "failed") if (allp["outcome"] == o).any()}}
    if args.options:
        out["options"] = {s_: score_options(allp.loc[allp["set"] == s_]) for s_ in ("fit", "test")}
    if args.package:
        out["options"] = {s_: score_package_options(allp.loc[allp["set"] == s_]) for s_ in ("fit", "test")}
        rt = allp.drop_duplicates("match")["runtime_s"]
        out["runtime_s"] = {"median": float(rt.median()), "max": float(rt.max())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out["all"][k] for k in ("passes_matched", "passes_with_sc_lines", "furthest_line_broken", "any_break", "first_line", "second_last_line", "last_line", "type_where_both_agree_on_the_line")}, indent=1))
    print(json.dumps(out["all"]["confusion"], indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
