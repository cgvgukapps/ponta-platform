"""
context_event_chains.py — what an event feed alone says about possession, and the patterns at scale.

A. **Event-only possession against tracking** (Metrica Sample Games 1 to 3; the seven IDSSE
   matches): possession chains from the events alone (``event_chains.py``, after Pappalardo et al.
   2019) against P1's tracking-based possession on the context grid, and, on IDSSE, both against
   the league's official frame-level possession.
B. **Patterns at scale** (Wyscout's open events, seven competitions of 2017/18): possessions per
   match, duration, events per possession, how possessions start and end, how far and how fast they
   move forward, and Wyscout's counter-attack tag. The same statistics from the same rule on the
   Metrica and IDSSE feeds, and from P1's tracking chains, sit beside them.

Wyscout data: Pappalardo et al. (2019), Scientific Data 6, 236, CC BY 4.0. IDSSE data: Deutsche
Fußball Liga (DFL); Bassek et al. (2025), Scientific Data 12, 195, CC BY 4.0. All data is read
outside the repository; no names are read (Wyscout's players file is never opened).

    python tools/context_event_chains.py --cache /tmp/ponta-context/cache --idsse ~/ponta-reference-data/idsse \
        --idsse-cache /tmp/ponta-context/idsse-cache --wyscout ~/ponta-reference-data/wyscout-opendata \
        --out /tmp/ponta-context/event_chains.json
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_measure import load
from event_chains import chains, from_metrica, from_wyscout, on_grid
from idsse import MATCH_IDS, load_idsse_match

COUNTER_ATTACK_TAG = 1901
FINAL_THIRD = 2 / 3


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def stats(ch: pd.DataFrame, matches: int, length: float = 105.0) -> dict:
    """Possession patterns from a table of chains (event or P1)."""
    if not len(ch):
        return {}
    d = ch["duration_s"].astype(float)
    gain = (ch["max_depth"] - ch["start_depth"]).astype(float) if "max_depth" in ch else pd.Series(dtype=float)
    long = (d >= 3) & gain.notna() if len(gain) else pd.Series(False, index=ch.index)
    start = ch["start_type"].astype(str).str.replace(r"^set_piece:", "restart:", regex=True)
    out = {
        "possessions_per_match": round(len(ch) / matches, 1),
        "duration_s": {"median": round(float(d.median()), 1), "p75": round(float(d.quantile(0.75)), 1), "p90": round(float(d.quantile(0.9)), 1)},
        "start": {k: round(float(v), 3) for k, v in start.map(lambda s: s if s.startswith("restart:") or s in ("regain", "won", "recovery") else "other")
                  .replace({"won": "regain", "recovery": "regain"}).value_counts(normalize=True).items()},
        "end": {k: round(float(v), 3) for k, v in ch["end_type"].value_counts(normalize=True).items()},
    }
    if "events" in ch:
        out["events_per_possession_median"] = float(ch["events"].median())
    if len(gain):
        out["reach_final_third"] = round(float((ch["max_depth"] >= FINAL_THIRD * length).mean()), 3)
        out["forward_speed_mps"] = {"median": round(float((gain[long] / d[long]).median()), 2),
                                    "p90": round(float((gain[long] / d[long]).quantile(0.9)), 2)} if long.any() else None
    return out


def part_a(cache: Path, idsse: Path, idsse_cache: Path) -> tuple[dict, dict]:
    report, pooled = {}, {"event": [], "p1": [], "matches": 0}
    jobs = [("metrica", k) for k in ("metrica_1", "metrica_2", "metrica_3")] + [("idsse", k) for k in MATCH_IDS]
    for source, key in jobs:
        official = None
        if source == "metrica":
            m = load(key, cache)
        else:
            m, official = load_idsse_match(idsse, key, idsse_cache)
        res = Registry([PossessionPackage()]).run(m)
        g = res.frames
        ts, per = g["frame_ts"].to_numpy(float), g["period"].to_numpy()
        f0 = int(m.frames["frame"].min())
        ev = from_metrica(m.events, m.fps, f0, m.pitch.length_m, {1: 1, 2: 1})
        ch = chains(ev)
        ev_lab = on_grid(ch, ts, per)
        p1 = g["p1_possession"].astype(object).to_numpy()
        both = np.isin(ev_lab, ["home", "away"]) & np.isin(p1, ["home", "away"])
        entry = {
            "event_chains": len(ch), "p1_chains": int(len(res.outputs["p1"].segments["chains"])),
            "event_vs_p1_agreement": round(float((ev_lab[both] == p1[both]).mean()), 4),
            "event_labelled_share_of_p1_labelled": round(float(np.isin(ev_lab, ["home", "away"])[np.isin(p1, ["home", "away"])].mean()), 4),
        }
        if official is not None:
            ob = g[["native_frame"]].merge(official.ball.rename(columns={"frame": "native_frame"}), on="native_frame", how="left")
            live, team = (ob["ball_status"] == 1).to_numpy(), ob["ball_team"].astype(object).to_numpy()
            for name, lab in (("event", ev_lab), ("p1", p1)):
                ok = live & np.isin(lab, ["home", "away"]) & np.isin(team, ["home", "away"])
                entry[f"{name}_vs_official"] = round(float((lab[ok] == team[ok]).mean()), 4)
                entry[f"{name}_labelled_share_in_play"] = round(float(np.isin(lab[live], ["home", "away"]).mean()), 4)
        p1c = res.outputs["p1"].segments["chains"].assign(max_depth=np.nan, start_depth=np.nan)
        entry["patterns_event"] = stats(ch, 1, m.pitch.length_m)
        entry["patterns_p1"] = stats(p1c.drop(columns=["max_depth", "start_depth"]), 1, m.pitch.length_m)
        report[key] = entry
        pooled["event"].append(ch)
        pooled["p1"].append(p1c.drop(columns=["max_depth", "start_depth"]))
        pooled["matches"] += 1
        print(key, json.dumps({k: v for k, v in entry.items() if not k.startswith("patterns")}), flush=True)
    summary = {"event_rule_on_these_feeds": stats(pd.concat(pooled["event"]), pooled["matches"]),
               "p1_tracking_chains": stats(pd.concat(pooled["p1"]), pooled["matches"])}
    return report, summary


def part_b(wyscout: Path) -> dict:
    out = {}
    with zipfile.ZipFile(wyscout / "events.zip") as z:
        for name in sorted(z.namelist()):
            comp = name.removeprefix("events_").removesuffix(".json")
            raw = json.loads(z.read(name))
            df = pd.DataFrame([{k: e[k] for k in ("matchId", "teamId", "eventName", "subEventName", "matchPeriod", "eventSec", "positions")}
                               | {"counter": any(t["id"] == COUNTER_ATTACK_TAG for t in e["tags"])} for e in raw])
            del raw
            parts, counter_poss = [], []
            for _, ev in df.groupby("matchId", sort=True):
                c = from_wyscout(ev)
                ch = chains(c)
                parts.append(ch)
                # share of possessions with at least one event Wyscout tagged as a counter-attack
                tagged = ev.loc[ev["counter"]]
                if len(ch):
                    hit = np.zeros(len(ch), bool)
                    per = ev["matchPeriod"].map({"1H": 1, "2H": 2, "E1": 3, "E2": 4}).to_numpy()
                    for p_, t_, tm in zip(per[ev["counter"].to_numpy()], tagged["eventSec"], tagged["teamId"].astype(str), strict=True):
                        hit |= (ch["period"].to_numpy() == p_) & (ch["team"].to_numpy() == tm) & (ch["start_t"].to_numpy() <= t_) & (ch["end_t"].to_numpy() >= t_)
                    counter_poss.append(hit.mean())
            allc = pd.concat(parts)
            out[comp] = stats(allc, df["matchId"].nunique()) | {"matches": int(df["matchId"].nunique()), "events": int(len(df)),
                                                               "counter_attack_possession_share": round(float(np.mean(counter_poss)), 4)}
            print(comp, json.dumps({k: out[comp][k] for k in ("matches", "events", "possessions_per_match", "duration_s")}), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--idsse", type=Path, required=True)
    ap.add_argument("--idsse-cache", type=Path, required=True)
    ap.add_argument("--wyscout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.cache, args.idsse, args.idsse_cache, args.wyscout, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    a, a_sum = part_a(args.cache, args.idsse, args.idsse_cache)
    b = part_b(args.wyscout)
    out = {"part_a_matches": a, "part_a_summary": a_sum, "part_b_wyscout": b}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"part_a_summary": a_sum}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
