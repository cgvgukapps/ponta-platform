"""
context_contest.py — P1 0.3.0's in-contest state against the founder's possession calls (Stage C, deliverable 1).

``p1_state`` adds FIFA's possession states beside ``p1_possession`` (unchanged): the ball is in contest where
both teams are at it, and, with a logged feed, from a logged duel, clearance, block or deflection until a team
holds it (``p1_min_switch_s``), for at most ``p1_contest_max_s``. This scores the state at the 50 moments the
founder judged blind on the German league's matches (``tools/data/possession_review_2026-11-18.json``; home,
away or nobody), for several caps, and reports how much in-play time is in contest on IDSSE, Metrica and
SkillCorner. A "nobody" call is right when the state is ``in_contest``; a team call when it is that team's
possession. The founder's calls are the reference; nothing is tuned beyond the cap's few values.

Since P1 0.4.0 it also scores ``p1_ball_state`` the same way: a "nobody" call is right on ``loose`` or
``contested``, a team call on that team's ``controlled_`` or ``in_flight_`` state; for ``p1_loose_max_s`` of
3, 5 and 8 s.

Reads the data outside the repository; writes JSON outside it; no names are read.

    python tools/context_contest.py --idsse ~/ponta-reference-data/idsse --idsse-cache /tmp/ponta-context/idsse-cache-v3 \
        --cache /tmp/ponta-context/cache --sc-data ~/ponta-reference-data/skillcorner-opendata --sc-cache /tmp/ponta-context/sc-cache \
        --out /tmp/ponta-context/contest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from matrix_engine.sample_data import repo_root
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.thresholds import values

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_measure import load
from context_reference_set import _load
from idsse import MATCH_IDS, load_idsse_match

CALLS = Path(__file__).resolve().parent / "data" / "possession_review_2026-11-18.json"
CAPS = (2.0, 4.0, 6.0)
LOOSE_CAPS = (3.0, 5.0, 8.0)


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def ball_right(state: str, call: str) -> bool:
    return state in ("loose", "contested") if call == "nobody" else state in (f"controlled_{call}", f"in_flight_{call}")


def tally(calls: dict, states: dict[str, str], right) -> dict:
    r = {"nobody_right": 0, "nobody": 0, "team_right": 0, "team": 0, "controls_right": 0, "controls": 0, "right": 0}
    for cid, c in calls.items():
        ok = right(states.get(cid), c["call"])
        r["right"] += ok
        grp = "controls" if c["kind"] == "agree" else ("nobody" if c["call"] == "nobody" else "team")
        r[grp] += 1
        r[f"{grp}_right"] += ok
    return r


def ball_shares(frames) -> dict:
    bs = frames["p1_ball_state"].astype(object).to_numpy()
    play = bs != "out_of_play"
    n = max(int(play.sum()), 1)
    out = {}
    for k in ("controlled_home", "controlled_away", "contested", "in_flight_home", "in_flight_away", "loose", "unknown"):
        out[k.replace("_home", "").replace("_away", "")] = out.get(k.replace("_home", "").replace("_away", ""), 0.0) + float(((bs == k) & play).sum() / n)
    return {k: round(v, 4) for k, v in out.items()}


def shares(frames) -> dict:
    st = frames["p1_state"].astype(object).to_numpy()
    src = frames["p1_state_source"].astype(object).to_numpy()
    play = st != "out_of_play"
    n = max(int(play.sum()), 1)
    return {"in_contest": round(float(((st == "in_contest") & play).sum() / n), 4),
            "in_contest_from_events": round(float(((st == "in_contest") & (src == "events") & play).sum() / n), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idsse", type=Path, required=True)
    ap.add_argument("--idsse-cache", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True, help="pickled regression matches (as context_measure.py)")
    ap.add_argument("--sc-data", type=Path, required=True)
    ap.add_argument("--sc-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.idsse, args.idsse_cache, args.cache, args.sc_data, args.sc_cache, args.out):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    calls = {c["id"]: c for c in json.loads(CALLS.read_text())["cases"]}
    out = {"caps": {}, "shares": {}, "ball_state": {}, "ball_state_shares": {}}
    states: dict[float, dict[str, str]] = {cap: {} for cap in CAPS}
    bstates: dict[float, dict[str, str]] = {cap: {} for cap in LOOSE_CAPS}
    for mid in MATCH_IDS:
        m, _ = load_idsse_match(args.idsse, mid, args.idsse_cache)
        for cap in CAPS:
            f = Registry([PossessionPackage()]).run(m, config=values({"p1_contest_max_s": cap})).frames
            st = f["p1_state"].astype(object).to_numpy()
            for cid in calls:
                if cid.startswith(mid + "-"):
                    states[cap][cid] = st[int(cid.split("-")[1])]
            if cap == 4.0:
                out["shares"][f"idsse_{mid}"] = shares(f)
        for cap in LOOSE_CAPS:
            f = Registry([PossessionPackage()]).run(m, config=values({"p1_loose_max_s": cap})).frames
            bs = f["p1_ball_state"].astype(object).to_numpy()
            for cid in calls:
                if cid.startswith(mid + "-"):
                    bstates[cap][cid] = bs[int(cid.split("-")[1])]
            if cap == 5.0:
                out["ball_state_shares"][f"idsse_{mid}"] = ball_shares(f)
        print(mid, flush=True)
    for cap in CAPS:
        r = {"nobody_called_in_contest": 0, "nobody": 0, "team_kept": 0, "team": 0, "controls_kept": 0, "controls": 0, "right": 0}
        for cid, c in calls.items():
            s = states[cap].get(cid)
            right = (s == "in_contest") if c["call"] == "nobody" else (s == f"in_possession_{c['call']}")
            r["right"] += right
            if c["kind"] == "agree":
                r["controls"] += 1
                r["controls_kept"] += right
            elif c["call"] == "nobody":
                r["nobody"] += 1
                r["nobody_called_in_contest"] += right
            else:
                r["team"] += 1
                r["team_kept"] += right
        out["caps"][str(cap)] = r
    for cap in LOOSE_CAPS:
        out["ball_state"][str(cap)] = tally(calls, bstates[cap], ball_right)
    for name in ("metrica_1", "metrica_2", "metrica_3"):
        f = Registry([PossessionPackage()]).run(load(name, args.cache)).frames
        out["shares"][name], out["ball_state_shares"][name] = shares(f), ball_shares(f)
    sc = _load(args.sc_data / "matches" / "1886347", args.sc_cache)
    f = Registry([PossessionPackage()]).run(sc).frames
    out["shares"]["skillcorner_1886347"], out["ball_state_shares"]["skillcorner_1886347"] = shares(f), ball_shares(f)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
