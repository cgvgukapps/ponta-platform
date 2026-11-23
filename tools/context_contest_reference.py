"""
context_contest_reference.py — grounding ``p1_contest_max_s`` in the leagues' own event coding (23 November 2026).

``p1_contest_max_s`` (4 s, agreed on 20 November 2026 from the founder's possession calls) caps how long the ball stays
in contest after a logged duel, clearance, block or deflection when no team takes control. The founder asked for
decisions grounded in standards and professional reference data rather than his calls, so this reads the reference:
in the German league's official events (IDSSE, 7 matches) and Metrica's logged events (3 matches), how long after a
contest event does the next **controlled action** come (a pass, a failed pass, a shot or a recovery: a player has the
ball)? Contests followed by a dead ball (a set piece or the ball out) before any controlled action are counted apart.

Reports the distribution of that time (median, 75th, 90th and 95th percentiles) and the share of contests resolved
within 2, 4 and 6 s, per source. Reads data outside the repository; prints JSON; no names are read.

    python tools/context_contest_reference.py --idsse ~/ponta-reference-data/idsse --idsse-cache /tmp/ponta-context/idsse-cache \
        --metrica-cache /tmp/ponta-context/cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matrix_engine.sample_data import repo_root

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_measure import load as load_sample
from idsse import MATCH_IDS, load_idsse_match

CONTEST = ("CHALLENGE", "CLEARANCE", "BLOCK", "DEFLECTION")
CONTROLLED = ("PASS", "BALL LOST", "SHOT", "RECOVERY")
DEAD = ("SET PIECE", "BALL OUT")


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


def gaps(m) -> tuple[list[float], int]:
    """Seconds from each contest event to the next controlled action; and how many contests met a dead ball first."""
    ev = m.events.dropna(subset=["Start Frame"]).sort_values("Start Frame", kind="stable")
    typ = ev["Type"].fillna("").astype(str).str.upper().to_numpy()
    sub = ev["Subtype"].fillna("").astype(str).str.upper().to_numpy() if "Subtype" in ev else np.full(len(ev), "")
    fr = ev["Start Frame"].to_numpy(float)
    per = ev["Period"].to_numpy()
    is_contest = np.array([t in CONTEST or any(c in s for c in ("CLEARANCE", "BLOCK", "DEFLECTION")) for t, s in zip(typ, sub, strict=True)])
    out, dead = [], 0
    for i in np.flatnonzero(is_contest):
        for j in range(i + 1, len(ev)):
            if per[j] != per[i]:
                break
            if is_contest[j]:
                continue  # another contest touch: still contested
            if typ[j] in DEAD:
                dead += 1
                break
            if typ[j] in CONTROLLED:
                out.append((fr[j] - fr[i]) / float(m.fps))
                break
    return out, dead


def summary(g: list[float], dead: int, n_matches: int) -> dict:
    a = np.asarray(g, float)
    return {"matches": n_matches, "contests_resolved_by_control": int(len(a)), "contests_ending_dead": dead,
            "seconds_p50_p75_p90_p95": [round(float(np.percentile(a, q)), 2) for q in (50, 75, 90, 95)] if len(a) else None,
            "resolved_within": {f"{s}s": round(float(np.mean(a <= s)), 3) for s in (2, 4, 6)} if len(a) else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idsse", type=Path, required=True)
    ap.add_argument("--idsse-cache", type=Path, required=True)
    ap.add_argument("--metrica-cache", type=Path, required=True)
    args = ap.parse_args()
    for p in (args.idsse, args.idsse_cache, args.metrica_cache):
        if not _outside(p):
            sys.exit(f"refusing to read sample data inside the repository: {p}")
    out = {}
    g_all, d_all = [], 0
    for mid in MATCH_IDS:
        m, _ = load_idsse_match(args.idsse, mid, args.idsse_cache)
        g, d = gaps(m)
        g_all += g
        d_all += d
    out["german_league_official"] = summary(g_all, d_all, len(MATCH_IDS))
    g_all, d_all = [], 0
    for key in ("metrica_1", "metrica_2", "metrica_3"):
        g, d = gaps(load_sample(key, args.metrica_cache))
        g_all += g
        d_all += d
    out["metrica_logged"] = summary(g_all, d_all, 3)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
