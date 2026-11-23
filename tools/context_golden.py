"""
context_golden.py — golden-output harness for context packages (P0).

Runs every registered context package on the regression set (Metrica Sample Games 1 to 3,
SkillCorner 1886347), writes the outputs to a directory **outside the repository**, and
captures or checks a per-match summary: for each package field, a hash of its values and
counts; the Data check entries; versions; runtime and storage.

    python tools/context_golden.py --out /tmp/ponta-context/golden --cache /tmp/ponta-context/cache --capture
    python tools/context_golden.py --out /tmp/ponta-context/golden --cache /tmp/ponta-context/cache --check

Summaries live in ``tests/golden/context/`` and are git-ignored (they are derived from
licensed or unverified sample data); only ``manifest.json`` (versions, no data) is tracked.
SkillCorner's final score is read from its match file as numbers only, standing in for an
upload; no player name is read.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from matrix_engine.parity import parity_for_match
from matrix_engine.sample_data import repo_root, sample_data_dir
from plaiq_engine.context.fifa_view import FifaViewPackage
from plaiq_engine.context.line_breaks import LineBreakPackage
from plaiq_engine.context.movement import MovementPackage
from plaiq_engine.context.phase import PhasePackage
from plaiq_engine.context.possession import GameStateInput, PossessionPackage
from plaiq_engine.context.probe import ProbePackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.space_value import SpaceValuePackage
from plaiq_engine.context.states import StatesPackage
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.context.thresholds import THRESHOLDS_VERSION
from plaiq_engine.context.zones import ZonesPackage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_measure import SC_ID, load

MATCHES = ("metrica_1", "metrica_2", "metrica_3", "skillcorner_1886347")
GOLDEN = repo_root() / "tests" / "golden" / "context"


def registry() -> Registry:
    return Registry([ProbePackage(), PossessionPackage(), StructurePackage(), ZonesPackage(), PhasePackage(), LineBreakPackage(),
                     MovementPackage(), SpaceValuePackage(), FifaViewPackage(), StatesPackage()])


def extras_for(key: str) -> dict:
    if not key.startswith("skillcorner"):
        return {}
    meta = json.loads((Path(sample_data_dir()) / "skillcorner" / "matches" / str(SC_ID) / f"{SC_ID}_match.json").read_text())
    return {"game_state": GameStateInput(final_score={"home": int(meta["home_team_score"]), "away": int(meta["away_team_score"])})}


def column_hash(s: pd.Series) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(s.astype(object).where(s.notna(), None).astype(str), index=False).values.tobytes()).hexdigest()[:16]


def summarise(res, seconds: float, sizes: dict) -> dict:
    f = res.frames
    return {
        "versions": res.versions,
        "thresholds_version": res.thresholds_version,
        "grid_points": int(len(f)),
        "fields": {c: {"hash": column_hash(f[c]), "nulls": int(f[c].isna().sum())} for c in f.columns},
        "segments": {f"{p}_{n}": {"rows": int(len(s)), "hash": hashlib.sha256(pd.util.hash_pandas_object(s, index=False).values.tobytes()).hexdigest()[:16]}
                     for p, o in res.outputs.items() for n, s in o.segments.items()},
        "entries": [dataclasses.asdict(e) for e in res.entries],
        "runtime_s": round(seconds, 1),
        "storage_bytes": sizes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--matches", nargs="*", default=list(MATCHES))
    args = ap.parse_args()
    out, cache = Path(args.out).resolve(), Path(args.cache).resolve()
    root = repo_root().resolve()
    for p in (out, cache):
        if p == root or root in p.parents:
            print(f"refusing to write inside the repository: {p}", file=sys.stderr)
            return 2
    GOLDEN.mkdir(parents=True, exist_ok=True)
    reg, failures = registry(), 0
    for key in args.matches:
        m = load(key, cache)
        t0 = time.time()
        res = reg.run(m, extras=extras_for(key))
        report = res.attach_to(parity_for_match(m))
        seconds = time.time() - t0
        paths = res.write(out / key)
        (out / key / "data_check.json").write_text(json.dumps(report.to_dict(), indent=1, default=str))
        sizes = {k: os.path.getsize(p) for k, p in paths.items()}
        summary = summarise(res, seconds, sizes)
        target = GOLDEN / f"{key}.json"
        if args.capture:
            target.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
            print(f"{key}: captured ({summary['grid_points']} grid points, {summary['runtime_s']} s, {sum(sizes.values()) / 1e6:.1f} MB)")
            continue
        if not target.exists():
            print(f"{key}: no golden summary (run --capture first)")
            failures += 1
            continue
        gold = json.loads(target.read_text())
        diffs = [c for c in sorted(set(gold["fields"]) | set(summary["fields"]))
                 if gold["fields"].get(c) != summary["fields"].get(c)]
        diffs += [s for s in sorted(set(gold["segments"]) | set(summary["segments"])) if gold["segments"].get(s) != summary["segments"].get(s)]
        if json.loads(json.dumps(summary["entries"], default=str)) != gold["entries"]:
            diffs.append("data check entries")
        print(f"{key}: {'OK' if not diffs else 'DIFFERS: ' + ', '.join(diffs)}")
        failures += bool(diffs)
    if args.capture:
        manifest = {"matches": list(args.matches), "packages": reg.order(), "versions": {n: v for n, v in res.versions.items()},
                    "thresholds_version": THRESHOLDS_VERSION}
        (GOLDEN / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
