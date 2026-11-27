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


#: Columns that only label versions: compared through ``versions``, never as values (a version bump is not a value change).
_LABELS = ("_version", "versions", "thresholds_version")


def _is_label(c: str) -> bool:
    return str(c).endswith(_LABELS)


def _frame_hash(df: pd.DataFrame) -> str:
    keep = df[[c for c in df.columns if not _is_label(c)]]
    return hashlib.sha256(pd.util.hash_pandas_object(keep.astype(object).where(keep.notna(), None).astype(str),
                                                     index=False).values.tobytes()).hexdigest()[:16]


def package_of(name: str) -> str:
    """The package a field, segment or threshold belongs to (``p4v_engaging_defenders`` is ``p4v``); grid columns are ``grid``."""
    head = str(name).split("_", 1)[0]
    return head if head.startswith("p") and head[1:2].isdigit() or head == "probe" else "grid"


def threshold_values() -> dict:
    from plaiq_engine.context.thresholds import THRESHOLDS

    return {name: t.value for name, t in THRESHOLDS.items()}


def summarise(res, seconds: float, sizes: dict, semantic: dict | None = None) -> dict:
    """
    A per-match summary, compared package by package (review SE-03): fields, segments and Data check entries, hashed
    without their version labels; the threshold values; and, when given, hashes of the semantic events and episodes.
    """
    f = res.frames
    return {
        "versions": res.versions,
        "thresholds_version": res.thresholds_version,
        "threshold_values": threshold_values(),
        "grid_points": int(len(f)),
        "fields": {c: {"hash": column_hash(f[c]), "nulls": int(f[c].isna().sum())} for c in f.columns if not _is_label(c)},
        "segments": {f"{p}_{n}": {"rows": int(len(s)), "hash": _frame_hash(s)} for p, o in res.outputs.items() for n, s in o.segments.items()},
        "entries": [{k: v for k, v in dataclasses.asdict(e).items() if not _is_label(k)} for e in res.entries],
        "semantic": semantic or {},
        "runtime_s": round(seconds, 1),
        "storage_bytes": sizes,
    }


def semantic_hashes(match, res) -> dict:
    """Hashes of the semantic events and episodes built on this context (the tables the AI reads)."""
    from plaiq_engine.match_analysis import MatchAnalysis
    from plaiq_engine.semantic_builder import SemanticBuilder
    from plaiq_engine.teams import TeamRegistry

    b = SemanticBuilder(MatchAnalysis(match), parity_for_match(match), teams=TeamRegistry.for_reference(), context=res)
    ev = b.semantic_events()
    st = b.semantic_states(ev)
    return {"semantic_events": {"rows": int(len(ev)), "hash": _frame_hash(ev.drop(columns=["engine_version"]))},
            "semantic_states": {"rows": int(len(st)), "hash": _frame_hash(st)}}


def compare(gold: dict, new: dict) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Differences by package, and the packages skipped with the reason. A package is skipped only when its own version or
    one of its own threshold values changed (an intended change); every other package must match.
    """
    skipped: dict[str, str] = {}
    for pkg in set(gold["versions"]) | set(new["versions"]):
        if gold["versions"].get(pkg) != new["versions"].get(pkg):
            skipped[pkg] = f"version {gold['versions'].get(pkg)} -> {new['versions'].get(pkg)}"
    gt, nt = gold.get("threshold_values", {}), new.get("threshold_values", {})
    for name in set(gt) | set(nt):
        if gt.get(name) != nt.get(name):
            pkg = package_of(name)
            pkg = pkg if pkg != "grid" else ("grid" if name.startswith("context_") else name.split("_", 1)[0])  # pf_: profiles
            skipped.setdefault(pkg, f"threshold {name} {gt.get(name)} -> {nt.get(name)}")
    diffs: dict[str, list[str]] = {}

    def note(name: str, what: str) -> None:
        pkg = package_of(name)
        if pkg not in skipped:
            diffs.setdefault(pkg, []).append(what)

    for c in sorted(set(gold["fields"]) | set(new["fields"])):
        if gold["fields"].get(c) != new["fields"].get(c):
            note(c, f"field {c}")
    for s_ in sorted(set(gold["segments"]) | set(new["segments"])):
        if gold["segments"].get(s_) != new["segments"].get(s_):
            note(s_, f"segment {s_}")
    ge = {(e["package"], e["metric"]): e for e in gold["entries"]}
    ne = {(e["package"], e["metric"]): e for e in json.loads(json.dumps(new["entries"], default=str))}
    for k in sorted(set(ge) | set(ne)):
        if ge.get(k) != ne.get(k):
            note(k[0] + "_entry", f"Data check entry {k[0]}.{k[1]}")
    if gold.get("semantic") and new.get("semantic") and not skipped:
        for t in sorted(set(gold["semantic"]) | set(new["semantic"])):
            if gold["semantic"].get(t) != new["semantic"].get(t):
                diffs.setdefault("semantic", []).append(t)
    return diffs, skipped


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
        summary = summarise(res, seconds, sizes, semantic_hashes(m, res))
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
        diffs, skipped = compare(gold, json.loads(json.dumps(summary, default=str)))
        flat = [d for pkg in sorted(diffs) for d in diffs[pkg]]
        note = f" (skipped, intended changes: {', '.join(f'{k}: {v}' for k, v in sorted(skipped.items()))})" if skipped else ""
        print(f"{key}: {'OK' if not flat else 'DIFFERS: ' + ', '.join(flat)}{note}")
        failures += bool(flat)
    if args.capture:
        # The tracked manifest records each summary's hash, so a local summary that drifted from the captured one is
        # visible (review SE-03): the summaries themselves stay out of the repository (derived from sample data).
        hashes = {k: hashlib.sha256((GOLDEN / f"{k}.json").read_bytes()).hexdigest() for k in MATCHES if (GOLDEN / f"{k}.json").exists()}
        manifest = {"matches": list(MATCHES), "packages": reg.order(), "versions": {n: v for n, v in res.versions.items()},
                    "thresholds_version": THRESHOLDS_VERSION, "summary_sha256": hashes}
        (GOLDEN / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
