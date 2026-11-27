"""
synthetic_golden.py — capture or check the synthetic engine golden (review finding SE-01).

    python tools/synthetic_golden.py --check
    python tools/synthetic_golden.py --capture

The golden is built from the licence-free synthetic match in ``tests/support/synthetic_match.py``
(fictional players, no sample data) and tracked in ``tests/golden/synthetic/engine.json``. CI checks
it through ``tests/cross_engine/test_synthetic_golden.py``. Recapture only when a change to an engine
value is intended, and record that in the commit message.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "support"))

from engine_snapshot import compare, snapshot
from plaiq_engine.match_analysis import MatchAnalysis
from synthetic_match import synthetic_match

GOLDEN = ROOT / "tests" / "golden" / "synthetic" / "engine.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args()
    snap = snapshot(MatchAnalysis(synthetic_match()))
    if args.capture:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(snap, indent=0, sort_keys=True) + "\n")
        print(f"captured {GOLDEN.relative_to(ROOT)} ({GOLDEN.stat().st_size / 1e3:.0f} kB)")
        return 0
    diffs = compare(json.loads(GOLDEN.read_text()), json.loads(json.dumps(snap)))
    print("OK" if not diffs else "DIFFERS:\n  " + "\n  ".join(diffs[:40]))
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
