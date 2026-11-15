"""
Demo: one sample match end to end, inspectable without any UI (extraction spec, step 7).

    MatrixPipeline (normalise) -> parity -> MatchAnalysis -> semantic_builder

Prints the data check (Parity Report) and the first rows of both Parquet files.

    python tools/demo.py metrica          # Metrica Sample Game 3 (EPTS)
    python tools/demo.py skillcorner      # SkillCorner open data, match 1886347
    python tools/demo.py metrica --out /tmp/ponta-demo

Outputs go to a temporary directory outside the repository by default: they are
derived from open sample data whose redistribution terms are still unverified, so
they must never be committed. Scoring a full match takes a few minutes.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from matrix_engine.match import load_metrica_epts_match, load_skillcorner_match
from matrix_engine.parity import parity_for_match
from matrix_engine.pipeline import MatrixPipeline
from matrix_engine.sample_data import repo_root, sample_data_dir
from plaiq_engine.match_analysis import MatchAnalysis
from plaiq_engine.roster.profiles import roster_profiles
from plaiq_engine.semantic_builder import SemanticBuilder


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("match", choices=["metrica", "skillcorner"])
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: a new temporary directory)")
    args = ap.parse_args()

    out = (args.out or Path(tempfile.mkdtemp(prefix="ponta-demo-"))).resolve()
    if str(out).startswith(str(repo_root())):
        log("ERROR: refusing to write sample-derived output inside the repository")
        return 2

    t0 = time.time()
    if args.match == "metrica":
        match = load_metrica_epts_match(sample_data_dir(), 3)
    else:
        match = load_skillcorner_match(os.path.join(sample_data_dir(), "skillcorner", "matches", "1886347"), 1886347)
    log(f"loaded {match.provider} {match.match_id}: {match.n_frames} frames, {len(match.events)} events ({time.time() - t0:.0f}s)")

    normalised = MatrixPipeline().normalise(match)
    log(f"normalised: {len(normalised.match.tracking)} player observations; Normalise only, positions unchanged")

    parity = parity_for_match(match)
    log("\nDATA CHECK (Parity Report)")
    for mid, m in parity.metrics.items():
        measure = f"  [{m.measure}]" if m.measure else ""
        log(f"  {m.state.upper():9} {m.label}: {m.reason}{measure}")
    log(f"  summary {parity.summary}; coverage {parity.coverage}")

    hj, aj = list(match.slots["home"]), list(match.slots["away"])
    ph, pa = roster_profiles(match.provider, match.match_id, hj, aj)
    analysis = MatchAnalysis(match, profiles_home=ph, profiles_away=pa)
    log("\nscoring decisions (a few minutes) ...")
    t1 = time.time()
    builder = SemanticBuilder(analysis, parity)
    paths = builder.write(out)
    log(f"done in {time.time() - t1:.0f}s")

    sem = pd.read_parquet(paths["semantic_events"])
    tr = pd.read_parquet(paths["trajectories"])
    with pd.option_context("display.width", 200, "display.max_columns", 14):
        log(f"\nsemantic_events.parquet: {len(sem)} rows")
        log(sem[["event_id", "minute_display", "actor_player_ref", "action_type", "action_outcome", "dq_score", "dq_band",
                 "opposition_pressure", "pitch_zone", "not_scored"]].head(12).to_string(index=False))
        log(f"\n{sem['action_type'].value_counts().to_dict()}")
        log(f"\ntrajectories.parquet: {len(tr)} rows")
        log(tr.head(5).to_string(index=False))
    log(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
