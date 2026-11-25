"""
Plaiq-via-Matrix equivalence test (extraction spec, step 4).

The prototype ran Plaiq on wide CSVs written by its offline ``preprocess.py``. The
platform runs it on a match loaded and normalised by Matrix
(``matrix_engine.match``), through the ``MatchAnalysis`` façade. This test runs the
platform path on the two golden matches and compares every output with the golden
files captured from the frozen prototype (``tools/golden/README.md``).

Two levels:

* **Inputs.** The wide frames, velocities and events the engine receives must equal
  the prototype's processed inputs. If they do, the engine (deterministic, unchanged)
  must produce the same outputs.
* **Outputs.** Every golden output is recomputed and compared on natural keys, never
  row order.

Tolerance. The only input difference is float storage: positions agree to within
``INPUT_ATOL`` metres (the prototype's EPTS positions passed through a CSV). Outputs
are compared with ``OUTPUT_ATOL``/``OUTPUT_RTOL``; non-numeric columns must match
exactly. Any difference beyond tolerance is a finding to record as an
``engine_version`` decision, not something to fix by editing formulas.

**Recorded departure (25 November 2026, review backlog A).** The production SkillCorner loader drops records with no
period and maps SkillCorner's stoppages to SET PIECE rows; the prototype did neither. This test loads the SkillCorner
match with ``prototype_inputs=True`` so it still checks the engine maths on the prototype's inputs.

Opt-in: loading and scoring both full matches takes several minutes, so the test
runs only with ``PONTA_EQUIVALENCE=1`` (and the sample data and golden files present).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matrix_engine.ingestion.second_spectrum_io import HIGH_PASS
from matrix_engine.match import load_metrica_epts_match, load_skillcorner_match
from matrix_engine.sample_data import processed_dir, repo_root, sample_data_dir

pytestmark = pytest.mark.skipif(os.environ.get("PONTA_EQUIVALENCE") != "1", reason="set PONTA_EQUIVALENCE=1 to run the equivalence test")

#: Positions agree to this many metres (float storage through the prototype's CSV).
INPUT_ATOL = 1e-5
#: Output tolerances, justified by the input difference above.
OUTPUT_ATOL = 1e-6
OUTPUT_RTOL = 1e-6

#: Inferred intent locations are positions: compared to 0.1 mm.
INTENT_ATOL = 1e-4

#: Differences found on 15 November 2026 and characterised in EXTRACTION_REPORT.md.
#: The only input difference is float storage (positions agree to 2e-6 m); in two of
#: 39,589 POV rows it tips a near-tie or a 4th-decimal rounding. No decision-quality
#: band changes. These rows are excluded from the strict comparison, and nothing else.
KNOWN_POV_DIFFERENCES = {
    "metrica_3": {
        (2202, "Away", "23"): "delta_pov 0.0462 -> 0.0463 (one rounding unit)",
        (3373, "Home", "6"): "carry: best alternative changes near a tie; decision quality 1.248 -> 1.003, both Optimal",
    },
}
#: Carries whose alternatives table differs (12 of 36,792 rows): near-equal hexes, one rounding unit,
#: and the 3373 near-tie above. Characterised in EXTRACTION_REPORT.md.
KNOWN_ALT_EVENTS = {"metrica_3": {825, 2370, 2452, 3373}}

GOLDEN = repo_root() / "tests" / "golden" / "prototype"
MATCHES = {
    "metrica_3": {"prefix": "Game_3", "fps": 25.0},
    "skillcorner_1886347": {"prefix": "SC_1886347", "fps": 10.0},
}


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"missing {path}")


def _load(label: str):
    if label == "metrica_3":
        return load_metrica_epts_match(sample_data_dir(), 3)
    # The prototype's inputs: the production loader now drops records with no period and adds restart rows (review
    # DE-01, DE-11, 25 November 2026), a recorded departure; this test checks the engine on the inputs it was captured on.
    return load_skillcorner_match(os.path.join(sample_data_dir(), "skillcorner", "matches", "1886347"), 1886347, prototype_inputs=True)


def _profiles(label: str, home_j: list[str], away_j: list[str]):
    """The same per-player profiles the golden capture used (``tools/golden``)."""
    from plaiq_engine.roster.profiles import roster_profiles

    provider, mid = label.split("_")
    return roster_profiles(provider, mid, home_j, away_j)


@pytest.fixture(scope="module", params=list(MATCHES))
def analysis(request):
    label = request.param
    _require(Path(sample_data_dir()))
    _require(GOLDEN / label)
    from plaiq_engine.match_analysis import MatchAnalysis

    match = _load(label)
    home_j, away_j = list(match.slots["home"]), list(match.slots["away"])
    ph, pa = _profiles(label, home_j, away_j)
    return label, MatchAnalysis(match, profiles_home=ph, profiles_away=pa)


def compare_frames(
    golden: pd.DataFrame, mine: pd.DataFrame, key: list[str] | None = None, *,
    skip: pd.Series | None = None, col_atol: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compare two frames column by column; returns the max absolute difference per numeric
    column. ``skip`` (a boolean mask over the sorted golden rows) excludes known,
    characterised rows; ``col_atol`` overrides the tolerance for named columns.
    """
    assert list(golden.columns) == list(mine.columns), f"columns differ: {set(golden.columns) ^ set(mine.columns)}"
    assert len(golden) == len(mine), f"row counts differ: {len(golden)} vs {len(mine)}"
    if key:
        golden = golden.sort_values(key, kind="stable").reset_index(drop=True)
        mine = mine.sort_values(key, kind="stable").reset_index(drop=True)
    else:
        golden = golden.reset_index(drop=True)
        mine = mine.reset_index(drop=True)
    if skip is not None:
        keep = ~skip(golden) if callable(skip) else ~skip
        golden, mine = golden.loc[keep].reset_index(drop=True), mine.loc[keep].reset_index(drop=True)
    col_atol = col_atol or {}
    diffs: dict[str, float] = {}
    for col in golden.columns:
        a, b = golden[col], mine[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(a):
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            assert np.array_equal(np.isnan(av), np.isnan(bv)), f"{col}: missing values differ"
            ok = ~np.isnan(av)
            diffs[col] = float(np.max(np.abs(av[ok] - bv[ok]))) if ok.any() else 0.0
            atol = col_atol.get(col, OUTPUT_ATOL)
            assert np.allclose(av[ok], bv[ok], atol=atol, rtol=OUTPUT_RTOL), f"{col}: max abs diff {diffs[col]}"
        else:
            sa = a.astype(str).where(a.notna(), "<NA>")
            sb = b.astype(str).where(b.notna(), "<NA>")
            mism = int((sa != sb).sum())
            assert mism == 0, f"{col}: {mism} values differ"
    return diffs


def test_inputs_equal_the_prototypes_processed_files(analysis):
    label, ma = analysis
    prefix = MATCHES[label]["prefix"]
    for team, mine in (("home", ma.home), ("away", ma.away)):
        ref = pd.read_csv(f"{processed_dir()}/{prefix}_tracking_{team}.csv", index_col="Frame")
        assert ref.index.equals(mine.index), f"{team}: frame grid differs"
        for col in ref.columns:
            a, b = ref[col].to_numpy(dtype=float), mine[col].to_numpy(dtype=float)
            assert np.array_equal(np.isnan(a), np.isnan(b)), f"{team} {col}: missing values differ"
            ok = ~np.isnan(a)
            assert np.allclose(a[ok], b[ok], atol=INPUT_ATOL, rtol=0), f"{team} {col}: max diff {np.max(np.abs(a[ok] - b[ok]))}"
    ev = pd.read_csv(f"{processed_dir()}/{prefix}_events.csv")
    # ``High Pass`` is a recorded addition (da92464, the pass-height flag carried through Matrix); the prototype had none.
    assert list(ev.columns) == [c for c in ma.match.events.columns if c != HIGH_PASS]
    assert len(ev) == len(ma.match.events)


def test_event_delta_v(analysis):
    label, ma = analysis
    golden = pd.read_parquet(GOLDEN / label / "event_delta_v.parquet")
    mine = ma.events.drop(columns=[HIGH_PASS], errors="ignore")  # a recorded addition, see above
    # The golden events were read back from CSV, so ids are typed int/float there and
    # str here; the engine normalises both to the same jersey string.
    for col in ("From", "To"):
        if col in golden.columns:
            golden[col] = golden[col].map(lambda v: "" if pd.isna(v) else str(int(v)) if isinstance(v, float) and v == int(v) else str(v))
            mine[col] = mine[col].map(lambda v: "" if pd.isna(v) else str(v))
    compare_frames(golden, mine)


def test_frame_pressure(analysis):
    label, ma = analysis
    golden = pd.read_parquet(GOLDEN / label / "frame_pressure.parquet")
    fa = ma.frame_analytics
    mine = fa.loc[fa["Frame"].isin(golden["Frame"].unique())]
    compare_frames(golden, mine, key=["Frame", "team", "jersey"])


def test_movement_value(analysis):
    label, ma = analysis
    golden = pd.read_parquet(GOLDEN / label / "movement_value.parquet")
    key = [c for c in ("event_idx", "team", "jersey") if c in golden.columns]
    compare_frames(golden, ma.movement_value, key=key or None)


def test_event_pov_and_decision_quality(analysis):
    from plaiq_engine.semantic_builder import dq_band

    label, ma = analysis
    key = ["event_idx", "team", "jersey"]
    golden = pd.read_parquet(GOLDEN / label / "pov.parquet")
    mine = ma.event_pov()
    known = set(KNOWN_POV_DIFFERENCES.get(label, {}))

    def known_rows(df: pd.DataFrame) -> pd.Series:
        return pd.Series([(int(r.event_idx), str(r.team), str(r.jersey)) in known for r in df[key].itertuples()], index=df.index)

    compare_frames(golden, mine, key=key, skip=known_rows, col_atol={"intent_x": INTENT_ATOL, "intent_y": INTENT_ATOL})

    # No decision-quality band changes anywhere, the known rows included.
    g = golden.sort_values(key, kind="stable").reset_index(drop=True)
    m = mine.sort_values(key, kind="stable").reset_index(drop=True)
    for col in ("decision_quality", "decision_quality_failed"):
        gb = g[col].map(lambda v: dq_band(v) if pd.notna(v) else "none")
        mb = m[col].map(lambda v: dq_band(v) if pd.notna(v) else "none")
        assert (gb == mb).all(), f"{col}: {(gb != mb).sum()} band changes"

    golden_alt = pd.read_parquet(GOLDEN / label / "pov_alternatives.parquet")
    alt_key = ["event_idx", "hex_x", "hex_y"] if not golden_alt.duplicated(["event_idx", "hex_x", "hex_y"]).any() else None
    known_ev = KNOWN_ALT_EVENTS.get(label, set())
    compare_frames(golden_alt, ma.event_alternatives(), key=alt_key, skip=lambda df: df["event_idx"].isin(known_ev))


def test_frame_surfaces(analysis):
    label, ma = analysis
    g = np.load(GOLDEN / label / "frame_surfaces.npz")
    worst = {"control": 0.0, "value": 0.0}
    for i, frame in enumerate(g["frames"]):
        control, t_ball, t_h, t_a = ma.frame_control(int(frame))
        assert t_h is not None
        d = float(np.max(np.abs(np.asarray(control, dtype=np.float32) - g["control"][i])))
        worst["control"] = max(worst["control"], d)
        for name, arr in (("t_home", t_h), ("t_away", t_a), ("t_ball", t_ball)):
            assert np.allclose(np.asarray(arr, dtype=np.float32), g[name][i], atol=1e-4, rtol=1e-5, equal_nan=True), f"frame {frame} {name}"
        v = ma.value_surface(int(frame))
        if str(g["value_possession"][i]):
            assert v is not None
            dv = np.abs(np.asarray(v, dtype=np.float32) - g["value"][i])
            worst["value"] = max(worst["value"], float(np.nanmax(dv)))
            assert np.allclose(np.asarray(v, dtype=np.float32), g["value"][i], atol=1e-4, rtol=1e-5, equal_nan=True), f"frame {frame} value"
        else:
            assert v is None
    # float32 storage in the golden file: 1e-4 is well above float32 rounding at these magnitudes.
    assert worst["control"] <= 1e-4, worst
    # Ball-missing frames: the engine returns no travel times, as in the prototype.
    for frame in g["ball_missing_frames"]:
        _, _, t_h, _ = ma.frame_control(int(frame))
        assert t_h is None
