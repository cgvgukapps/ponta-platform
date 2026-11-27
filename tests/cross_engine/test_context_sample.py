"""Context packages P0 to P4 on real sample data (Metrica Sample Game 1). Skips without the data (fails with PONTA_REQUIRE_SAMPLE_DATA)."""

import json
import os
import sys
from pathlib import Path

import pytest
from matrix_engine.match import load_metrica_csv_match
from matrix_engine.parity import parity_for_match
from matrix_engine.sample_data import repo_root, sample_data_dir

sys.path.insert(0, str(Path(repo_root()) / "tools"))
from context_golden import compare, registry, semantic_hashes, summarise


@pytest.fixture(scope="module")
def result():
    path = sample_data_dir()
    if not os.path.isdir(path):
        message = f"sample data not found at {path} (set PONTA_SAMPLE_DATA)"
        if os.environ.get("PONTA_REQUIRE_SAMPLE_DATA"):
            pytest.fail(message)
        pytest.skip(message)
    m = load_metrica_csv_match(path, 1)
    return m, registry().run(m)


def test_every_grid_point_is_labelled_with_a_source(result):
    _, res = result
    f = res.frames
    assert f["p1_possession_source"].isin(["observed", "inferred", "events", "unknown"]).all()
    assert (f["p1_possession"] != "unknown").mean() > 0.95
    assert f["p1_game_state_source"].eq("events").all()
    assert f["p1_score_home"].notna().all()


def test_data_check_entries_attach(result):
    m, res = result
    report = res.attach_to(parity_for_match(m))
    assert set(report.packages) == {"p1", "p2", "p3", "p4", "p4v", "p5", "p6", "p7", "p8", "probe"}
    assert {e.metric for e in report.packages["p5"]} == {"line_breaks", "relationships"}
    assert {e.metric for e in report.packages["p6"]} == {"movement"}
    assert {e.metric for e in report.packages["p7"]} == {"space", "adjusted_choice"}
    assert {e.metric for e in report.packages["p4v"]} == {"engagements", "fifa_view", "fifa_block"}
    assert {e.metric for e in report.packages["p8"]} == {"episodes"}
    assert {e.metric for e in report.packages["p1"]} == {"possession", "game_state", "possession_state"}
    assert {e.metric for e in report.packages["p2"]} == {"team_shape", "formation"}
    assert report.summary == parity_for_match(m).summary  # base metrics unchanged


def test_matches_the_golden_summary_package_by_package(result):
    """
    Review SE-03: compared package by package, so a version bump skips only that package (and one whose own thresholds
    changed); every other package must match. A missing or drifted summary fails under PONTA_REQUIRE_SAMPLE_DATA.
    Set PONTA_GOLDEN_ALL=1 to check Metrica's games 2 and 3 and SkillCorner's match too, and PONTA_GOLDEN_SEMANTIC=1 to
    compare the semantic events and episodes as well (both slow).
    """
    import hashlib

    m, res = result
    root = repo_root() / "tests" / "golden" / "context"
    manifest = json.loads((root / "manifest.json").read_text())
    strict = bool(os.environ.get("PONTA_REQUIRE_SAMPLE_DATA"))
    keys = list(manifest["matches"]) if os.environ.get("PONTA_GOLDEN_ALL") else ["metrica_1"]
    for key in keys:
        gold_path = root / f"{key}.json"
        if not gold_path.exists():
            (pytest.fail if strict else pytest.skip)(f"no golden summary for {key} (tools/context_golden.py --capture)")
        want = manifest.get("summary_sha256", {}).get(key)
        if want and hashlib.sha256(gold_path.read_bytes()).hexdigest() != want:
            (pytest.fail if strict else pytest.skip)(f"the local {key} summary is not the one the manifest records; recapture")
        if key == "metrica_1":
            mm, rr = m, res
        else:
            from context_golden import extras_for, load

            mm = load(key, Path(os.environ.get("PONTA_CONTEXT_CACHE", "/tmp/ponta-context/cache")))
            rr = registry().run(mm, extras=extras_for(key))
        semantic = semantic_hashes(mm, rr) if os.environ.get("PONTA_GOLDEN_SEMANTIC") else None  # about 7 minutes a match
        diffs, skipped = compare(json.loads(gold_path.read_text()),
                                 json.loads(json.dumps(summarise(rr, 0.0, {}, semantic), default=str)))
        assert not diffs, f"{key}: changed without a version bump: {diffs} (skipped as intended: {skipped})"


def test_stage_b_on_sample_data(result):
    _, res = result
    f = res.frames
    p2 = res.outputs["p2"]
    assert p2.measurements["orientation_consistent"]
    assert p2.measurements["shape_coverage_in_play"] > 0.95
    # Shape medians sit inside the published ranges for official matches (Rico-González et al. 2021).
    assert 25 <= f["p2_home_length_m"].median() <= 50
    assert 30 <= f["p2_home_width_m"].median() <= 50
    assert f["p2_home_formation_out"].notna().all()
    assert set(f["p4_phase"].dropna()) <= {"build_up", "create", "finish", "direct", "quick_break", "transition", "set_play", "chaotic"}
    assert (f["p4_phase"].notna() | f["p4_team"].isna()).mean() > 0.95
