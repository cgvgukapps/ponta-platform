"""Context packages P0 to P4 on real sample data (Metrica Sample Game 1). Skips without the data."""

import json
import os
import sys
from pathlib import Path

import pytest
from matrix_engine.match import load_metrica_csv_match
from matrix_engine.parity import parity_for_match
from matrix_engine.sample_data import repo_root, sample_data_dir

sys.path.insert(0, str(Path(repo_root()) / "tools"))
from context_golden import registry, summarise


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
    assert set(report.packages) == {"p1", "p2", "p3", "p4", "p5", "probe"}
    assert {e.metric for e in report.packages["p5"]} == {"line_breaks"}
    assert {e.metric for e in report.packages["p1"]} == {"possession", "game_state", "possession_state"}
    assert {e.metric for e in report.packages["p2"]} == {"team_shape", "formation"}
    assert report.summary == parity_for_match(m).summary  # base metrics unchanged


def test_matches_the_golden_summary_when_captured(result):
    _, res = result
    gold_path = repo_root() / "tests" / "golden" / "context" / "metrica_1.json"
    if not gold_path.exists():
        pytest.skip("no golden summary captured (tools/context_golden.py --capture)")
    gold = json.loads(gold_path.read_text())
    if gold["versions"] != res.versions or gold["thresholds_version"] != res.thresholds_version:
        pytest.skip("golden summary is for other package or threshold versions; recapture")
    assert summarise(res, 0.0, {})["fields"] == gold["fields"]


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
