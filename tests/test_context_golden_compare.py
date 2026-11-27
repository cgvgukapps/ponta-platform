"""The context golden compares package by package (review SE-03): a version bump skips only its own package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from context_golden import compare, package_of


def _summary(**over):
    s = {"versions": {"p1": "0.6.0", "p2": "0.3.0", "p4": "0.6.0"}, "threshold_values": {"p1_x": 1.0, "p7_cost": 0.5},
         "fields": {"p1_possession": {"hash": "a"}, "p2_home_length_m": {"hash": "b"}, "p4_phase": {"hash": "c"}, "clock_s": {"hash": "d"}},
         "segments": {"p4_phases": {"rows": 3, "hash": "e"}}, "entries": [{"package": "p2", "metric": "team_shape", "state": "full"}],
         "semantic": {"semantic_events": {"hash": "f"}}}
    for k, v in over.items():
        s[k] = {**s[k], **v} if isinstance(v, dict) else v
    return s


def test_identical_summaries_match():
    assert compare(_summary(), _summary()) == ({}, {})


def test_a_version_bump_skips_only_its_package():
    new = _summary(versions={"p4": "0.7.0"}, fields={"p4_phase": {"hash": "C"}, "p2_home_length_m": {"hash": "B"}})
    diffs, skipped = compare(_summary(), new)
    assert set(skipped) == {"p4"} and diffs == {"p2": ["field p2_home_length_m"]}  # p2 changed without a bump: caught


def test_a_threshold_change_skips_its_package_and_grid_changes_are_caught():
    new = _summary(threshold_values={"p1_x": 2.0}, fields={"p1_possession": {"hash": "A"}, "clock_s": {"hash": "D"}})
    diffs, skipped = compare(_summary(), new)
    assert set(skipped) == {"p1"} and diffs == {"grid": ["field clock_s"]}


def test_entries_and_semantic_tables_are_compared():
    new = _summary(entries=[{"package": "p2", "metric": "team_shape", "state": "degraded"}], semantic={"semantic_events": {"hash": "F"}})
    diffs, _ = compare(_summary(), new)
    assert diffs == {"p2": ["Data check entry p2.team_shape"], "semantic": ["semantic_events"]}


def test_package_names():
    assert [package_of(n) for n in ("p4v_engaging_defenders", "p1_chains", "probe_x", "clock_s", "pf_min_matches_x")] == \
        ["p4v", "p1", "probe", "grid", "grid"]


def test_a_profiles_threshold_does_not_skip_the_grid():
    new = _summary(threshold_values={"pf_min_matches_x": 4.0}, fields={"clock_s": {"hash": "D"}})
    diffs, skipped = compare(_summary(), new)
    assert "grid" not in skipped and diffs == {"grid": ["field clock_s"]}
