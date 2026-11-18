"""Stage B: P2 team structure, P3 relative zones, P4 phase of play (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest
from context_fixture import eleven_a_side
from plaiq_engine.context.phase import (
    PhasePackage,
    _smooth,
    block_of,
    corner_at_start,
    deepest_three,
)
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.structure import (
    TEMPLATES,
    StructurePackage,
    _persist,
    assign_lines,
    attack_signs,
    fit_formation,
)
from plaiq_engine.context.thresholds import THRESHOLDS, values
from plaiq_engine.context.zones import ZonesPackage, band_of, lane_of

CFG = values()


def _run(match, packages=None):
    return Registry(packages or [PossessionPackage(), StructurePackage(), ZonesPackage(), PhasePackage()]).run(match)


@pytest.fixture(scope="module")
def standard():
    return _run(eleven_a_side())


# -- building blocks -------------------------------------------------------------------

def test_lines_are_the_exact_three_means_split():
    depths = np.array([30, 31, 29, 32, 45, 47, 46, 48, 63, 64], float)
    lab = assign_lines(depths)
    assert list(lab) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert list(assign_lines(np.array([1.0, 2.0]))) == [0, 0]


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_every_template_recognises_itself(name):
    xy = np.array([(d, w) for _, d, w in TEMPLATES[name]], float)
    fit = fit_formation(xy)
    assert fit["best"] == name
    assert fit["roles"] == [r for r, _, _ in TEMPLATES[name]]
    assert fit["margin"] > CFG["p2_formation_min_margin"]


def test_a_formation_change_must_persist():
    labels = ["4-4-2", "4-4-2", "4-3-3", "4-4-2", "4-3-3", "4-3-3", "4-3-3"]
    out = _persist(labels, [120.0] * 7, 300.0)
    assert out[:4] == ["4-4-2"] * 4  # one 4-3-3 window is not a change
    assert out[4:] == ["4-3-3"] * 3  # three in a row (360 s) is
    assert _persist(["unclear", "4-4-2"], [120.0, 120.0], 300.0) == ["4-4-2", "4-4-2"]


def test_lanes_follow_the_box_edges():
    lanes = lane_of(np.array([30.0, 15.0, 0.0, -15.0, -30.0, np.nan]))
    assert list(lanes) == ["left_wing", "left_half_space", "centre", "right_half_space", "right_wing", None]


def test_bands_against_the_defending_lines():
    d = np.array([10.0, 22.0, 35.0, 55.0, 80.0])
    one = np.ones(5)
    assert list(band_of(d, 20 * one, 25 * one, 45 * one, 60 * one)) == ["behind_defence", "in_defensive_line", "between_lines",
                                                                           "between_mid_att", "in_front"]


def test_blocks_from_the_deepest_three():
    D = np.array([[5.0, 20.0, 22.0, 24.0, 50.0, np.nan], [5.0, 40.0, 41.0, 42.0, 60.0, 70.0]])
    lines = np.array([[3, 0, 0, 0, 1, -1], [3, 0, 0, 0, 1, 2]], dtype=np.int8)
    d3 = deepest_three(D, lines)
    assert d3 == pytest.approx([22.0, 41.0])
    assert list(block_of(np.array([10.0, 30.0, 50.0, np.nan]), 26.0, 47.0)) == ["low", "medium", "high", None]


def test_short_phases_merge_into_the_one_before():
    lab = np.array(["create"] * 6 + ["direct"] * 2 + ["create"] * 6, dtype=object)
    assert set(_smooth(lab, 5)) == {"create"}
    assert list(_smooth(np.array(["direct", "create", "create", "create"], dtype=object), 2)) == ["create"] * 4


def test_corner_rule():
    bx, by = np.array([np.nan, 51.0, 48.0]), np.array([np.nan, 32.0, 30.0])
    assert corner_at_start(bx, by, np.arange(3), 105.0, 68.0, 4.0)
    assert not corner_at_start(bx, by, np.arange(1), 105.0, 68.0, 4.0)


REVIEW_REFINEMENTS = ("p4_finish_min_depth_share", "p4_build_up_max_depth_share", "p4_break_end_depth_share")
CORNER_SETUP = ("p4_corner_setup_crowd_min", "p4_corner_setup_radius_m", "p4_corner_setup_hold_s")


def test_stage_b_thresholds_are_agreed_with_a_reason_and_source():
    stage_b = [t for name, t in THRESHOLDS.items() if name.startswith(("p2_", "p3_", "p4_")) and name not in REVIEW_REFINEMENTS + CORNER_SETUP]
    assert len(stage_b) == 22
    for t in stage_b:
        assert t.reason and t.status == "agreed"
        assert "agreed by the founder, 17 November 2026" in t.source
    for name in REVIEW_REFINEMENTS:
        t = THRESHOLDS[name]
        assert t.status == "measured" and "replay review" in t.source  # the founder has not agreed them yet
    for name in CORNER_SETUP:
        assert THRESHOLDS[name].status in ("proposed", "measured")  # the founder has not agreed them yet
    for name in ("p4_low_block_max_m", "p4_high_block_min_m", "p4_break_min_gain_m"):
        assert "CONTEXT_STRUCTURE_REPORT.md" in THRESHOLDS[name].source


# -- the packages on a synthetic match ---------------------------------------------------

def test_orientation_is_read_from_the_data():
    flipped = eleven_a_side(home_attacks_positive=False, seconds=60.0)
    from plaiq_engine.context.grid import build_grid
    s = attack_signs(flipped, build_grid(flipped, 5.0)).set_index("team")["attack_sign"]
    assert s["home"] == -1 and s["away"] == 1


def test_structure_finds_the_formations_and_roles(standard):
    f = standard.frames
    assert (f["p2_home_formation_in"] == "4-4-2").all()
    assert (f["p2_away_formation_out"] == "4-3-3").all()
    roles = standard.outputs["p2"].segments["roles"]
    home = roles.loc[(roles["team"] == "home") & (roles["state"] == "in")].drop_duplicates("player_id").set_index("player_id")["role"]
    assert home["home_2"] == "LB" and home["home_10"] == "LCF"
    gk = standard.outputs["p2"].segments["goalkeepers"]
    assert set(gk["player_id"]) == {"home_1", "away_1"}


def test_shape_measures_match_the_layout(standard):
    f = standard.frames
    assert f["p2_home_outfield_n"].eq(10).all()
    assert f["p2_home_length_m"].median() == pytest.approx(34.0, abs=2.0)  # 30 to 64 m
    assert f["p2_home_width_m"].median() == pytest.approx(48.0, abs=2.5)
    assert f["p2_home_def_mid_gap_m"].median() == pytest.approx(16.5, abs=1.5)


def test_the_same_shapes_when_the_pitch_is_mirrored(standard):
    mirrored = _run(eleven_a_side(home_attacks_positive=False))
    for c in ("p2_home_length_m", "p2_away_hull_m2", "p2_home_formation_in", "p3_ball_band", "p3_ball_lane", "p4_phase"):
        a, b = standard.frames[c], mirrored.frames[c]
        if a.dtype.kind == "f":
            np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), atol=1e-6)
        else:
            assert (a.astype(object).fillna("-") == b.astype(object).fillna("-")).all(), c


def test_zones_for_a_midfielder_on_the_ball(standard):
    f = standard.frames
    held = f["p3_attacking_team"] == "home"
    assert held.mean() > 0.9
    # Home_7 is the left-centre midfielder (depth 46, width +8): centre lane, middle third,
    # between the away team's midfield and attacking lines.
    assert set(f.loc[held, "p3_ball_lane"]) == {"centre"}
    assert set(f.loc[held, "p3_ball_third"]) == {"middle"}
    assert set(f.loc[held, "p3_ball_band"]) == {"between_mid_att"}
    assert not f.loc[held, "p3_ball_in_pocket"].any()


def test_established_play_against_a_medium_block(standard):
    f = standard.frames
    held = f["p4_team"] == "home"
    assert set(f.loc[held, "p4_block"]) == {"medium"}       # away back four about 31 m from goal
    assert set(f.loc[held, "p4_phase"]) == {"create"}
    assert set(f.loc[held, "p4_out_phase"]) == {"medium_block"}
    phases = standard.outputs["p4"].segments["phases"]
    assert len(phases) >= 1 and (phases["phase"] == "create").all()


def test_package_entries_and_versions(standard):
    metrics = {(e.package, e.metric): e.state for e in standard.entries}
    assert metrics[("p2", "team_shape")] == "full"
    assert ("p3", "relative_zones") in metrics and ("p4", "phase_of_play") in metrics
    assert {"p2_version", "p3_version", "p4_version"} <= set(standard.frames.columns)


def test_no_coordinates_in_stage_b_fields(standard):
    names = [c for c in standard.frames.columns if c.startswith(("p2_", "p3_", "p4_"))]
    assert not [c for c in names if c.endswith(("_x", "_y"))]


def test_deterministic():
    a, b = _run(eleven_a_side(seconds=150.0)), _run(eleven_a_side(seconds=150.0))
    pd.testing.assert_frame_equal(a.frames, b.frames)
