"""P1: possession labels and sources, dead ball, game state, chains, the upload input and the Data check."""

import numpy as np
import pandas as pd
import pytest
from context_fixture import FPS, event, tiny_match
from plaiq_engine.context.possession import (
    PossessionPackage,
    build_chains,
    goals_and_cards,
    label_possession,
    load_game_state_input,
)
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.thresholds import values

CFG = values()


def _arrays(n, rate=5.0):
    grid = pd.DataFrame({"frame_ts": np.round(np.arange(n) / rate, 3), "period": 1, "minute": 1, "native_frame": np.arange(n) * 5 + 1})
    ball = pd.DataFrame({"ball_detected": np.ones(n, bool), "ball_speed": np.full(n, 2.0)})
    near = pd.DataFrame({"near_home": np.full(n, 9.0), "near_away": np.full(n, 9.0)})
    return grid, ball, near, np.full(n, None, dtype=object)


def test_observed_control_and_contested():
    grid, ball, near, ev = _arrays(6)
    near.loc[0:1, "near_home"] = 0.5            # home controls
    near.loc[2, ["near_home", "near_away"]] = 0.5  # both close: contested
    lab, src, _ = label_possession(grid, ball, near, ev, None, CFG)
    assert list(lab[:3]) == ["home", "home", "contested"]
    assert list(lab[3:]) == ["home"] * 3  # ball still detected, nobody controls: the last team keeps it
    assert set(src) == {"observed"}


def test_a_brush_past_an_opponent_does_not_switch_possession():
    grid, ball, near, ev = _arrays(12)
    near.loc[0:3, "near_home"] = 0.5
    near.loc[4, "near_away"] = 0.5           # 0.2 s of away control: shorter than p1_min_switch_s
    near.loc[5:6, "near_home"] = 0.5
    near.loc[7:11, "near_away"] = 0.5        # 1 s of away control: a real switch
    lab, _, _ = label_possession(grid, ball, near, ev, None, CFG)
    assert list(lab[:7]) == ["home"] * 7
    assert list(lab[7:]) == ["away"] * 5     # relabelled from when away took control


def test_a_touch_by_the_other_team_makes_the_ball_loose_when_switched_on():
    """The founder's rule: a deflection or clearance by the other team leaves possession open until someone controls."""
    grid, ball, near, ev = _arrays(12)
    near.loc[0:3, "near_home"] = 0.5
    near.loc[4, "near_away"] = 0.5          # away touches it for 0.2 s: a block, not a switch
    near.loc[8:11, "near_home"] = 0.5       # home controls again
    on = values({"p1_loose_on_opponent_touch": 1.0})
    lab, _, _ = label_possession(grid, ball, near, ev, None, on)
    assert list(lab[:4]) == ["home"] * 4 and list(lab[4:8]) == ["loose"] * 4 and list(lab[8:]) == ["home"] * 4
    off, _, _ = label_possession(grid, ball, near, ev, None, CFG)
    assert set(off) == {"home"}             # off (the agreed behaviour): the team that had the ball keeps it


def test_fast_ball_is_in_flight_not_controlled():
    grid, ball, near, ev = _arrays(3)
    near.loc[:, "near_away"] = 0.5
    ball.loc[:, "ball_speed"] = CFG["p1_max_control_ball_speed_mps"] + 5
    lab, _, control = label_possession(grid, ball, near, ev, None, CFG)
    assert all(c is None for c in control) and set(lab) == {"unknown"}


def test_gaps_are_carried_then_handed_to_events_then_unknown():
    grid, ball, near, ev = _arrays(40)
    near.loc[0:4, "near_home"] = 0.5
    ball.loc[5:, "ball_detected"] = False
    ev[25:30] = "away"
    lab, src, _ = label_possession(grid, ball, near, ev, None, CFG)
    gap = CFG["p1_max_carry_gap_s"]
    carried = (grid["frame_ts"] - grid["frame_ts"][4] <= gap) & (grid.index >= 5)
    assert set(src[carried]) == {"inferred"} and set(lab[carried]) == {"home"}
    assert list(src[25:30]) == ["events"] * 5 and list(lab[25:30]) == ["away"] * 5
    assert set(lab[30:]) == {"unknown"} and set(src[30:]) == {"unknown"}


def test_dead_ball_wins_over_everything_and_resets():
    grid, ball, near, ev = _arrays(6)
    near.loc[:, "near_home"] = 0.5
    dead = np.array([False, False, True, True, False, False])
    lab, src, _ = label_possession(grid, ball, near, ev, dead, CFG)
    assert list(lab) == ["home", "home", "dead_ball", "dead_ball", "home", "home"]
    assert list(src[2:4]) == ["events", "events"]


def test_goals_tokens_kick_off_confirmation_and_red_cards():
    evs = [
        event("SHOT", "Home", 100, "ON TARGET-GOAL"), event("SET PIECE", "Away", 400, "KICK OFF"),
        event("SHOT", "Away", 600, "WOODWORK-GOAL"),            # a GOAL token: counted, but no kick-off follows
        event("SHOT", "Home", 800, "ON TARGET, SAVED"),
        event("SET PIECE", "Home", 900, "GOAL KICK"),           # never a goal
        event("CARD", "Away", 950, "RED"), event("CARD", "Home", 960, "YELLOW"),
    ]
    m = tiny_match(evs, seconds=50.0)
    goals, reds = goals_and_cards(m, CFG)
    assert [(g["team"], g["confirmed"]) for g in goals] == [("home", True), ("away", False)]
    assert [r["team"] for r in reds] == ["away"]


def test_game_state_from_events_upload_final_only_and_unknown():
    m = tiny_match([event("SHOT", "Home", 100, "ON TARGET-GOAL"), event("SET PIECE", "Away", 200, "KICK OFF")], seconds=10.0)
    reg = Registry([PossessionPackage()])
    f = reg.run(m).frames
    assert f.loc[f["native_frame"] < 100, "p1_score_home"].eq(0).all() and f.loc[f["native_frame"] >= 100, "p1_score_home"].eq(1).all()
    assert f["p1_game_state_source"].eq("events").all()

    sc = tiny_match([event("PASS", "Home", 10)], seconds=10.0, provider="skillcorner")
    up = load_game_state_input({"final_score": {"home": 1, "away": 0}, "goals": [{"team": "home", "period": 1, "clock_s": 4.0}]})
    f = reg.run(sc, extras={"game_state": up}).frames
    assert f["p1_game_state_source"].eq("upload").all()
    assert f.loc[f["clock_s"] < 4.0, "p1_score_home"].eq(0).all() and f.loc[f["clock_s"] >= 4.0, "p1_score_home"].eq(1).all()

    res = reg.run(sc, extras={"game_state": load_game_state_input({"final_score": {"home": 1, "away": 0}})})
    assert res.frames["p1_game_state_source"].eq("final_only").all() and res.frames["p1_score_home"].isna().all()
    assert {e.metric: e.state for e in res.entries}["game_state"] == "degraded"

    res = reg.run(sc)
    assert {e.metric: e.state for e in res.entries}["game_state"] == "blocked"


def test_upload_input_is_validated():
    with pytest.raises(ValueError, match="add up"):
        load_game_state_input({"final_score": {"home": 2, "away": 0}, "goals": [{"team": "home", "period": 1, "clock_s": 1.0}]})
    with pytest.raises(ValueError, match="team"):
        load_game_state_input({"goals": [{"team": "Home FC", "period": 1, "clock_s": 1.0}]})
    with pytest.raises(ValueError, match="unknown fields"):
        load_game_state_input({"score": 1})
    with pytest.raises(ValueError, match="period"):
        load_game_state_input({"red_cards": [{"team": "away", "period": 7, "clock_s": 1.0}]})


def test_dead_ball_and_set_piece_from_logged_events_only():
    evs = [event("PASS", "Home", 1, end_frame=30), event("BALL OUT", "Home", 60), event("SET PIECE", "Away", 150, "THROW IN"),
           event("PASS", "Away", 150, end_frame=170)]
    f = Registry([PossessionPackage()]).run(tiny_match(evs, seconds=10.0, ball=lambda t: (0.5, 0.0))).frames
    dead = f.loc[f["p1_dead_ball"].fillna(False)]
    assert len(dead) and dead["native_frame"].between(60, 149).all() and dead["p1_set_piece"].eq("throw_in").all()
    sc = Registry([PossessionPackage()]).run(tiny_match(evs, seconds=10.0, provider="skillcorner")).frames
    assert sc["p1_dead_ball"].isna().all()  # derived feed: dead ball unknown, not false


def test_chains_start_and_end_types():
    m = tiny_match([event("RECOVERY", "Away", 1 + int(2.0 * FPS)), event("SHOT", "Away", 1 + int(3.6 * FPS))], seconds=6.0)
    grid = pd.DataFrame({"frame_ts": np.round(np.arange(30) / 5.0, 3), "period": 1, "minute": 1})
    lab = np.array(["home"] * 10 + ["away"] * 10 + ["dead_ball"] * 5 + ["home"] * 5, dtype=object)
    src = np.full(30, "observed", dtype=object)
    setp = np.array([None] * 20 + ["corner_kick"] * 5 + [None] * 5, dtype=object)
    chain, segs = build_chains(grid, lab, src, setp, m, CFG, [])
    assert list(segs["team"]) == ["home", "away", "home"]
    assert list(segs["start_type"]) == ["period_start", "recovery", "set_piece:corner_kick"]
    assert list(segs["end_type"]) == ["lost", "shot", "period_end"]
    assert (chain[20:25] == -1).all()


def test_a_set_piece_starts_only_the_taking_teams_chain():
    """A corner headed clear: the defending team's chain starts 'won', not with the corner."""
    m = tiny_match([], seconds=6.0)
    grid = pd.DataFrame({"frame_ts": np.round(np.arange(30) / 5.0, 3), "period": 1, "minute": 1})
    lab = np.array(["home"] * 10 + ["dead_ball"] * 5 + ["away"] * 5 + ["home"] * 10, dtype=object)
    src = np.full(30, "observed", dtype=object)
    setp = np.array([None] * 10 + ["corner_kick"] * 5 + [None] * 15, dtype=object)
    setp_team = np.array([None] * 10 + ["home"] * 5 + [None] * 15, dtype=object)
    _, segs = build_chains(grid, lab, src, setp, m, CFG, [], setp_team)
    assert list(segs["team"]) == ["home", "away", "home"]
    assert list(segs["start_type"]) == ["period_start", "won", "won"]
    _, same = build_chains(grid, lab, src, setp, m, CFG, [], np.where(setp_team == "home", "away", setp_team))
    assert same["start_type"].iloc[1] == "set_piece:corner_kick"


def test_package_output_has_every_field_and_no_names():
    m = tiny_match([event("PASS", "Home", 1)], seconds=10.0, ball=lambda t: (0.5, 0.0) if t < 6 else None)
    res = Registry([PossessionPackage()]).run(m)
    f = res.frames
    assert f["p1_possession"].isin(["home", "away", "contested", "dead_ball", "unknown"]).all()
    assert f["p1_possession_source"].isin(["observed", "inferred", "events", "unknown"]).all()
    assert f["p1_version"].eq(PossessionPackage.version).all()
    assert not f.astype(str).apply(lambda s: s.str.contains("Player ")).any().any()
    entries = {e.metric: e for e in res.entries}
    assert entries["possession"].coverage.endswith("of match time labelled")
    pd.testing.assert_frame_equal(f, Registry([PossessionPackage()]).run(m).frames)  # deterministic
