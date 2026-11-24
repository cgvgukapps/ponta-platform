"""The PFF reference reader's event translation and frame sampling (synthetic rows; no PFF data needed)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pff import EVENT_COLUMNS, Reference, subsample, to_p1_events

FPS = 30.0


def _ev(frame, gev="OTB", pev=None, team="home", **kw):
    base = dict.fromkeys(EVENT_COLUMNS)
    base |= {"frame": frame, "period": 1, "game_event_id": frame, "possession_event_id": frame, "gev": gev, "pev": pev,
             "team": team, "setpiece": "O" if gev == "OTB" else None, "x": float(frame) / 100, "y": 0.0}
    return base | kw


def test_pff_events_become_p1_vocabulary():
    ev = pd.DataFrame([
        _ev(0, gev="FIRSTKICKOFF", pev="PA", setpiece="K", player="home_9", target="home_6", pass_outcome="C"),
        _ev(60, pev="PA", player="home_6", target="home_10", pass_outcome="D"),           # intercepted
        _ev(90, pev="IT", team="away", player="away_4"),                                  # the interceptor's touch
        _ev(120, pev="CH", team="away", challenge_type="T", challenger="home_8", challenger_team="home", winner_team="away"),
        _ev(150, pev="SH", team="away", player="away_11", shot_outcome="G"),
        _ev(160, gev="OUT", team=None, out_type="A"),
        _ev(300, gev="OTB", pev="PA", setpiece="K", player="home_9", target="home_5", pass_outcome="C"),
        _ev(330, gev="OUT", team=None, out_type="T"),
        _ev(400, pev="PA", team="away", setpiece="T", player="away_2", target="away_7", pass_outcome="C"),
        _ev(430, pev="IT", team="away", player="away_7", foul_type="I", culprit_team="home", card="Y"),
    ])
    out = to_p1_events(ev, FPS)
    rows = list(zip(out["Type"], out["Subtype"].fillna(""), out["Team"], out["Start Frame"], strict=True))
    assert rows == [
        ("SET PIECE", "KICK OFF", "Home", 0),
        ("PASS", "", "Home", 0),
        ("BALL LOST", "", "Home", 60),
        ("RECOVERY", "INTERCEPTION", "Away", 90),   # a change of team in open play, after an intercepted pass
        ("CHALLENGE", "TACKLE-LOST", "Home", 120),  # the challenger's side, lost to the carrier
        ("SHOT", "ON TARGET-GOAL", "Away", 150),
        ("SET PIECE", "KICK OFF", "Home", 300),
        ("PASS", "", "Home", 300),
        ("BALL OUT", "", "Home", 330),              # PFF's out event has no team: the throw-in's opponents put it out
        ("SET PIECE", "THROW IN", "Away", 400),
        ("PASS", "", "Away", 400),
        ("FAULT RECEIVED", "", "Away", 430),        # the culprit is home: the foul is received by away
        ("CARD", "YELLOW", "Home", 430),
    ]
    passes = out.loc[out["Type"].isin(["PASS", "BALL LOST"])]
    assert list(passes["End Frame"]) == [60, 90, 330, 430]   # a pass ends where the next event starts
    assert passes["End X"].notna().all()
    assert list(out.loc[out["Type"] == "PASS", "From"]) == ["9", "9", "2"] and list(out.loc[out["Type"] == "PASS", "To"]) == ["6", "5", "7"]
    assert not [c for c in out.columns if "name" in c.lower()]


def test_an_own_goal_is_credited_to_the_scorers():
    ev = pd.DataFrame([_ev(0, pev="CL", team="home", player="home_3", pass_outcome=None), _ev(20, gev="OUT", team=None, out_type="A")])
    out = to_p1_events(ev, FPS)
    shots = out.loc[out["Type"] == "SHOT"]
    assert list(zip(shots["Subtype"], shots["Team"], strict=True)) == [("OWN GOAL-GOAL", "Away")]


def test_a_pass_end_is_left_unknown_when_the_next_event_is_far_or_in_another_period():
    ev = pd.DataFrame([_ev(0, pev="PA", player="home_6", target="home_10", pass_outcome="C"),
                       _ev(int(20 * FPS), pev="PA", player="home_10", target="home_6", pass_outcome="C"),
                       _ev(int(20 * FPS) + 30, pev="IT", period=2)])
    out = to_p1_events(ev, FPS)
    assert out.loc[out["Type"] == "PASS", "End X"].isna().all()


def test_subsample_renumbers_frames_so_gaps_keep_their_seconds():
    from matrix_engine.match import NormalisedMatch
    from matrix_engine.pitch_config import PitchConfig
    from matrix_engine.schema import coerce_schema

    frames = pd.DataFrame({"frame": np.arange(300, 330), "period": 1, "time_s": np.arange(30) / FPS})
    tracking = coerce_schema(pd.DataFrame({"timestamp": frames["time_s"], "frame": frames["frame"], "player_id": "home_1", "team": "home",
                                           "x": 0.0, "y": 0.0, "source": "pff", "sample_rate_hz": FPS}))
    m = NormalisedMatch(provider="pff", match_id="t", fps=FPS, pitch=PitchConfig(fps=FPS), frames=frames, tracking=tracking,
                        ball=frames[["frame"]].assign(x=0.0, y=0.0), events=pd.DataFrame())
    ref = Reference(events=pd.DataFrame([_ev(301), _ev(329)]), visible_share=1.0, ball_share=1.0, orientation_agrees=True)
    m3, ref3 = subsample(m, ref, 3)
    assert m3.fps == FPS / 3 and m3.pitch.fps == FPS / 3
    assert list(m3.frames["frame"]) == list(range(100, 110))
    assert set(m3.tracking["frame"]) == set(range(100, 110)) and (m3.tracking["sample_rate_hz"] == FPS / 3).all()
    assert list(ref3.events["frame"]) == [100, 110]   # to the nearest kept frame
    # Ten kept frames at a third of the rate span the same second as thirty native ones.
    assert (m3.frames["frame"].iloc[-1] - m3.frames["frame"].iloc[0]) / m3.fps == np.round((frames["frame"].iloc[-3] - 300) / FPS, 6)


def test_a_period_whose_players_disagree_with_the_ball_is_dropped():
    from matrix_engine.match import NormalisedMatch
    from matrix_engine.pitch_config import PitchConfig
    from pff import _keep_consistent

    n = 30
    f = np.arange(2 * n)
    period = np.where(f < n, 1, 2)
    frames = pd.DataFrame({"frame": f, "period": period, "time_s": f / FPS})
    # The ball is at x = 10 throughout. Home's player stands on it in period 1 and away's in period 2.
    home_x, away_x = np.where(period == 1, 10.0, -30.0), np.where(period == 1, -30.0, 10.0)
    tracking = pd.DataFrame({"frame": np.r_[f, f], "player_id": ["home_9"] * len(f) + ["away_4"] * len(f),
                             "team": ["home"] * len(f) + ["away"] * len(f), "x": np.r_[home_x, away_x], "y": 0.0})
    m = NormalisedMatch(provider="pff", match_id="t", fps=FPS, pitch=PitchConfig(fps=FPS), frames=frames, tracking=tracking,
                        ball=pd.DataFrame({"frame": f, "x": 10.0, "y": 0.0}), events=pd.DataFrame())
    ev = pd.DataFrame([_ev(int(i), pev="PA", period=int(period[i]), player="home_9", target="home_9", pass_outcome="C") for i in f])
    ref = Reference(events=ev, visible_share=1.0, ball_share=1.0, orientation_agrees=True)
    m2, ref2 = _keep_consistent(m, ref)
    assert ref2.dropped_periods == (2,) and ref2.consistency == {1: (1.0, n), 2: (0.0, n)}
    assert set(m2.frames["period"]) == {1} and m2.tracking["frame"].max() < n and m2.ball["frame"].max() < n
    assert set(ref2.events["period"]) == {1}
