"""The IDSSE reference reader's event translation and kick timing (synthetic rows; no data needed)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from context_idsse import kick_frame
from idsse import to_p1_events


def _ev(**kw):
    base = {"frame": 0, "period": 1, "kind": "Play", "team": "home", "x": 0.0, "y": 0.0, "phase_id": "1",
            "decision_frame": None, "evaluation": "successfullyCompleted", "result": None, "card": None,
            "possession_change": None, "counter_attack": None, "player": "home_8"}
    return base | kw


def test_league_events_become_p1_vocabulary():
    ev = pd.DataFrame([
        _ev(frame=100, kind="ThrowIn", team="away", decision_frame=60),
        _ev(frame=150, kind="Play", team="away", evaluation="unsuccessful"),
        _ev(frame=200, kind="TacklingGame", team="home", possession_change="true", evaluation=None),
        _ev(frame=300, kind="ShotAtGoal", team="home", result="SuccessfulShot", evaluation=None),
        _ev(frame=400, kind="Caution", team="away", card="yellowRed", evaluation=None),
    ])
    out = to_p1_events(ev)
    rows = list(zip(out["Type"], out["Subtype"].fillna(""), out["Team"], out["Start Frame"], strict=True))
    assert rows == [
        ("BALL OUT", "", "Home", 60),          # the ball went dead when the referee decided: the other team put it out
        ("SET PIECE", "THROW IN", "Away", 100),
        ("PASS", "", "Away", 100),
        ("BALL LOST", "", "Away", 150),
        ("CHALLENGE", "", "Home", 200),
        ("RECOVERY", "", "Home", 200),
        ("SHOT", "ON TARGET-GOAL", "Home", 300),
        ("CARD", "SECOND YELLOW", "Away", 400),
    ]
    assert set(out["Period"]) == {1}
    passes = out["Type"].isin(["PASS", "BALL LOST"])
    assert list(out.loc[passes, "End Frame"]) == [150, 200]  # a pass ends where the next event starts


def test_a_restart_is_timed_when_the_ball_comes_alive():
    ball = pd.DataFrame({"frame": np.arange(0, 400), "ball_status": np.r_[np.ones(100), np.zeros(80), np.ones(220)].astype(int)})
    assert kick_frame(150, ball) == 180   # logged at 150, the ball is live from 180
    assert kick_frame(300, ball) == 300   # no dead-to-live change nearby: the logged frame
