"""Possession chains from events alone (tools/event_chains.py), on a hand-built sequence."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from event_chains import chains, on_grid


def _e(t, team, kind, restart=None, depth=50.0, period=1):
    return {"period": period, "t": t, "team": team, "kind": kind, "restart": restart, "depth_m": depth}


def test_the_play_action_rule():
    ev = pd.DataFrame([
        _e(0, "home", "restart", "kick_off"), _e(2, "home", "pass", depth=55), _e(4, "away", "duel"), _e(4.2, "home", "duel"),
        _e(5, "home", "pass", depth=70),            # duels do not change possession
        _e(7, "away", "pass", depth=30),            # the other team acts: home lost it
        _e(9, "away", "shot", depth=90),            # a shot ends the possession
        _e(9.5, "home", "save", depth=2),           # the keeper's save belongs to the shot
        _e(12, "home", "pass", depth=10),           # the keeper plays it: a regain
        _e(15, None, "interruption"),               # ball out
        _e(30, "away", "restart", "throw_in", depth=60), _e(31, "away", "pass", depth=62),
        _e(40, "home", "pass", period=2),
    ])
    ch = chains(ev)
    assert list(zip(ch["team"], ch["start_type"], ch["end_type"], strict=True)) == [
        ("home", "restart:kick_off", "lost"), ("away", "regain", "shot"), ("home", "regain", "out"),
        ("away", "restart:throw_in", "period_end"), ("home", "period_start", "period_end")]
    assert ch["max_depth"].iloc[0] == 70 and ch["events"].iloc[0] == 3
    lab = on_grid(ch, np.array([1.0, 8.0, 20.0, 30.5]), np.array([1, 1, 1, 1]))
    assert list(lab) == ["home", "away", None, "away"]  # nobody between the ball going out and the throw-in
