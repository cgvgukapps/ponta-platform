"""A tiny synthetic match for context-package tests: fictional players, no sample data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matrix_engine.match import from_wide_frames
from matrix_engine.pitch_config import PitchConfig

FPS = 25.0
EVENT_COLUMNS = ["Type", "Subtype", "Period", "From", "To", "Team", "Start Frame", "Start Time [s]", "Start X", "Start Y",
                 "End Frame", "End Time [s]", "End X", "End Y"]


def event(type_: str, team: str, frame: int, subtype=np.nan, period: int = 1, end_frame=None) -> dict:
    return {"Type": type_, "Subtype": subtype, "Period": period, "From": "Player 1", "To": np.nan, "Team": team,
            "Start Frame": frame, "Start Time [s]": (frame - 1) / FPS, "Start X": 0.0, "Start Y": 0.0,
            "End Frame": frame if end_frame is None else end_frame, "End Time [s]": ((frame if end_frame is None else end_frame) - 1) / FPS,
            "End X": np.nan, "End Y": np.nan}


def tiny_match(events: list[dict] | None = None, *, seconds: float = 20.0, provider: str = "metrica", ball=None, periods=(1,)):
    """
    Two players a side standing still: home_1 at (0, 0), away_1 at (10, 0), the others far away.
    ``ball(t)`` returns (x, y) or None (undetected) for time t in seconds; default: always undetected.
    """
    n_per = int(seconds * FPS)
    frames = np.arange(1, n_per * len(periods) + 1)
    period = np.repeat(np.array(periods), n_per)
    t = (frames - 1) / FPS
    idx = pd.Index(frames, name="Frame")
    clock = {"Period": period, "Time [s]": t}
    home = pd.DataFrame({**clock, "Home_1_x": 0.0, "Home_1_y": 0.0, "Home_2_x": -20.0, "Home_2_y": 10.0}, index=idx)
    away = pd.DataFrame({**clock, "Away_1_x": 10.0, "Away_1_y": 0.0, "Away_2_x": 20.0, "Away_2_y": -10.0}, index=idx)
    bx, by = np.full(len(t), np.nan), np.full(len(t), np.nan)
    if ball is not None:
        for i, ti in enumerate(t):
            p = ball(ti)
            if p is not None:
                bx[i], by[i] = p
    away["ball_x"], away["ball_y"] = bx, by
    ev = pd.DataFrame(events or [], columns=EVENT_COLUMNS)
    return from_wide_frames(home, away, ev, provider=provider, match_id="t1", fps=FPS, pitch=PitchConfig(fps=FPS))
