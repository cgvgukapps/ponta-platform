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


def eleven_a_side(*, home: str = "4-4-2", away: str = "4-3-3", seconds: float = 360.0, home_attacks_positive: bool = True,
                  ball_holder: str | None = "Home_7", events: list[dict] | None = None):
    """
    Two fictional teams of eleven in template formations (``structure.TEMPLATES``), each player
    swaying about a metre around their slot, for ``seconds`` in one period. The ball sits at
    ``ball_holder``'s feet (None: undetected). ``home_attacks_positive=False`` mirrors the
    pitch, as Metrica Sample Game 2 is normalised.
    """
    from plaiq_engine.context.structure import TEMPLATES

    pitch = PitchConfig(fps=FPS)
    half = pitch.length_m / 2
    n = int(seconds * FPS)
    frames = np.arange(1, n + 1)
    t = (frames - 1) / FPS
    idx = pd.Index(frames, name="Frame")
    clock = {"Period": np.ones(n, dtype=int), "Time [s]": t}
    sgn = 1.0 if home_attacks_positive else -1.0

    def side(name: str, formation: str, sign: float, phase: float) -> pd.DataFrame:
        cols = {}
        slots = [("GK", 5.0, 0.0)] + list(TEMPLATES[formation])
        for k, (_, depth, width) in enumerate(slots, start=1):
            sway = np.sin(2 * np.pi * t / 30.0 + phase + k)
            cols[f"{name}_{k}_x"] = sign * (depth + sway - half)
            cols[f"{name}_{k}_y"] = sign * (width + 0.8 * np.cos(2 * np.pi * t / 25.0 + k))
        return pd.DataFrame({**clock, **cols}, index=idx)

    home_df = side("Home", home, sgn, 0.0)
    away_df = side("Away", away, -sgn, 1.0)
    if ball_holder is None:
        away_df["ball_x"], away_df["ball_y"] = np.nan, np.nan
    else:
        src = home_df if ball_holder.startswith("Home") else away_df
        away_df["ball_x"], away_df["ball_y"] = src[f"{ball_holder}_x"].to_numpy() + 0.3, src[f"{ball_holder}_y"].to_numpy()
    ev = pd.DataFrame(events or [], columns=EVENT_COLUMNS)
    return from_wide_frames(home_df, away_df, ev, provider="metrica", match_id="t11", fps=FPS, pitch=pitch)
