"""
A 10-second synthetic match with fictional players and no sample data (licence-free).

Shared by the semantic builder tests and the synthetic engine golden
(``tests/golden/synthetic/``, ``tools/synthetic_golden.py``).
"""

import numpy as np
import pandas as pd
from matrix_engine.match import from_wide_frames
from matrix_engine.pitch_config import PitchConfig

N = 250  # 10 s at 25 Hz
FPS = 25.0


def _team(prefix: str, sign: float, rng) -> dict:
    cols = {}
    base = [(-48, 0)] + [(-30 + 6 * i, -25 + 5 * i) for i in range(10)]  # a goalkeeper deepest, then outfield
    for k, (bx, by) in enumerate(base, start=1):
        drift = np.linspace(0, 3, N)[:, None] * rng.normal(0, 1, 2)
        cols[f"{prefix}_{k}_x"] = sign * (bx + drift[:, 0])
        cols[f"{prefix}_{k}_y"] = sign * (by + drift[:, 1])
    return cols


def synthetic_match():
    """A 10-second synthetic match: fictional players, no sample data. Also feeds the UI's sample match page."""
    rng = np.random.default_rng(3)
    idx = pd.Index(np.arange(1, N + 1), name="Frame")
    clock = {"Period": np.ones(N, int), "Time [s]": np.arange(N) / FPS}
    home = pd.DataFrame({**clock, **_team("Home", 1.0, rng)}, index=idx)
    away = pd.DataFrame({**clock, **_team("Away", -1.0, rng)}, index=idx)
    ball_x = np.interp(np.arange(N), [0, 60, 120, 180, 249], [-24, -12, 0, 6, 12])
    ball_y = np.interp(np.arange(N), [0, 60, 120, 180, 249], [-20, -10, -5, 0, 5])
    away["ball_x"], away["ball_y"] = ball_x, ball_y
    events = pd.DataFrame([
        # A set piece then the pass that takes it: the pass is not scored.
        {"Type": "SET PIECE", "Subtype": "THROW IN", "Period": 1, "From": "Player 3", "To": np.nan, "Team": "Home", "Start Frame": 1, "Start Time [s]": 0.0, "Start X": -18.0, "Start Y": -15.0, "End Frame": 1, "End Time [s]": 0.0, "End X": np.nan, "End Y": np.nan},
        {"Type": "PASS", "Subtype": np.nan, "Period": 1, "From": "Player 3", "To": "Player 4", "Team": "Home", "Start Frame": 1, "Start Time [s]": 0.0, "Start X": -18.0, "Start Y": -15.0, "End Frame": 40, "End Time [s]": 1.56, "End X": -12.0, "End Y": -10.0},
        {"Type": "PASS", "Subtype": np.nan, "Period": 1, "From": "Player 4", "To": "Player 6", "Team": "Home", "Start Frame": 61, "Start Time [s]": 2.4, "Start X": -12.0, "Start Y": -10.0, "End Frame": 120, "End Time [s]": 4.76, "End X": 0.0, "End Y": -5.0},
        {"Type": "CARRY", "Subtype": np.nan, "Period": 1, "From": "Player 6", "To": np.nan, "Team": "Home", "Start Frame": 121, "Start Time [s]": 4.8, "Start X": 0.0, "Start Y": -5.0, "End Frame": 180, "End Time [s]": 7.16, "End X": 6.0, "End Y": 0.0},
        {"Type": "PASS", "Subtype": np.nan, "Period": 1, "From": "Player 1", "To": "Player 2", "Team": "Home", "Start Frame": 181, "Start Time [s]": 7.2, "Start X": 6.0, "Start Y": 0.0, "End Frame": 220, "End Time [s]": 8.76, "End X": 10.0, "End Y": 4.0},
        {"Type": "RECOVERY", "Subtype": np.nan, "Period": 1, "From": "Player 5", "To": np.nan, "Team": "Away", "Start Frame": 230, "Start Time [s]": 9.16, "Start X": 12.0, "Start Y": 5.0, "End Frame": 230, "End Time [s]": 9.16, "End X": 12.0, "End Y": 5.0},
    ])
    return from_wide_frames(home, away, events, provider="synthetic", match_id="1", fps=FPS, pitch=PitchConfig(fps=FPS))
