"""
match_analysis.py — the ``MatchAnalysis`` façade.

The prototype's Streamlit app drove the Plaiq engine through about thirty ad hoc
imports and its own caching (inventory: ``docs/MATCH_ANALYSIS_INVENTORY.md``). This
class is the one entry point that replaces that. It **wraps; it does not rewrite**:
every number comes from the same engine functions, called in the same order and with
the same arguments as the prototype (the order verified by
``tools/golden/capture_prototype_golden.py``).

Input is a Matrix ``NormalisedMatch`` (decision recorded in
``docs/ARCHITECTURE.md``). The façade rebuilds the wide frames the engine reads,
derives velocities from positions exactly as the prototype's preprocessing did, and
tags events with ΔV, then computes on demand and caches.

Rendering helpers (Plotly trace builders) are deliberately not exposed: they belong to
the web pitch, not the engine.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
from matrix_engine.ingestion.metrica_velocities import calc_player_velocities
from matrix_engine.match import NormalisedMatch, to_wide_frames

from plaiq_engine.analytics.hex_analytics import (
    assign_all_players,
    compute_pressure,
    infer_possession,
    tag_possession,
)
from plaiq_engine.analytics.hex_value import (
    compute_movement_delta_v,
    compute_movement_value,
    compute_value,
    pressure_field,
    smooth_control,
    tag_events_delta_v,
    xT_surface,
)
from plaiq_engine.analytics.pass_option_value import (
    _is_failed_pass_event,
    compute_event_pov,
)
from plaiq_engine.analytics.pitch_control import (
    DEFAULT_PROFILE,
    build_pitch_hex_grid,
    compute_frame_control,
)
from plaiq_engine.statsbomb.sb_pov_adapter import _nearest_hex

#: Hexagon circumradius giving one square metre per hexagon, as in the prototype app.
HEX_RADIUS: float = float(np.sqrt(2.0 / (3.0 * np.sqrt(3))))

#: Natural keys used to sort engine output. ``compute_event_pov`` returns rows in an
#: order that varies between runs; values do not. Sorting makes output deterministic.
POV_KEY: tuple[str, ...] = ("event_idx", "team", "jersey")
ALT_KEY: tuple[str, ...] = ("event_idx", "hex_x", "hex_y")


# ---------------------------------------------------------------------------
# Public wrappers for two engine helpers that the app used under private names.
# Thin on purpose: the engine code is not edited.
# ---------------------------------------------------------------------------

def is_failed_pass_event(event_type: str, subtype, ball_xy_start, ball_xy_end) -> bool:
    """Whether a BALL LOST event is scored as a failed pass (see ``pass_option_value``)."""
    return _is_failed_pass_event(event_type, subtype, ball_xy_start, ball_xy_end)


def nearest_hex(xy: np.ndarray, hex_centers: np.ndarray) -> int | None:
    """Index of the hexagon centre nearest to ``xy``."""
    return _nearest_hex(xy, hex_centers)


def default_profiles(jerseys: list[str]) -> dict[str, dict]:
    """Every jersey on the engine's default physical profile."""
    return {j: DEFAULT_PROFILE for j in jerseys}


class MatchAnalysis:
    """
    Plaiq analysis of one match.

    Parameters
    ----------
    match : NormalisedMatch
        The match as Matrix normalised it.
    profiles_home, profiles_away : dict, optional
        Per-jersey physical profiles (``v_max``, ``a_max``, ``reaction_time``).
        Default: the engine's default profile for every jersey.
    """

    def __init__(
        self,
        match: NormalisedMatch,
        *,
        profiles_home: dict | None = None,
        profiles_away: dict | None = None,
    ) -> None:
        self.match = match
        home, away = to_wide_frames(match)
        # Velocities from positions, as the prototype's preprocessing did (same function,
        # same frame rate): the default window at 25 Hz, auto-scaled at other rates.
        self.home = calc_player_velocities(home, fps=match.fps)
        self.away = calc_player_velocities(away, fps=match.fps)
        self.home_jerseys = list(match.slots.get("home", ()))
        self.away_jerseys = list(match.slots.get("away", ()))
        self.profiles_home = profiles_home or default_profiles(self.home_jerseys)
        self.profiles_away = profiles_away or default_profiles(self.away_jerseys)
        self.R = HEX_RADIUS

    # -- geometry and static surfaces ------------------------------------------

    @cached_property
    def hex_centers(self) -> np.ndarray:
        return build_pitch_hex_grid(self.R)

    @cached_property
    def xt_home(self) -> np.ndarray:
        return xT_surface(self.hex_centers, attacking_right=True)

    @cached_property
    def xt_away(self) -> np.ndarray:
        return xT_surface(self.hex_centers, attacking_right=False)

    # -- events -----------------------------------------------------------------

    @cached_property
    def events(self) -> pd.DataFrame:
        """The event feed tagged with event ΔV."""
        return tag_events_delta_v(self.match.events.copy(), self.xt_home, self.xt_away, self.hex_centers, self.R)

    @cached_property
    def possession(self) -> pd.Series:
        """Frame-level possession inferred from the events."""
        return infer_possession(self.events, len(self.home))

    # -- frame analytics ----------------------------------------------------------

    @cached_property
    def hex_assignment(self) -> pd.DataFrame:
        return assign_all_players(self.home, self.away, self.home_jerseys, self.away_jerseys, self.R)

    @cached_property
    def frame_analytics(self) -> pd.DataFrame:
        """Per frame and player: hex, pressure and possession tag."""
        press = compute_pressure(self.hex_assignment)
        tagged = tag_possession(press, self.possession)
        return tagged.merge(self.hex_assignment[["Frame", "team", "jersey", "q", "r"]], on=["Frame", "team", "jersey"], how="left")

    @cached_property
    def frame_analytics_with_value(self) -> pd.DataFrame:
        """Frame analytics plus the movement ΔV columns the decision engine reads."""
        return compute_movement_delta_v(self.frame_analytics, self.xt_home, self.xt_away, self.hex_centers, self.R)

    def pressure(self, frame: int) -> pd.DataFrame:
        """Pressure per player at one frame."""
        fa = self.frame_analytics
        return fa.loc[fa["Frame"] == frame].reset_index(drop=True)

    # -- event-level analytics -------------------------------------------------------

    @cached_property
    def movement_value(self) -> pd.DataFrame:
        """Off-ball movement value per event and player."""
        return compute_movement_value(self.frame_analytics_with_value, self.events, self.xt_home, self.xt_away, self.hex_centers, self.R)

    @cached_property
    def _pov(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        pov, alts = compute_event_pov(
            self.frame_analytics_with_value, self.events, self.home, self.away,
            self.home_jerseys, self.away_jerseys, self.xt_home, self.xt_away,
            self.hex_centers, self.R, self.profiles_home, self.profiles_away, fps=self.match.fps,
        )
        pov = pov.sort_values(list(POV_KEY), kind="stable").reset_index(drop=True)
        if len(alts) and not alts.duplicated(list(ALT_KEY)).any():
            alts = alts.sort_values(list(ALT_KEY), kind="stable").reset_index(drop=True)
        return pov, alts

    def event_pov(self) -> pd.DataFrame:
        """Pass option value, ΔPOV and decision quality per event and player, sorted on its natural key."""
        return self._pov[0]

    def event_alternatives(self) -> pd.DataFrame:
        """The scored alternatives behind each decision."""
        return self._pov[1]

    # -- per-frame surfaces (computed on demand) ------------------------------------------

    def frame_control(self, frame: int) -> tuple:
        """
        Control surface at one frame: ``(control, t_ball, t_home, t_away)``.

        ``t_home`` is ``None`` when the ball was not detected at that frame; the
        engine then returns an uncontrolled grid, as the prototype did.
        """
        control, t_ball, t_h, t_a, _, _ = compute_frame_control(
            self.home.loc[frame], self.away.loc[frame], self.home_jerseys, self.away_jerseys,
            self.hex_centers, profiles_home=self.profiles_home, profiles_away=self.profiles_away,
        )
        return control, t_ball, t_h, t_a

    def value_surface(self, frame: int) -> np.ndarray | None:
        """
        Effective value per hexagon for the team in possession at ``frame``.

        ``None`` when the ball was not detected or no team is in possession.
        """
        _, t_ball, t_h, t_a = self.frame_control(frame)
        side = self.possession.get(frame)
        if t_h is None or side not in ("Home", "Away"):
            return None
        vh, va = smooth_control(t_h, t_a, t_ball)
        fp = self.frame_analytics
        fp = fp.loc[fp["Frame"] == frame, ["team", "jersey", "pressure"]]
        disc = pressure_field(self.home.loc[frame], self.away.loc[frame], self.home_jerseys, self.away_jerseys, self.hex_centers, fp, side)
        return np.asarray(compute_value(self.xt_home if side == "Home" else self.xt_away, vh if side == "Home" else va, disc))
