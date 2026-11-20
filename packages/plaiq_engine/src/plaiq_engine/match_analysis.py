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
from scipy.spatial import cKDTree

from plaiq_engine.analytics.decision_quality_v2 import (
    DQ_V2_RINGS,
    DQ_V2_VERSION,
    GAIN_V2_VERSION,
    decision_gain,
    decision_quality_cells,
    dq_v2_band,
    option_cells_probabilities,
    option_value,
)
from plaiq_engine.analytics.event_stitch import (
    ball_xy_from_tracking,
    build_pass_receipt_lookup,
    clean_jersey,
)
from plaiq_engine.analytics.hex_analytics import (
    assign_all_players,
    cartesian_to_hex,
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
    _build_defender_data,
    _is_failed_pass_event,
    compute_event_pov,
)
from plaiq_engine.analytics.pass_probability import (
    parameters_for,
    pass_arrays,
    probability_from_arrays,
    projected_target,
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

    def _pass_moment(self, e: pd.Series) -> dict | str:
        """
        The ball and players at the moment of one pass, as the pass completion probability takes them, or the
        reason there is none. Shared by ``pass_completion`` and ``decision_quality_v2`` so both read the same moment.
        """
        team = str(e.get("Team", "")).capitalize()
        own, opp = (self.home, self.away) if team == "Home" else (self.away, self.home)
        jerseys, ojer = (self.home_jerseys, self.away_jerseys) if team == "Home" else (self.away_jerseys, self.home_jerseys)
        prof, oprof = (self.profiles_home, self.profiles_away) if team == "Home" else (self.profiles_away, self.profiles_home)
        passer, target = clean_jersey(e.get("From")), clean_jersey(e.get("To"))
        start = e.get("Start Frame")
        if not target:
            return "no logged target for this pass"
        if pd.isna(start) or int(start) not in own.index:
            return "no tracking at the pass"
        f = int(start)
        # The ball where it is tracked, else at the passer's feet. Never the event's coordinates, which are
        # not always in the tracking's frame (Metrica CSV second halves, PLAIQ_REFERENCES_REPORT.md).
        ball = np.array([self.away.at[f, "ball_x"], self.away.at[f, "ball_y"]], float) if "ball_x" in self.away.columns else np.full(2, np.nan)
        if not np.isfinite(ball).all() and f"{team}_{passer}_x" in own.columns:
            ball = np.array([own.at[f, f"{team}_{passer}_x"], own.at[f, f"{team}_{passer}_y"]], float)
        if not np.isfinite(ball).all():
            return "neither the ball nor the passer is tracked at the pass"
        col = f"{team}_{target}"
        rec = np.array([own.at[f, f"{col}_x"], own.at[f, f"{col}_y"]], float) if f"{col}_x" in own.columns else np.full(2, np.nan)
        if not np.isfinite(rec).all():
            return "the target is not tracked at the pass"
        vel = np.array([own.at[f, f"{col}_vx"], own.at[f, f"{col}_vy"]], float) if f"{col}_vx" in own.columns else np.zeros(2)
        vel = np.where(np.isfinite(vel), vel, 0.0)
        opp_team = "Away" if team == "Home" else "Home"
        return {"frame": f, "team": team, "passer": passer, "target": target, "ball": ball, "receiver": rec, "velocity": vel,
                "own": own.loc[f], "mates_jerseys": [j for j in jerseys if str(j) != passer], "profiles": prof,
                "defenders": _build_defender_data(opp.loc[f], opp_team, ojer, oprof)}

    @cached_property
    def pass_completion(self) -> pd.DataFrame:
        """
        Pass completion probability per pass (``analytics/pass_probability.py``, adopted 19 November 2026).

        One row per ``PASS`` and ``BALL LOST`` event: the probability, known at the moment of the pass, that it
        reaches the logged target, with that target projected along their run to the ball's arrival, from the
        tracked ball (or the passer, when the ball is not detected) at the event's start frame. Uses the
        provider's fitted parameter set where one exists, else the default set. A pass without a logged target
        (most providers log none for a failed pass) or without tracking at the pass gets no probability and a
        reason; Plaiq's inferred failed-pass receiver is never used (it is not validated). Labelled estimated.
        """
        params = parameters_for(self.match.provider)
        cols = ["event_idx", "pass_completion_probability", "p_survive", "share_at_target", "parameter_set", "reason"]
        rows, batch, where = [], [], []
        for idx, e in self.events.iterrows():
            if str(e.get("Type", "")).upper() not in ("PASS", "BALL LOST"):
                continue
            if str(e.get("Team", "")).capitalize() not in ("Home", "Away"):
                continue
            row = {"event_idx": idx, "pass_completion_probability": np.nan, "p_survive": np.nan, "share_at_target": np.nan,
                   "parameter_set": params.name, "reason": None}
            rows.append(row)
            m = self._pass_moment(e)
            if isinstance(m, str):
                row["reason"] = m
                continue
            batch.append({"ball": m["ball"], "target": projected_target(m["ball"], m["receiver"], m["velocity"], params.model.flight),
                          "attackers": _build_defender_data(m["own"], m["team"], m["mates_jerseys"], m["profiles"]),
                          "defenders": m["defenders"], "high": False})
            where.append(row)
        if batch:
            n = max(1, max(max(len(p["attackers"]), len(p["defenders"])) for p in batch))
            r = probability_from_arrays(pass_arrays(batch, params.model, max_players=n), params.model)
            for i, row in enumerate(where):
                row["pass_completion_probability"] = float(r["p_complete"][i])
                row["p_survive"], row["share_at_target"] = float(r["p_survive"][i]), float(r["share_at_target"][i])
        return pd.DataFrame(rows, columns=cols).sort_values("event_idx", kind="stable").reset_index(drop=True)

    @cached_property
    def _hex_tree(self) -> cKDTree:
        return cKDTree(self.hex_centers)

    def _xt_at(self, team: str, points) -> np.ndarray:
        """Plaiq's threat for ``team`` (``Home`` or ``Away``) at each point (nearest hexagon)."""
        surface = self.xt_home if team == "Home" else self.xt_away
        return surface[self._hex_tree.query(np.asarray(points, float).reshape(-1, 2))[1]]

    @cached_property
    def _hex_axial(self) -> tuple[np.ndarray, np.ndarray]:
        q, r = cartesian_to_hex(self.hex_centers[:, 0], self.hex_centers[:, 1], self.R)
        return np.asarray(q, int), np.asarray(r, int)

    @cached_property
    def decision_quality_v2(self) -> pd.DataFrame:
        """
        Decision quality v2 (``analytics/decision_quality_v2.py`` 1.1.0, adopted beside v1 19 November 2026).

        One row per ``PASS`` and ``BALL LOST`` event that ``pass_completion`` covers. At the moment of the pass
        every team-mate (not the passer) is an option: **the best ball into their run**, the best of the hex
        cells within ``DQ_V2_RINGS`` of the cell their run reaches when the ball arrives, each cell worth the
        completion probability of a pass to it times its value. The score is the chosen option's expected value
        over the best one's, **0 to 1** (1.0: the best option was chosen), with the agreed bands (``dq_v2_band``).
        The chosen option is the logged target, scored like every other, so completed and failed passes take one
        formula and the outcome is never an input. ``chosen_cell`` and ``best_cell`` are grid cell indices for
        storage and the replay, never for the AI layer. v1 (``event_pov``) is unchanged and stays the default.
        Labelled estimated.
        """
        params = parameters_for(self.match.provider)
        q_all, r_all = self._hex_axial
        h = self.hex_centers
        cols = ["event_idx", "dq_v2", "dq_v2_band", "ev_chosen", "ev_best", "p_chosen", "chosen_cell", "best_option", "best_cell",
                "best_is_chosen", "options", "cells", "version", "parameter_set", "reason"]
        pc = self.pass_completion.set_index("event_idx")
        rows = []
        for idx, r in pc.iterrows():
            row = {"event_idx": idx, "dq_v2": np.nan, "dq_v2_band": None, "ev_chosen": np.nan, "ev_best": np.nan, "p_chosen": np.nan,
                   "chosen_cell": None, "best_option": None, "best_cell": None, "best_is_chosen": None, "options": 0, "cells": 0,
                   "version": DQ_V2_VERSION, "parameter_set": r["parameter_set"], "reason": None}
            rows.append(row)
            if pd.isna(r["pass_completion_probability"]):
                row["reason"] = r["reason"]
                continue
            m = self._pass_moment(self.events.loc[idx])
            names, mates = [], []
            for j in m["mates_jerseys"]:
                d = _build_defender_data(m["own"], m["team"], [j], m["profiles"])
                if d:
                    names.append(str(j))
                    mates.append(d[0])
            if m["target"] not in names:
                row["reason"] = "the target is not tracked at the pass"
                continue
            cells, p = option_cells_probabilities(m["ball"], mates, m["defenders"], params.model, h, q_all, r_all, self.R, rings=DQ_V2_RINGS)
            xt = self.xt_home if m["team"] == "Home" else self.xt_away
            xt_ball = float(self._xt_at(m["team"], m["ball"])[0])
            values = [np.array([option_value(m["ball"], h[c], xt_ball, xt[c]) for c in cs]) for cs in cells]
            k = names.index(m["target"])
            out = decision_quality_cells(p, values, k)
            kc = out["best_cell"][k]
            row.update({"ev_chosen": out["ev_chosen"], "ev_best": out["ev_best"], "p_chosen": float(p[k][kc]), "chosen_cell": int(cells[k][kc]),
                        "options": out["options"], "cells": int(sum(len(c) for c in cells))})
            if not np.isfinite(out["dq"]):
                row["reason"] = "no option on offer has any value"
                continue
            b = out["best_option"]
            row.update({"dq_v2": out["dq"], "dq_v2_band": dq_v2_band(out["dq"]), "best_option": names[b],
                        "best_cell": int(cells[b][out["best_cell"][b]]), "best_is_chosen": out["best_is_chosen"]})
        return pd.DataFrame(rows, columns=cols).sort_values("event_idx", kind="stable").reset_index(drop=True)

    @cached_property
    def gain_v2(self) -> pd.DataFrame:
        """
        Gain version 2 (``analytics/decision_quality_v2.decision_gain``, 2.0.0, adopted beside version 1
        20 November 2026): Gain from Choice's own parts. For every pass Choice scores, the value realised where
        the ball arrived (``option_value`` at the tracked ball at version 1's arrival frame, the stitched receipt,
        else the receiver; **0 for a failed pass**) over Choice's best expected value on offer at release, clipped
        0 to 1.5. ``execution`` is the realised value over the chosen option's expected value, so Gain = Choice x
        execution. Version 1 (``event_pov``) stays the default Gain. Passes only (version 1 also scores carries).
        Labelled estimated.
        """
        cols = ["event_idx", "gain_v2", "realised_value", "execution", "arrival_frame", "version", "reason"]
        d2 = self.decision_quality_v2.set_index("event_idx")
        receipts = build_pass_receipt_lookup(self.events)
        rows = []
        for idx, c in d2.iterrows():
            row = {"event_idx": idx, "gain_v2": np.nan, "realised_value": np.nan, "execution": np.nan, "arrival_frame": None,
                   "version": GAIN_V2_VERSION, "reason": None}
            rows.append(row)
            if pd.isna(c["dq_v2"]):
                row["reason"] = c["reason"]
                continue
            e = self.events.loc[idx]
            m = self._pass_moment(e)
            if str(e.get("Type", "")).upper() == "PASS":
                r = receipts.get(idx) or {}
                f = int(r["receipt_start_frame"]) if r.get("stitch_valid") and r.get("receipt_start_frame") is not None else int(e["End Frame"])
                own = self.home if m["team"] == "Home" else self.away
                col = f"{m['team']}_{m['target']}"
                rec = np.array([own.at[f, f"{col}_x"], own.at[f, f"{col}_y"]], float) if f in own.index and f"{col}_x" in own.columns else None
                arrival = ball_xy_from_tracking(self.home, self.away, f, fallback=rec)
                if arrival is None or not np.isfinite(arrival).all():
                    row["reason"] = "neither the ball nor the receiver is tracked where the pass arrived"
                    continue
                row["arrival_frame"] = f
                realised = option_value(m["ball"], arrival, float(self._xt_at(m["team"], m["ball"])[0]), float(self._xt_at(m["team"], arrival)[0]))
            else:
                realised = 0.0
            row.update({"gain_v2": decision_gain(realised, float(c["ev_best"])), "realised_value": realised,
                        "execution": realised / float(c["ev_chosen"]) if c["ev_chosen"] > 0 else np.nan})
        return pd.DataFrame(rows, columns=cols).sort_values("event_idx", kind="stable").reset_index(drop=True)

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
