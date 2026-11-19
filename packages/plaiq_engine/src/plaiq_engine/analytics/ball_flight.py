"""
ball_flight.py — Plaiq's ball flight and defender timing as explicit parameters (a candidate beside v1).

Targeted pitch control (``pass_option_value.targeted_pcz``) asks two questions about a pass: does a
defender reach a point on the way before the ball (interception risk), and who reaches the receiver
first (control). Both rest on three assumptions set by hand, never measured:

- **ball speed:** a pass over ``d`` metres travels at ``MIN_BALL_SPEED + BALL_SPEED_GRAD × d``
  (5 m/s plus 0.2 m/s per metre) on average;
- **timing along the pass:** each point on the way is timed as if it were the end of its own,
  shorter pass (``"point"``), so a 40 m ball reaches its 10 m mark in 1.43 s, when that ball moving
  at its own average pace would reach it in 0.77 s;
- **defenders:** every player starts after a 0.7 s reaction time and runs at up to 9.0 m/s,
  accelerating at 5.5 m/s².

``FlightModel`` makes each of these a parameter, so they can be calibrated on tracking data
(``tools/plaiq_calibration.py``, ``ponta-platform/PLAIQ_REFERENCES_REPORT.md``). With its defaults
(``V1``) ``targeted_pcz_flight`` returns exactly what ``targeted_pcz`` returns. The calibrated
alternatives:

- ``waypoints="pass"`` times each point along this pass: the pass takes ``time_to(D)`` and covers
  the fraction ``s / D`` of its length after the fraction ``τ`` of that time, with
  ``s / D = (1 + c)τ − cτ²`` (``c = decel``: 0 is constant speed, 1 stops at the end);
- ``reaction_time``, ``v_max`` and ``a_max``, when set, replace every defender's profile value.

A candidate: ``targeted_pcz``, POV, decision quality and every persisted score are untouched until a
calibration is measured and agreed (engine governance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plaiq_engine.analytics.pass_option_value import (
    _CTRL_SIGMOID_K,
    _corridor_waypoints,
)
from plaiq_engine.analytics.pitch_control import (
    BALL_SPEED_GRAD,
    DEFAULT_PROFILE,
    MIN_BALL_SPEED,
    player_travel_time,
)

BALL_FLIGHT_VERSION = "0.1.0"


@dataclass(frozen=True)
class FlightModel:
    """Ball flight and defender timing for targeted pitch control. The defaults are Plaiq v1."""

    min_speed: float = MIN_BALL_SPEED   # m/s, average pass speed extrapolated to d = 0
    speed_grad: float = BALL_SPEED_GRAD  # m/s per metre of pass length
    waypoints: str = "point"            # "point" (v1) or "pass"
    decel: float = 0.0                  # within-pass shape for waypoints="pass", 0 to 1
    reaction_time: float | None = None  # replaces every defender's reaction time when set
    v_max: float | None = None
    a_max: float | None = None

    def time_to(self, d) -> np.ndarray:
        """Time for a pass of length ``d`` to arrive (v1's speed model, with this model's parameters)."""
        d = np.asarray(d, float)
        return d / np.maximum(self.min_speed + self.speed_grad * d, 1e-6)

    def waypoint_times(self, length: float, s) -> np.ndarray:
        """When the ball of a pass of ``length`` metres passes the points ``s`` metres along it."""
        s = np.asarray(s, float)
        if self.waypoints == "point":
            return self.time_to(s)
        if self.waypoints != "pass":
            raise ValueError(f"waypoints must be 'point' or 'pass', not {self.waypoints!r}")
        frac = np.clip(s / max(length, 1e-6), 0.0, 1.0)
        c = float(np.clip(self.decel, 0.0, 1.0))
        tau = frac if c < 1e-9 else ((1 + c) - np.sqrt(np.maximum((1 + c) ** 2 - 4 * c * frac, 0.0))) / (2 * c)
        return tau * float(self.time_to(length))

    def profile(self, d: dict) -> tuple[float, float, float]:
        """A defender's (v_max, a_max, reaction_time), this model's values first."""
        return (self.v_max if self.v_max is not None else float(d.get("v_max", DEFAULT_PROFILE["v_max"])),
                self.a_max if self.a_max is not None else float(d.get("a_max", DEFAULT_PROFILE["a_max"])),
                self.reaction_time if self.reaction_time is not None else float(d.get("reaction_time", DEFAULT_PROFILE["reaction_time"])))


V1 = FlightModel()


def targeted_pcz_flight(ball_xy: np.ndarray, receiver_xy: np.ndarray, defender_data: list, flight: FlightModel = V1) -> tuple[float, float]:
    """``targeted_pcz`` with the ball flight and defender timing of ``flight``: (interception_risk, V_ctrl_receiver)."""
    if not defender_data:
        return 0.0, 0.5
    ball_xy, receiver_xy = np.asarray(ball_xy, float), np.asarray(receiver_xy, float)
    length = float(np.linalg.norm(receiver_xy - ball_xy))
    waypoints = _corridor_waypoints(ball_xy, receiver_xy)
    t_ball_wpts = flight.waypoint_times(length, np.linalg.norm(waypoints - ball_xy, axis=1)) if len(waypoints) else np.array([])
    t_ball_recv = float(flight.time_to(length))
    t_def_wpts, t_def_recv = [], []
    for dd in defender_data:
        pos, vel = np.asarray(dd["pos"], float), np.asarray(dd.get("vel", [0.0, 0.0]), float)
        vmax, amax, rt = flight.profile(dd)
        if len(waypoints):
            t_def_wpts.append(player_travel_time(pos, vel, waypoints, vmax, amax, rt))
        t_def_recv.append(player_travel_time(pos, vel, receiver_xy[None, :], vmax, amax, rt)[0])
    risk = float((np.min(t_def_wpts, axis=0) <= t_ball_wpts).mean()) if t_def_wpts else 0.0
    gap = min(t_def_recv) - t_ball_recv
    ctrl = float(1.0 / (1.0 + np.exp(-float(np.clip(gap * _CTRL_SIGMOID_K, -10.0, 10.0)))))
    return risk, ctrl
