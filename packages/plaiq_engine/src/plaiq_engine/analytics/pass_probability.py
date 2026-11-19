"""
pass_probability.py — a probabilistic, height-aware pass completion model (a candidate beside targeted PCZ).

Targeted pitch control (``pass_option_value.targeted_pcz``) gives availability as
``V_ctrl × (1 − interception_risk)``: the share of points on the pass's line some defender reaches
before the ball, and a fixed sigmoid of the time gap at the receiver. Step 6 of the Plaiq review
(``ponta-platform/PLAIQ_REFERENCES_REPORT.md``) measured the physics under it and found that
correcting the constants does not close the gap to a provider's completion model: the formula has no
ball height, no reach or time to control, and is not a probability.

This model follows the structure of the published pitch-control model (Spearman, 2018, "Beyond
expected goals"), written for one pass:

- **arrival:** each player reaches a point after a reaction time plus Plaiq's kinematic travel time
  (``pitch_control.player_travel_time``); real arrival is uncertain, so the chance a player has
  arrived by time ``t`` is a logistic curve around that time with spread ``sigma``;
- **in flight:** the ball travels the pass's line with the measured flight (``MEASURED_FLIGHT``,
  step 6: 8.5 m/s plus 0.165 m/s per metre, slowing along the pass). While it passes each metre,
  every defender who has arrived there may take it at rate ``lam_flight``; the chance the ball
  survives is ``exp(−∫ lam_flight Σ_j f_j dt)``. A high ball can only be reached in the first and last
  ``high_reach`` of its length;
- **at the target:** from the ball's arrival, both teams' players (the passer excluded) arrive and
  control it as in Spearman's model, ``dP_team / dt = (1 − P_home − P_away) × lam × Σ f_i``, over
  ``horizon_s``; the passing team's share of control is the chance it keeps the ball there;
- **the probability** the pass is completed is survival × the passing team's share at the target;
- **the target** can be where the receiver stands at the pass, or ``projected_target``: where a
  receiver running at constant velocity will be when the ball arrives (a ball played into a run).

The receiving team and the defenders have their own reaction times (the receiver knows where the
ball is going). Parameters are fitted by likelihood on tracking (``tools/plaiq_pass_probability.py``);
``PassModel()`` holds Spearman's published values with the measured flight, unfitted. A candidate:
targeted PCZ, POV, decision quality and every persisted score are untouched (engine governance).

**Adopted 19 November 2026** (founder) as the pass completion probability metric, beside v1 and labelled
estimated. Parameters are fitted per provider (``PARAMETER_SETS``, ``parameters_for``); a provider without
its own set uses the default set and the Data check marks the metric Degraded. The shipped sets use the
projected target and no height flag. Matrix's events now carry a provider's height flag (``High Pass``,
19 November 2026), but it is not read here: SkillCorner logs it on completed passes only, so it gives away
the outcome, and on IDSSE, where it is logged on every pass, it made no difference held out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from plaiq_engine.analytics.ball_flight import FlightModel
from plaiq_engine.analytics.pitch_control import player_travel_time

#: 0.1.0 (19 November 2026): candidate. 1.0.0 (19 November 2026): adopted by the founder as the pass
#: completion probability metric, beside v1, labelled estimated, with the parameter sets below.
PASS_PROBABILITY_VERSION = "1.0.0"

#: The ball flight measured on the seven IDSSE matches (step 6, fitted on the even matches).
MEASURED_FLIGHT = FlightModel(min_speed=8.527, speed_grad=0.1654, waypoints="pass", decel=0.39)

_K = np.pi / np.sqrt(3.0)
_INF = 1e6


@dataclass(frozen=True)
class PassModel:
    """Parameters of the pass completion model. The defaults are Spearman's published values, unfitted."""

    sigma: float = 0.45        # s, spread of arrival times
    lam: float = 4.3           # 1/s, rate of controlling a ball that has arrived, at the target
    lam_flight: float = 4.3    # 1/s, rate of taking a ball in flight
    rt_att: float = 0.7        # s, the passing team's reaction time
    rt_def: float = 0.7        # s, the defenders' reaction time
    high_reach: float = 0.15   # share of a high pass's length reachable at each end
    v_max: float = 9.0
    a_max: float = 5.5
    flight: FlightModel = field(default=MEASURED_FLIGHT)
    step_m: float = 1.0
    horizon_s: float = 3.0
    dt_s: float = 0.04


@dataclass
class PassArrays:
    """Per-pass travel times without reaction (padded with ``_INF``), so parameters can be fitted cheaply."""

    t_ball: np.ndarray      # (N, K) time the ball passes each path point
    dt_ball: np.ndarray     # (N, K) time the ball spends in each step (0 = padding)
    frac: np.ndarray        # (N, K) position along the pass, 0 to 1
    tau_def_path: np.ndarray  # (N, D, K) defenders' kinematic time to each path point
    t_target: np.ndarray    # (N,) ball arrival at the target
    tau_att: np.ndarray     # (N, A) passing team's kinematic time to the target
    tau_def: np.ndarray     # (N, D) defenders' kinematic time to the target
    high: np.ndarray        # (N,) bool


@dataclass(frozen=True)
class ParameterSet:
    """A fitted, named parameter set and the evidence behind it (``ponta-platform/PLAIQ_REFERENCES_REPORT.md``)."""

    name: str
    model: PassModel
    fitted_on: str
    held_out: str


#: Fitted parameter sets by provider. ``default`` serves every provider without its own set.
PARAMETER_SETS: dict[str, ParameterSet] = {
    "default": ParameterSet(
        name="default-idsse-2026-11-19",
        model=PassModel(sigma=0.442, lam=30.0, lam_flight=0.231, rt_att=0.0, rt_def=0.585),
        fitted_on="IDSSE (German league, TRACAB 25 Hz): 1,852 open-play passes of the 4 even matches; projected target, no height",
        held_out="IDSSE 3 odd matches: log-loss 0.301, AUC 0.773; SkillCorner 20 matches: log-loss 0.373, AUC 0.838 (over-predicts by 5 points)"),
    "skillcorner": ParameterSet(
        name="skillcorner-2026-11-19",
        model=PassModel(sigma=0.541, lam=30.0, lam_flight=0.469, rt_att=0.665, rt_def=0.863),
        fitted_on="SkillCorner open data: 6,173 passes of the 10 even matches; projected target, no height",
        held_out="SkillCorner 10 odd matches: log-loss 0.356, AUC 0.838, calibrated within 0.05 by band (SkillCorner's own model 0.283, 0.894)"),
}


def parameters_for(provider: str) -> ParameterSet:
    """The provider's fitted parameter set, else the default one."""
    return PARAMETER_SETS.get(str(provider).lower(), PARAMETER_SETS["default"])


def _kinematic(players: list, points: np.ndarray, model: PassModel) -> np.ndarray:
    if not len(players) or not len(points):
        return np.full((len(players), len(points)), _INF)
    return np.array([player_travel_time(np.asarray(p["pos"], float), np.asarray(p.get("vel", [0.0, 0.0]), float), points,
                                        model.v_max, model.a_max, 0.0) for p in players])


def pass_arrays(passes: list[dict], model: PassModel, max_players: int = 11, max_points: int | None = None) -> PassArrays:
    """
    Travel times for a batch of passes. Each pass is ``{"ball": (x, y), "target": (x, y), "attackers": [...],
    "defenders": [...], "high": bool}``; players as ``{pos, vel}`` (``_build_defender_data``), the passer excluded.
    """
    lengths = [float(np.hypot(*(np.asarray(p["target"], float) - np.asarray(p["ball"], float)))) for p in passes]
    k_max = max_points or max(1, int(max(lengths, default=1.0) // model.step_m))
    n = len(passes)
    out = PassArrays(np.zeros((n, k_max)), np.zeros((n, k_max)), np.zeros((n, k_max)), np.full((n, max_players, k_max), _INF),
                     np.zeros(n), np.full((n, max_players), _INF), np.full((n, max_players), _INF), np.zeros(n, bool))
    for i, (p, length) in enumerate(zip(passes, lengths, strict=True)):
        ball, target = np.asarray(p["ball"], float), np.asarray(p["target"], float)
        out.high[i] = bool(p.get("high", False))
        out.t_target[i] = float(model.flight.time_to(length))
        att, dfn = p["attackers"][:max_players], p["defenders"][:max_players]
        out.tau_att[i, :len(att)] = _kinematic(att, target[None, :], model)[:, 0] if att else []
        out.tau_def[i, :len(dfn)] = _kinematic(dfn, target[None, :], model)[:, 0] if dfn else []
        k = min(int(length // model.step_m), k_max)
        if k < 1 or length < 1e-6:
            continue
        s = (np.arange(k) + 0.5) * (length / k)
        u = (target - ball) / length
        pts = ball + np.outer(s, u)
        edges = np.concatenate([[0.0], (np.arange(1, k) * (length / k)), [length]])
        te = model.flight.waypoint_times(length, edges)
        out.t_ball[i, :k] = model.flight.waypoint_times(length, s)
        out.dt_ball[i, :k] = np.diff(te)
        out.frac[i, :k] = s / length
        if dfn:
            out.tau_def_path[i, :len(dfn), :k] = _kinematic(dfn, pts, model)
    return out


def _arrived(t, tau, sigma):
    return 1.0 / (1.0 + np.exp(-np.clip(_K / sigma * (t - tau), -50, 50)))


def probability_from_arrays(a: PassArrays, model: PassModel) -> dict[str, np.ndarray]:
    """Completion probability, flight survival and the passing team's share at the target, per pass."""
    reach = ~a.high[:, None] | (a.frac <= model.high_reach) | (a.frac >= 1.0 - model.high_reach)
    f = _arrived(a.t_ball[:, None, :], model.rt_def + a.tau_def_path, model.sigma)      # (N, D, K)
    hazard = model.lam_flight * (f.sum(axis=1) * a.dt_ball * reach).sum(axis=1)        # (N,)
    survive = np.exp(-hazard)
    p_att, p_def = np.zeros(len(a.t_target)), np.zeros(len(a.t_target))
    for t in np.arange(0.0, model.horizon_s, model.dt_s):
        now = a.t_target + t
        fa = _arrived(now[:, None], model.rt_att + a.tau_att, model.sigma).sum(axis=1)
        fd = _arrived(now[:, None], model.rt_def + a.tau_def, model.sigma).sum(axis=1)
        free = np.clip(1.0 - p_att - p_def, 0.0, 1.0)
        da, dd = free * model.lam * fa * model.dt_s, free * model.lam * fd * model.dt_s
        scale = np.where(da + dd > free, free / np.maximum(da + dd, 1e-12), 1.0)  # never more than the free share
        p_att, p_def = p_att + da * scale, p_def + dd * scale
    total = p_att + p_def
    share = np.where(total > 1e-9, p_att / np.maximum(total, 1e-9), 0.5)
    return {"p_complete": survive * share, "p_survive": survive, "share_at_target": share}


def projected_target(ball_xy, receiver_xy, receiver_vel, flight: FlightModel = MEASURED_FLIGHT, rounds: int = 8) -> np.ndarray:
    """
    Where a receiver running at constant velocity will be when the ball arrives: the pass's target known at
    the moment of the pass (a ball played into a run), from positions and velocities only, never the outcome.
    """
    ball, rec, vel = np.asarray(ball_xy, float), np.asarray(receiver_xy, float), np.asarray(receiver_vel, float)
    target = rec.copy()
    for _ in range(rounds):
        target = rec + vel * float(flight.time_to(np.linalg.norm(target - ball)))
    return target


def pass_probability(ball_xy, target_xy, attackers: list, defenders: list, model: PassModel | None = None, high: bool = False) -> dict[str, float]:
    """One pass: ``attackers`` are the passing team without the passer, ``defenders`` the opponents (``{pos, vel}``)."""
    model = model or PassModel()
    p = {"ball": ball_xy, "target": target_xy, "attackers": attackers, "defenders": defenders, "high": high}
    n = max(len(attackers), len(defenders), 1)
    r = probability_from_arrays(pass_arrays([p], model, max_players=n), model)
    return {k: float(v[0]) for k, v in r.items()}
