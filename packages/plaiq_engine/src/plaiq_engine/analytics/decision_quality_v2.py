"""
decision_quality_v2.py — decision quality as expected value at the moment of the pass (a candidate beside v1).

Decision quality v1 (``pass_option_value``) judges a completed pass by its receiver's option value at
the end of the pass, and a failed pass by the option value at an inferred meeting point. It judges the
intention, not the outcome, which is right, but measured on 12,012 SkillCorner passes it rated 89% of
failed passes optimal against 58% of completed ones, and it never weighs the risk the passer took
(``ponta-platform/PLAIQ_REFERENCES_REPORT.md``).

v2 keeps the intention and adds the risk known at the time, with one formula for every pass:

- **the moment:** the frame the ball is played; the outcome is never an input;
- **the chosen option:** where the ball was played to (the reception point, or for a failed pass the
  intended target);
- **the alternatives:** every team-mate (not the passer) where they stand at that moment;
- **each option's expected value:** ``availability × value`` (``option_split.option_components``:
  the chance the pass arrives, and what it is worth if it does);
- **the score:** the chosen option's expected value over the best expected value on offer, chosen
  included, so 1.0 means the best option was chosen.

The unified bands (poor below 0.55, suboptimal to 0.80, good to 0.95, optimal above) are kept as they
are, untuned. A candidate metric: v1, POV and every persisted score are untouched until v2 is measured
and agreed (engine governance).

**0.1.0 measured, not adopted** (19 November 2026): it became mostly a measure of risk and lost v1's link
to chances, because it compared the destination with where team-mates stood, so a ball into a run looked
like a ball into space. **0.2.0 adds two narrower candidates** whose risk term is the adopted pass
completion probability (``pass_probability``, a calibrated probability fitted per provider) instead of
targeted pitch control's availability:

- ``decision_quality_risk`` (**v1 with risk**): v1's score times the chance the chosen pass is completed.
  v1's numerator is a value already realised (the receiver's option value once the ball has arrived)
  while its denominator weighs every alternative by its risk; the factor puts the chosen pass on the same
  expected-value footing and leaves v1's end-of-pass credit (a ball played into a run) as it is;
- ``decision_quality_projected`` (**projected**): each team-mate's option is a pass into their run
  (``projected_target``), worth its completion probability times its value there
  (``option_value``); the chosen option's expected value over the best one's. One formula for completed
  and failed passes, the outcome never an input.

``decision_quality_v2`` (0.1.0) is unchanged.

**1.0.0 adopted beside v1** (founder decision, 19 November 2026): **projected** ships as decision quality
v2 (``MatchAnalysis.decision_quality_v2``; semantic events schema 1.3), labelled estimated. Its maths is
0.2.0's, unchanged. It runs **0 to 1** and has **no bands** yet: v1's bands (cut on v1's 0 to 1.5 scale) are
not applied to it until bands for v2 are measured and agreed. v1 stays the default decision quality.
``decision_quality_v2`` (0.1.0) and ``decision_quality_risk`` (0.2.0) stay here as measured, not adopted.

**Hex-cell candidates** (19 November 2026, measured, not adopted; the shipped score stays 1.0.0): the same
expected value, computed on Plaiq's hexagonal grid. Each team-mate's option is **the best ball into their
run**: the cells within ``rings`` of the cell their run reaches when the ball arrives (``run_cells``), each
worth the completion probability of a pass to the cell's centre times the cell's value; the option is its best
cell. The chosen option is scored the same way, so placement within the cluster (execution) is not charged to
the choice. Ring 0 is the run's single cell (the grid alone); rings 1 and 2 hold 7 and 19 cells per
team-mate. Only cells a run is heading into are evaluated, never every reachable cell (about a fifth of the
pitch on a typical pass), so passes into space no run is heading for are still not options.

**1.1.0 adopted** (founder decision, 19 November 2026): **ring 1** replaces the point-based 1.0.0 as decision
quality v2, beside v1, labelled estimated (``MatchAnalysis.decision_quality_v2``). **Bands agreed the same
day**, with descriptive names that say where the choice sits against the best option rather than judge the
player (``dq_v2_band``): best option (0.95 and above), near best (0.60 to 0.94), below best (0.45 to 0.59),
well below best (under 0.45). They are not version 1's bands and never share its names. ``decision_quality_projected``
(1.0.0) stays as the record of the point-based score.
"""

from __future__ import annotations

import numpy as np

from plaiq_engine.analytics.hex_analytics import cartesian_to_hex
from plaiq_engine.analytics.option_split import option_components
from plaiq_engine.analytics.pass_option_value import (
    _distance_decay,
    _progressive_factor,
)
from plaiq_engine.analytics.pass_probability import (
    _INF,
    PassArrays,
    PassModel,
    _kinematic,
    probability_from_arrays,
    projected_target,
)

#: 0.1.0 (19 November 2026): expected value at the pass, measured, not adopted. 0.2.0 (19 November 2026):
#: adds the two narrower candidates with the pass completion probability as the risk term. 1.0.0
#: (19 November 2026): projected adopted beside v1 by the founder, maths unchanged from 0.2.0. 1.1.0
#: (19 November 2026): the hex-cell candidate at ring 1 adopted in its place, with bands.
DQ_V2_VERSION = "1.1.0"

#: The rings of cells around each run that the adopted score evaluates (7 cells per team-mate).
DQ_V2_RINGS = 1

#: Gain version 2 (the one-model Gain, ``decision_gain``), adopted beside version 1 by the founder 20 November 2026.
#: Version 1 (``pass_option_value``) stays the default Gain. Same cut-offs and band names as version 1.
GAIN_V2_VERSION = "2.0.0"

#: Bands agreed by the founder 19 November 2026 (measured: ``ponta-platform/PLAIQ_REFERENCES_REPORT.md``,
#: "Candidate bands" and "on Plaiq's hex grid"). Descriptive names, distinct from version 1's.
DQ_V2_BANDS: tuple[tuple[float, str], ...] = ((0.95, "best_option"), (0.60, "near_best"), (0.45, "below_best"))
DQ_V2_LOWEST_BAND = "well_below_best"


def dq_v2_band(score: float | None) -> str | None:
    """The agreed band of a decision quality v2 score (0 to 1); None without a score."""
    if score is None or not np.isfinite(score):
        return None
    for cut, band in DQ_V2_BANDS:
        if score >= cut:
            return band
    return DQ_V2_LOWEST_BAND


def decision_quality_v2(
    ball_xy: np.ndarray,
    target_xy: np.ndarray,
    xT_ball: float,
    xT_target: float,
    teammates_xy: list,
    xT_teammates: list,
    defender_data: list,
    *,
    value_key: str = "value",
) -> dict:
    """
    Decision quality of one pass at the moment it is played. ``teammates_xy`` excludes the passer;
    ``value_key`` is ``value`` (POV's value part) or ``value_xt`` (the threat at the option alone).
    """
    def ev(xy, xt):
        c = option_components(ball_xy, np.asarray(xy, float), xT_ball, xt, defender_data)
        v = c["value"] if value_key == "value" else xt
        return c["availability"] * v, c

    chosen, cc = ev(target_xy, xT_target)
    alts = [ev(xy, xt)[0] for xy, xt in zip(teammates_xy, xT_teammates, strict=True)]
    best = max([chosen, *alts]) if alts else chosen
    dq = float(chosen / best) if best > 0 else np.nan
    return {"dq": dq, "ev_chosen": float(chosen), "ev_best": float(best), "availability_chosen": cc["availability"],
            "value_chosen": cc["value"], "options": len(alts)}


def decision_quality_risk(v1_dq: float, p_complete: float) -> float:
    """v1 with risk: v1's decision quality times the chosen pass's completion probability (0 to 1.5)."""
    if v1_dq is None or p_complete is None or not (np.isfinite(v1_dq) and np.isfinite(p_complete)):
        return np.nan
    return float(np.clip(v1_dq * p_complete, 0.0, 1.5))


def option_value(ball_xy, target_xy, xT_ball: float, xT_target: float) -> float:
    """POV's value part for a pass to ``target_xy``: what it is worth if it arrives (``option_split``)."""
    ball, target = np.asarray(ball_xy, float), np.asarray(target_xy, float)
    return float(xT_target * _progressive_factor(xT_ball, xT_target) * (1.0 - _distance_decay(ball, target)))


def option_probabilities(ball_xy, mates: list[dict], defenders: list[dict], model: PassModel) -> tuple[np.ndarray, np.ndarray]:
    """
    Every team-mate's pass at the moment of the pass: the target projected along their run and the chance
    the pass is completed. ``mates`` are the passing team without the passer (``{pos, vel}``); each pass is
    computed as ``MatchAnalysis.pass_completion`` computes the chosen one (the same attackers and defenders).
    """
    ball = np.asarray(ball_xy, float)
    if not mates:
        return np.zeros((0, 2)), np.zeros(0)
    targets = np.array([projected_target(ball, m["pos"], m.get("vel", np.zeros(2)), model.flight) for m in mates])
    # One moment, the same players: identical arrays to pass_arrays, each player's travel times computed once.
    return targets, probability_from_arrays(pass_arrays_one_moment(ball, targets, mates, defenders, model), model)["p_complete"]


def decision_quality_projected(p_complete: np.ndarray, value: np.ndarray, chosen: int) -> dict:
    """
    Projected: the chosen option's expected value (completion probability × value) over the best option's,
    0 to 1, where 1.0 means the best pass on offer was chosen.
    """
    ev = np.asarray(p_complete, float) * np.asarray(value, float)
    if not len(ev) or not (0 <= chosen < len(ev)):
        return {"dq": np.nan, "ev_chosen": np.nan, "ev_best": np.nan, "options": int(len(ev))}
    best = float(ev.max())
    return {"dq": float(ev[chosen] / best) if best > 0 else np.nan, "ev_chosen": float(ev[chosen]), "ev_best": best,
            "best_is_chosen": bool(ev[chosen] >= best - 1e-12), "options": int(len(ev))}


# -- hex-cell candidates (measured, not adopted) ---------------------------------------------------------------

def ring_distance(q: np.ndarray, r: np.ndarray, q0: int, r0: int) -> np.ndarray:
    """Axial ring distance from cell ``(q0, r0)``."""
    dq, dr = np.asarray(q) - q0, np.asarray(r) - r0
    return (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) // 2


def run_cells(target_xy, hex_centers: np.ndarray, q_all: np.ndarray, r_all: np.ndarray, R: float, rings: int) -> np.ndarray:
    """
    Indices of the grid cells within ``rings`` of the cell containing ``target_xy`` (a run's point at the ball's
    arrival). A run that ends off the pitch is centred on the nearest cell on it first, so every option is a
    full ring clipped only at the pitch's edge (fixed 19 November 2026: before, such a run got a partial ring
    or a single cell depending on how far off the pitch it ended).
    """
    q0, r0 = (int(np.asarray(v).ravel()[0]) for v in cartesian_to_hex(np.array([target_xy[0]]), np.array([target_xy[1]]), R))
    if not ((q_all == q0) & (r_all == r0)).any():
        c = int(np.argmin(np.linalg.norm(hex_centers - np.asarray(target_xy, float), axis=1)))
        q0, r0 = int(q_all[c]), int(r_all[c])
    return np.where(ring_distance(q_all, r_all, q0, r0) <= rings)[0]


def option_cells_probabilities(ball_xy, mates: list[dict], defenders: list[dict], model: PassModel, hex_centers: np.ndarray,
                               q_all: np.ndarray, r_all: np.ndarray, R: float, rings: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    For every team-mate: the cells around where their run reaches when the ball arrives, and the completion
    probability of a pass to each cell's centre (one batch; the same attackers and defenders as
    ``option_probabilities``).
    """
    ball = np.asarray(ball_xy, float)
    cells = [run_cells(projected_target(ball, m["pos"], m.get("vel", np.zeros(2)), model.flight), hex_centers, q_all, r_all, R, rings)
             for m in mates]
    if not cells:
        return [], []
    targets = hex_centers[np.concatenate(cells)]
    p = probability_from_arrays(pass_arrays_one_moment(ball, targets, mates, defenders, model), model)["p_complete"]
    split = np.cumsum([len(c) for c in cells])[:-1]
    return cells, np.split(np.asarray(p, float), split)


def pass_arrays_one_moment(ball_xy, targets: np.ndarray, attackers: list, defenders: list, model: PassModel) -> PassArrays:
    """
    ``pass_arrays`` for many passes from one moment (one ball, the same players): identical arrays, but each
    player's travel times are computed once over every target and waypoint instead of once per pass.
    """
    ball, targets = np.asarray(ball_xy, float), np.asarray(targets, float).reshape(-1, 2)
    max_players = max(1, len(attackers), len(defenders))
    lengths = np.hypot(*(targets - ball).T)
    k_max = max(1, int(lengths.max(initial=1.0) // model.step_m))
    n = len(targets)
    out = PassArrays(np.zeros((n, k_max)), np.zeros((n, k_max)), np.zeros((n, k_max)), np.full((n, max_players, k_max), _INF),
                     np.zeros(n), np.full((n, max_players), _INF), np.full((n, max_players), _INF), np.zeros(n, bool))
    paths = []
    for i, (target, length) in enumerate(zip(targets, lengths, strict=True)):
        out.t_target[i] = float(model.flight.time_to(float(length)))
        k = min(int(length // model.step_m), k_max)
        if k < 1 or length < 1e-6:
            paths.append(np.zeros((0, 2)))
            continue
        s = (np.arange(k) + 0.5) * (length / k)
        edges = np.concatenate([[0.0], (np.arange(1, k) * (length / k)), [length]])
        out.t_ball[i, :k] = model.flight.waypoint_times(float(length), s)
        out.dt_ball[i, :k] = np.diff(model.flight.waypoint_times(float(length), edges))
        out.frac[i, :k] = s / length
        paths.append(ball + np.outer(s, (target - ball) / length))
    if attackers:
        out.tau_att[:, :len(attackers)] = _kinematic(attackers, targets, model).T
    if defenders:
        out.tau_def[:, :len(defenders)] = _kinematic(defenders, targets, model).T
        sizes = [len(x) for x in paths]
        if sum(sizes):
            tau = _kinematic(defenders, np.concatenate(paths), model)
            for i, (lo, k) in enumerate(zip(np.cumsum([0, *sizes[:-1]]), sizes, strict=True)):
                if k:
                    out.tau_def_path[i, :len(defenders), :k] = tau[:, lo:lo + k]
    return out


def decision_quality_cells(p_cells: list[np.ndarray], value_cells: list[np.ndarray], chosen: int) -> dict:
    """
    Each option's expected value is its best cell (probability × value); the score is the chosen option's over
    the best option's, 0 to 1. ``best_cell`` is each option's best cell (a position in its list), ``best_option``
    the option with the highest expected value.
    """
    evs = [np.asarray(p, float) * np.asarray(v, float) for p, v in zip(p_cells, value_cells, strict=True)]
    ev = np.array([float(e.max()) if len(e) else 0.0 for e in evs])
    best_cell = [int(np.argmax(e)) if len(e) else -1 for e in evs]
    if not len(ev) or not (0 <= chosen < len(ev)):
        return {"dq": np.nan, "ev_chosen": np.nan, "ev_best": np.nan, "options": int(len(ev)), "best_cell": best_cell}
    best = float(ev.max())
    return {"dq": float(ev[chosen] / best) if best > 0 else np.nan, "ev_chosen": float(ev[chosen]), "ev_best": best,
            "best_is_chosen": bool(ev[chosen] >= best - 1e-12), "options": int(len(ev)), "best_option": int(np.argmax(ev)),
            "best_cell": best_cell}


# -- one model, two outputs: Gain from Choice's own parts (a candidate, measured, not adopted) ------------------

def decision_gain(realised_value: float, ev_best: float) -> float:
    """
    **Gain** in the one-model candidate: the value the pass realised over the best expected value on offer when
    it was played (Choice's denominator, ``ev_best``), clipped 0 to 1.5 as version 1 is. The realised value is
    ``option_value`` at the cell where the ball arrived, and 0 for a failed pass: Gain is the outcome, Choice
    the decision, from one option set and one value surface. Gain / Choice is what the execution and the run
    added to the choice. **Adopted beside version 1 as Gain version 2** (2.0.0, 20 November 2026,
    ``MatchAnalysis.gain_v2``); version 1 stays the default Gain.
    """
    if realised_value is None or ev_best is None or not (np.isfinite(realised_value) and np.isfinite(ev_best)) or ev_best <= 0:
        return np.nan
    return float(np.clip(realised_value / ev_best, 0.0, 1.5))
