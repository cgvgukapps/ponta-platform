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
"""

from __future__ import annotations

import numpy as np

from plaiq_engine.analytics.option_split import option_components
from plaiq_engine.analytics.pass_option_value import (
    _distance_decay,
    _progressive_factor,
)
from plaiq_engine.analytics.pass_probability import (
    PassModel,
    pass_arrays,
    probability_from_arrays,
    projected_target,
)

#: 0.1.0 (19 November 2026): expected value at the pass, measured, not adopted. 0.2.0 (19 November 2026):
#: adds the two narrower candidates with the pass completion probability as the risk term. 1.0.0
#: (19 November 2026): projected adopted beside v1 by the founder, maths unchanged from 0.2.0.
DQ_V2_VERSION = "1.0.0"


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
    batch = [{"ball": ball, "target": t, "attackers": mates, "defenders": defenders, "high": False} for t in targets]
    n = max(1, len(mates), len(defenders))
    return targets, probability_from_arrays(pass_arrays(batch, model, max_players=n), model)["p_complete"]


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
