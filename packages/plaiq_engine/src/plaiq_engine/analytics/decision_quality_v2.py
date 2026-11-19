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
"""

from __future__ import annotations

import numpy as np

from plaiq_engine.analytics.option_split import option_components

DQ_V2_VERSION = "0.1.0"


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
