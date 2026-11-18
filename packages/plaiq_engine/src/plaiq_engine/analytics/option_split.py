"""
option_split.py — pass option value split into availability and value (a new metric beside POV).

Pass option value (``pass_option_value.player_pov``, unchanged) multiplies two different things:

    POV = xT_receiver × V_ctrl × (1 − interception_risk) × progressive_factor × (1 − distance_decay)

- **availability** ``= V_ctrl × (1 − interception_risk)``: is the pass on? Control of the ball at the
  receiver (targeted PCZ) and the chance it is not cut out on the way;
- **value** ``= xT_receiver × progressive_factor × (1 − distance_decay)``: what the pass is worth if
  it arrives.

So ``POV = availability × value`` (before POV's clip to [0, 2]). Measured against references
(``ponta-platform/PLAIQ_REFERENCES_REPORT.md``), the two halves behave differently: availability
predicts completion (AUC 0.78 on SkillCorner's passes) while POV hardly tracks a provider's threat.
Keeping them apart lets each be judged, and replaced, on its own (``ponta-blueprint/plaiq-review.md``,
step 3). ``value_gain`` (the change in threat, ``xT_receiver − xT_ball``) is offered beside POV's value
as a candidate. A new metric: POV and decision quality are untouched (engine governance).
"""

from __future__ import annotations

import numpy as np

from plaiq_engine.analytics.pass_option_value import (
    _MAX_PASS_DISTANCE,
    _distance_decay,
    _progressive_factor,
    targeted_pcz,
)

OPTION_SPLIT_VERSION = "0.1.0"


def option_components(
    ball_xy: np.ndarray,
    receiver_xy: np.ndarray,
    xT_ball: float,
    xT_receiver: float,
    defender_data: list,
) -> dict[str, float]:
    """
    Availability, value and their parts for one passing option, from the same components POV uses.
    Beyond ``_MAX_PASS_DISTANCE`` the option is unavailable (POV scores it 0).
    """
    d = float(np.linalg.norm(np.asarray(receiver_xy, float) - np.asarray(ball_xy, float)))
    prog = _progressive_factor(xT_ball, xT_receiver)
    decay = _distance_decay(np.asarray(ball_xy, float), np.asarray(receiver_xy, float))
    value = float(xT_receiver * prog * (1.0 - decay))
    if d > _MAX_PASS_DISTANCE:
        risk, ctrl = 1.0, 0.0
    else:
        risk, ctrl = targeted_pcz(np.asarray(ball_xy, float), np.asarray(receiver_xy, float), defender_data)
    availability = float(ctrl * (1.0 - risk))
    return {
        "availability": availability, "control": float(ctrl), "interception_risk": float(risk),
        "value": value, "value_gain": float(xT_receiver - xT_ball), "distance_m": d,
        "pov": float(np.clip(availability * value, 0.0, 2.0)),
    }
