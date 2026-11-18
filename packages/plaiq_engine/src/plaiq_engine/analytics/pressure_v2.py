"""
pressure_v2.py — directional, count-aware pressure on a player (a new metric beside 1v1 and zonal pressure).

Plaiq's 1v1 pressure (``hex_analytics.compute_pressure``) reads only the nearest opponent's distance.
FIFA's EFI pressure on the ball uses the defender's distance, their angle to the player (in front
counts more than behind) and how many defenders press (``ponta-blueprint/fifa-alignment.md``). This
follows the pressure zone of Andrienko et al. (2017, "Visual analytics of pressure in football",
Data Mining and Knowledge Discovery 31): each defender's pressure falls with distance inside a zone
whose reach depends on the angle,

    D(θ) = D_back + (D_front − D_back) · (1 + cos θ) / 2
    p_i  = (1 − d_i / D(θ_i)) ** q      when d_i < D(θ_i), else 0

where θ is the angle between the player's direction of attack and the line from the player to the
defender (θ = 0: the defender is goal-side, in front; θ = π: behind). Pressure on the player is the
sum over defenders (count-aware) or, for comparison, the largest single contribution.

Published defaults (Andrienko et al.): reach 9 m in front, 3 m behind, q = 1.75. A new metric:
1v1 pressure, zonal pressure, the hex value model and decision quality are untouched.
"""

from __future__ import annotations

import numpy as np

PRESSURE_V2_VERSION = "0.1.0"
D_FRONT_M, D_BACK_M, EXPONENT = 9.0, 3.0, 1.75


def defender_pressure(player_xy, attack_dir, defenders_xy, *, d_front: float = D_FRONT_M, d_back: float = D_BACK_M,
                      q: float = EXPONENT) -> np.ndarray:
    """Each defender's pressure on the player (0 outside their zone). ``attack_dir`` need not be unit length."""
    p = np.asarray(player_xy, float)
    a = np.asarray(attack_dir, float)
    a = a / (np.linalg.norm(a) or 1.0)
    dxy = np.asarray(defenders_xy, float).reshape(-1, 2) - p
    ok = np.isfinite(dxy).all(axis=1)
    out = np.zeros(len(dxy))
    if not ok.any():
        return out
    d = np.hypot(dxy[ok, 0], dxy[ok, 1])
    cos = np.where(d > 0, (dxy[ok] @ a) / np.where(d > 0, d, 1.0), 1.0)
    reach = d_back + (d_front - d_back) * (1.0 + cos) / 2.0
    out[ok] = np.clip(1.0 - d / reach, 0.0, None) ** q  # zero outside the zone
    return out


def pressure(player_xy, attack_dir, defenders_xy, *, aggregate: str = "sum", **kw) -> float:
    """Pressure on the player: the sum of the defenders' pressures (count-aware) or the largest one."""
    c = defender_pressure(player_xy, attack_dir, defenders_xy, **kw)
    return float(c.sum() if aggregate == "sum" else (c.max() if len(c) else 0.0))
