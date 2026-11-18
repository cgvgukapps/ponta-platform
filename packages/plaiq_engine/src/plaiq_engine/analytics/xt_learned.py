"""
xt_learned.py — an expected-threat surface learned from events (a candidate beside the geometric V_xT).

Plaiq's threat surface (``hex_value.xT_surface``) is a geometric formula: the goal mouth's angle with a
distance decay and a progression bonus. This learns the surface from data instead, by the expected
threat method of Singh (2019): the pitch is split into ``nx`` × ``ny`` zones and, from a set of on-ball
actions (moves and shots), each zone's value is the chance of scoring from it within the possession:

    xT(z) = s(z) · g(z) + m(z) · Σ_z' T(z → z') · xT(z')

where s and m are the shares of actions in z that are shots and moves, g the goal rate of shots from z,
and T the share of moves from z that arrive, successfully, in z' (a failed move ends the possession,
value 0). Solved by iteration from xT = 0.

Coordinates are the attacking team's frame: depth ``x`` in metres from its own goal line (0 to
``length``) and width ``y`` in metres from the centre line (−width / 2 to +width / 2). Zones are
fractions of the pitch, so pitches of different sizes share one surface. A candidate metric: the
geometric surface, POV and decision quality are untouched until it is measured and agreed.
"""

from __future__ import annotations

import numpy as np

XT_LEARNED_VERSION = "0.1.0"
NX, NY = 16, 12


def zone(x, y, length: float, width: float, nx: int = NX, ny: int = NY) -> tuple[np.ndarray, np.ndarray]:
    """Zone indices for positions in the attacking frame (clipped to the pitch)."""
    ix = np.clip((np.asarray(x, float) / length * nx).astype(int), 0, nx - 1)
    iy = np.clip(((np.asarray(y, float) + width / 2) / width * ny).astype(int), 0, ny - 1)
    return ix, iy


def fit_xt(start_x, start_y, end_x, end_y, is_shot, success, goal, *, length: float = 105.0, width: float = 68.0,
           nx: int = NX, ny: int = NY, iterations: int = 200, tol: float = 1e-9) -> dict:
    """
    Learn the surface from actions. ``is_shot``: a shot (else a move); ``success``: a move that reached
    its end (ignored for shots); ``goal``: a shot that scored. Returns the surface and its parts.
    """
    is_shot, success, goal = (np.asarray(a, bool) for a in (is_shot, success, goal))
    sx, sy = zone(start_x, start_y, length, width, nx, ny)
    shots = np.zeros((nx, ny)); goals = np.zeros((nx, ny)); moves = np.zeros((nx, ny))
    np.add.at(shots, (sx[is_shot], sy[is_shot]), 1)
    np.add.at(goals, (sx[is_shot & goal], sy[is_shot & goal]), 1)
    np.add.at(moves, (sx[~is_shot], sy[~is_shot]), 1)
    ok = ~is_shot & success & np.isfinite(np.asarray(end_x, float)) & np.isfinite(np.asarray(end_y, float))
    ex, ey = zone(np.asarray(end_x, float)[ok], np.asarray(end_y, float)[ok], length, width, nx, ny)
    trans = np.zeros((nx * ny, nx * ny))
    np.add.at(trans, (sx[ok] * ny + sy[ok], ex * ny + ey), 1)
    total = shots + moves
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(total > 0, shots / total, 0.0)
        m = np.where(total > 0, moves / total, 0.0)
        g = np.where(shots > 0, goals / shots, 0.0)
        t = trans / np.where(moves.reshape(-1, 1) > 0, moves.reshape(-1, 1), 1.0)
    xt = np.zeros(nx * ny)
    sg, mf = (s * g).reshape(-1), m.reshape(-1)
    for _ in range(iterations):
        new = sg + mf * (t @ xt)
        if np.max(np.abs(new - xt)) < tol:
            xt = new
            break
        xt = new
    return {"xt": xt.reshape(nx, ny), "shot_share": s, "move_share": m, "goal_rate": g, "actions": total,
            "length": length, "width": width, "version": XT_LEARNED_VERSION}


def lookup(surface: dict, x, y, length: float, width: float) -> np.ndarray:
    """The learned threat at positions (attacking frame, metres) on a pitch of the given size."""
    nx, ny = surface["xt"].shape
    ix, iy = zone(x, y, length, width, nx, ny)
    return surface["xt"][ix, iy]
