"""
restarts.py — restarts found from player positions when there is no event feed.

Broadcast tracking often loses the ball at a restart: it is detected only once it is in the air.
Players stay tracked, and a restart has a shape: for a corner, one attacking player at a corner
flag while a crowd of both teams waits in and around that penalty area (the founder's proposal,
17 November 2026). ``corner_setups`` finds those moments from positions alone; the ball is not
used.

Positions are given in each team's own frame (``structure.team_frames``): depth in metres from
the team's own goal line, width with the team's left positive. Code computes; nothing here is
validated by analysts.
"""

from __future__ import annotations

import numpy as np

#: Penalty area: 16.5 m deep, 40.32 m wide.
BOX_DEPTH_M = 16.5
BOX_HALF_WIDTH_M = 20.16


def _runs(cond: np.ndarray, period: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index runs where ``cond`` holds, split at period changes."""
    out: list[tuple[int, int]] = []
    start = None
    for k in range(len(cond)):
        if cond[k] and start is not None and period[k] != period[k - 1]:
            out.append((start, k - 1))
            start = k
        elif cond[k] and start is None:
            start = k
        elif not cond[k] and start is not None:
            out.append((start, k - 1))
            start = None
    if start is not None:
        out.append((start, len(cond) - 1))
    return out


def corner_counts(att_depth: np.ndarray, att_width: np.ndarray, def_depth: np.ndarray, def_width: np.ndarray,
                  length: float, width: float, radius: float, box_margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per grid point, for one attacking team at the end it attacks: attackers within ``radius`` of a
    corner flag, defenders within ``radius`` of one, and players of both teams in or within
    ``box_margin`` of the penalty area. Arrays are (grid points, players); NaN means not observed.
    """
    with np.errstate(invalid="ignore"):
        a_flag = np.hypot(length - att_depth, width / 2 - np.abs(att_width)) <= radius
        d_flag = np.hypot(def_depth, width / 2 - np.abs(def_width)) <= radius
        a_box = (att_depth >= length - BOX_DEPTH_M - box_margin) & (np.abs(att_width) <= BOX_HALF_WIDTH_M + box_margin)
        d_box = (def_depth <= BOX_DEPTH_M + box_margin) & (np.abs(def_width) <= BOX_HALF_WIDTH_M + box_margin)
    return a_flag.sum(axis=1), d_flag.sum(axis=1), a_box.sum(axis=1) + d_box.sum(axis=1)


def corner_setups(att_depth: np.ndarray, att_width: np.ndarray, def_depth: np.ndarray, def_width: np.ndarray,
                  ts: np.ndarray, period: np.ndarray, length: float, width: float, *,
                  radius: float, crowd_min: int, hold_s: float, box_margin: float = 2.0) -> list[tuple[float, float]]:
    """
    Corner set-ups for one attacking team: spans of at least ``hold_s`` in which exactly one
    attacker (and no defender) stands within ``radius`` of a corner flag at the end it attacks,
    while at least ``crowd_min`` players of both teams are in or near that penalty area.
    Returns (start, end) times; the end of a set-up is about when the corner is taken.
    """
    a_flag, d_flag, crowd = corner_counts(att_depth, att_width, def_depth, def_width, length, width, radius, box_margin)
    cond = (a_flag == 1) & (d_flag == 0) & (crowd >= crowd_min)
    return [(float(ts[s]), float(ts[e])) for s, e in _runs(cond, period) if ts[e] - ts[s] >= hold_s]
