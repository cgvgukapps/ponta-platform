"""
restarts.py — restarts found from player positions when there is no event feed.

Broadcast tracking often loses the ball at a restart: it is detected only once it is in the air.
Players stay tracked, and a restart has a shape: for a corner, one attacking player at a corner
flag while a crowd of both teams waits in and around that penalty area (the founder's proposal,
17 November 2026). ``corner_setups`` finds those moments from positions alone; the ball is not
used. ``throw_in_setups`` (18 November 2026) finds throw-ins: one player standing beyond a
touchline while play has slowed, with the ball near them.

Goal kicks are not found: on broadcast tracking about three in four have a gap in the tracking
(a replay or a close-up) in the 15 s before them, so the set-up is rarely seen
(``CONTEXT_STRUCTURE_REPORT.md``).

Positions are given in each team's own frame (``structure.team_frames``): depth in metres from
the team's own goal line, width with the team's left positive. Code computes; nothing here is
validated by analysts.
"""

from __future__ import annotations

import warnings

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


# ---------------------------------------------------------------------------
# Throw-ins (18 November 2026)
# ---------------------------------------------------------------------------

#: A thrower within this distance of a goal line is at a corner, not a throw-in.
THROW_IN_EDGE_M = 3.0
#: Gaps of up to this long in a thrower's run are bridged (tracking flickers at the touchline).
THROW_IN_BRIDGE_S = 1.0
#: The stillness and ball tests read the last this-many seconds of the set-up.
THROW_IN_LOOK_S = 2.0
#: A set-up belongs to a possession when it ends at most this long before the possession starts,
#: or at most this long after (the window the detector was measured with).
SETUP_BEFORE_S, SETUP_AFTER_S = 6.0, 3.0


def _bridged(cond: np.ndarray, period: np.ndarray, gap: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in _runs(cond, period):
        if out and s - out[-1][1] - 1 <= gap and period[s] == period[out[-1][1]]:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def median_speed(depths: list[np.ndarray], widths: list[np.ndarray], ts: np.ndarray, period: np.ndarray) -> np.ndarray:
    """Median speed (m/s) of all observed players at each grid point; NaN at a period's first point."""
    parts = []
    for d, w in zip(depths, widths, strict=True):
        v = np.hypot(np.diff(d, axis=0), np.diff(w, axis=0)) / np.diff(ts)[:, None]
        v = np.vstack([np.full((1, d.shape[1]), np.nan), v])
        v[np.r_[False, period[1:] != period[:-1]]] = np.nan
        parts.append(v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.hstack(parts), axis=1)


def throw_in_setups(team_depth: np.ndarray, team_width: np.ndarray, other_depth: np.ndarray, other_width: np.ndarray,
                    ball_depth: np.ndarray, ball_width: np.ndarray, speed: np.ndarray, ts: np.ndarray, period: np.ndarray,
                    length: float, width: float, *, out_m: float, hold_s: float, still_mps: float, ball_m: float,
                    hz: float = 5.0) -> list[tuple[float, float]]:
    """
    Throw-in set-ups for one team, from positions: spans of at least ``hold_s`` in which exactly one
    of its players stands at least ``out_m`` beyond a touchline (away from the corners) and no
    opponent does; in the set-up's last seconds play has slowed (median player speed at most
    ``still_mps``) and the ball is within ``ball_m`` of that player. Positions are in the team's
    own frame, the ball's too. Returns (start, end) times; the end is about when the ball is thrown.
    """
    with np.errstate(invalid="ignore"):
        def outside(d, w):
            return (np.abs(w) - width / 2 >= out_m) & (d >= THROW_IN_EDGE_M) & (d <= length - THROW_IN_EDGE_M)
        mine, theirs = outside(team_depth, team_width), outside(other_depth, other_width)
    cond = (mine.sum(axis=1) == 1) & (theirs.sum(axis=1) == 0)
    look = int(round(THROW_IN_LOOK_S * hz))
    out = []
    for s, e in _bridged(cond, period, int(round(THROW_IN_BRIDGE_S * hz))):
        if ts[e] - ts[s] < hold_s:
            continue
        k = max(s, e - look)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if not np.nanmean(speed[k:e + 1]) <= still_mps:
                continue
        near = False
        for i in range(k, e + 1):
            cols = np.where(mine[i])[0]
            if len(cols) and np.isfinite(ball_depth[i]) and np.hypot(team_depth[i, cols[0]] - ball_depth[i], team_width[i, cols[0]] - ball_width[i]) <= ball_m:
                near = True
                break
        if near:
            out.append((float(ts[s]), float(ts[e])))
    return out
