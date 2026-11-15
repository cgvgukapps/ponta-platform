"""
statsbomb/sb_dq.py — Decision Quality (DQ) for StatsBomb events.

Implements two DQ metrics:

Pass DQ
-------
  DQ = POV(actual_receiver, end_snapshot)
       / max(POV(all_teammates, start_snapshot))

  Coverage: ~688 pass chains per 90-min match where both start and end
  freeze-frames are available via the stitch graph.

Carry DQ
--------
  Simplified ring-alternative model (no tracking velocities):

    DQ = xT(actual_end_hex) / max(xT(ring_alternative_hexes))

  The ring of alternatives is built from the hex at carry Start X/Y, with
  radius equal to the ring the actual End hex lies on.  Defenders in ff_start
  are used to attenuate xT via pitch control.

  Coverage: 845 carries/match with both coords + FF.

Both functions return dict with keys: dq, confidence.  Additional keys vary
per function (see docstrings).

Event Player Row Builder
------------------------
  sb_event_player_rows() builds a per-player summary table for an event,
  mirroring §9.2 of the analytics spec.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from plaiq_engine.statsbomb.sb_pov_adapter import (
    sb_all_teammates,
    sb_defender_data,
    _nearest_hex,
)

# Minimum non-zero DQ denominator to avoid div-by-zero
_DQ_FLOOR = 0.001

# Carry ring tolerance: alternatives within ±1 ring of the actual end ring
_RING_TOL = 1


def sb_pass_dq(
    event: dict | pd.Series,
    next_event: Optional[dict | pd.Series],
    ff_start: Optional[dict],
    ff_end: Optional[dict],
    xT_home: np.ndarray,
    xT_away: np.ndarray,
    hex_centers: np.ndarray,
    R: float,
) -> dict:
    """
    Compute Pass Decision Quality for a StatsBomb PASS event.

    DQ = POV(actual_receiver_at_end) / max(POV(all_teammates_at_start))

    Parameters
    ----------
    event       : current PASS event
    next_event  : next stitched event (provides receiver start position)
    ff_start    : freeze frame at the PASS event (decision snapshot)
    ff_end      : freeze frame at the next stitched event (end snapshot),
                  or None — falls back to ff_start
    xT_home, xT_away : xT surfaces per hex
    hex_centers : (N, 2) hex centre array
    R           : hex circumradius

    Returns
    -------
    dict with keys:
      dq             float   — Pass DQ [0, 1+]; NaN if not computable
      pov_receiver   float   — POV of the actual receiver
      max_pov_start  float   — max POV at pass decision time
      confidence     str     — "full" | "partial" | "none"
      reason         str     — human-readable explanation
    """
    def _fail(reason: str) -> dict:
        return dict(dq=float("nan"), pov_receiver=float("nan"),
                    max_pov_start=float("nan"), confidence="none",
                    reason=reason)

    if ff_start is None:
        return _fail("no start FF")

    start_players = ff_start.get("players", [])
    if not start_players:
        return _fail("empty start FF")

    xT_surfaces = (xT_home, xT_away)
    ball_xy = _event_ball_start(event)
    if ball_xy is None:
        return _fail("no ball location")

    # max POV at start
    try:
        all_tm_start = sb_all_teammates(
            start_players, ball_xy, xT_surfaces, hex_centers
        )
    except Exception as exc:
        return _fail(f"sb_all_teammates error: {exc}")

    if not all_tm_start:
        return _fail("no visible teammates")

    max_pov_start = max((t["pov"] for t in all_tm_start), default=0.0)

    # Receiver position
    recv_xy = _receiver_xy(event, next_event)
    if recv_xy is None:
        return _fail("no receiver position")

    # POV for actual receiver — use ff_end if available, else ff_start
    ff_for_recv = ff_end if ff_end is not None else ff_start
    recv_players = ff_for_recv.get("players", [])
    defender_data = sb_defender_data(recv_players, "opponent")

    recv_np = np.array([float(recv_xy[0]), float(recv_xy[1])])
    ball_np = np.array([float(ball_xy[0]), float(ball_xy[1])])

    hidx = _nearest_hex(ball_np, hex_centers)
    ridx = _nearest_hex(recv_np, hex_centers)
    xT_ball = float(xT_home[hidx]) if hidx is not None else 0.0
    xT_recv = float(xT_home[ridx]) if ridx is not None else 0.0

    try:
        from plaiq_engine.analytics.pass_option_value import player_pov
        pov_receiver = float(player_pov(
            ball_np, recv_np, xT_ball, xT_recv, defender_data
        ))
    except Exception as exc:
        return _fail(f"player_pov error: {exc}")

    denom = max(float(max_pov_start), _DQ_FLOOR)
    dq = float(np.clip(pov_receiver / denom, 0.0, 2.0))

    confidence = "full" if ff_end is not None else "partial"

    return {
        "dq":            dq,
        "pov_receiver":  pov_receiver,
        "max_pov_start": float(max_pov_start),
        "confidence":    confidence,
        "reason":        "ok",
    }


def sb_carry_dq(
    event: dict | pd.Series,
    ff_start: Optional[dict],
    xT_home: np.ndarray,
    xT_away: np.ndarray,
    hex_centers: np.ndarray,
    R: float,
) -> dict:
    """
    Compute Carry Decision Quality for a StatsBomb CARRY event.

    DQ = xT(actual_end_hex) / max(xT(ring_alternative_hexes_at_start))

    Defenses in ff_start attenuate xT via the pitch-control surface but
    velocities are assumed zero (no tracking data).

    Parameters
    ----------
    event       : CARRY event with "Start X", "Start Y", "End X", "End Y"
    ff_start    : freeze frame at the carry event
    xT_home, xT_away, hex_centers, R : grid/surface args

    Returns
    -------
    dict with keys:
      dq           float   — Carry DQ [0, 1.5]; NaN if not computable
      v_actual     float   — xT of actual end hex
      v_best       float   — max xT of ring alternatives
      n_alts       int     — number of ring alternatives
      confidence   str
      reason       str
    """
    def _fail(reason: str) -> dict:  # noqa: E301
        return dict(dq=float("nan"), v_actual=float("nan"),
                    v_best=float("nan"), n_alts=0,
                    confidence="none", reason=reason)

    start_xy = _event_coords(event, "Start X", "Start Y")
    end_xy   = _event_coords(event, "End X",   "End Y")
    if start_xy is None or end_xy is None:
        return _fail("missing carry coords")

    start_np = np.array([float(start_xy[0]), float(start_xy[1])])
    end_np   = np.array([float(end_xy[0]),   float(end_xy[1])])

    if np.allclose(start_np, end_np, atol=0.01):
        return _fail("ring-0 carry (no movement)")

    # Axial hex indices for start and end
    tree = cKDTree(hex_centers)
    q_all, r_all = _axial_coords(hex_centers, R)

    end_idx   = int(tree.query(end_np)[1])
    start_idx = int(tree.query(start_np)[1])

    end_q, end_r   = q_all[end_idx],   r_all[end_idx]
    start_q, start_r = q_all[start_idx], r_all[start_idx]

    dq_val = int(end_q - start_q)
    dr_val = int(end_r - start_r)
    chosen_ring = int((abs(dq_val) + abs(dr_val) + abs(dq_val + dr_val)) // 2)

    if chosen_ring < 1:
        return _fail("ring-0 carry (same hex)")

    # Ring alternatives
    ring_min = max(1, chosen_ring - _RING_TOL)
    ring_max = chosen_ring + _RING_TOL

    dq_arr = q_all - start_q
    dr_arr = r_all - start_r
    ring_dist = (np.abs(dq_arr) + np.abs(dr_arr) + np.abs(dq_arr + dr_arr)) // 2
    alt_mask = (ring_dist >= ring_min) & (ring_dist <= ring_max)
    alt_idxs = np.where(alt_mask)[0]

    if alt_idxs.size == 0:
        alt_idxs = np.array([end_idx], dtype=int)

    # xT at each alternative (nearest-hex lookup)
    xT_alts = _xT_at_positions(hex_centers[alt_idxs], xT_home, hex_centers)
    v_best   = float(np.max(xT_alts)) if xT_alts.size else 0.0
    v_actual = float(_xT_at_positions(hex_centers[[end_idx]], xT_home, hex_centers)[0])

    v_best_floored = max(v_best, _DQ_FLOOR)
    dq_val = float(np.clip(v_actual / v_best_floored, 0.0, 1.5))

    return {
        "dq":          dq_val,
        "v_actual":    v_actual,
        "v_best":      v_best,
        "n_alts":      int(alt_idxs.size),
        "confidence":  "full" if ff_start is not None else "none",
        "reason":      "ok",
    }


def sb_event_player_rows(
    event: dict | pd.Series,
    ff_players_identified: list[dict],
    ball_xy: list[float] | np.ndarray,
    xT_home: np.ndarray,
    xT_away: np.ndarray,
    hex_centers: np.ndarray,
) -> pd.DataFrame:
    """
    Build a per-player summary DataFrame for an event, mirroring §9.2.

    One row per identified player visible in the freeze frame.

    Columns
    -------
    name         str    player name (or "Unknown")
    team         str    "Home" / "Away"
    role         str    "Actor" / "Teammate" / "Opponent"
    xT_pos       float  xT at player's current location
    pressure     float  distance to nearest opponent (proxy)
    pov          float  POV if teammate, else NaN
    confidence   str    "strong" / "medium" / "weak" / None
    """
    ball_np = np.array([float(ball_xy[0]), float(ball_xy[1])])

    defenders_opp = sb_defender_data(ff_players_identified, "opponent")

    rows = []
    for p in ff_players_identified:
        loc  = p.get("location", [0.0, 0.0])
        p_np = np.array([float(loc[0]), float(loc[1])])

        hidx  = _nearest_hex(p_np, hex_centers)
        xT_p  = float(xT_home[hidx]) if hidx is not None else 0.0

        # Pressure proxy: distance to nearest opponent
        opp_positions = [np.array([float(o["pos"][0]), float(o["pos"][1])])
                         for o in defenders_opp] if not p.get("actor") else []
        if opp_positions:
            pressure = float(min(
                np.linalg.norm(p_np - op) for op in opp_positions
            ))
        else:
            pressure = float("nan")

        # POV only for visible teammates
        pov_val = float("nan")
        if p.get("teammate") and not p.get("actor"):
            ridx = _nearest_hex(p_np, hex_centers)
            hidx_ball = _nearest_hex(ball_np, hex_centers)
            xT_ball_v = float(xT_home[hidx_ball]) if hidx_ball is not None else 0.0
            xT_recv_v = float(xT_home[ridx]) if ridx is not None else 0.0
            try:
                from plaiq_engine.analytics.pass_option_value import player_pov
                pov_val = float(player_pov(
                    ball_np, p_np,
                    xT_ball_v, xT_recv_v,
                    defenders_opp,
                ))
            except Exception:
                pov_val = float("nan")

        if p.get("actor"):
            role = "Actor"
        elif p.get("teammate"):
            role = "Teammate"
        else:
            role = "Opponent"

        is_home = p.get("teammate")
        team = "Home" if is_home else "Away"

        rows.append({
            "name":       p.get("player_name") or "Unknown",
            "team":       team,
            "role":       role,
            "xT_pos":     xT_p,
            "pressure":   pressure,
            "pov":        pov_val,
            "confidence": p.get("confidence"),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _event_ball_start(event) -> Optional[list[float]]:
    x = event.get("Start X") if hasattr(event, "get") else None
    y = event.get("Start Y") if hasattr(event, "get") else None
    if x is None or y is None:
        return None
    try:
        return [float(x), float(y)]
    except (ValueError, TypeError):
        return None


def _event_coords(event, x_col: str, y_col: str) -> Optional[list[float]]:
    x = event.get(x_col) if hasattr(event, "get") else None
    y = event.get(y_col) if hasattr(event, "get") else None
    if x is None or y is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return [float(x), float(y)]
    except (ValueError, TypeError):
        return None


def _receiver_xy(
    event, next_event: Optional[dict | pd.Series]
) -> Optional[list[float]]:
    """Pass end coords from event End X/Y, or next_event Start X/Y."""
    end = _event_coords(event, "End X", "End Y")
    if end is not None:
        return end
    if next_event is not None:
        return _event_coords(next_event, "Start X", "Start Y")
    return None


def _axial_coords(
    hex_centers: np.ndarray,
    R: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian hex centres to axial (q, r) integer coords."""
    q_frac = (2.0 / 3.0) * hex_centers[:, 0] / R
    r_frac = (-1.0 / 3.0) * hex_centers[:, 0] / R + (np.sqrt(3) / 3.0) * hex_centers[:, 1] / R
    return np.round(q_frac).astype(int), np.round(r_frac).astype(int)


def _xT_at_positions(
    positions: np.ndarray,
    xT_home: np.ndarray,
    hex_centers: np.ndarray,
) -> np.ndarray:
    """Return xT at each of the given positions using nearest-hex lookup."""
    if len(positions) == 0:
        return np.array([])
    tree = cKDTree(hex_centers)
    _, idxs = tree.query(positions)
    return xT_home[idxs].astype(float)
