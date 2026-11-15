#!/usr/bin/env python3
"""
event_stitch.py — Link consecutive events for pass-receipt arrival frames.

When a pass event's End Frame ends before the ball arrives, the **next
event's start** is the observed receipt moment: the receiver has the ball
and the next action begins.  Used by Event-step viz (receiver ghost) and
pass POV / decision quality (end-frame scoring).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

_MAX_BALL_GAP_M = 2.0
_MAX_RECEIPT_DT_S = 8.0


def clean_jersey(val) -> str:
    """Normalise jersey labels: strip 'Player' prefix and float '.0' suffix."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, float) and val == int(val):
        val = int(val)
    return str(val).replace("Player", "").strip()


def _ball_gap_m(a: pd.Series, b: pd.Series) -> float:
    ax, ay = a.get("End X"), a.get("End Y")
    bx, by = b.get("Start X"), b.get("Start Y")
    if pd.isna(ax) or pd.isna(ay) or pd.isna(bx) or pd.isna(by):
        return float("nan")
    return float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))


def _dt_s(a: pd.Series, b: pd.Series) -> float:
    end_t = a.get("End Time [s]", a.get("Start Time [s]", np.nan))
    start_t = b.get("Start Time [s]", np.nan)
    if pd.isna(end_t) or pd.isna(start_t):
        return float("nan")
    return float(start_t) - float(end_t)


def evaluate_pass_receipt(
    pass_event: pd.Series,
    next_event: pd.Series,
    pass_idx: int,
    next_idx: int,
) -> dict:
    """
    Decide whether ``next_event`` is the receipt of ``pass_event``.

    Returns a dict with keys:
        pass_event_idx, receipt_event_idx, receipt_start_frame,
        receipt_start_x, receipt_start_y, ball_gap_m, dt_s,
        stitch_valid, identity_link
    """
    to_j = clean_jersey(pass_event.get("To"))
    from_j = clean_jersey(next_event.get("From"))
    ball_gap = _ball_gap_m(pass_event, next_event)
    dt = _dt_s(pass_event, next_event)

    same_period = pass_event.get("Period") == next_event.get("Period")
    same_team = pass_event.get("Team") == next_event.get("Team")
    pass_receipt = bool(to_j) and to_j == from_j

    if pass_receipt:
        identity_link = "pass_receipt"
    else:
        identity_link = "none"

    stitch_valid = (
        same_period
        and same_team
        and pass_receipt
        and not math.isnan(ball_gap)
        and ball_gap < _MAX_BALL_GAP_M
        and not math.isnan(dt)
        and 0.0 <= dt <= _MAX_RECEIPT_DT_S
    )

    sf_raw = next_event.get("Start Frame")
    receipt_frame = int(sf_raw) if pd.notna(sf_raw) else None
    sx, sy = next_event.get("Start X"), next_event.get("Start Y")

    return {
        "pass_event_idx":       pass_idx,
        "receipt_event_idx":    next_idx,
        "receipt_start_frame":  receipt_frame,
        "receipt_start_x":      float(sx) if pd.notna(sx) else np.nan,
        "receipt_start_y":      float(sy) if pd.notna(sy) else np.nan,
        "ball_gap_m":           ball_gap,
        "dt_s":                 dt,
        "stitch_valid":         stitch_valid,
        "identity_link":        identity_link,
    }


def _next_same_period_index(
    sorted_indices: list,
    events: pd.DataFrame,
    pos: int,
    period,
) -> int | None:
    for j in range(pos + 1, len(sorted_indices)):
        b_idx = sorted_indices[j]
        if events.loc[b_idx, "Period"] == period:
            return b_idx
    return None


def build_pass_receipt_lookup(events_df: pd.DataFrame) -> dict[int, dict]:
    """
    Precompute pass → receipt links for all PASS rows in ``events_df``.

    Returns
    -------
    dict mapping pass event .index → receipt info dict from
    :func:`evaluate_pass_receipt`.
    """
    if events_df.empty:
        return {}

    ev = events_df.sort_values(
        ["Period", "Start Frame", "Start Time [s]"], kind="stable",
    )
    sorted_indices = ev.index.tolist()
    lookup: dict[int, dict] = {}

    for i, a_idx in enumerate(sorted_indices):
        a = ev.loc[a_idx]
        if "PASS" not in str(a.get("Type", "")).upper():
            continue
        b_idx = _next_same_period_index(
            sorted_indices, ev, i, a["Period"],
        )
        if b_idx is None:
            continue
        b = ev.loc[b_idx]
        lookup[a_idx] = evaluate_pass_receipt(a, b, a_idx, b_idx)

    return lookup


def ball_xy_from_tracking(
    home_tracking: pd.DataFrame,
    away_tracking: pd.DataFrame,
    frame: int,
    fallback: np.ndarray | None = None,
) -> np.ndarray | None:
    """Tracked ball at ``frame``, or ``fallback`` when absent."""
    for df in (away_tracking, home_tracking):
        if frame not in df.index:
            continue
        row = df.loc[frame]
        bx, by = row.get("ball_x"), row.get("ball_y")
        if pd.notna(bx) and pd.notna(by):
            return np.array([float(bx), float(by)])
    return fallback


def patch_pass_receipt_receiver(
    end_row_h: pd.Series | None,
    end_row_a: pd.Series | None,
    home_tracking: pd.DataFrame,
    away_tracking: pd.DataFrame,
    pass_event: pd.Series,
    receipt: dict,
) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Move the intended receiver's ghost position to the receipt frame.

    Other players stay at the pass end-frame positions in ``end_row_*``.
    Creates end rows from the pass End Frame when missing.
    """
    if not receipt or not receipt.get("stitch_valid"):
        return end_row_h, end_row_a

    rf = receipt.get("receipt_start_frame")
    recv_j = clean_jersey(pass_event.get("To"))
    if rf is None or not recv_j:
        return end_row_h, end_row_a

    rf = int(rf)
    team = str(pass_event.get("Team", "Home"))
    prefix = "Home" if team == "Home" else "Away"
    tdf = home_tracking if team == "Home" else away_tracking

    if rf not in tdf.index:
        return end_row_h, end_row_a

    rx = tdf.loc[rf, f"{prefix}_{recv_j}_x"]
    ry = tdf.loc[rf, f"{prefix}_{recv_j}_y"]
    if pd.isna(rx) or pd.isna(ry):
        return end_row_h, end_row_a

    if end_row_h is None or end_row_a is None:
        ef_raw = pass_event.get("End Frame")
        if pd.isna(ef_raw):
            return end_row_h, end_row_a
        ef = int(ef_raw)
        if ef not in home_tracking.index or ef not in away_tracking.index:
            return end_row_h, end_row_a
        end_row_h = home_tracking.loc[ef].copy()
        end_row_a = away_tracking.loc[ef].copy()

    target = end_row_h if team == "Home" else end_row_a
    target[f"{prefix}_{recv_j}_x"] = float(rx)
    target[f"{prefix}_{recv_j}_y"] = float(ry)

    return end_row_h, end_row_a
