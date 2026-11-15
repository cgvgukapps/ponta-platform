"""
statsbomb/stitch.py — Event stitching for StatsBomb open data.

Links consecutive events within a possession into chains using index ordering,
possession ID, and ball spatial continuity.  Key findings from the WC-final
data audit:

  - 75% of same-possession consecutive pairs have ball gap < 2 m (End→Start).
  - Pass → recipient chains: 911 per 90-min match (median ball gap 0 m).
  - Carry continuation (same player): 707 per match (median gap 0 m).
  - ~35% of FF events have a valid stitch to a next event that also has a FF,
    giving a usable "end-of-event" freeze-frame proxy.
  - Use sb_event_index ordering, NOT timestamps — ~110 pairs per match have
    negative dt due to overlapping events (CHALLENGE during CARRY, etc.).

The stitch graph is intentionally lightweight: a DataFrame of edge attributes
per consecutive pair, not a full graph structure.  Downstream consumers
(identity.py, sb_dq.py) call get_possession_chains() or end_proxy_ff()
directly.
"""
from __future__ import annotations

import math

import pandas as pd

# Maximum ball gap (metres) between End(A) and Start(B) for a stitch to be
# considered spatially valid.
_MAX_BALL_GAP_M = 2.0


def build_stitch_graph(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a stitch graph over consecutive StatsBomb events.

    Parameters
    ----------
    events_df : pd.DataFrame
        PLAIQ-schema events with at minimum:
        ``sb_event_index``, ``Period``, ``sb_possession``,
        ``Start X``, ``Start Y``, ``End X``, ``End Y``,
        ``Type``, ``From``, ``To``, ``sb_event_id``.

    Returns
    -------
    pd.DataFrame with one row per consecutive (A, B) pair and columns:

    ev_a_pos       int   — positional index in events_df (iloc)
    ev_b_pos       int
    ev_a_idx       int   — DataFrame .index value for event A
    ev_b_idx       int   — DataFrame .index value for event B
    ball_gap_m     float — Euclidean distance End(A)→Start(B) in metres
    dt_s           float — Start Time B − End Time A  (may be negative)
    same_period    bool
    same_poss      bool
    identity_link  str   — "pass_receipt" | "carry_cont" | "none"
    stitch_valid   bool  — same period + same poss + ball_gap < threshold
    """
    if events_df.empty or "sb_event_index" not in events_df.columns:
        return pd.DataFrame(columns=[
            "ev_a_pos", "ev_b_pos", "ev_a_idx", "ev_b_idx",
            "ball_gap_m", "dt_s", "same_period", "same_poss",
            "identity_link", "stitch_valid",
        ])
    ev = events_df.sort_values("sb_event_index").reset_index(drop=False)
    n = len(ev)
    if n < 2:
        return pd.DataFrame(columns=[
            "ev_a_pos", "ev_b_pos", "ev_a_idx", "ev_b_idx",
            "ball_gap_m", "dt_s", "same_period", "same_poss",
            "identity_link", "stitch_valid",
        ])

    rows = []
    for i in range(n - 1):
        a = ev.iloc[i]
        b = ev.iloc[i + 1]

        same_period = (a["Period"] == b["Period"])
        same_poss   = (a["sb_possession"] == b["sb_possession"])

        # Ball spatial gap
        ax, ay = a.get("End X"), a.get("End Y")
        bx, by = b.get("Start X"), b.get("Start Y")
        if pd.notna(ax) and pd.notna(ay) and pd.notna(bx) and pd.notna(by):
            ball_gap = float(math.hypot(float(bx) - float(ax),
                                        float(by) - float(ay)))
        else:
            ball_gap = float("nan")

        # Time gap (may be negative for overlapping events)
        end_t_a   = a.get("End Time [s]",   a.get("Start Time [s]", float("nan")))
        start_t_b = b.get("Start Time [s]", float("nan"))
        if pd.notna(end_t_a) and pd.notna(start_t_b):
            dt_s = float(start_t_b) - float(end_t_a)
        else:
            dt_s = float("nan")

        # Identity link type
        a_type = str(a.get("Type", ""))
        a_to   = str(a.get("To") or "")
        b_from = str(b.get("From") or "")
        if a_type == "PASS" and a_to and a_to == b_from:
            identity_link = "pass_receipt"
        elif a_type == "CARRY" and str(a.get("From") or "") == b_from and b_from:
            identity_link = "carry_cont"
        else:
            identity_link = "none"

        stitch_valid = (
            same_period
            and same_poss
            and not math.isnan(ball_gap)
            and ball_gap < _MAX_BALL_GAP_M
        )

        rows.append({
            "ev_a_pos":      i,
            "ev_b_pos":      i + 1,
            "ev_a_idx":      int(a["index"]),
            "ev_b_idx":      int(b["index"]),
            "ball_gap_m":    ball_gap,
            "dt_s":          dt_s,
            "same_period":   same_period,
            "same_poss":     same_poss,
            "identity_link": identity_link,
            "stitch_valid":  stitch_valid,
        })

    return pd.DataFrame(rows)


def get_possession_chains(
    events_df: pd.DataFrame,
    stitch_df: pd.DataFrame,
) -> list[list[int]]:
    """
    Group events into possession-level chains using the stitch graph.

    Returns
    -------
    list of lists, each inner list containing DataFrame index values (.index)
    for one possession sequence in chronological order.

    A new chain starts whenever ``stitch_valid`` is False (possession change,
    half boundary, or large ball gap).
    """
    if events_df.empty or "sb_event_index" not in events_df.columns:
        return []
    ev = events_df.sort_values("sb_event_index").reset_index(drop=False)
    if ev.empty:
        return []

    valid_b_positions: set[int] = set(
        stitch_df.loc[stitch_df["stitch_valid"], "ev_b_pos"].tolist()
    )

    chains: list[list[int]] = []
    current: list[int] = [int(ev.iloc[0]["index"])]

    for i in range(1, len(ev)):
        if i in valid_b_positions:
            current.append(int(ev.iloc[i]["index"]))
        else:
            chains.append(current)
            current = [int(ev.iloc[i]["index"])]

    chains.append(current)
    return chains


def end_proxy_ff(
    ev_df_idx: int,
    events_df: pd.DataFrame,
    stitch_df: pd.DataFrame,
    freeze_frames: dict,
) -> dict | None:
    """
    Return the freeze frame from the next stitched event as an "end snapshot"
    for the event at position ``ev_df_idx`` (DataFrame .index value).

    The next event's freeze frame is the closest available proxy for the
    spatial state at the end of the current event — covers ~35% of FF events
    (1,262 in a typical 90-min match with 360 data).

    Parameters
    ----------
    ev_df_idx    : int  — .index value of the event in events_df
    events_df    : pd.DataFrame — full events DataFrame
    stitch_df    : pd.DataFrame — output of build_stitch_graph()
    freeze_frames: dict — keyed by sb_event_id UUID string

    Returns
    -------
    Freeze-frame dict ({"players": [...], "visible_area": ...}) or None.
    """
    # Find the stitch edge where ev_df_idx is event A
    edge = stitch_df[
        (stitch_df["ev_a_idx"] == ev_df_idx) & stitch_df["stitch_valid"]
    ]
    if edge.empty:
        return None

    b_idx = int(edge.iloc[0]["ev_b_idx"])
    if b_idx not in events_df.index:
        return None

    b_uuid = str(events_df.loc[b_idx, "sb_event_id"] or "")
    return freeze_frames.get(b_uuid)


def stitch_coverage_summary(
    events_df: pd.DataFrame,
    stitch_df: pd.DataFrame,
    freeze_frames: dict,
) -> dict:
    """
    Return a dict of coverage statistics useful for diagnostics:

    total_events        int
    ff_events           int
    valid_stitches      int
    pct_valid_stitches  float
    dual_ff_pairs       int   — stitched pairs where both A and B have a FF
    pct_dual_ff_of_ff   float — dual_ff_pairs / ff_events
    pass_receipt_links  int
    carry_cont_links    int
    """
    n_total = len(events_df)
    ff_uuids = set(str(u) for u in freeze_frames)
    n_ff = events_df["sb_event_id"].astype(str).isin(ff_uuids).sum()

    valid = stitch_df[stitch_df["stitch_valid"]]
    n_valid = len(valid)

    dual_ff = 0
    for _, row in valid.iterrows():
        a_uuid = str(events_df.loc[row["ev_a_idx"], "sb_event_id"])
        b_uuid = str(events_df.loc[row["ev_b_idx"], "sb_event_id"])
        if a_uuid in ff_uuids and b_uuid in ff_uuids:
            dual_ff += 1

    return {
        "total_events":        n_total,
        "ff_events":           int(n_ff),
        "valid_stitches":      n_valid,
        "pct_valid_stitches":  round(100 * n_valid / max(1, n_total - 1), 1),
        "dual_ff_pairs":       dual_ff,
        "pct_dual_ff_of_ff":   round(100 * dual_ff / max(1, int(n_ff)), 1),
        "pass_receipt_links":  int((stitch_df["identity_link"] == "pass_receipt").sum()),
        "carry_cont_links":    int((stitch_df["identity_link"] == "carry_cont").sum()),
    }
