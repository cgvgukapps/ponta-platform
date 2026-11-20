"""
relationships.py — context package P5 (second part): relationships on every grid point.

Stage C (``ponta-blueprint/context-relationships-task.md``, deliverable 2). Computed per grid point for the team
out of possession against the ball and the team in possession (P1's ball state: a team controls the ball, or it
travels within one team's possession). Ponta's definitions, after FIFA's where one exists:

- **carrier pressure:** Plaiq's 1v1 pressure on the ball carrier (the controlling team's player nearest the
  ball), 1 to 10 from the hex distance to the nearest opponent (``hex_analytics.compute_pressure``'s formula,
  on Plaiq's hex grid in the same coordinates), so the score equals Plaiq's for that player and frame;
- **goal-side defenders:** defending outfield players nearer their own goal than the ball (by depth);
- **offside line:** the second-last defending player's distance from its own goal line (goalkeeper included);
- **free attackers:** attacking outfield players with no defending player within ``p5_free_radius_m``;
- **overload at the ball:** attacking minus defending outfield players within ``p5_overload_radius_m`` of the ball;
- **marking distance:** the mean, over defending outfield players, of the distance to the nearest attacker.

Null where no team has the ball (loose, contested, out of play, unknown) or the ball is undetected. Positions are
used only to compute these numbers; no coordinate leaves the package. Estimated, not yet validated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plaiq_engine.analytics.hex_analytics import cartesian_to_hex
from plaiq_engine.context.structure import OTHER, TEAMS

#: Plaiq's hexagon circumradius (one square metre per cell), as in ``match_analysis.HEX_RADIUS``.
HEX_R = float(np.sqrt(2.0 / (3.0 * np.sqrt(3))))

COLUMNS = ("p5_carrier_pressure", "p5_goal_side_defenders", "p5_offside_line_m", "p5_free_attackers",
           "p5_overload_at_ball", "p5_marking_distance_m")


def pressure_score(hex_distance: np.ndarray) -> np.ndarray:
    """Plaiq's 1v1 pressure score from a hex distance: 10 at one cell or less, 1 at ten or more (NaN kept)."""
    d = np.asarray(hex_distance, float)
    return np.where(np.isnan(d), np.nan, np.maximum(1, 11 - np.maximum(1, d)))


def hex_distance(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    """Ring distance on Plaiq's hex grid between points ``a`` (n, 2) and ``b`` (n, m, 2); NaN where either is missing."""
    qa, ra = cartesian_to_hex(a_xy[..., 0], a_xy[..., 1], HEX_R)
    qb, rb = cartesian_to_hex(b_xy[..., 0], b_xy[..., 1], HEX_R)
    dq, dr = np.asarray(qb, float) - np.asarray(qa, float)[:, None], np.asarray(rb, float) - np.asarray(ra, float)[:, None]
    d = (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) / 2
    bad = ~np.isfinite(a_xy).all(axis=-1)[:, None] | ~np.isfinite(b_xy).all(axis=-1)
    return np.where(bad, np.nan, d)


def relationships_on_grid(tf: dict, sign: dict, per: np.ndarray, ball_xy: np.ndarray, ball_state: np.ndarray,
                          keepers: dict, block: np.ndarray, length: float, cfg: dict) -> pd.DataFrame:
    """
    The relationship fields on every grid point. ``tf`` are P2's team frames (own-frame depth and width per player),
    ``sign`` the attack sign per (period, team), ``ball_xy`` the ball in absolute coordinates, ``ball_state`` P1's,
    ``keepers`` P2's goalkeeper per (team, period, window) and ``block`` the window per grid point.
    """
    n = len(per)
    out = {c: np.full(n, np.nan) for c in COLUMNS}
    s_of = np.array([[sign.get((int(p), t), 1) for t in TEAMS] for p in per], float)  # (n, 2)
    absol, own_depth, gk_col = {}, {}, {}
    for j, t in enumerate(TEAMS):
        f = tf[t]
        s = s_of[:, j][:, None]
        absol[t] = np.stack([s * (f.depth - length / 2), s * f.width], axis=-1)  # (n, players, 2), absolute
        own_depth[t] = f.depth
        gk = np.full(n, -1)
        for i in range(n):
            pid = keepers.get((t, int(per[i]), int(block[i])))
            gk[i] = f.ids.index(pid) if pid in f.ids else -1
        gk_col[t] = gk
    attacking = np.full(n, None, dtype=object)
    for t in TEAMS:
        attacking[np.isin(ball_state, [f"controlled_{t}", f"in_flight_{t}"])] = t
    ok = np.isfinite(ball_xy).all(axis=1)
    for t in TEAMS:
        rows = np.flatnonzero((attacking == t) & ok)
        if not len(rows):
            continue
        d = OTHER[t]
        A, D = absol[t][rows], absol[d][rows]  # absolute positions
        a_out = np.ones(A.shape[:2], bool)
        a_out[np.arange(len(rows)), np.clip(gk_col[t][rows], 0, None)] &= gk_col[t][rows] < 0
        d_out = np.ones(D.shape[:2], bool)
        d_out[np.arange(len(rows)), np.clip(gk_col[d][rows], 0, None)] &= gk_col[d][rows] < 0
        a_seen, d_seen = np.isfinite(A).all(axis=-1), np.isfinite(D).all(axis=-1)
        b = ball_xy[rows]
        # Carrier: the attacking player nearest the ball; Plaiq's 1v1 pressure on them.
        da = np.where(a_seen, np.linalg.norm(A - b[:, None, :], axis=-1), np.inf)
        carrier = np.argmin(da, axis=1)
        has = np.isfinite(da.min(axis=1))
        c_xy = A[np.arange(len(rows)), carrier]
        hd = hex_distance(c_xy, D)
        hd = np.where(d_seen, hd, np.nan)
        md = np.nanmin(np.where(np.isfinite(hd), hd, np.inf), axis=1)
        out["p5_carrier_pressure"][rows] = np.where(has & np.isfinite(md), pressure_score(np.where(np.isfinite(md), md, np.nan)), np.nan)
        # Depth in the defending team's own frame: its goal at 0. The ball's depth in that frame.
        dd = own_depth[d][rows]
        s_d = s_of[rows, TEAMS.index(d)]
        ball_d = s_d * b[:, 0] + length / 2
        out["p5_goal_side_defenders"][rows] = ((dd < ball_d[:, None]) & d_out & d_seen).sum(axis=1)
        srt = np.sort(np.where(d_seen, dd, np.inf), axis=1)
        out["p5_offside_line_m"][rows] = np.where(np.isfinite(srt[:, 1]), srt[:, 1], np.nan)
        # Free attackers, overload and marking distance (outfield players).
        pair = np.linalg.norm(A[:, :, None, :] - D[:, None, :, :], axis=-1)  # (r, attackers, defenders)
        pair = np.where(a_seen[:, :, None] & d_seen[:, None, :], pair, np.inf)
        nearest_def_any = pair.min(axis=2)  # any defender, the goalkeeper included, marks a free player
        out["p5_free_attackers"][rows] = ((nearest_def_any > cfg["p5_free_radius_m"]) & a_out & a_seen).sum(axis=1)
        r = cfg["p5_overload_radius_m"]
        near_a = (np.linalg.norm(A - b[:, None, :], axis=-1) <= r) & a_out & a_seen
        near_d = (np.linalg.norm(D - b[:, None, :], axis=-1) <= r) & d_out & d_seen
        out["p5_overload_at_ball"][rows] = near_a.sum(axis=1) - near_d.sum(axis=1)
        mark = np.where(a_out[:, :, None], pair, np.inf).min(axis=1)  # each defender's nearest attacker
        mark = np.where(d_out & d_seen & np.isfinite(mark), mark, np.nan)
        with np.errstate(all="ignore"):
            out["p5_marking_distance_m"][rows] = np.nanmean(mark, axis=1)
    frame = pd.DataFrame({c: np.round(out[c], 3) for c in COLUMNS})
    for c in ("p5_carrier_pressure", "p5_goal_side_defenders", "p5_free_attackers", "p5_overload_at_ball"):
        frame[c] = pd.array(np.where(np.isfinite(out[c]), out[c], np.nan), dtype="Float64").round().astype("Int64")
    return frame


# ---------------------------------------------------------------------------
# FIFA EFI aggregates per team and half (P5 0.2.0)
# ---------------------------------------------------------------------------

LANES = ("left_wing", "left_half_space", "centre", "right_half_space", "right_wing")
BEHIND_MIDFIELD = ("between_lines", "in_defensive_line", "behind_defence")
MIRROR_THIRD = {"defensive": "attacking", "middle": "middle", "attacking": "defensive"}


def efi_aggregates(frames: pd.DataFrame, chains: pd.DataFrame, passes: pd.DataFrame, end_index: np.ndarray, cfg: dict) -> pd.DataFrame:
    """
    FIFA EFI measures Ponta can compute from tracking, per team and half, in long form (team, period, metric, value):

    - ``receptions_behind_midfield`` / ``receptions_behind_defence``: completed passes received (P3's band at the
      reception) beyond the defending midfield line / beyond the defensive line;
    - ``final_third_entries`` and ``final_third_entries_<lane>``: possession chains whose ball first reaches the
      attacking third, by P3's lane at the entry (one per chain);
    - ``def_line_m_<in|out>_<third>`` and ``length_m_<in|out>_<third>``: median defensive line height and team length,
      in or out of possession, by the ball's third in the team's own view;
    - ``ball_recovery_time_s``: median time from losing a possession to the team's next possession in the half;
    - ``forced_turnovers``: possessions of the other team that end lost with its carrier's pressure at least
      ``p5_forced_turnover_min_pressure`` at the last moment a team had the ball.

    ``passes`` are P5's line-break rows; ``end_index`` the grid point where each is received (-1 when unknown).
    """
    rows = []
    per = frames["period"].to_numpy()
    att = frames["p3_attacking_team"].astype(object).to_numpy()
    third = frames["p3_ball_third"].astype(object).to_numpy()
    lane = frames["p3_ball_lane"].astype(object).to_numpy()
    band = frames["p3_ball_band"].astype(object).to_numpy()
    chain = frames["p1_chain_id"].to_numpy(dtype=float, na_value=np.nan)
    press = frames["p5_carrier_pressure"].to_numpy(dtype=float, na_value=np.nan)
    ts = frames["frame_ts"].to_numpy(float)

    def add(team, p, metric, value):
        if value is not None and not (isinstance(value, float) and not np.isfinite(value)):
            rows.append({"team": team, "period": int(p), "metric": metric, "value": round(float(value), 3)})

    for team in TEAMS:
        for p in sorted(pd.unique(per)):
            inp = per == p
            done = passes.loc[(passes["team"] == team) & (passes["period"] == p) & (passes["outcome"] == "completed")]
            ends = [int(end_index[i]) for i in done.index if end_index[i] >= 0]
            bands = [band[k] for k in ends]
            add(team, p, "receptions_behind_midfield", sum(b in BEHIND_MIDFIELD for b in bands))
            add(team, p, "receptions_behind_defence", sum(b == "behind_defence" for b in bands))
            entries = {ln: 0 for ln in LANES}
            mine = inp & (att == team) & np.isfinite(chain)
            for c in pd.unique(chain[mine]):
                k = np.flatnonzero(mine & (chain == c) & (third == "attacking"))
                if len(k) and lane[k[0]] in entries:
                    entries[lane[k[0]]] += 1
            add(team, p, "final_third_entries", sum(entries.values()))
            for ln, v in entries.items():
                add(team, p, f"final_third_entries_{ln}", v)
            for state, sel in (("in", inp & (att == team)), ("out", inp & (att == OTHER[team]))):
                own_third = third if state == "in" else np.array([MIRROR_THIRD.get(x) for x in third], dtype=object)
                for th in ("defensive", "middle", "attacking"):
                    k = sel & (own_third == th)
                    for f in ("def_line_m", "length_m"):
                        v = frames[f"p2_{team}_{f}"].to_numpy(dtype=float, na_value=np.nan)[k]
                        add(team, p, f"{f}_{state}_{th}", float(np.nanmedian(v)) if np.isfinite(v).any() else None)
            ch = chains.loc[chains["period"] == p].sort_values("start_ts")
            gaps = []
            for r in ch.loc[(ch["team"] == team) & (ch["end_type"] == "lost")].itertuples():
                nxt = ch.loc[(ch["team"] == team) & (ch["start_ts"] > r.end_ts)]
                if len(nxt):
                    gaps.append(float(nxt["start_ts"].iloc[0] - r.end_ts))
            add(team, p, "ball_recovery_time_s", float(np.median(gaps)) if gaps else None)
            forced = 0
            for r in ch.loc[(ch["team"] == OTHER[team]) & (ch["end_type"] == "lost")].itertuples():
                k = np.flatnonzero(inp & (ts >= r.start_ts) & (ts <= r.end_ts) & np.isfinite(press))
                if len(k) and press[k[-1]] >= cfg["p5_forced_turnover_min_pressure"]:
                    forced += 1
            add(team, p, "forced_turnovers", forced)
    return pd.DataFrame(rows, columns=["team", "period", "metric", "value"])
