"""
movement.py — context package P6: movement types.

Stage C (``ponta-blueprint/context-relationships-task.md``, deliverable 3). Every **run** of every outfield player:
a spell at ``p6_run_min_speed_mps`` or faster (15 km/h, a common running band and the slowest average speed among
SkillCorner's labelled runs) lasting at least ``p6_run_min_s``. Runs by the team in possession (P1's ball state:
controlled or in flight within its possession, at the run's start) are off-ball runs of the carrier's team-mates;
runs by the other team are defensive runs. Each run gets, by rules fixed before any scoring:

**FIFA's categories, relative to P2's units and the defending shape** (``ponta-blueprint/fifa-alignment.md``):
- ``area_end``: where the run ends against the defending units (``line_breaks.units_at``): ``in_front`` of the first
  unit, ``in_between`` units, or ``in_behind`` the last; an in-behind run is ``from_final_unit`` when it starts
  within ``p6_move_min_m`` of the last unit's line, else ``from_in_front``;
- ``shape``: ``in_to_out``, ``out_to_in``, ``inside`` or ``outside`` the defending team's convex hull (start to end).
  (SkillCorner's ``in_to_out`` and ``out_to_in`` flags are moves between channels, not the shape; ``channel_move``
  gives that reading: ``outward``, ``inward`` or ``same`` across P3's centre, half-space and wide bands.)

**Not off-ball runs** (version 0.2.0, after the founder's run review): a run counts as in possession when its team
has the ball for more of the run than the other team (time with nobody on the ball counts for neither; else the state
at the run's start); the carrier's own movement (the team's nearest player to the ball, within ``p6_carry_radius_m``,
for at least half the run) is a ``carry``, not typed or counted as an off-ball run. (A run that ends with the
player receiving the ball stays its type: the founder typed such runs, and SkillCorner labels them too.)

**The type**, in priority order (the names follow SkillCorner's run vocabulary so they can be measured; Ponta's
rules): ``cross_receiver`` (the ball wide in the attacking third, the run ending in the box), ``behind`` (ending in
behind and gaining depth), ``overlap`` / ``underlap`` (starting behind the carrier and finishing ahead of them,
outside or inside them), ``coming_short`` (closing on the carrier and ending ahead of them), ``dropping_off`` (losing
depth), ``pulling_wide`` / ``pulling_half_space`` (moving sideways into the wing or a half-space), ``support``
(ending behind or level with the carrier), else ``run_ahead_of_the_ball``. Defensive runs are ``pressing`` (ending
within ``p6_press_end_m`` of the carrier, having closed on them; defensive runs may be as slow as
``p6_press_min_speed_mps``), ``recovery`` (towards the own goal with the ball
nearer it than the runner) or ``other``.

Offers made only by a signal or body shape are not visible in tracking and are never claimed. Positions are used to
classify; none leaves the package. Estimated, not yet validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from plaiq_engine.context.line_breaks import inside_shape, units_at
from plaiq_engine.context.possession import ball_on_grid
from plaiq_engine.context.registry import (
    ContextField,
    ContextInput,
    ContextPackage,
    PackageOutput,
)
from plaiq_engine.context.structure import (
    OTHER,
    TEAMS,
    team_frames,
    window_index,
    windows_from_segments,
)

BOX_DEPTH_M = 16.5
BOX_HALF_WIDTH_M = 20.16
HALF_SPACE_M = 9.16
RUN_COLUMNS = ("team", "player_id", "period", "start_ts", "end_ts", "duration_s", "in_possession", "type", "area_end",
               "behind_from", "shape", "channel_move", "gain_m", "lateral_m")


def channel_band(width: float) -> int:
    """0 centre, 1 half-space, 2 wide: P3's lane edges by distance from the centre line."""
    a = abs(width)
    return 0 if a < HALF_SPACE_M else (1 if a <= BOX_HALF_WIDTH_M else 2)


def runs_of(speed: np.ndarray, ts: np.ndarray, per: np.ndarray, min_speed: float, min_s: float) -> list[tuple[int, int]]:
    """Index spans (start, end inclusive) where ``speed`` stays at or above ``min_speed`` for at least ``min_s``."""
    out, i, n = [], 0, len(speed)
    fast = np.nan_to_num(speed, nan=0.0) >= min_speed
    while i < n:
        if not fast[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and fast[j + 1] and per[j + 1] == per[i]:
            j += 1
        if ts[j] - ts[i] + 1e-9 >= min_s:
            out.append((i, j))
        i = j + 1
    return out


def classify_attacking(run: dict, cfg: dict, length: float) -> str:
    """The type of an off-ball run from its geometry (attacking frame: depth towards the opponent's goal)."""
    m = cfg["p6_move_min_m"]
    d0, d1, w0, w1 = run["d0"], run["d1"], run["w0"], run["w1"]
    cd0, cd1, cw0 = run["carrier_d0"], run["carrier_d1"], run["carrier_w0"]
    bd, bw = run["ball_d0"], run["ball_w0"]
    in_box = d1 >= length - BOX_DEPTH_M and abs(w1) <= BOX_HALF_WIDTH_M
    ball_wide = any(np.isfinite(d) and d >= 2 * length / 3 and abs(w) > BOX_HALF_WIDTH_M
                    for d, w in ((bd, bw), (run.get("ball_d1", np.nan), run.get("ball_w1", np.nan))))
    if ball_wide and in_box:  # the ball wide in the attacking third at the run's start or end
        return "cross_receiver"
    if run["area_end"] == "in_behind" and d1 - d0 >= m:
        return "behind"
    if np.isfinite(cd0) and d0 < cd0 - 1.0 and d1 > cd1 + 1.0 and np.sign(w0) == np.sign(cw0):
        return "overlap" if abs(w1) > abs(run["carrier_w1"]) else "underlap"
    dist0 = np.hypot(d0 - cd0, w0 - cw0) if np.isfinite(cd0) else np.nan
    dist1 = np.hypot(d1 - cd1, w1 - run["carrier_w1"]) if np.isfinite(cd1) else np.nan
    if np.isfinite(dist0) and dist0 - dist1 >= m and d1 > cd1 and d1 - d0 < m:  # towards the ball, not a forward run
        return "coming_short"
    if d1 - d0 <= -m:
        return "dropping_off"
    sideways = abs(w1 - w0) >= max(m, abs(d1 - d0))  # a pulling run is mainly sideways
    if sideways and abs(w1) - abs(w0) >= m and abs(w1) > BOX_HALF_WIDTH_M:
        return "pulling_wide"
    if sideways and HALF_SPACE_M <= abs(w1) <= BOX_HALF_WIDTH_M:
        return "pulling_half_space"
    if np.isfinite(cd1) and d1 <= cd1 + 2.0:
        return "support"
    return "run_ahead_of_the_ball"


@dataclass
class _Fields:
    items: tuple[ContextField, ...] = field(default=(
        ContextField("p6_home_runs", "players", "estimated",
                     "Home outfield players in a run (at least p6_run_min_speed_mps for at least p6_run_min_s) at this grid point."),
        ContextField("p6_away_runs", "players", "estimated", "Away outfield players in a run at this grid point."),
        ContextField("p6_runs_in_behind", "players", "estimated",
                     "Players of the team in possession whose current run ends in behind the defending team's last unit (FIFA's category)."),
    ))


class MovementPackage(ContextPackage):
    name = "p6"
    version = "0.2.0"
    depends_on = ("p1", "p2")
    fields = _Fields().items

    def compute(self, ctx: ContextInput) -> PackageOutput:
        m, g, cfg = ctx.match, ctx.grid, ctx.config
        p1, p2 = ctx.upstream["p1"], ctx.upstream["p2"]
        n, L = len(g), m.pitch.length_m
        ts, per = g["frame_ts"].to_numpy(float), g["period"].to_numpy()
        tf = team_frames(m, g, p2.segments["orientation"])
        sign = {(int(r.period), r.team): r.attack_sign for r in p2.segments["orientation"].itertuples()}
        block = window_index(g, cfg)
        keepers = windows_from_segments(p2.segments).gk
        bs = p1.frames["p1_ball_state"].astype(object).to_numpy()
        ball = ball_on_grid(m, g)[["ball_x", "ball_y"]].to_numpy(float)
        attacking = np.full(n, None, dtype=object)
        for t in TEAMS:
            attacking[np.isin(bs, [f"controlled_{t}", f"in_flight_{t}"])] = t
        # The ball and each team's carrier in each team's own (attacking) frame.
        s_t = {t: np.array([sign.get((int(p), t), 1) for p in per], float) for t in TEAMS}
        ball_own = {t: (s_t[t] * ball[:, 0] + L / 2, s_t[t] * ball[:, 1]) for t in TEAMS}
        carrier, near_ball = {}, {}
        for t in TEAMS:
            dd = np.hypot(tf[t].depth - ball_own[t][0][:, None], tf[t].width - ball_own[t][1][:, None])
            dd = np.where(np.isfinite(dd), dd, np.inf)
            carrier[t] = np.where(np.isfinite(dd.min(axis=1)), dd.argmin(axis=1), -1)
            near_ball[t] = dd.min(axis=1)

        def gk_col(t, k):
            pid = keepers.get((t, int(per[k]), int(block[k])))
            return tf[t].ids.index(pid) if pid in tf[t].ids else None

        rows, counts = [], {t: np.zeros(n, int) for t in TEAMS}
        behind_now = np.zeros(n, int)
        step = np.r_[np.diff(ts), np.nan]
        for t in TEAMS:
            f = tf[t]
            vd = np.vstack([np.diff(f.depth, axis=0), np.full((1, f.depth.shape[1]), np.nan)]) / step[:, None]
            vw = np.vstack([np.diff(f.width, axis=0), np.full((1, f.width.shape[1]), np.nan)]) / step[:, None]
            speed = pd.DataFrame(np.hypot(vd, vw)).rolling(3, center=True, min_periods=1).mean().to_numpy()
            for j, pid in enumerate(f.ids):
                spans = runs_of(speed[:, j], ts, per, cfg["p6_run_min_speed_mps"], cfg["p6_run_min_s"])
                if cfg["p6_press_min_speed_mps"] < cfg["p6_run_min_speed_mps"]:
                    # Defensive runs may be slower (a press is often a jog): spans at the lower speed while the other
                    # team has the ball, not already covered by a run.
                    slow = runs_of(speed[:, j], ts, per, cfg["p6_press_min_speed_mps"], cfg["p6_run_min_s"])
                    spans += [(a, b) for a, b in slow if attacking[a] == OTHER[t] and not any(a <= y and b >= x for x, y in spans)]
                for a, b in sorted(spans):
                    gk = gk_col(t, a)
                    if gk == j:
                        continue
                    counts[t][a:b + 1] += 1
                    d0, d1, w0, w1 = f.depth[a, j], f.depth[b, j], f.width[a, j], f.width[b, j]
                    if not np.isfinite([d0, d1, w0, w1]).all():
                        continue
                    span = np.arange(a, b + 1)
                    # 0.2.0 (the founder's run review): possession over the run, not at its first instant; and the
                    # carrier's own movement is a carry, not an off-ball run.
                    n_own, n_their = int((attacking[span] == t).sum()), int((attacking[span] == OTHER[t]).sum())
                    if n_own == n_their:  # no team has the ball during the run, or as long each: the state at its start
                        own, theirs = attacking[a] == t, attacking[a] == OTHER[t]
                    else:
                        own, theirs = n_own > n_their, n_their > n_own
                    on_ball = (carrier[t][span] == j) & (near_ball[t][span] <= cfg["p6_carry_radius_m"])
                    run = {"team": t, "player_id": pid, "period": int(per[a]), "start_ts": float(ts[a]), "end_ts": float(ts[b]),
                           "duration_s": round(float(ts[b] - ts[a]), 2), "in_possession": bool(own), "type": None,
                           "area_end": None, "behind_from": None, "shape": None,
                           "channel_move": {1: "outward", -1: "inward", 0: "same"}[int(np.sign(channel_band(w1) - channel_band(w0)))]
                           if np.isfinite([w0, w1]).all() else None,
                           "gain_m": round(float(d1 - d0), 2), "lateral_m": round(float(abs(w1) - abs(w0)), 2)}
                    if own and on_ball.mean() >= 0.5:
                        run["type"] = "carry"
                    elif own:
                        d = OTHER[t]
                        units = units_at(tf[d].depth[b], tf[d].width[b], L, gk_col(d, b))
                        units0 = units_at(tf[d].depth[a], tf[d].width[a], L, gk_col(d, a))
                        last = units[0][:, 0].max() if len(units[0]) else np.nan
                        first = units[2][:, 0].min() if len(units[2]) else np.nan
                        if np.isfinite(last) and d1 > last:
                            run["area_end"] = "in_behind"
                            run["behind_from"] = "from_final_unit" if abs(d0 - last) <= cfg["p6_move_min_m"] else "from_in_front"
                        elif np.isfinite(first) and d1 < first:
                            run["area_end"] = "in_front"
                        elif np.isfinite(last):
                            run["area_end"] = "in_between"
                        i0, i1 = inside_shape(units0, (d0, w0)), inside_shape(units, (d1, w1))
                        if i0 is not None and i1 is not None:
                            run["shape"] = {(True, False): "in_to_out", (False, True): "out_to_in",
                                            (True, True): "inside", (False, False): "outside"}[(i0, i1)]
                        c0, c1 = carrier[t][a], carrier[t][b]
                        geo = {"d0": d0, "d1": d1, "w0": w0, "w1": w1, "area_end": run["area_end"],
                               "carrier_d0": f.depth[a, c0] if c0 >= 0 and c0 != j else np.nan,
                               "carrier_w0": f.width[a, c0] if c0 >= 0 and c0 != j else np.nan,
                               "carrier_d1": f.depth[b, c1] if c1 >= 0 and c1 != j else np.nan,
                               "carrier_w1": f.width[b, c1] if c1 >= 0 and c1 != j else np.nan,
                               "ball_d0": ball_own[t][0][a], "ball_w0": ball_own[t][1][a],
                               "ball_d1": ball_own[t][0][b], "ball_w1": ball_own[t][1][b]}
                        run["type"] = classify_attacking(geo, cfg, L)
                        if run["area_end"] == "in_behind":
                            behind_now[a:b + 1] += 1
                    elif theirs:
                        o = OTHER[t]
                        co = carrier[o][b]
                        if co >= 0:
                            # The carrier in this (defending) team's own frame: mirror both axes.
                            cdep, cwid = L - tf[o].depth[b, co], -tf[o].width[b, co]
                            ca = carrier[o][a] if carrier[o][a] >= 0 else co
                            cdep0, cwid0 = L - tf[o].depth[a, ca], -tf[o].width[a, ca]
                            closed = np.hypot(d0 - cdep0, w0 - cwid0) - np.hypot(d1 - cdep, w1 - cwid)
                            if np.hypot(d1 - cdep, w1 - cwid) <= cfg["p6_press_end_m"] and closed > 0:
                                run["type"] = "pressing"
                        if run["type"] is None:
                            bd = ball_own[t][0][a]
                            run["type"] = "recovery" if (d1 - d0 <= -cfg["p6_move_min_m"] and np.isfinite(bd) and bd < d0) else "other"
                    rows.append(run)
        runs = pd.DataFrame(rows, columns=list(RUN_COLUMNS)).sort_values(["start_ts", "team", "player_id"], kind="stable").reset_index(drop=True)
        frames = pd.DataFrame({"p6_home_runs": counts["home"], "p6_away_runs": counts["away"], "p6_runs_in_behind": behind_now})
        att_runs = runs.loc[runs["in_possession"] & ~runs["type"].isin(["carry"])]
        played = sum(float(np.isfinite(tf[t].depth).sum()) for t in TEAMS) * (float(np.nanmedian(np.diff(ts))) if n > 1 else 0.0)
        measurements = {
            "runs": int(len(runs)), "off_ball_runs": int(len(att_runs)),
            "carries": int((runs["type"] == "carry").sum()),
            "type_share": {k: round(float(v), 4) for k, v in att_runs["type"].value_counts(normalize=True).items()},
            "area_end_share": {k: round(float(v), 4) for k, v in att_runs["area_end"].value_counts(normalize=True).items()},
            "shape_share": {k: round(float(v), 4) for k, v in att_runs["shape"].value_counts(normalize=True).items()},
            "defensive_type_share": {k: round(float(v), 4) for k, v in runs.loc[~runs["in_possession"], "type"].value_counts(normalize=True).items()},
            "player_time_in_runs_share": round(float(runs["duration_s"].sum() / played), 4) if played else None,
        }
        p2_state = next((x.state for x in p2.entries if x.metric == "team_shape"), "blocked")
        state = "blocked" if not len(runs) else ("full" if p2_state == "full" else "degraded")
        cov = measurements["player_time_in_runs_share"]
        entries = [self.entry(
            "movement", "Movement types", state,
            {"full": "Runs are found from player speeds and typed against the defending units and shape.",
             "degraded": "Runs are typed, but team shape is degraded (players out of view), so units and the shape can be incomplete.",
             "blocked": "No runs: the tracking has no players moving at running speed."}[state]
            + " Offers made by a signal or body shape are not visible in tracking. Ponta's rules after FIFA's categories: estimated, not yet validated.",
            coverage=f"{(cov or 0) * 100:.1f}% of player time in runs", unlock=None if state == "full" else "Full-pitch tracking of both teams.")]
        return PackageOutput(frames=frames, entries=entries, segments={"runs": runs}, measurements=measurements)
