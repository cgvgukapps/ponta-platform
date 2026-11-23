"""
fifa_view.py — context package P4v: a FIFA-vocabulary view of P4's phases, with engagements, presses and FIFA's
published mid-block rule.

Stage C (``ponta-blueprint/context-relationships-task.md``, deliverable 5; ``ponta-blueprint/fifa-alignment.md``).
P4 keeps SkillCorner's vocabulary, which is what Ponta measures against; this package publishes FIFA's names beside
it and adds the splits FIFA makes that tracking supports. P4's agreed outputs are unchanged. Ponta's rules, after
FIFA's definitions where one exists:

- **Engagements (the pressing measure).** A defending outfield player is engaging the ball carrier (P5's carrier: the
  controlling team's player nearest the ball) while within ``p4v_engage_radius_m`` of the carrier and closing on them
  at ``p4v_engage_closing_mps`` or more (over ``p4v_closing_lag_s``, the same carrier throughout); spells shorter than
  ``p4v_engage_min_s`` are dropped. Each engagement is typed at its start, in priority order:
  ``recovery`` (the defender is running towards its own goal at ``p4v_recovery_back_mps`` or more: FIFA's recovery),
  ``counter_press`` (within ``p4v_counter_press_s`` of its team losing the ball: FIFA's counter-press), ``press``
  (the ball beyond ``p4v_pressure_max_depth_share`` of the pitch from the defender's goal) and ``pressure`` (nearer).
- **Presses:** at least ``p4v_press_min_players`` different defenders' engagements (not ``pressure``), each starting
  within ``p4v_press_gap_s`` of the previous one's end, with no change of possession between them. A press is a
  counter-press when its first engagement is one.
- **The FIFA view of phases**, per grid point. For the team in possession, P4's phase by FIFA's name; ``build_up``
  splits into ``build_up_opposed`` (P5's carrier pressure at ``p4v_opposed_min_pressure`` or more in the last
  ``p4v_opposed_hold_s`` on the same carrier; adding the engagements over-calls opposed) and ``build_up_unopposed``. For the team
  out of possession, P4's block by FIFA's name, split into press and block: ``high_press`` / ``mid_press`` /
  ``low_press`` while a press is under way, ``counter_press`` while a counter-press is, otherwise ``high_block`` /
  ``mid_block`` / ``low_block``.
- **FIFA's mid-block rule** (World Cup 2022 article, the only published phase thresholds), for the team out of
  possession: no outfield player moving forward or backward faster than ``p4v_fifa_max_speed_kmh``; at least
  ``p4v_fifa_min_behind`` outfield players behind the ball; at least ``p4v_fifa_min_players`` outfield players in the
  middle and defensive thirds (``low_block`` when they are all in the defensive third, ``mid_block`` otherwise); held
  for ``p4v_fifa_hold_s`` of the previous ``p4v_fifa_window_s``. FIFA's "players form horizontal lines" is not
  checked. Beside P4's block, which comes from the three deepest defenders.

Positions are used only to compute these labels and counts; no coordinate leaves the package. Estimated, not yet
validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

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

#: P4's in-possession phases by FIFA's names (``build_up`` is split by the engagement rule).
IN_FIFA = {"build_up": "build_up", "create": "progression", "finish": "final_third", "direct": "long_ball",
           "quick_break": "counter_attack", "transition": "attacking_transition", "set_play": "set_play",
           "chaotic": "in_contest"}
#: P4's out-of-possession phases by FIFA's names (blocks are split into press and block).
OUT_FIFA = {"high_block": "high_block", "medium_block": "mid_block", "low_block": "low_block",
            "defending_transition": "defensive_transition", "defending_quick_break": "defensive_transition",
            "defending_direct": "defending_long_ball", "defending_set_play": "defending_set_play", "chaotic": "in_contest"}
PRESS_OF_BLOCK = {"high_block": "high_press", "mid_block": "mid_press", "low_block": "low_press"}
ENGAGEMENT_TYPES = ("recovery", "counter_press", "press", "pressure")


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs, end inclusive, of the True runs in a 1-d boolean array."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1, strict=True))


def held(ok: np.ndarray, ts: np.ndarray, hold_s: float, window_s: float) -> np.ndarray:
    """True where ``ok`` was true for at least ``hold_s`` of the ``window_s`` up to and including each point."""
    ok = np.asarray(ok, bool)
    if not len(ok):
        return ok
    step = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0
    c = np.r_[0, np.cumsum(ok)]
    j = np.searchsorted(ts, ts - window_s + 1e-9)
    count = c[np.arange(1, len(ok) + 1)] - c[j]
    return ok & (count * step >= hold_s - 1e-9)


def type_engagement(back_mps: float, since_loss_s: float, ball_depth_m: float, length: float, cfg: dict) -> str:
    """An engagement's type at its start, in priority order: recovery, counter-press, press, pressure."""
    if np.isfinite(back_mps) and back_mps >= cfg["p4v_recovery_back_mps"]:
        return "recovery"
    if np.isfinite(since_loss_s) and since_loss_s <= cfg["p4v_counter_press_s"]:
        return "counter_press"
    if np.isfinite(ball_depth_m) and ball_depth_m < cfg["p4v_pressure_max_depth_share"] * length:
        return "pressure"
    return "press"


def chain_presses(eng: pd.DataFrame, possession_change_ts: dict, cfg: dict) -> pd.DataFrame:
    """Group engagements into presses (see the module docstring). ``eng`` holds one team's engagements."""
    cols = ["team", "start_ts", "end_ts", "engagements", "players", "type"]
    e = eng.loc[eng["type"] != "pressure"].sort_values("start_ts", kind="stable")
    out = []
    cur: list = []

    def close():
        if cur and len({r.player for r in cur}) >= cfg["p4v_press_min_players"]:
            out.append({"team": cur[0].team, "start_ts": cur[0].start_ts, "end_ts": max(r.end_ts for r in cur),
                        "engagements": len(cur), "players": len({r.player for r in cur}),
                        "type": "counter_press" if cur[0].type == "counter_press" else "press"})

    for r in e.itertuples():
        if cur:
            end = max(x.end_ts for x in cur)
            changes = possession_change_ts.get(r.team, np.array([]))
            changed = bool(((changes > cur[0].start_ts) & (changes <= r.start_ts)).any())
            if r.start_ts - end > cfg["p4v_press_gap_s"] or changed or r.period != cur[0].period:
                close()
                cur = []
        cur.append(r)
    close()
    return pd.DataFrame(out, columns=cols)


@dataclass
class _Fields:
    items: tuple[ContextField, ...] = field(default=(
        ContextField("p4v_phase", "label", "estimated",
                     "The team in possession's phase in FIFA's vocabulary (P4's phase by FIFA's name; build-up split into opposed and unopposed). Null where P4 has no phase."),
        ContextField("p4v_out_phase", "label", "estimated",
                     "The team out of possession's phase in FIFA's vocabulary: high, mid or low press or block, counter-press, defensive transition, and P4's other defending phases by name."),
        ContextField("p4v_engaging_defenders", "players", "estimated",
                     "Defending outfield players engaging the ball carrier (within p4v_engage_radius_m and closing at p4v_engage_closing_mps or more). Null when no team controls the ball."),
        ContextField("p4v_fifa_block", "label", "estimated",
                     "FIFA's published block rule for the team out of possession: mid_block or low_block where it holds; null elsewhere. Beside P4's block, never replacing it."),
    ))


class FifaViewPackage(ContextPackage):
    name = "p4v"
    version = "0.1.0"
    depends_on = ("p1", "p2", "p4", "p5")
    fields = _Fields().items

    def compute(self, ctx: ContextInput) -> PackageOutput:
        m, g, cfg = ctx.match, ctx.grid, ctx.config
        p1, p2, p4, p5 = (ctx.upstream[k] for k in ("p1", "p2", "p4", "p5"))
        n, L = len(g), m.pitch.length_m
        ts, per = g["frame_ts"].to_numpy(float), g["period"].to_numpy()
        step = float(np.median(np.diff(ts))) if n > 1 else 0.2
        lag = max(1, int(round(cfg["p4v_closing_lag_s"] / step)))
        tf = team_frames(m, g, p2.segments["orientation"])
        sign = {(int(r.period), r.team): r.attack_sign for r in p2.segments["orientation"].itertuples()}
        s_of = np.array([[sign.get((int(p), t), 1 if t == "home" else -1) for t in TEAMS] for p in per], float).reshape(n, 2)
        block = window_index(g, cfg)
        keepers = windows_from_segments(p2.segments).gk
        ball = ball_on_grid(m, g)[["ball_x", "ball_y"]].to_numpy(float)
        bs = p1.frames["p1_ball_state"].astype(object).to_numpy()

        # Who last had the ball, and when each team lost it (P1's possession, ignoring contested and dead moments).
        poss = p1.frames["p1_possession"].astype(object).to_numpy()
        last = np.full(n, None, dtype=object)
        cur = None
        lost_at = {t: np.full(n, np.nan) for t in TEAMS}
        change_ts = {t: [] for t in TEAMS}
        last_loss = dict.fromkeys(TEAMS, np.nan)
        for i in range(n):
            if poss[i] in TEAMS and poss[i] != cur:
                if cur is not None:
                    last_loss[cur] = ts[i]
                    change_ts[cur].append(ts[i])
                    change_ts[OTHER[cur]].append(ts[i])
                cur = poss[i]
            last[i] = cur
            for t in TEAMS:
                lost_at[t][i] = last_loss[t]
        change_ts = {t: np.asarray(v, float) for t, v in change_ts.items()}

        engaging_count = np.full(n, np.nan)
        carrier_key = np.full(n, -1, dtype=np.int64)  # who carries the ball: team index x 1000 + player column
        rows = []
        for t in TEAMS:
            d = OTHER[t]
            ctl = bs == f"controlled_{t}"
            ok = ctl & np.isfinite(ball).all(axis=1)
            j_t, j_d = TEAMS.index(t), TEAMS.index(d)
            A = np.stack([s_of[:, j_t][:, None] * (tf[t].depth - L / 2), s_of[:, j_t][:, None] * tf[t].width], axis=-1)
            D = np.stack([s_of[:, j_d][:, None] * (tf[d].depth - L / 2), s_of[:, j_d][:, None] * tf[d].width], axis=-1)
            da = np.where(np.isfinite(A).all(axis=-1), np.linalg.norm(A - ball[:, None, :], axis=-1), np.inf)
            carrier = np.where(ok & np.isfinite(da.min(axis=1)), np.argmin(np.where(np.isfinite(da), da, np.inf), axis=1), -1)
            c_xy = np.where((carrier >= 0)[:, None], A[np.arange(n), np.clip(carrier, 0, None)], np.nan)
            dist = np.linalg.norm(D - c_xy[:, None, :], axis=-1)  # (n, defenders), NaN where either is missing
            # Outfield only: the goalkeeper of each window is excluded.
            gk = np.array([tf[d].ids.index(keepers[(d, int(per[i]), int(block[i]))])
                           if keepers.get((d, int(per[i]), int(block[i]))) in tf[d].ids else -1 for i in range(n)])
            outfield = np.ones(dist.shape, bool)
            outfield[np.arange(n), np.clip(gk, 0, None)] &= gk < 0
            # Closing speed and running back over the lag, with the same carrier and period at both ends.
            same = np.zeros(n, bool)
            same[lag:] = (carrier[lag:] >= 0) & (carrier[lag:] == carrier[:-lag]) & (per[lag:] == per[:-lag]) & \
                (ts[lag:] - ts[:-lag] <= lag * step * 1.5)
            closing = np.full(dist.shape, np.nan)
            back = np.full(dist.shape, np.nan)
            dt = np.r_[np.full(lag, np.nan), ts[lag:] - ts[:-lag]][:, None]
            closing[lag:] = (dist[:-lag] - dist[lag:]) / dt[lag:]
            back[lag:] = (tf[d].depth[:-lag] - tf[d].depth[lag:]) / dt[lag:]
            closing[~same] = np.nan
            eng = (dist <= cfg["p4v_engage_radius_m"]) & (closing >= cfg["p4v_engage_closing_mps"]) & outfield
            min_pts = max(1, int(round(cfg["p4v_engage_min_s"] / step)))
            ball_d = s_of[:, j_d] * ball[:, 0] + L / 2  # the ball's depth in the defending team's frame
            kept = np.zeros(dist.shape, bool)
            for j, pid in enumerate(tf[d].ids):
                for a, b in runs(eng[:, j]):
                    if b - a + 1 < min_pts:
                        continue
                    kept[a:b + 1, j] = True
                    since = ts[a] - lost_at[d][a]
                    rows.append({"team": d, "player": pid, "period": int(per[a]), "start_ts": float(ts[a]), "end_ts": float(ts[b]),
                                 "type": type_engagement(back[a, j], since, ball_d[a], L, cfg),
                                 "start_distance_m": round(float(dist[a, j]), 2), "closing_mps": round(float(closing[a, j]), 2),
                                 "back_mps": round(float(back[a, j]), 2) if np.isfinite(back[a, j]) else None,
                                 "since_loss_s": round(float(since), 2) if np.isfinite(since) else None})
            rows_t = carrier >= 0
            carrier_key[rows_t] = j_t * 1000 + carrier[rows_t]
            engaging_count[rows_t] = kept[rows_t].sum(axis=1)
        eng_cols = ["team", "player", "period", "start_ts", "end_ts", "type", "start_distance_m", "closing_mps", "back_mps", "since_loss_s"]
        engagements = pd.DataFrame(rows, columns=eng_cols).sort_values(["start_ts", "team", "player"], kind="stable").reset_index(drop=True)
        presses = pd.concat([chain_presses(engagements.loc[engagements["team"] == t], change_ts, cfg) for t in TEAMS], ignore_index=True)
        presses = presses.sort_values("start_ts", kind="stable").reset_index(drop=True)

        # The FIFA view of phases.
        f4 = p4.frames
        team4 = f4["p4_team"].astype(object).to_numpy()
        ph = f4["p4_phase"].astype(object).to_numpy()
        out4 = f4["p4_out_phase"].astype(object).to_numpy()
        press_by = {t: np.full(n, None, dtype=object) for t in TEAMS}
        for r in presses.itertuples():
            sel = (ts >= r.start_ts - 1e-9) & (ts <= r.end_ts + 1e-9)
            press_by[r.team][sel] = r.type
        cp_eng = {t: np.zeros(n, bool) for t in TEAMS}
        for r in engagements.loc[engagements["type"] == "counter_press"].itertuples():
            cp_eng[r.team] |= (ts >= r.start_ts - 1e-9) & (ts <= r.end_ts + 1e-9)
        pressure = p5.frames["p5_carrier_pressure"].to_numpy(dtype=float, na_value=np.nan)
        opposed_now = pressure >= cfg["p4v_opposed_min_pressure"]  # engagements as well over-call opposed (measured)
        # Held for p4v_opposed_hold_s, but only while the same player carries the ball: pressure on the previous
        # carrier does not make the next one's build-up opposed.
        hold = max(1, int(round(cfg["p4v_opposed_hold_s"] / step)))
        opposed = opposed_now.copy()
        for k in range(1, min(hold, n)):
            same_carrier = (carrier_key[k:] >= 0) & (carrier_key[k:] == carrier_key[:-k]) & (per[k:] == per[:-k])
            opposed[k:] |= opposed_now[:-k] & same_carrier

        phase = np.full(n, None, dtype=object)
        out_phase = np.full(n, None, dtype=object)
        for i in range(n):
            if ph[i] is None or ph[i] not in IN_FIFA:
                continue
            phase[i] = ("build_up_opposed" if opposed[i] else "build_up_unopposed") if ph[i] == "build_up" else IN_FIFA[ph[i]]
            d = OTHER.get(team4[i])
            base = OUT_FIFA.get(out4[i])
            if d is None or base is None:
                continue
            if press_by[d][i] == "counter_press" or cp_eng[d][i]:
                out_phase[i] = "counter_press"
            elif press_by[d][i] == "press" and base in PRESS_OF_BLOCK:
                out_phase[i] = PRESS_OF_BLOCK[base]
            else:
                out_phase[i] = base

        # FIFA's published block rule, for the team out of possession.
        fifa = np.full(n, None, dtype=object)
        vmax = cfg["p4v_fifa_max_speed_kmh"] / 3.6
        for d in TEAMS:
            t = OTHER[d]
            j_d = TEAMS.index(d)
            dep = tf[d].depth
            gk = np.array([tf[d].ids.index(keepers[(d, int(per[i]), int(block[i]))])
                           if keepers.get((d, int(per[i]), int(block[i]))) in tf[d].ids else -1 for i in range(n)])
            outfield = np.isfinite(dep)
            outfield[np.arange(n), np.clip(gk, 0, None)] &= gk < 0
            v = np.full(dep.shape, np.nan)
            v[lag:] = np.abs(dep[lag:] - dep[:-lag]) / np.r_[ts[lag:] - ts[:-lag]][:, None]
            still = np.where(outfield, np.nan_to_num(v, nan=np.inf) <= vmax, True).all(axis=1) & (outfield.sum(axis=1) >= cfg["p4v_fifa_min_players"])
            ball_d = s_of[:, j_d] * ball[:, 0] + L / 2
            behind = (outfield & (dep < ball_d[:, None])).sum(axis=1) >= cfg["p4v_fifa_min_behind"]
            in_def = (outfield & (dep < L / 3)).sum(axis=1)
            in_mid_def = (outfield & (dep < 2 * L / 3)).sum(axis=1)
            k = int(cfg["p4v_fifa_min_players"])
            has_ball = (team4 == t) & np.isfinite(ball_d)
            base_ok = has_ball & still & behind
            low = held(base_ok & (in_def >= k), ts, cfg["p4v_fifa_hold_s"], cfg["p4v_fifa_window_s"])
            mid = held(base_ok & (in_mid_def >= k), ts, cfg["p4v_fifa_hold_s"], cfg["p4v_fifa_window_s"]) & ~low
            fifa[has_ball & mid] = "mid_block"
            fifa[has_ball & low] = "low_block"

        frames = pd.DataFrame({"p4v_phase": phase, "p4v_out_phase": out_phase,
                               "p4v_engaging_defenders": pd.array(np.where(np.isfinite(engaging_count), engaging_count, np.nan), dtype="Float64").astype("Int64"),
                               "p4v_fifa_block": fifa})
        p2_state = next((x.state for x in p2.entries if x.metric == "team_shape"), "blocked")
        state = "blocked" if not np.isfinite(engaging_count).any() else ("full" if p2_state == "full" else "degraded")
        measurements = {
            "engagements": int(len(engagements)),
            "engagements_by_type": {k: int((engagements["type"] == k).sum()) for k in ENGAGEMENT_TYPES},
            "presses": int(len(presses)), "counter_presses": int((presses["type"] == "counter_press").sum()) if len(presses) else 0,
            "fifa_block_share": {k: float((fifa == k).mean()) if n else 0.0 for k in ("mid_block", "low_block")},
        }
        why = {"full": "From tracking: every defender's distance to the carrier and closing speed.",
               "degraded": "Players are out of view at times (team shape degraded), so engagements are missed there.",
               "blocked": "No team controls a detected ball with players observed, so no engagement can be measured."}[state]
        entries = [
            self.entry("engagements", "Engagements and presses", state, why + " Estimated, not yet validated.",
                       coverage=f"{len(engagements)} engagements, {len(presses)} presses"),
            self.entry("fifa_view", "FIFA view of phases", "degraded" if state == "blocked" else state,
                       "P4's phases by FIFA's names, with build-up opposed or unopposed and press against block. Estimated, not yet validated.",
                       coverage=f"{int(pd.notna(phase).sum())} grid points with a phase"),
            self.entry("fifa_block", "FIFA's mid-block rule", "blocked" if state == "blocked" else "full",
                       "FIFA's published thresholds (World Cup 2022); the horizontal-lines condition is not checked. Beside P4's block.",
                       coverage=f"{int(pd.notna(fifa).sum())} grid points in a FIFA block"),
        ]
        return PackageOutput(frames=frames, entries=entries, segments={"engagements": engagements, "presses": presses},
                             measurements=measurements)
