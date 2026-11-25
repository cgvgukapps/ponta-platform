"""
space_value.py — context package P7: space, value and game-state context.

Stage C (``ponta-blueprint/context-relationships-task.md``, deliverable 4). Three parts, on Plaiq's honeycomb:

- **Space.** Every ``p7_space_every_s`` (a measured rate: Plaiq's pitch control costs about 25 ms per frame over the
  whole grid), on grid points with the ball detected: how much of each team's **high-value space** (the cells in the
  top ``p7_high_value_share`` of its threat surface, ``hex_value.xT_surface``) the other team does **not** control,
  weighted by threat, from Plaiq's per-frame pitch control (``MatchAnalysis.frame_control``: each cell +1 home, −1
  away, 0 nobody). (A team rarely controls its own high-value space outright: it lies near the opponent's goal, where
  the defenders control a median 57% and nobody 38%; so the open share is the informative one.) And, for the team
  with the ball, the **most dangerous space the defence does not control** as a lane and third in that team's frame
  (never a cell index).
- **The game-state risk weight** ``λ`` for the team with the ball, a versioned lookup by goal difference (P1's score)
  and time: level 1; leading ``p7_lambda_leading`` (from ``p7_late_minute`` ``p7_lambda_leading_late``); trailing
  ``p7_lambda_trailing`` (late ``p7_lambda_trailing_late``). The values are a judgement (a leading team weighs losing
  the ball more, a chasing team accepts risk), labelled estimated until an expert study measures them. With the score
  unknown ``λ`` is 1 and the Data check says so.
- **Context-adjusted Choice**, a new score beside Choice (which stays the default): from Choice's own parts at each
  pass (``MatchAnalysis.choice_options``: every option's ring-1 cells, completion probability ``p`` and value), each
  cell is worth ``U = p × value − w × λ × (1 − p) × cost``, where ``cost`` is the other team's threat at that cell (what
  losing the ball there gives away) and ``w`` is ``p7_cost_weight`` (0.5 proposed; at 1 the score leans hard to safe
  passes, at 0 it is Choice). An option is its best cell; the score is the chosen option's ``U`` over the best
  option's, when that is positive. The weight is named in the segment row. Since 0.3.0 (25 November 2026, review
  SM-06) the score stays on its 0 to 1 scale: where the chosen option's ``U`` is negative (worth less than its risk)
  it is 0 with that reason in the segment row; before, the ratio went negative (to about -50 on one match).

Estimated, not yet validated (Gate 3b, the expert study, decides whether the adjusted score is ever shown first).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from plaiq_engine.context.registry import (
    ContextField,
    ContextInput,
    ContextPackage,
    PackageOutput,
)
from plaiq_engine.context.structure import OTHER, TEAMS
from plaiq_engine.context.zones import lane_of

THIRDS = ("defensive", "middle", "attacking")


def game_state_weight(goal_diff: float, minute: float, cfg: dict) -> float:
    """``λ`` for a team from its goal difference and the minute; 1 when the score is unknown."""
    if goal_diff is None or not np.isfinite(goal_diff):
        return 1.0
    late = np.isfinite(minute) and minute >= cfg["p7_late_minute"]
    if goal_diff > 0:
        return cfg["p7_lambda_leading_late"] if late else cfg["p7_lambda_leading"]
    if goal_diff < 0:
        return cfg["p7_lambda_trailing_late"] if late else cfg["p7_lambda_trailing"]
    return 1.0


#: Why an adjusted Choice is 0 (P7 0.3.0).
CHOSEN_BELOW_RISK = "the chosen option was worth less than its risk: adjusted Choice is 0"


def adjusted_choice(opts: dict, cost_surface: np.ndarray, lam: float, cost_weight: float = 1.0) -> dict:
    """
    Context-adjusted Choice for one pass from Choice's options: each cell's ``U = p × value − w × λ × (1 − p) × cost``; an
    option is its best cell; the chosen option's ``U`` over the best option's when that is positive.
    """
    us = []
    for cs, p, v in zip(opts["cells"], opts["p"], opts["values"], strict=True):
        p, v = np.asarray(p, float), np.asarray(v, float)
        u = p * v - cost_weight * lam * (1.0 - p) * cost_surface[np.asarray(cs, int)]
        us.append(float(u.max()) if len(u) else -np.inf)
    us = np.array(us)
    best, chosen = float(us.max()), float(us[opts["chosen"]])
    if best <= 0:
        return {"adjusted": np.nan, "u_chosen": chosen, "u_best": best, "reason": "no option is worth more than its risk"}
    if chosen < 0:
        # P7 0.3.0 (review SM-06): a ratio of a negative to a positive utility is not on the 0 to 1 scale; the chosen
        # option was worth less than its risk, so the score is 0 and the row says why.
        return {"adjusted": 0.0, "u_chosen": chosen, "u_best": best, "reason": CHOSEN_BELOW_RISK}
    return {"adjusted": min(chosen / best, 1.0), "u_chosen": chosen, "u_best": best, "reason": None}


@dataclass
class _Fields:
    items: tuple[ContextField, ...] = field(default=(
        ContextField("p7_home_high_value_open", "share", "estimated",
                     "The share of home's high-value space (the top p7_high_value_share of its threat surface) that away does not control, weighted by threat, from Plaiq's pitch control. Every p7_space_every_s with the ball detected; null elsewhere."),
        ContextField("p7_away_high_value_open", "share", "estimated", "The same for away (the share home does not control)."),
        ContextField("p7_open_danger_zone", "label", "estimated",
                     "For the team with the ball: the lane and third (its own frame) of the most threatening cell the defence does not control. Never a cell index. At the same grid points as the control shares."),
        ContextField("p7_game_state_weight", "weight", "estimated",
                     "The game-state risk weight for the team with the ball (a versioned lookup by goal difference and time; a judgement, not yet measured). 1 when the score is unknown. Null when no team has the ball."),
        ContextField("p7_adjusted_choice", "0 to 1", "estimated",
                     "At the grid point where a pass starts: context-adjusted Choice, 0 to 1, beside Choice (the default); 0 when the chosen option was worth less than its risk. Null elsewhere, or when no option is worth more than its risk."),
    ))


class SpaceValuePackage(ContextPackage):
    name = "p7"
    version = "0.3.0"
    depends_on = ("p1", "p2")
    fields = _Fields().items

    def compute(self, ctx: ContextInput) -> PackageOutput:
        from plaiq_engine.match_analysis import (
            MatchAnalysis,  # Plaiq's control, threat surfaces and Choice's options
        )

        m, g, cfg = ctx.match, ctx.grid, ctx.config
        p1, p2 = ctx.upstream["p1"], ctx.upstream["p2"]
        n, L = len(g), m.pitch.length_m
        ts, per, nf = g["frame_ts"].to_numpy(float), g["period"].to_numpy(), g["native_frame"].to_numpy(np.int64)
        minute = g["minute"].to_numpy(float)
        sign = {(int(r.period), r.team): r.attack_sign for r in p2.segments["orientation"].itertuples()}
        f1 = p1.frames
        bs = f1["p1_ball_state"].astype(object).to_numpy()
        attacking = np.full(n, None, dtype=object)
        for t in TEAMS:
            attacking[np.isin(bs, [f"controlled_{t}", f"in_flight_{t}"])] = t
        score = {t: f1[f"p1_score_{t}"].to_numpy(dtype=float, na_value=np.nan) for t in TEAMS}
        gs_source = str(f1["p1_game_state_source"].iloc[0]) if n else "unknown"
        known = gs_source in ("events", "upload")

        # Game-state weight for the team with the ball, on every grid point.
        lam = np.full(n, np.nan)
        for i in np.flatnonzero(attacking != None):
            t = attacking[i]
            diff = score[t][i] - score[OTHER[t]][i] if known else np.nan
            lam[i] = game_state_weight(diff, minute[i], cfg)

        # One MatchAnalysis per match where the caller shares it (extras["analysis"]): Choice is then computed once.
        ma = ctx.extras.get("analysis") or MatchAnalysis(m)
        H = ma.hex_centers
        xt = {"home": ma.xt_home, "away": ma.xt_away}
        hv = {t: xt[t] >= np.quantile(xt[t], 1.0 - cfg["p7_high_value_share"]) for t in TEAMS}
        sgn = {"home": 1, "away": -1}
        hv_ctl = {t: np.full(n, np.nan) for t in TEAMS}
        danger = np.full(n, None, dtype=object)
        frames_ok = set(ma.home.index) & set(ma.away.index)
        det = f1["p1_ball_detected"].to_numpy(bool)
        last_t = -np.inf
        for i in range(n):
            if ts[i] - last_t < cfg["p7_space_every_s"] - 1e-9 or not det[i] or int(nf[i]) not in frames_ok:
                continue
            last_t = ts[i]
            control = ma.frame_control(int(nf[i]))[0]
            if not np.any(control):  # the engine returns an uncontrolled grid when it has no ball at that frame
                continue
            for t in TEAMS:
                w = xt[t][hv[t]]
                hv_ctl[t][i] = float((w * (control[hv[t]] != sgn[OTHER[t]])).sum() / w.sum())
            t = attacking[i]
            if t is not None:
                open_ = control != sgn[OTHER[t]]
                if open_.any():
                    c = int(np.argmax(np.where(open_, xt[t], -np.inf)))
                    s = sign.get((int(per[i]), t), 1)
                    depth, width = s * H[c, 0] + L / 2, s * H[c, 1]
                    lane = lane_of(np.array([width]))[0]
                    third = THIRDS[min(2, int(depth // (L / 3)))]
                    danger[i] = f"{lane}_{third}"

        # Context-adjusted Choice from Choice's own options.
        d2 = ma.decision_quality_v2.set_index("event_idx")
        rows, adj = [], np.full(n, np.nan)
        for idx, opts in ma.choice_options.items():
            e = ma.events.loc[idx]
            k = int(np.clip(np.searchsorted(nf, int(e["Start Frame"])), 0, n - 1))
            t = opts["team"].lower()
            diff = score[t][k] - score[OTHER[t]][k] if known else np.nan
            weight = game_state_weight(diff, minute[k], cfg)
            r = adjusted_choice(opts, xt[OTHER[t]], weight, cfg["p7_cost_weight"])
            base = d2.at[idx, "dq_v2"] if idx in d2.index else np.nan
            rows.append({"event_idx": int(idx), "team": t, "ts": float(ts[k]), "choice": round(float(base), 4) if np.isfinite(base) else None,
                         "adjusted": round(float(r["adjusted"]), 4) if np.isfinite(r["adjusted"]) else None, "weight": weight,
                         "cost_weight": cfg["p7_cost_weight"],
                         "goal_diff": None if not np.isfinite(diff) else int(diff), "minute": int(minute[k]),
                         "game_state": "known" if known else "unknown", "reason": r["reason"]})
            if np.isfinite(r["adjusted"]) and not np.isfinite(adj[k]):
                adj[k] = r["adjusted"]
        segs = pd.DataFrame(rows, columns=["event_idx", "team", "ts", "choice", "adjusted", "weight", "cost_weight", "goal_diff", "minute",
                                           "game_state", "reason"]).sort_values("event_idx", kind="stable").reset_index(drop=True)

        frames = pd.DataFrame({"p7_home_high_value_open": np.round(hv_ctl["home"], 4),
                               "p7_away_high_value_open": np.round(hv_ctl["away"], 4),
                               "p7_open_danger_zone": danger, "p7_game_state_weight": np.round(lam, 3),
                               "p7_adjusted_choice": np.round(adj, 4)})
        sampled = np.isfinite(hv_ctl["home"])
        measurements = {
            "space_points": int(sampled.sum()),
            "high_value_open_median": {t: float(np.nanmedian(hv_ctl[t])) if sampled.any() else None for t in TEAMS},
            "passes_adjusted": int(segs["adjusted"].notna().sum()), "passes": int(len(segs)),
            "passes_chosen_below_risk": int((segs["reason"] == CHOSEN_BELOW_RISK).sum()),
            "adjusted_minus_choice_median": float((segs["adjusted"].astype(float) - segs["choice"].astype(float)).median()) if len(segs) else None,
            "game_state": "known" if known else "unknown",
        }
        p2_state = next((x.state for x in p2.entries if x.metric == "team_shape"), "blocked")
        space_state = "blocked" if not sampled.any() else ("full" if p2_state == "full" else "degraded")
        entries = [
            self.entry("space", "Space and value", space_state,
                       {"full": "Open high-value space comes from Plaiq's pitch control on the honeycomb, sampled every few seconds.",
                        "degraded": "Computed, but players are out of view at times (team shape degraded), so control is approximate there.",
                        "blocked": "The ball is never detected with both teams' positions, so no control surface."}[space_state]
                       + " Estimated, not yet validated.",
                       coverage=f"{int(sampled.sum())} sampled grid points"),
            self.entry("adjusted_choice", "Context-adjusted Choice", "degraded" if not known or not len(segs) else "full",
                       ("The game-state weight uses the score and minute; its values are a judgement until the expert study."
                        if known else "The score is unknown, so the game-state weight is neutral (1): the adjusted score adds only the cost of losing the ball.")
                       + " Choice stays the default; the adjusted score is shown beside it.",
                       unlock=None if known else "Add the logged event feed, or upload goal times and red cards.",
                       coverage=f"{int(segs['adjusted'].notna().sum())} of {len(segs)} scored passes"),
        ]
        return PackageOutput(frames=frames, entries=entries, segments={"adjusted_choice": segs}, measurements=measurements)
