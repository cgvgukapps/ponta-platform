"""
zones.py — context package P3: relative zones.

Where the ball and the attacking players are relative to the defending team, on every grid
point where one team has the ball (``ponta-blueprint/context-packages.md``, P3;
``ponta-blueprint/context-structure-task.md``). The attacking team is P1's team in
possession; the defending team's lines come from P2.

* **Lane** of the ball, in the attacking team's frame (its left positive): wings outside the
  penalty-area edges (more than 20.16 m from the centre line), half-spaces between the
  penalty-area and goal-area edges (9.16 to 20.16 m), centre inside the goal-area edges. The
  edges are the Laws of the Game's fixed markings; the five-lane split is a coaching
  convention, not a provider definition.
* **Third** of the ball along the attacking direction (thirds of the pitch length).
* **Ball against the defending lines**, measured as depth from the defending team's goal
  line: ``behind_defence`` (beyond the deepest outfield defender), ``in_defensive_line``
  (between the deepest defender and the defensive line's centroid), ``between_lines``
  (between the defensive and midfield lines' centroids), ``between_mid_att`` (between the
  midfield and attacking lines' centroids), ``in_front`` (ahead of the attacking line's
  centroid). FIFA's lines use the deepest member; using centroids for the bands between
  lines is Ponta's choice.
* **Inside the defending team's hull** (its outfield players' convex hull).
* **The pocket**: between the lines, in a half-space and inside the defending hull. There is
  no published definition; this one is Ponta's.
* **Counts** of the attacking team's outfield players behind the defence, between the lines
  and in the pocket.

All labels are estimated (rules on positions), not yet validated. Nothing here reaches the AI
as coordinates.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

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
    line_members,
    team_frames,
    team_states,
    window_index,
    windows_from_segments,
)

PENALTY_EDGE_M = 20.16  # half of the penalty area's 40.32 m width
GOAL_AREA_EDGE_M = 9.16  # half of the goal area's 18.32 m width

BANDS = ("behind_defence", "in_defensive_line", "between_lines", "between_mid_att", "in_front")
LANES = ("left_wing", "left_half_space", "centre", "right_half_space", "right_wing")
THIRDS = ("defensive", "middle", "attacking")


def lane_of(width: np.ndarray) -> np.ndarray:
    """Five lanes from the width (left positive); None where the width is unknown."""
    out = np.full(len(width), None, dtype=object)
    w = np.asarray(width, float)
    ok = np.isfinite(w)
    out[ok & (w > PENALTY_EDGE_M)] = "left_wing"
    out[ok & (w > GOAL_AREA_EDGE_M) & (w <= PENALTY_EDGE_M)] = "left_half_space"
    out[ok & (np.abs(w) <= GOAL_AREA_EDGE_M)] = "centre"
    out[ok & (w < -GOAL_AREA_EDGE_M) & (w >= -PENALTY_EDGE_M)] = "right_half_space"
    out[ok & (w < -PENALTY_EDGE_M)] = "right_wing"
    return out


def band_of(depth: np.ndarray, deepest: np.ndarray, c_def: np.ndarray, c_mid: np.ndarray, c_att: np.ndarray) -> np.ndarray:
    """Band against the defending lines; all depths from the defending team's own goal line."""
    d = np.asarray(depth, float)
    out = np.full(d.shape, None, dtype=object)
    ok = np.isfinite(d) & np.isfinite(deepest) & np.isfinite(c_def) & np.isfinite(c_mid) & np.isfinite(c_att)
    out[ok & (d < deepest)] = "behind_defence"
    out[ok & (d >= deepest) & (d < c_def)] = "in_defensive_line"
    out[ok & (d >= c_def) & (d < c_mid)] = "between_lines"
    out[ok & (d >= c_mid) & (d < c_att)] = "between_mid_att"
    out[ok & (d >= c_att)] = "in_front"
    return out


def _inside(eq: np.ndarray | None, pts: np.ndarray) -> np.ndarray:
    if eq is None:
        return np.zeros(len(pts), dtype=bool)
    return np.all(pts @ eq[:, :2].T + eq[:, 2] <= 1e-9, axis=1)


@dataclass
class _Fields:
    items: tuple[ContextField, ...] = field(default=(
        ContextField("p3_attacking_team", "label", "estimated", "The team in possession (P1) that zones are measured for; null when neither team has the ball."),
        ContextField("p3_ball_lane", "label", "estimated",
                     "The ball's lane in the attacking team's frame: left_wing, left_half_space, centre, right_half_space, right_wing (penalty-area and goal-area edges)."),
        ContextField("p3_ball_third", "label", "estimated", "The ball's third along the attacking direction: defensive, middle, attacking."),
        ContextField("p3_ball_band", "label", "estimated",
                     "The ball against the defending team's lines: behind_defence, in_defensive_line, between_lines, between_mid_att, in_front. Bands between lines use line centroids (Ponta's choice). Estimated, not yet validated."),
        ContextField("p3_ball_in_hull", "bool", "estimated", "The ball is inside the convex hull of the defending team's outfield players."),
        ContextField("p3_ball_in_pocket", "bool", "estimated",
                     "The ball is between the lines, in a half-space and inside the defending hull. Ponta's definition (none is published). Estimated, not yet validated."),
        ContextField("p3_att_behind_defence", "players", "estimated", "Attacking outfield players beyond the defending team's deepest outfield player."),
        ContextField("p3_att_between_lines", "players", "estimated", "Attacking outfield players between the defending team's defensive and midfield line centroids."),
        ContextField("p3_att_in_pocket", "players", "estimated", "Attacking outfield players in the pocket (between the lines, in a half-space, inside the defending hull)."),
    ))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


class ZonesPackage(ContextPackage):
    name = "p3"
    version = "0.1.0"
    depends_on = ("p1", "p2")
    fields = _Fields().items

    def compute(self, ctx: ContextInput) -> PackageOutput:
        m, g, cfg = ctx.match, ctx.grid, ctx.config
        p1, p2 = ctx.upstream["p1"], ctx.upstream["p2"]
        n = len(g)
        L = m.pitch.length_m
        signs = p2.segments["orientation"]
        tf = team_frames(m, g, signs)
        states = team_states(g, p1.frames, p1.segments.get("chains", pd.DataFrame()), cfg)
        block = window_index(g, cfg)
        lines = line_members(g, tf, states, block, windows_from_segments(p2.segments))

        poss = p1.frames["p1_possession"].to_numpy(object)
        att = np.array([v if v in TEAMS else None for v in poss], dtype=object)
        ball = ball_on_grid(m, g)
        bx, by = ball["ball_x"].to_numpy(float), ball["ball_y"].to_numpy(float)
        per = g["period"].to_numpy()
        sign_of = {(int(r.period), r.team): r.attack_sign for r in signs.itertuples()}

        lane = np.full(n, None, dtype=object)
        third = np.full(n, None, dtype=object)
        band = np.full(n, None, dtype=object)
        in_hull = pd.array([None] * n, dtype="boolean")
        in_pocket = pd.array([None] * n, dtype="boolean")
        cnt = {k: pd.array([None] * n, dtype="Int64") for k in ("behind", "between", "pocket")}
        min_n = int(cfg["p2_min_outfield_players"])

        for team in TEAMS:
            rows = np.where(att == team)[0]
            if not len(rows):
                continue
            dfd = OTHER[team]
            s_att = np.array([sign_of.get((int(p), team), 1 if team == "home" else -1) for p in per[rows]], float)
            # Ball in the attacking team's frame, and as depth from the defending team's goal line.
            a_depth = s_att * bx[rows] + L / 2
            a_width = s_att * by[rows]
            d_depth = L - a_depth
            lane[rows] = lane_of(a_width)
            t_idx = np.clip(np.floor(a_depth / (L / 3)), 0, 2)
            third[rows] = np.where(np.isfinite(a_depth), np.array(THIRDS, dtype=object)[np.nan_to_num(t_idx).astype(int)], None)

            D, W, Ld = tf[dfd].depth[rows], tf[dfd].width[rows], lines[dfd][rows]
            outf = np.isfinite(D) & (Ld != 3)
            enough = outf.sum(axis=1) >= min_n
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                deepest = np.nanmin(np.where(outf, D, np.nan), axis=1)
                cents = [np.nanmean(np.where(outf & (Ld == k), D, np.nan), axis=1) for k in range(3)]
            deepest = np.where(enough, deepest, np.nan)
            band[rows] = band_of(d_depth, deepest, *cents)

            # Attacking players as depth from the defending goal line (outfield only).
            Da, Wa, La = tf[team].depth[rows], tf[team].width[rows], lines[team][rows]
            a_out = np.isfinite(Da) & (La != 3)
            ad = L - Da  # attacker depth from the defending team's goal line
            # Attacking width in the attacking frame; lanes are judged there.
            a_bands = band_of(ad, deepest[:, None], cents[0][:, None], cents[1][:, None], cents[2][:, None])
            a_half = (np.abs(Wa) > GOAL_AREA_EDGE_M) & (np.abs(Wa) <= PENALTY_EDGE_M)

            for k, i in enumerate(rows):
                if not enough[k]:
                    continue
                pts = np.column_stack([D[k, outf[k]], W[k, outf[k]]])
                try:
                    eq = ConvexHull(pts).equations
                except QhullError:
                    eq = None
                # The defending frame mirrors the attacking frame: depth d -> L - d, width w -> -w.
                if np.isfinite(d_depth[k]):
                    inside = bool(_inside(eq, np.array([[d_depth[k], -a_width[k]]]))[0])
                    in_hull[i] = inside
                    in_pocket[i] = inside and band[i] == "between_lines" and lane[i] in ("left_half_space", "right_half_space")
                sel = a_out[k]
                cnt["behind"][i] = int((a_bands[k][sel] == "behind_defence").sum())
                btw = sel & (a_bands[k] == "between_lines")
                cnt["between"][i] = int(btw.sum())
                if btw.any():
                    ins = _inside(eq, np.column_stack([ad[k, btw], -Wa[k, btw]]))
                    cnt["pocket"][i] = int((ins & a_half[k, btw]).sum())
                else:
                    cnt["pocket"][i] = 0

        frames = pd.DataFrame({
            "p3_attacking_team": att, "p3_ball_lane": lane, "p3_ball_third": third, "p3_ball_band": band,
            "p3_ball_in_hull": in_hull, "p3_ball_in_pocket": in_pocket,
            "p3_att_behind_defence": cnt["behind"], "p3_att_between_lines": cnt["between"], "p3_att_in_pocket": cnt["pocket"],
        })

        # -- measurements and the Data check ---------------------------------------
        has_team = att != None
        labelled = has_team & (band != None)
        coverage = float(labelled.sum() / max(int(has_team.sum()), 1))
        lab = pd.Series(band[labelled])
        measurements = {
            "coverage_of_possession_time": round(coverage, 4),
            "ball_band_share": {b: round(float((lab == b).mean()), 4) for b in BANDS} if len(lab) else {},
            "ball_lane_share": {ln: round(float((pd.Series(lane[has_team]) == ln).mean()), 4) for ln in LANES},
            "ball_in_pocket_share": round(float(np.nanmean(in_pocket.to_numpy(dtype=float, na_value=np.nan)[labelled])), 4) if labelled.any() else None,
            "mean_attackers_between_lines": round(float(np.nanmean(cnt["between"].to_numpy(dtype=float, na_value=np.nan))), 3) if has_team.any() else None,
        }
        p2_state = next((e.state for e in p2.entries if e.metric == "team_shape"), "blocked")
        if p2_state == "blocked" or coverage == 0.0:
            state, why = "blocked", "Zones need both teams' shapes and a detected ball; neither is available."
        elif p2_state == "degraded" or coverage < cfg["p3_min_coverage"]:
            state = "degraded"
            why = ("Team shape is degraded for this match, so zones are missing where players are out of view." if p2_state == "degraded"
                   else f"The ball or the defending shape is missing for more than {_pct(1 - cfg['p3_min_coverage'])} of possession time.")
        else:
            state, why = "full", "The ball and the defending team's shape are available for nearly all possession time."
        entries = [self.entry("relative_zones", "Ball and players against the defending lines", state,
                              why + " Bands and the pocket are Ponta's rules: estimated, not yet validated.",
                              coverage=f"{_pct(coverage)} of possession time",
                              unlock=None if state == "full" else "Full-pitch tracking with a detected ball.")]
        return PackageOutput(frames=frames, entries=entries, measurements=measurements)
