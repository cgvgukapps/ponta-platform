"""
line_breaks.py — context package P5 (first part): line breaks.

Which of the opposition's lines (units) each pass broke, after the definitions FIFA publishes in
its Football Language and Enhanced Football Intelligence documents (``ponta-blueprint/
fifa-alignment.md``); Ponta's own rule and wording:

- **units** are the defending team's outfield lines at the moment of the pass: the observed
  outfield players grouped by depth (exact one-dimensional 3-means, ``structure.assign_lines``).
  A unit counts only with at least ``p5_min_unit_players`` players;
- a unit's **line** is its deepest player (nearest its own goal). A pass **breaks** the unit when
  the ball starts in front of that line and is received beyond it;
- **through** when the ball crosses the line between the unit's two widest players, **around**
  otherwise (FIFA's third type, over, needs the ball's height and is not labelled);
- units are named from the attacking team's view: the first met is ``first``, the deepest
  ``last``, and with three units the middle one ``second_last``.

Passes are the event feed's completed passes (``PASS``) with start and end positions. A failed
pass (``BALL LOST``) is listed with its line breaks unknown: FIFA judges it by where the intended
team-mate was, and Ponta has no target for it yet (Plaiq's failed-pass inference finds the nearest
viable receiver, not the target; ``CONTEXT_STRUCTURE_REPORT.md``, line breaks).

Measured against SkillCorner's line-break labels on its 20 open-data matches: any line broken
κ 0.68 / 0.66 (fit / held out), furthest line κ 0.52; completed passes 80% on the furthest line.
Estimated, not yet validated by analysts.

**Version 0.2.0 (Stage C, 20 November 2026):**
- **failed passes** are judged by the logged target's point at the ball's arrival: the target's run
  projected from the moment of the pass (``pass_probability.projected_target``, the moment and run
  Choice uses for the chosen option). A failed pass without a logged target stays unknown;
- **received inside or outside the shape:** whether that point (completed: the pass's end) is inside the
  convex hull of the defending outfield players at the moment of the pass;
- **line-breaking options:** at each pass, the team-mates (not the passer) whose projected run lands beyond
  at least one unit and whose pass completion probability (``pass_probability``, adopted) is at least
  ``p5_lb_option_min_p``. Their count is per pass.
- **open passing lanes:** at each pass, team-mates whose pass completion probability is at least
  ``p5_open_lane_min_p``;
- **relationships on every grid point** (``context/relationships.py``): carrier pressure, goal-side defenders,
  the offside line, free attackers, the overload at the ball and the marking distance.
The moment of the pass comes from ``MatchAnalysis`` (the ball tracked, else the passer; players at the start
frame), as for the pass completion probability and Choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from plaiq_engine.analytics.decision_quality_v2 import option_probabilities
from plaiq_engine.analytics.pass_option_value import _build_defender_data
from plaiq_engine.analytics.pass_probability import (
    PassModel,
    parameters_for,
    projected_target,
)
from plaiq_engine.context.possession import ball_on_grid
from plaiq_engine.context.registry import (
    ContextField,
    ContextInput,
    ContextPackage,
    PackageOutput,
)
from plaiq_engine.context.relationships import efi_aggregates, relationships_on_grid
from plaiq_engine.context.structure import (
    OTHER,
    assign_lines,
    team_frames,
    window_index,
    windows_from_segments,
)

UNIT_ORDER = (2, 1, 0)  # attack, midfield, defence: the order the attacking team meets them


def units_at(depth: np.ndarray, width: np.ndarray, length: float, keeper: int | None) -> dict[int, np.ndarray]:
    """
    The defending team's outfield players at one moment, grouped into three units by depth and
    given in the attacking team's frame (depth towards the defending team's goal, width).
    ``depth`` and ``width`` are the defending team's own frame; ``keeper`` (P2's goalkeeper for the
    window, by column) is left out.
    """
    outf = np.isfinite(depth)
    if keeper is not None:
        outf[keeper] = False
    out = {c: np.empty((0, 2)) for c in UNIT_ORDER}
    if outf.sum() < 3:
        return out
    lab = assign_lines(depth[outf])
    d, w = depth[outf], width[outf]
    for c in UNIT_ORDER:
        sel = lab == c
        out[c] = np.column_stack([length - d[sel], -w[sel]])
    return out


def breaks(start: tuple[float, float], end: tuple[float, float], units: dict[int, np.ndarray], min_players: int = 2) -> dict:
    """
    Units broken by a ball moving from ``start`` to ``end`` (attacking frame). Per unit: None when
    it has fewer than ``min_players`` players, False when not broken, else ``through`` or ``around``.
    """
    out = {}
    (d0, w0), (d1, w1) = start, end
    for code, pts in units.items():
        if len(pts) < min_players:
            out[code] = None
            continue
        line = float(pts[:, 0].max())  # the unit's deepest player, nearest its own goal
        if not (d0 < line <= d1):
            out[code] = False
            continue
        u = (line - d0) / (d1 - d0) if d1 != d0 else 0.0
        w = w0 + u * (w1 - w0)
        out[code] = "through" if pts[:, 1].min() <= w <= pts[:, 1].max() else "around"
    return out


def inside_shape(units: dict[int, np.ndarray], point: tuple[float, float]) -> bool | None:
    """Whether ``point`` (attacking frame) is inside the convex hull of the defending outfield players; None if no hull."""
    pts = np.vstack([u for u in units.values() if len(u)]) if any(len(u) for u in units.values()) else np.empty((0, 2))
    if len(pts) < 3:
        return None
    try:
        eq = ConvexHull(pts).equations
    except QhullError:
        return None
    return bool((eq[:, :2] @ np.asarray(point, float) + eq[:, 2] <= 1e-9).all())


def pass_moment_options(ma, e: pd.Series, model: PassModel) -> dict | None:
    """
    The moment of one pass from ``MatchAnalysis`` (absolute coordinates): the ball, the logged target's projected
    point, and every team-mate's projected point with its completion probability. None without a moment.
    """
    m = ma._pass_moment(e)
    if isinstance(m, str):
        return None
    names, mates = [], []
    for j in m["mates_jerseys"]:
        d = _build_defender_data(m["own"], m["team"], [j], m["profiles"])
        if d:
            names.append(str(j))
            mates.append(d[0])
    targets, p = option_probabilities(m["ball"], mates, m["defenders"], model)
    target = projected_target(m["ball"], m["receiver"], m["velocity"], model.flight)
    return {"ball": m["ball"], "target": target, "options": list(zip(names, targets, p, strict=True))}


def unit_names(b: dict) -> dict[int, str]:
    """first / second_last / last for the units that count, from the attacking team's view."""
    valid = [c for c in UNIT_ORDER if b.get(c) is not None]
    if len(valid) == 3:
        return dict(zip(valid, ("first", "second_last", "last"), strict=True))
    if len(valid) == 2:
        return dict(zip(valid, ("first", "last"), strict=True))
    return {valid[0]: "last"} if valid else {}


def furthest(b: dict) -> str:
    """The deepest unit broken (``first`` / ``second_last`` / ``last``), or ``none``."""
    names = unit_names(b)
    for code in (0, 1, 2):
        if b.get(code):
            return names[code]
    return "none"


@dataclass
class _Fields:
    items: tuple[ContextField, ...] = field(default=(
        ContextField("p5_pass_units_broken", "count", "estimated",
                     "At the grid point where a completed pass starts: how many of the opposition's units (lines of at least two players, grouped at that moment) it broke. Null elsewhere. Estimated, not yet validated."),
        ContextField("p5_pass_furthest_line", "label", "estimated",
                     "At the grid point where a completed pass starts: the deepest unit it broke (first, second_last, last) or none; unknown for a failed pass (no target yet). Null elsewhere."),
        ContextField("p5_pass_line_break_type", "label", "estimated",
                     "How the deepest broken unit was broken: through (between its two widest players) or around. Null when no unit was broken."),
        ContextField("p5_pass_received_inside_shape", "bool", "estimated",
                     "At the grid point where a pass starts: whether the ball was received (a failed pass: the logged target's projected point) inside the convex hull of the defending outfield players at the moment of the pass. Null elsewhere. Since P5 0.2.0."),
        ContextField("p5_pass_line_breaking_options", "count", "estimated",
                     "At the grid point where a pass starts: team-mates (not the passer) whose projected run lands beyond at least one unit and whose pass completion probability is at least p5_lb_option_min_p. Null elsewhere. Since P5 0.2.0."),
        ContextField("p5_pass_open_lanes", "count", "estimated",
                     "At the grid point where a pass starts: team-mates (not the passer) whose pass completion probability is at least p5_open_lane_min_p. Null elsewhere. Since P5 0.2.0."),
        ContextField("p5_carrier_pressure", "1 to 10", "estimated",
                     "Plaiq's 1v1 pressure on the ball carrier (the controlling team's player nearest the ball), from the hex distance to the nearest opponent. Null when no team has the ball. Since P5 0.2.0."),
        ContextField("p5_goal_side_defenders", "players", "estimated",
                     "Defending outfield players nearer their own goal than the ball. Null when no team has the ball."),
        ContextField("p5_offside_line_m", "m", "estimated",
                     "The second-last defending player's distance from its own goal line (goalkeeper included). Null when no team has the ball."),
        ContextField("p5_free_attackers", "players", "estimated",
                     "Attacking outfield players with no defending player within p5_free_radius_m. Ponta's definition (none is published)."),
        ContextField("p5_overload_at_ball", "players", "estimated",
                     "Attacking minus defending outfield players within p5_overload_radius_m of the ball. Ponta's definition (none is published)."),
        ContextField("p5_marking_distance_m", "m", "estimated",
                     "Mean distance from each defending outfield player to the nearest attacker. Null when no team has the ball."),
    ))


class LineBreakPackage(ContextPackage):
    name = "p5"
    version = "0.2.0"
    depends_on = ("p1", "p2", "p3")  # P1 0.4.0's ball state decides where relationships apply; P3's zones feed EFI
    fields = _Fields().items

    def compute(self, ctx: ContextInput) -> PackageOutput:
        m, g, cfg = ctx.match, ctx.grid, ctx.config
        p2 = ctx.upstream["p2"]
        n, L = len(g), m.pitch.length_m
        min_players = int(cfg["p5_min_unit_players"])
        tf = team_frames(m, g, p2.segments["orientation"])
        sign = {(int(r.period), r.team): r.attack_sign for r in p2.segments["orientation"].itertuples()}
        nf, per, ts = g["native_frame"].to_numpy(np.int64), g["period"].to_numpy(), g["frame_ts"].to_numpy(float)
        block = window_index(g, cfg)
        keepers = windows_from_segments(p2.segments).gk

        ev = m.events
        cols_ok = not ev.empty and {"Type", "Team", "Start Frame", "Start X", "Start Y"} <= set(ev.columns)
        passes = ev.loc[ev["Type"].isin(["PASS", "BALL LOST"])] if cols_ok else ev.iloc[0:0]

        model = parameters_for(m.provider).model
        min_p = cfg["p5_lb_option_min_p"]
        ma = None
        if len(passes):
            from plaiq_engine.match_analysis import (
                MatchAnalysis,  # the moment of the pass, as for Choice
            )
            ma = MatchAnalysis(m)

        def frame_point(xy, s):
            return (s * float(xy[0]) + L / 2, s * float(xy[1]))

        units_out = np.full(n, None, dtype=object)
        inside_out = np.full(n, None, dtype=object)
        options_out = np.full(n, None, dtype=object)
        lanes_out = np.full(n, None, dtype=object)
        lane_p = cfg["p5_open_lane_min_p"]
        furthest_out = np.full(n, None, dtype=object)
        type_out = np.full(n, None, dtype=object)
        rows = []
        for idx, e in passes.iterrows():
            team = str(e["Team"]).lower()
            if team not in OTHER or not np.isfinite(e["Start X"]):
                continue
            k = int(np.clip(np.searchsorted(nf, int(e["Start Frame"])), 0, n - 1))
            s = sign.get((int(per[k]), team), 1)
            start = (s * float(e["Start X"]) + L / 2, s * float(e["Start Y"]))
            d = OTHER[team]
            gk = keepers.get((d, int(per[k]), int(block[k])))
            units = units_at(tf[d].depth[k], tf[d].width[k], L, tf[d].ids.index(gk) if gk in tf[d].ids else None)
            known = sum(len(units[c]) >= min_players for c in UNIT_ORDER)
            row = {"event_row": int(idx), "period": int(per[k]), "team": team, "ts": float(ts[k]),
                   "outcome": "completed" if e["Type"] == "PASS" else "failed", "units": known,
                   "first": None, "second_last": None, "last": None, "furthest": None, "type": None, "units_broken": None,
                   "target": None, "received_inside": None, "line_breaking_options": None, "open_lanes": None}
            has_end = "End X" in e and np.isfinite(e.get("End X", np.nan))
            mo = pass_moment_options(ma, ma.events.loc[idx], model) if ma is not None and known else None
            end = None
            if e["Type"] == "PASS" and has_end and known:
                end, row["target"] = (s * float(e["End X"]) + L / 2, s * float(e["End Y"])), "pass_end"
            elif e["Type"] == "BALL LOST" and mo is not None:
                end, row["target"] = frame_point(mo["target"], s), "logged_target_projected"
            if mo is not None:
                start_m = frame_point(mo["ball"], s)
                row["line_breaking_options"] = int(sum(1 for _, t, p in mo["options"]
                                                        if p >= min_p and any(breaks(start_m, frame_point(t, s), units, min_players).values())))
                options_out[k] = row["line_breaking_options"] if options_out[k] is None else options_out[k]
                row["open_lanes"] = int(sum(1 for _, _, p in mo["options"] if p >= lane_p))
                lanes_out[k] = row["open_lanes"] if lanes_out[k] is None else lanes_out[k]
            if end is not None:
                row["received_inside"] = inside_shape(units, end)
                if inside_out[k] is None:
                    inside_out[k] = row["received_inside"]
                b = breaks(start, end, units, min_players)
                names = unit_names(b)
                for code, name in names.items():
                    row[name] = bool(b[code])
                row["furthest"] = furthest(b)
                row["type"] = next((b[c] for c in (0, 1, 2) if b.get(c)), None)
                row["units_broken"] = int(sum(bool(v) for v in b.values()))
                if units_out[k] is None:
                    units_out[k], furthest_out[k], type_out[k] = row["units_broken"], row["furthest"], row["type"]
            if end is None and e["Type"] == "BALL LOST" and units_out[k] is None:
                furthest_out[k] = "unknown"  # a failed pass with no logged target
            rows.append(row)
        segs = pd.DataFrame(rows, columns=["event_row", "period", "team", "ts", "outcome", "units", "first", "second_last", "last",
                                           "furthest", "type", "units_broken", "target", "received_inside", "line_breaking_options", "open_lanes"])
        frames = pd.DataFrame({"p5_pass_units_broken": pd.array(units_out, dtype="Int64"),
                               "p5_pass_furthest_line": furthest_out, "p5_pass_line_break_type": type_out,
                               "p5_pass_received_inside_shape": pd.array(inside_out, dtype="boolean"),
                               "p5_pass_line_breaking_options": pd.array(options_out, dtype="Int64"),
                               "p5_pass_open_lanes": pd.array(lanes_out, dtype="Int64")})
        p1f = ctx.upstream["p1"].frames
        ball = ball_on_grid(m, g)
        rel = relationships_on_grid(tf, sign, per, ball[["ball_x", "ball_y"]].to_numpy(float),
                                    p1f["p1_ball_state"].astype(object).to_numpy(), keepers, block, L, cfg)
        frames = pd.concat([frames, rel], axis=1)
        # EFI aggregates per team and half, from P1's chains, P2's lines, P3's zones and the pass rows.
        p3f = ctx.upstream["p3"].frames
        allf = pd.concat([g[["frame_ts", "period"]].reset_index(drop=True), p1f.reset_index(drop=True),
                          ctx.upstream["p2"].frames.reset_index(drop=True), p3f.reset_index(drop=True), rel], axis=1)
        allf = allf.loc[:, ~allf.columns.duplicated()]
        end_idx = np.full(len(segs), -1)
        if len(segs) and "End Frame" in ev.columns:
            ef = ev.loc[segs["event_row"], "End Frame"].to_numpy(dtype=float)
            ok = np.isfinite(ef)
            end_idx[ok] = np.clip(np.searchsorted(nf, ef[ok].astype(np.int64)), 0, n - 1)
        efi = efi_aggregates(allf, ctx.upstream["p1"].segments.get("chains", pd.DataFrame()), segs.reset_index(drop=True), end_idx, cfg)

        # -- measurements and the Data check ---------------------------------------
        done = segs.loc[segs["furthest"].notna()]
        measurements = {
            "passes": int((segs["outcome"] == "completed").sum()), "failed_passes": int((segs["outcome"] == "failed").sum()),
            "labelled_share_of_completed": round(float(len(done) / max(int((segs["outcome"] == "completed").sum()), 1)), 4),
            "furthest_share": {k: round(float(v), 4) for k, v in done["furthest"].value_counts(normalize=True).items()},
            "type_share": {k: round(float(v), 4) for k, v in done["type"].value_counts(normalize=True).items()},
            "failed_labelled_share": round(float(((segs["outcome"] == "failed") & segs["furthest"].notna()).sum()
                                                 / max(int((segs["outcome"] == "failed").sum()), 1)), 4),
            "received_inside_share": round(float(segs["received_inside"].dropna().astype(bool).mean()), 4) if segs["received_inside"].notna().any() else None,
            "line_breaking_options_mean": round(float(segs["line_breaking_options"].dropna().mean()), 3) if segs["line_breaking_options"].notna().any() else None,
            "relationships_share_of_grid": round(float(rel["p5_goal_side_defenders"].notna().mean()), 4),
            "carrier_pressure_median": float(rel["p5_carrier_pressure"].dropna().median()) if rel["p5_carrier_pressure"].notna().any() else None,
        }
        p2_state = next((x.state for x in p2.entries if x.metric == "team_shape"), "blocked")
        if not len(done):
            state, why = "blocked", "Line breaks need passes with start and end positions and the defending team's players; none are available."
        elif p2_state != "full":
            state, why = "degraded", "Line breaks are labelled, but team shape is degraded (players out of view), so some units are incomplete."
        else:
            state, why = "full", "Line breaks are labelled for completed passes from the defending team's units at each pass."
        entries = [self.entry("line_breaks", "Line breaks", state,
                              why + " Failed passes are judged by the logged target's projected point where a target is logged, else unknown. Ponta's rule after FIFA's definitions: estimated, not yet validated.",
                              coverage=f"{measurements['labelled_share_of_completed'] * 100:.1f}% of completed passes labelled",
                              unlock=None if state == "full" else "Full-pitch tracking and an event feed with pass end positions.")]
        cov = measurements["relationships_share_of_grid"]
        rel_state = "blocked" if not cov else ("full" if p2_state == "full" else "degraded")
        entries.append(self.entry(
            "relationships", "Relationships", rel_state,
            {"full": "Pressure, goal-side, offside line, free players, overload and marking are computed wherever a team has the ball.",
             "degraded": "Computed wherever a team has the ball, but team shape is degraded (players out of view), so counts can be short.",
             "blocked": "No grid point has a team with the ball and a detected ball."}[rel_state]
            + " Ponta's definitions after FIFA's where one exists: estimated, not yet validated.",
            coverage=f"{cov * 100:.1f}% of grid points", unlock=None if rel_state == "full" else "Full-pitch tracking with the ball detected."))
        measurements["efi_rows"] = int(len(efi))
        return PackageOutput(frames=frames, entries=entries, segments={"line_breaks": segs, "efi": efi}, measurements=measurements)
