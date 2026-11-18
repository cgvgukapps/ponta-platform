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
    ))


class LineBreakPackage(ContextPackage):
    name = "p5"
    version = "0.1.0"
    depends_on = ("p1", "p2")
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

        units_out = np.full(n, None, dtype=object)
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
                   "first": None, "second_last": None, "last": None, "furthest": None, "type": None, "units_broken": None}
            has_end = "End X" in e and np.isfinite(e.get("End X", np.nan))
            if e["Type"] == "PASS" and has_end and known:
                end = (s * float(e["End X"]) + L / 2, s * float(e["End Y"]))
                b = breaks(start, end, units, min_players)
                names = unit_names(b)
                for code, name in names.items():
                    row[name] = bool(b[code])
                row["furthest"] = furthest(b)
                row["type"] = next((b[c] for c in (0, 1, 2) if b.get(c)), None)
                row["units_broken"] = int(sum(bool(v) for v in b.values()))
                if units_out[k] is None:
                    units_out[k], furthest_out[k], type_out[k] = row["units_broken"], row["furthest"], row["type"]
            elif e["Type"] == "BALL LOST" and units_out[k] is None:
                furthest_out[k] = "unknown"
            rows.append(row)
        segs = pd.DataFrame(rows, columns=["event_row", "period", "team", "ts", "outcome", "units", "first", "second_last", "last",
                                           "furthest", "type", "units_broken"])
        frames = pd.DataFrame({"p5_pass_units_broken": pd.array(units_out, dtype="Int64"),
                               "p5_pass_furthest_line": furthest_out, "p5_pass_line_break_type": type_out})

        # -- measurements and the Data check ---------------------------------------
        done = segs.loc[segs["furthest"].notna()]
        measurements = {
            "passes": int((segs["outcome"] == "completed").sum()), "failed_passes": int((segs["outcome"] == "failed").sum()),
            "labelled_share_of_completed": round(float(len(done) / max(int((segs["outcome"] == "completed").sum()), 1)), 4),
            "furthest_share": {k: round(float(v), 4) for k, v in done["furthest"].value_counts(normalize=True).items()},
            "type_share": {k: round(float(v), 4) for k, v in done["type"].value_counts(normalize=True).items()},
        }
        p2_state = next((x.state for x in p2.entries if x.metric == "team_shape"), "blocked")
        if not len(done):
            state, why = "blocked", "Line breaks need passes with start and end positions and the defending team's players; none are available."
        elif p2_state != "full":
            state, why = "degraded", "Line breaks are labelled, but team shape is degraded (players out of view), so some units are incomplete."
        else:
            state, why = "full", "Line breaks are labelled for completed passes from the defending team's units at each pass."
        entries = [self.entry("line_breaks", "Line breaks", state,
                              why + " Failed passes are listed with line breaks unknown (no target yet). Ponta's rule after FIFA's definitions: estimated, not yet validated.",
                              coverage=f"{measurements['labelled_share_of_completed'] * 100:.1f}% of completed passes labelled",
                              unlock=None if state == "full" else "Full-pitch tracking and an event feed with pass end positions.")]
        return PackageOutput(frames=frames, entries=entries, segments={"line_breaks": segs}, measurements=measurements)
