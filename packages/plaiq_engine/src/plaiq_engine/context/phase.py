"""
phase.py — context package P4: phase of play.

A deterministic phase label on every grid point where one team has the ball
(``ponta-blueprint/context-packages.md``, P4; ``ponta-blueprint/context-structure-task.md``).
The vocabulary follows SkillCorner's published phases of play (specification of 16 February
2025) so the labels can be compared with a provider's; the rules and every number are
Ponta's, measured on the regression set, and the provider's labels are never an input.

Within each P1 possession chain, in priority order:

1. ``set_play``: the chain starts with a logged corner or a free kick in the attacking half;
   up to ``p4_set_play_max_s``. Without a logged event feed only corners are detected: the
   ball comes within ``p4_corner_radius_m`` of a corner flag in the chain's first
   ``p4_corner_window_s``. (Since 0.2.0 corners come from player positions instead,
   ``restarts.corner_setups``.)
2. ``direct``: from the own half, the ball gains at least ``p4_direct_min_gain_m`` towards
   goal within ``p4_direct_window_s`` (SkillCorner: a long ball of at least 32 m). Labelled
   from the kick to the furthest point.
3. ``chaotic``: the chain is shorter than ``p4_chaotic_max_s``, or possession is contested.
4. ``quick_break`` (regain in the opponent's half) or ``transition`` (regain in the own
   half): the chain begins by winning the ball from the other team in open play (since
   0.3.0, without a logged feed, not straight after a throw-in found from positions) and the
   ball gains at least ``p4_break_min_gain_m`` within ``p4_break_eval_s``. It lasts until
   ``p4_recovered_players`` defending outfield players are goal-side of the ball (after the
   evaluation time), the chain ends, or ``p4_break_max_s``.
5. Established play, from the defending team's block: the mean depth of its three deepest
   outfield players (SkillCorner's measure) below ``p4_low_block_max_m`` is a low block
   (``finish``), at or beyond ``p4_high_block_min_m`` a high block (``build_up``), between
   them a medium block (``create``). Since version 0.2.0 (the founder's replay review) where
   the ball is refines the block: ``finish`` needs the ball past ``p4_finish_min_depth_share``
   of the pitch (else ``create``), and with the ball inside ``p4_build_up_max_depth_share`` of
   it the team is building up whatever the block (a quarter of the pitch since 0.4.0, 20 November 2026; a
   third before). A quick break or transition that began
   further back ends when the ball reaches ``p4_break_end_depth_share`` of the pitch (0, off, since 0.5.0:
   the founder's decision of 23 November 2026, after the rule showed no effect against SkillCorner).

Runs shorter than ``p4_min_phase_s`` merge into the run before them in the same chain. The
team out of possession gets the paired label (``defending_transition`` …), except in established
play (build-up, create, finish), where its label is its own block: ``high_block``, ``medium_block``
or ``low_block`` from ``p4_block`` (version 0.6.0, 25 November 2026, review finding AN-02: before,
the label was copied from the attacking phase, so a build-up forced by the ball's depth read as a
high block when the block was medium). ``p4_block`` is the defending team's block on every
possession grid point, whatever the phase. Phases are estimated (rules on positions), not yet validated.

Without a logged feed (version 0.3.0): no phase is labelled while a throw-in set-up found from
positions is under way (the ball is dead), and a corner set-up is ``set_play`` for the taking
team from the set-up's start.

The ``restarts`` segment lists the restarts behind the labels: from the logged feed (every logged
set piece, team as logged) or, without one, the corners and throw-ins found from player positions (version 0.3.0).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from plaiq_engine.context.possession import _restart_name, ball_on_grid, logged_events
from plaiq_engine.context.registry import (
    ContextField,
    ContextInput,
    ContextPackage,
    PackageOutput,
)
from plaiq_engine.context.restarts import (
    SETUP_AFTER_S,
    SETUP_BEFORE_S,
    corner_setups,
    median_speed,
    throw_in_setups,
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

IN_PHASES = ("build_up", "create", "finish", "direct", "quick_break", "transition", "set_play", "chaotic")
PAIRED = {"build_up": "high_block", "create": "medium_block", "finish": "low_block", "direct": "defending_direct",
          "quick_break": "defending_quick_break", "transition": "defending_transition",
          "set_play": "defending_set_play", "chaotic": "chaotic"}
BLOCK_PHASE = {"low": "finish", "medium": "create", "high": "build_up"}
BLOCK_OUT = {"low": "low_block", "medium": "medium_block", "high": "high_block"}
SET_PLAY_STARTS = ("corner_kick", "free_kick")


def deepest_three(depth: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Mean depth of the three deepest outfield players per grid point (NaN with fewer than three)."""
    D = np.where(np.isfinite(depth) & (lines != 3), depth, np.inf)
    s = np.sort(D, axis=1)[:, :3]
    out = s.mean(axis=1) if s.shape[1] == 3 else np.full(len(D), np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def block_of(d3: np.ndarray, low_max: float, high_min: float) -> np.ndarray:
    """Low, medium or high block from the three deepest defenders' mean depth (m from own goal line)."""
    out = np.full(len(d3), None, dtype=object)
    ok = np.isfinite(d3)
    out[ok & (d3 < low_max)] = "low"
    out[ok & (d3 >= low_max) & (d3 < high_min)] = "medium"
    out[ok & (d3 >= high_min)] = "high"
    return out


def corner_at_start(bx: np.ndarray, by: np.ndarray, first_rows: np.ndarray, length: float, width: float, radius: float) -> bool:
    """True when the ball comes within ``radius`` of a corner flag at any detected point of ``first_rows``."""
    fx, fy = bx[first_rows], by[first_rows]
    ok = np.isfinite(fx) & np.isfinite(fy)
    if not ok.any():
        return False
    d = np.hypot(length / 2 - np.abs(fx[ok]), width / 2 - np.abs(fy[ok]))
    return bool(d.min() <= radius)


def _forward_max(a: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Max of ``a[i:i+w+1]`` (NaN-aware) and its offset, per i."""
    n = len(a)
    pad = np.r_[a, np.full(w, np.nan)]
    win = np.lib.stride_tricks.sliding_window_view(pad, w + 1)[:n]
    filled = np.where(np.isfinite(win), win, -np.inf)
    off = filled.argmax(axis=1)
    mx = filled[np.arange(n), off]
    return np.where(np.isfinite(mx), mx, np.nan), off


def _smooth(labels: np.ndarray, min_pts: int) -> np.ndarray:
    """Merge runs shorter than ``min_pts`` into the run before them (the next one for a leading run)."""
    lab = labels.copy()
    if len(lab) == 0 or min_pts <= 1:
        return lab
    changed = True
    while changed:
        changed = False
        starts = np.r_[0, np.where(lab[1:] != lab[:-1])[0] + 1]
        ends = np.r_[starts[1:], len(lab)]
        if len(starts) == 1:
            break
        for s, e in zip(starts, ends, strict=True):
            if e - s < min_pts:
                lab[s:e] = lab[s - 1] if s > 0 else lab[e]
                changed = True
                break
    return lab


@dataclass
class _Fields:
    items: tuple[ContextField, ...] = field(default=(
        ContextField("p4_team", "label", "estimated", "The team in possession the phase belongs to (P1); null when neither team has the ball."),
        ContextField("p4_phase", "label", "estimated",
                     "Phase of play of the team in possession: build_up, create, finish, direct, quick_break, transition, set_play, chaotic. Ponta's rules after SkillCorner's published vocabulary. Estimated, not yet validated."),
        ContextField("p4_out_phase", "label", "estimated",
                     "The phase of the team out of possession: in established play its own block (high_block, medium_block, low_block, from p4_block); otherwise the paired label (defending_direct, defending_quick_break, defending_transition, defending_set_play, chaotic)."),
        ContextField("p4_block", "label", "estimated",
                     "The defending team's block from its three deepest outfield players' mean depth: low, medium or high (boundaries are Ponta's measured thresholds)."),
        ContextField("p4_defending_depth_m", "m", "observed", "Mean distance of the defending team's three deepest outfield players from their own goal line."),
        ContextField("p4_phase_id", "id", "estimated", "The phase segment this grid point belongs to; null outside a phase."),
    ))


class PhasePackage(ContextPackage):
    name = "p4"
    version = "0.6.0"
    depends_on = ("p1", "p2")
    fields = _Fields().items

    def compute(self, ctx: ContextInput) -> PackageOutput:
        m, g, cfg = ctx.match, ctx.grid, ctx.config
        p1, p2 = ctx.upstream["p1"], ctx.upstream["p2"]
        n, L = len(g), m.pitch.length_m
        hz = cfg["context_grid_rate_hz"]
        ts, per = g["frame_ts"].to_numpy(float), g["period"].to_numpy()
        signs = p2.segments["orientation"]
        sign_of = {(int(r.period), r.team): r.attack_sign for r in signs.itertuples()}
        chains = p1.segments.get("chains", pd.DataFrame())
        tf = team_frames(m, g, signs)
        lines = line_members(g, tf, team_states(g, p1.frames, chains, cfg), window_index(g, cfg), windows_from_segments(p2.segments))

        poss = p1.frames["p1_possession"].to_numpy(object)
        cid = p1.frames["p1_chain_id"].to_numpy(dtype=float, na_value=np.nan)
        ball = ball_on_grid(m, g)
        bx, by = ball["ball_x"].to_numpy(float), ball["ball_y"].to_numpy(float)
        W = m.pitch.width_m
        logged = logged_events(m)

        d3 = {t: deepest_three(tf[t].depth, lines[t]) for t in TEAMS}
        # Ball depth towards goal for each team (its own frame) and defending players' depths.
        a_depth, a_width = {}, {}
        for t in TEAMS:
            s = np.array([sign_of.get((int(p), t), 1 if t == "home" else -1) for p in per], float)
            a_depth[t] = s * bx + L / 2
            a_width[t] = s * by

        team = np.array([v if v in TEAMS else None for v in poss], dtype=object)
        phase = np.full(n, None, dtype=object)
        block = np.full(n, None, dtype=object)
        dd = np.full(n, np.nan)
        for t in TEAMS:
            rows = team == t
            dd[rows] = d3[OTHER[t]][rows]
        for t in TEAMS:
            rows = team == t
            block[rows] = block_of(dd[rows], cfg["p4_low_block_max_m"], cfg["p4_high_block_min_m"])
        contested = poss == "contested"

        # Established play everywhere first, then the chain rules on top.
        for k in np.where(team != None)[0]:
            phase[k] = BLOCK_PHASE.get(block[k])
        # Where the ball is refines the block: finish needs the ball far enough forward (else
        # create); a ball deep in the team's own end is build-up whatever the block (0 = off).
        fin_min, bu_max = cfg["p4_finish_min_depth_share"] * L, cfg["p4_build_up_max_depth_share"] * L
        for t in TEAMS:
            rows = team == t
            dep = a_depth[t]
            known = rows & np.isfinite(dep)
            if fin_min > 0:
                phase[known & (phase == "finish") & (dep < fin_min)] = "create"
            if bu_max > 0:
                phase[known & np.isin(phase, ["create", "finish"]) & (dep < bu_max)] = "build_up"
        phase[contested] = "chaotic"

        w_direct = int(round(cfg["p4_direct_window_s"] * hz))
        w_eval = int(round(cfg["p4_break_eval_s"] * hz))
        fmax = {t: _forward_max(a_depth[t], w_direct) for t in TEAMS}
        prev_team: dict[int, str] = {}
        ch = chains.sort_values("start_ts").reset_index(drop=True) if len(chains) else chains
        for j in range(len(ch)):
            c = ch.loc[j]
            prev_team[int(c["chain_id"])] = ch.loc[j - 1, "team"] if j > 0 and ch.loc[j - 1, "period"] == c["period"] else None
        dead = p1.frames["p1_dead_ball"].to_numpy(dtype=float, na_value=np.nan)
        # Without a logged feed, corners from player positions (the founder's proposal): the end
        # of each set-up is about when the corner is taken.
        setups: dict[str, np.ndarray] = {}
        dead_spans: list[tuple[float, float]] = []  # throw-in set-ups found from positions: the ball is dead
        corner_spans: list[tuple[str, float, float]] = []  # corner set-ups: part of the set play
        if not logged and cfg["p4_corner_setup_crowd_min"] > 0:
            for t in TEAMS:
                found = corner_setups(tf[t].depth, tf[t].width, tf[OTHER[t]].depth, tf[OTHER[t]].width, ts, per, L, W,
                                      radius=cfg["p4_corner_setup_radius_m"], crowd_min=int(cfg["p4_corner_setup_crowd_min"]),
                                      hold_s=cfg["p4_corner_setup_hold_s"])
                setups[t] = np.array([e for _, e in found], dtype=float)
                corner_spans += [(t, a, b) for a, b in found]
        # Without a logged feed, throw-ins from player positions (0.3.0): a possession that starts
        # with one is a restart, not a regain.
        throw_ins: dict[str, np.ndarray] = {}
        if not logged and cfg["p4_throw_in_hold_s"] > 0:
            speed = median_speed([tf[t].depth for t in TEAMS], [tf[t].width for t in TEAMS], ts, per)
            for t in TEAMS:
                found = throw_in_setups(tf[t].depth, tf[t].width, tf[OTHER[t]].depth, tf[OTHER[t]].width, a_depth[t], a_width[t],
                                        speed, ts, per, L, W, out_m=cfg["p4_throw_in_out_m"], hold_s=cfg["p4_throw_in_hold_s"],
                                        still_mps=cfg["p4_throw_in_still_mps"], ball_m=cfg["p4_throw_in_ball_m"], hz=hz)
                throw_ins[t] = np.array([e for _, e in found], dtype=float)
                dead_spans += found

        for c in ch.itertuples() if len(ch) else ():
            t = c.team
            idx = np.where(cid == c.chain_id)[0]
            if not len(idx) or t not in TEAMS:
                continue
            s_i, e_i = idx[0], idx[-1]
            span = np.arange(s_i, e_i + 1)
            in_team = span[team[span] == t]
            start = str(c.start_type)
            first = a_depth[t][in_team[: max(1, int(hz))]] if len(in_team) else np.array([np.nan])
            start_depth = a_depth[t][s_i] if np.isfinite(a_depth[t][s_i]) else (first[np.isfinite(first)][0] if np.isfinite(first).any() else np.nan)
            # 1. Set play
            sp = start.split(":", 1)[1] if start.startswith("set_piece:") else ""
            kick = ts[s_i]
            if setups:
                # A set-up that ends just before or during this chain: the set play runs from its end.
                hits = setups.get(t, np.array([]))
                hits = hits[(hits >= ts[s_i] - cfg["p4_corner_setup_hold_s"] - 1.0) & (hits <= ts[e_i])]
                if len(hits):
                    sp, kick = "corner_kick", float(hits[0])
            elif not logged and corner_at_start(bx, by, in_team[: max(1, int(round(cfg["p4_corner_window_s"] * hz)))], L, W, cfg["p4_corner_radius_m"]):
                sp = "corner_kick"
            if sp.startswith(SET_PLAY_STARTS) and (sp.startswith("corner_kick") or (np.isfinite(start_depth) and start_depth >= L / 2)):
                end = kick + cfg["p4_set_play_max_s"]
                sel = in_team[(ts[in_team] >= kick) & (ts[in_team] <= end)]
                phase[sel] = "set_play"
                set_end = sel[-1] if len(sel) else s_i
            else:
                set_end = -1
            # 4. Quick break or transition (written before direct and chaotic, which override it)
            gap_dead = False
            if s_i > 0:
                prev = slice(max(0, s_i - int(round(cfg["p1_chain_event_window_s"] * hz))), s_i)
                gap_dead = bool(np.nan_to_num(dead[prev], nan=0.0).any())
            thrown = throw_ins.get(t, np.array([]))
            restarted = bool(((thrown >= ts[s_i] - SETUP_BEFORE_S) & (thrown <= ts[s_i] + SETUP_AFTER_S)).any())
            regain = (start in ("won", "recovery") and prev_team.get(int(c.chain_id)) == OTHER[t] and not gap_dead and not restarted)
            if regain and np.isfinite(start_depth) and set_end < 0:
                look = in_team[(in_team >= s_i) & (in_team <= s_i + w_eval)]
                gain = np.nanmax(a_depth[t][look]) - start_depth if len(look) and np.isfinite(a_depth[t][look]).any() else np.nan
                if np.isfinite(gain) and gain >= cfg["p4_break_min_gain_m"]:
                    label = "quick_break" if start_depth >= L / 2 else "transition"
                    stop = ts[s_i] + cfg["p4_break_max_s"]
                    D_def = tf[OTHER[t]].depth
                    end_depth = cfg["p4_break_end_depth_share"] * L
                    for k in in_team:
                        if ts[k] > stop:
                            break
                        if end_depth > 0 and start_depth < end_depth and np.isfinite(a_depth[t][k]) and a_depth[t][k] >= end_depth:
                            break
                        if k > s_i + w_eval and np.isfinite(a_depth[t][k]):
                            ball_def_depth = L - a_depth[t][k]
                            goal_side = np.isfinite(D_def[k]) & (lines[OTHER[t]][k] != 3) & (D_def[k] < ball_def_depth)
                            if goal_side.sum() >= cfg["p4_recovered_players"]:
                                break
                        phase[k] = label
            # 3. Chaotic: short chains
            if c.duration_s < cfg["p4_chaotic_max_s"] and set_end < 0:
                phase[in_team] = "chaotic"
            # 2. Direct: long gains from the own half
            mx, off = fmax[t]
            for k in in_team:
                if k <= set_end:
                    continue
                if np.isfinite(a_depth[t][k]) and a_depth[t][k] < L / 2 and np.isfinite(mx[k]) and mx[k] - a_depth[t][k] >= cfg["p4_direct_min_gain_m"]:
                    stop = min(k + int(off[k]), e_i)
                    rng = np.arange(k, stop + 1)
                    phase[rng[team[rng] == t]] = "direct"

        # Smooth within chains, then build phase segments.
        min_pts = int(round(cfg["p4_min_phase_s"] * hz))
        for c in ch.itertuples() if len(ch) else ():
            idx = np.where((cid == c.chain_id) & (team != None))[0]
            if len(idx) > 1:
                sub = phase[idx].astype(object)
                ok = np.array([v is not None for v in sub])
                if ok.all():
                    phase[idx] = _smooth(sub, min_pts)
        phase[team == None] = None
        # Restart set-ups found from positions (no logged feed). P1 carries possession through the
        # stoppage, so: while a throw-in set-up is under way the ball is dead and no phase is
        # labelled; a corner set-up is part of the set play for the taking team (the founder's
        # review: "a corner being taken" is a set play).
        for a, b in dead_spans:
            phase[(ts >= a) & (ts < b)] = None
        for t, a, b in corner_spans:
            span = (ts >= a) & (ts < b)
            phase[span & (team == t)] = "set_play"
            phase[span & (team != t)] = None
        # In established play the defending team's label is its own block (AN-02); elsewhere the paired label.
        out_phase = np.array([BLOCK_OUT.get(b, PAIRED[v]) if v in BLOCK_PHASE.values() else PAIRED.get(v) if v is not None else None
                              for v, b in zip(phase, block, strict=True)], dtype=object)

        pid = np.full(n, -1, dtype=np.int64)
        rows, cur = [], -1
        for k in range(n):
            if phase[k] is None:
                continue
            if k == 0 or phase[k] != phase[k - 1] or team[k] != team[k - 1] or per[k] != per[k - 1] or cid[k] != cid[k - 1]:
                cur += 1
                rows.append({"phase_id": cur, "period": int(per[k]), "team": team[k], "phase": phase[k], "start_ts": float(ts[k]),
                             "end_ts": float(ts[k]), "chain_id": int(cid[k]) if np.isfinite(cid[k]) else None})
            pid[k] = cur
            rows[-1]["end_ts"] = float(ts[k])
        segs = pd.DataFrame(rows, columns=["phase_id", "period", "team", "phase", "start_ts", "end_ts", "chain_id"])
        segs["duration_s"] = (segs["end_ts"] - segs["start_ts"]).round(3)
        segs["chain_id"] = segs["chain_id"].astype("Int64")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            frames = pd.DataFrame({
                "p4_team": team, "p4_phase": phase, "p4_out_phase": out_phase, "p4_block": block,
                "p4_defending_depth_m": np.round(dd, 3),
                "p4_phase_id": pd.array([int(v) if v >= 0 else None for v in pid], dtype="Int64"),
            })

        # -- restarts behind the labels ----------------------------------------------
        rrows = []
        if logged:
            # Every logged set piece, team as logged (a corner cleared at the first touch included).
            ev = m.events
            nf = g["native_frame"].to_numpy(np.int64)
            for i in ev.index[ev["Type"].astype(str) == "SET PIECE"] if "Type" in ev.columns else ():
                f0 = ev.at[i, "Start Frame"] if "Start Frame" in ev.columns else np.nan
                if pd.isna(f0):
                    continue
                k = int(min(np.searchsorted(nf, int(f0)), n - 1))
                tm = str(ev.at[i, "Team"]).lower() if "Team" in ev.columns else ""
                rrows.append({"type": _restart_name(ev.at[i, "Subtype"]), "team": tm if tm in TEAMS else None,
                              "period": int(per[k]), "ts": float(ts[k]), "source": "events"})
        else:
            for kind, found in (("corner_kick", setups), ("throw_in", throw_ins)):
                for t, times in found.items():
                    for x in times:
                        k = int(min(np.searchsorted(ts, x), n - 1))
                        rrows.append({"type": kind, "team": t, "period": int(per[k]), "ts": float(x), "source": "tracking"})
        restarts = pd.DataFrame(sorted(rrows, key=lambda r: (r["ts"], r["team"])), columns=["type", "team", "period", "ts", "source"])

        # -- measurements and the Data check ---------------------------------------
        in_play = poss != "dead_ball"
        labelled = phase != None
        coverage = float(labelled[in_play].mean()) if in_play.any() else 0.0
        src = p1.frames["p1_possession_source"].to_numpy(object)
        inferred = float(((src != "observed") & labelled).sum() / max(int(labelled.sum()), 1))
        share = pd.Series(phase[labelled]).value_counts(normalize=True)
        measurements = {
            "coverage_in_play": round(coverage, 4),
            "phase_share": {p: round(float(share.get(p, 0.0)), 4) for p in IN_PHASES},
            "block_share": {b: round(float((block[labelled] == b).mean()), 4) for b in ("low", "medium", "high")} if labelled.any() else {},
            "segments": int(len(segs)),
            "median_duration_s": {p: round(float(segs.loc[segs["phase"] == p, "duration_s"].median()), 2)
                                  for p in IN_PHASES if (segs["phase"] == p).any()},
            "set_play_source": "events" if logged else "corners_from_tracking",
            "restarts": {"source": "events" if logged else "tracking",
                         "counts": {k: int(v) for k, v in restarts["type"].value_counts().sort_index().items()}},
        }
        p2_state = next((e.state for e in p2.entries if e.metric == "team_shape"), "blocked")
        if coverage == 0.0 or p2_state == "blocked":
            state, why = "blocked", "Phases need possession and both teams' shapes; they are not available."
        elif p2_state == "degraded" or inferred > cfg["p1_degraded_inferred_share"] or not logged:
            state = "degraded"
            parts = []
            if p2_state == "degraded":
                parts.append("team shape is degraded")
            if inferred > cfg["p1_degraded_inferred_share"]:
                parts.append(f"{_pct(inferred)} of labelled time rests on inferred possession")
            if not logged:
                parts.append("without a logged event feed only corners are found as set plays, and only corners and throw-ins as restarts, from player positions")
            why = "Phases are labelled but " + ", ".join(parts) + "."
        else:
            state, why = "full", "Phases are labelled from observed possession and both teams' shapes."
        entries = [self.entry("phase_of_play", "Phase of play", state, why + " Phases are Ponta's rules: estimated, not yet validated.",
                              coverage=f"{_pct(coverage)} of in-play time",
                              unlock=None if state == "full" else "Full-pitch tracking and the logged event feed.")]
        return PackageOutput(frames=frames, entries=entries, segments={"phases": segs, "restarts": restarts}, measurements=measurements)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"
