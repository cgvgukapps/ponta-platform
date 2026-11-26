"""
states.py — context package P8: semantic states (episodes).

Stage D (``ponta-blueprint/context-states-task.md``). The AI layer's evidence for anything longer than one action: the
match cut into **episodes**, each a passage of play with one team in possession, one phase and one shape, described by
labels and aggregates from P1 to P7 and P4v, with a short summary written by code from the row's own fields.

**The episode rule.** P4's phase segments already sit within one team's possession chain (P1) and one phase. An
episode is a run of grid points with the same P4 phase segment, split further where either team's formation changes
(P2, when ``p8_split_formation`` is 1) or where P4v's FIFA phase changes (when ``p8_split_fifa_phase`` is 1, a finer cut:
build-up opposed and unopposed). Runs shorter than ``p8_min_state_s`` merge into the run before them in the same chain,
or the one after for a chain's first run; a chain with a single short run keeps it as an episode when
``p8_keep_short_chains`` is 1, and has none when it is 0. Dead-ball time and time with no
team in possession belong to no episode.

**The row** (segment ``states``): labels and aggregates only, never a coordinate, cell index or name. Places are P3
zones; players are not named; distances (a team's length, a line's height) are labelled metrics. Each row carries the
Data check state of every package for the match and the package versions. Every number and label the ``summary``
quotes is a field of the same row (``plaiq_engine.citations.check_quoted_numbers``). Episodes are estimated: their
phase is.

**Version 0.2.0** (after the founder's episode review, 23 November 2026): each row says how the episode started
(``start_reason``: continued from the previous phase in the same chain, won the ball, the goalkeeper had the ball, a set
piece named in ``start_detail``, a restart or a kick-off, from P1's chain and the logged first action) and whether the
team in possession made a logged clearance (``cleared``); an episode whose last logged action by that team is a clearance
ends ``cleared``. The defending side of contested and set-play spells is no longer phrased as an opponent's phase.

**Version 0.2.1** (23 November 2026): P4's ``set_play`` is summarised as "play from a set piece". SkillCorner's published
convention, which P4 follows, keeps the play after the kick in the set-play phase (its set-play phases last a median
4.8 s, up to 20 s), so the words now say that rather than suggesting a still ball.

**Version 0.3.0** (25 November 2026, review backlog A: AI-01, AN-03): an episode continues the one before when both
belong to the same P1 chain in the same period, whether or not the grid points between them have a phase; only a
chain's last episode takes the chain's end (lost, shot, goal, dead ball), and an episode followed by the same chain
after a stretch with no phase ends ``contested_kept`` when that stretch is mostly a contest for the ball (P1 contested; a
median 0.4 s on the reference matches), else ``phase_unknown``. Where P1 has no logged restart for an episode's chain, a restart
P4 found from player positions by the same team within ``p8_restart_before_s`` / ``p8_restart_after_s`` starts it as
a set piece.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from plaiq_engine.analytics.event_stitch import clean_jersey
from plaiq_engine.context.registry import (
    ContextField,
    ContextInput,
    ContextPackage,
    PackageOutput,
)
from plaiq_engine.context.structure import (
    OTHER,
    TEAMS,
    window_index,
    windows_from_segments,
)
from plaiq_engine.context.thresholds import THRESHOLDS_VERSION

THIRDS = ("defensive", "middle", "attacking")
LINE_ORDER = {"first": 1, "second_last": 2, "last": 3}
#: How an episode ended (``end_reason``).
END_REASONS = ("phase_changed", "formation_changed", "contested_kept", "phase_unknown", "possession_lost", "cleared", "shot", "goal", "ball_out_of_play",
               "foul_or_set_piece", "dead_ball", "end_of_period")
#: How an episode started (``start_reason``; 0.2.0). ``start_detail`` names the set piece.
START_REASONS = ("continued", "won_the_ball", "goalkeeper", "set_piece", "restart", "kick_off", "unknown")
START_TEXT = {"won_the_ball": "won the ball", "goalkeeper": "the goalkeeper had the ball", "restart": "a restart", "kick_off": "a kick-off"}
SET_PIECE_TEXT = {"corner_kick": "a corner", "throw_in": "a throw-in", "goal_kick": "a goal kick", "free_kick": "a free kick",
                  "free_kick_direct": "a free kick", "free_kick_indirect": "a free kick", "penalty": "a penalty", "kick_off": "a kick-off"}
#: P1 set pieces after a dead ball, by end reason.
OUT_OF_PLAY = {"throw_in", "corner_kick", "goal_kick"}
#: Readable names for the summary (FIFA's where P4v gives one).
PHASE_TEXT = {"build_up_opposed": "build-up (opposed)", "build_up_unopposed": "build-up (unopposed)", "build_up": "build-up",
              "progression": "progression", "final_third": "final-third play", "long_ball": "a long ball",
              "counter_attack": "a counter-attack", "attacking_transition": "an attacking transition",
              "set_play": "play from a set piece", "in_contest": "a contested spell", "create": "progression", "finish": "final-third play",
              "direct": "a long ball", "quick_break": "a counter-attack", "transition": "an attacking transition", "chaotic": "a contested spell"}
#: The defending side of contested, set-play and long-ball spells is not phrased as an opponent's phase (0.2.0, the founder's review).
OUT_TEXT = {"high_block": "a high block", "mid_block": "a mid-block", "low_block": "a low block", "high_press": "a high press",
            "mid_press": "a mid press", "low_press": "a low press", "counter_press": "a counter-press",
            "defensive_transition": "a defence in transition"}
END_TEXT = {"phase_changed": "the phase changed", "formation_changed": "a formation changed",
            "contested_kept": "a brief contest for the ball, which the same team kept",
            "phase_unknown": "the phase could not be read for a while; the same team kept the ball", "possession_lost": "possession lost",
            "cleared": "the ball was cleared",
            "shot": "a shot", "goal": "a goal", "ball_out_of_play": "the ball went out of play",
            "foul_or_set_piece": "a foul or set piece", "dead_ball": "a stoppage", "end_of_period": "the end of the period"}

#: P6's attacking run types (and carries), one count column each; P4v's engagement types likewise.
RUN_TYPES = ("behind", "coming_short", "dropping_off", "support", "overlap", "underlap", "pulling_wide", "pulling_half_space",
             "run_ahead_of_the_ball", "cross_receiver", "carry")
ENGAGEMENT_TYPES = ("recovery", "counter_press", "press", "pressure")
#: Every Data check metric of the context packages for the match, one column each (``data_check_p1_possession`` …).
DC_METRICS = (("p1", "possession"), ("p1", "possession_state"), ("p1", "game_state"), ("p2", "team_shape"), ("p2", "formation"),
              ("p3", "relative_zones"), ("p4", "phase_of_play"), ("p4v", "engagements"), ("p4v", "fifa_view"), ("p4v", "fifa_block"),
              ("p5", "line_breaks"), ("p5", "relationships"), ("p6", "movement"), ("p7", "space"), ("p7", "adjusted_choice"),
              ("p8", "episodes"))

STATE_COLUMNS: tuple[str, ...] = (
    "state_id", "match_id", "period", "start_s", "end_s", "duration_s", "minute_display",
    "team", "chain_id", "start_reason", "start_detail", "cleared", "share_controlled", "share_in_flight", "share_loose_or_contested", "score_diff", "red_card_diff",
    "game_state_source",
    "phase", "out_phase", "fifa_phase", "fifa_out_phase", "block", "opposed_pct", "press_share", "counter_press_share",
    "fifa_mid_block_share", "fifa_low_block_share",
    "formation_in", "formation_out", "team_length_m", "team_width_m", "opp_length_m", "opp_width_m",
    "opp_def_line_m", "opp_def_mid_gap_m", "opp_mid_att_gap_m",
    "start_lane", "start_third", "end_lane", "end_third", "share_defensive_third", "share_middle_third", "share_attacking_third",
    "passes", "passes_completed", "passes_breaking_a_line", "furthest_line_broken",
    "carrier_pressure_median", "carrier_pressure_max", "goal_side_defenders_median", "overload_at_ball_median",
    "runs", "runs_in_behind", *(f"runs_{t}" for t in RUN_TYPES), "defensive_runs",
    "open_high_value_share", "opp_open_high_value_share", "open_danger_zone",
    "engagements", *(f"engagements_{t}" for t in ENGAGEMENT_TYPES), "presses", "counter_press",
    "end_reason", "led_to_shot",
    "first_event_idx", "last_event_idx", "events",
    "ball_detected_share", "min_outfield_observed", *(f"data_check_{p}_{m}" for p, m in DC_METRICS), "data_check", "thresholds_version",
    "versions", "label", "summary",
)


def _mode(a: np.ndarray):
    v = [x for x in a if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return None
    vals, counts = np.unique(np.asarray(v, dtype=object).astype(str), return_counts=True)
    return str(vals[np.argmax(counts)])  # ties: the first in sorted order, so deterministic


def _num(a, fn, digits: int = 1):
    x = np.asarray(a, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    v = float(fn(x))
    return int(round(v)) if digits == 0 else round(v, digits)


def segment(keys: list, chain: np.ndarray, ts: np.ndarray, period: np.ndarray, min_s: float, step: float,
            keep_short: bool = True) -> list[list[int]]:
    """
    Episodes as [start, end] index pairs (end inclusive). ``keys[i]`` is the splitting key at grid point i (None: no
    episode there). Runs of one key, contiguous and in one period; short runs merge within their chain.
    """
    runs: list[list] = []
    for i, k in enumerate(keys):
        if k is None:
            continue
        if runs and runs[-1][2] == k and runs[-1][1] == i - 1 and period[i] == period[i - 1]:
            runs[-1][1] = i
        else:
            runs.append([i, i, k, chain[i]])

    def dur(r):
        return ts[r[1]] - ts[r[0]] + step

    changed = True
    while changed:
        changed = False
        for j, r in enumerate(runs):
            if dur(r) >= min_s - 1e-9:
                continue
            prev = runs[j - 1] if j > 0 else None
            nxt = runs[j + 1] if j + 1 < len(runs) else None
            if prev is not None and prev[3] == r[3] and prev[1] == r[0] - 1:
                prev[1] = r[1]
            elif nxt is not None and nxt[3] == r[3] and nxt[0] == r[1] + 1:
                nxt[0] = r[0]
            else:
                continue
            runs.pop(j)
            changed = True
            break
    return [[int(r[0]), int(r[1])] for r in runs if keep_short or dur(r) >= min_s - 1e-9]


def _n(k: int, one: str, many: str) -> str:
    return f"{k} {one if k == 1 else many}"


def summary_text(row: dict) -> str:
    """One or two sentences from the row's own fields (fixed templates; every number is a field of the row)."""
    team = str(row["team"]).capitalize()
    other = "Away" if row["team"] == "home" else "Home"
    if row["phase"] == "build_up" and row["opposed_pct"] is not None:
        # A build-up episode can mix opposed and unopposed spells: say how much was opposed rather than its majority.
        phase = "build-up, unopposed" if row["opposed_pct"] == 0 else f"build-up, opposed {row['opposed_pct']}% of the time"
    else:
        phase = PHASE_TEXT.get(row["fifa_phase"] or row["phase"], str(row["fifa_phase"] or row["phase"]))
    parts = [f"{team}, {phase}, {row['duration_s']} s"]
    if row["start_third"] and row["end_third"]:
        parts[-1] += (f", in the {row['start_third']} third" if row["start_third"] == row["end_third"]
                      else f", from the {row['start_third']} third to the {row['end_third']} third")
    out = OUT_TEXT.get(row["fifa_out_phase"] or "", None)
    if out:
        parts[-1] += f", against {out}"
    s = parts[0] + "."
    if row["start_reason"] == "set_piece":
        s += f" Started from {SET_PIECE_TEXT.get(row['start_detail'], 'a set piece')}."
    elif row["start_reason"] in START_TEXT:
        s += f" Started: {START_TEXT[row['start_reason']]}."
    if row["passes"]:  # null without an event feed: the summary then says nothing about passes
        s += (f" {_n(row['passes'], 'pass', 'passes')}, {row['passes_completed']} completed, "
              f"{row['passes_breaking_a_line']} broke a line.")
    if row["carrier_pressure_max"] is not None:
        s += f" Pressure on the ball carrier peaked at {row['carrier_pressure_max']} on the 1 to 10 scale."
    if row["engagements"]:
        s += f" {other} engaged the ball carrier {_n(row['engagements'], 'time', 'times')}" + (", including a counter-press." if row["counter_press"] else ".")
    if row["runs_in_behind"]:
        s += f" {_n(row['runs_in_behind'], 'run', 'runs')} in behind."
    if row["cleared"] and row["end_reason"] != "cleared":
        s += f" {team} cleared the ball."
    s += f" Ended: {END_TEXT.get(row['end_reason'], row['end_reason'])}."
    return s + " Context estimated, not yet validated."


class _Timed:
    """A segment table as numpy arrays sorted by a time column, sliced by time with a binary search."""

    def __init__(self, df: pd.DataFrame, tcol: str, cols: tuple[str, ...]):
        d = df.sort_values(tcol, kind="stable")
        self.t = d[tcol].to_numpy(float)
        self.a = {c: (d[c].to_numpy(bool) if c == "in_possession" else d[c].to_numpy(object if c not in ("start_ts", "end_ts") else float))
                  for c in cols}

    def within(self, t0: float, t1: float, mask: np.ndarray | None = None) -> np.ndarray:
        lo, hi = np.searchsorted(self.t, t0 - 1e-9), np.searchsorted(self.t, t1 + 1e-9, side="right")
        idx = np.arange(lo, hi)
        return idx if mask is None else idx[mask[lo:hi]]


class _Cache(dict):
    """Columns extracted once per match (episodes slice them)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def __missing__(self, key):
        v = self[key] = self.fn(key)
        return v


@dataclass
class _Fields:
    items: tuple[ContextField, ...] = field(default=(
        ContextField("p8_state_id", "id", "estimated",
                     "The episode this grid point belongs to (segment states); null in dead-ball time and when no team has the ball."),
    ))


class StatesPackage(ContextPackage):
    name = "p8"
    version = "0.3.0"
    depends_on = ("p1", "p2", "p3", "p4", "p4v", "p5", "p6", "p7")
    fields = _Fields().items

    def compute(self, ctx: ContextInput) -> PackageOutput:
        m, g, cfg, up = ctx.match, ctx.grid, ctx.config, ctx.upstream
        n = len(g)
        ts, per = g["frame_ts"].to_numpy(float), g["period"].to_numpy()
        nf = g["native_frame"].to_numpy(np.int64)
        minute = g["minute"].to_numpy()
        step = float(np.median(np.diff(ts))) if n > 1 else 0.2
        match_id = str(g["match_id"].iloc[0]) if n else ""
        f1, f2, f3, f4 = (up[k].frames for k in ("p1", "p2", "p3", "p4"))
        f4v, f5, f7 = up["p4v"].frames, up["p5"].frames, up["p7"].frames

        team = f4["p4_team"].astype(object).to_numpy()
        pid = f4["p4_phase_id"].to_numpy(dtype=float, na_value=np.nan)
        chain = f1["p1_chain_id"].to_numpy(dtype=float, na_value=np.nan)
        fifa = f4v["p4v_phase"].astype(object).to_numpy()
        form = {t: {k: f2[f"p2_{t}_formation_{k}"].astype(object).to_numpy() for k in ("in", "out")} for t in TEAMS}
        keys: list = []
        for i in range(n):
            if not np.isfinite(pid[i]) or team[i] not in TEAMS:
                keys.append(None)
                continue
            k = [int(pid[i])]
            if cfg["p8_split_formation"]:
                k += [form[team[i]]["in"][i], form[OTHER[team[i]]]["out"][i]]
            if cfg["p8_split_fifa_phase"]:
                k.append(fifa[i])
            keys.append(tuple(k))
        spans = segment(keys, chain, ts, per, cfg["p8_min_state_s"], step, keep_short=bool(cfg["p8_keep_short_chains"]))

        # Upstream segments, by time.
        chains = up["p1"].segments["chains"].set_index("chain_id")
        LB = _Timed(up["p5"].segments["line_breaks"], "ts", ("team", "furthest", "outcome", "units_broken"))
        RU = _Timed(up["p6"].segments["runs"], "start_ts", ("team", "in_possession", "area_end", "type"))
        EN = _Timed(up["p4v"].segments["engagements"], "start_ts", ("team", "type"))
        PR = _Timed(up["p4v"].segments["presses"], "start_ts", ("team", "start_ts", "end_ts", "type"))
        ev = m.events
        ev_frames = ev["Start Frame"].to_numpy(dtype=float) if (not ev.empty and "Start Frame" in ev) else np.array([])
        # Without an event feed passes and events are unknown, not zero: counts are null so an aggregate never reads
        # "no passes" as a pattern.
        has_feed = bool(len(ev_frames))
        has_passes = has_feed and {"Type", "Start X"} <= set(ev.columns)
        # Logged clearances (Metrica: a subtype; the German league: an event type) and who made each action.
        typ = ev["Type"].astype(str).str.upper() if "Type" in ev else pd.Series("", index=ev.index)
        sub = ev["Subtype"].astype(str).str.upper() if "Subtype" in ev else pd.Series("", index=ev.index)
        is_clear = (typ.str.contains("CLEARANCE") | sub.str.contains("CLEARANCE")).to_numpy()
        ev_team = ev["Team"].astype(str).str.lower().to_numpy() if "Team" in ev else np.full(len(ev), "", dtype=object)
        actor = np.array([f"{tm}_{j}" if (j := clean_jersey(fr)) else None for tm, fr in
                          zip(ev_team, ev["From"] if "From" in ev else [None] * len(ev), strict=True)], dtype=object)
        keepers = windows_from_segments(up["p2"].segments).gk
        win = window_index(g, cfg)
        ev_ts = np.interp(ev_frames, nf.astype(float), ts) if len(ev_frames) and n else np.array([])
        data_check = json.dumps({f"{e.package}.{e.metric}": e.state for k in sorted(up) for e in up[k].entries}, sort_keys=True)
        versions = json.dumps({**{k: next((e.package_version for e in up[k].entries), None) for k in sorted(up)},
                               "p8": self.version, "thresholds": THRESHOLDS_VERSION}, sort_keys=True)
        dc_state = {(e.package, e.metric): e.state for k in up for e in up[k].entries}
        dc_cols = {f"data_check_{p}_{m}": dc_state.get((p, m)) for p, m in DC_METRICS if p != "p8"}
        set_piece = f1["p1_set_piece"].astype(object).to_numpy() if "p1_set_piece" in f1 else np.full(n, None, dtype=object)
        poss_all = f1["p1_possession"].astype(object).to_numpy()
        rs = up["p4"].segments.get("restarts", pd.DataFrame(columns=["type", "team", "period", "ts", "source"]))
        rs = rs.loc[rs["source"] == "tracking"] if len(rs) else rs
        rs_ts, rs_team = rs["ts"].to_numpy(float), rs["team"].astype(object).to_numpy()
        rs_per, rs_type = rs["period"].to_numpy(), rs["type"].astype(object).to_numpy()

        allf = pd.concat([f1, f2, f3, f4, f4v, f5, f7], axis=1)
        O = _Cache(lambda c: allf[c].astype(object).to_numpy())
        F = _Cache(lambda c: allf[c].to_numpy(dtype=float, na_value=np.nan))
        B = _Cache(lambda c: allf[c].to_numpy(bool))
        state_of = np.full(n, None, dtype=object)
        rows = []
        for idx, (a, b) in enumerate(spans):
            sl = slice(a, b + 1)
            t = team[a]
            o = OTHER[t]
            t0, t1 = float(ts[a]), float(ts[b])
            sid = f"{match_id}:{int(per[a])}:{idx:04d}"
            state_of[sl] = sid
            bs = O["p1_ball_state"][sl]
            score = {x: F[f"p1_score_{x}"][sl] for x in TEAMS}
            reds = {x: F[f"p1_red_{x}"][sl] for x in TEAMS}
            out_v = O["p4v_out_phase"][sl]
            fblock = O["p4v_fifa_block"][sl]
            lane = O["p3_ball_lane"][sl]
            third = O["p3_ball_third"][sl]
            known_third = [x for x in third if x in THIRDS]
            in_lb = LB.within(t0, t1, LB.a["team"] == t)
            furthest = [x for x in LB.a["furthest"][in_lb] if x in LINE_ORDER]
            r_any = RU.within(t0, t1)
            r_att = r_any[(RU.a["team"][r_any] == t) & RU.a["in_possession"][r_any]]
            r_def = r_any[(RU.a["team"][r_any] == o) & ~RU.a["in_possession"][r_any]]
            e_in = EN.within(t0, t1, EN.a["team"] == o)
            p_in = np.flatnonzero((PR.a["team"] == o) & (PR.a["start_ts"] <= t1 + 1e-9) & (PR.a["end_ts"] >= t0 - 1e-9))
            run_types = pd.Series(RU.a["type"][r_att]).value_counts().sort_index() if len(r_att) else pd.Series(dtype=int)
            eng_types = pd.Series(EN.a["type"][e_in]).value_counts().sort_index() if len(e_in) else pd.Series(dtype=int)
            ev_in = np.flatnonzero((ev_ts >= t0 - 1e-9) & (ev_ts <= t1 + step - 1e-9)) if len(ev_ts) else np.array([], int)

            # How it ended.
            nxt = spans[idx + 1] if idx + 1 < len(spans) else None
            cid = chain[a]
            # 0.3.0 (review AI-01): the chain continues when the next episode has the same chain id in the same period,
            # contiguous or not; only the chain's last episode takes its end (lost, shot, goal, dead ball).
            if nxt is not None and np.isfinite(cid) and chain[nxt[0]] == cid and per[nxt[0]] == per[b]:
                if nxt[0] == b + 1:
                    end_reason = "phase_changed" if pid[nxt[0]] != pid[b] else "formation_changed"
                else:
                    # The grid points between: mostly a contest for the ball (P1 contested), or no phase for another reason.
                    between = poss_all[b + 1:nxt[0]]
                    end_reason = "contested_kept" if len(between) and np.mean(between == "contested") >= 0.5 else "phase_unknown"
            else:
                c_end = chains.at[int(cid), "end_type"] if np.isfinite(cid) and int(cid) in chains.index else None
                if c_end == "lost":
                    end_reason = "possession_lost"
                elif c_end in ("shot", "goal"):
                    end_reason = c_end
                elif c_end == "period_end":
                    end_reason = "end_of_period"
                elif c_end == "dead_ball":
                    after = [x for x in set_piece[b + 1:min(n, b + 1 + int(30 / step))] if x is not None]
                    sp = str(after[0]) if after else None
                    end_reason = ("ball_out_of_play" if sp in OUT_OF_PLAY else "foul_or_set_piece" if sp else "dead_ball")
                else:
                    end_reason = "dead_ball"
            led = bool(np.isfinite(cid) and int(cid) in chains.index and chains.at[int(cid), "end_type"] in ("shot", "goal"))
            # 0.2.0: how it started, and clearances (the founder's review, 23 November 2026).
            own = [j for j in ev_in if ev_team[j] == t]
            cleared = bool(any(is_clear[j] for j in own))
            if own and is_clear[own[-1]] and end_reason in ("possession_lost", "ball_out_of_play", "dead_ball", "phase_changed"):
                end_reason = "cleared"
            prev = spans[idx - 1] if idx > 0 else None
            start_detail = None
            if prev is not None and np.isfinite(cid) and chain[prev[0]] == cid and per[prev[1]] == per[a]:
                start_reason = "continued"
            else:
                s_type = str(chains.at[int(cid), "start_type"]) if np.isfinite(cid) and int(cid) in chains.index else "unknown"
                if s_type.startswith("set_piece:"):
                    start_reason, start_detail = "set_piece", s_type.split(":", 1)[1].lower()
                    if start_detail == "kick_off":
                        start_reason = "kick_off"
                elif s_type in ("recovery", "won"):
                    first_by = actor[own[0]] if own else None
                    gk = keepers.get((t, int(per[a]), int(win[a])))
                    start_reason = "goalkeeper" if first_by is not None and first_by == gk else "won_the_ball"
                elif s_type == "period_start":
                    start_reason = "kick_off"
                elif s_type == "restart":
                    start_reason = "restart"
                else:
                    start_reason = "unknown"
                # 0.3.0 (review AN-03): without a logged restart, one P4 found from positions by this team.
                if start_reason in ("won_the_ball", "restart", "unknown") and len(rs_ts):
                    near = np.flatnonzero((rs_team == t) & (rs_per == per[a]) & (rs_ts >= t0 - cfg["p8_restart_before_s"])
                                          & (rs_ts <= t0 + cfg["p8_restart_after_s"]))
                    if len(near):
                        start_reason, start_detail = "set_piece", str(rs_type[near[0]])

            row = {
                "state_id": sid, "match_id": match_id, "period": int(per[a]), "start_s": round(t0, 2), "end_s": round(t1 + step, 2),
                "duration_s": round(t1 + step - t0, 1), "minute_display": f"{int(minute[a])}'",
                "team": t, "chain_id": int(cid) if np.isfinite(cid) else None,
                "start_reason": start_reason, "start_detail": start_detail, "cleared": cleared if has_feed else None,
                "share_controlled": round(float(np.mean(bs == f"controlled_{t}")), 3),
                "share_in_flight": round(float(np.mean(bs == f"in_flight_{t}")), 3),
                "share_loose_or_contested": round(float(np.mean(np.isin(bs, ["loose", "contested"]))), 3),
                "score_diff": _num(score[t] - score[o], np.median, 0), "red_card_diff": _num(reds[t] - reds[o], np.median, 0),
                "game_state_source": str(f1["p1_game_state_source"].iloc[a]),
                "phase": _mode(O["p4_phase"][sl]), "out_phase": _mode(O["p4_out_phase"][sl]),
                "fifa_phase": _mode(fifa[sl]), "fifa_out_phase": _mode(out_v),
                "opposed_pct": (int(round(100 * float(np.mean(fifa[sl] == "build_up_opposed")) / max(1e-9, float(np.mean(np.isin(fifa[sl], ["build_up_opposed", "build_up_unopposed"]))))))
                                if np.isin(fifa[sl], ["build_up_opposed", "build_up_unopposed"]).any() else None), "block": _mode(O["p4_block"][sl]),
                "press_share": round(float(np.mean([str(x).endswith("_press") and x != "counter_press" for x in out_v])), 3),
                "counter_press_share": round(float(np.mean(out_v == "counter_press")), 3),
                "fifa_mid_block_share": round(float(np.mean(fblock == "mid_block")), 3),
                "fifa_low_block_share": round(float(np.mean(fblock == "low_block")), 3),
                "formation_in": _mode(form[t]["in"][sl]), "formation_out": _mode(form[o]["out"][sl]),
                "team_length_m": _num(F[f"p2_{t}_length_m"][sl], np.median),
                "team_width_m": _num(F[f"p2_{t}_width_m"][sl], np.median),
                "opp_length_m": _num(F[f"p2_{o}_length_m"][sl], np.median),
                "opp_width_m": _num(F[f"p2_{o}_width_m"][sl], np.median),
                "opp_def_line_m": _num(F[f"p2_{o}_def_line_m"][sl], np.median),
                "opp_def_mid_gap_m": _num(F[f"p2_{o}_def_mid_gap_m"][sl], np.median),
                "opp_mid_att_gap_m": _num(F[f"p2_{o}_mid_att_gap_m"][sl], np.median),
                "start_lane": next((x for x in lane if x is not None), None),
                "start_third": known_third[0] if known_third else None,
                "end_lane": next((x for x in lane[::-1] if x is not None), None),
                "end_third": known_third[-1] if known_third else None,
                **{f"share_{x}_third": round(float(np.mean(third == x)), 3) for x in THIRDS},
                "passes": int(len(in_lb)) if has_passes else None,
                "passes_completed": int((LB.a["outcome"][in_lb] == "completed").sum()) if has_passes else None,
                "passes_breaking_a_line": int((np.nan_to_num(LB.a["units_broken"][in_lb].astype(float)) > 0).sum()) if has_passes else None,
                "furthest_line_broken": max(furthest, key=LINE_ORDER.get) if furthest else None,
                "carrier_pressure_median": _num(F["p5_carrier_pressure"][sl], np.median, 0),
                "carrier_pressure_max": _num(F["p5_carrier_pressure"][sl], np.max, 0),
                "goal_side_defenders_median": _num(F["p5_goal_side_defenders"][sl], np.median, 0),
                "overload_at_ball_median": _num(F["p5_overload_at_ball"][sl], np.median, 0),
                "runs": int(len(r_att)), "runs_in_behind": int((RU.a["area_end"][r_att] == "in_behind").sum()),
                **{f"runs_{k}": int(run_types.get(k, 0)) for k in RUN_TYPES},
                "defensive_runs": int(len(r_def)),
                "open_high_value_share": _num(F[f"p7_{t}_high_value_open"][sl], np.median, 3),
                "opp_open_high_value_share": _num(F[f"p7_{o}_high_value_open"][sl], np.median, 3),
                "open_danger_zone": _mode(O["p7_open_danger_zone"][sl]),
                "engagements": int(len(e_in)),
                **{f"engagements_{k}": int(eng_types.get(k, 0)) for k in ENGAGEMENT_TYPES},
                "presses": int(len(p_in)),
                "counter_press": bool((PR.a["type"][p_in] == "counter_press").any() or (EN.a["type"][e_in] == "counter_press").any()),
                "end_reason": end_reason, "led_to_shot": led,
                "first_event_idx": int(ev.index[ev_in[0]]) if len(ev_in) else None,
                "last_event_idx": int(ev.index[ev_in[-1]]) if len(ev_in) else None, "events": int(len(ev_in)) if has_feed else None,
                "ball_detected_share": round(float(B["p1_ball_detected"][sl].mean()), 3),
                "min_outfield_observed": _num(np.minimum(F["p2_home_outfield_n"][sl],
                                                         F["p2_away_outfield_n"][sl]), np.min, 0),
                **dc_cols, "data_check": data_check, "thresholds_version": THRESHOLDS_VERSION, "versions": versions, "label": "estimated",
            }
            row["summary"] = summary_text(row)
            rows.append(row)
        states = pd.DataFrame(rows, columns=list(STATE_COLUMNS))

        frames = pd.DataFrame({"p8_state_id": state_of})
        p4_state = next((e.state for e in up["p4"].entries if e.metric == "phase_of_play"), "blocked")
        p2_state = next((e.state for e in up["p2"].entries if e.metric == "team_shape"), "blocked")
        state = "blocked" if states.empty else ("full" if p4_state == "full" and p2_state == "full" else "degraded")
        in_play = np.isin(O["p1_possession"], list(TEAMS))
        covered = float(pd.notna(state_of)[in_play].mean()) if in_play.any() else 0.0
        measurements = {"episodes": int(len(states)),
                        "median_duration_s": float(states["duration_s"].median()) if len(states) else None,
                        "in_possession_time_covered": round(covered, 4),
                        "end_reasons": {k: int((states["end_reason"] == k).sum()) for k in END_REASONS} if len(states) else {}}
        states["data_check_p8_episodes"] = state
        entries = [self.entry("episodes", "Episodes", state,
                              {"full": "Cut on the team in possession, its chain, the phase and the formations; described from P1 to P7.",
                               "degraded": "Cut and described, but the phase or team shape is degraded for this match, so episodes are approximate.",
                               "blocked": "No phase is labelled for this match, so no episode can be cut."}[state]
                              + " Estimated, not yet validated.",
                              coverage=f"{len(states)} episodes, {covered:.0%} of possession time")]
        return PackageOutput(frames=frames, entries=entries, segments={"states": states}, measurements=measurements)
