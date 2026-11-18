"""
possession.py — context package P1: possession and game state.

Possession on every grid point (``ponta-blueprint/context-packages.md``, P1):

* ``home`` / ``away`` / ``contested`` / ``dead_ball`` / ``unknown``, each with its source:
* ``observed`` — the ball is detected. A team is in control when its nearest player is
  within ``p1_control_radius_m`` and the ball moves slower than
  ``p1_max_control_ball_speed_mps``; both teams within the radius is ``contested``. While
  the ball stays detected but nobody controls it (a pass in flight, a loose ball), the last
  controlling team keeps possession.
* ``inferred`` — the ball is not detected; the last observed possession is carried forward
  for at most ``p1_max_carry_gap_s``.
* ``events`` — beyond that, the event feed's possession (``infer_possession``, event
  anchored) decides, where there is one.
* ``unknown`` — nothing supports a label.

``dead_ball`` comes from the event feed only: the time between the last action and a set
piece restart. Without logged set pieces (SkillCorner's feed is derived from tracking) dead
ball is unknown, never guessed.

Game state on every grid point: score and red cards per team, from a logged event feed
(goals are confirmed by the other team's kick-off), from an uploaded game-state file, or
unknown. Possession chains (segments) record how each possession began and ended.

Possession is a label the engine assigns from distances (estimated); distances, detection
and the event-derived fields are observed. Nothing here reaches the AI as coordinates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.match import NormalisedMatch

from plaiq_engine.analytics.hex_analytics import infer_possession
from plaiq_engine.context.registry import (
    ContextField,
    ContextInput,
    ContextPackage,
    PackageOutput,
)

TEAMS = ("home", "away")
OTHER = {"home": "away", "away": "home"}


# ---------------------------------------------------------------------------
# Uploaded game state (matches without a logged event feed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimedEvent:
    team: str
    period: int
    clock_s: float


@dataclass(frozen=True)
class GameStateInput:
    """
    Game state supplied by the club when there is no logged event feed.

    ``clock_s`` is the match clock in the same convention as the tracking provider's frame
    clock (``NormalisedMatch.frames.time_s``). Either field may be absent: a final score
    without goal times gives the result but not the score at each moment.
    """

    final_score: dict[str, int] | None = None
    goals: tuple[TimedEvent, ...] = ()
    red_cards: tuple[TimedEvent, ...] = ()


def load_game_state_input(data: str | Path | dict) -> GameStateInput:
    """Parse and validate an uploaded game-state JSON (a path or an already-parsed dict)."""
    raw = data if isinstance(data, dict) else json.loads(Path(data).read_text())
    unknown = set(raw) - {"final_score", "goals", "red_cards"}
    if unknown:
        raise ValueError(f"unknown fields: {sorted(unknown)}")

    def timed(items, what: str) -> tuple[TimedEvent, ...]:
        out = []
        for i, it in enumerate(items or []):
            team, period, clock = it.get("team"), it.get("period"), it.get("clock_s")
            if team not in TEAMS:
                raise ValueError(f"{what}[{i}].team must be 'home' or 'away'")
            if period not in (1, 2, 3, 4):
                raise ValueError(f"{what}[{i}].period must be 1 to 4")
            if not isinstance(clock, (int, float)) or clock < 0:
                raise ValueError(f"{what}[{i}].clock_s must be a non-negative number")
            out.append(TimedEvent(team, int(period), float(clock)))
        return tuple(sorted(out, key=lambda e: (e.period, e.clock_s, e.team)))

    final = raw.get("final_score")
    if final is not None:
        if set(final) != set(TEAMS) or not all(isinstance(final[t], int) and final[t] >= 0 for t in TEAMS):
            raise ValueError("final_score must give non-negative integers for 'home' and 'away'")
    goals = timed(raw.get("goals"), "goals")
    if final is not None and goals:
        counted = {t: sum(g.team == t for g in goals) for t in TEAMS}
        if counted != {t: final[t] for t in TEAMS}:
            raise ValueError(f"goals {counted} do not add up to final_score {final}")
    return GameStateInput(final_score=dict(final) if final is not None else None, goals=goals,
                          red_cards=timed(raw.get("red_cards"), "red_cards"))


# ---------------------------------------------------------------------------
# Building blocks (also used by the measurement tool)
# ---------------------------------------------------------------------------

def _tokens(subtype) -> list[str]:
    if subtype is None or (isinstance(subtype, float) and np.isnan(subtype)):
        return []
    return [t.strip() for t in re.split(r"[-,]", str(subtype).upper()) if t.strip()]


def logged_events(match: NormalisedMatch) -> bool:
    """True when the event feed is logged by an event provider (not derived from tracking)."""
    return not match.events.empty and match.provider != "skillcorner"


def ball_on_grid(match: NormalisedMatch, grid: pd.DataFrame) -> pd.DataFrame:
    """Ball position and speed at each grid point's native frame; NaN where undetected."""
    b = match.ball.sort_values("frame", kind="stable").reset_index(drop=True)
    f = b["frame"].to_numpy(dtype=np.int64)
    x, y = b["x"].to_numpy(float), b["y"].to_numpy(float)
    speed = np.full(len(b), np.nan)
    if len(b) > 1:
        fwd = np.r_[f[1:] - f[:-1] == 1, False]
        bwd = np.r_[False, f[1:] - f[:-1] == 1]
        d_next = np.r_[np.hypot(np.diff(x), np.diff(y)), np.nan]
        d_prev = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
        speed = np.where(fwd, d_next, np.where(bwd, d_prev, np.nan)) * match.fps
    bt = pd.DataFrame({"native_frame": f, "ball_x": x, "ball_y": y, "ball_speed": speed})
    out = grid[["native_frame"]].merge(bt, on="native_frame", how="left")
    out["ball_detected"] = out["ball_x"].notna()
    return out


def nearest_on_grid(match: NormalisedMatch, grid: pd.DataFrame, ball: pd.DataFrame) -> pd.DataFrame:
    """Distance from the ball to each team's nearest observed player, where the ball is detected."""
    seen = ball.loc[ball["ball_detected"], ["native_frame", "ball_x", "ball_y"]].drop_duplicates("native_frame")
    tr = match.tracking.loc[match.tracking["frame"].isin(seen["native_frame"]), ["frame", "team", "x", "y"]]
    tr = tr.merge(seen.rename(columns={"native_frame": "frame"}), on="frame", how="inner")
    tr["d"] = np.hypot(tr["x"] - tr["ball_x"], tr["y"] - tr["ball_y"])
    near = tr.groupby(["frame", "team"])["d"].min().unstack("team")
    for t in TEAMS:
        if t not in near.columns:
            near[t] = np.nan
    near = near[list(TEAMS)].rename(columns={"home": "near_home", "away": "near_away"}).reset_index().rename(columns={"frame": "native_frame"})
    return grid[["native_frame"]].merge(near, on="native_frame", how="left")[["near_home", "near_away"]]


def event_possession_on_grid(match: NormalisedMatch, grid: pd.DataFrame) -> np.ndarray:
    """Event-anchored possession (the existing engine rule) at each grid point; None where unknown."""
    if match.events.empty or "Start Frame" not in match.events.columns:
        return np.full(len(grid), None, dtype=object)
    total = int(max(match.frames["frame"].max(), grid["native_frame"].max()))
    s = infer_possession(match.events, total)
    vals = s.reindex(grid["native_frame"].to_numpy()).to_numpy(dtype=object)
    return np.array([v.lower() if isinstance(v, str) else None for v in vals], dtype=object)


def _restart_name(subtype) -> str:
    toks = [t for t in _tokens(subtype) if t != "RETAKEN"]
    return "_".join(re.sub(r"[^A-Z]+", "_", t).strip("_").lower() for t in toks) or "set_piece"


def dead_ball_on_grid(match: NormalisedMatch, grid: pd.DataFrame, cfg: dict, *, teams: bool = False):
    """
    Dead-ball flag and the coming restart's type, from logged set pieces. ``None`` when the
    feed has no set pieces (dead ball is then unknown, not false). With ``teams=True`` also the
    team taking the coming restart (``home``/``away``), as a third array.
    """
    ev = match.events
    setp = np.full(len(grid), None, dtype=object)
    setp_team = np.full(len(grid), None, dtype=object)
    if not logged_events(match) or "Type" not in ev.columns or not (ev["Type"].astype(str) == "SET PIECE").any():
        return (None, setp, setp_team) if teams else (None, setp)
    ev = ev.sort_values(["Start Frame"], kind="stable").reset_index(drop=True)
    dead = np.zeros(len(grid), dtype=bool)
    nf = grid["native_frame"].to_numpy()
    max_frames = cfg["p1_dead_ball_max_s"] * match.fps
    for i in ev.index[ev["Type"].astype(str) == "SET PIECE"]:
        if i == 0 or ev.at[i - 1, "Period"] != ev.at[i, "Period"]:
            continue  # the period's opening kick-off: no stoppage before it
        prev = ev.loc[i - 1]
        stop = prev["End Frame"] if pd.notna(prev.get("End Frame")) else prev["Start Frame"]
        restart = ev.at[i, "Start Frame"]
        if pd.isna(stop) or pd.isna(restart) or not (0 < restart - stop <= max_frames):
            continue
        m = (nf >= stop) & (nf < restart)
        dead[m] = True
        setp[m] = _restart_name(ev.at[i, "Subtype"])
        team = str(ev.at[i, "Team"]).lower() if "Team" in ev.columns else ""
        setp_team[m] = team if team in TEAMS else None
    return (dead, setp, setp_team) if teams else (dead, setp)


def goals_and_cards(match: NormalisedMatch, cfg: dict) -> tuple[list[dict], list[dict]]:
    """
    Goals and red cards from a logged event feed. A shot is a goal when its subtype has a
    ``GOAL`` token (``ON TARGET-GOAL``, ``ON TARGET, GOAL``; never ``GOAL KICK``). A goal is
    confirmed when the other team kicks off within ``p1_goal_kickoff_window_s``.
    """
    if not logged_events(match):
        return [], []
    ev = match.events.sort_values("Start Frame", kind="stable")
    types = ev["Type"].astype(str)
    kick = ev.loc[(types == "SET PIECE") & ev["Subtype"].map(lambda s: "KICK OFF" in _tokens(s))]
    window = cfg["p1_goal_kickoff_window_s"] * match.fps
    goals = []
    for _, r in ev.loc[types == "SHOT"].iterrows():
        toks = _tokens(r["Subtype"])
        if "GOAL" not in toks:
            continue
        team = str(r["Team"]).lower()
        after = kick.loc[(kick["Start Frame"] > r["Start Frame"]) & (kick["Start Frame"] <= r["Start Frame"] + window)]
        confirmed = bool((after["Team"].astype(str).str.lower() == OTHER.get(team)).any())
        goals.append({"team": team, "frame": int(r["Start Frame"]), "subtype": str(r["Subtype"]), "confirmed": confirmed})
    reds = []
    for _, r in ev.loc[types == "CARD"].iterrows():
        toks = _tokens(r["Subtype"])
        if "RED" in toks or "SECOND YELLOW" in toks:
            reds.append({"team": str(r["Team"]).lower(), "frame": int(r["Start Frame"])})
    return goals, reds


def game_state_on_grid(match: NormalisedMatch, grid: pd.DataFrame, cfg: dict, upload: GameStateInput | None) -> tuple[dict, str, dict]:
    """Score and red cards per team at each grid point, the source, and details for the report."""
    n = len(grid)
    nf = grid["native_frame"].to_numpy()
    cols = {k: np.full(n, np.nan) for k in ("score_home", "score_away", "red_home", "red_away")}
    info: dict = {}
    if logged_events(match):
        goals, reds = goals_and_cards(match, cfg)
        for t in TEAMS:
            gf = np.array(sorted(g["frame"] for g in goals if g["team"] == t), dtype=np.int64)
            rf = np.array(sorted(c["frame"] for c in reds if c["team"] == t), dtype=np.int64)
            cols[f"score_{t}"] = np.searchsorted(gf, nf, side="right").astype(float)
            cols[f"red_{t}"] = np.searchsorted(rf, nf, side="right").astype(float)
        info = {"goals": goals, "red_cards": reds, "unconfirmed_goals": sum(not g["confirmed"] for g in goals)}
        return cols, "events", info
    if upload is not None and upload.goals:
        per, clock = grid["period"].to_numpy(), grid["clock_s"].to_numpy()
        for t in TEAMS:
            for what, items in (("score", upload.goals), ("red", upload.red_cards)):
                acc = np.zeros(n)
                for e in items:
                    if e.team == t:
                        acc += ((per > e.period) | ((per == e.period) & (clock >= e.clock_s))).astype(float)
                cols[f"{what}_{t}"] = acc
        info = {"final_score": upload.final_score, "goals_uploaded": len(upload.goals), "red_cards_uploaded": len(upload.red_cards)}
        return cols, "upload", info
    if upload is not None and upload.final_score is not None:
        return cols, "final_only", {"final_score": upload.final_score}
    return cols, "unknown", {}


def label_possession(grid: pd.DataFrame, ball: pd.DataFrame, near: pd.DataFrame, event_poss: np.ndarray,
                     dead: np.ndarray | None, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Possession label, source and the raw control call (``home``/``away``/``contested``/None,
    detected frames only) at each grid point. Runs period by period, in time order.
    """
    n = len(grid)
    lab = np.full(n, "unknown", dtype=object)
    src = np.full(n, "unknown", dtype=object)
    r, vmax, gap_max = cfg["p1_control_radius_m"], cfg["p1_max_control_ball_speed_mps"], cfg["p1_max_carry_gap_s"]
    nh, na = near["near_home"].to_numpy(float), near["near_away"].to_numpy(float)
    det = ball["ball_detected"].to_numpy(bool)
    spd = ball["ball_speed"].to_numpy(float)
    slow = ~(spd > vmax)  # unknown speed counts as slow
    home_in = det & (nh <= r) & slow
    away_in = det & (na <= r) & slow
    control = np.full(n, None, dtype=object)
    control[home_in & away_in] = "contested"
    control[home_in & ~away_in] = "home"
    control[away_in & ~home_in] = "away"

    ts, per = grid["frame_ts"].to_numpy(float), grid["period"].to_numpy()
    min_switch = cfg["p1_min_switch_s"]
    state, last_seen, cur_per = None, -np.inf, None
    pend_team, pend_start = None, None  # a switch to the other team, waiting for p1_min_switch_s of control
    for i in range(n):
        if per[i] != cur_per:
            cur_per, state, last_seen, pend_team = per[i], None, -np.inf, None
        if dead is not None and dead[i]:
            lab[i], src[i], state, pend_team = "dead_ball", "events", None, None
            continue
        if det[i]:
            last_seen = ts[i]
            c = control[i]
            if c in TEAMS and (state is None or c == state):
                state, pend_team = c, None
                lab[i], src[i] = c, "observed"
            elif c in TEAMS:  # the other team controls: switch only once it has held the ball long enough
                if pend_team != c:
                    pend_team, pend_start = c, i
                if ts[i] - ts[pend_start] + 1e-9 >= min_switch:
                    state, pend_team = c, None
                    lab[pend_start:i + 1] = c  # the switch happened when the new team took control
                    src[pend_start:i + 1] = "observed"
                else:
                    lab[i], src[i] = state, "observed"
            elif c == "contested":
                lab[i], src[i] = "contested", "observed"
            elif state is not None:
                lab[i], src[i] = state, "observed"
                pend_team = None
            continue
        if state is not None and ts[i] - last_seen <= gap_max:
            lab[i], src[i] = state, "inferred"
        elif event_poss[i] in TEAMS:
            lab[i], src[i] = event_poss[i], "events"
    return lab, src, control


def build_chains(grid: pd.DataFrame, lab: np.ndarray, src: np.ndarray, setp: np.ndarray, match: NormalisedMatch,
                 cfg: dict, goals: list[dict], setp_team: np.ndarray | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Possession chains: maximal stretches held by one team. Contested and unknown points do not
    break a chain; the other team taking the ball, a dead ball or the end of a period do.

    A chain after a dead ball starts with the set piece only when its team took the set piece
    (``setp_team``); when the other team wins the first touch (a corner headed clear), its chain
    starts ``won``. Before P1 0.2.0 the set piece went to whichever team had the ball first.
    """
    n = len(grid)
    ts, per, minute = grid["frame_ts"].to_numpy(float), grid["period"].to_numpy(), grid["minute"].to_numpy()
    chain = np.full(n, -1, dtype=np.int64)
    f0 = int(match.frames["frame"].min())
    ev = match.events if logged_events(match) or not match.events.empty else pd.DataFrame()
    if not ev.empty and "Start Frame" in ev.columns:
        ev = ev.assign(_ts=(ev["Start Frame"] - f0) / match.fps, _team=ev["Team"].astype(str).str.lower(), _type=ev["Type"].astype(str))
    goal_ts = {(g["team"], round((g["frame"] - f0) / match.fps, 2)) for g in goals}

    def near_event(team: str, t: float, types: set[str], window: float, before_only: bool = False) -> pd.DataFrame:
        if ev.empty:
            return ev
        lo = t - window
        hi = t if before_only else t + window
        return ev.loc[(ev["_team"] == team) & ev["_type"].isin(types) & (ev["_ts"] >= lo) & (ev["_ts"] <= hi)]

    rows, cid = [], 0
    cur = None  # [team, start_i, start_type]
    last_break, pending_sp, pending_team = "period_start", None, None

    def close(end_i: int, end_type: str):
        nonlocal cur, cid
        team, s_i, s_type = cur
        if end_type in ("lost", "dead_ball", "period_end"):
            shots = near_event(team, ts[end_i], {"SHOT"}, cfg["p1_chain_shot_window_s"], before_only=True)
            if not shots.empty:
                is_goal = any((team, round(t, 2)) in goal_ts for t in shots["_ts"])
                end_type = "goal" if is_goal else "shot"
        seg = slice(s_i, end_i + 1)
        chain[seg] = cid
        obs = float(np.mean(src[seg] == "observed"))
        rows.append({"chain_id": cid, "period": int(per[s_i]), "team": team, "start_ts": float(ts[s_i]), "end_ts": float(ts[end_i]),
                     "duration_s": round(float(ts[end_i] - ts[s_i]), 3), "start_minute": int(minute[s_i]),
                     "start_type": s_type, "end_type": end_type, "observed_share": round(obs, 4)})
        cid += 1
        cur = None

    for i in range(n):
        if i > 0 and per[i] != per[i - 1]:
            if cur is not None:
                close(i - 1, "period_end")
            last_break, pending_sp, pending_team = "period_start", None, None
        lab_i = lab[i]
        if lab_i == "dead_ball":
            if cur is not None:
                close(i - 1, "dead_ball")
            last_break, pending_sp = "dead_ball", setp[i]
            pending_team = setp_team[i] if setp_team is not None else None
            continue
        if lab_i in TEAMS:
            if cur is not None and cur[0] == lab_i:
                continue
            if cur is not None:
                close(i - 1, "lost")
                last_break = "other_team"
            if last_break == "period_start":
                s_type = "period_start"
            elif last_break == "dead_ball":
                if pending_sp and pending_team is not None and pending_team != lab_i:
                    s_type = "won"  # the other team's restart; this team won the first touch
                else:
                    s_type = f"set_piece:{pending_sp}" if pending_sp else "restart"
            elif last_break == "other_team":
                rec = near_event(lab_i, ts[i], {"RECOVERY"}, cfg["p1_chain_event_window_s"])
                s_type = "recovery" if not rec.empty else "won"
            else:
                s_type = "unknown"
            cur = [lab_i, i, s_type]
            last_break, pending_sp = "in_chain", None
    if cur is not None:
        close(n - 1, "period_end")
    segs = pd.DataFrame(rows, columns=["chain_id", "period", "team", "start_ts", "end_ts", "duration_s", "start_minute",
                                       "start_type", "end_type", "observed_share"])
    return chain, segs


def _runs(mask: np.ndarray, ts: np.ndarray, per: np.ndarray) -> list[float]:
    """Durations (s) of consecutive True runs within each period."""
    out, start = [], None
    for i in range(len(mask)):
        if mask[i] and (start is None or per[i] != per[start]):
            if start is not None:
                out.append(ts[i - 1] - ts[start])
            start = i
        elif not mask[i] and start is not None:
            out.append(ts[i - 1] - ts[start])
            start = None
    if start is not None:
        out.append(ts[len(mask) - 1] - ts[start])
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------
# The package
# ---------------------------------------------------------------------------

@dataclass
class _Fields:
    items: tuple[ContextField, ...] = field(default=(
        ContextField("p1_possession", "label", "estimated",
                     "Which team has the ball: home, away, contested, dead_ball or unknown. Assigned by rules from distances, the ball and the event feed."),
        ContextField("p1_possession_source", "label", "observed",
                     "Where the possession label came from: observed (ball detected), inferred (carried through a short detection gap), events (event feed), unknown."),
        ContextField("p1_ball_detected", "bool", "observed", "The ball has a position at the grid point's native frame."),
        ContextField("p1_ball_speed_mps", "m/s", "observed", "Ball speed from consecutive native frames; null when undetected or isolated."),
        ContextField("p1_nearest_home_m", "m", "observed", "Distance from the ball to the nearest observed home player; null when the ball is undetected."),
        ContextField("p1_nearest_away_m", "m", "observed", "Distance from the ball to the nearest observed away player; null when the ball is undetected."),
        ContextField("p1_chain_id", "id", "estimated", "Possession chain this grid point belongs to; null outside a chain (dead ball, before the first possession)."),
        ContextField("p1_dead_ball", "bool", "observed", "Between the last action and a logged set-piece restart; null when the feed has no set pieces."),
        ContextField("p1_set_piece", "label", "observed", "The restart a dead ball is waiting for (throw_in, corner_kick, free_kick_indirect …); null otherwise."),
        ContextField("p1_score_home", "goals", "observed", "Home goals so far; null when the score at this moment is unknown."),
        ContextField("p1_score_away", "goals", "observed", "Away goals so far; null when the score at this moment is unknown."),
        ContextField("p1_red_home", "cards", "observed", "Home red cards so far; null when unknown."),
        ContextField("p1_red_away", "cards", "observed", "Away red cards so far; null when unknown."),
        ContextField("p1_game_state_source", "label", "observed", "Where the score and cards came from: events, upload, final_only (result known, timeline not) or unknown."),
    ))


class PossessionPackage(ContextPackage):
    name = "p1"
    version = "0.2.0"
    fields = _Fields().items

    def compute(self, ctx: ContextInput) -> PackageOutput:
        m, g, cfg = ctx.match, ctx.grid, ctx.config
        upload = ctx.extras.get("game_state")
        ball = ball_on_grid(m, g)
        near = nearest_on_grid(m, g, ball)
        event_poss = event_possession_on_grid(m, g)
        dead, setp, setp_team = dead_ball_on_grid(m, g, cfg, teams=True)
        lab, src, control = label_possession(g, ball, near, event_poss, dead, cfg)
        gs, gs_source, gs_info = game_state_on_grid(m, g, cfg, upload)
        chain, chains = build_chains(g, lab, src, setp, m, cfg, gs_info.get("goals", []) if gs_source == "events" else [], setp_team)

        frames = pd.DataFrame({
            "p1_possession": lab,
            "p1_possession_source": src,
            "p1_ball_detected": ball["ball_detected"].to_numpy(bool),
            "p1_ball_speed_mps": np.round(ball["ball_speed"].to_numpy(float), 3),
            "p1_nearest_home_m": np.round(near["near_home"].to_numpy(float), 3),
            "p1_nearest_away_m": np.round(near["near_away"].to_numpy(float), 3),
            "p1_chain_id": pd.array([int(c) if c >= 0 else None for c in chain], dtype="Int64"),
            "p1_dead_ball": pd.array(dead if dead is not None else [None] * len(g), dtype="boolean"),
            "p1_set_piece": setp,
            "p1_score_home": pd.array(gs["score_home"], dtype="Float64").astype("Int64"),
            "p1_score_away": pd.array(gs["score_away"], dtype="Float64").astype("Int64"),
            "p1_red_home": pd.array(gs["red_home"], dtype="Float64").astype("Int64"),
            "p1_red_away": pd.array(gs["red_away"], dtype="Float64").astype("Int64"),
            "p1_game_state_source": gs_source,
        })

        # -- measurements and the Data check ---------------------------------------
        ts, per = g["frame_ts"].to_numpy(float), g["period"].to_numpy()
        in_play = lab != "dead_ball"
        n_play = max(int(in_play.sum()), 1)
        shares = {s: round(float(((src == s) & in_play).sum() / n_play), 4) for s in ("observed", "inferred", "events", "unknown")}
        not_observed = 1.0 - shares["observed"]
        gaps = np.array(_runs(~ball["ball_detected"].to_numpy(bool), ts, per))
        undetected_time = float(gaps.sum()) if len(gaps) else 0.0
        gap_bands = {}
        for lo, hi, name in ((0, 1, "under_1s"), (1, 3, "1_to_3s"), (3, 10, "3_to_10s"), (10, np.inf, "over_10s")):
            sel = (gaps >= lo) & (gaps < hi) if len(gaps) else np.array([], bool)
            gap_bands[name] = round(float(gaps[sel].sum() / undetected_time), 4) if undetected_time else 0.0
        labelled = float(((lab != "unknown")).mean()) if len(lab) else 0.0
        measurements = {
            "grid_points": int(len(g)),
            "ball_detected_share": round(float(ball["ball_detected"].mean()), 4) if len(g) else 0.0,
            "ball_detected_share_by_period": {int(p): round(float(ball["ball_detected"][per == p].mean()), 4) for p in sorted(pd.unique(per))},
            "detection_gaps": {"count": int(len(gaps)), "median_s": round(float(np.median(gaps)), 2) if len(gaps) else None,
                               "p90_s": round(float(np.percentile(gaps, 90)), 2) if len(gaps) else None,
                               "max_s": round(float(gaps.max()), 2) if len(gaps) else None,
                               "undetected_time_by_gap_length": gap_bands},
            "possession_source_share_in_play": shares,
            "label_share": {k: round(float((lab == k).mean()), 4) for k in ("home", "away", "contested", "dead_ball", "unknown")},
            "control_share_detected": {k: round(float((control == k).sum() / max(int(ball["ball_detected"].sum()), 1)), 4)
                                       for k in ("home", "away", "contested")},
            "dead_ball_known": dead is not None,
            "chains": {"count": int(len(chains)),
                       "known_start_share": round(float((~chains["start_type"].isin(["unknown"])).mean()), 4) if len(chains) else None,
                       "known_end_share": round(float((~chains["end_type"].isin(["unknown"])).mean()), 4) if len(chains) else None,
                       "start_types": chains["start_type"].str.split(":").str[0].value_counts().to_dict(),
                       "end_types": chains["end_type"].value_counts().to_dict()},
            "game_state": {"source": gs_source, **{k: v for k, v in gs_info.items() if k != "goals"},
                           "goals": [{k: g_[k] for k in ("team", "subtype", "confirmed")} for g_ in gs_info.get("goals", [])]},
        }

        degraded_at = cfg["p1_degraded_inferred_share"]
        poss_state = "full" if not_observed <= degraded_at else "degraded"
        entries = [self.entry(
            "possession", "Possession", poss_state,
            "The ball is detected for most of the match, so possession is observed." if poss_state == "full"
            else f"More than {_pct(degraded_at)} of in-play time is not observed (the ball was undetected), so possession there is inferred or taken from the event feed.",
            measure=f"{_pct(shares['observed'])} observed, {_pct(shares['inferred'])} inferred, {_pct(shares['events'])} from events, {_pct(shares['unknown'])} unknown",
            unlock=None if poss_state == "full" else "A tracking export with better ball detection.",
            coverage=f"{_pct(labelled)} of match time labelled")]
        if gs_source == "events":
            unconfirmed = gs_info.get("unconfirmed_goals", 0)
            gs_state = "full" if dead is not None and not unconfirmed else "degraded"
            why = "Score, red cards and dead-ball time come from the logged event feed."
            if unconfirmed:
                why = f"{unconfirmed} goal(s) in the event feed are not followed by the other team's kick-off, so the score timeline may be wrong."
            entries.append(self.entry("game_state", "Game state", gs_state, why,
                                      measure=f"{len(gs_info['goals'])} goal(s), {len(gs_info['red_cards'])} red card(s)",
                                      coverage=f"{_pct(1.0)} of match time"))
        elif gs_source == "upload":
            entries.append(self.entry("game_state", "Game state", "degraded" if dead is None else "full",
                                      "Score and red cards come from the uploaded game state." + (" Dead-ball time is unknown (no logged set pieces)." if dead is None else ""),
                                      coverage=f"{_pct(1.0)} of match time", unlock="Add the logged event feed for this match." if dead is None else None))
        elif gs_source == "final_only":
            fs = gs_info["final_score"]
            entries.append(self.entry("game_state", "Game state", "degraded",
                                      "Only the final score is known, so the score at each moment, red cards and dead-ball time are unknown.",
                                      measure=f"final score {fs['home']}–{fs['away']}",
                                      unlock="Upload goal times and red cards, or add the logged event feed.", coverage="0.0% of match time"))
        else:
            entries.append(self.entry("game_state", "Game state", "blocked",
                                      "No logged event feed and no uploaded game state, so score, cards and dead-ball time are unknown.",
                                      unlock="Upload the final score, goal times and red cards, or add the logged event feed.",
                                      coverage="0.0% of match time"))
        return PackageOutput(frames=frames, entries=entries, segments={"chains": chains}, measurements=measurements)
