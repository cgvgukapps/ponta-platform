"""
pff.py — read PFF FC's FIFA World Cup 2022 data as Ponta reference data (development and measurement only).

PFF FC released broadcast tracking (Sportlogiq, smoothed by PFF) and its event data for all 64 matches of the 2022
FIFA World Cup, on request through its access form. **No licence is stated; the specifications are marked proprietary.**
Under the founder's decision of 23 November 2026 (``ponta-blueprint/legal-compliance.md``) it may be used for internal
development and measurement only: every artefact derived from it names PFF as its source, and nothing derived from it
reaches a customer, pilot, demo, the website or any public claim without PFF's written approval. The files stay
outside every repository (``~/ponta-reference-data/pff-worldcup2022``).

This is a **reference reader for measurement**, not a supported provider (Ponta's providers are Metrica, SkillCorner,
StatsBomb and STATSports). It builds a ``NormalisedMatch`` so the context packages run unchanged, plus PFF's own
event table for comparisons (pressure on the ball, lines broken, possession sequences).

- **No names are read.** Players are ``home_<shirt>`` / ``away_<shirt>``; PFF's player ids are used only to find a
  player's side, shirt and position group in the roster. Team and competition ids only, never names; no video links.
- **Tracking:** PFF's smoothed player positions (a Kalman filter over Sportlogiq's broadcast tracking, PFF tracking
  specification 2.2). Players off camera are estimated by the provider, as in SkillCorner's open data; the share of
  visible positions is kept for the Data check report, and each row carries PFF's flag (schema column ``observed``;
  pickles cached before 24 November 2026 do not). **The ball is the tracked ball** (``balls``), not PFF's smoothed
  ball, which PFF places at the ball carrier from its own events: using it would make Ponta's possession agree with
  PFF's events by construction. ``ball="smoothed"`` reads it instead.
- **Consistency:** a period is dropped when its player tracking disagrees with PFF's own ball and events (the player
  nearest the ball is on the acting team less often than chance). This removes extra time in 10506 and 10517. PFF did
  not track extra time in 10510 and 10511; those events have no frame and are counted, not used.
- **Coordinates:** centred metres. Each period is rotated so the home team attacks +x, from where the home goalkeeper
  stands (checked against the metadata's ``homeTeamStartLeft``).
- **Clock:** the video frame number is the frame; ``time_s`` is the period's elapsed time plus 0, 45, 90 or 105 minutes.
  Events sit on the exact frame PFF links them to (``possession_event_id`` / ``game_event_id`` in the tracking).
- **Events** are translated into the vocabulary P1 reads (Metrica's): ``SET PIECE``, ``PASS`` / ``BALL LOST`` (with
  ``CROSS``), ``SHOT``, ``CHALLENGE`` (``WON`` / ``LOST``), ``CLEARANCE``, ``CARRY``, ``RECOVERY`` (a change of team
  in open play; ``INTERCEPTION`` after an intercepted or blocked pass), ``BALL OUT``, ``FAULT RECEIVED``, ``CARD``.
  A pass or carry ends where the next event starts: PFF logs the receiving touch as its own event (``IT``, added
  1 May 2025 "to make XY coordinates available for the moment players receive the ball").

Parsed matches are cached as pickles in ``cache`` (outside the repository).
"""

from __future__ import annotations

import bz2
import dataclasses
import json
import pickle
import sys as _sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.match import NormalisedMatch
from matrix_engine.pitch_config import PitchConfig
from matrix_engine.schema import coerce_schema, validate_schema

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_tag import tag as cache_tag

PROVIDER = "pff"
#: The reader's version, in its cache key (cache_tag): 2 since the observed flag and the period check.
READER_VERSION = "2"
SOURCE = "PFF FC, FIFA World Cup 2022 (no licence stated; internal development only)"
PERIOD_OFFSET_S = {1: 0.0, 2: 45 * 60.0, 3: 90 * 60.0, 4: 105 * 60.0}
#: PFF's set-piece codes (``setpieceType``) as P1's set-piece subtypes. Drop balls (D) are not set pieces for P1.
SETPIECE = {"K": "KICK OFF", "C": "CORNER KICK", "F": "FREE KICK", "G": "GOAL KICK", "P": "PENALTY", "T": "THROW IN"}
#: PFF's shot outcomes (``shotOutcomeType``) as P1's shot subtypes.
SHOT_RESULT = {"G": "ON TARGET-GOAL", "S": "ON TARGET-SAVED", "L": "ON TARGET-SAVED", "B": "ON TARGET-BLOCKED",
               "F": "OFF TARGET-SAVED", "O": "OFF TARGET-OUT", "C": "OFF TARGET-BLOCKED"}
CARD = {"Y": "YELLOW", "R": "RED", "S": "SECOND YELLOW"}
#: Duels (both sides compete for a loose ball) as opposed to tackles and dribbles (one side has the ball).
KICKOFFS = ("FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF")
#: Restarts that follow the ball crossing a line (the team taking them did not put it out).
OUT_RESTARTS = ("T", "G", "C")
DUELS = {"FIFTY": "GROUND", "A": "AERIAL", "G": "GROUND", "K": "GROUND"}
#: A pass or carry's end is the next event's start only when it comes within this many seconds.
END_WITHIN_S = 10.0
#: A period is kept only if, at PFF's passes, crosses and shots, the tracked player nearest the tracked ball is on the
#: acting team more often than chance (0.5). Engineering check, not a football threshold: in two extra-time matches
#: (10506, 10517) the players' tracking does not line up with PFF's own ball and events (about 0.2), and no time
#: offset within five minutes repairs it; normal time scores 0.6 to 0.9.
CONSISTENT_MIN_SHARE = 0.5
CONSISTENT_MIN_EVENTS = 20
EVENT_COLUMNS = ("frame", "period", "game_event_id", "possession_event_id", "gev", "pev", "team", "setpiece", "x", "y",
                 "player", "target", "pass_outcome", "shot_outcome", "challenge_type", "challenger", "challenger_team",
                 "winner_team", "out_type", "out_team", "foul_type", "culprit_team", "card", "pressure", "initial_pressure",
                 "pressure_team", "lines_broken", "sequence", "duration")


@dataclass
class Reference:
    """PFF's own event table (comparison data only; no names) and what the reader saw."""

    events: pd.DataFrame
    visible_share: float          # share of player positions PFF marks visible rather than estimated
    ball_share: float             # share of frames with a tracked ball
    orientation_agrees: bool      # the goalkeeper-based rotation matches the metadata's homeTeamStartLeft
    dropped_periods: tuple[int, ...] = ()   # periods removed because players and ball disagree (see CONSISTENT_MIN_SHARE)
    events_without_frames: int = 0          # PFF events with no tracking frame (extra time PFF did not track, for example)
    consistency: dict | None = None         # per period: (share of on-ball events matched, events checked)


def match_ids(data_dir: Path) -> list[str]:
    return sorted(p.stem for p in (Path(data_dir) / "Metadata").glob("*.json"))


def _json(path: Path):
    with open(path, encoding="utf-8") as fh:
        out = json.load(fh)
    return out[0] if isinstance(out, list) and len(out) == 1 and isinstance(out[0], dict) else out


def read_metadata(data_dir: Path, gid: str) -> dict:
    """Team ids, competition, season, frame rate, pitch and starting side. Names and video links are not read."""
    md = _json(Path(data_dir) / "Metadata" / f"{gid}.json")
    pitch = (md.get("stadium") or {}).get("pitches") or [{}]
    return {"teams": {"home": str(md["homeTeam"]["id"]), "away": str(md["awayTeam"]["id"])},
            "competition": f"{PROVIDER}:{md['competition']['id']}", "season": str(md.get("season") or ""),
            "fps": float(md.get("fps") or 29.97), "length": float(pitch[-1].get("length") or 105.0),
            "width": float(pitch[-1].get("width") or 68.0), "home_start_left": md.get("homeTeamStartLeft"),
            "home_start_left_et": md.get("homeTeamStartLeftExtraTime")}


def read_roster(data_dir: Path, gid: str, teams: dict[str, str]) -> dict[str, dict]:
    """PFF player id to side, shirt and position group. Names are not read."""
    side = {v: k for k, v in teams.items()}
    out = {}
    for r in _json(Path(data_dir) / "Rosters" / f"{gid}.json"):
        team = side.get(str((r.get("team") or {}).get("id")))
        pid = str((r.get("player") or {}).get("id"))
        if team and r.get("shirtNumber") not in (None, ""):
            out[pid] = {"team": team, "shirt": str(int(float(r["shirtNumber"]))), "position": r.get("positionGroupType"),
                        "started": bool(r.get("started"))}
    return out


def read_tracking(path: Path, ball: str = "tracked") -> dict:
    """Players' and the ball's positions per frame in PFF's coordinates, and the frames PFF links each event to."""
    pl: dict[str, list] = {"frame": [], "team": [], "shirt": [], "x": [], "y": [], "visible": []}
    bl: dict[str, list] = {"frame": [], "x": [], "y": []}
    fr: dict[str, list] = {"frame": [], "period": [], "elapsed": []}
    pev_frame: dict[int, int] = {}
    gev_frame: dict[int, int] = {}
    last = None
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            f, period = t.get("frameNum"), t.get("period")
            if f is None or period is None:
                continue
            if t.get("possession_event_id") is not None:
                pev_frame.setdefault(int(t["possession_event_id"]), int(f))
            if t.get("game_event_id") is not None:
                gev_frame.setdefault(int(t["game_event_id"]), int(f))
            if f == last:
                continue  # PFF duplicates a frame that carries more than one possession event
            last = f
            fr["frame"].append(int(f))
            fr["period"].append(int(period))
            fr["elapsed"].append(float(t.get("periodElapsedTime") or 0.0))
            for team, key in (("home", "homePlayersSmoothed"), ("away", "awayPlayersSmoothed")):
                for p in t.get(key) or []:
                    if p.get("x") is None or p.get("y") is None or p.get("jerseyNum") in (None, ""):
                        continue
                    pl["frame"].append(int(f))
                    pl["team"].append(team)
                    pl["shirt"].append(str(int(float(p["jerseyNum"]))))
                    pl["x"].append(float(p["x"]))
                    pl["y"].append(float(p["y"]))
                    pl["visible"].append(p.get("visibility") == "VISIBLE")
            b = t.get("ballsSmoothed") if ball == "smoothed" else (t.get("balls") or [None])[0]
            if b and b.get("x") is not None and b.get("y") is not None:
                bl["frame"].append(int(f))
                bl["x"].append(float(b["x"]))
                bl["y"].append(float(b["y"]))
    players = pd.DataFrame(pl).drop_duplicates(["frame", "team", "shirt"])  # one position per shirt and frame
    return {"players": players, "ball": pd.DataFrame(bl), "frames": pd.DataFrame(fr), "pev_frame": pev_frame, "gev_frame": gev_frame}


def _signs(players: pd.DataFrame, frames: pd.DataFrame, roster: dict) -> dict[int, int]:
    """Per period, -1 when the home goalkeeper stands at +x (the home team attacks -x: rotate that period), else +1."""
    gk = {p["shirt"] for p in roster.values() if p["team"] == "home" and p["position"] == "GK"}
    period_of = pd.Series(frames["period"].to_numpy(), index=frames["frame"].to_numpy())
    home = players.loc[(players["team"] == "home") & players["shirt"].isin(gk)]
    home = home.assign(period=home["frame"].map(period_of))
    out = {}
    for period, g in home.groupby("period"):
        main = g.groupby("shirt")["x"].agg(["size", "mean"]).sort_values("size").iloc[-1]
        out[int(period)] = -1 if main["mean"] > 0 else 1
    return out


def _orientation_agrees(signs: dict[int, int], md: dict) -> bool:
    left, left_et = md["home_start_left"], md["home_start_left_et"]
    want = {}
    if left is not None:
        want |= {1: 1 if left else -1, 2: -1 if left else 1}
    if left_et is not None:
        want |= {3: 1 if left_et else -1, 4: -1 if left_et else 1}
    return all(signs.get(p) == s for p, s in want.items() if p in signs)


def _pid(v) -> str | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(int(float(v)))


def read_events(path: Path, roster: dict, pev_frame: dict, gev_frame: dict, video_clock: tuple[np.ndarray, np.ndarray],
                signs: dict[int, int]) -> pd.DataFrame:
    """PFF's events on tracking frames, in Ponta's coordinates, with ids for player and side. No names."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    vt, vf = video_clock

    def ref(pid) -> str | None:
        p = roster.get(_pid(pid) or "")
        return f"{p['team']}_{p['shirt']}" if p else None

    def team_of(pid) -> str | None:
        p = roster.get(_pid(pid) or "")
        return p["team"] if p else None

    rows = []
    for e in raw:
        ge, pe = e.get("gameEvents") or {}, e.get("possessionEvents") or {}
        fo = e.get("fouls") or {}
        pev_id, gev_id = e.get("possessionEventId"), e.get("gameEventId")
        frame = pev_frame.get(int(pev_id)) if pev_id is not None else None
        if frame is None and gev_id is not None:
            frame = gev_frame.get(int(gev_id))
        if frame is None and e.get("startTime") is not None:  # no linked frame: the nearest video time within 2 s
            k = int(np.clip(np.searchsorted(vt, float(e["startTime"]) * 1000.0), 0, len(vt) - 1))
            frame = int(vf[k]) if abs(vt[k] - float(e["startTime"]) * 1000.0) <= 2000 else None
        if frame is None:
            continue
        period = int(ge.get("period") or 0) or None
        s = signs.get(period, 1)
        b = (e.get("ball") or [None])[0] if isinstance(e.get("ball"), list) else e.get("ball")
        x = s * float(b["x"]) if b and b.get("x") is not None else np.nan
        y = s * float(b["y"]) if b and b.get("y") is not None else np.nan
        team = None if ge.get("homeTeam") is None else ("home" if ge["homeTeam"] else "away")
        pev = pe.get("possessionEventType")
        actor = {"PA": pe.get("passerPlayerId"), "CR": pe.get("crosserPlayerId"), "SH": pe.get("shooterPlayerId"),
                 "CL": pe.get("clearerPlayerId"), "BC": pe.get("ballCarrierPlayerId")}.get(pev) or ge.get("playerId")
        target = pe.get("receiverPlayerId") or pe.get("targetPlayerId") if pev in ("PA", "CR") else None
        challenger = pe.get("challengerPlayerId") if pev == "CH" else None
        culprit = fo.get("finalCulpritPlayerId") or fo.get("onFieldCulpritPlayerID") or fo.get("onFieldCulpritPlayerId")
        pressure = pe.get("pressureType")
        rows.append({
            "frame": int(frame), "period": period, "game_event_id": gev_id, "possession_event_id": pev_id,
            "gev": ge.get("gameEventType"), "pev": pev, "team": team, "setpiece": ge.get("setpieceType"), "x": x, "y": y,
            "player": ref(actor), "target": ref(target),
            "pass_outcome": pe.get("passOutcomeType") if pev == "PA" else pe.get("crossOutcomeType") if pev == "CR" else None,
            "shot_outcome": pe.get("shotOutcomeType") if pev == "SH" else None,
            "challenge_type": pe.get("challengeType") if pev == "CH" else None,
            "challenger": ref(challenger), "challenger_team": team_of(challenger),
            "winner_team": team_of(pe.get("challengeWinnerPlayerId")) if pev == "CH" else None,
            "out_type": ge.get("outType") if ge.get("gameEventType") == "OUT" else None,
            "out_team": team if ge.get("gameEventType") == "OUT" else None,
            "foul_type": fo.get("foulType"), "culprit_team": team_of(culprit),
            "card": fo.get("finalFoulOutcomeType") or fo.get("onFieldFoulOutcomeType"),
            "pressure": pressure, "initial_pressure": (e.get("initialTouch") or {}).get("initialPressureType"),
            "pressure_team": team_of(pe.get("pressurePlayerId")) or team_of((e.get("initialTouch") or {}).get("initialPressurePlayerId")),
            "lines_broken": pe.get("linesBrokenType"), "sequence": e.get("sequence"), "duration": e.get("duration"),
        })
    df = pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
    return df.sort_values(["frame", "possession_event_id"], kind="stable", na_position="first").reset_index(drop=True)


def _shirt(ref) -> str | None:
    return ref.split("_", 1)[1] if isinstance(ref, str) and "_" in ref else None


def to_p1_events(ev: pd.DataFrame, fps: float) -> pd.DataFrame:
    """PFF's events in the vocabulary P1 reads (Metrica's), team as ``Home`` / ``Away``, players as shirt numbers."""
    out: list[dict] = []
    other = {"home": "away", "away": "home"}
    text = [c for c in ev.columns if not pd.api.types.is_numeric_dtype(ev[c])]
    ev = ev.assign(**{c: ev[c].astype(object).where(ev[c].notna(), None) for c in text})  # missing text is None, never NaN

    def add(kind, sub, team, r, **kw):
        out.append({"Type": kind, "Subtype": sub, "Team": team.capitalize() if team else None, "Period": r.period,
                    "Start Frame": int(r.frame), "End Frame": int(r.frame), "Start X": r.x, "Start Y": r.y,
                    "From": kw.get("frm"), "To": kw.get("to"), "_ends": kw.get("ends", False), "_seq": len(out)})

    prev_team, prev_pass_outcome, dead, seen_gev = None, None, True, set()
    goals_by_shot: list[tuple[int, str]] = []
    pending_out = None  # the last ball-out-of-play event, until the restart says who put it out
    for r in ev.itertuples():
        if r.gev in KICKOFFS or r.gev in ("END", "OUT"):
            dead = True
        if r.gev == "OUT":
            pending_out = None
            if r.out_type in ("H", "A"):
                scorer = "home" if r.out_type == "H" else "away"
                if not any(team == scorer and 0 <= r.frame - f <= 15 * fps for f, team in goals_by_shot):
                    add("SHOT", "OWN GOAL-GOAL", scorer, r)  # no logged goal shot by the scorers: an own goal
            elif r.out_type == "T":
                pending_out = r
            prev_team = None
        if (r.gev == "OTB" or r.gev in KICKOFFS) and r.team:
            first_of_game_event = r.game_event_id not in seen_gev
            seen_gev.add(r.game_event_id)
            if first_of_game_event and r.setpiece in SETPIECE:
                if pending_out is not None and r.setpiece in OUT_RESTARTS:
                    add("BALL OUT", None, other[r.team], pending_out)  # PFF's out event has no team: the restart's opponents put it out
                add("SET PIECE", SETPIECE[r.setpiece], r.team, r)
            elif first_of_game_event and r.setpiece in ("O", None) and prev_team == other[r.team] and not dead:
                add("RECOVERY", "INTERCEPTION" if prev_pass_outcome in ("D", "B") else None, r.team, r)
            if first_of_game_event:
                pending_out = None
            dead = False
            pev = r.pev
            if pev in ("PA", "CR"):
                done = r.pass_outcome == "C"
                add("PASS" if done else "BALL LOST", "CROSS" if pev == "CR" else None, r.team, r, frm=_shirt(r.player),
                    to=_shirt(r.target), ends=True)
                prev_pass_outcome = r.pass_outcome
            elif pev == "SH":
                sub = SHOT_RESULT.get(r.shot_outcome, "OTHER")
                add("SHOT", sub, r.team, r, frm=_shirt(r.player))
                if r.shot_outcome == "G":
                    goals_by_shot.append((int(r.frame), r.team))
            elif pev == "CH":
                won = r.winner_team
                if r.challenge_type in DUELS:
                    for t in ("home", "away"):
                        res = "WON" if won == t else "LOST" if won else None
                        add("CHALLENGE", f"{DUELS[r.challenge_type]}-{res}" if res else DUELS[r.challenge_type], t, r)
                elif r.challenger_team:
                    res = "WON" if won == r.challenger_team else "LOST" if won else None
                    add("CHALLENGE", f"TACKLE-{res}" if res else "TACKLE", r.challenger_team, r, frm=_shirt(r.challenger))
            elif pev == "CL":
                add("CLEARANCE", None, r.team, r, frm=_shirt(r.player))
            elif pev == "BC":
                add("CARRY", None, r.team, r, frm=_shirt(r.player), ends=True)
            if pev not in ("PA", "CR"):
                prev_pass_outcome = None if pev not in ("IT", None) else prev_pass_outcome
            prev_team = r.team
        if r.foul_type == "I" and r.culprit_team:
            add("FAULT RECEIVED", None, other[r.culprit_team], r)
            if r.card in CARD:
                add("CARD", CARD[r.card], r.culprit_team, r)
    cols = ["Type", "Subtype", "Team", "Period", "Start Frame", "End Frame", "Start X", "Start Y", "End X", "End Y", "From", "To"]
    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["End X"], df["End Y"] = np.nan, np.nan
    # A pass or carry ends where the next PFF event (a later frame, the same period, within END_WITHIN_S) starts.
    ef, ep = ev["frame"].to_numpy(), ev["period"].to_numpy()
    ex, ey = ev["x"].to_numpy(float), ev["y"].to_numpy(float)
    for i in np.flatnonzero(df["_ends"].to_numpy()):
        f, p = int(df.at[i, "Start Frame"]), df.at[i, "Period"]
        k = int(np.searchsorted(ef, f, side="right"))
        if k < len(ef) and ep[k] == p and ef[k] - f <= END_WITHIN_S * fps:
            df.at[i, "End Frame"], df.at[i, "End X"], df.at[i, "End Y"] = int(ef[k]), ex[k], ey[k]
    return df.sort_values(["Start Frame", "_seq"], kind="stable")[cols].reset_index(drop=True)


def _slots(tracking: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    out = {}
    for team in ("home", "away"):
        ids = tracking.loc[tracking["team"] == team, "player_id"].unique()
        out[team] = tuple(sorted((i.split("_", 1)[1] for i in ids), key=lambda j: (len(j), j)))
    return out


def _with_times(events: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    clock = frames.set_index("frame")["time_s"]
    return events.assign(**{"Start Time [s]": events["Start Frame"].map(clock), "End Time [s]": events["End Frame"].map(clock)})


def subsample(m: NormalisedMatch, ref: Reference, step: int) -> tuple[NormalisedMatch, Reference]:
    """
    Every ``step``-th video frame, renumbered (frame // step) so frame gaps still convert to seconds at the new rate
    (29.97 / 3 = 9.99 Hz, the rate of SkillCorner's broadcast tracking). Events move to the nearest kept frame.
    """
    if step <= 1:
        return m, ref
    keep = m.frames.loc[m.frames["frame"] % step == 0]
    kept = set(keep["frame"].to_numpy().tolist())
    fps = m.fps / step
    tr = m.tracking.loc[m.tracking["frame"].isin(kept)].copy()
    tr["frame"] //= step
    tr["sample_rate_hz"] = fps
    bl = m.ball.loc[m.ball["frame"].isin(kept)].copy()
    bl["frame"] //= step
    ev = ref.events.assign(frame=(ref.events["frame"] / step).round().astype(int))
    m2 = dataclasses.replace(m, fps=fps, pitch=dataclasses.replace(m.pitch, fps=fps),
                             frames=keep.assign(frame=keep["frame"] // step).reset_index(drop=True),
                             tracking=tr.reset_index(drop=True), ball=bl.reset_index(drop=True))
    return m2, dataclasses.replace(ref, events=ev)


def period_consistency(m: NormalisedMatch, events: pd.DataFrame) -> dict[int, tuple[float, int]]:
    """Per period: the share of passes, crosses and shots where the tracked player nearest the tracked ball is on the
    acting team, and how many such events had a tracked ball and players on their frame."""
    ev = events.loc[events["pev"].isin(["PA", "CR", "SH"]) & events["team"].notna(), ["frame", "period", "team"]]
    b = m.ball.drop_duplicates("frame").set_index("frame")
    ev = ev.loc[ev["frame"].isin(b.index)]
    tr = m.tracking.loc[m.tracking["frame"].isin(ev["frame"]), ["frame", "team", "x", "y"]]
    tr = tr.assign(d=np.hypot(tr["x"].to_numpy() - b.loc[tr["frame"], "x"].to_numpy(), tr["y"].to_numpy() - b.loc[tr["frame"], "y"].to_numpy()))
    nearest = tr.sort_values("d").drop_duplicates("frame").set_index("frame")["team"]
    ev = ev.assign(near=ev["frame"].map(nearest)).dropna(subset=["near"])
    return {int(p): (float((g["near"] == g["team"]).mean()), int(len(g))) for p, g in ev.groupby("period")}


def _keep_consistent(m: NormalisedMatch, ref: Reference) -> tuple[NormalisedMatch, Reference]:
    """Remove periods whose player tracking disagrees with PFF's own ball and events (see ``CONSISTENT_MIN_SHARE``)."""
    cons = period_consistency(m, ref.events)
    drop = tuple(sorted(p for p, (share, n) in cons.items() if n >= CONSISTENT_MIN_EVENTS and share < CONSISTENT_MIN_SHARE))
    ref = dataclasses.replace(ref, consistency={p: (round(s, 3), n) for p, (s, n) in cons.items()}, dropped_periods=drop)
    if not drop:
        return m, ref
    keep = m.frames.loc[~m.frames["period"].isin(drop)]
    kept = set(keep["frame"].to_numpy().tolist())
    m = dataclasses.replace(m, frames=keep.reset_index(drop=True),
                            tracking=m.tracking.loc[m.tracking["frame"].isin(kept)].reset_index(drop=True),
                            ball=m.ball.loc[m.ball["frame"].isin(kept)].reset_index(drop=True))
    return m, dataclasses.replace(ref, events=ref.events.loc[~ref.events["period"].isin(drop)].reset_index(drop=True))


def _raw_event_count(path: Path) -> int:
    return len(json.loads(Path(path).read_text(encoding="utf-8")))


def load_pff_match(data_dir: Path, gid: str, cache: Path, ball: str = "tracked", step: int = 1) -> tuple[NormalisedMatch, Reference]:
    """
    One PFF match as a ``NormalisedMatch`` and PFF's own event table. Cached as a pickle at the native rate (outside the
    repository); ``step`` > 1 keeps every ``step``-th frame (see ``subsample``).
    """
    data_dir = Path(data_dir)
    cache.mkdir(parents=True, exist_ok=True)
    pk = cache / f"pff_{gid}_{ball}.{cache_tag(PROVIDER, READER_VERSION)}.pkl"
    md = read_metadata(data_dir, gid)
    if pk.exists():
        with open(pk, "rb") as fh:
            m, ref = pickle.load(fh)
        # The P1 event table is rebuilt from PFF's events, so a mapping change needs no re-parse.
        ref = dataclasses.replace(ref, events_without_frames=_raw_event_count(data_dir / "Event Data" / f"{gid}.json") - len(ref.events))
        m, ref = _keep_consistent(m, ref)
        m, ref = subsample(m, ref, step)
        events = _with_times(to_p1_events(ref.events, m.fps), m.frames)
        return dataclasses.replace(m, events=events, teams=md["teams"], competition=md["competition"], season=md["season"]), ref
    roster = read_roster(data_dir, gid, md["teams"])
    tr = read_tracking(data_dir / "Tracking Data" / f"{gid}.jsonl.bz2", ball=ball)
    frames = tr["frames"]
    frames = frames.assign(time_s=frames["elapsed"] + frames["period"].map(PERIOD_OFFSET_S).fillna(0.0))[["frame", "period", "time_s"]]
    signs = _signs(tr["players"], frames, roster)
    period_of = pd.Series(frames["period"].to_numpy(), index=frames["frame"].to_numpy())
    pl, bl = tr["players"], tr["ball"]
    s_pl = pl["frame"].map(period_of).map(signs).fillna(1).to_numpy()
    s_bl = bl["frame"].map(period_of).map(signs).fillna(1).to_numpy() if len(bl) else np.ones(0)
    clock = pd.Series(frames["time_s"].to_numpy(), index=frames["frame"].to_numpy())
    tracking = pd.DataFrame({"timestamp": pl["frame"].map(clock).to_numpy(), "frame": pl["frame"].to_numpy(),
                             "player_id": (pl["team"] + "_" + pl["shirt"]).to_numpy(), "team": pl["team"].to_numpy(),
                             "x": pl["x"].to_numpy() * s_pl, "y": pl["y"].to_numpy() * s_pl, "source": PROVIDER,
                             "sample_rate_hz": md["fps"], "observed": pl["visible"].to_numpy(dtype=bool)})
    tracking = validate_schema(coerce_schema(tracking), name=PROVIDER)
    ball_df = pd.DataFrame({"frame": bl["frame"].to_numpy(), "x": bl["x"].to_numpy() * s_bl, "y": bl["y"].to_numpy() * s_bl}) \
        if len(bl) else pd.DataFrame({"frame": pd.Series(dtype="int64"), "x": pd.Series(dtype=float), "y": pd.Series(dtype=float)})
    # The video clock for events PFF does not link to a frame (video milliseconds per frame, from the frame rate).
    vf = frames["frame"].to_numpy()
    video_clock = (vf / md["fps"] * 1000.0, vf)
    ev = read_events(data_dir / "Event Data" / f"{gid}.json", roster, tr["pev_frame"], tr["gev_frame"], video_clock, signs)
    m = NormalisedMatch(provider=PROVIDER, match_id=str(gid), fps=md["fps"],
                        pitch=PitchConfig(length_m=md["length"], width_m=md["width"], fps=md["fps"]),
                        frames=frames.reset_index(drop=True), tracking=tracking, ball=ball_df,
                        events=pd.DataFrame(), teams=md["teams"], competition=md["competition"], season=md["season"])
    m = dataclasses.replace(m, slots=_slots(m.tracking))
    ref = Reference(events=ev, visible_share=float(pl["visible"].mean()) if len(pl) else 0.0,
                    ball_share=len(bl) / max(len(frames), 1), orientation_agrees=_orientation_agrees(signs, md),
                    events_without_frames=_raw_event_count(data_dir / "Event Data" / f"{gid}.json") - len(ev))
    with open(pk, "wb") as fh:
        pickle.dump((m, ref), fh)
    m, ref = _keep_consistent(m, ref)
    m, ref = subsample(m, ref, step)
    return dataclasses.replace(m, events=_with_times(to_p1_events(ref.events, m.fps), m.frames)), ref
