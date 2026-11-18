"""
idsse.py — read the IDSSE dataset (official German league data) as Ponta reference data.

Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of spatiotemporal and event data in
elite soccer*, Scientific Data 12, 195 (https://doi.org/10.1038/s41597-025-04505-y). Data: Deutsche
Fußball Liga (DFL). CC BY 4.0. Seven matches of the 2022/23 Bundesliga and 2. Bundesliga, each with
official match information, events and TRACAB positions (25 Hz). Download with
``tools/fetch_reference_data.py idsse``; the files stay outside every repository.

This is a **reference reader for measurement**, not a supported provider (Ponta's providers are
Metrica, SkillCorner, StatsBomb and STATSports). It builds a ``NormalisedMatch`` so the context
packages run unchanged, plus the official references P1 to P4 can be compared with:

- per frame: the official ball status (in play or dead) and the team in possession;
- per event: the official possession phase id (``BallPossessionPhase``) and every restart with
  the referee's decision time (the ball went dead) and the time it was taken;
- per player: the official playing position and each team's line-up.

**No names are read.** Players are ``home_<shirt>`` / ``away_<shirt>``; persons are looked up by
the league's id only, and club names are never read. Coordinates: centred metres, the home team
attacking +x in both halves (each half is rotated when the home goalkeeper defends +x), events in
the same frame. Events are translated into the vocabulary P1 reads (``PASS``, ``SHOT``,
``SET PIECE``, ``RECOVERY``, ``BALL LOST``, ``BALL OUT``, ``CARD``). Parsed matches are cached as
pickles in ``cache`` (outside the repository).
"""

from __future__ import annotations

import dataclasses
import pickle
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.match import NormalisedMatch
from matrix_engine.pitch_config import PitchConfig
from matrix_engine.schema import coerce_schema, validate_schema

FPS = 25.0
PROVIDER = "idsse"
PERIODS = {"firstHalf": 1, "secondHalf": 2}
MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")

_FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)" T="([^"]+)" X="([-\d.]+)" Y="([-\d.]+)"')
_BALL = re.compile(r'Z="([-\d.]+)".*BallPossession="(\d)" BallStatus="(\d)"')
_PITCH = re.compile(r'<PitchSize X="([\d.]+)" Y="([\d.]+)"')

#: Restart events and the subtype P1 reads.
RESTARTS = {"KickOff": "KICK OFF", "ThrowIn": "THROW IN", "GoalKick": "GOAL KICK", "CornerKick": "CORNER KICK",
            "FreeKick": "FREE KICK", "Penalty": "PENALTY"}
#: Shot outcomes (the child element of ``ShotAtGoal``) as P1's shot subtypes.
SHOT_RESULT = {"SuccessfulShot": "ON TARGET-GOAL", "SavedShot": "ON TARGET-SAVED", "ShotWide": "OFF TARGET-OUT",
               "BlockedShot": "BLOCKED", "ShotWoodWork": "OFF TARGET-WOODWORK", "OtherShot": "OTHER"}
CARD = {"yellow": "YELLOW", "red": "RED", "yellowRed": "SECOND YELLOW"}


@dataclass
class Official:
    """The league's own references for one match (comparison data only)."""

    ball: pd.DataFrame                 # frame, period, ball_status (1 in play, 0 dead), ball_team (home/away/None)
    events: pd.DataFrame               # every parsed event: frame, period, kind, team, phase_id, decision_frame, details
    roles: pd.DataFrame                # player_id, team, position (league code), starting
    lineups: dict[str, str] = field(default_factory=dict)
    result: str = ""


def _files(data_dir: Path, match_id: str) -> dict[str, Path]:
    out = {}
    for kind, tag in (("info", "matchinformation"), ("events", "events_raw"), ("positions", "positions_raw")):
        hits = sorted(data_dir.glob(f"DFL_*_{tag}_*_DFL-MAT-{match_id}.xml"))
        if not hits:
            raise FileNotFoundError(f"{tag} file for {match_id} not found in {data_dir}")
        out[kind] = hits[0]
    return out


def read_info(path: Path) -> dict:
    """Teams, players (league id to team, shirt and position) and line-ups. Names are not read."""
    root = ET.parse(path).getroot()
    g = root.find(".//General")
    side = {g.get("HomeTeamId"): "home", g.get("GuestTeamId"): "away"}
    persons, lineups = {}, {}
    for t in root.iter("Team"):
        team = side[t.get("TeamId")]
        lineups[team] = t.get("LineUp") or ""
        for p in t.iter("Player"):
            persons[p.get("PersonId")] = {"team": team, "player_id": f"{team}_{p.get('ShirtNumber')}",
                                          "position": p.get("PlayingPosition"), "starting": p.get("Starting") == "true"}
    return {"team_of": side, "persons": persons, "lineups": lineups, "result": g.get("Result") or ""}


def read_positions(path: Path, persons: dict, team_of: dict) -> dict:
    """Players' and the ball's positions per frame (league coordinates: centred metres)."""
    rows: dict[str, list] = {"frame": [], "period": [], "player_id": [], "team": [], "x": [], "y": []}
    ball: dict[str, list] = {"frame": [], "period": [], "t": [], "x": [], "y": [], "z": [], "poss": [], "status": []}
    pitch = (105.0, 68.0)
    current = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "<Frame " in line:
                if current is None:
                    continue
                m = _FRAME.search(line)
                if not m:
                    continue
                kind, period, who = current
                if kind == "ball":
                    b = _BALL.search(line)
                    ball["frame"].append(int(m.group(1)))
                    ball["period"].append(period)
                    ball["t"].append(m.group(2))
                    ball["x"].append(float(m.group(3)))
                    ball["y"].append(float(m.group(4)))
                    ball["z"].append(float(b.group(1)) if b else np.nan)
                    ball["poss"].append(int(b.group(2)) if b else 0)
                    ball["status"].append(int(b.group(3)) if b else 0)
                else:
                    rows["frame"].append(int(m.group(1)))
                    rows["period"].append(period)
                    rows["player_id"].append(who[0])
                    rows["team"].append(who[1])
                    rows["x"].append(float(m.group(3)))
                    rows["y"].append(float(m.group(4)))
            elif "<FrameSet" in line:
                fs = _FRAMESET.search(line)
                if not fs:
                    current = None
                    continue
                section, team_id, person = fs.groups()
                period = PERIODS.get(section)
                if team_id == "BALL":
                    current = ("ball", period, None)
                elif person in persons and team_id in team_of:
                    p = persons[person]
                    current = ("player", period, (p["player_id"], p["team"]))
                else:
                    current = None  # referees
            elif "<PitchSize" in line:
                m = _PITCH.search(line)
                if m:
                    pitch = (float(m.group(1)), float(m.group(2)))
    tracking = pd.DataFrame(rows)
    b = pd.DataFrame(ball)
    b["t"] = pd.to_datetime(b["t"], utc=True)
    return {"tracking": tracking, "ball": b, "pitch": pitch}


def _flip_signs(tracking: pd.DataFrame, persons: dict) -> dict[int, int]:
    """Per period, -1 when the home goalkeeper defends +x (rotate that half), else +1."""
    gk = {p["player_id"] for p in persons.values() if p["team"] == "home" and p["position"] == "TW"}
    out = {}
    for period, g in tracking.loc[tracking["player_id"].isin(gk)].groupby("period"):
        # The keeper who played most of the half.
        main = g.groupby("player_id")["x"].agg(["size", "mean"]).sort_values("size").iloc[-1]
        out[int(period)] = -1 if main["mean"] > 0 else 1
    return out


def _event_frame(when: str | None, clock: tuple[np.ndarray, np.ndarray]) -> int | None:
    """The frame nearest an event time (any UTC offset); None when no tracked frame is within 2 s."""
    if not when:
        return None
    t_ns, frames = clock
    t = int(datetime.fromisoformat(when).timestamp() * 1e9)
    k = int(np.clip(np.searchsorted(t_ns, t), 0, len(t_ns) - 1))
    lo = max(k - 1, 0)
    k = lo if abs(t_ns[lo] - t) <= abs(t_ns[k] - t) else k
    if abs(t_ns[k] - t) > 2e9:
        return None  # outside the tracked halves
    return int(frames[k])


def read_events(path: Path, info: dict, clock: tuple[np.ndarray, np.ndarray], period_of: dict[int, int], length: float, width: float,
                signs: dict[int, int]) -> pd.DataFrame:
    """Every event the reader uses, with the league's ids for team and possession phase. No names."""
    root = ET.parse(path).getroot()
    team_of, persons = info["team_of"], info["persons"]
    rows = []
    for e in root.iter("Event"):
        kids = list(e)
        if not kids:
            continue
        top = kids[0]
        kind = top.tag
        if kind in ("Delete",):
            continue
        frame = _event_frame(e.get("EventTime"), clock)
        if frame is None:
            continue
        period = period_of.get(frame)
        sign = signs.get(period, 1)
        x = e.get("X-Position")
        y = e.get("Y-Position")
        # Event coordinates run from a corner (0 to length, 0 to width); rotate like the tracking.
        ex = sign * (float(x) - length / 2) if x is not None else np.nan
        ey = sign * (float(y) - width / 2) if y is not None else np.nan
        play = top.find("Play") if kind in RESTARTS else (top if kind == "Play" else None)
        team_id = top.get("Team") or top.get("WinnerTeam") or (play.get("Team") if play is not None else None)
        row = {"frame": frame, "period": period, "kind": kind, "team": team_of.get(team_id), "x": ex, "y": ey,
               "phase_id": (play if play is not None else top).get("BallPossessionPhase"),
               "decision_frame": _event_frame(top.get("DecisionTimestamp"), clock) if kind in RESTARTS else None,
               "evaluation": play.get("Evaluation") if play is not None else None,
               "result": next((c.tag for c in top if c.tag in SHOT_RESULT), None) if kind == "ShotAtGoal" else None,
               "card": top.get("CardColor") if kind == "Caution" else None,
               "possession_change": top.get("PossessionChange") if kind == "TacklingGame" else None,
               "counter_attack": top.get("CounterAttack") if kind == "ShotAtGoal" else None,
               "player": persons.get((play if play is not None else top).get("Player"), {}).get("player_id")}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("frame", kind="stable").reset_index(drop=True)


def to_p1_events(ev: pd.DataFrame) -> pd.DataFrame:
    """The league's events in the vocabulary P1 reads (Metrica's), team as ``Home`` / ``Away``."""
    out = []

    def add(kind, sub, team, start, end=None, x=np.nan, y=np.nan):
        out.append({"Type": kind, "Subtype": sub, "Team": team.capitalize() if team else None, "Period": period,
                    "Start Frame": int(start), "End Frame": int(end if end is not None else start), "Start X": x, "Start Y": y})

    for r in ev.itertuples():
        period = int(r.period)
        other = {"home": "away", "away": "home"}.get(r.team)
        if r.kind in RESTARTS:
            if r.decision_frame is not None and r.decision_frame < r.frame and r.kind in ("ThrowIn", "GoalKick", "CornerKick"):
                add("BALL OUT", None, other, r.decision_frame, x=r.x, y=r.y)
            elif r.decision_frame is not None and r.decision_frame < r.frame and r.kind in ("FreeKick", "Penalty"):
                add("FAULT RECEIVED", None, r.team, r.decision_frame, x=r.x, y=r.y)
            add("SET PIECE", RESTARTS[r.kind], r.team, r.frame, x=r.x, y=r.y)
            add("PASS" if r.evaluation == "successfullyCompleted" else "BALL LOST", None, r.team, r.frame, x=r.x, y=r.y)
        elif r.kind == "Play":
            add("PASS" if r.evaluation == "successfullyCompleted" else "BALL LOST", None, r.team, r.frame, x=r.x, y=r.y)
        elif r.kind == "ShotAtGoal":
            add("SHOT", SHOT_RESULT.get(r.result, "OTHER"), r.team, r.frame, x=r.x, y=r.y)
        elif r.kind == "TacklingGame":
            add("CHALLENGE", None, r.team, r.frame, x=r.x, y=r.y)
            if r.possession_change == "true":
                add("RECOVERY", None, r.team, r.frame, x=r.x, y=r.y)
        elif r.kind == "BallClaiming":
            add("RECOVERY", None, r.team, r.frame, x=r.x, y=r.y)
        elif r.kind == "Caution" and r.card in CARD:
            add("CARD", CARD[r.card], r.team, r.frame)
    cols = ["Type", "Subtype", "Team", "Period", "Start Frame", "End Frame", "Start X", "Start Y"]
    df = pd.DataFrame(out, columns=cols).sort_values("Start Frame", kind="stable").reset_index(drop=True)
    # A pass ends where the next event starts (the league logs no reception time).
    nxt = df["Start Frame"].shift(-1)
    passes = df["Type"].isin(["PASS", "BALL LOST"])
    df.loc[passes, "End Frame"] = nxt[passes].fillna(df.loc[passes, "Start Frame"]).astype(int)
    return df


def _slots(tracking: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Per team, the shirt numbers as Plaiq's wide frames expect them (``home_7`` gives ``7``)."""
    out = {}
    for team in ("home", "away"):
        ids = tracking.loc[tracking["team"] == team, "player_id"].unique()
        out[team] = tuple(sorted((i.split("_", 1)[1] for i in ids), key=lambda j: (len(j), j)))
    return out


def load_idsse_match(data_dir: Path, match_id: str, cache: Path) -> tuple[NormalisedMatch, Official]:
    """One IDSSE match as a ``NormalisedMatch`` and the league's references. Cached as a pickle."""
    cache.mkdir(parents=True, exist_ok=True)
    pk = cache / f"idsse_{match_id}.pkl"
    if pk.exists():
        with open(pk, "rb") as fh:
            m, official = pickle.load(fh)
        # The P1 event table is rebuilt from the league's events, so a mapping change needs no re-parse.
        return dataclasses.replace(m, events=to_p1_events(official.events), slots=_slots(m.tracking)), official
    f = _files(Path(data_dir), match_id)
    info = read_info(f["info"])
    pos = read_positions(f["positions"], info["persons"], info["team_of"])
    length, width = pos["pitch"]
    tr, ball = pos["tracking"], pos["ball"]
    signs = _flip_signs(tr, info["persons"])
    s_tr = tr["period"].map(signs).fillna(1).to_numpy()
    tr = tr.assign(x=tr["x"] * s_tr, y=tr["y"] * s_tr)
    s_b = ball["period"].map(signs).fillna(1).to_numpy()
    ball = ball.assign(x=ball["x"] * s_b, y=ball["y"] * s_b)

    clock = ball[["frame", "period", "t"]].sort_values("frame").reset_index(drop=True)
    frames = clock[["frame", "period"]].copy()
    first = frames.groupby("period")["frame"].transform("min")
    offset = frames["period"].map({1: 0.0, 2: 45 * 60.0}).fillna(0.0)
    frames["time_s"] = (frames["frame"] - first) / FPS + offset
    period_of = dict(zip(frames["frame"], frames["period"], strict=True))

    t_ns = clock["t"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype("int64")
    ev = read_events(f["events"], info, (t_ns, clock["frame"].to_numpy()), period_of, length, width, signs)
    tracking = pd.DataFrame({"timestamp": tr["frame"].map(dict(zip(frames["frame"], frames["time_s"], strict=True))),
                             "frame": tr["frame"], "player_id": tr["player_id"], "team": tr["team"], "x": tr["x"], "y": tr["y"],
                             "source": PROVIDER, "sample_rate_hz": FPS})
    tracking = validate_schema(coerce_schema(tracking), name=PROVIDER)
    seen = ball["status"].notna()
    m = NormalisedMatch(provider=PROVIDER, match_id=match_id, fps=FPS, pitch=PitchConfig(length_m=length, width_m=width, fps=FPS),
                        frames=frames.reset_index(drop=True), tracking=tracking,
                        ball=ball.loc[seen, ["frame", "x", "y"]].reset_index(drop=True), events=to_p1_events(ev))
    team_of_poss = {1: "home", 2: "away"}
    official = Official(
        ball=pd.DataFrame({"frame": ball["frame"], "period": ball["period"], "ball_status": ball["status"],
                           "ball_team": ball["poss"].map(team_of_poss)}),
        events=ev,
        roles=pd.DataFrame([{"player_id": p["player_id"], "team": p["team"], "position": p["position"], "starting": p["starting"]}
                            for p in info["persons"].values()]),
        lineups=info["lineups"], result=info["result"])
    with open(pk, "wb") as fh:
        pickle.dump((m, official), fh)
    return m, official
