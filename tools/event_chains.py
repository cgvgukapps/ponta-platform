"""
event_chains.py — possessions from an event feed alone, after Pappalardo et al. (2019).

Re-implemented from the description of the "play action" in Pappalardo, Cintia, Rossi et al.
(2019), *A public data set of spatio-temporal match events in soccer competitions*, Scientific Data
6, 236 (https://doi.org/10.1038/s41597-019-0247-7), and its replication notebook (CC BY 4.0). Their
code is not copied. The rule:

- a possession begins when a team gains the ball (a restart, a regain, a kick-off);
- it ends at an interruption (ball out, whistle, foul, offside, end of a period), at a shot (a save
  that follows belongs to it), or when the other team acts on the ball;
- duels come in pairs, one per team, and never change possession on their own.

Events are given in a common form (``EVENT_COLUMNS``); adapters translate Metrica's, the German
league's (IDSSE) and Wyscout's feeds. Depth is metres from the acting team's own goal line towards
the opponent's. This is a tool for measurement (how much an event feed alone says about possession
and its patterns); it is not part of the engines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: kind: pass, shot, save, duel, restart, interruption, foul, offside, other (an on-ball action).
EVENT_COLUMNS = ("period", "t", "team", "kind", "restart", "depth_m")
ENDING = {"interruption": "out", "foul": "foul", "offside": "offside"}


def chains(events: pd.DataFrame) -> pd.DataFrame:
    """Possession chains from events in the common form, in time order within each period."""
    rows = []
    cur = None

    def close(end_type: str, t_end: float) -> None:
        nonlocal cur
        if cur is not None:
            cur.update(end_type=end_type, end_t=t_end)
            rows.append(cur)
        cur = None

    for period, ev in events.sort_values(["period", "t"], kind="stable").groupby("period", sort=True):
        cur, started = None, False
        for r in ev.itertuples(index=False):
            kind, team = r.kind, r.team
            if kind in ENDING:
                close(ENDING[kind], r.t)
                continue
            if kind == "duel" or team is None or (isinstance(team, float) and np.isnan(team)):
                continue
            if kind == "save" and cur is None and rows and rows[-1]["end_type"] == "shot":
                continue  # the keeper's save belongs to the shot
            if kind == "restart":
                close("out", r.t)  # a restart with no logged interruption before it
                cur = {"period": int(period), "team": team, "start_t": r.t, "start_type": f"restart:{r.restart}",
                       "start_depth": r.depth_m, "max_depth": r.depth_m, "events": 1, "end_type": None}
                started = True
                continue
            if cur is not None and team != cur["team"]:
                close("lost", r.t)
            if cur is None:
                cur = {"period": int(period), "team": team, "start_t": r.t, "start_type": "regain" if started else "period_start",
                       "start_depth": r.depth_m, "max_depth": r.depth_m, "events": 0, "end_type": None}
                started = True
            cur["events"] += 1
            if np.isfinite(r.depth_m):
                cur["max_depth"] = np.nanmax([cur["max_depth"], r.depth_m])
            if kind == "shot":
                close("shot", r.t)
        if cur is not None:
            close("period_end", float(ev["t"].max()))
    out = pd.DataFrame(rows, columns=["period", "team", "start_t", "end_t", "start_type", "end_type", "events", "start_depth", "max_depth"])
    out["duration_s"] = out["end_t"] - out["start_t"]
    return out


def on_grid(ch: pd.DataFrame, ts: np.ndarray, period: np.ndarray) -> np.ndarray:
    """The team in possession at each grid time from event chains; None between chains."""
    lab = np.full(len(ts), None, dtype=object)
    for c in ch.itertuples():
        m = (period == c.period) & (ts >= c.start_t) & (ts < c.end_t)
        lab[m] = c.team
    return lab


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def from_metrica(ev: pd.DataFrame, fps: float, f0: int, length: float, home_sign: dict[int, int]) -> pd.DataFrame:
    """Metrica-vocabulary events (also the IDSSE reader's P1 table). ``home_sign``: +1 when home attacks +x."""
    kind_of = {"PASS": "pass", "BALL LOST": "pass", "RECOVERY": "other", "CARRY": "other", "CHALLENGE": "duel",
               "SHOT": "shot", "SET PIECE": "restart", "BALL OUT": "interruption", "FAULT RECEIVED": "foul"}
    rows = []
    for r in ev.to_dict("records"):
        t = r.get("Type")
        if t not in kind_of:
            continue
        team = str(r["Team"]).lower() if isinstance(r.get("Team"), str) else None
        period = int(r["Period"])
        x = r.get("Start X", np.nan)
        s = home_sign.get(period, 1) * (1 if team == "home" else -1)
        depth = s * float(x) + length / 2 if x is not None and np.isfinite(x) else np.nan
        sub = str(r.get("Subtype")) if t == "SET PIECE" else ""
        rows.append({"period": period, "t": (int(r["Start Frame"]) - f0) / fps, "team": team, "kind": kind_of[t],
                     "restart": sub.lower().replace(" ", "_") or None, "depth_m": depth})
    return pd.DataFrame(rows, columns=list(EVENT_COLUMNS))


WYSCOUT_KIND = {"Pass": "pass", "Shot": "shot", "Save attempt": "save", "Duel": "duel", "Free Kick": "restart",
                "Interruption": "interruption", "Foul": "foul", "Offside": "offside", "Others on the ball": "other",
                "Goalkeeper leaving line": "other"}
WYSCOUT_RESTART = {"Corner": "corner_kick", "Free Kick": "free_kick", "Free kick cross": "free_kick", "Free kick shot": "free_kick",
                   "Goal kick": "goal_kick", "Penalty": "penalty", "Throw in": "throw_in"}
WYSCOUT_PERIOD = {"1H": 1, "2H": 2, "E1": 3, "E2": 4, "P": 5}


def from_wyscout(ev: pd.DataFrame, length: float = 105.0) -> pd.DataFrame:
    """One match of Wyscout's open events. Positions run 0 to 100 towards the acting team's goal."""
    x0 = ev["positions"].map(lambda p: p[0]["x"] if p else np.nan)
    return pd.DataFrame({
        "period": ev["matchPeriod"].map(WYSCOUT_PERIOD).fillna(5).astype(int),
        "t": ev["eventSec"].astype(float),
        "team": ev["teamId"].astype(str),
        "kind": ev["eventName"].map(WYSCOUT_KIND).fillna("other"),
        "restart": ev["subEventName"].map(WYSCOUT_RESTART),
        "depth_m": x0.astype(float) / 100.0 * length,
    }).loc[lambda d: d["period"] <= 4].reset_index(drop=True)
