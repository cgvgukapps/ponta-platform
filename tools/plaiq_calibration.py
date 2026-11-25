"""
plaiq_calibration.py — Plaiq's ball flight, player kinematics and reaction time calibrated on tracking.

Step 6 of ``ponta-blueprint/plaiq-review.md``. Targeted pitch control (``targeted_pcz``) rests on a ball
speed model (5 m/s plus 0.2 m/s per metre), a timing rule along the pass (each point timed as its own
pass) and one defender profile (reaction 0.7 s, top speed 9.0 m/s, acceleration 5.5 m/s²), all set by
hand. The German league's data (IDSSE, 7 matches, TRACAB at 25 Hz with the ball's height, and the
league's own pass events with the intended recipient) lets each be measured:

1. **Ball flight (physics only).** For every open-play pass the recipient touched first: the kick (the
   ball leaves the passer: within 1.5 m, then beyond 3 m within 0.6 s, nearest the logged time) and
   the first touch (any player but the passer within 1.0 m of the ball, the ball at most 2.5 m high).
   Average speed against length gives the model's two parameters; the ball's progress along each pass
   gives how it slows (``s / D = (1 + c)τ − cτ²``).
2. **Players (physics only).** Each player's top speed (the fastest half-second) and peak acceleration
   (the 99.9th percentile of the speed gained over 0.4 s) from the tracking.
3. **Reaction time (outcomes).** Reaction time is an effective parameter (perception, the first step,
   controlling the ball), so it is fitted: availability (``V_ctrl × (1 − interception_risk)``) for the
   pass to the recipient where they stand at the kick, against whether the pass was completed (AUC),
   over a grid of reaction times for each flight model.

Fit on the even matches, tested on the odd ones; nothing is adopted here. With ``--sc-data`` the
chosen candidates are also checked on SkillCorner's 20 open-data matches (a different league and
provider, broadcast tracking at 10 Hz), on the passes the harness used (the targeted player at the
pass). Reference data is comparison data only (founder decision 2); data, caches and output stay
outside the repository; no names are read.

    python tools/plaiq_calibration.py --idsse ~/ponta-reference-data/idsse --cache /tmp/ponta-context \
        --out /tmp/ponta-context/calibration.json [--sc-data ~/ponta-reference-data/skillcorner-opendata]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.sample_data import repo_root
from plaiq_engine.analytics.ball_flight import V1, FlightModel, targeted_pcz_flight
from plaiq_engine.analytics.pass_option_value import _build_defender_data
from plaiq_engine.match_analysis import MatchAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_tag import tag
from idsse import FPS, MATCH_IDS, load_idsse_match
from plaiq_references import auc

KICK_NEAR_M, KICK_AWAY_M, KICK_AWAY_FRAMES = 1.5, 3.0, 15
KICK_WINDOW = (-50, 37)          # frames around the logged time (-2 s to +1.5 s)
TOUCH_M, TOUCH_MAX_Z_M = 1.0, 2.5
TOUCH_FROM, TOUCH_TO = 3, 150    # frames after the kick (0.12 s to 6 s)
REACTIONS = tuple(round(x, 1) for x in np.arange(0.0, 1.21, 0.1))


def _outside(path: Path) -> bool:
    root = repo_root()
    return not (path.resolve() == root or root in path.resolve().parents)


# ---------------------------------------------------------------------------
# Extraction (IDSSE)
# ---------------------------------------------------------------------------

def _grid(m, official):
    """Ball and players as arrays over every frame from the first to the last."""
    f0, f1 = int(m.frames["frame"].min()), int(m.frames["frame"].max())
    full = pd.RangeIndex(f0, f1 + 1)
    b = m.ball.set_index("frame")[["x", "y"]].reindex(full)
    ob = official.ball.set_index("frame").reindex(full)
    pos = m.tracking.pivot_table(index="frame", columns="player_id", values=["x", "y"]).reindex(full)
    ids = list(pos["x"].columns)
    return {"f0": f0, "bx": b["x"].to_numpy(), "by": b["y"].to_numpy(), "bz": ob["ball_z"].to_numpy(),
            "live": ob["ball_status"].to_numpy() == 1, "ids": ids, "px": pos["x"].to_numpy().T, "py": pos["y"].to_numpy().T}


def kick(g: dict, passer: str, frame: int) -> int | None:
    """The frame the ball leaves the passer, nearest the logged time; None when not seen."""
    if passer not in g["ids"]:
        return None
    i = g["ids"].index(passer)
    lo, hi = frame - g["f0"] + KICK_WINDOW[0], frame - g["f0"] + KICK_WINDOW[1]
    lo, hi = max(lo, 0), min(hi, len(g["bx"]) - KICK_AWAY_FRAMES - 1)
    if hi <= lo:
        return None
    w = np.arange(lo, hi + KICK_AWAY_FRAMES + 1)
    d = np.hypot(g["bx"][w] - g["px"][i][w], g["by"][w] - g["py"][i][w])
    out = []
    for k in range(hi - lo):
        if d[k] <= KICK_NEAR_M and d[k + 1] > KICK_NEAR_M and np.nanmax(d[k + 1:k + 1 + KICK_AWAY_FRAMES]) > KICK_AWAY_M:
            out.append(lo + k)
    if not out:
        return None
    return int(min(out, key=lambda x: abs(x - (frame - g["f0"]))))


def first_touch(g: dict, passer: str, k: int) -> tuple[int | None, str | None]:
    """(index, player id) of the first touch after the kick, (index, None) when the ball goes dead, else (None, None)."""
    w = np.arange(k + TOUCH_FROM, min(k + TOUCH_TO, len(g["bx"])))
    if not len(w):
        return None, None
    dist = np.hypot(g["px"][:, w] - g["bx"][w], g["py"][:, w] - g["by"][w])
    dist[g["ids"].index(passer)] = np.inf
    low = ~(g["bz"][w] > TOUCH_MAX_Z_M)
    touched = (dist <= TOUCH_M) & low
    hit = np.where(touched.any(axis=0))[0]
    dead = np.where(~g["live"][w])[0]
    t_hit = hit[0] if len(hit) else np.inf
    t_dead = dead[0] if len(dead) else np.inf
    if t_hit == np.inf and t_dead == np.inf:
        return None, None
    if t_dead < t_hit:
        return int(w[int(t_dead)]), None
    j = int(t_hit)
    return int(w[j]), g["ids"][int(np.nanargmin(np.where(touched[:, j], dist[:, j], np.inf)))]


def _progress(g: dict, k: int, e: int) -> tuple[float, float, float]:
    """The fraction of the pass's length covered after 25%, 50% and 75% of its time (projected on its line)."""
    start = np.array([g["bx"][k], g["by"][k]])
    v = np.array([g["bx"][e], g["by"][e]]) - start
    length = float(np.linalg.norm(v))
    out = []
    for tau in (0.25, 0.5, 0.75):
        x = k + tau * (e - k)
        a, b = int(np.floor(x)), int(np.ceil(x))
        p = np.array([np.interp(x, [a, b], [g["bx"][a], g["bx"][b]]), np.interp(x, [a, b], [g["by"][a], g["by"][b]])])
        out.append(float((p - start) @ v / length ** 2))
    return tuple(out)


def idsse_passes(data: Path, cache: Path, match_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every open-play pass with a recipient (kick, first touch, flight, the situation at the kick) and the players' kinematics."""
    m, official = load_idsse_match(data, match_id, cache / "idsse-cache")
    ma = MatchAnalysis(m)
    g = _grid(m, official)
    ev = official.events
    ev = ev.loc[(ev["kind"] == "Play") & ev["open_play"].astype(bool) & ev["recipient"].notna()
                & ev["evaluation"].isin(["successfullyCompleted", "unsuccessful"])]
    rows = []
    for r in ev.itertuples():
        k = kick(g, r.player, int(r.frame))
        if k is None or not g["live"][k] or not np.isfinite(g["bx"][k]):
            rows.append({"match": match_id, "completed": r.evaluation == "successfullyCompleted", "kick": False})
            continue
        e, who = first_touch(g, r.player, k)
        team = r.team
        prefix, oprefix = ("Home", "Away") if team == "home" else ("Away", "Home")
        own, opp = (ma.home, ma.away) if team == "home" else (ma.away, ma.home)
        ojer = ma.away_jerseys if team == "home" else ma.home_jerseys
        oprof = ma.profiles_away if team == "home" else ma.profiles_home
        mates = [j for j in (ma.home_jerseys if team == "home" else ma.away_jerseys) if j != r.player.split("_", 1)[1]]
        mprof = ma.profiles_home if team == "home" else ma.profiles_away
        f = g["f0"] + k
        shirt = r.recipient.split("_", 1)[1]
        col = f"{prefix}_{shirt}"
        rec = (own.at[f, f"{col}_x"], own.at[f, f"{col}_y"]) if f in own.index and f"{col}_x" in own.columns else (np.nan, np.nan)
        rec_end = (own.at[g["f0"] + e, f"{col}_x"], own.at[g["f0"] + e, f"{col}_y"]) \
            if e is not None and (g["f0"] + e) in own.index and f"{col}_x" in own.columns else (np.nan, np.nan)
        row = {"match": match_id, "completed": r.evaluation == "successfullyCompleted", "kick": True, "height": r.height,
               "kick_offset_s": (f - int(r.frame)) / FPS, "ball": (float(g["bx"][k]), float(g["by"][k])), "recipient_at_kick": rec,
               "recipient_at_end": rec_end, "defenders": _build_defender_data(opp.loc[f], oprefix, ojer, oprof) if f in opp.index else [],
               "attackers": _build_defender_data(own.loc[f], prefix, mates, mprof) if f in own.index else [],
               "touch_by": None if e is None else ("out" if who is None else "recipient" if who == r.recipient
                                                  else "team" if who.startswith(team) else "opponent")}
        if e is not None and who is not None:
            row["length_m"] = float(np.hypot(g["bx"][e] - g["bx"][k], g["by"][e] - g["by"][k]))
            row["time_s"] = (e - k) / FPS
            row["peak_z_m"] = float(np.nanmax(g["bz"][k:e + 1])) if np.isfinite(g["bz"][k:e + 1]).any() else np.nan
            if row["length_m"] >= 5 and e - k >= 4:
                row["progress"] = _progress(g, k, e)
        rows.append(row)
    kin = []
    for side, prefix, jer in ((ma.home, "Home", ma.home_jerseys), (ma.away, "Away", ma.away_jerseys)):
        for j in jer:
            v = side.get(f"{prefix}_{j}_speed")
            if v is None or v.notna().sum() < 15000:  # at least 10 minutes on the pitch
                continue
            s = v.to_numpy(float)
            acc = (s[10:] - s[:-10]) / 0.4
            standing = (s[:-25] < 1.5) & np.isfinite(s[25:])
            kin.append({"match": match_id, "player": f"{prefix}_{j}", "top_speed": float(v.rolling(12, min_periods=12).mean().max()),
                        "peak_accel": float(np.nanpercentile(acc[np.isfinite(acc)], 99.9)),
                        "accel_from_standing_1s": float(np.nanpercentile((s[25:] - s[:-25])[standing], 99.5)) if standing.any() else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(kin)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_speed(p: pd.DataFrame) -> dict:
    """Least squares on the passes' flight times for speed = a + b × d (the v1 form)."""
    from scipy.optimize import least_squares
    d, t = p["length_m"].to_numpy(), p["time_s"].to_numpy()
    res = least_squares(lambda x: d / (x[0] + x[1] * d) - t, x0=[5.0, 0.2], bounds=([0.5, 0.0], [30.0, 2.0]), loss="soft_l1", f_scale=0.2)
    return {"min_speed": round(float(res.x[0]), 3), "speed_grad": round(float(res.x[1]), 4)}


def fit_decel(p: pd.DataFrame) -> float:
    taus = np.array([0.25, 0.5, 0.75])
    prog = np.array(p["progress"].tolist())
    cs = np.linspace(0, 1, 101)
    err = [np.nanmean((prog - ((1 + c) * taus - c * taus ** 2)) ** 2) for c in cs]
    return round(float(cs[int(np.argmin(err))]), 2)


def flight_errors(p: pd.DataFrame, flight: FlightModel) -> dict:
    pred = flight.time_to(p["length_m"].to_numpy())
    err = pred - p["time_s"].to_numpy()
    return {"passes": int(len(p)), "median_abs_error_s": round(float(np.median(np.abs(err))), 3),
            "mean_error_s": round(float(np.mean(err)), 3), "rmse_s": round(float(np.sqrt(np.mean(err ** 2))), 3)}


def availability(rows: pd.DataFrame, flight: FlightModel, target: str = "recipient_at_kick") -> np.ndarray:
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows.itertuples()):
        rec = np.asarray(getattr(r, target), float)
        if not np.isfinite(rec).all() or not r.defenders:
            continue
        risk, ctrl = targeted_pcz_flight(np.asarray(r.ball, float), rec, r.defenders, flight)
        out[i] = ctrl * (1 - risk)
    return out


def calibration_table(avail: np.ndarray, completed: np.ndarray) -> list[dict]:
    ok = np.isfinite(avail)
    a, c = avail[ok], completed[ok]
    edges = (0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01)
    return [{"availability": f"{lo:g}-{min(hi, 1):g}", "passes": int(((a >= lo) & (a < hi)).sum()),
             "completed": round(float(c[(a >= lo) & (a < hi)].mean()), 3) if ((a >= lo) & (a < hi)).any() else None}
            for lo, hi in pairwise(edges)]


# ---------------------------------------------------------------------------
# SkillCorner transfer (the harness's passes)
# ---------------------------------------------------------------------------

def skillcorner_rows(match_dir: Path, cache: Path) -> pd.DataFrame:
    from context_reference_set import _load
    m = _load(match_dir, cache)
    ma = MatchAnalysis(m)
    meta = json.loads((match_dir / f"{match_dir.name}_match.json").read_text())
    home = meta["home_team"]["id"]
    number = {p["id"]: str(p["number"]) for p in meta["players"]}
    ref = pd.read_csv(match_dir / f"{match_dir.name}_dynamic_events.csv", low_memory=False,
                      usecols=["event_type", "end_type", "frame_end", "team_id", "pass_outcome", "player_targeted_id", "player_targeted_xpass_completion",
                               "high_pass"])
    ref = ref.loc[(ref["event_type"] == "player_possession") & (ref["end_type"] == "pass") & ref["pass_outcome"].isin(["successful", "unsuccessful"])]
    ev = m.events.loc[m.events["Type"].isin(["PASS", "BALL LOST"])]
    rows = []
    for r in ref.itertuples():
        team = "Home" if r.team_id == home else "Away"
        cand = ev.loc[(ev["Team"] == team) & ((ev["Start Frame"] - r.frame_end).abs() <= 2)]
        target = number.get(r.player_targeted_id)
        if cand.empty or target is None:
            continue
        e = cand.iloc[0]
        f = int(e["Start Frame"])
        own, opp = (ma.home, ma.away) if team == "Home" else (ma.away, ma.home)
        if f not in own.index or f"{team}_{target}_x" not in own.columns or not np.isfinite(e["Start X"]):
            continue
        rec = (own.at[f, f"{team}_{target}_x"], own.at[f, f"{team}_{target}_y"])
        oprefix = "Away" if team == "Home" else "Home"
        ojer, oprof = (ma.away_jerseys, ma.profiles_away) if team == "Home" else (ma.home_jerseys, ma.profiles_home)
        mates = [j for j in (ma.home_jerseys if team == "Home" else ma.away_jerseys) if str(j) != str(e["From"])]
        mprof = ma.profiles_home if team == "Home" else ma.profiles_away
        rows.append({"match": match_dir.name, "completed": r.pass_outcome == "successful", "sc_xpass": r.player_targeted_xpass_completion,
                     "ball": (float(e["Start X"]), float(e["Start Y"])), "recipient_at_kick": rec,
                     "defenders": _build_defender_data(opp.loc[f], oprefix, ojer, oprof),
                     "attackers": _build_defender_data(own.loc[f], team, mates, mprof), "high": str(r.high_pass) == "True"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--idsse", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sc-data", type=Path)
    args = ap.parse_args()
    for p in [args.idsse, args.cache, args.out] + ([args.sc_data] if args.sc_data else []):
        if not _outside(p):
            sys.exit(f"refusing to read or write sample data inside the repository: {p}")
    rows_pk, kin_pk = args.cache / f"calibration_passes.{tag()}.pkl", args.cache / f"calibration_kinematics.{tag()}.pkl"
    if rows_pk.exists() and kin_pk.exists():
        passes, kin = pd.read_pickle(rows_pk), pd.read_pickle(kin_pk)
    else:
        parts = [idsse_passes(args.idsse, args.cache, mid) for mid in MATCH_IDS]
        passes = pd.concat([p for p, _ in parts], ignore_index=True)
        kin = pd.concat([k for _, k in parts], ignore_index=True)
        passes.to_pickle(rows_pk)
        kin.to_pickle(kin_pk)
        print("extracted", len(passes), "passes", flush=True)
    half = {mid: ("fit" if i % 2 == 0 else "test") for i, mid in enumerate(MATCH_IDS)}
    passes["half"] = passes["match"].map(half)
    kin["half"] = kin["match"].map(half)
    seen = passes.loc[passes["kick"]]

    # 1. Ball flight (physics): passes the recipient touched first.
    flown = seen.loc[(seen["touch_by"] == "recipient") & (seen["time_s"] >= 0.2) & (seen["length_m"] >= 3)]
    speed = {h: fit_speed(flown.loc[(flown["half"] == "fit") & ((flown["height"] == h) if h != "all" else True)]) for h in ("all", "flat", "high")}
    decel = fit_decel(flown.loc[(flown["half"] == "fit") & flown["progress"].notna()])
    measured = FlightModel(**speed["all"])
    measured_pass = replace(measured, waypoints="pass", decel=decel)
    test_flown = flown.loc[flown["half"] == "test"]
    prog_test = np.array(test_flown["progress"].dropna().tolist())
    by_length = []
    for lo, hi in ((3, 10), (10, 20), (20, 30), (30, 45), (45, 80)):
        g = flown.loc[(flown["length_m"] >= lo) & (flown["length_m"] < hi)]
        if len(g):
            by_length.append({"length_m": f"{lo}-{hi}", "passes": int(len(g)), "median_speed_mps": round(float((g["length_m"] / g["time_s"]).median()), 2),
                              "v1_speed_mps": round(float(V1.min_speed + V1.speed_grad * g["length_m"].median()), 2),
                              "measured_speed_mps": round(float(measured.min_speed + measured.speed_grad * g["length_m"].median()), 2)})

    # 2. Players (physics).
    k_fit = kin.loc[kin["half"] == "fit"]
    players = {"players": int(len(kin)), "top_speed_mps": {q: round(float(kin["top_speed"].quantile(x)), 2) for q, x in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))},
               "peak_accel_mps2": {q: round(float(kin["peak_accel"].quantile(x)), 2) for q, x in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))},
               "speed_gain_first_second_from_standing_mps": round(float(kin["accel_from_standing_1s"].median()), 2)}
    kinematics = {"v1": (None, None), "measured": (round(float(k_fit["top_speed"].median()), 2), round(float(k_fit["peak_accel"].median()), 2))}

    # 3. Reaction time (outcomes): availability for the pass to the recipient at the kick.
    usable = seen.loc[seen["defenders"].map(len) > 0].reset_index(drop=True)
    comp = usable["completed"].to_numpy()
    fit_mask = (usable["half"] == "fit").to_numpy()
    flights = {"v1": V1, "measured_speed": measured, "measured_speed_pass_timing": measured_pass}
    grid = []
    for fname, base in flights.items():
        for kname, (vmax, amax) in kinematics.items():
            for rt in REACTIONS:
                fl = replace(base, reaction_time=rt, v_max=vmax, a_max=amax)
                a = availability(usable, fl)
                grid.append({"flight": fname, "kinematics": kname, "reaction_time": rt,
                             "auc_fit": auc(a[fit_mask], comp[fit_mask]), "auc_test": auc(a[~fit_mask], comp[~fit_mask])})
            print(fname, kname, "done", flush=True)
    grid = pd.DataFrame(grid)
    v1_row = {"flight": "v1", "kinematics": "v1", "reaction_time": 0.7}
    best = {}
    for fname in flights:
        for kname in kinematics:
            g = grid.loc[(grid["flight"] == fname) & (grid["kinematics"] == kname)]
            best[f"{fname}/{kname}"] = g.loc[g["auc_fit"].idxmax()].to_dict()
    chosen_key = max(best, key=lambda k: best[k]["auc_fit"])
    chosen = best[chosen_key]
    cflight = replace(flights[chosen["flight"]], reaction_time=chosen["reaction_time"], v_max=kinematics[chosen["kinematics"]][0],
                      a_max=kinematics[chosen["kinematics"]][1])
    a_v1, a_ch = availability(usable, V1), availability(usable, cflight)
    a_v1_end, a_ch_end = availability(usable, V1, "recipient_at_end"), availability(usable, cflight, "recipient_at_end")
    test = ~fit_mask
    out = {
        "passes": {"open_play_with_recipient": int(len(passes)), "kick_found": int(passes["kick"].sum()),
                   "first_touch": seen["touch_by"].value_counts(dropna=False).to_dict(),
                   "kick_offset_from_logged_time_s": {q: round(float(seen["kick_offset_s"].quantile(x)), 2) for q, x in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))}},
        "ball_flight": {"received_by_recipient": int(len(flown)), "fit": speed, "decel": decel, "v1": {"min_speed": V1.min_speed, "speed_grad": V1.speed_grad},
                        "test_errors": {"v1": flight_errors(test_flown, V1), "measured": flight_errors(test_flown, measured),
                                        **{f"measured_{h}": flight_errors(test_flown.loc[test_flown["height"] == h], FlightModel(**speed[h])) for h in ("flat", "high")},
                                        **{f"measured_all_on_{h}": flight_errors(test_flown.loc[test_flown["height"] == h], measured) for h in ("flat", "high")}},
                        "by_length": by_length,
                        "progress_test_mean": {"tau_0.25_0.5_0.75": [round(float(x), 3) for x in np.nanmean(prog_test, axis=0)],
                                               "constant_speed": [0.25, 0.5, 0.75],
                                               "fitted_decel": [round((1 + decel) * t - decel * t * t, 3) for t in (0.25, 0.5, 0.75)]}},
        "players": players, "kinematics_options": {k: {"v_max": v[0], "a_max": v[1]} for k, v in kinematics.items()},
        "reaction_grid": grid.to_dict(orient="records"),
        "best_per_model": best, "chosen": {"key": chosen_key, **chosen, "flight_model": asdict(cflight)},
        "v1": grid.loc[(grid["flight"] == "v1") & (grid["kinematics"] == "v1") & (grid["reaction_time"] == 0.7)].iloc[0].to_dict(),
        "target_at_end_of_pass": {"auc_test_v1": auc(a_v1_end[test], comp[test]), "auc_test_chosen": auc(a_ch_end[test], comp[test])},
        "calibration_test": {"v1": calibration_table(a_v1[test], comp[test]), "chosen": calibration_table(a_ch[test], comp[test])},
        "test_mean_availability": {"v1": {"completed": round(float(np.nanmean(a_v1[test & comp])), 3), "failed": round(float(np.nanmean(a_v1[test & ~comp])), 3)},
                                   "chosen": {"completed": round(float(np.nanmean(a_ch[test & comp])), 3), "failed": round(float(np.nanmean(a_ch[test & ~comp])), 3)}},
        "v1_row": v1_row,
    }
    if args.sc_data:
        sc_pk = args.cache / f"calibration_skillcorner.{tag()}.pkl"
        if sc_pk.exists():
            sc = pd.read_pickle(sc_pk)
        else:
            sc = pd.concat([skillcorner_rows(d, args.cache / "sc-cache") for d in sorted((args.sc_data / "matches").iterdir())
                            if (d / f"{d.name}_dynamic_events.csv").exists()], ignore_index=True)
            sc.to_pickle(sc_pk)
        sc = sc.loc[sc["defenders"].map(len) > 0].reset_index(drop=True)
        c = sc["completed"].to_numpy()
        sc_out = {"passes": int(len(sc)), "auc_sc_xpass": auc(sc["sc_xpass"].to_numpy(), c)}
        for name, fl in (("v1", V1), ("chosen", cflight)):
            sc_out[f"auc_{name}"] = auc(availability(sc, fl), c)
        for fname, base in flights.items():
            fl = replace(base, reaction_time=chosen["reaction_time"], v_max=kinematics[chosen["kinematics"]][0], a_max=kinematics[chosen["kinematics"]][1])
            sc_out[f"auc_{fname}_at_chosen_reaction"] = auc(availability(sc, fl), c)
        out["skillcorner_transfer"] = sc_out
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "reaction_grid"}, indent=1, default=str))


if __name__ == "__main__":
    main()
