"""
structure.py — context package P2: team structure.

Each team's shape on every grid point and its lines, formation and player roles per window
(``ponta-blueprint/context-packages.md``, P2; ``ponta-blueprint/context-structure-task.md``).

**Team frame.** Every measure is taken in the team's own frame: depth is metres from its own
goal line towards the opponent's, width is metres from the centre line with the team's left
positive. The direction of attack is read from the data for each team and period (the side
its goalkeeper defends), not assumed from the normaliser: Metrica Sample Game 2 is
normalised with the home team attacking −x, against the normaliser's stated convention.

**Per grid point and team** (outfield players observed at the grid point's native frame; the
goalkeeper is the team's deepest player on average over the window's two minutes):

* length, width, centroid depth, stretch index (mean distance to the centroid) and
  convex-hull area (Rico-González et al. 2021 give the definitions and official-match ranges);
* line heights: the deepest member of the defensive, midfield and attacking lines (FIFA's
  convention), and the gaps between line centroids (Forcher et al.).

**Per window** (two minutes within a period, split by possession state, using only
possessions of at least ``p2_min_possession_s``; Shaw and Glickman 2019):

* line membership: exact one-dimensional 3-means on players' mean depth (Forcher and Goes);
* the formation template that fits the players' mean positions best (centred, scaled per
  axis, assigned to template slots with the Hungarian algorithm; Bialkowski et al. 2016),
  its margin over the second best, and each player's role from the slot. Below
  ``p2_formation_min_margin`` the window is ``unclear``;
* the team's current formation: a change is accepted only when it holds for
  ``p2_formation_persist_s`` (SoccerCPD, Kim et al. 2022).

**Per player** (version 0.2.0, 18 November 2026): the main role for the match, read from the
out-of-possession windows (the team's nominal shape), else from all windows (``main_roles``).

Formations and roles are Ponta's labels from rules (estimated, not yet validated); shape
measures are observed. Nothing here reaches the AI as coordinates: depths and distances are
measures, not positions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matrix_engine.match import NormalisedMatch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError

from plaiq_engine.context.registry import (
    ContextField,
    ContextInput,
    ContextPackage,
    PackageOutput,
)

TEAMS = ("home", "away")
OTHER = {"home": "away", "away": "home"}
LINES = ("def", "mid", "att")

# ---------------------------------------------------------------------------
# Formation templates: (role, depth from own goal in m, width in m with left positive).
# Nominal shapes; only their relative layout matters (positions are centred and scaled
# per axis before matching).
# ---------------------------------------------------------------------------

_BACK4 = (("LB", 32, 24), ("LCB", 30, 8), ("RCB", 30, -8), ("RB", 32, -24))
_BACK3 = (("LCB", 30, 13), ("CB", 29, 0), ("RCB", 30, -13))

TEMPLATES: dict[str, tuple[tuple[str, float, float], ...]] = {
    "4-4-2": _BACK4 + (("LM", 48, 24), ("LCM", 46, 8), ("RCM", 46, -8), ("RM", 48, -24), ("LCF", 64, 7), ("RCF", 64, -7)),
    "4-3-3": _BACK4 + (("DM", 42, 0), ("LCM", 50, 12), ("RCM", 50, -12), ("LW", 62, 24), ("CF", 66, 0), ("RW", 62, -24)),
    "4-2-3-1": _BACK4 + (("LDM", 43, 8), ("RDM", 43, -8), ("LW", 58, 24), ("AM", 57, 0), ("RW", 58, -24), ("CF", 66, 0)),
    "4-1-4-1": _BACK4 + (("DM", 42, 0), ("LM", 52, 24), ("LCM", 52, 9), ("RCM", 52, -9), ("RM", 52, -24), ("CF", 66, 0)),
    "4-1-2-1-2": _BACK4 + (("DM", 42, 0), ("LCM", 48, 14), ("RCM", 48, -14), ("AM", 56, 0), ("LCF", 64, 7), ("RCF", 64, -7)),
    "3-5-2": _BACK3 + (("LWB", 44, 28), ("DM", 42, 0), ("LCM", 50, 12), ("RCM", 50, -12), ("RWB", 44, -28), ("LCF", 64, 7), ("RCF", 64, -7)),
    "3-4-3": _BACK3 + (("LWB", 46, 28), ("LCM", 46, 8), ("RCM", 46, -8), ("RWB", 46, -28), ("LW", 62, 20), ("CF", 66, 0), ("RW", 62, -20)),
    "3-4-2-1": _BACK3 + (("LWB", 46, 28), ("LDM", 44, 8), ("RDM", 44, -8), ("RWB", 46, -28), ("LAM", 58, 12), ("RAM", 58, -12), ("CF", 66, 0)),
    "5-3-2": (("LWB", 34, 27),) + _BACK3 + (("RWB", 34, -27), ("LCM", 46, 13), ("CM", 45, 0), ("RCM", 46, -13), ("LCF", 62, 7), ("RCF", 62, -7)),
    "5-4-1": (("LWB", 34, 27),) + _BACK3 + (("RWB", 34, -27), ("LM", 48, 22), ("LCM", 46, 7), ("RCM", 46, -7), ("RM", 48, -22), ("CF", 62, 0)),
}


def _scale(xy: np.ndarray) -> np.ndarray:
    """Centre and scale each axis to unit standard deviation (a team's spread varies by phase)."""
    c = xy - xy.mean(axis=0)
    sd = c.std(axis=0)
    sd[sd == 0] = 1.0
    return c / sd


_TEMPLATE_XY = {k: _scale(np.array([(d, w) for _, d, w in v], float)) for k, v in TEMPLATES.items()}


def _assign(p: np.ndarray, name: str) -> tuple[float, list[str]]:
    t = _TEMPLATE_XY[name]
    cost = ((p[:, None, :] - t[None, :, :]) ** 2).sum(axis=2)
    r, c = linear_sum_assignment(cost)
    roles = [""] * len(p)
    for i, j in zip(r, c, strict=True):
        roles[i] = TEMPLATES[name][j][0]
    return float(cost[r, c].mean()), roles


def roles_for(xy: np.ndarray, name: str) -> list[str]:
    """Each player's slot in one named template (rows in the order of ``xy``)."""
    return _assign(_scale(np.asarray(xy, float)), name)[1]


def fit_formation(xy: np.ndarray) -> dict:
    """
    Best and second-best templates for players' mean positions (depth, width), by the
    Hungarian assignment's mean squared distance after scaling. Returns the roles in the
    order of ``xy``'s rows.
    """
    p = _scale(np.asarray(xy, float))
    scores = [(*_assign(p, name), name) for name in _TEMPLATE_XY]
    scores = [(c, name, roles) for c, roles, name in scores]
    scores.sort(key=lambda s: s[0])  # stable: ties keep the template order
    (c1, best, roles), (c2, second, _) = scores[0], scores[1]
    return {"best": best, "cost": c1, "second": second, "second_cost": c2,
            "margin": (c2 - c1) / c2 if c2 > 0 else 0.0, "roles": roles}


def assign_lines(depths: np.ndarray) -> np.ndarray:
    """
    Exact one-dimensional 3-means: the split of players sorted by depth into three
    contiguous groups with the least within-group sum of squares. Deterministic.
    Returns 0 (defence), 1 (midfield) or 2 (attack) per player; fewer than three players
    are all defence.
    """
    d = np.asarray(depths, float)
    n = len(d)
    out = np.zeros(n, dtype=np.int64)
    if n < 3:
        return out
    order = np.argsort(d, kind="stable")
    s = d[order]

    def sse(a: int, b: int) -> float:
        g = s[a:b]
        return float(((g - g.mean()) ** 2).sum())

    best = None
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            v = sse(0, i) + sse(i, j) + sse(j, n)
            if best is None or v < best[0] - 1e-12:
                best = (v, i, j)
    _, i, j = best
    lab = np.zeros(n, dtype=np.int64)
    lab[i:j], lab[j:] = 1, 2
    out[order] = lab
    return out


# ---------------------------------------------------------------------------
# Positions in the team frame
# ---------------------------------------------------------------------------

@dataclass
class TeamFrames:
    """One team's outfield-and-goalkeeper positions on the grid, in its own frame."""

    ids: list[str]
    depth: np.ndarray  # (grid points, players) metres from own goal line; NaN where unobserved
    width: np.ndarray  # same shape; metres from the centre line, team's left positive


def attack_signs(match: NormalisedMatch, grid: pd.DataFrame) -> pd.DataFrame:
    """
    Each team's direction of attack per period (+1 towards +x), from the side its goalkeeper
    defends: the player with the largest absolute mean x among those seen for at least half
    the period. ``consistent`` is False when both teams appear to attack the same way.
    """
    t = match.tracking[["frame", "team", "player_id", "x"]].merge(
        grid[["native_frame", "period"]].rename(columns={"native_frame": "frame"}), on="frame", how="inner")
    rows = []
    for p in sorted(grid["period"].unique()):
        n = int((grid["period"] == p).sum())
        sign = {}
        for team in TEAMS:
            g = t.loc[(t["period"] == p) & (t["team"] == team)].groupby("player_id")["x"].agg(["mean", "size"])
            g = g.loc[g["size"] >= 0.5 * n]
            if g.empty:
                sign[team] = 1 if team == "home" else -1
                continue
            gk = g["mean"].abs().sort_index().idxmax()
            sign[team] = -1 if g.at[gk, "mean"] > 0 else 1
        for team in TEAMS:
            rows.append({"period": int(p), "team": team, "attack_sign": int(sign[team]),
                         "consistent": sign["home"] == -sign["away"]})
    return pd.DataFrame(rows, columns=["period", "team", "attack_sign", "consistent"])


def team_frames(match: NormalisedMatch, grid: pd.DataFrame, signs: pd.DataFrame) -> dict[str, TeamFrames]:
    half = match.pitch.length_m / 2.0
    nf = grid["native_frame"].to_numpy(np.int64)
    uniq = np.unique(nf)
    per = grid["period"].to_numpy()
    sign_of = {(int(r.period), r.team): r.attack_sign for r in signs.itertuples()}
    out = {}
    tr = match.tracking.loc[match.tracking["frame"].isin(uniq), ["frame", "team", "player_id", "x", "y"]]
    for team in TEAMS:
        tt = tr.loc[tr["team"] == team]
        ids = sorted(tt["player_id"].unique())
        col = {pid: i for i, pid in enumerate(ids)}
        rows_u = np.searchsorted(uniq, tt["frame"].to_numpy(np.int64))
        cols = tt["player_id"].map(col).to_numpy(np.int64)
        Xu = np.full((len(uniq), len(ids)), np.nan)
        Yu = np.full((len(uniq), len(ids)), np.nan)
        Xu[rows_u, cols] = tt["x"].to_numpy(float)
        Yu[rows_u, cols] = tt["y"].to_numpy(float)
        idx = np.searchsorted(uniq, nf)
        X, Y = Xu[idx], Yu[idx]
        s = np.array([sign_of.get((int(p), team), 1 if team == "home" else -1) for p in per], float)[:, None]
        out[team] = TeamFrames(ids=ids, depth=s * X + half, width=s * Y)
    return out


# ---------------------------------------------------------------------------
# Possession states and windows
# ---------------------------------------------------------------------------

def team_states(grid: pd.DataFrame, p1: pd.DataFrame, chains: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    """
    ``in`` / ``out`` of possession per team at each grid point, counting only possessions
    (P1 chains) of at least ``p2_min_possession_s``; None elsewhere.
    """
    lab = p1["p1_possession"].to_numpy(object)
    cid = p1["p1_chain_id"].to_numpy(dtype=float, na_value=np.nan)
    long_ids = set(chains.loc[chains["duration_s"] >= cfg["p2_min_possession_s"], "chain_id"].astype(int)) if len(chains) else set()
    ok = np.array([not np.isnan(c) and int(c) in long_ids for c in cid])
    out = {}
    for team in TEAMS:
        s = np.full(len(grid), None, dtype=object)
        s[ok & (lab == team)] = "in"
        s[ok & (lab == OTHER[team])] = "out"
        out[team] = s
    return out


def window_index(grid: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Two-minute blocks within each period (block 0 starts at the period's first grid point)."""
    ts, per = grid["frame_ts"].to_numpy(float), grid["period"].to_numpy()
    out = np.zeros(len(grid), dtype=np.int64)
    for p in np.unique(per):
        m = per == p
        out[m] = np.floor((ts[m] - ts[m].min()) / cfg["p2_window_s"]).astype(np.int64)
    return out


def _persist(labels: list[str | None], durations: list[float], min_s: float) -> list[str | None]:
    """Accept a formation change only when the new formation holds, unbroken, for ``min_s``."""
    out: list[str | None] = []
    current: str | None = None
    i, n = 0, len(labels)
    while i < n:
        lab = labels[i]
        if lab is None or lab == "unclear" or lab == current:
            out.append(current)
            i += 1
            continue
        j, run = i, 0.0
        while j < n and labels[j] in (lab, "unclear", None):
            run += durations[j] if labels[j] == lab else 0.0
            j += 1
        if current is None or run >= min_s:
            current = lab
        out.append(current)
        i += 1
    # The first accepted formation also covers the windows before it.
    first = next((x for x in out if x is not None), None)
    return [x if x is not None else first for x in out]


@dataclass
class Windows:
    formations: pd.DataFrame
    roles: pd.DataFrame
    gk: dict[tuple[str, int, int], str]  # (team, period, block) -> goalkeeper id


def analyse_windows(grid: pd.DataFrame, tf: dict[str, TeamFrames], states: dict[str, np.ndarray], block: np.ndarray, cfg: dict) -> Windows:
    ts, per = grid["frame_ts"].to_numpy(float), grid["period"].to_numpy()
    step = 1.0 / cfg["context_grid_rate_hz"]
    frows, rrows, gks = [], [], {}
    for team in TEAMS:
        f = tf[team]
        for p in sorted(np.unique(per)):
            for b in sorted(np.unique(block[per == p])):
                m_block = (per == p) & (block == b)
                seen = np.isfinite(f.depth[m_block])
                pres = seen.mean(axis=0)
                cand = np.where(pres >= cfg["p2_min_presence_share"])[0]
                if len(cand) == 0:
                    continue
                mean_depth = np.nanmean(np.where(seen, f.depth[m_block], np.nan)[:, cand], axis=0)
                gk = int(cand[np.argmin(mean_depth)])
                gks[(team, int(p), int(b))] = f.ids[gk]
                for state in ("in", "out", "all"):
                    m = m_block & (states[team] == state) if state != "all" else m_block & (states[team] != None)
                    n = int(m.sum())
                    if n * step < cfg["p2_min_window_data_s"]:
                        continue
                    sseen = np.isfinite(f.depth[m])
                    spres = sseen.mean(axis=0)
                    players = [i for i in np.argsort(-spres, kind="stable") if spres[i] >= cfg["p2_min_presence_share"] and i != gk][:10]
                    players = sorted(players)
                    if len(players) < cfg["p2_min_outfield_players"]:
                        continue
                    xy = np.column_stack([np.nanmean(f.depth[m][:, players], axis=0), np.nanmean(f.width[m][:, players], axis=0)])
                    lines = assign_lines(xy[:, 0])
                    fit = fit_formation(xy)
                    label = fit["best"] if fit["margin"] >= cfg["p2_formation_min_margin"] else "unclear"
                    start, end = float(ts[m_block].min()), float(ts[m_block].max())
                    frows.append({"team": team, "period": int(p), "block": int(b), "state": state, "start_ts": start, "end_ts": end,
                                  "data_s": round(n * step, 1), "players": len(players), "best": fit["best"],
                                  "cost": round(fit["cost"], 4), "second": fit["second"], "margin": round(fit["margin"], 4), "formation": label})
                    for k, i in enumerate(players):
                        rrows.append({"team": team, "period": int(p), "block": int(b), "state": state, "start_ts": start, "end_ts": end,
                                      "player_id": f.ids[i], "line": LINES[lines[k]], "role": fit["roles"][k],
                                      "_d": xy[k, 0], "_w": xy[k, 1]})
    formations = pd.DataFrame(frows, columns=["team", "period", "block", "state", "start_ts", "end_ts", "data_s", "players", "best",
                                              "cost", "second", "margin", "formation"])
    formations["current"] = None
    for (team, state), g in formations.groupby(["team", "state"], sort=True):
        g = g.sort_values(["period", "block"])
        formations.loc[g.index, "current"] = _persist(g["formation"].tolist(), g["data_s"].tolist(), cfg["p2_formation_persist_s"])
    roles = pd.DataFrame(rrows, columns=["team", "period", "block", "state", "start_ts", "end_ts", "player_id", "line", "role", "_d", "_w"])
    # Roles follow the team's current (persisted) formation, so one window's odd fit does not
    # relabel players; the window's own best fit stays in ``formations``.
    if len(roles):
        cur = formations.set_index(["team", "period", "block", "state"])["current"]
        for key, g in roles.groupby(["team", "period", "block", "state"], sort=True):
            name = cur.get(key)
            if name:
                roles.loc[g.index, "role"] = roles_for(g[["_d", "_w"]].to_numpy(float), name)
    roles = roles.drop(columns=["_d", "_w"])
    return Windows(formations=formations, roles=roles, gk=gks)


def main_roles(roles: pd.DataFrame, formations: pd.DataFrame) -> pd.DataFrame:
    """
    Each player's main role for the match: the slot they hold for the most window data **out of
    possession**, where a team keeps its nominal shape; in possession, wide players and a forward
    push on and a 4-4-2 often fits as a 4-2-3-1. Players with no out-of-possession window fall back
    to all windows. Ties go to the alphabetically first role, so the result is deterministic.

    Returns one row per player: ``team``, ``player_id``, ``role``, ``state`` (``out`` or ``all``,
    the windows read), ``role_s`` (window data in that role) and ``share`` (of the player's data
    in that state).
    """
    cols = ["team", "player_id", "role", "state", "role_s", "share"]
    if roles.empty:
        return pd.DataFrame(columns=cols)
    r = roles.merge(formations[["team", "period", "block", "state", "data_s"]], on=["team", "period", "block", "state"])
    out: dict[str, dict] = {}
    for state in ("all", "out"):  # "out" is read last, so it wins where it exists
        s = r.loc[r["state"] == state]
        if s.empty:
            continue
        per = s.groupby(["team", "player_id", "role"])["data_s"].sum().reset_index()
        tot = per.groupby("player_id")["data_s"].sum()
        best = per.sort_values(["data_s", "role"], ascending=[False, True]).drop_duplicates("player_id")
        for b in best.itertuples():
            out[b.player_id] = {"team": b.team, "player_id": b.player_id, "role": b.role, "state": state,
                                "role_s": round(float(b.data_s), 1), "share": round(float(b.data_s / tot[b.player_id]), 4)}
    return pd.DataFrame(sorted(out.values(), key=lambda d: (d["team"], d["player_id"])), columns=cols)


def windows_from_segments(segments: dict[str, pd.DataFrame]) -> Windows:
    """Rebuild P2's window results from its segment tables (used by packages that depend on P2)."""
    gk = {(r.team, int(r.period), int(r.block)): r.player_id for r in segments["goalkeepers"].itertuples()}
    return Windows(formations=segments["formations"], roles=segments["roles"], gk=gk)


def line_members(grid: pd.DataFrame, tf: dict[str, TeamFrames], states: dict[str, np.ndarray], block: np.ndarray,
                 win: Windows) -> dict[str, np.ndarray]:
    """
    Line of each player at each grid point: 0, 1, 2 (defence, midfield, attack), 3 for the
    goalkeeper, -1 when unknown. Uses the window of the team's current state, falling back
    to the block's all-possession window.
    """
    per = grid["period"].to_numpy()
    code = {ln: i for i, ln in enumerate(LINES)}
    out = {}
    lookup: dict[tuple, dict[str, int]] = {}
    for r in win.roles.itertuples():
        lookup.setdefault((r.team, r.period, r.block, r.state), {})[r.player_id] = code[r.line]
    for team in TEAMS:
        f = tf[team]
        col = {pid: i for i, pid in enumerate(f.ids)}
        L = np.full(f.depth.shape, -1, dtype=np.int8)
        names = (None, "in", "out")
        sc = np.array([names.index(v) for v in states[team]], dtype=np.int64)
        for p_, b_, s_ in sorted(set(zip(per.tolist(), block.tolist(), sc.tolist(), strict=True))):
            m = (per == p_) & (block == b_) & (sc == s_)
            mapping = lookup.get((team, p_, b_, names[s_])) or lookup.get((team, p_, b_, "all")) or {}
            for pid, ln in mapping.items():
                L[m, col[pid]] = ln
            gk = win.gk.get((team, p_, b_))
            if gk is not None:
                L[m, col[gk]] = 3
        out[team] = L
    return out


# ---------------------------------------------------------------------------
# Per-frame shape
# ---------------------------------------------------------------------------

def _hull_area(xs: np.ndarray, ys: np.ndarray) -> float:
    pts = np.column_stack([xs, ys])
    if len(pts) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts).volume)
    except QhullError:
        return 0.0


def shape_measures(tf: TeamFrames, lines: np.ndarray, min_players: int) -> dict[str, np.ndarray]:
    D, W = tf.depth, tf.width
    outfield = np.isfinite(D) & (lines != 3)
    Do, Wo = np.where(outfield, D, np.nan), np.where(outfield, W, np.nan)
    n = outfield.sum(axis=1)
    ok = n >= min_players
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        length = np.nanmax(Do, axis=1) - np.nanmin(Do, axis=1)
        width = np.nanmax(Wo, axis=1) - np.nanmin(Wo, axis=1)
        cx, cy = np.nanmean(Do, axis=1), np.nanmean(Wo, axis=1)
        stretch = np.nanmean(np.hypot(Do - cx[:, None], Wo - cy[:, None]), axis=1)
        deepest = np.nanmin(Do, axis=1)
        line_h, line_c = {}, {}
        for k, ln in enumerate(LINES):
            Dl = np.where(outfield & (lines == k), D, np.nan)
            line_h[ln] = np.nanmin(Dl, axis=1)
            line_c[ln] = np.nanmean(Dl, axis=1)
    hull = np.full(len(D), np.nan)
    for i in np.where(ok)[0]:
        sel = outfield[i]
        hull[i] = _hull_area(D[i, sel], W[i, sel])
    nan = lambda a: np.where(ok, a, np.nan)
    return {
        "outfield_n": n.astype(np.int64),
        "length_m": nan(length), "width_m": nan(width), "centroid_depth_m": nan(cx), "stretch_m": nan(stretch),
        "hull_m2": nan(hull), "def_line_m": nan(deepest),
        "mid_line_m": nan(line_h["mid"]), "att_line_m": nan(line_h["att"]),
        "def_mid_gap_m": nan(line_c["mid"] - line_c["def"]), "mid_att_gap_m": nan(line_c["att"] - line_c["mid"]),
    }


def current_formation(grid: pd.DataFrame, block: np.ndarray, formations: pd.DataFrame, team: str, state: str) -> np.ndarray:
    """The team's current formation in ``state`` at each grid point, carried across blocks without a window."""
    per = grid["period"].to_numpy()
    out = np.full(len(grid), None, dtype=object)
    f = formations.loc[(formations["team"] == team) & (formations["state"] == state)]
    by = {(int(r.period), int(r.block)): r.current for r in f.itertuples()}
    last = None
    keys = pd.DataFrame({"p": per, "b": block}).drop_duplicates()
    for k in keys.itertuples(index=False):
        v = by.get((int(k.p), int(k.b)))
        if v is not None:
            last = v
        out[(per == k.p) & (block == k.b)] = last
    first = next((v for v in out if v is not None), None)
    return np.array([v if v is not None else first for v in out], dtype=object)


# ---------------------------------------------------------------------------
# The package
# ---------------------------------------------------------------------------

_MEASURES = (
    ("outfield_n", "players", "observed", "Outfield players of the team observed at this grid point (the goalkeeper excluded)."),
    ("length_m", "m", "observed", "Depth from the team's deepest to its highest outfield player."),
    ("width_m", "m", "observed", "Distance between the team's two widest outfield players."),
    ("centroid_depth_m", "m", "observed", "Mean distance of the outfield players from the team's own goal line."),
    ("stretch_m", "m", "observed", "Stretch index: mean distance of the outfield players from their centroid."),
    ("hull_m2", "m²", "observed", "Area of the convex hull around the outfield players."),
    ("def_line_m", "m", "observed", "Defensive line height: the deepest outfield player's distance from the team's own goal line."),
    ("mid_line_m", "m", "estimated", "Midfield line height: the deepest midfield-line player's distance from the own goal line. Line membership is Ponta's 3-means grouping per window."),
    ("att_line_m", "m", "estimated", "Attacking line height: the deepest attacking-line player's distance from the own goal line."),
    ("def_mid_gap_m", "m", "estimated", "Distance between the defensive and midfield lines' centroids (depth)."),
    ("mid_att_gap_m", "m", "estimated", "Distance between the midfield and attacking lines' centroids (depth)."),
    ("formation_in", "label", "estimated", "The team's current formation in possession (best-fitting template, held at least the persistence time). Estimated, not yet validated."),
    ("formation_out", "label", "estimated", "The team's current formation out of possession. Estimated, not yet validated."),
)


@dataclass
class _Fields:
    items: tuple[ContextField, ...] = field(default=tuple(
        ContextField(f"p2_{team}_{name}", unit, kind, definition.replace("The team", f"The {team} team").replace("the team", f"the {team} team"))
        for team in TEAMS for name, unit, kind, definition in _MEASURES))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


class StructurePackage(ContextPackage):
    name = "p2"
    version = "0.2.0"
    depends_on = ("p1",)
    fields = _Fields().items

    def compute(self, ctx: ContextInput) -> PackageOutput:
        m, g, cfg = ctx.match, ctx.grid, ctx.config
        p1 = ctx.upstream["p1"]
        signs = attack_signs(m, g)
        tf = team_frames(m, g, signs)
        states = team_states(g, p1.frames, p1.segments.get("chains", pd.DataFrame()), cfg)
        block = window_index(g, cfg)
        win = analyse_windows(g, tf, states, block, cfg)
        lines = line_members(g, tf, states, block, win)

        cols: dict[str, object] = {}
        shape = {}
        for team in TEAMS:
            sm = shape_measures(tf[team], lines[team], int(cfg["p2_min_outfield_players"]))
            shape[team] = sm
            for name, *_ in _MEASURES:
                key = f"p2_{team}_{name}"
                if name == "formation_in":
                    cols[key] = current_formation(g, block, win.formations, team, "in")
                elif name == "formation_out":
                    cols[key] = current_formation(g, block, win.formations, team, "out")
                elif name == "outfield_n":
                    cols[key] = sm[name]
                else:
                    cols[key] = np.round(sm[name], 3)
        frames = pd.DataFrame(cols)

        # -- measurements and the Data check ---------------------------------------
        in_play = p1.frames["p1_possession"].to_numpy(object) != "dead_ball"
        has_shape = np.isfinite(shape["home"]["length_m"]) & np.isfinite(shape["away"]["length_m"])
        coverage = float(has_shape[in_play].mean()) if in_play.any() else 0.0
        consistent = bool(signs["consistent"].all()) if len(signs) else False
        fm = win.formations
        clear = fm.loc[fm["state"] != "all"]
        clear_share = float((clear["formation"] != "unclear").mean()) if len(clear) else 0.0

        def summary(team: str, key: str) -> dict:
            a = shape[team][key][in_play & has_shape]
            a = a[np.isfinite(a)]
            return {"median": round(float(np.median(a)), 2), "p10": round(float(np.percentile(a, 10)), 2),
                    "p90": round(float(np.percentile(a, 90)), 2)} if len(a) else {}

        measurements = {
            "attack_signs": signs.to_dict("records"),
            "orientation_consistent": consistent,
            "shape_coverage_in_play": round(coverage, 4),
            "shape": {t: {k: summary(t, k) for k in ("length_m", "width_m", "centroid_depth_m", "stretch_m", "hull_m2",
                                                     "def_line_m", "mid_line_m", "att_line_m", "def_mid_gap_m", "mid_att_gap_m")}
                      for t in TEAMS},
            "windows": {"count": int(len(clear)), "clear_share": round(clear_share, 4),
                        "margin_median": round(float(clear["margin"].median()), 4) if len(clear) else None},
            "formations": {f"{t}_{s}": fm.loc[(fm["team"] == t) & (fm["state"] == s), "current"].value_counts().to_dict()
                           for t in TEAMS for s in ("in", "out")},
            "formation_changes": {f"{t}_{s}": int((fm.loc[(fm["team"] == t) & (fm["state"] == s)].sort_values(["period", "block"])["current"]
                                                   .pipe(lambda c: (c != c.shift()).sum() - 1)) if ((fm["team"] == t) & (fm["state"] == s)).any() else 0)
                                  for t in TEAMS for s in ("in", "out")},
        }

        if coverage == 0.0:
            state = "blocked"
            why = "No in-play moment has enough outfield players observed for a team shape."
        elif not consistent:
            state = "degraded"
            why = "Both teams appear to attack the same way in at least one period, so the team frame is uncertain."
        elif coverage < cfg["p2_min_shape_coverage"]:
            state = "degraded"
            why = f"Fewer than {_pct(cfg['p2_min_shape_coverage'])} of in-play moments have at least {int(cfg['p2_min_outfield_players'])} outfield players observed per team."
        else:
            state = "full"
            why = "Both teams' outfield players are observed for nearly all of in-play time."
        unlock = None if state == "full" else "Full-pitch tracking (all 22 players in view)."
        entries = [
            self.entry("team_shape", "Team shape", state, why, coverage=f"{_pct(coverage)} of in-play time", unlock=unlock),
            self.entry("formation", "Formation and roles", state,
                       "Formations and roles are Ponta's template fits to mean positions per two-minute window: estimated, not yet validated."
                       if state != "blocked" else why,
                       measure=f"{_pct(clear_share)} of windows fit one template clearly", coverage=f"{_pct(coverage)} of in-play time",
                       unlock=unlock),
        ]
        gk = pd.DataFrame([{"team": t, "period": p_, "block": b_, "player_id": pid} for (t, p_, b_), pid in sorted(win.gk.items())],
                          columns=["team", "period", "block", "player_id"])
        segments = {"formations": win.formations, "roles": win.roles, "main_roles": main_roles(win.roles, win.formations),
                    "orientation": signs, "goalkeepers": gk}
        return PackageOutput(frames=frames, entries=entries, segments=segments, measurements=measurements)
