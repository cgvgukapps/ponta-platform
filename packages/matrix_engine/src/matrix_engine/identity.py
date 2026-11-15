#!/usr/bin/env python3
"""
identity.py — work out which player in source A is which player in source B.

Two vendors tracking the same match will not agree on player identifiers. One
keys on shirt number, another on an internal vision ID, a third on a wearable's
serial. Reconciling them by hand is a chunk of the manual work Matrix exists to
remove, so it has to be inferred from the only thing both feeds genuinely share:
where the players actually were.

Matching is done on **trajectories, not single frames**. At any one instant
several players stand within a couple of metres of each other and a nearest-dot
rule picks the wrong one routinely; over a few hundred frames each player's path
is essentially unique. The cost between a candidate pair is the mean distance
between their trajectories over the times both were observed.

The assignment itself is solved globally with the Hungarian algorithm rather
than greedily. Greedy matching lets one confident pair strand two others into a
mutually wrong swap; the global optimum over the whole cost matrix does not.

Matching is per team. Teams are given by both providers, and using that halves
the problem while removing the possibility of matching a defender to an
opponent who happened to run a similar path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

#: Pairs whose mean trajectory distance exceeds this are refused, even if the
#: assignment would otherwise pick them. Set well above realistic sensor
#: disagreement so it only rejects genuinely implausible matches.
DEFAULT_MAX_MATCH_DISTANCE_M: float = 12.0

#: Minimum shared observations before a pair's cost is considered meaningful.
DEFAULT_MIN_OVERLAP: int = 20


@dataclass
class IdentityResult:
    """
    Outcome of matching one source's players onto a reference's.

    Attributes
    ----------
    mapping : dict[str, str]
        ``other`` player_id -> ``reference`` player_id, for confident matches
        only.
    unmatched : list[str]
        ``other`` player_ids left unassigned — no candidate cleared the
        distance threshold, or they lacked overlapping observations.
    costs : pd.DataFrame
        One row per accepted match with its mean trajectory distance, the
        margin to that player's runner-up, and the overlap count. The margin is
        the useful diagnostic: a small one means the match was nearly a coin
        flip between two candidates.
    cost_matrices : dict[str, pd.DataFrame]
        Full per-team cost matrix, retained for inspection.
    """

    mapping: dict[str, str]
    unmatched: list[str]
    costs: pd.DataFrame
    cost_matrices: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def accuracy_against(self, truth: dict[str, str]) -> float:
        """
        Fraction of ``truth``'s pairs this mapping got right.

        Scored over every entry in ``truth``, so an unmatched player counts as
        a miss — declining to answer is not the same as being right.
        """
        if not truth:
            return float("nan")
        correct = sum(1 for k, v in truth.items() if self.mapping.get(k) == v)
        return correct / len(truth)


def match_identities(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    *,
    max_distance_m: float = DEFAULT_MAX_MATCH_DISTANCE_M,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
    time_tolerance_s: float = 1e-6,
) -> IdentityResult:
    """
    Match ``other``'s players onto ``reference``'s by trajectory similarity.

    Both frames must already be on a common clock — run ``sync.synchronize``
    first. Rows are joined on exact timestamps (within ``time_tolerance_s``),
    which the shared resampling grid guarantees.

    Parameters
    ----------
    reference, other : pd.DataFrame
        Matrix-schema tracking, typically two slices of one ``SyncResult``.
    max_distance_m : float
        Reject a match whose mean trajectory distance exceeds this.
    min_overlap : int
        Require at least this many shared timestamps before trusting a cost.
    time_tolerance_s : float
        Rounding applied to timestamps before joining, to absorb float error.

    Returns
    -------
    IdentityResult
    """
    if reference.empty or other.empty:
        return IdentityResult({}, sorted(other["player_id"].unique().tolist()),
                              _empty_costs(), {})

    ref_wide = _pivot(reference, time_tolerance_s)
    other_wide = _pivot(other, time_tolerance_s)

    ref_teams = _team_lookup(reference)
    other_teams = _team_lookup(other)

    mapping: dict[str, str] = {}
    matched_rows: list[dict] = []
    cost_matrices: dict[str, pd.DataFrame] = {}
    matched_other: set[str] = set()

    for team in sorted(set(other_teams.values())):
        ref_players = sorted(p for p, t in ref_teams.items() if t == team)
        other_players = sorted(p for p, t in other_teams.items() if t == team)
        if not ref_players or not other_players:
            continue

        cost, overlap = _cost_matrix(ref_wide, other_wide, ref_players, other_players)
        cost_matrices[team] = pd.DataFrame(cost, index=other_players, columns=ref_players)

        # linear_sum_assignment cannot take infinities; substitute a finite
        # sentinel above any acceptable cost and reject those pairs afterwards.
        sentinel = max_distance_m * 1000.0
        solvable = np.where(np.isfinite(cost), cost, sentinel)
        rows, cols = linear_sum_assignment(solvable)

        for r, c in zip(rows, cols):
            other_id = other_players[r]
            ref_id = ref_players[c]
            pair_cost = cost[r, c]
            if not np.isfinite(pair_cost) or pair_cost > max_distance_m:
                continue
            if overlap[r, c] < min_overlap:
                continue
            mapping[other_id] = ref_id
            matched_other.add(other_id)
            matched_rows.append(
                {
                    "other_id": other_id,
                    "reference_id": ref_id,
                    "team": team,
                    "distance_m": float(pair_cost),
                    "runner_up_margin_m": _runner_up_margin(cost[r], c),
                    "overlap": int(overlap[r, c]),
                }
            )

    unmatched = sorted(set(other_teams) - matched_other)
    costs = (
        pd.DataFrame(matched_rows).sort_values("distance_m").reset_index(drop=True)
        if matched_rows
        else _empty_costs()
    )
    return IdentityResult(mapping, unmatched, costs, cost_matrices)


def split_sources(synced: pd.DataFrame, reference: str, other: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: split a combined synced frame back into two single-source frames."""
    return (
        synced.loc[synced["source"] == reference],
        synced.loc[synced["source"] == other],
    )


def apply_mapping(
    df: pd.DataFrame, mapping: dict[str, str], *, drop_unmatched: bool = True
) -> pd.DataFrame:
    """
    Relabel a source's players into the reference's ID space.

    Parameters
    ----------
    drop_unmatched : bool
        Drop players with no mapping. Keeping them would mean carrying a player
        that fusion cannot pair with anything, under an ID that collides with
        nothing — usually noise rather than signal.
    """
    out = df.copy()
    mapped = out["player_id"].map(mapping)
    if drop_unmatched:
        out = out.loc[mapped.notna()].copy()
        out["player_id"] = mapped.loc[mapped.notna()].to_numpy()
    else:
        out["player_id"] = mapped.fillna(out["player_id"]).to_numpy()
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_costs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["other_id", "reference_id", "team", "distance_m",
                 "runner_up_margin_m", "overlap"]
    )


def _team_lookup(df: pd.DataFrame) -> dict[str, str]:
    return (
        df.drop_duplicates("player_id")
        .set_index("player_id")["team"]
        .astype(str)
        .to_dict()
    )


def _pivot(df: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reshape long tracking into (x, y) wide frames indexed by rounded timestamp.

    Rounding to ``tolerance`` makes the join between sources exact despite the
    float arithmetic that produced the shared grid.
    """
    work = df.loc[:, ["timestamp", "player_id", "x", "y"]].copy()
    decimals = max(0, int(round(-np.log10(tolerance))))
    work["timestamp"] = work["timestamp"].round(decimals)
    work = work.drop_duplicates(subset=["timestamp", "player_id"], keep="first")
    x = work.pivot(index="timestamp", columns="player_id", values="x")
    y = work.pivot(index="timestamp", columns="player_id", values="y")
    return x, y


def _cost_matrix(
    ref_wide: tuple[pd.DataFrame, pd.DataFrame],
    other_wide: tuple[pd.DataFrame, pd.DataFrame],
    ref_players: list[str],
    other_players: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean trajectory distance between every (other, reference) player pair.

    Returns ``(cost, overlap)``; cost is ``inf`` where the pair never shared a
    timestamp.
    """
    ref_x, ref_y = ref_wide
    other_x, other_y = other_wide

    common_times = ref_x.index.intersection(other_x.index)
    n_other, n_ref = len(other_players), len(ref_players)
    cost = np.full((n_other, n_ref), np.inf)
    overlap = np.zeros((n_other, n_ref), dtype=int)
    if len(common_times) == 0:
        return cost, overlap

    rx = ref_x.reindex(index=common_times, columns=ref_players).to_numpy(dtype=float)
    ry = ref_y.reindex(index=common_times, columns=ref_players).to_numpy(dtype=float)
    ox = other_x.reindex(index=common_times, columns=other_players).to_numpy(dtype=float)
    oy = other_y.reindex(index=common_times, columns=other_players).to_numpy(dtype=float)

    for i in range(n_other):
        dx = ox[:, i][:, None] - rx
        dy = oy[:, i][:, None] - ry
        distance = np.sqrt(dx * dx + dy * dy)
        valid = np.isfinite(distance)
        counts = valid.sum(axis=0)
        totals = np.where(valid, distance, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, totals / np.maximum(counts, 1), np.inf)
        cost[i] = means
        overlap[i] = counts
    return cost, overlap


def _runner_up_margin(row: np.ndarray, chosen: int) -> float:
    """Gap between the chosen pairing's cost and the next best for that player."""
    finite = np.isfinite(row)
    if finite.sum() < 2:
        return float("inf")
    others = np.delete(row, chosen)
    others = others[np.isfinite(others)]
    if others.size == 0:
        return float("inf")
    return float(others.min() - row[chosen])
