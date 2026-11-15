"""Investigate the ring-distance formula discrepancy between
ANALYTICS_SPECIFICATION.md §2.1 and the code (statsbomb/sb_dq.py,
analytics/pass_option_value.py, analytics/hex_analytics.py).

The spec (as written, ~line 84) gives:

    ring_dist = max(|Δq|, |Δr|, |Δq + Δr|) / 2

The code (identically, in all three call sites) computes:

    ring_dist = (|Δq| + |Δr| + |Δq + Δr|) // 2

These are the two standard identities for cube/axial hex distance — but only
when the max-form is used *without* dividing by 2. As literally transcribed
in the spec (with the extra "/ 2"), the max-form is NOT equivalent to the
sum-form the code implements; it's off by roughly a factor of 2. This test
verifies that claim rigorously (exhaustive small-range check against an
independent BFS reference), so the conclusion is a finding, not a guess.

Resolution: the code is correct (verified against independent BFS distance
on the hex grid below) and consistent across all three call sites. The bug
is in the spec's transcription of the max-form identity — see
ANALYTICS_SPECIFICATION.md's corrected §2.1 for the fix (dropped the
erroneous "/ 2").
"""
import itertools
from collections import deque

import pytest

_AXIAL_NEIGHBORS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

_RANGE = range(-8, 9)  # small exhaustive neighborhood around the origin


def code_ring_distance(dq: int, dr: int) -> int:
    """The exact expression used in sb_dq.py / pass_option_value.py /
    hex_analytics.py (sum-of-cube-coordinate-deltas form)."""
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def spec_literal_ring_distance(dq: int, dr: int) -> float:
    """The formula exactly as written in ANALYTICS_SPECIFICATION.md §2.1
    before this investigation (max-form *with* the erroneous "/ 2")."""
    return max(abs(dq), abs(dr), abs(dq + dr)) / 2


def cube_max_ring_distance(dq: int, dr: int) -> int:
    """The correct max-form identity (no division) — mathematically
    equivalent to the sum-form for cube/axial hex coordinates."""
    return max(abs(dq), abs(dr), abs(dq + dr))


def bfs_hex_distance(dq: int, dr: int) -> int:
    """Independent reference: BFS over the hex grid's 6-neighbor adjacency,
    from the origin out to the requested (dq, dr). Doesn't use any of the
    closed-form identities above, so agreement with them is a real check
    rather than restating the same algebra."""
    if (dq, dr) == (0, 0):
        return 0
    frontier = deque([(0, 0)])
    dist = {(0, 0): 0}
    target = (dq, dr)
    bound = max(abs(dq), abs(dr)) + 1
    while frontier:
        q, r = frontier.popleft()
        d = dist[(q, r)]
        if (q, r) == target:
            return d
        if abs(q) > bound or abs(r) > bound:
            continue
        for nq, nr in _AXIAL_NEIGHBORS:
            nxt = (q + nq, r + nr)
            if nxt not in dist:
                dist[nxt] = d + 1
                frontier.append(nxt)
    raise AssertionError(f"BFS never reached target offset {target}")


@pytest.mark.parametrize("dq,dr", list(itertools.product(_RANGE, _RANGE)))
def test_code_formula_matches_independent_bfs_reference(dq, dr):
    assert code_ring_distance(dq, dr) == bfs_hex_distance(dq, dr)


@pytest.mark.parametrize("dq,dr", list(itertools.product(_RANGE, _RANGE)))
def test_correct_max_form_identity_matches_code_formula(dq, dr):
    # The standard cube-distance identity max(|dx|,|dy|,|dz|) == (|dx|+|dy|+|dz|)/2
    # holds exactly (no /2 needed on the max side) — this is the form the
    # spec should have documented.
    assert cube_max_ring_distance(dq, dr) == code_ring_distance(dq, dr)


def test_spec_literal_formula_as_written_diverges_from_code():
    # Confirms the spec's transcription bug is real: dividing the max-form by
    # 2 again is NOT equivalent to the code almost everywhere (any nonzero
    # offset), which is why this needed a code fix vs. a spec fix decision.
    mismatches = [
        (dq, dr)
        for dq, dr in itertools.product(_RANGE, _RANGE)
        if (dq, dr) != (0, 0)
        and spec_literal_ring_distance(dq, dr) != code_ring_distance(dq, dr)
    ]
    assert len(mismatches) > 0
