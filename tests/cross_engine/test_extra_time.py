"""
Extra time in the loaders (review DE-06, 27 November 2026): teams change ends between the halves of extra time too, so
periods 3 and 4 cannot share a sign, and their clocks follow the earlier periods'. Synthetic four-period fixtures only.
"""

import copy

import numpy as np
import pandas as pd
import pytest
from matrix_engine.ingestion.epts_io import _check_periods
from matrix_engine.ingestion.metrica_io import to_single_playing_direction
from matrix_engine.ingestion.second_spectrum_io import _period_time_seconds
from matrix_engine.ingestion.skillcorner_io import _parse_players, tracking_to_wide
from skillcorner_fixture import (
    AWAY_ID,
    HOME_GK_ID,
    HOME_ID,
    make_match_json,
    make_tracking_records,
)


def _metrica(keeper_x_by_period: dict[int, float]):
    rows = [(p, x) for p, x in keeper_x_by_period.items() for _ in range(10)]
    period = [p for p, _ in rows]
    home = pd.DataFrame({"Period": period, "Home_1_x": [x for _, x in rows], "Home_1_y": 0.0,
                         "Home_2_x": [np.sign(x) * 10.0 if np.isfinite(x) else np.nan for _, x in rows], "Home_2_y": 1.0})
    away = pd.DataFrame({"Period": period, "Away_1_x": [-x for _, x in rows], "Away_1_y": 0.0})
    events = pd.DataFrame({"Period": [1, 2, 3, 4], "Start X": [1.0, 2.0, 3.0, 4.0], "Start Y": 0.0, "End X": 0.0, "End Y": 0.0})
    return home, away, events


def test_metrica_orients_each_extra_time_half_from_the_data():
    # Raw: the home keeper at +45 in period 1, the other end in period 2, stays there for extra time's first half, and
    # changes ends again for its second. After the fix the keeper is at +45 in every period.
    h, a, e = to_single_playing_direction(*_metrica({1: 45.0, 2: -45.0, 3: -45.0, 4: 45.0}))
    assert h.groupby("Period")["Home_1_x"].mean().tolist() == [45.0, 45.0, 45.0, 45.0]
    assert e["Start X"].tolist() == [1.0, -2.0, -3.0, 4.0]  # events follow their own period
    assert a.groupby("Period")["Away_1_x"].mean().tolist() == [-45.0, -45.0, -45.0, -45.0]


def test_metrica_fails_loudly_when_an_extra_time_direction_cannot_be_read():
    with pytest.raises(ValueError, match="period 3"):
        to_single_playing_direction(*_metrica({1: 45.0, 2: -45.0, 3: np.nan, 4: 45.0}))


def test_the_clock_runs_through_extra_time():
    periods = {1: (1, 100), 2: (101, 200), 3: (201, 250), 4: (251, 300)}
    assert _period_time_seconds(150, 2, periods, 10.0) == 14.9  # unchanged for the second half (the old formula)
    assert _period_time_seconds(201, 3, periods, 10.0) == 20.0
    assert _period_time_seconds(260, 4, periods, 10.0) == 25.9
    with pytest.raises(ValueError):
        _period_time_seconds(320, 5, periods, 10.0)


def test_skillcorner_reads_the_direction_of_every_period():
    meta = make_match_json(home_attacks_right_in_h1=False)
    hp = _parse_players([p for p in meta["players"] if p["team_id"] == HOME_ID])
    ap = _parse_players([p for p in meta["players"] if p["team_id"] == AWAY_ID])
    base = make_tracking_records(n_p1_frames=3, n_p2_frames=3)
    extra = []
    for k, (period, sign) in enumerate(((3, 1), (4, -1))):
        for r in base[:3]:
            r = copy.deepcopy(r)
            r["frame"], r["period"] = 100000 + 10 * k + r["frame"], period
            for pd_ in r["player_data"]:
                pd_["x"] *= sign  # extra time's first half as the first half, its second as the second
            extra.append(r)
    signs = {1: -1, 2: 1, 3: -1, 4: 1}
    home, _ = tracking_to_wide(base + extra, hp, ap, h1_home_sign=-1, period_signs=signs)
    gk = [c for c in home.columns if c.endswith("_x")]
    keeper = next(c for c in gk if np.allclose(home.loc[home["Period"] == 1, c].abs(), 40.0))
    by_period = home.groupby("Period")[keeper].mean()
    assert by_period.nunique() == 1, by_period.to_dict()  # one end in all four periods
    old, _ = tracking_to_wide(base + extra, hp, ap, h1_home_sign=-1)  # the old rule: every later period as the second
    assert old.groupby("Period")[keeper].mean().nunique() == 2  # extra time's first half was mirrored
    assert HOME_GK_ID


def test_epts_needs_both_halves_and_refuses_extra_time():
    with pytest.raises(ValueError):
        _check_periods({1: (1, 100)})
    with pytest.raises(NotImplementedError):
        _check_periods({1: (1, 100), 2: (101, 200), 3: (201, 250)})
    _check_periods({1: (1, 100), 2: (101, 200)})
