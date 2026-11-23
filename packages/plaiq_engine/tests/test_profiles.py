"""Stage E: team match profiles, baselines and pattern citations (fictional players and teams, no sample data)."""

import json

import numpy as np
import pandas as pd
import pytest
from context_fixture import eleven_a_side
from plaiq_engine import profiles as P
from plaiq_engine.citations import resolve_profile_citation
from plaiq_engine.context.fifa_view import FifaViewPackage
from plaiq_engine.context.line_breaks import LineBreakPackage
from plaiq_engine.context.movement import MovementPackage
from plaiq_engine.context.phase import PhasePackage
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.space_value import SpaceValuePackage
from plaiq_engine.context.states import StatesPackage
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.context.thresholds import values
from plaiq_engine.context.zones import ZonesPackage
from plaiq_engine.teams import TeamRegistry


@pytest.fixture(scope="module")
def match_and_result():
    m = eleven_a_side(seconds=60.0)
    res = Registry([PossessionPackage(), StructurePackage(), ZonesPackage(), PhasePackage(), LineBreakPackage(), MovementPackage(),
                    SpaceValuePackage(), FifaViewPackage(), StatesPackage()]).run(m)
    return m, res


def test_profile_has_every_catalogue_measure_for_both_teams(match_and_result):
    m, res = match_and_result
    keys = TeamRegistry().keys_for(m)
    pr = P.match_profiles(m, res, keys)
    assert list(pr.columns) == list(P.PROFILE_COLUMNS)
    assert len(pr) == 2 * len(P.CATALOGUE)
    for team in ("home", "away"):
        assert set(pr.loc[pr["team"] == team, "measure"]) == set(P.MEASURES)
    assert pr["team_key"].str.fullmatch(r"t_[0-9a-f]{12}").all()
    assert pr["basis"].isin(["published", "common", "ponta"]).all() and pr["label"].isin(["observed", "estimated"]).all()
    assert pr["data_check"].isin(["full", "degraded", "blocked"]).all()
    assert pr.loc[pr["value"].isna(), "data_check"].eq("blocked").all()  # no value is never shown as a usable value
    shares = pr.loc[pr["measure"].str.contains("share") & pr["value"].notna(), "value"]
    assert shares.between(0, 1).all()
    assert not pr.loc[pr["measure"] == "ppda", "value"].notna().any()  # no event feed: PPDA is unknown, not zero
    v = json.loads(pr["versions"].iloc[0])
    assert v["profiles"] == P.PROFILE_VERSION and "thresholds" in v
    assert not [c for c in pr.columns if c.endswith(("_x", "_y", "_cell")) or "name" in c]


def _synthetic_profiles(versions="v1"):
    rng = np.random.default_rng(7)
    rows = []
    for t, centre in (("t_a", 0.30), ("t_b", 0.40), ("t_c", 0.50)):
        for i in range(6):
            rows.append({"match_id": f"m{t}{i}", "team": "home", "team_key": t, "opp_team_key": "t_x", "competition": "c", "season": "s",
                         "measure": "possession_share", "value": centre + rng.normal(0, 0.02), "denominator": "share", "sample_minutes": 90,
                         "sample_n": 0, "label": "estimated", "basis": "common", "source": "", "data_check": "full", "has_feed": True,
                         "versions": versions})
    return pd.DataFrame(rows, columns=list(P.PROFILE_COLUMNS))


def test_baseline_shrinks_and_refuses_mixed_versions():
    pr = _synthetic_profiles()
    b = P.baseline(pr, "t_a", "possession_share")
    assert b["team_matches"] == 6 and b["set_teams"] == 3
    assert b["team_mean"] <= b["shrunk"] <= b["set_mean"]  # pulled towards the set, not past it
    mixed = pd.concat([pr, _synthetic_profiles("v2").assign(match_id=lambda d: d["match_id"] + "z")])
    assert "refused" in P.baseline(mixed, "t_a", "possession_share")
    blocked = pr.assign(data_check=np.where(pr["match_id"] == "mt_a0", "blocked", "full"))
    assert P.baseline(blocked, "t_a", "possession_share")["team_matches"] == 5


def test_compare_gives_a_verdict_only_above_the_measured_minimum(monkeypatch):
    pr = _synthetic_profiles()
    cfg = values()
    assert P.compare(pr, "mt_a0", "t_a", "possession_share", cfg)["verdict"] is None  # minimum not measured yet
    monkeypatch.setitem(P.MIN_MATCHES, "possession_share", 10)
    assert P.compare(pr, "mt_a0", "t_a", "possession_share", cfg)["verdict"] == "not enough matches yet"
    monkeypatch.setitem(P.MIN_MATCHES, "possession_share", 3)
    out = P.compare(pr, "mt_a0", "t_a", "possession_share", cfg)
    assert out["verdict"] in ("above your usual", "about your usual", "below your usual")
    high = pr.copy()
    high.loc[high["match_id"] == "mt_a0", "value"] = 0.9
    assert P.compare(high, "mt_a0", "t_a", "possession_share", cfg)["verdict"] == "above your usual"


def test_profile_citations(monkeypatch):
    pr = _synthetic_profiles()
    ids = [f"mt_a{i}" for i in range(4)]
    assert not resolve_profile_citation(pr, "t_a", "possession_share", ids).ok  # minimum not measured yet
    monkeypatch.setitem(P.MIN_MATCHES, "possession_share", 3)
    assert resolve_profile_citation(pr, "t_a", "possession_share", ids).ok
    assert not resolve_profile_citation(pr, "t_a", "possession_share", ids[:2]).ok
    assert not resolve_profile_citation(pr, "t_a", "not_a_measure", ids).ok
    assert not resolve_profile_citation(pr, "t_b", "possession_share", ids).ok  # those matches are not t_b's
