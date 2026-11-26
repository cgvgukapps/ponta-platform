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
    assert pr.loc[pr["data_check"] == "blocked", "value"].isna().all()  # and a Blocked row never carries one (AI-05)
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


def test_ppda_counts_only_the_defending_teams_actions():
    """Duels logged for both players, or for their winner, count only for the team without the ball (AN-01)."""
    typ = np.array(["CHALLENGE", "CHALLENGE", "CHALLENGE", "RECOVERY", "RECOVERY", "FAULT RECEIVED", "FAULT RECEIVED", "PASS", "CHALLENGE"])
    sub = np.array(["TACKLE-WON", "TACKLE-LOST", "AERIAL-WON", "INTERCEPTION", "", "", "", "", "GROUND-WON"])
    team = np.array(["home", "away", "home", "home", "home", "away", "home", "away", "home"])
    poss = np.array(["away", "away", "home", "away", "away", "away", "home", "away", "contested"], dtype=object)
    got = P.defensive_actions(typ, sub, team, poss, "home")
    # home's tackle while away had the ball; not away's own duel, not home's duel on its own ball; home's interception;
    # not a plain recovery; home's foul (away fouled while in possession); not a foul home received; not a duel while
    # the ball was contested (both sides' duels would count).
    assert got.tolist() == [True, False, False, True, False, True, False, False, False]


def test_ppda_is_blocked_for_a_feed_without_defender_actions(match_and_result):
    """A feed outside PPDA_FEEDS (SkillCorner's dynamic events log no defender-side actions) gives no PPDA, Blocked."""
    from dataclasses import replace

    from context_fixture import event

    m, res = match_and_result
    ev = pd.DataFrame([event("PASS", "Away", 10 + i) for i in range(30)] + [event("CHALLENGE", "Home", 50 + i) for i in range(12)])
    pr = P.match_profiles(replace(m, events=ev, provider="skillcorner"), res, TeamRegistry().keys_for(m))
    row = pr.loc[pr["measure"] == "ppda"]
    assert row["value"].isna().all() and row["data_check"].eq("blocked").all()
    assert "skillcorner" not in P.PPDA_FEEDS


def test_chain_measures_are_degraded_without_restarts(match_and_result):
    """No feed, so no restarts: chain starts and ends are not known to be losses and regains (review AN-03)."""
    m, res = match_and_result
    assert res.outputs["p1"].measurements["dead_ball_known"] is False
    pr = P.match_profiles(m, res, TeamRegistry().keys_for(m)).set_index(["team", "measure"])
    for name in P.DEAD_BALL_MEASURES:
        assert pr.loc[("home", name), "data_check"] in ("degraded", "blocked"), name


def test_a_blocked_match_gets_no_verdict_and_degraded_is_disclosed(monkeypatch):
    """compare() refuses a Blocked match row; compare, baseline and citations carry the worst Data check state (AI-05)."""
    pr = _synthetic_profiles()
    cfg = values()
    monkeypatch.setitem(P.MIN_MATCHES, "possession_share", 3)
    high = pr.copy()
    high.loc[high["match_id"] == "mt_a0", ["value", "data_check"]] = [0.95, "blocked"]
    out = P.compare(high, "mt_a0", "t_a", "possession_share", cfg)
    assert out["verdict"] is None and out["data_check"] == "blocked"
    deg = pr.assign(data_check=np.where(pr["match_id"] == "mt_a1", "degraded", "full"))
    assert P.baseline(deg, "t_a", "possession_share")["data_check"] == "degraded"
    assert P.compare(deg, "mt_a0", "t_a", "possession_share", cfg)["data_check"] == "degraded"
    r = resolve_profile_citation(deg, "t_a", "possession_share", [f"mt_a{i}" for i in range(4)])
    assert r.ok and r.data_check == "degraded"


def test_unlicensed_derived_minimums_are_off_by_default():
    """PFF-derived values never govern a comparison unless the switch is on, and every verdict names its source (GC-03)."""
    from plaiq_engine.context.thresholds import THRESHOLDS

    cfg = values()
    assert cfg["pf_allow_unlicensed_derived"] == 0.0
    tagged = {k for k, t in THRESHOLDS.items() if t.licence and not t.licence.startswith("skillcorner")}
    assert tagged and all(k.endswith("_pff") for k in tagged)
    for m in P.MIN_MATCHES_UNLICENSED:
        n, src = P.min_matches(m, cfg)
        assert n == P.MIN_MATCHES.get(m) and (src is None or src.startswith("skillcorner"))  # no licensed value: no verdict
        n, src = P.min_matches(m, values({"pf_allow_unlicensed_derived": 1.0}))
        assert n == max(P.MIN_MATCHES.get(m, 0), P.MIN_MATCHES_UNLICENSED[m])
    pr = _synthetic_profiles()
    out = P.compare(pr, "mt_a0", "t_a", "field_tilt", cfg)
    assert "minimum_source" in out or out.get("reason")


def _noise_profiles(rng, teams=12, per_team=6, icc=0.3, measure="possession_share"):
    sd_b, sd_w = np.sqrt(icc), np.sqrt(1 - icc)
    rows = []
    for t in range(teams):
        centre = rng.normal(0, sd_b)
        for i in range(per_team):
            rows.append({"match_id": f"m{t}_{i}", "team": "home", "team_key": f"t{t}", "opp_team_key": "tx", "competition": "c", "season": "s",
                         "measure": measure, "value": centre + rng.normal(0, sd_w), "denominator": "", "sample_minutes": 90, "sample_n": 0,
                         "label": "estimated", "basis": "common", "source": "", "data_check": "full", "has_feed": True, "versions": "v"})
    return pd.DataFrame(rows, columns=list(P.PROFILE_COLUMNS))


def test_the_verdict_fires_at_its_stated_false_alarm_rate(monkeypatch):
    """Under pure noise "above" or "below your usual" comes up about pf_verdict_false_alarm of the time (review ST-05)."""
    monkeypatch.setitem(P.MIN_MATCHES, "possession_share", 3)
    cfg = values()
    rng = np.random.default_rng(5)
    flags = total = 0
    for _ in range(60):
        pr = _noise_profiles(rng)
        for t in range(12):
            out = P.compare(pr, f"m{t}_0", f"t{t}", "possession_share", cfg)
            total += 1
            flags += out["verdict"] in ("above your usual", "below your usual")
            assert out["usual_range"][0] < out["usual_range"][1] and out["false_alarm_rate"] == cfg["pf_verdict_false_alarm"]
    rate = flags / total
    assert 0.06 <= rate <= 0.14, rate  # 10% stated; the old 1 SD rule gave about a third


def test_shrinkage_weight_is_the_method_of_moments():
    """The set's weight is within over between variance with between from (MSB - MSW) / n0 (review ST-06)."""
    rng = np.random.default_rng(9)
    ks = []
    for _ in range(200):
        b = P.baseline(_noise_profiles(rng, teams=12, per_team=3, icc=0.1), "t0", "possession_share")
        ks.append(np.inf if b["set_weight"] is None else b["set_weight"])  # no between-team spread: the set mean
    # True weight (1 - ICC) / ICC = 9; the old estimator (variance of the team means) gave a median near 2.
    assert 6.0 <= float(np.median(ks)) <= 20.0, np.median(ks)
