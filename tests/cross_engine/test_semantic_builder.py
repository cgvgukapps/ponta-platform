"""semantic_builder on a small synthetic match: schema, determinism, privacy, not-scored rules."""

import numpy as np
import pandas as pd
import pytest
from matrix_engine.parity import ParityInputs, classify
from plaiq_engine.match_analysis import MatchAnalysis
from plaiq_engine.semantic_builder import (
    SEMANTIC_COLUMNS,
    SemanticBuilder,
    dq_band,
    nested_record,
    pitch_zone,
)
from plaiq_engine.teams import TeamRegistry
from synthetic_match import FPS, synthetic_match


@pytest.fixture(scope="module")
def built():
    analysis = MatchAnalysis(synthetic_match())
    parity = classify(ParityInputs(provider="synthetic", mode="full_tracking", sample_rate_hz=FPS, teams_tracked=("away", "home"),
                                   player_dropout=0.0, has_events=True, has_shot_events=False))
    builder = SemanticBuilder(analysis, parity, teams=TeamRegistry.for_reference())
    return analysis, builder, builder.semantic_events()


def test_schema_columns_and_versions(built):
    _, _, sem = built
    assert list(sem.columns) == list(SEMANTIC_COLUMNS)
    assert sem["schema_version"].eq("2.2").all()  # ... 2.0 opaque ids; 2.1 shots, turnovers, runs; 2.2 near-ball share
    assert sem["engine_version"].str.startswith("plaiq-").all()
    assert sem["match_id"].nunique() == 1 and sem["match_id"].str.fullmatch(r"m1_[0-9a-f]{12}").all()
    assert sem["event_id"].is_unique


def test_deterministic(built):
    _, builder, sem = built
    pd.testing.assert_frame_equal(sem, builder.semantic_events())


def test_no_names_and_no_coordinate_arrays(built):
    _, _, sem = built
    refs = pd.concat([sem[c] for c in ("actor_player_ref", "target_player_ref", "best_option_player_ref", "inferred_intended_ref")]).dropna()
    assert refs.str.fullmatch(r"(home|away)_\w+").all()
    assert not sem.astype(str).apply(lambda s: s.str.contains("Player ")).any().any()
    for col in sem.columns:
        assert not sem[col].map(lambda v: isinstance(v, (list, tuple, np.ndarray))).any(), col
    assert not {"x", "y", "Start X", "End X"} & set(sem.columns)


def test_goalkeeper_and_set_piece_are_not_scored(built):
    _, builder, sem = built
    assert "home_1" in builder.goalkeepers
    gk = sem.loc[(sem["actor_player_ref"] == "home_1") & (sem["action_type"] == "pass")]
    assert len(gk) == 1 and gk["not_scored"].all() and gk["dq_score"].isna().all() and gk["dq_band"].eq("not_scored").all()
    sp = sem.loc[(sem["event_idx"] == 1) & (sem["action_type"] == "pass")]
    assert len(sp) == 1
    assert sp["not_scored"].all() and sp["not_scored_reason"].str.startswith("set piece").all()


def test_physiological_context_is_null_not_zero(built):
    _, _, sem = built
    assert sem["physiological_context"].isna().all()
    assert sem["parity_physiological"].eq("blocked").all()


def test_scores_equal_the_engine(built):
    analysis, _, sem = built
    pov = analysis.event_pov()
    scored = sem.loc[(sem["action_type"].isin(["pass", "carry"])) & sem["dq_score"].notna()]
    assert len(scored) >= 1
    for _, r in scored.iterrows():
        team, jersey = r["actor_player_ref"].split("_", 1)
        hit = pov.loc[(pov["event_idx"] == r["event_idx"]) & (pov["team"].str.lower() == team) & (pov["jersey"].astype(str) == jersey)]
        assert abs(float(hit["decision_quality"].iloc[0]) - r["dq_score"]) < 1e-4
        assert r["dq_band"] == dq_band(r["dq_score"])


def test_defensive_action_and_zones(built):
    _, _, sem = built
    d = sem.loc[sem["action_type"] == "defensive_action"]
    assert len(d) == 1 and d["action_outcome"].iloc[0] == "won" and d["actor_player_ref"].iloc[0] == "away_5"
    assert pitch_zone(30.0, 0.0, "home") == "centre_final_third"
    assert pitch_zone(30.0, 0.0, "away") == "centre_defensive_third"
    assert pitch_zone(0.0, 25.0, "home") == "left_wing_middle_third"


def test_bands_are_the_engines():
    assert [dq_band(v) for v in (0.2, 0.55, 0.79, 0.8, 0.95, 1.3)] == [
        "low_gain", "partial_gain", "partial_gain", "most_gain", "full_gain", "full_gain"]
    assert dq_band(None) is None


def test_nested_record_shape(built):
    _, _, sem = built
    rec = nested_record(sem.iloc[0].to_dict())
    assert set(rec) >= {"actor", "action", "spatial_intelligence", "physiological_context", "quality_gate"}


def test_trajectories_have_refs_and_no_physiological_columns(built):
    _, builder, _ = built
    tr = builder.trajectories()
    assert {"frame", "time_s", "player_ref", "x", "y", "speed", "source_attribution"} <= set(tr.columns)
    assert not {"heart_rate", "hr", "load"} & set(tr.columns)
    assert tr["player_ref"].str.fullmatch(r"(home|away)_\w+|ball").all()


def test_context_columns_are_null_without_context(built):
    _, _, sem = built
    for c in ("possession_chain_id", "possession_source", "score_diff", "red_card_diff"):
        assert sem[c].isna().all(), c


def test_context_columns_from_p1(built):
    from plaiq_engine.context.possession import PossessionPackage
    from plaiq_engine.context.registry import Registry

    analysis, builder, plain = built
    ctx = Registry([PossessionPackage()]).run(analysis.match)
    sem = SemanticBuilder(analysis, builder.parity, context=ctx, teams=TeamRegistry.for_reference()).semantic_events()
    base_cols = [c for c in SEMANTIC_COLUMNS if c not in ("possession_chain_id", "possession_source", "score_diff", "red_card_diff")]
    pd.testing.assert_frame_equal(sem[base_cols], plain[base_cols])  # existing columns unchanged
    assert sem["possession_source"].dropna().isin(["observed", "inferred", "events", "unknown"]).all()
    assert sem["possession_chain_id"].dropna().str.fullmatch(r"m1_[0-9a-f]{12}:ch\d{4}").all()
    assert sem["score_diff"].dropna().eq(0).all()  # no goals in the synthetic match
    rec = nested_record(sem.iloc[0].to_dict())
    assert set(rec["match_context"]) == {"possession_chain_id", "possession_source", "score_diff", "red_card_diff"}


def test_context_columns_and_states_from_every_package(built):
    """Schema 1.7: the Stage B and C context and the episode on each row; earlier columns unchanged; the episode table."""
    import sys
    from pathlib import Path

    from plaiq_engine.citations import check_quoted_numbers, resolve_state_citation
    from plaiq_engine.semantic_builder import CONTEXT_COLUMNS

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    from context_golden import registry

    analysis, builder, plain = built
    ctx = registry().run(analysis.match, extras={"analysis": analysis})
    b = SemanticBuilder(analysis, builder.parity, context=ctx, teams=TeamRegistry.for_reference())
    sem = b.semantic_events()
    p1 = ("possession_chain_id", "possession_source", "score_diff", "red_card_diff")
    base_cols = [c for c in SEMANTIC_COLUMNS if c not in CONTEXT_COLUMNS and c not in p1]  # team keys are context-independent
    pd.testing.assert_frame_equal(sem[base_cols], plain[base_cols])  # existing columns unchanged
    assert sem["context_versions"].notna().all()
    assert sem["state_id"].dropna().str.fullmatch(r"m1_[0-9a-f]{12}:\d:\d{4}").all()
    assert sem["zone_third"].dropna().isin(["defensive", "middle", "attacking"]).all()
    assert not [c for c in sem.columns if c.endswith(("_x", "_y", "_cell"))]
    st = b.semantic_states(sem)
    assert len(st) >= 1 and st["state_id"].is_unique
    for r in st.to_dict("records"):
        assert check_quoted_numbers(r["summary"], r) == [], r["summary"]
        assert resolve_state_citation(st, r["match_id"], r["state_id"], r["start_s"], r["end_s"]).ok
    assert set(sem["state_id"].dropna()) <= set(st["state_id"])
    for frame in (sem, st):  # schema 2.0: no provider id in any value of either AI table
        assert not any(b.provider_match_id in str(v) for v in frame.to_numpy().ravel())
    # Schema 2.1: one row per P6 run, typed; roles from P2 where no roster; movement values in their own table.
    p6 = ctx.outputs["p6"].segments["runs"]
    runs = sem.loc[sem["action_type"] == "run"]
    assert len(runs) == len(p6) and runs["event_idx"].isna().all() and runs["event_id"].is_unique
    assert runs["run_type"].notna().all() or not len(runs)
    assert "off_ball_run" not in set(sem["action_type"])
    mv = b.movement_values()
    assert len(mv) and set(mv["source_event_id"]) <= set(sem["event_id"]) and mv["event_id"].is_unique
    assert {"shots", "ended_in_shot"} <= set(st.columns) and st["shots"].eq(0).all() and not st["ended_in_shot"].any()
    # Schema 1.9 (review AI-10): an episode's end ids are semantic events inside it, so each resolves.
    from plaiq_engine.citations import resolve_event_citation

    ts_of = sem.set_index("event_id")["timestamp_s"]
    for col in ("first_event_id", "last_event_id"):
        ids = st[col].dropna()
        assert len(ids) >= 1
        for i in ids:
            assert resolve_event_citation(sem, b.match_id, i, float(ts_of[i])).ok
    inside = sem.loc[sem["state_id"].notna()]
    assert set(st["first_event_id"].dropna()) <= set(inside["event_id"])


def test_team_keys_are_opaque_and_stable(built):
    """Schema 1.8: every row carries pseudonymous team keys; an anonymous side gets a key for its match only."""
    from plaiq_engine.teams import TeamRegistry

    analysis, builder, sem = built
    assert sem["actor_team_key"].str.fullmatch(r"t1_[0-9a-f]{12}").all()
    assert (sem["actor_team_key"] != sem["opponent_team_key"]).all()
    reg = TeamRegistry.for_reference()
    assert reg.key("skillcorner", "1802") == reg.key("skillcorner", "1802") != reg.key("skillcorner", "866")
    assert reg.key("skillcorner", "1802") != TeamRegistry(secret="tenant-a-secret-0123").key("skillcorner", "1802")
    assert reg.key("synthetic", None, match_id="1", side="home") != reg.key("synthetic", None, match_id="2", side="home")
    reg.merge("idsse", "DFL-CLU-000008", reg.key("skillcorner", "1802"))
    assert reg.key("idsse", "DFL-CLU-000008") == reg.key("skillcorner", "1802")
    assert "1802" not in reg.key("skillcorner", "1802")


def test_pass_completion_probability_on_passes_only(built):
    analysis, _, sem = built
    passes = sem.loc[sem["action_type"] == "pass"]
    scored = passes.loc[passes["pass_completion_probability"].notna()]
    assert len(scored) == 2  # the open-play passes; the set-piece pass is not scored
    assert scored["pass_completion_probability"].between(0, 1).all()
    assert scored["pass_completion_status"].eq("estimated").all()
    assert passes["pass_completion_parameters"].eq("default-idsse-2026-11-19").all()  # no set fitted for this provider
    sp = passes.loc[passes["event_idx"] == 1].iloc[0]
    assert sp["pass_completion_probability"] is None or pd.isna(sp["pass_completion_probability"])
    assert sp["pass_completion_status"].startswith("set piece")
    others = sem.loc[sem["action_type"] != "pass", "pass_completion_probability"]
    assert others.isna().all()
    pc = analysis.pass_completion.set_index("event_idx")
    for r in scored.itertuples():
        assert r.pass_completion_probability == pytest.approx(pc.at[r.event_idx, "pass_completion_probability"], abs=1e-4)


def test_pass_completion_ignores_the_pass_height_flag():
    """SkillCorner logs High Pass on completed passes only, so the outcome model must not read it."""
    from dataclasses import replace

    from matrix_engine.ingestion.second_spectrum_io import HIGH_PASS
    m = replace(synthetic_match(), provider="skillcorner")
    ev = m.events.copy()
    ev[HIGH_PASS] = pd.array([pd.NA, True, True, pd.NA, True, pd.NA], dtype="boolean")
    with_flag = MatchAnalysis(replace(m, events=ev)).pass_completion
    without = MatchAnalysis(m).pass_completion
    pd.testing.assert_frame_equal(with_flag, without)


def test_pass_completion_providers_agree_across_engines():
    from matrix_engine.parity import PASS_COMPLETION_FITTED_PROVIDERS
    from plaiq_engine.analytics.pass_probability import PARAMETER_SETS
    assert set(PASS_COMPLETION_FITTED_PROVIDERS) == set(PARAMETER_SETS) - {"default"}


def test_decision_quality_v2_beside_v1(built):
    analysis, _, sem = built
    passes = sem.loc[sem["action_type"] == "pass"].set_index("event_idx")
    # The open-play pass is scored, 0 to 1, labelled estimated, with no band; v1's columns are untouched.
    r = passes.loc[2]
    assert 0.0 <= r["dq_v2_score"] <= 1.0
    assert r["dq_v2_status"] == "estimated" and r["dq_v2_version"] == "1.1.0"
    from plaiq_engine.analytics.decision_quality_v2 import dq_v2_band
    assert r["dq_v2_band"] == dq_v2_band(r["dq_v2_score"]) and r["dq_v2_band"] in {"best_option", "near_best", "below_best", "well_below_best"}
    assert r["dq_v2_best_option_ref"].startswith("home_") and r["dq_v2_best_option_ref"] != "home_4"  # never the passer
    assert r["dq_band"] == dq_band(r["dq_score"])
    # Not scored where v1 is not: the set piece and the goalkeeper, each with its reason.
    assert passes.loc[1, "dq_v2_status"].startswith("set piece") and pd.isna(passes.loc[1, "dq_v2_score"])
    assert passes.loc[4, "dq_v2_status"].startswith("goalkeeper") and pd.isna(passes.loc[4, "dq_v2_score"])
    assert sem.loc[sem["action_type"] != "pass", "dq_v2_score"].isna().all()
    # The engine's numbers: the chosen and best options are cells, each the best of 7 around its run.
    d2 = analysis.decision_quality_v2.set_index("event_idx")
    assert r["dq_v2_score"] == pytest.approx(d2.at[2, "dq_v2"], abs=1e-4)
    assert d2.at[2, "dq_v2"] == pytest.approx(d2.at[2, "ev_chosen"] / d2.at[2, "ev_best"])
    assert 0.0 <= d2.at[2, "p_chosen"] <= 1.0 and d2.at[2, "cells"] == 7 * d2.at[2, "options"]
    assert d2.at[2, "chosen_cell"] is not None and d2.at[2, "best_cell"] is not None
    assert "chosen_cell" not in sem.columns and "best_cell" not in sem.columns  # cells are coordinates: never in the AI's rows
    assert bool(d2.at[2, "best_is_chosen"]) == (d2.at[2, "dq_v2"] >= 1.0 - 1e-12)
    rec = nested_record(passes.reset_index().loc[lambda d: d["event_idx"] == 2].iloc[0].to_dict())
    assert set(rec["spatial_intelligence"]["decision_quality_v2"]) == {"score", "band", "status", "best_option_player_ref", "version"}


def test_decision_quality_v2_ignores_the_outcome():
    """One formula for completed and failed passes: the same pass, logged as lost, scores the same and gets a row."""
    from dataclasses import replace

    m = synthetic_match()
    ev = m.events.copy()
    ev.loc[2, "Type"] = "BALL LOST"
    lost = MatchAnalysis(replace(m, events=ev))
    done = MatchAnalysis(m)
    assert lost.decision_quality_v2.at[1, "event_idx"] == 2
    assert lost.decision_quality_v2.at[1, "dq_v2"] == pytest.approx(done.decision_quality_v2.at[1, "dq_v2"], abs=1e-12)
    parity = classify(ParityInputs(provider="synthetic", mode="full_tracking", sample_rate_hz=FPS, teams_tracked=("away", "home"),
                                   player_dropout=0.0, has_events=True, has_shot_events=False))
    sem = SemanticBuilder(lost, parity, teams=TeamRegistry.for_reference()).semantic_events()
    row = sem.loc[(sem["event_idx"] == 2) & (sem["action_type"] == "pass")]
    assert len(row) == 1 and row["action_outcome"].iloc[0] == "lost" and row["dq_v2_status"].iloc[0] == "estimated"


def test_a_lost_pass_never_carries_a_version_1_score():
    """Version 1's failed-pass score stays unreported (review SM-02): null score, band not_scored, a reason."""
    from dataclasses import replace

    m = synthetic_match()
    ev = m.events.copy()
    ev.loc[2, "Type"] = "BALL LOST"
    analysis = MatchAnalysis(replace(m, events=ev))
    parity = classify(ParityInputs(provider="synthetic", mode="full_tracking", sample_rate_hz=FPS, teams_tracked=("away", "home"),
                                   player_dropout=0.0, has_events=True, has_shot_events=False))
    sem = SemanticBuilder(analysis, parity, teams=TeamRegistry.for_reference()).semantic_events()
    lost = sem.loc[sem["action_outcome"] == "lost"]
    assert len(lost) >= 1
    assert lost["dq_score"].isna().all() and lost["dq_band"].eq("not_scored").all() and lost["not_scored"].all()
    assert lost["not_scored_reason"].str.startswith("lost pass").all()


def test_gain_v2_beside_version_1(built):
    analysis, _, sem = built
    passes = sem.loc[sem["action_type"] == "pass"].set_index("event_idx")
    r = passes.loc[2]
    assert 0.0 <= r["gain_v2_score"] <= 1.5 and r["gain_v2_status"] == "estimated" and r["gain_v2_version"] == "2.0.0"
    assert r["gain_v2_band"] == dq_band(r["gain_v2_score"])  # version 1's cut-offs and names
    assert passes.loc[1, "gain_v2_status"].startswith("set piece") and pd.isna(passes.loc[1, "gain_v2_score"])
    assert passes.loc[4, "gain_v2_status"].startswith("goalkeeper") and pd.isna(passes.loc[4, "gain_v2_score"])
    assert sem.loc[sem["action_type"] != "pass", "gain_v2_score"].isna().all()
    # Gain = Choice x execution, from one model.
    g = analysis.gain_v2.set_index("event_idx").loc[2]
    d = analysis.decision_quality_v2.set_index("event_idx").loc[2]
    assert g["gain_v2"] == pytest.approx(min(1.5, d["dq_v2"] * g["execution"]))
    rec = nested_record(passes.reset_index().loc[lambda x: x["event_idx"] == 2].iloc[0].to_dict())
    assert set(rec["spatial_intelligence"]["gain_v2"]) == {"score", "band", "status", "version"}


def test_gain_v2_is_zero_for_a_failed_pass():
    from dataclasses import replace

    m = synthetic_match()
    ev = m.events.copy()
    ev.loc[2, "Type"] = "BALL LOST"
    g = MatchAnalysis(replace(m, events=ev)).gain_v2.set_index("event_idx")
    assert g.at[2, "gain_v2"] == 0.0 and g.at[2, "realised_value"] == 0.0


def test_estimated_positions_are_not_counted_as_observed():
    """Where the provider flags estimates, players_observed counts observations only and trajectories carry the flag
    (schema 1.9, review AI-03, DE-05); the context Data check degrades positional entries (GC-01)."""
    from matrix_engine.match import with_observed
    from plaiq_engine.context.possession import PossessionPackage
    from plaiq_engine.context.registry import Registry
    from plaiq_engine.context.structure import StructurePackage

    m = synthetic_match()
    away = sorted({p for p in m.tracking["player_id"] if p.startswith("away_")})
    est = pd.DataFrame([(f, p) for f in m.frames["frame"] for p in away], columns=["frame", "player_id"])  # away all estimated
    m2 = with_observed(m, est)
    analysis = MatchAnalysis(m2)
    parity = classify(ParityInputs(provider="synthetic", mode="full_tracking", sample_rate_hz=FPS, teams_tracked=("away", "home"),
                                   player_dropout=0.0, has_events=True, has_shot_events=False))
    b = SemanticBuilder(analysis, parity, teams=TeamRegistry.for_reference())
    sem = b.semantic_events()
    assert sem["players_observed"].dropna().eq(11).all()  # the home eleven only
    tr = b.trajectories()
    assert tr.loc[tr["team"] == "away", "observed"].eq(False).all() and tr.loc[tr["team"] == "home", "observed"].eq(True).all()
    assert tr.loc[tr["team"] == "ball", "observed"].isna().all()
    res = Registry([PossessionPackage(), StructurePackage()]).run(m2)
    shape = next(e for e in res.entries if e.package == "p2" and e.metric == "team_shape")
    assert shape.state != "full" and "estimated by the provider" in shape.reason


# -- schema 2.0: identity fails closed, no provider ids (review backlog B) ------------------------------------------


def test_the_registry_fails_closed(monkeypatch):
    from plaiq_engine.teams import REFERENCE_SECRET, PseudonymError, TeamRegistry

    with pytest.raises(TypeError):
        SemanticBuilder(MatchAnalysis(synthetic_match()), None)  # no teams: refused
    for bad in ("", "short"):
        with pytest.raises(PseudonymError):
            TeamRegistry(secret=bad)
    monkeypatch.setenv("PONTA_ENV", "production")
    with pytest.raises(PseudonymError):
        TeamRegistry.for_reference()
    assert TeamRegistry(secret="a-tenant-secret-of-length").secret != REFERENCE_SECRET  # a real secret still works


def test_a_name_in_the_feed_fails_the_build():
    from dataclasses import replace

    from plaiq_engine.teams import PseudonymError, TeamRegistry

    m = synthetic_match()
    ev = m.events.copy()
    ev.loc[2, "From"] = "Jane Example"
    b = SemanticBuilder(MatchAnalysis(replace(m, events=ev)), _parity(), teams=TeamRegistry.for_reference())
    with pytest.raises(PseudonymError):
        b.semantic_events()


def _parity():
    return classify(ParityInputs(provider="synthetic", mode="full_tracking", sample_rate_hz=FPS, teams_tracked=("away", "home"),
                                 player_dropout=0.0, has_events=True, has_shot_events=False))


def test_no_provider_id_in_any_ai_facing_value():
    """A provider match id, team id or competition id never appears in a value (review AI-04: the test checks values)."""
    from dataclasses import replace

    from plaiq_engine.teams import TeamRegistry

    m = replace(synthetic_match(), provider="skillcorner", match_id="1886347", teams={"home": "4177", "away": "866"},
                competition="61", season="2024/2025")
    b = SemanticBuilder(MatchAnalysis(m), _parity(), teams=TeamRegistry.for_reference())
    sem = b.semantic_events()
    text = [str(v) for v in sem.to_numpy().ravel()]
    for token in ("1886347", "4177", "skillcorner_", "skillcorner:", "2024/2025"):
        assert not any(token in v for v in text), token
    assert sem["competition"].str.fullmatch(r"c1_[0-9a-f]{10}").all() and sem["season"].str.fullmatch(r"s1_[0-9a-f]{10}").all()
    prov = b.provenance()  # the provider's ids are kept apart, for audits only
    assert prov["provider_match_id"].eq("1886347").all() and set(prov["event_id"]) >= set(sem["event_id"].str.split("_run_").str[0])


def test_event_ids_survive_an_added_event():
    """An event added elsewhere in the match changes no other id (review DE-03: the id was the row position)."""
    from dataclasses import replace

    from plaiq_engine.teams import TeamRegistry

    m = synthetic_match()
    reg = TeamRegistry.for_reference()
    before = SemanticBuilder(MatchAnalysis(m), _parity(), teams=reg)._event_ids()
    ev = m.events.copy()
    extra = ev.iloc[[2]].assign(**{"Type": "CHALLENGE", "Subtype": "AERIAL-LOST", "Start Frame": 70, "End Frame": 70})
    ev2 = pd.concat([ev.iloc[:3], extra, ev.iloc[3:]], ignore_index=True)
    after = SemanticBuilder(MatchAnalysis(replace(m, events=ev2)), _parity(), teams=reg)._event_ids()
    old = [before[i] for i in range(len(ev))]
    new = [after[i] for i in range(len(ev2)) if i != 3]
    assert old == new and len(set(after.values())) == len(ev2)


def test_nested_record_carries_every_column(built):
    """Every flat column reaches the nested view the AI reads, bar a short, named list (review SE-15)."""
    _, _, sem = built
    rec = nested_record(sem.iloc[0].to_dict())

    def leaves(d, prefix=""):
        out = {}
        for k, v in d.items():
            out.update(leaves(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
        return out

    flat = set(leaves(rec))
    nested_names = {k.rsplit(".", 1)[-1] for k in flat}
    aliases = {"actor_player_ref": "player_ref", "actor_role": "role", "actor_role_source": "role_source", "action_detail": "detail", "action_type": "type", "action_outcome": "outcome",
               "actor_team": "team", "actor_team_key": "team_key", "parity_spatial": "spatial_intelligence",
               "parity_physiological": "physiological_context", "pass_completion_probability": "probability",
               "pass_completion_status": "status", "pass_completion_parameters": "parameters",
               "dq_v2_score": "score", "dq_v2_band": "band", "dq_v2_status": "status", "dq_v2_best_option_ref": "best_option_player_ref",
               "dq_v2_version": "version", "gain_v2_score": "score", "gain_v2_band": "band", "gain_v2_status": "status",
               "gain_v2_version": "version"}
    excluded = {"event_idx"}  # the loader's row position: internal, not for the AI
    missing = [c for c in SEMANTIC_COLUMNS if c not in excluded and c not in nested_names and aliases.get(c) not in nested_names]
    assert not missing, missing


def test_shots_turnovers_set_pieces_fouls_and_cards_get_rows():
    """Schema 2.1 (review AI-06): every shot, every lost ball and the context rows; the end of a half is not a turnover."""
    from dataclasses import replace

    m = synthetic_match()
    ev = m.events.copy()
    base = ev.iloc[5].to_dict()  # the away recovery at frame 230
    extra = [
        {**base, "Type": "SHOT", "Subtype": "HEAD-ON TARGET-GOAL", "Team": "Away", "From": "Player 9", "Start Frame": 240, "End Frame": 240},
        {**base, "Type": "BALL LOST", "Subtype": "THEFT", "Team": "Home", "From": "Player 6", "Start Frame": 150, "End Frame": 150},
        {**base, "Type": "BALL LOST", "Subtype": "END HALF", "Team": "Home", "From": "Player 2", "Start Frame": 249, "End Frame": 249},
        {**base, "Type": "FAULT RECEIVED", "Subtype": np.nan, "Team": "Home", "From": "Player 4", "Start Frame": 100, "End Frame": 100},
        {**base, "Type": "CARD", "Subtype": "YELLOW", "Team": "Away", "From": "Player 7", "Start Frame": 101, "End Frame": 101},
    ]
    ev = pd.concat([ev, pd.DataFrame(extra)], ignore_index=True)
    sem = SemanticBuilder(MatchAnalysis(replace(m, events=ev)), _parity(), teams=TeamRegistry.for_reference()).semantic_events()
    by = sem.set_index("action_type")
    shot = sem.loc[sem["action_type"] == "shot"].iloc[0]
    assert shot["action_outcome"] == "goal" and shot["action_detail"] == "head" and shot["actor_player_ref"] == "away_9"
    assert shot["dq_score"] is None or pd.isna(shot["dq_score"])  # no decision or shot-quality score on a shot
    t = sem.loc[sem["action_type"] == "turnover"]
    assert len(t) == 1 and t["action_detail"].iloc[0] == "dispossessed" and t["actor_player_ref"].iloc[0] == "home_6"
    sp = sem.loc[sem["action_type"] == "set_piece"]
    assert len(sp) == 1 and sp["action_detail"].iloc[0] == "throw_in"
    assert sem.loc[sem["action_type"] == "card", "action_detail"].tolist() == ["yellow"]
    assert sem.loc[sem["action_type"] == "foul_won", "actor_player_ref"].tolist() == ["home_4"]
    assert "home_2" not in set(sem.loc[sem["action_type"] == "turnover", "actor_player_ref"])  # END HALF: no row
    assert sem["timestamp_s"].is_monotonic_increasing and len(by)


def test_every_failed_pass_gets_a_row_scored_or_not():
    """Schema 2.1: a failed pass no model scores (here a goalkeeper's) still gets a lost-pass row (before, it vanished)."""
    from dataclasses import replace

    m = synthetic_match()
    ev = m.events.copy()
    ev.loc[4, "Type"] = "BALL LOST"  # the goalkeeper's pass, which no model scores
    sem = SemanticBuilder(MatchAnalysis(replace(m, events=ev)), _parity(), teams=TeamRegistry.for_reference()).semantic_events()
    row = sem.loc[sem["event_idx"] == 4]
    assert len(row) == 1 and row["action_type"].iloc[0] == "pass" and row["action_outcome"].iloc[0] == "lost"
    assert row["dq_band"].iloc[0] == "not_scored"


def test_runs_and_estimated_roles_from_the_context():
    """Schema 2.1: P6's runs become one row each, and P2's main role fills actor_role where no roster gives one."""
    from types import SimpleNamespace

    m = synthetic_match()
    frames = pd.DataFrame({"match_id": "synthetic_1", "frame_ts": np.arange(0, 10, 0.2), "native_frame": np.arange(1, 251, 5)})
    runs = pd.DataFrame([{"team": "home", "player_id": "home_9", "period": 1, "start_ts": 2.0, "end_ts": 3.4, "duration_s": 1.4,
                          "in_possession": True, "type": "run_in_behind", "area_end": "in_behind"},
                         {"team": "away", "player_id": "away_4", "period": 1, "start_ts": 6.0, "end_ts": 7.0, "duration_s": 1.0,
                          "in_possession": False, "type": "recovery_run", "area_end": None}])
    roles = pd.DataFrame([{"team": "home", "player_id": "home_9", "role": "ST", "state": "out", "role_s": 300.0, "share": 0.9},
                          {"team": "home", "player_id": "home_4", "role": "CM", "state": "out", "role_s": 300.0, "share": 0.9}])
    ctx = SimpleNamespace(frames=frames, outputs={"p2": SimpleNamespace(segments={"main_roles": roles}),
                                                  "p6": SimpleNamespace(segments={"runs": runs})}, thresholds_version="x")
    sem = SemanticBuilder(MatchAnalysis(m), _parity(), teams=TeamRegistry.for_reference(), context=ctx,
                          roles={"home_4": "Captain's role"}).semantic_events()
    r = sem.loc[sem["action_type"] == "run"].set_index("actor_player_ref")
    assert set(r.index) == {"home_9", "away_4"} and r.at["home_9", "run_type"] == "run_in_behind" and r.at["home_9", "run_area"] == "in_behind"
    assert r.at["home_9", "timestamp_s"] == pytest.approx(2.0, abs=0.05) and r["event_idx"].isna().all()
    assert r.at["home_9", "actor_role"] == "ST" and r.at["home_9", "actor_role_source"] == "estimated"
    h4 = sem.loc[sem["actor_player_ref"] == "home_4"].iloc[0]
    assert h4["actor_role"] == "Captain's role" and h4["actor_role_source"] == "roster"  # the roster wins
