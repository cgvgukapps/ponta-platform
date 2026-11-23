"""semantic_builder on a small synthetic match: schema, determinism, privacy, not-scored rules."""

import numpy as np
import pandas as pd
import pytest
from matrix_engine.match import from_wide_frames
from matrix_engine.parity import ParityInputs, classify
from matrix_engine.pitch_config import PitchConfig
from plaiq_engine.match_analysis import MatchAnalysis
from plaiq_engine.semantic_builder import (
    SEMANTIC_COLUMNS,
    SemanticBuilder,
    dq_band,
    nested_record,
    pitch_zone,
)

N = 250  # 10 s at 25 Hz
FPS = 25.0


def _team(prefix: str, sign: float, rng) -> dict:
    cols = {}
    base = [(-48, 0)] + [(-30 + 6 * i, -25 + 5 * i) for i in range(10)]  # a goalkeeper deepest, then outfield
    for k, (bx, by) in enumerate(base, start=1):
        drift = np.linspace(0, 3, N)[:, None] * rng.normal(0, 1, 2)
        cols[f"{prefix}_{k}_x"] = sign * (bx + drift[:, 0])
        cols[f"{prefix}_{k}_y"] = sign * (by + drift[:, 1])
    return cols


def synthetic_match():
    """A 10-second synthetic match: fictional players, no sample data. Also feeds the UI's sample match page."""
    rng = np.random.default_rng(3)
    idx = pd.Index(np.arange(1, N + 1), name="Frame")
    clock = {"Period": np.ones(N, int), "Time [s]": np.arange(N) / FPS}
    home = pd.DataFrame({**clock, **_team("Home", 1.0, rng)}, index=idx)
    away = pd.DataFrame({**clock, **_team("Away", -1.0, rng)}, index=idx)
    ball_x = np.interp(np.arange(N), [0, 60, 120, 180, 249], [-24, -12, 0, 6, 12])
    ball_y = np.interp(np.arange(N), [0, 60, 120, 180, 249], [-20, -10, -5, 0, 5])
    away["ball_x"], away["ball_y"] = ball_x, ball_y
    events = pd.DataFrame([
        # A set piece then the pass that takes it: the pass is not scored.
        {"Type": "SET PIECE", "Subtype": "THROW IN", "Period": 1, "From": "Player 3", "To": np.nan, "Team": "Home", "Start Frame": 1, "Start Time [s]": 0.0, "Start X": -18.0, "Start Y": -15.0, "End Frame": 1, "End Time [s]": 0.0, "End X": np.nan, "End Y": np.nan},
        {"Type": "PASS", "Subtype": np.nan, "Period": 1, "From": "Player 3", "To": "Player 4", "Team": "Home", "Start Frame": 1, "Start Time [s]": 0.0, "Start X": -18.0, "Start Y": -15.0, "End Frame": 40, "End Time [s]": 1.56, "End X": -12.0, "End Y": -10.0},
        {"Type": "PASS", "Subtype": np.nan, "Period": 1, "From": "Player 4", "To": "Player 6", "Team": "Home", "Start Frame": 61, "Start Time [s]": 2.4, "Start X": -12.0, "Start Y": -10.0, "End Frame": 120, "End Time [s]": 4.76, "End X": 0.0, "End Y": -5.0},
        {"Type": "CARRY", "Subtype": np.nan, "Period": 1, "From": "Player 6", "To": np.nan, "Team": "Home", "Start Frame": 121, "Start Time [s]": 4.8, "Start X": 0.0, "Start Y": -5.0, "End Frame": 180, "End Time [s]": 7.16, "End X": 6.0, "End Y": 0.0},
        {"Type": "PASS", "Subtype": np.nan, "Period": 1, "From": "Player 1", "To": "Player 2", "Team": "Home", "Start Frame": 181, "Start Time [s]": 7.2, "Start X": 6.0, "Start Y": 0.0, "End Frame": 220, "End Time [s]": 8.76, "End X": 10.0, "End Y": 4.0},
        {"Type": "RECOVERY", "Subtype": np.nan, "Period": 1, "From": "Player 5", "To": np.nan, "Team": "Away", "Start Frame": 230, "Start Time [s]": 9.16, "Start X": 12.0, "Start Y": 5.0, "End Frame": 230, "End Time [s]": 9.16, "End X": 12.0, "End Y": 5.0},
    ])
    return from_wide_frames(home, away, events, provider="synthetic", match_id="1", fps=FPS, pitch=PitchConfig(fps=FPS))


@pytest.fixture(scope="module")
def built():
    analysis = MatchAnalysis(synthetic_match())
    parity = classify(ParityInputs(provider="synthetic", mode="full_tracking", sample_rate_hz=FPS, teams_tracked=("away", "home"),
                                   player_dropout=0.0, has_events=True, has_shot_events=False))
    builder = SemanticBuilder(analysis, parity)
    return analysis, builder, builder.semantic_events()


def test_schema_columns_and_versions(built):
    _, _, sem = built
    assert list(sem.columns) == list(SEMANTIC_COLUMNS)
    assert sem["schema_version"].eq("1.8").all()  # 1.1 P1; 1.2 pass completion; 1.3 DQ v2; 1.4 v2 cells; 1.5 v1 names; 1.6 Gain v2; 1.7 context; 1.8 teams
    assert sem["engine_version"].str.startswith("plaiq-").all()
    assert sem["match_id"].eq("synthetic_1").all()
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
    sem = SemanticBuilder(analysis, builder.parity, context=ctx).semantic_events()
    base_cols = [c for c in SEMANTIC_COLUMNS if c not in ("possession_chain_id", "possession_source", "score_diff", "red_card_diff")]
    pd.testing.assert_frame_equal(sem[base_cols], plain[base_cols])  # existing columns unchanged
    assert sem["possession_source"].dropna().isin(["observed", "inferred", "events", "unknown"]).all()
    assert sem["possession_chain_id"].dropna().str.fullmatch(r"synthetic_1_ch\d{4}").all()
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
    b = SemanticBuilder(analysis, builder.parity, context=ctx)
    sem = b.semantic_events()
    p1 = ("possession_chain_id", "possession_source", "score_diff", "red_card_diff")
    base_cols = [c for c in SEMANTIC_COLUMNS if c not in CONTEXT_COLUMNS and c not in p1]  # team keys are context-independent
    pd.testing.assert_frame_equal(sem[base_cols], plain[base_cols])  # existing columns unchanged
    assert sem["context_versions"].notna().all()
    assert sem["state_id"].dropna().str.fullmatch(r"synthetic_1:\d:\d{4}").all()
    assert sem["zone_third"].dropna().isin(["defensive", "middle", "attacking"]).all()
    assert not [c for c in sem.columns if c.endswith(("_x", "_y", "_cell"))]
    st = b.semantic_states(sem)
    assert len(st) >= 1 and st["state_id"].is_unique
    for r in st.to_dict("records"):
        assert check_quoted_numbers(r["summary"], r) == [], r["summary"]
        assert resolve_state_citation(st, r["match_id"], r["state_id"], r["start_s"], r["end_s"]).ok
    assert set(sem["state_id"].dropna()) <= set(st["state_id"])


def test_team_keys_are_opaque_and_stable(built):
    """Schema 1.8: every row carries pseudonymous team keys; an anonymous side gets a key for its match only."""
    from plaiq_engine.teams import TeamRegistry

    analysis, builder, sem = built
    assert sem["actor_team_key"].str.fullmatch(r"t_[0-9a-f]{12}").all()
    assert (sem["actor_team_key"] != sem["opponent_team_key"]).all()
    reg = TeamRegistry()
    assert reg.key("skillcorner", "1802") == reg.key("skillcorner", "1802") != reg.key("skillcorner", "866")
    assert reg.key("skillcorner", "1802") != TeamRegistry(secret="tenant-a").key("skillcorner", "1802")
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
    sem = SemanticBuilder(lost, parity).semantic_events()
    row = sem.loc[(sem["event_idx"] == 2) & (sem["action_type"] == "pass")]
    assert len(row) == 1 and row["action_outcome"].iloc[0] == "lost" and row["dq_v2_status"].iloc[0] == "estimated"


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
