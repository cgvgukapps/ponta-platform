# Extraction report

The extraction of the prototype engines into this monorepo, following `ponta-blueprint/platform-extraction-task.md`. Written 15 November 2026. The prototype (`PONTA-PROTOTYPE`, tag `prototype-validation-final`) was not modified.

## Result in one paragraph

Both engines run from the monorepo with the prototype's tests carried over and green. Plaiq run on data loaded and normalised by Matrix reproduces the prototype's golden outputs for Metrica Sample Game 3 and SkillCorner match 1886347. On the Metrica match, two of 39,589 decision rows and 12 of 36,792 alternative rows (four carries) differ, for a characterised reason; no decision-quality band changes. A Parity Report (`parity.py`) and a semantic-event builder (`semantic_builder.py`) exist with their own tests. One finding needs a human decision: the prototype's SkillCorner parity panel is out of date with its own ingestion.

## Module split

| Package | Contents |
|---|---|
| `matrix_engine` | The prototype's `matrix_engine/*`; `ingestion/` (Metrica CSV and EPTS, SkillCorner, Second Spectrum, Genius events, velocities); `pitch_config`; new `match.py`, `pipeline.py`, `parity.py`, `sample_data.py` |
| `plaiq_engine` | `analytics/`, `statsbomb/`, `roster/`, `ingestion/statsbomb_io.py`; new `match_analysis.py`, `semantic_builder.py`, `roster/profiles.py` |

`roster/` is in `plaiq_engine` (Matrix never imports it; it imports Plaiq's `DEFAULT_PROFILE`). `pitch_config` is in `matrix_engine` (both import it). Dependency direction holds: `plaiq_engine` imports `matrix_engine`, never the reverse; `tools/check_imports.py` enforces it in CI, including imports inside functions. Not ported: `app.py`, `replay.py`, `preprocess.py`. Decisions with reasons: `docs/ARCHITECTURE.md`, items 1 to 10.

**History.** Moved files were copied, not history-preserving (`git filter-repo` is not installed). Full history stays in the prototype at the tag.

## Tests

| Location | Carried | Now | New |
|---|---|---|---|
| `packages/matrix_engine` | 245 | 261 | 16: `NormalisedMatch` round trip, `MatrixPipeline` (including identity with the engine), 11 parity rules |
| `packages/plaiq_engine` | 673 | 677 | 4: the public wrappers |
| `tests/` (repo level) | 55 (46 cross-engine, 9 import graph) | 77 | 22: 10 semantic builder, 12 equivalence (opt-in) |
| **Total** | **973** (964 from the prototype + 9 skeleton) | **1,015** | |

None deleted, skipped or weakened. Full local run, `PONTA_REQUIRE_SAMPLE_DATA=1 pytest -q`: 1,003 passed, 12 skipped (the equivalence tests, opt-in with `PONTA_EQUIVALENCE=1` because scoring both full matches takes about 13 minutes). CI runs without sample data and skips the data-dependent tests.

Test changes beyond imports: three path checks in `test_statsports_provider.py` now use the real repo root, so their "nothing from this dataset is in the repo" scans still cover the whole repo; `test_skillcorner_io.py` and `test_ss_roster.py` moved to `tests/cross_engine/` because they import `roster`.

## Plaiq-via-Matrix equivalence

**Path.** `load_metrica_epts_match` / `load_skillcorner_match` (same ingestion calls as the prototype's `preprocess.py`) → `NormalisedMatch` (Matrix long schema plus ball, events and frame clock) → `MatchAnalysis` rebuilds the wide frames and derives velocities → engine. Compared on natural keys, never row order.

**Inputs.** Identical frame grids, column order, missing-value patterns and events for both matches. Positions differ by at most 1.9e-6 m on Metrica Game 3 (float storage through the prototype's CSV) and 1.8e-15 on SkillCorner. SkillCorner event ids arrive as strings here and as int/float from the prototype's CSV; the engine normalises both to the same jersey string, so this has no effect.

**Outputs.**

| Output | SkillCorner 1886347 | Metrica Game 3 |
|---|---|---|
| Events with ΔV | equal | equal |
| Frame pressure | equal | equal |
| Movement value | equal | equal |
| Control and value surfaces (50 SkillCorner, 48 Metrica sampled frames) and 5 ball-missing frames each | equal | equal |
| POV and decision quality (39,589 rows on Metrica) | equal | 2 rows differ, below |
| Decision alternatives (36,792 rows on Metrica) | equal | 12 rows in 4 carries differ, below |

Tolerance: 1e-6 absolute and relative on every numeric column; inferred intent locations to 0.1 mm (they moved by at most 0.05 mm in 21 rows); surfaces to 1e-4 (stored as float32 in the golden file); non-numeric columns exactly.

**Differences beyond tolerance (Metrica Game 3 only).**

1. Event 2202 (interception, Away 23): `delta_pov` 0.0462 → 0.0463, one unit in the fourth decimal the engine rounds to.
2. Event 3373 (carry, Home 6, about 88:56): the best alternative carry changes near a tie (`v_best_carry` 0.0750 → 0.0933), so decision quality moves 1.248 → 1.003. Both are **Optimal**.
3. Alternatives tables for four carries: 3373 (the near-tie above, 1 row); 2370 (Home 7, 8 of 216 rows, values up to 0.0025, and the best-alternative flag swaps between two near-equal hexes without changing the carry's decision quality); 2452 (Home 4, 2 of 71 rows, up to 0.005); 825 (Home 9, 1 row, one rounding unit). 12 of 36,792 rows in all.

Cause: the 2-micrometre input difference tips near-ties and rounding boundaries in the carry alternatives. **No decision-quality band changes** across all 1,760 scored rows. No formula was edited. The test lists these rows and events explicitly and asserts no band change anywhere, so any new difference fails.

**Decision for a human:** accept these as the platform's baseline (`engine_version` unchanged), or treat event 3373 as evidence that carry decision quality is sensitive near ties and add it to the engine-validity work in `engine-governance.md`. Recommendation: accept, and record the sensitivity.

## Parity Report (`matrix_engine/parity.py`)

Twelve metrics in six families, each with state, reason, measured figure and what would change it; JSON-serialisable, deterministic, versioned. Rules implemented and tested: GPS-only blocks opposition pressure and control and keeps physical summaries; event-only degrades decision quality as a separate confidence class; optical below 10 Hz or with more than 10% of player observations missing is Degraded with the figure; no physiological source blocks load and fatigue in every mode. Coverage is reported: ball-detected share and player-observation share.

The prototype's SkillCorner panel (`app.py`, fixed text) is reproduced exactly for its stated premise: a feed with no shot events.

**Finding, decision for a human.** The SkillCorner open-data events do contain shots: 23, mapped by the prototype's own adapter from dynamic events. Measured on the real match, pass option value and event ΔV come out **Full**, where the panel said Degraded. Shot quality stays Blocked: shots derived from tracking lack the detail a shot-quality model needs. The rules were not bent to match the stale text. Decide whether the Data check should show Full for these two metrics on SkillCorner, or whether tracking-derived shots should count as Degraded.

Not yet in the report: the share of passes and carries actually scored (needs Plaiq output; Matrix cannot import Plaiq). It belongs in the semantic layer or a combined report.

## Semantic builder (`plaiq_engine/semantic_builder.py`)

One row per pass (including scored failed passes), carry, off-ball run (movements with a positive engine `run_bonus`) and defensive action (recoveries, challenges). Deterministic ids `evt_{provider}_{match}_{event}`, explicit sort. Pseudonymous refs (`home_10`), categorical `pitch_zone`, no names and no coordinate arrays (tested). `physiological_context` is null. Goalkeepers (found with the engine's `find_goalkeeper`) and passes taken at a set piece are `not_scored` with a reason. Scores equal the engine's (tested). `trajectories.parquet` carries positions, engine velocities and the ball for the replay, with no physiological columns.

Decisions and deviations from `ai-layer.md`:

- Flat columns rather than nested structs, so the typed query tool filters directly; `nested_record` gives the nested view.
- `best_option_dq_score` is not produced: by construction it would always be 1.0. `best_option_pov` and `chosen_option_pov` are produced instead.
- Pressure is the engine's 1 to 10 scale (`opposition_pressure`); `ai-layer.md`'s example shows a 0 to 1 index. The schema should change, not the engine.
- `inference_confidence` is the engine's category (strong, medium, weak), not a number.
- No per-frame confidence score for a single source: Matrix computes confidence only when reconciling. Rows carry `ball_detected` and `players_observed` instead.

## Demo

`python tools/demo.py metrica` (or `skillcorner`) runs normalise → parity → `MatchAnalysis` → semantic builder and prints the data check and the first rows of both Parquet files. Output goes to a temporary directory outside the repo; the script refuses to write inside it. About 8 minutes for Metrica, 2 for SkillCorner.

## Other findings

- `to_single_playing_direction` (Metrica CSV, games 1 and 2) flips event rows by index label using a frame number, so event coordinates in the second half may not be flipped. Not in the golden path (Game 3 uses EPTS). Reported, not fixed.
- Loading Metrica Game 3 from EPTS takes about 100 seconds; worth caching before the worker exists.

## Left for a human

1. Accept the characterised Metrica differences as the baseline (recommended).
2. SkillCorner Data check: Full or Degraded for pass option value and event ΔV.
3. Update `ai-layer.md`'s semantic schema for the deviations listed above.
4. How CI gets sample data remains "skip in CI, full run locally" until the licence gate (G3) is settled.

## Decisions recorded (15 November 2026)

1. The characterised Metrica differences are the platform baseline; `engine_version` is unchanged. The carry near-tie (event 3373) is noted for the engine-validity work in `engine-governance.md`.
2. The SkillCorner Data check follows the measured data: pass option value and event ΔV are Full; shot quality stays Blocked because the shots are derived from tracking.
3. `ai-layer.md`'s semantic event schema was updated to the implemented builder, and its invented "pressure index ≥ 0.7" example replaced: the engine's pressure is on a 1 to 10 scale and no named threshold exists yet.
