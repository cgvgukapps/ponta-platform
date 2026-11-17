# Architecture

The architecture is specified in the planning repo, [ponta-blueprint](https://github.com/cgvgukapps/ponta-blueprint). This file is a pointer, so the design has one source of truth.

- `architecture.md`: canonical store (Parquet and DuckDB), API, worker, monorepo and repo strategy, build order.
- `data-tiers.md`: data tiers and the Parity Report.
- `ai-layer.md`: semantic football events and the AI guardrails.
- `platform-extraction-task.md`: the spec for the current extraction work.
- `engine-governance.md`: claim policy and ingestion quality gates.

## Decisions made during extraction (differences from the blueprint's starting proposal)

Recorded here because they are driven by the prototype's measured import graph, not by preference.

1. **`roster/` belongs to `plaiq_engine`, not `matrix_engine`.** Nothing in the prototype's `matrix_engine` imports `roster`, and `roster` imports `analytics.pitch_control` (for `DEFAULT_PROFILE`). Placing it in Matrix would make Matrix depend on Plaiq.
2. **`pitch_config` lives in `matrix_engine`**, the lower layer, because both engines import it. Matrix owns coordinate and pitch geometry; Plaiq imports it.
3. **`roster.build_roster` lazily imported `replay.load_game`**, a matplotlib demo script that is not ported. It gets a local equivalent loader (a plain CSV read). This is a code change but not an engine-maths change; it is covered by an equivalence check.
4. **`resolve_identities` no longer takes `prev_event`.** The parameter was unused in the body (only `prev_ff_identified` drives Tier 4), and its only caller, the prototype's `app.py`, is not ported and always passed `None`. Removed from the signature and docstring. No behaviour change; `prev_ff_identified` is still keyword-passed everywhere. Flagged originally in the prototype's `CLEANUP_REPORT.md`.
5. **`MatchAnalysis` takes a Matrix `NormalisedMatch`, not provider frames.** The Matrix schema carries player tracking only, so `matrix_engine/match.py` adds a `NormalisedMatch`: the vendor frame clock, players in the long schema, the ball, the event feed and the vendor slot order, all in Matrix coordinates. Its loaders call the same ingestion functions the prototype's `preprocess.py` called. `MatchAnalysis` rebuilds the wide frames the engine reads and derives velocities from positions, as the prototype did. The equivalence test checks this path against the golden outputs.
6. **Metrica Game 3 needs an EPTS loader.** The Matrix Metrica provider reads the CSV games (1 and 2) only; Game 3, the golden match, is EPTS. `load_metrica_epts_match` wraps `epts_io.load_game3`.
7. **`MatrixPipeline`: one source is Normalise only.** `normalise()` validates and measures coverage and changes no position (no resampling, interpolation or fusion), matching Tier 1 in `data-tiers.md`. `run()` reconciles two or more sources with the unchanged `UnifiedTrackingEngine` flow.
8. **Rendering helpers stay out of the engine packages.** The Plotly trace builders `app.py` imported (`control_to_hex_traces`, `value_to_hex_traces`, `build_render_grid`, `player_zone_outlines`, `corridor_traces_for_frame`) are reference for the web pitch only.
9. **Private names get thin public wrappers.** `is_failed_pass_event` and `nearest_hex` in `plaiq_engine/match_analysis.py` call `_is_failed_pass_event` and `_nearest_hex` unchanged; engine code is not edited.
10. **Decision-quality bands live in `semantic_builder.py`.** The engine documents 0.55 / 0.80 / 0.95 but has no band function (the app applied them). `dq_band` is defined once, citing the spec.

## Context packages (added 17 November 2026)

Context packages (`ponta-blueprint/context-packages.md`) describe the whole match on a fixed grid. Stage A (P0 foundations, P1 possession and game state) and Stage B (P2 team structure, P3 relative zones, P4 phase of play) are built; see `CONTEXT_FOUNDATIONS_REPORT.md`, `CONTEXT_STRUCTURE_REPORT.md` and `docs/context/vocabulary.md`.

11. **Context packages live in `plaiq_engine/context/`.** Later packages need Plaiq's pressure and pass logic, and the semantic builder reads P1, so the packages sit on the Plaiq side of the one-way dependency. `matrix_engine` knows only the Data check entry type (`parity.PackageEntry`, `with_package_entries`) and never imports package code; the import check still passes.
12. **The grid is keyed by elapsed time, not the vendor clock.** `frame_ts` is seconds since the first recorded frame, because SkillCorner's clock restarts at 45:00 in the second half and would repeat keys. The vendor clock is kept as `clock_s` and the football minute as `minute`.
13. **One frame table, one block per package.** `context_frames.parquet` holds the grid columns, then each package's fields (prefixed with its name) and a `<package>_version` column. Segment tables (P1's possession chains) are written as `<package>_<name>.parquet`.
14. **Thresholds in one versioned config.** `plaiq_engine/context/thresholds.py` holds every named value with its status (`proposed`, `measured`, `agreed`); only the founder sets `agreed`.
15. **Semantic events schema 1.1.** Four nullable columns from P1 (`possession_chain_id`, `possession_source`, `score_diff`, `red_card_diff`), null unless the builder is given a `ContextResult`. Existing columns are unchanged.
16. **Direction of attack is read from the data in Stage B.** `structure.attack_signs` takes each team's direction per period from the side its goalkeeper defends rather than trusting the provider convention. It found that Metrica Sample Game 2 was normalised with the home team attacking −x (the Metrica CSV loader flipped only the second half and assumed the first-half direction). Fixed on 17 November 2026 with founder approval: `metrica_io.orient_home_attacking_positive` rotates the match when the home goalkeeper defends +x, called by both Metrica CSV loaders after the half-time flip. The prototype's own function is unchanged. Stage B keeps reading direction from the data, so it does not depend on any loader's convention.
17. **Downstream packages rebuild P2's per-frame lines from its segments.** P3 and P4 call `team_frames` and `line_members` with P2's `roles`, `goalkeepers` and `orientation` tables rather than passing arrays between packages, so every package output stays a plain table. It costs runtime (Stage B is about 13 s of the 16 s per match).
18. **Provider labels are comparison data only.** `tools/context_references.py` reads SkillCorner's roles and phases (no names) for `tools/context_structure_measure.py` and `tools/context_phase_sweep.py`; no package reads them.
