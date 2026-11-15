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
