# ponta-platform

The Ponta production monorepo. Ponta is one AI-native football intelligence product: **Matrix** (normalise, reconcile, fuse) is the built-in data engine, **Plaiq** is the spatial decision engine, and grounded AI reads the semantic events they produce. Strategy and architecture live in the planning repo, [ponta-blueprint](https://github.com/cgvgukapps/ponta-blueprint) (`architecture.md`, `data-tiers.md`, `ai-layer.md`, `roadmap.md`).

`PONTA-PROTOTYPE` is frozen at tag `prototype-validation-final` as the validation record. This repo is where production work happens.

## Layout

```
packages/
  matrix_engine/   normalise, reconcile, fuse, Parity Report; owns pitch_config
  plaiq_engine/    hex grid, pitch control, POV, decision quality, event-only mode, roster, semantic builder
apps/
  api/             FastAPI (later)
  worker/          ingestion and batch jobs (later)
  web/             frontend, imported from Ponta_Platform_UI (later)
tools/
  check_imports.py     import-graph rules, run in CI
  golden/              golden outputs captured from the prototype, before any code moved
tests/                 repo-level tests
docs/                  points back to the blueprint
```

## Rules

- **Dependency direction:** `plaiq_engine` may import `matrix_engine`. `matrix_engine` never imports `plaiq_engine`. Enforced by `tools/check_imports.py`, including lazy imports inside functions.
- **No Streamlit** (or any UI framework) under `packages/`, and no imports of the prototype scripts `app`, `replay`, `preprocess`.
- **Engine maths is not changed by extraction.** Differences found are reported and recorded as an `engine_version` decision, not silently fixed.
- **The unlicensed STATSports sample is never copied into this repo**, into fixtures, or into any output.
- **No player names** in any generated artefact. Pseudonymous references only.

## Setup

Python 3.13. Install the pinned dependencies, then both packages **in one command** (so the local packages resolve each other and pip never looks up an internal name on the public index):

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e packages/matrix_engine -e packages/plaiq_engine
```

Checks:

```bash
python tools/check_imports.py
ruff check packages tools tests
pytest -q
```

## Sample data

Tests that need open sample data (Metrica sample games, SkillCorner open data) read it from outside this repo. It is **not** copied here: the redistribution terms for Metrica and SkillCorner are still open (blueprint, `website-v2-plan.md`, gate G3). By default it is read from `../PONTA-PROTOTYPE/sample-data`; set `PONTA_SAMPLE_DATA` to point elsewhere.

## Golden outputs

`tools/golden/README.md` describes the outputs captured from the frozen prototype and how to regenerate them.
