# Artefacts derived from unlicensed reference data

> **Status: kept current (started 25 November 2026, review backlog A, GC-03).** Every artefact derived from PFF FC's
> World Cup 2022 data or StatsBomb's open data is listed here. Under the founder's decision of 23 November 2026
> (`ponta-blueprint/legal-compliance.md`), such artefacts are for internal development and measurement only: each
> names its source, and none reaches a customer, pilot, demo, the website or any public claim until the founder has
> the provider's written approval. Add a row whenever a new one is made; remove it only when it is deleted or
> re-measured on licensed data.

## In this repository

| Artefact | Source | How it is kept apart |
|---|---|---|
| `pf_min_matches_*_pff` in `plaiq_engine/context/thresholds.py` (5, 5, 6, 7 matches) | PFF World Cup 2022 | `licence` tag; used only with `pf_allow_unlicensed_derived` on (default off). With it off, comparisons use the SkillCorner values and every verdict names its `minimum_source`. Test: `test_unlicensed_derived_minimums_are_off_by_default`. |
| `CONTEXT_PROFILES_REPORT.md` section 8 (Stage E on PFF) and the decision in section 9 | PFF World Cup 2022 | The section names PFF as its source; internal report. |
| `tools/pff.py` (reader) | PFF World Cup 2022 format | A reference reader, not a supported provider; reads ids only, never player names. |

No artefact in this repository is derived from StatsBomb's open data today.

## Outside every repository (`~/ponta-reference-data/cache/`)

| Artefact | Source |
|---|---|
| `pff-matches/`, `pff-observed-check/` (parsed match caches) | PFF World Cup 2022 |
| `stage-e/pff_*.parquet`, `stage-e-pff-trial/`, `profiles_pff.json`, `pff_consistency.json` | PFF World Cup 2022 |

The data itself lives at `~/ponta-reference-data/pff-worldcup2022`, outside every repository.
