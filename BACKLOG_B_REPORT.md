# Review backlog B: before Phase 2

> **Status: done except running the pre-registered confirmation (27 November 2026).** Backlog B of the independent review (`ponta-blueprint/work-review.md`, section 5) is the work before the AI layer. This report records each step as it lands: what changed, what it fixes and how it was checked. Measured figures are not expected to move in steps 1 to 3; any that do are reported here.

## 1. A synthetic engine golden in CI (SE-01)

**The gap.** The only value-level check of Plaiq's metrics was the prototype equivalence test, which is opt-in and whose goldens live outside the repository. A change that shifted every decision score passed CI.

**What changed.** The licence-free synthetic match (fictional players, 10 s, no sample data) moved to `tests/support/synthetic_match.py`. `tests/golden/synthetic/engine.json` (38 kB, tracked) holds its headline values: the decision values and their alternatives, movement value, pass completion, Choice, Gain version 2, 1v1 pressure at five frames, and the control and value surfaces at those frames (summary statistics and every 97th cell). `tests/cross_engine/test_synthetic_golden.py` compares them on every push, and checks the scales: pressure 1 to 10, Gain 0 to 1.5, Choice 0 to 1, each band agreeing with its score. Recapture with `python tools/synthetic_golden.py --capture`, only for an intended change.

**Checked.** Moving one constant in `pass_option_value.py` (the carry pressure discount, 0.30 to 0.31) fails the test; restored, it passes, also with `PONTA_SAMPLE_DATA` pointing nowhere (CI mode).

## 2. Identity and pseudonyms (AI-04, DE-03, DE-15, SE-11, AI-11, GC-04, SE-15)

**The gaps.** Provider ids sat in every AI-facing row (`match_id` `skillcorner_1886347`, `competition` `skillcorner:61`), so a match id and a shirt number resolved to a named player by lookup. `event_id` was the row position in the loader's table, so any mapping change shifted later ids and could turn a stored citation into a different action. The team-key secret defaulted to a public constant, keys carried no version, a name in an event feed would have become a player ref, and STATSports ids used one public salt for every tenant.

**What changed (semantic schema 2.0, profiles 0.3.0).**

- `plaiq_engine/teams.py`: `TeamRegistry` needs its secret (`TeamRegistry.for_reference()` for tools and tests). The reference secret is refused when `PONTA_ENV` is `production`; a tenant secret needs 16 characters or more. Every key carries its key version (`t1_`, `m1_`, `c1_`, `s1_`). `identity(match)` returns the match's opaque keys.
- The semantic builder needs the registry. `match_id` is the match key on every event, episode and profile row; competition and season are opaque keys; `state_id` and `possession_chain_id` are built on the match key.
- `event_id` comes from the event's content: the match key, period, start frame, type, actor and its order among identical events. Adding an event elsewhere leaves every other id unchanged (tested).
- The provider's ids move to `provenance()` (`event_provenance.parquet`): event id, provider, provider match id and the provider's event id where the feed has one. It is not AI-facing; in the product it sits with Postgres.
- A player ref must be a shirt number; a name in the feed fails the build (`PseudonymError`). The StatsBomb event-only path still keys players by name inside its own identity code; it does not reach the builder, and StatsBomb is "Coming soon". Mapping StatsBomb's `player_id` to a shirt number is a condition of building that loader.
- `nested_record` gains a `team` block, and a test checks every flat column reaches the nested view (the loader's row position, `event_idx`, is the one named exclusion).
- STATSports player ids take a per-tenant `id_salt` (an HMAC under it); the public salt is refused in production.

**Checked.** A value-level test builds the synthetic match as SkillCorner match 1886347 with real-looking team, competition and season ids and finds none of them in any value of the semantic events or episodes; the provenance table holds them. Values are unchanged: the synthetic engine golden and the context goldens pass.

**Still to do in the product.** Keep tenant secrets in Supabase Vault; treat the registry's merges file as tenant-confidential; the API never returns `event_provenance` to a model.

## 3. Rows the AI needed (AI-06, AI-07): semantic schema 2.1

**The gaps.** Of 21 realistic coach questions the review tried, 8 could not be answered or would be answered misleadingly. There were no shot rows, a lost ball that was not a scored pass vanished (28% of turnovers on SkillCorner match 1886347, every dispossession among them), roles were always null, and `off_ball_run` rows (one per team-mate the engine rewarded at every event) made up 84% of the table at about 8 times P6's run count, under a name that said runs.

**What changed.**

- **Shots:** one row per shot, with the outcome where the feed gives it (goal, saved, blocked, off target; SkillCorner gives none) and `action_detail` `head` for headers. No shot-quality figure is claimed.
- **Turnovers:** every lost ball that is not a failed pass, with `action_detail` dispossessed, clearance, offside or other. **Every failed pass** now gets a lost-pass row, scored or not. The end of a half is not a turnover.
- **Context rows:** set pieces (`set_piece`, the kind in `action_detail`), fouls won (`foul_won`) and cards (`card`, yellow or red). None carries a score.
- **Runs:** P6's runs, one row each (`run`), with SkillCorner's run type in `run_type` and FIFA's area at the end in `run_area`. "Runs in behind by `home_9`" now has one meaning.
- **Movement values** leave the event table for `movement_values()` (`movement_values.parquet`): Plaiq's off-ball movement value per team-mate and event, each tied to its source event by `source_event_id`.
- **Roles:** `actor_role` falls back to P2's main role, with `actor_role_source` saying `roster` or `estimated`; a roster role wins.
- **Episodes:** `shots` and `ended_in_shot` (a shot inside this episode), beside the chain-level `led_to_shot`.
- Rows are ordered by time, then by the event's position.

**Measured on SkillCorner match 1886347** (raw feed: 735 passes, 487 carries, 215 lost balls, 98 set pieces, 23 shots, 15 fouls won). Schema 2.1 gives 4,348 rows (before, 9,296, of which 7,223 were `off_ball_run`): 735 completed and 167 lost passes (before 155: twelve unscored failed passes were missing), 48 turnovers, all dispossessions (before none), 487 carries, 23 shots (before none), 98 set pieces, 15 fouls won and 2,775 runs, one per P6 run of both teams in and out of possession. The 7,223 movement values sit in their own table. 96% of rows carry an estimated role; the rest are players P2 gave no main role. Of 526 episodes, 14 contain a shot, against 19 with the chain-level `led_to_shot`; 9 of the 23 shots fall outside any episode (P8 cuts episodes within possessions, and a shot from a set piece may not open one), which is a question for P8, not fixed here.

**Found on the way.** The match page payload's "Passes that lost the ball" group filtered on the Gain score, which lost passes have not carried since schema 1.9, so it was always empty. It now lists every lost pass (payload 1.2).

**Not done here.** The score difference on SkillCorner (its match file gives only the final score, so `score_diff` stays null) and a validated shot-quality measure. The UI prototype still expects `off_ball_run` in its sample payload.

## 4. One field dictionary for every AI-facing column (AI-08, AN-10)

**The gaps.** No machine-readable definition existed for schema 1.7 onwards or for the 116 episode columns. Event rows carried only decision quality's Data check state. Carrier pressure (weighted kappa 0.76 against SkillCorner) and line-breaking options (kappa at most 0.16) were both just "estimated", and "kept from player-facing views" lived in prose inside definitions. The leak tests checked column names, so a cell index added under a neutral name would have passed.

**What changed.** `plaiq_engine/field_dictionary.py` (version 1.0.0) has one entry for each of 272 columns and measures across `semantic_events`, `semantic_states`, `movement_values`, `profiles` and the 46 profile measures. Each entry gives a definition, unit and scale, kind (observed, estimated, identifier or metadata), a typed validation status with its figure and report, the Data check metric it inherits, and whether it is quotable and player-facing. `tools/field_dictionary.py` writes `docs/ai_fields.json` and `docs/ai_fields.md` from it.

**The rules, tested** (`tests/cross_engine/test_field_dictionary.py`):

- Each AI table has exactly the dictionary's columns. `check_columns` is the allowlist the API will apply, and a column added under a neutral name fails it.
- Every claimed check names its figure. An estimated field cannot be "not applicable".
- **Only observed values are player-facing** until an analyst validates an estimated one (Gate 2b). `player_facing_view` applies this.
- A field that disagrees with reference data is neither quotable nor player-facing. Today that is three profile rates: press, pressure and recovery engagements per minute, which run against PFF's pressure rate. Identifiers are cited, never quoted.
- No entry claims analyst validation; the test says so if that changes.
- The generated JSON matches the module.

**Status counts** (160 estimated fields): 28 agree with reference data, 54 agree in part, 3 disagree, 13 are measured against outcomes (Gain, Choice, adjusted Choice and their medians), and 62 are not yet measured. Agreement figures are those reported so far: measured on a development set, with the profile agreements from before backlog A (`CONTEXT_PROFILES_REPORT.md` section 3).

**Not done here.** Typed-tool results are not built yet. When they are, each attaches the state of its fields' `data_check` metric from the match's Data check. The episode summary's quoted fields are not yet checked against the quotable flag; none of the fields it quotes is a disagreeing one.

## 5. The citation's confidence slot is an evidence grade (AI-03)

**The gap.** The open decision proposed filling the confidence slot from `ball_detected` and `players_observed`. On broadcast tracking both were constant (22 players and the ball "detected" on all 8,600 rows of SkillCorner match 1886347), so any number built from them would have printed high confidence on every claim, against the match's own Data check.

**What changed (semantic schema 2.2).**

- The slot is a small categorical object, not a probability: `{field, data_check, label, validation, observed_near_ball}`. It gives the cited field's Data check state in this match, observed or estimated, its validation status from the field dictionary, and a band of the row's observed near-ball share. The band is `all_observed` at 90% or more (the Data check's own rule: no more than 10% of observations missing or estimated), `partly_estimated` below that, and `not_flagged` where the provider does not mark estimates.
- `citations.match_states(parity)` gives every state by the dictionary's metric names. `evidence_grade(row, field, states)` computes the grade. `check_grade` and `resolve_event_citation(..., grade=, states=)` are the verifier's check that a model's grade is that one.
- A field that may not be quoted (identifiers, disagreeing fields) cannot be graded, so it cannot be cited as a claim. A state the Data check did not report is never shown as Full.
- Events gain `observed_near_ball_share`: the share of positions within 15 m of the ball that the provider observed at the action's start.

**Measured on SkillCorner match 1886347** (events without context): 1,568 of 1,573 rows have a share (median 1.0, 10th percentile 0.86). The bands are 1,341 `all_observed`, 227 `partly_estimated` and 5 `not_flagged`. `players_observed` now varies (median 16, standard deviation 4).

**Decision recorded.** This closes the open decision on what fills the confidence slot for single-source matches, in the form backlog B named (approved by the founder, 27 November 2026). `ponta-blueprint/ai-layer.md` carries it.

## 6. Data engineering (DE-04, DE-06, DE-08, DE-09, DE-12)

### 6a. Extra time (DE-06)

**The gap.** Three supported loaders flipped every period after the first alike, so in a cup match with extra time one of its halves would be mirrored for every Plaiq metric, and SkillCorner event times after the second half were wrong. EPTS fell back to Sample Game 3's own frame bounds for any match without them. No open sample match goes to extra time, so nothing caught it.

**What changed.**

- **SkillCorner** reads the home team's direction for every period from `home_team_side`. On all 20 open matches that list alternates, so their output is unchanged.
- **Metrica's CSV** orients periods 3 and 4 by where the home goalkeeper stands against period 1, read from the data (the files give no per-period direction). The load fails when that cannot be read. Periods 1 and 2 are unchanged, as their regression tests require.
- **The clock helper** times periods 3 and 4 after the earlier periods; the second half's times are unchanged.
- **EPTS** needs both halves' bounds from its metadata and refuses extra time, since its metadata gives the direction for the first half only.

**Checked** on synthetic four-period fixtures (`tests/cross_engine/test_extra_time.py`). Under the old rule, extra time's first half was mirrored on SkillCorner and its second half on Metrica; both now hold one end in all four periods.

### 6b. The Data check reads the data, not the caller (DE-04; Data check schema 1.2)

**The gaps.** The mode defaulted to full tracking, so a wearable export run without one was described as optical and as partly usable for opposition pressure ("Positions are only one team is tracked."). Ball detection never entered a rule. Whether events were inferred from tracking was decided by the provider's name.

**What changed.**

- `infer_mode(match)`: no positions gives event-only; positions with no ball and not both teams (a wearable export) give GPS-only; anything else gives full tracking. A `mode` passed that contradicts the data raises.
- **The ball at passes and carries.** Over the whole match, ball detection mixes in dead-ball time: 59% to 74% on the sample matches, even Metrica's optical games. At the start of passes and carries it is 99.9% to 100% on Metrica's three games and SkillCorner's match 1886347. The ball-dependent metrics (control zones, movement value, pass option value, decision quality, pass completion, Choice and so Gain version 2) are Degraded when the ball is missing at more than 10% of those starts. This is the player-observation rule's threshold, an engineering choice. Pressure reads players and is not affected. The sample matches are unchanged.
- `NormalisedMatch.events_derived_from_tracking` is set by the adapter (SkillCorner's loader). Matches cached before the flag existed fall back to the provider name.
- Reasons read as sentences ("Positions are limited: only one team is tracked.").

### 6c. One clock (DE-09; Matrix 0.2.2)

**The gap.** The Matrix schema had no clock check, so a feed counting each period from its own start would interpolate across the halves on resampling. The reference readers for PFF and the German league put the second half at a fixed 45:00, so a first half's stoppage time overlapped it.

**Found on the way, and worse.** SkillCorner's own timestamps also restart the second half at 45:00. The main SkillCorner loader kept them for tracking but timed events from frame counts on elapsed time, so **every second-half event sat 78.1 s after its own tracking frame** on match 1886347. Episode times come from the tracking clock and event times from the events, so the two AI tables disagreed by 78 s after half time on SkillCorner. Joins by frame (the Data check, the context packages, the measured agreements) were not affected; anything read by time was.

**What changed.**

- `matrix_engine.clock.elapsed_match_time`: each period starts where the previous one ended.
- The SkillCorner loader rebases its tracking clock to elapsed time, so event and tracking times now agree exactly in both halves. The prototype-equivalence inputs keep the old clock.
- The PFF reader (version 3) and the German league's reader (version 2) use elapsed time; their caches re-parse on next use.
- `validate_schema` refuses a timestamp that runs backwards in frame order within a source and player.
- Matrix is 0.2.2 (an output change), which also re-keys the versioned caches.

**Checked.** On the context goldens only `clock_s` changes, and only on SkillCorner match 1886347; every other field and the three Metrica games are unchanged (recaptured). The full suite passes.

**Not done here.** One shared SkillCorner reader for the Matrix provider and the loader (they now agree on the clock and on dropping null periods, but remain two code paths), a separate football-clock column for display, and `period` as a required schema column.

### 6d. What a load dropped and assumed (DE-12)

**The gap.** The adapters drop records with no period, skip events with no frame, set unreadable timestamps to 0.0 and read or assume each period's direction, and none of it was counted. A worker could not tell a club what its upload lost.

**What changed.** `matrix_engine.ingest.IngestReport` lists rows read, rows dropped by reason, fallbacks used and where each period's direction came from. The SkillCorner loader fills it at each of its drop points (records without a period, missing or unreadable timestamps, events without a frame or period, events with no mapped type, players without a shirt number) and attaches it as `match.ingest`. The Data check carries it in its coverage, beside the ball-at-passes share. On match 1886347 it reports 59,061 tracking records read, 19 without a period dropped, one event with no mapped type, and both periods' directions from the match file.

**Not done here.** The Metrica, PFF and German league readers do not report yet, and no state changes on a count: a threshold waits for measurements of what real uploads drop. The worker's error taxonomy (input invalid, provider quirk, engine failure) belongs with the worker.

### 6e. Pinned Parquet schemas and a manifest (DE-08)

**The gap.** pandas inferred each file's types from its rows. On one SkillCorner match `actor_role`, `score_diff` and `physiological_context` were Arrow `null`, while counts were `int64` in one file and `double` in another. A DuckDB query over many matches would need `union_by_name` and casts, or fail. Nothing tested the files with the engine that will read them, and DuckDB was not installed.

**What changed.** `plaiq_engine/parquet_schemas.py` pins one Arrow schema per table: `semantic_events`, `semantic_states`, `movement_values` and `event_provenance`. Integers are nullable `int64`, measures `double`, flags `bool` and everything else `string`. The types were inferred from Metrica game 1 and SkillCorner match 1886347 with full context, then fixed by hand. Counts the context packages can return as fractions are `double` everywhere. A fraction in an integer column fails the write.

`SemanticBuilder.write` writes through the schemas and adds `manifest.json`, which lists every file with its rows and SHA-256 hash, plus the match key, schema version, engine version and field dictionary version. Trajectories are written in about eight row groups for the replay's range reads. DuckDB 2.0.0 is pinned as a development dependency (2.0.1 was a week old).

**Checked.** Two matches' files read together in DuckDB without `union_by_name` (`tests/cross_engine/test_parquet_schemas.py`), and each manifest hash matches its file.

## 7. Tests that run where they matter (SE-02, SE-03, SE-12, DE-13)

**Reconcile and Fuse in CI (SE-02).** Every Matrix fixture started from a real Metrica window, so on a clean runner 135 of 275 tests skipped, and sync, fusion and identity ran at 8% to 12% line coverage. Where the sample data is absent, `matrix_engine.testing.synthetic_ground_truth` now gives a seeded synthetic match in the window's shape. The ball wanders on a scale of seconds, each team's block shifts with it, and each player moves about a formation slot (median speed about 2 m/s, 99th percentile about 7.5). A first version that moved too smoothly left the clock alignment with too little team-level signal to lock onto, and failed six tests; with block shifts that follow the ball, they pass. In CI mode 220 Matrix tests pass and 16 skip (those that read real provider files), against 140 and 135 before. The Metrica window stays the local, data-required run.

**Context goldens that fail rather than skip (SE-03).** The golden test skipped whenever any package or threshold version changed, so a recapture blessed intended and unintended changes together. It now compares package by package. A package is skipped only when its own version or one of its own threshold values changed; every other package must match, or the test fails and names it. The comparison covers grid fields, segments, Data check entries and, new, hashes of the semantic events and episodes. Version labels are left out of the hashes, so a bump is not mistaken for a value change. Under `PONTA_REQUIRE_SAMPLE_DATA` a missing summary fails, as does one whose hash differs from the one the tracked manifest records. `PONTA_GOLDEN_ALL=1` checks all four matches. The rules have their own data-free tests (`tests/test_context_golden_compare.py`).

**A pinned STATSports sample (SE-12).** CI fetched a 51 MB unlicensed file from a third party's `main` branch with no hash. The reference sample is now pinned to the repository's commit of 4 August 2022 and checked against its SHA-256; a copy that differs is refused. The default is refused in production, where the worker passes the club's upload. The 17 tests that fetch it are marked `network`, and CI runs `-m "not network"`.

**Edge-case fixtures (DE-13).** Added, beside the existing null-period and estimated-position ones: extra time (step 6a), clocks that restart each period (6c), several identical events on one frame (distinct, stable ids), and an event with no frame.

**Found on the way.** The engine's possession pass failed on an event with no start frame. Such an event is now left out at `MatchAnalysis.events`, keeping every other event's label, and counted in `events_without_frame`. The supported loaders already drop them, so the sample outputs and the engine golden are unchanged.

## 8. A pre-registered confirmation (ST-01, ST-02)

**The gap.** One even/odd split of the same 20 SkillCorner matches was the "held-out" half for every tuning round. The Gate 2a bars were restated after the first reading, and at least one rule changed after a held-out reading.

**What changed.** `CONFIRMATION_PREREGISTRATION.md` is written before any confirmation data is opened. It records:

- The frozen versions: Matrix 0.2.2, Plaiq 0.2.0, the package versions, thresholds 0.30.0, semantic schema 2.2.
- The data, chosen in advance:
  - **Part A, internal, can run now:** PFF's World Cup set, which never set or tuned a P1 to P8 rule. It confirms possession, line breaks and carrier pressure against PFF's own labels.
  - **Part B, when a licensed, never-used set exists:** phases, roles, formations, episodes, runs and engagements.
- The bars. Gate 2a's agreed bars are kept unchanged; the others are proposed below each development figure, and they wait for the founder's agreement.
- An analysis plan: one run, every figure reported, nothing tuned after looking.

**Meanwhile, leave one team out.** The frozen rules were scored on SkillCorner's 20 matches leaving out each of the 13 teams in turn. Pooled phase kappa stays between 0.637 and 0.654 (0.647 on all 20). Each team's own matches range from 0.595 to 0.678. Roles grouped stay between 86% and 89%. No team carries the headline figures. This is not re-tuned cross-validation: the rules were tuned on all 20 matches, so it shows dependence on single teams, not performance on new data.

**Not done here.** Running Part A (it needs the founder's agreement on the bars first) and Part B (it needs data).

## Where the tests stand (27 November 2026)

- **Local, with sample data** (`PONTA_REQUIRE_SAMPLE_DATA=1 pytest -q`): 1,243 pass, plus the 12 opt-in equivalence tests (1,193 before backlog B).
- **CI mode** (no sample data, `-m "not network"`): 1,206 pass and 32 skip (tests that read real provider or sample files). The 17 tests that fetch the pinned STATSports sample are deselected.
- **Slow opt-in checks:** `PONTA_GOLDEN_ALL=1` runs the context golden on all four matches, `PONTA_GOLDEN_SEMANTIC=1` compares the semantic tables too (about 5 minutes a match), and `PONTA_EQUIVALENCE=1` runs the prototype equivalence test.
- The builder's pressure lookup now indexes the frame table once instead of filtering it per event (values unchanged). Building a full match's semantic events took about 4 minutes before this change.
