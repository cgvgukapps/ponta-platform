# Context Foundations Report (P0 and P1)

17 November 2026. Task: `ponta-blueprint/context-foundations-task.md`; specification: `ponta-blueprint/context-packages.md`. Stage A of the context packages: P0 (foundations) and P1 (possession and game state), with the measurements needed to agree the first thresholds.

## Result in one paragraph

Both packages are built and run on all four regression matches in about 2.5 s each, adding about 0.8 MB per match. Every grid point carries a possession label with its source, and game state with any unknown parts marked. The outputs are byte-identical across runs. At the measured settings (1.5 m control radius, 16 m/s ball-speed limit, 0.4 s minimum switch), proximity possession agrees with the engine's existing event-based possession on **93.6% to 95.7%** of compared time across the three Metrica matches, with a similar number of possession changes. It also agrees with SkillCorner's own player possessions on **95.1%** of their frames. This shows the two methods are consistent; it does not prove either is accurate, because the comparison is itself a rule over events, not a hand-labelled truth. **Gate 1 is not declared passed:** the founder sets the bar from these numbers.

## What was built

| Piece | Where |
|---|---|
| Context grid (5 Hz, keyed by `match_id` and `frame_ts`) | `plaiq_engine/context/grid.py` |
| Package contract and registry (fields with unit, observed or estimated tag and definition; dependency order; version stamps; frame and segment output) | `plaiq_engine/context/registry.py` |
| Thresholds config with status (`proposed`, `measured`, `agreed`) | `plaiq_engine/context/thresholds.py` (version 0.2.0) |
| Data check entries from packages (additive; base report unchanged) | `matrix_engine/parity.py`: `PackageEntry`, `with_package_entries` |
| Probe package (proves the path) | `plaiq_engine/context/probe.py` |
| P1 possession and game state, chains, uploaded game state | `plaiq_engine/context/possession.py` |
| Semantic events schema 1.1 (four context columns) | `plaiq_engine/semantic_builder.py` |
| Measurements | `tools/context_measure.py` |
| Golden harness | `tools/context_golden.py`; summaries in `tests/golden/context/` (git-ignored; manifest tracked) |
| Definitions | `docs/context/vocabulary.md`; decisions in `docs/ARCHITECTURE.md` (items 11 to 15) |

**Placement:** `plaiq_engine/context/`, because later packages need Plaiq's pressure and pass logic and the semantic builder reads P1; `matrix_engine` only defines the Data check entry type. The import check passes.

**Tests:** 26 new: 3 Data check tests, 7 framework tests, 11 possession tests, 2 new semantic-builder tests, and 3 sample-data tests (Metrica Game 1, including a golden comparison). One existing assertion changed: the semantic schema version is now `1.1`, a deliberate bump. No test was removed or weakened.

## Measurements

### Ball detection

| Match | Grid points | Ball detected | 1st half | 2nd half | Gaps | Median gap | 90th percentile | Undetected time in gaps over 10 s |
|---|---|---|---|---|---|---|---|---|
| Metrica 1 | 29,002 | 60.9% | 57.4% | 64.3% | 129 | 13.6 s | 37.1 s | 93.3% |
| Metrica 2 | 28,232 | 59.0% | 71.4% | 47.4% | 147 | 7.2 s | 45.4 s | 91.6% |
| Metrica 3 | 28,753 | 67.8% | 68.3% | 67.3% | 130 | 8.9 s | 38.6 s | 83.9% |
| SkillCorner 1886347 | 29,531 | 73.6% | 83.3% | 64.9% | 66 | 17.7 s | 41.3 s | 94.4% |

Almost all undetected-ball time is in long gaps, and on the Metrica matches those coincide with dead ball (the ball is out of play). Only about 1% of undetected time is in gaps under 3 s, so the carry-gap setting matters little.

### Control sweep (agreement with the reference, and possession changes)

Reference: the engine's event-based possession (`infer_possession`) for Metrica, and SkillCorner's own player possessions (comparison only, decision 2) for SkillCorner. Agreement is over grid points where both give a team. Ball-speed limit 16 m/s; "changes" counts switches between teams.

| Match (reference changes) | Switch rule | 1.0 m | 1.5 m | 2.0 m | 2.5 m |
|---|---|---|---|---|---|
| Metrica 1 (198) | none | 92.5%, 364 | 90.3%, 456 | 88.7%, 558 | 87.7%, 596 |
| | 0.4 s | 93.3%, 168 | **93.7%, 178** | 93.4%, 204 | 92.0%, 236 |
| | 0.8 s | 92.0%, 156 | 92.5%, 150 | 92.6%, 160 | 92.2%, 158 |
| Metrica 2 (175) | none | 93.6%, 365 | 91.9%, 459 | 90.1%, 567 | 89.4%, 593 |
| | 0.4 s | 95.2%, 147 | **95.7%, 163** | 95.2%, 175 | 93.9%, 211 |
| | 0.8 s | 94.6%, 137 | 94.8%, 145 | 94.7%, 139 | 94.7%, 141 |
| Metrica 3 (199) | none | 92.0%, 479 | 90.5%, 591 | 89.0%, 701 | 88.3%, 727 |
| | 0.4 s | 91.4%, 185 | **93.6%, 213** | 92.7%, 231 | 92.1%, 255 |
| | 0.8 s | 89.6%, 169 | 90.8%, 171 | 92.1%, 181 | 92.5%, 183 |
| SkillCorner (173) | none | 92.5%, 394 | 95.5%, 564 | 96.7%, 650 | 97.5%, 740 |
| | 0.4 s | 91.1%, 202 | **95.1%, 248** | 96.6%, 286 | 97.7%, 330 |
| | 0.8 s | 87.4%, 148 | 93.0%, 194 | 95.7%, 230 | 96.9%, 246 |

Readings:

- **Without a switch rule, possession flickers.** Changes double or triple against the reference (450 to 590 against about 190), because a ball passing near an opponent flips it. A 0.4 s rule brings the count close to the reference and gives the best Metrica agreement.
- **The ball-speed limit matters little.** At 1.5 m and 0.4 s, 16 m/s beats 8 and 12 m/s on all three Metrica matches by 0.1 to 1.4 points.
- **Broadcast tracking wants a larger radius.** SkillCorner agrees best at 2.0 to 2.5 m. Its comparison covers only 23% of the match (the provider's player-possession frames) and those are controlled moments, which favour a larger radius. Its change count is above the provider's at every radius.

### Possession sources and the Data check (at the measured settings)

| Match | Observed | Inferred | From events | Unknown | Dead ball (share of match) | Possession | Game state |
|---|---|---|---|---|---|---|---|
| Metrica 1 | 85.0% | 2.2% | 12.0% | 0.8% | 32.4% | Full | Full: 3 goals, all confirmed |
| Metrica 2 | 82.3% | 2.7% | 14.3% | 0.7% | 29.7% | Full | Full: 5 goals, all confirmed |
| Metrica 3 | 89.0% | 1.4% | 8.9% | 0.7% | 25.9% | Full | Full: 2 goals, both confirmed |
| SkillCorner | 73.6% | 3.2% | 23.1% | 0.0% | unknown | **Degraded** | **Degraded:** final score 2–0 only |

Shares are of in-play time. SkillCorner's feed is derived from tracking, so it has no logged set pieces: its dead-ball time is unknown and is counted as in play. That alone pushes its not-observed share to 26%, over the 25% line.

### Possession chains

| Match | Chains | Start types | End types |
|---|---|---|---|
| Metrica 1 | 220 | recovery 86, set piece 75, won 57, period start 2 | lost 136, dead ball 71, shot 8, goal 3, period end 2 |
| Metrica 2 | 210 | set piece 77, recovery 71, won 60, period start 2 | lost 125, dead ball 72, shot 8, goal 3, period end 2 |
| Metrica 3 | 259 | won 88, set piece 85, recovery 84, period start 2 | lost 167, dead ball 82, shot 7, goal 1, period end 2 |
| SkillCorner | 249 | won 247, period start 2 | lost 241, shot 6, period end 2 |

Every chain has a known start and end type. "Goal" ends are fewer than goals (for example 1 of 2 in Metrica 3) when the possession that scored was first broken by a contested or lost frame before the shot window. That is a known limit of chain end typing, not of the score.

## Proposed thresholds (for the founder to agree or change)

| Threshold | Value | Status | Evidence |
|---|---|---|---|
| `context_grid_rate_hz` | 5 Hz | proposed | Work plan; about 29,000 grid points per match, 2.5 s, 0.8 MB |
| `p1_control_radius_m` | 1.5 m | measured | Best Metrica agreement with the switch rule; **open:** a per-provider radius for broadcast tracking (2.0 to 2.5 m for SkillCorner) |
| `p1_max_control_ball_speed_mps` | 16 m/s | measured | Slightly better than 8 or 12 m/s on all Metrica matches |
| `p1_min_switch_s` | 0.4 s | measured | Brings possession changes close to the reference; without it changes double |
| `p1_max_carry_gap_s` | 3 s | measured | About 1% of undetected time is in gaps under 3 s; little effect |
| `p1_degraded_inferred_share` | 25% | measured | Metrica 11 to 18% not observed (Full), SkillCorner 26% (Degraded) |
| `p1_dead_ball_max_s`, `p1_goal_kickoff_window_s` | 180 s | proposed | All goals in the three Metrica matches are confirmed within it |
| `p1_chain_event_window_s`, `p1_chain_shot_window_s` | 1 s, 3 s | proposed | Not swept yet |

**Gate 1 bar (decision needed).** A natural proposal is "at least 93% agreement with event-based possession on each Metrica match, with possession changes within 15% of the reference". All three Metrica matches meet that today (93.6%, 95.7%, 93.7%; changes −10%, −7%, +7%). The bar is the founder's call.

## Game-state derivation

- **Metrica 1 and 2 (CSV):** goals from `SHOT` subtypes with a `GOAL` token (`ON TARGET-GOAL`, `HEAD-ON TARGET-GOAL`). The ambiguous `WOODWORK-GOAL` subtype in Game 1 is on a **`BALL OUT`** event, not a shot, so it is not a goal. Every goal is confirmed by the other team's kick-off. Cards in both games are yellow only.
- **Metrica 3 (EPTS):** subtypes use commas (`ON TARGET, GOAL`); the same token rule finds both goals, both confirmed. One yellow card.
- **SkillCorner:** the feed is derived from tracking (no set pieces, no goal markers), so the final score (2–0) is read from the match file as numbers only, as an uploaded game state would be. The score at each moment, red cards and dead ball are unknown, and the Data check says so. The upload input (goal times, red cards) would make it Full except for dead ball.

## Runtime and storage

About 2.5 s per match for both packages (1.9 to 2.7 s), excluding loading. The frame file and chains add about 0.8 MB per match. The measurement sweep (80 settings per match) takes 45 to 80 s per match.

## Gate results

- **Full local gate** (`PONTA_REQUIRE_SAMPLE_DATA=1 pytest -q`): 1,029 passed, 12 skipped (the opt-in equivalence tests), up from 1,003 passed: exactly the 26 new tests. Without sample data, the new data-dependent tests skip as CI expects.
- **Import check and ruff:** both pass.
- **Golden harness:** captured on all four matches; a second, independent run in a separate output directory matches on every field, segment and Data check entry.

## Differences from the specification, and limits

- **Inference from player movement is not implemented.** During detection gaps the last state is carried for 3 s, then the event feed's possession is used. Because almost all undetected time is in long gaps that coincide with dead ball, the measured effect is small on these matches, but it will matter for feeds with shorter, in-play gaps.
- **Loose and in-flight balls stay with the last controlling team** and are tagged `observed` (the ball is seen, the label is carried). A separate `in_flight` source could be added if analysts want that distinction.
- **The comparison is a rule, not a truth.** Event-based possession carries the last event forward; agreement shows consistency. A small hand-labelled sample would measure accuracy.
- **SkillCorner's player names are in its event file;** the measurement tool reads only frame, team and type columns, and no name appears in any output.
- **Metrica's second-half event-coordinate issue** (`EXTRACTION_REPORT.md`) does not affect P1, which uses event frames and teams only.

## Decisions (founder, 17 November 2026)

1. **Gate 1 bar agreed:** at least 93% agreement with event-based possession on each Metrica match, with possession changes within 15% of the event feed's. **Gate 1 is passed:** 93.7%, 95.7% and 93.6%; changes −10%, −7% and +7%.
2. **Thresholds agreed:** every value in "Proposed thresholds" above, unchanged. Their status in `plaiq_engine/context/thresholds.py` is now `agreed`, each naming this decision.

Still open (questions in the report, not proposals), with the engineer's recommendation:

3. **Control radius per provider:** recommended, but only once a second broadcast match or a design partner's feed confirms the SkillCorner result; one match is not enough to set a provider-specific value.
4. **Hand-labelled possession sample:** recommended before Stage B, and cheap if folded into the first analyst session that Gate 2 needs anyway.
5. **Movement-based inference for detection gaps:** recommended later, when a feed with frequent in-play gaps appears; on the regression set its effect would be small.

## Decisions for a human (as first asked)

1. The Gate 1 bar (the proposal above, or another).
2. Agree, or change, the measured thresholds; status moves to `agreed` only by the founder's decision.
3. One control radius, or one per provider (broadcast tracking agrees better at 2.0 to 2.5 m).
4. Whether a small hand-labelled possession sample is worth doing before Stage B, to measure accuracy rather than consistency.
5. Whether to add movement-based inference for detection gaps now or when a feed with in-play gaps appears.

## Data sources and licences

*(Added 25 November 2026, review GC-14.)* Reference data this report measures against, all kept outside every repository:

- **SkillCorner open data** (20 matches): MIT licence; credit SkillCorner.
- **Metrica Sports sample data** (games 1 to 3): published by Metrica for research; licence not stated (`ponta-blueprint/legal-compliance.md`). Never copied into a repository.
