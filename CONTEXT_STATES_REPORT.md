# Context States and Episodes — Report (Stage D, P8 0.1.0 and 0.2.0)

Stage D of `ponta-blueprint/context-packages.md`, built to `ponta-blueprint/context-states-task.md` (23 November 2026).

**On the founder's judgements:** one expert's reading, an indication beside the provider references, never ground truth. Scores against them are written as agreement with the founder's call, and no rule is tuned to them alone.

## 1. What was built

- **P8, the episode package** (`plaiq_engine/context/states.py`; depends on P1 to P7 and P4v; estimated):
  - **The rule:** P4's phase segments (already within one team's possession chain and one phase), split where either team's formation changes (`p8_split_formation` 1) and, when `p8_split_fifa_phase` is 1, where P4v's FIFA phase changes. Runs under `p8_min_state_s` (2 s, proposed) merge into the run before them in the same chain, or the one after for a chain's first run. A chain too short to merge keeps its single episode (`p8_keep_short_chains` 1). Dead-ball time and time with no team in possession belong to no episode. `state_id` is `"{match_id}:{period}:{index}"`.
  - **The row** (segment `states`, 101 columns): identity and time; possession and game state; P4 and P4v phases with press, counter-press and FIFA-block shares and the opposed share of build-up; formations and both teams' length, width and line gaps; the ball's P3 zone at the start and end and the share in each third; passes, completed passes and line breaks; carrier pressure (median and peak, 1 to 10), goal-side defenders and the overload at the ball; runs by type (one column per P6 type, and carries); engagements by type (one column per P4v type), presses and whether a counter-press occurred; open high-value space for both teams and the open danger zone; how it ended (`end_reason`) and whether the chain led to a shot; the event rows at its ends; coverage (ball detected, fewest outfield players seen); **one Data check column per metric** (16) and the detail as JSON; versions; the label (estimated); and the **summary**.
  - **The summary** is written by code from fixed templates and quotes only the row's own fields. It marks the context as estimated and uses no causal words. Example (German league): "Home, a counter-attack, 7.2 s, in the attacking third, against a defence in transition. 1 pass, 0 completed, 0 broke a line. Pressure on the ball carrier peaked at 7 on the 1 to 10 scale. Away engaged the ball carrier 1 time. Ended: the ball went out of play. Context estimated, not yet validated."
  - **Without an event feed** passes and events are null, never zero (found by an independent audit, below), so an aggregate cannot read "no passes" as a pattern.
- **Semantic events at schema 1.7** (`semantic_builder.py`): each action carries, at its start, the episode (`state_id`), P4 and P4v phases (`phase_team` names whose they are), the block, both formations, the ball's zone in the actor's attacking direction, line breaks for passes, goal-side defenders, the overload at the ball, runs in behind, context-adjusted Choice, open high-value space, engaging defenders and the context versions. Earlier columns and values are unchanged (tested). The nested JSON gains a `context` block.
- **`semantic_states`** (`SemanticBuilder.semantic_states`, written as `semantic_states.parquet`): P8's rows plus the semantic event ids at each end and, for the passes in each episode, the median Gain (0 to 1.5), Choice (0 to 1) and context-adjusted Choice, named apart and never combined.
- **Citation resolution** (`plaiq_engine/citations.py`): `resolve_event_citation`, `resolve_state_citation` (times within 0.25 s) and `check_quoted_numbers` (every number in a sentence must be a field of the cited row, as the sentence rounds it, or a scale end: 0, 1, 1.5, 10).
- **Thresholds 0.22.0:** `p8_min_state_s` 2 s, `p8_split_formation` 1, `p8_split_fifa_phase` 0, `p8_keep_short_chains` 1, all proposed.
- **Tools:** `tools/context_states.py` (the measurements) and `tools/context_states_review.py` (the founder's episode review page and its scoring).

## 2. Against SkillCorner's phases of play (20 matches; held out, fit in brackets)

SkillCorner's `phases_of_play` file is a published episode segmentation (comparison only). Boundaries are each side's starts and ends.

| Variant | Episodes per match (SkillCorner's phases) | Median duration (SkillCorner's) | SkillCorner boundaries found within 2 s | P8 boundaries matched within 2 s | Phase κ, same team |
|---|---|---|---|---|---|
| **Base (2 s minimum)** | 686 (671) against 439 | 4.2 s (4.4) against 4.9 | **78.6%** (77.5%) | **58.5%** (58.7%) | **0.64** (0.66) |
| No formation split | 686 (670) | 4.2 s | 78.6% | 58.5% | 0.64 |
| FIFA phase split | 728 (721) | 4.0 s | 80.3% | 56.8% | 0.63 |
| 1 s minimum | 738 (721) | 3.8 s | 80.4% | 59.2% | 0.64 |
| 3 s minimum | 648 (632) | 4.6 s | 74.8% | 58.1% | 0.62 |
| 5 s minimum | 579 (559) | 5.4 s | 63.9% | 56.6% | 0.58 |
| Drop short chains | 540 (536) | 5.2 s | 76.6% | 57.2% | 0.64 |

- Within 1 s: base finds 56.3% of SkillCorner's boundaries, and 37.8% of P8's match one.
- **Episodes cover 93% of SkillCorner's phase time; 72% to 74% of episode time lies inside one of its phases** (SkillCorner leaves gaps between phases).
- **The formation split changes nothing** on these matches: formations rarely change mid-possession.
- **The minimum duration is a trade-off with no clear winner:** 1 s is marginally better on every measure but cuts about 50 more episodes a match; 5 s loses boundaries. **Dropping short chains** brings counts and durations nearest SkillCorner's at little cost in agreement.
- **P8 cuts about half as many episodes again as SkillCorner has phases.** Most of the difference is short possessions (P1 changes the team in possession more often than SkillCorner's phases do) and P4's phase changes inside a SkillCorner phase.
- **About 31% of possession time on SkillCorner's broadcast data has no episode:** the ball is out of camera, so P4 has no phase there. The Data check coverage says so.

## 3. Metrica and the German league

| | Episodes per match | Median duration | Possession time in an episode | Possession changes with an episode starting within 2 s |
|---|---|---|---|---|
| Metrica 1 / 2 / 3 (logged events) | 609 / 616 / 710 | 4.2 / 3.8 / 3.4 s | 100% | 79% / 77% / 81% |
| German league, 7 matches (official possession) | 651 to 768 | 3.0 to 3.4 s | 100% | 79% to 85% |

- **End reasons** (Metrica 1): possession lost 264, phase changed 167, foul or set piece 112, ball out of play 35, shot 22, goal 7, end of period 2. Without a logged feed (SkillCorner) the restart reasons collapse into possession lost.
- **The German league needed one loader fix in the measurement tools:** its event table has no pass end points, which Plaiq's event tagging expects, so the tools add empty `End X` and `End Y` columns. The full context stack had not run on those matches before. The engine is unchanged.

## 4. Checks

- **Every summary passes the quoted-number check** on all 30 matches (0 failures), and no AI-facing column holds a coordinate, cell index or name (tested, and checked on every match).
- **Deterministic:** a second run gives equal episodes (tested); golden outputs recaptured and a second run matches.
- **Runtime:** P8 takes a median 2.0 s a match (max 3.7 s), inside the 10 s budget (first version 24 s; columns are now extracted once and segment tables sliced by binary search, with identical output). **Storage:** about 130 KB a match for the episode table.

## 5. An independent audit (23 November 2026)

A separate agent audited the codebase, read-only, for whether it will let users discover patterns. Its findings, with what was done:
- **Fixed in P8:** pass and event counts are null without an event feed; run types, engagement types and the Data check are separate columns, not JSON.
- **The biggest blocker to patterns across matches:** no stable team or player identity across matches (a team is `home` or `away`, a player `home_7`, within one match), no per-team, per-match profile and no baselines. Planned as the next stage (`ponta-blueprint/context-packages.md`).
- **Later:** one smoothed, shared kinematics layer for the context packages (they take speeds from raw differences, which inflates jitter), and a richer event row (players bypassed, open lanes and line-breaking options copied from P5, distance and angle to goal as bands, never as numbers that would amount to a coordinate). Its suggestion to wire in pressure version 2 conflicts with the recorded decision not to adopt it; it needs a new decision first.

## 6. The founder's episode review

`tools/context_states_review.py` built a page of 38 episodes from three German league matches (5 per phase; 3 counter-attacks in the 3 to 30 s range), each replayed with its summary, judged for boundaries and summary accuracy. The calls, without positions, are in `tools/data/episode_review_2026-11-23.json`.

| | Agrees with the founder's call |
|---|---|
| Boundaries: start and end with the passage | **36 of 38** (2 start too late) |
| Summary accurate / partly / wrong | **28** / 8 / 2 |
| By phase, summary accurate | long ball 5/5, counter-attack 3/3, build-up 4/5, final third 4/5, progression 3/5, attacking transition 3/5, set play 3/5, contested 3/5 |

**What the notes point to** (none is tuned to; each needs the references as well):
- **Clearances read as attacking play (4 cases).** A ball cleared out of the defending half is summarised as an attacking transition or progression; one note describes a defensive recovery and a clearance. P4 labels the cleared team's next spell from how far the ball travels, and the summary does not say the episode ended in a clearance. P1 already reads logged clearances (0.3.0).
- **Set play after the ball is moving (2 cases).** The founder's reading: a set play starts from a still ball, so play some seconds after the kick is not one. P4 follows SkillCorner's convention, where the set-play phase continues after the kick up to `p4_set_play_max_s`. An upstream P4 question, not P8's.
- **How an episode starts is missing.** A goalkeeper's catch and throw read as an opposed build-up; a contested header won by a team reads as progression; a contested spell runs on until one team clearly regains. The row has how it ended but not how it started (P1's chains carry `start_type`).
- **Two boundaries start late:** a set-play episode starts after the kick (P1's chain starts at the first touch after the restart), and one progression covers the other team's recovery first.
- **Wording:** "against a contested spell" and "against a set play" read badly; the defending side of those phases should not be phrased as an opponent's phase.

## 6a. P8 0.2.0, after the review (23 November 2026)

**The founder's decisions:** keep the 2 s minimum (`p8_min_state_s` agreed; thresholds 0.23.0); build P8 0.2.0 as proposed.

**Built:**
- **How the episode started** (`start_reason`, `start_detail`): continued from the previous phase in the same chain; won the ball; the goalkeeper had the ball (the team's first logged action is the goalkeeper's, from P2's goalkeeper for that window); a set piece, named (throw-in, free kick, corner, goal kick); a restart; a kick-off. From P1's chain start and the logged first action. The summary says it: "Started: the goalkeeper had the ball.", "Started from a throw-in."
- **Clearances:** `cleared` is true when the team in possession made a logged clearance in the episode (Metrica logs them as a subtype, the German league as an event type); an episode whose last logged action by that team is a clearance ends `cleared` ("Ended: the ball was cleared."). Null without an event feed.
- **Wording:** the defending side of contested, set-play and long-ball spells is no longer phrased as an opponent's phase ("against a contested spell" is gone).

**Measured** (Metrica 1 to 3 and the German league's 7 matches, about 6,800 episodes):
- **Start reasons:** won the ball 3,542, continued 1,642, set piece 1,496, kick-off 94, goalkeeper 39.
- **Clearances are rare in the feeds,** so `cleared` is rare: under 3% of episodes in every phase on Metrica, and on the German league 5.5% of attacking transitions, 2.7% of contested spells, 1.7% of long balls. Ended `cleared`: 1 to 13 a match.
- **On the founder's clearance cases** (an indication): two of the three now carry the clearance (one ends "cleared", one says the team cleared the ball); the third has no logged clearance, so no rule from the feed can catch it. A clearance read from tracking (a long, high, unaimed ball out of the defending third under pressure) would need its own measurement first.
- Every summary still passes the quoted-number check; no coordinate, cell or name column; storage about 150 KB a match.
- **Not changed:** P4's set-play window (play after the kick stays a set play under SkillCorner's convention) is an upstream question to measure against SkillCorner's set-play phases before any decision; the episode boundaries.

## 6b. The open questions, decided by the standards (23 November 2026)

The founder's decision order (published definitions, then professional reference data, then outcomes, his own view only to break a tie) applied to the two questions the review left:

- **P4's set-play window stays.** SkillCorner's published phases specification, which P4 follows, keeps the play after the kick in the set-play phase for up to 20 s, and its own phases-of-play files do so: 360 set-play phases on its 20 matches, median 4.8 s, 76% longer than 2 s. The founder's reading (a set play is the still ball) is a different convention, not evidence against this one. **P8 0.2.1** words the phase as "play from a set piece", which says what the convention means.
- **Clearances stay logged-only.** A tracking-only candidate (`tools/context_clearance_tracking.py`: the end of a controlled spell, a fast ball from the kicking team's defensive zone that gains distance and is not collected by a team-mate) finds at most 6% of the German league's logged clearances, at 6% precision or less, across 27 combinations of speed, zone and distance on 4 matches; the best of them finds 4% at 3% precision on the other 3. A clearance is usually a single first-time touch, so the team never "controls" the ball in P1's sense; reading clearances from tracking needs a touch detector, which P1 does not have.

## 7. Deviations from the spec

- The AI layer's typed query tool, state filters, episode playlists and CI questions stay in `ai-workflows-task.md` (no AI layer exists yet), as the spec says.
- The episode row carries both teams' shape measures under `team_` and `opp_` names rather than all P2 line metrics.
- Episodes are not cut at the minimum duration when a chain is too short to merge (`p8_keep_short_chains`).

## 8. Decisions for the founder

1. *(Decided 23 November 2026: keep 2 s.)* **The episode rule:** keep the 2 s minimum, or choose 1 s (marginally better agreement, more episodes) or dropping short chains (counts and durations nearest SkillCorner's). The formation split can stay (no cost) or go (no effect).
2. *(Done in 0.2.0, above.)* **The summary, after the review:** add how the episode started (P1's `start_type`: regain, set piece, restart, period start) and an end reason "cleared" when the last logged action is a clearance; fix the wording for contested and set-play spells; measure how often transitions and progressions end in a logged clearance on Metrica and the German league before changing anything upstream.
4. **P4's set-play window** (upstream, agreed earlier): whether play after the kick stays a set play (SkillCorner's convention) or the phase ends once the ball is moving (the founder's reading); measure against SkillCorner's set-play phases first.
3. **The thresholds** (`p8_min_state_s`, the three switches): agree, or keep proposed.

## Re-measured after the review's backlog A (25 and 26 November 2026)

The review of 25 November 2026 (`ponta-blueprint/work-review.md`, AI-01, AN-03, AI-10) found episode reasons wrong whenever a chain had a gap in the grid: on SkillCorner 1886347, 265 of 677 episodes ended "possession lost" or "shot" while the same team kept the ball, and restarts read as "won the ball". **P8 0.3.0** continues a chain by its id, gives the chain's end only to its last episode, ends an episode split by a brief contest `contested_kept` (a median 0.4 s gap) or else `phase_unknown`, and starts episodes at restarts: SkillCorner's recorded stoppages (P1 0.6.0) and, without a logged restart, those P4 finds from positions. Episode end ids come from the semantic events, so each resolves.

| | Before | After |
|---|---|---|
| Episodes a match, SkillCorner | about 680 | **about 545** |
| Against SkillCorner's phases of play, fit / test half: phase κ | 0.64 (test) | **0.658 / 0.652** |
| Boundaries found within 2 s, fit / test | 79% | **78.1% / 78.5%** (precision 64%) |
| Episodes followed by the same chain yet given a terminal end (1886347) | 265 | **0** |
| Restart starts on 1886347 (set piece) | 0 | **72 of 526** |
| Summaries failing the number check | 0 (loose check) | **0 (field-bound check)** |

The fit and test halves are halves of one development set used across tuning rounds, not an independent confirmation (review ST-01).

## Data sources and licences

*(Added 25 November 2026, review GC-14.)* Reference data this report measures against, all kept outside every repository:

- **SkillCorner open data** (20 matches): MIT licence; credit SkillCorner.
- **Metrica Sports sample data** (games 1 to 3): published by Metrica for research; licence not stated (`ponta-blueprint/legal-compliance.md`). Never copied into a repository.
- **The German league's open data (IDSSE)** (7 matches): CC BY 4.0. Data from the DFL (Deutsche Fußball Liga), published by Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12, 195.
