# Context States and Episodes — Report (Stage D, P8 0.1.0)

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

`tools/context_states_review.py` built a page of 38 episodes from three German league matches (5 per phase; 3 counter-attacks in the 3 to 30 s range), each replayed with its summary, judged for boundaries and summary accuracy. Results pending.

## 7. Deviations from the spec

- The AI layer's typed query tool, state filters, episode playlists and CI questions stay in `ai-workflows-task.md` (no AI layer exists yet), as the spec says.
- The episode row carries both teams' shape measures under `team_` and `opp_` names rather than all P2 line metrics.
- Episodes are not cut at the minimum duration when a chain is too short to merge (`p8_keep_short_chains`).

## 8. Decisions for the founder

1. **The episode rule:** keep the 2 s minimum, or choose 1 s (marginally better agreement, more episodes) or dropping short chains (counts and durations nearest SkillCorner's). The formation split can stay (no cost) or go (no effect).
2. **The summary templates,** after the episode review.
3. **The thresholds** (`p8_min_state_s`, the three switches): agree, or keep proposed.
