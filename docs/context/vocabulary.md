# Context vocabulary (P0 and P1)

Definitions of every field the context packages produce, as built on 17 November 2026. Package versions: `probe` 0.1.0, `p1` 0.1.0; thresholds 0.2.0. The specification is `ponta-blueprint/context-packages.md`; the measurements behind the thresholds are in `CONTEXT_FOUNDATIONS_REPORT.md`.

**Observed** means a fact read from the data (a distance, whether the ball was seen, an event). **Estimated** means a label the engine assigns by rules. Nothing in the frame table reaches the AI layer directly; the semantic events carry only the labels listed at the end.

## Grid (P0)

| Field | Unit | Meaning |
|---|---|---|
| `match_id` | id | `{provider}_{match}`, the same id as the semantic events |
| `frame_ts` | s | Time since the first recorded frame; monotonic across periods; the grid key with `match_id` |
| `period` | number | 1, 2 (3, 4 for extra time) |
| `native_frame` | frame | The vendor frame nearest the grid point (within half a native period); used for refinement |
| `clock_s` | s | The vendor's own clock at that frame (SkillCorner restarts at 45:00 in the second half) |
| `minute` | minute | Football minute: first half from 1, second half from 46 |
| `<package>_version` | version | The version of the package that wrote that block |

Rate: `context_grid_rate_hz` (5 Hz). Nothing is interpolated: a ball or player not observed at the native frame stays unobserved.

## P1 Possession and game state

| Field | Unit | Kind | Meaning |
|---|---|---|---|
| `p1_possession` | label | estimated | `home`, `away`, `contested`, `dead_ball` or `unknown` |
| `p1_possession_source` | label | observed | `observed` (ball detected), `inferred` (carried through a short detection gap), `events` (the event feed's possession), `unknown` |
| `p1_ball_detected` | bool | observed | The ball has a position at the native frame |
| `p1_ball_speed_mps` | m/s | observed | From consecutive native frames; null when undetected or isolated |
| `p1_nearest_home_m`, `p1_nearest_away_m` | m | observed | Ball to each team's nearest observed player; null when the ball is undetected |
| `p1_chain_id` | id | estimated | The possession chain the grid point belongs to; null outside a chain |
| `p1_dead_ball` | bool | observed | Between the last action and a logged set-piece restart; null (unknown) when the feed has no logged set pieces |
| `p1_set_piece` | label | observed | The restart a dead ball is waiting for (`throw_in`, `corner_kick`, `free_kick_indirect`, `kick_off` …) |
| `p1_score_home`, `p1_score_away` | goals | observed | Goals so far; null when the score at that moment is unknown |
| `p1_red_home`, `p1_red_away` | cards | observed | Red cards so far; null when unknown |
| `p1_game_state_source` | label | observed | `events`, `upload`, `final_only` or `unknown` |

**Rules.**

- **Control:** a team is in control when its nearest player is within `p1_control_radius_m` of a detected ball moving slower than `p1_max_control_ball_speed_mps`; both teams within the radius is `contested`.
- **Switching:** possession passes to the other team only after it has controlled the ball for `p1_min_switch_s`, and the switch is dated from when it took control.
- **Loose ball:** while the ball stays detected and nobody controls it, the last controlling team keeps possession (`observed`).
- **Detection gaps:** the last state is carried for at most `p1_max_carry_gap_s` (`inferred`), then the event feed's possession is used (`events`), else `unknown`.
- **Dead ball:** from logged events only, the span between the last action before a set piece and the set piece (at most `p1_dead_ball_max_s`), excluding a period's opening kick-off.
- **Goals:** a `SHOT` whose subtype has a `GOAL` token (`ON TARGET-GOAL`, `ON TARGET, GOAL`; never `GOAL KICK`). A goal is confirmed when the other team kicks off within `p1_goal_kickoff_window_s`. Red cards are `CARD` events with `RED` or `SECOND YELLOW`.
- **Game state without a logged feed:** from an uploaded JSON (`final_score`, `goals` and `red_cards` with team, period and `clock_s`), validated against the final score.
- **Chains:** a team's stretch of possession; contested and unknown points do not break it. A chain ends when the other team takes the ball, at a dead ball or at the end of a period. Start types are `period_start`, `set_piece:<type>`, `restart`, `recovery` (a recovery event at the start) and `won`. End types are `lost`, `dead_ball`, `period_end`, `shot` and `goal` (a shot by the team just before the end).

**Data check.**

- **Possession:** `full` when the share of in-play time not observed is at most `p1_degraded_inferred_share`, otherwise `degraded`. The measure lists observed, inferred, events and unknown shares; coverage is the share of match time labelled.
- **Game state:**
  - `full` with a logged feed and every goal confirmed;
  - `degraded` with unconfirmed goals, an upload without dead-ball information, or a final score only;
  - `blocked` when nothing is known.

**Known limits.**

- **Inference:** possession during long detection gaps falls back to the event feed. Inference from player movement is not implemented yet.
- **Derived feeds:** SkillCorner's feed is derived from tracking, so dead ball is unknown and counted as in play.
- **Precision:** the control radius is tuned on 25 Hz Metrica data; broadcast tracking agrees better with a larger radius.

## In the semantic events (schema 1.1)

`possession_chain_id` (`{match_id}_chNNNN`), `possession_source`, `score_diff` and `red_card_diff` for the actor's team, at the action's start frame. All are labels or small integers, never coordinates; all are null without context.
