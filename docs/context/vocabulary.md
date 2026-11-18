# Context vocabulary (P0 to P4)

Definitions of every field the context packages produce, as built on 17 November 2026. Package versions: `probe` and `p3` 0.1.0, `p1` and `p2` 0.2.0, `p4` 0.3.0 (18 November 2026); thresholds 0.9.0 (every Stage A and B value agreed by the founder, the last four on 18 November 2026; `p1_loose_on_opponent_touch` proposed and off). The specification is `ponta-blueprint/context-packages.md` (Stage B: `ponta-blueprint/context-structure-task.md`); the measurements behind the thresholds are in `CONTEXT_FOUNDATIONS_REPORT.md` (P1) and `CONTEXT_STRUCTURE_REPORT.md` (P2 to P4).

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
| `p1_possession` | label | estimated | `home`, `away`, `contested`, `dead_ball` or `unknown`; `loose` only when `p1_loose_on_opponent_touch` is on (off by default) |
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
- **Chains:** a team's stretch of possession; contested and unknown points do not break it. A chain ends when the other team takes the ball, at a dead ball or at the end of a period. Start types are `period_start`, `set_piece:<type>`, `restart`, `recovery` (a recovery event at the start) and `won`. Since P1 0.2.0 a chain starts with a set piece only when its team took it; the other team winning the first touch (a corner headed clear) starts `won`. End types are `lost`, `dead_ball`, `period_end`, `shot` and `goal` (a shot by the team just before the end).

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

## Team frame (P2 to P4)

Every Stage B measure is taken in a team's own frame: **depth** is metres from its own goal line towards the opponent's, **width** is metres from the centre line with the team's left positive. Each team's direction of attack is read per period from the side its goalkeeper defends (the player with the largest mean distance from halfway), not assumed from the provider convention (it caught Metrica Sample Game 2 normalised the wrong way round, since fixed in the loader). The goalkeeper in each two-minute block is the team's deepest player on average; every other player is outfield.

## P2 Team structure

Per team (`home`, `away`), on every grid point; null when fewer than `p2_min_outfield_players` outfield players are observed.

| Field | Unit | Kind | Meaning |
|---|---|---|---|
| `p2_<team>_outfield_n` | players | observed | Outfield players observed |
| `p2_<team>_length_m` | m | observed | Deepest to highest outfield player (depth) |
| `p2_<team>_width_m` | m | observed | Widest to widest outfield player |
| `p2_<team>_centroid_depth_m` | m | observed | Mean outfield depth |
| `p2_<team>_stretch_m` | m | observed | Stretch index: mean distance to the outfield centroid |
| `p2_<team>_hull_m2` | m² | observed | Convex-hull area of the outfield players |
| `p2_<team>_def_line_m` | m | observed | Defensive line height: the deepest outfield player's depth |
| `p2_<team>_mid_line_m`, `p2_<team>_att_line_m` | m | estimated | The deepest member of the midfield and attacking lines |
| `p2_<team>_def_mid_gap_m`, `p2_<team>_mid_att_gap_m` | m | estimated | Depth between line centroids |
| `p2_<team>_formation_in`, `p2_<team>_formation_out` | label | estimated | The team's current formation in and out of possession |

**Rules.**

- **Windows:** two minutes (`p2_window_s`) within a period, split by the team's possession state (`in`, `out`, and `all` as a fallback). Only P1 chains of at least `p2_min_possession_s` count; a window needs `p2_min_window_data_s` of data, and a player counts when seen for `p2_min_presence_share` of it.
- **Lines:** exact one-dimensional 3-means on the players' mean depth: the split into three contiguous groups with the least within-group sum of squares.
- **Formation:** the players' mean positions, centred and scaled per axis, are assigned to each template's slots with the Hungarian algorithm (templates: 4-4-2, 4-3-3, 4-2-3-1, 4-1-4-1, 4-1-2-1-2, 3-5-2, 3-4-3, 3-4-2-1, 5-3-2, 5-4-1). The window's formation is the best fit, or `unclear` when it beats the second best by less than `p2_formation_min_margin`. The current formation changes only when a new one holds for `p2_formation_persist_s`.
- **Roles:** each player's slot in the current formation's template (`LB`, `LCB`, `CB`, `RCB`, `RB`, `LWB`, `RWB`, `DM`, `LDM`, `RDM`, `LCM`, `CM`, `RCM`, `LM`, `RM`, `LAM`, `RAM`, `AM`, `LW`, `RW`, `CF`, `LCF`, `RCF`). Ponta's `LM` and `RM` are the wide midfielders of a flat four (SkillCorner's `LM` and `RM` are a central pair).
- **Main role** (P2 0.2.0): each player's role for the match, the slot held for the most window data out of possession, where a team keeps its nominal shape; all windows when the player has no out-of-possession window. In possession a 4-4-2 often fits as a 4-2-3-1, so a forward who drops reads as `AM` in those windows.

**Segments.** `p2_formations` (team, period, block, state, data seconds, best and second template, margin, window formation, current formation), `p2_roles` (team, period, block, state, player id, line, role), `p2_main_roles` (team, player id, role, the state read, seconds in that role, share of the player's data in that state), `p2_orientation` (attack sign per team and period), `p2_goalkeepers`. No positions are written.

**Data check.** `team_shape` and `formation`: `full` when both teams have a shape for at least `p2_min_shape_coverage` of in-play time and the orientation is consistent; `degraded` below it (broadcast tracking loses players out of view); `blocked` with no shape at all.

## P3 Relative zones

Measured for the team in possession (P1) against the team out of possession, on every grid point where one team has the ball.

| Field | Unit | Kind | Meaning |
|---|---|---|---|
| `p3_attacking_team` | label | estimated | The team in possession |
| `p3_ball_lane` | label | estimated | `left_wing`, `left_half_space`, `centre`, `right_half_space`, `right_wing` in the attacking team's frame: wings beyond the penalty-area edges (20.16 m from the centre line), half-spaces between those and the goal-area edges (9.16 m), centre inside |
| `p3_ball_third` | label | estimated | `defensive`, `middle`, `attacking` along the attacking direction |
| `p3_ball_band` | label | estimated | Against the defending lines (depth from the defending goal line): `behind_defence` (beyond the deepest defender), `in_defensive_line` (up to the defensive line's centroid), `between_lines` (defensive to midfield centroid), `between_mid_att`, `in_front` |
| `p3_ball_in_hull` | bool | estimated | Inside the defending outfield players' convex hull |
| `p3_ball_in_pocket` | bool | estimated | Between the lines, in a half-space and inside the defending hull (Ponta's definition; none is published) |
| `p3_att_behind_defence`, `p3_att_between_lines`, `p3_att_in_pocket` | players | estimated | Attacking outfield players in each zone |

**Data check.** `relative_zones` inherits P2's state and is `degraded` when the ball or the defending shape is missing for more than `1 − p3_min_coverage` of possession time.

## P4 Phase of play

| Field | Unit | Kind | Meaning |
|---|---|---|---|
| `p4_team` | label | estimated | The team in possession |
| `p4_phase` | label | estimated | `build_up`, `create`, `finish`, `direct`, `quick_break`, `transition`, `set_play`, `chaotic` |
| `p4_out_phase` | label | estimated | The paired label for the other team: `high_block`, `medium_block`, `low_block`, `defending_direct`, `defending_quick_break`, `defending_transition`, `defending_set_play`, `chaotic` |
| `p4_block` | label | estimated | The defending team's block, `low`, `medium` or `high`, whatever the phase |
| `p4_defending_depth_m` | m | observed | Mean depth of the defending team's three deepest outfield players |
| `p4_phase_id` | id | estimated | The phase segment |

**Rules** (within each P1 chain, highest priority first; the vocabulary follows SkillCorner's published phases, the rules and numbers are Ponta's):

1. **Set play:** a chain starting with a logged corner, or a free kick in the attacking half, for up to `p4_set_play_max_s`. Without a logged feed, only corners, found from player positions (the founder's rule, P4 0.2.0): one attacker within `p4_corner_setup_radius_m` of a corner flag and no defender, while at least `p4_corner_setup_crowd_min` players of both teams are in that penalty area, for at least `p4_corner_setup_hold_s`; the set play runs from the end of the set-up, and since P4 0.3.0 the set-up itself is `set_play` for the taking team. With `p4_corner_setup_crowd_min` at 0 the older ball rule is used (the ball within `p4_corner_radius_m` of a flag in the chain's first `p4_corner_window_s`).
2. **Direct:** from the own half, the ball gains `p4_direct_min_gain_m` towards goal within `p4_direct_window_s`.
3. **Chaotic:** a chain shorter than `p4_chaotic_max_s`, or contested possession.
4. **Quick break or transition:** a chain won from the other team in open play (regain in the opponent's half or the own half; since P4 0.3.0, without a logged feed, not a chain that starts with its team's throw-in found from positions) where the ball gains `p4_break_min_gain_m` within `p4_break_eval_s`. It ends when `p4_recovered_players` defenders are goal-side of the ball, at the chain's end, after `p4_break_max_s`, or (P4 0.2.0) when a break that began further back reaches `p4_break_end_depth_share` of the pitch (the final third).
5. **Established play:** the defending team's block from `p4_defending_depth_m`: below `p4_low_block_max_m` low (`finish`), at or above `p4_high_block_min_m` high (`build_up`), otherwise medium (`create`). Since P4 0.2.0 the ball's position refines it: `finish` needs the ball past `p4_finish_min_depth_share` of the pitch (the attacking half), otherwise `create`; with the ball inside `p4_build_up_max_depth_share` (the own third) the phase is `build_up` whatever the block.

Runs shorter than `p4_min_phase_s` merge into the run before them in the same chain.

**Restarts without a logged feed (P4 0.3.0).** Throw-ins are found from positions: exactly one player of the team (and none of the other) at least `p4_throw_in_out_m` beyond a touchline, away from the corners, for at least `p4_throw_in_hold_s`; in the set-up's last 2 s the median speed of all players is at most `p4_throw_in_still_mps` and the ball comes within `p4_throw_in_ball_m` of that player. While a throw-in set-up is under way the ball is dead and no phase is labelled. Goal kicks and free kicks are not found.

**Segments.** `p4_phases` (phase id, period, team, phase, start, end, duration, chain id); `p4_restarts` (type, team, period, time, source: `events` from the logged feed, or `tracking` for corners and throw-ins found from positions).

**Data check.** `phase_of_play`: `degraded` when team shape is degraded, when more than `p1_degraded_inferred_share` of labelled time rests on inferred possession, or without a logged event feed (only corners are found as set plays, and only corners and throw-ins as restarts); coverage is the share of in-play time labelled.

**Known limits (Stage B).**

- **Validation:** formations, roles, zones and phases are estimated, not yet validated by analysts (Gate 2b). Reference agreement is in `CONTEXT_STRUCTURE_REPORT.md`.
- **Roles:** after P2 0.2.0 most remaining disagreements with SkillCorner are a central midfielder given a defensive-midfield slot, and a forward who drops given an attacking-midfield slot (`CONTEXT_STRUCTURE_REPORT.md`).
- **Restarts without events:** the position rules find about 70% of corners (about 55% of detections right) and about 55% of throw-ins on broadcast tracking (about 61% right; 87% right on Metrica's full-pitch tracking). Goal kicks and free kicks are not found; most goal kicks have a gap in broadcast tracking before them.
- **Runtime:** P3 and P4 rebuild the team frames from P2's segments; about 13 s of the 16 s per match is Stage B.

## In the semantic events (schema 1.1)

`possession_chain_id` (`{match_id}_chNNNN`), `possession_source`, `score_diff` and `red_card_diff` for the actor's team, at the action's start frame. All are labels or small integers, never coordinates; all are null without context.
