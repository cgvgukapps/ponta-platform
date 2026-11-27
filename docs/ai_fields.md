# AI field dictionary (version 1.1.0)

> Generated from `packages/plaiq_engine/src/plaiq_engine/field_dictionary.py` by `tools/field_dictionary.py`; do not edit by hand. Figures are measured on a development set, not held out. Only observed values are player-facing until an analyst validates an estimated one (Gate 2b).

## semantic_events

| Column | Definition | Unit | Kind | Validation | Data check | Quotable | Player-facing |
|---|---|---|---|---|---|---|---|
| `schema_version` | Semantic schema version of the row. |  | metadata |  |  | no | no |
| `engine_version` | Plaiq and Matrix versions that produced the row. |  | metadata |  |  | no | no |
| `event_id` | Opaque event id, derived from the event's content; cited, never quoted. |  | identifier |  |  | no | no |
| `match_id` | Opaque match key (m1_...), never the provider's id. |  | identifier |  |  | no | no |
| `event_idx` | The loader's row position; internal, for joins only. |  | identifier |  |  | no | no |
| `timestamp_s` | Match clock at the action's start. | s | observed |  |  | yes | yes |
| `minute_display` | The match minute as mm:ss. |  | observed |  |  | yes | yes |
| `period` | Period of play (1, 2; 3 and 4 in extra time). |  | observed |  |  | yes | yes |
| `actor_player_ref` | The acting player as side and shirt number (home_10); the client adds the name from the club's roster. |  | identifier |  |  | no | no |
| `actor_team` | home or away. |  | observed |  |  | yes | yes |
| `actor_role` | The actor's role: the club's roster where given, else P2's main role for the match. |  | estimated | agrees with reference data (roles agree with SkillCorner's positions 83% grouped, 90% merged, pooled over 583 players (measured on a development set; CONTEXT_STRUCTURE_REPORT.md)) | p2.formation | yes | no |
| `actor_role_source` | Where actor_role came from: roster or estimated. |  | metadata |  |  | no | no |
| `action_type` | pass, carry, shot, turnover, defensive_action, set_piece, foul_won, card, or run (run rows are P6's estimate). |  | observed |  |  | yes | yes |
| `action_outcome` | completed or lost (passes, carries), won or lost (defensive actions), the shot's outcome where the feed gives one. |  | observed |  |  | yes | yes |
| `action_detail` | The feed's subtype in Ponta's words: header, set-piece kind, dispossessed, clearance, offside, card colour. |  | observed |  |  | yes | yes |
| `target_player_ref` | The logged receiver of a pass (side and shirt number). |  | observed |  |  | yes | yes |
| `inferred_intended_ref` | The engine's inferred intended receiver of a failed pass. |  | estimated | not yet measured | decision_quality | yes | no |
| `inference_confidence` | How strongly the intended receiver is inferred (weak, medium, strong). | category | estimated | not yet measured | decision_quality | yes | no |
| `dq_score` | Gain (decision quality version 1): forward movement on the threat surface relative to the passer's position. | 0 to 1.5 | estimated | measured against outcomes (measured against outcomes and providers' analogues, not against analysts; construct validity untested; the passer is the best option in about 65% of scored passes (PLAIQ_REFERENCES_REPORT.md)) | decision_quality | yes | no |
| `dq_band` | Gain's band: low_gain, partial_gain, most_gain, full_gain, or not_scored. | category | estimated | measured against outcomes (measured against outcomes and providers' analogues, not against analysts; construct validity untested; the passer is the best option in about 65% of scored passes (PLAIQ_REFERENCES_REPORT.md)) | decision_quality | yes | no |
| `chosen_option_pov` | Pass option value of the option the passer chose. | value | estimated | agrees in part (availability predicts completion at AUC 0.77; value tracks SkillCorner's threat at 0.78 (measured on a development set; PLAIQ_REFERENCES_REPORT.md)) | pass_option_value | yes | no |
| `best_option_player_ref` | The team-mate with the highest pass option value at release. |  | estimated | agrees in part (availability predicts completion at AUC 0.77; value tracks SkillCorner's threat at 0.78 (measured on a development set; PLAIQ_REFERENCES_REPORT.md)) | pass_option_value | yes | no |
| `best_option_pov` | The highest pass option value on offer at release. | value | estimated | agrees in part (availability predicts completion at AUC 0.77; value tracks SkillCorner's threat at 0.78 (measured on a development set; PLAIQ_REFERENCES_REPORT.md)) | pass_option_value | yes | no |
| `xt_delta` | Change in threat (value surface) from the action's start to its end. | threat | estimated | agrees in part (the value surface tracks SkillCorner's threat at 0.78 (measured on a development set; PLAIQ_REFERENCES_REPORT.md)) | event_delta_v | yes | no |
| `movement_value` | Off-ball movement value (null in this table since schema 2.1; see movement_values). | value | estimated | not yet measured | movement_value | yes | no |
| `opposition_pressure` | 1v1 pressure on the actor at the action's start. | 1 to 10 | estimated | agrees with reference data (1v1 pressure against SkillCorner's five pressure levels: weighted kappa 0.80 (measured on a development set; PLAIQ_REFERENCES_REPORT.md)) | pressure_1v1 | yes | no |
| `pitch_zone` | Channel and third of the action's start, in the actor's attacking direction. | category | observed |  |  | yes | yes |
| `mode` | Analysis mode (full_tracking). |  | metadata |  |  | no | no |
| `pass_completion_probability` | Estimated probability the pass is completed. | 0 to 1 | estimated | agrees with reference data (log-loss 0.327, AUC 0.843 on SkillCorner's matches (SkillCorner's own model 0.894; measured on a development set; PLAIQ_REFERENCES_REPORT.md)) | pass_completion | yes | no |
| `pass_completion_status` | estimated, or why the pass has no completion probability. |  | metadata |  |  | no | no |
| `pass_completion_parameters` | The fitted parameter set used. |  | metadata |  |  | no | no |
| `dq_v2_score` | Choice (decision quality version 2): how the chosen option compared with the best on offer. | 0 to 1 | estimated | measured against outcomes (measured against outcomes and SkillCorner's analogue, not against analysts (PLAIQ_REFERENCES_REPORT.md)) | decision_quality_v2 | yes | no |
| `dq_v2_band` | Choice's band: best_option, near_best, below_best, well_below_best (never Gain's names). | category | estimated | measured against outcomes (measured against outcomes and SkillCorner's analogue, not against analysts (PLAIQ_REFERENCES_REPORT.md)) | decision_quality_v2 | yes | no |
| `dq_v2_status` | estimated, or why Choice is not scored. |  | metadata |  |  | no | no |
| `dq_v2_best_option_ref` | The best option by Choice (side and shirt number). |  | estimated | measured against outcomes (measured against outcomes and SkillCorner's analogue, not against analysts (PLAIQ_REFERENCES_REPORT.md)) | decision_quality_v2 | yes | no |
| `dq_v2_version` | Choice's version. |  | metadata |  |  | no | no |
| `gain_v2_score` | Gain version 2: the realised value of the pass over the best option's expected value. | 0 to 1.5 | estimated | measured against outcomes (measured against outcomes and SkillCorner's analogue, not against analysts (PLAIQ_REFERENCES_REPORT.md)) | gain_v2 | yes | no |
| `gain_v2_band` | Gain version 2's band, on version 1's cut-offs and names. | category | estimated | measured against outcomes (measured against outcomes and SkillCorner's analogue, not against analysts (PLAIQ_REFERENCES_REPORT.md)) | gain_v2 | yes | no |
| `gain_v2_status` | estimated, or why Gain version 2 is not scored. |  | metadata |  |  | no | no |
| `gain_v2_version` | Gain version 2's version. |  | metadata |  |  | no | no |
| `not_scored` | True when the decision is deliberately not scored (goalkeeper, set piece, lost pass for Gain). |  | metadata |  |  | yes | no |
| `not_scored_reason` | Why it is not scored. |  | metadata |  |  | yes | no |
| `run_type` | A run's type in SkillCorner's vocabulary (run rows only). | category | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `run_area` | Where the run ends against the defending units: in_front, in_between or in_behind (FIFA's categories). | category | estimated | not yet measured | p6.movement | yes | no |
| `physiological_context` | Null: no physiological source exists. |  | metadata |  |  | no | no |
| `ball_detected` | Whether the ball was detected at the action's start. |  | observed |  |  | yes | yes |
| `players_observed` | Players the provider observed (not estimated) at the action's start. | players | observed |  |  | yes | yes |
| `observed_near_ball_share` | Share of positions within 15 m of the ball the provider observed at the action's start (null where it does not flag estimates). | share | observed |  |  | yes | yes |
| `parity_spatial` | Data check state of decision quality for this match. |  | metadata |  |  | yes | no |
| `parity_physiological` | Data check state of load and fatigue for this match. |  | metadata |  |  | yes | no |
| `possession_chain_id` | The possession chain (P1), built on the match key. |  | identifier |  |  | no | no |
| `possession_source` | How P1 assigned possession: observed, inferred, events or unknown. |  | metadata |  |  | no | no |
| `score_diff` | Actor's team score minus the opponent's, where the feed gives the running score. | goals | observed |  | p1.game_state | yes | yes |
| `red_card_diff` | Actor's team red cards minus the opponent's. | cards | observed |  | p1.game_state | yes | yes |
| `state_id` | The episode (P8) the action belongs to. |  | identifier |  |  | no | no |
| `phase_team` | The team in possession by P4. |  | estimated | agrees with reference data (82% to 89% agreement with the German league's official frame-level possession (CONTEXT_FOUNDATIONS_REPORT.md)) | p1.possession | yes | no |
| `phase` | Phase of play of the team in possession (SkillCorner's vocabulary). | category | estimated | agrees with reference data (phase of play against SkillCorner's phases: kappa 0.66 / 0.63 by match half, Gate 2a met (measured on a development set; CONTEXT_STRUCTURE_REPORT.md)) | p4.phase_of_play | yes | no |
| `out_phase` | The defending team's own block in established play, else the pair of the attacking phase. | category | estimated | not yet measured | p4.phase_of_play | yes | no |
| `fifa_phase` | FIFA's name for the phase, with build-up opposed or unopposed. | category | estimated | agrees in part (build-up opposed against SkillCorner's engagements: kappa 0.52 after one fix (0.36 before it; measured on a development set)) | p4v.fifa_view | yes | no |
| `fifa_out_phase` | FIFA's name for the out-of-possession phase (press against block). | category | estimated | not yet measured | p4v.fifa_view | yes | no |
| `block` | The defending block (high, mid, low) by P4's depth bands. | category | estimated | not yet measured | p4.phase_of_play | yes | no |
| `formation_in` | The actor's team's formation in possession. | category | estimated | agrees with reference data (formations agree with SkillCorner's on 20 of 20 matches (measured on a development set; CONTEXT_STRUCTURE_REPORT.md)) | p2.formation | yes | no |
| `formation_out` | The opponent's formation out of possession. | category | estimated | agrees with reference data (formations agree with SkillCorner's on 20 of 20 matches (measured on a development set; CONTEXT_STRUCTURE_REPORT.md)) | p2.formation | yes | no |
| `zone_lane` | The ball's lane relative to the defending shape. | category | estimated | not yet measured | p3.relative_zones | yes | no |
| `zone_third` | The ball's third relative to the defending shape. | category | estimated | not yet measured | p3.relative_zones | yes | no |
| `zone_band` | The ball's band between the defending lines (team in possession only). | category | estimated | not yet measured | p3.relative_zones | yes | no |
| `units_broken` | Defending units the pass broke. | units | estimated | agrees with reference data (line breaks against SkillCorner's labels: kappa 0.56 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p5.line_breaks | yes | no |
| `furthest_line_broken` | The furthest defending line the pass broke. | category | estimated | agrees with reference data (line breaks against SkillCorner's labels: kappa 0.56 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p5.line_breaks | yes | no |
| `received_inside_shape` | Whether the pass was received inside the defending shape. |  | estimated | not yet measured | p5.relationships | yes | no |
| `goal_side_defenders` | Defenders goal-side of the ball. | players | estimated | agrees with reference data (goal-side defenders agree with SkillCorner's 81% (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p5.relationships | yes | no |
| `overload_at_ball` | Attackers minus defenders near the ball. | players | estimated | not yet measured | p5.relationships | yes | no |
| `runs_in_behind` | Runs in behind under way. | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `adjusted_choice` | Choice adjusted for the game state and the cost of losing the ball. | 0 to 1 | estimated | measured against outcomes (at half cost weight: shot AUC 0.58, completion AUC 0.63 on SkillCorner's matches; the game-state weights are a judgement (CONTEXT_RELATIONSHIPS_REPORT.md section 5)) | p7.adjusted_choice | yes | no |
| `open_high_value_share` | Open share of the team's high-value space. | share | estimated | agrees in part (no direct label; orders by SkillCorner's phase as expected (build-up 0.24 to set plays 0.71; CONTEXT_RELATIONSHIPS_REPORT.md)) | p7.space | yes | no |
| `engaging_defenders` | Defenders engaging the carrier (P4v). | players | estimated | agrees in part (58% of SkillCorner's engagements found at 65% precision, type kappa 0.54; counter-press 51% found (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md section 6)) | p4v.engagements | yes | no |
| `context_versions` | Context package and threshold versions. |  | metadata |  |  | no | no |
| `actor_team_key` | Opaque key of the actor's team (t1_...), stable across the club's matches. |  | identifier |  |  | no | no |
| `opponent_team_key` | Opaque key of the opponent. |  | identifier |  |  | no | no |
| `competition` | Opaque competition key (c1_...). |  | identifier |  |  | no | no |
| `season` | Opaque season key (s1_...). |  | identifier |  |  | no | no |

## semantic_states

| Column | Definition | Unit | Kind | Validation | Data check | Quotable | Player-facing |
|---|---|---|---|---|---|---|---|
| `state_id` | Opaque episode id, built on the match key. |  | identifier |  |  | no | no |
| `match_id` | Opaque match key. |  | identifier |  |  | no | no |
| `period` | Period of play. |  | observed |  |  | yes | yes |
| `start_s` | Episode start. | s | estimated | agrees with reference data (78.5% of SkillCorner's phase boundaries found within 2 s, phase kappa 0.65 (measured on a development set; CONTEXT_STATES_REPORT.md)) | p8.episodes | yes | no |
| `end_s` | Episode end. | s | estimated | agrees with reference data (78.5% of SkillCorner's phase boundaries found within 2 s, phase kappa 0.65 (measured on a development set; CONTEXT_STATES_REPORT.md)) | p8.episodes | yes | no |
| `duration_s` | Episode length. | s | estimated | agrees with reference data (78.5% of SkillCorner's phase boundaries found within 2 s, phase kappa 0.65 (measured on a development set; CONTEXT_STATES_REPORT.md)) | p8.episodes | yes | no |
| `minute_display` | Episode start as mm:ss. |  | estimated | agrees with reference data (78.5% of SkillCorner's phase boundaries found within 2 s, phase kappa 0.65 (measured on a development set; CONTEXT_STATES_REPORT.md)) | p8.episodes | yes | no |
| `team` | The team in possession. |  | estimated | agrees with reference data (82% to 89% agreement with the German league's official frame-level possession (CONTEXT_FOUNDATIONS_REPORT.md)) | p1.possession | yes | no |
| `chain_id` | The possession chain. |  | identifier |  |  | no | no |
| `start_reason` | How the episode started (won the ball, goalkeeper, set piece, restart, kick-off, continued). | category | estimated | not yet measured | p8.episodes | yes | no |
| `start_detail` | The start's detail (set-piece kind). | category | estimated | not yet measured | p8.episodes | yes | no |
| `cleared` | Whether the team made a logged clearance in the episode. |  | observed |  |  | yes | yes |
| `share_controlled` | Share of the episode with the ball controlled (P1's ball state). | share | estimated | not yet measured | p1.possession_state | yes | no |
| `share_in_flight` | Share of the episode with the ball in flight (P1's ball state). | share | estimated | not yet measured | p1.possession_state | yes | no |
| `share_loose_or_contested` | Share of the episode with the ball loose or contested (P1's ball state). | share | estimated | not yet measured | p1.possession_state | yes | no |
| `score_diff` | The team's score minus the opponent's. | goals | observed |  | p1.game_state | yes | yes |
| `red_card_diff` | The team's red cards minus the opponent's. | cards | observed |  | p1.game_state | yes | yes |
| `game_state_source` | Where the score came from (running or final_only). |  | metadata |  |  | no | no |
| `phase` | The episode's phase of play. | category | estimated | agrees with reference data (78.5% of SkillCorner's phase boundaries found within 2 s, phase kappa 0.65 (measured on a development set; CONTEXT_STATES_REPORT.md)) | p4.phase_of_play | yes | no |
| `out_phase` | The defending team's phase or block. | category | estimated | not yet measured | p4.phase_of_play | yes | no |
| `fifa_phase` | FIFA's name for the phase. | category | estimated | agrees in part (build-up opposed against SkillCorner's engagements: kappa 0.52 after one fix (0.36 before it; measured on a development set)) | p4v.fifa_view | yes | no |
| `fifa_out_phase` | FIFA's name for the out-of-possession phase. | category | estimated | not yet measured | p4v.fifa_view | yes | no |
| `block` | The defending block. | category | estimated | not yet measured | p4.phase_of_play | yes | no |
| `opposed_pct` | Share of build-up time that was opposed. | percent | estimated | agrees in part (build-up opposed against SkillCorner's engagements: kappa 0.52 after one fix (0.36 before it; measured on a development set)) | p4v.fifa_view | yes | no |
| `press_share` | Share of the episode inside a press (P4v). | share | estimated | agrees in part (58% of SkillCorner's engagements found at 65% precision, type kappa 0.54; counter-press 51% found (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md section 6)) | p4v.engagements | yes | no |
| `counter_press_share` | Share of the episode inside a counter-press (P4v). | share | estimated | agrees in part (58% of SkillCorner's engagements found at 65% precision, type kappa 0.54; counter-press 51% found (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md section 6)) | p4v.engagements | yes | no |
| `fifa_mid_block_share` | Share of the episode meeting FIFA's mid block rule. | share | estimated | not yet measured | p4v.fifa_block | yes | no |
| `fifa_low_block_share` | Share of the episode meeting FIFA's low block rule. | share | estimated | not yet measured | p4v.fifa_block | yes | no |
| `formation_in` | Formation in possession. | category | estimated | agrees with reference data (formations agree with SkillCorner's on 20 of 20 matches (measured on a development set; CONTEXT_STRUCTURE_REPORT.md)) | p2.formation | yes | no |
| `formation_out` | Formation out of possession. | category | estimated | agrees with reference data (formations agree with SkillCorner's on 20 of 20 matches (measured on a development set; CONTEXT_STRUCTURE_REPORT.md)) | p2.formation | yes | no |
| `team_length_m` | The team's length (median over the episode). | m | estimated | not yet measured | p2.team_shape | yes | no |
| `team_width_m` | The team's width (median over the episode). | m | estimated | not yet measured | p2.team_shape | yes | no |
| `opp_length_m` | The opponent's length (median over the episode). | m | estimated | not yet measured | p2.team_shape | yes | no |
| `opp_width_m` | The opponent's width (median over the episode). | m | estimated | not yet measured | p2.team_shape | yes | no |
| `opp_def_line_m` | The opponent's defensive line height (median over the episode). | m | estimated | not yet measured | p2.team_shape | yes | no |
| `opp_def_mid_gap_m` | Gap between the opponent's defensive and midfield lines (median over the episode). | m | estimated | not yet measured | p2.team_shape | yes | no |
| `opp_mid_att_gap_m` | Gap between the opponent's midfield and attacking lines (median over the episode). | m | estimated | not yet measured | p2.team_shape | yes | no |
| `start_lane` | The ball's lane at the episode's start, relative to the defending shape. | category | estimated | not yet measured | p3.relative_zones | yes | no |
| `start_third` | The ball's third at the episode's start, relative to the defending shape. | category | estimated | not yet measured | p3.relative_zones | yes | no |
| `end_lane` | The ball's lane at the episode's end, relative to the defending shape. | category | estimated | not yet measured | p3.relative_zones | yes | no |
| `end_third` | The ball's third at the episode's end, relative to the defending shape. | category | estimated | not yet measured | p3.relative_zones | yes | no |
| `share_defensive_third` | Share of the episode with the ball in the defensive third. | share | estimated | not yet measured | p3.relative_zones | yes | no |
| `share_middle_third` | Share of the episode with the ball in the middle third. | share | estimated | not yet measured | p3.relative_zones | yes | no |
| `share_attacking_third` | Share of the episode with the ball in the attacking third. | share | estimated | not yet measured | p3.relative_zones | yes | no |
| `passes` | Passes in the episode. | passes | observed |  |  | yes | yes |
| `passes_completed` | Completed passes in the episode. | passes | observed |  |  | yes | yes |
| `passes_breaking_a_line` | Passes that broke a defending line. | passes | estimated | agrees with reference data (line breaks against SkillCorner's labels: kappa 0.56 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p5.line_breaks | yes | no |
| `furthest_line_broken` | The furthest line broken. | category | estimated | agrees with reference data (line breaks against SkillCorner's labels: kappa 0.56 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p5.line_breaks | yes | no |
| `carrier_pressure_median` | Carrier pressure (median). | 1 to 10 | estimated | agrees with reference data (carrier pressure against SkillCorner's pressure levels: weighted kappa 0.76 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p5.relationships | yes | no |
| `carrier_pressure_max` | Carrier pressure (max). | 1 to 10 | estimated | agrees with reference data (carrier pressure against SkillCorner's pressure levels: weighted kappa 0.76 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p5.relationships | yes | no |
| `goal_side_defenders_median` | Defenders goal-side of the ball (median). | players | estimated | agrees with reference data (goal-side defenders agree with SkillCorner's 81% (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p5.relationships | yes | no |
| `overload_at_ball_median` | Attackers minus defenders near the ball (median). | players | estimated | not yet measured | p5.relationships | yes | no |
| `runs` | Runs (P6). | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `runs_in_behind` | Runs (P6): runs in behind. | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `runs_behind` | Runs (P6): runs behind. | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `runs_coming_short` | Runs (P6): runs coming short. | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `runs_dropping_off` | Runs (P6): runs dropping off. | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `runs_support` | Runs (P6): runs support. | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `runs_overlap` | Runs (P6): runs overlap. | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `runs_underlap` | Runs (P6): runs underlap. | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `runs_pulling_wide` | Runs (P6): runs pulling wide. | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `runs_pulling_half_space` | Runs (P6): runs pulling half space. | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `runs_run_ahead_of_the_ball` | Runs (P6): runs run ahead of the ball. | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `runs_cross_receiver` | Runs (P6): runs cross receiver. | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `runs_carry` | Runs (P6): runs carry. | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `defensive_runs` | Runs (P6): defensive runs. | runs | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `open_high_value_share` | Open share of the team's high-value space (median). | share | estimated | agrees in part (no direct label; orders by SkillCorner's phase as expected (build-up 0.24 to set plays 0.71; CONTEXT_RELATIONSHIPS_REPORT.md)) | p7.space | yes | no |
| `opp_open_high_value_share` | Open share of the opponent's high-value space (median). | share | estimated | agrees in part (no direct label; orders by SkillCorner's phase as expected (build-up 0.24 to set plays 0.71; CONTEXT_RELATIONSHIPS_REPORT.md)) | p7.space | yes | no |
| `open_danger_zone` | The most dangerous open zone (lane and third). | category | estimated | not yet measured | p7.space | yes | no |
| `engagements` | Defensive engagements (P4v). | count | estimated | agrees in part (58% of SkillCorner's engagements found at 65% precision, type kappa 0.54; counter-press 51% found (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md section 6)) | p4v.engagements | yes | no |
| `engagements_recovery` | Defensive engagements (P4v): engagements recovery. | count | estimated | agrees in part (58% of SkillCorner's engagements found at 65% precision, type kappa 0.54; counter-press 51% found (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md section 6)) | p4v.engagements | yes | no |
| `engagements_counter_press` | Defensive engagements (P4v): engagements counter press. | count | estimated | agrees in part (58% of SkillCorner's engagements found at 65% precision, type kappa 0.54; counter-press 51% found (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md section 6)) | p4v.engagements | yes | no |
| `engagements_press` | Defensive engagements (P4v): engagements press. | count | estimated | agrees in part (58% of SkillCorner's engagements found at 65% precision, type kappa 0.54; counter-press 51% found (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md section 6)) | p4v.engagements | yes | no |
| `engagements_pressure` | Defensive engagements (P4v): engagements pressure. | count | estimated | agrees in part (58% of SkillCorner's engagements found at 65% precision, type kappa 0.54; counter-press 51% found (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md section 6)) | p4v.engagements | yes | no |
| `presses` | Defensive engagements (P4v): presses. | count | estimated | agrees in part (58% of SkillCorner's engagements found at 65% precision, type kappa 0.54; counter-press 51% found (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md section 6)) | p4v.engagements | yes | no |
| `counter_press` | Defensive engagements (P4v): counter press. | count | estimated | agrees in part (58% of SkillCorner's engagements found at 65% precision, type kappa 0.54; counter-press 51% found (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md section 6)) | p4v.engagements | yes | no |
| `end_reason` | How the episode ended (phase changed, possession lost, shot, goal, ball out of play ...). | category | estimated | not yet measured | p8.episodes | yes | no |
| `led_to_shot` | Whether the episode's possession chain ended in a shot or goal (chain level). |  | observed |  |  | yes | yes |
| `first_event_idx` | P8's first logged-event index; internal. |  | identifier |  |  | no | no |
| `last_event_idx` | P8's last logged-event index; internal. |  | identifier |  |  | no | no |
| `events` | Logged events in the episode. | events | observed |  |  | yes | yes |
| `ball_detected_share` | Share of the episode with the ball detected. | share | observed |  |  | yes | yes |
| `min_outfield_observed` | Fewest outfield players observed at any point in the episode. | players | observed |  |  | yes | yes |
| `data_check_p1_possession` | Data check state of p1 possession for this episode. |  | metadata |  |  | yes | no |
| `data_check_p1_possession_state` | Data check state of p1 possession_state for this episode. |  | metadata |  |  | yes | no |
| `data_check_p1_game_state` | Data check state of p1 game_state for this episode. |  | metadata |  |  | yes | no |
| `data_check_p2_team_shape` | Data check state of p2 team_shape for this episode. |  | metadata |  |  | yes | no |
| `data_check_p2_formation` | Data check state of p2 formation for this episode. |  | metadata |  |  | yes | no |
| `data_check_p3_relative_zones` | Data check state of p3 relative_zones for this episode. |  | metadata |  |  | yes | no |
| `data_check_p4_phase_of_play` | Data check state of p4 phase_of_play for this episode. |  | metadata |  |  | yes | no |
| `data_check_p4v_engagements` | Data check state of p4v engagements for this episode. |  | metadata |  |  | yes | no |
| `data_check_p4v_fifa_view` | Data check state of p4v fifa_view for this episode. |  | metadata |  |  | yes | no |
| `data_check_p4v_fifa_block` | Data check state of p4v fifa_block for this episode. |  | metadata |  |  | yes | no |
| `data_check_p5_line_breaks` | Data check state of p5 line_breaks for this episode. |  | metadata |  |  | yes | no |
| `data_check_p5_relationships` | Data check state of p5 relationships for this episode. |  | metadata |  |  | yes | no |
| `data_check_p6_movement` | Data check state of p6 movement for this episode. |  | metadata |  |  | yes | no |
| `data_check_p7_space` | Data check state of p7 space for this episode. |  | metadata |  |  | yes | no |
| `data_check_p7_adjusted_choice` | Data check state of p7 adjusted_choice for this episode. |  | metadata |  |  | yes | no |
| `data_check_p8_episodes` | Data check state of p8 episodes for this episode. |  | metadata |  |  | yes | no |
| `data_check` | The worst Data check state in the episode. |  | metadata |  |  | yes | no |
| `thresholds_version` | Thresholds version. |  | metadata |  |  | no | no |
| `versions` | Package versions. |  | metadata |  |  | no | no |
| `label` | estimated: every episode field is Ponta's estimate unless named observed here. |  | metadata |  |  | yes | no |
| `first_event_id` | First semantic event inside the episode. |  | identifier |  |  | no | no |
| `last_event_id` | Last semantic event inside the episode. |  | identifier |  |  | no | no |
| `gain_median` | Median Gain of the episode's passes. | 0 to 1.5 | estimated | measured against outcomes (measured against outcomes and providers' analogues, not against analysts; construct validity untested; the passer is the best option in about 65% of scored passes (PLAIQ_REFERENCES_REPORT.md)) | decision_quality | yes | no |
| `choice_median` | Median Choice of the episode's passes. | 0 to 1 | estimated | measured against outcomes (measured against outcomes and SkillCorner's analogue, not against analysts (PLAIQ_REFERENCES_REPORT.md)) | decision_quality_v2 | yes | no |
| `adjusted_choice_median` | Median adjusted Choice of the episode's passes. | 0 to 1 | estimated | measured against outcomes (at half cost weight: shot AUC 0.58, completion AUC 0.63 on SkillCorner's matches; the game-state weights are a judgement (CONTEXT_RELATIONSHIPS_REPORT.md section 5)) | p7.adjusted_choice | yes | no |
| `shots` | Shots inside the episode. | shots | observed |  |  | yes | yes |
| `ended_in_shot` | Whether a shot fell inside this episode (episode level, unlike led_to_shot). |  | observed |  |  | yes | yes |
| `team_key` | Opaque key of the team in possession. |  | identifier |  |  | no | no |
| `opp_team_key` | Opaque key of the opponent. |  | identifier |  |  | no | no |
| `competition` | Opaque competition key. |  | identifier |  |  | no | no |
| `season` | Opaque season key. |  | identifier |  |  | no | no |
| `summary` | A one-paragraph summary written by code from the episode's own fields (passes the field-bound number check). | text | estimated | not yet measured | p8.episodes | yes | no |
| `schema_version` | Semantic schema version. |  | metadata |  |  | no | no |

## movement_values

| Column | Definition | Unit | Kind | Validation | Data check | Quotable | Player-facing |
|---|---|---|---|---|---|---|---|
| `schema_version` | Semantic schema version. |  | metadata |  |  | no | no |
| `event_id` | The source event's id plus the player. |  | identifier |  |  | no | no |
| `source_event_id` | The event this movement value belongs to (resolves in semantic_events). |  | identifier |  |  | no | no |
| `match_id` | Opaque match key. |  | identifier |  |  | no | no |
| `timestamp_s` | The source event's start. | s | observed |  |  | yes | yes |
| `minute_display` | The source event's start as mm:ss. |  | observed |  |  | yes | yes |
| `period` | Period of play. |  | observed |  |  | yes | yes |
| `player_ref` | The moving team-mate (side and shirt number). |  | identifier |  |  | no | no |
| `team` | home or away. |  | observed |  |  | yes | yes |
| `movement_value` | Plaiq's off-ball movement value for the team-mate at this event. | value | estimated | not yet measured | movement_value | yes | no |
| `xt_delta` | Change in threat from the team-mate's movement. | threat | estimated | agrees in part (the value surface tracks SkillCorner's threat at 0.78 (measured on a development set; PLAIQ_REFERENCES_REPORT.md)) | movement_value | yes | no |
| `opposition_pressure` | 1v1 pressure on the team-mate. | 1 to 10 | estimated | agrees with reference data (1v1 pressure against SkillCorner's five pressure levels: weighted kappa 0.80 (measured on a development set; PLAIQ_REFERENCES_REPORT.md)) | pressure_1v1 | yes | no |
| `state_id` | The episode at the source event. |  | identifier |  |  | no | no |
| `label` | estimated. |  | metadata |  |  | yes | no |

## profiles

| Column | Definition | Unit | Kind | Validation | Data check | Quotable | Player-facing |
|---|---|---|---|---|---|---|---|
| `match_id` | Opaque match key. |  | identifier |  |  | no | no |
| `team` | home or away. |  | observed |  |  | yes | yes |
| `team_key` | Opaque team key. |  | identifier |  |  | no | no |
| `opp_team_key` | Opaque opponent key. |  | identifier |  |  | no | no |
| `competition` | Opaque competition key. |  | identifier |  |  | no | no |
| `season` | Opaque season key. |  | identifier |  |  | no | no |
| `measure` | The measure's name (see profile_measures). |  | metadata |  |  | yes | no |
| `value` | The measure's value; its validation is the measure's (profile_measures). | per measure | estimated | not yet measured |  | yes | no |
| `denominator` | What the value is per (per 90 minutes of ball in play, share of possession time ...). |  | metadata |  |  | yes | no |
| `sample_minutes` | Minutes the value rests on. | minutes | observed |  |  | yes | yes |
| `sample_n` | Count the value rests on. | count | observed |  |  | yes | yes |
| `label` | observed or estimated. |  | metadata |  |  | yes | no |
| `basis` | published, common or ponta. |  | metadata |  |  | yes | no |
| `source` | The definition's source. |  | metadata |  |  | yes | no |
| `data_check` | Data check state of the measure's inputs (a Blocked row has no value). |  | metadata |  |  | yes | no |
| `has_feed` | Whether the match had an event feed. |  | metadata |  |  | no | no |
| `versions` | Versions the profile was built from. |  | metadata |  |  | no | no |

## profile_measures

| Column | Definition | Unit | Kind | Validation | Data check | Quotable | Player-facing |
|---|---|---|---|---|---|---|---|
| `efi_receptions_behind_midfield_p90` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); per 90 minutes of play. | per 90 minutes of play | estimated | agrees in part (Spearman 0.64 with PFF's midfield-line breaks (section 8; internal only)) | p3.relative_zones | yes | no |
| `efi_receptions_behind_defence_p90` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); per 90 minutes of play. | per 90 minutes of play | estimated | agrees in part (Spearman 0.42 with PFF's defensive-line breaks (section 8; internal only)) | p3.relative_zones | yes | no |
| `efi_final_third_entries_p90` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); per 90 minutes of play. | per 90 minutes of play | estimated | not yet measured | p3.relative_zones | yes | no |
| `efi_final_third_entries_left_wing_share` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); share of entries. | share of entries | estimated | not yet measured | p3.relative_zones | yes | no |
| `efi_final_third_entries_left_half_space_share` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); share of entries. | share of entries | estimated | not yet measured | p3.relative_zones | yes | no |
| `efi_final_third_entries_centre_share` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); share of entries. | share of entries | estimated | not yet measured | p3.relative_zones | yes | no |
| `efi_final_third_entries_right_half_space_share` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); share of entries. | share of entries | estimated | not yet measured | p3.relative_zones | yes | no |
| `efi_final_third_entries_right_wing_share` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); share of entries. | share of entries | estimated | not yet measured | p3.relative_zones | yes | no |
| `efi_def_line_m_out_defensive` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); metres from own goal line. | metres from own goal line | estimated | not yet measured | p2.team_shape | yes | no |
| `efi_def_line_m_out_middle` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); metres from own goal line. | metres from own goal line | estimated | not yet measured | p2.team_shape | yes | no |
| `efi_def_line_m_out_attacking` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); metres from own goal line. | metres from own goal line | estimated | not yet measured | p2.team_shape | yes | no |
| `efi_length_m_in_middle` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); metres. | metres | estimated | not yet measured | p2.team_shape | yes | no |
| `efi_length_m_out_middle` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); metres. | metres | estimated | not yet measured | p2.team_shape | yes | no |
| `efi_ball_recovery_time_s` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); seconds. | seconds | estimated | not yet measured | p1.possession | yes | no |
| `efi_forced_turnovers_p90` | FIFA Enhanced Football Intelligence (EFI) definitions, as computed by P5 (relationships.efi_aggregates); per 90 minutes of play. | per 90 minutes of play | estimated | not yet measured | p1.possession | yes | no |
| `ppda` | PPDA as published by Colin Trainor (2014); defensive actions mapped per provider; passes per defensive action. | passes per defensive action | estimated | agrees in part (-0.72 with PFF's pressure rate, the expected direction; levels 5 to 8 on Metrica and 5 to 19 on the German league, within the published range; Blocked on SkillCorner and PFF (section 10)) | p1.possession | yes | no |
| `possession_share` | Time-based possession share, a common public definition; share of in-play time. | share of in-play time | estimated | agrees with reference data (r 0.91 with the German league's official possession (14 team-matches); r 0.95 with PFF's on-ball time (CONTEXT_PROFILES_REPORT.md sections 3 and 8)) | p1.possession | yes | no |
| `field_tilt` | Field tilt, a common public definition; share. | share | estimated | not yet measured | p1.possession | yes | no |
| `phase_share_build_up` | Ponta P4 (SkillCorner's phase vocabulary); share of possession time with a phase. | share of possession time with a phase | estimated | agrees with reference data (Spearman 0.90 with SkillCorner's phase shares (40 team-matches; section 3)) | p4.phase_of_play | yes | no |
| `phase_share_create` | Ponta P4 (SkillCorner's phase vocabulary); share of possession time with a phase. | share of possession time with a phase | estimated | agrees in part (Spearman 0.69 with SkillCorner's progression share (section 3)) | p4.phase_of_play | yes | no |
| `phase_share_finish` | Ponta P4 (SkillCorner's phase vocabulary); share of possession time with a phase. | share of possession time with a phase | estimated | agrees with reference data (Spearman 0.91 with SkillCorner's final-third share (section 3)) | p4.phase_of_play | yes | no |
| `phase_share_direct` | Ponta P4 (SkillCorner's phase vocabulary); share of possession time with a phase. | share of possession time with a phase | estimated | agrees in part (Spearman 0.66 with SkillCorner's long-ball share (section 3)) | p4.phase_of_play | yes | no |
| `phase_share_quick_break` | Ponta P4 (SkillCorner's phase vocabulary); share of possession time with a phase. | share of possession time with a phase | estimated | agrees in part (Spearman 0.32 with SkillCorner's counter-attack share (section 3)) | p4.phase_of_play | yes | no |
| `phase_share_transition` | Ponta P4 (SkillCorner's phase vocabulary); share of possession time with a phase. | share of possession time with a phase | estimated | agrees in part (Spearman 0.47 with SkillCorner's transition share (section 3)) | p4.phase_of_play | yes | no |
| `phase_share_set_play` | Ponta P4 (SkillCorner's phase vocabulary); share of possession time with a phase. | share of possession time with a phase | estimated | agrees in part (Spearman 0.53 with SkillCorner's set-play share (section 3)) | p4.phase_of_play | yes | no |
| `phase_share_chaotic` | Ponta P4 (SkillCorner's phase vocabulary); share of possession time with a phase. | share of possession time with a phase | estimated | agrees in part (Spearman 0.68 with SkillCorner's contested share (section 3)) | p4.phase_of_play | yes | no |
| `out_share_high_block` | Ponta P4 (the defending team's own block; FIFA's block names); share of out-of-possession time. | share of out-of-possession time | estimated | not yet measured | p4.phase_of_play | yes | no |
| `out_share_mid_block` | Ponta P4 (the defending team's own block; FIFA's block names); share of out-of-possession time. | share of out-of-possession time | estimated | not yet measured | p4.phase_of_play | yes | no |
| `out_share_low_block` | Ponta P4 (the defending team's own block; FIFA's block names); share of out-of-possession time. | share of out-of-possession time | estimated | not yet measured | p4.phase_of_play | yes | no |
| `press_share_out` | Ponta P4v; share of out-of-possession time. | share of out-of-possession time | estimated | agrees in part (58% of SkillCorner's engagements found at 65% precision, type kappa 0.54; counter-press 51% found (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md section 6)) | p4v.engagements | yes | no |
| `counter_press_share_out` | Ponta P4v; share of out-of-possession time. | share of out-of-possession time | estimated | agrees in part (58% of SkillCorner's engagements found at 65% precision, type kappa 0.54; counter-press 51% found (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md section 6)) | p4v.engagements | yes | no |
| `build_up_opposed_share` | Ponta P4v; share of build-up time. | share of build-up time | estimated | agrees in part (build-up opposed against SkillCorner's engagements: kappa 0.52 after one fix (0.36 before it; measured on a development set)) | p4v.fifa_view | yes | no |
| `presses_per_min_out` | Ponta P4v; per minute out of possession. | per minute out of possession | estimated | agrees in part (Spearman 0.57 with SkillCorner's pressing chains (section 3); -0.05 with PFF's pressure rate) | p4v.engagements | yes | no |
| `engagements_recovery_per_min_out` | Ponta P4v; per minute out of possession. | per minute out of possession | estimated | disagrees with reference data: not quotable (engagement rates run against PFF's pressure rate: pressure type -0.47, presses -0.05, all engagements -0.22 (CONTEXT_PROFILES_REPORT.md section 8; PFF World Cup 2022, internal only)) | p4v.engagements | no | no |
| `engagements_counter_press_per_min_out` | Ponta P4v; per minute out of possession. | per minute out of possession | estimated | agrees in part (Spearman 0.47 with PFF's pressure rate (section 8; internal only)) | p4v.engagements | yes | no |
| `engagements_press_per_min_out` | Ponta P4v; per minute out of possession. | per minute out of possession | estimated | disagrees with reference data: not quotable (engagement rates run against PFF's pressure rate: pressure type -0.47, presses -0.05, all engagements -0.22 (CONTEXT_PROFILES_REPORT.md section 8; PFF World Cup 2022, internal only)) | p4v.engagements | no | no |
| `engagements_pressure_per_min_out` | Ponta P4v; per minute out of possession. | per minute out of possession | estimated | disagrees with reference data: not quotable (engagement rates run against PFF's pressure rate: pressure type -0.47, presses -0.05, all engagements -0.22 (CONTEXT_PROFILES_REPORT.md section 8; PFF World Cup 2022, internal only)) | p4v.engagements | no | no |
| `chains_lost_share` | Ponta P1; share of chains. | share of chains | estimated | not yet measured | p1.possession | yes | no |
| `chains_shot_share` | Ponta P1; share of chains. | share of chains | estimated | not yet measured | p1.possession | yes | no |
| `chains_dead_ball_share` | Ponta P1; share of chains. | share of chains | estimated | not yet measured | p1.possession | yes | no |
| `runs_in_behind_per_min_in` | Ponta P6 (FIFA's area categories); per minute in possession. | per minute in possession | estimated | agrees in part (75% to 80% of SkillCorner's runs found; run type kappa 0.41 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p6.movement | yes | no |
| `carrier_pressure_median_in` | Ponta P5 (Plaiq pressure); 1 to 10. | 1 to 10 | estimated | agrees with reference data (carrier pressure against SkillCorner's pressure levels: weighted kappa 0.76 (measured on a development set; CONTEXT_RELATIONSHIPS_REPORT.md)) | p5.relationships | yes | no |
| `carrier_pressure_median_out` | Ponta P5 (Plaiq pressure); 1 to 10. | 1 to 10 | estimated | agrees in part (Spearman 0.53 with PFF's pressure rate (section 8; internal only)) | p5.relationships | yes | no |
| `open_high_value_share_median` | Ponta P7 (pitch control and threat on the honeycomb); share. | share | estimated | agrees in part (no direct label; orders by SkillCorner's phase as expected (build-up 0.24 to set plays 0.71; CONTEXT_RELATIONSHIPS_REPORT.md)) | p7.space | yes | no |
| `gain_median` | Plaiq Gain (version 1); 0 to 1.5. | 0 to 1.5 | estimated | measured against outcomes (measured against outcomes and providers' analogues, not against analysts; construct validity untested; the passer is the best option in about 65% of scored passes (PLAIQ_REFERENCES_REPORT.md)) | p5.line_breaks | yes | no |
| `choice_median` | Plaiq Choice (version 2); 0 to 1. | 0 to 1 | estimated | measured against outcomes (measured against outcomes and SkillCorner's analogue, not against analysts (PLAIQ_REFERENCES_REPORT.md)) | p5.line_breaks | yes | no |

