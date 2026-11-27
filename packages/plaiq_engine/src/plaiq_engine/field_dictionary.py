"""
field_dictionary.py — one machine-readable dictionary of every AI-facing column (review backlog B: AI-08, AN-10).

Four tables reach the AI layer: ``semantic_events``, ``semantic_states`` (episodes), ``movement_values`` and
``profiles`` (with one entry per profile measure, table ``profile_measures``). Every column has exactly one entry here
(tested), giving:

* ``definition``, ``unit`` (scale included, for example ``1 to 10``);
* ``kind``: ``observed`` (the provider's feed, or a deterministic bucket of it), ``estimated`` (computed by Ponta),
  ``identifier`` or ``metadata``;
* ``validation``: what the value has been checked against, typed rather than prose (AN-10):
  ``analyst_validated`` (none yet: Gate 2b), ``reference_agrees`` and ``reference_partial`` (measured against
  professional reference data on a development set, agreeing well or only in part), ``reference_disagrees``,
  ``outcome_measured`` (no labelled reference; measured against outcomes), ``no_reference``, or ``not_applicable``;
* ``evidence``: the figure and where it is reported (required unless ``no_reference`` or ``not_applicable``);
* ``data_check``: the Data check metric whose state the value inherits (a ``parity`` metric, or ``p4.phase_of_play``
  style for a context package), so a typed tool attaches the state to every result;
* ``quotable``: whether the AI may state the value in prose (identifiers, versions and disagreeing fields may not);
* ``player_facing``: whether the value may appear in a view a player sees. Only observed values may, until an analyst
  has validated an estimated one (Gate 2b): the rule that unvalidated fields reach analysts only labelled
  "estimated, not yet validated" is enforced here, not left to prose.

Figures are measured on a development set (founder decision, 25 November 2026), not held out. ``python
tools/field_dictionary.py`` writes ``docs/ai_fields.json`` and ``docs/ai_fields.md`` from this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

DICTIONARY_VERSION = "1.1.0"  # 27 November 2026: semantic schema 2.2 (1.0.0: 2.1), profiles 0.3.0

KINDS = ("observed", "estimated", "identifier", "metadata")
VALIDATION = ("analyst_validated", "reference_agrees", "reference_partial", "reference_disagrees", "outcome_measured",
              "no_reference", "not_applicable")
TABLES = ("semantic_events", "semantic_states", "movement_values", "profiles", "profile_measures")


@dataclass(frozen=True)
class Field:
    table: str
    name: str
    definition: str
    unit: str | None
    kind: str
    validation: str
    evidence: str | None
    data_check: str | None
    quotable: bool
    player_facing: bool

    def to_dict(self) -> dict:
        return asdict(self)


# -- evidence, quoted once ---------------------------------------------------------------------------------------
_DEV = "measured on a development set"
E_PRESSURE = f"1v1 pressure against SkillCorner's five pressure levels: weighted kappa 0.80 ({_DEV}; PLAIQ_REFERENCES_REPORT.md)"
E_CARRIER = f"carrier pressure against SkillCorner's pressure levels: weighted kappa 0.76 ({_DEV}; CONTEXT_RELATIONSHIPS_REPORT.md)"
E_COMPLETION = f"log-loss 0.327, AUC 0.843 on SkillCorner's matches (SkillCorner's own model 0.894; {_DEV}; PLAIQ_REFERENCES_REPORT.md)"
E_OPTIONS = f"availability predicts completion at AUC 0.77; value tracks SkillCorner's threat at 0.78 ({_DEV}; PLAIQ_REFERENCES_REPORT.md)"
E_GAIN = ("measured against outcomes and providers' analogues, not against analysts; construct validity untested; the passer "
          "is the best option in about 65% of scored passes (PLAIQ_REFERENCES_REPORT.md)")
E_CHOICE = "measured against outcomes and SkillCorner's analogue, not against analysts (PLAIQ_REFERENCES_REPORT.md)"
E_ADJUSTED = ("at half cost weight: shot AUC 0.58, completion AUC 0.63 on SkillCorner's matches; the game-state weights are a "
              "judgement (CONTEXT_RELATIONSHIPS_REPORT.md section 5)")
E_PHASE = f"phase of play against SkillCorner's phases: kappa 0.66 / 0.63 by match half, Gate 2a met ({_DEV}; CONTEXT_STRUCTURE_REPORT.md)"
E_FORMATION = f"formations agree with SkillCorner's on 20 of 20 matches ({_DEV}; CONTEXT_STRUCTURE_REPORT.md)"
E_ROLE = f"roles agree with SkillCorner's positions 83% grouped, 90% merged, pooled over 583 players ({_DEV}; CONTEXT_STRUCTURE_REPORT.md)"
E_POSSESSION = "82% to 89% agreement with the German league's official frame-level possession (CONTEXT_FOUNDATIONS_REPORT.md)"
E_LINE_BREAKS = f"line breaks against SkillCorner's labels: kappa 0.56 ({_DEV}; CONTEXT_RELATIONSHIPS_REPORT.md)"
E_GOAL_SIDE = f"goal-side defenders agree with SkillCorner's 81% ({_DEV}; CONTEXT_RELATIONSHIPS_REPORT.md)"
E_RUNS = f"75% to 80% of SkillCorner's runs found; run type kappa 0.41 ({_DEV}; CONTEXT_RELATIONSHIPS_REPORT.md)"
E_ENGAGE = (f"58% of SkillCorner's engagements found at 65% precision, type kappa 0.54; counter-press 51% found ({_DEV}; "
            "CONTEXT_RELATIONSHIPS_REPORT.md section 6)")
E_OPPOSED = f"build-up opposed against SkillCorner's engagements: kappa 0.52 after one fix (0.36 before it; {_DEV})"
E_SPACE = "no direct label; orders by SkillCorner's phase as expected (build-up 0.24 to set plays 0.71; CONTEXT_RELATIONSHIPS_REPORT.md)"
E_EPISODES = f"78.5% of SkillCorner's phase boundaries found within 2 s, phase kappa 0.65 ({_DEV}; CONTEXT_STATES_REPORT.md)"
E_PFF_ENGAGE = ("engagement rates run against PFF's pressure rate: pressure type -0.47, presses -0.05, all engagements -0.22 "
                "(CONTEXT_PROFILES_REPORT.md section 8; PFF World Cup 2022, internal only)")
E_XT = f"the value surface tracks SkillCorner's threat at 0.78 ({_DEV}; PLAIQ_REFERENCES_REPORT.md)"

_FIELDS: list[Field] = []


def _f(table, name, definition, unit=None, kind="estimated", validation="no_reference", evidence=None, data_check=None,
       quotable=None, player_facing=None):
    if quotable is None:
        quotable = kind in ("observed", "estimated") and validation != "reference_disagrees"
    if player_facing is None:
        player_facing = kind == "observed" or validation == "analyst_validated"
    _FIELDS.append(Field(table, name, definition, unit, kind, validation, evidence, data_check, bool(quotable), bool(player_facing)))


def _ident(table, name, definition):
    _f(table, name, definition, kind="identifier", validation="not_applicable", quotable=False, player_facing=False)


def _meta(table, name, definition, unit=None, quotable=False):
    _f(table, name, definition, unit, kind="metadata", validation="not_applicable", quotable=quotable, player_facing=False)


def _obs(table, name, definition, unit=None, data_check=None):
    _f(table, name, definition, unit, kind="observed", validation="not_applicable", data_check=data_check)


# -- semantic_events -------------------------------------------------------------------------------------------
T = "semantic_events"
_meta(T, "schema_version", "Semantic schema version of the row.")
_meta(T, "engine_version", "Plaiq and Matrix versions that produced the row.")
_ident(T, "event_id", "Opaque event id, derived from the event's content; cited, never quoted.")
_ident(T, "match_id", "Opaque match key (m1_...), never the provider's id.")
_ident(T, "event_idx", "The loader's row position; internal, for joins only.")
_obs(T, "timestamp_s", "Match clock at the action's start.", "s")
_obs(T, "minute_display", "The match minute as mm:ss.")
_obs(T, "period", "Period of play (1, 2; 3 and 4 in extra time).")
_ident(T, "actor_player_ref", "The acting player as side and shirt number (home_10); the client adds the name from the club's roster.")
_obs(T, "actor_team", "home or away.")
_f(T, "actor_role", "The actor's role: the club's roster where given, else P2's main role for the match.", validation="reference_agrees",
   evidence=E_ROLE, data_check="p2.formation")
_meta(T, "actor_role_source", "Where actor_role came from: roster or estimated.")
_obs(T, "action_type", "pass, carry, shot, turnover, defensive_action, set_piece, foul_won, card, or run (run rows are P6's estimate).")
_obs(T, "action_outcome", "completed or lost (passes, carries), won or lost (defensive actions), the shot's outcome where the feed gives one.")
_obs(T, "action_detail", "The feed's subtype in Ponta's words: header, set-piece kind, dispossessed, clearance, offside, card colour.")
_obs(T, "target_player_ref", "The logged receiver of a pass (side and shirt number).")
_f(T, "inferred_intended_ref", "The engine's inferred intended receiver of a failed pass.", data_check="decision_quality")
_f(T, "inference_confidence", "How strongly the intended receiver is inferred (weak, medium, strong).", unit="category", data_check="decision_quality")
_f(T, "dq_score", "Gain (decision quality version 1): forward movement on the threat surface relative to the passer's position.", "0 to 1.5",
   validation="outcome_measured", evidence=E_GAIN, data_check="decision_quality")
_f(T, "dq_band", "Gain's band: low_gain, partial_gain, most_gain, full_gain, or not_scored.", "category", validation="outcome_measured",
   evidence=E_GAIN, data_check="decision_quality")
_f(T, "chosen_option_pov", "Pass option value of the option the passer chose.", "value", validation="reference_partial", evidence=E_OPTIONS,
   data_check="pass_option_value")
_f(T, "best_option_player_ref", "The team-mate with the highest pass option value at release.", validation="reference_partial",
   evidence=E_OPTIONS, data_check="pass_option_value")
_f(T, "best_option_pov", "The highest pass option value on offer at release.", "value", validation="reference_partial", evidence=E_OPTIONS,
   data_check="pass_option_value")
_f(T, "xt_delta", "Change in threat (value surface) from the action's start to its end.", "threat", validation="reference_partial",
   evidence=E_XT, data_check="event_delta_v")
_f(T, "movement_value", "Off-ball movement value (null in this table since schema 2.1; see movement_values).", "value",
   data_check="movement_value")
_f(T, "opposition_pressure", "1v1 pressure on the actor at the action's start.", "1 to 10", validation="reference_agrees",
   evidence=E_PRESSURE, data_check="pressure_1v1")
_obs(T, "pitch_zone", "Channel and third of the action's start, in the actor's attacking direction.", "category")
_meta(T, "mode", "Analysis mode (full_tracking).")
_f(T, "pass_completion_probability", "Estimated probability the pass is completed.", "0 to 1", validation="reference_agrees",
   evidence=E_COMPLETION, data_check="pass_completion")
_meta(T, "pass_completion_status", "estimated, or why the pass has no completion probability.")
_meta(T, "pass_completion_parameters", "The fitted parameter set used.")
_f(T, "dq_v2_score", "Choice (decision quality version 2): how the chosen option compared with the best on offer.", "0 to 1",
   validation="outcome_measured", evidence=E_CHOICE, data_check="decision_quality_v2")
_f(T, "dq_v2_band", "Choice's band: best_option, near_best, below_best, well_below_best (never Gain's names).", "category",
   validation="outcome_measured", evidence=E_CHOICE, data_check="decision_quality_v2")
_meta(T, "dq_v2_status", "estimated, or why Choice is not scored.")
_f(T, "dq_v2_best_option_ref", "The best option by Choice (side and shirt number).", validation="outcome_measured", evidence=E_CHOICE,
   data_check="decision_quality_v2")
_meta(T, "dq_v2_version", "Choice's version.")
_f(T, "gain_v2_score", "Gain version 2: the realised value of the pass over the best option's expected value.", "0 to 1.5",
   validation="outcome_measured", evidence=E_CHOICE, data_check="gain_v2")
_f(T, "gain_v2_band", "Gain version 2's band, on version 1's cut-offs and names.", "category", validation="outcome_measured",
   evidence=E_CHOICE, data_check="gain_v2")
_meta(T, "gain_v2_status", "estimated, or why Gain version 2 is not scored.")
_meta(T, "gain_v2_version", "Gain version 2's version.")
_meta(T, "not_scored", "True when the decision is deliberately not scored (goalkeeper, set piece, lost pass for Gain).", quotable=True)
_meta(T, "not_scored_reason", "Why it is not scored.", quotable=True)
_f(T, "run_type", "A run's type in SkillCorner's vocabulary (run rows only).", "category", validation="reference_partial", evidence=E_RUNS,
   data_check="p6.movement")
_f(T, "run_area", "Where the run ends against the defending units: in_front, in_between or in_behind (FIFA's categories).", "category",
   data_check="p6.movement")
_meta(T, "physiological_context", "Null: no physiological source exists.")
_obs(T, "ball_detected", "Whether the ball was detected at the action's start.")
_obs(T, "players_observed", "Players the provider observed (not estimated) at the action's start.", "players")
_obs(T, "observed_near_ball_share", "Share of positions within 15 m of the ball the provider observed at the action's start "
     "(null where it does not flag estimates).", "share")
_meta(T, "parity_spatial", "Data check state of decision quality for this match.", quotable=True)
_meta(T, "parity_physiological", "Data check state of load and fatigue for this match.", quotable=True)
_ident(T, "possession_chain_id", "The possession chain (P1), built on the match key.")
_meta(T, "possession_source", "How P1 assigned possession: observed, inferred, events or unknown.")
_obs(T, "score_diff", "Actor's team score minus the opponent's, where the feed gives the running score.", "goals", data_check="p1.game_state")
_obs(T, "red_card_diff", "Actor's team red cards minus the opponent's.", "cards", data_check="p1.game_state")
_ident(T, "state_id", "The episode (P8) the action belongs to.")
_f(T, "phase_team", "The team in possession by P4.", validation="reference_agrees", evidence=E_POSSESSION, data_check="p1.possession")
_f(T, "phase", "Phase of play of the team in possession (SkillCorner's vocabulary).", "category", validation="reference_agrees",
   evidence=E_PHASE, data_check="p4.phase_of_play")
_f(T, "out_phase", "The defending team's own block in established play, else the pair of the attacking phase.", "category",
   data_check="p4.phase_of_play")
_f(T, "fifa_phase", "FIFA's name for the phase, with build-up opposed or unopposed.", "category", validation="reference_partial",
   evidence=E_OPPOSED, data_check="p4v.fifa_view")
_f(T, "fifa_out_phase", "FIFA's name for the out-of-possession phase (press against block).", "category", data_check="p4v.fifa_view")
_f(T, "block", "The defending block (high, mid, low) by P4's depth bands.", "category", data_check="p4.phase_of_play")
_f(T, "formation_in", "The actor's team's formation in possession.", "category", validation="reference_agrees", evidence=E_FORMATION,
   data_check="p2.formation")
_f(T, "formation_out", "The opponent's formation out of possession.", "category", validation="reference_agrees", evidence=E_FORMATION,
   data_check="p2.formation")
_f(T, "zone_lane", "The ball's lane relative to the defending shape.", "category", data_check="p3.relative_zones")
_f(T, "zone_third", "The ball's third relative to the defending shape.", "category", data_check="p3.relative_zones")
_f(T, "zone_band", "The ball's band between the defending lines (team in possession only).", "category", data_check="p3.relative_zones")
_f(T, "units_broken", "Defending units the pass broke.", "units", validation="reference_agrees", evidence=E_LINE_BREAKS,
   data_check="p5.line_breaks")
_f(T, "furthest_line_broken", "The furthest defending line the pass broke.", "category", validation="reference_agrees",
   evidence=E_LINE_BREAKS, data_check="p5.line_breaks")
_f(T, "received_inside_shape", "Whether the pass was received inside the defending shape.", data_check="p5.relationships")
_f(T, "goal_side_defenders", "Defenders goal-side of the ball.", "players", validation="reference_agrees", evidence=E_GOAL_SIDE,
   data_check="p5.relationships")
_f(T, "overload_at_ball", "Attackers minus defenders near the ball.", "players", data_check="p5.relationships")
_f(T, "runs_in_behind", "Runs in behind under way.", "runs", validation="reference_partial", evidence=E_RUNS, data_check="p6.movement")
_f(T, "adjusted_choice", "Choice adjusted for the game state and the cost of losing the ball.", "0 to 1", validation="outcome_measured",
   evidence=E_ADJUSTED, data_check="p7.adjusted_choice")
_f(T, "open_high_value_share", "Open share of the team's high-value space.", "share", validation="reference_partial", evidence=E_SPACE,
   data_check="p7.space")
_f(T, "engaging_defenders", "Defenders engaging the carrier (P4v).", "players", validation="reference_partial", evidence=E_ENGAGE,
   data_check="p4v.engagements")
_meta(T, "context_versions", "Context package and threshold versions.")
_ident(T, "actor_team_key", "Opaque key of the actor's team (t1_...), stable across the club's matches.")
_ident(T, "opponent_team_key", "Opaque key of the opponent.")
_ident(T, "competition", "Opaque competition key (c1_...).")
_ident(T, "season", "Opaque season key (s1_...).")

# -- semantic_states (episodes) ----------------------------------------------------------------------------------
T = "semantic_states"
_ident(T, "state_id", "Opaque episode id, built on the match key.")
_ident(T, "match_id", "Opaque match key.")
_obs(T, "period", "Period of play.")
for c, d in (("start_s", "Episode start"), ("end_s", "Episode end"), ("duration_s", "Episode length")):
    _f(T, c, d + ".", "s", validation="reference_agrees", evidence=E_EPISODES, data_check="p8.episodes")
_f(T, "minute_display", "Episode start as mm:ss.", validation="reference_agrees", evidence=E_EPISODES, data_check="p8.episodes")
_f(T, "team", "The team in possession.", validation="reference_agrees", evidence=E_POSSESSION, data_check="p1.possession")
_ident(T, "chain_id", "The possession chain.")
_f(T, "start_reason", "How the episode started (won the ball, goalkeeper, set piece, restart, kick-off, continued).", "category",
   data_check="p8.episodes")
_f(T, "start_detail", "The start's detail (set-piece kind).", "category", data_check="p8.episodes")
_obs(T, "cleared", "Whether the team made a logged clearance in the episode.")
for c in ("share_controlled", "share_in_flight", "share_loose_or_contested"):
    _f(T, c, f"Share of the episode with the ball {c[6:].replace('_', ' ')} (P1's ball state).", "share", data_check="p1.possession_state")
_obs(T, "score_diff", "The team's score minus the opponent's.", "goals", data_check="p1.game_state")
_obs(T, "red_card_diff", "The team's red cards minus the opponent's.", "cards", data_check="p1.game_state")
_meta(T, "game_state_source", "Where the score came from (running or final_only).")
_f(T, "phase", "The episode's phase of play.", "category", validation="reference_agrees", evidence=E_EPISODES, data_check="p4.phase_of_play")
_f(T, "out_phase", "The defending team's phase or block.", "category", data_check="p4.phase_of_play")
_f(T, "fifa_phase", "FIFA's name for the phase.", "category", validation="reference_partial", evidence=E_OPPOSED, data_check="p4v.fifa_view")
_f(T, "fifa_out_phase", "FIFA's name for the out-of-possession phase.", "category", data_check="p4v.fifa_view")
_f(T, "block", "The defending block.", "category", data_check="p4.phase_of_play")
_f(T, "opposed_pct", "Share of build-up time that was opposed.", "percent", validation="reference_partial", evidence=E_OPPOSED,
   data_check="p4v.fifa_view")
for c in ("press_share", "counter_press_share"):
    _f(T, c, f"Share of the episode inside a {c[:-6].replace('_', '-')} (P4v).", "share", validation="reference_partial",
       evidence=E_ENGAGE, data_check="p4v.engagements")
for c in ("fifa_mid_block_share", "fifa_low_block_share"):
    _f(T, c, f"Share of the episode meeting FIFA's {c[5:8]} block rule.", "share", data_check="p4v.fifa_block")
for c in ("formation_in", "formation_out"):
    _f(T, c, f"Formation {'in' if c.endswith('in') else 'out of'} possession.", "category", validation="reference_agrees",
       evidence=E_FORMATION, data_check="p2.formation")
for c, d in (("team_length_m", "The team's length"), ("team_width_m", "The team's width"), ("opp_length_m", "The opponent's length"),
             ("opp_width_m", "The opponent's width"), ("opp_def_line_m", "The opponent's defensive line height"),
             ("opp_def_mid_gap_m", "Gap between the opponent's defensive and midfield lines"),
             ("opp_mid_att_gap_m", "Gap between the opponent's midfield and attacking lines")):
    _f(T, c, d + " (median over the episode).", "m", data_check="p2.team_shape")
for c in ("start_lane", "start_third", "end_lane", "end_third"):
    _f(T, c, f"The ball's {c.split('_')[1]} at the episode's {c.split('_')[0]}, relative to the defending shape.", "category",
       data_check="p3.relative_zones")
for c in ("share_defensive_third", "share_middle_third", "share_attacking_third"):
    _f(T, c, f"Share of the episode with the ball in the {c[6:].replace('_', ' ')}.", "share", data_check="p3.relative_zones")
_obs(T, "passes", "Passes in the episode.", "passes")
_obs(T, "passes_completed", "Completed passes in the episode.", "passes")
_f(T, "passes_breaking_a_line", "Passes that broke a defending line.", "passes", validation="reference_agrees", evidence=E_LINE_BREAKS,
   data_check="p5.line_breaks")
_f(T, "furthest_line_broken", "The furthest line broken.", "category", validation="reference_agrees", evidence=E_LINE_BREAKS,
   data_check="p5.line_breaks")
for c in ("carrier_pressure_median", "carrier_pressure_max"):
    _f(T, c, f"Carrier pressure ({c.rsplit('_', 1)[1]}).", "1 to 10", validation="reference_agrees", evidence=E_CARRIER,
       data_check="p5.relationships")
_f(T, "goal_side_defenders_median", "Defenders goal-side of the ball (median).", "players", validation="reference_agrees",
   evidence=E_GOAL_SIDE, data_check="p5.relationships")
_f(T, "overload_at_ball_median", "Attackers minus defenders near the ball (median).", "players", data_check="p5.relationships")
for c in ("runs", "runs_in_behind", "runs_behind", "runs_coming_short", "runs_dropping_off", "runs_support", "runs_overlap",
          "runs_underlap", "runs_pulling_wide", "runs_pulling_half_space", "runs_run_ahead_of_the_ball", "runs_cross_receiver",
          "runs_carry", "defensive_runs"):
    _f(T, c, "Runs (P6)" + ("" if c == "runs" else f": {c.replace('_', ' ')}") + ".", "runs", validation="reference_partial",
       evidence=E_RUNS, data_check="p6.movement")
_f(T, "open_high_value_share", "Open share of the team's high-value space (median).", "share", validation="reference_partial",
   evidence=E_SPACE, data_check="p7.space")
_f(T, "opp_open_high_value_share", "Open share of the opponent's high-value space (median).", "share", validation="reference_partial",
   evidence=E_SPACE, data_check="p7.space")
_f(T, "open_danger_zone", "The most dangerous open zone (lane and third).", "category", data_check="p7.space")
for c in ("engagements", "engagements_recovery", "engagements_counter_press", "engagements_press", "engagements_pressure", "presses",
          "counter_press"):
    _f(T, c, f"Defensive engagements (P4v){'' if c == 'engagements' else ': ' + c.replace('_', ' ')}.", "count",
       validation="reference_partial", evidence=E_ENGAGE, data_check="p4v.engagements")
_f(T, "end_reason", "How the episode ended (phase changed, possession lost, shot, goal, ball out of play ...).", "category",
   data_check="p8.episodes")
_obs(T, "led_to_shot", "Whether the episode's possession chain ended in a shot or goal (chain level).")
_ident(T, "first_event_idx", "P8's first logged-event index; internal.")
_ident(T, "last_event_idx", "P8's last logged-event index; internal.")
_obs(T, "events", "Logged events in the episode.", "events")
_obs(T, "ball_detected_share", "Share of the episode with the ball detected.", "share")
_obs(T, "min_outfield_observed", "Fewest outfield players observed at any point in the episode.", "players")
for c in ("p1_possession", "p1_possession_state", "p1_game_state", "p2_team_shape", "p2_formation", "p3_relative_zones",
          "p4_phase_of_play", "p4v_engagements", "p4v_fifa_view", "p4v_fifa_block", "p5_line_breaks", "p5_relationships",
          "p6_movement", "p7_space", "p7_adjusted_choice", "p8_episodes"):
    _meta(T, f"data_check_{c}", f"Data check state of {c.replace('_', ' ', 1)} for this episode.", quotable=True)
_meta(T, "data_check", "The worst Data check state in the episode.", quotable=True)
_meta(T, "thresholds_version", "Thresholds version.")
_meta(T, "versions", "Package versions.")
_meta(T, "label", "estimated: every episode field is Ponta's estimate unless named observed here.", quotable=True)
_ident(T, "first_event_id", "First semantic event inside the episode.")
_ident(T, "last_event_id", "Last semantic event inside the episode.")
_f(T, "gain_median", "Median Gain of the episode's passes.", "0 to 1.5", validation="outcome_measured", evidence=E_GAIN,
   data_check="decision_quality")
_f(T, "choice_median", "Median Choice of the episode's passes.", "0 to 1", validation="outcome_measured", evidence=E_CHOICE,
   data_check="decision_quality_v2")
_f(T, "adjusted_choice_median", "Median adjusted Choice of the episode's passes.", "0 to 1", validation="outcome_measured",
   evidence=E_ADJUSTED, data_check="p7.adjusted_choice")
_obs(T, "shots", "Shots inside the episode.", "shots")
_obs(T, "ended_in_shot", "Whether a shot fell inside this episode (episode level, unlike led_to_shot).")
_ident(T, "team_key", "Opaque key of the team in possession.")
_ident(T, "opp_team_key", "Opaque key of the opponent.")
_ident(T, "competition", "Opaque competition key.")
_ident(T, "season", "Opaque season key.")
_f(T, "summary", "A one-paragraph summary written by code from the episode's own fields (passes the field-bound number check).",
   "text", data_check="p8.episodes")
_meta(T, "schema_version", "Semantic schema version.")

# -- movement_values -----------------------------------------------------------------------------------------
T = "movement_values"
_meta(T, "schema_version", "Semantic schema version.")
_ident(T, "event_id", "The source event's id plus the player.")
_ident(T, "source_event_id", "The event this movement value belongs to (resolves in semantic_events).")
_ident(T, "match_id", "Opaque match key.")
_obs(T, "timestamp_s", "The source event's start.", "s")
_obs(T, "minute_display", "The source event's start as mm:ss.")
_obs(T, "period", "Period of play.")
_ident(T, "player_ref", "The moving team-mate (side and shirt number).")
_obs(T, "team", "home or away.")
_f(T, "movement_value", "Plaiq's off-ball movement value for the team-mate at this event.", "value", data_check="movement_value")
_f(T, "xt_delta", "Change in threat from the team-mate's movement.", "threat", validation="reference_partial", evidence=E_XT,
   data_check="movement_value")
_f(T, "opposition_pressure", "1v1 pressure on the team-mate.", "1 to 10", validation="reference_agrees", evidence=E_PRESSURE,
   data_check="pressure_1v1")
_ident(T, "state_id", "The episode at the source event.")
_meta(T, "label", "estimated.", quotable=True)

# -- profiles ------------------------------------------------------------------------------------------------
T = "profiles"
_ident(T, "match_id", "Opaque match key.")
_obs(T, "team", "home or away.")
_ident(T, "team_key", "Opaque team key.")
_ident(T, "opp_team_key", "Opaque opponent key.")
_ident(T, "competition", "Opaque competition key.")
_ident(T, "season", "Opaque season key.")
_meta(T, "measure", "The measure's name (see profile_measures).", quotable=True)
_f(T, "value", "The measure's value; its validation is the measure's (profile_measures).", "per measure")
_meta(T, "denominator", "What the value is per (per 90 minutes of ball in play, share of possession time ...).", quotable=True)
_obs(T, "sample_minutes", "Minutes the value rests on.", "minutes")
_obs(T, "sample_n", "Count the value rests on.", "count")
_meta(T, "label", "observed or estimated.", quotable=True)
_meta(T, "basis", "published, common or ponta.", quotable=True)
_meta(T, "source", "The definition's source.", quotable=True)
_meta(T, "data_check", "Data check state of the measure's inputs (a Blocked row has no value).", quotable=True)
_meta(T, "has_feed", "Whether the match had an event feed.")
_meta(T, "versions", "Versions the profile was built from.")

# -- profile_measures (one per catalogue measure) -----------------------------------------------------------------
T = "profile_measures"
_PM: dict[str, tuple[str, str | None]] = {
    "possession_share": ("reference_agrees", "r 0.91 with the German league's official possession (14 team-matches); r 0.95 with PFF's "
                                             "on-ball time (CONTEXT_PROFILES_REPORT.md sections 3 and 8)"),
    "phase_share_build_up": ("reference_agrees", "Spearman 0.90 with SkillCorner's phase shares (40 team-matches; section 3)"),
    "phase_share_finish": ("reference_agrees", "Spearman 0.91 with SkillCorner's final-third share (section 3)"),
    "phase_share_create": ("reference_partial", "Spearman 0.69 with SkillCorner's progression share (section 3)"),
    "phase_share_chaotic": ("reference_partial", "Spearman 0.68 with SkillCorner's contested share (section 3)"),
    "phase_share_direct": ("reference_partial", "Spearman 0.66 with SkillCorner's long-ball share (section 3)"),
    "phase_share_set_play": ("reference_partial", "Spearman 0.53 with SkillCorner's set-play share (section 3)"),
    "phase_share_transition": ("reference_partial", "Spearman 0.47 with SkillCorner's transition share (section 3)"),
    "phase_share_quick_break": ("reference_partial", "Spearman 0.32 with SkillCorner's counter-attack share (section 3)"),
    "presses_per_min_out": ("reference_partial", "Spearman 0.57 with SkillCorner's pressing chains (section 3); -0.05 with PFF's pressure rate"),
    "engagements_counter_press_per_min_out": ("reference_partial", "Spearman 0.47 with PFF's pressure rate (section 8; internal only)"),
    "engagements_pressure_per_min_out": ("reference_disagrees", E_PFF_ENGAGE),
    "engagements_press_per_min_out": ("reference_disagrees", E_PFF_ENGAGE),
    "engagements_recovery_per_min_out": ("reference_disagrees", E_PFF_ENGAGE),
    "efi_receptions_behind_midfield_p90": ("reference_partial", "Spearman 0.64 with PFF's midfield-line breaks (section 8; internal only)"),
    "efi_receptions_behind_defence_p90": ("reference_partial", "Spearman 0.42 with PFF's defensive-line breaks (section 8; internal only)"),
    "ppda": ("reference_partial", "-0.72 with PFF's pressure rate, the expected direction; levels 5 to 8 on Metrica and 5 to 19 on the "
                                  "German league, within the published range; Blocked on SkillCorner and PFF (section 10)"),
    "carrier_pressure_median_out": ("reference_partial", "Spearman 0.53 with PFF's pressure rate (section 8; internal only)"),
    "carrier_pressure_median_in": ("reference_agrees", E_CARRIER),
    "gain_median": ("outcome_measured", E_GAIN),
    "choice_median": ("outcome_measured", E_CHOICE),
    "runs_in_behind_per_min_in": ("reference_partial", E_RUNS),
    "open_high_value_share_median": ("reference_partial", E_SPACE),
    "build_up_opposed_share": ("reference_partial", E_OPPOSED),
    "press_share_out": ("reference_partial", E_ENGAGE),
    "counter_press_share_out": ("reference_partial", E_ENGAGE),
}


def _profile_measures() -> None:
    from plaiq_engine.profiles import CATALOGUE

    for m in CATALOGUE:
        validation, evidence = _PM.get(m.name, ("no_reference", None))
        dc = ".".join(m.inputs[0]) if m.inputs else None
        _f(T, m.name, f"{m.source}; {m.denominator}.", m.denominator, kind="observed" if m.label == "observed" else "estimated",
           validation=validation, evidence=evidence, data_check=dc)


_profile_measures()


def fields(table: str | None = None) -> list[Field]:
    """Every entry, or one table's."""
    return [f for f in _FIELDS if table is None or f.table == table]


def lookup(table: str, name: str) -> Field:
    hit = [f for f in _FIELDS if f.table == table and f.name == name]
    if not hit:
        raise KeyError(f"{table}.{name} is not in the field dictionary")
    return hit[0]


def columns(table: str) -> tuple[str, ...]:
    return tuple(f.name for f in fields(table))


def check_columns(table: str, cols) -> None:
    """Raise unless ``cols`` are exactly the dictionary's columns for ``table`` (the allowlist that replaces name-based leak tests)."""
    known, got = set(columns(table)), list(cols)
    unknown = [c for c in got if c not in known]
    missing = [c for c in known if c not in set(got)]
    if unknown or missing:
        raise ValueError(f"{table}: columns not in the field dictionary {unknown}; dictionary columns missing {missing}")


def player_facing_view(df, table: str):
    """Only the columns a player may see (observed values until an analyst validates an estimated one)."""
    check_columns(table, df.columns)
    return df[[c for c in df.columns if lookup(table, c).player_facing]]


def as_json() -> dict:
    return {"version": DICTIONARY_VERSION, "kinds": list(KINDS), "validation": list(VALIDATION),
            "fields": [f.to_dict() for f in _FIELDS]}
