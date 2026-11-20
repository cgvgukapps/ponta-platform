"""
thresholds.py — every named threshold the context packages use, in one versioned place.

Each value carries a status (``ponta-blueprint/context-packages.md``, "Measure before setting a threshold"):

* ``proposed`` — chosen by the engineer, with the reason; not yet measured.
* ``measured`` — proposed from this repository's measurements on the regression set (the
  ``source`` names the measurement).
* ``agreed`` — set by the founder. Code never sets this status on its own; every agreed value names the decision in ``source``.

The P1 values were agreed by the founder on 17 November 2026 (``CONTEXT_FOUNDATIONS_REPORT.md``,
"Decisions"). Version 0.3.0 added the Stage B thresholds (P2 to P4), proposed or measured
(``CONTEXT_STRUCTURE_REPORT.md``). Version 0.4.0 records the founder's agreement of every Stage B
value on 17 November 2026; no value changed, so package output is unchanged. Version 0.5.0 adds three
P4 refinements from the founder's replay review and the founder's corner rule (corners from
player positions when there is no event feed), measured against the review, SkillCorner's 20
open-data matches and Metrica's logged corners (P4 0.2.0). Version 0.6.0 records the founder's
agreement of those six values on 18 November 2026; no value changed, so package output is unchanged.
Version 0.7.0 adds four throw-in values (throw-ins from player positions when there is no event
feed), measured on SkillCorner's 20 open-data matches and Metrica's logged throw-ins (P4 0.3.0).
Version 0.8.0 records the founder's agreement of those four values on 18 November 2026; no value
changed, so package output is unchanged.
Version 0.9.0 adds ``p1_loose_on_opponent_touch`` (the founder's loose-ball rule), proposed and off,
so package output is unchanged (``CONTEXT_STRUCTURE_REPORT.md``, possession round).
Version 0.10.0 adds ``p5_min_unit_players`` for P5's line breaks (proposed, from FIFA's definition).
Version 0.11.0 adds ``p2_template_4_2_2_2`` (a box-midfield template for P2), proposed and off, so
package output is unchanged (``CONTEXT_STRUCTURE_REPORT.md``, Gate 2a round).
Version 0.13.0 adds ``p1_contest_max_s`` for P1 0.3.0's in-contest state (proposed; the state is a new
field, so earlier outputs are unchanged).
Version 0.12.0 moves ``p4_build_up_max_depth_share`` from a third to a quarter of the pitch, agreed by
the founder on 20 November 2026 (P4 0.4.0; ``CONTEXT_STRUCTURE_REPORT.md``, Gate 2a round).

Nothing threshold-like is hard-coded elsewhere in the context packages. Changing a value
changes package output: bump ``THRESHOLDS_VERSION`` and the version of every package that
reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Status = Literal["proposed", "measured", "agreed"]

THRESHOLDS_VERSION = "0.13.0"


@dataclass(frozen=True)
class Threshold:
    name: str
    value: float
    unit: str
    status: Status
    reason: str
    source: str | None = None


_ALL: tuple[Threshold, ...] = (
    Threshold("context_grid_rate_hz", 5.0, "Hz", "agreed",
              "The founder's work plan proposes 5 Hz: fine enough for possession and phase, a fifth of Metrica's 25 Hz rows.", "Proposed in CONTEXT_FOUNDATIONS_REPORT.md; agreed by the founder, 17 November 2026"),
    Threshold("p1_control_radius_m", 1.5, "m", "agreed",
              "A team is in control when its nearest player is within this distance of a detected ball. Best agreement with event-anchored possession on the Metrica matches (with a 0.4 s switch); broadcast tracking (SkillCorner) agrees better at 2.0 to 2.5 m, so a per-provider radius is an open decision.",
              "CONTEXT_FOUNDATIONS_REPORT.md, control sweep; agreed by the founder, 17 November 2026"),
    Threshold("p1_max_control_ball_speed_mps", 16.0, "m/s", "agreed",
              "A ball moving faster than this is in flight, not under control, even near a player. 16 m/s agreed slightly better than 8 or 12 on all three Metrica matches; the limit matters little.",
              "CONTEXT_FOUNDATIONS_REPORT.md, control sweep; agreed by the founder, 17 November 2026"),
    Threshold("p1_min_switch_s", 0.4, "s", "agreed",
              "Possession passes to the other team only after it has controlled the ball this long, so a ball brushing past an opponent does not flip possession. Without it, team switches double against the event feed's count.",
              "CONTEXT_FOUNDATIONS_REPORT.md, control sweep; agreed by the founder, 17 November 2026"),
    Threshold("p1_max_carry_gap_s", 3.0, "s", "agreed",
              "During a ball-detection gap, the last observed possession is carried forward this long; after that the event feed decides, or the label is unknown. About 1% of undetected-ball time is in gaps under 3 s, so the value matters little.",
              "CONTEXT_FOUNDATIONS_REPORT.md, detection gaps; agreed by the founder, 17 November 2026"),
    Threshold("p1_loose_on_opponent_touch", 0.0, "switch (1 on, 0 off)", "proposed",
              "When the other team touches the ball without holding it for p1_min_switch_s (a deflection, block or clearance), possession is loose until a player of either team gains control (the founder's rule, 18 November 2026). Off: the team that had the ball keeps it.",
              "CONTEXT_STRUCTURE_REPORT.md, possession round: the founder's blind replay check on the IDSSE matches"),
    Threshold("p1_contest_max_s", 4.0, "s", "proposed",
              "After a logged duel, clearance, block or deflection, the ball is in contest until a team has held control for p1_min_switch_s, for at most this long (a ball undetected or travelling without anyone controlling it).",
              "Stage C (context-relationships-task.md), deliverable 1; measured against the founder's possession replay calls"),
    Threshold("p1_degraded_inferred_share", 0.25, "share of in-play time", "agreed",
              "Possession is Degraded when more than this share of in-play time is not observed (inferred, from events or unknown). The Metrica matches sit at 11 to 18%; SkillCorner at 26%, partly because its dead-ball time is unknown and so counted as in play.",
              "CONTEXT_FOUNDATIONS_REPORT.md, possession sources; agreed by the founder, 17 November 2026"),
    Threshold("p1_dead_ball_max_s", 180.0, "s", "agreed",
              "A stoppage longer than this before a restart is not treated as one dead-ball interval (it is more likely missing events).", "Proposed in CONTEXT_FOUNDATIONS_REPORT.md; agreed by the founder, 17 November 2026"),
    Threshold("p1_goal_kickoff_window_s", 180.0, "s", "agreed",
              "A goal is confirmed when the other team kicks off within this time after the shot.", "Proposed in CONTEXT_FOUNDATIONS_REPORT.md; agreed by the founder, 17 November 2026"),
    Threshold("p1_chain_event_window_s", 1.0, "s", "agreed",
              "Events within this time of a possession chain's start or end name how it began or ended.", "Proposed in CONTEXT_FOUNDATIONS_REPORT.md; agreed by the founder, 17 November 2026"),
    Threshold("p1_chain_shot_window_s", 3.0, "s", "agreed",
              "A shot by the team in possession within this time of the chain's end makes the end type a shot.", "Proposed in CONTEXT_FOUNDATIONS_REPORT.md; agreed by the founder, 17 November 2026"),
    # -- P2 team structure (Stage B; agreed by the founder, 17 November 2026) ------------
    Threshold("p2_window_s", 120.0, "s", "agreed",
              "Formation and line windows are two minutes within a period: long enough for stable mean positions, short enough to see changes.",
              "Shaw and Glickman 2019 (two-minute windows per possession state); agreed by the founder, 17 November 2026"),
    Threshold("p2_min_possession_s", 5.0, "s", "agreed",
              "Only possessions at least this long feed a window's in- or out-of-possession shape; shorter ones are transitions.",
              "Shaw and Glickman 2019 (possessions under 5 s excluded); agreed by the founder, 17 November 2026"),
    Threshold("p2_min_window_data_s", 10.0, "s", "agreed",
              "A window needs this much in-state time to fit lines and a formation; less is skipped.", "Proposed in CONTEXT_STRUCTURE_REPORT.md; agreed by the founder, 17 November 2026"),
    Threshold("p2_min_presence_share", 0.5, "share of window", "agreed",
              "A player counts in a window when observed for at least this share of it (handles substitutions).", "Proposed in CONTEXT_STRUCTURE_REPORT.md; agreed by the founder, 17 November 2026"),
    Threshold("p2_min_outfield_players", 8.0, "players", "agreed",
              "A team's shape is measured only with at least this many outfield players observed (allows two red cards or out-of-view players).", "Proposed in CONTEXT_STRUCTURE_REPORT.md; agreed by the founder, 17 November 2026"),
    Threshold("p2_min_shape_coverage", 0.9, "share of in-play time", "agreed",
              "Team shape is Degraded when fewer than this share of in-play grid points have a shape for both teams.", "Proposed in CONTEXT_STRUCTURE_REPORT.md; agreed by the founder, 17 November 2026"),
    Threshold("p2_template_4_2_2_2", 0.0, "switch (1 on, 0 off)", "proposed",
              "Fit a 4-2-2-2 (two holding midfielders, two wide attackers pushed up behind two strikers) beside the ten standard templates. Off: the standard ten only.",
              "CONTEXT_STRUCTURE_REPORT.md, Gate 2a round: match 1959846's home team defends in this shape, and with no template for it P2 fitted a 3-4-2-1. Measured 20 November 2026 and not adopted: 1959846 still misses both role bars, and the grouped bar is met on 18 matches instead of 19"),
    Threshold("p2_formation_min_margin", 0.1, "relative cost", "agreed",
              "A window's formation is unclear when the best template's fit is less than this much better than the second best ((c2 - c1) / c2).", "Proposed in CONTEXT_STRUCTURE_REPORT.md; agreed by the founder, 17 November 2026"),
    Threshold("p2_formation_persist_s", 300.0, "s of window data", "agreed",
              "A formation change is accepted only when the new formation holds for this long.",
              "Kim et al. 2022, SoccerCPD (formation changes persist at least 5 minutes); agreed by the founder, 17 November 2026"),
    # -- P3 relative zones -----------------------------------------------------------------
    Threshold("p3_min_coverage", 0.75, "share of possession time", "agreed",
              "Zones are Degraded when the ball or the defending shape is missing for more than a quarter of possession time (matches P1's 25%).", "Proposed in CONTEXT_STRUCTURE_REPORT.md; agreed by the founder, 17 November 2026"),
    # -- P4 phase of play -------------------------------------------------------------------
    Threshold("p4_low_block_max_m", 26.0, "m from own goal line", "agreed",
              "The defending team is in a low block when its three deepest outfield players average less than this from their goal line (about 10 m outside the penalty area). Equal pitch thirds (35 m) agreed far worse with the reference (kappa 0.20).",
              "CONTEXT_STRUCTURE_REPORT.md, phase sweep against SkillCorner 1886347 (one match; the value holds on both halves); agreed by the founder, 17 November 2026"),
    Threshold("p4_high_block_min_m", 47.0, "m from own goal line", "agreed",
              "The defending team is in a high block when its three deepest outfield players average at least this from their goal line (a few metres short of halfway).",
              "CONTEXT_STRUCTURE_REPORT.md, phase sweep against SkillCorner 1886347 (one match; the value holds on both halves); agreed by the founder, 17 November 2026"),
    Threshold("p4_set_play_max_s", 20.0, "s", "agreed",
              "A set play lasts at most this long after the restart.", "SkillCorner phases of play specification, 16 February 2025 (set play up to 20 s); agreed by the founder, 17 November 2026"),
    Threshold("p4_corner_radius_m", 4.0, "m", "agreed",
              "Without a logged event feed, a possession whose first ball position is within this distance of a corner flag is a corner (the corner arc is 1 m; tracking at restarts is noisy).", "Proposed in CONTEXT_STRUCTURE_REPORT.md; agreed by the founder, 17 November 2026"),
    Threshold("p4_corner_window_s", 1.0, "s", "agreed",
              "Without a logged event feed, the time at the start of a possession in which the ball must come near a corner flag for a corner.", "Proposed in CONTEXT_STRUCTURE_REPORT.md; agreed by the founder, 17 November 2026"),
    Threshold("p4_direct_min_gain_m", 32.0, "m", "agreed",
              "A ball from the own half that gains at least this much towards goal is a direct phase.",
              "SkillCorner phases of play specification, 16 February 2025 (long ball of at least 32 m); agreed by the founder, 17 November 2026"),
    Threshold("p4_direct_window_s", 3.0, "s", "agreed",
              "The time within which the direct gain must happen (a long ball's flight, not a run).", "Proposed in CONTEXT_STRUCTURE_REPORT.md; agreed by the founder, 17 November 2026"),
    Threshold("p4_chaotic_max_s", 5.0, "s", "agreed",
              "A possession chain shorter than this is chaotic (SkillCorner also spares chains with three or more passes; passes are not counted here).",
              "SkillCorner phases of play specification, 16 February 2025 (chaotic unless at least 5 s or 3 passes); agreed by the founder, 17 November 2026"),
    Threshold("p4_break_eval_s", 3.0, "s", "agreed",
              "After a regain, the ball's gain towards goal is measured over this time to decide whether it is a quick break or transition.", "Proposed in CONTEXT_STRUCTURE_REPORT.md; agreed by the founder, 17 November 2026"),
    Threshold("p4_break_min_gain_m", 15.0, "m", "agreed",
              "The gain towards goal within the evaluation time that makes a regain a quick break or transition. 15 m agreed slightly better than 5 or 10 m on both halves.",
              "CONTEXT_STRUCTURE_REPORT.md, phase sweep against SkillCorner 1886347 (one match; the value holds on both halves); agreed by the founder, 17 November 2026"),
    Threshold("p4_break_max_s", 15.0, "s", "agreed",
              "A quick break or transition lasts at most this long.", "Proposed in CONTEXT_STRUCTURE_REPORT.md; agreed by the founder, 17 November 2026"),
    Threshold("p4_recovered_players", 7.0, "players", "agreed",
              "A quick break or transition ends when this many defending outfield players are goal-side of the ball.", "Proposed in CONTEXT_STRUCTURE_REPORT.md; agreed by the founder, 17 November 2026"),
    # -- P4 refinements from the founder's replay review (17 November 2026; agreed 18 November 2026) --
    Threshold("p4_finish_min_depth_share", 0.5, "share of pitch length", "agreed",
              "Against a low block, finish needs the ball in the attacking half; otherwise it is create. Requiring the final third instead lost agreement with SkillCorner (kappa 0.55 against 0.61). 0 turns the check off.",
              "CONTEXT_STRUCTURE_REPORT.md, founder's replay review (50 moments) and SkillCorner's 20 open-data matches; agreed by the founder, 18 November 2026"),
    Threshold("p4_build_up_max_depth_share", 0.25, "share of pitch length", "agreed",
              "With the ball within this share of the pitch from the team's own goal line, the team is building up whatever the block (the reviewer: playing from the back is build-up). 0 turns the check off. A quarter since 0.12.0 (a third before).",
              "CONTEXT_STRUCTURE_REPORT.md, founder's replay review (50 moments) and SkillCorner's 20 open-data matches; the rule agreed by the founder, 18 November 2026 (a third); "
              "a quarter measured in the Gate 2a round (pooled kappa 0.670 / 0.638 from 0.609 / 0.610, held out by match, no match below 0.50) and agreed by the founder, 20 November 2026"),
    Threshold("p4_break_end_depth_share", 0.667, "share of pitch length", "agreed",
              "A quick break or transition that began further back ends when the ball reaches the final third; the attack is then labelled by the block (usually finish). 0 turns the check off.",
              "CONTEXT_STRUCTURE_REPORT.md, founder's replay review (50 moments) and SkillCorner's 20 open-data matches; agreed by the founder, 18 November 2026"),
    # -- Corners from player positions (the founder's proposal, 17 November 2026; agreed 18 November 2026) --
    Threshold("p4_corner_setup_crowd_min", 12.0, "players", "agreed",
              "Without a logged feed, a corner set-up needs at least this many players of both teams in that penalty area while one attacker stands at a corner flag. Finds 73% of SkillCorner's corners (55% of detections right) against 4% for the ball rule. 0 turns the position rule off (the ball rule is used).",
              "CONTEXT_STRUCTURE_REPORT.md, corner sweep: fitted on 10 SkillCorner matches, read on the other 10 and on Metrica's logged corners; agreed by the founder, 18 November 2026"),
    Threshold("p4_corner_setup_radius_m", 4.0, "m", "agreed",
              "The taker stands within this distance of the corner flag (and no defender does). 2 and 3 m lost corners on broadcast tracking.",
              "CONTEXT_STRUCTURE_REPORT.md, corner sweep: fitted on 10 SkillCorner matches, read on the other 10 and on Metrica's logged corners; agreed by the founder, 18 November 2026"),
    Threshold("p4_corner_setup_hold_s", 1.0, "s", "agreed",
              "The set-up must hold this long, which separates it from open play near the corner flag.",
              "CONTEXT_STRUCTURE_REPORT.md, corner sweep: fitted on 10 SkillCorner matches, read on the other 10 and on Metrica's logged corners; agreed by the founder, 18 November 2026"),
    # -- Throw-ins from player positions (measured and agreed 18 November 2026) --
    Threshold("p4_throw_in_hold_s", 2.0, "s", "agreed",
              "Without a logged feed, a throw-in set-up is one player of the team (and none of the other) standing beyond a touchline for at least this long. 0 turns throw-in detection off.",
              "CONTEXT_STRUCTURE_REPORT.md, restart sweep: fitted on 10 SkillCorner matches, read on the other 10 and on Metrica's logged throw-ins; agreed by the founder, 18 November 2026"),
    Threshold("p4_throw_in_out_m", 0.0, "m", "agreed",
              "The thrower stands at least this far beyond the touchline (0: on the line or beyond). 0.5 and 1 m lost throw-ins; -0.5 m let in wide players.",
              "CONTEXT_STRUCTURE_REPORT.md, restart sweep: fitted on 10 SkillCorner matches, read on the other 10 and on Metrica's logged throw-ins; agreed by the founder, 18 November 2026"),
    Threshold("p4_throw_in_still_mps", 2.0, "m/s", "agreed",
              "Play has slowed: the median speed of all observed players over the set-up's last 2 s is at most this.",
              "CONTEXT_STRUCTURE_REPORT.md, restart sweep: fitted on 10 SkillCorner matches, read on the other 10 and on Metrica's logged throw-ins; agreed by the founder, 18 November 2026"),
    Threshold("p4_throw_in_ball_m", 5.0, "m", "agreed",
              "The ball is within this distance of the thrower at some point in the set-up's last 2 s. Without a ball test, about half of all detections are wrong (wide players on the line in open play).",
              "CONTEXT_STRUCTURE_REPORT.md, restart sweep: fitted on 10 SkillCorner matches, read on the other 10 and on Metrica's logged throw-ins; agreed by the founder, 18 November 2026"),
    # -- P5 line breaks (18 November 2026; proposed, from FIFA's definition) --
    Threshold("p5_min_unit_players", 2.0, "players", "proposed",
              "A defending line (unit) counts for line breaks only with at least this many players, as FIFA's Football Language defines a unit line.",
              "FIFA Football Language, penetrative action; ponta-blueprint/fifa-alignment.md; CONTEXT_STRUCTURE_REPORT.md, line breaks"),
    Threshold("p4_min_phase_s", 1.0, "s", "agreed",
              "A phase shorter than this merges into the phase before it in the same possession.", "Proposed in CONTEXT_STRUCTURE_REPORT.md; agreed by the founder, 17 November 2026"),
)

THRESHOLDS: dict[str, Threshold] = {t.name: t for t in _ALL}


def values(overrides: dict[str, float] | None = None) -> dict[str, float]:
    """Current values by name, with optional overrides (used by measurement sweeps only)."""
    out = {k: t.value for k, t in THRESHOLDS.items()}
    if overrides:
        unknown = set(overrides) - set(out)
        if unknown:
            raise KeyError(f"unknown thresholds: {sorted(unknown)}")
        out.update(overrides)
    return out
