"""
thresholds.py — every named threshold the context packages use, in one versioned place.

Each value carries a status (``ponta-blueprint/context-packages.md``, "Measure before setting a threshold"):

* ``proposed`` — chosen by the engineer, with the reason; not yet measured.
* ``measured`` — proposed from this repository's measurements on the regression set (the
  ``source`` names the measurement).
* ``agreed`` — set by the founder. Code never sets this status on its own; every agreed value names the decision in ``source``.

The P1 values were agreed by the founder on 17 November 2026 (``CONTEXT_FOUNDATIONS_REPORT.md``,
"Decisions"). Version 0.3.0 adds the Stage B thresholds (P2 to P4), proposed or measured
(``CONTEXT_STRUCTURE_REPORT.md``); P1's values are unchanged.

Nothing threshold-like is hard-coded elsewhere in the context packages. Changing a value
changes package output: bump ``THRESHOLDS_VERSION`` and the version of every package that
reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Status = Literal["proposed", "measured", "agreed"]

THRESHOLDS_VERSION = "0.3.0"


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
    # -- P2 team structure (Stage B; proposed until measured and agreed) ------------------
    Threshold("p2_window_s", 120.0, "s", "proposed",
              "Formation and line windows are two minutes within a period: long enough for stable mean positions, short enough to see changes.",
              "Shaw and Glickman 2019 (two-minute windows per possession state)"),
    Threshold("p2_min_possession_s", 5.0, "s", "proposed",
              "Only possessions at least this long feed a window's in- or out-of-possession shape; shorter ones are transitions.",
              "Shaw and Glickman 2019 (possessions under 5 s excluded)"),
    Threshold("p2_min_window_data_s", 10.0, "s", "proposed",
              "A window needs this much in-state time to fit lines and a formation; less is skipped.", None),
    Threshold("p2_min_presence_share", 0.5, "share of window", "proposed",
              "A player counts in a window when observed for at least this share of it (handles substitutions).", None),
    Threshold("p2_min_outfield_players", 8.0, "players", "proposed",
              "A team's shape is measured only with at least this many outfield players observed (allows two red cards or out-of-view players).", None),
    Threshold("p2_min_shape_coverage", 0.9, "share of in-play time", "proposed",
              "Team shape is Degraded when fewer than this share of in-play grid points have a shape for both teams.", None),
    Threshold("p2_formation_min_margin", 0.1, "relative cost", "proposed",
              "A window's formation is unclear when the best template's fit is less than this much better than the second best ((c2 - c1) / c2).", None),
    Threshold("p2_formation_persist_s", 300.0, "s of window data", "proposed",
              "A formation change is accepted only when the new formation holds for this long.",
              "Kim et al. 2022, SoccerCPD (formation changes persist at least 5 minutes)"),
    # -- P3 relative zones -----------------------------------------------------------------
    Threshold("p3_min_coverage", 0.75, "share of possession time", "proposed",
              "Zones are Degraded when the ball or the defending shape is missing for more than a quarter of possession time (matches P1's 25%).", None),
    # -- P4 phase of play -------------------------------------------------------------------
    Threshold("p4_low_block_max_m", 26.0, "m from own goal line", "measured",
              "The defending team is in a low block when its three deepest outfield players average less than this from their goal line (about 10 m outside the penalty area). Equal pitch thirds (35 m) agreed far worse with the reference (kappa 0.20).",
              "CONTEXT_STRUCTURE_REPORT.md, phase sweep against SkillCorner 1886347 (one match; the value holds on both halves)"),
    Threshold("p4_high_block_min_m", 47.0, "m from own goal line", "measured",
              "The defending team is in a high block when its three deepest outfield players average at least this from their goal line (a few metres short of halfway).",
              "CONTEXT_STRUCTURE_REPORT.md, phase sweep against SkillCorner 1886347 (one match; the value holds on both halves)"),
    Threshold("p4_set_play_max_s", 20.0, "s", "proposed",
              "A set play lasts at most this long after the restart.", "SkillCorner phases of play specification, 16 February 2025 (set play up to 20 s)"),
    Threshold("p4_corner_radius_m", 4.0, "m", "proposed",
              "Without a logged event feed, a possession whose first ball position is within this distance of a corner flag is a corner (the corner arc is 1 m; tracking at restarts is noisy).", None),
    Threshold("p4_corner_window_s", 1.0, "s", "proposed",
              "Without a logged event feed, the time at the start of a possession in which the ball must come near a corner flag for a corner.", None),
    Threshold("p4_direct_min_gain_m", 32.0, "m", "proposed",
              "A ball from the own half that gains at least this much towards goal is a direct phase.",
              "SkillCorner phases of play specification, 16 February 2025 (long ball of at least 32 m)"),
    Threshold("p4_direct_window_s", 3.0, "s", "proposed",
              "The time within which the direct gain must happen (a long ball's flight, not a run).", None),
    Threshold("p4_chaotic_max_s", 5.0, "s", "proposed",
              "A possession chain shorter than this is chaotic (SkillCorner also spares chains with three or more passes; passes are not counted here).",
              "SkillCorner phases of play specification, 16 February 2025 (chaotic unless at least 5 s or 3 passes)"),
    Threshold("p4_break_eval_s", 3.0, "s", "proposed",
              "After a regain, the ball's gain towards goal is measured over this time to decide whether it is a quick break or transition.", None),
    Threshold("p4_break_min_gain_m", 15.0, "m", "measured",
              "The gain towards goal within the evaluation time that makes a regain a quick break or transition. 15 m agreed slightly better than 5 or 10 m on both halves.",
              "CONTEXT_STRUCTURE_REPORT.md, phase sweep against SkillCorner 1886347 (one match; the value holds on both halves)"),
    Threshold("p4_break_max_s", 15.0, "s", "proposed",
              "A quick break or transition lasts at most this long.", None),
    Threshold("p4_recovered_players", 7.0, "players", "proposed",
              "A quick break or transition ends when this many defending outfield players are goal-side of the ball.", None),
    Threshold("p4_min_phase_s", 1.0, "s", "proposed",
              "A phase shorter than this merges into the phase before it in the same possession.", None),
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
