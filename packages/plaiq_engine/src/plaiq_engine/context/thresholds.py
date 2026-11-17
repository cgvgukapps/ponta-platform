"""
thresholds.py — every named threshold the context packages use, in one versioned place.

Each value carries a status (``ponta-blueprint/context-packages.md``, "Measure before setting a threshold"):

* ``proposed`` — chosen by the engineer, with the reason; not yet measured.
* ``measured`` — proposed from this repository's measurements on the regression set (the
  ``source`` names the measurement).
* ``agreed`` — set by the founder. Code never sets this status on its own; every agreed value names the decision in ``source``.

All values below were agreed by the founder on 17 November 2026 (``CONTEXT_FOUNDATIONS_REPORT.md``,
"Decisions"). The status changed, not the values, so ``THRESHOLDS_VERSION`` is unchanged.

Nothing threshold-like is hard-coded elsewhere in the context packages. Changing a value
changes package output: bump ``THRESHOLDS_VERSION`` and the version of every package that
reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Status = Literal["proposed", "measured", "agreed"]

THRESHOLDS_VERSION = "0.2.0"


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
