"""
parity.py — the Analytics Parity Report (shown in the product as the "Data check").

For every match, each metric gets a state, **Full**, **Degraded** or **Blocked**, a
plain-language reason, the measured figure where there is one, and what would change
it. The report is computed deterministically from the data and its provenance, never
by a model, and it is JSON-serialisable (stored as JSONB on the match record). The
brief and Ask Ponta read it before saying anything (``ponta-blueprint/data-tiers.md``).

It generalises the prototype's SkillCorner "Analytics Parity" panel (``app.py``,
SkillCorner sidebar), which was fixed text. The rules below reproduce that panel's
classification exactly for SkillCorner open data (tested), and extend it to the other
modes:

* **GPS only** (one team, no ball, no events): opposition pressure, control zones and
  everything that needs the ball or events are Blocked; physical summaries are Full.
* **Event only**: decision quality is Degraded and flagged as a separate confidence
  class; anything needing continuous positions is Blocked or Degraded.
* **Optical tracking**: Full, Degraded with the measured figure when the frame rate
  is low or players drop out, and the event feed's limits applied on top.
* **No physiological source**: every load or fatigue metric is Blocked.

Changing a rule here changes what the product claims: it is a material change, to be
versioned and backfilled visibly (``ponta-blueprint/engine-governance.md``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import metadata
from typing import Literal

import numpy as np
import pandas as pd

from matrix_engine.match import NormalisedMatch

SCHEMA_VERSION = "1.0"

State = Literal["full", "degraded", "blocked"]
Mode = Literal["full_tracking", "event_only", "gps_only"]

#: Below this frame rate, continuous metrics are Degraded. SkillCorner's 10 Hz is
#: Full in the prototype's own classification, so the threshold sits below it.
LOW_RATE_HZ = 10.0
#: Above this share of missing player observations, position-based metrics are Degraded.
MAX_PLAYER_DROPOUT = 0.10

#: Providers with a pass completion parameter set fitted on their own data. Mirrors
#: ``plaiq_engine.analytics.pass_probability.PARAMETER_SETS`` (Matrix does not import Plaiq; a
#: cross-engine test keeps the two in step). Others use the default set fitted on another source.
PASS_COMPLETION_FITTED_PROVIDERS: tuple[str, ...] = ("skillcorner",)

#: Metric id -> (family, label). Order is the display order.
METRICS: dict[str, tuple[str, str]] = {
    "hex_grid": ("spatial", "Hexagonal analysis grid"),
    "xt_surface": ("spatial", "Threat surface (xT)"),
    "pressure_1v1": ("pressure", "1v1 pressure"),
    "pressure_zonal": ("pressure", "Zonal pressure"),
    "control_zones": ("control", "Control zones"),
    "movement_value": ("movement", "Off-ball movement value"),
    "pass_option_value": ("decisions", "Pass option value"),
    "decision_quality": ("decisions", "Decision quality: Gain"),
    "pass_completion": ("decisions", "Pass completion probability"),
    "decision_quality_v2": ("decisions", "Decision quality: Choice"),
    "gain_v2": ("decisions", "Decision quality: Gain (version 2)"),
    "event_delta_v": ("value", "Event value change (ΔV)"),
    "shot_quality": ("value", "Shot quality"),
    "physical_summary": ("physical", "Physical summary (distance, speed)"),
    "load_and_fatigue": ("physical", "Load and fatigue context"),
}


@dataclass(frozen=True)
class ParityInputs:
    """
    Everything the classification depends on. Measured from the data where possible.

    Attributes
    ----------
    provider : str
    mode : str
        ``full_tracking`` (positions for both teams and the ball), ``event_only``
        (an event feed, optionally with freeze-frames) or ``gps_only`` (wearable
        positions for one team).
    sample_rate_hz : float or None
        Tracking rate; ``None`` for event-only data.
    teams_tracked : tuple of str
        Teams whose positions are present (``home``, ``away``).
    ball_detected_share : float or None
        Share of frames with a ball position.
    player_dropout : float or None
        Share of expected player observations missing (11 per tracked team per frame
        in play).
    has_events : bool
        An event feed is present.
    has_shot_events : bool
        The event feed contains shots.
    events_derived_from_tracking : bool
        Events were inferred from tracking (SkillCorner dynamic events), not logged
        by an event provider.
    has_freeze_frames : bool
        Event-only data carries player snapshots (StatsBomb 360).
    physiological_channels : tuple of str
        Heart rate, IMU and similar channels present. None are ingested today.
    fusion_tier : int
        1 single source, 2 two unpaired sources, 3 paired sources.
    """

    provider: str
    mode: Mode
    sample_rate_hz: float | None = None
    teams_tracked: tuple[str, ...] = ()
    ball_detected_share: float | None = None
    player_dropout: float | None = None
    has_events: bool = False
    has_shot_events: bool = False
    events_derived_from_tracking: bool = False
    has_freeze_frames: bool = False
    physiological_channels: tuple[str, ...] = ()
    fusion_tier: int = 1


@dataclass(frozen=True)
class MetricParity:
    family: str
    label: str
    state: State
    reason: str
    measure: str | None = None
    unlock: str | None = None


@dataclass(frozen=True)
class PackageEntry:
    """
    One Data check line contributed by a context package (``ponta-blueprint/context-packages.md``).

    Context packages live outside ``matrix_engine`` (some need Plaiq), so they build these
    entries themselves and attach them with :func:`with_package_entries`; ``matrix_engine``
    never imports package code. ``coverage`` is the share of match time or events the
    package labelled, as a plain line (for example ``"98.2% of match time labelled"``).
    """

    package: str
    package_version: str
    metric: str
    label: str
    state: State
    reason: str
    measure: str | None = None
    unlock: str | None = None
    coverage: str | None = None


@dataclass
class AnalyticsParityReport:
    provider: str
    mode: Mode
    fusion_tier: int
    inputs: dict
    metrics: dict[str, MetricParity]
    coverage: dict
    schema_version: str = SCHEMA_VERSION
    engine_version: str = field(default_factory=lambda: _engine_version())
    #: Context package entries, by package name. Empty unless packages added some; the
    #: base metrics, their summary and the serialised form are unchanged when empty.
    packages: dict[str, list[PackageEntry]] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, int]:
        out = {"full": 0, "degraded": 0, "blocked": 0}
        for m in self.metrics.values():
            out[m.state] += 1
        return out

    def states(self) -> dict[str, State]:
        return {k: m.state for k, m in self.metrics.items()}

    def package_summary(self) -> dict[str, int]:
        """State counts over context package entries only (the base summary is separate)."""
        out = {"full": 0, "degraded": 0, "blocked": 0}
        for entries in self.packages.values():
            for e in entries:
                out[e.state] += 1
        return out

    def to_dict(self) -> dict:
        """JSON-serialisable form, as stored on the match record."""
        d = self._base_dict()
        if self.packages:
            d["packages"] = {name: [asdict(e) for e in entries] for name, entries in sorted(self.packages.items())}
            d["package_summary"] = self.package_summary()
        return d

    def _base_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "provider": self.provider,
            "mode": self.mode,
            "fusion_tier": self.fusion_tier,
            "inputs": self.inputs,
            "summary": self.summary,
            "coverage": self.coverage,
            "metrics": {k: asdict(m) for k, m in self.metrics.items()},
        }


def with_package_entries(report: AnalyticsParityReport, entries: list[PackageEntry]) -> AnalyticsParityReport:
    """
    Return a copy of ``report`` with context package entries attached, grouped by package.

    The base metrics are untouched. A package's entries replace any earlier entries from
    the same package, so re-running a package does not duplicate its lines.
    """
    by_pkg: dict[str, list[PackageEntry]] = {k: list(v) for k, v in report.packages.items()}
    new: dict[str, list[PackageEntry]] = {}
    for e in entries:
        new.setdefault(e.package, []).append(e)
    by_pkg.update(new)
    return AnalyticsParityReport(
        provider=report.provider,
        mode=report.mode,
        fusion_tier=report.fusion_tier,
        inputs=report.inputs,
        metrics=report.metrics,
        coverage=report.coverage,
        schema_version=report.schema_version,
        engine_version=report.engine_version,
        packages=by_pkg,
    )


def _engine_version() -> str:
    try:
        return f"matrix-{metadata.version('ponta-matrix-engine')}"
    except metadata.PackageNotFoundError:  # pragma: no cover - only outside an install
        return "matrix-unknown"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(inputs: ParityInputs) -> AnalyticsParityReport:
    """Apply the parity rules to measured inputs. Pure and deterministic."""
    m: dict[str, MetricParity] = {}

    def put(mid: str, state: State, reason: str, measure: str | None = None, unlock: str | None = None) -> None:
        fam, label = METRICS[mid]
        m[mid] = MetricParity(fam, label, state, reason, measure, unlock)

    both_teams = {"home", "away"} <= set(inputs.teams_tracked)
    add_tracking = "Add an optical tracking export that covers both teams and the ball."

    # Static geometry: computable whatever the source.
    put("hex_grid", "full", "The analysis grid is fixed pitch geometry.")
    put("xt_surface", "full", "The threat surface is a fixed model of the pitch.")

    if inputs.mode == "gps_only":
        why = "Wearable positions cover one team only, with no ball and no opposition."
        for mid in ("pressure_1v1", "pressure_zonal", "control_zones"):
            put(mid, "blocked", why, unlock=add_tracking)
        for mid in ("movement_value", "pass_option_value", "decision_quality", "pass_completion", "decision_quality_v2", "event_delta_v"):
            put(mid, "blocked", "Needs the ball, both teams and an event feed.", unlock=add_tracking)
        put("shot_quality", "blocked", "No event feed.", unlock="Add an event feed that includes shots.")
        put("physical_summary", "full", "Distance and speed from the wearable positions.",
            measure=f"{inputs.sample_rate_hz:g} Hz" if inputs.sample_rate_hz else None)

    elif inputs.mode == "event_only":
        if inputs.has_freeze_frames:
            snap = "Player positions exist only in freeze-frame snapshots, with no velocities."
            put("pressure_1v1", "degraded", snap, unlock=add_tracking)
            put("pressure_zonal", "degraded", snap, unlock=add_tracking)
            put("control_zones", "degraded", snap + " Control is computed with players standing still.", unlock=add_tracking)
        else:
            for mid in ("pressure_1v1", "pressure_zonal", "control_zones"):
                put(mid, "blocked", "The event feed has no player positions.", unlock=add_tracking)
        put("movement_value", "blocked", "Needs continuous positions between events.", unlock=add_tracking)
        put("pass_option_value", "degraded", "Options are estimated from event locations and any snapshots.", unlock=add_tracking)
        put("decision_quality", "degraded",
            "Event-only estimate: a separate confidence class, shown as a range and never on the same band as tracked scores.",
            unlock=add_tracking)
        put("pass_completion", "blocked", "Needs every player's position and speed at the pass.", unlock=add_tracking)
        put("decision_quality_v2", "blocked", "Needs every player's position and speed at the pass.", unlock=add_tracking)
        put("event_delta_v", "full" if inputs.has_shot_events else "degraded",
            "Event start and end locations are in the feed." if inputs.has_shot_events else "No shot events, so the value of shots is missing.")
        put("shot_quality", "full" if inputs.has_shot_events else "blocked",
            "Shots are in the event feed." if inputs.has_shot_events else "No shot events in this feed.")
        put("physical_summary", "blocked", "No positions over time, so no distance or speed.",
            unlock="Add tracking or a wearable export for the same match.")

    else:  # full_tracking
        rate = inputs.sample_rate_hz or 0.0
        drop = inputs.player_dropout or 0.0
        measure = f"{rate:g} Hz, {_pct(drop)} of player observations missing"
        reasons = []
        if rate < LOW_RATE_HZ:
            reasons.append(f"tracked at {rate:g} Hz, below {LOW_RATE_HZ:g} Hz")
        if drop > MAX_PLAYER_DROPOUT:
            reasons.append(f"{_pct(drop)} of player observations are missing")
        if not both_teams:
            reasons.append("only one team is tracked")
        pos_state: State = "degraded" if reasons else "full"
        pos_reason = ("Positions are " + "; ".join(reasons) + ".") if reasons else "Both teams are tracked throughout."
        pos_unlock = "Use a higher-rate or more complete tracking export." if reasons else None
        for mid in ("pressure_1v1", "pressure_zonal", "control_zones", "movement_value"):
            put(mid, pos_state, pos_reason, measure=measure, unlock=pos_unlock)

        if not inputs.has_events:
            for mid in ("pass_option_value", "decision_quality", "pass_completion", "decision_quality_v2", "event_delta_v"):
                put(mid, "blocked", "No event feed, so no passes or carries to score.", unlock="Add the event feed for this match.")
            put("shot_quality", "blocked", "No event feed.", unlock="Add an event feed that includes shots.")
        else:
            no_shots = "No shot events, so terminal (shot) value is missing."
            put("pass_option_value", "degraded" if not inputs.has_shot_events else pos_state,
                no_shots if not inputs.has_shot_events else pos_reason, measure=measure,
                unlock="Add an event feed that includes shots." if not inputs.has_shot_events else pos_unlock)
            if inputs.events_derived_from_tracking:
                put("decision_quality", "degraded",
                    "Events are derived from tracking, so only the decisions that feed covers are scored.",
                    unlock="Add a logged event feed for this match.")
            else:
                put("decision_quality", pos_state, pos_reason, measure=measure, unlock=pos_unlock)
            if inputs.provider.lower() in PASS_COMPLETION_FITTED_PROVIDERS:
                put("pass_completion", pos_state, pos_reason, measure=measure, unlock=pos_unlock)
            else:
                put("pass_completion", "degraded",
                    "Fitted on another source's tracking, not this provider's: it ranks passes, but its probabilities may be off by "
                    "several points. Passes without a logged target get no probability.",
                    measure=measure, unlock="Fit the pass completion parameters on this provider's matches.")
            # Decision quality v2 weighs every option by the pass completion probability, so it inherits that
            # metric's fit, and like v1 it scores only the passes the event feed covers.
            if inputs.provider.lower() not in PASS_COMPLETION_FITTED_PROVIDERS:
                put("decision_quality_v2", "degraded",
                    "Its risk term, the pass completion probability, is fitted on another source's tracking. Passes without a "
                    "logged target, including most failed passes, get no score.",
                    measure=measure, unlock="Fit the pass completion parameters on this provider's matches.")
            elif inputs.events_derived_from_tracking:
                put("decision_quality_v2", "degraded",
                    "Events are derived from tracking, so only the decisions that feed covers are scored.",
                    unlock="Add a logged event feed for this match.")
            else:
                put("decision_quality_v2", pos_state, pos_reason, measure=measure, unlock=pos_unlock)
            put("event_delta_v", "full" if inputs.has_shot_events else "degraded",
                "Event start and end locations are in the feed." if inputs.has_shot_events else "No shot events, so the value of shots is missing.",
                unlock=None if inputs.has_shot_events else "Add an event feed that includes shots.")
            if inputs.has_shot_events and not inputs.events_derived_from_tracking:
                put("shot_quality", "full", "Shots are in the logged event feed.")
            elif inputs.has_shot_events:
                put("shot_quality", "blocked",
                    "Shots in this feed are derived from tracking, without the detail a shot-quality model needs.",
                    unlock="Add a logged event feed for this match.")
            else:
                put("shot_quality", "blocked", "No shot events in this feed.", unlock="Add an event feed that includes shots.")
        put("physical_summary", "degraded",
            "Distance and speed derived from optical positions, not measured by a wearable.",
            measure=f"{rate:g} Hz", unlock="Add a wearable export for the same match.")

    # Load and fatigue: no physiological ingestion exists, so always Blocked.
    if inputs.physiological_channels:
        put("load_and_fatigue", "blocked",
            "Physiological channels are present but Ponta does not ingest them yet.",
            measure=", ".join(inputs.physiological_channels), unlock="In development.")
    else:
        put("load_and_fatigue", "blocked", "No physiological source (heart rate or IMU) for this match.",
            unlock="Pair a wearable export with this match. Paired-source analysis is in development.")

    # Gain version 2 is computed from Choice's parts (its best expected value on offer), so it has Choice's state.
    v2 = m["decision_quality_v2"]
    put("gain_v2", v2.state, v2.reason, measure=v2.measure, unlock=v2.unlock)

    ordered = {k: m[k] for k in METRICS}
    coverage = {
        "ball_detected_share": inputs.ball_detected_share,
        "player_observation_share": None if inputs.player_dropout is None else 1.0 - inputs.player_dropout,
    }
    return AnalyticsParityReport(
        provider=inputs.provider,
        mode=inputs.mode,
        fusion_tier=inputs.fusion_tier,
        inputs=asdict(inputs) | {"teams_tracked": list(inputs.teams_tracked), "physiological_channels": list(inputs.physiological_channels)},
        metrics=ordered,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Measuring inputs from a match
# ---------------------------------------------------------------------------

def player_dropout(match: NormalisedMatch, *, per_team: int = 11) -> float:
    """
    Share of expected player observations missing, over frames in play.

    Expected is ``per_team`` players for each tracked team in every frame where at
    least one player of that team was observed; extra observations (substitutes
    briefly overlapping) do not offset missing ones.
    """
    tr = match.tracking
    if tr.empty:
        return 1.0
    counts = tr.groupby(["team", "frame"]).size().clip(upper=per_team)
    return float(1.0 - counts.sum() / (len(counts) * per_team))


def inputs_from_match(
    match: NormalisedMatch,
    *,
    mode: Mode = "full_tracking",
    physiological_channels: tuple[str, ...] = (),
    fusion_tier: int = 1,
) -> ParityInputs:
    """Measure the parity inputs from a Matrix-normalised match."""
    ev = match.events
    types = ev["Type"].astype(str).str.upper() if "Type" in ev.columns else pd.Series(dtype=str)
    return ParityInputs(
        provider=match.provider,
        mode=mode,
        sample_rate_hz=match.fps,
        teams_tracked=tuple(sorted(set(match.tracking["team"].unique()))),
        ball_detected_share=round(match.ball_detected_share(), 4),
        player_dropout=round(player_dropout(match), 4),
        has_events=not ev.empty,
        has_shot_events=bool(np.any(types.str.contains("SHOT"))),
        events_derived_from_tracking=match.provider == "skillcorner",
        physiological_channels=physiological_channels,
        fusion_tier=fusion_tier,
    )


def parity_for_match(match: NormalisedMatch, **kwargs) -> AnalyticsParityReport:
    """Measure and classify in one step."""
    return classify(inputs_from_match(match, **kwargs))
