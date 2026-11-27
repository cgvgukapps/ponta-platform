"""
semantic_builder.py — compile a match into semantic football events.

One row per pass, carry, shot, turnover, defensive action, run (P6) and set piece, foul won
or card, written to ``semantic_events.parquet`` (schema 2.1). Plaiq's per-event movement values for
team-mates are a table of their own (``movement_values.parquet``). This is the only thing the AI layer reads for evidence
(``ponta-blueprint/ai-layer.md``, "The semantic football event"). It also writes
``trajectories.parquet`` for the replay, which the AI layer never reads.

Rules this module enforces:

* **Pure and deterministic.** Same match, same output: ids are derived from the
  event's content (schema 2.0), rows are explicitly sorted, nothing depends on engine row order.
* **No names, no provider ids and no coordinate arrays** in semantic events. Players are
  pseudonymous references (``home_10``, a shirt number or the build fails); ``match_id``,
  team, competition and season are opaque keys (``plaiq_engine.teams``); location is a
  categorical ``pitch_zone``. The provider's own ids are kept apart in ``provenance()``,
  which the AI layer never reads.
* **Conditional sections.** ``physiological_context`` is null unless the match has a
  physiological source and the fields exist. Nothing is zero-filled.
* **Not scored, with a reason.** Goalkeepers and set pieces carry no decision
  quality: ``not_scored`` is set and the score fields are null.
* Every row carries ``schema_version``, ``engine_version``, ``match_id``,
  ``timestamp_s`` and its data-check annotation.

Columns are flat (``actor_player_ref``, ``dq_score`` …) so the typed query tool can
filter them directly; ``nested_record`` gives the nested JSON view of ``ai-layer.md``.

Numbers come from the engine through ``MatchAnalysis``; this module selects and
labels them. The decision-quality bands are the engine's (``ANALYTICS_SPECIFICATION.md``
section 7.1).
"""

from __future__ import annotations

import hashlib
import json
import re
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.ingestion.metrica_io import find_goalkeeper
from matrix_engine.parity import NEAR_BALL_M, AnalyticsParityReport

from plaiq_engine.analytics.event_stitch import clean_jersey
from plaiq_engine.context.states import STATE_COLUMNS
from plaiq_engine.match_analysis import MatchAnalysis
from plaiq_engine.teams import PseudonymError, TeamRegistry

#: 1.1 (17 November 2026): adds the context columns from package P1 (possession chain and
#: source, score and red-card difference for the actor's team). Earlier columns are unchanged.
#: 1.2 (19 November 2026): adds the pass completion probability (``MatchAnalysis.pass_completion``,
#: adopted by the founder 19 November 2026), its status and its parameter set. Earlier columns are unchanged.
#: 1.3 (19 November 2026): adds decision quality v2, projected (``MatchAnalysis.decision_quality_v2``, adopted
#: beside v1 by the founder 19 November 2026): its score (0 to 1, no bands), status, best option and version.
#: A failed pass that v2 scores now gets a row even when v1 has no failed-pass score. Earlier columns are unchanged.
#: 1.4 (19 November 2026): decision quality v2 1.1.0 (hex cells, ring 1) with its agreed band, ``dq_v2_band``
#: (best_option, near_best, below_best, well_below_best; never version 1's names). Earlier columns are unchanged.
#: 1.5 (19 November 2026): version 1's bands get descriptive names (founder decision): full_gain, most_gain,
#: partial_gain, low_gain replace optimal, good, suboptimal, poor at the same cut-offs. Version 1 is shown as
#: "Gain" and version 2 as "Choice". Values change; columns do not.
#: 1.6 (20 November 2026): Gain version 2 (``MatchAnalysis.gain_v2``, adopted beside version 1): ``gain_v2_score``
#: (0 to 1.5, version 1's cut-offs and band names), ``gain_v2_band``, ``gain_v2_status``, ``gain_v2_version``. Passes only.
#: 1.7 (23 November 2026, Stage D): the Stage B and C context at each action's start, when a ContextResult is given
#: (phase, FIFA phase, block, formations, the ball's zone, line breaks, relationships, runs, adjusted Choice, open space,
#: engaging defenders), the episode it belongs to (``state_id``, P8) and the context versions. All estimated; labels and
#: counts only, never coordinates. Earlier columns are unchanged.
#: 1.8 (23 November 2026, Stage E): ``actor_team_key`` and ``opponent_team_key`` (pseudonymous team keys, stable across
#: matches; ``plaiq_engine.teams``), ``competition`` and ``season`` (the provider's ids, never names), on every row and
#: on ``semantic_states``. Earlier columns are unchanged.
#: 1.9 (25 November 2026, review backlog A): values change, columns do not. A lost pass no longer carries version 1's
#: failed-pass score: ``dq_score`` is null and ``dq_band`` ``not_scored``, with ``not_scored_reason`` saying why
#: (Choice and Gain version 2 still score it). ``semantic_states``' ``first_event_id`` and ``last_event_id`` are the first
#: and last semantic events inside the episode, so each resolves in ``semantic_events`` (before, P8's logged-event index).
#: ``players_observed`` counts only positions the provider observed where it flags estimates (before, any position), and
#: ``trajectories`` gains ``observed`` (null for the ball and where the provider does not say).
#: 2.0 (27 November 2026, review backlog B: AI-04, DE-03, GC-04, SE-15): identifiers change, columns do not. ``match_id``
#: is the opaque match key (``m1_…``, a keyed hash under the tenant secret) on every row, episode and profile, never the
#: provider's id; ``competition`` and ``season`` are opaque keys (``c1_…``, ``s1_…``); team keys carry their key version
#: (``t1_…``). ``event_id`` is derived from the event's content (match key, period, start frame, type, actor and order
#: among identical ones), so it survives re-ingestion and mapping changes elsewhere in the match; ``state_id`` and
#: ``possession_chain_id`` are built on the match key. Player refs must be shirt numbers (a name fails the build).
#: ``nested_record`` gains a ``team`` block. The provider's ids sit in ``provenance()`` (``event_provenance.parquet``),
#: which is not AI-facing.
#: 2.1 (27 November 2026, review backlog B: AI-06, AI-07): rows the AI needed and did not have. **Shots** (``shot``; outcome
#: goal, saved, blocked or off target where the feed says, ``action_detail`` ``head`` for headers; no shot-quality claim),
#: **turnovers** (``turnover``, every lost ball that is not a failed pass: ``action_detail`` dispossessed, clearance or
#: offside), **every failed pass** (before, only those a model scored), **set pieces** (``set_piece``, the kind in
#: ``action_detail``), **fouls won** (``foul_won``) and **cards** (``card``, yellow or red), and **P6's runs** (``run``, one
#: row per run with ``run_type`` and ``run_area``, FIFA's in front, in between or in behind). The ``off_ball_run`` rows
#: (one per team-mate the engine rewarded at each event, about 8 times P6's runs) leave this table for
#: ``movement_values()``. ``actor_role`` falls back to P2's main role, with ``actor_role_source`` (roster or estimated).
#: New columns: ``action_detail``, ``actor_role_source``, ``run_type``, ``run_area``; ``semantic_states`` gains ``shots`` and
#: ``ended_in_shot`` (a shot inside this episode, unlike the chain-level ``led_to_shot``).
#: 2.2 (27 November 2026, review AI-03): ``observed_near_ball_share``, the share of player positions within the Data
#: check's near-ball radius (15 m) the provider observed rather than estimated at the action's start (null where the
#: provider does not flag estimates). It feeds the citation's evidence grade (``citations.evidence_grade``).
SCHEMA_VERSION = "2.2"

#: Why a lost pass has no version 1 score.
LOST_PASS_REASON = "lost pass: Gain version 1 does not charge a pass for failing; see Choice and Gain version 2"

#: The label every pass completion probability carries (measured against provider references, not analysts).
PASS_COMPLETION_LABEL = "estimated"
#: The label every Gain version 2 score carries (measured against outcomes and a provider's analogue, not analysts).
GAIN_V2_LABEL = "estimated"
#: The label every decision quality v2 score carries (measured against outcomes and a provider's analogue, not analysts).
DQ_V2_LABEL = "estimated"

#: Decision-quality bands (ANALYTICS_SPECIFICATION.md 7.1): unified across event types. Version 1, shown as "Gain":
#: forward movement on the threat surface relative to the passer's position (description corrected 25 November 2026,
#: review SM-01; it is not the share of the best option's value). Cut-offs unchanged; names descriptive since schema 1.5
#: (they were optimal, good, suboptimal and poor).
BAND_CUTS: tuple[tuple[float, str], ...] = ((0.95, "full_gain"), (0.80, "most_gain"), (0.55, "partial_gain"))
LOWEST_BAND = "low_gain"

#: A movement value is kept for a team-mate the engine rewarded: run_bonus above this.
RUN_BONUS_MIN = 0.0
#: Defensive actions taken from the event feed.
DEFENSIVE_TYPES = {"RECOVERY", "CHALLENGE"}
#: A shot's outcome from the feed's subtype (first match wins); None where the feed gives none (SkillCorner).
SHOT_OUTCOMES: tuple[tuple[str, str], ...] = (("GOAL", "goal"), ("SAVED", "saved"), ("BLOCKED", "blocked"),
                                              ("WOODWORK", "off_target"), ("OFF TARGET", "off_target"))
#: A lost ball that is not a failed pass, by the feed's subtype (first match wins). Any other lost ball is a failed pass.
TURNOVER_DETAIL: tuple[tuple[str, str], ...] = (("THEFT", "dispossessed"), ("FORCED", "dispossessed"), ("CHALLENGE", "dispossessed"),
                                                ("CLEARANCE", "clearance"), ("OFFSIDE", "offside"), ("REFEREE", "other"),
                                                ("WOODWORK", "other"))
#: Lost-ball rows that are not a team losing the ball (the end of a half).
NOT_A_TURNOVER = ("END HALF",)
#: Set-piece kinds (``action_detail`` of a ``set_piece`` row).
SET_PIECE_KINDS = {"THROW IN": "throw_in", "FREE KICK": "free_kick", "CORNER KICK": "corner_kick", "GOAL KICK": "goal_kick",
                   "KICK OFF": "kick_off", "PENALTY": "penalty"}
#: Pitch length thirds (team frame, metres from the centre line).
THIRD_EDGE_M = 17.5
#: Channels across the pitch (team frame, metres from the centre line, positive = left).
CHANNEL_EDGES_M = (20.0, 9.0)

#: Schema 1.7's context columns (all estimated). ``phase`` and ``out_phase`` belong to ``phase_team``, the team in
#: possession; zones are in the actor's attacking direction (``zone_band`` only for the team in possession).
CONTEXT_COLUMNS: tuple[str, ...] = (
    "state_id", "phase_team", "phase", "out_phase", "fifa_phase", "fifa_out_phase", "block",
    "formation_in", "formation_out", "zone_lane", "zone_third", "zone_band",
    "units_broken", "furthest_line_broken", "received_inside_shape", "goal_side_defenders", "overload_at_ball",
    "runs_in_behind", "adjusted_choice", "open_high_value_share", "engaging_defenders", "context_versions",
)
MIRROR_THIRD = {"defensive": "attacking", "middle": "middle", "attacking": "defensive"}
#: Schema 1.8's team identity columns (opaque keys since schema 2.0).
TEAM_COLUMNS: tuple[str, ...] = ("actor_team_key", "opponent_team_key", "competition", "season")
#: ``semantic_states`` (schema 1.7): P8's episode row plus the event ids at its ends and the pass decision medians.
STATE_EXTRA_COLUMNS: tuple[str, ...] = (*STATE_COLUMNS[:STATE_COLUMNS.index("summary")], "first_event_id", "last_event_id",
                                        "gain_median", "choice_median", "adjusted_choice_median", "shots", "ended_in_shot",
                                        "team_key", "opp_team_key", "competition", "season", "summary", "schema_version")

#: ``movement_values`` (schema 2.1): Plaiq's off-ball movement value for each team-mate the engine rewarded at an event.
MOVEMENT_COLUMNS: tuple[str, ...] = ("schema_version", "event_id", "source_event_id", "match_id", "timestamp_s", "minute_display",
                                     "period", "player_ref", "team", "movement_value", "xt_delta", "opposition_pressure",
                                     "state_id", "label")

SEMANTIC_COLUMNS: tuple[str, ...] = (
    "schema_version", "engine_version", "event_id", "match_id", "event_idx", "timestamp_s", "minute_display", "period",
    "actor_player_ref", "actor_team", "actor_role", "actor_role_source",
    "action_type", "action_outcome", "action_detail", "target_player_ref", "inferred_intended_ref", "inference_confidence",
    "dq_score", "dq_band", "chosen_option_pov", "best_option_player_ref", "best_option_pov",
    "xt_delta", "movement_value", "opposition_pressure", "pitch_zone", "mode",
    "pass_completion_probability", "pass_completion_status", "pass_completion_parameters",
    "dq_v2_score", "dq_v2_band", "dq_v2_status", "dq_v2_best_option_ref", "dq_v2_version",
    "gain_v2_score", "gain_v2_band", "gain_v2_status", "gain_v2_version",
    "not_scored", "not_scored_reason",
    "run_type", "run_area",
    "physiological_context",
    "ball_detected", "players_observed", "observed_near_ball_share", "parity_spatial", "parity_physiological",
    # Context package P1 (null unless a ContextResult is given): labels, never coordinates.
    "possession_chain_id", "possession_source", "score_diff", "red_card_diff",
    # Context packages P2 to P8 (schema 1.7; null unless a ContextResult is given): labels and counts, never coordinates.
    *CONTEXT_COLUMNS,
    # Team identity (schema 1.8; opaque keys since 2.0): stable across matches, never the provider's ids.
    *TEAM_COLUMNS,
)


_SHIRT = re.compile(r"\d{1,3}")


def _first(table: tuple[tuple[str, str], ...], sub: str) -> str | None:
    """The label of the first keyword found in a feed subtype."""
    return next((label for key, label in table if key in sub), None)


def dq_band(score: float | None) -> str | None:
    if score is None or not np.isfinite(score):
        return None
    for cut, band in BAND_CUTS:
        if score >= cut:
            return band
    return LOWEST_BAND


def _engine_version() -> str:
    def v(dist: str) -> str:
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:  # pragma: no cover
            return "unknown"

    return f"plaiq-{v('ponta-plaiq-engine')} / matrix-{v('ponta-matrix-engine')}"


def _minute(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def pitch_zone(x: float, y: float, team: str) -> str | None:
    """Categorical zone in the team's own frame (home attacks +x; left is +y when attacking +x)."""
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    fx, fy = (x, y) if team == "home" else (-x, -y)
    third = "defensive_third" if fx < -THIRD_EDGE_M else "final_third" if fx > THIRD_EDGE_M else "middle_third"
    wide, half = CHANNEL_EDGES_M
    if fy > wide:
        channel = "left_wing"
    elif fy > half:
        channel = "left_half_space"
    elif fy >= -half:
        channel = "centre"
    elif fy >= -wide:
        channel = "right_half_space"
    else:
        channel = "right_wing"
    return f"{channel}_{third}"


class SemanticBuilder:
    """
    Build semantic events and trajectories for one analysed match.

    Parameters
    ----------
    analysis : MatchAnalysis
    parity : AnalyticsParityReport
        The match's data check; its states annotate every row.
    roles : dict, optional
        ``{player_ref: role}`` from the club's roster. Unknown roles stay null.
    context : ContextResult, optional
        Context package outputs for the same match (``plaiq_engine.context``). When given,
        each row carries the P1 context at its start frame; otherwise those columns are null.
    teams : TeamRegistry
        Required (schema 2.0, fails closed): the tenant's pseudonym registry, or
        ``TeamRegistry.for_reference()`` for tools and tests. Every key in the output comes from it.
    """

    def __init__(self, analysis: MatchAnalysis, parity: AnalyticsParityReport, *, teams: TeamRegistry,
                 roles: dict[str, str] | None = None, context=None) -> None:
        if not isinstance(teams, TeamRegistry):
            raise PseudonymError("SemanticBuilder needs a TeamRegistry (TeamRegistry.for_reference() for tools and tests)")
        self.a = analysis
        self.parity = parity
        self.roles = roles or {}
        m = analysis.match
        self.identity = teams.identity(m)
        self.match_id = self.identity.match_key
        self.provider_match_id = f"{m.provider}_{m.match_id}"  # provenance only, never in an AI-facing row
        self.team_keys = self.identity.team_keys
        self.competition = self.identity.competition_key
        self.season = self.identity.season_key
        self._ctx_match_id = (str(context.frames["match_id"].iloc[0])
                              if context is not None and "match_id" in getattr(context, "frames", pd.DataFrame()).columns
                              and len(context.frames) else None)
        self._ctx = self._context_lookup(context)
        self._ctx2 = self._context_lookup_17(context)
        self._context = context
        self.engine_version = _engine_version()
        self.goalkeepers = self._goalkeepers()
        self._ids: dict[int, str] | None = None
        self._est_roles = self._estimated_roles(context)

    @staticmethod
    def _estimated_roles(context) -> dict[str, str]:
        """P2's main role per player ref (estimated), for rows the roster gives no role for (schema 2.1, review AI-06)."""
        p2 = getattr(context, "outputs", {}).get("p2") if context is not None else None
        mr = None if p2 is None else p2.segments.get("main_roles")
        if mr is None or not len(mr):
            return {}
        return {str(r.player_id).lower(): str(r.role) for r in mr.itertuples() if isinstance(r.role, str) and r.role}

    def _role(self, ref: str | None) -> tuple[str | None, str | None]:
        if not ref:
            return None, None
        if ref in self.roles:
            return self.roles[ref], "roster"
        if ref in self._est_roles:
            return self._est_roles[ref], "estimated"
        return None, None

    def _state(self, sid):
        """P8's state id rebased on the match key (P8 builds it on the context grid's provider-form id)."""
        if sid is None or (not isinstance(sid, str) and pd.isna(sid)):
            return None
        sid = str(sid)
        cm = self._ctx_match_id
        return self.match_id + sid[len(cm):] if cm and sid.startswith(cm + ":") else sid

    def _event_ids(self) -> dict[int, str]:
        """
        ``event_idx -> event_id`` from each event's content (schema 2.0, review DE-03): the match key, period, start
        frame, type, actor ref and the event's order among identical ones. Adding or removing another event leaves every
        other id unchanged, which the loader's row position did not.
        """
        if self._ids is None:
            ev = self.a.events
            seen: dict[tuple, int] = {}
            ids: dict[int, str] = {}
            for idx, e in ev.iterrows():
                team = str(e.get("Team", "")).lower()
                actor = self._ref(team, e.get("From")) if team in ("home", "away") else None
                start = e.get("Start Frame")
                key = (int(e["Period"]) if pd.notna(e.get("Period")) else -1, int(start) if pd.notna(start) else -1,
                       str(e.get("Type", "")).upper(), team, actor or "")
                seq = seen.get(key, 0)
                seen[key] = seq + 1
                raw = "|".join([self.match_id, *map(str, key), str(seq)])
                ids[int(idx)] = "ev_" + hashlib.sha256(raw.encode()).hexdigest()[:16]
            self._ids = ids
        return self._ids

    def provenance(self) -> pd.DataFrame:
        """
        The provider's ids behind each event (``event_provenance.parquet``): for audits, de-duplication and joins back to
        the vendor's file. **Not AI-facing**: the API keeps it beside Postgres and never puts it in a prompt.
        """
        ev = self.a.events
        pid = next((c for c in ("provider_event_id", "sb_event_id", "event_id", "Event Id") if c in ev.columns), None)
        ids = self._event_ids()
        return pd.DataFrame({
            "event_id": [ids[int(i)] for i in ev.index], "event_idx": ev.index.astype(int),
            "match_id": self.match_id, "provider": self.a.match.provider, "provider_match_id": str(self.a.match.match_id),
            "provider_event_id": (ev[pid].astype(object).where(ev[pid].notna(), None).astype(str).where(ev[pid].notna(), None)
                                  if pid else None),
            "schema_version": SCHEMA_VERSION,
        })

    # -- helpers ---------------------------------------------------------------

    def _context_lookup(self, context):
        """P1 context by native frame: the grid point at or before each frame."""
        if context is None or "p1_possession" not in getattr(context, "frames", pd.DataFrame()).columns:
            return None
        f = context.frames.sort_values("native_frame", kind="stable")
        return {
            "frame": f["native_frame"].to_numpy(),
            "chain": f["p1_chain_id"].to_numpy(dtype=object),
            "source": f["p1_possession_source"].to_numpy(dtype=object),
            "score": {t: f[f"p1_score_{t}"].to_numpy(dtype=object) for t in ("home", "away")},
            "red": {t: f[f"p1_red_{t}"].to_numpy(dtype=object) for t in ("home", "away")},
        }

    def _context_fields(self, frame: int | None, team: str) -> dict:
        empty = {"possession_chain_id": None, "possession_source": None, "score_diff": None, "red_card_diff": None}
        c = self._ctx
        if c is None or frame is None or team not in ("home", "away"):
            return empty
        i = int(np.searchsorted(c["frame"], frame, side="right")) - 1
        if i < 0:
            return empty
        other = "away" if team == "home" else "home"

        def diff(block):
            a, b = block[team][i], block[other][i]
            return None if a is None or b is None or pd.isna(a) or pd.isna(b) else int(a) - int(b)

        chain = c["chain"][i]
        return {
            "possession_chain_id": None if chain is None or pd.isna(chain) else f"{self.match_id}:ch{int(chain):04d}",
            "possession_source": None if c["source"][i] is None else str(c["source"][i]),
            "score_diff": diff(c["score"]),
            "red_card_diff": diff(c["red"]),
        }

    def _context_lookup_17(self, context):
        """Schema 1.7's context (P2 to P8) by native frame; per-pass fields from the P5 and P7 segments."""
        if context is None or "p4_phase" not in getattr(context, "frames", pd.DataFrame()).columns:
            return None
        f = context.frames.sort_values("native_frame", kind="stable")

        def col(name, ffill=False):
            if name not in f:
                return np.full(len(f), None, dtype=object)
            s = f[name].astype(object).where(f[name].notna(), None)
            return (s.ffill() if ffill else s).to_numpy(dtype=object)

        out = {"frame": f["native_frame"].to_numpy(),
               **{k: col(k) for k in ("p8_state_id", "p4_team", "p4_phase", "p4_out_phase", "p4v_phase", "p4v_out_phase", "p4_block",
                                       "p3_attacking_team", "p3_ball_lane", "p3_ball_third", "p3_ball_band", "p5_goal_side_defenders",
                                       "p5_overload_at_ball", "p6_runs_in_behind", "p4v_engaging_defenders")},
               **{f"p2_{t}_formation_{k}": col(f"p2_{t}_formation_{k}") for t in ("home", "away") for k in ("in", "out")},
               **{f"p7_{t}_high_value_open": col(f"p7_{t}_high_value_open", ffill=True) for t in ("home", "away")}}
        lb = context.outputs["p5"].segments.get("line_breaks") if "p5" in context.outputs else None
        out["passes"] = {} if lb is None else {int(r.event_row): r for r in lb.itertuples()}
        adj = context.outputs["p7"].segments.get("adjusted_choice") if "p7" in context.outputs else None
        out["adjusted"] = {} if adj is None else dict(zip(adj["event_idx"].astype(int), adj["adjusted"], strict=True))
        versions = {c[:-len("_version")]: str(f[c].iloc[0]) for c in f.columns if c.endswith("_version") and len(f)}
        out["versions"] = json.dumps({**versions, "thresholds": getattr(context, "thresholds_version", None)}, sort_keys=True)
        return out

    def _context_fields_17(self, idx: int, frame: int | None, team: str) -> dict:
        empty = dict.fromkeys(CONTEXT_COLUMNS)
        c = self._ctx2
        if c is None or frame is None or team not in ("home", "away"):
            return empty
        i = int(np.searchsorted(c["frame"], frame, side="right")) - 1
        if i < 0:
            return empty
        other = "away" if team == "home" else "home"

        def num(v):
            return None if v is None or pd.isna(v) else (int(v) if float(v).is_integer() else round(float(v), 4))

        lane, third, band = c["p3_ball_lane"][i], c["p3_ball_third"][i], c["p3_ball_band"][i]
        if c["p3_attacking_team"][i] not in (None, team):  # zones are the team in possession's: mirror them for the other
            third = MIRROR_THIRD.get(third, third)
            lane = None if lane is None else (lane.replace("left", "@").replace("right", "left").replace("@", "right"))
            band = None
        p = c["passes"].get(idx)
        adj = c["adjusted"].get(idx)
        return {
            "state_id": self._state(c["p8_state_id"][i]), "phase_team": c["p4_team"][i], "phase": c["p4_phase"][i], "out_phase": c["p4_out_phase"][i],
            "fifa_phase": c["p4v_phase"][i], "fifa_out_phase": c["p4v_out_phase"][i], "block": c["p4_block"][i],
            "formation_in": c[f"p2_{team}_formation_in"][i], "formation_out": c[f"p2_{other}_formation_out"][i],
            "zone_lane": lane, "zone_third": third, "zone_band": band,
            "units_broken": None if p is None else num(p.units_broken),
            "furthest_line_broken": None if p is None or p.furthest is None or pd.isna(p.furthest) else str(p.furthest),
            "received_inside_shape": None if p is None or p.received_inside is None or pd.isna(p.received_inside) else bool(p.received_inside),
            "goal_side_defenders": num(c["p5_goal_side_defenders"][i]), "overload_at_ball": num(c["p5_overload_at_ball"][i]),
            "runs_in_behind": num(c["p6_runs_in_behind"][i]),
            "adjusted_choice": None if adj is None or pd.isna(adj) else round(float(adj), 4),
            "open_high_value_share": num(c[f"p7_{team}_high_value_open"][i]),
            "engaging_defenders": num(c["p4v_engaging_defenders"][i]),
            "context_versions": c["versions"],
        }

    def _goalkeepers(self) -> set[str]:
        out = set()
        for team, wide in (("home", self.a.home), ("away", self.a.away)):
            prefix = team.capitalize()
            xcols = [c for c in wide.columns if c.startswith(f"{prefix}_") and c.endswith("_x")]
            live = wide.loc[wide[xcols].notna().sum(axis=1) >= min(11, len(xcols))]
            if len(live):
                out.add(f"{team}_{find_goalkeeper(live, prefix)}")
        return out

    def _ref(self, team: str, jersey) -> str | None:
        """``home_10``: the side and a shirt number. Anything else (a name, a provider id) fails the build (review GC-04)."""
        j = clean_jersey(jersey)
        if not j:
            return None
        if not _SHIRT.fullmatch(j):
            raise PseudonymError(f"a player ref must be a shirt number; the feed gave a non-numeric value for {team.lower()}")
        return f"{team.lower()}_{j}"

    def _observed_by_frame(self) -> pd.Series | None:
        """Players observed (not estimated) per native frame, where the provider flags estimates; None otherwise."""
        if not hasattr(self, "_obs_cache"):
            tr = self.a.match.tracking
            self._obs_cache = (tr.loc[tr["observed"].astype(bool) & tr["x"].notna()].groupby("frame").size()
                               if "observed" in tr.columns else None)
        return self._obs_cache

    def _near_ball_by_frame(self) -> pd.Series | None:
        """Share of positions within ``NEAR_BALL_M`` of the ball the provider observed, per native frame (schema 2.2)."""
        if not hasattr(self, "_near_cache"):
            m = self.a.match
            tr, ball = m.tracking, m.ball
            if "observed" not in tr.columns or tr.empty or ball.empty:
                self._near_cache = None
            else:
                b = ball.drop_duplicates("frame").set_index("frame")
                t = tr.loc[tr["frame"].isin(b.index) & tr["x"].notna(), ["frame", "x", "y", "observed"]]
                d = np.hypot(t["x"].to_numpy(float) - b["x"].reindex(t["frame"]).to_numpy(float),
                             t["y"].to_numpy(float) - b["y"].reindex(t["frame"]).to_numpy(float))
                near = t.loc[d <= NEAR_BALL_M]
                self._near_cache = near.groupby("frame")["observed"].mean() if len(near) else None
        return self._near_cache

    def _set_piece_frames(self) -> set[int]:
        ev = self.a.events
        sp = ev.loc[ev["Type"].astype(str).str.upper() == "SET PIECE", "Start Frame"]
        return {int(f) for f in sp.dropna()}

    def _row_base(self, idx: int, ev: pd.Series, team: str, ref: str | None, kind: str, event_id: str | None = None) -> dict:
        start = ev.get("Start Frame")
        frame = int(start) if pd.notna(start) else None
        ball = self.a.away.loc[frame, "ball_x"] if frame is not None and frame in self.a.away.index else np.nan
        observed = self._observed_by_frame()
        if frame is not None and observed is not None:
            obs = int(observed.get(frame, 0))  # the provider's own observations only (review AI-03, DE-05)
        elif frame is not None and frame in self.a.home.index:
            obs = int(self.a.home.loc[frame].filter(like="_x").drop(["ball_x"], errors="ignore").notna().sum()
                      + self.a.away.loc[frame].filter(like="_x").drop(["ball_x"], errors="ignore").notna().sum())
        else:
            obs = None
        t = float(ev.get("Start Time [s]", np.nan))
        return {
            "schema_version": SCHEMA_VERSION,
            "engine_version": self.engine_version,
            "event_id": event_id or self._event_ids()[int(idx)],
            "match_id": self.match_id,
            "event_idx": None if idx < 0 else idx,
            "timestamp_s": round(t, 2),
            "minute_display": _minute(t),
            "period": int(ev["Period"]) if pd.notna(ev.get("Period")) else None,
            "actor_player_ref": ref,
            "actor_team": team.lower(),
            "actor_role": self._role(ref)[0],
            "actor_role_source": self._role(ref)[1],
            "action_detail": None, "run_type": None, "run_area": None,
            "mode": "full_tracking",
            "physiological_context": None,
            "ball_detected": bool(np.isfinite(ball)) if frame is not None else None,
            "players_observed": obs,
            "observed_near_ball_share": (None if frame is None or (nb := self._near_ball_by_frame()) is None or frame not in nb.index
                                         else round(float(nb.loc[frame]), 3)),
            "parity_spatial": self.parity.metrics["decision_quality"].state,
            "parity_physiological": self.parity.metrics["load_and_fatigue"].state,
            **self._context_fields(frame, team.lower()),
            **self._context_fields_17(idx, frame, team.lower()),
            "actor_team_key": self.team_keys.get(team.lower()),
            "opponent_team_key": self.team_keys.get({"home": "away", "away": "home"}.get(team.lower(), "")),
            "competition": self.competition,
            "season": self.season,
        }

    # -- semantic events ----------------------------------------------------------

    def semantic_events(self) -> pd.DataFrame:
        a = self.a
        ev = a.events
        pov = a.event_pov()
        pov_by_event = {k: g for k, g in pov.groupby("event_idx", sort=False)}
        pc = a.pass_completion.set_index("event_idx")
        dq2 = a.decision_quality_v2.set_index("event_idx")
        g2 = a.gain_v2.set_index("event_idx")
        set_piece_frames = self._set_piece_frames()
        rows: list[dict] = []

        for idx, e in ev.iterrows():
            etype = str(e.get("Type", "")).upper()
            team = str(e.get("Team", "")).lower()
            if team not in ("home", "away"):
                continue
            actor = self._ref(team, e.get("From"))
            g = pov_by_event.get(idx)
            actor_pov = None
            if g is not None and actor:
                hit = g.loc[(g["team"].str.lower() == team) & (g["jersey"].astype(str) == actor.split("_", 1)[1])]
                actor_pov = hit.iloc[0] if len(hit) else None

            sub = str(e.get("Subtype", "") if pd.notna(e.get("Subtype")) else "").upper()
            turnover = _first(TURNOVER_DETAIL, sub) if etype == "BALL LOST" else None
            if etype == "BALL LOST" and any(k in sub for k in NOT_A_TURNOVER):
                continue
            # Every failed pass gets a row (schema 2.1; before, only those Choice or version 1 scored, review AI-06).
            if etype in ("PASS", "CARRY") or (etype == "BALL LOST" and turnover is None):
                kind = "pass" if etype in ("PASS", "BALL LOST") else "carry"
                row = self._row_base(idx, e, team, actor, kind)
                outcome = "lost" if etype == "BALL LOST" else "completed"
                dq = None
                if actor_pov is not None and etype != "BALL LOST":
                    # Version 1's failed-pass score (``decision_quality_failed``) is never written:
                    # version 1 does not charge a pass for failing, so it read as Full gain on most
                    # lost passes. It stays unreported (PLAIQ_REFERENCES_REPORT.md; review SM-02).
                    dq = actor_pov.get("decision_quality")
                dq = float(dq) if dq is not None and pd.notna(dq) else None
                best_ref, best_pov, chosen_pov = None, None, None
                if g is not None and kind == "pass":
                    mates = g.loc[(g["team"].str.lower() == team) & (g["jersey"].astype(str) != (actor or "_").split("_", 1)[-1])]
                    if len(mates) and mates["pov_start"].notna().any():
                        b = mates.loc[mates["pov_start"].idxmax()]
                        best_ref, best_pov = self._ref(team, b["jersey"]), float(b["pov_start"])
                    target = self._ref(team, e.get("To"))
                    if target is not None:
                        t_rows = g.loc[(g["team"].str.lower() == team) & (g["jersey"].astype(str) == target.split("_", 1)[1])]
                        if len(t_rows) and pd.notna(t_rows.iloc[0]["pov_start"]):
                            chosen_pov = float(t_rows.iloc[0]["pov_start"])

                not_scored, reason = False, None
                start = e.get("Start Frame")
                if actor in self.goalkeepers:
                    not_scored, reason = True, "goalkeeper: the decision model was not designed for goalkeeper distribution"
                elif pd.notna(start) and int(start) in set_piece_frames:
                    not_scored, reason = True, "set piece: the decision model is built for open play"
                elif outcome == "lost":
                    not_scored, reason = True, LOST_PASS_REASON
                if not_scored:
                    dq = None

                pcp, pcs, pcset = None, None, None
                if kind == "pass" and idx in pc.index:
                    r = pc.loc[idx]
                    pcset = r["parameter_set"]
                    if pd.notna(start) and int(start) in set_piece_frames:
                        pcs = "set piece: the model is fitted on open-play passes"
                    elif pd.notna(r["pass_completion_probability"]):
                        pcp, pcs = round(float(r["pass_completion_probability"]), 4), PASS_COMPLETION_LABEL
                    else:
                        pcs = r["reason"]
                d2s, d2b, d2st, d2best, d2v = None, None, None, None, None
                g2s, g2b, g2st, g2v = None, None, None, None
                if kind == "pass" and idx in dq2.index:
                    r2 = dq2.loc[idx]
                    d2v = r2["version"]
                    if actor in self.goalkeepers:
                        d2st = "goalkeeper: the decision model was not designed for goalkeeper distribution"
                    elif pd.notna(start) and int(start) in set_piece_frames:
                        d2st = "set piece: the decision model is built for open play"
                    elif pd.notna(r2["dq_v2"]):
                        d2s, d2st, d2best = round(float(r2["dq_v2"]), 4), DQ_V2_LABEL, self._ref(team, r2["best_option"])
                        d2b = r2["dq_v2_band"]
                    else:
                        d2st = r2["reason"]
                if kind == "pass" and idx in g2.index:
                    rg = g2.loc[idx]
                    g2v = rg["version"]
                    if d2st is not None and d2st != DQ_V2_LABEL:
                        g2st = d2st  # not scored where Choice is not: goalkeeper, set piece, no completion probability
                    elif pd.notna(rg["gain_v2"]):
                        g2s, g2st = round(float(rg["gain_v2"]), 4), GAIN_V2_LABEL
                        g2b = dq_band(g2s)
                    else:
                        g2st = rg["reason"]
                inferred = actor_pov.get("inferred_receiver") if actor_pov is not None else None
                conf = actor_pov.get("inference_confidence") if actor_pov is not None else None
                row.update({
                    "action_type": kind,
                    "action_outcome": outcome,
                    "target_player_ref": self._ref(team, e.get("To")) if kind == "pass" else None,
                    "inferred_intended_ref": self._ref(team, inferred) if inferred not in (None, "") and pd.notna(inferred) else None,
                    "inference_confidence": conf if conf not in (None, "") and pd.notna(conf) else None,
                    "dq_score": None if dq is None else round(dq, 4),
                    "dq_band": dq_band(dq) if dq is not None else ("not_scored" if not_scored else None),
                    "chosen_option_pov": chosen_pov,
                    "best_option_player_ref": best_ref,
                    "best_option_pov": best_pov,
                    "xt_delta": float(e["delta_v"]) if pd.notna(e.get("delta_v")) else None,
                    "movement_value": None,
                    "opposition_pressure": self._pressure(e, team, actor),
                    "pitch_zone": pitch_zone(float(e.get("Start X", np.nan)), float(e.get("Start Y", np.nan)), team),
                    "pass_completion_probability": pcp, "pass_completion_status": pcs, "pass_completion_parameters": pcset,
                    "gain_v2_score": g2s, "gain_v2_band": g2b, "gain_v2_status": g2st, "gain_v2_version": g2v,
                    "dq_v2_score": d2s, "dq_v2_band": d2b, "dq_v2_status": d2st, "dq_v2_best_option_ref": d2best, "dq_v2_version": d2v,
                    "not_scored": not_scored,
                    "not_scored_reason": reason,
                })
                rows.append(row)

            elif etype in DEFENSIVE_TYPES:
                outcome = "won" if etype == "RECOVERY" or "WON" in sub else "lost" if "LOST" in sub else None
                row = self._row_base(idx, e, team, actor, "defensive")
                row.update({
                    "action_type": "defensive_action", "action_outcome": outcome,
                    "target_player_ref": None, "inferred_intended_ref": None, "inference_confidence": None,
                    "dq_score": None, "dq_band": None, "chosen_option_pov": None, "best_option_player_ref": None, "best_option_pov": None,
                    "xt_delta": float(e["delta_v"]) if pd.notna(e.get("delta_v")) else None, "movement_value": None,
                    "opposition_pressure": self._pressure(e, team, actor),
                    "pitch_zone": pitch_zone(float(e.get("Start X", np.nan)), float(e.get("Start Y", np.nan)), team),
                    "not_scored": False, "not_scored_reason": None,
                })
                rows.append(row)

            elif etype == "SHOT" or turnover is not None or etype in ("SET PIECE", "FAULT RECEIVED", "CARD"):
                kind, outcome, detail = {
                    "SHOT": ("shot", _first(SHOT_OUTCOMES, sub), "head" if "HEAD" in sub else None),
                    "BALL LOST": ("turnover", "lost", turnover),
                    "SET PIECE": ("set_piece", None, next((v for k, v in SET_PIECE_KINDS.items() if k in sub), None)),
                    "FAULT RECEIVED": ("foul_won", None, None),
                    "CARD": ("card", None, "red" if "RED" in sub else "yellow" if "YELLOW" in sub else None),
                }[etype]
                row = self._row_base(idx, e, team, actor, kind)
                row.update({
                    "action_type": kind, "action_outcome": outcome, "action_detail": detail,
                    "target_player_ref": None, "inferred_intended_ref": None, "inference_confidence": None,
                    "dq_score": None, "dq_band": None, "chosen_option_pov": None, "best_option_player_ref": None, "best_option_pov": None,
                    "xt_delta": float(e["delta_v"]) if kind != "card" and pd.notna(e.get("delta_v")) else None, "movement_value": None,
                    "opposition_pressure": self._pressure(e, team, actor) if kind in ("shot", "turnover") else None,
                    "pitch_zone": pitch_zone(float(e.get("Start X", np.nan)), float(e.get("Start Y", np.nan)), team),
                    "not_scored": False, "not_scored_reason": None,
                })
                rows.append(row)

        rows += self._run_rows()
        out = pd.DataFrame(rows, columns=list(SEMANTIC_COLUMNS))
        return out.sort_values(["timestamp_s", "event_idx", "action_type", "event_id"], kind="stable",
                               na_position="last").reset_index(drop=True)

    def _run_rows(self) -> list[dict]:
        """
        P6's runs, one row each (schema 2.1, review AI-07): the runner, SkillCorner's run type and FIFA's area at the end
        (in front, in between, in behind), at the run's start. Estimated, like every context field.
        """
        ctx = self._context
        runs = None if ctx is None or "p6" not in getattr(ctx, "outputs", {}) else ctx.outputs["p6"].segments.get("runs")
        if runs is None or not len(runs):
            return []
        f = ctx.frames.sort_values("frame_ts", kind="stable")
        ts, native = f["frame_ts"].to_numpy(float), f["native_frame"].to_numpy()
        clock = self.a.away.get("Time [s]", None)
        out, seen = [], {}
        for r in runs.sort_values(["start_ts", "team", "player_id"], kind="stable").itertuples():
            team = str(r.team).lower()
            jersey = str(r.player_id).split("_", 1)[-1]
            ref = self._ref(team, jersey)
            if ref is None:
                continue
            i = min(max(int(np.searchsorted(ts, float(r.start_ts), side="left")), 0), len(ts) - 1)
            nf = int(native[i])
            t = float(clock.get(nf, np.nan)) if clock is not None else np.nan
            key = (int(r.period), nf, ref)
            seq = seen.get(key, 0)
            seen[key] = seq + 1
            eid = "ev_" + hashlib.sha256("|".join([self.match_id, "run", *map(str, key), str(seq)]).encode()).hexdigest()[:16]
            e = pd.Series({"Start Frame": nf, "Start Time [s]": t, "Period": int(r.period), "Start X": np.nan, "Start Y": np.nan})
            row = self._row_base(-1, e, team, ref, "run", event_id=eid)
            row.update({
                "action_type": "run", "action_outcome": None, "action_detail": None,
                "target_player_ref": None, "inferred_intended_ref": None, "inference_confidence": None,
                "dq_score": None, "dq_band": None, "chosen_option_pov": None, "best_option_player_ref": None, "best_option_pov": None,
                "xt_delta": None, "movement_value": None, "opposition_pressure": None, "pitch_zone": None,
                "not_scored": False, "not_scored_reason": None,
                "run_type": None if pd.isna(r.type) else str(r.type), "run_area": None if pd.isna(r.area_end) else str(r.area_end),
            })
            out.append(row)
        return out

    def movement_values(self) -> pd.DataFrame:
        """
        Plaiq's off-ball movement value for each team-mate the engine rewarded at an event (schema 2.1): the rows that were
        ``off_ball_run`` in the event table, which counted about 8 times P6's runs under a name that said runs (review AI-07).
        ``event_id`` is the source event's id plus the player; ``source_event_id`` resolves in ``semantic_events``.
        """
        ev, mv = self.a.events, self.a.movement_value
        ids = self._event_ids()
        rows = []
        for _, r in mv.loc[mv["run_bonus"] > RUN_BONUS_MIN].iterrows():
            idx = int(r["event_idx"])
            if idx not in ev.index:
                continue
            e = ev.loc[idx]
            team = str(r["team"]).lower()
            ref = self._ref(team, r["jersey"])
            if ref is None or ref == self._ref(str(e.get("Team", "")).lower(), e.get("From")):
                continue
            t = float(e.get("Start Time [s]", np.nan))
            start = e.get("Start Frame")
            rows.append({
                "schema_version": SCHEMA_VERSION, "event_id": f"{ids[idx]}_mv_{ref}", "source_event_id": ids[idx],
                "match_id": self.match_id, "timestamp_s": round(t, 2), "minute_display": _minute(t),
                "period": int(e["Period"]) if pd.notna(e.get("Period")) else None, "player_ref": ref, "team": team,
                "movement_value": float(r["movement_value"]) if pd.notna(r["movement_value"]) else None,
                "xt_delta": float(r["delta_xT_run"]) if pd.notna(r["delta_xT_run"]) else None,
                "opposition_pressure": float(r["pressure"]) if pd.notna(r["pressure"]) else None,
                "state_id": self._context_fields_17(idx, int(start) if pd.notna(start) else None, team)["state_id"],
                "label": "estimated",
            })
        out = pd.DataFrame(rows, columns=list(MOVEMENT_COLUMNS))
        return out.sort_values(["timestamp_s", "event_id"], kind="stable").reset_index(drop=True)

    def _pressure(self, e: pd.Series, team: str, ref: str | None) -> float | None:
        """Pressure on the actor at the start frame, on the engine's 1 to 10 scale."""
        start = e.get("Start Frame")
        if ref is None or pd.isna(start):
            return None
        fa = self.a.frame_analytics
        hit = fa.loc[(fa["Frame"] == int(start)) & (fa["team"].str.lower() == team) & (fa["jersey"].astype(str) == ref.split("_", 1)[1]), "pressure"]
        return float(hit.iloc[0]) if len(hit) and pd.notna(hit.iloc[0]) else None

    # -- trajectories ----------------------------------------------------------------

    def trajectories(self) -> pd.DataFrame:
        """
        Player and ball positions for the replay (never read by the AI layer). Velocity
        and speed are the engine's (``calc_player_velocities``). No physiological
        columns: no source for them exists.
        """
        parts = []
        for team, wide in (("home", self.a.home), ("away", self.a.away)):
            prefix = team.capitalize()
            for j in self.a.match.slots.get(team, ()):
                x = wide[f"{prefix}_{j}_x"].to_numpy(dtype=float)
                ok = np.isfinite(x)
                if not ok.any():
                    continue
                parts.append(pd.DataFrame({
                    "frame": wide.index.to_numpy()[ok], "period": wide["Period"].to_numpy()[ok],
                    "time_s": wide["Time [s]"].to_numpy(dtype=float)[ok], "player_ref": f"{team}_{j}", "team": team,
                    "x": x[ok], "y": wide[f"{prefix}_{j}_y"].to_numpy(dtype=float)[ok],
                    "vx": wide.get(f"{prefix}_{j}_vx", pd.Series(np.nan, index=wide.index)).to_numpy(dtype=float)[ok],
                    "vy": wide.get(f"{prefix}_{j}_vy", pd.Series(np.nan, index=wide.index)).to_numpy(dtype=float)[ok],
                    "speed": wide.get(f"{prefix}_{j}_speed", pd.Series(np.nan, index=wide.index)).to_numpy(dtype=float)[ok],
                }))
        ball = self.a.match.ball
        if len(ball):
            fr = self.a.match.frames.set_index("frame")
            parts.append(pd.DataFrame({
                "frame": ball["frame"].to_numpy(), "period": fr.loc[ball["frame"], "period"].to_numpy(),
                "time_s": fr.loc[ball["frame"], "time_s"].to_numpy(dtype=float), "player_ref": "ball", "team": "ball",
                "x": ball["x"].to_numpy(dtype=float), "y": ball["y"].to_numpy(dtype=float),
                "vx": np.nan, "vy": np.nan, "speed": np.nan,
            }))
        out = pd.concat(parts, ignore_index=True)
        # Whether the provider observed each player position or estimated it (null for the ball, and where the provider
        # does not say), so the replay can show estimated players apart (schema 1.9, review DE-05).
        tr = self.a.match.tracking
        if "observed" in tr.columns:
            flag = tr.assign(player_ref=tr["player_id"].astype(str))[["frame", "player_ref", "observed"]]
            out = out.merge(flag, on=["frame", "player_ref"], how="left")
            out["observed"] = out["observed"].astype("boolean")
        else:
            out["observed"] = pd.array([pd.NA] * len(out), dtype="boolean")
        out["source_attribution"] = self.a.match.provider
        out["match_id"] = self.match_id
        out["schema_version"] = SCHEMA_VERSION
        return out.sort_values(["frame", "team", "player_ref"], kind="stable").reset_index(drop=True)

    # -- output ---------------------------------------------------------------------------

    def semantic_states(self, events: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        The episodes (P8's ``states`` segment) for the AI layer, with the semantic event ids at each end and, for the
        passes in each episode, the median Gain (0 to 1.5), Choice (0 to 1) and context-adjusted Choice, each named and
        never combined. Empty when no ContextResult with P8 was given.
        """
        ctx = self._context
        if ctx is None or "p8" not in getattr(ctx, "outputs", {}):
            return pd.DataFrame(columns=list(STATE_EXTRA_COLUMNS))
        st = ctx.outputs["p8"].segments["states"].copy()
        st["state_id"] = st["state_id"].map(self._state)
        st["match_id"] = self.match_id
        ev = self.semantic_events() if events is None else events

        # The first and last semantic events in each episode, from the table the AI reads, so every id resolves
        # (schema 1.9, review AI-10: P8's logged-event indexes also counted events that are not semantic rows).
        inside = ev.loc[ev["state_id"].notna()].sort_values(["timestamp_s", "event_idx"], kind="stable") if len(ev) else ev
        ends = inside.groupby("state_id")["event_id"].agg(["first", "last"]) if len(inside) else pd.DataFrame(columns=["first", "last"])
        st["first_event_id"] = st["state_id"].map(ends["first"]).astype(object).where(lambda x: x.notna(), None)
        st["last_event_id"] = st["state_id"].map(ends["last"]).astype(object).where(lambda x: x.notna(), None)
        passes = ev.loc[ev["action_type"] == "pass"]
        by_state = {k: g for k, g in passes.groupby("state_id")}
        for col, src in (("gain_median", "dq_score"), ("choice_median", "dq_v2_score"), ("adjusted_choice_median", "adjusted_choice")):
            st[col] = [None if (g := by_state.get(sid)) is None or g[src].dropna().empty else round(float(g[src].dropna().astype(float).median()), 3)
                       for sid in st["state_id"]]
        shots = ev.loc[(ev["action_type"] == "shot") & ev["state_id"].notna()].groupby("state_id").size()
        st["shots"] = st["state_id"].map(shots).fillna(0).astype(int)
        st["ended_in_shot"] = st["shots"] > 0  # a shot inside this episode (schema 2.1), unlike the chain's led_to_shot
        st["team_key"] = st["team"].map(self.team_keys)
        st["opp_team_key"] = st["team"].map({"home": self.team_keys.get("away"), "away": self.team_keys.get("home")})
        st["competition"], st["season"] = self.competition, self.season
        st["schema_version"] = SCHEMA_VERSION
        return st[[c for c in STATE_EXTRA_COLUMNS]]

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        """
        Write ``semantic_events.parquet``, ``trajectories.parquet``, ``event_provenance.parquet`` (not AI-facing) and, with
        P8's context, ``semantic_states.parquet``.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {"semantic_events": out / "semantic_events.parquet", "trajectories": out / "trajectories.parquet",
                 "event_provenance": out / "event_provenance.parquet"}
        events = self.semantic_events()
        events.to_parquet(paths["semantic_events"], index=False)
        self.trajectories().to_parquet(paths["trajectories"], index=False)
        self.provenance().to_parquet(paths["event_provenance"], index=False)
        paths["movement_values"] = out / "movement_values.parquet"
        self.movement_values().to_parquet(paths["movement_values"], index=False)
        states = self.semantic_states(events)
        if len(states):
            paths["semantic_states"] = out / "semantic_states.parquet"
            states.to_parquet(paths["semantic_states"], index=False)
        return paths


def nested_record(row: dict) -> dict:
    """One flat semantic-event row in the nested JSON shape of ``ai-layer.md``."""
    return {
        "schema_version": row["schema_version"],
        "engine_version": row["engine_version"],
        "event_id": row["event_id"],
        "match_id": row["match_id"],
        "timestamp_s": row["timestamp_s"],
        "minute_display": row["minute_display"],
        "period": row["period"],
        "actor": {"player_ref": row["actor_player_ref"], "role": row["actor_role"], "role_source": row.get("actor_role_source")},
        "team": {"team": row.get("actor_team"), "team_key": row.get("actor_team_key"),  # schema 2.0 (review SE-15)
                 "opponent_team_key": row.get("opponent_team_key"), "competition": row.get("competition"), "season": row.get("season")},
        "action": {
            "type": row["action_type"], "outcome": row["action_outcome"], "detail": row.get("action_detail"),
            "run_type": row.get("run_type"), "run_area": row.get("run_area"), "target_player_ref": row["target_player_ref"],
            "inferred_intended_ref": row["inferred_intended_ref"], "inference_confidence": row["inference_confidence"],
        },
        "spatial_intelligence": {
            "dq_score": row["dq_score"], "dq_band": row["dq_band"], "best_option_player_ref": row["best_option_player_ref"],
            "best_option_pov": row["best_option_pov"], "chosen_option_pov": row["chosen_option_pov"], "xt_delta": row["xt_delta"],
            "movement_value": row["movement_value"], "opposition_pressure": row["opposition_pressure"],
            "pitch_zone": row["pitch_zone"], "mode": row["mode"],
            "not_scored": row["not_scored"], "not_scored_reason": row["not_scored_reason"],
            "pass_completion": {"probability": row.get("pass_completion_probability"), "status": row.get("pass_completion_status"),
                                "parameters": row.get("pass_completion_parameters")},
            "gain_v2": {"score": row.get("gain_v2_score"), "band": row.get("gain_v2_band"), "status": row.get("gain_v2_status"),
                        "version": row.get("gain_v2_version")},
            "decision_quality_v2": {"score": row.get("dq_v2_score"), "band": row.get("dq_v2_band"), "status": row.get("dq_v2_status"),
                                    "best_option_player_ref": row.get("dq_v2_best_option_ref"), "version": row.get("dq_v2_version")},
        },
        "physiological_context": row["physiological_context"],
        "match_context": {
            "possession_chain_id": row.get("possession_chain_id"), "possession_source": row.get("possession_source"),
            "score_diff": row.get("score_diff"), "red_card_diff": row.get("red_card_diff"),
        },
        "context": {k: row.get(k) for k in CONTEXT_COLUMNS},  # schema 1.7: Stage B to D context, all estimated
        "quality_gate": {
            "ball_detected": row["ball_detected"], "players_observed": row["players_observed"],
            "observed_near_ball_share": row.get("observed_near_ball_share"),
            "parity": {"spatial_intelligence": row["parity_spatial"], "physiological_context": row["parity_physiological"]},
        },
    }
