"""
semantic_builder.py — compile a match into semantic football events.

One row per pass, carry, off-ball run and defensive action, written to
``semantic_events.parquet``. This is the only thing the AI layer reads for evidence
(``ponta-blueprint/ai-layer.md``, "The semantic football event"). It also writes
``trajectories.parquet`` for the replay, which the AI layer never reads.

Rules this module enforces:

* **Pure and deterministic.** Same match, same output: ids are derived from the
  event index, rows are explicitly sorted, nothing depends on engine row order.
* **No names and no coordinate arrays** in semantic events. Players are pseudonymous
  references (``home_10``); location is a categorical ``pitch_zone``.
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

from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from matrix_engine.ingestion.metrica_io import find_goalkeeper
from matrix_engine.parity import AnalyticsParityReport

from plaiq_engine.analytics.event_stitch import clean_jersey
from plaiq_engine.match_analysis import MatchAnalysis

SCHEMA_VERSION = "1.0"

#: Decision-quality bands (ANALYTICS_SPECIFICATION.md 7.1): unified across event types.
BAND_CUTS: tuple[tuple[float, str], ...] = ((0.95, "optimal"), (0.80, "good"), (0.55, "suboptimal"))

#: An off-ball run is a movement the engine rewarded: run_bonus above this.
RUN_BONUS_MIN = 0.0
#: Defensive actions taken from the event feed.
DEFENSIVE_TYPES = {"RECOVERY", "CHALLENGE"}
#: Pitch length thirds (team frame, metres from the centre line).
THIRD_EDGE_M = 17.5
#: Channels across the pitch (team frame, metres from the centre line, positive = left).
CHANNEL_EDGES_M = (20.0, 9.0)

SEMANTIC_COLUMNS: tuple[str, ...] = (
    "schema_version", "engine_version", "event_id", "match_id", "event_idx", "timestamp_s", "minute_display", "period",
    "actor_player_ref", "actor_team", "actor_role",
    "action_type", "action_outcome", "target_player_ref", "inferred_intended_ref", "inference_confidence",
    "dq_score", "dq_band", "chosen_option_pov", "best_option_player_ref", "best_option_pov",
    "xt_delta", "movement_value", "opposition_pressure", "pitch_zone", "mode",
    "not_scored", "not_scored_reason",
    "physiological_context",
    "ball_detected", "players_observed", "parity_spatial", "parity_physiological",
)


def dq_band(score: float | None) -> str | None:
    if score is None or not np.isfinite(score):
        return None
    for cut, band in BAND_CUTS:
        if score >= cut:
            return band
    return "poor"


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
    """

    def __init__(self, analysis: MatchAnalysis, parity: AnalyticsParityReport, *, roles: dict[str, str] | None = None) -> None:
        self.a = analysis
        self.parity = parity
        self.roles = roles or {}
        self.match_id = f"{analysis.match.provider}_{analysis.match.match_id}"
        self.engine_version = _engine_version()
        self.goalkeepers = self._goalkeepers()

    # -- helpers ---------------------------------------------------------------

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
        j = clean_jersey(jersey)
        return f"{team.lower()}_{j}" if j else None

    def _set_piece_frames(self) -> set[int]:
        ev = self.a.events
        sp = ev.loc[ev["Type"].astype(str).str.upper() == "SET PIECE", "Start Frame"]
        return {int(f) for f in sp.dropna()}

    def _row_base(self, idx: int, ev: pd.Series, team: str, ref: str | None, kind: str) -> dict:
        start = ev.get("Start Frame")
        frame = int(start) if pd.notna(start) else None
        ball = self.a.away.loc[frame, "ball_x"] if frame is not None and frame in self.a.away.index else np.nan
        if frame is not None and frame in self.a.home.index:
            obs = int(self.a.home.loc[frame].filter(like="_x").drop(["ball_x"], errors="ignore").notna().sum()
                      + self.a.away.loc[frame].filter(like="_x").drop(["ball_x"], errors="ignore").notna().sum())
        else:
            obs = None
        t = float(ev.get("Start Time [s]", np.nan))
        return {
            "schema_version": SCHEMA_VERSION,
            "engine_version": self.engine_version,
            "event_id": f"evt_{self.match_id}_{idx:05d}" + ("" if kind != "run" else f"_run_{ref}"),
            "match_id": self.match_id,
            "event_idx": idx,
            "timestamp_s": round(t, 2),
            "minute_display": _minute(t),
            "period": int(ev["Period"]) if pd.notna(ev.get("Period")) else None,
            "actor_player_ref": ref,
            "actor_team": team.lower(),
            "actor_role": self.roles.get(ref) if ref else None,
            "mode": "full_tracking",
            "physiological_context": None,
            "ball_detected": bool(np.isfinite(ball)) if frame is not None else None,
            "players_observed": obs,
            "parity_spatial": self.parity.metrics["decision_quality"].state,
            "parity_physiological": self.parity.metrics["load_and_fatigue"].state,
        }

    # -- semantic events ----------------------------------------------------------

    def semantic_events(self) -> pd.DataFrame:
        a = self.a
        ev = a.events
        pov = a.event_pov()
        pov_by_event = {k: g for k, g in pov.groupby("event_idx", sort=False)}
        mv = a.movement_value
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

            if etype in ("PASS", "CARRY") or (etype == "BALL LOST" and actor_pov is not None and pd.notna(actor_pov.get("decision_quality_failed"))):
                kind = "pass" if etype in ("PASS", "BALL LOST") else "carry"
                row = self._row_base(idx, e, team, actor, kind)
                outcome = "lost" if etype == "BALL LOST" else "completed"
                dq = None
                if actor_pov is not None:
                    dq = actor_pov.get("decision_quality")
                    if (dq is None or pd.isna(dq)) and etype == "BALL LOST":
                        dq = actor_pov.get("decision_quality_failed")
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
                if not_scored:
                    dq = None

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
                    "not_scored": not_scored,
                    "not_scored_reason": reason,
                })
                rows.append(row)

            elif etype in DEFENSIVE_TYPES:
                sub = str(e.get("Subtype", "")).upper()
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

        # Off-ball runs: movements the engine rewarded, by players other than the actor.
        runs = mv.loc[mv["run_bonus"] > RUN_BONUS_MIN]
        for _, r in runs.iterrows():
            idx = int(r["event_idx"])
            if idx not in ev.index:
                continue
            e = ev.loc[idx]
            team = str(r["team"]).lower()
            ref = self._ref(team, r["jersey"])
            if ref is None or ref == self._ref(str(e.get("Team", "")), e.get("From")):
                continue
            row = self._row_base(idx, e, team, ref, "run")
            row.update({
                "action_type": "off_ball_run", "action_outcome": None,
                "target_player_ref": None, "inferred_intended_ref": None, "inference_confidence": None,
                "dq_score": None, "dq_band": None, "chosen_option_pov": None, "best_option_player_ref": None, "best_option_pov": None,
                "xt_delta": float(r["delta_xT_run"]) if pd.notna(r["delta_xT_run"]) else None,
                "movement_value": float(r["movement_value"]) if pd.notna(r["movement_value"]) else None,
                "opposition_pressure": float(r["pressure"]) if pd.notna(r["pressure"]) else None,
                "pitch_zone": None,
                "not_scored": False, "not_scored_reason": None,
            })
            rows.append(row)

        out = pd.DataFrame(rows, columns=list(SEMANTIC_COLUMNS))
        return out.sort_values(["event_idx", "action_type", "event_id"], kind="stable").reset_index(drop=True)

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
        out["source_attribution"] = self.a.match.provider
        out["match_id"] = self.match_id
        out["schema_version"] = SCHEMA_VERSION
        return out.sort_values(["frame", "team", "player_ref"], kind="stable").reset_index(drop=True)

    # -- output ---------------------------------------------------------------------------

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        """Write ``semantic_events.parquet`` and ``trajectories.parquet`` to ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {"semantic_events": out / "semantic_events.parquet", "trajectories": out / "trajectories.parquet"}
        self.semantic_events().to_parquet(paths["semantic_events"], index=False)
        self.trajectories().to_parquet(paths["trajectories"], index=False)
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
        "actor": {"player_ref": row["actor_player_ref"], "role": row["actor_role"]},
        "action": {
            "type": row["action_type"], "outcome": row["action_outcome"], "target_player_ref": row["target_player_ref"],
            "inferred_intended_ref": row["inferred_intended_ref"], "inference_confidence": row["inference_confidence"],
        },
        "spatial_intelligence": {
            "dq_score": row["dq_score"], "dq_band": row["dq_band"], "best_option_player_ref": row["best_option_player_ref"],
            "best_option_pov": row["best_option_pov"], "chosen_option_pov": row["chosen_option_pov"], "xt_delta": row["xt_delta"],
            "movement_value": row["movement_value"], "opposition_pressure": row["opposition_pressure"],
            "pitch_zone": row["pitch_zone"], "mode": row["mode"],
            "not_scored": row["not_scored"], "not_scored_reason": row["not_scored_reason"],
        },
        "physiological_context": row["physiological_context"],
        "quality_gate": {
            "ball_detected": row["ball_detected"], "players_observed": row["players_observed"],
            "parity": {"spatial_intelligence": row["parity_spatial"], "physiological_context": row["parity_physiological"]},
        },
    }
