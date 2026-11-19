#!/usr/bin/env python3
"""
skillcorner_io.py — SkillCorner broadcast tracking adapter (custom, no Kloppy).

Loads SkillCorner open data format:
  {id}_match.json                      — lineup, pitch size, match metadata
  {id}_tracking_extrapolated.jsonl     — 10 fps tracking (may be in Git LFS)
  {id}_dynamic_events.csv             — derived dynamic events

Output conventions match second_spectrum_io / epts_io / metrica_io:
  home_df  index=Frame, cols: Period, Time [s], Home_{jersey}_x/y
  away_df  same for Away_{jersey}_* plus ball_x, ball_y
  events_df  PLAIQ event schema (EVENT_COLUMNS from second_spectrum_io), plus the
             optional HIGH_PASS column from SkillCorner's high_pass flag
  meta dict: team_a_id, team_b_id, team_a_name, team_b_name,
             team_a_players, team_b_players, periods, frame_rate, pitch

Coordinate system:
  SkillCorner tracking is already in metres, centred at (0,0), x = long axis,
  y = short axis — identical to PLAIQ's internal convention.

Playing direction normalisation:
  PLAIQ requires HOME attacks +x in BOTH halves (STATIC_HOME_AWAY invariant).
  The `home_team_side` field in match.json (["right_to_left"|"left_to_right", ...])
  tells us which direction home attacks each period. We negate (x, y) for every
  frame and event in periods where home attacks -x, guaranteeing home always
  attacks +x after ingest.

10 fps vs 25 fps:
  SkillCorner open data is broadcast tracking at 10 fps. Analytics formulas are
  provider-agnostic (they consume metres, not frame counts), but the velocity
  smoothing window is adjusted at the preprocess / app layers. See preprocess.py
  and app.py for details.

Player identity:
  Tracking player_id = player.id in match.json (integer). trackable_object is
  the internal vision ID and is NOT used for roster joins. A fallback lookup by
  trackable_object is provided in case a given dataset uses that column instead.

Dynamic events:
  Only player_possession events are mapped to PLAIQ events. passing_option and
  other analytics-only event types are skipped. x/y coordinates in the CSV are
  in real metres at the match's actual pitch dimensions and do NOT require
  scaling when the tracking data uses the same pitch dimensions.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from matrix_engine.ingestion.second_spectrum_io import (
    EVENT_COLUMNS,
    HIGH_PASS,
    _period_time_seconds,
)
from matrix_engine.pitch_config import PitchConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SC_FPS: float = 10.0
SC_NOMINAL_PITCH_LENGTH: float = 105.0
SC_NOMINAL_PITCH_WIDTH: float = 68.0

# SkillCorner dynamic_events: event_type values to process
_SC_ON_BALL_TYPE = "player_possession"

# Mapping from dynamic_events end_type / carry logic → PLAIQ Type
_END_TYPE_MAP = {
    "pass":          None,   # resolved via pass_outcome
    "shot":          "SHOT",
    "tackle":        "BALL LOST",
    "interception":  "BALL LOST",
    "out_of_play":   "BALL OUT",
    "foul":          "FAULT RECEIVED",
    "header":        None,   # resolved via pass_outcome if present
}

# Subtypes for set-piece start types
_START_TYPE_SUBTYPE = {
    "corner_kick": "CORNER KICK",
    "throw_in":    "THROW IN",
    "free_kick":   "FREE KICK",
    "goal_kick":   "GOAL KICK",
    "kick_off":    "KICK OFF",
    "penalty":     "PENALTY",
}


# ---------------------------------------------------------------------------
# PitchConfig factory for SkillCorner metadata
# ---------------------------------------------------------------------------

def pitch_config_from_sc_meta(meta: dict) -> PitchConfig:
    """Build a PitchConfig from a parsed SkillCorner match meta dict."""
    return PitchConfig(
        length_m=float(meta.get("pitch_length", SC_NOMINAL_PITCH_LENGTH)),
        width_m=float(meta.get("pitch_width", SC_NOMINAL_PITCH_WIDTH)),
        fps=SC_FPS,
    )


# ---------------------------------------------------------------------------
# Phase 1a — Match metadata
# ---------------------------------------------------------------------------

def load_skillcorner_meta(match_json_path: str) -> dict:
    """
    Parse a SkillCorner {id}_match.json file.

    Returns
    -------
    dict with keys:
      match_id, pitch_length, pitch_width, fps,
      home_team_id, home_team_name,
      away_team_id, away_team_name,
      home_players, away_players,   # list of player dicts (see _parse_players)
      periods,                       # {period_int: (start_frame, end_frame)}
      home_team_side,                # ["right_to_left"|"left_to_right", ...]
      h1_home_sign,                  # +1 if home attacks +x in H1 else -1
    """
    with open(match_json_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    home_team_id = raw["home_team"]["id"]
    away_team_id = raw["away_team"]["id"]

    all_players = raw.get("players", [])
    home_players = _parse_players(
        [p for p in all_players if p.get("team_id") == home_team_id]
    )
    away_players = _parse_players(
        [p for p in all_players if p.get("team_id") == away_team_id]
    )

    periods: dict[int, tuple[int, int]] = {}
    for mp in raw.get("match_periods", []):
        pid = int(mp["period"])
        periods[pid] = (int(mp["start_frame"]), int(mp["end_frame"]))

    # home_team_side: ["right_to_left"|"left_to_right", ...]
    home_team_side = raw.get("home_team_side", [])
    h1_side = home_team_side[0] if home_team_side else "left_to_right"
    # "right_to_left" means home attacks -x in H1 → need to flip
    h1_home_sign = -1 if h1_side == "right_to_left" else 1

    return {
        "match_id":       raw["id"],
        "pitch_length":   float(raw.get("pitch_length", SC_NOMINAL_PITCH_LENGTH)),
        "pitch_width":    float(raw.get("pitch_width", SC_NOMINAL_PITCH_WIDTH)),
        "fps":            SC_FPS,
        "home_team_id":   home_team_id,
        "home_team_name": raw["home_team"].get("name", raw["home_team"].get("short_name", "Home")),
        "away_team_id":   away_team_id,
        "away_team_name": raw["away_team"].get("name", raw["away_team"].get("short_name", "Away")),
        "home_players":   home_players,
        "away_players":   away_players,
        "players_raw":    all_players,   # raw player list for id→jersey translation
        "periods":        periods,
        "home_team_side": home_team_side,
        "h1_home_sign":   h1_home_sign,
        # downstream compatibility
        "team_a_id":      home_team_id,
        "team_b_id":      away_team_id,
        "team_a_name":    raw["home_team"].get("name", raw["home_team"].get("short_name", "Home")),
        "team_b_name":    raw["away_team"].get("name", raw["away_team"].get("short_name", "Away")),
        "team_a_players": home_players,
        "team_b_players": away_players,
        "frame_rate":     SC_FPS,
    }


def _parse_players(players: list[dict]) -> list[dict]:
    """
    Normalise a list of SkillCorner player dicts into PLAIQ roster format.

    The `playing_time` block gives per-player on-pitch frame ranges; players
    with null playing_time (squad members who never played) are still included
    so that the tracking adapter can map their jersey column if they appear.
    """
    out: list[dict] = []
    for p in players:
        role = p.get("player_role") or {}
        position = role.get("acronym") or role.get("name") or "?"

        # Normalise position group strings to PLAIQ position codes
        position = _normalise_position(position)

        jersey = p.get("number")
        name = (
            p.get("short_name")
            or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
            or f"#{jersey}"
        )

        pt_total = (p.get("playing_time") or {}).get("total") or {}
        start_frame = pt_total.get("start_frame")
        end_frame   = pt_total.get("end_frame")

        out.append({
            "player_id":       p["id"],            # integer SC player ID
            "trackable_object": p.get("trackable_object"),
            "shirt":           str(jersey) if jersey is not None else "?",
            "jersey_no":       jersey,
            "name":            name,
            "short_name":      name.split()[-1] if name else str(jersey),
            "position_type":   position,
            "start_frame":     start_frame,
            "end_frame":       end_frame,
        })
    return out


def _normalise_position(acronym: str) -> str:
    """Map SkillCorner position acronyms to PLAIQ position codes."""
    _MAP = {
        "GK": "GK", "Goalkeeper": "GK",
        "CB": "CB", "LCB": "LCB", "RCB": "RCB",
        "LB": "LB", "RB": "RB",
        "DM": "CDM", "LDM": "CDM", "RDM": "CDM",
        "CM": "CM", "AM": "CAM",
        "LW": "LW", "RW": "RW",
        "CF": "ST", "ST": "ST",
        "SUB": "?",
    }
    return _MAP.get(acronym, acronym or "?")


# ---------------------------------------------------------------------------
# Phase 1b — Tracking data (JSONL stream)
# ---------------------------------------------------------------------------

def load_skillcorner_tracking(tracking_jsonl_path: str) -> list[dict]:
    """
    Stream-parse a SkillCorner tracking_extrapolated.jsonl file.

    Each line is a JSON object with keys:
      frame, timestamp, period, ball_data, player_data, possession, ...

    Returns
    -------
    list of frame dicts (preserves original order).

    Note: the file may be in Git LFS. Use the media URL or `git lfs pull`
    to obtain the actual binary. If the file is a LFS pointer stub (< 512 bytes)
    this function will raise an informative error.
    """
    records: list[dict] = []
    with open(tracking_jsonl_path, "r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()
        if first_line.startswith("version https://git-lfs"):
            raise RuntimeError(
                f"{tracking_jsonl_path!r} is a Git LFS pointer. "
                "Run `git lfs pull` or download via the media URL: "
                "https://media.githubusercontent.com/media/SkillCorner/opendata/master/"
                f"data/matches/<id>/<id>_tracking_extrapolated.jsonl"
            )
        if first_line:
            records.append(json.loads(first_line))
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Phase 1c — Playing direction detection
# ---------------------------------------------------------------------------

def detect_playing_direction_from_meta(meta: dict) -> int:
    """
    Return h1_home_sign from pre-parsed metadata.

    Uses the `home_team_side` field in match.json (most reliable source).
    Returns +1 if home attacks +x in H1, -1 if home attacks -x in H1.
    """
    return meta["h1_home_sign"]


def detect_playing_direction_from_tracking(
    records: list[dict],
    home_player_ids: set,
    gk_player_ids: set,
    *,
    n_frames: int = 300,
) -> int:
    """
    Fallback: infer home attacking direction in H1 from goalkeeper position.

    If a home GK is found in the first ``n_frames`` of period 1, their median
    x-position determines direction: GK in +x half → home attacks -x (h1=-1),
    GK in -x half → home attacks +x (h1=+1).

    Returns +1 or -1.
    """
    p1_records = [r for r in records if r.get("period") == 1][:n_frames]
    gk_xs: list[float] = []
    for rec in p1_records:
        for pd_item in rec.get("player_data") or []:
            pid = pd_item.get("player_id")
            if pid in home_player_ids and pid in gk_player_ids:
                x = pd_item.get("x")
                if x is not None and not np.isnan(float(x)):
                    gk_xs.append(float(x))
    if not gk_xs:
        return 1  # cannot determine, default: home attacks +x
    median_x = float(np.median(gk_xs))
    # GK stays near own goal; if median x > 0, home goal is on the right
    # → home attacks -x (right-to-left).
    return -1 if median_x > 0 else 1


# ---------------------------------------------------------------------------
# Phase 1d — Convert JSONL records to PLAIQ wide-format DataFrames
# ---------------------------------------------------------------------------

def tracking_to_wide(
    records: list[dict],
    home_players: list[dict],
    away_players: list[dict],
    *,
    h1_home_sign: int = 1,
    normalize_direction: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert SkillCorner JSONL records to PLAIQ wide-format tracking DataFrames.

    Parameters
    ----------
    records          : output of load_skillcorner_tracking()
    home_players     : parsed home player list from load_skillcorner_meta()
    away_players     : parsed away player list from load_skillcorner_meta()
    h1_home_sign     : +1 if home attacks +x in H1; -1 if home attacks -x
    normalize_direction : if True, apply sign flips so home always attacks +x

    Returns
    -------
    (home_df, away_df) — wide DataFrames indexed by Frame with Period, Time [s],
    Home_{jersey}_x/y columns and ball_x/ball_y on away_df (SS convention).
    """
    # Build lookup: player_id (int) → player dict
    home_by_id: dict[int, dict] = {}
    home_by_to: dict[int, dict] = {}
    for p in home_players:
        if p["player_id"] is not None:
            home_by_id[p["player_id"]] = p
        if p.get("trackable_object") is not None:
            home_by_to[p["trackable_object"]] = p

    away_by_id: dict[int, dict] = {}
    away_by_to: dict[int, dict] = {}
    for p in away_players:
        if p["player_id"] is not None:
            away_by_id[p["player_id"]] = p
        if p.get("trackable_object") is not None:
            away_by_to[p["trackable_object"]] = p

    home_jerseys = [p["shirt"] for p in home_players if p["shirt"] != "?"]
    away_jerseys = [p["shirt"] for p in away_players if p["shirt"] != "?"]

    home_rows: list[dict] = []
    away_rows: list[dict] = []
    frame_ids: list[int] = []

    for rec in records:
        frame_id = int(rec["frame"])
        _period_raw = rec.get("period")
        period = int(_period_raw) if _period_raw is not None else 1
        ts = rec.get("timestamp", 0.0)
        if ts is None:
            time_s = 0.0
        else:
            try:
                time_s = float(ts)
            except (ValueError, TypeError):
                # Handle "HH:MM:SS.ss" wall-clock strings from SkillCorner JSONL
                ts_str = str(ts)
                parts = ts_str.split(":")
                try:
                    if len(parts) == 3:
                        time_s = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                    elif len(parts) == 2:
                        time_s = int(parts[0]) * 60 + float(parts[1])
                    else:
                        time_s = float(parts[0])
                except Exception:
                    time_s = 0.0

        # Sign for this period: period 1 uses h1_home_sign; period 2 uses opposite
        if normalize_direction:
            if period == 1:
                sign = h1_home_sign
            else:
                # In real football, teams swap ends at half time.
                # If H1 home attacked -x (sign=-1), H2 home attacks +x (sign=+1).
                sign = -h1_home_sign
        else:
            sign = 1

        home_row: dict = {"Period": period, "Time [s]": round(time_s, 4)}
        away_row: dict = {"Period": period, "Time [s]": round(time_s, 4)}

        for jersey in home_jerseys:
            home_row[f"Home_{jersey}_x"] = np.nan
            home_row[f"Home_{jersey}_y"] = np.nan
        for jersey in away_jerseys:
            away_row[f"Away_{jersey}_x"] = np.nan
            away_row[f"Away_{jersey}_y"] = np.nan

        ball_x, ball_y = np.nan, np.nan
        bd = rec.get("ball_data")
        if bd and bd.get("x") is not None:
            ball_x = float(bd["x"]) * sign
            ball_y = float(bd.get("y", 0.0)) * sign

        for pd_item in rec.get("player_data") or []:
            pid = pd_item.get("player_id")
            x   = pd_item.get("x")
            y   = pd_item.get("y")
            if x is None or y is None:
                continue
            fx = float(x) * sign
            fy = float(y) * sign

            # Try home lookup
            player = home_by_id.get(pid) or home_by_to.get(pid)
            if player is not None:
                jersey = player["shirt"]
                if jersey != "?" and f"Home_{jersey}_x" in home_row:
                    home_row[f"Home_{jersey}_x"] = fx
                    home_row[f"Home_{jersey}_y"] = fy
                continue

            # Try away lookup
            player = away_by_id.get(pid) or away_by_to.get(pid)
            if player is not None:
                jersey = player["shirt"]
                if jersey != "?" and f"Away_{jersey}_x" in away_row:
                    away_row[f"Away_{jersey}_x"] = fx
                    away_row[f"Away_{jersey}_y"] = fy

        home_row["ball_x"] = ball_x
        home_row["ball_y"] = ball_y
        away_row["ball_x"] = ball_x
        away_row["ball_y"] = ball_y

        home_rows.append(home_row)
        away_rows.append(away_row)
        frame_ids.append(frame_id)

    home_df = pd.DataFrame(home_rows, index=frame_ids)
    away_df = pd.DataFrame(away_rows, index=frame_ids)
    home_df.index.name = "Frame"
    away_df.index.name = "Frame"

    return home_df, away_df


def _build_periods_from_df(home_df: pd.DataFrame) -> dict[int, tuple[int, int]]:
    """Derive {period: (min_frame, max_frame)} from a wide tracking DataFrame."""
    result: dict[int, tuple[int, int]] = {}
    for period in home_df["Period"].unique():
        sub = home_df[home_df["Period"] == int(period)]
        result[int(period)] = (int(sub.index.min()), int(sub.index.max()))
    return result


# ---------------------------------------------------------------------------
# Phase 1e — Dynamic events → PLAIQ events CSV
# ---------------------------------------------------------------------------

def load_skillcorner_events(
    dynamic_events_path: str,
    *,
    home_team_id: int,
    away_team_id: int,
    periods: dict[int, tuple[int, int]],
    fps: float = SC_FPS,
    pitch_length: float = SC_NOMINAL_PITCH_LENGTH,
    pitch_width: float = SC_NOMINAL_PITCH_WIDTH,
    h1_home_sign: int = 1,
    normalize_direction: bool = True,
    player_id_to_jersey: dict | None = None,
) -> pd.DataFrame:
    """
    Map SkillCorner dynamic_events.csv to a PLAIQ events DataFrame.

    Only ``player_possession`` events are converted. ``passing_option`` and
    other analytics-derived event types are skipped.

    Parameters
    ----------
    dynamic_events_path : path to {id}_dynamic_events.csv
    home_team_id        : from load_skillcorner_meta()
    away_team_id        : from load_skillcorner_meta()
    periods             : {period: (min_frame, max_frame)} from tracking
    fps                 : tracking fps (default 10)
    pitch_length        : actual match pitch length in metres
    pitch_width         : actual match pitch width in metres
    h1_home_sign        : +1 or -1 — same sign used for tracking normalisation
    normalize_direction : apply same sign flips as tracking normalisation

    Returns
    -------
    PLAIQ events DataFrame with EVENT_COLUMNS.
    """
    if not os.path.exists(dynamic_events_path):
        return pd.DataFrame(columns=EVENT_COLUMNS)

    df = pd.read_csv(dynamic_events_path, low_memory=False)

    if df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    # Normalise column names to lowercase for robustness
    df.columns = [c.strip().lower() for c in df.columns]

    # Only on-ball events
    event_type_col = _find_col(df, ["event_type"])
    if event_type_col:
        df = df[df[event_type_col].str.lower() == _SC_ON_BALL_TYPE].copy()
    if df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    # Coordinate scaling: SC events are in real venue metres (same as tracking).
    # Scale is 1.0 unless the event CSV was generated against a nominal pitch.
    # In practice SC open data uses real metres so scale ≈ 1.0; kept for safety.
    scale_x = pitch_length / SC_NOMINAL_PITCH_LENGTH
    scale_y = pitch_width  / SC_NOMINAL_PITCH_WIDTH

    rows: list[dict] = []

    frame_start_col = _find_col(df, ["frame_start"])
    frame_end_col   = _find_col(df, ["frame_end"])
    period_col      = _find_col(df, ["period"])
    player_col      = _find_col(df, ["player_id"])
    team_col        = _find_col(df, ["team_id"])
    x_start_col     = _find_col(df, ["x_start"])
    y_start_col     = _find_col(df, ["y_start"])
    x_end_col       = _find_col(df, ["x_end"])
    y_end_col       = _find_col(df, ["y_end"])
    end_type_col    = _find_col(df, ["end_type"])
    start_type_col  = _find_col(df, ["start_type"])
    carry_col       = _find_col(df, ["carry"])
    pass_outcome_col = _find_col(df, ["pass_outcome"])
    targeted_id_col      = _find_col(df, ["player_targeted_id"])
    target_x_col         = _find_col(df, ["player_targeted_x_reception"])
    target_y_col         = _find_col(df, ["player_targeted_y_reception"])
    target_x_pass_col    = _find_col(df, ["player_targeted_x_pass"])
    target_y_pass_col    = _find_col(df, ["player_targeted_y_pass"])
    high_pass_col        = _find_col(df, ["high_pass"])

    def _to_jersey(pid_int):
        if pid_int is None:
            return np.nan
        if player_id_to_jersey:
            jersey = player_id_to_jersey.get(int(pid_int))
            if jersey is not None:
                return str(jersey)
        return str(pid_int)

    def _apply_team_rotation(x, y, team):
        """Negate both axes for Away events (team-relative → PLAIQ-normalised)."""
        if team == "Away":
            return -x, -y
        return x, y

    def _append_event(evt_type, evt_subtype, period_id, from_id, to_id, team,
                      sf, sf_t, sx, sy, ef, ef_t, ex, ey, high=pd.NA):
        rows.append({
            HIGH_PASS:        high,
            "Type":           evt_type,
            "Subtype":        evt_subtype if evt_subtype is not None else np.nan,
            "Period":         period_id,
            "From":           from_id,
            "To":             to_id,
            "Team":           team,
            "Start Frame":    sf,
            "Start Time [s]": sf_t,
            "Start X":        sx,
            "Start Y":        sy,
            "End Frame":      ef,
            "End Time [s]":   ef_t,
            "End X":          ex,
            "End Y":          ey,
        })

    for _, row in df.iterrows():
        start_frame = _int_or_none(row.get(frame_start_col))
        end_frame   = _int_or_none(row.get(frame_end_col))
        period_id   = _int_or_none(row.get(period_col))

        if start_frame is None or period_id is None:
            continue
        if end_frame is None:
            end_frame = start_frame

        end_type     = str(row.get(end_type_col) or "").lower()
        start_type   = str(row.get(start_type_col) or "").lower()
        is_carry     = _bool_val(row.get(carry_col))
        pass_outcome = str(row.get(pass_outcome_col) or "").lower()
        player_id    = _int_or_none(row.get(player_col))
        team_id_raw  = _int_or_none(row.get(team_col))

        # Team side
        if team_id_raw == home_team_id:
            team = "Home"
        elif team_id_raw == away_team_id:
            team = "Away"
        else:
            team = np.nan

        start_t = _period_time_seconds(start_frame, period_id, periods, fps)
        end_t   = _period_time_seconds(end_frame,   period_id, periods, fps)

        from_id   = _to_jersey(player_id)
        to_id_raw = _int_or_none(row.get(targeted_id_col))
        to_id     = _to_jersey(to_id_raw)

        # Raw scaled coordinates (before team rotation)
        sx_raw = _float_or_nan(row.get(x_start_col)) * scale_x
        sy_raw = _float_or_nan(row.get(y_start_col)) * scale_y
        ex_raw = _float_or_nan(row.get(x_end_col)) * scale_x
        ey_raw = _float_or_nan(row.get(y_end_col)) * scale_y

        # Targeted reception / pass-aim coords (for pass end point)
        trx_raw = (
            _float_or_nan(row.get(target_x_col)) * scale_x
            if target_x_col else np.nan
        )
        try_raw = (
            _float_or_nan(row.get(target_y_col)) * scale_y
            if target_y_col else np.nan
        )
        if np.isnan(trx_raw) and target_x_pass_col:
            trx_raw = _float_or_nan(row.get(target_x_pass_col)) * scale_x
            try_raw = _float_or_nan(row.get(target_y_pass_col)) * scale_y

        # SkillCorner's own high-pass flag, on passes only (unknown when the column or the value is missing).
        high = _high_pass(row.get(high_pass_col)) if high_pass_col and end_type in ("pass", "header") else pd.NA

        if is_carry:
            # ----------------------------------------------------------------
            # Carry events: emit a CARRY row (possession trajectory) THEN the
            # completion event (PASS / BALL LOST / SHOT …) at the carry end.
            # ----------------------------------------------------------------
            carry_sx, carry_sy = _apply_team_rotation(sx_raw, sy_raw, team)
            carry_ex, carry_ey = _apply_team_rotation(ex_raw, ey_raw, team)
            if np.isnan(carry_ex):
                carry_ex, carry_ey = carry_sx, carry_sy

            carry_subtype = _START_TYPE_SUBTYPE.get(start_type)
            _append_event(
                "CARRY", carry_subtype, period_id, from_id, np.nan, team,
                start_frame, start_t, carry_sx, carry_sy,
                end_frame, end_t, carry_ex, carry_ey,
            )

            # Completion event at the end of the carry
            comp_type, comp_sub = _map_sc_event(end_type, start_type, pass_outcome)
            if comp_type is not None:
                # Start of completion = where the carry ended (carrier release point)
                comp_sx, comp_sy = carry_ex, carry_ey
                # End of completion = targeted reception point (or fallback)
                comp_ex_raw = trx_raw if not np.isnan(trx_raw) else ex_raw
                comp_ey_raw = try_raw if not np.isnan(try_raw) else ey_raw
                comp_ex, comp_ey = _apply_team_rotation(comp_ex_raw, comp_ey_raw, team)
                if np.isnan(comp_ex):
                    comp_ex, comp_ey = comp_sx, comp_sy
                _append_event(
                    comp_type, comp_sub, period_id, from_id, to_id, team,
                    end_frame, end_t, comp_sx, comp_sy,
                    end_frame, end_t, comp_ex, comp_ey, high,
                )
        else:
            # ----------------------------------------------------------------
            # Non-carry: single event as before.
            # ----------------------------------------------------------------
            plaiq_type, subtype = _map_sc_event(end_type, start_type, pass_outcome)
            if plaiq_type is None:
                continue

            sx, sy = _apply_team_rotation(sx_raw, sy_raw, team)

            # For pass/ball-lost prefer the reception/targeted point as End X/Y
            if plaiq_type in ("PASS", "BALL LOST") and not np.isnan(trx_raw):
                ex_r, ey_r = trx_raw, try_raw
            else:
                ex_r, ey_r = ex_raw, ey_raw
            if np.isnan(ex_r):
                ex_r, ey_r = sx_raw, sy_raw

            ex, ey = _apply_team_rotation(ex_r, ey_r, team)

            _append_event(
                plaiq_type, subtype, period_id, from_id, to_id, team,
                start_frame, start_t, sx, sy,
                end_frame, end_t, ex, ey, high,
            )

    result = pd.DataFrame(rows, columns=[*EVENT_COLUMNS, HIGH_PASS])
    result[HIGH_PASS] = result[HIGH_PASS].astype("boolean")
    if not result.empty:
        result = result.sort_values(
            ["Start Frame", "Period"], kind="stable"
        ).reset_index(drop=True)
    return result


def _map_sc_event(
    end_type: str,
    start_type: str,
    pass_outcome: str,
) -> tuple[str | None, str | None]:
    """
    Map a SkillCorner player_possession ``end_type`` / ``pass_outcome`` to a
    (PLAIQ Type, PLAIQ Subtype) pair for the *completion* event.

    Carry events are generated separately in the loop before calling this.
    Returns (None, None) to indicate no completion event should be emitted.
    """
    subtype: str | None = _START_TYPE_SUBTYPE.get(start_type)

    if end_type in ("pass", "header"):
        if pass_outcome == "successful":
            return "PASS", subtype
        elif pass_outcome in ("unsuccessful", "incomplete"):
            return "BALL LOST", "INCOMPLETE PASS"
        else:
            # Unknown outcome — treat as PASS if outcome not null else skip
            return ("PASS", subtype) if pass_outcome else (None, None)

    if end_type == "shot":
        return "SHOT", subtype

    # possession_loss covers tackles, interceptions, and unstructured loss
    if end_type in ("possession_loss", "tackle", "interception"):
        return "BALL LOST", "CHALLENGE"

    if end_type == "out_of_play":
        return "BALL OUT", subtype

    if end_type == "foul_suffered":
        return "FAULT RECEIVED", subtype

    if end_type == "clearance":
        return "BALL OUT", subtype

    return None, None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name present in df, else None."""
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _int_or_none(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _float_or_nan(val) -> float:
    try:
        f = float(val)
        return f if not np.isnan(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _high_pass(val):
    """SkillCorner's ``high_pass`` as a nullable flag: ``True``, ``False`` or ``pd.NA`` when not logged."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return pd.NA
    if isinstance(val, str):
        v = val.strip().lower()
        return True if v == "true" else False if v == "false" else pd.NA
    return bool(val)


def _bool_val(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return False


# ---------------------------------------------------------------------------
# Phase 1f — Adapter class (mirrors SecondSpectrumAdapter)
# ---------------------------------------------------------------------------

class SkillCornerAdapter:
    """
    Ingest SkillCorner open data: tracking + dynamic events + lineup metadata.

    Parameters
    ----------
    match_dir              : directory containing the three SkillCorner files
    match_id               : integer match ID (used to construct filenames)
    normalize_playing_direction : flip coordinates so home always attacks +x
    """

    DEFAULT_MATCH_FILE    = "{id}_match.json"
    DEFAULT_TRACKING_FILE = "{id}_tracking_extrapolated.jsonl"
    DEFAULT_EVENTS_FILE   = "{id}_dynamic_events.csv"

    def __init__(
        self,
        match_dir: str,
        match_id: int,
        *,
        normalize_playing_direction: bool = True,
        match_file: str | None = None,
        tracking_file: str | None = None,
        events_file: str | None = None,
    ):
        self.match_dir  = match_dir
        self.match_id   = int(match_id)
        self.normalize  = normalize_playing_direction

        def _resolve(override, template):
            if override:
                return os.path.join(match_dir, override)
            return os.path.join(match_dir, template.replace("{id}", str(match_id)))

        self.match_path    = _resolve(match_file,    self.DEFAULT_MATCH_FILE)
        self.tracking_path = _resolve(tracking_file, self.DEFAULT_TRACKING_FILE)
        self.events_path   = _resolve(events_file,   self.DEFAULT_EVENTS_FILE)

    # ------------------------------------------------------------------
    def load_meta(self) -> dict:
        return load_skillcorner_meta(self.match_path)

    def load_tracking_raw(self) -> list[dict]:
        return load_skillcorner_tracking(self.tracking_path)

    # ------------------------------------------------------------------
    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """
        Full ingest pipeline.

        Returns
        -------
        home_df, away_df, events_df, meta
        """
        meta    = self.load_meta()
        records = self.load_tracking_raw()

        h1_sign = meta["h1_home_sign"] if self.normalize else 1

        home_df, away_df = tracking_to_wide(
            records,
            meta["home_players"],
            meta["away_players"],
            h1_home_sign=h1_sign,
            normalize_direction=self.normalize,
        )

        # Derive period frame bounds from actual tracking data
        periods = _build_periods_from_df(home_df)
        meta["periods"] = periods

        # Build player_id → jersey mapping from meta so From/To columns in
        # events use jersey numbers (matching tracking column names like Home_4_x).
        pid_to_jersey: dict[int, int] = {
            p["id"]: p["number"]
            for p in meta.get("players_raw", [])
            if p.get("id") is not None and p.get("number") is not None
        }

        events_df = load_skillcorner_events(
            self.events_path,
            home_team_id=meta["home_team_id"],
            away_team_id=meta["away_team_id"],
            periods=periods,
            fps=SC_FPS,
            pitch_length=meta["pitch_length"],
            pitch_width=meta["pitch_width"],
            h1_home_sign=h1_sign,
            normalize_direction=self.normalize,
            player_id_to_jersey=pid_to_jersey,
        )

        # Add pitch config to meta for downstream consumers
        meta["pitch"] = pitch_config_from_sc_meta(meta)
        meta["provider"] = "skillcorner"

        return home_df, away_df, events_df, meta


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def load_skillcorner_game(
    match_dir: str,
    match_id: int,
    *,
    normalize_playing_direction: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Convenience wrapper around :class:`SkillCornerAdapter`.

    Parameters
    ----------
    match_dir              : directory containing {id}_match.json etc.
    match_id               : SkillCorner match ID (integer)
    normalize_playing_direction : flip so home always attacks +x

    Returns
    -------
    home_df, away_df, events_df, meta
    """
    return SkillCornerAdapter(
        match_dir,
        match_id,
        normalize_playing_direction=normalize_playing_direction,
    ).load()
