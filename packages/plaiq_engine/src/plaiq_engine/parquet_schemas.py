"""
parquet_schemas.py — one pinned Arrow schema per AI-facing table (review DE-08, 27 November 2026).

pandas inferred each file's types from its rows, so an all-null column was Arrow ``null`` in one match and ``int64`` or
``string`` in another, and a count was ``int64`` where every value was whole and ``double`` where one was not. Reading
many matches together in DuckDB then needed ``union_by_name`` and casts, or failed. Every table is now written through
the schema here: integers are nullable ``int64``, measures ``double``, flags ``bool`` and everything else ``string``.
Counts that the context packages can return as fractions on a resampled grid (units broken, goal-side defenders,
overload, engaging defenders) are ``double`` everywhere. Column order is the table's.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

INT: dict[str, frozenset[str]] = {
    "semantic_events": frozenset({"event_idx", "period", "players_observed", "score_diff", "red_card_diff", "runs_in_behind"}),
    "semantic_states": frozenset({
        "period", "chain_id", "score_diff", "red_card_diff", "passes", "passes_completed", "passes_breaking_a_line", "runs",
        "runs_in_behind", "runs_behind", "runs_coming_short", "runs_dropping_off", "runs_support", "runs_overlap",
        "runs_underlap", "runs_pulling_wide", "runs_pulling_half_space", "runs_run_ahead_of_the_ball", "runs_cross_receiver",
        "runs_carry", "defensive_runs", "engagements", "engagements_recovery", "engagements_counter_press",
        "engagements_press", "engagements_pressure", "presses", "events", "min_outfield_observed", "shots",
        "first_event_idx", "last_event_idx"}),
    "movement_values": frozenset({"period"}),
    "event_provenance": frozenset({"event_idx"}),
}
FLOAT: dict[str, frozenset[str]] = {
    "semantic_events": frozenset({
        "timestamp_s", "dq_score", "chosen_option_pov", "best_option_pov", "xt_delta", "movement_value", "opposition_pressure",
        "pass_completion_probability", "dq_v2_score", "gain_v2_score", "observed_near_ball_share", "units_broken",
        "goal_side_defenders", "overload_at_ball", "adjusted_choice", "open_high_value_share", "engaging_defenders"}),
    "semantic_states": frozenset({
        "start_s", "end_s", "duration_s", "share_controlled", "share_in_flight", "share_loose_or_contested", "opposed_pct",
        "press_share", "counter_press_share", "fifa_mid_block_share", "fifa_low_block_share", "team_length_m", "team_width_m",
        "opp_length_m", "opp_width_m", "opp_def_line_m", "opp_def_mid_gap_m", "opp_mid_att_gap_m", "share_defensive_third",
        "share_middle_third", "share_attacking_third", "carrier_pressure_median", "carrier_pressure_max",
        "goal_side_defenders_median", "overload_at_ball_median", "open_high_value_share", "opp_open_high_value_share",
        "ball_detected_share", "gain_median", "choice_median", "adjusted_choice_median"}),
    "movement_values": frozenset({"timestamp_s", "movement_value", "xt_delta", "opposition_pressure"}),
    "event_provenance": frozenset(),
}
BOOL: dict[str, frozenset[str]] = {
    "semantic_events": frozenset({"not_scored", "ball_detected", "received_inside_shape"}),
    "semantic_states": frozenset({"cleared", "counter_press", "led_to_shot", "ended_in_shot"}),
    "movement_values": frozenset(),
    "event_provenance": frozenset(),
}


def _columns(table: str) -> tuple[str, ...]:
    from plaiq_engine.semantic_builder import (
        MOVEMENT_COLUMNS,
        PROVENANCE_COLUMNS,
        SEMANTIC_COLUMNS,
        STATE_EXTRA_COLUMNS,
    )

    return {"semantic_events": SEMANTIC_COLUMNS, "semantic_states": STATE_EXTRA_COLUMNS, "movement_values": MOVEMENT_COLUMNS,
            "event_provenance": PROVENANCE_COLUMNS}[table]


def arrow_schema(table: str) -> pa.Schema:
    """The pinned schema of ``table``."""
    def typ(c):
        return pa.int64() if c in INT[table] else pa.float64() if c in FLOAT[table] else pa.bool_() if c in BOOL[table] else pa.string()

    return pa.schema([pa.field(c, typ(c), nullable=True) for c in _columns(table)])


def _coerce(s: pd.Series, t: pa.DataType) -> list:
    out = []
    for v in s.tolist():
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            out.append(None)
        elif pa.types.is_int64(t):
            f = float(v)
            if not f.is_integer():
                raise ValueError(f"{s.name}: {v!r} is not a whole number")
            out.append(int(f))
        elif pa.types.is_float64(t):
            out.append(float(v))
        elif pa.types.is_boolean(t):
            out.append(bool(v))
        else:
            out.append(str(v))
    return out


def to_arrow(df: pd.DataFrame, table: str) -> pa.Table:
    """``df`` as an Arrow table with the pinned schema; a column the schema does not know, or a missing one, raises."""
    schema = arrow_schema(table)
    if list(df.columns) != schema.names:
        raise ValueError(f"{table}: columns differ from the pinned schema: {set(df.columns) ^ set(schema.names)}")
    return pa.table({f.name: pa.array(_coerce(df[f.name], f.type), type=f.type) for f in schema}, schema=schema)


def write_table(df: pd.DataFrame, table: str, path: Path) -> dict:
    """Write through the pinned schema; returns the manifest entry (rows, sha256)."""
    pq.write_table(to_arrow(df, table), path)
    return {"file": Path(path).name, "rows": int(len(df)), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def write_manifest(out_dir: Path, entries: dict[str, dict], **meta) -> Path:
    """``manifest.json`` beside the files: each file, its rows and hash, and the versions it was written with."""
    p = Path(out_dir) / "manifest.json"
    p.write_text(json.dumps({**meta, "files": entries}, indent=1, sort_keys=True) + "\n")
    return p
