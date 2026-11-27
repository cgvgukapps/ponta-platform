"""Pinned Parquet schemas: matches read together in DuckDB without union_by_name or casts (review DE-08)."""

import hashlib
import json
from dataclasses import replace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from matrix_engine.parity import ParityInputs, classify
from plaiq_engine.match_analysis import MatchAnalysis
from plaiq_engine.parquet_schemas import arrow_schema, to_arrow
from plaiq_engine.semantic_builder import SemanticBuilder
from plaiq_engine.teams import TeamRegistry
from synthetic_match import FPS, synthetic_match

duckdb = pytest.importorskip("duckdb")


def _write(match, out):
    parity = classify(ParityInputs(provider="synthetic", mode="full_tracking", sample_rate_hz=FPS, teams_tracked=("away", "home"),
                                   player_dropout=0.0, has_events=True, has_shot_events=False))
    return SemanticBuilder(MatchAnalysis(match), parity, teams=TeamRegistry.for_reference()).write(out)


@pytest.fixture(scope="module")
def two(tmp_path_factory):
    a = _write(synthetic_match(), tmp_path_factory.mktemp("a"))
    m = synthetic_match()
    ev = m.events.copy()
    ev.loc[2, "Type"] = "BALL LOST"  # different content, so some columns differ in which values are null
    b = _write(replace(m, match_id="2", events=ev), tmp_path_factory.mktemp("b"))
    return a, b


def test_every_file_has_the_pinned_schema(two):
    for paths in two:
        for table in ("semantic_events", "movement_values", "event_provenance"):
            assert pq.read_schema(paths[table]).remove_metadata() == arrow_schema(table), table
    s = arrow_schema("semantic_events")
    assert s.field("physiological_context").type == pa.string() and s.field("movement_value").type == pa.float64()
    assert s.field("event_idx").type == pa.int64() and s.field("units_broken").type == pa.float64()


def test_duckdb_reads_both_matches_without_union_by_name(two):
    files = [str(p["semantic_events"]) for p in two]
    con = duckdb.connect()
    rows = con.execute("select match_id, count(*), avg(opposition_pressure) from read_parquet(?) group by match_id", [files]).fetchall()
    assert len(rows) == 2
    assert sum(r[1] for r in rows) == sum(len(pd.read_parquet(f)) for f in files)
    kinds = con.execute("describe select * from read_parquet(?)", [files]).fetchall()
    assert {k[0]: k[1] for k in kinds}["players_observed"] == "BIGINT"


def test_the_manifest_lists_every_file_with_its_hash(two):
    paths = two[0]
    man = json.loads(paths["manifest"].read_text())
    assert man["schema_version"] and man["match_id"].startswith("m1_")
    for table, entry in man["files"].items():
        if "sha256" in entry:
            assert hashlib.sha256(paths[table].read_bytes()).hexdigest() == entry["sha256"]
        assert entry["rows"] == len(pd.read_parquet(paths[table]))


def test_a_fraction_in_an_integer_column_is_refused():
    df = pd.DataFrame({c: [None] for c in arrow_schema("movement_values").names}).assign(period=[1.5])
    with pytest.raises(ValueError, match="whole number"):
        to_arrow(df, "movement_values")
