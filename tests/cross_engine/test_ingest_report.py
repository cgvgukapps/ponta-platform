"""What the SkillCorner loader read, dropped and assumed is counted and reaches the Data check (review DE-12)."""

import copy

from matrix_engine.ingestion.skillcorner_io import SkillCornerAdapter
from matrix_engine.match import load_skillcorner_match
from matrix_engine.parity import parity_for_match
from skillcorner_fixture import MATCH_ID, make_tracking_records, write_fixture_dir


def test_drops_and_fallbacks_are_counted(tmp_path):
    records = make_tracking_records(n_p1_frames=5, n_p2_frames=5)
    extra = copy.deepcopy(records[:2])
    for k, r in enumerate(extra):
        r["frame"], r["period"] = 900 + k, None  # half-time records: dropped, now counted
    bad = copy.deepcopy(records[2])
    bad["timestamp"] = "not a time"
    records = [*records[:2], bad, *records[3:], *extra]
    match_dir = write_fixture_dir(str(tmp_path), tracking_records=records)
    _, _, _, meta = SkillCornerAdapter(match_dir, MATCH_ID).load()
    rep = meta["ingest"]
    assert rep.rows_read["tracking_records"] == len(records)
    assert rep.dropped["tracking_records_without_period"] == 2
    assert rep.dropped["tracking_timestamp_unparsable"] == 1 and rep.fallbacks
    assert set(rep.orientation) == {1, 2} and all(v.startswith("match file") for v in rep.orientation.values())


def test_the_report_reaches_the_data_check(tmp_path):
    match_dir = write_fixture_dir(str(tmp_path))
    m = load_skillcorner_match(match_dir, MATCH_ID)
    cov = parity_for_match(m).coverage
    assert cov["ingest"]["provider"] == "skillcorner" and "orientation" in cov["ingest"]
