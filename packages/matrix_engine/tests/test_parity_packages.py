"""Context package entries in the Data check: additive, grouped by package, base report unchanged."""

from matrix_engine.parity import (
    PackageEntry,
    ParityInputs,
    classify,
    with_package_entries,
)


def _base():
    return classify(ParityInputs(provider="metrica", mode="full_tracking", sample_rate_hz=25.0, teams_tracked=("away", "home"),
                                 ball_detected_share=0.7, player_dropout=0.0, has_events=True, has_shot_events=True))


def _entry(pkg="p1", metric="possession", state="full"):
    return PackageEntry(package=pkg, package_version="0.1.0", metric=metric, label="Possession", state=state,
                        reason="r", measure="m", coverage="99% of match time labelled")


def test_base_report_is_unchanged_without_entries():
    r = _base()
    d = r.to_dict()
    assert "packages" not in d and "package_summary" not in d
    assert r.packages == {}


def test_entries_are_added_beside_the_base_metrics():
    r = _base()
    r2 = with_package_entries(r, [_entry(), _entry(metric="game_state", state="degraded")])
    assert r2.metrics == r.metrics and r2.summary == r.summary
    d = r2.to_dict()
    assert [e["metric"] for e in d["packages"]["p1"]] == ["possession", "game_state"]
    assert d["package_summary"] == {"full": 1, "degraded": 1, "blocked": 0}
    assert {k: v for k, v in d.items() if k not in ("packages", "package_summary")} == r.to_dict()


def test_rerunning_a_package_replaces_its_entries():
    r = with_package_entries(_base(), [_entry(), _entry(pkg="probe", metric="x")])
    r = with_package_entries(r, [_entry(state="degraded")])
    assert [e.state for e in r.packages["p1"]] == ["degraded"]
    assert set(r.packages) == {"p1", "probe"}
