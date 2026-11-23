"""P0: the context grid, the package registry, the thresholds config and the probe package."""

import hashlib

import numpy as np
import pandas as pd
import pytest
from context_fixture import tiny_match
from matrix_engine.parity import ParityInputs, classify
from plaiq_engine.context import thresholds
from plaiq_engine.context.grid import GRID_COLUMNS, build_grid
from plaiq_engine.context.probe import ProbePackage
from plaiq_engine.context.registry import (
    ContextField,
    ContextPackage,
    PackageOutput,
    Registry,
)


def test_grid_rate_keys_and_minutes():
    m = tiny_match(seconds=10.0, periods=(1, 2), ball=lambda t: (0.5, 0.0))
    g = build_grid(m, 5.0)
    assert list(g.columns) == list(GRID_COLUMNS)
    assert g["frame_ts"].is_monotonic_increasing and g["frame_ts"].is_unique
    assert (g["frame_ts"] >= 0).all() and not np.signbit(g["frame_ts"]).any()
    assert len(g) == 100  # 2 periods x 10 s x 5 Hz
    assert np.allclose(np.diff(g.loc[g["period"] == 1, "frame_ts"]), 0.2)
    assert g.loc[g["period"] == 1, "minute"].eq(1).all() and g.loc[g["period"] == 2, "minute"].eq(46).all()
    assert set(g["native_frame"]) <= set(m.frames["frame"])
    assert g["match_id"].eq("metrica_t1").all()


def test_grid_rejects_a_bad_rate():
    with pytest.raises(ValueError):
        build_grid(tiny_match(seconds=1.0), 0)


class _Pkg(ContextPackage):
    def __init__(self, name, deps=(), fields=None):
        self.name, self.version, self.depends_on = name, "0.0.1", deps
        self.fields = fields if fields is not None else (ContextField(f"{name}_v", "x", "observed", "d"),)

    def compute(self, ctx):
        return PackageOutput(frames=pd.DataFrame({f.name: np.zeros(len(ctx.grid)) for f in self.fields}))


def test_registry_orders_by_dependency_then_name():
    reg = Registry([_Pkg("c", ("a",)), _Pkg("b"), _Pkg("a")])
    assert reg.order() == ["a", "b", "c"]


def test_registry_refuses_cycles_missing_dependencies_and_unprefixed_fields():
    with pytest.raises(ValueError, match="cycle"):
        Registry([_Pkg("a", ("b",)), _Pkg("b", ("a",))]).order()
    with pytest.raises(ValueError, match="missing"):
        Registry([_Pkg("a", ("zz",))]).order()
    with pytest.raises(ValueError, match="prefixed"):
        Registry([_Pkg("a", fields=(ContextField("other", "x", "observed", "d"),))])


def test_registry_checks_declared_columns():
    class Bad(_Pkg):
        def compute(self, ctx):
            return PackageOutput(frames=pd.DataFrame({"bad_other": np.zeros(len(ctx.grid))}))
    with pytest.raises(ValueError, match="do not match"):
        Registry([Bad("bad")]).run(tiny_match(seconds=1.0))


def test_probe_runs_end_to_end_and_stamps_versions(tmp_path):
    m = tiny_match(seconds=10.0, ball=lambda t: (0.5, 0.0) if t < 5 else None)
    res = Registry([ProbePackage()]).run(m)
    assert res.frames["probe_version"].eq("0.1.0").all()
    assert res.frames["probe_ball_detected"].mean() == pytest.approx(0.5, abs=0.03)
    report = res.attach_to(classify(ParityInputs(provider="metrica", mode="full_tracking", sample_rate_hz=25.0, teams_tracked=("away", "home"))))
    assert [e.metric for e in report.packages["probe"]] == ["ball_detected"]
    p1 = res.write(tmp_path / "a")["context_frames"]
    p2 = res.write(tmp_path / "b")["context_frames"]
    assert hashlib.sha256(p1.read_bytes()).hexdigest() == hashlib.sha256(p2.read_bytes()).hexdigest()


def test_every_threshold_names_its_basis():
    """The founder's decision order (23 November 2026): every value says what it rests on; the tag list names only real thresholds."""
    for t in thresholds.THRESHOLDS.values():
        assert t.basis in ("published", "reference", "engineering", "judgement"), t.name
    assert set(thresholds._BASIS) <= set(thresholds.THRESHOLDS), set(thresholds._BASIS) - set(thresholds.THRESHOLDS)
    assert thresholds.THRESHOLDS["p4v_fifa_max_speed_kmh"].basis == "published"
    assert thresholds.THRESHOLDS["p7_cost_weight"].basis == "judgement"


def test_thresholds_have_status_and_sources():
    for t in thresholds.THRESHOLDS.values():
        assert t.status in ("proposed", "measured", "agreed")
        if t.status in ("measured", "agreed"):
            assert t.source, t.name
        if t.status == "agreed":
            assert "agreed by the founder" in t.source, f"{t.name}: agreed values must name the founder's decision"
    assert thresholds.values({"p1_control_radius_m": 2.0})["p1_control_radius_m"] == 2.0
    with pytest.raises(KeyError):
        thresholds.values({"no_such_threshold": 1.0})
