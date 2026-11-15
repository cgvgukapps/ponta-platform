"""NormalisedMatch round trip and the MatrixPipeline façade."""

import numpy as np
import pandas as pd
import pytest
from matrix_engine import UnifiedTrackingEngine
from matrix_engine.match import from_wide_frames, to_wide_frames
from matrix_engine.pipeline import MatrixPipeline
from matrix_engine.pitch_config import PitchConfig
from matrix_engine.schema import validate_schema
from matrix_engine.testing import DegradationParams, degrade


def _wide(team: str, jerseys: list[str], n: int = 8, seed: int = 0, ball: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prefix = team.capitalize()
    idx = pd.Index(np.arange(101, 101 + n), name="Frame")
    cols = {"Period": np.r_[np.ones(n // 2, int), np.full(n - n // 2, 2)], "Time [s]": np.arange(n) * 0.04}
    for j in jerseys:
        x = rng.uniform(-50, 50, n)
        y = rng.uniform(-30, 30, n)
        x[1] = np.nan  # a dropped observation
        y[1] = np.nan
        cols[f"{prefix}_{j}_x"] = x
        cols[f"{prefix}_{j}_y"] = y
    if ball:
        bx = rng.uniform(-50, 50, n)
        by = rng.uniform(-30, 30, n)
        bx[2] = by[2] = np.nan  # ball not detected
        cols["ball_x"], cols["ball_y"] = bx, by
    return pd.DataFrame(cols, index=idx)


@pytest.fixture
def wide_pair():
    home = _wide("home", ["11", "1", "7"], seed=1)
    away = _wide("away", ["28", "18"], seed=2, ball=True)
    events = pd.DataFrame({"Type": ["PASS"], "From": ["Player 7"], "Start Frame": [102]})
    return home, away, events


def test_round_trip_reproduces_positions_ball_order_and_grid(wide_pair):
    home, away, events = wide_pair
    m = from_wide_frames(home, away, events, provider="test", match_id="1", fps=25.0, pitch=PitchConfig())
    validate_schema(m.tracking)
    assert m.slots == {"home": ("11", "1", "7"), "away": ("28", "18")}
    assert m.n_frames == len(home)
    assert len(m.ball) == len(away) - 1

    h2, a2 = to_wide_frames(m)
    assert h2.index.equals(home.index) and a2.index.equals(away.index)
    for src, out in ((home, h2), (away, a2)):
        for col in src.columns:
            np.testing.assert_array_equal(src[col].to_numpy(), out[col].to_numpy())
    # Vendor slot order survives the round trip.
    assert [c for c in h2.columns if c.startswith("Home_")] == [c for c in home.columns if c.startswith("Home_")]
    # Both rebuilt frames carry the ball.
    np.testing.assert_array_equal(h2["ball_x"].to_numpy(), away["ball_x"].to_numpy())


def test_tracking_has_no_rows_for_unobserved_players(wide_pair):
    home, away, events = wide_pair
    m = from_wide_frames(home, away, events, provider="test", match_id="1", fps=25.0, pitch=PitchConfig())
    assert not m.tracking["frame"].eq(102).any()  # frame 102 is the dropped observation for every player
    assert np.isfinite(m.tracking[["x", "y"]].to_numpy()).all()


def test_normalise_changes_no_position(wide_pair):
    home, away, events = wide_pair
    m = from_wide_frames(home, away, events, provider="test", match_id="1", fps=25.0, pitch=PitchConfig())
    out = MatrixPipeline().normalise(m)
    assert out.match is m
    pd.testing.assert_frame_equal(out.match.tracking, m.tracking)
    assert set(out.coverage["source"]) == {"test"}


def test_run_needs_two_sources(metrica_home):
    with pytest.raises(ValueError):
        MatrixPipeline().run({"metrica": metrica_home})


def test_run_is_identical_to_the_engine(metrica_home):
    params = DegradationParams(target_hz=10.0, position_noise_m=0.5, dropout_rate=0.1, n_dropout_bursts=2, dropout_burst_s=4.0,
                               shuffle_identities=True, time_offset_s=2.5, time_drift_rate=0.0, seed=7)
    synthetic = degrade(metrica_home, params)
    name = synthetic.data["source"].iloc[0]

    engine = UnifiedTrackingEngine()
    engine.add_source("metrica", metrica_home)
    engine.add_source(name, synthetic.data)
    expected = engine.reconcile(reference="metrica")

    got = MatrixPipeline().run({"metrica": metrica_home, name: synthetic.data}, reference="metrica")
    pd.testing.assert_frame_equal(got.fused, expected.fused)
    assert got.reference == expected.reference
