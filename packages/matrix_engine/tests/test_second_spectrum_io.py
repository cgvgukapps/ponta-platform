"""Golden / smoke tests for the native-metre Second Spectrum adapter."""

import numpy as np
import pytest
from matrix_engine.ingestion.metrica_velocities import calc_player_velocities
from matrix_engine.ingestion.second_spectrum_io import (
    SecondSpectrumAdapter,
    load_secondspectrum_game,
)
from ss_fixture import write_ss_fixture


@pytest.fixture
def ss_dir(tmp_path):
    write_ss_fixture(str(tmp_path), pitch_length=105.0, pitch_width=68.0, fps=25)
    return str(tmp_path)


def test_loads_native_metres_and_metadata(ss_dir):
    home, away, events, meta = load_secondspectrum_game(ss_dir)

    assert meta["provider"] == "secondspectrum"
    assert meta["team_a_name"] == "HOMEFC"
    assert meta["team_b_name"] == "AWAYFC"
    assert meta["frame_rate"] == 25.0
    assert meta["pitch"].length_m == 105.0
    assert meta["pitch"].width_m == 68.0

    # Both halves present.
    assert sorted(home["Period"].unique().tolist()) == [1, 2]

    # Wide-format columns match the Metrica/EPTS contract.
    for jersey in (1, 2, 3, 4):
        assert f"Home_{jersey}_x" in home.columns
        assert f"Home_{jersey}_y" in home.columns
        assert f"Away_{jersey}_x" in away.columns
    assert "ball_x" in away.columns and "ball_y" in away.columns
    assert list(events.columns)  # event scaffold exists


def test_coordinates_are_metres_within_pitch(ss_dir):
    home, away, _, meta = load_secondspectrum_game(ss_dir)
    half_l, half_w = meta["pitch"].half_length, meta["pitch"].half_width

    x_cols = [c for c in home.columns if c.endswith("_x")]
    y_cols = [c for c in home.columns if c.endswith("_y")]
    assert np.nanmax(np.abs(home[x_cols].values)) <= half_l + 1e-6
    assert np.nanmax(np.abs(home[y_cols].values)) <= half_w + 1e-6
    # Native metres, not 0-1 normalised: spread must be on the metre scale.
    assert np.nanmax(np.abs(home[x_cols].values)) > 5.0


def test_persistent_player_id_is_opta_id(ss_dir):
    _, _, _, meta = load_secondspectrum_game(ss_dir)
    gk = next(p for p in meta["team_a_players"] if p["shirt"] == "1")
    assert gk["player_id"] == "h1"  # Second Spectrum persistent id (optaId)


def test_home_attacks_positive_x_in_both_halves(ss_dir):
    """After STATIC_HOME_AWAY the home keeper stays on -x in every frame."""
    home, away, _, _ = load_secondspectrum_game(ss_dir, normalize_playing_direction=True)

    assert (home["Home_1_x"] < 0).all()   # home GK defends -x (home attacks +x)
    assert (away["Away_1_x"] > 0).all()   # away GK defends +x


def test_without_normalization_direction_is_not_fixed(ss_dir):
    """Raw orientation keeps the physical halftime side-swap (sanity check)."""
    home, _, _, _ = load_secondspectrum_game(ss_dir, normalize_playing_direction=False)

    p1_gk = home.loc[home["Period"] == 1, "Home_1_x"]
    p2_gk = home.loc[home["Period"] == 2, "Home_1_x"]
    # Sides swap at halftime in the raw feed, so the keeper changes sign.
    assert (p1_gk < 0).all()
    assert (p2_gk > 0).all()


def test_velocities_compute_on_ss_output(ss_dir):
    home, away, _, _ = load_secondspectrum_game(ss_dir)
    home = calc_player_velocities(home)
    speed_cols = [c for c in home.columns if c.endswith("_speed")]
    assert len(speed_cols) == 4
    finite = np.isfinite(home[speed_cols].values)
    assert finite.any()


def test_adapter_limit(ss_dir):
    adapter = SecondSpectrumAdapter(ss_dir, limit=3)
    home, _, _, _ = adapter.load()
    assert len(home) == 3
