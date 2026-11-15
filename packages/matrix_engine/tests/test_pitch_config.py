"""Lock the pitch-geometry single source so the Metrica baseline is unchanged."""

from matrix_engine.pitch_config import (
    DEFAULT_FPS,
    DEFAULT_PITCH,
    FIELD_DIMENSIONS,
    FIELD_LENGTH,
    FIELD_WIDTH,
    PitchConfig,
)


def test_historical_metrica_defaults_unchanged():
    # These literals were previously hardcoded across app.py, pitch_control.py,
    # hex_value.py, pass_option_value.py and metrica_io.py. The refactor must
    # keep them bit-for-bit identical so Metrica output does not move.
    assert FIELD_LENGTH == 106.0
    assert FIELD_WIDTH == 68.0
    assert FIELD_DIMENSIONS == (106.0, 68.0)
    assert DEFAULT_FPS == 25.0
    assert DEFAULT_PITCH.half_length == 53.0
    assert DEFAULT_PITCH.half_width == 34.0


def test_pitchconfig_derived_properties():
    cfg = PitchConfig(length_m=105.0, width_m=68.0, fps=25.0)
    assert cfg.half_length == 52.5
    assert cfg.half_width == 34.0
    assert cfg.dimensions == (105.0, 68.0)
    assert cfg.home_attacks_positive_x is True


def test_from_kloppy_metadata_reads_dimensions():
    class _CS:
        pitch_length = 104.85
        pitch_width = 67.97

    class _Meta:
        coordinate_system = _CS()
        frame_rate = 25

    cfg = PitchConfig.from_kloppy_metadata(_Meta())
    assert cfg.length_m == 104.85
    assert cfg.width_m == 67.97
    assert cfg.fps == 25.0


def test_from_kloppy_metadata_falls_back_when_missing():
    class _Meta:
        coordinate_system = None
        frame_rate = None

    cfg = PitchConfig.from_kloppy_metadata(_Meta())
    assert cfg.length_m == 106.0
    assert cfg.width_m == 68.0
    assert cfg.fps == 25.0
