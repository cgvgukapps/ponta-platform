#!/usr/bin/env python3
"""
pitch_config.py — single source of truth for pitch geometry, frame rate, and
playing-direction convention.

Historically these constants (106 m × 68 m, 25 fps, half-pitch 53/34) were
duplicated as literals across app.py, pitch_control.py, hex_value.py,
pass_option_value.py and metrica_io.py. Centralising them here lets the
ingestion layer carry real per-match pitch dimensions (Genius Sports / Second
Spectrum reports actual stadium dimensions, often 105 × 68) without rewriting
the analytics modules.

The module-level FIELD_LENGTH / FIELD_WIDTH / FIELD_DIMENSIONS / DEFAULT_FPS
mirror the historical Metrica-sample defaults (106 × 68, 25 fps) so existing
behaviour is bit-for-bit unchanged when no explicit PitchConfig is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass

# Historical Metrica-sample defaults. Kept as the fallback so nothing downstream
# changes numerically until a loader supplies a real PitchConfig.
_DEFAULT_LENGTH_M = 106.0
_DEFAULT_WIDTH_M = 68.0
_DEFAULT_FPS = 25.0


@dataclass(frozen=True)
class PitchConfig:
    """Geometry + timing for a single match's coordinate system.

    Coordinates everywhere downstream are metres, centred at (0, 0), with the
    home team attacking +x in both halves (after playing-direction
    normalisation). ``length_m`` spans x ∈ [-half_length, +half_length] and
    ``width_m`` spans y ∈ [-half_width, +half_width].
    """

    length_m: float = _DEFAULT_LENGTH_M
    width_m: float = _DEFAULT_WIDTH_M
    fps: float = _DEFAULT_FPS
    home_attacks_positive_x: bool = True

    @property
    def half_length(self) -> float:
        return self.length_m / 2.0

    @property
    def half_width(self) -> float:
        return self.width_m / 2.0

    @property
    def dimensions(self) -> tuple[float, float]:
        return (self.length_m, self.width_m)

    @classmethod
    def from_skillcorner_meta(cls, meta: dict) -> PitchConfig:
        """Build a PitchConfig from a parsed SkillCorner match meta dict.

        Reads ``pitch_length`` and ``pitch_width`` (real venue metres) and
        hard-codes fps=10 (SkillCorner broadcast tracking rate).
        """
        return cls(
            length_m=float(meta.get("pitch_length", _DEFAULT_LENGTH_M)),
            width_m=float(meta.get("pitch_width", _DEFAULT_WIDTH_M)),
            fps=10.0,
        )

    @classmethod
    def from_kloppy_metadata(cls, metadata) -> PitchConfig:
        """Build a PitchConfig from a Kloppy dataset's metadata.

        Falls back to the Metrica-sample defaults when the provider does not
        report pitch dimensions (Kloppy itself defaults to 105 × 68 in that
        case, but we keep our historical 106 × 68 fallback for parity).
        """
        length = _DEFAULT_LENGTH_M
        width = _DEFAULT_WIDTH_M
        cs = getattr(metadata, "coordinate_system", None)
        if cs is not None:
            length = getattr(cs, "pitch_length", None) or length
            width = getattr(cs, "pitch_width", None) or width
        fps = getattr(metadata, "frame_rate", None) or _DEFAULT_FPS
        return cls(length_m=float(length), width_m=float(width), fps=float(fps))


# Default instance + flat aliases for backward compatibility with modules that
# previously hardcoded these literals.
DEFAULT_PITCH = PitchConfig()

FIELD_LENGTH = DEFAULT_PITCH.length_m
FIELD_WIDTH = DEFAULT_PITCH.width_m
FIELD_DIMENSIONS = DEFAULT_PITCH.dimensions
DEFAULT_FPS = DEFAULT_PITCH.fps

# StatsBomb open-data default pitch. StatsBomb normalises every match to
# 120 × 80 "units" (approx. yards); we use FIFA standard 105 × 68 m as the
# real-world mapping. fps=None because StatsBomb is event-based (no fixed
# frame rate). Import this from statsbomb_io where needed; the constant here
# keeps pitch_config self-contained.
SB_PITCH = PitchConfig(length_m=105.0, width_m=68.0, fps=25.0)
