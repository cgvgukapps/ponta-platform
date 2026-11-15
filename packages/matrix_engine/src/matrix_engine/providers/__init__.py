"""
providers — per-vendor adapters that emit the common Matrix tracking schema.

Each adapter is a thin translation layer over an existing, tested loader in
``ingestion/``. Adapters own vendor quirks (coordinate systems, clock
conventions, identifier formats); everything downstream of this package sees
only ``matrix_engine.schema``'s long format.
"""

from matrix_engine.providers.metrica import load_metrica_source, metrica_pitch
from matrix_engine.providers.skillcorner import (
    load_skillcorner_source,
    skillcorner_pitch,
)
from matrix_engine.providers.statsports import (
    DEFAULT_STATSPORTS_SOURCE,
    StatsportsUnavailable,
    load_statsports_source,
    statsports_pitch,
)

__all__ = [
    "DEFAULT_STATSPORTS_SOURCE",
    "StatsportsUnavailable",
    "load_metrica_source",
    "load_skillcorner_source",
    "load_statsports_source",
    "metrica_pitch",
    "skillcorner_pitch",
    "statsports_pitch",
]
