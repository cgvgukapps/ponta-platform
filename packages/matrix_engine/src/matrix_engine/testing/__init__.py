"""
testing — fixtures and generators used to validate Matrix without real
multi-vendor data.

Everything here exists because the repo has one real recording per match and no
wearable data. It is validation scaffolding, not part of the reconciliation
pipeline, and nothing in ``matrix_engine`` outside this package imports it.
"""

from matrix_engine.testing.synthetic_source import (
    DegradationParams,
    SyntheticSource,
    degrade,
)

__all__ = ["DegradationParams", "SyntheticSource", "degrade"]
