"""
pipeline.py — the ``MatrixPipeline`` façade.

One entry point over what Matrix already does, chosen by how many sources a match has:

* **One source (Tier 1): Normalise only.** The match is validated against the schema
  and its coverage is summarised. Positions are not resampled, interpolated or
  fused: a single source has nothing to reconcile against, and Tier 1 makes no
  reconciliation claim (``ponta-blueprint/data-tiers.md``).
* **Two or more sources (Tiers 2 and 3): reconcile.** Exactly the existing
  ``UnifiedTrackingEngine`` flow (providers → sync → identity → sync (refine) →
  fusion → confidence → quality), unchanged. ``run`` returns its
  ``ReconciliationResult``; this façade adds nothing to the maths.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from matrix_engine import ReconciliationResult, UnifiedTrackingEngine
from matrix_engine.match import NormalisedMatch
from matrix_engine.quality import coverage_summary, detect_gaps
from matrix_engine.schema import validate_schema


@dataclass(frozen=True)
class NormaliseResult:
    """
    The Tier 1 output: the match as loaded, validated, with its coverage measured.

    Attributes
    ----------
    match : NormalisedMatch
        Unchanged positions, ball and events.
    coverage : pd.DataFrame
        ``quality.coverage_summary`` for the single source.
    gaps : pd.DataFrame
        ``quality.detect_gaps`` for the single source: stretches where a player
        stopped being reported.
    """

    match: NormalisedMatch
    coverage: pd.DataFrame
    gaps: pd.DataFrame


class MatrixPipeline:
    """
    Façade over Matrix.

    Parameters
    ----------
    **engine_options
        Passed to ``UnifiedTrackingEngine`` for multi-source runs (``target_hz``,
        ``max_offset_s``, ``n_windows``, ``max_interp_gap_s``,
        ``refine_with_identity``). Ignored for a single source.
    """

    def __init__(self, **engine_options) -> None:
        self.engine_options = engine_options

    def normalise(self, match: NormalisedMatch) -> NormaliseResult:
        """Tier 1: validate one source and measure its coverage. No position is changed."""
        validate_schema(match.tracking, name=match.provider)
        sources = {match.provider: match.tracking}
        return NormaliseResult(match=match, coverage=coverage_summary(sources), gaps=detect_gaps(sources))

    def run(self, sources: Mapping[str, pd.DataFrame], *, reference: str | None = None) -> ReconciliationResult:
        """
        Tiers 2 and 3: reconcile two or more Matrix-schema sources.

        Behaviour-identical to building a ``UnifiedTrackingEngine`` with the same
        options, adding the sources in order and calling ``reconcile``.
        """
        if len(sources) < 2:
            raise ValueError("run() reconciles two or more sources; use normalise() for one")
        engine = UnifiedTrackingEngine(**self.engine_options)
        for name, df in sources.items():
            engine.add_source(name, df)
        return engine.reconcile(reference=reference)
