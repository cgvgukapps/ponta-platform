#!/usr/bin/env python3
"""
matrix_engine — Matrix, the Unified Tracking Engine (prototype).

Matrix reconciles tracking data from independent providers into one dataset
with a shared clock, a shared player ID space, fused positions, a per-frame
confidence score, and source attribution. This package is the validation
prototype for that idea, built on the tested ingestion layer already in this
repo. It is UI-agnostic — nothing here imports Streamlit — and depends on
``ingestion``/``roster``/``pitch_config`` but never the reverse, so it can be
lifted into its own package later without a rewrite.

Pipeline
--------

``providers`` -> ``sync`` -> ``identity`` -> ``sync`` (refine) -> ``fusion``
-> ``confidence`` -> ``quality`` -> ``export``

:class:`UnifiedTrackingEngine` is the façade over that sequence::

    engine = UnifiedTrackingEngine()
    engine.add_source("metrica", metrica_df)
    engine.add_source("gps", gps_df)
    result = engine.reconcile()
    result.export("out/fused.csv")

Sync runs twice by design. The first pass aligns clocks without knowing who is
who, using signals that relabelling cannot change; the second re-estimates the
offset from players that identity matching has since paired up, which is far
sharper. See ``sync.py`` for why that ordering is forced.

What this prototype does and does not show is set out in ``README.md`` in this
package — read it before quoting any number from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from matrix_engine import confidence as _confidence
from matrix_engine import export as _export
from matrix_engine import fusion as _fusion
from matrix_engine import identity as _identity
from matrix_engine import quality as _quality
from matrix_engine import sync as _sync
from matrix_engine.confidence import score_confidence
from matrix_engine.export import add_kinematics, export_fused
from matrix_engine.fusion import FusionResult, estimate_source_noise, fuse
from matrix_engine.identity import IdentityResult, apply_mapping, match_identities
from matrix_engine.quality import (
    DriftReport,
    QualityReport,
    assess_quality,
    native_quality_summary,
)
from matrix_engine.schema import (
    OPTIONAL_COLUMNS,
    SchemaError,
    validate_pitch_bounds,
    validate_schema,
)
from matrix_engine.sync import ClockAlignment, SyncResult, refine_alignment, synchronize

__all__ = [
    "OPTIONAL_COLUMNS",
    "ClockAlignment",
    "DriftReport",
    "FusionResult",
    "IdentityResult",
    "QualityReport",
    "ReconciliationResult",
    "SchemaError",
    "SyncResult",
    "UnifiedTrackingEngine",
    "add_kinematics",
    "apply_mapping",
    "assess_quality",
    "estimate_source_noise",
    "export_fused",
    "fuse",
    "match_identities",
    "native_quality_summary",
    "refine_alignment",
    "score_confidence",
    "synchronize",
    "validate_pitch_bounds",
    "validate_schema",
]


@dataclass
class ReconciliationResult:
    """
    Everything a reconciliation run produced.

    Attributes
    ----------
    fused : pd.DataFrame
        One row per player per frame: fused position, confidence, source
        attribution, disagreement and gap distance.
    sync : SyncResult
        The final synchronisation, including the alignment used per source.
    identity : dict[str, IdentityResult]
        Per non-reference source, how its players were matched to the
        reference's.
    fusion : FusionResult
        Weights and per-source noise estimates fusion settled on.
    quality : QualityReport
        Drift verdicts and dropout gaps.
    reference : str
        Source whose clock and player IDs the output is expressed in.
    """

    fused: pd.DataFrame
    sync: SyncResult
    identity: dict[str, IdentityResult]
    fusion: FusionResult
    quality: QualityReport
    reference: str
    _kinematics: pd.DataFrame | None = field(default=None, repr=False)

    def with_kinematics(self) -> pd.DataFrame:
        """Fused output plus velocity and acceleration, computed once and cached."""
        if self._kinematics is None:
            self._kinematics = _export.add_kinematics(self.fused)
        return self._kinematics

    def export(self, path: str, **kwargs) -> str:
        """Write the standardised Matrix CSV. See ``export.export_fused``."""
        return _export.export_fused(self.with_kinematics(), path, **kwargs)

    def identity_accuracy(self, truth: dict[str, dict[str, str]]) -> dict[str, float]:
        """
        Score identity matching against a known mapping, per source.

        Only meaningful for synthetic validation, where the true mapping is
        known — see ``matrix_engine/testing/synthetic_source.py``.
        """
        return {
            source: self.identity[source].accuracy_against(mapping)
            for source, mapping in truth.items()
            if source in self.identity
        }


class UnifiedTrackingEngine:
    """
    Façade over the reconciliation pipeline.

    Parameters
    ----------
    target_hz : float, optional
        Output grid rate. Defaults to the highest native rate among the
        sources, so the best feed's temporal resolution survives.
    max_offset_s : float
        Half-width of the clock-offset search.
    n_windows : int
        Windows used for offset and drift estimation.
    max_interp_gap_s : float
        Longest dropout that will be interpolated across rather than left empty.
    refine_with_identity : bool
        Run the second, identity-aware alignment pass. On by default; turning
        it off is mainly useful for measuring what it contributes.
    """

    def __init__(
        self,
        *,
        target_hz: float | None = None,
        max_offset_s: float = 15.0,
        n_windows: int = 6,
        max_interp_gap_s: float = 15.0,
        refine_with_identity: bool = True,
    ) -> None:
        self.target_hz = target_hz
        self.max_offset_s = max_offset_s
        self.n_windows = n_windows
        self.max_interp_gap_s = max_interp_gap_s
        self.refine_with_identity = refine_with_identity
        self._sources: dict[str, pd.DataFrame] = {}

    # -- inputs -------------------------------------------------------------

    def add_source(self, name: str, df: pd.DataFrame) -> UnifiedTrackingEngine:
        """
        Register a source, validating it against the common schema up front.

        Rejecting a malformed feed here rather than three stages later is the
        main practical benefit of having a schema at all.
        """
        validate_schema(df, name=name)
        if df.empty:
            raise ValueError(f"source '{name}' is empty")
        if name in self._sources:
            raise ValueError(f"source '{name}' has already been added")
        self._sources[name] = df
        return self

    @property
    def sources(self) -> dict[str, pd.DataFrame]:
        return dict(self._sources)

    # -- pipeline -----------------------------------------------------------

    def reconcile(self, *, reference: str | None = None) -> ReconciliationResult:
        """
        Run the full pipeline over the registered sources.

        Parameters
        ----------
        reference : str, optional
            Source defining the output clock and player ID space. Defaults to
            the highest-rate source.

        Returns
        -------
        ReconciliationResult
        """
        if not self._sources:
            raise ValueError("no sources added")
        if reference is None:
            reference = max(
                self._sources,
                key=lambda n: float(self._sources[n]["sample_rate_hz"].max()),
            )
        if reference not in self._sources:
            raise KeyError(f"reference '{reference}' not among {sorted(self._sources)}")

        others = [n for n in self._sources if n != reference]

        # Noise is measured on native-rate data: after resampling, the jitter
        # of an upsampled source describes the interpolator, not the sensor.
        noise = {
            name: _fusion.estimate_source_noise(df)
            for name, df in self._sources.items()
        }

        sync_result = self._synchronize(reference=reference)

        identity_results: dict[str, IdentityResult] = {}
        alignments = dict(sync_result.alignments)
        for name in others:
            ref_slice, other_slice = _identity.split_sources(
                sync_result.data, reference, name
            )
            identity_results[name] = _identity.match_identities(ref_slice, other_slice)

            if self.refine_with_identity and name in alignments:
                alignments[name] = _sync.refine_alignment(
                    self._sources[reference],
                    self._sources[name],
                    identity_results[name].mapping,
                    alignments[name],
                    n_windows=self.n_windows,
                )

        if self.refine_with_identity and others:
            sync_result = self._synchronize(reference=reference, alignments=alignments)

        aligned = [sync_result.data.loc[sync_result.data["source"] == reference]]
        for name in others:
            aligned.append(
                _identity.apply_mapping(
                    sync_result.data.loc[sync_result.data["source"] == name],
                    identity_results[name].mapping,
                )
            )
        combined = pd.concat(aligned, ignore_index=True)

        fusion_result = _fusion.fuse(combined, noise=noise)
        fused = _confidence.score_confidence(fusion_result.data, noise=noise)
        quality_report = _quality.assess_quality(self._sources, sync_result.alignments)

        return ReconciliationResult(
            fused=fused,
            sync=sync_result,
            identity=identity_results,
            fusion=fusion_result,
            quality=quality_report,
            reference=reference,
        )

    # -- internals ----------------------------------------------------------

    def _synchronize(
        self,
        *,
        reference: str,
        alignments: dict[str, ClockAlignment] | None = None,
    ) -> SyncResult:
        return _sync.synchronize(
            self._sources,
            reference=reference,
            target_hz=self.target_hz,
            max_offset_s=self.max_offset_s,
            n_windows=self.n_windows,
            max_interp_gap_s=self.max_interp_gap_s,
            alignments=alignments,
        )
