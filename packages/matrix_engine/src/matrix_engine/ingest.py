"""
ingest.py — what an adapter read, dropped and assumed (review DE-12, 27 November 2026).

Adapters make reasonable local choices (drop a record with no period, skip an event with no frame, read the playing
direction from metadata or assume it). None was counted, so a worker could not tell a club that, say, 5% of events or a
period's direction fell back to a default; the PFF reader's faults of 24 November were found by hand for that reason.
An ``IngestReport`` travels with the ``NormalisedMatch`` (``match.ingest``) and appears in the Data check's coverage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class IngestReport:
    """Counts by reason, the fallbacks used, and where each period's playing direction came from."""

    provider: str
    rows_read: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)
    orientation: dict[int, str] = field(default_factory=dict)

    def count(self, reason: str, n: int = 1) -> None:
        if n:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(n)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["orientation"] = {str(k): v for k, v in sorted(self.orientation.items())}
        return d
