"""
teams.py — pseudonymous identities across matches (Stage E, ``ponta-blueprint/context-profiles-task.md``).

A ``team_key`` is opaque and stable: a keyed hash (HMAC-SHA256) of the provider and the provider's own team id, under a
secret the caller supplies (one per tenant in the product). The same club from the same provider gets the same key in
every match; nothing in the key reveals the club, its name or the provider's id. One club across two providers is
joined only by an explicit merge in the registry, never inferred. Sides with no provider team id (Metrica's anonymous
sample games) get a key of their own for that match only, so they never link across matches.

**Match, competition and season keys** (review backlog B, AI-04, 27 November 2026) are made the same way, so no
provider id reaches an AI-facing table: ``match_id`` in the semantic events, episodes and profiles is the match key,
and ``event_id`` and ``state_id`` are built from it. The provider's own ids stay with the caller (Postgres in the
product; the per-match provenance file, which is not AI-facing).

**Fails closed** (review DE-15, SE-11, AI-11): the secret is required. ``TeamRegistry.for_reference()`` uses the
published reference secret for tools and tests, which gives no protection, and is refused when ``PONTA_ENV`` is
``production``. A product secret must be at least 16 characters. Every key carries its key version (``t1_``, ``m1_``,
``c1_``, ``s1_``), so a rotated secret is visible in the key instead of silently breaking stored baselines. The merges
file maps provider ids to keys, so it is tenant-confidential.

Player identity across matches is not built here (Phase 5 of the roadmap).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

#: The secret for the reference sets and tests: not a secret, so keys made with it carry no protection. The product
#: passes a per-tenant secret.
REFERENCE_SECRET = "ponta-reference-data"
#: The key version written into every key; bump it when the secret or the derivation changes.
KEY_VERSION = 1
MIN_SECRET_CHARS = 16


class PseudonymError(ValueError):
    """A pseudonym could not be made safely (no secret, the reference secret in production, a non-numeric player ref)."""


def _production() -> bool:
    return os.environ.get("PONTA_ENV", "").strip().lower() in ("production", "prod")


@dataclass(frozen=True)
class MatchIdentity:
    """The opaque keys for one match: what AI-facing tables carry in place of the provider's ids."""

    match_key: str
    team_keys: dict[str, str]
    competition_key: str | None
    season_key: str | None


@dataclass
class TeamRegistry:
    """Opaque keys from provider ids, with explicit merges of one club across providers."""

    secret: str
    merges: dict[str, str] = field(default_factory=dict)  # "provider:id" -> canonical team_key
    key_version: int = KEY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.secret, str) or not self.secret:
            raise PseudonymError("a pseudonym secret is required (TeamRegistry.for_reference() for tools and tests)")
        if self.secret == REFERENCE_SECRET:
            if _production():
                raise PseudonymError("the reference secret is public and is refused when PONTA_ENV is production")
        elif len(self.secret) < MIN_SECRET_CHARS:
            raise PseudonymError(f"a tenant secret must be at least {MIN_SECRET_CHARS} characters")

    @classmethod
    def for_reference(cls, merges: dict[str, str] | None = None) -> TeamRegistry:
        """The registry for reference sets and tests: the published secret, so its keys protect nothing."""
        return cls(secret=REFERENCE_SECRET, merges=dict(merges or {}))

    @property
    def is_reference(self) -> bool:
        return self.secret == REFERENCE_SECRET

    def _hash(self, prefix: str, raw: str, n: int = 12) -> str:
        return f"{prefix}{self.key_version}_" + hmac.new(self.secret.encode(), raw.encode(), hashlib.sha256).hexdigest()[:n]

    def key(self, provider: str, provider_team_id: str | None, *, match_id: str = "", side: str = "") -> str:
        """The team key; an anonymous side (no provider id) is keyed to its match and side only."""
        if provider_team_id is None or str(provider_team_id) == "":
            raw = f"anonymous:{provider}:{match_id}:{side}"
        else:
            raw = f"{provider}:{provider_team_id}"
            if raw in self.merges:
                return self.merges[raw]
        return self._hash("t", raw)

    def keys_for(self, match) -> dict[str, str]:
        """``{"home": key, "away": key}`` for a ``NormalisedMatch``."""
        teams = getattr(match, "teams", None) or {}
        return {side: self.key(match.provider, teams.get(side), match_id=str(match.match_id), side=side) for side in ("home", "away")}

    def match_key(self, provider: str, provider_match_id) -> str:
        """The opaque key of one provider match (``m1_…``)."""
        return self._hash("m", f"match:{provider}:{provider_match_id}")

    def competition_key(self, provider: str, competition) -> str | None:
        return None if competition is None or str(competition) == "" else self._hash("c", f"competition:{provider}:{competition}", 10)

    def season_key(self, provider: str, competition, season) -> str | None:
        return None if season is None or str(season) == "" else self._hash("s", f"season:{provider}:{competition}:{season}", 10)

    def identity(self, match) -> MatchIdentity:
        """Every opaque key for a ``NormalisedMatch``."""
        comp, season = getattr(match, "competition", None), getattr(match, "season", None)
        return MatchIdentity(match_key=self.match_key(match.provider, match.match_id), team_keys=self.keys_for(match),
                             competition_key=self.competition_key(match.provider, comp),
                             season_key=self.season_key(match.provider, comp, season))

    def merge(self, provider: str, provider_team_id: str, team_key: str) -> None:
        """Record that a provider's team is the club already keyed ``team_key`` (an explicit, reviewed decision)."""
        self.merges[f"{provider}:{provider_team_id}"] = team_key

    def save(self, path: Path) -> None:
        """The merges only (the secret is never written). The file maps provider ids to keys: tenant-confidential."""
        Path(path).write_text(json.dumps({"merges": self.merges}, indent=1, sort_keys=True))

    @classmethod
    def load(cls, path: Path, secret: str) -> TeamRegistry:
        p = Path(path)
        merges = json.loads(p.read_text()).get("merges", {}) if p.exists() else {}
        return cls(secret=secret, merges=merges)
