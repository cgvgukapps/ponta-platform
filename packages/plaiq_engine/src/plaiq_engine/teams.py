"""
teams.py — a pseudonymous team identity across matches (Stage E, ``ponta-blueprint/context-profiles-task.md``).

A ``team_key`` is opaque and stable: a keyed hash (HMAC-SHA256) of the provider and the provider's own team id, under a
secret the caller supplies (one per tenant in the product). The same club from the same provider gets the same key in
every match; nothing in the key reveals the club, its name or the provider's id. One club across two providers is
joined only by an explicit merge in the registry, never inferred. Sides with no provider team id (Metrica's anonymous
sample games) get a key of their own for that match only, so they never link across matches.

Player identity across matches is not built here (Phase 5 of the roadmap).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from pathlib import Path

#: The secret for the reference sets and tests: not a secret, so keys made with it carry no protection. The product
#: passes a per-tenant secret.
REFERENCE_SECRET = "ponta-reference-data"


@dataclass
class TeamRegistry:
    """Team keys from ``(provider, provider_team_id)``, with explicit merges of one club across providers."""

    secret: str = REFERENCE_SECRET
    merges: dict[str, str] = field(default_factory=dict)  # "provider:id" -> canonical team_key

    def key(self, provider: str, provider_team_id: str | None, *, match_id: str = "", side: str = "") -> str:
        """The team key; an anonymous side (no provider id) is keyed to its match and side only."""
        if provider_team_id is None or str(provider_team_id) == "":
            raw = f"anonymous:{provider}:{match_id}:{side}"
        else:
            raw = f"{provider}:{provider_team_id}"
            if raw in self.merges:
                return self.merges[raw]
        return "t_" + hmac.new(self.secret.encode(), raw.encode(), hashlib.sha256).hexdigest()[:12]

    def keys_for(self, match) -> dict[str, str]:
        """``{"home": key, "away": key}`` for a ``NormalisedMatch``."""
        teams = getattr(match, "teams", None) or {}
        return {side: self.key(match.provider, teams.get(side), match_id=str(match.match_id), side=side) for side in ("home", "away")}

    def merge(self, provider: str, provider_team_id: str, team_key: str) -> None:
        """Record that a provider's team is the club already keyed ``team_key`` (an explicit, reviewed decision)."""
        self.merges[f"{provider}:{provider_team_id}"] = team_key

    def save(self, path: Path) -> None:
        """The merges only (the secret is never written)."""
        Path(path).write_text(json.dumps({"merges": self.merges}, indent=1, sort_keys=True))

    @classmethod
    def load(cls, path: Path, secret: str = REFERENCE_SECRET) -> TeamRegistry:
        p = Path(path)
        merges = json.loads(p.read_text()).get("merges", {}) if p.exists() else {}
        return cls(secret=secret, merges=merges)
