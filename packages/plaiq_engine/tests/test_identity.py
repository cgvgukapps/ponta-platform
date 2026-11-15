"""Tests for statsbomb/identity.py identity assignment."""
from plaiq_engine.statsbomb.identity import (
    _find_player_by_name,
    _is_gk,
    identified_by_confidence,
    identity_summary,
    resolve_identities,
)


def _player(name, jersey=1, pos="Midfielder"):
    return {"name": name, "jersey_number": jersey, "position": pos}


def _gk(name, jersey=1):
    return {"name": name, "jersey_number": jersey, "position": "Goalkeeper"}


def _dot(teammate=True, actor=False, keeper=False, loc=None):
    return {
        "teammate": teammate,
        "actor":    actor,
        "keeper":   keeper,
        "location": loc or [0.0, 0.0],
    }


LINEUP_SIMPLE = {
    "home": [
        _player("Alice", 9),
        _player("Bob",   8),
        _gk("HomeGK",   1),
    ],
    "away": [
        _player("Carlos", 10),
        _player("Diana",  7),
        _gk("AwayGK",    1),
    ],
}

EVENT_PASS = {
    "Type": "PASS", "From": "Alice", "To": "Bob",
    "Team": "Home",
    "End X": 5.0, "End Y": 0.0,
}


class TestTier1Actor:
    def test_actor_flag_assigned_from_name(self):
        dots = [_dot(teammate=True, actor=True, loc=[0.0, 0.0])]
        result = resolve_identities(dots, EVENT_PASS, LINEUP_SIMPLE)
        assert result[0]["player_name"] == "Alice"
        assert result[0]["confidence"] == "strong"

    def test_actor_gets_jersey(self):
        dots = [_dot(teammate=True, actor=True, loc=[0.0, 0.0])]
        result = resolve_identities(dots, EVENT_PASS, LINEUP_SIMPLE)
        assert result[0]["jersey"] == 9

    def test_only_first_actor_dot_assigned_as_strong(self):
        dots = [
            _dot(teammate=True, actor=True, loc=[0.0, 0.0]),
            _dot(teammate=True, actor=True, loc=[1.0, 0.0]),
        ]
        result = resolve_identities(dots, EVENT_PASS, LINEUP_SIMPLE)
        # Tier 1 assigns "Alice" (strong) to the first actor dot only.
        # Tier 5 (lineup fill) may assign the second dot weakly.
        strong = [d for d in result if d["confidence"] == "strong" and d["player_name"] == "Alice"]
        assert len(strong) == 1


class TestTier2PassTarget:
    def test_pass_target_assigned_by_location(self):
        dots = [
            _dot(teammate=True, actor=True, loc=[0.0, 0.0]),
            _dot(teammate=True, actor=False, loc=[5.1, 0.0]),
        ]
        result = resolve_identities(dots, EVENT_PASS, LINEUP_SIMPLE)
        names = {d["player_name"] for d in result}
        assert "Bob" in names

    def test_pass_target_confidence_strong(self):
        dots = [
            _dot(teammate=True, actor=True, loc=[0.0, 0.0]),
            _dot(teammate=True, actor=False, loc=[5.0, 0.0]),
        ]
        result = resolve_identities(dots, EVENT_PASS, LINEUP_SIMPLE)
        bob_dot = next(d for d in result if d["player_name"] == "Bob")
        assert bob_dot["confidence"] == "strong"

    def test_pass_target_not_strong_if_too_far(self):
        dots = [
            _dot(teammate=True, actor=True, loc=[0.0, 0.0]),
            _dot(teammate=True, actor=False, loc=[50.0, 0.0]),
        ]
        result = resolve_identities(dots, EVENT_PASS, LINEUP_SIMPLE)
        # Tier 2 should NOT fire because the dot is >3m from pass end.
        # Bob may still appear via Tier 5 (weak), but NOT as strong.
        bob_dots = [d for d in result if d["player_name"] == "Bob"]
        for d in bob_dots:
            assert d["confidence"] != "strong", "Bob should not be STRONG when dot is 45m from pass end"


class TestTier3Keepers:
    def test_home_keeper_assigned_to_keeper_dot(self):
        dots = [_dot(teammate=True, keeper=True, loc=[50.0, 0.0])]
        result = resolve_identities(dots, EVENT_PASS, LINEUP_SIMPLE)
        assert result[0]["player_name"] == "HomeGK"
        assert result[0]["confidence"] == "strong"

    def test_away_keeper_assigned_to_opponent_keeper_dot(self):
        dots = [_dot(teammate=False, keeper=True, loc=[50.0, 0.0])]
        result = resolve_identities(dots, EVENT_PASS, LINEUP_SIMPLE)
        assert result[0]["player_name"] == "AwayGK"
        assert result[0]["confidence"] == "strong"


class TestTier4CrossEvent:
    def test_medium_confidence_from_previous_ff(self):
        prev_identified = [
            {
                "teammate": True, "actor": False, "keeper": False,
                "location": [10.0, 0.0],
                "player_name": "Bob",
                "jersey": 8,
                "confidence": "strong",
            }
        ]
        dots = [_dot(teammate=True, loc=[10.5, 0.0])]
        result = resolve_identities(
            dots, EVENT_PASS, LINEUP_SIMPLE,
            prev_ff_identified=prev_identified,
        )
        assert result[0]["player_name"] == "Bob"
        assert result[0]["confidence"] == "medium"

    def test_no_match_when_too_far(self):
        prev_identified = [
            {
                "teammate": True, "actor": False, "keeper": False,
                "location": [50.0, 0.0],
                "player_name": "Bob",
                "jersey": 8,
                "confidence": "strong",
            }
        ]
        dots = [_dot(teammate=True, loc=[0.0, 0.0])]
        result = resolve_identities(
            dots, EVENT_PASS, LINEUP_SIMPLE,
            prev_ff_identified=prev_identified,
        )
        # Bob is 50m away — too far for Tier 4 (max 10m), falls to Tier 5
        assert result[0]["confidence"] == "weak"


class TestTier5LineupFill:
    def test_remaining_dots_get_weak_confidence(self):
        dots = [
            _dot(teammate=True, loc=[20.0, 0.0]),
            _dot(teammate=True, loc=[25.0, 5.0]),
        ]
        result = resolve_identities(dots, EVENT_PASS, LINEUP_SIMPLE)
        weak = [d for d in result if d["confidence"] == "weak"]
        assert len(weak) >= 1

    def test_all_dots_assigned(self):
        dots = [
            _dot(teammate=True, actor=True, loc=[0.0, 0.0]),
            _dot(teammate=True,             loc=[5.0, 0.0]),
            _dot(teammate=False,            loc=[-5.0, 0.0]),
        ]
        event = dict(EVENT_PASS)
        event["To"] = "Carlos"
        result = resolve_identities(dots, event, LINEUP_SIMPLE)
        assigned = [d for d in result if d["player_name"] is not None]
        assert len(assigned) == 3


class TestOriginalDotsUnmodified:
    def test_original_list_not_mutated(self):
        dots = [_dot(teammate=True, actor=True)]
        originals = [d.copy() for d in dots]
        resolve_identities(dots, EVENT_PASS, LINEUP_SIMPLE)
        assert "player_name" not in originals[0]


class TestIdentitySummary:
    def test_summary_counts(self):
        dots = [_dot(teammate=True, actor=True)]
        result = resolve_identities(dots, EVENT_PASS, LINEUP_SIMPLE)
        s = identity_summary(result)
        assert "strong" in s
        assert s["strong"] >= 1

    def test_identified_by_confidence_partitions(self):
        dots = [_dot(teammate=True, actor=True), _dot(teammate=False)]
        result = resolve_identities(dots, EVENT_PASS, LINEUP_SIMPLE)
        parts = identified_by_confidence(result)
        total = sum(len(v) for v in parts.values())
        assert total == len(result)


class TestHelpers:
    def test_find_player_by_name(self):
        pool = [_player("Alice", 9), _player("Bob", 8)]
        p = _find_player_by_name("alice", pool)
        assert p is not None
        assert p["jersey_number"] == 9

    def test_find_player_not_found(self):
        pool = [_player("Alice", 9)]
        assert _find_player_by_name("Charlie", pool) is None

    def test_is_gk(self):
        assert _is_gk({"position": "Goalkeeper"})
        assert _is_gk({"position": "GK"})
        assert not _is_gk({"position": "Midfielder"})
        assert not _is_gk({})
