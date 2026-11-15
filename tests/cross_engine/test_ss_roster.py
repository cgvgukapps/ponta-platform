"""Second Spectrum roster is built from persistent player_id, not slot inference."""

import pytest
from matrix_engine.ingestion.second_spectrum_io import load_secondspectrum_game
from plaiq_engine.roster.build_roster import build_secondspectrum_roster
from ss_fixture import write_ss_fixture


@pytest.fixture
def ss_loaded(tmp_path):
    write_ss_fixture(str(tmp_path), pitch_length=105.0, pitch_width=68.0, fps=25)
    return load_secondspectrum_game(str(tmp_path))


def test_roster_uses_persistent_player_ids(ss_loaded):
    home, away, _events, meta = ss_loaded
    roster = build_secondspectrum_roster(7, meta, home, away)

    assert roster["game_id"] == 7
    assert roster["provider"] == "secondspectrum"
    assert roster["teams"]["Home"]["team_name"] == "HOMEFC"

    home_players = roster["teams"]["Home"]["players"]
    assert len(home_players) == 4

    by_id = {p["player_id"]: p for p in home_players}
    # Persistent Second Spectrum ids (optaId), not "Player N" slot labels.
    assert set(by_id) == {"h1", "h2", "h3", "h4"}

    gk = by_id["h1"]
    assert gk["primary_slot"] == 1
    assert gk["name_source"] == "secondspectrum"
    # Single, unambiguous spell spanning both halves (no slot reuse).
    assert len(gk["spells"]) == 1
    spell = gk["spells"][0]
    assert spell["slot"] == 1
    assert spell["slot_agreement"] == 1.0
    assert spell["period_span_start"] == 1
    assert spell["period_span_end"] == 2
    assert spell["end_s"] > spell["start_s"]


def test_roster_position_from_metadata(ss_loaded):
    home, away, _events, meta = ss_loaded
    roster = build_secondspectrum_roster(7, meta, home, away)
    gk = next(p for p in roster["teams"]["Home"]["players"] if p["primary_slot"] == 1)
    # Kloppy maps the SS "GK" position string onto its goalkeeper PositionType.
    assert "GOALKEEPER" in gk["position"].upper() or gk["position"] == "GK"
