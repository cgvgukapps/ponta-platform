"""
skillcorner_fixture.py — Synthetic SkillCorner test data helpers.

Creates minimal in-memory fixtures (match.json dict, tracking JSONL records,
dynamic_events CSV string) that match the real SkillCorner open-data format
without requiring any downloaded files.

Usage
-----
    from skillcorner_fixture import (
        make_match_json,
        make_tracking_records,
        make_dynamic_events_csv,
        write_fixture_dir,
    )
"""

from __future__ import annotations

import json
import os

# ---------------------------------------------------------------------------
# Match IDs + player IDs used across fixtures
# ---------------------------------------------------------------------------

MATCH_ID   = 9990001
HOME_ID    = 111
AWAY_ID    = 222

HOME_GK_ID = 1001   # jersey 1
HOME_DF_ID = 1002   # jersey 5
HOME_FW_ID = 1003   # jersey 9

AWAY_GK_ID = 2001   # jersey 1  (away)
AWAY_DF_ID = 2002   # jersey 5  (away)

# Frames: period 1 = 10–30, period 2 = 50–70
P1_START = 10
P1_END   = 30
P2_START = 50
P2_END   = 70


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_match_json(
    *,
    home_attacks_right_in_h1: bool = False,
) -> dict:
    """
    Minimal match.json dict.

    Parameters
    ----------
    home_attacks_right_in_h1 : if True, h1 side = "left_to_right" (home +x);
                               if False, h1 side = "right_to_left" (home -x).
    """
    h1 = "left_to_right" if home_attacks_right_in_h1 else "right_to_left"
    h2 = "right_to_left" if home_attacks_right_in_h1 else "left_to_right"
    return {
        "id":                MATCH_ID,
        "home_team_score":   2,
        "away_team_score":   0,
        "date_time":         "2025-01-01T12:00:00Z",
        "home_team":         {"id": HOME_ID, "name": "Home FC", "short_name": "Home FC", "acronym": "HOM"},
        "away_team":         {"id": AWAY_ID, "name": "Away FC", "short_name": "Away FC", "acronym": "AWY"},
        "pitch_length":      104.0,
        "pitch_width":       68.0,
        "home_team_side":    [h1, h2],
        "ball":              {"trackable_object": 99},
        "status":            "closed",
        "match_periods": [
            {"period": 1, "name": "period_1", "start_frame": P1_START, "end_frame": P1_END,
             "duration_frames": P1_END - P1_START, "duration_minutes": 45.0},
            {"period": 2, "name": "period_2", "start_frame": P2_START, "end_frame": P2_END,
             "duration_frames": P2_END - P2_START, "duration_minutes": 45.0},
        ],
        "players": [
            # Home players
            _player(HOME_GK_ID, HOME_ID, 1,  "GK", "Goalkeeper",    is_gk=True),
            _player(HOME_DF_ID, HOME_ID, 5,  "CB", "Center Back",   is_gk=False),
            _player(HOME_FW_ID, HOME_ID, 9,  "CF", "Center Forward",is_gk=False),
            # Away players
            _player(AWAY_GK_ID, AWAY_ID, 1,  "GK", "Goalkeeper",    is_gk=True),
            _player(AWAY_DF_ID, AWAY_ID, 5,  "CB", "Center Back",   is_gk=False),
        ],
    }


def _player(pid, team_id, number, acronym, name, *, is_gk):
    return {
        "id":           pid,
        "first_name":   f"Player{pid}",
        "last_name":    "Test",
        "short_name":   f"P{number}",
        "number":       number,
        "team_id":      team_id,
        "trackable_object": pid + 10000,
        "player_role":  {"id": 0 if is_gk else 2, "acronym": acronym,
                         "name": name, "position_group": "Other" if is_gk else "Outfield"},
        "playing_time": {
            "total": {
                "start_frame": P1_START,
                "end_frame":   P2_END,
                "minutes_played": 90.0,
            },
            "by_period": [],
        },
    }


def make_tracking_records(
    *,
    n_p1_frames: int = 5,
    n_p2_frames: int = 5,
    gk_x_p1: float = -40.0,   # raw GK x in period 1 (negative = home attacks +x)
    home_df_x: float = -20.0,
    away_df_x: float = 20.0,
    ball_x: float = 0.5,
    ball_y: float = 0.3,
) -> list[dict]:
    """
    Minimal synthetic tracking records (JSONL list of dicts).

    GK position in period 1 encodes the expected playing direction:
      gk_x_p1 < 0  →  home GK on left   → home attacks +x  (h1_sign = +1)
      gk_x_p1 > 0  →  home GK on right  → home attacks -x  (h1_sign = -1)
    """
    records: list[dict] = []

    def _frame(frame_id, period, t, gk_x_sign=1):
        return {
            "frame":     frame_id,
            "timestamp": t,
            "period":    period,
            "ball_data": {"x": ball_x, "y": ball_y, "z": 0.0},
            "player_data": [
                {"player_id": HOME_GK_ID, "x": gk_x_p1 * gk_x_sign, "y": 0.0,      "is_detected": True},
                {"player_id": HOME_DF_ID, "x": home_df_x,            "y": -5.0,     "is_detected": True},
                {"player_id": HOME_FW_ID, "x": 30.0,                 "y": 2.0,      "is_detected": True},
                {"player_id": AWAY_GK_ID, "x": away_df_x + 20.0,    "y": 0.0,      "is_detected": True},
                {"player_id": AWAY_DF_ID, "x": away_df_x,            "y": 5.0,      "is_detected": True},
            ],
            "possession": {"player_id": HOME_FW_ID, "group": "home"},
        }

    for i in range(n_p1_frames):
        fid = P1_START + i
        t   = i / 10.0
        records.append(_frame(fid, 1, t, gk_x_sign=1))

    for i in range(n_p2_frames):
        fid = P2_START + i
        t   = 45.0 + i / 10.0
        records.append(_frame(fid, 2, t, gk_x_sign=-1))  # flipped in P2

    return records


def make_dynamic_events_csv(
    *,
    match_id: int = MATCH_ID,
    home_team_id: int = HOME_ID,
    away_team_id: int = AWAY_ID,
) -> str:
    """
    Minimal dynamic_events CSV string with:
    - one successful pass (player_possession, end_type=pass, outcome=successful)
    - one failed pass (player_possession, end_type=pass, outcome=unsuccessful)
    - one carry (player_possession, carry=True, end_type=none)
    - one passing_option (should be skipped)
    """
    header = (
        "event_id,match_id,frame_start,frame_end,period,"
        "event_type,player_id,team_id,"
        "x_start,y_start,x_end,y_end,"
        "carry,end_type,start_type,pass_outcome,"
        "player_targeted_id,player_targeted_x_reception,player_targeted_y_reception,"
        "fully_extrapolated"
    )
    rows = [
        # Successful pass from home player 9 to away player 5
        f"ev_1,{match_id},{P1_START},{P1_START+2},1,"
        f"player_possession,{HOME_FW_ID},{home_team_id},"
        f"30.0,2.0,45.0,3.0,"
        f"False,pass,unknown,successful,"
        f"{AWAY_DF_ID},44.0,4.0,False",

        # Failed pass from home player 5
        f"ev_2,{match_id},{P1_START+3},{P1_START+4},1,"
        f"player_possession,{HOME_DF_ID},{home_team_id},"
        f"-20.0,-5.0,-10.0,0.0,"
        f"False,pass,unknown,unsuccessful,"
        f",,False",

        # Carry by away player 5 (end_type = '' means no terminal pass/shot)
        f"ev_3,{match_id},{P2_START},{P2_START+5},2,"
        f"player_possession,{AWAY_DF_ID},{away_team_id},"
        f"20.0,5.0,10.0,3.0,"
        f"True,,unknown,,"
        f",,False",

        # passing_option — should be skipped
        f"ev_4,{match_id},{P1_START+1},{P1_START+2},1,"
        f"passing_option,{AWAY_GK_ID},{away_team_id},"
        f"40.0,0.0,40.0,0.0,"
        f"False,,,,,"
        f",,False",
    ]
    return "\n".join([header] + rows)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def write_fixture_dir(
    tmp_dir: str,
    match_id: int = MATCH_ID,
    *,
    home_attacks_right_in_h1: bool = False,
    tracking_records: list[dict] | None = None,
) -> str:
    """
    Write all three SkillCorner fixture files to ``tmp_dir`` and return the path.

    The tracking JSONL is written as a real JSONL file (one JSON object per line).
    """
    if tracking_records is None:
        tracking_records = make_tracking_records()

    match_dir = os.path.join(tmp_dir, str(match_id))
    os.makedirs(match_dir, exist_ok=True)

    # match.json
    with open(os.path.join(match_dir, f"{match_id}_match.json"), "w") as f:
        json.dump(make_match_json(home_attacks_right_in_h1=home_attacks_right_in_h1), f)

    # tracking JSONL
    with open(os.path.join(match_dir, f"{match_id}_tracking_extrapolated.jsonl"), "w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in tracking_records)

    # dynamic_events CSV
    with open(os.path.join(match_dir, f"{match_id}_dynamic_events.csv"), "w") as f:
        f.write(make_dynamic_events_csv(
            match_id=match_id,
            home_team_id=HOME_ID,
            away_team_id=AWAY_ID,
        ))

    return match_dir
