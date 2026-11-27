#!/usr/bin/env python3
"""
epts_io.py  —  Reader for the FIFA EPTS tracking format (Game 3).

Produces (home_df, away_df, events_df) identical in structure to the
wide-format CSVs from Games 1 & 2, so the rest of the pipeline
(metrica_velocities, hex_analytics, pitch_control, app.py) works unchanged.

Tracking file format
--------------------
<frame_id>:<x1>,<y1>;<x2>,<y2>;...;<x22>,<y22>:<ball_x>,<ball_y>

All coordinates are normalised 0–1.  The 22 player columns are ordered as:
  cols  0–10  →  Team A starters  (first 11 players in XML order)
  cols 11–21  →  Team B starters  (first 11 players in XML order)

Substitutes (remaining XML entries) are NOT in the tracking stream.

Output column conventions (identical to Games 1 & 2)
------------------------------------------------------
home_df  index: Frame
         Period,  Time [s]
         Home_{shirt}_x,  Home_{shirt}_y          — metric metres, centred at (0,0)
away_df  identical structure for Away_{shirt}_*  + ball_x, ball_y
events_df columns: Type, Subtype, Period, From, To, Team,
                   Start Frame, Start Time [s], Start X, Start Y,
                   End Frame,   End Time [s],   End X,   End Y
"""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from matrix_engine.pitch_config import DEFAULT_FPS, FIELD_LENGTH, FIELD_WIDTH

FPS          = int(DEFAULT_FPS)


# ---------------------------------------------------------------------------
# Metadata parser
# ---------------------------------------------------------------------------

def parse_metadata(xml_path: str) -> dict:
    """
    Parse the EPTS XML metadata file.

    Returns a dict with keys:
        team_a_id, team_b_id
        team_a_players  : [{shirt, name, position_type, player_id}]  — XML order
        team_b_players  : same
        periods         : {1: (start_frame, end_frame), 2: ...}
        team_a_attack_h1: 'left_to_right' | 'right_to_left'
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # ---- Teams ----
    teams = {}
    for team in root.findall('.//{*}Team') or root.findall('.//Team'):
        tid   = team.get('id')
        tname = team.findtext('{*}Name') or team.findtext('Name') or tid
        teams[tid] = tname

    team_ids = list(teams.keys())
    team_a_id = team_ids[0] if team_ids else 'FIFATMA'
    team_b_id = team_ids[1] if len(team_ids) > 1 else 'FIFATMB'

    # ---- Players ----
    team_a_players: list = []
    team_b_players: list = []

    for p in root.findall('.//{*}Player') or root.findall('.//Player'):
        shirt    = p.findtext('{*}ShirtNumber') or p.findtext('ShirtNumber') or '?'
        pname    = p.findtext('{*}Name')         or p.findtext('Name')        or f'Player {shirt}'
        pid      = p.get('id', '')
        team_id  = p.get('teamId', '')
        pos_type = None

        for pp in p.findall('.//{*}ProviderParameter') or p.findall('.//ProviderParameter'):
            n = pp.findtext('{*}Name') or pp.findtext('Name')
            v = pp.findtext('{*}Value') or pp.findtext('Value')
            if n == 'position_type':
                pos_type = v

        entry = dict(shirt=shirt, name=pname, position_type=pos_type, player_id=pid)
        if team_id == team_a_id:
            team_a_players.append(entry)
        else:
            team_b_players.append(entry)

    # ---- Period timing ----
    params: dict = {}
    for pp in root.findall('.//{*}ProviderParameter') or root.findall('.//ProviderParameter'):
        n = pp.findtext('{*}Name') or pp.findtext('Name')
        v = pp.findtext('{*}Value') or pp.findtext('Value')
        if n and v:
            params[n] = v

    periods = {}
    for half, key_start, key_end in [
        (1, 'first_half_start',  'first_half_end'),
        (2, 'second_half_start', 'second_half_end'),
    ]:
        s = params.get(key_start)
        e = params.get(key_end)
        if s and e:
            try:
                periods[half] = (int(s), int(e))
            except ValueError:
                pass

    # ---- Attack direction for Team A in H1 ----
    # The XML typically has one entry per team; take the first occurrence
    attack_dirs = []
    for pp in root.findall('.//{*}ProviderParameter') or root.findall('.//ProviderParameter'):
        n = pp.findtext('{*}Name') or pp.findtext('Name')
        v = pp.findtext('{*}Value') or pp.findtext('Value')
        if n == 'attack_direction_first_half' and v:
            attack_dirs.append(v)

    team_a_attack_h1 = attack_dirs[0] if attack_dirs else 'left_to_right'

    return dict(
        team_a_id       = team_a_id,
        team_b_id       = team_b_id,
        team_a_name     = teams.get(team_a_id, 'Home'),
        team_b_name     = teams.get(team_b_id, 'Away'),
        team_a_players  = team_a_players,
        team_b_players  = team_b_players,
        periods         = periods,
        team_a_attack_h1= team_a_attack_h1,
    )


# ---------------------------------------------------------------------------
# Tracking reader
# ---------------------------------------------------------------------------

def _to_metric(arr: np.ndarray) -> np.ndarray:
    """Normalised 0-1 coordinates → metric metres centred at (0, 0)."""
    out = arr.copy().astype(float)
    out[:, 0] = (arr[:, 0] - 0.5) *  FIELD_LENGTH
    out[:, 1] = (arr[:, 1] - 0.5) * -FIELD_WIDTH   # y flip for pitch orientation
    return out


def _check_periods(periods: dict) -> None:
    """Both halves' frame bounds from the metadata, and no extra time (which this format cannot orient)."""
    missing = [p for p in (1, 2) if p not in periods]
    if missing:
        raise ValueError(f"EPTS metadata gives no frame bounds for period(s) {missing}")
    if any(int(p) > 2 for p in periods):
        raise NotImplementedError("EPTS extra time: the metadata gives the playing direction for the first half only")


def load_epts_tracking(tracking_path: str, meta: dict) -> tuple:
    """
    Parse the EPTS compact tracking file.

    Returns
    -------
    home_df : DataFrame  (index=Frame, cols: Period, Time [s], Home_{shirt}_x/y)
    away_df : DataFrame  (index=Frame, cols: Period, Time [s], Away_{shirt}_x/y,
                          ball_x, ball_y)
    """
    starters_a = meta['team_a_players'][:11]   # first 11 = cols 0–10
    starters_b = meta['team_b_players'][:11]   # first 11 = cols 11–21
    periods    = meta['periods']
    attack_h1  = meta['team_a_attack_h1']

    # Build period lookup: frame → (period, time_within_period). The bounds come from the file's metadata only (review
    # DE-06, 27 November 2026: the fallback was Sample Game 3's own bounds, wrong for any other match), and extra time
    # fails loudly: the metadata gives the direction for the first half only, so periods 3 and 4 cannot be oriented.
    _check_periods(periods)
    p1_start, p1_end = periods[1]
    p2_start, p2_end = periods[2]

    # Pre-allocate lists
    frames: list      = []
    all_positions: list = []   # shape: (N, 22, 2)
    balls: list       = []

    with open(tracking_path, 'r') as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(':')
            if len(parts) != 3:
                continue

            frame_id = int(parts[0])
            pos_strs = parts[1].split(';')
            ball_str = parts[2]

            # Parse 22 player positions
            row_pos = []
            for ps in pos_strs[:22]:
                xy = ps.split(',')
                try:
                    row_pos.append((float(xy[0]), float(xy[1])))
                except (ValueError, IndexError):
                    row_pos.append((np.nan, np.nan))

            # Pad if fewer than 22
            while len(row_pos) < 22:
                row_pos.append((np.nan, np.nan))

            # Ball
            bxy = ball_str.split(',')
            try:
                bx, by = float(bxy[0]), float(bxy[1])
            except (ValueError, IndexError):
                bx, by = np.nan, np.nan

            frames.append(frame_id)
            all_positions.append(row_pos)
            balls.append((bx, by))

    frames       = np.array(frames, dtype=np.int32)
    positions    = np.array(all_positions, dtype=np.float32)   # (N, 22, 2)
    balls_arr    = np.array(balls, dtype=np.float32)           # (N, 2)

    N = len(frames)

    # ---- Period and Time ----
    period_col = np.where(frames <= p1_end, 1, 2).astype(np.int8)
    # Time in seconds from period start, at 25fps
    p1_s = p1_start
    p2_s = p2_start
    time_col = np.where(
        period_col == 1,
        (frames - p1_s) / FPS,
        (frames - p2_s) / FPS + (p1_end - p1_s + 1) / FPS,
    )

    # ---- Play-direction normalisation ----
    # Normalise so Team A always attacks left→right (positive x).
    # For each period, flip x&y if Team A is going right→left.
    flip_h1 = (attack_h1 == 'right_to_left')
    flip_h2 = not flip_h1   # teams switch sides at half-time

    flip_mask = np.where(period_col == 1, flip_h1, flip_h2)  # (N,)

    # Apply flip to all 22 player positions and ball
    pos_metric = np.empty_like(positions, dtype=np.float32)
    for i in range(N):
        xy = positions[i]             # (22, 2) in 0-1 normalised
        if flip_mask[i]:
            xy = 1.0 - xy
        # Convert 0-1 → metric centred
        pos_metric[i, :, 0] = (xy[:, 0] - 0.5) *  FIELD_LENGTH
        pos_metric[i, :, 1] = (xy[:, 1] - 0.5) * -FIELD_WIDTH

    ball_metric = np.empty_like(balls_arr, dtype=np.float32)
    for i in range(N):
        bxy = balls_arr[i]
        if flip_mask[i]:
            bxy = 1.0 - bxy
        ball_metric[i, 0] = (bxy[0] - 0.5) *  FIELD_LENGTH
        ball_metric[i, 1] = (bxy[1] - 0.5) * -FIELD_WIDTH

    # Mark NaN in original → NaN in metric
    nan_mask = np.isnan(positions)
    pos_metric[nan_mask] = np.nan
    ball_metric[np.isnan(balls_arr)] = np.nan

    # ---- Build DataFrames ----
    base = {'Period': period_col, 'Time [s]': time_col.round(4)}

    # Home (Team A)
    home_data = base.copy()
    for idx, player in enumerate(starters_a):
        s = player['shirt']
        home_data[f'Home_{s}_x'] = pos_metric[:, idx, 0]
        home_data[f'Home_{s}_y'] = pos_metric[:, idx, 1]

    # Away (Team B)
    away_data = base.copy()
    for idx, player in enumerate(starters_b):
        s = player['shirt']
        away_data[f'Away_{s}_x'] = pos_metric[:, 11 + idx, 0]
        away_data[f'Away_{s}_y'] = pos_metric[:, 11 + idx, 1]

    away_data['ball_x'] = ball_metric[:, 0]
    away_data['ball_y'] = ball_metric[:, 1]

    home_df = pd.DataFrame(home_data, index=frames)
    away_df = pd.DataFrame(away_data, index=frames)
    home_df.index.name = 'Frame'
    away_df.index.name = 'Frame'

    return home_df, away_df


# ---------------------------------------------------------------------------
# Events reader
# ---------------------------------------------------------------------------

def load_epts_events(events_path: str, meta: dict,
                     home_df: pd.DataFrame, away_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse EPTS JSON events into the standard events DataFrame.

    Normalises event coordinates using the same flip logic applied to tracking.
    """
    with open(events_path, 'r') as fh:
        raw = json.load(fh)

    evts = raw['data'] if isinstance(raw, dict) else raw

    team_a_id  = meta['team_a_id']
    periods    = meta['periods']
    attack_h1  = meta['team_a_attack_h1']
    flip_h1    = (attack_h1 == 'right_to_left')

    _check_periods(periods)
    p1_start, p1_end = periods[1]
    p2_start = periods[2][0]

    def frame_to_time(frame, period):
        if period == 1:
            return round((frame - p1_start) / FPS, 4)
        return round((frame - p2_start) / FPS + (p1_end - p1_start + 1) / FPS, 4)

    def _norm_coord(val, period, axis):
        """Normalise a 0-1 event coordinate to metric, applying direction flip."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        flip = flip_h1 if period == 1 else (not flip_h1)
        v = 1.0 - val if flip else val
        if axis == 'x':
            return round((v - 0.5) *  FIELD_LENGTH, 4)
        return round((v - 0.5) * -FIELD_WIDTH, 4)

    rows = []
    for e in evts:
        period      = e.get('period', 1)
        team_id     = (e.get('team') or {}).get('id', '')
        team_label  = 'Home' if team_id == team_a_id else 'Away'
        etype       = (e.get('type') or {}).get('name', '')
        # subtypes can be a dict or list
        st_raw = e.get('subtypes')
        if isinstance(st_raw, dict):
            subtype = st_raw.get('name', '')
        elif isinstance(st_raw, list) and st_raw:
            subtype = ', '.join(s.get('name', '') for s in st_raw)
        else:
            subtype = ''

        from_raw    = e.get('from') or {}
        to_raw      = e.get('to')   or {}
        from_player = from_raw.get('name', '') if from_raw else ''
        to_player   = to_raw.get('name', '')   if to_raw   else np.nan

        start = e.get('start') or {}
        end   = e.get('end')   or {}

        sf = start.get('frame', np.nan)
        ef = end.get('frame',   np.nan)
        st = start.get('time',  np.nan) if start.get('time') is not None else (frame_to_time(sf, period) if not np.isnan(sf) else np.nan)
        et = end.get('time',    np.nan) if end.get('time')   is not None else (frame_to_time(ef, period) if not np.isnan(ef) else np.nan)

        sx = _norm_coord(start.get('x'), period, 'x')
        sy = _norm_coord(start.get('y'), period, 'y')
        ex = _norm_coord(end.get('x'),   period, 'x')
        ey = _norm_coord(end.get('y'),   period, 'y')

        rows.append({
            'Type':           etype,
            'Subtype':        subtype if subtype else np.nan,
            'Period':         period,
            'From':           from_player if from_player else np.nan,
            'To':             to_player,
            'Team':           team_label,
            'Start Frame':    sf,
            'Start Time [s]': st,
            'Start X':        sx,
            'Start Y':        sy,
            'End Frame':      ef,
            'End Time [s]':   et,
            'End X':          ex,
            'End Y':          ey,
        })

    df = pd.DataFrame(rows)

    # Rename columns to match Games 1 & 2 convention used in app.py
    df = df.rename(columns={
        'Start Frame':    'Start Frame',
        'Start Time [s]': 'Start Time [s]',
        'End Frame':      'End Frame',
        'End Time [s]':   'End Time [s]',
    })

    return df


# ---------------------------------------------------------------------------
# Convenience: load everything at once
# ---------------------------------------------------------------------------

def load_game3(data_dir: str) -> tuple:
    """
    Load Game 3 (EPTS format) and return (home_df, away_df, events_df, meta).

    Parameters
    ----------
    data_dir : path to .../Sample_Game_3/

    Returns
    -------
    home_df, away_df : wide-format tracking DataFrames, metric coords
    events_df        : standard events DataFrame
    meta             : raw metadata dict (for player_db population, etc.)
    """
    import os
    xml_path      = os.path.join(data_dir, 'Sample_Game_3_metadata.xml')
    tracking_path = os.path.join(data_dir, 'Sample_Game_3_tracking.txt')
    events_path   = os.path.join(data_dir, 'Sample_Game_3_events.json')

    meta                = parse_metadata(xml_path)
    home_df, away_df    = load_epts_tracking(tracking_path, meta)
    events_df           = load_epts_events(events_path, meta, home_df, away_df)

    return home_df, away_df, events_df, meta
