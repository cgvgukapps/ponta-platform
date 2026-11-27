#!/usr/bin/env python3
"""
metrica_io.py

Reading and preprocessing Metrica Sports sample tracking and event data.
Applies coordinate conversion, playing direction normalisation, and column renaming.

Data source: https://github.com/metrica-sports/sample-data
"""

import csv

import numpy as np
import pandas as pd

from matrix_engine.pitch_config import (
    FIELD_DIMENSIONS,  # metres (length, width); Metrica default 106 x 68
)

MAX_SPEED = 12.0  # m/s — above this is treated as a tracking error


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_match_data(data_dir: str, game_id: int):
    """
    Read all data for a game: home tracking, away tracking, and events.
    Returns (tracking_home, tracking_away, events) as DataFrames.
    """
    tracking_home = read_tracking_data(data_dir, game_id, "Home")
    tracking_away = read_tracking_data(data_dir, game_id, "Away")
    events = read_event_data(data_dir, game_id)
    return tracking_home, tracking_away, events


def read_event_data(data_dir: str, game_id: int) -> pd.DataFrame:
    """Read event CSV and return as a DataFrame."""
    path = f"{data_dir}/Sample_Game_{game_id}/Sample_Game_{game_id}_RawEventsData.csv"
    events = pd.read_csv(path)
    return events


def read_tracking_data(data_dir: str, game_id: int, team: str) -> pd.DataFrame:
    """
    Read a tracking CSV for the given team ('Home' or 'Away').

    The raw CSV has a 3-row header:
      Row 1: team name repeated per player
      Row 2: jersey numbers
      Row 3: column names (Period, Frame, Time [s], then paired x/y per player)

    Output column names follow the pattern: <Team>_<jersey>_x / <Team>_<jersey>_y
    e.g. Home_1_x, Home_1_y, Away_15_x ...
    Ball columns are ball_x and ball_y.
    Index is set to Frame.
    """
    path = f"{data_dir}/Sample_Game_{game_id}/Sample_Game_{game_id}_RawTrackingData_{team}_Team.csv"

    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # row 1: team names — skip
        jerseys = [x for x in next(reader) if x != ""]  # row 2: jersey numbers
        columns = next(reader)  # row 3: base column names

    # Build clean column names for each player pair
    for i, jersey in enumerate(jerseys):
        columns[i * 2 + 3] = f"{team}_{jersey}_x"
        columns[i * 2 + 4] = f"{team}_{jersey}_y"

    columns[-2] = "ball_x"
    columns[-1] = "ball_y"

    tracking = pd.read_csv(path, names=columns, index_col="Frame", skiprows=3)
    return tracking


def merge_tracking_data(home: pd.DataFrame, away: pd.DataFrame) -> pd.DataFrame:
    """Merge home and away tracking DataFrames on Frame index (drops duplicate ball columns)."""
    return home.drop(columns=["ball_x", "ball_y"]).merge(away, left_index=True, right_index=True)


# ---------------------------------------------------------------------------
# Coordinate transformations
# ---------------------------------------------------------------------------

def to_metric_coordinates(data: pd.DataFrame, field_dimen=FIELD_DIMENSIONS) -> pd.DataFrame:
    """
    Convert positions from Metrica normalised units (0–1, origin top-left)
    to metres with the origin at the centre circle.

    x: left edge = -53m, right edge = +53m
    y: top edge  = +34m, bottom edge = -34m  (y-axis flipped vs raw data)
    """
    x_cols = [c for c in data.columns if c.endswith("_x")]
    y_cols = [c for c in data.columns if c.endswith("_y")]

    data[x_cols] = (data[x_cols] - 0.5) * field_dimen[0]
    # Multiply by -1 to flip y: Metrica origin is top-left, not bottom-left
    data[y_cols] = -1 * (data[y_cols] - 0.5) * field_dimen[1]
    return data


def to_single_playing_direction(
    home: pd.DataFrame, away: pd.DataFrame, events: pd.DataFrame
) -> tuple:
    """
    Flip all coordinates in the second half so that each team always attacks
    in the same direction across both halves.

    Must be called AFTER to_metric_coordinates.

    Each table is split by its own ``Period`` column. *Fixed 19 November 2026:* the
    prototype sliced every table from the first second-half *tracking frame label*
    (about 70,000), which on the events table (indexed by event row) selected
    nothing, so second-half event coordinates were never flipped and sat about 55 m
    from the tracked ball (``ponta-platform/PLAIQ_REFERENCES_REPORT.md``). Tracking
    output is unchanged: its rows from the first second-half frame on are exactly
    those with ``Period >= 2``.

    *Extra time (review DE-06, 27 November 2026):* teams change ends between the halves of extra time too, so periods 3
    and 4 are not flipped alike. Each is flipped when the home goalkeeper defends the other end from period 1, read from
    the data (Metrica's files give no per-period direction); when it cannot be read, the load fails rather than guess.
    Periods 1 and 2 are unchanged.
    """
    flip_periods = {2}
    if "Period" in home.columns:
        later = sorted(int(p) for p in pd.unique(home["Period"].dropna()) if int(p) >= 3)
        if later:
            first = _home_keeper_end(home, 1)
            for p in later:
                end = _home_keeper_end(home, p)
                if first is None or end is None:
                    raise ValueError(f"cannot tell which way the home team attacks in period {p}: no goalkeeper found")
                if end != first:
                    flip_periods.add(p)
    for df in [home, away, events]:
        coord_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
        # Events use different column names
        if "Start X" in df.columns:
            coord_cols = ["Start X", "Start Y", "End X", "End Y"]
        if "Period" not in df.columns or not coord_cols:
            continue
        df.loc[df["Period"].isin(flip_periods), coord_cols] *= -1

    return home, away, events


def _home_keeper_end(home: pd.DataFrame, period: int, min_presence: float = 0.5) -> int | None:
    """+1 or -1: the end (sign of x) the home goalkeeper stands at in ``period``, before any flip; None when unclear."""
    part = home.loc[home["Period"] == period]
    x_cols = [c for c in part.columns if c.startswith("Home_") and c.endswith("_x")]
    if part.empty or not x_cols:
        return None
    seen = part[x_cols].notna().mean()
    means = part[[c for c in x_cols if seen[c] >= min_presence]].mean()
    if means.empty or not np.isfinite(means.abs().max()) or means.abs().max() < 20.0:
        return None
    return 1 if means[means.abs().idxmax()] > 0 else -1


def home_attack_sign(home: pd.DataFrame, min_presence: float = 0.5) -> int:
    """
    Which way the home team attacks in the first half: +1 towards +x, -1 towards -x.

    Read from the home goalkeeper, taken as the player with the largest absolute
    mean x over the first half among players seen in at least ``min_presence`` of
    its frames. A goalkeeper near +x defends that end, so the team attacks -x.
    More robust than the single kick-off frame that ``find_playing_direction`` reads.
    """
    first = home.loc[home.Period == 1] if "Period" in home.columns else home
    x_cols = [c for c in first.columns if c.startswith("Home_") and c.endswith("_x")]
    seen = first[x_cols].notna().mean()
    means = first[[c for c in x_cols if seen[c] >= min_presence]].mean()
    if means.empty or not np.isfinite(means.abs().max()):
        return 1
    return -1 if means[means.abs().idxmax()] > 0 else 1


def orient_home_attacking_positive(
    home: pd.DataFrame, away: pd.DataFrame, events: pd.DataFrame
) -> tuple:
    """
    Rotate the whole match by 180 degrees if the home team attacks -x.

    ``to_single_playing_direction`` only makes each team attack the same way in
    both halves; it assumes the home team starts attacking +x. Metrica Sample
    Game 2 starts the other way, so without this step its home team attacks -x
    for the whole match, against Matrix's convention. Call it AFTER
    ``to_single_playing_direction``. A no-op when the home team already attacks +x.
    """
    if home_attack_sign(home) == 1:
        return home, away, events
    for df in [home, away, events]:
        coord_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
        if "Start X" in df.columns:
            coord_cols = ["Start X", "Start Y", "End X", "End Y"]
        df.loc[:, coord_cols] *= -1
    return home, away, events


def find_goalkeeper(team: pd.DataFrame, team_name: str) -> str:
    """
    Identify the goalkeeper as the player furthest from the centre circle at kick-off.
    Returns the jersey number string.
    """
    x_cols = [c for c in team.columns if c.endswith("_x") and c.startswith(team_name)]
    gk_col = team.iloc[0][x_cols].abs().idxmax()
    return gk_col.split("_")[1]


def find_playing_direction(team: pd.DataFrame, team_name: str) -> int:
    """
    Determine playing direction at kick-off based on goalkeeper position.
    Returns +1 (left→right) or -1 (right→left).
    """
    gk_jersey = find_goalkeeper(team, team_name)
    gk_x_col = f"{team_name}_{gk_jersey}_x"
    return int(-np.sign(team.iloc[0][gk_x_col]))
