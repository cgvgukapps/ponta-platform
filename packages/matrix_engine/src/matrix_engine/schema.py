#!/usr/bin/env python3
"""
schema.py — the common normalized tracking schema every Matrix source speaks.

Matrix reconciles tracking data from structurally different vendors (optical,
wearable, broadcast). Every provider adapter in ``matrix_engine/providers/``
converts its vendor's native layout into the single long-format schema defined
here, and every stage downstream (sync, identity, fusion) both consumes and
produces it. Validating at each boundary is what makes "vendor-agnostic"
mean something concrete rather than aspirational.

Long format, one row per observed player per frame:

    timestamp  frame  player_id  team  x      y     source     sample_rate_hz
    0.04       1      home_11    home  -12.3  4.1   metrica    25.0

Design decisions worth stating explicitly:

Rows are observations, not a dense grid.
  If a source did not see a player at a frame, there is no row. This makes
  frame dropout and partial vendor coverage (SkillCorner broadcast tracking
  only sees ~8 of 22 players per frame) representable without NaN sentinels,
  and it is why ``x``/``y`` are required to be finite.

Coordinates are metres, centred at (0, 0), home attacking +x.
  This is the convention already established in ``pitch_config.py`` and used
  throughout the existing Plaiq engine. Providers are responsible for getting
  their vendor into it — that conversion is the point of the adapter.

``player_id`` is a string in the source's own label space.
  Matrix does not assume two sources agree on player identifiers — reconciling
  them is ``identity.py``'s job. Provider adapters namespace their IDs
  (e.g. ``home_11``) so that a merged multi-source frame stays unambiguous.

Optional columns
----------------

``native_quality`` — a source's own opinion of a sample, in [0, 1].
  Some vendors ship per-sample diagnostics alongside the position: a GPS unit
  knows its satellite geometry and reported horizontal accuracy in a way no
  downstream consumer can reconstruct. Where a provider can turn those into a
  single 0–1 score it does, and ``confidence.py`` folds it into the trust score.

  Only ``providers/statsports.py`` populates this today, because it is the only
  adapter here reading a feed that carries per-sample diagnostics at all —
  Metrica's and SkillCorner's sample exports ship positions without them. The
  column is therefore optional rather than required, and absent is not a defect.

  When it is absent, confidence falls back to exactly what it used before:
  multi-source coverage, gap distance and inter-source agreement. That fallback
  is weaker in a specific way worth naming — those three signals can only infer
  trouble from *disagreement* or *dropout*, so a single source quietly degrading
  while still reporting positions is invisible to them. ``native_quality`` is
  the only signal in the schema that can catch that case, which is why it is
  carried through the pipeline rather than dropped at the adapter boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

#: Required columns, in canonical order.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "frame",
    "player_id",
    "team",
    "x",
    "y",
    "source",
    "sample_rate_hz",
)

#: Column -> accepted pandas dtype kind(s). ``kind`` is used rather than an
#: exact dtype so that int32/int64 and float32/float64 both pass.
_DTYPE_KINDS: dict[str, tuple[str, ...]] = {
    "timestamp": ("f",),
    "frame": ("i", "u"),
    "player_id": ("O", "U", "S"),
    "team": ("O", "U", "S"),
    "x": ("f",),
    "y": ("f",),
    "source": ("O", "U", "S"),
    "sample_rate_hz": ("f",),
}

#: Optional columns a provider may add. Absent is valid; present must be
#: well-formed. Kept separate from :data:`REQUIRED_COLUMNS` so that
#: ``coerce_schema`` still orders the required set first.
OPTIONAL_COLUMNS: tuple[str, ...] = ("native_quality",)

#: Accepted dtype kinds for the optional columns, checked only when present.
_OPTIONAL_DTYPE_KINDS: dict[str, tuple[str, ...]] = {
    "native_quality": ("f",),
}

#: Inclusive bounds on ``native_quality``. A vendor diagnostic is normalised to
#: this range by its provider so that the confidence model does not have to know
#: whose scale it is reading.
NATIVE_QUALITY_BOUNDS: tuple[float, float] = (0.0, 1.0)

#: Team label for a source that genuinely cannot say which side a player is on.
#: A single-club wearable export is the motivating case: it contains one team's
#: players and no opponent, so ``home``/``away`` would be a guess dressed up as
#: data. Providers use this instead of inventing a split.
UNKNOWN_TEAM: str = "unknown"

#: Valid values for the ``team`` column.
VALID_TEAMS: frozenset[str] = frozenset({"home", "away", UNKNOWN_TEAM})

#: How far outside the nominal pitch rectangle a coordinate may fall before it
#: is treated as a unit/coordinate-system error rather than a player stepping
#: over the touchline. Real tracking routinely puts players a few metres off
#: the pitch (throw-ins, keepers behind the goal line, benches).
PITCH_BOUNDS_MARGIN_M: float = 10.0


class SchemaError(ValueError):
    """Raised when a DataFrame does not conform to the Matrix tracking schema."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame, *, name: str = "tracking frame") -> pd.DataFrame:
    """
    Check ``df`` against the Matrix common schema, raising on any violation.

    Returns the DataFrame unchanged so the call can be inlined at a boundary::

        return validate_schema(build_frame(...), name="metrica")

    Raises
    ------
    SchemaError
        With a message naming the offending column and the specific problem.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaError(f"{name}: expected a pandas DataFrame, got {type(df).__name__}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{name}: missing required column(s) {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}"
        )

    for col, kinds in _DTYPE_KINDS.items():
        kind = df[col].dtype.kind
        if kind not in kinds:
            raise SchemaError(
                f"{name}: column '{col}' has dtype {df[col].dtype} (kind '{kind}'), "
                f"expected one of kinds {list(kinds)}"
            )

    for col, kinds in _OPTIONAL_DTYPE_KINDS.items():
        if col not in df.columns:
            continue
        kind = df[col].dtype.kind
        if kind not in kinds:
            raise SchemaError(
                f"{name}: optional column '{col}' has dtype {df[col].dtype} "
                f"(kind '{kind}'), expected one of kinds {list(kinds)}"
            )

    if df.empty:
        # An empty frame with the right columns/dtypes is structurally valid;
        # the value checks below have nothing to inspect.
        return df

    for col in ("timestamp", "x", "y", "sample_rate_hz"):
        values = df[col].to_numpy(dtype=float, copy=False)
        if not np.isfinite(values).all():
            n_bad = int((~np.isfinite(values)).sum())
            raise SchemaError(
                f"{name}: column '{col}' contains {n_bad} non-finite value(s). "
                "Rows represent actual observations — omit the row instead of "
                "writing NaN for a player a source did not see."
            )

    bad_teams = sorted(set(df["team"].unique()) - VALID_TEAMS)
    if bad_teams:
        raise SchemaError(
            f"{name}: column 'team' contains invalid value(s) {bad_teams}. "
            f"Expected only {sorted(VALID_TEAMS)}"
        )

    if (df["sample_rate_hz"] <= 0).any():
        raise SchemaError(f"{name}: column 'sample_rate_hz' must be strictly positive")

    if df["player_id"].eq("").any():
        raise SchemaError(f"{name}: column 'player_id' contains empty string(s)")

    if df["source"].eq("").any():
        raise SchemaError(f"{name}: column 'source' contains empty string(s)")

    if "native_quality" in df.columns:
        # NaN is permitted here, unlike in the required columns: it means "no
        # vendor opinion for this row", which is a real state once sources with
        # and without diagnostics are fused together. Only the values that do
        # exist have to be on scale.
        quality = df["native_quality"].to_numpy(dtype=float, copy=False)
        known = quality[~np.isnan(quality)]
        low, high = NATIVE_QUALITY_BOUNDS
        if known.size and (known.min() < low or known.max() > high):
            raise SchemaError(
                f"{name}: column 'native_quality' has values outside "
                f"[{low}, {high}] (observed {known.min():.3f}..{known.max():.3f}). "
                "Providers normalise their vendor's diagnostic onto this scale."
            )
        if known.size and not np.isfinite(known).all():
            raise SchemaError(
                f"{name}: column 'native_quality' contains infinite value(s); "
                "use NaN for 'unknown'"
            )

    duplicated = df.duplicated(subset=["source", "player_id", "frame"])
    if duplicated.any():
        n_dup = int(duplicated.sum())
        raise SchemaError(
            f"{name}: {n_dup} duplicate (source, player_id, frame) row(s) — "
            "each source may report a player at most once per frame"
        )

    return df


def validate_pitch_bounds(
    df: pd.DataFrame,
    pitch,
    *,
    margin_m: float = PITCH_BOUNDS_MARGIN_M,
    name: str = "tracking frame",
) -> pd.DataFrame:
    """
    Check that coordinates plausibly live in ``pitch``'s metric coordinate system.

    This is a *units* check, not a containment check — it catches a provider
    that emitted normalised 0–1 coordinates, yards, or an uncentred origin,
    while tolerating players legitimately a few metres off the pitch.

    Parameters
    ----------
    pitch : pitch_config.PitchConfig
        Supplies ``half_length`` / ``half_width``.
    margin_m : float
        Slack allowed beyond the pitch rectangle on every side.
    """
    if df.empty:
        return df

    x_limit = pitch.half_length + margin_m
    y_limit = pitch.half_width + margin_m

    x = df["x"].to_numpy(dtype=float, copy=False)
    y = df["y"].to_numpy(dtype=float, copy=False)

    if np.abs(x).max() > x_limit:
        raise SchemaError(
            f"{name}: max |x| = {np.abs(x).max():.2f} m exceeds "
            f"{x_limit:.2f} m (half-length {pitch.half_length:.1f} m + "
            f"{margin_m:.1f} m margin). Coordinates are probably not in the "
            "expected metric, centre-origin system."
        )
    if np.abs(y).max() > y_limit:
        raise SchemaError(
            f"{name}: max |y| = {np.abs(y).max():.2f} m exceeds "
            f"{y_limit:.2f} m (half-width {pitch.half_width:.1f} m + "
            f"{margin_m:.1f} m margin). Coordinates are probably not in the "
            "expected metric, centre-origin system."
        )
    return df


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def empty_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the correct columns and dtypes."""
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="float64"),
            "frame": pd.Series(dtype="int64"),
            "player_id": pd.Series(dtype="object"),
            "team": pd.Series(dtype="object"),
            "x": pd.Series(dtype="float64"),
            "y": pd.Series(dtype="float64"),
            "source": pd.Series(dtype="object"),
            "sample_rate_hz": pd.Series(dtype="float64"),
        }
    )


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast a DataFrame that already has the required columns into canonical
    dtypes and column order, dropping rows with non-finite coordinates.

    Provider adapters use this as the last step before ``validate_schema`` so
    that each one does not hand-roll the same set of casts. Columns beyond the
    required set (e.g. a provider's native velocities) are preserved, after
    the required ones.
    """
    extras = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    out = df.loc[:, list(REQUIRED_COLUMNS) + extras].copy()

    out["timestamp"] = out["timestamp"].astype("float64")
    out["x"] = out["x"].astype("float64")
    out["y"] = out["y"].astype("float64")
    out["sample_rate_hz"] = out["sample_rate_hz"].astype("float64")
    out["player_id"] = out["player_id"].astype("object")
    out["team"] = out["team"].astype("object")
    out["source"] = out["source"].astype("object")
    if "native_quality" in out.columns:
        out["native_quality"] = out["native_quality"].astype("float64")

    finite = (
        np.isfinite(out["timestamp"].to_numpy())
        & np.isfinite(out["x"].to_numpy())
        & np.isfinite(out["y"].to_numpy())
    )
    out = out.loc[finite]
    out["frame"] = out["frame"].astype("int64")

    return out.reset_index(drop=True)
