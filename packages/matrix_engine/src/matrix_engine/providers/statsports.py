#!/usr/bin/env python3
"""
statsports.py — adapter for a real STATSports Apex GPS/IMU raw export.

Every other source in this prototype is optical. This adapter exists so that
Matrix's "vendor-agnostic" claim is tested against a genuinely different class
of sensor — a wearable that measures position by satellite, samples at its own
rate, carries no team labels, and ships per-sample diagnostics about its own
signal quality. Those diagnostics are the interesting part: they are the only
input in the whole engine that lets a *single* source report that it is having
trouble, which ``confidence.py`` otherwise can only infer from disagreement
between sources or from outright dropout.

Data handling — read this before using the module
-------------------------------------------------

The default source is a raw URL into a public student capstone repository. It is
real, name-attributed player data with no stated license and no evidence of
consent for third-party reuse, so this module is built to touch it as lightly as
possible:

* **Fetched on access, never vendored.** There is no "download this first" step
  and nothing is written into the repository tree.
* **No disk copy unless asked for.** ``cache_dir`` is ``None`` by default, so a
  plain call streams the file into memory and keeps nothing. Passing a cache
  directory is a convenience for repeated local runs; :func:`default_cache_dir`
  points at the OS temp directory and :func:`purge_cache` deletes it. Anything
  it writes is disposable and safe to remove at any time.
* **Anonymised by construction.** ``Player Display Name`` is replaced with a
  salted hash before any other transformation runs, and it is dropped from the
  frame in the same step, so no caller can reach it. ``anonymize=False`` is
  rejected rather than honoured.
* **Unavailability is normal.** The source can vanish — and arguably should.
  Anything that stops the fetch raises :class:`StatsportsUnavailable`, which
  callers (and this repo's tests) treat as "skip", never as a failure.

What this export actually contains, as measured
-----------------------------------------------

Design choices below are driven by inspection of the real file rather than by
the vendor's documentation, which is not public. From 325,036 rows / 18 players:

**Native rate is 3.333 Hz** (0.3 s between samples), not the 10 Hz an Apex unit
samples at internally — the export has already been decimated. It is measured
per player rather than assumed, which is the point of :func:`load_statsports_source`
reporting ``native_rate_hz`` instead of hard-coding one.

**One ~92-minute session**, all 18 players on a common wall clock (18:08–19:40).
Sustained stretches with a median speed near zero and the squad clustered in a
few tens of metres mean this is not 92 minutes of continuous match play; it
looks like a session including warm-up and stoppages. Nothing here verifies
which parts are match play.

**52 rows carry ``Lat``/``Lon`` of exactly 0** — null-island sentinels, dropped
as missing rather than projected into the Atlantic.

**``No. of Satellites`` and ``Quality of Signal`` are 0 on 15% of rows, together,
and those rows are not degraded.** This was the surprise. The obvious reading is
"no satellite fix, position is dead-reckoned", so it was checked: those rows show
median second-difference jitter of 0.091 m against 0.086 m elsewhere, an
identical rate of frozen positions, and — decisively — no position jump when the
field starts reporting again (0.266 m, versus a 0.261 m baseline step). A genuine
fix loss followed by reacquisition snaps back. This does not. The fields are
simply unpopulated on those samples, so the satellite term is treated as *absent*
there rather than as evidence of a problem. Penalising them would have made
``native_quality`` mostly a detector of missing telemetry.

**``Quality of Signal`` carries no measurable information** about position error
in this export. Non-zero values run 173–342 on an undocumented scale, and its
Spearman correlation against three independent error proxies is |rho| <= 0.035.
It is parsed and passed through under ``with_raw_quality`` but deliberately
excluded from the score: including it drowned the fields that do work, which is
how the first version of this module ended up with a quality score that had no
relationship to data quality at all.

**``Hacc`` is the field that works**, despite not being in the column list this
adapter was originally specified against. It is 0 on 99.2% of rows and 1–7 on
the rest, and those rows show 2.2x the median jitter and 3.8x the median
disagreement between the vendor's own reported speed and the speed implied by its
own positions. ``Hdop`` is a useful second signal (>= 0.9 gives 3.6x the median
jitter) but has almost no spread: 90% of rows sit at exactly 0.40.

See ``matrix_engine/STATSPORTS_ADDENDUM.md`` for what all of that does and does
not prove.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from matrix_engine.pitch_config import PitchConfig
from matrix_engine.schema import (
    UNKNOWN_TEAM,
    coerce_schema,
    validate_schema,
)

#: Raw export this adapter was written against. Fetched on access; see the
#: module docstring for the handling rules that come with it.
DEFAULT_STATSPORTS_SOURCE = (
    "https://raw.githubusercontent.com/bakero08/Statsports-data-analytics-UCD/main/maindata.csv"
)

#: ``source`` value this adapter stamps on every row.
SOURCE_NAME = "statsports"

#: Environment overrides for ``source``, checked in this order. A local file
#: wins over an alternate URL, and either wins over the default.
ENV_SAMPLE_PATH = "MATRIX_STATSPORTS_SAMPLE_PATH"
ENV_SAMPLE_URL = "MATRIX_STATSPORTS_SAMPLE_URL"

#: Seconds to wait on the network before giving up and reporting unavailable.
FETCH_TIMEOUT_S: float = 120.0

# -- raw column names -------------------------------------------------------
# The real header also contains a typo ("Gyro Yro X" where "Gyro X" is meant).
# It is not read here, but it is a reminder that this adapter targets one
# specific export rather than a documented vendor format.

COL_NAME = "Player Display Name"
COL_TIME = "Time"
COL_LAT = "Lat"
COL_LON = "Lon"
COL_SPEED = "Speed (m/s)"
COL_HACC = "Hacc"
COL_HDOP = "Hdop"
COL_QOS = "Quality of Signal"
COL_SATELLITES = "No. of Satellites"

#: Columns without which the file is not this export at all.
_ESSENTIAL_COLUMNS = (COL_NAME, COL_TIME, COL_LAT, COL_LON)

# -- native_quality model ---------------------------------------------------
# Weights and breakpoints below were set from the measurements in the module
# docstring, and every one of them is a keyword argument on
# :func:`compute_native_quality` so another export can be re-fitted without
# editing this file.

#: ``Hacc`` at which the accuracy term reaches 0. Observed values are 0–7.
HACC_WORST_M: float = 5.0

#: ``Hdop`` band over which the dilution term falls from 1 to 0. 0.8 is the
#: 99th percentile of the observed distribution; 0.9 is where jitter jumps.
HDOP_GOOD: float = 0.8
HDOP_WORST: float = 2.0

#: Satellite count band. Observed non-zero counts are 14–22, so this term is
#: near-saturated in practice and is weighted accordingly.
SATELLITES_MIN: float = 6.0
SATELLITES_GOOD: float = 18.0

#: Relative weight of each term, renormalised over whichever are populated.
#: ``Hacc`` dominates because it is the only one with a large measured effect.
QUALITY_WEIGHT_HACC: float = 0.6
QUALITY_WEIGHT_HDOP: float = 0.3
QUALITY_WEIGHT_SATELLITES: float = 0.1

# -- pitch geometry plausibility -------------------------------------------

#: Laws-of-the-game range for a full-size pitch, used only as a sanity check on
#: the recovered span. Being inside it does not validate the transform.
PLAUSIBLE_LENGTH_M: tuple[float, float] = (90.0, 120.0)
PLAUSIBLE_WIDTH_M: tuple[float, float] = (45.0, 90.0)

#: Grid the ``Time`` field resolves to (0.1 s), used to number frames. The
#: native 3.33 Hz sampling lands on every third grid point, so frame numbers
#: are strictly increasing but not consecutive.
FRAME_GRID_HZ: float = 10.0

#: Salt for player-ID hashing on the reference sample: fixed rather than random so IDs are stable across runs and
#: processes, which identity matching needs. It is public, so it protects nothing: the product passes a per-tenant
#: ``id_salt`` (review DE-15, 27 November 2026), and this default is refused when ``PONTA_ENV`` is ``production``.
_ID_SALT = "matrix_engine.providers.statsports"
#: Shortest per-tenant salt accepted.
MIN_SALT_CHARS = 16

#: Length of the hash prefix used as an ID. 10 hex characters is 40 bits — far
#: beyond collision range for a squad, and short enough to read in a test.
_ID_LENGTH = 10


class StatsportsUnavailable(RuntimeError):
    """
    The export could not be read, for any reason.

    Covers no override configured, no network, a timeout, a 404 because the
    source repository was taken down, and a file that parsed but is not this
    export. Callers are expected to treat all of those the same way: skip the
    work rather than fail. Depending on a live URL into someone else's
    unlicensed repository means unavailability is an ordinary outcome, not an
    error condition.
    """


@dataclass(frozen=True)
class PitchEstimate:
    """
    Recovered pitch frame — an *estimate*, with no survey behind it.

    Attributes
    ----------
    origin_lat, origin_lon : float
        Centre of the equirectangular projection: the mean of the session's
        own fixes.
    rotation_deg : float
        Angle of the recovered long axis, anticlockwise from east. The output
        x-axis is aligned to it.
    length_m, width_m : float
        Full extent of the point cloud along the recovered long and short axes.
    trimmed_length_m, trimmed_width_m : float
        The same after discarding the 5% of samples furthest from the centre.
        A diagnostic for how much of the span is off-pitch excursion, not a
        better estimate of the pitch.
    method : str
        Orientation estimator used.
    plausible : bool
        Whether ``length_m``/``width_m`` fall inside :data:`PLAUSIBLE_LENGTH_M`
        and :data:`PLAUSIBLE_WIDTH_M`.
    warning : str
        Why not, when ``plausible`` is False; empty otherwise.
    """

    origin_lat: float
    origin_lon: float
    rotation_deg: float
    length_m: float
    width_m: float
    trimmed_length_m: float
    trimmed_width_m: float
    method: str
    plausible: bool
    warning: str


# ---------------------------------------------------------------------------
# Source resolution and fetching
# ---------------------------------------------------------------------------

def resolve_source(source: str | None = None) -> str:
    """
    Decide what to read, honouring the environment overrides.

    A local path in :data:`ENV_SAMPLE_PATH` wins, then an alternate URL in
    :data:`ENV_SAMPLE_URL`, then the explicit ``source`` argument, then
    :data:`DEFAULT_STATSPORTS_SOURCE`.
    """
    local = os.environ.get(ENV_SAMPLE_PATH, "").strip()
    if local:
        return local
    url = os.environ.get(ENV_SAMPLE_URL, "").strip()
    if url:
        return url
    return source or DEFAULT_STATSPORTS_SOURCE


def default_cache_dir() -> str:
    """
    A disposable cache location, inside the OS temp directory.

    Deliberately not inside the repository and deliberately not durable: it is
    somewhere the operating system is free to clear, and :func:`purge_cache`
    removes it outright. Nothing depends on its contents surviving.
    """
    return os.path.join(tempfile.gettempdir(), "matrix_statsports_cache")


def purge_cache(cache_dir: str | None = None) -> bool:
    """
    Delete a cache directory written by :func:`load_statsports_source`.

    Returns True if something was removed. Safe to call at any time — a cached
    copy is only ever a re-fetchable convenience.
    """
    target = cache_dir or default_cache_dir()
    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
        return True
    return False


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _cached_path(source: str, cache_dir: str) -> str:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"statsports_{digest}.csv")


def _fetch_to_cache(source: str, cache_dir: str) -> str:
    """Download ``source`` into ``cache_dir`` unless it is already there."""
    path = _cached_path(source, cache_dir)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(cache_dir, exist_ok=True)
    request = urllib.request.Request(source, headers={"User-Agent": "matrix_engine"})
    # Write via a temporary name so an interrupted download cannot be mistaken
    # for a complete one on the next run.
    handle, staging = tempfile.mkstemp(dir=cache_dir, suffix=".part")
    os.close(handle)
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
            with open(staging, "wb") as out:
                shutil.copyfileobj(response, out)
        os.replace(staging, path)
    except Exception:
        if os.path.exists(staging):
            os.remove(staging)
        raise
    return path


def _read_raw(source: str, max_rows: int | None, cache_dir: str | None) -> pd.DataFrame:
    """Read the export into memory, converting every failure into unavailability."""
    target = source
    try:
        if cache_dir and _is_url(source):
            target = _fetch_to_cache(source, cache_dir)
        elif _is_url(source):
            request = urllib.request.Request(source, headers={"User-Agent": "matrix_engine"})
            with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
                target = io.BytesIO(response.read())
        elif not os.path.exists(source):
            raise StatsportsUnavailable(
                f"STATSports source '{source}' does not exist. Set "
                f"${ENV_SAMPLE_PATH} to a local copy, ${ENV_SAMPLE_URL} to an "
                "alternate URL, or leave both unset to fetch the default."
            )
        raw = pd.read_csv(target, nrows=max_rows)
    except StatsportsUnavailable:
        raise
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise StatsportsUnavailable(
            f"could not read STATSports source '{source}': {exc}. This is an "
            "expected outcome — the default source is a third-party repository "
            "that may be offline or removed."
        ) from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise StatsportsUnavailable(
            f"STATSports source '{source}' did not parse as CSV: {exc}"
        ) from exc

    missing = [c for c in _ESSENTIAL_COLUMNS if c not in raw.columns]
    if missing:
        raise StatsportsUnavailable(
            f"STATSports source '{source}' is missing column(s) {missing} — "
            f"got {list(raw.columns)[:6]}... This adapter targets one specific "
            "raw Apex export layout."
        )
    return raw


# ---------------------------------------------------------------------------
# Anonymisation
# ---------------------------------------------------------------------------

def anonymize_name(name: str, salt: str | None = None) -> str:
    """
    Map a display name to a stable opaque player ID.

    Deterministic, so the same player gets the same ID across runs and across
    processes — identity matching and any cross-session comparison need that.

    This is de-identification for handling hygiene, not anonymity against a
    motivated attacker: because it is unsalted-per-run and deterministic,
    somebody who already holds the squad list can hash each candidate name and
    confirm a match. What it does guarantee is that no name survives in any
    DataFrame, CSV, log line or report produced from this data, which is the
    property that matters for a dataset that should not have been republished
    in the first place.

    With a per-tenant ``salt`` (the product), the id is a keyed hash (HMAC-SHA256) under it, so the same player gets a
    different id in each tenant and the squad-list attack needs the tenant's secret. Without one, the public reference
    salt is used, which is refused when ``PONTA_ENV`` is ``production``.
    """
    if salt is None:
        if os.environ.get("PONTA_ENV", "").strip().lower() in ("production", "prod"):
            raise ValueError("a per-tenant id_salt is required when PONTA_ENV is production")
        digest = hashlib.sha256(f"{_ID_SALT}:{name}".encode()).hexdigest()
    else:
        if len(salt) < MIN_SALT_CHARS:
            raise ValueError(f"id_salt must be at least {MIN_SALT_CHARS} characters")
        digest = hmac.new(salt.encode(), f"statsports:{name}".encode(), hashlib.sha256).hexdigest()
    return f"sp_{digest[:_ID_LENGTH]}"


# ---------------------------------------------------------------------------
# native_quality
# ---------------------------------------------------------------------------

def compute_native_quality(
    raw: pd.DataFrame,
    *,
    hacc_worst_m: float = HACC_WORST_M,
    hdop_good: float = HDOP_GOOD,
    hdop_worst: float = HDOP_WORST,
    satellites_min: float = SATELLITES_MIN,
    satellites_good: float = SATELLITES_GOOD,
    weight_hacc: float = QUALITY_WEIGHT_HACC,
    weight_hdop: float = QUALITY_WEIGHT_HDOP,
    weight_satellites: float = QUALITY_WEIGHT_SATELLITES,
) -> pd.Series:
    """
    Condense the vendor's per-sample diagnostics into one score in [0, 1].

    Each available diagnostic becomes a sub-score that is 1 when the reading is
    good and 0 when it is bad, and the result is their weighted mean over
    whichever are populated on that row. Linear ramps throughout: the underlying
    readings are not calibrated against a position error in metres, so any
    curvature would be invented precision.

    ``Quality of Signal`` is not used — see the module docstring for the
    measurement behind that decision. Rows where the satellite count is zero are
    treated as *not reporting* that field rather than as reporting zero
    satellites, again for a measured reason, so their score comes from the
    remaining terms.

    Returns
    -------
    pd.Series
        Float score indexed like ``raw``. NaN where the export carried none of
        the diagnostics for that row, meaning "the vendor said nothing".
    """
    n = len(raw)
    numerator = np.zeros(n, dtype=float)
    denominator = np.zeros(n, dtype=float)

    if COL_HACC in raw.columns:
        hacc = pd.to_numeric(raw[COL_HACC], errors="coerce").to_numpy(dtype=float)
        term = np.clip(1.0 - hacc / max(hacc_worst_m, 1e-9), 0.0, 1.0)
        present = np.isfinite(hacc)
        numerator += np.where(present, weight_hacc * term, 0.0)
        denominator += np.where(present, weight_hacc, 0.0)

    if COL_HDOP in raw.columns:
        hdop = pd.to_numeric(raw[COL_HDOP], errors="coerce").to_numpy(dtype=float)
        span = max(hdop_worst - hdop_good, 1e-9)
        term = np.clip((hdop_worst - hdop) / span, 0.0, 1.0)
        # HDOP is a dilution factor; zero or negative is not a reading.
        present = np.isfinite(hdop) & (hdop > 0)
        numerator += np.where(present, weight_hdop * term, 0.0)
        denominator += np.where(present, weight_hdop, 0.0)

    if COL_SATELLITES in raw.columns:
        satellites = pd.to_numeric(raw[COL_SATELLITES], errors="coerce").to_numpy(dtype=float)
        span = max(satellites_good - satellites_min, 1e-9)
        term = np.clip((satellites - satellites_min) / span, 0.0, 1.0)
        present = np.isfinite(satellites) & (satellites > 0)
        numerator += np.where(present, weight_satellites * term, 0.0)
        denominator += np.where(present, weight_satellites, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(denominator > 0, numerator / denominator, np.nan)
    return pd.Series(np.clip(score, 0.0, 1.0), index=raw.index, name="native_quality")


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------

def _metres_per_degree(lat_deg: float) -> tuple[float, float]:
    """WGS84 series approximation of metres per degree at ``lat_deg``."""
    phi = np.radians(lat_deg)
    per_lat = 111132.92 - 559.82 * np.cos(2 * phi) + 1.175 * np.cos(4 * phi)
    per_lon = 111412.84 * np.cos(phi) - 93.5 * np.cos(3 * phi)
    return float(per_lat), float(per_lon)


def _pca_angle(points: np.ndarray) -> float:
    """Angle of the largest-variance direction, in radians."""
    centred = points - points.mean(axis=0)
    covariance = np.cov(centred.T)
    values, vectors = np.linalg.eigh(covariance)
    principal = vectors[:, int(np.argmax(values))]
    return float(np.arctan2(principal[1], principal[0]))


def _min_area_angle(points: np.ndarray) -> float:
    """
    Angle of the long side of the minimum-area bounding rectangle.

    By the rotating-calipers result the optimal rectangle has a side collinear
    with a convex-hull edge, so only the hull edge directions are candidates —
    exact, and a few dozen rotations rather than a sweep.
    """
    try:
        from scipy.spatial import ConvexHull

        hull = points[ConvexHull(points).vertices]
    except Exception:
        # Degenerate or collinear cloud: no rectangle to speak of.
        return _pca_angle(points)

    best_area = np.inf
    best_angle = 0.0
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    for dx, dy in edges:
        if dx == 0.0 and dy == 0.0:
            continue
        angle = float(np.arctan2(dy, dx))
        rotated = hull @ _rotation(-angle).T
        width = float(np.ptp(rotated[:, 0]))
        height = float(np.ptp(rotated[:, 1]))
        area = width * height
        if area < best_area:
            best_area = area
            best_angle = angle if width >= height else angle + np.pi / 2.0
    return best_angle


def _rotation(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def _orient(
    east: np.ndarray,
    north: np.ndarray,
    method: str,
    rotation_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotate the cloud so its long axis is x, and centre it on the box middle."""
    points = np.column_stack([east, north])
    if rotation_deg is not None:
        angle = float(np.radians(rotation_deg))
    elif method == "pca":
        angle = _pca_angle(points)
    elif method == "min_area_rect":
        angle = _min_area_angle(points)
    else:
        raise ValueError(
            f"unknown orientation method '{method}'; expected 'min_area_rect' or 'pca'"
        )

    rotated = points @ _rotation(-angle).T
    if rotation_deg is None and np.ptp(rotated[:, 1]) > np.ptp(rotated[:, 0]):
        # PCA can hand back the short axis when the cloud is near-isotropic,
        # which is exactly what this data does. Keep x as the longer extent.
        angle += np.pi / 2.0
        rotated = points @ _rotation(-angle).T

    x = rotated[:, 0] - (rotated[:, 0].max() + rotated[:, 0].min()) / 2.0
    y = rotated[:, 1] - (rotated[:, 1].max() + rotated[:, 1].min()) / 2.0
    return x, y, float(np.degrees(angle) % 180.0)


def _pitch_estimate(
    lat0: float,
    lon0: float,
    x: np.ndarray,
    y: np.ndarray,
    rotation_deg: float,
    method: str,
    trim_fraction: float = 0.05,
) -> PitchEstimate:
    """Summarise the recovered frame and sanity-check it against real pitches."""
    length = float(np.ptp(x))
    width = float(np.ptp(y))

    radius = np.hypot(x, y)
    keep = radius <= np.quantile(radius, 1.0 - trim_fraction)
    trimmed_length = float(np.ptp(x[keep])) if keep.any() else length
    trimmed_width = float(np.ptp(y[keep])) if keep.any() else width

    problems = []
    if not PLAUSIBLE_LENGTH_M[0] <= length <= PLAUSIBLE_LENGTH_M[1]:
        problems.append(
            f"long-axis span {length:.1f} m is outside the "
            f"{PLAUSIBLE_LENGTH_M[0]:.0f}–{PLAUSIBLE_LENGTH_M[1]:.0f} m range for a pitch"
        )
    if not PLAUSIBLE_WIDTH_M[0] <= width <= PLAUSIBLE_WIDTH_M[1]:
        problems.append(
            f"short-axis span {width:.1f} m is outside the "
            f"{PLAUSIBLE_WIDTH_M[0]:.0f}–{PLAUSIBLE_WIDTH_M[1]:.0f} m range for a pitch"
        )

    warning = ""
    if problems:
        warning = (
            "; ".join(problems)
            + ". The projection is uncalibrated and this session includes "
            "off-pitch movement, so the span covers more ground than the pitch "
            "itself; treat the coordinate frame as approximate."
        )
        warnings.warn(f"statsports: {warning}", RuntimeWarning, stacklevel=3)

    return PitchEstimate(
        origin_lat=float(lat0),
        origin_lon=float(lon0),
        rotation_deg=float(rotation_deg),
        length_m=length,
        width_m=width,
        trimmed_length_m=trimmed_length,
        trimmed_width_m=trimmed_width,
        method=method,
        plausible=not problems,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_statsports_source(
    source: str = DEFAULT_STATSPORTS_SOURCE,
    *,
    player_names: list[str] | None = None,
    anonymize: bool = True,
    max_rows: int | None = None,
    cache_dir: str | None = None,
    orientation: str = "min_area_rect",
    rotation_deg: float | None = None,
    with_raw_quality: bool = False,
    id_salt: str | None = None,
) -> pd.DataFrame:
    """
    Load a raw STATSports Apex export as Matrix-schema tracking.

    Parameters
    ----------
    source : str
        URL or local path. Overridden by :data:`ENV_SAMPLE_PATH` /
        :data:`ENV_SAMPLE_URL` when either is set; see :func:`resolve_source`.
    player_names : list[str], optional
        Restrict to these display names. Matched against the raw column and
        then discarded with it — the names are used to filter, never retained.
    anonymize : bool
        Must be True. The parameter exists to make the guarantee explicit;
        passing False raises rather than disabling anonymisation, because there
        is no legitimate path for a real name to leave this function.
    max_rows : int, optional
        Cap on rows read from the source, applied before ``player_names``
        filtering. The export stores one contiguous block per player, so a small
        cap yields the first player or two — enough for rate, quality and jitter
        work without transferring 51 MB.
    cache_dir : str, optional
        Disposable directory for a downloaded copy. ``None`` (the default) keeps
        nothing on disk. See :func:`default_cache_dir` and :func:`purge_cache`.
    orientation : {'min_area_rect', 'pca'}
        Pitch-axis estimator. ``min_area_rect`` by default because PCA is
        measurably unreliable here: this cloud's principal-axis eigenvalue ratio
        is 1.04, and the PCA angle swings from 68° to −169° depending on which
        portion of the session is included, while the minimum-area rectangle
        returns 62.5° from every subset tried.
    rotation_deg : float, optional
        Pin the pitch axis instead of estimating it. Worth knowing about: the
        estimate is fitted to whatever subset was loaded, so loading two
        different sets of players produces two different coordinate frames, and
        positions from separate calls are not comparable unless the rotation is
        pinned. Pass the ``rotation_deg`` from a full-session load to put a
        subset into the same frame.
    id_salt : str, optional
        The tenant's salt for player ids (required when ``PONTA_ENV`` is ``production``); see :func:`anonymize_name`.
    with_raw_quality : bool
        Also return the vendor's raw diagnostic columns
        (``vendor_speed_m_s``, ``hacc``, ``hdop``, ``quality_of_signal``,
        ``satellites``), so the quality model can be re-checked against them
        without re-reading the source.

    Returns
    -------
    pd.DataFrame
        Matrix schema plus ``native_quality``. ``df.attrs['statsports']``
        carries a :class:`PitchEstimate`, the measured per-player sample rates,
        and the session's wall-clock start, none of which fit in a column.

    Raises
    ------
    StatsportsUnavailable
        The source could not be read. Callers should skip, not fail.
    ValueError
        ``anonymize=False``, or an unknown ``orientation``.
    """
    if not anonymize:
        raise ValueError(
            "anonymize=False is not supported: this adapter reads real, "
            "name-attributed player data and no code path may emit those names. "
            "Filter with player_names= if you need a specific player."
        )

    resolved = resolve_source(source)
    raw = _read_raw(resolved, max_rows, cache_dir)

    if player_names is not None:
        wanted = set(player_names)
        raw = raw.loc[raw[COL_NAME].astype(str).isin(wanted)]
        if raw.empty:
            raise StatsportsUnavailable(
                f"none of the {len(wanted)} requested player name(s) appear in "
                f"the first {max_rows if max_rows else 'all'} row(s) of '{resolved}'"
            )

    # Anonymise before anything else reads the frame, and drop the name column
    # in the same step so nothing downstream can reach it.
    player_id = raw[COL_NAME].astype(str).map(lambda n: anonymize_name(n, id_salt))
    quality = compute_native_quality(raw)

    work = pd.DataFrame(
        {
            "player_id": player_id.to_numpy(),
            "wall_clock_s": pd.to_timedelta(raw[COL_TIME].astype(str)).dt.total_seconds(),
            "lat": pd.to_numeric(raw[COL_LAT], errors="coerce"),
            "lon": pd.to_numeric(raw[COL_LON], errors="coerce"),
            "native_quality": quality.to_numpy(),
        }
    )
    if with_raw_quality:
        for out_name, col in (
            ("vendor_speed_m_s", COL_SPEED),
            ("hacc", COL_HACC),
            ("hdop", COL_HDOP),
            ("quality_of_signal", COL_QOS),
            ("satellites", COL_SATELLITES),
        ):
            if col in raw.columns:
                work[out_name] = pd.to_numeric(raw[col], errors="coerce").to_numpy()
    del raw, player_id

    # Exactly-zero coordinates are null-island sentinels, not positions off the
    # west coast of Africa.
    work = work.loc[
        work["lat"].notna()
        & work["lon"].notna()
        & (work["lat"] != 0.0)
        & (work["lon"] != 0.0)
        & work["wall_clock_s"].notna()
    ]
    if work.empty:
        raise StatsportsUnavailable(
            f"'{resolved}' contained no rows with usable coordinates and timestamps"
        )
    work = work.sort_values(["player_id", "wall_clock_s"], kind="stable").reset_index(drop=True)

    lat0 = float(work["lat"].mean())
    lon0 = float(work["lon"].mean())
    per_lat, per_lon = _metres_per_degree(lat0)
    east = (work["lon"].to_numpy(dtype=float) - lon0) * per_lon
    north = (work["lat"].to_numpy(dtype=float) - lat0) * per_lat
    x, y, recovered_deg = _orient(east, north, orientation, rotation_deg)
    estimate = _pitch_estimate(
        lat0, lon0, x, y, recovered_deg,
        orientation if rotation_deg is None else f"pinned({rotation_deg:.2f} deg)",
    )

    session_start_s = float(work["wall_clock_s"].min())
    timestamp = work["wall_clock_s"].to_numpy(dtype=float) - session_start_s
    rates = _measure_sample_rates(work["player_id"].to_numpy(), timestamp)
    native_rate_hz = float(np.median(list(rates.values()))) if rates else np.nan
    if not np.isfinite(native_rate_hz) or native_rate_hz <= 0:
        raise StatsportsUnavailable(
            f"could not measure a sample rate from '{resolved}' — "
            "too few samples per player"
        )

    frame = np.rint(timestamp * FRAME_GRID_HZ).astype("int64")

    out = pd.DataFrame(
        {
            "timestamp": timestamp,
            "frame": frame,
            "player_id": work["player_id"].to_numpy(),
            "team": UNKNOWN_TEAM,
            "x": x,
            "y": y,
            "source": SOURCE_NAME,
            "sample_rate_hz": native_rate_hz,
            "native_quality": work["native_quality"].to_numpy(dtype=float),
        }
    )
    for extra in work.columns:
        if extra in ("player_id", "wall_clock_s", "lat", "lon", "native_quality"):
            continue
        out[extra] = work[extra].to_numpy()

    out = coerce_schema(out)
    # A player is occasionally sampled twice inside one 0.1 s grid cell; the
    # schema allows one observation per frame, so keep the first.
    out = out.drop_duplicates(subset=["source", "player_id", "frame"], keep="first")
    out = out.sort_values(["timestamp", "player_id"], kind="stable").reset_index(drop=True)
    validate_schema(out, name=SOURCE_NAME)

    out.attrs["statsports"] = {
        "pitch": estimate,
        "native_rate_hz": native_rate_hz,
        "player_rates_hz": rates,
        "session_start_s": session_start_s,
        "session_duration_s": float(timestamp.max() - timestamp.min()),
        "n_players": int(out["player_id"].nunique()),
        "source": resolved,
    }
    return out


def _measure_sample_rates(
    player_ids: np.ndarray, timestamp: np.ndarray
) -> dict[str, float]:
    """
    Measure each player's native rate from its own consecutive-sample deltas.

    Measured rather than assumed on purpose: this export turned out to be
    3.33 Hz, not the 10 Hz an Apex unit samples at, and a hard-coded rate would
    have silently mis-scaled every gap and noise estimate downstream.
    """
    rates: dict[str, float] = {}
    order = np.argsort(player_ids, kind="stable")
    ids = player_ids[order]
    times = timestamp[order]
    boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    for block_ids, block_times in zip(
        np.split(ids, boundaries), np.split(times, boundaries)
    ):
        if block_times.size < 2:
            continue
        deltas = np.diff(np.sort(block_times))
        deltas = deltas[deltas > 0]
        if deltas.size == 0:
            continue
        rates[str(block_ids[0])] = float(1.0 / np.median(deltas))
    return rates


def statsports_pitch(df: pd.DataFrame) -> PitchConfig:
    """
    A :class:`PitchConfig` built from the span recovered by the loader.

    Only for keeping this provider interchangeable with the others — it is
    derived from the data, so validating the same data against it is circular
    and proves nothing about whether the transform is correct.
    """
    meta = df.attrs.get("statsports")
    if not meta:
        raise ValueError(
            "frame carries no statsports metadata; pass the output of "
            "load_statsports_source"
        )
    estimate: PitchEstimate = meta["pitch"]
    return PitchConfig(
        length_m=estimate.length_m,
        width_m=estimate.width_m,
        fps=float(meta["native_rate_hz"]),
    )
