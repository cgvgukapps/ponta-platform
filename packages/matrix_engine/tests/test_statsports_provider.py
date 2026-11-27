#!/usr/bin/env python3
"""
Tests for the real STATSports Apex provider adapter.

Two tiers, deliberately:

**Unconditional.** The anonymisation guarantee, the quality-score model and the
coordinate maths are tested against inline synthetic fixtures, so they run on
every machine with no network and no configuration. The anonymisation test in
particular must never be skippable — it is the one protecting real people's
names, and a guarantee that only holds when a URL happens to be reachable is not
a guarantee.

**Gated on real data.** Everything asserting a property *of the export* needs the
export. Those tests skip — never fail — when it cannot be fetched, because the
default source is a raw URL into a third-party repository that has no license and
may be removed at any time, quite reasonably. A suite that goes red the day that
happens would be asserting the wrong thing.

The real data is fetched once per session into pytest's own temp directory, which
pytest deletes afterwards. Nothing lands in the repo tree.
"""

import io
import os

import numpy as np
import pandas as pd
import pytest
from matrix_engine.confidence import score_confidence
from matrix_engine.providers.statsports import (
    COL_HACC,
    COL_HDOP,
    COL_LAT,
    COL_LON,
    COL_NAME,
    COL_QOS,
    COL_SATELLITES,
    COL_SPEED,
    COL_TIME,
    DEFAULT_STATSPORTS_SOURCE,
    ENV_SAMPLE_PATH,
    ENV_SAMPLE_URL,
    SOURCE_NAME,
    PitchEstimate,
    StatsportsUnavailable,
    anonymize_name,
    compute_native_quality,
    default_cache_dir,
    load_statsports_source,
    purge_cache,
    resolve_source,
    statsports_pitch,
)
from matrix_engine.quality import detect_gaps, native_quality_summary
from matrix_engine.sample_data import repo_root as _repo_root
from matrix_engine.schema import UNKNOWN_TEAM, SchemaError, validate_schema

#: Obviously fake names for the unconditional fixtures. Never a real person, so
#: this file can assert on them in plain text.
FAKE_NAMES = ["Wibble Fakename", "Zorb Testinson"]

#: Rows read for the "a small cap is enough" test. The export stores one
#: contiguous block per player, so this is the first player or two.
SMALL_MAX_ROWS = 20000


# ---------------------------------------------------------------------------
# Synthetic fixtures — these tests always run
# ---------------------------------------------------------------------------

def _fake_export(n_per_player: int = 400, names: list[str] | None = None) -> pd.DataFrame:
    """
    A stand-in with the real export's column layout and a plausible pitch walk.

    Players trace laps of a rectangle so that the recovered orientation and span
    are predictable, which is what makes the coordinate maths testable without
    the real file.
    """
    names = names or FAKE_NAMES
    rows = []
    for index, name in enumerate(names):
        t = np.arange(n_per_player) * 0.3
        # A 100 m x 60 m lap, at the latitude of the real session so the
        # projection constants are exercised in the right regime.
        phase = 2 * np.pi * t / (n_per_player * 0.3)
        east = 50.0 * np.cos(phase)
        north = 30.0 * np.sin(phase) + 3.0 * index
        lat = 53.3058 + north / 111_291.0
        lon = -6.2279 + east / 66_674.0
        clock = pd.to_timedelta(65280.1 + t, unit="s")
        rows.append(
            pd.DataFrame(
                {
                    COL_NAME: name,
                    COL_TIME: [str(v)[-15:] for v in clock.astype(str)],
                    COL_LAT: lat,
                    COL_LON: lon,
                    COL_SPEED: np.abs(np.gradient(east, 0.3)),
                    COL_HACC: 0,
                    COL_HDOP: 0.4,
                    COL_QOS: 310,
                    COL_SATELLITES: 20,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def isolated_env(monkeypatch):
    """
    Ignore any override the developer has configured.

    Tests that pass an explicit path must read *that* path. Without this, running
    the suite with ``MATRIX_STATSPORTS_SAMPLE_PATH`` set would silently redirect
    every synthetic fixture at the real export.
    """
    monkeypatch.delenv(ENV_SAMPLE_PATH, raising=False)
    monkeypatch.delenv(ENV_SAMPLE_URL, raising=False)


@pytest.fixture
def fake_export_path(tmp_path, isolated_env):
    path = tmp_path / "fake_statsports.csv"
    _fake_export().to_csv(path, index=False)
    return str(path)


def test_a_real_name_never_survives_into_the_output(fake_export_path):
    """
    The guarantee that matters most, and the reason it is not gated on network.

    Checks every value of every column, not just ``player_id``, because the
    failure being guarded against is a name surviving somewhere unexpected — a
    stray passthrough column, an index, a source label.
    """
    df = load_statsports_source(fake_export_path, with_raw_quality=True)

    haystack = "\n".join(str(v) for v in df.to_numpy().ravel())
    haystack += "\n" + "\n".join(map(str, df.columns))
    haystack += "\n" + "\n".join(map(str, df.index))
    haystack += "\n" + str(df.attrs)

    for name in FAKE_NAMES:
        assert name not in haystack
        for token in name.split():
            assert token not in haystack, f"name fragment '{token}' leaked into output"

    assert COL_NAME not in df.columns
    assert df["player_id"].nunique() == len(FAKE_NAMES)
    assert df["player_id"].str.startswith("sp_").all()


def test_anonymisation_cannot_be_switched_off(fake_export_path):
    with pytest.raises(ValueError, match="anonymize=False is not supported"):
        load_statsports_source(fake_export_path, anonymize=False)


def test_player_ids_are_stable_and_collision_free():
    """Identity matching needs the same player to get the same ID every time."""
    assert anonymize_name("Wibble Fakename") == anonymize_name("Wibble Fakename")
    assert anonymize_name("Wibble Fakename") != anonymize_name("Zorb Testinson")

    ids = {anonymize_name(f"Player Number {i}") for i in range(500)}
    assert len(ids) == 500


def test_a_tenant_salt_gives_tenant_specific_ids(monkeypatch):
    """Review DE-15: the public salt is for the reference sample only; each tenant's ids differ and need its secret."""
    a, b = anonymize_name("Wibble Fakename", "tenant-a-salt-0123456789"), anonymize_name("Wibble Fakename", "tenant-b-salt-0123456789")
    assert a != b != anonymize_name("Wibble Fakename") and a.startswith("sp_")
    with pytest.raises(ValueError):
        anonymize_name("Wibble Fakename", "short")
    monkeypatch.setenv("PONTA_ENV", "production")
    with pytest.raises(ValueError):
        anonymize_name("Wibble Fakename")
    assert anonymize_name("Wibble Fakename", "tenant-a-salt-0123456789") == a


def test_filtering_by_name_keeps_only_that_player_and_still_anonymises(fake_export_path):
    df = load_statsports_source(fake_export_path, player_names=[FAKE_NAMES[0]])
    assert df["player_id"].nunique() == 1
    assert df["player_id"].iloc[0] == anonymize_name(FAKE_NAMES[0])
    assert FAKE_NAMES[0] not in "\n".join(str(v) for v in df.to_numpy().ravel())


def test_an_unknown_player_name_is_reported_as_unavailable(fake_export_path):
    with pytest.raises(StatsportsUnavailable, match="requested player name"):
        load_statsports_source(fake_export_path, player_names=["Nobody At All"])


def test_output_conforms_to_the_common_schema(fake_export_path):
    df = load_statsports_source(fake_export_path)
    validate_schema(df, name="statsports")

    assert (df["source"] == SOURCE_NAME).all()
    assert (df["team"] == UNKNOWN_TEAM).all()
    assert df["native_quality"].between(0.0, 1.0).all()
    assert not df.duplicated(subset=["source", "player_id", "frame"]).any()
    assert df["frame"].dtype.kind == "i"


def test_team_is_unknown_rather_than_a_guessed_home_away_split(fake_export_path):
    """
    A single-club export contains no opponent, so inventing a split would be
    fabrication. ``unknown`` is a schema-legal team label for exactly this case.
    """
    df = load_statsports_source(fake_export_path)
    assert set(df["team"].unique()) == {UNKNOWN_TEAM}
    validate_schema(df, name="statsports")


def test_zero_coordinates_are_dropped_rather_than_projected(tmp_path, isolated_env):
    """``Lat``/``Lon`` of exactly 0 is a null-island sentinel, not a position."""
    raw = _fake_export(n_per_player=200)
    raw.loc[raw.index[:20], [COL_LAT, COL_LON]] = 0.0
    path = tmp_path / "nulls.csv"
    raw.to_csv(path, index=False)

    df = load_statsports_source(str(path))
    assert len(df) == len(raw) - 20
    # Nothing survived near the projection of (0, 0), which would be thousands
    # of kilometres away and would wreck the recovered span.
    assert np.abs(df["x"]).max() < 500.0


def test_sample_rate_is_measured_not_assumed(tmp_path, isolated_env):
    """A 5 Hz file must report 5 Hz, with no 3.33 or 10 Hz default leaking in."""
    raw = _fake_export(n_per_player=300)
    t = np.arange(300) * 0.2
    for name in FAKE_NAMES:
        mask = raw[COL_NAME] == name
        raw.loc[mask, COL_TIME] = [
            str(v)[-15:] for v in pd.to_timedelta(65280.0 + t, unit="s").astype(str)
        ]
    path = tmp_path / "five_hz.csv"
    raw.to_csv(path, index=False)

    df = load_statsports_source(str(path))
    assert df["sample_rate_hz"].iloc[0] == pytest.approx(5.0, abs=0.01)
    rates = df.attrs["statsports"]["player_rates_hz"]
    assert len(rates) == len(FAKE_NAMES)
    assert all(v == pytest.approx(5.0, abs=0.01) for v in rates.values())


def test_the_recovered_frame_is_metric_centred_and_long_axis_first(fake_export_path):
    df = load_statsports_source(fake_export_path)
    estimate = df.attrs["statsports"]["pitch"]

    assert isinstance(estimate, PitchEstimate)
    assert estimate.length_m >= estimate.width_m
    # The fixture laps a 100 x 60 m rectangle, offsetting each player 3 m north,
    # so the recoverable extent is 100 m by 60 + 3 * (players - 1).
    assert estimate.length_m == pytest.approx(100.0, abs=2.0)
    assert estimate.width_m == pytest.approx(60.0 + 3.0 * (len(FAKE_NAMES) - 1), abs=2.0)
    # Centred: the extremes are symmetric about zero.
    assert df["x"].mean() == pytest.approx(0.0, abs=2.0)
    assert abs(df["x"].max() + df["x"].min()) < 1e-6
    assert abs(df["y"].max() + df["y"].min()) < 1e-6


def test_a_pinned_rotation_puts_separate_loads_in_one_frame(fake_export_path):
    """
    The orientation is fitted to whatever was loaded, so two subsets otherwise
    land in two different coordinate systems — a quiet way to corrupt a fusion.
    """
    full = load_statsports_source(fake_export_path)
    pinned = full.attrs["statsports"]["pitch"].rotation_deg

    one = load_statsports_source(fake_export_path, player_names=[FAKE_NAMES[0]])
    one_pinned = load_statsports_source(
        fake_export_path, player_names=[FAKE_NAMES[0]], rotation_deg=pinned
    )

    assert one_pinned.attrs["statsports"]["pitch"].rotation_deg == pytest.approx(pinned)
    # The unpinned subset is free to disagree; the pinned one may not.
    assert one.attrs["statsports"]["pitch"].method == "min_area_rect"
    assert one_pinned.attrs["statsports"]["pitch"].method.startswith("pinned")


def test_an_implausible_span_warns_rather_than_passing_silently(tmp_path, isolated_env):
    """A span nothing like a pitch must be flagged, not quietly accepted."""
    raw = _fake_export(n_per_player=200)
    # Spread the walk over ~1 km, far too big for any pitch.
    raw[COL_LAT] = 53.3058 + np.linspace(0, 0.009, len(raw))
    raw[COL_LON] = -6.2279 + np.linspace(0, 0.004, len(raw))
    path = tmp_path / "huge.csv"
    raw.to_csv(path, index=False)

    with pytest.warns(RuntimeWarning, match="outside the"):
        df = load_statsports_source(str(path))

    estimate = df.attrs["statsports"]["pitch"]
    assert estimate.plausible is False
    assert estimate.warning
    assert "uncalibrated" in estimate.warning


def test_orientation_method_must_be_one_we_implement(fake_export_path):
    with pytest.raises(ValueError, match="unknown orientation method"):
        load_statsports_source(fake_export_path, orientation="magic")


def test_statsports_pitch_reflects_the_recovered_span(fake_export_path):
    df = load_statsports_source(fake_export_path)
    pitch = statsports_pitch(df)
    estimate = df.attrs["statsports"]["pitch"]

    assert pitch.length_m == pytest.approx(estimate.length_m)
    assert pitch.width_m == pytest.approx(estimate.width_m)
    assert pitch.fps == pytest.approx(df["sample_rate_hz"].iloc[0])

    with pytest.raises(ValueError, match="no statsports metadata"):
        statsports_pitch(pd.DataFrame({"x": [1.0]}))


# ---------------------------------------------------------------------------
# native_quality model — synthetic, always runs
# ---------------------------------------------------------------------------

def test_quality_score_is_one_for_a_clean_reading_and_falls_as_readings_worsen():
    raw = pd.DataFrame(
        {
            COL_HACC: [0, 0, 0, 1, 3, 7],
            COL_HDOP: [0.4, 0.4, 0.8, 0.4, 1.4, 2.5],
            COL_SATELLITES: [20, 18, 18, 18, 16, 14],
        }
    )
    quality = compute_native_quality(raw)

    assert quality.iloc[0] == pytest.approx(1.0)
    assert quality.iloc[1] == pytest.approx(1.0)
    assert quality.is_monotonic_decreasing
    assert quality.between(0.0, 1.0).all()


def test_a_zero_satellite_count_is_treated_as_unreported_not_as_zero_satellites():
    """
    Measured, not assumed: in the real export 15% of rows carry
    ``No. of Satellites`` and ``Quality of Signal`` of 0 together, and those rows
    show no elevated jitter and no position jump when the fields resume. They are
    unpopulated telemetry, so the term drops out instead of scoring zero.
    """
    reported = pd.DataFrame({COL_HACC: [0], COL_HDOP: [0.4], COL_SATELLITES: [20]})
    unreported = pd.DataFrame({COL_HACC: [0], COL_HDOP: [0.4], COL_SATELLITES: [0]})

    assert compute_native_quality(reported).iloc[0] == pytest.approx(1.0)
    assert compute_native_quality(unreported).iloc[0] == pytest.approx(1.0)


def test_quality_is_nan_when_the_export_carries_no_diagnostics_at_all():
    """NaN means 'the vendor said nothing', which is not the same as 'bad'."""
    quality = compute_native_quality(pd.DataFrame({"unrelated": [1, 2, 3]}))
    assert quality.isna().all()


def test_horizontal_accuracy_dominates_the_score():
    """
    ``Hacc`` carries by far the strongest measured relationship to position
    error (2.2x median jitter, 3.8x median speed disagreement), so it is
    weighted to move the score more than the others.
    """
    base = {COL_HACC: [0], COL_HDOP: [0.4], COL_SATELLITES: [20]}

    def score(**worsened):
        return compute_native_quality(pd.DataFrame({**base, **worsened})).iloc[0]

    clean = score()
    # Each of these is the worst reading the real export contains for that field.
    assert clean == pytest.approx(1.0)
    assert score(**{COL_HACC: [7]}) < score(**{COL_HDOP: [2.5]}) < score(
        **{COL_SATELLITES: [6]}
    ) < clean


def test_quality_of_signal_is_ignored_because_it_carries_no_measurable_signal():
    """
    Its scale is undocumented (0, then 173–342) and its Spearman correlation
    against three independent error proxies is |rho| <= 0.035. Including it
    swamped the fields that do work, so it is parsed but excluded.
    """
    base = {COL_HACC: [0, 0], COL_HDOP: [0.4, 0.4], COL_SATELLITES: [20, 20]}
    with_qos = compute_native_quality(pd.DataFrame({**base, COL_QOS: [342, 173]}))
    assert with_qos.iloc[0] == pytest.approx(with_qos.iloc[1])


def test_the_quality_model_can_be_refitted_without_editing_the_module():
    raw = pd.DataFrame({COL_HACC: [2], COL_HDOP: [0.4], COL_SATELLITES: [20]})
    default = compute_native_quality(raw).iloc[0]
    stricter = compute_native_quality(raw, hacc_worst_m=2.0).iloc[0]
    assert stricter < default


# ---------------------------------------------------------------------------
# Source resolution and unavailability — always runs
# ---------------------------------------------------------------------------

def test_the_default_source_is_the_live_url_with_no_download_step(isolated_env):
    """No 'download this first' step: unconfigured means fetch, not fail."""
    assert resolve_source() == DEFAULT_STATSPORTS_SOURCE
    assert DEFAULT_STATSPORTS_SOURCE.startswith("https://")


def test_environment_overrides_take_precedence_in_order(monkeypatch, isolated_env):
    monkeypatch.setenv(ENV_SAMPLE_URL, "https://example.invalid/other.csv")
    assert resolve_source() == "https://example.invalid/other.csv"

    monkeypatch.setenv(ENV_SAMPLE_PATH, "/tmp/local.csv")
    assert resolve_source() == "/tmp/local.csv"


def test_an_unreachable_source_is_unavailable_rather_than_an_exception_to_debug(
    monkeypatch, isolated_env
):
    """
    The failure mode this has to get right: the source repository is removed and
    the suite keeps working. Every route to 'cannot read it' must land on the
    one exception type callers know to skip on.
    """
    monkeypatch.setenv(ENV_SAMPLE_URL, "https://raw.githubusercontent.invalid/gone.csv")
    with pytest.raises(StatsportsUnavailable):
        load_statsports_source()


def test_a_missing_local_file_is_unavailable(tmp_path, isolated_env):
    with pytest.raises(StatsportsUnavailable, match="does not exist"):
        load_statsports_source(str(tmp_path / "absent.csv"))


def test_a_file_that_is_not_this_export_is_unavailable(tmp_path, isolated_env):
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
    with pytest.raises(StatsportsUnavailable, match="missing column"):
        load_statsports_source(str(path))


def test_an_empty_file_is_unavailable(tmp_path, isolated_env):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(StatsportsUnavailable):
        load_statsports_source(str(path))


def test_the_cache_lives_outside_the_repository_and_can_be_purged(tmp_path):
    """
    Nothing derived from this dataset may become a durable artifact, so the
    cache is opt-in, lives in the OS temp directory, and deletes on request.
    """
    repo_root = str(_repo_root())
    assert not default_cache_dir().startswith(repo_root)

    assert purge_cache(str(tmp_path / "never_created")) is False
    made = tmp_path / "cache"
    made.mkdir()
    (made / "statsports_x.csv").write_text("data")
    assert purge_cache(str(made)) is True
    assert not made.exists()


def test_no_disk_copy_is_made_unless_a_cache_is_requested(fake_export_path, tmp_path):
    cache = tmp_path / "unused_cache"
    load_statsports_source(fake_export_path)
    assert not cache.exists()
    assert not os.path.exists(default_cache_dir()) or not os.listdir(default_cache_dir())


# ---------------------------------------------------------------------------
# Pipeline integration on synthetic data — always runs
# ---------------------------------------------------------------------------

def test_native_quality_reaches_the_confidence_score(fake_export_path):
    """
    The capability this whole change exists for: a single source with no
    disagreement and no dropout still gets marked down where its own sensor
    said the reading was poor.
    """
    df = load_statsports_source(fake_export_path)
    df.loc[df.index[:50], "native_quality"] = 0.0

    scored = score_confidence(df)
    assert "native_quality_factor" in scored.columns

    poor = scored.loc[scored["native_quality"] == 0.0, "confidence"]
    good = scored.loc[scored["native_quality"] == 1.0, "confidence"]
    assert poor.max() < good.min()


def test_quality_report_surfaces_the_sources_own_verdict(fake_export_path):
    df = load_statsports_source(fake_export_path)
    df.loc[df.index[: int(0.3 * len(df))], "native_quality"] = 0.2

    summary = native_quality_summary({SOURCE_NAME: df})
    assert list(summary["source"]) == [SOURCE_NAME]
    row = summary.iloc[0]
    assert row["flagged_fraction"] == pytest.approx(0.3, abs=0.02)
    assert row["flagged"]
    assert "native quality" in row["reason"]


def test_sources_without_diagnostics_are_absent_from_the_summary_not_scored_perfect():
    """Reporting nothing is different from reporting good, and must look different."""
    plain = pd.DataFrame(
        {
            "timestamp": [0.0, 0.04],
            "frame": [0, 1],
            "player_id": ["home_1", "home_1"],
            "team": ["home", "home"],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "source": ["metrica", "metrica"],
            "sample_rate_hz": [25.0, 25.0],
        }
    )
    assert native_quality_summary({"metrica": plain}).empty


def test_native_quality_must_stay_on_scale(fake_export_path):
    df = load_statsports_source(fake_export_path)
    df.loc[df.index[0], "native_quality"] = 1.5
    with pytest.raises(SchemaError, match="outside"):
        validate_schema(df, name="statsports")


# ---------------------------------------------------------------------------
# Real export — skipped cleanly when it cannot be fetched
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def statsports_cache(tmp_path_factory):
    """
    A disposable cache for the session, managed by pytest.

    Means the 51 MB export is transferred at most once per run, and that pytest
    deletes the copy afterwards without this file having to remember to.
    """
    return str(tmp_path_factory.mktemp("statsports_cache"))


@pytest.fixture(scope="session")
def real_statsports(statsports_cache):
    """
    The real export, or a skip.

    Session-scoped because fetching and projecting 325k rows is several seconds
    and every test here treats it as read-only.
    """
    try:
        return load_statsports_source(cache_dir=statsports_cache, with_raw_quality=True)
    except StatsportsUnavailable as exc:
        pytest.skip(f"real STATSports export unavailable: {exc}")


@pytest.fixture(scope="session")
def real_jitter(real_statsports):
    """
    Per-row second-difference magnitude — an error proxy that owes nothing to
    the vendor's quality fields, which is what makes it usable as a check on them.

    Player motion is smooth across three samples 0.3 s apart, so most of the
    curvature at this scale is measurement error rather than real movement. Only
    triples at the native interval count, so a dropout cannot masquerade as noise.
    """
    values = pd.Series(np.nan, index=real_statsports.index, dtype=float)
    for _, block in real_statsports.groupby("player_id", sort=False):
        block = block.sort_values("timestamp")
        t = block["timestamp"].to_numpy(dtype=float)
        if len(t) < 3:
            continue
        x = block["x"].to_numpy(dtype=float)
        y = block["y"].to_numpy(dtype=float)
        nominal = float(np.median(np.diff(t)))
        consecutive = np.abs(np.diff(t) - nominal) < 0.05
        usable = np.zeros(len(block), dtype=bool)
        usable[1:-1] = consecutive[:-1] & consecutive[1:]
        curvature = np.full(len(block), np.nan)
        curvature[1:-1] = np.hypot(
            x[2:] - 2 * x[1:-1] + x[:-2], y[2:] - 2 * y[1:-1] + y[:-2]
        )
        curvature[~usable] = np.nan
        values.loc[block.index] = curvature
    return values


def test_the_real_export_satisfies_the_common_schema(real_statsports):
    validate_schema(real_statsports, name="statsports")
    assert len(real_statsports) > 100_000
    assert real_statsports["player_id"].nunique() >= 10
    assert real_statsports["native_quality"].between(0.0, 1.0).all()
    assert (real_statsports["team"] == UNKNOWN_TEAM).all()


def test_no_real_name_appears_anywhere_in_the_real_output(real_statsports):
    """
    The synthetic anonymisation test proves the mechanism; this proves it held on
    the actual file. Player IDs are opaque hashes and nothing in the frame looks
    like a personal name.
    """
    ids = real_statsports["player_id"].unique()
    assert all(pid.startswith("sp_") for pid in ids)
    assert all(len(pid) == len("sp_") + 10 for pid in ids)

    object_columns = real_statsports.select_dtypes(include="object")
    values = set(pd.unique(object_columns.to_numpy().ravel()))
    assert not any(" " in str(v) for v in values), "a space-separated value looks like a name"


def test_the_real_native_rate_is_measured_at_a_third_of_apex_sampling(real_statsports):
    """
    3.33 Hz, not the 10 Hz an Apex unit samples at internally: this export was
    decimated before publication. Assuming 10 Hz would have mis-scaled every
    gap threshold and noise estimate downstream, which is why the rate is
    measured per player rather than declared.
    """
    meta = real_statsports.attrs["statsports"]
    assert meta["native_rate_hz"] == pytest.approx(10.0 / 3.0, rel=0.02)
    rates = np.array(list(meta["player_rates_hz"].values()))
    assert len(rates) >= 10
    assert np.allclose(rates, 10.0 / 3.0, rtol=0.05)
    assert meta["session_duration_s"] > 5000.0


def test_the_recovered_span_is_pitch_length_but_flags_its_own_width(real_statsports):
    """
    The honest outcome of an uncalibrated transform. The long axis lands squarely
    in the range for a real pitch; the short axis comes out near 98 m, wider than
    any legal pitch, because the session includes off-pitch movement. The adapter
    flags that rather than passing silently — and trimming the outer 5% of
    samples brings both axes back into range, which shows where the excess is
    coming from without pretending the transform has been validated.
    """
    estimate = real_statsports.attrs["statsports"]["pitch"]

    assert 90.0 <= estimate.length_m <= 120.0
    assert estimate.width_m > 90.0
    assert estimate.plausible is False
    assert estimate.warning and "uncalibrated" in estimate.warning

    assert estimate.trimmed_width_m < estimate.width_m
    assert estimate.trimmed_length_m < estimate.length_m
    assert estimate.method == "min_area_rect"


def test_the_minimum_area_rectangle_beats_pca_on_this_cloud(real_statsports, statsports_cache):
    """
    Why the default is not PCA, as the task originally specified. This point
    cloud is near-isotropic — the two principal standard deviations are 18.9 m
    and 18.1 m — so 'the direction of greatest variance' is not the pitch's long
    axis in any stable sense, and PCA returns a visibly worse rectangle.
    """
    rect = real_statsports.attrs["statsports"]["pitch"]
    pca = load_statsports_source(
        cache_dir=statsports_cache, orientation="pca"
    ).attrs["statsports"]["pitch"]

    assert rect.length_m * rect.width_m < pca.length_m * pca.width_m
    assert rect.width_m < pca.width_m


def test_native_quality_predicts_real_position_trouble(real_statsports, real_jitter):
    """
    The substitute for Track B's ground-truth RMSE check, since no paired
    optical source exists for this data. If the vendor's self-reported quality
    means anything, the samples it marks down must be measurably worse by a
    yardstick that never saw those fields — here, trajectory curvature.

    The effect is large and, more convincingly, monotone in the threshold: the
    worse the reported quality, the worse the measured jitter.
    """
    frame = real_statsports.assign(jitter=real_jitter).dropna(subset=["jitter"])
    clean = frame.loc[frame["native_quality"] >= 0.9999, "jitter"]
    assert len(clean) > 10_000

    ratios = []
    for threshold in (0.9, 0.8, 0.7):
        flagged = frame.loc[frame["native_quality"] < threshold, "jitter"]
        assert len(flagged) > 200, f"too few samples below {threshold} to judge"
        ratios.append(flagged.median() / clean.median())

    # Flagged samples are clearly worse, not marginally worse.
    assert ratios[-1] > 2.0
    # And the relationship is a dose-response, not one odd bucket.
    assert ratios == sorted(ratios)


def test_native_quality_also_predicts_disagreement_with_the_vendors_own_speed(
    real_statsports,
):
    """
    A second, independent yardstick: the export ships its own ``Speed (m/s)``,
    and where the positions are bad the speed implied by differencing them
    drifts away from what the unit reported. Two unrelated proxies agreeing
    makes it much harder to argue the first was an artefact of how jitter was
    defined.
    """
    frame = real_statsports.sort_values(["player_id", "timestamp"])
    errors = []
    for _, block in frame.groupby("player_id", sort=False):
        t = block["timestamp"].to_numpy(dtype=float)
        if len(t) < 3:
            continue
        deltas = np.diff(t)
        nominal = float(np.median(deltas))
        consecutive = np.abs(deltas - nominal) < 0.05
        step = np.hypot(np.diff(block["x"].to_numpy()), np.diff(block["y"].to_numpy()))
        derived = np.where(consecutive, step / np.where(deltas > 0, deltas, np.nan), np.nan)
        error = np.full(len(block), np.nan)
        error[1:] = np.abs(derived - block["vendor_speed_m_s"].to_numpy()[1:])
        errors.append(pd.Series(error, index=block.index))

    frame = frame.assign(speed_error=pd.concat(errors)).dropna(subset=["speed_error"])
    clean = frame.loc[frame["native_quality"] >= 0.9999, "speed_error"]
    flagged = frame.loc[frame["native_quality"] < 0.7, "speed_error"]

    assert len(flagged) > 200
    assert flagged.median() > 3.0 * clean.median()


def test_the_vendors_quality_verdict_reaches_the_quality_report(real_statsports):
    summary = native_quality_summary({SOURCE_NAME: real_statsports})
    row = summary.iloc[0]

    assert row["reported"] == len(real_statsports)
    assert 0.9 < row["mean_quality"] <= 1.0
    assert row["min_quality"] < 0.7
    assert row["flagged_samples"] > 0
    # This session was mostly clean, so it should not trip the source-level flag.
    assert not row["flagged"]


def test_dropping_null_island_rows_surfaces_them_as_real_dropout_gaps(real_statsports):
    """
    The two quality mechanisms turn out to complement each other here, which was
    not the expected result.

    The export is otherwise dense: every player is sampled at 3.33 Hz with no
    missing intervals, so ``detect_gaps`` would have nothing to say. But the 52
    rows carrying coordinates of exactly (0, 0) are not scattered — they come in
    bursts, and one is 46 consecutive samples long. Refusing to project them
    turns the vendor's own unusable rows into an honest 14-second dropout that
    ``detect_gaps`` reports and ``sync`` then prices into ``gap_s``, rather than
    a 5000 km round trip to the Atlantic that fusion would have averaged in.
    """
    gaps = detect_gaps({SOURCE_NAME: real_statsports})

    assert not gaps.empty
    assert gaps["duration_s"].max() > 10.0
    # Everything reported traces back to the ~52 dropped sentinel rows, so the
    # total missing sample count stays tiny against 325k rows.
    assert gaps["missed_samples"].sum() < 100
    assert (gaps["source"] == SOURCE_NAME).all()


def test_a_small_row_cap_is_enough_for_rate_and_quality_work(
    real_statsports, statsports_cache
):
    """
    Rows are stored in one contiguous block per player, so a cap yields whole
    players rather than a thin slice of all of them — which is what makes it a
    usable way to work without transferring the entire file.

    Depends on ``real_statsports`` purely to inherit its skip: any test that
    reaches the network has to route through the one fixture that knows how to
    turn unavailability into a skip, or it fails the day the source disappears.
    """
    df = load_statsports_source(cache_dir=statsports_cache, max_rows=SMALL_MAX_ROWS)
    validate_schema(df, name="statsports")

    assert len(df) <= SMALL_MAX_ROWS
    assert df["player_id"].nunique() <= 3
    assert df["sample_rate_hz"].iloc[0] == pytest.approx(10.0 / 3.0, rel=0.02)


def test_reading_the_real_export_leaves_nothing_in_the_repository(real_statsports):
    """
    Guards the handling rule directly rather than trusting it: after a real
    load, no file from this dataset may exist anywhere in the repo tree.
    """
    repo_root = str(_repo_root())
    suspicious = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in {".git", ".venv", "__pycache__", "node_modules"}]
        for name in files:
            lowered = name.lower()
            if "maindata" in lowered or "statsport" in lowered and lowered.endswith(".csv"):
                suspicious.append(os.path.join(root, name))
    assert suspicious == []


def test_the_cache_is_a_plain_refetchable_file_outside_the_repo(real_statsports, statsports_cache):
    repo_root = str(_repo_root())
    cached = [f for f in os.listdir(statsports_cache) if f.endswith(".csv")]
    assert cached, "a cache_dir was supplied, so the download should have landed there"
    assert not os.path.realpath(statsports_cache).startswith(os.path.realpath(repo_root))

    # Readable as the CSV it claims to be, and deletable at any time.
    sample = pd.read_csv(os.path.join(statsports_cache, cached[0]), nrows=5)
    assert COL_NAME in sample.columns


def test_a_second_load_reuses_the_cached_copy(statsports_cache, real_statsports):
    """The cache exists to avoid re-transferring 51 MB, so prove it does that."""
    path = os.path.join(
        statsports_cache,
        next(f for f in os.listdir(statsports_cache) if f.endswith(".csv")),
    )
    before = os.path.getmtime(path)
    load_statsports_source(cache_dir=statsports_cache, max_rows=5000)
    assert os.path.getmtime(path) == before


def test_a_local_override_reads_the_same_way_as_the_url(statsports_cache, real_statsports, monkeypatch):
    """
    Anyone reproducing this will hold their own copy, so the local-override path
    has to behave identically to the fetch path.
    """
    path = os.path.join(
        statsports_cache,
        next(f for f in os.listdir(statsports_cache) if f.endswith(".csv")),
    )
    monkeypatch.setenv(ENV_SAMPLE_PATH, path)
    assert resolve_source() == path

    df = load_statsports_source(max_rows=SMALL_MAX_ROWS)
    validate_schema(df, name="statsports")
    assert df["sample_rate_hz"].iloc[0] == pytest.approx(10.0 / 3.0, rel=0.02)


def test_a_truncated_stream_still_parses_or_reports_unavailable(
    statsports_cache, real_statsports, tmp_path, isolated_env
):
    """
    A half-transferred file must not become silently wrong data. Either it
    parses as a shorter export, or it reports unavailable.
    """
    path = os.path.join(
        statsports_cache,
        next(f for f in os.listdir(statsports_cache) if f.endswith(".csv")),
    )
    with open(path, "rb") as handle:
        head = handle.read(400_000)
    truncated = tmp_path / "truncated.csv"
    truncated.write_bytes(head[: head.rfind(b"\n")])

    try:
        df = load_statsports_source(str(truncated))
    except StatsportsUnavailable:
        return
    validate_schema(df, name="statsports")
    assert len(df) > 0


def test_the_export_is_one_session_on_a_single_shared_clock(real_statsports):
    """
    Context for every number quoted from this data: it is one ~92-minute
    session with every player on the same wall clock, not a season of matches.
    That is why there is no cross-match validation here.
    """
    meta = real_statsports.attrs["statsports"]
    assert 5000.0 < meta["session_duration_s"] < 6000.0

    spans = real_statsports.groupby("player_id")["timestamp"].agg(["min", "max"])
    # Every player overlaps the session; one starts late, as a substitute would.
    assert (spans["max"] > 0.95 * meta["session_duration_s"]).all()
    assert (spans["min"] < 0.05 * meta["session_duration_s"]).sum() >= len(spans) - 2


def test_no_stray_bytesio_or_name_column_survives_a_real_load(real_statsports):
    expected = {
        "timestamp", "frame", "player_id", "team", "x", "y", "source",
        "sample_rate_hz", "native_quality", "vendor_speed_m_s", "hacc", "hdop",
        "quality_of_signal", "satellites",
    }
    assert set(real_statsports.columns) == expected
    assert COL_NAME not in real_statsports.columns
    assert isinstance(real_statsports, pd.DataFrame)
    assert not isinstance(real_statsports.attrs["statsports"]["source"], io.IOBase)
