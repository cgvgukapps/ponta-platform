"""
Shared fixtures for the matrix_engine tests.

The expensive objects — a Metrica window and the full reconciliation runs built
from it — are session-scoped, because loading a sample game is a few seconds and
a whole pipeline run a few more. Everything derived from them is read-only in
the tests, so sharing is safe.
"""

import os

import numpy as np
import pytest

from matrix_engine import UnifiedTrackingEngine
from matrix_engine.providers import load_metrica_source
from matrix_engine.sample_data import sample_data_dir
from matrix_engine.testing import DegradationParams, degrade

#: Metrica sample game and window used by every Track B test. 180 s is long
#: enough for six alignment windows to be meaningful and short enough to keep
#: the suite quick.
GAME_ID = 1
WINDOW_START = 600.0
WINDOW_END = 780.0

#: SkillCorner frames read in Track A. The full 59k-frame file takes ~10 s to
#: parse; 3000 frames is 300 s of match, plenty to prove normalisation.
SKILLCORNER_MATCH_ID = 1886347
SKILLCORNER_MAX_FRAMES = 3000


@pytest.fixture(scope="session")
def data_dir():
    return sample_data_dir()


@pytest.fixture(scope="session")
def skillcorner_dir(data_dir):
    return os.path.join(data_dir, "skillcorner", "matches", str(SKILLCORNER_MATCH_ID))


@pytest.fixture(scope="session")
def skillcorner_match_id():
    return SKILLCORNER_MATCH_ID


@pytest.fixture(scope="session")
def skillcorner_max_frames():
    return SKILLCORNER_MAX_FRAMES


@pytest.fixture(scope="session")
def metrica_home(data_dir):
    """One team, one window of real Metrica tracking — the Track B ground truth."""
    return load_metrica_source(
        data_dir, GAME_ID, teams=("home",),
        start_time=WINDOW_START, end_time=WINDOW_END,
    )


@pytest.fixture(scope="session")
def metrica_both(data_dir):
    """Both teams, for tests that care about team handling."""
    return load_metrica_source(
        data_dir, GAME_ID, teams=("home", "away"),
        start_time=WINDOW_START, end_time=WINDOW_END,
    )


def _params(**overrides):
    """Standard degradation, overridable per fixture."""
    base = dict(
        target_hz=10.0,
        position_noise_m=0.5,
        dropout_rate=0.10,
        n_dropout_bursts=2,
        dropout_burst_s=4.0,
        shuffle_identities=True,
        time_offset_s=2.5,
        time_drift_rate=0.0,
        seed=7,
    )
    base.update(overrides)
    return DegradationParams(**base)


def _run(ground_truth, synthetic):
    engine = UnifiedTrackingEngine()
    engine.add_source("metrica", ground_truth)
    engine.add_source(synthetic.data["source"].iloc[0], synthetic.data)
    return engine.reconcile(reference="metrica")


#: Drift injected into the "drifting sensor" fixture, in seconds per second.
#:
#: Larger than a real sensor's, deliberately. What drift detection can resolve
#: is the error a clock *accumulates*, not its rate, and this window is 180 s
#: rather than a 90-minute match. 2.5e-3 over ~150 s accumulates about 0.37 s —
#: roughly what a realistically bad 70 ppm clock would accumulate across a full
#: match. Testing a full-match rate over a three-minute window would be testing
#: that a quantity below the noise floor cannot be measured, which is true but
#: not interesting. ``test_quality.py`` covers the sensitivity limit directly.
DRIFT_RATE = 2.5e-3


@pytest.fixture(scope="session")
def drifted(metrica_home):
    """Ground truth plus a degraded source whose clock drifts, and the run over both."""
    synthetic = degrade(metrica_home, _params(time_drift_rate=DRIFT_RATE))
    return {
        "truth": metrica_home,
        "synthetic": synthetic,
        "result": _run(metrica_home, synthetic),
    }


@pytest.fixture(scope="session")
def clean(metrica_home):
    """The same, with a constant offset but no drift — the negative control."""
    synthetic = degrade(metrica_home, _params(time_drift_rate=0.0, seed=11))
    return {
        "truth": metrica_home,
        "synthetic": synthetic,
        "result": _run(metrica_home, synthetic),
    }


@pytest.fixture(scope="session")
def dual_degraded(metrica_home):
    """
    Two independently degraded sources, neither of them ground truth.

    The headline Track B scenario fuses real Metrica against a degraded copy of
    itself, which means one input *is* the answer and fusion can win simply by
    trusting it. This fixture removes that advantage: both feeds are noisy, so
    beating them requires the weighting to actually combine information.
    """
    source_a = degrade(
        metrica_home,
        _params(position_noise_m=0.6, dropout_rate=0.05, n_dropout_bursts=1,
                dropout_burst_s=3.0, time_offset_s=1.2, shuffle_identities=False, seed=1),
        source_name="gps_a",
    )
    source_b = degrade(
        metrica_home,
        _params(position_noise_m=0.6, dropout_rate=0.05, n_dropout_bursts=1,
                dropout_burst_s=3.0, time_offset_s=-0.8, shuffle_identities=True, seed=2),
        source_name="gps_b",
    )
    engine = UnifiedTrackingEngine(target_hz=10.0)
    engine.add_source("gps_a", source_a.data)
    engine.add_source("gps_b", source_b.data)
    result = engine.reconcile(reference="gps_a")
    return {
        "truth": metrica_home,
        "a": source_a,
        "b": source_b,
        # gps_a is the reference and its clock was left uncorrected, so the
        # output timeline is the true clock shifted by gps_a's offset.
        "truth_clock_shift": source_a.params.time_offset_s,
        "result": result,
    }


@pytest.fixture(scope="session")
def position_rmse():
    """
    Callable scoring a frame's positions against ground truth.

    Ground truth is interpolated onto whatever timestamps the frame carries, so
    the same helper works for a resampled source and for fused output. Returns
    ``(rmse_metres, n_points_compared)``.
    """

    def _rmse(df, truth, *, clock_shift: float = 0.0):
        lookup = {}
        for player_id, block in truth.groupby("player_id", sort=False):
            block = block.sort_values("timestamp")
            lookup[player_id] = (
                block["timestamp"].to_numpy(dtype=float) + clock_shift,
                block["x"].to_numpy(dtype=float),
                block["y"].to_numpy(dtype=float),
            )

        squared = []
        for player_id, block in df.groupby("player_id", sort=False):
            entry = lookup.get(player_id)
            if entry is None:
                continue
            t_true, x_true, y_true = entry
            t = block["timestamp"].to_numpy(dtype=float)
            inside = (t >= t_true[0]) & (t <= t_true[-1])
            if not inside.any():
                continue
            t = t[inside]
            dx = np.interp(t, t_true, x_true) - block["x"].to_numpy(dtype=float)[inside]
            dy = np.interp(t, t_true, y_true) - block["y"].to_numpy(dtype=float)[inside]
            squared.append(dx**2 + dy**2)

        if not squared:
            return float("nan"), 0
        errors = np.concatenate(squared)
        return float(np.sqrt(errors.mean())), int(errors.size)

    return _rmse
