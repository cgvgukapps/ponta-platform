"""One clock convention (review DE-09): elapsed match time, and the schema refuses a clock that runs backwards."""

import numpy as np
import pandas as pd
import pytest
from matrix_engine.clock import elapsed_match_time, runs_backwards
from matrix_engine.schema import SchemaError, coerce_schema, validate_schema


def test_a_period_relative_clock_becomes_elapsed_time():
    period = pd.Series([1, 1, 1, 2, 2, 3])
    within = pd.Series([0.0, 10.0, 2801.0, 0.0, 5.0, 0.0])  # a first half with 101 s of stoppage time
    t = elapsed_match_time(period, within, 1.0)
    assert t.tolist() == [0.0, 10.0, 2801.0, 2802.0, 2807.0, 2808.0]
    assert not runs_backwards(np.arange(6), t.to_numpy())
    old = within + period.map({1: 0.0, 2: 2700.0, 3: 5400.0})  # the fixed 45:00 the readers used
    assert runs_backwards(np.arange(6), old.to_numpy())


def _tracking(times):
    n = len(times)
    return coerce_schema(pd.DataFrame({"timestamp": times, "frame": np.arange(n), "player_id": "home_9", "team": "home",
                                       "x": 0.0, "y": 0.0, "source": "s", "sample_rate_hz": 10.0}))


def test_the_schema_refuses_a_clock_that_runs_backwards():
    validate_schema(_tracking([0.0, 0.1, 0.2, 0.3]))
    with pytest.raises(SchemaError, match="runs backwards"):
        validate_schema(_tracking([0.0, 2801.0, 2700.0, 2700.1]))
