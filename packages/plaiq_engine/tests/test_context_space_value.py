"""P7 space, value and game-state context on the synthetic eleven-a-side match (fictional players, no sample data)."""

import numpy as np
from context_fixture import eleven_a_side
from plaiq_engine.context.possession import PossessionPackage
from plaiq_engine.context.registry import Registry
from plaiq_engine.context.space_value import (
    SpaceValuePackage,
    adjusted_choice,
    game_state_weight,
)
from plaiq_engine.context.structure import StructurePackage
from plaiq_engine.context.thresholds import values

CFG = values()


def test_game_state_weight_lookup():
    assert game_state_weight(0, 30, CFG) == 1.0
    assert game_state_weight(1, 30, CFG) == CFG["p7_lambda_leading"] and game_state_weight(2, 80, CFG) == CFG["p7_lambda_leading_late"]
    assert game_state_weight(-1, 30, CFG) == CFG["p7_lambda_trailing"] and game_state_weight(-1, 90, CFG) == CFG["p7_lambda_trailing_late"]
    assert game_state_weight(float("nan"), 30, CFG) == 1.0  # the score unknown: neutral


def test_adjusted_choice_weighs_the_cost_of_losing_the_ball():
    cost = np.array([0.0, 0.02, 0.30])  # losing the ball at cell 2 gives away a lot
    opts = {"cells": [np.array([0]), np.array([2])], "p": [np.array([0.95]), np.array([0.6])],
            "values": [np.array([0.010]), np.array([0.030])], "chosen": 1}
    # Choice alone: 0.6 x 0.030 = 0.018 against 0.0095, so the risky pass is the best option.
    level = adjusted_choice(opts, cost, 1.0)
    assert level["u_best"] > 0 and level["adjusted"] < 1.0  # with the cost of losing it, the safe pass is better
    protect = adjusted_choice(opts, cost, 1.5)
    assert protect["adjusted"] < level["adjusted"] or np.isnan(protect["adjusted"])  # a leading team weighs the risk more
    assert adjusted_choice({**opts, "chosen": 0}, cost, 1.0)["adjusted"] == 1.0
    # The cost weight scales the cost term: at 0 the score is the value ranking alone, and a lower weight leans less to safety.
    assert adjusted_choice(opts, cost, 1.0, 0.0)["u_best"] == max(float(np.max(np.asarray(p) * np.asarray(v))) for p, v in zip(opts["p"], opts["values"], strict=True))
    assert adjusted_choice(opts, cost, 1.0, 0.5)["u_chosen"] >= level["u_chosen"]


def test_package_fields_and_entries():
    res = Registry([PossessionPackage(), StructurePackage(), SpaceValuePackage()]).run(eleven_a_side(seconds=60.0))
    f = res.frames
    open_ = f["p7_home_high_value_open"].dropna()
    assert len(open_) >= 5 and open_.between(0, 1).all()
    assert f["p7_game_state_weight"].dropna().eq(1.0).all()  # no logged feed: the score is unknown, so neutral
    assert f["p7_open_danger_zone"].dropna().str.endswith(("defensive", "middle", "attacking")).all()
    states = {e.metric: e.state for e in res.entries}
    assert states["space"] in ("full", "degraded") and states["adjusted_choice"] == "degraded"
    assert not [c for c in f.columns if c.startswith("p7_") and c.endswith(("_x", "_y", "_cell"))]
    assert res.outputs["p7"].segments["adjusted_choice"]["cost_weight"].eq(0.5).all() or res.outputs["p7"].segments["adjusted_choice"].empty
