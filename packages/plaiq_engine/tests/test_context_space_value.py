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
    assert protect["u_chosen"] < level["u_chosen"]  # a leading team weighs the risk more
    assert protect["adjusted"] <= level["adjusted"] or np.isnan(protect["adjusted"])  # both 0 here: worth less than the risk
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


def test_adjusted_choice_stays_on_its_0_to_1_scale():
    """Random options never leave 0 to 1 (review SM-06); a chosen option worth less than its risk scores 0 and says so."""
    from plaiq_engine.context.space_value import CHOSEN_BELOW_RISK

    # The reviewer's case: the chosen pass is risky and worth little, the best one is safe.
    cost = np.array([0.4, 0.1])
    opts = {"cells": [np.array([0]), np.array([1])], "p": [np.array([0.3]), np.array([0.9])],
            "values": [np.array([0.2]), np.array([0.05])], "chosen": 0}
    r = adjusted_choice(opts, cost, 1.0)
    assert r["u_chosen"] < 0 < r["u_best"] and r["adjusted"] == 0.0 and r["reason"] == CHOSEN_BELOW_RISK
    rng = np.random.default_rng(11)
    for _ in range(500):
        k = int(rng.integers(2, 6))
        cost = rng.uniform(0, 0.5, k)
        opts = {"cells": [np.array([i]) for i in range(k)], "p": [rng.uniform(0, 1, 1) for _ in range(k)],
                "values": [rng.uniform(0, 0.3, 1) for _ in range(k)], "chosen": int(rng.integers(0, k))}
        a = adjusted_choice(opts, cost, float(rng.uniform(0.5, 1.5)), float(rng.uniform(0, 1)))["adjusted"]
        assert np.isnan(a) or 0.0 <= a <= 1.0
