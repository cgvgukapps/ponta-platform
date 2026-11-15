"""Pass-receipt stitching via next-event start."""

import numpy as np
import pandas as pd

from plaiq_engine.analytics.event_stitch import (
    build_pass_receipt_lookup,
    evaluate_pass_receipt,
    patch_pass_receipt_receiver,
)


def _pass_then_carry():
    return pd.DataFrame([
        {
            "Type": "PASS", "Team": "Home", "From": "14", "To": "15",
            "Period": 1,
            "Start Frame": 100, "End Frame": 107,
            "Start Time [s]": 10.0, "End Time [s]": 10.7,
            "Start X": 0.0, "Start Y": 0.0,
            "End X": 30.0, "End Y": 0.0,
        },
        {
            "Type": "CARRY", "Team": "Home", "From": "15", "To": np.nan,
            "Period": 1,
            "Start Frame": 121, "End Frame": 130,
            "Start Time [s]": 12.1, "End Time [s]": 13.0,
            "Start X": 30.0, "Start Y": 0.0,
            "End X": 32.0, "End Y": 1.0,
        },
    ])


def test_pass_receipt_stitch_valid():
    ev = _pass_then_carry()
    rec = evaluate_pass_receipt(ev.iloc[0], ev.iloc[1], 0, 1)
    assert rec["stitch_valid"] is True
    assert rec["identity_link"] == "pass_receipt"
    assert rec["receipt_start_frame"] == 121


def test_build_lookup_indexes_pass():
    ev = _pass_then_carry()
    ev.index = [10, 11]
    lkp = build_pass_receipt_lookup(ev)
    assert 10 in lkp
    assert lkp[10]["stitch_valid"] is True


def test_patch_receiver_updates_end_row_only():
    ev = _pass_then_carry()
    ev.index = [0, 1]
    rec = build_pass_receipt_lookup(ev)[0]

    home = pd.DataFrame([
        {"Frame": 107, "Home_14_x": 0.0, "Home_14_y": 0.0,
         "Home_15_x": 5.0, "Home_15_y": 0.0, "ball_x": 30.0, "ball_y": 0.0},
        {"Frame": 121, "Home_14_x": 1.0, "Home_14_y": 0.0,
         "Home_15_x": 29.5, "Home_15_y": 0.2, "ball_x": 30.1, "ball_y": 0.0},
    ]).set_index("Frame")
    away = home.copy()

    end_h, end_a = patch_pass_receipt_receiver(
        None, None, home, away, ev.iloc[0], rec,
    )
    assert end_h["Home_15_x"] == 29.5
    assert end_h["Home_14_x"] == 0.0
