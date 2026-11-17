"""
probe.py — a trivial context package that proves the P0 path end to end.

It writes one observed field (whether the ball was detected at the grid point's native
frame) and one Data check entry with the coverage line. It is registered in tests and by
the golden harness, never in the product pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from plaiq_engine.context.registry import (
    ContextField,
    ContextInput,
    ContextPackage,
    PackageOutput,
)


class ProbePackage(ContextPackage):
    name = "probe"
    version = "0.1.0"
    fields = (ContextField("probe_ball_detected", "bool", "observed",
                           "True when the ball has a position at the grid point's native frame."),)

    def compute(self, ctx: ContextInput) -> PackageOutput:
        seen = np.isin(ctx.grid["native_frame"].to_numpy(), ctx.match.ball["frame"].to_numpy())
        share = float(seen.mean()) if len(seen) else 0.0
        return PackageOutput(
            frames=pd.DataFrame({"probe_ball_detected": seen}),
            entries=[self.entry("ball_detected", "Ball detected (probe)", "full",
                                "Test package: ball detection on the context grid.",
                                measure=f"{100 * share:.1f}% of grid points", coverage=f"{len(seen)} grid points")],
            measurements={"ball_detected_share": round(share, 4)},
        )
