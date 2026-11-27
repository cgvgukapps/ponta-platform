"""
A value-level snapshot of the headline engine outputs on the licence-free synthetic match.

Review finding SE-01: the only value-level comparison of Plaiq's metrics was the opt-in prototype
equivalence test, whose goldens live outside the repository. This snapshot is small, derived from
fictional players only, and tracked in ``tests/golden/synthetic/engine.json`` so CI checks it on
every push. Recapture with ``python tools/synthetic_golden.py --capture`` only when a change to an
engine value is intended, and say so in the commit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Frames whose per-frame surfaces are snapshotted (the synthetic match runs frames 1 to 250).
FRAMES = (1, 60, 120, 180, 240)
#: Every this-many hex cell of a surface is kept, beside the surface's summary statistics.
CELL_STEP = 97
DECIMALS = 9


def _num(v):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), DECIMALS)
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    return v if isinstance(v, str) else str(v)


def _table(df: pd.DataFrame) -> dict:
    """Columns and rows, with numbers rounded; arrays and nested objects dropped."""
    keep = [c for c in df.columns if not df[c].map(lambda v: isinstance(v, (list, tuple, dict, np.ndarray))).any()]
    rows = [[_num(v) for v in r] for r in df[keep].itertuples(index=False, name=None)]
    return {"columns": [str(c) for c in keep], "rows": rows}


def _surface(arr) -> dict | None:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float).ravel()
    fin = a[np.isfinite(a)]
    return {"n": int(a.size), "finite": int(fin.size),
            "sum": _num(fin.sum()) if fin.size else None,
            "min": _num(fin.min()) if fin.size else None, "max": _num(fin.max()) if fin.size else None,
            "sample": [_num(v) for v in a[::CELL_STEP]]}


def snapshot(analysis) -> dict:
    """The snapshot of one ``MatchAnalysis``."""
    fa = analysis.frame_analytics
    press = fa.loc[fa["Frame"].isin(FRAMES)].sort_values(["Frame", "team", "jersey"], kind="stable")
    press = press[[c for c in ("Frame", "team", "jersey", "pressure", "pressure_zonal", "in_possession") if c in press.columns]]
    out = {
        "event_pov": _table(analysis.event_pov()),
        "event_alternatives": _table(analysis.event_alternatives()),
        "movement_value": _table(analysis.movement_value),
        "pass_completion": _table(analysis.pass_completion),
        "decision_quality_v2": _table(analysis.decision_quality_v2),
        "gain_v2": _table(analysis.gain_v2),
        "pressure": _table(press.reset_index(drop=True)),
        "control": {},
        "value_surface": {},
    }
    for f in FRAMES:
        control, *_ = analysis.frame_control(f)
        out["control"][str(f)] = _surface(control)
        out["value_surface"][str(f)] = _surface(analysis.value_surface(f))
    return out


def compare(gold, new, path: str = "", rtol: float = 1e-6, atol: float = 1e-7) -> list[str]:
    """Differences between two snapshots, as readable paths (empty when they agree)."""
    if isinstance(gold, dict) and isinstance(new, dict):
        diffs = [f"{path}/{k}: missing" for k in gold.keys() - new.keys()]
        diffs += [f"{path}/{k}: new" for k in new.keys() - gold.keys()]
        for k in sorted(gold.keys() & new.keys()):
            diffs += compare(gold[k], new[k], f"{path}/{k}", rtol, atol)
        return diffs
    if isinstance(gold, list) and isinstance(new, list):
        if len(gold) != len(new):
            return [f"{path}: length {len(gold)} -> {len(new)}"]
        out = []
        for i, (g, n) in enumerate(zip(gold, new)):
            out += compare(g, n, f"{path}[{i}]", rtol, atol)
            if len(out) > 20:
                return out
        return out
    if isinstance(gold, float) or isinstance(new, float):
        if gold is None or new is None or not isinstance(gold, (int, float)) or not isinstance(new, (int, float)):
            return [] if gold == new else [f"{path}: {gold!r} -> {new!r}"]
        return [] if abs(gold - new) <= atol + rtol * abs(gold) else [f"{path}: {gold!r} -> {new!r}"]
    return [] if gold == new else [f"{path}: {gold!r} -> {new!r}"]
