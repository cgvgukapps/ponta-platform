#!/usr/bin/env python3
"""
Capture golden outputs from the frozen PONTA-PROTOTYPE.

Purpose
-------
The platform extraction must be behaviour-preserving. Before any code moves,
this script records what the prototype's Plaiq pipeline produces for the open
sample matches, so the Plaiq-via-Matrix equivalence test can compare against
the prototype's *direct-ingest* path.

What it does
------------
Mirrors, step for step, the loaders in the prototype's ``app.py`` (which cannot
be imported headlessly: it calls Streamlit at import time). Per match it
writes:

  event_delta_v.parquet     events tagged with ΔV (static xT)
  movement_value.parquet    compute_movement_value, per (event, player)
  pov.parquet               compute_event_pov, per-player POV + decision quality
  pov_alternatives.parquet  compute_event_pov, carry alternatives
  frame_pressure.parquet    per-player pressure rows for the sampled frames
  frame_surfaces.npz        control / travel-time / value arrays, sampled frames

plus one ``manifest.json`` at the output root (prototype commit and tag,
library versions, sample parameters, row counts, sha256 of every file).

Safety
------
* Read-only on the prototype. Bytecode writing is disabled. The prototype
  working tree must be clean unless ``--allow-dirty`` is given.
* Open sample data only (Metrica sample games, SkillCorner open data).
  The unlicensed STATSports export is not touched.
* No roster names are read or written. Rows carry jersey slots only.

Usage
-----
Run with the prototype's own interpreter, which has the pinned dependencies::

    /Users/cicerovieira/PONTA-PROTOTYPE/.venv/bin/python \\
        tools/golden/capture_prototype_golden.py

    # quick wiring check, NOT golden (first 15 events only):
    ... capture_prototype_golden.py --smoke --out /tmp/golden-smoke

Defaults: Metrica game 3 and SkillCorner match 1886347.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True

DEFAULT_PROTOTYPE = Path("/Users/cicerovieira/PONTA-PROTOTYPE")
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "tests" / "golden" / "prototype"
DEFAULT_MATCHES = ["metrica:3", "skillcorner:1886347"]
SCHEMA_VERSION = 1


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(proto: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(proto), *args], capture_output=True, text=True)
    return r.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prototype", type=Path, default=DEFAULT_PROTOTYPE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--matches", nargs="+", default=DEFAULT_MATCHES, help="provider:id, e.g. metrica:3 skillcorner:1886347")
    ap.add_argument("--frames", type=int, default=40, help="evenly spaced frames to sample (default 40)")
    ap.add_argument("--event-frames", type=int, default=10, help="extra frames at evenly spaced event starts (default 10)")
    ap.add_argument("--smoke", action="store_true", help="wiring check only: first 15 events, not golden")
    ap.add_argument("--allow-dirty", action="store_true", help="allow an uncommitted prototype working tree")
    args = ap.parse_args()

    proto: Path = args.prototype.resolve()
    if not (proto / "analytics").is_dir():
        log(f"ERROR: {proto} does not look like PONTA-PROTOTYPE")
        return 2

    dirty = git(proto, "status", "--porcelain")
    if dirty and not args.allow_dirty:
        log("ERROR: prototype working tree is not clean; refusing (use --allow-dirty to override):")
        print(dirty)
        return 2
    if args.smoke:
        log("SMOKE MODE: outputs are truncated and must not be used as golden data")

    sys.path.insert(0, str(proto))
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd

    from analytics.hex_analytics import assign_all_players, compute_pressure, infer_possession, tag_possession
    from analytics.hex_value import (
        compute_movement_delta_v,
        compute_movement_value,
        compute_value,
        pressure_field,
        smooth_control,
        tag_events_delta_v,
        xT_surface,
    )
    from analytics.pass_option_value import compute_event_pov
    from analytics.pitch_control import DEFAULT_PROFILE, build_pitch_hex_grid, compute_frame_control
    from roster.player_db import get_profiles, load_sc_roster

    processed = proto / "sample-data" / "data" / "processed"

    # ---- shared, matches app.get_pcz_grid() / get_xT_surfaces() --------------
    R = float(np.sqrt(2.0 / (3.0 * np.sqrt(3))))
    hex_centers = build_pitch_hex_grid(R)
    xT_home = xT_surface(hex_centers, attacking_right=True)
    xT_away = xT_surface(hex_centers, attacking_right=False)

    def jerseys(df, team):  # app.get_player_jerseys
        xs = [c for c in df.columns if c.startswith(f"{team}_") and c.endswith("_x")]
        return [c.split("_")[1] for c in xs]

    def sc_profiles(match_id, team_label, js):  # app.load_sc_pov_scores_cache
        roster = load_sc_roster(match_id)
        if roster is None:
            return {j: DEFAULT_PROFILE for j in js}
        players = {str(p.get("primary_slot", "")): p for p in roster["teams"].get(team_label, {}).get("players", [])}
        out = {}
        for j in js:
            p = players.get(str(j))
            out[j] = (
                {k: p.get(k, DEFAULT_PROFILE[k]) for k in ("v_max", "a_max", "reaction_time")} if p else DEFAULT_PROFILE
            )
        return out

    def load_match(kind, mid):
        """app.load_game / app.load_sc_match, including the ΔV tag."""
        prefix = f"Game_{mid}" if kind == "metrica" else f"SC_{mid}"
        home = pd.read_csv(processed / f"{prefix}_tracking_home.csv", index_col="Frame")
        away = pd.read_csv(processed / f"{prefix}_tracking_away.csv", index_col="Frame")
        events = pd.read_csv(processed / f"{prefix}_events.csv")
        events = tag_events_delta_v(events, xT_home, xT_away, hex_centers, R)
        return home, away, events

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "smoke (NOT golden)" if args.smoke else "golden",
        "prototype": {
            "path": str(proto),
            "commit": git(proto, "rev-parse", "HEAD"),
            "tag": git(proto, "describe", "--tags", "--exact-match", "HEAD") or None,
            "tree_clean": not dirty,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "sampling": {"even_frames": args.frames, "event_frames": args.event_frames, "method": "np.linspace over common frames; evenly spaced event Start Frames"},
        "pov_fps": {"metrica": "pitch_config.DEFAULT_FPS (default argument, as in app.load_pov_scores)", "skillcorner": 10.0},
        "canonicalisation": {
            "pov.parquet": "sorted by (event_idx, team, jersey); the prototype's row order is not deterministic across runs, values are",
            "pov_alternatives.parquet": "sorted by (event_idx, hex_x, hex_y) when that key is unique",
        },
        "matches": {},
        "files": {},
    }
    t_total = time.time()

    for spec in args.matches:
        kind, mid_s = spec.split(":")
        mid = int(mid_s)
        label = f"{kind}_{mid}"
        mdir = args.out / label
        mdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        log(f"== {label}: loading")
        home, away, events = load_match(kind, mid)
        home_j, away_j = jerseys(home, "Home"), jerseys(away, "Away")
        n_frames = len(home)
        log(f"   {n_frames} frames, {len(events)} events, {len(home_j)}+{len(away_j)} player slots")

        # analytics (app.load_analytics / load_sc_analytics_cache)
        log("   assigning hexes and computing pressure")
        hex_df = assign_all_players(home, away, home_j, away_j, R)
        press_df = compute_pressure(hex_df)
        poss = infer_possession(events, n_frames)
        tagged = tag_possession(press_df, poss)
        analytics = tagged.merge(hex_df[["Frame", "team", "jersey", "q", "r"]], on=["Frame", "team", "jersey"], how="left")

        log("   movement ΔV")
        analytics_val = compute_movement_delta_v(analytics, xT_home, xT_away, hex_centers, R)
        ev_for_value = events.head(15) if args.smoke else events
        log(f"   movement value ({len(ev_for_value)} events)")
        movement = compute_movement_value(analytics_val, ev_for_value, xT_home, xT_away, hex_centers, R)

        if kind == "metrica":
            prof_h = get_profiles(mid, "home", home_j)
            prof_a = get_profiles(mid, "away", away_j)
            pov_kwargs = {}
        else:
            prof_h = sc_profiles(mid, "Home", home_j)
            prof_a = sc_profiles(mid, "Away", away_j)
            pov_kwargs = {"fps": 10.0}

        log("   event POV and decision quality (slow)")
        t_pov = time.time()
        pov_df, alt_df = compute_event_pov(
            analytics_val, ev_for_value, home, away, home_j, away_j,
            xT_home, xT_away, hex_centers, R, prof_h, prof_a, **pov_kwargs,
        )
        log(f"   POV done in {time.time() - t_pov:.0f}s: {len(pov_df)} pov rows, {len(alt_df)} alternative rows")

        # The prototype's row ORDER from compute_event_pov varies between runs (values do
        # not). Verified across independent runs: identical after sorting on the natural
        # key. Canonicalise so golden files are byte-stable and diffable, and so the
        # equivalence test never depends on engine row order.
        pov_key = ["event_idx", "team", "jersey"]
        assert not pov_df.duplicated(pov_key).any(), "pov natural key is not unique"
        pov_df = pov_df.sort_values(pov_key, kind="stable").reset_index(drop=True)
        alt_key = ["event_idx", "hex_x", "hex_y"]
        if len(alt_df) and not alt_df.duplicated(alt_key).any():
            alt_df = alt_df.sort_values(alt_key, kind="stable").reset_index(drop=True)

        # ---- deterministic frame sample --------------------------------------
        # The ball is undetected in a large share of frames (about a quarter to a third
        # in these samples). The app returns an uncontrolled grid there, so surfaces are
        # sampled only from ball-present frames, and a few ball-missing frames are
        # recorded separately as an edge case for the equivalence test.
        common = np.array(sorted(set(home.index) & set(away.index)))
        ball_ok_mask = away.loc[common, "ball_x"].notna().to_numpy() & away.loc[common, "ball_y"].notna().to_numpy()
        ball_ok = common[ball_ok_mask]
        ball_missing = common[~ball_ok_mask]
        even = ball_ok[np.linspace(0, len(ball_ok) - 1, args.frames).astype(int)]
        starts = np.sort(events["Start Frame"].dropna().astype(int).unique())
        starts = starts[np.isin(starts, ball_ok)]
        ev_pick = starts[np.linspace(0, len(starts) - 1, min(args.event_frames, len(starts))).astype(int)] if len(starts) else np.array([], dtype=int)
        sample = np.array(sorted(set(even.tolist()) | set(ev_pick.tolist())))
        miss_pick = (
            ball_missing[np.linspace(0, len(ball_missing) - 1, min(5, len(ball_missing))).astype(int)]
            if len(ball_missing) else np.array([], dtype=int)
        )
        log(f"   sampling {len(sample)} ball-present frames for surfaces, {len(miss_pick)} ball-missing frames as edge cases")

        frames_ok, ctrl, th, ta, tb, val, val_poss = [], [], [], [], [], [], []
        for f in sample:
            row_h, row_a = home.loc[f], away.loc[f]
            control, t_ball, t_h, t_a, _, _ = compute_frame_control(
                row_h, row_a, home_j, away_j, hex_centers, profiles_home=prof_h, profiles_away=prof_a,
            )
            if t_h is None:  # ball not detected: the app returns an uncontrolled grid
                continue
            side = poss.get(f)
            v = np.full(len(hex_centers), np.nan, dtype=np.float32)
            if side in ("Home", "Away"):
                vh, va = smooth_control(t_h, t_a, t_ball)
                v_ctrl = vh if side == "Home" else va
                fp = analytics[analytics["Frame"] == f][["team", "jersey", "pressure"]]
                disc = pressure_field(row_h, row_a, home_j, away_j, hex_centers, fp, side)
                v = np.asarray(compute_value(xT_home if side == "Home" else xT_away, v_ctrl, disc), dtype=np.float32)
            frames_ok.append(int(f))
            ctrl.append(np.asarray(control, dtype=np.float32))
            th.append(np.asarray(t_h, dtype=np.float32))
            ta.append(np.asarray(t_a, dtype=np.float32))
            tb.append(np.asarray(t_ball, dtype=np.float32))
            val.append(v)
            val_poss.append(side if side in ("Home", "Away") else "")

        # ---- write -----------------------------------------------------------
        outputs = {
            "event_delta_v.parquet": events,
            "movement_value.parquet": movement,
            "pov.parquet": pov_df,
            "pov_alternatives.parquet": alt_df,
            "frame_pressure.parquet": analytics[analytics["Frame"].isin(frames_ok)].reset_index(drop=True),
        }
        for name, df in outputs.items():
            df.to_parquet(mdir / name, index=False)
        np.savez_compressed(
            mdir / "frame_surfaces.npz",
            frames=np.array(frames_ok, dtype=np.int64),
            hex_centers=hex_centers.astype(np.float32),
            control=np.stack(ctrl), t_home=np.stack(th), t_away=np.stack(ta), t_ball=np.stack(tb),
            value=np.stack(val), value_possession=np.array(val_poss),
            # Frames where the ball is undetected; the prototype returns an all-zero control
            # grid and no travel times for these. Edge case for the equivalence test.
            ball_missing_frames=miss_pick.astype(np.int64),
        )

        manifest["matches"][label] = {
            "provider": kind,
            "match_id": mid,
            "frames": int(n_frames),
            "events_total": int(len(events)),
            "events_used_for_value_and_pov": int(len(ev_for_value)),
            "players": {"home": home_j, "away": away_j},
            "hex_count": int(len(hex_centers)),
            "frames_ball_present": int(len(ball_ok)),
            "frames_ball_missing": int(len(ball_missing)),
            "ball_missing_frames_recorded": [int(f) for f in miss_pick],
            "sampled_frames_requested": int(len(sample)),
            "sampled_frames_with_surfaces": len(frames_ok),
            "frames_with_value_surface": int(sum(1 for p in val_poss if p)),
            "rows": {n: int(len(d)) for n, d in outputs.items()},
            "seconds": round(time.time() - t0, 1),
        }
        for p in sorted(mdir.iterdir()):
            manifest["files"][f"{label}/{p.name}"] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
        log(f"== {label}: done in {time.time() - t0:.0f}s")

    manifest["total_seconds"] = round(time.time() - t_total, 1)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    log(f"wrote {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
