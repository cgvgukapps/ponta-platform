# Golden outputs from the frozen prototype

Recorded **before any code moved**, so the extraction can be proven behaviour-preserving. The Plaiq-via-Matrix equivalence test compares against these.

- Source: `PONTA-PROTOTYPE` tag `prototype-validation-final` (commit `41bd812`, 964 tests passing).
- Data: open sample data only. Metrica Sample Game 3 and SkillCorner open-data match 1886347. The unlicensed STATSports export is not used.
- No player names. Rows carry jersey slots and pseudonymous ids (`Player 8`).

## Regenerate

Use the prototype's own interpreter (pinned dependencies). The prototype working tree must be clean.

```bash
/Users/cicerovieira/PONTA-PROTOTYPE/.venv/bin/python tools/golden/capture_prototype_golden.py
```

About 10 minutes (Metrica about 7.5, SkillCorner about 2.5). Output goes to `tests/golden/prototype/`. Only `manifest.json` is tracked in git; the data files (about 12 MB, mostly `frame_surfaces.npz`) are ignored. A quick wiring check that is **not** golden: `--smoke --out /tmp/golden-smoke` (about 1 minute).

## What is captured, per match

| File | Prototype function | Content |
|---|---|---|
| `event_delta_v.parquet` | `tag_events_delta_v` | Events with ΔV (static xT) |
| `movement_value.parquet` | `compute_movement_value` | Per (event, player) |
| `pov.parquet` | `compute_event_pov` | Per-player POV and decision quality |
| `pov_alternatives.parquet` | `compute_event_pov` | Carry alternatives |
| `frame_pressure.parquet` | `compute_pressure` + `tag_possession` | Per-player pressure, sampled frames |
| `frame_surfaces.npz` | `compute_frame_control`, `smooth_control`, `pressure_field`, `compute_value` | Control, travel-time and value arrays, sampled frames (7,463 hexes) |

The loaders mirror `app.py` step for step (it cannot be imported headlessly because it calls Streamlit at import time). SkillCorner uses `fps=10.0` for POV, exactly as the app does; Metrica uses the default frame rate.

## Findings recorded while capturing

1. **Row order of `compute_event_pov` is not deterministic across runs; values are.** Two independent runs were identical once sorted on `(event_idx, team, jersey)`, which is unique. The golden `pov.parquet` is stored sorted. Equivalence tests must compare on that key, never on row order. **The semantic builder must sort explicitly** and cannot rely on engine row order.
2. **The ball is undetected in a large share of frames**: 32.2% of Metrica Game 3 and 26.4% of the SkillCorner match. The prototype returns an all-zero control grid and no travel times for those frames. Surfaces are sampled only from ball-present frames; five ball-missing frames per match are recorded in `frame_surfaces.npz` (`ball_missing_frames`) as an edge case the Matrix path must match.
3. **Decision quality ranges up to 1.5** in both matches (minimum 0.0 and 0.027), so the value is not a 0 to 1 ratio. Check this against the rating bands before the UI or AI layer assumes a 0 to 1 scale.
4. **SkillCorner event `To` values are floats-as-strings** (`'1.0'`) while `From` is `'1'`. An identity-key type quirk to normalise, not to copy.
5. **SkillCorner has 18 + 18 player slots** (substitutes), but 22 players per frame in `frame_pressure`.

## Sampling

Frames: 40 evenly spaced ball-present frames plus up to 10 at evenly spaced event starts (deduplicated), and 5 ball-missing frames. Deterministic (`np.linspace` over sorted frame ids). Parameters are recorded in `manifest.json`.

## Tolerance guidance for the equivalence test

Not set here. The tolerance must be explicit and justified in `EXTRACTION_REPORT.md`. Compare event-level tables on their natural keys. Compare surfaces with `np.allclose` and report the maximum difference per array, rather than choosing a tolerance first.
