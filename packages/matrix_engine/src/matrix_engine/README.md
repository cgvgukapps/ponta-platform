# Matrix prototype — what it proves and what it doesn't

Validation prototype for **Matrix**, the vendor-agnostic tracking reconciliation
engine described in `ponta-blueprint/Matrix.md`. It exists to answer one
question before any further investment: *does the reconciliation approach
actually work?* It is not a production build, and several of the numbers below
are better than the real problem warrants — where that is true, it says so.

Everything in this document runs offline against data already in `sample-data/`.
No vendor credentials, no network calls, no new dependencies. (The STATSports
follow-up described below is the one exception: it fetches an external file, and
skips cleanly when it cannot.)

```
python scripts/matrix_demo.py     # end-to-end run + metrics + plot
pytest tests/matrix_engine        # 223 tests
pytest                            # 382 tests (159 pre-existing, unchanged)
```

A follow-up added a real STATSports GPS provider, which partly addresses the
"no wearable data" gap below and adds an optional `native_quality` signal to the
confidence model. Its results and limits are in
[`STATSPORTS_ADDENDUM.md`](STATSPORTS_ADDENDUM.md); everything in this document
is unchanged by it, and 17 of the 382 tests skip when that external export is not
reachable.

## Headline results

Measured on Metrica sample game 1, 600–900 s, home team (82,500 ground-truth
player-frames at 25 Hz). Reproduce with `python scripts/matrix_demo.py`.

| Capability | Result |
|---|---|
| Clock offset recovery | **3 ms** error against an injected 2.65 s offset |
| Clock drift recovery | **+1.009e-03 s/s** against an injected 1.0e-03 s/s (0.9% error) |
| Identity re-matching | **100%** (11/11), median runner-up margin 14.7 m |
| Fusion — truth + degraded copy | 0.633 m → **0.001 m** RMSE (99.8% reduction) |
| Fusion — two degraded sources | 0.611 m → **0.461 m** RMSE (24.5% vs the better input) |
| Confidence separation | **0.181** inside injected dropout gaps vs **0.850** elsewhere |
| Drift detection | Flagged at **27.4 sigma** on the drifted run, **not flagged** on the clean run |
| Runtime | **2.3 s** for a 5-minute, 11-player reconciliation |

## What was actually proven

### 1. Normalisation works across genuinely different real vendors (Track A)

Metrica sample game 1 and SkillCorner open-data match 1886347 are different
fixtures in different formats, and both come through their adapters into one
schema:

| | Metrica | SkillCorner |
|---|---|---|
| Format | 3-header-row CSV, normalised 0–1, top-left origin | JSONL, metres, per-period clock |
| Rate | 25 Hz | 10 Hz |
| Pitch | 106 × 68 m | 104 × 68 m (read from match metadata) |
| Output | 99,000 rows, 22 players | 63,008 rows, 22 players |
| x range | −49.98 … 56.10 m | −51.72 … 50.71 m |

After normalisation the two are structurally interchangeable: identical columns
and dtypes, both in centred metres, both direction-normalised so home attacks
+x, both stackable into one valid frame. `test_providers.py` asserts this
rather than eyeballing it, including a units check that would catch a vendor
left in normalised or yard coordinates.

This is the least glamorous result and the most certain one. It is also the
capability with the clearest commercial value, since it is the part clubs
currently do by hand.

### 2. Clock alignment works, and needs two passes to be useful

Alignment has to happen before identity matching (you cannot compare player 7
to player 7 when the feeds disagree on who player 7 is), but the sharpest
alignment signal is matched players. Matrix resolves this by running sync
twice:

| | Offset error | Window-to-window scatter | Drift estimate (true 1.0e-03) |
|---|---|---|---|
| Pass 1 — team centroid/spread | 310 ms | 863 ms | +8.2e-05 |
| Pass 2 — matched players | **4.6 ms** | **5.0 ms** | **+1.04e-03** |

The first pass uses per-team centroid and spread, which are invariant to
relabelling players. It is good enough to bootstrap identity matching and
nothing more. The second pass re-estimates from matched trajectories and is
roughly 70× tighter, which is what makes the drift slope meaningful — on the
clean run the coarse pass reports a spurious −1.16e-03 s/s, while the refined
pass reports −7.7e-05 s/s.

Running sync once would have produced a plausible-looking pipeline with an
alignment error large enough to dominate the fusion result.

### 3. Fusion reduces position error — with an important caveat

Two scenarios, because they prove different things.

**Ground truth + one degraded copy** (the scenario in the task spec): degraded
source alone 0.633 m RMSE, fused 0.001 m. That 99.8% reduction is real but
close to circular — one input *is* the ground truth, so a system that merely
identified the better source would score the same. It does confirm that
inverse-variance weighting correctly recognises which feed to trust (the weight
ratio lands at 2500:4, matching the 0.020 m vs 0.495 m estimated noise), which
would be a real failure if it got it wrong. It is not evidence of sensor
fusion.

**Two independently degraded sources, neither of them the truth** — the honest
test:

| Source | RMSE vs ground truth |
|---|---|
| gps_a alone | 0.696 m |
| gps_b alone | 0.611 m |
| **fused** | **0.461 m** |

A 24.5% reduction against the better of the two inputs, where averaging two
equally-noisy independent estimates caps out at 29.3%. The shortfall is real
dropout, interpolation across gaps, and residual clock error. Landing near the
theoretical limit is the actual evidence that the fusion math is right.

Source noise is estimated from the data, not declared — read off the second
difference of each trajectory, recovering 0.495 m for an injected 0.5 m and
staying within 15% across 0.3–1.0 m.

### 4. Confidence and quality flags track the real failures

Confidence is the product of three separately-reported factors (coverage, gap
distance, inter-source agreement), so a low score can be explained rather than
just observed. Inside deliberately injected dropout gaps it averages 0.181
against 0.850 elsewhere, and it falls monotonically with gap length.

Drift detection requires a slope that is both statistically distinguishable
from zero *and* large enough to matter. On the drifted run it fires at 27.4
sigma; on a clean run with the same 2.5 s constant offset it does not fire.
Both halves are tested, because a detector that flags everything is worthless.

Dropout detection found all 22 injected bursts (longest 4.20 s) plus the
incidental gaps from random dropout, measured on native-rate data — after
resampling the gaps are interpolated away and undetectable, which the tests pin
down explicitly.

## What was not proven

**No real multi-vendor overlap.** The single largest gap. Every fusion,
identity and drift number above comes from a synthetic second source derived
from the same recording. Two real vendors tracking one match would disagree in
ways this generator does not simulate: different calibration, systematic
coordinate bias, disagreeing on *which* players are on the pitch, and errors
correlated between players rather than independent.

**The identity result is much easier than the real problem.** Matching stays at
100% with 25 m of injected noise — larger than the pitch is wide — because
costs are mean distances over ~1800 frames and independent zero-mean noise is
exactly the error model averaging destroys. Real cross-vendor identity failures
come from correlated errors: a mirrored or rotated axis, sustained tracking
swaps between two nearby players, a player absent for a long stretch, or a
centre-back pair with near-identical paths. None are simulated. Treat "100%
accuracy" as evidence the Hungarian-on-trajectory-distance approach is sound in
principle, not as a measured expectation for real vendor pairs. The
runner-up margin (14.7 m under standard degradation, falling to 2.8 m at 25 m
noise) is the quantity to watch on real data.

**Drift detection has a stated sensitivity floor.** What is detectable is
*accumulated* error, not drift rate: on this 248 s span the floor is 8.07e-04
s/s. A realistic sensor drift of ~1e-04 s/s (100 ppm) is invisible over three
minutes and obvious over ninety, so the tests inject 2.5e-03 to accumulate a
comparable error in a short window. Detecting real drift needs full-match
recordings, which this prototype does not run.

**No wearable or GPS data exists in this repo.** `synthetic_gps` is a label on
downsampled, noised optical data. Nothing here has touched a real accelerometer
stream, its sampling irregularities, or its failure modes. *Partly addressed
since — `providers/statsports.py` now ingests a real STATSports Apex export, so
the ingest and its quality signals are real. There is still no optical source for
the same session, so every fusion, identity and drift number in this document
still comes from the synthetic generator. See
[`STATSPORTS_ADDENDUM.md`](STATSPORTS_ADDENDUM.md).*

**No performance or scale validation.** 2.3 s for 5 minutes of one team.
Nothing was measured at full-match, full-squad, or full-season scale, and the
identity cost matrix is O(players²) per team with a dense pairwise distance
computation that has not been profiled.

**Only two sources at a time.** The code paths generalise to more (weights,
attribution bitmasks and confidence all take an arbitrary count), but every
test and both demo scenarios use exactly two.

**Fusion is a weighted average, not a state estimator.** No motion model, no
Kalman filter, no outlier rejection beyond the noise-floor and gap weighting.
Deliberate for a first prototype, but a real system with a badly-behaved sensor
would want a filter that can reject an implausible jump rather than average it
in.

**Nothing legal was addressed.** The ToS and GDPR questions raised in
`Matrix.md` — whether fusing two vendors' proprietary feeds into a derived
dataset is permitted at all — are untouched, and remain the risk that could
invalidate the product independent of whether the engineering works.

## What would be needed to validate further

In rough order of how much each would tell you:

1. **One match recorded by two real vendors.** Everything uncertain above
   collapses to a measurement with this. Most realistically an optical provider
   plus a club's own GPS for a single match — the club already has both, so it
   needs a data-sharing agreement rather than new instrumentation.
2. **A real wearable export** (STATSports, Catapult, KINEXON) with its native
   rate, gaps and per-player files, to replace the `synthetic_gps` stand-in.
   *Partly done — see [`STATSPORTS_ADDENDUM.md`](STATSPORTS_ADDENDUM.md). The
   ingest is real; without a paired optical source it cannot replace the
   stand-in for fusion validation.*
3. **Full 90-minute recordings**, to test drift detection at realistic rates
   and to find out what the pipeline costs at scale.
4. **Ground truth for a segment** — surveyed positions or a high-accuracy
   reference over even a few minutes — to score fused output against something
   other than one of its own inputs.
5. **A vendor pair known to disagree about personnel**, to exercise the
   unmatched-player path, which is currently only tested synthetically.

## Dependencies

**None added.** The package uses `numpy`, `pandas` and `scipy`, all already in
`requirements.txt`. `scipy` supplies `linear_sum_assignment` for the Hungarian
assignment in `identity.py` and `savgol_filter` for the derivatives in
`export.py`, matching how `ingestion/metrica_velocities.py` already
differentiates. The demo script uses `matplotlib` (already a dependency) with
the `Agg` backend so it writes a file and never opens a window.

`matrix_engine/` imports nothing from Streamlit and nothing from `analytics/`.
It depends on `ingestion/`, `roster/` and `pitch_config.py` and never the
reverse, so extracting it into its own package later is a move, not a rewrite.

## Layout

```
matrix_engine/
  __init__.py           UnifiedTrackingEngine façade + ReconciliationResult
  schema.py             common normalized schema, validator, pitch-bounds check
  providers/
    metrica.py          wraps ingestion/metrica_io + metrica_velocities
    skillcorner.py      wraps ingestion/skillcorner_io
    statsports.py       real GPS export, fetched on access (see the addendum)
  sync.py               two-pass clock alignment + resampling to a common grid
  identity.py           trajectory-distance matching, Hungarian assignment
  fusion.py             noise estimation + inverse-variance weighted fusion
  confidence.py         per-frame coverage x gap x agreement score
  quality.py            drift verdicts + dropout gap detection
  export.py             kinematics + standardised CSV output
  testing/
    synthetic_source.py degradation generator (validation scaffolding only)
```

Pipeline order: `providers → sync → identity → sync (refine) → fusion →
confidence → quality → export`.

## Notes for whoever picks this up

- **`tests/matrix_engine/test_identity_matching.py`** is named that way, not
  `test_identity.py`, because `tests/statsbomb/test_identity.py` already exists
  and pytest requires unique test module basenames when test directories are
  not packages. Existing tests were not to be modified, so the new file yielded.
- **Provider loaders are `lru_cache`d** on the raw read. The Metrica CSVs are
  ~145k rows and the SkillCorner JSONL is 89 MB; without caching the test suite
  spends most of its time re-parsing. Callers must not mutate what they get back.
- **`load_skillcorner_source(..., max_frames=N)`** bounds the JSONL read. The
  full file takes ~10 s to parse; the tests read 3000 frames.
- **The Metrica sample data has genuine glitches** — occasional single-frame
  position jumps whose second derivative reaches 170 m/s². `export.py` caps
  implausible accelerations the same way `metrica_velocities.py` caps speeds.
- **`synthetic_source.SyntheticSource` carries the answer key** (true identity
  mapping, injected gaps, pre-noise signal). Only tests and the demo's scoring
  step may read it; the pipeline is only ever handed `.data`.
- **`providers/statsports.py` reads an external, unlicensed dataset over the
  network** and is the only part of this package that does. It keeps no copy by
  default, anonymises names before anything else touches the frame, and raises
  `StatsportsUnavailable` — which its tests treat as a skip — whenever the source
  cannot be reached. `MATRIX_STATSPORTS_SAMPLE_PATH` points it at a local copy.
  Read the handling rules in
  [`STATSPORTS_ADDENDUM.md`](STATSPORTS_ADDENDUM.md) before using it.
