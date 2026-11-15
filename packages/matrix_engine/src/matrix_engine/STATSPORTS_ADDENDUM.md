# Real wearable data — STATSports addendum

Follow-up to `matrix_engine/README.md`, which closed with this gap:

> **No wearable or GPS data exists in this repo.** `synthetic_gps` is a label on
> downsampled, noised optical data.

That is now partly addressed. `matrix_engine/providers/statsports.py` reads a
genuine raw STATSports Apex export — real GPS positions, real IMU channels, real
per-sample signal diagnostics — and normalises it into the same schema Metrica
and SkillCorner speak. **It does not close the larger gap**: there is still no
optical tracking of the same session, so there is still nothing to fuse against
and still no ground truth. What follows is scoped accordingly.

```
pytest tests/matrix_engine       # 223 tests (17 skip without the export)
pytest                           # 382 with the export reachable, 365 + 17 skipped without
```

## Provenance and handling — read before reproducing

The default source is `maindata.csv` from the public GitHub repository
`bakero08/Statsports-data-analytics-UCD`, a student capstone project. It is real,
name-attributed data for 18 named UCD AFC players, including heart rate, with **no
stated license and no evidence that STATSports, the club or the players consented
to third-party reuse**.

This is not an open-data programme like Metrica's or SkillCorner's samples, and it
is not treated as one:

- **Nothing is vendored.** The adapter fetches on access. No copy of this dataset
  is in this repository, in its history, or produced by running its tests.
- **No disk copy by default.** `cache_dir=None` streams into memory. A cache is
  opt-in, lives in the OS temp directory, and `purge_cache()` deletes it. The
  cache used while validating this work was deleted afterwards.
- **Anonymised by construction.** `Player Display Name` is replaced by a salted
  hash prefix (`sp_3044506ada`) and dropped in the same step, before any other
  transformation. `anonymize=False` raises rather than obeying. A test that never
  skips asserts that no name — and no whitespace-separated fragment of one —
  appears in any value, column, index or attribute of the output.
- **Tests skip, never fail, when it is unreachable.** Including the case where the
  repository is taken down, which would be a reasonable thing for it to be.

A caveat on the anonymisation, stated plainly because the word invites more
confidence than it deserves: the hash is deterministic, so anyone who already
holds the squad list can hash each candidate name and confirm a match. It
guarantees that no name leaks out of this code; it is not anonymity against
someone who knows who was on the pitch that evening.

Reproducing this means sourcing your own copy under the same constraints. None of
this makes the dataset licensed, and none of it should be read as a
recommendation to use it.

## What the export actually is

Measured, not taken from documentation — STATSports publishes no public spec for
this format.

| | Value |
|---|---|
| Rows | 325,036 raw, 324,984 after dropping unusable coordinates |
| Players | 18, all on one shared wall clock |
| Session | one continuous block, 18:08–19:40 (5,519.8 s) |
| Native rate | **3.333 Hz** (0.3 s), identical for all 18 players |
| Columns | position, speed, heart rate, `Hacc`, `Hdop`, `Quality of Signal`, satellites, 3-axis accelerometer, 3-axis gyro |
| Team labels | none |

Three things about it are worth recording because they contradict what a reader
would reasonably assume:

**The rate is 3.33 Hz, not 10 Hz.** An Apex unit samples at 10 Hz internally;
this export was decimated before publication. The adapter measures the rate from
consecutive per-player deltas rather than declaring one, which matters because a
hard-coded 10 Hz would have silently mis-scaled every gap threshold, noise
estimate and interpolation weight downstream by a factor of three.

**It is probably not 92 minutes of match play.** The session runs 18:08–19:40,
which is before a typical League of Ireland kick-off, and it contains long
stretches where the median speed is near zero and the whole squad sits inside a
few tens of metres. Whatever it is — warm-up, a training session, pre-match plus
part of a match — nothing here identifies which portions are match play, so no
claim in this document depends on that.

**The header contains a typo** (`Gyro Yro X` where `Gyro X` is meant). Not
consequential, since that column is unused, but a reminder that this adapter
targets one file rather than a documented vendor format.

## What was proven

### 1. A real wearable vendor's raw export parses cleanly into the common schema

324,984 rows pass `validate_schema`, in centred metres, with measured sample
rates, opaque stable player IDs, and the optional `native_quality` column. It
stacks with Metrica and SkillCorner output without special-casing. The Track A
claim — that the schema is a real contract and not a description of the two
optical sources it was written against — now has a genuinely different class of
sensor behind it: satellite positioning, its own sample rate, no team labels, and
per-sample diagnostics neither optical source provides.

Two vendor quirks the adapter handles rather than propagates:

- **52 rows carry coordinates of exactly (0, 0)** — null-island sentinels. They
  are dropped, not projected. Projecting them would have put players 5,900 km
  into the Atlantic and inflated the recovered pitch span by four orders of
  magnitude.
- **`team` is `unknown`, not a guessed `home`/`away` split.** This is one club's
  export; there is no opponent in it. `schema.py` gained `UNKNOWN_TEAM` as a
  legal team value for exactly this case, because the alternative was fabricating
  a split and calling it data.

An unexpected benefit of dropping the sentinel rows: they arrive in bursts, one
of them 46 consecutive samples long, so refusing them converts the vendor's own
unusable rows into an honest 14.1-second dropout that `detect_gaps` reports and
`sync` prices into `gap_s`. The two quality mechanisms turned out to complement
each other rather than one being idle.

### 2. The vendor's own quality signal is real, usable, and measurably meaningful

This is the substantive new capability. Before this change, `confidence.py` could
only infer trouble from *disagreement between sources* or *dropout*. A single
source quietly degrading while still emitting plausible positions at a steady
rate was invisible to it. `native_quality` is the only signal in the schema that
can catch that, and it now flows provider → `sync` → `fusion` → `confidence` →
`quality`.

Since there is no ground truth to score against, the claim "this score means
something" was tested against two error proxies that never see the quality
fields: trajectory curvature (second difference of position over consecutive
native samples), and the disagreement between the vendor's own reported
`Speed (m/s)` and the speed implied by differencing its own positions.

| `native_quality` | share of rows | median curvature | vs clean | median speed disagreement | vs clean |
|---|---|---|---|---|---|
| = 1.0 (clean) | 97.5% | 0.086 m | — | 0.095 m/s | — |
| < 0.9 | 0.88% | 0.162 m | **1.9×** | 0.304 m/s | **3.2×** |
| < 0.8 | 0.71% | 0.194 m | **2.3×** | 0.370 m/s | **3.9×** |
| < 0.7 | 0.57% | 0.236 m | **2.7×** | 0.482 m/s | **5.1×** |

The monotone dose-response across thresholds is the convincing part, more than
any single ratio: the worse the unit says a sample is, the worse it measurably
is, on two unrelated yardsticks. On means rather than medians the effect is
larger still (4.2× curvature, 9.6× speed disagreement at `< 0.7`), because bad
samples are bad in the tail.

### 3. The lat/lon → pitch transform produces a plausible pitch *length*, and flags its own width

Projection is equirectangular about the session's own GPS centroid (distortion is
negligible at pitch scale), then a rotation to put the recovered long axis on x.

| | long axis | short axis |
|---|---|---|
| Recovered span | **118.4 m** | **97.6 m** |
| Plausible for a pitch | 90–120 m ✓ | 45–90 m ✗ |
| After discarding the outer 5% of samples | 93.1 m | 91.9 m |

The long axis lands inside the range for a real pitch. The short axis does not:
at 97.6 m it is wider than any legal pitch, because the session includes movement
off the pitch and the transform has no idea where the touchlines are. The adapter
**warns and records `plausible=False`** rather than passing silently, and a test
asserts that it does. Trimming the outermost 5% of samples brings both axes into
range, which locates where the excess comes from — but it is reported as a
diagnostic, not as a better estimate, because tuning a trim fraction until a
plausibility check passes would be fitting the check rather than measuring the
pitch.

### 4. PCA is the wrong orientation estimator here, and that is measurable

The task this work followed specified PCA on the projected point cloud. Building
it revealed that it does not work on this data, so the default is the
minimum-area bounding rectangle instead (exact, via convex-hull edge directions).
PCA remains available as `orientation="pca"`.

| | minimum-area rectangle | PCA |
|---|---|---|
| Recovered span | 118.4 × 97.6 m | 114.3 × 103.0 m |
| Bounding-box area | **11,556 m²** | 11,773 m² |
| Axis angle, dropping the first 20 / 30 / 35 min | **constant to within 0.5°** | 61.5° / 45.5° / −168.9° |

(Angles are each estimator's own axis convention, so only their *stability* is
comparable across the two columns. The adapter reports the recovered long axis at
152.4° from east for the full session.)

The reason is in the data, not in the implementation: the two principal standard
deviations are 18.9 m and 18.1 m, a ratio of 1.04. The cloud is very nearly
isotropic, so "the direction of greatest variance" is not a stable quantity, and
PCA's answer swings by more than 100° depending on which part of the session is
included. Over a full match, players collectively cover the pitch too evenly for
variance to reveal its long axis. The minimum-area rectangle returns 62.5° from
every subset tried.

## What was not proven

**Still zero real fusion validation.** The headline gap is unchanged. No optical
source covers this session, so there is no second feed to reconcile against, no
real cross-vendor identity matching, no real clock alignment between independent
sensors, and no ground truth to score a fused position against. Everything in
`README.md` about fusion, identity and drift still rests on a synthetic second
source derived from the same optical recording. This work replaced a synthetic
*ingest*, not a synthetic *validation*.

**The coordinate transform is an unverified estimate, not a calibration.** No
pitch survey, no corner coordinates, no recorded orientation. The plausibility
check above is a smell test that the short axis already fails. Beyond that:

- **The attacking direction is undetermined.** Which way is +x is arbitrary, and
  there is no half-time flip handling, because nothing in the data says when
  halves started or which way the team was playing.
- **The frame is fitted to whatever subset is loaded.** Load two different sets
  of players and you get two different coordinate systems. That is a quiet way to
  corrupt a fusion, so `rotation_deg=` exists to pin the frame — but the footgun
  is inherent to estimating geometry from data rather than knowing it.

**`native_quality` is a plausible model, not a calibrated one.** It maps three
diagnostics onto [0, 1] with linear ramps and fixed weights. The ramps and
weights were fitted by inspection of this one session; nothing calibrates the
score against a position error in metres, because that would need ground truth.
The dose-response above shows the *ordering* is meaningful. It does not license
reading `native_quality = 0.6` as any particular number of metres.

**Two of the three specified quality fields carry no measurable signal, and the
one that does was not in the specification.**

| Field | Spread in this export | Max \|Spearman ρ\| vs 3 error proxies | In the score |
|---|---|---|---|
| `Hacc` | 0 on 99.2% of rows, 1–7 on the rest | 0.089 | **yes, weight 0.6** |
| `Hdop` | 0.40 on 90% of rows, max 12.4 | 0.060 | yes, weight 0.3 |
| `No. of Satellites` | 0 or 14–22 | 0.026 | yes, weight 0.1 |
| `Quality of Signal` | 0 or 173–342, undocumented scale | 0.035 | **no** |

`Quality of Signal` is excluded. Its scale is undocumented, and bucketing it
against curvature gives a flat, non-monotone result (medians 0.078–0.091 m across
every band) rather than a dose-response. The first version of this module
included it and produced a quality score with *no* relationship to data quality
at all — pooled Spearman 0.007, and a within-player flagged-versus-clean jitter
ratio of 1.00 across 18 players. Including a plausible-sounding field with no
signal drowned the field that had one. It is still parsed and available via
`with_raw_quality=True`.

Note also that Spearman is a weak statistic on all of these, `Hacc` included,
because 97.5% of rows tie at the same score. The group comparison and the
dose-response in the table above are the load-bearing evidence, not the
correlation coefficients.

**Zero satellites does not mean no fix, and assuming it did would have been a
bug.** 15% of rows carry `No. of Satellites` and `Quality of Signal` of 0
together. The obvious reading is a lost fix with a dead-reckoned position, so it
was checked, and the data says otherwise:

| | rows with 0 satellites | rows with a satellite count |
|---|---|---|
| Median curvature | 0.091 m | 0.086 m |
| Frozen-position rate | 1.8% | 2.0% |
| Position step when the field resumes | 0.266 m | 0.261 m (baseline) |

A genuine fix loss followed by reacquisition snaps the position back. This does
not, at all. The fields are simply unpopulated on those samples, so the satellite
term is treated as absent there and the score is renormalised over the remaining
terms. Penalising them would have turned `native_quality` into a detector of
missing telemetry, and since those rows are 15% of the file it would have swamped
the 2.5% that carry a real signal.

**One club, one session, one export layout.** Everything here is fitted to a
single ~92-minute session from one team. Other STATSports customers may export
different columns, different rates, or different units, and this adapter does not
attempt to detect or adapt to that — it fails loudly with
`StatsportsUnavailable` when the columns are not the ones it knows.

**Heart rate and IMU channels are ignored.** The export carries heart rate,
3-axis accelerometer and 3-axis gyroscope. None is used. Heart rate in particular
is physiological data with no bearing on tracking reconciliation, so the adapter
does not carry it forward at all.

## Effect on the existing prototype: none

The `schema.py`, `sync.py`, `fusion.py`, `confidence.py` and `quality.py` changes
are additive, and this is asserted rather than asserted-of.

- **Confidence is bit-identical.** `test_native_quality_additive.py` reimplements
  the original three-factor formula from its definition and asserts
  `array_equal` against current output for all three original Track B scenarios.
- **No column appears where it does not apply.** The `native_quality_factor`
  column is omitted, not filled with ones, for sources without diagnostics — so
  no consumer can start depending on it accidentally — and the fused column set
  for those sources is asserted unchanged.
- **A perfect-quality column is exactly neutral**, and a NaN reading is neutral
  rather than a penalty, which is what happens on every row a source *without*
  diagnostics covered alone.
- **The demo's numbers are unchanged**: scenario 1 still 0.633 m → 0.001 m
  (99.8%), scenario 2 still 0.611 m → 0.461 m (24.5%), identity still 100%,
  offset error still 3 ms.
- **All 320 pre-existing tests still pass**, untouched.

## Where this leaves the validation gap

`README.md` listed what would be needed next. This work delivers item 2 —
*"a real wearable export with its native rate, gaps and per-player files"* —
about halfway: the ingest is real, and the native rate, sentinel rows and vendor
diagnostics turned out to be genuinely more informative than the synthetic
stand-in was. It also turned up a capability gap worth having found (confidence
could not use a source's own opinion of itself) and two assumptions that the real
data refuted (PCA for orientation, zero satellites meaning no fix).

Items 1, 3, 4 and 5 are untouched, and item 1 — *one match recorded by two real
vendors* — remains the single measurement that would collapse most of the
remaining uncertainty. A real wearable feed with no optical counterpart cannot
substitute for it, and this addendum should not be read as progress toward it.
