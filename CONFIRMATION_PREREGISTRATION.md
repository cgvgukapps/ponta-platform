# Pre-registered confirmation of the context packages

> **Status: written 27 November 2026, before any confirmation data was opened (review backlog B, ST-01, ST-02). Bars agreed by the founder on 27 November 2026; they are not changed for this confirmation. Part A's comparisons were fixed in writing (below) before they were run.**

## Why

Every agreement figure for P1 to P8 so far comes from one development set. One even/odd split of SkillCorner's 20 open matches served as the "held-out" half for every tuning round, the Gate 2a bars were restated after the first reading, and at least one rule was changed after a held-out reading (build-up opposed, kappa 0.36 before the fix and 0.52 after). After that many looks the odd half is a second development set (Dwork et al. 2015, "The reusable holdout"). Since 25 November 2026 every figure is reported externally as "measured on a development set". This confirmation measures the frozen packages once, on matches never used, against bars written first.

## What is frozen

The confirmation runs the versions current at the freeze commit and nothing later:

| Part | Version |
|---|---|
| Matrix engine | 0.2.2 |
| Plaiq engine | 0.2.0 |
| Context packages | P1 0.6.0, P2 0.2.0, P3 0.1.0, P4 0.6.0, P4v 0.1.0, P5 0.2.0, P6 0.2.0, P7 0.3.0, P8 0.3.0 (as in `tests/golden/context/manifest.json` at the freeze) |
| Thresholds | 0.30.0 |
| Semantic schema | 2.2 |

The freeze is the commit that adds this file. Any change to a frozen part after it makes a new version. A new version needs a new confirmation on data this one did not use.

## Data, chosen before looking

- **Part A (internal; can run now).** PFF FC's World Cup 2022 set (64 matches). It was used for Stage E's stability and team-level agreement, **never to set or tune a P1 to P8 rule**. It confirms the event-level fields PFF labels itself: possession (P1, against PFF's possession events), line breaks (P5, against PFF's line-break flags) and carrier pressure (P5, against PFF's pressure events). Under the founder's decision of 23 November 2026, its results are internal only, name their source, and reach no public claim without PFF's written approval.
- **Part B (when data exists).** A licensed set never used: a design partner's season, or open data released after the freeze. It confirms the fields only SkillCorner-style or analyst labels can check: phases (P4), roles and formations (P2), episodes (P8), runs (P6) and engagements (P4v). Part B needs a source of labels: SkillCorner's own on new SkillCorner matches, or one or two analysts' (Gate 2b).

All matches in the chosen set are used. None are dropped after looking, except for a load failure, which is reported with its cause.

## Bars (agreed by the founder, 27 November 2026)

Each bar sits below its development figure, because a confirmation on new data usually reads lower. Where an agreed gate bar exists it is kept unchanged.

| Field | Measure | Development figure | Bar | Basis of the bar |
|---|---|---|---|---|
| P4 phase of play | Cohen's kappa against the reference's phases, pooled, each half | 0.66 / 0.63 | **0.55 on each half** | Gate 2a's agreed bar, unchanged |
| P4 phase of play | Kappa on each match | lowest match half 0.52 | **no match below 0.45** | proposed (Gate 2a's floor of 0.50, less a development margin) |
| P2 roles | Share grouped / central midfield merged, pooled | 83% / 90% | **75% / 85%** | Gate 2a's agreed bars, unchanged |
| P2 formations | Formation changes per team half | 0 to 1 | **2 or fewer on 90% of matches** | Gate 2a's bar of 2, per match |
| P8 episodes | Reference phase boundaries matched within 2 s | 78.5% | **70%** | proposed |
| P1 possession | Agreement with the reference's possession, by time | 82% to 89% (German league) | **80%** | proposed |
| P5 carrier pressure | Weighted kappa against the reference's pressure levels | 0.76 | **0.60** | proposed |
| P5 line breaks | Kappa against the reference's line-break labels | 0.56 | **0.45** | proposed |
| P6 runs | Reference runs found / run type kappa | 75% to 80% / 0.41 | **70% / 0.35** | proposed |
| P4v engagements | Found / precision | 58% / 65% | **50% / 55%** | proposed |
| Stage E verdict | Share of ordinary matches called above or below usual | 11% (5 of 47) | **5% to 15%** | the agreed 10% false-alarm rate, with sampling room |

## Part A: how each comparison is made (fixed 27 November 2026, before it was run)

PFF's labels are not the development references' labels, so each Part A measure is stated here in full. The PFF data was opened before for Stage E's team-level figures (pressure rates, lines broken per team, possession shares), never for an event-level comparison with P1 or P5, and none of these definitions was chosen by looking at one. The tool is `tools/confirmation_pff.py`; one match is run first only to check the tool runs, and nothing below changes after that.

- **Data and rate.** All 64 matches through `tools/pff.py` (reader 3): PFF's tracked ball, not its smoothed ball; every third frame (about 10 Hz, the rate of SkillCorner's data, on which P5's development figures were measured, and the rate Stage E used on PFF). Periods the reader drops because players and ball disagree (extra time in 10506 and 10517), and events PFF did not link to a frame, are counted and reported, not used. P1, P2, the zones and P5 run as frozen, at the frozen thresholds, with PFF's events as their event input, as every provider's events are; P1 reads those events, so possession is not independent of PFF in the way a separate feed would be (the same holds for the German league's figure).
- **P1 possession (bar 80%).** PFF's team in possession on P1's grid: the team of the latest PFF on-the-ball event (a kick-off or an `OTB` event) in the period. A grid point is dead (left out) before the first kick-off, after an `OUT` or `END` event until the next on-the-ball event, and between any event and a set-piece restart that follows it with no `OUT` in between (a foul or offside stoppage). The measure is the German league's: the share of live grid points where PFF names a team at which P1 names the same team, so P1's contested, unknown or dead labels count as misses. Also reported: agreement where both name a team.
- **P5 carrier pressure (bar 0.60).** PFF's `pressureType` on the acting player's passes, crosses, shots and carries (`PA`, `CR`, `SH`, `BC`), on the frame PFF links each to. PFF does not publish what its four codes mean; following Stage E's existing reading (`tools/context_profiles.py`), `P` and `A` are pressed and `N` and `L` are not. P5's carrier pressure at the nearest grid point is cut at the development cut point between no pressure and low pressure (3 on the 1v1 scale): 3 or more is pressed. With two levels the pre-registered weighted kappa is Cohen's kappa. **This is a coarser measure than the development figure (five SkillCorner levels, quadratic-weighted), so the two are not directly comparable.** Also reported: the kappa with `L` counted as pressed, and P5's median pressure under each PFF code (descriptive, no bar).
- **P5 line breaks (bar 0.45).** PFF's `linesBrokenType` on passes (`PA`; PFF labels no crosses), matched to P5's pass rows by team and start frame within two frames, where P5 labels the pass. PFF names lines by role (`A` attack, `M` midfield, `D` defence), so the furthest line broken is compared by role: none, attack, midfield or defence (P5's units in the same roles; PFF's furthest is `D` if present, else `M`, else `A`). Cohen's kappa on the four labels. Also reported: whether any line was broken (kappa).
- **Pooling and intervals.** Each figure is pooled over all matches (points, passes or actions), with a 90% interval from 2,000 resamples of matches, and each match's own figure. Outputs stay outside the repository and name PFF as their source.

## Analysis plan

- The tools and settings are the ones the development figures used (`tools/context_reference_set.py`, `tools/context_states.py`, `tools/context_fifa_view.py`, `tools/context_profiles.py`), run once at the frozen versions, with outputs kept outside the repository.
- Every figure is reported, met or not, with its 90% interval from resampling matches.
- Nothing is tuned after the data is opened. A rule found wanting is reported as such, and its fix is a new version for the next confirmation.
- The result is written up in this document, under a dated "Result" heading, with the commit it ran at.

## Meanwhile: how much do the figures depend on one team? (27 November 2026)

No fresh data exists yet, so the frozen rules were scored on SkillCorner's 20 matches leaving out one of the 13 teams at a time. **This is not cross-validation with re-tuning:** the rules were tuned on all 20 matches. It shows whether the headline rests on a few teams, not how the rules would do on new data.

| Measure | All 20 matches | Without each team in turn | Each team's own matches |
|---|---|---|---|
| Phase kappa (point-weighted over matches) | 0.643 (pooled 0.647) | 0.637 to 0.654 | 0.595 to 0.678 (median 0.655) |
| Roles grouped (time-weighted) | 0.883 | 0.863 to 0.893 | 0.768 to 0.960 |

No team moves the pooled phase figure by more than 0.01. The lowest team's own matches (4 matches) read 0.595, above the proposed per-match floor. Source: `~/ponta-reference-data/cache/rerun-2026-11-25/reference_set.json` (backlog A re-measurement; Matrix 0.2.2's clock change does not touch these frame-based figures).
