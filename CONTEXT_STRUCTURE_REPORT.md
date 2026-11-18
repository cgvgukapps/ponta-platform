# Context Structure Report (P2, P3, P4)

17 November 2026. Task: `ponta-blueprint/context-structure-task.md`; specification: `ponta-blueprint/context-packages.md`. Stage B of the context packages, built under founder decision 4a (proceed against reference labels; analyst agreement still gates player-facing use).

## Result in one paragraph

All three packages are built on the Stage A registry. They run on all four regression matches; Stage B adds about 13 s and 3.3 MB per match (16 s and 4 MB with P1). Outputs are identical across runs.

- **Roles:** each player's main role agrees with SkillCorner's role by position group for **79% of playing time** (70% of players). With the three central-midfield roles merged it is **91% of playing time** (85% of players). Most misses involve the defensive, central and attacking midfield split, or an attacking midfielder against a forward or wide player.
- **Phases:** in-possession phases agree with SkillCorner's on **73.1%** of compared time, **Cohen's κ 0.60** (0.58 on the first half, 0.61 on the second). Published agreement between human analysts is about κ 0.7.
- **Formations** are stable, with at most one change per team per half on every match.
- **Shape measures** fall inside published ranges for official matches.
- **Zones** wait for the founder's spot check.

These are agreements with one provider's labels on one match, not accuracy against a hand-labelled truth.

**Update, 17 November 2026 (later the same day).** The founder agreed the bars and every Stage B threshold, approved the Metrica Game 2 fix (done) and the SkillCorner open-data download. Measured on all 20 open-data matches ([below](#across-skillcorners-open-data-20-matches)):
- **Formations** meet the bar on every match.
- **Phases** hold up pooled (κ 0.61; 0.62 and 0.59 by half), but only 11 of 20 matches clear 0.55 on both halves.
- **Roles** do not generalise: the median is 75.4% by playing time, and 12 of 20 matches clear 75%. Most misses are wide midfielders given central slots.

**Zones:** the founder's spot check passed, 47 of 50 judged right (bar 45).

**Gate 2a is met on the reference match, zones included. It is not met across the wider set for roles, and only pooled for phases.**

**Update, 18 November 2026.**
- **Phases:** revised after the founder's replay review (P1 and P4 0.2.0; [below](#the-founders-phase-replay-review-and-the-refinements-17-and-18-november-2026)).
- **Roles:** revised (P2 0.2.0; [below](#the-role-revision-p2-020-18-november-2026)). Most of the role misses were the comparison misreading SkillCorner's LM and RM, which are central midfielders in its vocabulary. Each player's main role is now read from the team's defensive shape.
- **Roles across the 20 matches:** median 92.2%; 19 of 20 meet the grouped bar and 17 the merged bar.
- **Phase bar:** restated by the founder as pooled κ at least 0.55 on each half (met, 0.61), with no match below 0.50 (three matches fall short by up to 0.02).
- **All six new thresholds agreed.**

## What was built

| Piece | Where |
|---|---|
| P2 team structure: per-frame shape and lines; per-window lines, formation, roles; current formation with persistence | `plaiq_engine/context/structure.py` |
| P3 relative zones: ball lane, third, band against the defending lines, hull, pocket; attacker counts | `plaiq_engine/context/zones.py` |
| P4 phase of play: rule-based state machine, paired out-of-possession phase, block, phase segments | `plaiq_engine/context/phase.py` |
| Stage B thresholds (P2 to P4), each with status, reason and source | `plaiq_engine/context/thresholds.py` (version 0.3.0; 0.4.0 records the founder's agreement, values unchanged) |
| Reference readers (SkillCorner roles, phases as points and as spans; no names) | `tools/context_references.py` |
| Phase sweep (half-split) | `tools/context_phase_sweep.py` |
| Measurements for this report | `tools/context_structure_measure.py` |
| Spot-check page for zones and phases (written outside the repository) | `tools/context_spotcheck.py` |
| Stage B on SkillCorner's open data (20 matches; reads and writes outside the repository) | `tools/context_reference_set.py` |
| Golden harness registers P2 to P4; manifest updated | `tools/context_golden.py`, `tests/golden/context/manifest.json` |
| Tests: 27 synthetic (building blocks, all ten templates, mirrored pitch, determinism) and one on sample data | `packages/plaiq_engine/tests/test_context_structure.py`, `tests/cross_engine/test_context_sample.py`, fixture `tests/support/context_fixture.py` (`eleven_a_side`) |
| Field definitions | `docs/context/vocabulary.md` (P0 to P4) |
| Architecture decisions 16 to 18 | `docs/ARCHITECTURE.md` |

## A data finding: Metrica Game 2's direction of play (fixed)

In Metrica Sample Game 2 the home goalkeeper's mean x is +39.7 m. The normalised data therefore has the home team attacking −x in both periods, against `PitchConfig.home_attacks_positive_x=True`. The Metrica CSV loader flips the second half and assumes the first-half direction; in Game 2 that assumption is wrong. Games 1 and 3 and SkillCorner are correct.

- **Stage B:** reads each team's direction per period from the data (the side its goalkeeper defends) and is unaffected.
- **Who it may affect:** anything that assumes the home team attacks +x on Game 2 (decision-quality direction, attacking-third labels in the semantic events). The prototype used the same loader, so its Game 2 outputs share the assumption.

**Fixed on 17 November 2026** (founder approval, after this report was first written):
- `metrica_io.orient_home_attacking_positive` rotates the whole match by 180 degrees when the home goalkeeper defends +x over the first half. Both Metrica CSV loaders (`match.load_metrica_csv_match`, `providers.metrica`) call it after the half-time flip; the prototype's `to_single_playing_direction` is unchanged.
- Regression tests in `packages/matrix_engine/tests/test_metrica_coords.py`.
- After the fix both home goalkeepers in Games 1 and 2 defend −x in both halves.
- **What changed downstream:** in the context golden summaries only Game 2's recorded attack signs; every P1 to P4 field is identical, as expected. Games 1 and 3 and SkillCorner are unchanged. No prototype parity reference covers Game 2, so the extraction proof is unaffected. Engine outputs on Game 2 that depend on direction (decision quality, attacking-third labels) now follow the documented convention and will differ from the prototype's Game 2 outputs.

## Measurements

All figures come from `tools/context_structure_measure.py` and `tools/context_phase_sweep.py`; JSON outputs are kept outside the repository.

### P2 shape (medians over in-play time, both teams)

| Match | Length m | Width m | Hull m² | Def line m | Def–mid gap m | Mid–att gap m | Shape coverage |
|---|---|---|---|---|---|---|---|
| Metrica 1 | 33.4 / 33.4 | 39.9 / 41.8 | 979 / 929 | 33.6 / 38.1 | 11.6 / 12.3 | 13.4 / 13.0 | 100% |
| Metrica 2 | 33.0 / 31.8 | 39.1 / 39.1 | 865 / 903 | 34.1 / 35.5 | 11.7 / 11.3 | 12.6 / 13.2 | 100% |
| Metrica 3 | 35.4 / 34.4 | 40.8 / 41.6 | 1,040 / 980 | 35.4 / 33.8 | 13.5 / 13.1 | 13.8 / 13.1 | 100% |
| SkillCorner | 33.1 / 30.5 | 36.9 / 37.2 | 855 / 758 | 39.9 / 26.6 | 12.1 / 10.5 | 12.5 / 12.0 | 73.6% |

**Published comparison values:**
- **Rico-González et al. 2021, official matches:** length 31–46 m, width 35–48 m, hull about 900 m².
- **Forcher et al.:** line distances 10.2 ± 3.8 m (defence to midfield) and 13.3 ± 4.2 m (midfield to attack).

Every median falls inside these ranges. The defensive line here is measured over all in-play time; the published 22–28 m is for defending only, so it is not comparable.

SkillCorner's shape coverage is 73.6% because broadcast tracking has no players during replays and camera cuts (its tracking covers about 72 of the match's minutes). Shape is therefore Degraded for that match, as it should be.

### P2 formations (current formation per team, state and half; changes in brackets)

| Match | Home in | Home out | Away in | Away out |
|---|---|---|---|---|
| Metrica 1 | 4-4-2 / 4-4-2 (0) | 4-4-2 / 4-4-2 (0) | 4-1-4-1 / 4-1-4-1 (0) | 4-4-2 / 4-4-2 (0) |
| Metrica 2 | 3-4-2-1 / 3-4-2-1 (0) | 4-4-2 / 4-4-2 (0) | 4-2-3-1 / 4-2-3-1 (0) | 4-4-2 / 4-4-2 (1 in H1) |
| Metrica 3 | 4-2-3-1 / 4-2-3-1 (0) | 4-2-3-1 / 4-2-3-1 (0) | 3-5-2 / 3-5-2 (0) | 4-1-2-1-2 / 4-4-2 (1 in H1) |
| SkillCorner | 4-2-3-1 / 4-2-3-1 (0) | 4-4-2 / 4-4-2 (0) | 3-4-2-1 / 3-4-2-1 (0) | 4-4-2 / 4-4-2 (1 in H1) |

**Plausibility:**
- **Out of possession:** shapes are mostly 4-4-2. A 4-2-3-1 defending as a 4-4-2 is the textbook pattern.
- **In possession:** shapes with three at the back (3-4-2-1, 3-5-2) are what a back four looks like when the full-backs push on.

Between 64% and 71% of windows fit one template clearly; the median margin over the second-best template is 0.18 to 0.19.

**The template library.** Ten formations from the literature. In-possession shapes such as 2-3-5 have no template yet, so they map to the nearest one with three at the back.

### P2 roles against SkillCorner (1886347; 27 outfield players who played)

Each player's main role is their most common role over the match's windows, weighted by window data.

| Windows used | Exact | Position group | Group, time-weighted | Central mid merged | Merged, time-weighted |
|---|---|---|---|---|---|
| All possession (used) | 48.1% | 70.4% | **79.3%** | 85.2% | **90.5%** |
| Out of possession | 29.6% | 70.4% | 77.1% | 88.9% | 95.3% |
| In possession | 48.1% | 70.4% | 75.9% | 77.8% | 82.0% |

**Groups:** left back (LB, LWB), right back, centre back, defensive mid, central mid, attacking mid, left wide (LW, LM), right wide, forward.

**Every disagreement by group** (SkillCorner → Ponta): AM→LCF, AM→LCM, AM→LDM, AM→LM, CF→AM, LDM→LCM, LDM→RCF, RDM→RCM.
- **Within central midfield (four):** AM→LCM, AM→LDM, LDM→LCM, RDM→RCM.
- **An attacking midfielder against a forward or wide player (three):** AM→LCF, AM→LM, CF→AM.
- **A substitute with under six minutes of data (one):** LDM→RCF.

**Literature:**
- **Formation:** about 75% expert agreement (Bialkowski et al. 2016).
- **Positions:** 86.5% (SoccerCPD, Kim et al. 2022).

The goalkeepers are found correctly and excluded from roles.

### P3 zones (share of possession time with a label)

| Match | Coverage | Behind defence | In def line | Between lines | Mid–att | In front | Ball in pocket | Attackers between lines (mean) |
|---|---|---|---|---|---|---|---|---|
| Metrica 1 | 84.9% | 3.1% | 4.1% | 20.9% | 38.1% | 33.9% | 4.7% | 2.8 |
| Metrica 2 | 81.7% | 2.2% | 2.3% | 20.1% | 39.5% | 35.9% | 4.4% | 2.7 |
| Metrica 3 | 89.1% | 1.9% | 3.2% | 18.2% | 37.7% | 39.0% | 4.8% | 3.0 |
| SkillCorner | 68.4% | 8.2% | 4.9% | 17.0% | 29.5% | 40.5% | 3.4% | 2.3 |

The lane shares are symmetric (wings 16–21% each, half-spaces 10–14%, centre 17–25%). Coverage is lower than possession time where the ball is undetected.

**There is no reference for zones.** Gate 2a for P3 is the founder's spot check: `tools/context_spotcheck.py` writes 50 cards (stratified by band across the four matches) with the pitch, both teams, the defending lines and hull, the labels, and tick boxes.

### P4 phases against SkillCorner (1886347)

SkillCorner labels each phase on its player-possession events. Read as spans (first to last possession of each phase) they cover 38.8% of grid time.

**Comparison base:** grid points where SkillCorner and Ponta give the same team in possession; teams agree on 89.9% of SkillCorner's labelled points, leaving 9,463 compared points. Out-of-possession phases are a fixed pairing in both, so only in-possession phases are compared.

| Measure | Whole match | First half | Second half |
|---|---|---|---|
| Accuracy | **73.1%** | 72.6% | 73.7% |
| Cohen's κ | **0.60** | 0.58 | 0.61 |

**Confusion matrix** (rows: SkillCorner; columns: Ponta; grid points):

| | build_up | create | finish | direct | transition | quick_break | set_play | chaotic |
|---|---|---|---|---|---|---|---|---|
| build_up | **1,364** | 552 | 0 | 1 | 11 | 0 | 0 | 0 |
| create | 483 | **3,647** | 368 | 89 | 64 | 0 | 0 | 11 |
| finish | 8 | 136 | **1,655** | 0 | 16 | 0 | 0 | 0 |
| direct | 19 | 24 | 14 | **106** | 0 | 0 | 0 | 0 |
| transition | 3 | 74 | 60 | 22 | **24** | 0 | 0 | 0 |
| quick_break | 0 | 10 | 47 | 0 | 0 | **0** | 0 | 17 |
| set_play | 0 | 1 | 248 | 0 | 0 | 0 | **53** | 0 |
| chaotic | 68 | 80 | 82 | 15 | 6 | 0 | 19 | **66** |

(disruption, 5 SkillCorner phases, has no Ponta equivalent and does not appear in the compared points.)

**Where and why they disagree:**
- **Established play (build_up, create, finish)** is 89% of the compared points and agrees well. Most misses are at the block boundaries, where a few metres of the defenders' depth flips the label.
- **Set plays without an event feed** account for most of the remaining loss. SkillCorner's set plays come from its own event detection; Ponta finds only corners from tracking (53 of 302 points). With a logged event feed (Metrica), set plays come from the logged restarts.
- **Transition, quick_break and chaotic** are rare (a few dozen points each) and agree poorly. Their rules depend on regains, which SkillCorner's derived feed cannot separate from restarts (every chain starts "won"), and on SkillCorner's count of passes, which Ponta does not have.
- **Equal pitch thirds for the blocks** (35 m and 69 m from goal) gave κ 0.20. The measured boundaries (26 m and 47 m) give 0.60.

### P4 sweeps (half-split; `tools/context_phase_sweep.py`)

Each group was varied from a starting setting of a 26 m low block, 45 m high block and 10 m break gain; the final setting combines the best of each.

| Threshold | Values tried | Best | κ first / second half at best |
|---|---|---|---|
| Low block below (m) | 22, 24, 26, 28, 30 | 26 on the whole match and the second half; 24 on the first half (0.589 / 0.560) | 0.577 / 0.591 |
| High block from (m) | 38, 41, 44, 47, 50 | 47 on both halves, for every low-block value up to 26 | 0.577 / 0.591 |
| Break gain (m) | 5, 10, 15 | 15 on both halves | 0.574 / 0.591 |
| Break maximum (s) | 8, 15 | no difference | |
| Chaotic below (s) | 0, 2, 3, 5 | 5 on the whole match (all within 0.01) | |
| Minimum phase (s) | 0, 1, 2 | no difference | |
| Direct window (s) | 2, 3, 4 | 3 on both halves | |

Only the two block boundaries matter much. They are fitted to one match and one provider. The half-split shows they are not noise within that match, not that they generalise.

### P4 on Metrica (logged events)

| Match | Labelled in-play time | build_up | create | finish | direct | transition | quick_break | set_play | chaotic |
|---|---|---|---|---|---|---|---|---|---|
| Metrica 1 | 94.5% | 17.6% | 34.9% | 30.2% | 3.8% | 3.0% | 0.7% | 6.8% | 3.0% |
| Metrica 2 | 93.4% | 9.8% | 39.2% | 38.3% | 3.0% | 1.5% | 0.2% | 5.0% | 3.0% |
| Metrica 3 | 94.7% | 13.1% | 47.9% | 22.6% | 3.5% | 1.8% | 1.1% | 5.5% | 4.4% |

**Set plays on Metrica** come from logged restarts, so comparing them with the same logged restarts would only measure the input. Instead, the **tracking-only corner rule** was tested against Metrica's logged corners:

- **Result:** it finds 6 of 25, with 1 false alarm (radius 4 m, first 1 s). Wider radii find up to 9 of 25 but raise false alarms to 5 to 7.
- **Why it misses:** in most missed corners the ball is not detected until it is in flight. That is why Data check marks phases Degraded without a logged feed.

## Thresholds

**All agreed by the founder on 17 November 2026** (thresholds version 0.4.0; no value changed). The Status column records the basis each value had when it was proposed.

| Threshold | Value | Status | Basis |
|---|---|---|---|
| `p2_window_s` | 120 s | proposed | Shaw and Glickman 2019 |
| `p2_min_possession_s` | 5 s | proposed | Shaw and Glickman 2019 |
| `p2_min_window_data_s` | 10 s | proposed | Engineering choice |
| `p2_min_presence_share` | 0.5 | proposed | Engineering choice (substitutions) |
| `p2_min_outfield_players` | 8 | proposed | Allows two missing players |
| `p2_min_shape_coverage` | 0.9 | proposed | Metrica 100%, SkillCorner 73.6% |
| `p2_formation_min_margin` | 0.10 | proposed | 64–71% of windows clear at this margin |
| `p2_formation_persist_s` | 300 s | proposed | SoccerCPD (Kim et al. 2022) |
| `p3_min_coverage` | 0.75 | proposed | Mirrors P1's 25%; Metrica 82–89%, SkillCorner 68% |
| `p4_low_block_max_m` | 26 m | **measured** | Sweep, both halves |
| `p4_high_block_min_m` | 47 m | **measured** | Sweep, both halves |
| `p4_break_min_gain_m` | 15 m | **measured** | Sweep, both halves |
| `p4_set_play_max_s` | 20 s | proposed | SkillCorner specification |
| `p4_direct_min_gain_m` | 32 m | proposed | SkillCorner specification |
| `p4_direct_window_s` | 3 s | proposed | Sweep (3 s best of 2, 3, 4) |
| `p4_chaotic_max_s` | 5 s | proposed | SkillCorner specification |
| `p4_break_eval_s` | 3 s | proposed | Engineering choice |
| `p4_break_max_s` | 15 s | proposed | Sweep shows no difference |
| `p4_recovered_players` | 7 | proposed | Engineering choice |
| `p4_min_phase_s` | 1 s | proposed | Sweep shows no difference |
| `p4_corner_radius_m` | 4 m | proposed | Corner rule test above |
| `p4_corner_window_s` | 1 s | proposed | Corner rule test above |

## Gate 2a bars

**Agreed by the founder on 17 November 2026.** "Measured" is the reference match 1886347; the wider set is in the next section.

| Package | Measure | Measured | Proposed bar |
|---|---|---|---|
| P2 roles | Position-group agreement with SkillCorner, weighted by playing time | 79.3% | **at least 75%** |
| P2 roles | Same, central midfield merged | 90.5% | **at least 85%** |
| P2 formations | Changes per team, state and half | at most 1 everywhere | **at most 2**, and plausible on the spot check |
| P3 zones | Founder's spot check (50 cards) | not yet done | **at least 45 of 50** zone labels judged right |
| P4 phases | Cohen's κ against SkillCorner's in-possession phases | 0.60 (0.58 / 0.61 by half) | **at least 0.55**, on each half |

The phase bar sits below published analyst agreement (κ about 0.7) because the reference is one provider's rules, not a consensus.

## Across SkillCorner's open data (20 matches)

**Source:** SkillCorner's open data (MIT licence, checked 17 November 2026):
- 20 matches from the 2024/25 Australian A-League, each with tracking, dynamic events and a phases-of-play file;
- downloaded to `~/ponta-reference-data/skillcorner-opendata`, outside every repository;
- SkillCorner credited as its README asks.

**Method:**
- `tools/context_reference_set.py` runs P1 to P4 with the agreed thresholds on every match.
- Roles are scored by the same method as on the reference match.
- Phases are scored against two references: spans from the dynamic events (the Gate 2a method) and SkillCorner's own phases-of-play file.
- Nothing was tuned on these matches.
- Two corrections to the comparison, neither touching the packages:
  - **LF and RF** (forwards in a front two) were missing from the role-group table and are now counted as centre forwards. This changes no figure on the reference match.
  - **The re-downloaded 1886347** has a revised dynamic-events file. Phases on it give κ 0.59 (0.58 and 0.59 by half, 9,628 points) against 0.60 (0.58 and 0.61, 9,463 points) on the older copy; roles are identical.

| Measure | Bar | Median (range) over 20 matches | Matches meeting the bar |
|---|---|---|---|
| Roles by position group, time-weighted | at least 75% | 75.4% (59.7% to 98.7%) | 12 |
| Same, central midfield merged | at least 85% | 76.3% (59.7% to 100%) | 7 |
| Formation changes per team, state and half | at most 2 | at most 1 on every match | 20 |
| Phase κ, lower of the two halves (event spans) | at least 0.55 | 0.56 (0.50 to 0.66) | 11 |
| All four bars | | | 5 |

**Phases pooled over all 20 matches:**
- **Event spans:** 193,453 compared points; 72.7% agreement, **κ 0.61** (0.62 first half, 0.59 second).
- **SkillCorner's phases-of-play file:** 265,278 points; 70.3% agreement, **κ 0.59** (0.60 and 0.58).
- So the phase rules and the block boundaries (26 m and 47 m) carry over to other matches at about the level measured on the reference match. One half of one match is a small sample (about 5,000 points), and per-half κ ranges from 0.50 to 0.72. **No match falls below 0.50.**

**Roles, where they miss** (165 of 583 players who played):
- **Wide midfielders given central slots** account for 72 misses (44%). SkillCorner calls them LM or RM; Ponta gives a central or defensive midfield slot.
- **Centre forwards given an attacking-midfield slot** account for 24 (15%).
- The rest are spread thinly: wingers against wing-backs or attacking midfielders, and full-backs against centre-backs.
- **Why merging central midfield does not help here:** SkillCorner uses no central-midfield role (CM, LCM, RCM) on these matches. Its midfielders are DM, AM, LM and RM. Merging central midfield helped on the reference match only because of that match's line-ups.
- **A hypothesis, not tested:** many of these teams play a flat midfield four or a front two. The template fit may prefer shapes with a midfield three, or a wide midfielder who tucks in reads as central over a two-minute window. SkillCorner's role is one label per player for the match; Ponta's is the main slot by playing time.

**Other measures on the 20 matches:**
- **Direction:** read consistently on every match.
- **Shape coverage:** shape covers a median 73% of in-play time (54% to 80%), so every Stage B entry is Degraded on broadcast tracking, as on the reference match.
- **Runtime:** P1 to P4 take a median 18.8 s per match (13 s to 27 s) on a laptop also running other jobs.

**What this means for Gate 2a:**
- **Formations:** met everywhere.
- **Phases:** met pooled and on the reference match; per match, the per-half bar is noisy.
- **Roles:** not met across the set. The misses are specific (wide midfield and front-two shapes), and the template fit is the place to fix them.
- **Recommended:** a short P2 role-fit revision measured on these 20 matches before Stage C builds on roles. Until then, roles stay "estimated, not yet validated" and are not used by later packages.
- **Phase bar:** the founder may also restate it for a set of matches, for example pooled κ at least 0.55 on each half and no match below 0.50.

## The founder's phase replay review and the refinements (17 and 18 November 2026)

**The review.**
- A still frame rarely shows a phase, so the founder (a former semi-professional footballer) judged the same 50 moments again as replays: 5 s before and 2 s after each, from `tools/context_spotcheck.py --replay`.
- Each case got a verdict and, often, a note on what the phase should be.
- **29 of 50** were judged right against P4 0.1.0, stricter than the 34 of 50 on stills.
- The judgments are kept, without positions, in `tools/data/phase_review_2026-11-17.json`. Each case records the labels the reviewer accepts, or rejects where the note gives no single answer.

**Method.**
- `tools/context_phase_review.py` scores each rule variant twice: on the 50 judgments, and against SkillCorner's phases pooled over its 20 open-data matches.
- One reviewer on 50 moments is a small sample, and rules tuned to it can overfit. A variant was adopted only if it agreed better with the review **and** lost nothing against SkillCorner.

**First and second rounds** (each refinement on its own, against P4 0.1.0; SkillCorner κ against event spans, pooled):

| Refinement | Review (of 50) | SkillCorner κ | Decision |
|---|---|---|---|
| None (P4 0.1.0) | 29 | 0.607 | |
| Finish needs the ball in the attacking half, else create | 30 | 0.622 | **adopted** |
| Finish needs the ball in the final third | 30 | 0.554 | dropped: SkillCorner's finish does not require it |
| Build-up whenever the ball is in the own third | 31 | 0.595 | **adopted** (in combination, below) |
| A break ends when the ball reaches the final third | 31 | 0.605 | **adopted** |
| Direct lasts the ball's whole forward flight | 28 | 0.603 | dropped: a carry after the reception also moves forward |
| The same, only while the ball travels at 8, 11 or 14 m/s | 28 to 29 | 0.607 | dropped: the missed long ball rolled at about 5 m/s, like a carry |
| Set play ends when the ball goes back into the own half | 29 | 0.608 | dropped: fixed one case, broke another |
| The three adopted together | 34 | 0.608 (0.604 / 0.610 by half) | |

**A P1 bug found on the way (fixed, P1 0.2.0).**
- After a dead ball, P1 gave the set piece to whichever team had the ball first. When the defending team won the first touch (a corner headed clear), its chain started `set_piece:corner_kick`, and P4 labelled the clearance a set play.
- Now a chain starts with a set piece only when its team took it; otherwise it starts `won`.
- Only Metrica's chain start types change; possession labels, and so Gate 1, are unchanged.
- One review case moved the wrong way because of the fix. Metrica 3 grid point 25151 is a clearance from the team's own box after the opponent's set piece. The reviewer had accepted "set play" there; it is now `chaotic`.

**Corners from player positions (the founder's proposal, P4 0.2.0).**
- Without an event feed the ball is usually undetected until a corner is in the air, so the ball rule found about 1 corner in 20 on SkillCorner.
- The founder proposed reading the set-up instead: one attacker at a corner flag while players crowd that penalty area. `plaiq_engine/context/restarts.py` implements it without using the ball.
- `tools/context_corner_sweep.py` swept the taker radius, box margin, crowd and hold time on 10 SkillCorner matches and read them on the other 10 and on Metrica. References (comparison only):
  - SkillCorner's restart markers (`corner_for`, `corner_against`): 138 corners;
  - Metrica's 26 logged corner kicks.

| Rule | SkillCorner, held-out matches: found / detections right | Metrica: found / detections right |
|---|---|---|
| Ball near a corner flag (P4 0.1.0) | 4% / 8% | 23% / 86% |
| **Set-up: taker within 4 m, at least 12 players in the box, held at least 1 s** | **73% / 55%** | **69% / 55%** |

- **Generalisation:** the settings were chosen on SkillCorner alone and hold on Metrica's different tracking.
- **False alarms:** about half the detections are false. The likely cause is throw-ins near the corner flag, which share the shape; not yet checked.
- **Scope:** the rule is used only without a logged feed; with events, logged corners are used.

**Final result (P1 0.2.0, P4 0.2.0, thresholds 0.5.0):**
- **Review:** 34 of 50, against 28 for the old rules with the corrected P1.
- **SkillCorner κ pooled over the 20 matches:** 0.610 against event spans (0.609 and 0.611 by half) and 0.619 against the phases-of-play file.
- **Per match** (`tools/context_reference_set.py`, event spans):
  - 13 of 20 matches clear κ 0.55 on both halves (11 before), and the median of the weaker half rose from 0.56 to 0.57;
  - the changes are not uniform: one match rose from 0.54 to 0.64 on its weaker half, while three fell just below 0.50 (0.53 to 0.48, 0.50 to 0.50, 0.55 to 0.50);
  - with roles and formations unchanged, 6 of 20 matches meet all four bars (5 before).
  - On the reference match 1886347 (newer copy of the events): κ 0.60 (0.61 and 0.58 by half).
- **Runtime:** a median 14 s per match for P1 to P4 on the 20 matches; two runs of about 150 s were the laptop, not the code (13.8 s when re-timed).
- **What the review says is still wrong:**
  - long balls and lost possessions while the ball travels (Metrica 1 grid point 9197, Metrica 2 grid point 4206, SkillCorner grid point 24345);
  - a goal kick credited to the receiving team while the ball is undetected (Metrica 2 grid point 8943), a P1 issue;
  - restarts other than corners without an event feed (SkillCorner grid point 16221, a throw-in);
  - recycling the ball backwards to rebuild, which the reviewer calls build-up (Metrica 2 grid point 5864, Metrica 3 grid point 11497);
  - a regain that gained less than 15 m in 3 s (Metrica 3 grid point 4277).
- **Thresholds:** the six new values are `measured`, not `agreed`, until the founder agrees them. *(Agreed by the founder on 18 November 2026; thresholds 0.6.0, no value changed.)*

## The role revision (P2 0.2.0, 18 November 2026)

Approved by the founder on 18 November 2026. Two changes, measured in this order.

**1. The comparison misread SkillCorner's vocabulary (a measurement fix; P2 unchanged).**
- SkillCorner's own position groups, in each player's `player_role`, file **LM and RM under "Midfield"** and **LF and RF under "Wide Attacker"**.
- Mean positions from the tracking agree. Over the 20 matches, SkillCorner's LM and RM stand a mean 2 to 4 m from the middle, as LDM and RDM do; its LW and RW stand about 11 m out, and LF and RF 11 to 15 m.
- In its line-ups a 4-4-2 is `LW LM RM RW`, so LM and RM are the central pair. Ponta's templates use LM and RM for a flat four's wide midfielders.
- The comparison mapped both vocabularies through Ponta's table, so it counted a central midfielder given a central slot as a miss. It also counted LF and RF as centre forwards, the correction recorded in the section above.
- `tools/context_structure_measure.py` now maps SkillCorner's roles through their own groups (`REF_GROUP`). The reference match 1886347 uses none of these four labels, so its figures do not change.
- **So most of the "wide midfielders given central slots" were Ponta right and the comparison wrong.**

**2. Each player's main role is read from the defensive shape (P2 0.2.0).**
- With the comparison fixed, the remaining misses were mostly one pattern: SkillCorner lists a 4-4-2, while Ponta fits a 4-2-3-1 or 4-3-3 *in possession*, when the wide players and one forward push on. Out of possession Ponta mostly fits the 4-4-2.
- The role comparison had read each player's most common slot over *all* windows, mixing both states.
- **The rule, from football convention** (a team's nominal formation is its defensive shape) and fixed before scoring: a player's main role is the slot held for the most window data **out of possession**, or over all windows when the player has no out-of-possession window. P2 now publishes it as the `main_roles` segment. Per-window roles are unchanged.
- **Adoption test, as for the corner rule:** compared on the even-index matches (fit) and read on the odd ones (held out). In possession was scored too, and was worse on both halves.

| Roles by position group, time-weighted (bar 75%) | All windows | **Main role (P2 0.2.0)** |
|---|---|---|
| Fit half (10 matches): median, matches meeting the bar | 77.5%, 6 | **85.0%, 9** |
| Held-out half (10 matches): median, matches meeting the bar | 85.3%, 10 | **97.6%, 10** |
| Central midfield merged (bar 85%), all 20: median, matches meeting the bar | 91.3%, 17 | **94.8%, 17** |
| Reference match 1886347: grouped / merged | 79.3% / 90.5% | 77.0% / 95.2% |

**Across the 20 matches, from the start of the day to now:**

| Roles, time-weighted | Before (17 November) | Comparison fixed | **P2 0.2.0** |
|---|---|---|---|
| By position group: median, matches meeting 75% | 75.4%, 12 | 81.5%, 16 | **92.2%, 19** |
| Central midfield merged: median, matches meeting 85% | 76.3%, 7 | 91.3%, 17 | **94.8%, 17** |
| Players whose main role misses (of 583) | 165 | 126 | **93** |

- **Weighting:** as before, each player is weighted by the window data behind their main role. Weighted by each player's playing time with data instead, the medians are 90.7% and 93.5%.
- **Where it got worse:**
  - On match 1959846 roles fell from 72% to 59% (merged 92% to 78%). The home team's out-of-possession shape fits a 3-4-2-1, while SkillCorner lists a 4-4-2.
  - On 1996435, 98.7% fell to 92.6%.
  - On the reference match the strict measure fell from 79.3% to 77.0%, still above the bar, while the merged measure rose from 90.5% to 95.2%.
- **Still below a bar:** grouped on 1959846; merged on 1959846, 2006229 (84.4%) and 2016236 (84.0%).
- **The remaining misses are spread thinly.** The largest groups are a central midfielder given a defensive-midfield slot (9), a forward given a central or attacking-midfield slot (9), and an attacking midfielder given a central-midfield slot (6).
- **Unchanged:** formations, lines and every per-frame field. P3 and P4 read P2's per-window roles, not the main role, so their outputs are unchanged and phase κ is unchanged (pooled 0.610).

**Gate 2a across the 20 matches after the revision** (`tools/context_reference_set.py`):

| Bar | Matches meeting it |
|---|---|
| Roles by position group, at least 75% | 19 |
| Roles, central midfield merged, at least 85% | 17 |
| Formation changes, at most 2 | 20 |
| Phase κ at least 0.55 on each half | 13 |
| All four | 10 |

## Runtime and storage

Measured alone by the golden harness:

| Match | P1 to P4 runtime | Storage (frame table and segments) |
|---|---|---|
| Metrica 1 | 16.2 s | 4.1 MB |
| Metrica 2 | 16.0 s | 4.0 MB |
| Metrica 3 | 17.5 s | 4.1 MB |
| SkillCorner | 15.7 s | 2.7 MB |

Stage A alone was about 2.5 s and 0.8 MB. Most of the new time is spent in P3 and P4:
- they rebuild P2's per-frame lines from its segment tables (architecture decision 17);
- they test the hull frame by frame.

Both are easy to optimise if the runtime budget at Gate 3a needs it.

**Other gates:**
- **Full local gate:** 1,057 passed, 12 skipped. *(After the Game 2 fix and the agreement of the thresholds: 1,060 passed, 12 skipped. After P1, P2 and P4 0.2.0: 1,068 passed, 12 skipped.)*
- **Golden outputs:** captured; a second run matches on all four matches.
- **P1 outputs:** unchanged (the only golden difference is the four new Data check entries per match).
- **Import check and lint:** both pass.

## Differences from the specification, and limits

- **P2:**
  - Role variants such as an inverted full-back are not detected; roles are template slots.
  - "Shape changes against the previous window" is covered by the formation table and its persisted changes, not by a separate diff.
- **P3:**
  - Zones are labelled for the ball, with counts of attacking players per zone; there is no per-player zone column, since it would multiply the frame table by 22.
  - "Wide of the full-back" is not built.
  - The pocket is defined as between the lines, in a half-space and inside the hull (no published definition exists), not "beside the 6".
- **P4:**
  - **Vocabulary:** follows SkillCorner's published phases so it can be measured against a reference. The blueprint's earlier list (progression, settled possession, counter-press, defensive transition) is not used.
  - **Pressure:** Plaiq pressure is not an input; the block rule matched the reference without it.
  - **Counter-press:** not labelled.
- **References:** one provider. *(Revised 17 November 2026.)* Measured on the reference match and then on all 20 of SkillCorner's open-data matches (section above); still one provider and one league.
- **No analyst check:** every Stage B label is "estimated, checked against references, not yet validated by analysts" until Gate 2b.

## Decisions for the founder

Taken on 17 November 2026: the five Gate 2a bars and every Stage B threshold agreed; the Metrica Game 2 fix approved and done; the SkillCorner open-data download approved and done (20 matches, not 10).

**Zone spot check, done 17 November 2026:**
- **Zones:** 47 of 50 judged right, so the bar is met. The three misses: Metrica 1 grid point 25974, Metrica 2 grid point 4809, SkillCorner grid point 16221.
- **Phases on the still cards:** 34 of 50 judged right. By label: build_up 8 of 8, create 9 of 12, finish 9 of 17, direct 4 of 6, set_play 2 of 4, transition 1 of 2, chaotic 1 of 1.
- **Why the phase figure is indicative only:** the founder noted that a still frame rarely shows a phase. `tools/context_spotcheck.py --replay` now writes the same 50 moments as replays (5 s before, 2 s after) with a phase tick and a note per case.
- **A pattern to check on the replays:** "finish" follows the defending team's low block, whether or not the ball is in the final third.

Taken on 18 November 2026:
1. ~~Phase replay check~~ done: 29 of 50 against P4 0.1.0; see the review section above.
2. **Roles across the 20 matches:** the role revision approved and done (P2 0.2.0; section above). Roles meet the grouped bar on 19 of 20 matches and the merged bar on 17.
3. **Phase bar for a set of matches:** restated as **pooled κ at least 0.55 on each half, and no match below 0.50 on either half**, the floor proposed before P4 0.2.0 was measured. Pooled it is met (0.609 and 0.611). **Three matches are below the floor and count as failing:** 2013725 (0.483 in the first half), 2015213 (0.496, second half) and 2017461 (0.497, second half). The per-match bar of 0.55 on each half is kept as a reading and is met on 13 of 20.
4. **The six new thresholds (P4 0.2.0):** agreed (thresholds 0.6.0).

**Gate 2a across the set, after these decisions:**
- **Met:** formations (20 of 20) and phases pooled.
- **Not met:** roles on three matches (one on the grouped bar, three on the merged bar) and the phase floor on three. All three phase failures are within 0.02 of the floor.
- **On the reference match:** every bar is met.

Still open:
5. **Phase replay check again (optional):**
   - The same 50 moments were re-rendered with P4 0.2.0 on 18 November 2026. The sample is drawn by the ball's P3 band, which did not change, so the moments are the same. The page holds positions, so it is kept outside the repository.
   - Seven cases changed label: Metrica 1 grid point 25371 (create to build_up), Metrica 2 grid points 4809 (finish to create), 18016 and 20789 (transition to finish), Metrica 3 grid point 25151 (set_play to chaotic), and SkillCorner grid points 7820 (create to build_up) and 20372 (finish to set_play).
   - Six of the seven now carry a label the founder accepted. Re-judging only these seven confirms the changes on the pitch.
