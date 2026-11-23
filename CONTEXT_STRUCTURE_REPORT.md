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
- **Restarts (later the same day, P4 0.3.0):**
  - Throw-ins are now found from player positions without an event feed: 55% found and 61% of detections right on held-out SkillCorner matches, and 60% and 87% on Metrica.
  - Goal kicks cannot be found on broadcast tracking.
  - The review rises to 36 of 50, and SkillCorner agreement is unchanged to slightly up.
  - Two matches remain below the 0.50 floor.

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
- **Review:** 34 of 50, against 28 for the old rules with the corrected P1. *(After the founder's re-check of the seven changed cases, 18 November 2026: 35 of 50, against 29.)*
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

## Restarts from player positions (P4 0.3.0, 18 November 2026)

**Why.** Without an event feed, P4 found corners from positions (the founder's rule) but no other restart. The founder's review flagged a throw-in labelled `finish` (SkillCorner grid point 16221) and a goal kick labelled `create` (Metrica 2 grid point 8943).

**What the references say first.** SkillCorner does not treat throw-ins or goal kicks as set plays. Its phase on the first possession after each restart, over the 20 matches:

| Restart | SkillCorner's phase on the next possession |
|---|---|
| Throw-in (705) | create 234, finish 197, chaotic 160, build_up 89; set_play 10 |
| Goal kick (178) | build_up 92, create 46, direct 32 |
| Corner (103) | set_play 95 |

- So labelling throw-ins or goal kicks `set_play` would lose agreement, and the adoption rule would drop it.
- Points in the 8 s after a restart disagree with SkillCorner about as often as elsewhere (27% after throw-ins against 28% overall). Breaks wrongly started by a restart account for about 300 of 194,000 compared points.
- **The aim was therefore restart facts on matches without a feed, plus small phase changes with a football reason, under the usual test.**

**Throw-ins: the detector** (`restarts.throw_in_setups`), the founder's corner idea carried over:
- exactly one player of the team, and none of the other, stands on or beyond a touchline away from the corners, for at least 2 s;
- play has slowed: the median speed of all players over the set-up's last 2 s is at most 2 m/s;
- the ball is within 5 m of that player in those 2 s.

The settings were swept by `tools/context_restart_sweep.py` on 10 SkillCorner matches (fit) and read on the other 10 and on Metrica's 110 logged throw-ins. A detection matches a reference of the same team when the set-up ends from 6 s before it to 3 s after it.

| Throw-ins | Found | Detections right |
|---|---|---|
| SkillCorner, fit half (384) | 51% | 63% |
| SkillCorner, held-out half (447) | 55% | 61% |
| Metrica, logged (110) | 60% | **87%** |

- **Without the ball test** (the best such rule on the fit half), about half of all detections are wrong on every set: wide players stand on the line in open play. With it, Metrica's precision rises to 87%. SkillCorner's stays near 62%; its broadcast ball position is filled in almost everywhere, so the test separates less there.
- **Stillness alone** traded finds for precision one for one; it is kept only together with the ball test, as chosen on the fit half.

**Goal kicks: not detected.** At best about 20% found and 30% right. The reason is the broadcast: **73% of SkillCorner's goal kicks have a tracking gap** (at least 2 s with fewer than 4 players tracked, or a jump in the clock) in the 15 s before, against 24% of throw-ins. Corners have the same gap rate (80%), but their set-up lasts long enough to be seen. Metrica logs no goal kicks, so the full-pitch case is unmeasured.

**What P4 0.3.0 does with them** (only without a logged feed):
1. **Publishes the restarts** as a `restarts` segment: type, team, period, time, source. With a feed it lists every logged set piece, team as logged; without one, the corners and throw-ins found from positions.
2. **A throw-in is not a regain:** a possession that starts with a detected throw-in of its own team cannot start a quick break or transition.
3. **Set-ups:**
   - While a throw-in set-up is under way the ball is dead, so no phase is labelled. P1 carries possession through stoppages without a feed, so before this P4 labelled the wait.
   - A corner set-up is `set_play` for the taking team from the set-up's start, following the founder's note on SkillCorner grid point 20372 ("a corner being taken"). An earlier try that left corner set-ups unlabelled broke that case and was dropped.

| Rules (scored by `tools/context_phase_review.py`) | Review (of 50) | SkillCorner κ, event spans (by half) | SkillCorner κ, phases file |
|---|---|---|---|
| P4 0.2.0 | 35 | 0.6103 (0.6085 / 0.6106) | 0.6193 |
| Throw-in is not a regain (2 alone) | 35 | 0.6105 (0.6089 / 0.6106) | 0.6196 |
| Corner set-ups left unlabelled (tried, dropped) | 34 | 0.6103 (0.6086 / 0.6106) | 0.6211 |
| **P4 0.3.0 (1 to 3)** | **36** | 0.6104 (0.6089 / 0.6104) | **0.6249** |

- **Review:** SkillCorner grid point 16221, the throw-in, now has no phase; the founder's note rejected only `finish`. Counting "no phase" as right there is a reading of the note, for the founder to confirm.
- **SkillCorner:**
  - against the phases-of-play file (which covers dead time too), κ rises by 0.006;
  - against event spans it is flat, with the second half 0.0002 lower: noise, but strictly not "nothing lost", so recorded here.
- **Per match (event spans):**
  - changes stay within ±0.008;
  - 13 of 20 still clear 0.55 on both halves;
  - 2015213 moves from 0.496 to 0.501, so **two matches remain below the 0.50 floor** (2013725 at 0.482, 2017461 at 0.490). Moves this small are noise, not a fix.
- **Unchanged:** matches with a logged feed (Metrica): P4's output there is unchanged.
- **Thresholds 0.7.0** *(agreed by the founder on 18 November 2026, thresholds 0.8.0)*: four new `measured` values (`p4_throw_in_hold_s` 2 s, `p4_throw_in_out_m` 0 m, `p4_throw_in_still_mps` 2 m/s, `p4_throw_in_ball_m` 5 m), for the founder to agree. The detector's edge (3 m from a goal line), bridging (1 s) and look-back (2 s) are module constants, as measured.
- **Not done:** free kicks (not attempted; 59% have a tracking gap before them), goal kicks (above), and P1 chain start types without a feed. P1 runs before team frames exist, so the restarts live in P4's segment.

## Against official league data: IDSSE (18 November 2026)

**The data.**
- Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12, 195. Data: **Deutsche Fußball Liga (DFL)**, CC BY 4.0; any publication must credit the DFL and cite the paper.
- Seven 2022/23 matches (two Bundesliga, five 2. Bundesliga), each with:
  - full-pitch TRACAB tracking at 25 Hz;
  - the league's official events;
  - match information, including each player's official playing position and each team's line-up.
- The ball's tracking carries the **official ball status** (in play or dead) and **team in possession on every frame**: a frame-level reference that SkillCorner and Metrica do not give.
- **Download and reading:**
  - `tools/fetch_reference_data.py idsse` downloads the files to `~/ponta-reference-data/idsse`, outside every repository, and checks each figshare MD5.
  - `tools/idsse.py` reads them as a `NormalisedMatch` without reading any name. It rotates halves so home attacks +x and translates the league's events into the vocabulary P1 reads.
  - This is a reference reader for measurement, not a supported provider.
- **Reader checks:**
  - the home keeper's mean x is −35 to −38 m in both halves;
  - an event sits a median 2.7 m from the tracked player who made it (37 m with the y axis mirrored);
  - the official team in possession matches the acting team on 81% of events.
- **Measured by** `tools/context_idsse.py`. Nothing was tuned on these matches: every threshold was set on SkillCorner and Metrica.

| Match | P1 vs official possession (ball in play): with events / tracking only | P1 dead ball: official dead time found / P1 dead right | Main roles vs official positions: grouped / merged | Official line-up → P2 out of possession |
|---|---|---|---|---|
| J03WMX | 88.0% / 86.9% | 74% / 94% | 95.2% / 95.5% | 4-2-3-1 → 4-2-3-1; 4-2-3-1 → 4-2-3-1 |
| J03WN1 | 81.7% / 79.5% | 73% / 94% | 91.2% / 97.4% | 4-3-3 → 4-3-3; 4-2-3-1 → 4-2-3-1 |
| J03WOH | 88.1% / 86.7% | 72% / 94% | 78.6% / 82.7% | 4-2-3-1 → 4-2-3-1; 4-2-3-1 → 4-1-2-1-2 |
| J03WOY | 85.6% / 83.5% | 83% / 96% | 97.4% / 97.4% | 4-2-3-1 → 4-2-3-1; 3-3-2-2 → 3-5-2 |
| J03WPY | 87.6% / 85.8% | 70% / 96% | 58.1% / 76.0% | 4-2-3-1 → 4-4-2; 4-1-3-2 → 5-3-2 |
| J03WQQ | 88.9% / 87.6% | 78% / 97% | 86.1% / 99.1% | 3-4-1-2 → 3-5-2; 3-3-2-2 → 5-3-2 |
| J03WR9 | 89.4% / 88.2% | 72% / 98% | 93.5% / 99.6% | 4-4-2 → 4-4-2; 4-2-3-1 → 4-2-3-1 |

**P1 possession.**
- **Agreement:** 82% to 89% (87.2% pooled), below Gate 1's 93%. Gate 1 measured agreement with possession inferred from events (a rule), so this is the first independent check, and it is lower.
- **Tracking alone** costs 1 to 2 points.
- **Where P1 disagrees:**
  - 74% agreement in the first half-second after a P1 switch (timing lag);
  - 82% with the ball at 4 to 10 m/s (short passes and contested balls);
  - 89% even more than 5 s after a switch, so part of the gap is the two sources defining possession differently.
- **Dead ball:** P1 finds 70% to 83% of the official dead time, and 94% to 98% of what it calls dead is dead.
- **The league logs restarts 1 to 3 s before the ball is played.** The ball first moves, and the official status turns live, a median 2.5 to 3.4 s after a logged corner and 1 to 2 s after a logged throw-in. A dead-ball span that ends at the logged restart therefore ends early.
- **Next:** a P1 round tuned against the official frame-level possession, fitted on some matches and read on the others. Gate 1 is not reopened by this; it is a finding for the founder.

**P2 roles and formations.**
- **Official positions to groups** (`POSITION_GROUP` in the tool):
  - most codes map directly;
  - LM / RM and DLM / DRM in a back three stand 46 to 48 m deep and 16 to 18 m wide, so they are wing-backs, grouped with full-backs;
  - HL / HR stand centrally, so central midfield.
- **Roles:** main roles agree at a median 91.2% (97.4% merged); 6 of 7 matches clear 75%.
- **Independent check of P2 0.2.0:** reading main roles from the defensive shape beats reading them from all windows here too (median 91.2% against 90.3%, merged 97.4% against 95.9%), though not on every match.
- **The weakest match (J03WPY, 58%):** the home 4-2-3-1 defends as a 4-4-2 and the away 4-1-3-2 fits a 5-3-2. Line-up labels that differ from the defensive shape are the main source of misses.
- **Formations:** the out-of-possession formation matches the official line-up's shape on 8 of 14 teams. Of the rest, four play shapes Ponta has no template for (3-3-2-2 twice, 4-1-3-2, 3-4-1-2) and two defend in another shape.

**P4 restarts from positions, on full-pitch tracking** (the events removed; each restart timed when the official ball status turns live):

| From positions (settings chosen on SkillCorner) | Found | Detections right |
|---|---|---|
| Corners (the founder's rule) | **93%** (53 of 57) | **67%** |
| Throw-ins | **83%** (256 of 308) | **79%** |

- Against the logged times (1 to 3 s early), corners scored 33% / 24% even though every set-up was found. The detector was right; the reference time was early.
- Extra corner detections are mostly a set-up that broke and reformed, or a second corner straight after the first.
- **Goal kicks:** the full-pitch tracking shows their set-up, so a goal-kick rule can now be measured here (not built yet).

**Not done here:** phases (the league gives no phase labels; its shots carry a counter-attack flag and a build-up type, a small check for later), and Plaiq's metrics (the league's shot pressure and expected goals are a possible comparison).

## Event data: possessions from events alone, and patterns at scale (18 November 2026)

**Data.**
- Wyscout's open events: Pappalardo et al. (2019), *Scientific Data* 6, 236, CC BY 4.0.
  - 1,941 matches of seven competitions in 2017/18 (five leagues, the 2018 World Cup, Euro 2016), about 3.25 million events.
  - Downloaded with `tools/fetch_reference_data.py wyscout` to `~/ponta-reference-data/wyscout-opendata`.
  - The players file holds real names and is never opened.
- **The rule:** possessions from events alone (`tools/event_chains.py`), re-implemented from the paper's "play action" (not its code). A possession starts when a team gains the ball. It ends at an interruption, a shot (the save belongs to it) or when the other team acts. Paired duels never switch it.
- **Measured by** `tools/context_event_chains.py`.

**A. How much does an event feed alone say about possession?**
- **Same feeds:** on the three Metrica matches, possession from events alone agrees with P1's tracking possession on 90% to 92% of the time both label (the events cover 82% to 97% of P1's labelled time).
- **Official reference:** on the seven IDSSE matches, against the league's frame-level possession with the ball in play:

| | Median agreement | Range |
|---|---|---|
| Events alone | 84.6% | 74.5% to 87.8% |
| P1 (tracking and events) | 88.0% | 81.7% to 89.4% |

- **So tracking adds about 3 points** of possession agreement over events alone on this reference. This measures the possession part of the "event-only analysis" decision, nothing more.

**B. Possession patterns at scale.** The same rule on Wyscout, set beside the same rule on our feeds and P1's tracking chains (10 matches: Metrica 1 to 3 and IDSSE):

| | Possessions per match | Median duration | Start: regain / throw-in / free kick / goal kick / corner | End: lost / out / foul / shot | Reach the final third |
|---|---|---|---|---|---|
| Wyscout, five leagues | 335 to 352 | 6.1 to 6.6 s | 69–72% / 13% / 7–10% / 5% / 3% | 65–68% / 17–19% / 6–8% / 6–7% | 38% to 41% |
| Wyscout, World Cup and Euro | 323 to 326 | 8.1 to 8.5 s | 71% / 13% / 7–8% / 5–6% / 3% | 65% / 18–19% / 8% / 7% | 42% to 43% |
| Event rule on our 10 matches | 295 | 7.7 s | 68% / 14% / 8% / 4% / 3% | 62% / 22% / 8% / 8% | 38% |
| P1 tracking chains, same 10 | 247 | 8.6 s | 71% / 14% / 7% / 4% / under 2% | 61% lost, 34% dead ball, 3% shot | (not measured) |

- **P1's possession starts match the norms of 1,941 matches.** Regains, throw-ins, free kicks and goal kicks start P1's tracking possessions in the same proportions as Wyscout's (71%, 14%, 7%, 4% against 69–72%, 13%, 7–10%, 5%).
- **P1 finds fewer, longer possessions** than an event rule: 247 against about 300 to 350 a match. A brief touch shorter than `p1_min_switch_s` does not switch possession in P1. This is by design, but it means "possessions per match" is not comparable between Ponta and an event provider without saying so.
- **Corner starts look low in P1** (under 2% against 3%). Since P1 0.2.0, a corner cleared at the first touch starts the defenders' possession as `won`. This is worth a look when P1 is next revised.
- **Shots end 3% of P1's possessions against 6% to 8% for the event rule.** The definitions differ: P1 ends a possession at the next change or stoppage, and a shot that goes out ends as `dead_ball`.
- **Counter-attacks:**
  - Wyscout tags 1.8% to 2.1% of possessions as counter-attacks.
  - On IDSSE the league flags 14 of 164 shots as counter-attacks. Ponta had a quick break or transition in the 5 s before 5 of them and before 2 of the 150 others. When Ponta labels a break the league mostly agrees (5 of 7), but it catches only about a third of counter-attacks.
  - A likely cause: since P4 0.2.0 a break ends when the ball reaches the final third, so the phase at the shot is `finish`.
  - At 60 of the 164 shots, P4's team at the shot frame is not the shooting team: the possession lag in the P1 finding above.
  - A small sample, and SkillCorner's transition is not the same concept as a counter-attack, so this is a direction for the next phase round, not a verdict.

**What this gives Ponta.**
- **Base rates** for the Data check (possessions and restarts per match; a feed far outside them is suspect).
- **A measured answer** for event-only possession.
- **Three leads for the next rounds:** P1's possession lag at switches and on shots; corner starts in P1; and counter-attacks in P4, now measurable against both the league's and Wyscout's flags.

## The possession round: P1 against the official possession (18 November 2026)

**Question.** P1 agrees with the league's frame-level possession on 82% to 89% of in-play time. Can its control thresholds close the gap without breaking what Gate 1 measured?

**Sweep** (`tools/context_possession_official.py`):
- **Axes:** control radius, ball-speed limit and minimum switch time (the axes of `context_measure.py`, 80 settings).
- **Fit and test:** chosen on four IDSSE matches, read on the other three.
- **Objective, fixed in advance:** accuracy over every in-play point where the league names a team. Contested, unknown or dead-ball labels count as misses, so a setting cannot gain by labelling less.
- **Guards:** Gate 1 on Metrica (at least 93% against event-anchored possession, switches within 15%), and agreement with SkillCorner's player possessions no more than 0.5 points lower.

| Setting | Official accuracy, fit / held out | Gate 1 | SkillCorner | Switches against the league's |
|---|---|---|---|---|
| Current: 1.5 m, 16 m/s, 0.4 s | 79.6% / 75.8% | passes | 95.1% | 0.51 |
| Best keeping the guards: 1.5 m, no speed limit, 0.4 s | 80.2% / 76.6% | passes | 95.1% | 0.52 |
| Best on the league's data alone: 1.0 m, no speed limit, no minimum switch | 83.5% / 80.4% | **fails** (switches 2.1 to 2.6 times the event feed's) | 92.5% | 1.00 |

**Why the gap does not close.**
- **The league switches possession about twice as often as P1** (2,804 against 1,440 switches on the seven matches). Its count is matched only with no minimum switch, which Gate 1's event reference and SkillCorner's possessions both reject.
- **The disagreement is not just brief spells.** Smoothing the league's possession with P1's own 0.4 s rule leaves agreement at 78.0%.
- **Of all in-play points the league names:** P1 names the other team on 11.4%, says contested on 4.7%, unknown on 3.1%, and dead ball (from events) on 2.9%.
- **The league's possession is anchored to its event log.** 93% of its switches have a logged event within 2 s, at a median offset of about 0.1 s, most often the new team's first action. It is the league's event-based possession on the frame clock, not an independent tracking truth. The 82% to 89% compares two definitions; it does not measure P1's error.

**Decision: no threshold change.**
- The best change that keeps the guards adds 0.7 to 0.9 points.
- A P1 change ripples into P2 to P4 and every measurement since Gate 1.

**Next: a judge.**
- `tools/context_possession_replay.py` writes a **blind** replay check: 40 moments where P1 and the league disagree for at least 1 s, and 10 control moments where they agree, drawn from all seven matches and shuffled.
- The founder says who has the ball at each moment (home, away or nobody yet) without seeing either label. The key is kept apart from the page.
- Scoring both sources against the founder's calls says which definition matches a football reading, and where P1's real errors are.

**The founder's blind calls (18 November 2026)**, kept without positions in `tools/data/possession_review_2026-11-18.json`:
- **Controls:** 10 of 10 called as both sources had them, so the judging is consistent.
- **The 40 disagreements:**

| | P1's team | The league's team | Nobody (ball in flight or loose) |
|---|---|---|---|
| All 40 | 11 | 15 | 14 |
| Disagreements of 1 to 2.2 s (25) | 11 | 9 | 5 |
| Disagreements of 2.4 s or more (15) | **0** | 6 | **9** |

- **Short disagreements are a draw.** They are the timing of a switch, and neither source is systematically better (11 against 9 with 25 cases).
- **Long disagreements are P1's errors.** P1 holds the old team while the ball is in flight or loose, or after the other team has it.
- **What marks them:**
  - in the controls, the nearest player was a median 1.2 m from the ball;
  - in the moments called the league's way or nobody's, 2.6 to 3.5 m, with the ball at about 9 to 10 m/s: a pass or clearance travelling.
  - Time since P1 last saw control does not separate them (2 s or less everywhere).
- **The league's definition is not better either.** It names a team at all 14 "nobody" moments, usually the eventual receiver.
- **Not tuned on these 40 calls** (too few to fit a rule). The open question is one of definition, for the founder:
  - **Keep team possession as it is.** It carries through the ball's flight, as SkillCorner's possessions and Gate 1's event reference do, and phases and chains depend on it.
  - **Add a separate observed ball-state field** on each grid point: controlled by home, controlled by away, in flight, or loose. That gives the AI "nobody had it yet" without changing possession.
  - **Either way, measure before agreeing:** the thresholds for "in flight" (ball speed, distance to the nearest player) would be swept against a larger judged sample, as for phases.

**The founder's rule and its test (18 November 2026).**
- **The founder's reading of the cases:** many are clearances or defensive touches. If the ball came from one team and bounced off the other, possession cannot be either team's until a player has gained control.
- **The evidence fits.** In 8 of the 14 "nobody" moments, the last player within 1 m of the ball was from the team P1 did not have in possession, 0.2 to 1.4 s earlier, with the ball then travelling at about 7 to 12 m/s. P1's minimum switch (0.4 s) ignores a touch that short by design, so it keeps the old team; the league gives the ball to the team that touched it.
- **Built as `p1_loose_on_opponent_touch`**, proposed and **off** (thresholds 0.9.0; P1's output unchanged). When on, a touch by the other team that does not last `p1_min_switch_s` makes the label `loose` until a player gains control. The original team's control ends it at once; the other team needs `p1_min_switch_s`. Chains are not broken by `loose`, and P4 labels no phase there.

| P1 | Off (agreed) | On |
|---|---|---|
| Founder's 50 judged moments | 21 | 22 |
| "Nobody" moments called loose | 0 of 14 | 5 of 14 |
| Team moments kept | 11 of 26 | 7 of 26 |
| Controls | 10 of 10 | 10 of 10 |
| Agreement where P1 names a team: league (IDSSE mean) | 87.0% | **92.2%** |
| Same: Metrica event-anchored (Gate 1 reference) | 93.6% to 95.7% | **96.6% to 98.2%** |
| Same: SkillCorner's player possessions | 95.1% | **96.6%** |
| Possession switches against the references | unchanged | unchanged |
| In-play time labelled `loose` | 0 | about 9% (IDSSE), 8% (SkillCorner) |

- **On every reference, the team P1 names is more often right when the rule is on,** because touches by the other side are no longer credited to the old team.
- **On the judged moments the rule is a wash** (22 against 21). It catches 5 of the 14 loose balls and loosens 4 moments the founder gave to P1's team. Proximity is not always a touch, and at 5 Hz a ball travelling 10 m/s moves 2 m between grid points.
- **Detecting touches at 25 Hz did no better** (18 to 21 of 50 for radii of 1 to 2 m and velocity changes of 3 to 5 m/s), so it was not pursued. Its prototype also missed the brief control that matters most.
- **Not adopted.** Turning it on changes P1's labels on about 9% of in-play time, and through them P2's possession states and P4's phases. That needs the founder's decision, the downstream checks (phase κ, the phase review) and a larger judged sample.

**Downstream checks with the rule on (18 November 2026, at the founder's request).** Measured with `--set p1_loose_on_opponent_touch=1` in `tools/context_phase_review.py` and `tools/context_reference_set.py`:

| | Off | On |
|---|---|---|
| Founder's phase review | 36 of 50 | **34** |
| SkillCorner κ, event spans, pooled (by half) | 0.6104 (0.6089 / 0.6104) | 0.6105 (0.6078 / 0.6116) |
| SkillCorner κ, phases-of-play file | 0.6249 | **0.6317** |
| Matches below the 0.50 phase floor | 2 | **4** (1996436, 2013725, 2015213, 2017461) |
| Roles: matches meeting 75% / 85% merged | 19 / 17 | 18 / 17 |
| Matches meeting all four bars | 10 | 9 |

- **Review moments affected:** 8 of the 50 become loose (no phase). The founder had accepted a phase at 6 of them, the same over-firing seen in the possession review.
- **Recommendation: keep it off.** The principle matches FIFA's "in contest" ("when neither team has controlled possession of the ball"), but a player near the ball is not a touch.
- **FIFA computes its possession control from the event sequence.** Where a logged feed exists, "in contest" can come from logged clearances, blocks and duels instead (see `ponta-blueprint/fifa-alignment.md`).

## Line breaks: a prototype for Stage C (18 November 2026)

**The rule** (`tools/context_line_breaks.py`, after FIFA's definitions in `ponta-blueprint/fifa-alignment.md`):
- **Units:** the defending team's outfield lines at the moment of the pass, each with at least two players. A unit's line is its deepest player.
- **Breaking a unit:** the ball starts in front of the line and is received beyond it.
- **Through or around:** through when the ball crosses the line between the unit's two widest players, around otherwise. FIFA's "over" needs the ball's height, which SkillCorner does not give.

**Measured against SkillCorner's line-break labels** on its 20 open-data matches, where SkillCorner knows the number of lines: 6,628 passes, 10 fit matches and 10 held out. Nothing was tuned except the choice of lines.

| | Any line broken, κ fit / held out | Furthest line broken, κ fit / held out |
|---|---|---|
| P2's lines (two-minute windows) | 0.64 / 0.63 | 0.48 / 0.49 |
| **Lines grouped at the pass moment** (FIFA's "at the moment of the pass") | **0.68 / 0.66** | **0.52 / 0.52** |

- **Adopted:** lines grouped at the pass moment.
- **Completed passes** (5,869): furthest line κ 0.54, 80% agreement.
- **Through or around**, where both name the same line: 82% agreement, κ 0.64.

**Failed passes and Plaiq's pass intent.**
- For a failed pass, FIFA judges the break by where the intended team-mate was. Plaiq's failed-pass inference gave an intended meeting point for 740 of the 759 failed passes.
- Using it changed nothing: furthest line κ 0.27 before, 0.26 after.
- **The reason:** Plaiq picks the *first* team-mate along the ball's line who would have controlled it, usually short of a long ball's target. Ponta then credits first-line breaks (170 against SkillCorner's 31) where SkillCorner credits last-line breaks (211 against 51). *(Corrected 18 November 2026, `PLAIQ_REFERENCES_REPORT.md`: on SkillCorner's data a failed pass's end point is already SkillCorner's own targeted player at the pass, not the interception point. The likelier reason for the low agreement is runs in behind: at the pass the runner is still in front of the back line, and SkillCorner credits the break.)*
- **Line breaks need the pass's target.** Plaiq's inference stays as it is for decision quality; changing it is an engine decision for the founder.

**Line-breaking options** (team-mates beyond at least one line from the ball, at the pass):
- **Geometry agrees.** For the team-mates SkillCorner itself lists as passing options on the reference match, Ponta and SkillCorner agree on which would break a line in 83% of 1,123 cases (κ 0.63).
- **Availability does not.**
  - Counting every team-mate beyond a line gives about 5 per pass, against SkillCorner's 0.9.
  - Keeping only those with Plaiq's option value at or above a threshold gives at best κ 0.11 on "any option" and rank correlation 0.19 (held out).
  - Plaiq's option value measures a pass's worth, not whether the team-mate is available. An availability test (for example Plaiq's control of the ball at the team-mate) has to be built and measured first.

**Built as P5 0.1.0 (18 November 2026, at the founder's request):** `plaiq_engine/context/line_breaks.py`.
- **Inputs:** completed passes with lines at the pass moment; failed passes listed with their breaks unknown.
- **Threshold:** `p5_min_unit_players` (2, FIFA's definition), proposed.
- **Its own output, scored against SkillCorner** (`tools/context_line_breaks.py --package`, 5,873 completed passes):
  - any line broken κ 0.65 / 0.63 (fit / held out);
  - furthest line κ 0.53 / 0.54 (79.7% agreement), the prototype's figure;
  - through or around 87% (κ 0.73).
- **Data check:** `line_breaks` is degraded on broadcast tracking.
- **Tests:** full local gate 1,080 passed, 12 skipped.

**Next, for the founder to decide:**
1. ~~Bring completed-pass line breaks into P5~~ done (P5 0.1.0).
2. A target rule for failed passes, in the context layer or as a Plaiq change.
3. An availability model for line-breaking options, measured against SkillCorner's option lists.

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

5. **Phase replay check again:** done on 18 November 2026.
   - The same 50 moments were re-rendered with P4 0.2.0 on 18 November 2026. The sample is drawn by the ball's P3 band, which did not change, so the moments are the same. The page holds positions, so it is kept outside the repository.
   - Seven cases changed label: Metrica 1 grid point 25371 (create to build_up), Metrica 2 grid points 4809 (finish to create), 18016 and 20789 (transition to finish), Metrica 3 grid point 25151 (set_play to chaotic), and SkillCorner grid points 7820 (create to build_up) and 20372 (finish to set_play).
   - Six of the seven already carried a label the founder had accepted.
   - **The founder re-judged the seven replays: all seven right.** On Metrica 3 grid point 25151 the note reads "this is right but the start frames show set_play, labelled moment is correct", so the P1 fix is confirmed there.
   - Kept in `tools/data/phase_review_2026-11-18_recheck.json`. `tools/context_phase_review.py` adds the re-check's accepted labels to the review's.
   - **The review now stands at 35 of 50 with P1 and P4 0.2.0.** The old rules with the corrected P1 also gain grid point 25151, so they score 29.

Taken on 18 November 2026 (later): the restarts round approved and done (P4 0.3.0; section above).

6. **The four throw-in thresholds (P4 0.3.0):** agreed (thresholds 0.8.0, no value changed).
7. **SkillCorner grid point 16221:** the founder confirmed that "no phase" (the ball is dead during the throw-in set-up) is right there, so the review stands at 36 of 50.

## Metrica CSV second-half events fixed (19 November 2026)

**The defect.** `matrix_engine.ingestion.metrica_io.to_single_playing_direction`, inherited from the prototype, flipped each table from the first second-half *tracking frame label* (about 70,000). On the events table, indexed by event row, that slice selected nothing. So the second-half event coordinates of Metrica's CSV games (Sample Games 1 and 2, and any Metrica CSV export through `providers/metrica.py`) were never flipped: they sat a median 55 m from the tracked ball.

**The fix.** Each table is now split by its own `Period` column. Measured on Games 1 and 2:
- tracking output is bit-identical;
- 817 and 856 event rows change, all in the second half;
- second-half pass starts now sit 0.35 m from the tracked ball, the same as first halves;
- Game 3 (EPTS) and SkillCorner never used this path and are unchanged.

**What changed downstream** (second halves of Games 1 and 2 only):
- **Context packages:** P1 to P4 are identical; the golden check differs only in P5's line-break fields. Second-half line breaks were undercounted, and the first halves are identical:

  | | Game 1 | Game 2 |
  |---|---|---|
  | Second half, before | 44 of 545 | 29 of 531 |
  | Second half, after | 155 | 134 |
  | First half | 125 of 511 | 167 of 666 |

  The goldens were recaptured; P5's version is unchanged because its rules did not change.
- **Plaiq:**

  | | Game 1 | Game 2 |
  |---|---|---|
  | Events with ΔV changed | 569 | 538 |
  | Decision-quality rows changed | 427 | 418 |
  | Failed-pass decision-quality rows changed | 74 | 54 |
  | Pass option value rows changed | 8,870 | 9,274 |
  | Scored decisions | 900 → 941 | 1,029 → 1,078 |

  Neither game is in the Plaiq equivalence goldens (Game 3 and SkillCorner), and the decision-quality bands were audited on Game 3, so no recorded Plaiq evidence changes.
- **Semantic events:** `pitch_zone` and every value above, on those halves.

**Not changed:** the frozen prototype carries the same defect and stays frozen. Its records cover Game 3 and SkillCorner, which are unaffected.

## The Gate 2a round: closing the set (20 November 2026)

**Where the set stood** (`tools/context_reference_set.py`, current code, reproduced): formations 20 of 20; phases pooled κ 0.609 and 0.610 by half; roles grouped 19 of 20 and merged 17 of 20; two matches below the founder's 0.50 phase floor (2013725 first half 0.482, 2017461 second half 0.490). 2015213, one of the three earlier failures, now sits at 0.501.

### Roles: 1959846 plays a shape P2 has no template for

- **Out of possession in the first half, the home team's mean positions** (depth from its own goal, width): a flat back four at 34 to 38 m (centre-backs at width 7 and −5 m, full-backs at ±15 m), two central midfielders at 42 to 46 m, two wide players pushed up to 50 to 53 m (width −12 and +18 m), and two strikers at 55 to 59 m. That is a 4-2-2-2, a box midfield. SkillCorner's 4-4-2 line-up is consistent with the tracking; P2's fit is not.
- **With no template for the shape, P2 fits a 3-4-2-1:** the right-back becomes a centre-back, the wide players wing-backs or attacking midfielders, the strikers attacking midfielders.
- **Candidate: a 4-2-2-2 template** (`p2_template_4_2_2_2`, proposed and off, thresholds 0.11.0), its coordinates set from the other templates before scoring. Measured on the 20 matches:

| Roles | Without (P2 0.2.0) | With the 4-2-2-2 |
|---|---|---|
| 1959846 grouped / merged | 58.7% / 78.4% | 62.4% / 82.9% (still below both bars) |
| Matches meeting the grouped bar (75%) | 19 | 18 |
| Matches meeting the merged bar (85%) | 17 | 18 |
| Grouped median | 92.2% | 86.7% |

  The template catches teams SkillCorner lists as 4-2-3-1 or 4-4-2 and costs four held-out matches 4 to 11 points; phases are unchanged. **Not adopted.** The switch stays off as the record.
- **2006229 (84.4%) and 2016236 (84.0%)** miss the merged bar by about a point, with misses spread across players (forwards read as central midfielders, a right-back as a centre-back, SkillCorner's "RM" as Ponta's wide midfielder). No single pattern; fixing them one by one would tune to noise.
- **Pooled over all 583 players:** 84.1% by position group (bar 75%) and 89.0% with central midfield merged (bar 85%).
- **Proposed set-level reading for roles, as for phases:** pooled over the set, at least 75% grouped and 85% merged (met), with the per-match bars kept as a reading (19 and 17 of 20) and 1959846 recorded as a known limit: P2 has no box-midfield template, and adding one costs more than it gains.

### Phases: the floor failures come from the own-third build-up rule

- **On both failing halves the largest disagreement is SkillCorner "create", Ponta "build_up"** (1,015 grid points on 2013725's first half; 286 on 2017461's second).
- **The cause is the founder's own-third rule** (build-up whenever the ball is in the own third, adopted from the replay review, P4 0.2.0). With it off, 2013725's first half rises from 0.482 to 0.567 and 2017461's second from 0.490 to 0.508.
- **Swept the boundary** (`p4_build_up_max_depth_share`) rather than dropping the rule; review scored with Metrica pickles rebuilt after the 19 November second-half fix:

| Build-up whenever the ball is within | Review (of 50) | SkillCorner κ by half (event spans) | Phases-of-play file | Matches with κ ≥ 0.55 on both halves | Lowest half |
|---|---|---|---|---|---|
| a third (agreed, 0.333) | 36 | 0.609 / 0.610 | 0.625 | 13 | 0.482 |
| 0.30 of the pitch | 35 | 0.653 / 0.637 | 0.655 | 16 | 0.510 |
| **a quarter (0.25)** | **35** | **0.670 / 0.638** | **0.652** | **18** | **0.524** |
| off (0) | 34 | 0.638 / 0.604 | 0.619 | | |

- **Held out by match** (points-weighted κ over each half of the matches, half 1 / half 2): a third gives 0.612 / 0.638 on the even matches and 0.593 / 0.573 on the odd; a quarter gives 0.672 / 0.659 and **0.656 / 0.609**. The gain holds on the matches it was not chosen on.
- **The one review case lost:** Metrica 1 grid point 25371, where the ball is between a quarter and a third of the pitch and the founder accepted build-up. The two "recycling backwards" cases (Metrica 2 grid point 5864, Metrica 3 grid point 11497) are not fixed by any boundary.
- **A quarter beats both a third and no rule:** the founder's rule is right in kind; the boundary was too far out.
- **Proposed for the founder:** `p4_build_up_max_depth_share` from 0.333 to 0.25 (P4 0.4.0). With it, no match is below the 0.50 floor (lowest 0.524), 18 of 20 meet κ 0.55 on both halves, and the founder's set-level phase bar is met. **The agreed value is unchanged until the founder agrees.**

### What closes Gate 2a across the set

- **Formations:** met (20 of 20).
- **Phases:** met pooled now; the floor is met with the boundary at a quarter.
- **Roles:** met on the proposed set-level reading; per match 19 and 17 of 20, 1959846 a recorded limit.
- **The review items left open** move to Stage C (`ponta-blueprint/context-relationships-task.md`): long balls and losses in flight need "in contest" from logged events (P1 0.3.0); a goal kick credited to the receiving team while the ball is undetected is the same P1 gap; recycling backwards and a short regain stay as review notes for Gate 2b.

**Decided by the founder on 20 November 2026:** both proposals agreed.
- **Build-up within a quarter of the pitch** (`p4_build_up_max_depth_share` 0.25; thresholds 0.12.0, P4 0.4.0). Golden outputs recaptured: on the four regression matches only P4's phase fields and the version stamps change.
- **Roles are judged on the set,** as phases are: pooled over every player at least 75% grouped and 85% with central midfield merged, with the per-match bars kept as a reading. 1959846's box midfield is a recorded limit of P2's templates.
- **Gate 2a across the 20 matches, on the agreed settings** (`tools/context_reference_set.py`, which now reports the set-level reading as `gate_2a_set`): phases pooled κ 0.670 and 0.638 by half, no match below 0.50 (lowest 0.524), 18 of 20 at κ 0.55 on both halves; roles pooled 84.1% grouped and 89.0% merged (per match 19 and 17 of 20); formations 20 of 20. **Gate 2a is met across the set.** Gate 2b (analysts) still comes before player-facing use.

## Three phase rules rechecked against SkillCorner alone (23 November 2026)

The founder asked that decisions rest on published definitions and professional reference data, with his own replay calls only breaking ties. Three P4 rules began in his replay review (18 November 2026) and were then measured with SkillCorner as well; each was rerun on SkillCorner's 20 matches alone, the rule on against off (`tools/context_reference_set.py --set`; phase κ against SkillCorner's phases-of-play file, pooled, with the held-out matches in brackets):

| Rule | On (as agreed) | Off | Other value |
|---|---|---|---|
| Build-up whatever the block within a quarter of the pitch (`p4_build_up_max_depth_share` 0.25) | **0.652** (0.637) | 0.619 (0.604) | |
| Final third needs the ball in the attacking half (`p4_finish_min_depth_share` 0.5) | **0.652** (0.637) | 0.638 (0.624) | 0.6: 0.644 (0.630) |
| A break ends at the final third (`p4_break_end_depth_share` 0.667) | 0.652 (0.637) | 0.652 (0.639) | 0.6: 0.652; 0.75: 0.652 |

- **The build-up and final-third rules stand on SkillCorner's data alone**; their basis is `reference`.
- **The break-end rule has no measurable effect** on SkillCorner's labels (switched off, the lowest match half rises from 0.524 to 0.546). It rests on the founder's review, so its basis is now `judgement`. Switching it off is a decision for the founder; nothing was changed.

**Decided by the founder on 23 November 2026:** the break-end rule is switched off (`p4_break_end_depth_share` 0; thresholds 0.24.0, P4 0.5.0). Breaks now end by the recovery rule (enough defenders goal-side), the end of the chain or `p4_break_max_s` (both engineering choices, not published values). Measured as shipped on SkillCorner's 20 matches: pooled phase κ 0.672 / 0.640 by half (was 0.670 / 0.638), against the phases-of-play file 0.652 (held out 0.639), lowest match half 0.546 (was 0.524); the Gate 2a bars still hold (18 of 20 matches at κ 0.55 on both halves, none below 0.50).
