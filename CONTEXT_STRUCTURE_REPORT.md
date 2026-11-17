# Context Structure Report (P2, P3, P4)

17 November 2026. Task: `ponta-blueprint/context-structure-task.md`; specification: `ponta-blueprint/context-packages.md`. Stage B of the context packages, built under founder decision 4a (proceed against reference labels; analyst agreement still gates player-facing use).

## Result in one paragraph

All three packages are built on the Stage A registry. They run on all four regression matches; Stage B adds about 13 s and 3.3 MB per match (16 s and 4 MB with P1). Outputs are identical across runs.

- **Roles:** each player's main role agrees with SkillCorner's role by position group for **79% of playing time** (70% of players). With the three central-midfield roles merged it is **91% of playing time** (85% of players). Most misses involve the defensive, central and attacking midfield split, or an attacking midfielder against a forward or wide player.
- **Phases:** in-possession phases agree with SkillCorner's on **73.1%** of compared time, **Cohen's κ 0.60** (0.58 on the first half, 0.61 on the second). Published agreement between human analysts is about κ 0.7.
- **Formations** are stable, with at most one change per team per half on every match.
- **Shape measures** fall inside published ranges for official matches.
- **Zones** wait for the founder's spot check.

These are agreements with one provider's labels on one match, not accuracy against a hand-labelled truth. **Gate 2a is not declared passed:** the founder sets the bars from these numbers.

## What was built

| Piece | Where |
|---|---|
| P2 team structure: per-frame shape and lines; per-window lines, formation, roles; current formation with persistence | `plaiq_engine/context/structure.py` |
| P3 relative zones: ball lane, third, band against the defending lines, hull, pocket; attacker counts | `plaiq_engine/context/zones.py` |
| P4 phase of play: rule-based state machine, paired out-of-possession phase, block, phase segments | `plaiq_engine/context/phase.py` |
| Stage B thresholds (P2 to P4), each with status, reason and source | `plaiq_engine/context/thresholds.py` (version 0.3.0; P1 values unchanged) |
| Reference readers (SkillCorner roles, phases as points and as spans; no names) | `tools/context_references.py` |
| Phase sweep (half-split) | `tools/context_phase_sweep.py` |
| Measurements for this report | `tools/context_structure_measure.py` |
| Spot-check page for zones and phases (written outside the repository) | `tools/context_spotcheck.py` |
| Golden harness registers P2 to P4; manifest updated | `tools/context_golden.py`, `tests/golden/context/manifest.json` |
| Tests: 27 synthetic (building blocks, all ten templates, mirrored pitch, determinism) and one on sample data | `packages/plaiq_engine/tests/test_context_structure.py`, `tests/cross_engine/test_context_sample.py`, fixture `tests/support/context_fixture.py` (`eleven_a_side`) |
| Field definitions | `docs/context/vocabulary.md` (P0 to P4) |
| Architecture decisions 16 to 18 | `docs/ARCHITECTURE.md` |

## A data finding: Metrica Game 2's direction of play

In Metrica Sample Game 2 the home goalkeeper's mean x is +39.7 m. The normalised data therefore has the home team attacking −x in both periods, against `PitchConfig.home_attacks_positive_x=True`. The Metrica CSV loader flips the second half and assumes the first-half direction; in Game 2 that assumption is wrong. Games 1 and 3 and SkillCorner are correct.

- **Stage B:** reads each team's direction per period from the data (the side its goalkeeper defends) and is unaffected.
- **The loader:** is not changed. This task makes no engine changes; the finding is reported here.
- **Who it may affect:** anything that assumes the home team attacks +x on Game 2 (decision-quality direction, attacking-third labels in the semantic events). The prototype used the same loader, so its Game 2 outputs share the assumption. **Recommended:** a small Matrix fix with a regression test, as its own task.

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

## Proposed thresholds (for the founder to agree or change)

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

## Proposed Gate 2a bars

| Package | Measure | Measured | Proposed bar |
|---|---|---|---|
| P2 roles | Position-group agreement with SkillCorner, weighted by playing time | 79.3% | **at least 75%** |
| P2 roles | Same, central midfield merged | 90.5% | **at least 85%** |
| P2 formations | Changes per team, state and half | at most 1 everywhere | **at most 2**, and plausible on the spot check |
| P3 zones | Founder's spot check (50 cards) | not yet done | **at least 45 of 50** zone labels judged right |
| P4 phases | Cohen's κ against SkillCorner's in-possession phases | 0.60 (0.58 / 0.61 by half) | **at least 0.55**, on each half |

The phase bar sits below published analyst agreement (κ about 0.7) because the reference is one provider's rules, not a consensus.

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
- **Full local gate:** 1,057 passed, 12 skipped.
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
- **References:** one match with role and phase labels, one provider. SkillCorner's open data reportedly has phases for 10 more matches under an MIT licence; using them would test whether the bars and the block boundaries generalise.
- **No analyst check:** every Stage B label is "estimated, checked against references, not yet validated by analysts" until Gate 2b.

## Decisions for the founder

1. **Gate 2a bars:** agree the five bars above, or change them.
2. **Thresholds:** agree the measured P4 values (26 m, 47 m, 15 m) and the proposed P2 to P4 values, or change them.
3. **Zone spot check:**
   - Open the page written by `tools/context_spotcheck.py`.
   - Tick the zone and phase boxes.
   - Copy the results back.
4. **Metrica Game 2 direction:** approve a separate small Matrix task to fix the loader, with a regression test, and check what downstream outputs change.
5. **More reference matches:** approve downloading SkillCorner's open data (10 matches with phases of play) outside the repositories, after checking its licence, to re-run the role and phase measurements.
