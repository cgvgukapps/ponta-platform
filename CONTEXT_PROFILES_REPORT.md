# Team Identity, Match Profiles and Baselines — Report (Stage E)

Stage E of `ponta-blueprint/context-packages.md`, built to `ponta-blueprint/context-profiles-task.md` (23 November 2026). Decisions follow the founder's order (published definitions, then professional reference data, then outcomes; the founder's view only to break a tie). *(25 November 2026: "held-out" halves here are halves of one development set used across tuning rounds, not an independent confirmation; see section 10.)*

## 1. What was built

- **Match metadata** (`matrix_engine/match.py`): `NormalisedMatch` gains `teams` (the provider's team id per side, never a name), `competition` (`"<provider>:<id>"`) and `season`. SkillCorner's match file and the German league's match information supply them; Metrica's sample games have none. Matches cached before Stage E still load (class defaults), and the loaders add the metadata.
- **Team keys** (`plaiq_engine/teams.py`): `team_key` is a keyed hash (HMAC-SHA256) of the provider and its team id under a secret the caller supplies (one per tenant in the product; a public reference secret here). The same club from the same provider gets the same key in every match, and the key reveals nothing about the club. One club across two providers is joined only by an explicit merge. A side with no provider id gets a key for its match only.
- **Semantic schema 1.8** (`semantic_builder.py`): every event carries `actor_team_key`, `opponent_team_key`, `competition` and `season`; `semantic_states` carries `team_key`, `opp_team_key`, `competition` and `season`. Earlier columns unchanged (tested).
- **Team match profiles** (`plaiq_engine/profiles.py`): one row per team, match and measure, in long form, with the value, denominator, sample minutes and count, label, basis, source, the Data check state of its inputs, whether the match had an event feed, and the versions. **46 measures** in a catalogue, by basis:
  - **published (FIFA EFI, 15):** receptions behind the midfield and defensive lines, final-third entries and their share by lane, defensive line height out of possession by ball third, team length in and out of possession, ball recovery time, forced turnovers;
  - **published (event, 1):** PPDA as Trainor (2014) defines it, from the logged feed;
  - **common public definitions (2):** time-based possession share, field tilt;
  - **Ponta's (28):** phase shares, block and press shares out of possession, build-up opposed share, presses and engagements by type per minute out of possession, how chains end, runs in behind, carrier pressure applied and received, open high-value space, median Gain and median Choice of passes (apart, never combined).
- **Baselines and comparisons:** a team's baseline (its other matches) and the set's (same competition and season), refusing mixed versions and leaving Blocked matches out; empirical Bayes shrinkage towards the set (weight = within-team variance over between-team variance); a verdict ("above / about / below your usual") only at or above the measure's minimum number of matches (`MIN_MATCHES`) and beyond `pf_verdict_sd` (1.0, proposed) within-team standard deviations.
- **Pattern citations** (`citations.resolve_profile_citation`): `[team_key, measure, match_ids]`, resolved only when the measure is in the catalogue, every cited match has a usable profile, the versions agree and the matches meet the measure's minimum.
- **Thresholds 0.25.0:** `pf_verdict_sd` (1.0, engineering, proposed) and `pf_reliability_target` (0.7, published: the conventional threshold for acceptable reliability, Nunnally 1978; proposed).
- **Tools:** `tools/context_profiles.py` (`build` caches each match's profiles outside the repository; `analyse` measures). Built for all 27 reference matches (SkillCorner's 20, the German league's 7): 2,484 profile rows, one set of versions, 2.9% Blocked: on the German league, whose events carry no pass end points, field tilt, receptions behind the lines and Gain and Choice are Blocked (14 team-matches each), and one SkillCorner PPDA.

## 2. What the reference data can show

- **SkillCorner's 20 matches:** 13 teams; 12 appear in two or more matches (one in 7, three in 4, four in 3, four in 2). Stability is measured on these.
- **The German league's 7 matches:** 2 Bundesliga matches between four clubs on the last matchday of 2022/23, and 5 home matches of one 2. Bundesliga club. Only one team recurs, always at home, so it cannot separate a team's habits from its home ground; used for agreement with the official data, not for stability.
- **Nothing here sets a norm.** A club's own season is what baselines are for.

## 3. Agreement with the references (comparison only)

| Measure | Reference | Agreement (team-matches) |
|---|---|---|
| Possession share | German league official possession | r 0.91, mean difference 2.4 percentage points (14) |
| Phase share: build-up / final third | SkillCorner's phases of play | Spearman 0.90 / 0.91 (40) |
| Phase share: progression / contested / long ball | the same | 0.69 / 0.68 / 0.66 |
| Phase share: set play / transition / counter-attack | the same | 0.53 / 0.47 / 0.32 |
| Presses per match | SkillCorner's pressing chains | 0.57 |

## 4. Stability: which measures describe a team

The intraclass correlation (ICC(1), one-way random effects, unequal group sizes) says how much of a measure's spread lies between teams rather than between a team's own matches; a 90% interval comes from resampling teams. Spearman–Brown gives the matches a team needs for its mean to reach reliability 0.7 (and 0.8).

| Measure (basis) | ICC (90% interval) | Matches for 0.7 / 0.8 |
|---|---|---|
| Time in a high block out of possession (Ponta) | **0.52** (0.23 to 0.70) | **3** / 4 |
| Field tilt (common) | **0.42** (0.12 to 0.61) | **4** / 6 |
| Defensive line height out of possession, ball in the middle third (FIFA EFI) | **0.40** (0.15 to 0.69) | **4** / 7 |
| Time in a low block (Ponta) | **0.36** (0.11 to 0.51) | **5** / 8 |
| Final-third entries down the left wing, share (FIFA EFI) | 0.40 (0.04 to 0.60) | 4 / 7 |
| Build-up share of possession (Ponta) | 0.35 (0.03 to 0.48) | 5 / 8 |
| Carrier pressure applied (Ponta) | 0.31 (0.03 to 0.46) | 6 / 10 |
| Pressure engagements per minute (Ponta) | 0.29 (0.02 to 0.60) | 6 / 10 |
| Defensive line height, ball in the attacking third (FIFA EFI) | 0.28 (0.05 to 0.47) | 7 / 11 |
| Median Choice of passes (Ponta) | 0.28 (0.09 to 0.37) | 7 / 11 |

- **Everything else is below 0.25 or has an interval reaching zero**: on two to seven matches a team, most measures vary as much within a team as between teams. Among them: possession share (0.22), PPDA (0.01), final-third entries (0.19), receptions behind the lines (0.19, 0.30), median Gain (0.09), counter-pressing (0.01 to 0.15), runs in behind (below zero).
- **Proposed minimum matches** (`MIN_MATCHES`, basis reference; *revised to 5, 5, 6, 7 on 24 November 2026 after PFF's set, section 8*): only the four measures whose interval lies above 0.1, the top four rows. With 46 measures tested on 12 teams, a few would clear a looser bar by chance, so the bar is set where chance is unlikely. Every other measure answers "the minimum is not measured yet" until a club's own matches show it is stable. Block height, field tilt and line height are also known style traits in published analysis, which supports reading them as team habits.

## 5. Does a baseline help?

Leaving one match out, the shrunk team baseline's error against the set mean's, on teams with three or more matches: the best reductions are field tilt 17%, length out of possession 15%, build-up share 13%, pressure engagements 11%, Choice 10%, time in a high block 9%; many measures are at or below zero. **With so few matches per team, a team's baseline is only modestly better than the set's**, which is what shrinkage is for: it keeps a short history from being read as a tendency.

## 6. Checks

- Every profile row has a label, a basis and a Data check state; no name, coordinate or cell in any column (tested). *(Corrected 25 November 2026, review AI-05 and AI-04: as first written this also said "no value is shown for a Blocked row" and "no provider id in any column". The first was false (28 of 71 Blocked rows kept a value); it is true since profiles 0.2.0 and tested both ways. The second is still false: `match_id` and `competition` carry the provider's ids. Opaque match and competition keys are backlog B, before the AI layer.)*
- The cache holds one set of versions; baselines refuse mixed versions (tested).
- Golden outputs recaptured for thresholds 0.25.0; a second run matches.

## 7. Deviations from the spec

- **Split-half reliability** is computed from the intraclass correlation (Spearman–Brown on the ICC) rather than by splitting each team's matches: only four SkillCorner teams have four or more matches, too few for an empirical split.
- **The German league's events have no pass end points,** so field tilt, receptions behind the lines, Gain and Choice are Blocked there; stability uses SkillCorner only in any case.
- The German league's agreement check covers possession share only; its PPDA would be computed from the same official events Ponta reads, so it is not an independent check.

## 8. A second reference: PFF FC's FIFA World Cup 2022 (24 November 2026)

> **Source: PFF FC, FIFA World Cup 2022 broadcast tracking and events. No licence is stated and the specifications are marked proprietary.** Used for internal development and measurement only, under the founder's decision of 23 November 2026 (`ponta-blueprint/legal-compliance.md`). Nothing in this section, or derived from it, goes to a customer, pilot, demo, the website or any public claim without PFF's written approval. The data stays outside every repository; profiles use team keys and pseudonymised player ids, never names.

**Why:** SkillCorner's set has 12 recurring teams on 2 to 7 matches, so its stability intervals are wide. PFF's set has **all 32 teams of a tournament, each on 3 to 7 matches (128 team-matches)**.

**How it was read** (`tools/pff.py`, a reference reader, not a supported provider):
- PFF's smoothed player positions and the **tracked** ball. PFF's own smoothed ball is placed from its events, so using it would make Ponta's possession agree with PFF by construction.
- Each period is rotated so the home team attacks +x, from where the home goalkeeper stands.
- PFF's events are translated into the event vocabulary P1 reads. Each event sits on the frame PFF links it to.
- Run at **9.99 Hz (every third video frame)**, SkillCorner's rate, to fit the machine (about 330 s a match). On one match against the native 29.97 fps, the median relative difference across the profile was 1.2%: 72% of values within 5% and 84% within 10%. The larger differences fall on shares under 0.1 and on whole-number medians. This is an engineering choice.
- **Tracking quality:**
  - Only a **median 38% of player positions are seen** on camera; PFF estimates the rest. The ball is tracked in 68% of frames.
  - When these profiles were built, the Data check marked every PFF profile row Full because every player has a position: it could not see that most positions are estimated. *(Fixed 24 November 2026, Data check schema 1.1: the provider's flag is kept, an estimated position counts as a missing one, and position-based metrics on PFF's and SkillCorner's broadcast tracking are Degraded with the measured share. The cached profiles predate the fix; a rebuild would carry the new states. At 9.99 Hz the PFF reader also falls just under the 10 Hz bar.)*
- **Extra time:**
  - A period is now kept only if the tracked player nearest the ball, at PFF's passes, crosses and shots, is on the acting team more often than chance. This is an engineering check.
  - Normal time scores a median 0.78. In two matches' extra time (10506, 10517) the score is 0.20 to 0.36: the players' tracking does not line up with PFF's own ball and events, and no time offset within five minutes repairs it. Those periods are dropped.
  - PFF did not track extra time in two other matches (10510, 10511); their extra-time events, about 700 a match, have no frame and are not used.
  - Elsewhere, at most 8 events a match have no frame.

**Agreement with PFF's own event data** (comparison only; 128 team-matches):

| Ponta measure | PFF's figure | Agreement |
|---|---|---|
| Possession share | share of on-ball time | r 0.95, mean difference 3.2 points |
| Receptions behind midfield / behind the defence (FIFA EFI) | passes and carries that break the midfield / defensive line | Spearman 0.64 / 0.42 |
| PPDA | pressures applied per minute of the opponents' on-ball time | −0.72 (the expected direction: fewer passes allowed per action means more pressure) |
| Carrier pressure applied (median) | the same | 0.53 |
| Counter-press engagements per minute out of possession | the same | 0.47 |
| Pressure engagements / presses / all engagements per minute out of possession | the same | **−0.47 / −0.05 / −0.22** |

- PFF's pressure rate rises with a team's own possession (Spearman 0.59): teams that keep the ball press hard when they lose it. A per-90 count would instead reward teams for defending longer, so the comparison uses the rate.
- **Ponta's engagement counts do not follow PFF's pressure rate.** Only the counter-press type does, while the pressure type runs the other way. PFF records pressure per on-ball event, within about 5 yards and closing, while P4v counts defenders' approaches. Some disagreement is expected from the units, but not the sign. This is unresolved, and engagement rates stay unvalidated (P4v's type κ against SkillCorner was 0.54).

**Stability on 32 teams** (ICC(1), 90% interval from resampling teams; matches for reliability 0.7 / 0.8):

| Measure | SkillCorner (12 teams) | PFF (32 teams) |
|---|---|---|
| Time in a high block (minimum 3, now 5) | 0.52 (0.23 to 0.70): 3 / 4 | 0.33 (0.17 to 0.47): **5** / 9 |
| Field tilt (minimum 4, now 5) | 0.42 (0.12 to 0.61): 4 / 6 | 0.36 (0.20 to 0.49): **5** / 7 |
| Line height, ball in the middle third (minimum 4, now 6) | 0.40 (0.15 to 0.69): 4 / 7 | 0.30 (0.06 to 0.47): **6** / 10 |
| Time in a low block (minimum 5, now 7) | 0.36 (0.11 to 0.51): 5 / 8 | 0.26 (0.10 to 0.40): **7** / 12 |
| Possession share | 0.22 (−0.16 to 0.49) | 0.37 (0.19 to 0.51): 4 / 7 |
| Time in a mid block | 0.01 (−0.31 to 0.19) | 0.40 (0.18 to 0.55): 4 / 6 |
| PPDA | 0.01 (−0.25 to 0.22) | 0.28 (0.10 to 0.39): 7 / 11 |
| Receptions behind midfield | 0.19 (−0.02 to 0.35) | 0.36 (0.19 to 0.47): 5 / 7 |
| Team length out of possession, ball in midfield | 0.27 (−0.16 to 0.55) | 0.33 (0.19 to 0.43): 5 / 8 |
| Long-ball and contested phase shares | 0.23 / 0.14 | 0.38 / 0.37: 4 / 5 |
| Median Choice of passes | 0.28 (0.09 to 0.37): 7 | 0.03 (−0.10 to 0.15): 66 |
| Median Gain of passes | 0.09 | 0.05 |

- **The four agreed measures are stable on both sets, and each needs more matches on PFF's:** high block 5 against 3, field tilt 5 against 4, line height 6 against 4, low block 7 against 5. PFF's intervals are narrower (32 teams against 12). On PFF, line height's interval reaches down to 0.06 and low block's to 0.10. Field tilt and the high block clear the report's bar on both sets.
- **Measures that look stable only on PFF's set** (possession share, mid block, PPDA, receptions behind midfield, length, long-ball and contested shares) are not stable on SkillCorner's. A World Cup spans a much wider range of team strength than one league, which raises the variance between teams. These measures are therefore not added; they stay at "the minimum is not measured yet".
- **Choice is not stable on PFF's set** (0.03 against SkillCorner's 0.28). Choice reads every player's position, and on PFF's set most positions are estimated by the provider. That may explain the gap, but it is not measured.
- **A baseline helps a little more on PFF's set:** the largest reductions in error are the contested share 16%, long-ball share 14%, mid block 14%, line height with the ball in the attacking third 13%, field tilt 11% and high block 9%. 17 of 46 measures are at or below zero.
- **Context:** these are national teams, 3 to 7 matches each over four weeks, against opponents drawn by the tournament. A club season differs on every count, so neither set is a norm.

## 9. Decisions for the founder

1. *(Agreed by the founder, 23 November 2026.)* **The minimum matches** for the four stable measures (3, 4, 4, 5), and every other measure without a verdict until a club's own data shows it is stable.
   - **Revised by the founder, 24 November 2026: the larger of the two sets, 5, 5, 6, 7** (high block, field tilt, line height with the ball in the middle third, low block), in `MIN_MATCHES`. PFF's set needs more matches for each, and its estimate rests on 32 teams against 12 (section 8).
   - No new measures until a club's own season shows them stable.
   - **Gate:** the revised values derive from PFF's data, so before they govern anything a customer, pilot or demo sees, PFF's written approval is needed or they are re-measured on licensed data (a design partner's season).
2. **`pf_verdict_sd`** (1.0 within-team standard deviation for "above / below your usual"): an engineering choice; to be measured on a design partner's season.
3. **More data:** a design partner's season is what makes baselines useful. PFF's set is now read (section 8); StatsBomb's open events are next once the founder has registered.

## Data sources and licences

*(Added 25 November 2026, review GC-14.)* Reference data this report measures against, all kept outside every repository:

- **SkillCorner open data** (20 matches): MIT licence; credit SkillCorner.
- **Metrica Sports sample data** (games 1 to 3): published by Metrica for research; licence not stated (`ponta-blueprint/legal-compliance.md`). Never copied into a repository.
- **The German league's open data (IDSSE)** (7 matches): CC BY 4.0. Data from the DFL (Deutsche Fußball Liga), published by Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12, 195.
- **PFF FC's FIFA World Cup 2022 data**: no licence; internal development and measurement only. Nothing derived from it reaches a customer, pilot, demo, the website or a public claim before PFF's written approval (`DERIVED_DATA_INVENTORY.md`).
