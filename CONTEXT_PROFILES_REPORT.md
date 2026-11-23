# Team Identity, Match Profiles and Baselines — Report (Stage E)

Stage E of `ponta-blueprint/context-packages.md`, built to `ponta-blueprint/context-profiles-task.md` (23 November 2026). Decisions follow the founder's order (published definitions, then professional reference data on held-out matches, then outcomes; his view only to break a tie).

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
- **Proposed minimum matches** (`MIN_MATCHES`, basis reference): only the four measures whose interval lies above 0.1, the top four rows. With 46 measures tested on 12 teams, a few would clear a looser bar by chance, so the bar is set where chance is unlikely. Every other measure answers "the minimum is not measured yet" until a club's own matches show it is stable. Block height, field tilt and line height are also known style traits in published analysis, which supports reading them as team habits.

## 5. Does a baseline help?

Leaving one match out, the shrunk team baseline's error against the set mean's, on teams with three or more matches: the best reductions are field tilt 17%, length out of possession 15%, build-up share 13%, pressure engagements 11%, Choice 10%, time in a high block 9%; many measures are at or below zero. **With so few matches per team, a team's baseline is only modestly better than the set's**, which is what shrinkage is for: it keeps a short history from being read as a tendency.

## 6. Checks

- Every profile row has a label, a basis and a Data check state; no value is shown for a Blocked row; no name, provider id, coordinate or cell in any column (tested).
- The cache holds one set of versions; baselines refuse mixed versions (tested).
- Golden outputs recaptured for thresholds 0.25.0; a second run matches.

## 7. Deviations from the spec

- **Split-half reliability** is computed from the intraclass correlation (Spearman–Brown on the ICC) rather than by splitting each team's matches: only four SkillCorner teams have four or more matches, too few for an empirical split.
- **The German league's events have no pass end points,** so field tilt, receptions behind the lines, Gain and Choice are Blocked there; stability uses SkillCorner only in any case.
- The German league's agreement check covers possession share only; its PPDA would be computed from the same official events Ponta reads, so it is not an independent check.

## 8. Decisions for the founder

1. *(Agreed by the founder, 23 November 2026.)* **The minimum matches** for the four stable measures (3, 4, 4, 5), and every other measure without a verdict until a club's own data shows it is stable.
2. **`pf_verdict_sd`** (1.0 within-team standard deviation for "above / below your usual"): an engineering choice; to be measured on a design partner's season.
3. **More data:** a design partner's season is what makes baselines useful; open sources with recurring teams (StatsBomb's open events, PFF FC's World Cup 2022 tracking) could be checked for licence and value first.
