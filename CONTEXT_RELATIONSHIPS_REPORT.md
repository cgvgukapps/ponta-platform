# Context Relationships, Movement and Space — Report (Stage C)

Stage C of `ponta-blueprint/context-packages.md`, built to `ponta-blueprint/context-relationships-task.md`. Written deliverable by deliverable; each section records what was built, how it was measured and what the founder decides.

**On the founder's judged moments** (the founder, 20 November 2026): they are one expert's reading, an indication to weigh beside the provider references, never ground truth. Scores against them are written as agreement with the founder's call, and no rule is tuned to them alone.

## 1. In contest from logged events (P1 0.3.0, 20 November 2026)

**Built** (`plaiq_engine/context/possession.py`):
- **`p1_state`**, FIFA's possession states beside `p1_possession` (unchanged): `in_possession_home`, `in_possession_away`, `in_contest`, `out_of_play`, `unknown`, with `p1_state_source`.
- **In contest** where both teams are at the ball (P1's `contested`) and, with a logged feed, from a logged duel (`CHALLENGE`), clearance, block or deflection until a team has held control for `p1_min_switch_s` (0.4 s), for at most `p1_contest_max_s` (proposed 4 s). Interceptions are not contests: they win the ball.
- **IDSSE's defensive clearances** (`OtherBallAction` with `DefensiveClearance`) now reach P1 as `CLEARANCE` events (`tools/idsse.py`). That type is not one the event-anchored possession reads, so `p1_possession` is unchanged there too.
- **Chains** gain `interruptions` (the other team's logged contest touches inside the chain: FIFA's "interrupted", touched by the opponent with no turnover) and `out_of_play` (`retained` or `lost` by which team restarts, for a chain ending in a dead ball).
- **Data check:** a `possession_state` entry, Full with logged contest events, Degraded without (in contest only where both teams are at the ball).
- **Thresholds 0.13.0** (`p1_contest_max_s`, proposed). P1 0.3.0. Golden outputs recaptured: on the four regression matches only P1's new fields, its chain table and version stamps change; P2 to P4 are untouched.

**Measured** (`tools/context_contest.py`) against the founder's 50 blind possession calls on the German league's matches (18 November 2026: home, away or nobody yet):

| | `p1_possession` (unchanged) | `p1_state`, cap 2 s or 4 s | `p1_state`, cap 6 s |
|---|---|---|---|
| "Nobody" moments called in contest | 0 of 14 | 2 | 3 |
| Team moments kept | 11 of 26 | 10 | 10 |
| Controls kept | 10 of 10 | 9 | 9 |
| Agrees with the founder's call, in total | 21 of 50 | 21 | 22 |

**In-play time in contest** (cap 4 s): 11% to 16% on the seven IDSSE matches (8.5% to 11.4% from logged events), 9% to 11% on Metrica Games 1 to 3 (5% to 8% from events), 2.7% on SkillCorner 1886347 (no logged feed: both teams at the ball only).

**Why it does not find the founder's "nobody" moments.** Of the 14:
- **2** follow a logged duel, and the state catches them;
- **about half** have the league logging the duel or the ball loss 0.3 to 0.8 s *after* the moment, when a player wins the ball; the founder saw it loose before;
- **the rest** are balls travelling after a misplaced pass, with no duel or clearance logged.

So FIFA's "in contest" (from the event log) and the founder's "nobody has it yet" (the ball in flight or loose) are different readings. The event log cannot place the second.

**Correction to the spec.** It said in contest would fix two phase review cases where a long ball was intercepted (Metrica 1 grid point 9197, Metrica 2 grid point 4206). The founder's notes accept `direct` at a moment just before the interception: that is a P4 phase question, not possession, and a pass in flight is in possession under FIFA's definitions too. Not fixed by this deliverable.

**For the founder:**
- **Keep `p1_state`** as FIFA's vocabulary, estimated: it is FIFA's own method (states from the event sequence), it changes no existing output, and it gives chains their "interrupted" and out-of-play outcomes.
- **Agree `p1_contest_max_s` at 4 s?** 2 s and 4 s give the same calls; 6 s catches one more loose ball.
- **Recommended next, separately:** an observed **ball-state** field (controlled by home, controlled by away, in flight, loose) from ball speed and the nearest player's distance, measured against the founder's calls with a larger judged sample. That is the reading behind "nobody has it yet", and the possession round already proposed it.

**Decided by the founder on 20 November 2026:** `p1_state` kept as FIFA's vocabulary (estimated), and `p1_contest_max_s` agreed at 4 s (thresholds 0.14.0, no value changed). The ball-state field is built next (section 2), before P5.

**Regrounded on 23 November 2026** (the founder asked that decisions rest on standards and professional reference data, not his calls; `tools/context_contest_reference.py`): in the leagues' own event coding, after a logged duel, clearance or block a player has the ball (a pass, a failed pass, a shot or a recovery) within 4 s in **84%** of 1,198 contests on the German league's 7 matches (median 1.0 s, 90th percentile 5.3 s) and **95%** of 743 on Metrica's 3 (405 and 196 more end in a dead ball first). The 4 s cap stands on that; its basis is now `reference`. No value changed.

## 2. Ball state (P1 0.4.0, 20 November 2026)

**Why.** Section 1 showed that FIFA's "in contest" (from the event log) does not place the founder's "nobody has it yet". The founder asked for the separate ball-state field the possession round proposed, before P5.

**What the founder's calls mean.** At the 50 judged moments, ball speed and the nearest player's distance do not separate "nobody" from team calls: moments given to a team include passes in flight at 14 to 18 m/s, 3 to 8 m from anyone, and so do three of the ten controls. A pass travelling to a team-mate stays the passing team's; "nobody" is a ball travelling with no team about to keep it (a clearance, a misplaced pass, a deflection).

**Built** (`plaiq_engine/context/possession.py`, `ball_state`): `p1_ball_state` on every grid point:
- `controlled_home` / `controlled_away`: P1's control call (nearest player within `p1_control_radius_m`, ball slower than `p1_max_control_ball_speed_mps`);
- `contested`: both teams at the ball;
- `in_flight_home` / `in_flight_away`: nobody controls it and the same team controls it before and after;
- `loose`: nobody controls it and the next team to control it is not the last, for at most `p1_loose_max_s` (proposed 5 s);
- `out_of_play`, or `unknown` (the ball undetected, or a longer spell with nobody near it).

It reads who controls the ball next, so it describes a finished match (batch), never a live one. `p1_possession` and `p1_state` are unchanged; thresholds 0.15.0; P1 0.4.0; golden outputs recaptured (only the new field, the version and the Data check entry change).

**Measured** against the founder's 50 calls (`tools/context_contest.py`; a "nobody" call agrees with `loose` or `contested`, a team call with that team's controlled or in-flight state):

| | `p1_possession` | `p1_state` (section 1) | **`p1_ball_state`**, cap 5 s (3 s / 8 s) |
|---|---|---|---|
| "Nobody" moments | 0 of 14 | 2 | **7** (6 / 7) |
| Team moments | 11 of 26 | 10 | **15** |
| Controls | 10 of 10 | 9 | **10** |
| Agrees with the founder's call, in total | 21 of 50 | 21 | **32** (31 / 32) |

**Share of in-play time** (cap 5 s): controlled 37% to 45%, in flight 28% to 31%, loose 14% to 21%, contested 4% to 6%, unknown 3% to 12% on the league's matches; loose 9% to 12% on Metrica; on SkillCorner 1886347 loose 8% and unknown 31% (broadcast tracking loses the ball).

**How the rule was chosen, and why 32 is optimistic.** Four versions were scored on these same 50 calls: "control" meaning any grid point of control (32 agreeing), or control held for 0.2, 0.4 or 0.6 s (21, 23, 24 agreeing). Held control fails because P1's 1.5 m control call flickers during a dribble, which leaves long false loose spells (about 30% of in-play time). The best of four on the same calls is an optimistic figure.

**A fresh blind check for the founder** (`tools/context_possession_replay.py --ball-state`): 50 new moments from the league's matches, drawn by state (15 loose, 15 in flight, 10 controlled, 10 contested), none within 10 s of a moment already judged. Same page and question (home, away or nobody yet); the key stays apart from the page. `--score` reads the calls. Kept outside the repository (it holds the league's positions).

**For the founder:** judge the fresh check; then agree `p1_loose_max_s` (5 s) and whether `p1_ball_state` is the "who has it" field the AI layer reads beside possession.

### The fresh check and P1 0.5.0 (20 November 2026)

**The founder's calls on the 50 fresh moments** (kept without positions in `tools/data/ball_state_review_2026-11-20.json`; an indication, not ground truth) agreed with P1 0.4.0's ball state on **27 of 50**, the honest figure beside the optimistic 32 on the reused set:

| P1 0.4.0's state | Agrees with the founder's call | The founder's calls |
|---|---|---|
| Controlled | 10 of 10 | the controlling team every time |
| In flight within a team's possession | 11 of 15 | that team 11, nobody 3, the other team 1 |
| Loose | 5 of 15 | home 5, away 5, nobody 5 |
| Contested (both teams at the ball) | 1 of 10 | a team 9 times |

- **Contested:** at 7 of the 10, the founder named the team that controlled the ball both before and after (a duel it kept); at 2 more, the team before.
- **Loose:** no pattern: nobody 5, the team about to get it 5, the team that had it 3, one each other way.

**Checked against the league's possession** on all seven matches, independently of the founder's calls:
- a duel the same team controls the ball before and after: the league names that team at **78%** of 3,188 moments;
- a duel across which the ball changes hands: the league names the team before at 66%, the team after at 34%;
- in flight within a team's possession: the league names that team at **90%** of 33,826 moments.

**P1 0.5.0:** a duel the same team controls the ball before and after is that team's (`controlled_home` / `controlled_away`); `contested` remains only where the ball changes hands across it (about 2% of in-play time). Loose is unchanged. With it the fresh calls agree on **33 of 50** (partly fitted to these calls; the league's 78% is the independent support) and the earlier set stays at 32. P5's relationships and P6's runs read the ball state, so their outputs move with it; P2 to P4 do not. Golden outputs recaptured.

**For the founder:** agree the duel rule and `p1_loose_max_s` (5 s); whether `p1_ball_state` is the AI's "who has the ball" field beside possession.

## 3. P5 relationships, completed (P5 0.2.0, 20 November 2026)

**Built** (`plaiq_engine/context/line_breaks.py`, `relationships.py`; P5 now depends on P1, P2 and P3):
- **Line breaks 0.2.0:** a failed pass is judged at the logged target's point at the ball's arrival (the target's run projected from the moment of the pass, as Choice does); without a logged target it stays unknown. Each pass also records whether it was received inside the defending team's shape (convex hull of its outfield players at the moment of the pass).
- **Line-breaking options and open lanes at each pass:** team-mates whose projected run lands beyond a unit with a completion probability of at least `p5_lb_option_min_p` (proposed 0.85), and team-mates with at least `p5_open_lane_min_p` (proposed 0.7). The moment of the pass is `MatchAnalysis`'s, the probabilities the adopted pass completion model's (batched: `option_probabilities` now uses `pass_arrays_one_moment`, identical arrays, which halved P5's runtime).
- **On every grid point where a team has the ball** (P1's ball state: controlled or in flight within its possession): the carrier's pressure (Plaiq's 1v1 pressure, the same formula on the same hex grid, so it equals Plaiq's value, which a test pins), goal-side defenders, the offside line (second-last defender), free attackers (no defender within `p5_free_radius_m`, proposed 5 m), the overload at the ball (within `p5_overload_radius_m`, proposed 10 m) and the mean marking distance.
- **FIFA EFI aggregates per team and half** (segment `efi`): receptions behind the midfield and defensive lines, final third entries by channel, defensive line height and team length by possession and ball third, ball recovery time, forced turnovers (pressure at least `p5_forced_turnover_min_pressure`, proposed 8, the measured high-pressure candidate).
- **Data check:** a `relationships` entry beside `line_breaks`. Thresholds 0.16.0 (five proposed). Golden outputs recaptured (only P5 changes).

**Measured on SkillCorner's 20 matches** (`tools/context_line_breaks.py --package`, `tools/context_relationships.py`; fit = even matches, held out = odd):

| | Fit | Held out |
|---|---|---|
| Line breaks, furthest line, κ (completed and failed passes) | 0.55 | 0.56 |
| Line breaks, any line, κ | 0.67 | 0.66 |
| Carrier pressure against SkillCorner's five levels, weighted κ (cut at 3, 6, 8, 10) | **0.76** (7,517 possessions) | **0.76** (6,987) |
| Goal-side of the engaging defender, agreement (κ) | 81.8% (0.55) | 81.4% (0.54) |

- **Failed passes** (759 with a SkillCorner label): any line broken 80% (κ 0.47), furthest line κ 0.44. Completed passes are unchanged from 0.1.0 (any line κ 0.64, furthest κ 0.54).
- **Line-breaking options do not agree with SkillCorner's count.** At 0.7, 0.85 and 0.95 the agreement on whether a pass has any line-breaking option is κ 0.13, 0.15 and 0.07 (held out 0.13, 0.16, 0.08). 0.85 brings the mean count nearest SkillCorner's (0.73 against 0.5). SkillCorner's passing options are its own notion of who made themselves available, not only who could be reached; the field is kept, proposed and unvalidated, and should not reach anyone until an analyst check.
- **Distributions with no reference** (median of match medians, 10th and 90th percentiles): free attackers 6 (4 to 8) with the 5 m radius, which is too generous for "free" to mean much; overload at the ball 0 (−1 to 2); marking distance 5.9 m (4.2 to 7.5); offside line 33.6 m from goal (10.9 to 52.4); goal-side defenders 9 (3 to 10). Relationships cover 57% of grid points (where a team has the ball).
- **Runtime:** P1, P2 and P5 together 25.5 s per match (median, at most 41 s) on the 20 matches, before P3's few seconds. P5 alone is about 16 s; the Gate 3 budget is 60 s for P5 to P7.

**For the founder:**
- **Agree** the line-break extension to failed passes, inside or outside the shape, carrier pressure, goal-side and the offside line, which agree with references or follow FIFA's definitions.
- **The free-player radius:** 5 m makes about six of ten attackers free at a typical moment. A larger radius, or a definition that also weighs the angle to the ball, needs a judged sample, as for possession.
- **Line-breaking options and open lanes** stay unvalidated (no reference agrees); an analyst check, or the founder's replay check, should decide them.
- **The forced-turnover pressure (8)** follows the open high-pressure decision.

**Decided by the founder on 20 November 2026:** the line-break extension to failed passes, inside or outside the shape, carrier pressure, goal-side and the offside line agreed. Free attackers, line-breaking options and open lanes stay unvalidated (their definitions say so) and are kept from player-facing views; their thresholds stay proposed.

## 4. P6 movement types (P6 0.1.0, 20 November 2026)

**Built** (`plaiq_engine/context/movement.py`; depends on P1 and P2):
- **Runs:** every outfield player's spells at `p6_run_min_speed_mps` (15 km/h, a common running band and the slowest average among SkillCorner's labelled runs) for at least `p6_run_min_s` (0.7 s, its shortest). Runs by the team in possession (P1's ball state at the start) are off-ball runs; the other team's are defensive runs.
- **FIFA's categories** for each off-ball run, against P2's units and the defending shape: `area_end` (in front of the first unit, in between, in behind; an in-behind run from the final unit or from in front of it) and `shape` (in to out, out to in, inside, outside the defending hull). Also `channel_move` (outward, inward, same across P3's centre, half-space and wide bands).
- **The type**, by rules fixed in priority order, named after SkillCorner's run vocabulary so it can be measured: cross receiver, behind, overlap, underlap, coming short, dropping off, pulling wide, pulling into the half-space, support, run ahead of the ball. Defensive runs: pressing, recovery, other.
- **Per grid point:** players in a run per team, and runs in behind by the team in possession. The `runs` segment lists every run (player, times, type, area, shape, channel move, depth gained, sideways movement). Data check entry `movement`.
- **Thresholds 0.17.0** (five, proposed). Golden outputs recaptured (only P6's outputs added). Runtime about 1.5 s per match.

**Rules refined once, before the held-out reading.** On a fit match three type rules over-fired; each was tightened by its definition: a pulling run must be mainly sideways (sideways movement at least the depth change), coming short must not gain depth, and a cross receiver may see the ball wide at the run's start or end. Type agreement rose from κ 0.31 to 0.42 on the fit matches and held at 0.40 on the held-out ones.

**Measured against SkillCorner's off-ball runs on its 20 matches** (`tools/context_movement.py`; a SkillCorner run is found when a P6 run of the same player overlaps at least half of the shorter of the two):

| | Fit | Held out |
|---|---|---|
| SkillCorner's runs found (recall) | 75.4% | 74.6% |
| P6's off-ball runs that SkillCorner labels (precision) | 39.4% | 40.3% |
| Type on found runs, κ (agreement) | **0.42** (50.8%) | **0.40** (49.5%) |
| In behind against SkillCorner's `behind`, κ | 0.31 | 0.37 |
| Channel move against SkillCorner's `in_to_out` / `out_to_in`, κ | 0.66 / 0.60 | 0.69 / 0.59 |
| Shape move (FIFA, the defending hull) against those flags, κ | 0.07 | 0.09 |
| SkillCorner's pressing engagements found by a P6 pressing run | 5% to 14% by subtype | 5% to 13% |

- **SkillCorner labels fewer runs than P6 finds** (P6 finds about 2.5 times as many): it evidently labels the runs it judges relevant to the play.
- **Type recall by SkillCorner type** (held out): run ahead of the ball 62%, support 56%, dropping off 56%, half-space 45%, coming short 45%, pulling wide 41%, cross receiver 36%, behind 34%, overlap 23%, underlap 22%. The main confusions: SkillCorner's `behind` runs that do not end beyond the last line (P6: run ahead of the ball), cross receivers that end level with the carrier (P6: support), and coming short against dropping off in both directions.
- **SkillCorner's in-to-out and out-to-in flags are channel moves, not the shape:** 750 of its 758 in-to-out runs move to a wider channel, 254 of its 262 out-to-in runs to a more central one. P6's `shape` follows FIFA (the defending team's shape) and is not comparable; `channel_move` reproduces SkillCorner's flags (κ about 0.6 to 0.7).
- **Pressing as a run does not match SkillCorner's engagements.** Its engagements are defenders closing on the carrier, mostly below running pace: letting defensive runs go down to 2.5 m/s (`p6_press_min_speed_mps`) lifts the share found only to about 14%. The default stays at 15 km/h; pressing and recovery are unvalidated.

**For the founder:**
- **The type agreement (κ 0.40 held out) is moderate.** The next check is a replay review of 50 runs across types, as for phases, before any threshold is agreed.
- **Pressing:** a measure from distance and closing speed (not a run) would match SkillCorner's engagements better; it belongs with the counter-press in the FIFA view of phases (deliverable 5), where pressing is needed.

### The founder's run review and P6 0.2.0 (20 November 2026)

**The review** (`tools/context_movement_review.py`; calls kept without positions in `tools/data/run_review_2026-11-20.json`): 50 runs from the German league's matches, five per P6 type, judged blind. The founder's calls are one expert's reading, an indication beside SkillCorner's labels.
- The founder's type agreed with P6 0.1.0's on **13 of the 46** runs given a type (28%; SkillCorner agrees on about 50%).
- **The notes pointed to definitions, not types:** six runs were the carrier's own movement with the ball, three were defensive (a track-back, runs after a clearance) though labelled as the team in possession, and one was a player moving to meet a pass already played.

**Changes (P6 0.2.0), from the spec's own definition** ("off-ball runs of the carrier's team-mates"):
- **Possession over the run:** a run belongs to the team that has the ball for more of it (time with nobody on the ball counts for neither), not the team at its first instant.
- **Carries:** the runner who is the team's nearest player to the ball and within `p6_carry_radius_m` (proposed 3 m) of it for at least half the run is carrying, `carry`, not an off-ball run. Wider than P1's 1.5 m control radius because a dribbler at speed pushes the ball ahead.
- **Tried and dropped:** labelling a run that ends with the ball `receiving`. It took five runs the founder had typed (coming short three times, behind, support), and SkillCorner keeps such runs too.
- **Not changed from the review:** the type rules. The confusions the founder's calls show (dropping off against coming short, run ahead against support) need SkillCorner's evidence as well before any rule moves.

**Measured on SkillCorner's 20 matches** (held out; fit in brackets):

| | 0.1.0 | 0.2.0, carry radius 1.5 m | **0.2.0, carry radius 3 m** |
|---|---|---|---|
| SkillCorner's runs found | 74.6% (75.4%) | 83.6% (83.6%) | **80.2%** (80.3%) |
| P6's off-ball runs that SkillCorner labels | 40.3% | 40.2% | **41.1%** |
| Type, κ | 0.40 (0.42) | 0.40 (0.42) | **0.41** (0.42) |
| P6 off-ball runs inside the runner's own SkillCorner possession | (not measured) | 5.1% (5.4%) | **1.2%** (1.3%) |
| P6 carries inside one of the runner's own SkillCorner possessions | | 90% (93%) | 79% (80%) |

- **Possession over the run** finds more of SkillCorner's runs (75% to 80% or more): runs that began as the ball changed hands now count for the right team.
- **The carry radius is a trade-off:** 3 m leaves almost no carries among off-ball runs (1.2%) but about one carry in five is not a SkillCorner possession; 1.5 m is the reverse (5.1%, one in ten). 3 m is proposed.
- **On the founder's 50 calls** with 0.2.0: all five noted carries are carries, one typed attacking run is now labelled defensive (eight were with a stricter first rule), and the type agrees on 11 of 31 typed off-ball runs (35%).
- Thresholds 0.17.0 gain `p6_carry_radius_m` (proposed). P6 0.2.0. Golden outputs recaptured (only P6 changes).

**For the founder:** agree the two definitional changes; choose the carry radius (3 m proposed, or 1.5 m); the type rules stay estimated, with their agreement moderate both with SkillCorner (κ 0.41) and with the founder's calls (35%).

**Decided by the founder on 20 November 2026 (sections 2 and 4):** the duel rule (P1 0.5.0) and `p1_loose_max_s` (5 s) agreed; `p1_ball_state` is the AI layer's "who has the ball" field beside possession; P6 0.2.0's possession over the run and the carry rule agreed, with `p6_carry_radius_m` at 3 m. Thresholds 0.18.0 (no value changed). P6's type rules stay estimated.

## 5. P7 space, value and game state (P7 0.1.0, 20 November 2026)

**Built** (`plaiq_engine/context/space_value.py`; depends on P1 and P2; all fields estimated, not yet validated):
- **Open high-value space:** every `p7_space_every_s` (5 s; Plaiq's pitch control costs about 25 ms a frame over the whole grid), with the ball detected, the share of each team's high-value space (the top `p7_high_value_share`, 20%, of its threat surface) that the other team does not control, weighted by threat, from Plaiq's per-frame control on the honeycomb. A team rarely controls its own high-value space outright (it lies near the opponent's goal, where the defenders control a median 57% and nobody 38%), so the open share is the informative measure, not control.
- **`p7_open_danger_zone`:** for the team with the ball, the lane and third (its own frame) of the most threatening cell the defence does not control. Never a cell index.
- **The game-state weight λ** for the team with the ball: a versioned lookup by goal difference (P1's score) and time. Level 1; leading 1.25 (from minute 75, 1.5); trailing 0.75 (late 0.5). A judgement, labelled estimated. With the score unknown λ is 1 and the Data check says so.
- **Context-adjusted Choice**, a new score beside Choice (Choice stays the default): from Choice's own options at each pass (their ring-1 cells, completion probability `p` and value, now kept by `MatchAnalysis.choice_options`), each cell is worth `U = p × value − λ × (1 − p) × cost`, where `cost` is the other team's threat at the cell (what losing the ball there gives away). An option is its best cell; the score is the chosen option's `U` over the best option's, and null with a reason when no option is worth more than its risk. Segment `adjusted_choice` names the weight, goal difference and minute on every row.
- **One shared analysis:** P5 and P7 reuse the semantic builder's `MatchAnalysis` (`extras["analysis"]`), so Choice is computed once per match.
- **Not in 0.1.0, against the spec:** per-cell control is not stored for the replay (the replay can recompute Plaiq's control for the frame it shows, so storing it waits for the replay's design); the danger zone uses P3's lane and a third rather than a P3 zone; P7 reads P1 and P2, not P4. The shot measure uses SkillCorner's `lead_to_shot` (as Choice was adopted on), not its `dangerous` flag.
- Thresholds 0.19.0 (seven, proposed). Golden outputs recaptured (P7 added). P7 takes a median 21 s a match (max 45 s) with Choice already computed; P5 to P7 together about 48 s with the shared analysis, inside the spec's 60 s budget (Choice's own 22 s is spent by the semantic builder either way).

**Measured on SkillCorner's 20 matches** (`tools/context_space_value.py`; comparison only; Choice's own groundings from `tools/plaiq_dq_v2.py`). SkillCorner's matches have no logged score, so λ is 1 here and the comparison isolates the cost-of-losing-the-ball term.

| 6,173 fit / 5,931 held-out passes | Choice, fit | Choice, held out | Adjusted, fit | Adjusted, held out |
|---|---|---|---|---|
| Possession leads to a shot, AUC (completed passes) | 0.633 | 0.609 | 0.580 | 0.558 |
| SkillCorner's decision analogue, Spearman | 0.295 | 0.292 | 0.087 | 0.066 |
| Completion, AUC (above 0.5 leans to safety) | 0.500 | 0.490 | **0.680** | **0.669** |
| Median | 0.78 | 0.79 | 0.71 | 0.72 |

- **The adjusted score leans to safety.** It ranks completed passes higher (AUC 0.67) and possessions that lead to shots lower than Choice does; its rank agreement with Choice is 0.66 held out. Choice itself is neutral on completion (0.49), as designed.
- **The cost term dominates at full weight.** SkillCorner's analogue (`xpass_completion × xthreat`) has no cost of losing the ball, so a lower agreement with it is expected; the drop in the shot AUC is the material finding.
- **λ's direction finds no support in SkillCorner's own labels.** The share of a team's possessions that SkillCorner calls `direct` barely moves with the game state: drawing 2.8% before minute 75 and 3.5% after, losing 3.0% and 3.0%, winning 2.5% and 2.9% (18,710 player possessions in all). Chasing teams do not play measurably more directly here. This tests one proxy, not the idea; λ stays a judgement.
- **Open high-value space by SkillCorner's phase** (median share at possession starts) orders sensibly: build-up 0.24, direct 0.24, create 0.28, chaotic 0.39, transition 0.39, disruption 0.43, finish 0.43, quick break 0.51, set play 0.71 (set plays start with the defence not yet set around the high-value cells).

**For the founder:**
- **Context-adjusted Choice:** keep it as built (a safety-leaning score, analysts only, "estimated, not yet validated"), or reweight the cost term (for example half weight, measured on the same groundings before any value is agreed). The expert study (Gate 3b) decides whether it is ever shown first.
- **λ values:** stay a proposed judgement; SkillCorner's direct-play share gives them no support either way.
- **The P7 thresholds** (sample rate, high-value share, late minute, four λ values): agree, or keep proposed.

### The cost term at lower weights (20 November 2026)

At the founder's request, the adjusted score again with the cost term scaled by a weight (`tools/context_space_value.py --cost-weights 0 0.25 0.5`; λ is 1 on SkillCorner). Held-out matches, fit in brackets:

| Cost weight | Shot AUC (completed) | SkillCorner analogue, Spearman | Completion AUC | Median |
|---|---|---|---|---|
| 0 (no cost) | 0.609 (0.633) | 0.292 (0.295) | 0.490 (0.500) | 0.79 |
| 0.25 | 0.593 (0.615) | 0.217 (0.231) | 0.581 (0.588) | 0.77 |
| **0.5** | **0.580** (0.600) | **0.161** (0.174) | **0.627** (0.639) | 0.76 |
| 1 (as built) | 0.558 (0.580) | 0.066 (0.087) | 0.669 (0.680) | 0.72 |
| Choice | 0.609 (0.633) | 0.292 (0.295) | 0.490 (0.500) | 0.79 |

- **Weight 0 reproduces Choice** (to the third decimal), so the adjusted score is Choice plus the cost term and nothing else.
- **The trade-off is smooth, with no sweet spot.** Each step of weight buys a lean to safe passes with a loss on the shot measure: half weight keeps about four fifths of the full weight's lean to safety (completion AUC 0.63 against 0.67) for under three fifths of its shot loss (0.029 against 0.051).
- **These references cannot choose the weight.** The shot outcome and SkillCorner's analogue reward value only; neither rewards avoiding a costly loss, so any cost term scores worse on them by construction. The weight is a judgement until a risk-side check (do losses where the cost is high lead to the opponent's shots more often?) or the expert study (Gate 3b) measures it.

**For the founder:** keep the weight at 1, or make it a versioned threshold `p7_cost_weight` (proposed 0.5, labelled estimated) on P7 0.2.0. Recommended: the threshold at 0.5, and the risk-side check on SkillCorner's and the German league's matches before any value is agreed.

### P7 0.2.0 and the risk-side check (21 November 2026)

**The founder's choice (21 November 2026):** `p7_cost_weight` at 0.5, proposed (thresholds 0.20.0, P7 0.2.0; each segment row names the weight). λ stays proposed. Golden outputs recaptured: only the adjusted score and P7's version change.

**The check** (`tools/context_risk_check.py`): do losses where the cost is high lead to the opponent's shots more often? A loss is the other team taking the ball in play (SkillCorner: its next player possession within 3 s; the German league: the official team in possession changing, each side holding the ball at least 1 s; never just after the loser's own shot). The cost is the regaining team's threat (Plaiq's `xT_surface`) at the ball when it regains. The outcome is a shot by the regaining team within 15 s (10 and 20 s agree) before the loser has the ball again.

| 15 s window | SkillCorner, 20 matches | German league, 7 matches |
|---|---|---|
| Losses (shot rate) | 2,330 (6.4%) | 1,786 (3.8%) |
| AUC of the cost | **0.76** | **0.81** |
| AUC of plain depth (how near its own goal the loser lost it) | 0.76 | 0.80 |
| Shot rate, lowest to highest cost quintile | 1.5%, 0.9%, 6.2%, 7.1%, **16.3%** | 0.6%, 0.6%, 2.0%, 3.4%, **12.3%** |
| Matches where the cost's AUC is above 0.5 | 20 of 20 | 7 of 7 |

**The term as the score applies it** (SkillCorner, the chosen option's best cell at weight 0.5):

| 15 s window | AUC |
|---|---|
| Failed passes (1,368): the cost at the option's cell | 0.73 (the cost at the actual loss point: 0.75) |
| Every pass Choice scored (14,860): "this pass is lost and the opponent shoots", by the charge `(1 − p) × cost` | **0.81** |
| The same, by `1 − p` alone | 0.75 |
| The same, by the cost alone | 0.60 |

- **The cost of losing the ball is real and the surface ranks it.** Losses in the top fifth of cost are followed by the opponent's shot more than ten times as often as the bottom two fifths on SkillCorner, and twenty times on the German league, in every match of both sources.
- **The threat surface adds little over depth** (AUC 0.76 against 0.76, 0.81 against 0.80): the cost is mostly where on the pitch the ball is lost. That is expected of a location surface, and harmless.
- **The charge the score applies is the right shape.** Before the outcome is known, `(1 − p) × cost` ranks the passes that end in a loss and an opponent's shot better than the risk of losing the ball alone or the cost alone (0.81 against 0.75 and 0.60). The option's cell stands in well for where the ball is actually lost (0.73 against 0.75).
- **What it does not settle:** the weight. The check shows the cost term measures a real risk; how much of it to weigh against value is an exchange rate, which the shot outcomes cannot set. `p7_cost_weight` stays proposed at 0.5 until the expert study (Gate 3b).

## 6. The FIFA view of phases (P4v 0.1.0, 21 to 23 November 2026)

**Built** (`plaiq_engine/context/fifa_view.py`, a package beside P4 so P4's agreed labels are unchanged; depends on P1, P2, P4 and P5; all fields estimated, not yet validated):
- **Engagements, the pressing measure:** a defending outfield player within `p4v_engage_radius_m` of the ball carrier (P5's carrier) and closing on them at `p4v_engage_closing_mps` or more, over `p4v_closing_lag_s` with the same carrier; spells under `p4v_engage_min_s` dropped. Typed at the start, in order: **recovery** (running towards its own goal at `p4v_recovery_back_mps` or more: FIFA's recovery), **counter-press** (within `p4v_counter_press_s` of its team losing the ball), **press** (the ball beyond `p4v_pressure_max_depth_share` of the pitch from its goal) and **pressure** (nearer). Segment `engagements`; `p4v_engaging_defenders` per grid point.
- **Presses:** engagements (not pressure) by at least `p4v_press_min_players` defenders, each within `p4v_press_gap_s` of the last, with no change of possession; a counter-press when the first is one. Segment `presses`.
- **FIFA's names** for P4's phases (`p4v_phase`, `p4v_out_phase`): build-up split into **opposed** (P5's carrier pressure at `p4v_opposed_min_pressure` or more in the last `p4v_opposed_hold_s`, on the same carrier) and **unopposed**; progression, final third, long ball, counter-attack, attacking transition, set play, in contest. Out of possession: **high, mid or low press** while a press is under way, **counter-press** while one is, otherwise high, mid or low block; defensive transition and the other defending phases by name.
- **FIFA's published mid-block rule** (`p4v_fifa_block`, World Cup 2022 article): no outfield player moving forward or backward faster than 10 km/h, at least four behind the ball, at least eight outfield players in the middle and defensive thirds (all in the defensive third: low block), held 1.6 s of 2.0 s. The "horizontal lines" condition is not checked.
- Thresholds 0.21.0: sixteen, all proposed (five are FIFA's published values). Golden outputs recaptured (P4v added). P4v takes about 1 s a match.

**Grounding the pressing measure** (SkillCorner's engagements on 8 matches, comparison only): its engagements start a median 6 to 8 m from the carrier, closing at a median 2.7 m/s, while other defenders sit a median 22 m away closing at 0.6 m/s. Its subtypes separate on three axes: recovery presses run back (towards their own goal at a median 3.5 m/s; nine in ten at 1 m/s or more); counter-presses come a median 2.6 s after the loss (77% within 5 s); "pressing" happens only in high and medium blocks and "pressure" mostly in low blocks and the defensive third.

**Chosen on the fit matches** (P4v rerun on the same upstream outputs): radius 8 m (6 to 12 m swept: 8 m finds 57% at 65% precision; 10 m 60% at 55%), the press/pressure split at 0.4 of the pitch (type κ 0.50 at a third, 0.53 at 0.4), a 4 s press gap (chain time found 22% at 1.5 s, 46% at 3 s, 55% at 4 s, precision unchanged), a 5 s counter-press window (3 to 6 s barely move κ), carrier pressure 6 for opposed (κ 0.50; 8: 0.27).

**Measured on SkillCorner's 20 matches with the chosen settings** (fit / held out):

| | Fit | Held out |
|---|---|---|
| SkillCorner's engagements found (8,963 / 8,481) | 57.3% | 57.8% |
| P4v's engagements that SkillCorner has | 65.0% | 65.2% |
| Type on found pairs, agreement (κ) | 66.1% (**0.54**) | 65.7% (**0.54**) |
| Type recall: recovery / press / counter-press / pressure | 92% / 77% / 53% / 38% | 90% / 75% / 51% / 42% |
| SkillCorner's pressing-chain time inside a P4v press | 46% | 45% |
| P4v press time inside a SkillCorner chain | 44% | 41% |
| SkillCorner's counter-press chains found by a P4v counter-press | 96 of 230 | 84 of 210 |
| Build-up opposed against SkillCorner's build-up possessions with an engagement (1,148 / 932; 69% / 67% opposed), agreement (κ) | 74.7% (**0.48**) | 77.5% (**0.52**) |

- **Recovery and press agree well; counter-press and pressure less.** Most of SkillCorner's counter-presses that P4v misses it calls pressure (about 330 a half): engagements more than 5 s after P1's change of possession, or deep in the defender's half. SkillCorner's loss moment and P1's differ, and its counter-press evidently is not only a time window.
- **Build-up opposed, after one fix found on the confirmation run.** Held over 2 s regardless of who had the ball, pressure on one carrier made the next one's build-up opposed too: 77% opposed against SkillCorner's 67%, κ 0.36 held out. The hold now applies only while the same player carries the ball (an opposed build-up is about the player on the ball, as FIFA puts it), giving 0.52 held out (pressure alone at any point of the possession: 0.53). The hold length itself was not swept.
- **Presses match SkillCorner's chains about halfway** both ways: a chain is its own grouping of engagements, and P4v finds about 58% of the engagements in it.

**FIFA's mid-block rule beside P4's block and SkillCorner's phases** (out-of-possession grid points; held out, fit in brackets):
- **FIFA's rule is much stricter:** it holds on 11% (10%) of the time out of possession as a mid-block and 4% (5%) as a low block. It needs every outfield player nearly still.
- **Where it holds, it points the same way:** inside P4's medium block it is a FIFA mid-block 31% (29%) of the time; inside P4's low block a FIFA low block 17% (17%) and a mid-block 6%; inside P4's high block a mid-block 5% (4%) and never a low block. Against SkillCorner's medium block 27% FIFA mid-block, its low block 15% FIFA low block.
- **Compactness:** during FIFA's mid-block the defending team's median width and length are 35.5 × 25.8 m (37.0 × 24.5), against FIFA's World Cup 2022 mid-block average of 40.1 × 27.2 m (P4's medium block: 36.0 × 28.1 m). A different competition and a median against a mean: an internal sanity check only, never a public comparison.

**For the founder:**
- Agree P4v's thresholds or keep them proposed (the engagement radius, closing speed, counter-press window, press gap, the press/pressure depth and carrier pressure 6 for opposed build-up). FIFA's five block values are published; adopting them as agreed is a vocabulary choice.
- The counter-press type is moderate (recall about 52%): a better loss moment (P1's change of possession against SkillCorner's) is the next lever, measured before any rule moves.
- The FIFA view is analysts-only and estimated, as every context field is until Gate 2b.
