# Plaiq against independent references

18 November 2026. Step 1 of `ponta-blueprint/plaiq-review.md`: measurement only, no engine change. Tool: `tools/plaiq_references.py`. References are comparison data only (founder decision 2): SkillCorner's per-pass labels (20 open-data matches), the German league's per-shot pressure and expected goals (IDSSE, 7 matches; data: Deutsche Fußball Liga), and Wyscout's open events (1,941 matches; Pappalardo et al. 2019). Plaiq ran with default player profiles. Agreement with a provider's model is consistency with an industry reference, not accuracy.

## Results

| Plaiq model | Reference | Result | Reading |
|---|---|---|---|
| **Completion**: targeted pitch control at the receiver × (1 − interception risk) | Whether 12,103 SkillCorner passes were completed (82% were) | AUC **0.78** (control alone 0.78, no-interception alone 0.73); SkillCorner's expected completion 0.90 | Plaiq ranks passes by how likely they are to arrive, but below a learned model |
| | SkillCorner's expected completion | Rank correlation 0.62 | |
| **Pass option value** of the targeted player at the pass | SkillCorner's threat for that player | Rank correlation **0.28** | Weak: POV is not tracking what SkillCorner calls threat |
| | SkillCorner's "dangerous" flag | AUC **0.53** | Close to chance |
| **1v1 pressure** on the shooter (nearest opponent) | The league's pressure on 164 shots | Rank correlation **0.30** | Weak |
| **Zonal pressure** on the shooter (three nearest opponents) | The same | Rank correlation **0.49** | Better: the number of pressers matters, as in FIFA's EFI pressure |
| **Threat surface** at the shot | The league's expected goals | Rank correlation 0.54; goals picked out with AUC 0.67, against 0.81 for the league's xG (16 goals) | A geometric surface is a coarse shot model |
| | Wyscout: goal rate of shots from each of 96 zones (zones with 30+ shots) | Rank correlation **0.81** | It ranks shooting zones well |
| | Wyscout: share of events after which the same team scores within 10 s, by zone (3.25 million events) | Rank correlation 0.51 | It ranks "danger of having the ball here" less well, the job an expected-threat surface does |

## Failed-pass target: not measurable on this data

- **On SkillCorner's data, a failed pass's end point is SkillCorner's own targeted player.** The reader (`matrix_engine.ingestion.skillcorner_io`) takes it from the targeted player's position at the pass, because the reception point is missing.
- **So the comparison is circular.** Plaiq's inferred receiver matched the targeted player on 90% of 2,099 failed passes, and "the team-mate nearest the end point" on 98%, but both start from SkillCorner's answer.
- **No independent test is possible here.** Metrica and the league's events give no intended receiver for a failed pass, so failed-pass target inference cannot be measured on the data held.
- **This corrects the line-break section of `CONTEXT_STRUCTURE_REPORT.md`.** On SkillCorner, Ponta's failed-pass end point was already the targeted player's position at the pass, not the interception point. The low agreement on failed passes (κ 0.27) is more likely runs in behind: at the pass the runner is still in front of the back line, and SkillCorner credits the break (211 last-line breaks against Ponta's 51).

## What it means for the review

1. **Pressure:** count the pressers. Zonal already beats 1v1 against the league's pressure (0.49 against 0.30). Pressure version 2 (angle and count, FIFA's inputs) has a reference to beat now.
2. **Threat:** a data-driven surface from Wyscout should be measured against these two baselines (0.51 and 0.81) and the league's xG.
3. **Pass option value** is the weakest model against the references. Its value part should be rebuilt, separating availability (0.78 AUC is a reasonable start) from value. The unit-aware and data-driven threat proposals feed this.
4. **Kinematics:** completion at 0.78 against SkillCorner's 0.90 leaves room. Calibrating ball speed and reaction time on the league's 25 Hz tracking is worth measuring.
5. **Failed-pass targets** need a reference first. A blind judged sample (the replay method) is the way to get one.

## Step 3: availability and value split (18 November 2026)

**The split.** `plaiq_engine/analytics/option_split.py` (version 0.1.0, a new metric beside POV; POV and decision quality untouched) writes pass option value as the product it is:
- **availability** = control at the receiver × (1 − interception risk);
- **value** = xT at the receiver × progressive factor × (1 − distance decay).

Their product equals POV exactly (tests in `test_option_split.py`).

**Measured** by `tools/plaiq_option_split.py`:
- **Data:** every SkillCorner pass on the 20 open-data matches; every team-mate of the passer scored as an option.
- **Fit and test:** a threshold chosen on 10 fit matches and read on the other 10.

| | Fit | Held out |
|---|---|---|
| **Availability** predicting completion of the targeted pass, AUC | **0.79** | **0.77** |
| POV predicting the same, AUC | 0.66 | 0.66 |
| Availability picking SkillCorner's listed options, AUC | 0.67 | 0.66 |
| **Value** of the targeted player against SkillCorner's threat, rank correlation | **0.79** | **0.78** |
| Same, against its "dangerous" flag, AUC | **0.87** | **0.88** |
| POV against SkillCorner's threat; against "dangerous" | 0.30; 0.54 | 0.26; 0.51 |
| xT at the receiver alone against threat; against "dangerous" | 0.89; 0.95 | 0.88; 0.95 |
| Line-breaking options: available (≥ 0.45, chosen on fit) and beyond a line, per pass, against SkillCorner's count | 4.9 against 0.6; κ 0.01 | 5.0 against 0.5; κ 0.01 |

**Reading.**
- **Each half is sound for its own question. Multiplying them hid both.** SkillCorner's threat is a value-if-received measure, and POV mixes in the chance of arrival.
- **Value tracks a provider's threat well** (0.78 held out), and availability predicts completion better than POV (0.77 against 0.66).
- **POV's progressive factor and distance decay lower agreement with threat** (0.78, against 0.88 for xT at the receiver alone). This is evidence for the data-driven threat step, not a reason to drop them yet: SkillCorner's threat is its own model.
- **Availability is not what makes a passing option.** SkillCorner lists far fewer options than there are open team-mates. Its options look like players offering to receive (FIFA's term), which is movement: P6's job, not Plaiq's. Line-breaking options wait for it.

## Step 4: pressure version 2 (18 November 2026)

**What was built.** `plaiq_engine/analytics/pressure_v2.py` (0.1.0) is a directional, count-aware pressure, added beside 1v1 and zonal pressure, which are untouched. It follows FIFA's EFI inputs and the pressure zone of Andrienko et al. (2017):
- a defender's pressure falls with distance inside a zone that reaches further in front of the player (9 m by default) than behind (3 m);
- the pressures are summed across defenders.

**How it was measured** (`tools/plaiq_pressure.py`), against two references:
- **SkillCorner's pressure level** at the start and end of every player possession: 33,270 moments on 20 matches. Settings were chosen on the fit matches.
- **The league's pressure on 164 shots,** an independent check nothing was fitted to.

| Pressure on the player | SkillCorner, held out: rank correlation / weighted κ (5 levels) | League shot pressure: rank correlation |
|---|---|---|
| **Plaiq 1v1** (nearest opponent) | **0.82 / 0.80** | 0.30 |
| **Plaiq zonal** (three nearest) | 0.80 / 0.78 | **0.49** |
| v2, published defaults (9 m front, 3 m behind, q 1.75, sum) | 0.76 / 0.68 | 0.43 |
| v2, best on fit (6 m front, 5 m behind, q 1.75, sum) | 0.80 / 0.73 | 0.41 |

**Reading.**
- **Pressure version 2 is not adopted.** It does not beat Plaiq's existing measures on either reference. The fit even moved it towards the same reach in front and behind, so direction did not help.
- **The two references measure different things,** as FIFA's two notions do:
  - **SkillCorner's level follows the nearest opponent.** That is the engaging defender of FIFA's coded direct, indirect or no pressure. Plaiq's 1v1 matches it best.
  - **The league's shot pressure follows how many defenders are around,** like EFI's collective "pressure on the ball". Plaiq's zonal pressure matches it best.
- **Correction to step 1:** 1v1 pressure looked weak against the league's shots (0.30) because that reference measures collective pressure. Against SkillCorner's level it is the best measure here.

**Named bands for 1v1 pressure** (the open "named high-pressure threshold"). These cut points on Plaiq's 1 to 10 scale reproduce SkillCorner's level shares on the fit matches; weighted κ 0.80 held out:

| 1v1 score | SkillCorner's level | Plaiq's specification's own reading |
|---|---|---|
| 1 to 2 | no pressure | 10 m or more, unchallenged |
| 3 to 5 | low | loose tracking |
| 6 to 7 | medium | nearby challenge |
| **8 to 9** | **high** | 1 to 2 m, immediate pressure |
| 10 | very high | contact |

- **For FIFA's three coded levels,** direct pressure ≈ 8 or more, indirect ≈ 3 to 7, none ≈ 1 to 2. That is a proposal for the founder to confirm on a judged sample, since FIFA publishes no distances.
- **For collective pressure,** zonal pressure is the measure to report.

## Step 5: a learned threat surface (18 November 2026)

**What was built.** `plaiq_engine/analytics/xt_learned.py` (0.1.0) is a candidate beside Plaiq's geometric `V_xT`, which is untouched. It is learned by the expected-threat method (Singh 2019) on a 16 × 12 grid:
- **Data:** Wyscout's events from 970 matches (even positions in match id order), 869,289 actions.
- **Moves:** passes and accelerations, successful when tagged accurate.
- **Shots:** penalties left out.
- **Kept outside the repository:** the learned surface.

**Measured** by `tools/plaiq_threat.py` on data it never saw:

| | Learned | Geometric |
|---|---|---|
| Wyscout, 971 held-out matches: team scores within 10 s of the event, AUC (1.6 million events) | **0.671** | 0.641 |
| Same, by zone, rank correlation | **0.59** | 0.54 |
| Wyscout held out: goal rate of shots by zone (42 zones with 30+ shots) | 0.83 | **0.89** |
| SkillCorner's threat for the targeted player (15,154 passes) | 0.87 | **0.89** |
| SkillCorner's "dangerous" flag, AUC | 0.94 | **0.95** |
| The league's xG (164 shots) | 0.53 | **0.54** |
| The league's goals, AUC | 0.62 | **0.67** |

**Reading.**
- **Not adopted.** The learned surface wins only on its own source, Wyscout, on matches it did not see, and loses slightly on every other reference. The likely reason is resolution: zones of about 6.6 × 5.7 m against a continuous surface, which matters near goal, where shots and expected goals are decided.
- **Plaiq's geometric threat surface holds up against three independent references,** so it stays.
- **For pass option value,** the gap between the threat at the receiver (0.88 against SkillCorner's threat) and POV's value part (0.78) comes from POV's progressive factor and distance decay, not from the surface. That is where a value change should look.
- **What the learned surface offers:** a scale in probability units (its maximum is 0.40, the chance of scoring within the possession), which the geometric surface, normalised to [0, 1], lacks. A finer grid or smoothing could close the resolution gap. Both are noted, not pursued.

## Decision quality grounded in maths, outcomes and context (18 November 2026)

**Why not a judge as the truth.** The founder decided a judge's rating should not be the truth for decision quality. So it is grounded three other ways by `tools/plaiq_dq_grounding.py`, on 12,012 SkillCorner passes (9,910 completed):
- **independent maths:** SkillCorner's own expected value of every option (expected completion × threat), giving its analogue of decision quality, the targeted option's value over the best one's;
- **what happened next;**
- **the context of the pass.**

The blind replay page (`tools/plaiq_dq_replay.py`) becomes a review tool for cases where these disagree, not a source of truth.

**1. Failed passes (a flaw found by the arithmetic).**
- **Plaiq rates 89% of failed passes "optimal"** (2,102 passes).
- The failed-pass score compares pass option value at the inferred meeting point with the best option at the pass. It never charges for the failure, and the inferred point is, by construction, where a team-mate would have won the ball.
- **Failed-pass decision quality should not be reported as it stands.**

**2. Outcomes for completed passes.**

| Plaiq band | Passes | Possession leads to a shot | Possession leads to a goal |
|---|---|---|---|
| Poor | 645 | 1.2% | 0.0% |
| Suboptimal | 1,155 | 1.6% | 0.35% |
| Good | 2,320 | 5.5% | 0.78% |
| Optimal | 5,790 | **9.1%** | **1.16%** |

- **As a predictor:** decision quality predicts a shot with AUC 0.60 and a goal with 0.62. SkillCorner's analogue gives 0.62 and 0.65.
- **Higher decision quality goes with lower completion:** it rewards ambitious, valuable passes. This is by design, but should be stated wherever the score is shown.

**3. Independent maths.**
- **Rank correlation with SkillCorner's analogue: 0.31.** Plaiq's score picks out the passes where SkillCorner says the best option was chosen with AUC 0.60.
- **Exact band agreement: 32%.**
- **Optimal versus best:** Plaiq calls 59% of these completed passes optimal; SkillCorner says the passer chose its best option on 37%. This measures engine governance's concern that the passer is "the best option" too often.
- **The two models define "best" differently:** SkillCorner's expected value is dominated by threat, while Plaiq weighs arrival more heavily.

**4. Context (pressure on the passer, SkillCorner's level).**

| Pressure | Passes | Median DQ | Poor or suboptimal | SkillCorner: chose best |
|---|---|---|---|---|
| None | 1,983 | 0.98 | 15% | 41% |
| Low | 2,033 | 0.95 | 24% | 32% |
| Medium | 3,153 | 0.97 | 19% | 34% |
| High | 1,607 | 1.02 | 13% | 41% |
| Very high | 168 | 1.01 | 12% | 41% |

Under high pressure fewer options are on, so both models judge the choice more kindly. That is sensible behaviour, not a penalty for being pressed.

**What this means.**
- **Completed-pass decision quality has outcome evidence** comparable to a provider's own model: "a defined, explainable score" can now cite it.
- **Failed-pass decision quality needs fixing before use.**
- **A candidate decision quality version 2,** to be measured against these same three groundings, fit and test by match: score the choice at the moment of the pass as expected value, using the option split (availability × value). That charges a failed pass for its availability and keeps one definition for completed and failed passes.

## Decision quality version 2, measured (19 November 2026)

**The design.** `plaiq_engine/analytics/decision_quality_v2.py` (0.1.0) is a candidate beside version 1, which is untouched. It judges the intention plus the risk known when the ball is played:
- the chosen destination's expected value (availability × value) over the best expected value among all team-mates at that moment;
- one formula for completed and failed passes, with the outcome never an input.

Measured by `tools/plaiq_dq_v2.py` on 12,112 SkillCorner passes (9,920 completed), with the threat-only value as a sensitivity check:

| | Version 1 | Version 2 | Version 2, threat only |
|---|---|---|---|
| Optimal share: completed / failed passes | 58% / **89%** | 30% / 19% | 37% / 17% |
| Median: completed / failed | 0.98 / 1.24 | 0.76 / 0.41 | 0.87 / 0.41 |
| Possession leads to a shot, completed passes, AUC | **0.60** | 0.51 | 0.48 |
| Possession leads to a shot, all passes, AUC | **0.55** | 0.53 | 0.50 |
| Possession leads to a goal, completed passes, AUC | **0.62** | 0.51 | 0.45 |
| Rank correlation with SkillCorner's option-value analogue | **0.31** | 0.15 | 0.06 |
| Picks out passes where SkillCorner's best option was chosen, AUC | **0.60** | 0.54 | 0.50 |
| Predicts completion, AUC | 0.23 | 0.69 | 0.74 |

The even and odd matches agree on every row.

**Reading.**
- **Not adopted; version 1 stays.** Version 2 removes the failed-pass leniency but becomes mostly a measure of risk: it predicts completion (0.69) and loses the link to chances (0.51) and to SkillCorner's analogue (0.15).
- **Why version 1 keeps the signal.** It values the receiver at the end of the pass, which credits playing the ball into a team-mate's run. Version 2 compares the destination with where team-mates stood at the pass, so a through ball into a run looks like a ball into space a defender could reach.
- **Plaiq's ball-speed model weighs on version 2.** At 5 m/s plus 0.2 m/s per metre, long forward passes look slower, and so riskier, than they are. *(Corrected 19 November 2026 by step 6: the model is 30% to 40% too slow on passes up to 30 m and close to right beyond 45 m, so the bias falls on short and medium passes.)* Calibrating it on the league's 25 Hz tracking (programme step 6) comes before any risk-weighted decision quality.
- **Version 1's failed-pass leniency stays a known flaw** (89% optimal). Until it is fixed, failed-pass decision quality should not be reported.
- **The candidate for the next attempt:** keep version 1's end-of-pass credit and add only the risk at the pass, after the ball-speed calibration.

## Step 6: ball flight, players and reaction time calibrated (19 November 2026)

**The question.** Targeted pitch control (`targeted_pcz`, which feeds availability, POV and decision quality) rests on hand-set physics:
- a ball speed of 5 m/s plus 0.2 m/s per metre of pass;
- each point along a pass timed as if it were the end of its own, shorter pass;
- one defender profile: reaction 0.7 s, top speed 9.0 m/s, acceleration 5.5 m/s².

`plaiq_engine/analytics/ball_flight.py` (0.1.0) makes each of these a parameter. With its defaults it returns exactly what `targeted_pcz` returns (tested). `tools/plaiq_calibration.py` measures them on the seven IDSSE matches (TRACAB at 25 Hz with the ball's height; the league's passes with the intended recipient), fitting on the even matches and testing on the odd ones. It then checks the result on SkillCorner's 20 matches.

**The passes.**
- 5,239 open-play passes with a logged recipient.
- **The kick is found on 3,260 (62%):** the ball leaves the passer (within 1.5 m, then beyond 3 m within 0.6 s), nearest the logged time.
  - The rate is 40% to 86% by match. It is the same for completed and failed passes (62% each), so it does not bias the outcome fit.
  - The kick lies a median 0.14 s after the logged time (10th to 90th percentile: −1.4 s to +1.2 s).
- **First touch:** the recipient 2,777, an opponent 260, another team-mate 198, out of play 22.

**1. Ball flight (physics only, 2,685 passes the recipient touched first).**

| Pass length | Passes | Median real speed | v1 model | Measured model |
|---|---|---|---|---|
| 3–10 m | 823 | 8.6 m/s | 6.4 | 9.7 |
| 10–20 m | 1,161 | 11.1 m/s | 7.8 | 10.9 |
| 20–30 m | 452 | 13.1 m/s | 9.8 | 12.5 |
| 30–45 m | 185 | 14.2 m/s | 12.1 | 14.4 |
| 45–80 m | 64 | 15.6 m/s | 15.2 | 17.0 |

- **Fitted on the even matches: 8.5 m/s plus 0.165 m/s per metre** (flat passes 8.6 + 0.178; high passes 6.2 + 0.213).
- **On the held-out matches it halves the timing error.** v1's ball arrives on average 0.37 s late (median error 0.42 s); the measured model's median error is 0.20 s, with a mean bias of −0.08 s.
- **Separate models for flat and high balls add little** (0.18 s against 0.18 s on flat balls, 0.33 s against 0.32 s on high balls).
- **The ball slows along a pass.** It covers 33%, 60% and 81% of its length in a quarter, half and three-quarters of its time (held out). `s / D = (1 + c)τ − cτ²` with **c = 0.39**, fitted on the even matches, predicts 32%, 60% and 82%.
- **v1's point-by-point timing has the opposite shape:** it makes the early part of a long pass slower than the pass's own average.

**2. Players (physics only, 204 player-matches with at least 10 minutes).**
- **Top speed** (the fastest half-second): median **8.4 m/s** (10th to 90th percentile 7.3 to 9.1).
- **Peak acceleration** (speed gained over 0.4 s, 99.9th percentile): median **4.7 m/s²** (4.3 to 5.3).
- **Plaiq's profile (9.0 m/s, 5.5 m/s²) sits at the fast end** of the league's players.

**3. Reaction time (fitted on outcomes).**

Availability for the pass to the recipient where they stand at the kick, against completion, AUC on the held-out matches:

| Ball flight | Players | Best reaction (fit) | AUC fit | AUC test |
|---|---|---|---|---|
| v1 | v1 (reaction 0.7 s) | — | 0.705 | **0.719** |
| v1 | v1 | 0.4 s | 0.708 | 0.728 |
| Measured speed | v1 | **0.0 s** | 0.736 | **0.739** |
| Measured speed | Measured | 0.0 s | 0.730 | 0.720 |
| Measured speed, slowing along the pass | v1 | 0.0 s | 0.724 | 0.702 |
| Measured speed, slowing along the pass | Measured | 0.0 s | 0.717 | 0.688 |

- **Measured speed at the v1 reaction time of 0.7 s: 0.715** on the held-out matches, against v1's 0.719.
- **With the receiver at the end of the pass** (v1's view of a completed pass): 0.798 for v1 against 0.815 for the best candidate.
- **SkillCorner (12,103 passes, a different league and provider):**
  - v1 **0.781**;
  - measured speed with reaction 0 s **0.793** (with slowing along the pass, 0.792);
  - v1's ball with reaction 0 s, 0.769;
  - SkillCorner's own model 0.896.
- **Observed completion by availability, held out.** The best candidate orders passes more cleanly than v1, but neither is a probability: passes rated below 0.1 are still completed about 70% of the time.

| Availability band | v1 | Best candidate |
|---|---|---|
| 0–0.1 | 71% | 68% |
| 0.1–0.3 | 70% | 82% |
| 0.3–0.5 | 93% | 88% |
| 0.5–0.7 | 89% | 94% |
| 0.7–0.9 | 95% | 96% |
| 0.9–1 | 96% | 97% |

**Reading.**
- **The ball-speed model is wrong and now measured.** Real passes are 30% to 40% faster than v1 assumes up to 30 m. 8.5 m/s plus 0.165 m/s per metre halves the timing error on held-out matches. This is physics, with no outcome in the fit.
- **The physical corrections do not, on their own, improve completion ranking:**
  - the faster ball at the v1 reaction time scores 0.715 against 0.719;
  - timing along the pass and the measured player speeds both lower the score.
- **The best fit sets the reaction time at the edge of the grid (0 s).** That is not a human reaction time. The parameter is absorbing errors elsewhere in the availability formula, so **reaction time cannot be calibrated as a physical quantity** with this formula.
- **The gain from recalibrating is small:** +0.02 AUC on held-out IDSSE matches and +0.01 on SkillCorner. **The gap to SkillCorner's model (0.79 against 0.90) is not in the physics constants. It is in the formula's structure:**
  - interception counts any waypoint a defender reaches first, with no ball height, reach or time to control;
  - control at the receiver is a fixed-steepness sigmoid;
  - neither is a probability.
- **Not adopted; v1 stays.** The measured ball flight and player figures are recorded as measured physics, to be used when the interception model is rebuilt.

**Proposed next (a candidate, measured the same way):** a probabilistic, height-aware interception and control model. It would integrate each player's chance of controlling the ball along its measured flight, after the published pitch-control method (Spearman 2018). It would use:
- the measured ball flight;
- the league's flat or high flag, where present, to say when a ball can be reached;
- the measured player speeds.

Reaction time and control time would be fitted only as part of that model.

## A probabilistic pass model (19 November 2026)

**The design.** `plaiq_engine/analytics/pass_probability.py` (0.1.0) is a candidate beside targeted pitch control, which is untouched. It rebuilds the structure step 6 found wanting, after the published pitch-control model (Spearman 2018):
- **arrival:** each player reaches a point after a reaction time plus Plaiq's kinematic travel time, with uncertainty (a logistic curve of spread `sigma`);
- **in flight:** the ball travels with the measured flight (step 6). Defenders who have arrived where it passes may take it at rate `lam_flight`; a high ball can be reached only in the first and last `high_reach` of its length;
- **at the target:** both teams arrive and control the ball at rate `lam`. The passing team's share of control is the chance it keeps the ball;
- **the result** is a completion probability: flight survival × share at the target;
- **the target:** where the recipient stands at the pass, or **projected along their run to the ball's arrival** (`projected_target`, from positions and velocities, never the outcome);
- **separate reaction times** for the passing team and the defenders.

`tools/plaiq_pass_probability.py` fits it by likelihood on the four even IDSSE matches (1,852 passes, 91% completed). It tests it on the three odd matches (1,053 passes, 90%) and on all of SkillCorner's 20 matches (12,103 passes, 82%). As a ceiling for that provider, it also refits on SkillCorner's even matches and tests on its odd ones. The baselines are:
- the base rate;
- v1's availability, mapped to a probability on the fit matches so it is judged fairly;
- SkillCorner's own expected completion.

**Results.** Lower log-loss and Brier are better; AUC is ranking.

| | IDSSE held out: log-loss / AUC | SkillCorner, all 20 (IDSSE-fitted): log-loss / AUC |
|---|---|---|
| Base rate | 0.332 / 0.50 | 0.514 / 0.50 |
| v1 availability (recalibrated) | 0.297 / 0.719 | 0.424 / 0.781 |
| Published parameters, unfitted | 0.533 / 0.772 | 0.541 / 0.807 |
| Fitted, target at the pass | 0.305 / 0.755 | 0.403 / 0.812 |
| **Fitted, projected target, height** | **0.296 / 0.762** | **0.370 / 0.842** |
| Fitted, projected target, no height | 0.301 / 0.773 | 0.373 / 0.838 |
| Fitted, projected target, measured player speeds | 0.296 / 0.768 | 0.370 / 0.842 |
| SkillCorner's own model | — | 0.279 / 0.896 |

Refitted on SkillCorner's even matches and tested on its odd ones (5,930 passes), log-loss / AUC:
- target at the pass: 0.348 / 0.832;
- **projected target: 0.321 / 0.859;**
- the IDSSE-fitted model on the same passes: 0.375 / 0.835;
- SkillCorner's own model: 0.283 / 0.894.

*(Corrected 19 November 2026: every SkillCorner result above that uses height reads SkillCorner's `high_pass` flag, which is logged on completed passes only, so it partly gives away the outcome. The refit's 0.321 / 0.859, its calibration column below and its parameters are not a valid measure of the model; its no-height refit, 0.356 / 0.838, is. The IDSSE-fitted rows move little (0.370 against 0.373). See "Pass height through Matrix" below.)*

Observed completion by predicted probability (projected target), predicted → completed:

| Predicted | IDSSE held out (IDSSE fit) | SkillCorner (IDSSE fit) | SkillCorner held out (its own refit) |
|---|---|---|---|
| Below 0.5 | 31 passes: 0.36 → 0.77 | 0.37 → 0.31 | 0.23 → 0.22 |
| 0.5–0.7 | 52: 0.61 → 0.37 | 0.61 → 0.46 | 0.61 → 0.67 |
| 0.7–0.8 | 47: 0.76 → 0.79 | 0.76 → 0.62 | 0.75 → 0.74 |
| 0.8–0.9 | 146: 0.86 → 0.87 | 0.86 → 0.76 | 0.86 → 0.84 |
| 0.9–0.95 | 188: 0.93 → 0.92 | 0.93 → 0.86 | 0.93 → 0.93 |
| 0.95–1 | 589: 0.98 → 0.96 | 0.98 → 0.97 | 0.98 → 0.98 |

**Fitted parameters** (IDSSE, projected target, height):
- arrival spread 0.60 s;
- control rate at the target 13.6 per s, in flight 0.51 per s;
- **reaction 0.25 s for the passing team and 1.01 s for the defenders;**
- reachable share of a high ball 0.47.

SkillCorner's refit gives arrival spread 0.55 s, rates 4.4 and 7.1 per s, reactions 0.78 s and 1.28 s, and a reachable share of 0.

**Reading.**
- **The structure was the gap.** On SkillCorner the model trained on the league's matches cuts v1's log-loss from 0.424 to 0.370 and lifts AUC from 0.781 to 0.842. Refitted on SkillCorner's own matches it reaches 0.321 / 0.859. That closes about half (transfer) to two-thirds (refit) of the AUC gap to SkillCorner's model.
- **Its output is a probability.** Refitted on the provider, predicted and observed completion agree within about 0.06 in every band.
- **Projecting the receiver along their run matters most.** It turns a through ball into a pass to where the receiver will be, and uses no outcome.
- **On IDSSE's held-out matches the gain is smaller:** AUC 0.719 to 0.762, with log-loss level with recalibrated v1. Those matches hold only 108 failed passes, and the lowest bands hold 31 and 52 passes, so the low end there is noisy.
- **Calibration depends on the data.** Fitted on the league, the model over-predicts SkillCorner's completion by about 6 points (0.875 against 0.819). A probability published to analysts would need fitting per provider, or on pooled data.
- **Not established:**
  - **Height:** the fitted reachable share is about 0.5 on the league's data (height makes little difference) and 0 on SkillCorner's (a high ball cannot be cut out in flight), and the two variants score alike;
  - **measured player speeds:** they score the same as v1's;
  - **the parameters as physical quantities:** some sit at their bounds in some variants. One pattern holds in every fit: **defenders effectively start 0.5 to 0.75 s after the passing team**, which reads as the receiver knowing where the ball is going.
- **Not adopted; v1 stays.** This is a candidate for the founder: a "pass completion probability" metric beside v1, labelled estimated. It would also be the availability half of the option split and the risk term of a narrower decision quality version 2.

## Pass completion probability adopted (19 November 2026)

**Founder decision, 19 November 2026:** the probabilistic pass model is adopted as the **pass completion probability**: a new metric beside v1 (targeted pitch control, POV and decision quality are unchanged), labelled estimated, with parameters fitted per provider.

**What ships.**
- **The model:** `analytics/pass_probability.py` 1.0.0. The receiver is projected along their run; there is no height flag, since normalised events carry none yet.
- **Parameter sets** (`PARAMETER_SETS`, `parameters_for`):
  - `skillcorner-2026-11-19`, fitted on SkillCorner's 10 even matches;
  - `default-idsse-2026-11-19`, fitted on IDSSE's 4 even matches, for every provider without its own set.
- **`MatchAnalysis.pass_completion`:** one row per `PASS` and `BALL LOST`.
  - **Positions:** the tracked ball at the event's start frame, or the passer when the ball is not detected, and never the event's coordinates (see below).
  - **The target** is the logged target. A pass without one gets no probability and a reason. Plaiq's inferred failed-pass receiver is never used, since it is not validated.
- **Semantic events, schema 1.2:** `pass_completion_probability`, `pass_completion_status` (`estimated`, or the reason there is none, including set pieces, since the model is fitted on open play) and `pass_completion_parameters`.
- **Data check:** a new "Pass completion probability" entry, now 13 metrics.
  - **Full or Degraded with the positions** where the provider has its own set.
  - **Degraded** where it uses the default set: "fitted on another source's tracking … may be off by several points".
  - **Blocked** without tracking or events.
  - A cross-engine test keeps Matrix's list of fitted providers in step with Plaiq's sets.

**The shipped path, end to end** (`tools/plaiq_pass_completion_check.py`). On SkillCorner's 20 matches, 16,041 of 17,547 passes have a logged target and get a probability. On the 10 held-out matches (7,749 passes):
- **log-loss 0.327, AUC 0.843**, Brier 0.100;
- mean predicted 0.841 against 0.842 completed;
- predicted against observed completion by band: 0.35 → 0.39, 0.61 → 0.63, 0.76 → 0.76, 0.86 → 0.85, 0.93 → 0.91, 0.98 → 0.98.

On the demo match (1886347), 881 of 884 pass rows carry a probability, averaging 0.90 on completed passes and 0.64 on lost ones. About 4 s a match.

**Correction to the section above:**
- The no-height model refitted on SkillCorner scores 0.356 / 0.838 held out, against 0.321 / 0.859 with SkillCorner's `high_pass` flag. **On SkillCorner the height flag helps;** it is on the league's data that it does not. *(Withdrawn 19 November 2026: SkillCorner logs the flag on completed passes only, so the gain is the outcome leaking in. See "Pass height through Matrix" below.)*
- Carrying a provider's height flag through Matrix's normalised events, and refitting with it, is the follow-up.

**Found on the way: Metrica CSV second-half events are not rotated.**
- **The defect:** Matrix's Metrica CSV loader (Sample Games 1 and 2) rotates the second-half tracking but not the second-half event coordinates. Second-half pass starts lie a median 55 m from the tracked ball and agree (0.35 m) only when mirrored.
- **Unaffected:** first halves, Metrica's EPTS Game 3 and SkillCorner (under 1.4 m in both halves).
- **The first symptom:** completed passes averaged 0.91 in first halves and 0.44 to 0.46 in second halves, until the metric switched to tracked positions (now 0.91 to 0.92 in both).
- **Other readers of event coordinates on those two games are affected,** for example the semantic events' `pitch_zone`.
- **Not fixed here.** Fixing it changes recorded outputs on those games, so it is reported for a decision. *(Fixed 19 November 2026 at the founder's go-ahead: each table is split by its own `Period`; see `CONTEXT_STRUCTURE_REPORT.md`, "Metrica CSV second-half events fixed".)*

## Pass height through Matrix (19 November 2026)

**What was asked:** carry a provider's pass-height flag into Matrix's normalised events and refit the pass completion parameters with it, on the strength of the SkillCorner refit (0.356 to 0.321 log-loss).

**Built:**
- **`High Pass`**, an optional, nullable event column (`second_spectrum_io.HIGH_PASS`; `True`, `False` or unknown). As with `native_quality` in the tracking schema, absent is valid, and a consumer treats a missing column as unknown, never as "not high".
- **SkillCorner:** from `high_pass`, on passes only.
- **StatsBomb:** from `pass_height`. "High Pass" is high; "Ground Pass" and "Low Pass" are not.
- **Metrica:** no column. Metrica logs no height, and `HEAD`, `CROSS` and `DEEP BALL` are not heights, so none is inferred.
- **Tests:** a loader test for each of SkillCorner and StatsBomb.

**Found: SkillCorner's flag records the outcome.** On the 20 open-data matches, `high_pass` (and `pass_range`) is present on completed passes only:
- 986 high and 11,821 not high, all of them completed;
- no unsuccessful pass carries the flag.

So on SkillCorner a pass's height is known only once its outcome is.
- **The earlier refit** read a missing flag as "not high", so every high pass in its data was completed. Its fitted reachable share of 0 ("a high ball cannot be cut out in flight") is that leak, not football.
- **The shipped path showed the same leak** when a with-height set was tried per pass:
  - held-out log-loss fell from 0.327 to 0.305;
  - on the 500 held-out high passes, all completed, it fell from 0.549 to 0.243;
  - on the other passes it barely moved (0.109 to 0.106);
  - and only completed passes can be flagged.
- **Where height is logged on every pass** (IDSSE: 390 of the 627 failed passes in step 6's set are high), height made no difference held out (0.296 against 0.301 log-loss; AUC 0.762 against 0.773).

**Decision taken here:**
- **The with-height set was not shipped.** The pass completion probability does not read `High Pass` (`pass_probability` stays 1.0.0, and a cross-engine test pins that the flag changes nothing).
- **The column stays in Matrix as a description of the pass,** with its provenance documented: on SkillCorner it is known only for completed passes, so it must never feed a model of the outcome. On StatsBomb it is logged on every pass, but StatsBomb is event-only, so the metric is Blocked there anyway.
- **Withdrawn:** "on SkillCorner the height flag helps", and the 0.321 / 0.859 refit as evidence for it (corrections marked above).

**Still open:**
- height from the tracking itself, for providers with ball height (TRACAB-like feeds);
- a provider that logs height on every pass and has tracking.

Neither is available in the reference data except IDSSE, where it did not help.

## Decision quality version 2 with the pass completion probability as its risk term (19 November 2026)

**What was asked:** use the adopted pass completion probability as the risk term of a narrower decision quality version 2.

**Two candidates** (`analytics/decision_quality_v2.py` 0.2.0, beside version 1 and the measured 0.1.0, all unchanged):
- **v1 with risk** (`decision_quality_risk`): version 1's score times the chosen pass's completion probability. Version 1's numerator is a value already realised, once the ball has arrived. Its denominator weighs every alternative by risk. The factor puts the chosen pass on the same footing and keeps version 1's credit for a ball played into a run.
- **Projected** (`decision_quality_projected`):
  - each team-mate's option is a pass into their run (`projected_target`);
  - each is worth its completion probability times POV's value part there (`option_value`);
  - the score is the chosen option's expected value over the best one's (0 to 1).
  - It is one formula for completed and failed passes, and the outcome is never an input.
  - It answers 0.1.0's flaw: comparing the destination with where team-mates stood, so a ball into a run looked like a ball into space.

**Measured** by `tools/plaiq_dq_v2.py` on the same 12,112 SkillCorner passes (9,920 completed). The completion probability's SkillCorner parameters were fitted on the even matches, so the odd matches are its held-out half.
- **Coverage:** projected scores 9,909 completed and 2,186 failed passes, because it needs only a logged, tracked target. Version 1 scores 9,910 and 2,102.
- **Check:** the projected candidate's chosen-pass probability equals the shipped `MatchAnalysis.pass_completion` exactly.

| | Version 1 | v1 with risk | **Projected** | Completion probability alone | 0.1.0 |
|---|---|---|---|---|---|
| Optimal share: completed / failed passes | 58% / **89%** | 31% / 23% | **31% / 27%** | 47% / 5% | 30% / 19% |
| Median: completed / failed | 0.98 / 1.24 | 0.88 / 0.70 | 0.78 / 0.75 | 0.94 / 0.62 | 0.76 / 0.41 |
| Possession leads to a shot, completed passes, AUC | 0.60 | 0.42 | **0.62** | 0.27 | 0.51 |
| Possession leads to a shot, all passes, AUC | 0.55 | 0.44 | **0.62** | 0.33 | 0.53 |
| Possession leads to a goal, completed passes, AUC (89 goals) | 0.62 | 0.35 | **0.63** | 0.23 | 0.51 |
| Rank correlation with SkillCorner's option-value analogue | **0.31** | 0.09 | 0.29 | −0.39 | 0.15 |
| Picks out passes where SkillCorner's best option was chosen, AUC | **0.60** | 0.51 | 0.58 | 0.37 | 0.54 |
| Predicts completion, AUC | 0.23 | 0.64 | **0.51** | 0.85 | 0.69 |

Shot rate by band, completed passes, poor → optimal: version 1 1.2%, 1.6%, 5.5%, 9.1%; projected 2.6%, 6.9%, 8.0%, 9.6%; v1 with risk 9.9%, 7.3%, 7.3%, 4.8% (reversed).

**By half** (even: fit half of the probability; odd: held out):

| | Version 1, even / odd | Projected, even / odd |
|---|---|---|
| Shot, completed passes, AUC | 0.61 / 0.60 | 0.63 / 0.61 |
| Shot, all passes, AUC | 0.55 / 0.55 | 0.63 / 0.61 |
| Goal, completed passes, AUC | 0.62 / 0.61 | 0.67 / 0.57 |
| Rank correlation with SkillCorner's analogue | 0.32 / 0.30 | 0.29 / 0.29 |
| Picks out SkillCorner's best, AUC | 0.60 / 0.60 | 0.58 / 0.58 |

**Reading.**
- **v1 with risk fails.** Multiplying by the completion probability turns the score into a measure of safety. It reverses the link to chances (shot AUC 0.42, and poor passes lead to shots twice as often as optimal ones) and loses SkillCorner's analogue (0.09). Not a candidate.
- **Projected is the first version 2 to keep version 1's signal and fix its flaw.**
  - **Failed passes:** they are no longer lenient (27% optimal against 31% for completed, where version 1 gives 89% against 58%), and one formula covers both.
  - **Chances:** shot AUC 0.62 against 0.60 on completed passes, and 0.62 against 0.55 on all passes, held on both halves. Shot rate rises steadily by band.
  - **SkillCorner's analogue:** close to version 1 (0.29 against 0.31; picking out its best 0.58 against 0.60).
  - **Completion:** neutral (AUC 0.51). It neither rewards safety (0.1.0, 0.69) nor penalises it (version 1, 0.23).
- **Where it is weaker or uncertain:**
  - The goal AUC rests on 89 goals and falls on the held-out half (0.57 against version 1's 0.61), so it is not established.
  - Its range is 0 to 1, not version 1's 0 to 1.5. The bands are kept untuned, which puts 31% of completed passes in optimal, near SkillCorner's own "chose its best" share of 37%.
  - Like version 1, it judges choices more kindly under pressure (median 0.70 unpressured, 0.89 under high pressure), because fewer options are on.
- **It inherits:**
  - Plaiq's threat surface for value (step 3: it hardly tracks a provider's threat; step 5's learned surface was not adopted);
  - the pass completion probability's per-provider fit, so on a provider without its own set it would be Degraded, as that metric is;
  - and it needs a logged target, as pass completion does (Metrica logs none for a failed pass).

**Not adopted; version 1 stays the default** (engine governance). Projected is a candidate for the founder: decision quality version 2 (0.2.0, projected), beside version 1 and labelled estimated. If adopted, failed-pass decision quality could be reported for the first time, from the same formula.

## Decision quality version 2 adopted (19 November 2026)

**Founder decision:** the projected candidate ships as decision quality version 2 (`decision_quality_v2` 1.0.0, the 0.2.0 maths unchanged), beside version 1, labelled estimated. Version 1 stays the default (`dq_score`, `dq_band`).

- **Engine:** `MatchAnalysis.decision_quality_v2`, one row per `PASS` and `BALL LOST` event that `pass_completion` covers: the score, the chosen and best expected values, the chosen pass's probability, the best option, the version and the parameter set, or the reason there is no score. Both properties read one shared moment of the pass (`_pass_moment`), so the chosen pass's probability is the shipped pass completion probability.
- **Semantic events, schema 1.3:** `dq_v2_score`, `dq_v2_status` (`estimated`, or the reason), `dq_v2_best_option_ref`, `dq_v2_version`; nested under `spatial_intelligence.decision_quality_v2`. Goalkeepers and set pieces are not scored, as in version 1. A failed pass that version 2 scores now gets a row even where version 1 has no failed-pass score.
- **Scale:** 0 to 1, no bands. Version 1's bands are cut on its 0 to 1.5 scale and are not applied; bands for version 2 wait until they are measured and agreed.
- **Data check:** a 14th metric, `decision_quality_v2`. It follows pass completion (Degraded on the default parameter set) and, like version 1, is Degraded where the events are derived from tracking; Blocked without tracking or events.
- **Checked end to end:** `tools/plaiq_dq_v2.py` now reads the shipped property instead of its own copy. On SkillCorner's 20 matches it reproduces the measured run: the same 12,112 passes, the same 12,095 scored, the chosen probabilities identical, scores within 2×10⁻⁷, and every grounding in the output identical to the table above.
- **Tests:** the synthetic match scores its open-play pass 0 to 1 with the shipped probability; the set piece and the goalkeeper are not scored; the same pass logged as lost scores the same and gets a row; the Data check states per provider and mode.
- **Not established, unchanged:** the goal link and any bands.

## Candidate bands for decision quality version 2 (19 November 2026)

Decision quality version 2 shipped 0 to 1 with no bands. `tools/plaiq_dq_v2_bands.py` measures candidate cut-off sets on the rows of the run above (12,095 scored SkillCorner passes, 9,909 completed). **Measured, not agreed.**

**Candidates:**
- **Version 1's cut-offs** (0.55, 0.80, 0.95), as they are: a baseline.
- **Meaning** (0.50, 0.75, 0.95): under half the best option's expected value is poor, under three-quarters suboptimal, within 5% of the best optimal.
- **Fitted** (0.45, 0.60, 0.90): a 0.05 grid chosen on the even matches only, maximising the likelihood of the possession leading to a shot (completed passes), every band at least 10% of passes, the rate rising band by band.
- **Fitted, top anchored** (0.45, 0.60, 0.95): the same with the top cut held at 0.95, so optimal keeps its meaning.

**Held-out half** (the 10 odd matches; nothing was fitted on them). Shot rate per band on completed passes, poor to optimal:

| Candidate | Cut-offs | Shot rate by band | Shares (all passes) | Rises in match resamples | Band AUC, shot |
|---|---|---|---|---|---|
| Version 1's cut-offs | 0.55 / 0.80 / 0.95 | 3.2%, 6.3%, 7.6%, 9.3% | 24 / 29 / 17 / 31% | 56% | 0.60 |
| Meaning | 0.50 / 0.75 / 0.95 | 3.1%, 5.4%, 7.8%, 9.3% | 19 / 27 / 23 / 31% | 80% | 0.60 |
| Fitted | 0.45 / 0.60 / 0.90 | 2.8%, 4.9%, 6.4%, 9.4% | 16 / 14 / 34 / 36% | 94% | 0.60 |
| **Fitted, top anchored** | **0.45 / 0.60 / 0.95** | **2.8%, 4.9%, 6.9%, 9.3%** | **16 / 14 / 39 / 31%** | **94%** | **0.60** |
| *Reference: version 1's bands on version 1* | 0.55 / 0.80 / 0.95 | 1.8%, 1.4%, 5.3%, 8.8% | 6 / 10 / 20 / 64% | 35% | 0.61 |

"Rises in match resamples" is the share of 1,000 resamples of the held-out matches (with replacement) in which the shot rate rises band by band; the Wilson intervals in the output treat passes as independent and overstate precision.

**Reading:**
- **Every candidate's shot rate rises band by band on both halves.** How reliably differs: the two fitted sets rise in 94% of held-out match resamples, the meaning set in 80%, version 1's cut-offs in 56%. Version 1's own bands on version 1 do not rise on the held-out half (poor 1.8% against suboptimal 1.4%).
- **The cut-offs barely change the overall ranking** (band AUC 0.60 for all): they differ in where the steps are, not in how much signal there is.
- **Where the steps are:** the shot rate climbs from about 0 below 0.2 to about 5% at 0.5 to 0.6, then flattens from about 0.7 (7 to 10%). The fitted lower cuts (0.45, 0.60) sit on the climb; above 0.6 the data separates little except the best option itself.
- **No failed-pass leniency in any candidate:** failed passes fall into the bands much as completed ones do (top anchored, held out: optimal 29% of failed against 31% of completed; version 1 on version 1: 88% against 59%).
- **The closest held-out steps are poor to suboptimal** (2.8% [1.9, 4.2] against 4.9% [3.5, 6.8] for the top-anchored set).
- **SkillCorner's analogue agrees weakly with any banding** (quadratic-weighted kappa 0.17 to 0.20 at the same cut-offs; 0.14 for version 1), as the rank correlation of 0.29 led us to expect; the share of passes where SkillCorner rates the chosen option its best rises band by band in every candidate (top anchored, held out: 26%, 35%, 35%, 45%).
- **Goals do not separate** on the held-out half (2, 6, 9 and 14 goals across the top-anchored bands); goals are not evidence for any set.

**Recommendation for the founder:** the **top-anchored set (0.45, 0.60, 0.95)**. It is as stable held out as the freely fitted set, keeps optimal as "the best option, within 5%", and puts the lower steps where the shot rate climbs. The meaning set (0.50, 0.75, 0.95) is the simpler one to explain, at the cost of a less reliable step between suboptimal and good.

**Limits:** one provider (SkillCorner's 20 open-data matches) and one outcome (the possession leading to a shot), which credits a pass for what happens after it. The bands describe the score; they do not validate decision quality as a construct. Before display, decide whether version 2's bands reuse version 1's names: with different cut-offs on a different scale, the same names invite merging the two.

## Decision quality version 2 on Plaiq's hex grid: the run-cell candidates (19 November 2026)

**Question:** can version 2 compute on the honeycomb without evaluating every cell? Every cell a team-mate can reach before the ball is not selective: on 150 passes of one match, a median of 1,407 of the pitch's 7,463 cells (1,174 when limited to within 45° of the run). The cells around where each team-mate's run reaches when the ball arrives are: a fixed 7 (one ring) or 19 (two rings) per team-mate. Version 1 offers no hex selection for passes (it scores team-mates where they stand); its kinematic cell selection serves carries only.

**Candidates** (`decision_quality_v2.py`, measured, not adopted; the shipped score stays 1.0.0): each team-mate's option is **the best ball into their run**, the best of the cells within 0, 1 or 2 rings of their run's cell, each worth the completion probability of a pass to the cell's centre times the cell's value (Plaiq's threat at the cell). The chosen option is scored the same way, so placement within the cluster (execution) is not charged to the choice. Ring 0 is the grid alone. `pass_arrays_one_moment` computes the same arrays as `pass_arrays` (a test pins exact equality) with each player's travel times taken once per moment, 4.5 times faster on a match.

**On the same 12,112 SkillCorner passes** (12,095 scored by every version):

| | Version 1 | Shipped (points) | Ring 0 | **Ring 1** | Ring 2 |
|---|---|---|---|---|---|
| Option evaluations per pass | | 10 | 10 | **70** | 190 |
| Rank correlation with the shipped score | | 1 | 0.993 | **0.979** | 0.961 |
| Optimal share (≥ 0.95): completed / failed | 58% / 89% | 31% / 27% | 31% / 27% | **31% / 30%** | 31% / 31% |
| Shot, completed passes, AUC | 0.60 | 0.62 | 0.62 | **0.62** | 0.62 |
| Shot, all passes, AUC | 0.55 | 0.62 | 0.62 | **0.62** | 0.62 |
| Goal, completed passes, AUC (held-out half) | 0.62 (0.61) | 0.63 (0.57) | 0.64 (0.58) | **0.66 (0.58)** | 0.66 (0.58) |
| Rank correlation with SkillCorner's analogue | 0.31 | 0.288 | 0.288 | **0.294** | 0.296 |
| Picks out SkillCorner's best, AUC | 0.60 | 0.577 | 0.578 | **0.581** | 0.582 |
| Predicts completion, AUC | 0.23 | 0.510 | 0.511 | **0.496** | 0.490 |
| Passes changing band (0.45 / 0.60 / 0.95) | | | 5% | **9%** | 14% |

**Paired resamples of the 20 matches** (300, difference from the shipped score, 95% interval):
- **Ring 1:** shot +0.002 [−0.003, +0.009]; goal +0.022 [0.000, +0.054]; SkillCorner analogue +0.006 [+0.002, +0.010]; picks its best +0.004 [+0.002, +0.007]; completion −0.014 [−0.019, −0.009].
- **Ring 2:** shot +0.002 [−0.007, +0.012]; goal +0.023 [−0.002, +0.049]; analogue +0.008 [+0.002, +0.014]; picks its best +0.006 [+0.001, +0.009]; completion −0.020 [−0.027, −0.013].

**Reading:**
- **The grid alone changes nothing** (ring 0 against the shipped score: rank correlation 0.993, every measure equal).
- **The rings are no worse anywhere that matters and slightly closer to the independent reference.** The link to shots is unchanged on both halves. Agreement with SkillCorner's analogue rises a little, outside the resampling noise, though still short of version 1's 0.31. The goal gain sits on the even half (held out: 0.576 against 0.574), so it is not established.
- **Failed passes gain more than completed ones** (mean +0.017 against +0.007 at ring 1): a hard ball into a run is credited with the best ball into that run, which is the intended split between choice and execution. Failed passes now look like completed ones in the optimal band (30% against 31%), not better, so there is no leniency.
- **The cost is a small lean towards risk:** completion AUC falls just below 0.5 (0.496 ring 1, 0.490 ring 2), small but outside the noise.
- **Ring 2 adds little over ring 1** (analogue +0.002) for nearly three times the cells, more lean towards risk and more passes changing band.
- **The recommended bands hold on ring 1.** Held out, 0.45 / 0.60 / 0.95 gives shots of 2.6%, 4.5%, 7.0%, 9.2% and rises in 99% of match resamples (three seeds: 99.2, 99.5, 99.2%), against 94% on the shipped score. A fit on ring 1's even matches picks 0.45 / 0.75 / 0.95, which holds out less well (79 to 81%). The resample share moves by a few points between seeds.

**Recommendation for the founder:** **ring 1** as decision quality version 2 1.1.0, replacing the point-based 1.0.0 beside version 1, labelled estimated. It puts version 2 on the honeycomb at 7 cells per team-mate, is no worse on any outcome, slightly closer to the reference, and keeps the recommended bands. Passes into space no run is heading for remain outside the option set.

**Note for the AI layer:** a cell on a 1 m² grid is effectively a coordinate. Cells can be stored, aggregated and drawn, but the AI layer should keep receiving zones and player references, never cell indices.

Evidence: `tools/plaiq_dq_v2.py` (columns `hex_r0`, `hex_r1`, `hex_r2`); bands on ring 1 with `tools/plaiq_dq_v2_bands.py` on the same rows.
