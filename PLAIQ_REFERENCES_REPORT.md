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
- **Plaiq's ball-speed model weighs on version 2.** At 5 m/s plus 0.2 m/s per metre, long forward passes look slower, and so riskier, than they are. Calibrating it on the league's 25 Hz tracking (programme step 6) comes before any risk-weighted decision quality.
- **Version 1's failed-pass leniency stays a known flaw** (89% optimal). Until it is fixed, failed-pass decision quality should not be reported.
- **The candidate for the next attempt:** keep version 1's end-of-pass credit and add only the risk at the pass, after the ball-speed calibration.
