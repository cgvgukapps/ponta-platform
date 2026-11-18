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
