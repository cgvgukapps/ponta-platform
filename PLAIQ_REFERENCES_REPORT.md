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
