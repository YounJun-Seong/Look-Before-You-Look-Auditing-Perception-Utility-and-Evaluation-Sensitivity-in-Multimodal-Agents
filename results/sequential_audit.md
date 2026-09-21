# Two-step sequential audit (preregistered, 1 pass; n=100)

## strict

### Q1 — value of a second acquisition (cross-replay oracle over {stop-after-a1, a1→CROP-TR, a1→HIGHRES} − one-step a1)

| a1 | one-step a1 | seq oracle | Δ [95% CI] |
|---|--:|--:|:--|
| center | 0.728 | 0.822 | +0.0933 [+0.0467, +0.1450] |
| ocr | 0.693 | 0.785 | +0.0917 [+0.0417, +0.1450] |

### Q2 — best follow-up by a1 outcome (R1-select, R2-score share of states; changed = a1 answer ≠ STOP answer)

| a1 | condition | n | stay | →TR | →HIGHRES | mean continuation gain |
|---|---|--:|--:|--:|--:|--:|
| center | changed=0,a1-correct=0 | 18 | 0.00 | 0.83 | 0.17 | +0.259 |
| center | changed=0,a1-correct=1 | 62 | 0.00 | 1.00 | 0.00 | +0.016 |
| center | changed=1,a1-correct=0 | 10 | 0.00 | 1.00 | 0.00 | +0.267 |
| center | changed=1,a1-correct=1 | 10 | 0.00 | 0.90 | 0.10 | +0.200 |
| ocr | changed=0,a1-correct=0 | 13 | 0.00 | 0.85 | 0.15 | +0.231 |
| ocr | changed=0,a1-correct=1 | 60 | 0.02 | 0.98 | 0.00 | +0.000 |
| ocr | changed=1,a1-correct=0 | 20 | 0.00 | 0.90 | 0.10 | +0.217 |
| ocr | changed=1,a1-correct=1 | 7 | 0.14 | 0.86 | 0.00 | +0.000 |

### Q3 — predicting 'a second acquisition helps' from post-a1 answer signals (labels cross-replay: R2)

| a1 | P(a2 helps) | AUC: answer-changed | AUC: a1 answer len |
|---|--:|--:|--:|
| center | 0.15 | 0.696 | 0.558 |
| ocr | 0.10 | 0.739 | 0.543 |

### Q4 — frontier points (mean over replays; extra vtok)

| arm/trajectory | acc | vtok |
|---|--:|--:|
| center (1-step) | 0.728 | 235 |
| ocr (1-step) | 0.693 | 0 |
| crop_tr (1-step) | 0.767 | 243 |
| highres (1-step) | 0.735 | 309 |
| center→crop_tr | 0.790 | 478 |
| center→highres | 0.737 | 544 |
| ocr→crop_tr | 0.750 | 243 |
| ocr→highres | 0.725 | 309 |

## lenient

### Q1 — value of a second acquisition (cross-replay oracle over {stop-after-a1, a1→CROP-TR, a1→HIGHRES} − one-step a1)

| a1 | one-step a1 | seq oracle | Δ [95% CI] |
|---|--:|--:|:--|
| center | 0.792 | 0.883 | +0.0917 [+0.0433, +0.1450] |
| ocr | 0.812 | 0.887 | +0.0750 [+0.0283, +0.1267] |

### Q2 — best follow-up by a1 outcome (R1-select, R2-score share of states; changed = a1 answer ≠ STOP answer)

| a1 | condition | n | stay | →TR | →HIGHRES | mean continuation gain |
|---|---|--:|--:|--:|--:|--:|
| center | changed=0,a1-correct=0 | 14 | 0.00 | 0.64 | 0.36 | +0.452 |
| center | changed=0,a1-correct=1 | 66 | 0.00 | 1.00 | 0.00 | +0.000 |
| center | changed=1,a1-correct=0 | 8 | 0.00 | 1.00 | 0.00 | +0.333 |
| center | changed=1,a1-correct=1 | 12 | 0.08 | 0.83 | 0.08 | +0.028 |
| ocr | changed=0,a1-correct=0 | 8 | 0.00 | 0.62 | 0.38 | +0.500 |
| ocr | changed=0,a1-correct=1 | 65 | 0.02 | 0.98 | 0.00 | +0.000 |
| ocr | changed=1,a1-correct=0 | 12 | 0.08 | 0.75 | 0.17 | +0.250 |
| ocr | changed=1,a1-correct=1 | 15 | 0.07 | 0.93 | 0.00 | -0.022 |

### Q3 — predicting 'a second acquisition helps' from post-a1 answer signals (labels cross-replay: R2)

| a1 | P(a2 helps) | AUC: answer-changed | AUC: a1 answer len |
|---|--:|--:|--:|
| center | 0.14 | 0.674 | 0.631 |
| ocr | 0.11 | 0.706 | 0.561 |

### Q4 — frontier points (mean over replays; extra vtok)

| arm/trajectory | acc | vtok |
|---|--:|--:|
| center (1-step) | 0.792 | 235 |
| ocr (1-step) | 0.812 | 0 |
| crop_tr (1-step) | 0.825 | 243 |
| highres (1-step) | 0.840 | 309 |
| center→crop_tr | 0.832 | 478 |
| center→highres | 0.838 | 544 |
| ocr→crop_tr | 0.828 | 243 |
| ocr→highres | 0.835 | 309 |
