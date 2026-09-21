# When does selective perception pay off?

Question-conditioned vs question-agnostic perception, computed from the Gate 0 logs. The dummy arm is read as two distinct layers, not one luck floor.

## MME-RealWorld-Lite (n = 98 with both arms)

| layer | what it buys | accuracy |
|---|---|--:|
| STOP (coarse only) | nothing | 32.7 |
| Control 1 — stochasticity floor | re-ask, re-encode (no new spatial evidence) | 30.6 |
| Control 2 — question-agnostic | 8 random crops | 41.8 |
| Structured | 8 question-conditioned crops | 43.9 |
| FULL | the whole image at native budget | 33.7 |

**Selectivity advantage** (structured − question-agnostic): **+2.0pp** — wins 8, losses 6, ties 84

| domain | n | selectivity advantage (pp) | wins | losses |
|---|--:|--:|--:|--:|
| Autonomous_Driving | 79 | +2.5 | 6 | 4 |
| Monitoring | 12 | +16.7 | 2 | 0 |
| Diagram and Table | 7 | -28.6 | 0 | 2 |

## TextVQA (n = 100 with both arms)

| layer | what it buys | accuracy |
|---|---|--:|
| STOP (coarse only) | nothing | 59.3 |
| Control 1 — stochasticity floor | re-ask, re-encode (no new spatial evidence) | 70.3 |
| Control 2 — question-agnostic | 8 random crops | 83.0 |
| Structured | 8 question-conditioned crops | 84.7 |
| FULL | the whole image at native budget | 74.7 |

**Selectivity advantage** (structured − question-agnostic): **+1.7pp** — wins 4, losses 2, ties 94

## What predicts the selectivity advantage

Per-sample outcome of question-conditioned crops minus question-agnostic crops (+1 / 0 / -1), binned by sample features.

### by native resolution (visual tokens)

| bin | n | advantage (pp) | wins | losses |
|---|--:|--:|--:|--:|
| 777–999 | 76 | +0.0 | 2 | 2 |
| 1036–1824 | 73 | +5.5 | 7 | 3 |
| 2535–8858 | 49 | +0.0 | 3 | 3 |

### by text density (OCR lines)

| bin | n | advantage (pp) | wins | losses |
|---|--:|--:|--:|--:|
| 0–0 | 131 | +2.3 | 7 | 4 |
| 1–1 | 23 | +13.0 | 4 | 1 |
| 2–552 | 44 | -4.5 | 1 | 3 |

### by text coverage (% of image area)

| bin | n | advantage (pp) | wins | losses |
|---|--:|--:|--:|--:|
| 0–0 | 131 | +2.3 | 7 | 4 |
| 0–0 | 1 | +0.0 | 0 | 0 |
| 0–26 | 66 | +1.5 | 5 | 4 |

### by whether the coarse view already answered it

| coarse | n | advantage (pp) | wins | losses |
|---|--:|--:|--:|--:|
| already right | 91 | -1.1 | 0 | 1 |
| wrong | 107 | +4.7 | 12 | 7 |
