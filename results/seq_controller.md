# (B) History-conditioned two-step controller + (C) transfer (dev role, n=100 states x 2 replays)

Controller features (fixed a priori): [a1 answer != STOP answer,
len(a1 answer)]; two ridge Δ-heads, argmax(0, Δ̂_TR, Δ̂_HI).
Same-replay decision+outcome; controller rows are 5-fold
cross-fitted by state. Oracle = cross-replay (audit Q1).

## strict

### a1 = center  (controller look-rate 1.00, incremental cost 270 tok)

| policy | acc |
|---|--:|
| stop after a1 | 0.728 |
| always →TR | 0.790 |
| always →HIGHRES | 0.737 |
| random @ rate | 0.763 |
| mixture (in-sample) | 0.790 |
| mixture (cross-fit) | 0.790 |
| **controller (cross-fit)** | **0.773** |
| sequential oracle (cross-replay) | 0.822 |

controller − mixture-CF = -0.0167 [-0.0367, +0.0000]; controller − stop-after-a1 = +0.0450 [+0.0117, +0.0800]

### a1 = ocr  (controller look-rate 0.91, incremental cost 244 tok)

| policy | acc |
|---|--:|
| stop after a1 | 0.693 |
| always →TR | 0.750 |
| always →HIGHRES | 0.725 |
| random @ rate | 0.733 |
| mixture (in-sample) | 0.750 |
| mixture (cross-fit) | 0.750 |
| **controller (cross-fit)** | **0.747** |
| sequential oracle (cross-replay) | 0.785 |

controller − mixture-CF = -0.0033 [-0.0250, +0.0167]; controller − stop-after-a1 = +0.0533 [+0.0117, +0.0983]

### (C) cross-action transfer (heads fitted on one a1 branch, applied unchanged to the other)

| direction | native CF acc | transferred acc | Δ (transfer − native) |
|---|--:|--:|--:|
| center→ocr | 0.747 | 0.753 | +0.0067 |
| ocr→center | 0.773 | 0.792 | +0.0183 |

## lenient

### a1 = center  (controller look-rate 1.00, incremental cost 293 tok)

| policy | acc |
|---|--:|
| stop after a1 | 0.792 |
| always →TR | 0.832 |
| always →HIGHRES | 0.838 |
| random @ rate | 0.835 |
| mixture (in-sample) | 0.837 |
| mixture (cross-fit) | 0.812 |
| **controller (cross-fit)** | **0.863** |
| sequential oracle (cross-replay) | 0.883 |

controller − mixture-CF = +0.0509 [+0.0210, +0.0830]; controller − stop-after-a1 = +0.0717 [+0.0333, +0.1100]

### a1 = ocr  (controller look-rate 0.94, incremental cost 268 tok)

| policy | acc |
|---|--:|
| stop after a1 | 0.812 |
| always →TR | 0.828 |
| always →HIGHRES | 0.835 |
| random @ rate | 0.830 |
| mixture (in-sample) | 0.832 |
| mixture (cross-fit) | 0.823 |
| **controller (cross-fit)** | **0.825** |
| sequential oracle (cross-replay) | 0.887 |

controller − mixture-CF = +0.0020 [-0.0310, +0.0349]; controller − stop-after-a1 = +0.0133 [-0.0250, +0.0517]

### (C) cross-action transfer (heads fitted on one a1 branch, applied unchanged to the other)

| direction | native CF acc | transferred acc | Δ (transfer − native) |
|---|--:|--:|--:|
| center→ocr | 0.825 | 0.843 | +0.0183 |
| ocr→center | 0.863 | 0.825 | -0.0383 |
