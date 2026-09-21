# Multi-replay robustness (prereg: paper/PREREG_MULTI_REPLAY.md)

Oracle accuracy (and headroom vs cross-fitted best fixed) on the
deterministic arm set, as the number of replays K grows. 100-state
outcome-blind subsets; strict scores; 10k state-paired bootstrap.

## mme

Reference (full sample, n=196): K=2 1→1 oracle 0.500 [0.429, 0.571]. Rows below are the fixed 100-state subset for every K.

| K | naive same-replay | 1→1 cross | 1→mean(R2..RK) | symmetric LOO | best fixed |
|---|--:|--:|--:|--:|--:|
| 2 (n=100) | 0.645 | 0.530 | 0.530 | 0.515 | 0.465 |
| 3 (n=100) | 0.663 | 0.530 | 0.510 | 0.503 | 0.447 |
| 4 (n=100) | 0.662 | 0.530 | 0.517 | 0.507 | 0.443 |
| 5 (n=100) | 0.664 | 0.530 | 0.527 | 0.515 | 0.484 |

Headroom (oracle − best fixed) with 95% CI of the oracle:

| K | 1→1 | 1→mean | symmetric |
|---|--:|--:|--:|
| 2 | +0.065 [0.430, 0.630] | +0.065 [0.430, 0.630] | +0.050 [0.425, 0.605] |
| 3 | +0.083 [0.430, 0.630] | +0.063 [0.420, 0.600] | +0.057 [0.420, 0.587] |
| 4 | +0.088 [0.430, 0.630] | +0.074 [0.427, 0.607] | +0.064 [0.424, 0.588] |
| 5 | +0.046 [0.430, 0.630] | +0.043 [0.440, 0.613] | +0.031 [0.434, 0.595] |

## textvqa

Reference (full sample, n=200): K=2 1→1 oracle 0.787 [0.733, 0.837]. Rows below are the fixed 100-state subset for every K.

| K | naive same-replay | 1→1 cross | 1→mean(R2..RK) | symmetric LOO | best fixed |
|---|--:|--:|--:|--:|--:|
| 2 (n=100) | 0.885 | 0.773 | 0.773 | 0.802 | 0.722 |
| 3 (n=100) | 0.888 | 0.773 | 0.805 | 0.822 | 0.721 |
| 4 (n=100) | 0.887 | 0.773 | 0.823 | 0.830 | 0.723 |
| 5 (n=100) | 0.888 | 0.773 | 0.830 | 0.836 | 0.723 |

Headroom (oracle − best fixed) with 95% CI of the oracle:

| K | 1→1 | 1→mean | symmetric |
|---|--:|--:|--:|
| 2 | +0.052 [0.697, 0.843] | +0.052 [0.697, 0.843] | +0.080 [0.737, 0.862] |
| 3 | +0.052 [0.697, 0.843] | +0.084 [0.738, 0.867] | +0.101 [0.761, 0.878] |
| 4 | +0.050 [0.697, 0.843] | +0.100 [0.761, 0.881] | +0.107 [0.770, 0.884] |
| 5 | +0.051 [0.697, 0.843] | +0.107 [0.768, 0.887] | +0.114 [0.777, 0.890] |
