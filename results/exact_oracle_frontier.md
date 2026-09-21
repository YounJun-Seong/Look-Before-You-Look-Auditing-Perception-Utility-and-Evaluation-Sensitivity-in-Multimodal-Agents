# Exact budget-constrained oracle (MCKP DP) vs Lagrangian sweep

Same cross-replay protocol and deterministic arms as the Lagrangian
frontier; the DP removes the relaxation on the selection objective.
`randomized` = upper concave envelope of the evaluated Lagrangian
points (a state-independent randomization over sweep operating
points). Strict scores as logged.

## mme  (n=196, STOP 0.352, always-HIGHRES 994 tok)

| budget | Lagrangian acc | exact DP acc | randomized hull | DP − Lagrangian |
|---|--:|--:|--:|--:|
| 5% | 0.474 | 0.490 | 0.476 | +0.0153 |
| 10% | 0.480 | 0.495 | 0.480 | +0.0153 |
| 20% | 0.480 | 0.495 | 0.480 | +0.0153 |
| 40% | 0.480 | 0.495 | 0.480 | +0.0153 |
| 100% | 0.480 | 0.495 | 0.480 | +0.0153 |

## textvqa  (n=200, STOP 0.623, always-HIGHRES 308 tok)

| budget | Lagrangian acc | exact DP acc | randomized hull | DP − Lagrangian |
|---|--:|--:|--:|--:|
| 5% | 0.735 | 0.762 | 0.761 | +0.0275 |
| 10% | 0.807 | 0.797 | 0.808 | -0.0108 |
| 20% | 0.813 | 0.806 | 0.813 | -0.0075 |
| 40% | 0.813 | 0.806 | 0.813 | -0.0075 |
| 100% | 0.813 | 0.806 | 0.813 | -0.0075 |
