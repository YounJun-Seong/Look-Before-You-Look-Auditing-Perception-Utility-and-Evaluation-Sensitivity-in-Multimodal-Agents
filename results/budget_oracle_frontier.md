# Budget-constrained cross-replay oracle frontier (post-hoc, Gate 0)

max_g E[y(g(s))] s.t. E[c(g(s))] <= B, computed by Lagrangian sweep
(select on one replay, score on the other, symmetrised). Deterministic
arms only (STOP, grid CROPs, OCR, HIGHRES); strict scores as logged.

## mme  (n=196, STOP 0.352, unconstrained oracle 0.480 @ 70 tok; always-HIGHRES cost 994 tok)

| budget | tok | oracle acc | Δ vs STOP | % of oracle gain |
|---|--:|--:|--:|--:|
| 5% | 50 | 0.474 | +0.122 | 96% |
| 10% | 99 | 0.480 | +0.128 | 100% |
| 20% | 199 | 0.480 | +0.128 | 100% |
| 40% | 398 | 0.480 | +0.128 | 100% |
| 100% | 994 | 0.480 | +0.128 | 100% |

## textvqa  (n=200, STOP 0.623, unconstrained oracle 0.813 @ 35 tok; always-HIGHRES cost 308 tok)

| budget | tok | oracle acc | Δ vs STOP | % of oracle gain |
|---|--:|--:|--:|--:|
| 5% | 15 | 0.735 | +0.112 | 59% |
| 10% | 31 | 0.807 | +0.184 | 97% |
| 20% | 62 | 0.813 | +0.190 | 100% |
| 40% | 123 | 0.813 | +0.190 | 100% |
| 100% | 308 | 0.813 | +0.190 | 100% |
