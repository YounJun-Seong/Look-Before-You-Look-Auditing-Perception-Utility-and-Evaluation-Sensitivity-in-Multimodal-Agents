# Budget-constrained cross-replay oracle frontier (post-hoc, Gate 0)

max_g E[y(g(s))] s.t. E[c(g(s))] <= B, computed by Lagrangian sweep
(select on one replay, score on the other, symmetrised). Deterministic
arms only (STOP, grid CROPs, OCR, HIGHRES); strict scores as logged.

## mme  (n=196, STOP 0.247, unconstrained oracle 0.355 @ 47 tok; always-HIGHRES cost 994 tok)

| budget | tok | oracle acc | Δ vs STOP | % of oracle gain |
|---|--:|--:|--:|--:|
| 5% | 50 | 0.355 | +0.107 | 100% |
| 10% | 99 | 0.355 | +0.107 | 100% |
| 20% | 199 | 0.355 | +0.107 | 100% |
| 40% | 398 | 0.355 | +0.107 | 100% |
| 100% | 994 | 0.355 | +0.107 | 100% |

## textvqa  (n=200, STOP 0.501, unconstrained oracle 0.723 @ 45 tok; always-HIGHRES cost 308 tok)

| budget | tok | oracle acc | Δ vs STOP | % of oracle gain |
|---|--:|--:|--:|--:|
| 5% | 15 | 0.636 | +0.135 | 61% |
| 10% | 31 | 0.702 | +0.201 | 90% |
| 20% | 62 | 0.723 | +0.223 | 100% |
| 40% | 123 | 0.723 | +0.223 | 100% |
| 100% | 308 | 0.723 | +0.223 | 100% |
