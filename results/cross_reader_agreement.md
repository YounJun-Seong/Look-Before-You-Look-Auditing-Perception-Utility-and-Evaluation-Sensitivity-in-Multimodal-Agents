# Cross-reader opportunity agreement (Qwen vs Phi-4, post-hoc)

help = some deterministic additional arm strictly beats STOP
(mean over both replays; grid CROPs, OCR, HIGHRES). Same states,
same views, strict scores as logged.

## mme  (n=196)

| Both | Qwen only | Phi-4 only | Neither |
|--:|--:|--:|--:|
| 35 (18%) | 36 (18%) | 34 (17%) | 91 (46%) |

- P(help): Qwen 0.36, Phi-4 0.35
- P(Phi-4 help | Qwen help) = 0.49; P(Qwen help | Phi-4 help) = 0.51
- Jaccard 0.33; Cohen's kappa 0.22
- Best-family agreement given both-help: 0.77 (n=35)

## textvqa  (n=200)

| Both | Qwen only | Phi-4 only | Neither |
|--:|--:|--:|--:|
| 22 (11%) | 44 (22%) | 51 (26%) | 83 (42%) |

- P(help): Qwen 0.33, Phi-4 0.36
- P(Phi-4 help | Qwen help) = 0.33; P(Qwen help | Phi-4 help) = 0.30
- Jaccard 0.19; Cohen's kappa -0.05
- Best-family agreement given both-help: 0.73 (n=22)
