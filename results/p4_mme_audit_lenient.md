# Gate 1 Audit Table

**Verdict: STOP** (pre-registered thresholds, SPEC §9)  
H_net = 3.1pp (CI low -2.0), P_policy = 2.8pp (CI low -2.2), DMG = 7.7%, diverse datasets 1/1

## Table 1 — accuracy and headroom (%)

| set | n | coarse STOP | full image | best fixed (cf) | naive oracle | oracle (cf) | dummy oracle | H net | P policy |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pooled | 196 | 26.0 | 35.7 | 34.9 | 49.0 | 37.8 | 31.6 | 3.1 | 2.8 |
| mme_realworld | 196 | 26.0 | 35.7 | 34.9 | 49.0 | 37.8 | 31.6 | 3.1 | 2.8 |

## Table 2 — best action type on decisive samples (%)

| set | n decisive | STOP | CROP | OCR | HIGHRES | max share |
|---|--:|--:|--:|--:|--:|--:|
| pooled | 59 | 25.4 | 50.8 | 22.0 | 1.7 | 50.8 |
| mme_realworld | 59 | 25.4 | 50.8 | 22.0 | 1.7 | 50.8 |

## Table 3 — exclusive solve and harm (%)

| set | excl STOP | excl CROP | excl OCR | excl HIGHRES | DMG | harm CROP | harm OCR | harm HIGHRES |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| pooled | 0.5 | 6.6 | 2.6 | 0.5 | 7.7 | 43.3 | 40.0 | 13.3 |
| mme_realworld | 0.5 | 6.6 | 2.6 | 0.5 | 7.7 | 43.3 | 40.0 | 13.3 |

## Table 4 — recovery overlap among base failures (counts)

| recovering action types | samples |
|---|--:|
| none | 122 |
| CROP+HIGHRES | 17 |
| CROP | 13 |
| OCR | 5 |
| CROP+HIGHRES+OCR | 5 |
| CROP+OCR | 2 |
| HIGHRES | 1 |
| HIGHRES+OCR | 1 |

## Table 5 — budget-bottleneck subset (coarse wrong AND full-image right)

n = 25 (12.8% of samples); recovered by at least one action: 88.0%

| | STOP | CROP | OCR | HIGHRES |
|---|--:|--:|--:|--:|
| solves the sample | 0.0 | 76.0 | 28.0 | 72.0 |
| is the best action | 0.0 | 63.6 | 31.8 | 4.5 |
| solves it exclusively | 0.0 | 12.0 | 4.0 | 4.0 |

recovery overlap inside the subset: CROP+HIGHRES=11, CROP+HIGHRES+OCR=5, none=3, CROP=3, HIGHRES=1, OCR=1, HIGHRES+OCR=1

## Table 6 — label reliability

- per-action R1/R2 agreement: 79.6%
- top-1 family agreement (decisive): 67.8%
- parse failures: 9 / 7000 rows
