# Gate 1 Audit Table

**Verdict: STOP** (pre-registered thresholds, SPEC §9)  
H_net = -1.7pp (CI low -9.0), P_policy = 11.9pp (CI low 7.1), DMG = 19.0%, diverse datasets 1/1

## Table 1 — accuracy and headroom (%)

| set | n | coarse STOP | full image | best fixed (cf) | naive oracle | oracle (cf) | dummy oracle | H net | P policy |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pooled | 200 | 50.5 | 63.3 | 62.4 | 80.3 | 74.3 | 69.7 | -1.7 | 11.9 |
| textvqa | 200 | 50.5 | 63.3 | 62.4 | 80.3 | 74.3 | 69.7 | -1.7 | 11.9 |

## Table 2 — best action type on decisive samples (%)

| set | n decisive | STOP | CROP | OCR | HIGHRES | max share |
|---|--:|--:|--:|--:|--:|--:|
| pooled | 103 | 36.9 | 30.1 | 30.1 | 2.9 | 36.9 |
| textvqa | 103 | 36.9 | 30.1 | 30.1 | 2.9 | 36.9 |

## Table 3 — exclusive solve and harm (%)

| set | excl STOP | excl CROP | excl OCR | excl HIGHRES | DMG | harm CROP | harm OCR | harm HIGHRES |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| pooled | 0.0 | 7.5 | 4.5 | 1.5 | 19.0 | 37.0 | 9.8 | 9.8 |
| textvqa | 0.0 | 7.5 | 4.5 | 1.5 | 19.0 | 37.0 | 9.8 | 9.8 |

## Table 4 — recovery overlap among base failures (counts)

| recovering action types | samples |
|---|--:|
| none | 43 |
| CROP+HIGHRES | 16 |
| CROP | 15 |
| CROP+HIGHRES+OCR | 12 |
| OCR | 9 |
| CROP+OCR | 7 |
| HIGHRES | 3 |
| HIGHRES+OCR | 3 |

## Table 5 — budget-bottleneck subset (coarse wrong AND full-image right)

n = 34 (17.0% of samples); recovered by at least one action: 94.1%

| | STOP | CROP | OCR | HIGHRES |
|---|--:|--:|--:|--:|
| solves the sample | 0.0 | 85.3 | 44.1 | 61.8 |
| is the best action | 0.0 | 53.1 | 46.9 | 0.0 |
| solves it exclusively | 0.0 | 17.6 | 5.9 | 0.0 |

recovery overlap inside the subset: CROP+HIGHRES=11, CROP+HIGHRES+OCR=9, CROP=6, CROP+OCR=3, none=2, OCR=2, HIGHRES+OCR=1

## Table 6 — label reliability

- per-action R1/R2 agreement: 92.7%
- top-1 family agreement (decisive): 79.6%
- parse failures: 0 / 7000 rows
