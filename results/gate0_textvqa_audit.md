# Gate 1 Audit Table

**Verdict: STOP** (pre-registered thresholds, SPEC §9)  
H_net = -2.7pp (CI low -9.3), P_policy = 9.8pp (CI low 4.7), DMG = 13.0%, diverse datasets 1/1

## Table 1 — accuracy and headroom (%)

| set | n | coarse STOP | full image | best fixed (cf) | naive oracle | oracle (cf) | dummy oracle | H net | P policy |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pooled | 200 | 60.0 | 75.5 | 69.1 | 88.2 | 78.8 | 83.0 | -2.7 | 9.8 |
| textvqa | 200 | 60.0 | 75.5 | 69.1 | 88.2 | 78.8 | 83.0 | -2.7 | 9.8 |

## Table 2 — best action type on decisive samples (%)

| set | n decisive | STOP | CROP | OCR | HIGHRES | max share |
|---|--:|--:|--:|--:|--:|--:|
| pooled | 81 | 32.1 | 45.7 | 19.8 | 2.5 | 45.7 |
| textvqa | 81 | 32.1 | 45.7 | 19.8 | 2.5 | 45.7 |

## Table 3 — exclusive solve and harm (%)

| set | excl STOP | excl CROP | excl OCR | excl HIGHRES | DMG | harm CROP | harm OCR | harm HIGHRES |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| pooled | 0.0 | 11.5 | 0.5 | 1.0 | 13.0 | 16.1 | 10.2 | 4.2 |
| textvqa | 0.0 | 11.5 | 0.5 | 1.0 | 13.0 | 16.1 | 10.2 | 4.2 |

## Table 4 — recovery overlap among base failures (counts)

| recovering action types | samples |
|---|--:|
| none | 27 |
| CROP | 23 |
| CROP+HIGHRES | 14 |
| CROP+HIGHRES+OCR | 10 |
| CROP+OCR | 5 |
| HIGHRES | 2 |
| OCR | 1 |

## Table 5 — budget-bottleneck subset (coarse wrong AND full-image right)

n = 32 (16.0% of samples); recovered by at least one action: 96.9%

| | STOP | CROP | OCR | HIGHRES |
|---|--:|--:|--:|--:|
| solves the sample | 0.0 | 90.6 | 34.4 | 62.5 |
| is the best action | 0.0 | 58.1 | 35.5 | 6.5 |
| solves it exclusively | 0.0 | 25.0 | 0.0 | 6.2 |

recovery overlap inside the subset: CROP+HIGHRES=10, CROP+HIGHRES+OCR=8, CROP=8, CROP+OCR=3, HIGHRES=2, none=1

## Table 6 — label reliability

- per-action R1/R2 agreement: 95.9%
- top-1 family agreement (decisive): 77.8%
- parse failures: 0 / 7000 rows


## Breakdown by dataset

| dataset | n | coarse | full | oracle cf | dummy | H net | best STOP | CROP | OCR | HIGHRES |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| _none | 200 | 60.0 | 75.5 | 78.8 | 83.0 | -2.7 | 32.1 | 45.7 | 19.8 | 2.5 |
