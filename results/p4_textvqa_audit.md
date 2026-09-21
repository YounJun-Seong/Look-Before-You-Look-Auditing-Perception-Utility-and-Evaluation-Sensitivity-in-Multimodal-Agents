# Gate 1 Audit Table

**Verdict: STOP** (pre-registered thresholds, SPEC §9)  
H_net = -0.3pp (CI low -7.3), P_policy = 10.0pp (CI low 6.0), DMG = 7.0%, diverse datasets 1/1

## Table 1 — accuracy and headroom (%)

| set | n | coarse STOP | full image | best fixed (cf) | naive oracle | oracle (cf) | dummy oracle | H net | P policy |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pooled | 200 | 25.8 | 33.7 | 33.3 | 49.7 | 43.3 | 45.3 | -0.3 | 10.0 |
| textvqa | 200 | 25.8 | 33.7 | 33.3 | 49.7 | 43.3 | 45.3 | -0.3 | 10.0 |

## Table 2 — best action type on decisive samples (%)

| set | n decisive | STOP | CROP | OCR | HIGHRES | max share |
|---|--:|--:|--:|--:|--:|--:|
| pooled | 61 | 23.0 | 32.8 | 37.7 | 6.6 | 37.7 |
| textvqa | 61 | 23.0 | 32.8 | 37.7 | 6.6 | 37.7 |

## Table 3 — exclusive solve and harm (%)

| set | excl STOP | excl CROP | excl OCR | excl HIGHRES | DMG | harm CROP | harm OCR | harm HIGHRES |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| pooled | 0.0 | 6.0 | 3.5 | 2.0 | 7.0 | 26.2 | 9.5 | 7.1 |
| textvqa | 0.0 | 6.0 | 3.5 | 2.0 | 7.0 | 26.2 | 9.5 | 7.1 |

## Table 4 — recovery overlap among base failures (counts)

| recovering action types | samples |
|---|--:|
| none | 111 |
| CROP | 12 |
| CROP+HIGHRES | 8 |
| CROP+OCR | 7 |
| OCR | 7 |
| CROP+HIGHRES+OCR | 5 |
| HIGHRES+OCR | 4 |
| HIGHRES | 4 |

## Table 5 — budget-bottleneck subset (coarse wrong AND full-image right)

n = 18 (9.0% of samples); recovered by at least one action: 83.3%

| | STOP | CROP | OCR | HIGHRES |
|---|--:|--:|--:|--:|
| solves the sample | 0.0 | 66.7 | 33.3 | 50.0 |
| is the best action | 0.0 | 53.3 | 40.0 | 6.7 |
| solves it exclusively | 0.0 | 22.2 | 5.6 | 5.6 |

recovery overlap inside the subset: CROP+HIGHRES=4, CROP=4, none=3, CROP+HIGHRES+OCR=3, HIGHRES=1, HIGHRES+OCR=1, OCR=1, CROP+OCR=1

## Table 6 — label reliability

- per-action R1/R2 agreement: 93.2%
- top-1 family agreement (decisive): 75.4%
- parse failures: 0 / 7000 rows
