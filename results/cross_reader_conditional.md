# Cross-reader opportunity agreement, conditional on baseline capability

Qwen2.5-VL-7B vs Phi-4-multimodal on the identical Gate-0 states and views.
help = some deterministic additional arm strictly beats STOP (mean over both
replays); PPU = best-arm gain over STOP. Strata by whether each reader's STOP
(mean over replays) is correct (>= 0.5). 10k state-paired bootstrap CIs.

## mme — strict scorer  (n=196; STOP acc Qwen 0.352, Phi-4 0.247)

| stratum | n | P(help) Q / P4 | Both | Q only | P4 only | Neither | Jaccard [CI] | κ [CI] | corr(PPU) Pearson [CI] | Spearman | best-family agree (n both) |
|---|--:|:--|--:|--:|--:|--:|:--|:--|:--|--:|:--|
| pooled | 196 | 0.36 / 0.35 | 35 | 36 | 34 | 91 | 0.33 [0.24, 0.42] | 0.22 [0.08, 0.36] | 0.21 [0.06, 0.36] | 0.22 | 0.77 (35) |
| both_stop_wrong | 92 | 0.45 / 0.27 | 19 | 22 | 6 | 45 | 0.40 [0.26, 0.55] | 0.36 [0.17, 0.54] | 0.42 [0.22, 0.59] | 0.40 | 0.74 (19) |
| both_stop_correct | 45 | 0.24 / 0.24 | 5 | 6 | 6 | 28 | 0.29 [0.08, 0.53] | 0.28 [-0.05, 0.58] | 0.28 [-0.05, 0.59] | 0.28 | 0.80 (5) |
| mixed | 59 | 0.32 / 0.56 | 11 | 8 | 22 | 18 | 0.27 [0.14, 0.41] | 0.02 [-0.20, 0.25] | -0.06 [-0.28, 0.18] | 0.00 | 0.82 (11) |
| qwen_correct_phi_wrong | 37 | 0.24 / 0.76 | 8 | 1 | 20 | 8 | 0.28 [0.12, 0.44] | 0.10 [-0.07, 0.28] | 0.26 [-0.05, 0.53] | 0.27 | 0.75 (8) |
| qwen_wrong_phi_correct | 22 | 0.45 / 0.23 | 3 | 7 | 2 | 10 | 0.25 [0.00, 0.54] | 0.14 [-0.22, 0.52] | 0.20 [-0.21, 0.55] | 0.20 | 1.00 (3) |

Per-family PPU correlation (Pearson, family-mean gain over STOP):

| stratum | CROP | OCR | HIGHRES |
|---|--:|--:|--:|
| pooled | 0.15 | 0.17 | 0.09 |
| both_stop_wrong | 0.50 | 0.36 | 0.21 |
| both_stop_correct | 0.10 | -0.11 | 0.06 |
| mixed | -0.05 | 0.21 | 0.07 |

## textvqa — strict scorer  (n=200; STOP acc Qwen 0.623, Phi-4 0.243)

| stratum | n | P(help) Q / P4 | Both | Q only | P4 only | Neither | Jaccard [CI] | κ [CI] | corr(PPU) Pearson [CI] | Spearman | best-family agree (n both) |
|---|--:|:--|--:|--:|--:|--:|:--|:--|:--|--:|:--|
| pooled | 200 | 0.33 / 0.36 | 22 | 44 | 51 | 83 | 0.19 [0.12, 0.26] | -0.05 [-0.18, 0.09] | 0.03 [-0.12, 0.19] | -0.02 | 0.73 (22) |
| both_stop_wrong | 70 | 0.73 / 0.33 | 19 | 32 | 4 | 15 | 0.35 [0.23, 0.47] | 0.11 [-0.05, 0.28] | 0.31 [0.14, 0.47] | 0.20 | 0.84 (19) |
| both_stop_correct | 53 | 0.08 / 0.19 | 1 | 3 | 9 | 40 | 0.08 [0.00, 0.25] | 0.04 [-0.15, 0.33] | -0.01 [-0.16, 0.25] | 0.01 | 0.00 (1) |
| mixed | 77 | 0.14 / 0.52 | 2 | 9 | 38 | 28 | 0.04 [0.00, 0.10] | -0.19 [-0.34, -0.04] | -0.29 [-0.40, -0.17] | -0.31 | 0.00 (2) |
| qwen_correct_phi_wrong | 75 | 0.12 / 0.53 | 2 | 7 | 38 | 28 | 0.04 [0.00, 0.11] | -0.14 [-0.29, -0.00] | -0.26 [-0.38, -0.12] | -0.27 | 0.00 (2) |
| qwen_wrong_phi_correct | 2 | 1.00 / 0.00 | 0 | 2 | 0 | 0 | 0.00 [—, —] | 0.00 [—, —] | — [—, —] | — | — (0) |

Per-family PPU correlation (Pearson, family-mean gain over STOP):

| stratum | CROP | OCR | HIGHRES |
|---|--:|--:|--:|
| pooled | -0.01 | 0.06 | 0.04 |
| both_stop_wrong | 0.11 | 0.47 | 0.09 |
| both_stop_correct | -0.25 | 0.05 | 0.28 |
| mixed | -0.12 | 0.01 | -0.02 |

## textvqa — lenient scorer  (n=200; STOP acc Qwen 0.726, Phi-4 0.501)

| stratum | n | P(help) Q / P4 | Both | Q only | P4 only | Neither | Jaccard [CI] | κ [CI] | corr(PPU) Pearson [CI] | Spearman | best-family agree (n both) |
|---|--:|:--|--:|--:|--:|--:|:--|:--|:--|--:|:--|
| pooled | 200 | 0.26 / 0.45 | 36 | 16 | 53 | 95 | 0.34 [0.25, 0.44] | 0.27 [0.14, 0.40] | 0.35 [0.19, 0.49] | 0.31 | 0.69 (36) |
| both_stop_wrong | 51 | 0.82 / 0.69 | 30 | 12 | 5 | 4 | 0.64 [0.50, 0.77] | 0.12 [-0.14, 0.39] | 0.48 [0.25, 0.69] | 0.44 | 0.77 (30) |
| both_stop_correct | 105 | 0.04 / 0.17 | 3 | 1 | 15 | 86 | 0.16 [0.00, 0.33] | 0.22 [-0.02, 0.46] | 0.18 [-0.04, 0.44] | 0.27 | 0.67 (3) |
| mixed | 44 | 0.14 / 0.82 | 3 | 3 | 33 | 5 | 0.08 [0.00, 0.17] | -0.12 [-0.30, 0.02] | -0.40 [-0.64, -0.02] | -0.39 | 0.00 (3) |
| qwen_correct_phi_wrong | 42 | 0.10 / 0.86 | 3 | 1 | 33 | 5 | 0.08 [0.00, 0.18] | -0.03 [-0.14, 0.05] | -0.15 [-0.45, 0.12] | -0.24 | 0.00 (3) |
| qwen_wrong_phi_correct | 2 | 1.00 / 0.00 | 0 | 2 | 0 | 0 | 0.00 [—, —] | 0.00 [—, —] | — [—, —] | — | — (0) |

Per-family PPU correlation (Pearson, family-mean gain over STOP):

| stratum | CROP | OCR | HIGHRES |
|---|--:|--:|--:|
| pooled | 0.14 | 0.18 | 0.08 |
| both_stop_wrong | 0.25 | 0.52 | 0.23 |
| both_stop_correct | 0.05 | 0.00 | 0.02 |
| mixed | -0.29 | 0.08 | -0.22 |
