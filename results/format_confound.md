# Answer-format confound: strict vs lenient scoring of the same generations

strict = official vqa_acc as logged (reproduced with 0 mismatches); lenient = strict OR gold is the first token / a whole-word substring of the normalised prediction. Labels, confidence ROC and the frozen v2.2 gate are recomputed on the rescored outcomes; nothing else changes.

| backbone / regime | scorer | STOP | HIGHRES | verbose STOP | HELP (prev.) | HELP = format flip | conf ROC [CI] | conf lift | v2.2 ROC | v2.2 lift |
|---|---|--:|--:|--:|--:|--:|:--:|--:|--:|--:|
| Qwen2.5-VL-7B / textvqa | strict | 65.0% | 77.9% | 43% | 35 (15.0%) | 23% | 0.796 [0.71, 0.88] | 3.23× | 0.669 | 1.52× |
| Qwen2.5-VL-7B / textvqa | lenient | 74.3% | 85.0% | 43% | 33 (13.6%) | 12% | 0.789 [0.69, 0.87] | 3.11× | 0.661 | 1.78× |
| Qwen2.5-VL-7B / vqav2 | strict | 42.9% | 65.3% | 40% | 44 (21.8%) | 64% | 0.423 [0.33, 0.52] | 0.73× | 0.703 | 1.77× |
| Qwen2.5-VL-7B / vqav2 | lenient | 75.9% | 81.0% | 40% | 21 (9.6%) | 29% | 0.655 [0.56, 0.75] | 0.50× | 0.617 | 1.49× |
| Phi-4-MM-5.6B / textvqa | strict | 66.4% | 84.7% | 27% | 42 (18.8%) | 12% | 0.839 [0.78, 0.90] | 2.78× | — | — |
| Phi-4-MM-5.6B / textvqa | lenient | 66.7% | 84.6% | 27% | 42 (18.5%) | 10% | 0.844 [0.78, 0.90] | 2.96× | — | — |
| Phi-4-MM-5.6B / vqav2 | strict | 76.1% | 77.3% | 5% | 10 (4.3%) | 0% | 0.818 [0.64, 0.96] | 7.02× | — | — |
| Phi-4-MM-5.6B / vqav2 | lenient | 76.7% | 77.4% | 5% | 10 (4.3%) | 0% | 0.818 [0.64, 0.95] | 6.99× | — | — |
| Phi-4-MM-5.6B (coarse 16) / vqav2 | strict | 71.8% | 78.1% | 4% | 21 (9.1%) | 5% | 0.814 [0.72, 0.90] | 4.21× | — | — |
| Phi-4-MM-5.6B (coarse 16) / vqav2 | lenient | 71.9% | 78.4% | 4% | 21 (9.1%) | 5% | 0.814 [0.72, 0.89] | 4.21× | — | — |
| Qwen2.5-VL-7B (one-word prompt) / vqav2 | strict | 78.1% | 83.7% | 3% | 20 (8.5%) | 5% | 0.715 [0.60, 0.82] | 2.36× | 0.607 | 0.59× |
| Qwen2.5-VL-7B (one-word prompt) / vqav2 | lenient | 78.5% | 83.9% | 3% | 19 (8.1%) | 0% | 0.732 [0.62, 0.84] | 2.61× | 0.588 | 0.65× |
| FRESH Qwen original / vqav2 | strict | 44.8% | 60.7% | 40% | 30 (16.8%) | 77% | 0.428 [0.32, 0.54] | 0.80× | — | — |
| FRESH Qwen original / vqav2 | lenient | 75.8% | 76.3% | 40% | 19 (8.6%) | 47% | 0.526 [0.38, 0.66] | 1.22× | — | — |
| FRESH Qwen one-word / vqav2 | strict | 78.0% | 80.7% | 3% | 14 (6.1%) | 7% | 0.897 [0.84, 0.95] | 4.65× | — | — |
| FRESH Qwen one-word / vqav2 | lenient | 77.8% | 80.7% | 3% | 15 (6.6%) | 7% | 0.906 [0.85, 0.95] | 5.07× | — | — |
| FRESH Qwen original / textvqa | strict | 67.5% | 77.7% | 43% | 29 (12.9%) | 21% | 0.786 [0.71, 0.86] | 3.21× | — | — |
| FRESH Qwen original / textvqa | lenient | 79.5% | 86.8% | 43% | 26 (11.5%) | 8% | 0.827 [0.75, 0.90] | 4.35× | — | — |
| FRESH Qwen one-word / textvqa | strict | 76.4% | 82.4% | 26% | 18 (8.0%) | 6% | 0.815 [0.72, 0.90] | 2.78× | — | — |
| FRESH Qwen one-word / textvqa | lenient | 78.2% | 83.6% | 26% | 19 (8.3%) | 5% | 0.822 [0.73, 0.90] | 3.17× | — | — |

- Qwen2.5-VL-7B / textvqa, strict scorer, one-word STOP answers only (n=134, HELP=18): confidence ROC **0.740**
- Qwen2.5-VL-7B / vqav2, strict scorer, one-word STOP answers only (n=120, HELP=14): confidence ROC **0.633**
- Qwen2.5-VL-7B (one-word prompt) / vqav2, strict scorer, one-word STOP answers only (n=229, HELP=18): confidence ROC **0.737**