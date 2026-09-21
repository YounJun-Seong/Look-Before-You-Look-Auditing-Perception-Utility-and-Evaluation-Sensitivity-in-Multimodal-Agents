# RQ1 re-check: policy headroom under format-tolerant scoring and symmetric cross-replay

TextVQA Gate 0 rollouts (`logs/tv_200.jsonl`, n=200, 24 rollouts/state, Qwen2.5-VL-7B, original prompt), re-audited with the frozen `ppu/audit.py`. strict = official vqa_acc as logged (reproduced, 0 mismatches); lenient = gold is the whole normalised answer, its first token, or a whole-word substring. Symmetric = per-state mean of (choose R1 → score R2) and (choose R2 → score R1) for both the oracle and the 2-fold cross-fitted best fixed policy; CI = paired bootstrap (10k).

| scorer | direction | coarse STOP | FULL | best fixed (cf) | oracle (cf) | dummy oracle | H_net [CI] | **P_policy [CI]** | DMG |
|---|---|--:|--:|--:|--:|--:|:--:|:--:|--:|
| strict | R1->R2 | 60.0 | 75.5 | 69.1 | 78.8 | 83.0 | -2.7 [-9.3, +4.0] | **+9.8** [+4.7, +14.8] | 13.0 |
| strict | R2->R1 | 64.7 | 75.0 | 70.1 | 83.8 | 85.0 | -1.3 [-6.3, +3.3] | **+13.8** [+9.7, +18.0] | 13.0 |
| strict | **symmetric** | | | 69.6 | 81.3 | | | **+11.8** [+8.0, +15.8] | |
| lenient | R1->R2 | 72.0 | 86.3 | 80.1 | 88.8 | 89.7 | -4.0 [-9.3, +1.3] | **+8.8** [+5.1, +12.7] | 11.5 |
| lenient | R2->R1 | 73.2 | 85.5 | 79.2 | 89.7 | 89.7 | -1.0 [-5.3, +3.0] | **+10.4** [+7.0, +14.2] | 11.5 |
| lenient | **symmetric** | | | 79.7 | 89.2 | | | **+9.6** [+6.5, +13.0] | |

Verbose (multi-word) coarse STOP answers in this log: 38%.

## Decisive-state best-action distribution (strict labels, R1→R2)

| scorer | n decisive | STOP | CROP | OCR | HIGHRES | exclusive OCR |
|---|--:|--:|--:|--:|--:|--:|
| strict | 81 | 32.1 | 45.7 | 19.8 | 2.5 | 0.5 |
| lenient | 67 | 34.3 | 43.3 | 20.9 | 1.5 | 0.5 |

## R2 paraphrase procedure (ppu/replay.py)

Deterministic, rule-based: 12 regex rewrites applied in order, first match wins; otherwise the fallback prefix. No LLM, no sampling. Rule usage on these 200 questions:

- `^what (.+)\?$`: 87
- `^what is (.+)\?$`: 57
- `fallback: 'Based on the image: …'`: 15
- `^who (.+)\?$`: 11
- `^how (.+)\?$`: 8
- `^is (.+)\?$`: 6
- `^which (.+)\?$`: 5
- `^what are (.+)\?$`: 3
- `^where (.+)\?$`: 3
- `^does (.+)\?$`: 3
- `^when (.+)\?$`: 2

Examples:

- what is the brand of this camera? → Looking at the image, what is the brand of this camera?
- how long has the drink on the right been aged? → Based on the image, how long has the drink on the right been aged?
- what is one of the brands being advertised? → Looking at the image, what is one of the brands being advertised?
- what is the company name to the left of the coors logo? → Looking at the image, what is the company name to the left of the coors logo?
- what playing card is shown? → Based on the image, what playing card is shown?
## MME-RealWorld-Lite (MCQ; strict only, lenient not applicable) — symmetric cross-replay

See `paper/gate0_mme_symmetric.json` (n=196, `logs/g0_mme200.jsonl`). R2 = fixed permutation of the answer options.

| direction | best fixed (cf) | oracle (cf) | H_net [CI] | **P_policy [CI]** |
|---|--:|--:|:--:|:--:|
| R1→R2 (reported) | 40.2 | 50.0 | +12.8 [+7.7, +17.9] | **+9.8** [+5.1, +14.7] |
| R2→R1 | 40.5 | 49.0 | +4.1 [−3.1, +11.2] | **+8.5** [+3.1, +14.0] |
| **symmetric** | | | | **+9.1** [+4.8, +13.6] |

Note: H_net (oracle net of the luck floor) is direction-sensitive on MME because the dummy
oracle also changes direction; P_policy (oracle − cross-fitted best fixed) is stable in both
regimes and in both directions.
