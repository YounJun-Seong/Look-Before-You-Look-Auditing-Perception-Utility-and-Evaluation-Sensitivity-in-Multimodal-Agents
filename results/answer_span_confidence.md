# Answer-span confidence (post-hoc re-collection)

HELP-vs-rest ROC, strict labels from the original logs (AMBIGUOUS excluded),
confidence statistics recomputed from per-token logprobs of regenerated STOP
replies (greedy, seed 1234, serial; `match` = share of regenerations that
byte-match the logged text). Span = text after the terminal `Answer:` marker
(whole reply when absent).

| run | n (HELP) | match | reply-mean ROC [CI] | span-mean ROC [CI] | span-min ROC [CI] | first-token ROC [CI] |
|---|--:|--:|:--:|:--:|:--:|:--:|
| Qwen / vqav2 (lockbox) | 202 (44) | 100% | 0.423 [0.33, 0.52] | 0.453 [0.36, 0.55] | 0.578 [0.49, 0.67] | 0.576 [0.49, 0.66] |
| Qwen / textvqa (lockbox) | 233 (35) | 100% | 0.796 [0.70, 0.87] | 0.775 [0.68, 0.86] | 0.768 [0.67, 0.86] | 0.696 [0.59, 0.80] |
| Qwen one-word / vqav2 (lockbox) | 236 (20) | 100% | 0.715 [0.60, 0.81] | 0.716 [0.60, 0.83] | 0.735 [0.63, 0.83] | 0.743 [0.63, 0.84] |
| FRESH Qwen / vqav2 | 179 (30) | 100% | 0.428 [0.32, 0.54] | 0.457 [0.35, 0.56] | 0.617 [0.51, 0.72] | 0.579 [0.46, 0.69] |
| FRESH Qwen one-word / vqav2 | 228 (14) | 100% | 0.897 [0.84, 0.94] | 0.905 [0.85, 0.95] | 0.890 [0.82, 0.94] | 0.898 [0.84, 0.95] |
| FRESH Qwen / textvqa | 225 (29) | 100% | 0.786 [0.71, 0.86] | 0.740 [0.63, 0.84] | 0.805 [0.70, 0.89] | 0.675 [0.56, 0.78] |
| FRESH Qwen one-word / textvqa | 225 (18) | 100% | 0.815 [0.72, 0.90] | 0.823 [0.73, 0.90] | 0.786 [0.67, 0.88] | 0.778 [0.68, 0.87] |

One-word-STOP-only ROCs are in paper/answer_span_confidence.json (`roc_oneword_only`).
