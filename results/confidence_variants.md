# Confidence-statistic variants (post-hoc, stored statistics only)

HELP-vs-rest ROC, strict scorer, AMBIGUOUS excluded. `mean_lp` is the
paper's confidence (whole-reply mean token logprob, averaged over both
STOP replays); `min_lp` is the whole-reply minimum token logprob. The
one-word-only columns restrict to states whose STOP answer is a single
word, where reply length cannot drive either statistic.

| backbone / regime | n (HELP) | mean-lp ROC [CI] | min-lp ROC [CI] | mean-lp ROC, one-word STOP only | min-lp ROC, one-word STOP only |
|---|--:|:--:|:--:|:--:|:--:|
| Qwen2.5-VL-7B / textvqa | 233 (35) | 0.796 [0.71, 0.87] | 0.786 [0.70, 0.86] | 0.740 (n=134, H=18) | 0.714 (n=134, H=18) |
| Qwen2.5-VL-7B / vqav2 | 202 (44) | 0.423 [0.33, 0.52] | 0.529 [0.44, 0.62] | 0.633 (n=120, H=14) | 0.694 (n=120, H=14) |
| Qwen2.5-VL-7B (one-word) / vqav2 | 236 (20) | 0.715 [0.60, 0.82] | 0.724 [0.61, 0.82] | 0.737 (n=229, H=18) | 0.729 (n=229, H=18) |
| Phi-4-MM-5.6B / textvqa | 223 (42) | 0.839 [0.78, 0.90] | 0.855 [0.79, 0.91] | 0.876 (n=165, H=31) | 0.905 (n=165, H=31) |
| Phi-4-MM-5.6B / vqav2 | 234 (10) | 0.818 [0.64, 0.96] | 0.829 [0.66, 0.95] | 0.834 (n=222, H=10) | 0.847 (n=222, H=10) |
| FRESH Qwen original / vqav2 | 179 (30) | 0.428 [0.32, 0.54] | 0.562 [0.46, 0.66] | 0.727 (n=107, H=5) | 0.651 (n=107, H=5) |
| FRESH Qwen one-word / vqav2 | 228 (14) | 0.897 [0.84, 0.95] | 0.872 [0.81, 0.93] | 0.901 (n=220, H=12) | 0.877 (n=220, H=12) |
| FRESH Qwen original / textvqa | 225 (29) | 0.786 [0.71, 0.86] | 0.847 [0.78, 0.90] | 0.789 (n=131, H=14) | 0.853 (n=131, H=14) |
| FRESH Qwen one-word / textvqa | 225 (18) | 0.815 [0.71, 0.90] | 0.771 [0.65, 0.87] | 0.812 (n=170, H=16) | 0.788 (n=170, H=16) |

Caveat: min_lp over the whole reply is length-sensitive in the opposite
direction from the mean (minima over more tokens are stochastically
smaller, and verbose replies are the flip-prone ones), so agreement
between the two statistics brackets the format sensitivity rather than
eliminating it. Answer-span confidence variants require a logprob
re-collection pass (scripts/collect_answer_span_conf.py).
