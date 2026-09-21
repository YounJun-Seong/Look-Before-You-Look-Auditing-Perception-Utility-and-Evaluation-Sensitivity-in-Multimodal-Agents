# Second backbone: format intervention on fresh VQAv2 (post-hoc)

Same 250 fresh VQAv2 states, views, decoding, and strict scorer as the fresh
Qwen runs; LLaVA-1.6-Mistral-7B served with the same pixel budgets
(Qwen-token-equivalent). Confidence = whole-reply mean token logprob of STOP
(mean over replays).

| backbone / prompt | n | STOP | HIGHRES | verbose STOP | yes/no on wh-q | HELP (prev.) | HELP = flip | conf ROC [CI] |
|---|--:|--:|--:|--:|--:|--:|--:|:--:|
| Qwen2.5-VL-7B / original | 250 | 44.8% | 60.7% | 40% | 1% | 30 (16.8%) | 77% | 0.428 [0.32, 0.54] |
| Qwen2.5-VL-7B / one-word | 250 | 78.0% | 80.7% | 3% | 6% | 14 (6.1%) | 7% | 0.897 [0.84, 0.95] |
| LLaVA-1.6-Mistral-7B / original | 250 | 82.7% | 84.7% | 6% | 2% | 6 (2.4%) | 0% | 0.777 [0.55, 0.95] |
| LLaVA-1.6-Mistral-7B / one-word | 250 | 65.5% | 64.6% | 2% | 38% | 4 (1.7%) | 25% | 0.752 [0.33, 0.98] |
| LLaVA (coarse 16) / original | 250 | 78.0% | 84.7% | 6% | 2% | 19 (7.9%) | 11% | 0.767 [0.63, 0.88] |
| LLaVA (coarse 16) / one-word | 250 | 62.8% | 64.6% | 1% | 40% | 12 (5.3%) | 0% | 0.715 [0.55, 0.87] |

## Reading

- **Qwen's pathology does not replicate on LLaVA.** Original-prompt LLaVA is
  not verbose (6% vs 40%), has no format-flip HELP states, and its confidence
  ROC is on the right side of chance — including at coarse 16, where a real
  perceptual bottleneck exists (STOP 78.0% vs HIGHRES 84.7%, HELP 7.9%).
  The below-chance regime is therefore not a property of the benchmark or
  the budget alone; it is reader x prompt dependent.
- **The repair is reader-dependent too.** The one-word prompt that fixes
  Qwen (ROC .428 -> .897) *breaks* LLaVA: it answers yes/no on 38-40% of
  wh-questions (vs 2% under the original prompt) and STOP accuracy drops
  17pp. The LLaVA one-word rows are a compliance failure of the
  intervention, not a valid utility measurement — report them as such.
- Claim to draw: there is no universal prompt fix; utility validity must be
  audited per reader/prompt configuration. Matches the Discussion's
  prevalence caveat.

Provenance: post-hoc; fresh VQAv2 set only (no lockbox contact); LLaVA runs
collected 2026-08-28 via scripts/run_llava_format_check.sh and
scripts/run_llava_c16.sh (5-state smoke gates passed; serial, greedy,
seed 1234; logs/serve_llava*.log).
