# Pre-registration: stronger deployable controllers — prospective evaluation

**Fixed before any evaluation state is collected.** Date: 2026-09-20.
Addendum to the 2026-08-30 freeze (reviewer-robustness round; see
`RESULTS_DIGEST_2026-08-30.md`, addendum 2026-09-20).

## Question

Is the opportunity–realization separation (oracle headroom that the frozen
ridge selector does not convert into a deployable advantage over the
state-independent mixture) an artifact of controller *capacity*? Either
outcome answers it: a nonlinear win relocates the bottleneck to head class;
a matching failure shows the separation is not specific to ridge.

## Frozen components (all committed before collection)

- **Controllers**: `models/strong_controllers_frozen.pkl` (sha256[:16]
  `5c60ccab3f7d08b8`), manifest `models/strong_controllers_frozen.json`.
  Three head classes on IDENTICAL inputs — the 16 F0 features of the
  frozen v2 selector (answer surface, question surface, STOP-confidence
  ECDF rank) — and identical targets (strict replay-mean score increment
  over STOP per family; CROP = uniform-grid expectation) and decision rule
  argmax_f [Δ̂_f − λ_B c_f], STOP = 0:
  * ridge (α=10), * gradient-boosted trees / xgboost (depth 3, 100 trees,
  lr .05), * 2-layer MLP (64-64, α=1e-3).
  Training pool: **1,350 TextVQA states** (Gate-0 200 + held-out 250 +
  prospective 600 + v3-prospective 300) — all previously analyzed sets;
  ridge is refit on the same pool so the capacity contrast is pool-matched.
  Per class one configuration from a disclosed grid (4/4/4) by 5-fold
  nested-CV policy value at ½× (tie ¼×), the v2 rule; λ(B) per class from
  the pooled out-of-fold predictions. Primary nonlinear controller fixed by
  the same rule: **xgb**.
- **Recorded pool-CV observations (before collection)**: at ½× ridge .732,
  xgb .730, mlp .719 (¼×: .708 / .703 / .699; 1×: .735 / .730 / .730) —
  nonlinear heads do NOT beat ridge in nested CV even with 3.4× the
  original training data; this is the prior for the prospective test.
- **Published reference**: frozen v2 (`selector_frozen_v2`, sha
  `d4e495063ac56b59`, dev-only n=396) evaluated unchanged; frozen dev-fitted
  mixture `models/mixture_frozen_textvqa.json` unchanged.
- **Evaluation set**: 400 fresh TextVQA states,
  `data/manifests/textvqa_strong.jsonl` (sha256[:16] `7312457434be55f3`),
  seed 20260920, excluding every sample_id in any existing log or TextVQA
  manifest (1,400 val ids), coarse-192 eligibility, outcome-blind.
- **Collection**: `scripts/collect_prospective.py --replays 1,2,3` — STOP
  (with logprobs), the 8 deterministic grid∪fallback CROPs, OCR (pre-pass
  cache), HIGHRES-512; greedy seed 1234, serial; replays 1–2 by the frozen
  `ppu/replay.py`, replay 3 by `ppu/replay_multi.py` ("In the image shown:
  {q}", the independent rewrite of PREREG_MULTI_REPLAY). Zero-transport-error
  + logprobs integrity gate; 5-state smoke first.

## Pre-declared analysis (single pass; `scripts/strong_controller_eval.py`)

Budgets B ∈ {¼, ½, 1} × mean always-HIGHRES tokens on the set; policies
evaluated on replays 1–2 exactly as PREREG_PROSPECTIVE_SELECTOR (selection
from pre-action information, mixture as per-state expectation).

- **Primary endpoint**: xgb − mixture at **½×, strict**, state-paired 10k
  bootstrap; success = CI excluding 0.
- **Co-secondary (capacity contrast)**: xgb − ridge(pool) and mlp −
  ridge(pool) at ½× strict, same bootstrap; effect sizes + CIs.
- Secondaries: all contrasts at ¼×/1×; lenient rescore of the same
  generations (frozen rule); frozen v2 and always-STOP/CROP/OCR/HIGHRES,
  random, best-fixed-at-B, Lagrangian oracle for context; realized token
  cost of every policy vs the nominal budget (budget-control check, as in
  PREREG_SELECTOR_V3); action mixes.
- **Replay-robustness extension** (`scripts/replay_pairwise_robustness.py`
  on this log, n=400): headroom R1→R2 / R1→R3 / R2→R3 (and reverses), STOP
  agreement, per-arm outcome correlation, PPU sign agreement — the
  100-state numbers of `paper/replay_pairwise_robustness.md` re-estimated
  at 4× the sample. No threshold; reported as protocol validation.

No changes after collection; failures reported as failures. Any hidden-state
(H) variant of these controllers is NOT part of this prereg (no GPU for the
extraction at freeze time) and, if later run, is a disclosed secondary.
