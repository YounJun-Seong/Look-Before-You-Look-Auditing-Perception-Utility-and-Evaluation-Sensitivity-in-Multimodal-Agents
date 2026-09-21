# Pre-registration: prospective selector-vs-mixture evaluation (TextVQA)

**Fixed before any prospective state is collected.** Date: 2026-08-31.

## Question

Does a budget-aware deployable multi-action selector outperform the strongest
deployable state-independent baseline — the dev-fitted randomized fixed
mixture — on fresh, never-touched states, at matched visual-token budgets?

## Frozen components (all committed before collection)

- **Selector**: `models/selector_frozen_v2.pkl` (sha256[:16]
  `d4e495063ac56b59`). Trained on the 396 Gate 0 dev states only. The full
  10-candidate grid (2 feature sets × 5 head classes) and the selection rule
  (5-fold dev CV policy value, TextVQA ½× budget, tie-break ¼×) are disclosed
  in `models/selector_frozen_v2.json`; winner: F0 features + ridge(α=0.3).
  Per-budget λ is chosen on dev inside the frozen artifact's λ grid.
- **Mixture**: `models/mixture_frozen_textvqa.json` — dev-fitted randomized
  fixed mixture (Gate 0 `tv_200` arm means, arms matched by fixed crop box):
  ¼× = 0.876·CROP-center + 0.124·OCR; ½× = 0.569·CROP-center +
  0.431·CROP-top-right; 1× = 1.0·CROP-top-right.
- **Evaluation set**: 600 TextVQA states,
  `data/manifests/textvqa_prospective.jsonl` (sha256[:16] `a535d8edce3aa822`),
  drawn with seed 20260831 from the val pool EXCLUDING all 950 previously
  used ids (Gate 0, fresh, lockbox, multi-replay, Phi-4, smokes); standard
  coarse-192 eligibility rule; no outcome of any kind consulted.
- Collection: `scripts/collect_prospective.py` — STOP (with logprobs),
  deterministic grid∪fallback CROPs, OCR (pre-pass cache), HIGHRES-512, both
  replays, greedy seed 1234. Zero-transport-error integrity gate; 5-state
  smoke with STOP-parse check before the full run.

## Pre-declared analysis (single pass)

Budgets B ∈ {¼, ½, 1} × mean always-HIGHRES tokens on the prospective set.

- **Primary endpoint**: selector − mixture accuracy at **½× budget, strict
  scorer**, state-paired 10k bootstrap; success = CI excluding 0.
- Secondary: same contrast at ¼× and 1×; lenient rescore of all contrasts
  (same generations, frozen lenient rule of `scripts/p4_lenient_rescore.py`);
  Random / best-fixed / confidence-threshold baselines and the
  budget-constrained oracle (Lagrangian + exact MCKP DP) for context.
- Selector actions are chosen from pre-action information only (frozen
  features + frozen λ(B)); the mixture samples arms per its frozen weights —
  evaluated as the per-state expectation over its weights.
- Failures are reported as failures; effect sizes + CIs for everything;
  "statistically significant" language reserved for the primary endpoint.

No changes after collection: selector, mixture weights, state ids, budgets,
endpoints, and scorers above are frozen as of this document's date.
