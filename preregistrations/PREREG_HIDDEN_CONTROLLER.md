# Pre-registration: hidden-state × nonlinear controllers — prospective evaluation

**Fixed before any evaluation state is collected.** Date: 2026-09-20 (design);
the frozen-artifact sha is appended by the pipeline after training and
before collection (see freeze record at the end). Addendum to the 2026-08-30
freeze; advisor's priority-1 item of 2026-09-20.

## Question

The paper has F0-surface features × {ridge, xgb, mlp} (PREREG_STRONG_CONTROLLER,
all ≈ ridge, none beats the mixture under strict scoring) and hidden state
H × ridge (PREREG_SELECTOR_V3, fails, λ-uncontrollable). The open cell is
**strong representation × strong head**. Does H × {xgb, mlp} close the
opportunity–realization separation? Either outcome answers it.

## Frozen components

- **Controllers**: `models/hidden_controllers_frozen.{pkl,json}` — full
  factorial {F0, H, H+F0} × {ridge, xgb, mlp} (9 cells), IDENTICAL targets
  (strict replay-mean Δ over STOP per family; CROP = uniform-grid
  expectation) and decision rule argmax_f [Δ̂_f − λ_B c_f], STOP = 0.
  H = mean-pooled last-layer hidden state (3584-d) of the exact STOP-time
  input (`scripts/extract_hidden_states.py`; deployable information only).
  Training pool: **1,750 TextVQA states** (Gate-0 200 + held-out 250 +
  prospective 600 + v3 300 + strong 400 — every previously analyzed set;
  every cell is trained on the same pool so representation and capacity
  contrasts are pool-matched). Per cell one configuration from a disclosed
  grid (ridge 4 α; xgb 4; mlp 4) by 5-fold nested-CV policy value at ½×
  (tie ¼×); λ(B) per cell from pooled out-of-fold predictions.
  **Primary controller = better of H/xgb, H/mlp by the same CV rule**,
  recorded in the manifest before collection.
- **Baselines**: frozen dev-fitted mixture `models/mixture_frozen_textvqa.json`
  (unchanged); frozen published v2 for context.
- **Evaluation set**: 400 fresh TextVQA states,
  `data/manifests/textvqa_hidden.jsonl` (sha256[:16] `af41a4ac740a9ecb`),
  seed 20260921, excluding every sample_id in any existing log or TextVQA
  manifest (1,800 val ids incl. the strong-controller set), coarse-192
  eligibility, outcome-blind.
- **Collection**: `scripts/collect_prospective.py --replays 1,2` (frozen
  protocol: STOP with logprobs, 8 grid∪fallback CROPs, OCR pre-pass cache,
  HIGHRES-512, greedy seed 1234, serial); zero-transport-error + logprobs
  integrity gate; 5-state smoke first. Hidden states of the fresh set are
  extracted AFTER collection from the same STOP-time inputs (no outcome
  enters the extraction).

## Pre-declared analysis (single pass; `scripts/hidden_controller_eval.py`)

Budgets B ∈ {¼, ½, 1} × mean always-HIGHRES tokens on the set.

- **Primary endpoint**: primary controller (H/xgb or H/mlp) − mixture at
  **½×, strict**, state-paired 10k bootstrap; success = CI excluding 0.
- **Co-secondary (representation and capacity contrasts, ½× strict)**:
  H/xgb − F0/xgb; H/xgb − H/ridge; H/mlp − H/ridge; H+F0/xgb − F0/xgb;
  H/ridge − F0/ridge; effect sizes + CIs.
- Secondaries: every cell vs mixture at ¼×/½×/1×; lenient rescore of all
  contrasts (frozen rule); realized token cost per cell vs nominal budget
  (the λ-controllability check that failed for H/ridge in v3); action
  mixes; Lagrangian oracle for context.
- Pre-collection nested-CV table of all 36 candidates is disclosed in the
  frozen manifest and is the prior for the prospective test.

No changes after collection; failures reported as failures.

> **Freeze record (appended by the pipeline before collection, 2026-09-20T15:59:37+09:00):** artifact sha256[:16] `ed93f19437264e56`, primary cell `H/mlp`. No fresh-set outcome existed at this time.
