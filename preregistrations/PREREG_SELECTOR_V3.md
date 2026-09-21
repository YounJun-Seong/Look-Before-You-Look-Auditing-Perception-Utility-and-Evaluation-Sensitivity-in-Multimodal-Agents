# Pre-registration: hidden-state selector (v3) prospective evaluation

**Fixed before any v3-prospective state is collected.** Date: 2026-09-01.

## Question

Does a representation-based deployable selector — the reader's own pre-action
hidden state — close the opportunity-realization separation that the
handcrafted-feature selector (v2) left open? Either outcome answers the
question: a win relocates the bottleneck to controller representation; a
matching failure shows the separation is not an artifact of weak handcrafted
features.

## Frozen components

- **Selector v3**: `models/selector_frozen_v3.pkl` (sha256[:16]
  `4fd1fa4fc1793b7a`). Features = mean-pooled last-layer hidden state of the
  exact STOP-time input (system prompt + coarse view + replay-1 question;
  `scripts/extract_hidden_states.py`) — deployable information only.
  Disclosed 6-candidate grid ({H, H+F0} × {ridge_3, ridge_30, mlp64});
  winner H + ridge(α=30) by the same 5-fold dev CV rule as v2 (TextVQA ½×
  policy value). λ(B) frozen on the TextVQA dev subset.
- **Recorded dev observations (before collection)**: (i) every H-based
  candidate scored BELOW the v2 reference (F0 + ridge0.3, CV ½× .7202) —
  best H candidate .7052; (ii) H-based value heads over-shoot the token
  budget out-of-fold (CV ¼× realized ≈176 tok vs the 77-tok cap that bound
  in-fold) — budget-control instability is itself a pre-stated finding to
  check prospectively.
- **Baselines**: frozen dev-fitted mixture (`models/mixture_frozen_textvqa.json`)
  and frozen selector v2 (sha `d4e495063ac56b59`), both untouched.
- **Evaluation set**: 300 fresh TextVQA states,
  `data/manifests/textvqa_v3_prospective.jsonl` (sha256[:16]
  `7bc5ec8a8eb9a61f`), seed 20260901, all 1,850 previously used ids
  excluded, outcome-blind. Collection identical to the v2 prospective run
  (`scripts/collect_prospective.py`, zero-transport-error + logprobs gates).

## Pre-declared analysis (single pass; `scripts/v3_prospective_eval.py`)

- **Primary**: v3 − frozen mixture at the ½× budget, strict scorer,
  state-paired 10k bootstrap; success = CI excluding 0.
- Secondaries (effect sizes + CIs, no significance language): ¼×/1×; v3 − v2
  head-to-head; lenient rescore of all contrasts; realized token costs vs
  the nominal budgets (the pre-stated budget-control check); action mixes.

No changes after collection; failures reported as failures.
