# Pre-registration: prospective test of the history-conditioned two-step controller

**Fixed before any prospective state is collected.** Date: 2026-08-31.

## Question

Does a minimal history-conditioned continuation controller beat the strongest
state-independent sequential policy on fresh states — or does the
opportunity-without-realizability pattern repeat in the sequential setting?
Either outcome answers the question; neither is a failure of the study.

## Frozen components

- **Controller**: `models/seq_controller_frozen.pkl` (sha256[:16]
  `cffa1043f455f7d1`). Per a1 branch, two ridge Δ-heads on the two fixed
  post-a1 features [a1 answer ≠ STOP answer, len(a1 answer)];
  decision argmax(0, Δ̂_TR, Δ̂_HI). Trained on the 100-state sequential dev
  set; strict-trained heads are primary, lenient-trained heads a disclosed
  secondary variant. No other tuning.
- **Baseline**: the frozen state-independent sequential mixture at the
  controller's dev realized cost — which resolves to **always→CROP-TR** on
  both branches (weights in the artifact).
- **Evaluation set**: 300 fresh TextVQA states,
  `data/manifests/textvqa_seq_prospective.jsonl` (sha256[:16]
  `4196e1abbead1506`), seed 20260831, all 1,550 previously used ids excluded,
  outcome-blind.
- Collection (`scripts/collect_seq_prospective.py`): per state × replays
  {1,2}: one-step arms STOP / CROP-center / CROP-TR / OCR / HIGHRES and the
  four enumerated trajectories SEQ#0-3 (same boxes, views, replay module,
  greedy seed 1234; image limit 3; zero-transport-error gate).

## Pre-declared analysis (single pass)

Per a1 branch (center, ocr), same-replay decision+outcome, mean over replays:

- **Primary endpoint**: a1 = center, strict scorer:
  controller(strict-heads) − frozen mixture (= always→TR), state-paired 10k
  bootstrap; success = CI excluding 0. Dev suggested strict parity and a
  lenient advantage — the strict primary is deliberately the hard test.
- Secondaries (all reported with CIs, no significance language): a1 = ocr;
  lenient rescore of the same generations; lenient-trained head variant;
  controller − stop-after-a1; the full hierarchy (stop-after-a1, always→TR,
  always→HIGHRES, random at the controller's rate, frozen mixture,
  controller, cross-replay sequential oracle); realized token costs.
- Dev-set expectations recorded for calibration: strict center −1.7pp vs
  mixture (ns), lenient center +5.1pp [+2.1, +8.3].

No changes after collection; failures reported as failures.
