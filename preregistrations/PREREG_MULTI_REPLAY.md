# Pre-registration: multi-replay robustness of the cross-replay protocol

**Fixed before collection.** Date: 2026-08-30. Answers the reviewer question
"R1 and R2 are still semantically correlated — why are two replays enough?"
by extending the replay axis, not by changing anything already frozen.

## Design

- Reader: frozen Qwen2.5-VL-7B (primary), same serving config, greedy.
- States: **100 per dataset** from the Gate 0 draws (MME-RealWorld-Lite,
  TextVQA), chosen outcome-blind: sorted state IDs, one fixed shuffle
  (seed 20260830), first 100. No reference to any logged outcome or label.
- Arms: the deterministic set of the primary audit — STOP, grid CROPs
  (identical boxes, reconstructed from the Gate 0 log by action key), OCR,
  HIGHRES. Same pixel views (MME 256/1280, TextVQA 192/512).
- Replays: **3, 4, 5 newly collected** (`ppu/replay_multi.py`: MCQ = fresh
  deterministic choice permutations, seeds 13/29/43; open-ended = fixed
  semantics-preserving question wrappers in the style of the frozen replay-2
  fallback). Replays 1-2 are reused from the Gate 0 logs, not recollected —
  the frozen `ppu/replay.py` is untouched.

## Pre-declared estimands (deterministic arms, strict scores)

For K = 2..5, with replay 1 as the selection replay:

1. **Same-replay (naive) oracle** — select and score on the same replay,
   averaged over replays (the inflation baseline).
2. **1→1 cross-replay oracle** — the paper's current protocol (K = 2).
3. **Held-out-mean oracle** — select on R1, evaluate on mean(R2..RK).
4. **Symmetric leave-one-out oracle** — select on replay r, evaluate on the
   mean of the others, averaged over r.

Each is reported as oracle accuracy and as headroom vs best fixed policy
(cross-fitted inside the subset), with 10k state-paired bootstrap CIs.

## Interpretation criterion

Protocol validation, not effect hunting: the question is whether the K = 2
cross-replay estimate is stable as K grows. Pre-declared reading: if the
held-out-mean and symmetric estimates at K = 4-5 remain within the K = 2
bootstrap CI (and the naive-vs-cross gap persists), two replays are judged
sufficient; systematic drift with K is reported as a protocol limitation
either way. No threshold on the headroom itself is set.

No changes after collection: subset seed, replay definitions, arm set, and
estimands above are frozen before any replay-3+ inference.
