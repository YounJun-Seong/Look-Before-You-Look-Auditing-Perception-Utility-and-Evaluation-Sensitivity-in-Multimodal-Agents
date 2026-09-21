# Pre-registration — fresh prospective lockbox for the answer-format confound (2026-08-23)

Written BEFORE any rollout on these states. Purpose: replicate, prospectively and
on never-used states, the finding that Qwen2.5-VL-7B's VQAv2 "confidence
reversal" is an answer-format / strict-scoring artefact.

## States
- `data/manifests/vqav2_fresh.jsonl`: 250 VQAv2 train states, strata quota
  A=100 / B=100 / C=50 (same rule as a1/confirm/lockbox), seed 20260823,
  excluding every VQAv2 id used anywhere in this project (780 ids).
- `data/manifests/textvqa_fresh.jsonl`: 250 TextVQA train states, seed 20260823,
  excluding the A1 draw and the sealed lockbox (500 ids). Gate 0 used the val split.
- SHA-1 of both manifests recorded in `logs/fresh_lockbox.log` before collection.

## Arms (Qwen2.5-VL-7B, frozen, greedy, serial, 2 replays, logprobs on)
| arm | regime | views | prompt |
|---|---|---|---|
| F1 | VQAv2 | 64→192 | p0.1 original ("short phrase") |
| F2 | VQAv2 | 64→192 | p0.2-oneword |
| F3 | TextVQA | 192→512 | p0.1 original |
| F4 | TextVQA | 192→512 | p0.2-oneword |

## Pre-specified analysis (scripts/format_confound_check.py, scripts/backbone_check.py, unchanged)
Labels: replicate-consistent 4-way; confidence = −mean token log-prob of STOP reply, R1/R2 averaged;
metrics: ROC-AUC, lift@1×, strict vs lenient rescoring, format-flip share, verbose share.

## Predictions (falsifiable)
P1. F1 strict: confidence ROC < 0.55 and verbose STOP share > 25%; ≥ 50% of HELP states are format flips.
P2. F1 lenient rescoring: ROC rises by ≥ 0.15 over F1 strict.
P3. F2 strict: verbose share < 10%, STOP accuracy ≥ F1 strict STOP + 20pp, confidence ROC > 0.60 with 95% CI lower bound > 0.5.
P4. F3 and F4 strict: confidence ROC > 0.70 (TextVQA signal is not format-driven); F4 ≈ F3 within 0.10.
P5. Under lenient scoring, F1 and F2 ROC differ by < 0.15 (scorer and prompt are two routes to the same correction).

Outcome: if P1–P4 hold, the confound replicates prospectively. If P3 fails (one-word ROC ≤ 0.6),
the one-word result of 2026-08-22 was not robust and the paper must say so.
