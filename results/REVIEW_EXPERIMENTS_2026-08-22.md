# Review-response experiments (2026-08-22)

Three additions requested after review, plus one unplanned finding that came out
of the first. All numbers are single-pass on the sealed lockboxes; v2.2 is
unchanged. Source tables: `backbone_check.md`, `format_confound.md`,
`feature_baselines_lockbox.md`, `full_population_eval.md`. Scripts under
`scripts/` with the same names. Phi-4 logs: `logs/lockbox_{textvqa,vqav2}_phi4.jsonl`,
`logs/lockbox_vqav2_phi4_c16.jsonl`.

## 1. Second backbone — Phi-4-multimodal-instruct (5.6B, SigLIP encoder)

Same lockbox state ids, same pixel views, labels regenerated from Phi-4's own
STOP/HIGHRES rollouts (2 replays, serial, logprobs on). Confidence only; no gate.

| backbone | regime | coarse→HIGHRES | STOP | HIGHRES | HELP prev. | conf ROC [95% CI] | lift@1× [95% CI] |
|---|---|---|--:|--:|--:|:--:|:--:|
| Qwen2.5-VL-7B | TextVQA | 192→512 | 65.0 | 77.9 | 15.0% | **0.796** [0.71, 0.87] | **3.23×** [2.15, 4.08] |
| Qwen2.5-VL-7B | VQAv2 | 64→192 | 42.9 | 65.3 | 21.8% | **0.423** [0.33, 0.52] | **0.73×** [0.28, 1.17] |
| Phi-4-MM | TextVQA | 192→512 | 66.4 | 84.7 | 18.8% | **0.839** [0.78, 0.90] | **2.78×** [2.15, 3.54] |
| Phi-4-MM | VQAv2 | 64→192 | 76.1 | 77.3 | 4.3% | 0.818 [0.64, 0.96] | 7.02× [1.80, 14.62] |
| Phi-4-MM | VQAv2 | **16**→192 | 71.8 | 78.1 | 9.1% | **0.814** [0.72, 0.89] | **4.21×** [1.92, 6.63] |

- TextVQA replicates cleanly: confidence is a strong perceptual-need signal on a
  second family (0.839 vs 0.796).
- **VQAv2 does NOT replicate the reversal.** Phi-4 confidence is well above
  chance (0.81) at both coarse budgets. The 64-tok run was near-saturated
  (STOP≈HIGHRES, 10 HELP); the 16-tok rerun restores headroom (+6.3pp, 21 HELP)
  and gives the same answer.

## 2. Why — the answer-format confound (unplanned, decisive)

Phi-4 answers VQAv2 in one word 95% of the time; Qwen answers in ≥2 words 40%
of the time ("No, it's not cold."). Official `vqa_acc` is exact match after
normalisation, so those score 0. Rescoring the *same logged generations*
leniently (gold = first token or whole-word substring):

| | scorer | STOP | HIGHRES | HELP (prev.) | HELP that are format flips | conf ROC | v2.2 ROC / lift |
|---|---|--:|--:|--:|--:|--:|--:|
| Qwen VQAv2 | strict | 42.9 | 65.3 | 44 (21.8%) | **64%** | **0.423** | 0.703 / 1.77× |
| Qwen VQAv2 | lenient | 75.9 | 81.0 | 21 (9.6%) | 29% | **0.655** [0.56, 0.75] | 0.617 / 1.49× |
| Qwen TextVQA | strict | 65.0 | 77.9 | 35 (15.0%) | 23% | 0.796 | 0.669 / 1.52× |
| Qwen TextVQA | lenient | 74.3 | 85.0 | 33 (13.6%) | 12% | 0.789 | 0.661 / 1.78× |
| Phi-4 (any) | either | — | — | — | 0–5% | unchanged | — |

Also, strict scorer, Qwen VQAv2 restricted to one-word STOP answers
(n=120, HELP=14): confidence ROC **0.633**.

**Reading.** The VQAv2 "confidence reversal" is substantially a scoring
artefact of Qwen's verbosity: 28 of 44 HELP states are "verbose STOP → one-word
HIGHRES". Under lenient scoring — which brings Qwen's STOP accuracy to Phi-4's
level (76%) — confidence is above chance on VQAv2 for Qwen too, and the
cross-regime gap shrinks from 0.37 to 0.13. The headline sentence *"answer
confidence is not perceptual uncertainty"* is not supported as a
backbone-independent phenomenon. What survives:

- confidence is a strong LOOK signal in the text-centric regime on both backbones;
- on VQAv2 its strength is backbone- and scorer-dependent, and the Qwen reversal
  is mostly format;
- v2.2's VQAv2 advantage shrinks but does not vanish under lenient labels
  (ROC 0.617, lift 1.49×); its top coefficient `ans_words +1.22` was partly
  learning "verbose → will be rescued by HIGHRES".

## 2b. Causal control — Qwen, VQAv2, one-word answer prompt (new generations)

Same Qwen2.5-VL-7B, same 250 lockbox states, same 64→192 views, serial, 2
replays; only the open-ended instruction changed to "Answer with a single word
or number … do not write a sentence" (`PPU_PROMPT_VARIANT=oneword`,
prompt version `p0.2-oneword`, log `logs/lockbox_vqav2_qwen_oneword.jsonl`).

| Qwen VQAv2 | prompt | verbose STOP | STOP | HIGHRES | HELP (prev.) | format flips | conf ROC [95% CI] | lift@1× [CI] |
|---|---|--:|--:|--:|--:|--:|:--:|:--:|
| original | p0.1 | 40% | 42.9 | 65.3 | 44 (21.8%) | 64% | **0.423** [0.33, 0.52] | 0.73× [0.28, 1.17] |
| **one-word** | p0.2 | **3%** | **78.1** | **83.7** | 20 (8.5%) | 5% | **0.715** [0.60, 0.82] | **2.36×** [0.51, 4.72] |
| (Phi-4, coarse 16) | — | 3% | 71.8 | 78.1 | 21 (9.1%) | 5% | 0.814 [0.72, 0.89] | 4.21× [1.92, 6.63] |

Constraining the answer format alone moves Qwen's VQAv2 confidence ROC from
0.42 to 0.72 (CI excludes 0.5), brings its STOP accuracy to the Phi-4 level,
and makes its label distribution look like Phi-4's. The reversal was the
format, not the regime. Strict and lenient scoring now agree (0.715 / 0.732).

The frozen v2.2 gate on these new labels: ROC 0.607, lift 0.59× — it does not
transfer to the one-word regime (its `ans_words` signal no longer exists). A
gate for format-controlled agents would have to be retrained; that is out of
scope for the closed Stage A.

## 3. Feature-group baselines (trained on dev, single pass on lockbox)

| variant | TextVQA ROC / lift | VQAv2 ROC / lift |
|---|---|---|
| confidence only | **0.796 / 3.23×** | 0.423 / 0.73× |
| question surface only | 0.458 / 0.76× | 0.526 / 1.15× |
| answer surface only | 0.549 / 1.14× | **0.733 / 2.09×** [1.49, 2.62] |
| Q + A, no confidence | 0.548 / 1.14× | 0.723 / 1.98× |
| full v2.2 (frozen) | 0.669 / 1.52× | 0.703 / 1.77× |

Answer-surface alone matches or beats v2.2 on VQAv2 (consistent with §2: the
signal is largely answer length). Confidence interactions are what lift
TextVQA from 1.14× to 1.52×. Honest phrasing: *cheap behavioural traces carry
most of the VQAv2 signal; confidence interactions keep the gate usable on
TextVQA.* (Hidden-state probe not run: vLLM server exposes no hidden states.)

## 4. Full-population deployment view (AMBIGUOUS included), Qwen

| VQAv2 n=250 (19% ambiguous) | PPU v2.2 | confidence | random | tok/state |
|---|--:|--:|--:|--:|
| 5% | **45.7** | 42.8 | 44.0 | 6 |
| 10% | **47.3** | 44.0 | 45.1 | 13 |
| 15% | **49.5** | 45.0 | 46.3 | 19 |
| 20% | **50.7** | 46.1 | 47.3 | 25 |

TextVQA: confidence > PPU ≈ random (+2pp at 20%). Ordering is unchanged by
including ambiguous states. Figures: `figures/fig2_fullpop_{vqav2,textvqa}.pdf`.
Caveat from §2: the VQAv2 numbers are under strict scoring and therefore
include format-rescue gains.

## What this means for the paper

1. Figure 1 as drawn (Qwen-only reversal) cannot be the headline. §2b is a
   causal control: with the answer format fixed, confidence is a reliable
   LOOK signal on VQAv2 for Qwen (0.72) as it is for Phi-4 (0.81) and on
   TextVQA for both (0.80 / 0.84). RQ3 becomes: *confidence-based uncertainty
   is a robust perceptual-need signal across regimes and backbones once
   answer format is controlled; the apparent reversal was an artefact of
   verbose answers under exact-match scoring.*
2. `fig1_backbones.pdf` shows both backbones; it is honest but no longer tells
   the original story.
3. Nothing here touches v2.2, the lockbox verdict, or the TextVQA claims.

---

# Addendum 2026-08-23 — reviewer priorities 1–3

## 5. RQ1 policy headroom under format-tolerant scoring and symmetric cross-replay (priority 1, 3)

`scripts/gate0_lenient_recheck.py` → `paper/gate0_lenient_recheck.md`. Original TextVQA Gate 0
rollouts (`logs/tv_200.jsonl`, n=200, 24 rollouts/state) re-audited with the frozen `ppu/audit.py`
after rescoring every generation; nothing regenerated.

| scorer | direction | best fixed (cf) | oracle (cf) | **P_policy [95% CI]** | DMG |
|---|---|--:|--:|:--:|--:|
| strict (paper) | R1→R2 | 69.1 | 78.8 | **+9.8** [+4.7, +14.8] | 13.0 |
| strict | R2→R1 | 70.1 | 83.8 | **+13.8** [+9.7, +18.0] | 13.0 |
| strict | symmetric | 69.6 | 81.3 | **+11.8** [+8.0, +15.8] | |
| lenient | R1→R2 | 80.1 | 88.8 | **+8.8** [+5.1, +12.7] | 11.5 |
| lenient | R2→R1 | 79.2 | 89.7 | **+10.4** [+7.0, +14.2] | |
| lenient | symmetric | 79.7 | 89.2 | **+9.6** [+6.5, +13.0] | |

MME (MCQ, option permutation; lenient n/a): +9.8 / +8.5 / symmetric **+9.1** [+4.8, +13.6].
Decisive-state action mix is unchanged under lenient labels (STOP 34 / CROP 43 / OCR 21 / HIGHRES 2).
**The +9.8pp headroom is robust to the scoring artefact and to the direction of cross-fitting.**
The R2 paraphrase procedure is documented and audited in the same file (12 deterministic regex
rewrites + fallback prefix; no LLM).

## 6. Fresh prospective lockbox for the format confound (priority 2)

Pre-registered in `paper/PREREG_FRESH_LOCKBOX.md` (predictions written, manifest SHA-1s logged,
before any rollout). 250 never-used VQAv2 states (same strata rule) + 250 never-used TextVQA train
states; Qwen2.5-VL-7B; four arms = {VQAv2, TextVQA} × {original prompt, one-word prompt}.
Verdict `scripts/fresh_lockbox_verdict.py` → `paper/fresh_lockbox_result.md`.

| | prediction | result |
|---|---|:--:|
| P1 | original VQAv2: ROC < 0.55, verbose > 25%, ≥50% of HELP are format flips | **PASS** (0.428, 40%, 77%) |
| P2 | lenient rescoring raises original-VQAv2 ROC by ≥ 0.15 | **FAIL** (0.428 → 0.526, +0.10) |
| P3 | one-word VQAv2: verbose < 10%, STOP +20pp, ROC > 0.60 with CI lower > 0.5 | **PASS** (3%, 44.8→78.0, **0.897** [0.84, 0.95]) |
| P4 | TextVQA ROC > 0.70 under both prompts, within 0.10 | **PASS** (0.786 / 0.815) |
| P5 | under lenient scoring, original and one-word VQAv2 ROC within 0.15 | **FAIL** (0.526 vs 0.906) |

Full table:

| arm | scorer | STOP | HIGHRES | verbose | HELP (prev.) | flips | conf ROC [CI] | lift@1× |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| VQAv2 original | strict | 44.8 | 60.7 | 40% | 30 (16.8%) | 77% | 0.428 [0.32, 0.54] | 0.80× |
| VQAv2 original | lenient | 75.8 | 76.3 | 40% | 19 (8.6%) | 47% | 0.526 [0.38, 0.66] | 1.22× |
| VQAv2 one-word | strict | 78.0 | 80.7 | 3% | 14 (6.1%) | 7% | **0.897** [0.84, 0.95] | 4.65× |
| TextVQA original | strict | 67.5 | 77.7 | 43% | 29 (12.9%) | 21% | 0.786 [0.71, 0.86] | 3.21× |
| TextVQA one-word | strict | 76.4 | 82.4 | 26% | 18 (8.0%) | 6% | 0.815 [0.72, 0.90] | 2.78× |

**Reading.** The confound replicates prospectively (P1), the causal prompt control replicates and is
stronger than on the sealed lockbox (0.897 vs 0.715; P3), and TextVQA is unaffected (P4). Two
predictions failed, and they fail in the same direction: **lenient rescoring is only a partial
correction** (+0.10, not +0.23 as on the sealed lockbox) and does not bring the original-prompt
arm anywhere near the one-word arm. The artefact is therefore not purely in the scorer: verbose
generation itself changes which states HIGHRES "helps" (47% of lenient-HELP states are still
format flips). The paper should present prompt-level format control as the remedy and
lenient rescoring as a diagnostic only. These two failures are reported as such.
