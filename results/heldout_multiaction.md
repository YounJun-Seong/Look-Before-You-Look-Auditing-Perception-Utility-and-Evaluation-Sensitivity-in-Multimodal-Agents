# Frozen multi-action policy — held-out evaluation (prereg result)

Single preregistered evaluation pass per `paper/PREREG_MULTIACTION_HELDOUT.md`,
run 2026-08-30. Numbers frozen in `paper/heldout_multiaction.json`
(seed 20260828, 10k state-paired bootstrap). n = 250 held-out fresh states per
dataset; STOP determinism check: **0 mismatches / 500 replays on both datasets**.

## Verdict against the pre-declared criteria

**1. Primary (TextVQA strict, multi-action − best-fixed-at-B > 0, CI excl. 0): PASS.**

| budget B | multi acc | best-fixed-at-B | Δ | 95% CI |
|---|--:|--:|--:|:--|
| ¼× (77 tok) | .745 | .675 | **+.070** | [+.027, +.114] ✓ |
| ½× (153 tok) | .774 | .675 | **+.099** | [+.055, +.144] ✓ |
| 1× (307 tok) | .774 | .777 | −.003 | [−.032, +.024] |

At ¼ and ½ of the always-HIGHRES token budget the frozen multi-action policy
beats the best fixed action available at that budget, CI excluding zero. At the
full budget it matches (does not beat) always-HIGHRES — the policy's value is
budget-constrained allocation, and the result says exactly that.

**2. Secondary (≥30% of oracle−STOP headroom recovered): PASS on both strict
datasets.**

| dataset / scorer | headroom (oracle−STOP) | λ=0 recovery | ½-budget recovery |
|---|--:|--:|--:|
| TextVQA strict | .167 | 55.6% | 59.2% |
| VQAv2 strict | .283 | 94.5% | 72.3% |
| VQAv2 lenient | .124 | 32.1% | 27.7% (below the 30% target; reported as such) |

**3. VQAv2 under both scorers — the pre-declared confound finding
materialized.** Under strict scoring the multi-action policy dominates
everything (Δ vs best-fixed +.103 to +.204, all CIs excluding 0). Under the
lenient rescore of the same generations the advantage collapses to +.026/+.034
(CIs touching 0 at ¼ and ½ budget; +.045 [+.007, +.083] at 1×), and the
ranking against the binary v2.2 gate **flips** (½ budget: v2.2 .802 vs
multi .792). The large strict-VQAv2 gain is therefore substantially the
audited format confound propagating into the policy comparison — exactly the
scenario criterion 3 pre-declared as a finding. TextVQA (strict-only, primary)
is unaffected.

## Honest caveats

- **Latency axis.** At low token budgets the policy substitutes OCR (0 visual
  tokens, CPU tool time): ¼-budget TextVQA mean added latency is ≈29 s/state
  vs ≈0.15 s for the confidence binary. Token-matched wins do not transfer to
  the latency axis; the two axes are reported separately per prereg and any
  claim must name its axis.
- **vs. binary baselines at matched tokens (TextVQA):** +.009 to +.015 with
  CIs including 0 — the policy beats best-fixed, not the tuned binaries, on
  TextVQA; on strict VQAv2 it beats the binaries too (½ budget vs v2.2:
  +.058 [+.006, +.109]), but see the lenient collapse above.

## Deviations log

1. **Collection interruption and re-collection (before evaluation).** The
   2026-08-28 vLLM engine died mid-run; `collect_heldout_arms.py` logged 3,660
   `APIConnectionError` rollouts (183 VQAv2 states) instead of aborting. Those
   rollouts were stripped (backup `logs/heldout_vqav2_arms.jsonl.bak_apierr`)
   and the 183 states re-collected 2026-08-30 under the identical frozen
   config; integrity check: 250/250 states, 0 errors, 0 STOP-determinism
   mismatches. No evaluation was run against the poisoned log.
2. **sklearn version shim.** The frozen artifacts were pickled under
   sklearn 1.9.0 (Python ≥3.11); no such interpreter remains on the host. The
   eval ran under sklearn 1.7.2 with a `predict_proba` attribute shim, after
   asserting the patched output equals sigmoid(frozen coefficients) to 1e-16
   and that pickle coefficients match the frozen JSON sidecar. Numerically
   identical; single-pass rule preserved.
3. The blinded human audit was withdrawn without execution (see
   `paper/PREREG_HUMAN_AUDIT.md`); it is independent of this evaluation.
