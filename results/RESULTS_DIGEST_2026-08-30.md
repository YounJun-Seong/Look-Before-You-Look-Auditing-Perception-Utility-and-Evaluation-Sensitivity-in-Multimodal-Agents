# Results digest — 2026-08-30 session

**EXPERIMENTS FROZEN (2026-08-30, end of session).** Everything below is
final; remaining work is writing, figures, and the Overleaf submission audit.
No further collection or analysis passes are planned or permitted without a
new dated addendum here.

> **Addendum 2026-08-31.** Three zero-model-inference supplement analyses
> authorized for the current submission (user decision; no new rollouts, no
> new readers, no sample-size increases): (i) cross-reader opportunity
> agreement (Qwen vs Phi-4, existing Gate-0-protocol logs), (ii) exact
> budget-constrained oracle via multiple-choice-knapsack DP alongside the
> Lagrangian sweep and the randomized hull, (iii) latency-axis summary from
> already-logged costs. Results appended as §9-§11. The learned controller,
> fresh 500-700-state evaluation, sequential experiment, third reader, and
> new datasets are explicitly deferred to a future cycle.

> **Addendum 2026-08-31 (2) — cycle reopened for this submission.** User
> decision: three additions go into the current submission after all:
> (i) a budget-aware deployable selector (trained on dev/Gate 0 only, frozen
> before evaluation), (ii) a prospective ~600-fresh-TextVQA-state evaluation
> of that selector against the dev-fitted randomized mixture (both frozen
> first; new prereg doc), (iii) one minimal sequential (two-step) perception
> experiment. Everything else in the 08-30 freeze stays frozen.

> **Addendum 2026-08-31 (3).** Follow-ups authorized: (A) realized-cost
> matched mixture on the prospective log (zero inference, post-hoc,
> disclosed), (B) a simple history-conditioned two-step controller built on
> the 100-state sequential set (dev role), with the fixed baseline hierarchy
> stop-after-a1 / always-second-look / random / state-independent sequential
> mixture / controller / sequential oracle, (C) cross-action transfer of the
> post-a1 signal (zero inference). If (B) is informative, the controller is
> frozen and prospectively tested once on ~300 fresh sequential states under
> a new prereg. Results as §14-§16.

> **Addendum 2026-09-01.** Depth round authorized (user's 12-point review):
> (i) stronger one-step controller (hidden-state representation), (ii)
> cost-aware sequential controller with history-confidence features, (iii)
> human semantic adjudication of strict/lenient disagreements — **DECLINED
> by the user 09-01; no sheets created; the paper uses "measurement
> dependence" wording** (distinct protocol from the withdrawn 08-30 audit),
> (iv) latency-cost Pareto, (v) cross-reader selector transfer, (vi)
> cross-dataset frozen-selector transfer. (iv)-(vi) are zero-inference on
> existing logs. Any evaluation that reuses an already-analyzed set is
> labeled a disclosed secondary analysis, never "prospective". §17+.

All numbers strict scorer unless noted. Source files linked per section; every
collection passed a zero-transport-error integrity gate.

## 1. Held-out frozen multi-action evaluation (preregistered, 1 pass)

`paper/heldout_multiaction.{json,md}`, prereg `PREREG_MULTIACTION_HELDOUT.md`.
n = 250 fresh states/dataset, STOP determinism 0 mismatches.

- **Criterion 1 (primary, TextVQA) PASS**: multi-action − best-fixed-at-B =
  +7.0pp [+2.7, +11.4] at ¼ budget, +9.9pp [+5.5, +14.4] at ½; parity at 1×.
- **Criterion 2 PASS**: headroom recovery TextVQA 55.6/59.2%, VQAv2 strict
  94.5/72.3% (λ=0 / ½-budget); VQAv2 lenient 32.1/27.7% (½-budget below the
  30% target, reported).
- **Criterion 3 — pre-declared confound materialized**: VQAv2 strict gains
  (+10~20pp) collapse to +2.6~4.5pp under lenient rescoring and the
  multi-action-vs-v2.2 ranking flips at ½ budget (v2.2 .802 vs multi .792).
- Caveats: low-budget policy substitutes OCR → latency axis differs
  (¼-budget TextVQA ≈29 s/state); vs tuned binaries on TextVQA +0.9~1.5pp
  (CIs include 0) — the beaten baseline is best-fixed. Against randomized
  state-independent mixtures see §8: positive but not significant on TextVQA;
  reversed on strict VQAv2 for scorer-artifact reasons.
- Deviations logged in the md (engine-death re-collection before eval;
  sklearn shim verified to 1e-16; human audit withdrawn separately).

## 2. Budget-constrained oracle frontier (post-hoc, Gate 0 logs)

`paper/budget_oracle_frontier.{json,md}` — max E[y] s.t. E[c] <= B, Lagrangian
sweep, cross-replay symmetrised, deterministic arms.

| budget (×always-HIGHRES tok) | MME Δ vs STOP (recovery) | TextVQA Δ (recovery) |
|---|--:|--:|
| 5% | +.122 (96%) | +.112 (59%) |
| 10% | +.128 (100%) | +.184 (97%) |
| 20% | +.128 (100%) | +.190 (100%) |

Headline: nearly all attainable adaptive gain is concentrated in ≤10% of the
full-perception token budget. Qwen and (see §5) Phi-4 alike.

## 3. Per-action damage decomposition (post-hoc, Gate 0 logs)

`paper/per_action_harm.{json,md}` (Qwen), `paper/p4_per_action_harm.*` (Phi-4).
P(harm | STOP correct), any / single-action:

| family | MME any/single | TextVQA any/single |
|---|:--|:--|
| CROP-grid | .449 / .169 | .154 / .049 |
| OCR | .080 / .080 | .085 / .085 |
| HIGHRES | .145 / .145 | .043 / .043 |
| any (DMG set) | .464 / .157 | .222 / .051 |

Aggregate DMG (13.8/13.0) is dominated by the grid-crop existential; single-
action harm is 4-17%. P(help | STOP wrong) ≫ harm on TextVQA (.782 vs .222).

## 4. Answer-span confidence (completed 08-28, unchanged)

`paper/answer_span_confidence.{json,md}` — regeneration byte-match 100%.
TextVQA stable (0.74-0.80 all variants); VQAv2-original reply-mean 0.42-0.43
(below chance) recovering only to 0.58-0.62 at span-min; one-word 0.72-0.90.

## 5. Second-reader replication — Phi-4-multimodal (prereg PREREG_SECOND_READER.md)

`paper/second_reader_replication.md`, audits `p4_{mme,textvqa}_audit.*`,
full 196/200 Gate-0 states, same views/replays/scorers, 0 transport errors.

| | Qwen P_policy | Phi-4 P_policy | Qwen DMG | Phi-4 DMG | STOP-opt (Q / P4) |
|---|--:|--:|--:|--:|:--|
| MME | +9.8 [+5.1,+14.7] | +2.8 [−2.2,+8.0] | 13.8 | 7.7 | 34.2% / 25.4% |
| TextVQA | +9.8 [+4.7,+14.8] | **+10.0 [+6.0,+14.0]** | 13.0 | 7.0 | 32.1% / 23.0% |

- **H1 PASS** (pooled +6.4pp; TextVQA CI excludes 0; MME positive, CI crosses
  0 — pre-declared as not-a-failure). **H2, H3, H4 PASS** (harm present in
  every family — Phi-4 MME OCR harm 40% vs Qwen 10.7%; STOP-optimal 23-25%;
  frontier saturates by 5-10% budget: `p4_budget_frontier.md`).
- Oracle action mix differs by reader (TextVQA: Qwen CROP-heavy 45.7% vs
  Phi-4 OCR-heavy 37.7%) → readers disagree on *which* action, both need
  state-dependent allocation.
- Interpretive caveat: Phi-4's low absolute accuracy under this protocol is
  substantially a format artifact — its STOP answers run 4.0 words mean / 32%
  >3 words here vs 1.5 / 5% in its own lockbox runs; strict scoring punishes
  the verbosity. Structural quantities, not absolute levels, are the
  comparanda.
- **Lenient rescore (offline, `p4_{ds}_audit_lenient.*`,
  `p4_budget_frontier_lenient.*`, script `p4_lenient_rescore.py`; strict
  reproduction asserted 0 mismatches).** MME is MCQ (extracted-letter,
  first-char match), so lenient == strict by construction — no surface-form
  axis. TextVQA lenient: STOP .258 → .505 (format artifact confirmed), best
  fixed (cf) .624, oracle (cf) .743, **P_policy +11.9 [+7.1, +16.7]** (CI
  excludes 0; strict was +10.0), STOP-optimal 36.9% (decisive n=103);
  frontier recovery 61% @5% / 90% @10% of budget. Structural replication
  survives — and strengthens — under the same-generation lenient rescore.

## 6. Phi-4 realizable policy (post-hoc; side collection byte-match 0/792)

`paper/p4_realizable.{json,md}`, confidence from
`data/span_conf/p4_{mme,textvqa}_stopconf.jsonl`.

Phi-4's own STOP-confidence selector is **indistinguishable from random
allocation at every matched budget** (conf−random point estimates −0.003 to
+0.013, all CIs crossing 0), while the budget-constrained oracle stands far
above both (e.g. TextVQA 10% budget: random .251, confidence .254, oracle
.396). The frozen v2.2 gate is Qwen-fitted and not transferred (protocol).
Reading: the *headroom* replicates across readers; the *zero-training
realizable signal* does not — confidence quality is reader-and-regime
specific, consistent with the paper's confidence findings.

## 7. Multi-replay robustness (prereg PREREG_MULTI_REPLAY.md)

`paper/multi_replay_robustness.{json,md}` — replays 3-5 newly collected
(`ppu/replay_multi.py`), 100-state outcome-blind subsets, deterministic arms.

- Naive same-replay oracle stays inflated at every K (MME .65-.66 vs cross
  .50-.53; TextVQA .885-.888 vs .773).
- K=4-5 estimates stay inside the K=2 bootstrap CI on both datasets →
  pre-declared reading: **two replays sufficient**.
- TextVQA held-out-mean/symmetric estimates *rise* with K (.773 → .830/.836):
  more evaluation replays denoise the outcome, so the published K=2 estimate
  is **conservative** — the protocol under-states, never over-states, the
  headroom. Headroom positive at every K (MME +.03~+.09, TextVQA +.05~+.11).

## 8. Randomized fixed-mixture baselines (post-hoc; the last analysis before freeze)

`paper/randomized_mixture_baseline.{json,md}` (in-sample + cross-fitted),
`paper/deployable_mixture_textvqa.{json,md}` (dev-fitted). LP over the fixed
arms {STOP, each grid/fallback CROP box, uniform CROP, OCR, HIGHRES}: with one
expected-token constraint the optimum mixes ≤2 arms (concave envelope).
Comparator strength ascends: in-sample (weights fitted on the eval sample —
baseline-favoring upper bound) → cross-fitted (2-fold, the fair realizable
state-independent policy) → deployable (weights fitted on Gate 0 dev, applied
unchanged to held-out; TextVQA only). Paired 10k bootstrap vs the frozen
selector at the prereg budget read-offs; recomputed selector accuracies
asserted equal to the frozen JSON.

**TextVQA strict — the primary cells (multi − mixture):**

| budget | vs in-sample | vs cross-fitted | vs deployable |
|---|:--|:--|:--|
| ¼× | +.009 [−.027,+.045] | +.020 [−.017,+.056] | +.010 [−.025,+.046] |
| ½× | +.012 [−.022,+.046] | +.018 [−.012,+.049] | **+.030 [−.003,+.063]** |
| 1× | −.011 [−.045,+.024] | −.011 [−.045,+.024] | +.028 [−.009,+.065] |

**VQAv2 (vs cross-fitted):** strict ¼× **−.153 [−.204,−.102]** and ½×
**−.072 [−.117,−.027]** — a fixed grid crop significantly beats the selector;
under lenient the reversal vanishes (¼× +.010 [−.017,+.038], ½× +.014
[−.017,+.045]). A third instance of the scorer artifact propagating into a
policy comparison.

Honest bottom line for the claims: the selector's significant cost-matched
advantage holds against **deterministic fixed** policies (the prereg
comparator); against **optimal state-independent randomized mixtures** the
TextVQA advantage is a consistent positive point estimate that strengthens as
the mixture is made fairer (+0.9/+1.2 → +2.0/+1.8 → +1.0/+3.0pp at ¼×/½×,
the deployable ½× borderline at CI lower bound −0.3pp) but does not clear the
95% level, and on strict VQAv2 the mixture wins for scorer-artifact reasons.
Claims must name the comparator class. Also recorded: the deployable mixture
transferred dev→held-out with no loss (dev fit .717 → held-out .744 at ½×),
so a transfer-loss argument is not available.

## 9. Cross-reader opportunity agreement (addendum 08-31, zero inference)

`paper/cross_reader_agreement.{json,md}`. help = some deterministic
additional arm strictly beats STOP (mean over replays), same states/views.

| | Both | Qwen only | Phi-4 only | Neither | Jaccard | κ |
|---|--:|--:|--:|--:|--:|--:|
| MME (196) | 18% | 18% | 17% | 46% | .33 | .22 |
| TextVQA (200) | 11% | 22% | 26% | 42% | .19 | **−.05** |

P(help) is nearly identical across readers (~.33-.36) but P(Phi-4 help |
Qwen help) = .33 on TextVQA ≈ the base rate — instance-level opportunity is
essentially **reader-independent in rate, reader-specific in location**.
Among both-help states, best-family agreement is 73-77%. One-line conclusion:
perception opportunity is structurally present in both readers but largely
non-overlapping at the instance level (the "structurally shared,
instance-specific" result).

## 10. Exact budget-constrained oracle (addendum 08-31, zero inference)

`paper/exact_oracle_frontier.{json,md}` — multiple-choice-knapsack DP on the
selection replay, scored cross-replay, vs the Lagrangian sweep and the
randomized hull (Gate 0 logs). The relaxation loophole is closed: at ≥10% of
the always-HIGHRES budget the Lagrangian read-off matches the randomized hull
to ≤1pp; at the 5% budget the exact DP is **higher** by +1.5pp (MME) /
+2.8pp (TextVQA) — the front-loaded-opportunity conclusion strengthens, it
does not weaken. (The DP's evaluated value can sit ~1pp below the hull at
loose budgets: exact optimization of the selection replay overfits selection
noise slightly — itself a small illustration of selection inflation.)

## 11. Latency at the token-matched operating points (addendum 08-31)

`paper/latency_summary.md` — already-logged latencies of the same frozen
operating points, no recomputation. Key rows: multi-action's low-token
operating points carry OCR tool time (TextVQA ¼×: 29.0 s/state vs 0.15 s for
the confidence binary at the same tokens; VQAv2 ¼×: 11.1 s vs 0.14 s), while
always-HIGHRES is 0.67 s. Any efficiency claim must name its axis; the two
axes are reported separately and never summed.

## 12. Prospective selector-vs-mixture evaluation (addendum 08-31(2); prereg)

`paper/prospective_eval.{json,md}`, prereg `PREREG_PROSPECTIVE_SELECTOR.md`.
n = 600 never-touched TextVQA states; frozen selector v2 (F0+ridge0.3, full
candidate grid disclosed) vs frozen dev-fitted mixture; 0 transport errors.

- **Primary endpoint (½× strict) FAILED**: selector − mixture = +0.9pp
  [−1.6, +3.4] (¼× +0.2, 1× +0.5; all CIs include 0). Reported as a failure.
- **Lenient rescore of the same generations: significant at every budget** —
  ¼× +3.0 [+0.8, +5.2], ½× +4.6 [+2.4, +6.8], 1× +3.0 [+0.7, +5.5]. The
  measurement-validity thesis reappears in the paper's own prospective test,
  in the opposite direction: strict scoring MASKS a real adaptive advantage
  (the selector leans on OCR/CROP whose answers pay strict format penalties).
- Selector also spends fewer tokens than the mixture at every budget
  (½×: 129 vs 154) — the comparison is conservative.
- Realizability gap persists: oracle .818 vs selector .720 (½× strict).
- One-line: the deployable selector does not beat the strongest
  state-independent baseline under the preregistered strict scorer, but
  significantly does at every budget once the same generations are
  format-tolerantly rescored — the scoring rule again decides the policy
  comparison.

## 13. Two-step sequential audit (addendum 08-31(2); prereg)

`paper/sequential_audit.{json,md}`, prereg `PREREG_SEQUENTIAL.md`. n = 100
outcome-blind states, 4 enumerated trajectories × 2 replays, 0 errors.

- **Q1 — sequential opportunity exists**: cross-replay oracle over
  {stay, →CROP-TR, →HIGHRES} beats stopping after a1 by **+9.3pp
  [+4.7, +14.5]** (a1 = center-crop) and **+9.2pp [+4.2, +14.5]** (a1 = OCR),
  strict; lenient similar.
- **Q2 — value is history-dependent**: continuation gain concentrates where
  a1 failed (+22~+27pp on a1-incorrect states) and is ≈0 where a1 succeeded —
  the agent-decision structure ("acquire more given what was learned") is
  real, not an artifact of averaging.
- **Q3 — a cheap post-a1 signal predicts it**: "a1 changed the STOP answer"
  gives AUC .70/.74 for "a second acquisition helps" (P(helps) ≈ 10-15%);
  answer length ≈ .55.
- **Q4 — fixed two-step sequences do NOT dominate**: at matched tokens the
  best 1-step arm (CROP-TR .767 @ 243) matches or beats fixed 2-step chains;
  the sequential value is adaptive (Q1/Q2), not a fixed pipeline.

## 14. Realized-cost matched mixture (addendum 08-31(3)A; zero inference)

`paper/costmatched_mixture.{json,md}`. Dev-fitted mixture re-solved at the
selector's REALIZED token cost (53/129/212) on the 600 prospective states.
Cost-matching strengthens the result: strict +1.9 [−0.1,+4.0] / +1.2
[−1.4,+3.8] / +0.9 [−1.7,+3.4] (¼× borderline; all ns), lenient **+2.7
[+0.9,+4.5] / +5.1 [+2.8,+7.4] / +3.6 [+1.4,+5.9] — all significant**. The
"same realized cost?" objection is closed.

## 15. History-conditioned two-step controller + transfer (addendum 08-31(3)B/C)

`paper/seq_controller.{json,md}`; dev role = the 100-state sequential set.
Controller: two fixed post-a1 features (answer-changed, answer length), two
ridge Δ-heads, argmax(0, Δ̂_TR, Δ̂_HI); 5-fold cross-fitted.

- Strict: controller beats stop-after-a1 (+4.5/+5.3pp, CIs exclude 0) but
  not the mixture (= always→TR): center −1.7 [−3.7, 0.0], ocr −0.3 (ns).
- **Lenient, center branch: controller .863 > mixture-CF .812 (+5.1
  [+2.1,+8.3]) and > both pure continuations (TR .832 / HI .838)** — the
  history-conditioned CHOICE between continuations beats every fixed one,
  approaching the oracle (.883). OCR branch ns.
- (C) Transfer: heads moved across a1 branches keep their value (Δ −3.8 to
  +1.9pp, 3 of 4 ≥ native) — the post-a1 signal is largely a generic
  residual-insufficiency signal, with mild action-specificity in one cell.
- Frozen for the prospective test: `models/seq_controller_frozen.pkl`
  (sha `cffa1043f455f7d1`); frozen matched-cost mixture resolves to
  **always→TR** on both branches.

## 16. Sequential prospective test (PREREG_SEQ_PROSPECTIVE.md) — complete

`paper/seq_prospective_eval.{json,md}`. 300 fresh states (manifest sha
`4196e1abbead1506`), 5,400 rollouts, 0 transport errors; single pass.

- **Primary endpoint FAILED**: center branch, strict, controller − frozen
  mixture (= always→TR) = **−1.7pp [−3.6, +0.2]**. The dev lenient-center
  advantage (+5.1pp) did **not** transfer (prospective lenient center −1.2pp
  [−2.9, +0.4]). Reported as a failure.
- Only positive cell (secondary, 1 of 8, multiplicity caveat): ocr branch,
  lenient scorer, lenient-trained heads: +2.7pp [+0.4, +5.1].
- Both frozen controllers collapse to look-rate 1.00 on fresh states — they
  effectively route TR-vs-HIGHRES, and that routing does not beat the best
  fixed continuation.
- **The sequential opportunity itself replicates prospectively, with CIs**
  (state-paired 10k bootstrap, appended to `seq_prospective_eval.{json,md}`;
  per-state rows in `seq_prospective_oracle_rows.csv`): cross-replay oracle −
  mixture(always→TR) = **+2.8pp [+1.1, +4.8]** (center strict), **+5.8pp
  [+3.4, +8.4]** (ocr strict); lenient +2.7 [+1.0, +4.7] / +6.2 [+3.8, +8.8];
  oracle − stop-after-a1 = +4.4 [+1.9, +6.9] / **+14.0 [+10.3, +17.8]**. All
  eight CIs exclude 0 — "prospectively replicates" is now CI-backed.
- Reading: the **Opportunity ≠ Realizability pattern reproduces in the
  sequential setting under a preregistered prospective test** — substantial
  sequential opportunity, history dependence and moderate predictability on
  dev, but no deployable advantage over the strongest state-independent
  continuation on fresh states. This is the pre-declared
  either-outcome-answers-the-question result, not a null to hide.

## 17. Cross-dataset frozen-selector transfer (addendum 09-01; zero inference)

`paper/cross_dataset_transfer.{json,md}`. Selector v2 applied to the VQAv2
held-out set with NO adaptation (TextVQA heads/scaler/ECDF/λ verbatim);
disclosed post-hoc (set previously analyzed in §1/§8).

- Strict: parity with the cross-fitted mixture (−0.9/−2.7/−1.9pp, CIs
  include 0); loses to the in-sample mixture at ½×/1× (that bound is
  baseline-favoring). Selector's action mix shifts sensibly (OCR-heavy on
  TextVQA → CROP/HIGHRES-heavy on VQAv2).
- Lenient: directionally positive at every budget (+2.8/+2.0/+1.5pp, CIs
  include 0) — the TextVQA prospective pattern (strict parity, lenient
  positive) reproduces on a second dataset without any tuning.
- Note: the oracle column here is the family-level allocation ceiling
  (4 arms; crop = uniform expectation), while mixtures draw on per-slot
  arms — stated to avoid an apples/oranges reading.
- One-line: the frozen selector transfers across datasets without
  adaptation and the strict-vs-lenient conclusion structure transfers with
  it; "TextVQA-specific" objection weakened.

## 18. Latency-cost Pareto + cross-reader selector transfer (addendum 09-01)

**(iv) Latency Pareto** — `paper/figures/figS_latency_pareto.{pdf,png}`
(zero inference; frozen held-out operating points on both cost axes). The
policy ordering changes with the constrained resource: on tokens the
multi-action sweep dominates at low budget; on wall-clock the λ_lat sweep
avoids OCR, the binaries reach always-HIGHRES cheaply, and OCR-heavy points
sit 1-2 orders of magnitude to the right. Supports "optimal action choice
depends on which resource the agent is constrained by"; axes never summed.

**(v) Cross-reader transfer** — `paper/cross_reader_transfer.{json,md}`
(zero inference; Qwen-trained selector v2 on Phi-4 states, Phi-4 confidence
from the verified side collection; verbatim vs conf-recalibrated variants,
which barely differ — the issue is the signal, not calibration).

- **MME: transfer fails** — the transferred selector nearly always stops
  (vtok ≤ 73) and loses to the cross-fitted mixture by −8.0 to −10.4pp
  (CIs exclude 0), leaving Phi-4's headroom unrealized.
- **TextVQA: a transferable component exists** — the transferred selector
  beats always-STOP by +7.9 to +10.4pp (all CIs exclude 0) and is
  directionally ahead of the CF-mixture (+1.3 to +2.7pp, CIs marginal).
- Combined with §9 (instance-level help agreement ≈ chance): what transfers
  is a coarse task-level signal of which action class helps, not
  instance-level perception need — perception need is substantially
  reader-specific, and policy transfer inherits that.

## 19. Cost-aware sequential controller v2 (addendum 09-01(B))

`paper/seq_v2_eval.{json,md}`; trained/frozen on dev-100
(`models/seq_controller_v2_frozen.pkl`, sha `b74b8b6e70332a52`; dev
a1-confidence regeneration byte-match 379/400, disclosed), evaluated on the
300-state sequential set as a DISCLOSED SECONDARY (set previously analyzed).
Removes all three disclosed v1 weaknesses: cost term (argmax[Δ̂ − λc]),
7 history features incl. pre/post confidence, and genuine stopping
(look-rate 0.83-0.85, or 0.30 under λ_B, vs v1's 1.00).

- Strict: parity with the frozen mixture (= always→TR) — center −0.6pp
  [−1.8, +0.6] (v1 was −1.7), ocr +0.4pp ns; **λ_B variant matches TR within
  −1.6pp ns at 80 vs 241 tok (⅓ the cost)**.
- Lenient: ocr branch v2 **+2.7pp [+0.5, +4.8] over the mixture** (secondary,
  consistent with the recurring strict-parity/lenient-positive pattern);
  λ_B keeps +1.6pp ns at ⅓ cost.
- Reading: with the cost term, richer history and real stopping in place,
  the qualitative conclusion is unchanged — the sequential
  opportunity-realization separation is not an artifact of a deliberately
  weak controller, and the cost-aware version buys large token savings at
  statistically indistinguishable accuracy.

## 20. Hidden-state selector v3 — prospective evaluation (PREREG_SELECTOR_V3)

`paper/v3_prospective_eval.{json,md}`; n = 300 fresh states (manifest sha
`7bc5ec8a`), 0 transport errors, single pass. Frozen v3 (H + ridge30, sha
`4fd1fa4f`) vs frozen mixture and frozen v2.

- **Primary (½× strict, v3 − mixture) FAILED**: +1.7pp [−1.9, +5.1].
  ¼× +2.9 [−0.7, +6.3]; 1× +1.7 [−2.0, +5.6] — all CIs include 0.
- **The pre-stated budget-control instability materialized exactly**: v3's
  realized cost is ≈174-179 tok at EVERY nominal budget — λ(B) does not
  bind with hidden-state value heads (dev CV had flagged this before
  collection). The ¼×/½× point estimates are therefore cost-confounded
  (v3 spends 2.3×/1.1× the mixture's tokens) and are not claimed; the only
  cost-fair cell is 1×, where v3 is cheaper than the mixture (179 vs 241
  tok) at +1.7pp ns strict / +3.5pp [−0.0, +7.2] borderline lenient, and
  v3 ≈ v2 (+0.2pp ns).
- Secondary v3 − v2 at ¼× (+4.5 [+1.0, +8.1]) is likewise cost-confounded
  (174 vs 53 tok) and not claimed as a win.
- **Reading**: the representation-based controller does not close the
  opportunity-realization separation either — "the failure is not unique to
  lightweight handcrafted features" is now CI-backed — and it surfaces a new
  finding: hidden-state value heads lose λ-budget controllability, a
  deployability failure mode the handcrafted selector does not have.
  Depth-round experiment queue is now COMPLETE (adjudication declined).

## Infrastructure notes (for the record)

- 2026-08-28 engine death poisoned 183 VQAv2 held-out states with
  APIConnectionError rollouts (collect logged instead of aborting); stripped
  (backup `.bak_apierr`) and re-collected before any evaluation. All drivers
  now carry a zero-transport-error integrity gate.
- Frozen sklearn-1.9.0 pickles evaluated under 1.7.2 via a predict_proba
  attribute shim, asserted equal to sigmoid(frozen coefficients) to 1e-16.
- Human audit withdrawn without execution (`PREREG_HUMAN_AUDIT.md`).
- Phi-4 weights were re-downloaded (HF cache had been purged); serving config
  reproduced from the 08-22 log (`scripts/serve_phi4.sh`).
- 2026-09-01: `textvqa_prospective.jsonl` was accidentally regenerated with a
  changed exclusion pool (an old builder script re-ran); restored immediately
  by reconstructing the original 950-id exclusion set — restored file matches
  the preregistered sha `a535d8edce3aa822` exactly and its ids match the
  collected log 600/600. No analysis touched the corrupted file.

> **Addendum 2026-09-20 — reviewer-robustness round (advisor's 3+1 list).**
> Authorized additions: (1) stronger deployable controller on a fresh set,
> (2) replay-pairwise robustness, (3) human semantic adjudication of
> strict/lenient disagreements — **DECLINED by the user 2026-09-20 (second
> time; no sheets created). Consequence per the advisor's note: the title's
> "Evaluation Confounds" should be softened to "Evaluation Sensitivity"
> unless independent adjudication is added.**
> (4) cross-reader agreement conditional on baseline capability. (2) and (4)
> are zero-inference and complete below (§21-§22); (1) is frozen and
> preregistered, collection queued behind GPU availability (§23).

## 21. Cross-reader agreement conditional on baseline capability (zero inference)

`paper/cross_reader_conditional.{json,md}`, script
`scripts/cross_reader_conditional.py`. Same Gate-0 states/views, Qwen vs
Phi-4; strata by whether each reader's STOP (mean over replays) is correct.

TextVQA strict (pooled Jaccard .19, κ −.05):

| stratum | n | P(help) Q / P4 | Jaccard [CI] | κ [CI] | corr(PPU) [CI] |
|---|--:|:--|:--|:--|:--|
| both STOP wrong | 70 | .73 / .33 | **.35 [.23, .47]** | .11 [−.05, .28] | **.31 [.14, .47]** |
| both STOP correct | 53 | .08 / .19 | .08 [.00, .25] | .04 | −.01 |
| mixed | 77 | .14 / .52 | .04 [.00, .10] | −.19 [−.34, −.04] | −.29 [−.40, −.17] |

TextVQA lenient (Phi-4 STOP .24 → .50; pooled Jaccard .34, κ .27, corr .35):
both-wrong n=51 Jaccard **.64 [.50, .77]**, κ .12 [−.14, .39], corr(PPU)
**.48 [.25, .69]**; both-correct n=105 Jaccard .16, corr .18. MME strict:
both-wrong n=92 Jaccard .40 [.26, .55], κ .36 [.17, .54], corr .42 [.22, .59]
(pooled .33 / .22 / .21). Per-family PPU correlation is highest for OCR in the
both-wrong stratum (TextVQA .47 strict / .52 lenient).

- Reading: a large share of the pooled disagreement is a capability
  artifact — conditional on both readers failing at STOP, the readers agree
  on *where* opportunity is far more often (Jaccard .35-.64) and on *how
  much* (corr .31-.48, CIs exclude 0). But chance-corrected agreement (κ)
  inside the both-wrong stratum stays ≈ .1 on TextVQA, because P(help) is
  high for both readers there; the "instance-specific" claim should be
  stated as "moderate overlap, weak beyond-base-rate agreement", not as
  independence. The mixed stratum shows the expected anti-correlation
  (opportunity belongs to whichever reader failed).

## 22. Replay-pairwise robustness (zero inference)

`paper/replay_pairwise_robustness.{json,md}`, script
`scripts/replay_pairwise_robustness.py`. Gate-0 replays 1-2 + the
PREREG_MULTI_REPLAY replays 3-5 (100-state outcome-blind subsets; replay 3
= independent rewrite "In the image shown: {q}"); every ordered pair (r→s).

TextVQA (n=100): headroom R1→R2 +.023 [−.050, +.100], R1→R3 +.067 [−.010,
+.143], R2→R3 **+.133 [+.063, +.207]**; reverses +.110 / +.120 / +.077.
All 20 ordered pairs positive (mean +.110, sd .035, 16/20 CIs exclude 0);
the published R1→R2 cell is the *smallest* of the 20 on this subset. MME
(n=100): all 20 positive (mean +.070, sd .044), R1→R2 +.110 among the
larger cells. Consistency (TextVQA): STOP agreement .92-.99 across pairs,
per-arm outcome correlation .88-.96 (R1-R2 .884, R1-R3 .926, R2-R3 .933),
binarized arm agreement .95-.98, PPU sign agreement among both-nonzero
pairs .90-1.00. K=2 on the larger logs: prospective 600 STOP agree .913,
arm corr .878 [.853, .899], headroom R1→R2 +.088 [+.056, +.120] and
R2→R1 +.086 [+.054, +.117]; v3-300 +.102 / +.108; Gate-0 200 +.073 / +.123.

- Reading: the headroom is not a property of the specific R1→R2 rewrite —
  the independent rewrite gives the same or larger headroom, replay
  outcomes are highly consistent, and the published protocol sits at the
  conservative end. n=100 for the 5-replay grid; the 400-state collection
  in §23 carries replay 3 so the R1/R2/R3 triangle is re-estimated at 4×.

## 23. Stronger deployable controllers — prospective test COMPLETE (1 pass)

Prereg `paper/PREREG_STRONG_CONTROLLER.md`; artifact
`models/strong_controllers_frozen.{pkl,json}` (sha `5c60ccab3f7d08b8`);
manifest `data/manifests/textvqa_strong.jsonl` (400 fresh states, sha
`7312457434be55f3`, 1,400 used val ids excluded). Same 16 F0 inputs, same
targets, same argmax[Δ̂ − λc] rule; head classes ridge / xgboost / MLP;
training pool 1,350 TextVQA states (Gate-0 + held-out + prospective + v3),
ridge refit on the same pool (capacity contrast is pool-matched); frozen v2
kept as the published reference. Pre-collection nested-CV at ½×: ridge
.732, xgb .730, mlp .719 — nonlinear heads do not beat ridge on 3.4× the
data. Primary endpoint: xgb − mixture, ½× strict; co-secondary xgb/mlp −
ridge. Collection (`scripts/run_strong_controller.sh`, replays 1,2,3) is
launched automatically by `scripts/wait_gpu_and_run.sh` when a GPU has
≥38 GB free (both GPUs occupied by other users' jobs at freeze time);
`scripts/strong_post.sh` runs the evaluation once. Also recorded: the v2
candidate grid already contained GBR/RF heads and the v3 grid an MLP head,
all below ridge in dev CV — disclosed in the frozen manifests but never a
headline; §23 makes it one.

**Result (2026-09-20 13:04; `paper/strong_controller_eval.{json,md}`,
log `logs/strong_textvqa.jsonl`, 400/400 states, 13,200 rollouts, 0
transport errors, 95 min).** Strict, n=400, always-HIGHRES 308 tok:

| budget | ridge (vtok) | xgb (vtok) | mlp (vtok) | v2 | mixture | xgb − mix [CI] | xgb − ridge [CI] | ridge − mix [CI] | oracle |
|---|---|---|---|--:|--:|:--|:--|:--|--:|
| ¼× | .691 (61) | .701 (54) | .691 (78) | .675 | .708 | −.006 [−.036, +.023] | +.010 [−.012, +.033] | −.017 [−.045, +.012] | .812 |
| ½× | .728 (141) | .726 (148) | .716 (160) | .699 | .713 | **+.013 [−.014, +.041]** | −.003 [−.020, +.016] | +.016 [−.012, +.044] | .812 |
| 1× | .729 (210) | .731 (216) | .727 (225) | .704 | .704 | +.027 [−.003, +.058] | +.002 [−.014, +.018] | +.025 [−.007, +.057] | .838 |

- **Primary endpoint (xgb − mixture, ½× strict) FAILED**: +1.3pp
  [−1.4, +4.1]. Reported as a failure.
- **Capacity contrast is null at every budget**: xgb − ridge −0.3 / +1.0 /
  +0.2pp, mlp − ridge −1.3 / 0.0 / −0.2pp, all CIs include 0 and are
  narrow (±2pp). The pool-refit ridge is as good as the nonlinear heads.
- The 3.4×-larger training pool helps the *published* v2 by +2~3pp
  (ridge-pool .728 vs v2 .699 at ½×) but the resulting policy still does
  not clear the mixture under strict scoring.
- Lenient (same generations): xgb − mixture **+3.3 [+0.7, +6.1]** (½×) and
  **+4.4 [+1.4, +7.5]** (1×), ridge − mixture +3.2 [+0.5, +5.9] / +4.8
  [+1.8, +7.8]; xgb − ridge still null (+0.1 / −0.3pp). The recurring
  strict-parity / lenient-positive pattern reproduces a third time on a
  fresh set, identically for linear and nonlinear heads.
- Budget control held for all three classes (realized 54-78 / 141-160 /
  210-225 tok at ¼/½/1×) — no v3-style λ instability with F0 inputs.
- Realizability gap persists: oracle .812 vs best deployable .728 (½×).
- **Reading (the reviewer's question answered)**: the opportunity–
  realization separation is *not* a controller-capacity artifact — with
  identical inputs, targets and decision rule, gradient-boosted and MLP
  heads trained on 1,350 states match ridge to within ±1pp on 400 fresh
  states, and none converts the +8~9pp oracle headroom into a strict-scored
  advantage over the state-independent mixture. Headline table for the
  paper: Ridge / XGBoost / MLP / Mixture / Oracle, this section.

**Replay triangle at n=400 (`paper/replay_pairwise_strong.{json,md}`,
replays 1-3 of the same log; §22 extension):** headroom R1→R2 +.095
[+.055, +.135], R1→R3 +.103 [+.068, +.138], R2→R3 +.113 [+.082, +.147]
(reverses +.095 / +.102 / +.102; all six CIs exclude 0, spread 1.8pp).
STOP agreement .945 / .955 / .980, per-arm outcome correlation .897 /
.922 / .933, PPU sign agreement among both-nonzero pairs .97 / .98 / 1.00.
The independent rewrite reproduces the published R1→R2 headroom to within
1pp at 4× the earlier sample — the "rewrite artifact" objection is closed
with CIs.

GPU released 13:02 (server shut down by the driver; no PPU process left on
either GPU).

## 24. Hidden-state × nonlinear controllers — prospective test COMPLETE (1 pass; advisor priority-1, 2026-09-21)

Prereg `paper/PREREG_HIDDEN_CONTROLLER.md` (freeze record appended before
collection); artifact `models/hidden_controllers_frozen.{pkl,json}` (sha
`ed93f19437264e56`; 9 cells {F0, H, H+F0} × {ridge, xgb, mlp}, pool n=1,750
= Gate-0 + held-out + prospective + v3 + strong; primary cell fixed by
nested CV = **H/mlp**); fresh set `data/manifests/textvqa_hidden.jsonl`
(400 states, sha `af41a4ac740a9ecb`, 1,800 used ids excluded); log
`logs/hidden_textvqa.jsonl` (8,800 rollouts, 0 transport errors, 59 min;
hidden states of the fresh set extracted after collection). Results
`paper/hidden_controller_eval.{json,md}`. Pipeline needed three launches
(GPU taken by other users mid-run; server memory 0.36 / max_len 8192);
nothing was retrained or re-evaluated.

Strict, ½× (always-HIGHRES 307 tok; oracle .808; mixture .714):

| cell | F0 | H | H+F0 |
|---|--:|--:|--:|
| ridge | **.749** (146) | .729 (163) | .732 (161) |
| xgb | .737 (141) | .705 (151) | .724 (131) |
| mlp | .744 (122) | .722 (171) | .707 (142) |

- **Primary endpoint (H/mlp − mixture, ½× strict) FAILED**: +0.8pp
  [−2.1, +3.8]. H/xgb − mixture −0.8 [−3.6, +2.0]. Reported as failures.
- **Representation contrast is NEGATIVE**: H/xgb − F0/xgb −3.1pp
  [−5.8, −0.6] (¼× −5.9 [−8.8, −3.0]; 1× −2.5 [−4.8, −0.2]); H/ridge −
  F0/ridge −1.9 [−4.2, +0.4]; H+F0/xgb − F0/xgb −1.3 [−3.3, +0.7]. The
  3584-d hidden state is a worse controller input than 16 surface
  features on fresh states, at every head class.
- **Capacity contrast on H is null-to-negative**: H/xgb − H/ridge −2.4
  [−4.8, −0.1]; H/mlp − H/ridge −0.8 [−3.3, +1.6].
- Secondary (not the preregistered primary; 12 contrasts × 3 budgets,
  multiplicity applies): the pool-1,750 **F0/ridge beats the mixture at ½×
  strict, +3.5pp [+0.8, +6.4]**, and F0/xgb at 1× +3.1 [+0.2, +6.0]; on the
  09-20 strong set the same F0/ridge contrast was +1.6 [−1.2, +4.4] ns.
  Read as: the surface-feature linear controller is at the edge of a
  strict-scored advantage once trained on ~1.7k states; it is not the
  representation or the head class that was missing.
- Lenient (same generations): every cell beats the mixture at ½× (+2.8 to
  +5.2pp, all CIs exclude 0), H − F0 contrasts null (−0.0 to −0.2pp). The
  strict-parity / lenient-positive pattern reproduces a fourth time.
- Budget control held for all nine cells (realized 41-73 / 122-171 /
  162-237 tok) — the v3 λ-instability does not recur with pooled OOF λ.
- **Reading (reviewer's question answered)**: the opportunity–realization
  separation is not caused by a weak representation or a weak head.
  Strong representation × strong head (H/xgb, H/mlp) does not beat the
  state-independent mixture under strict scoring on 400 fresh states and
  is significantly worse than the same head on surface features. Combined
  with §23, the full 3 × 3 grid is now on fresh data: the best deployable
  cell is the simplest one (F0/ridge), and the oracle headroom (+8~9pp
  over it) remains unrealized.

GPU released 10:55 (server shut down by the driver; no PPU process left).

## 25. Deployable-action restricted oracle (zero inference; 2026-09-21)

`paper/deployable_oracle.{json,md}`, script `scripts/deployable_oracle.py`.
Robustness check of the structural oracle: the audit's oracle selects among
the individual arms {STOP, CROP#0..7, OCR, HIGHRES}; the selectors choose a
family with CROP = uniform-grid expectation. The oracle is recomputed on
exactly that deployable space A^dep = {STOP, CROP-family, OCR, HIGHRES}
(family outcome/cost = the selectors' uniform-grid expectation), same
protocol (select on R1, ties → cheaper, score on R2), same cross-fitted
best-fixed baseline (audit rule, 11 fixed arms), so H_full − H_dep is purely
the oracle's action space. Deterministic arms only.

Strict, headroom H = oracle − best fixed (pp), unconstrained:

| set | n | H_full [CI] | H_dep [CI] | full − dep [CI] | dep/full |
|---|--:|:--|:--|:--|--:|
| TextVQA Gate-0 (Qwen) | 200 | +8.9 [+4.0, +13.9] | +4.6 [−0.1, +9.3] | +4.3 [+1.7, +7.1] | .51 |
| MME Gate-0 (Qwen) | 196 | +8.2 [+3.3, +13.2] | +3.0 [−1.1, +7.2] | +5.2 [+2.1, +8.5] | .37 |
| TextVQA held-out | 250 | +7.9 [+4.2, +11.6] | +4.4 [+1.1, +7.6] | +3.5 [+1.8, +5.5] | .56 |
| TextVQA prospective | 600 | +9.8 [+7.1, +12.6] | **+5.6 [+3.1, +8.2]** | +4.2 [+2.8, +5.7] | .57 |
| TextVQA v3-prospective | 300 | +11.4 [+8.0, +15.1] | +6.7 [+4.0, +9.4] | +4.7 [+2.5, +7.1] | .59 |
| TextVQA strong | 400 | +9.4 [+6.3, +12.5] | +6.3 [+3.6, +9.0] | +3.1 [+1.2, +5.0] | .67 |
| TextVQA hidden | 400 | +7.8 [+4.7, +10.9] | +4.1 [+1.5, +6.8] | +3.7 [+2.1, +5.4] | .53 |
| TextVQA Gate-0 (Phi-4) | 200 | +9.9 [+5.7, +13.9] | +7.5 [+4.1, +10.8] | +2.4 [+0.4, +4.5] | .76 |
| MME Gate-0 (Phi-4) | 196 | +2.3 [−2.9, +7.6] | +1.5 [−3.0, +6.1] | +0.8 [−1.4, +2.9] | .66 |

Budget-constrained (Lagrangian, symmetrised) oracle accuracy at ½×, full /
dep: prospective .819 / .777, v3 .816 / .772, strong .810 / .777, hidden
.809 / .777 (strict). Lenient: dep/full ratios .43–.70, same pattern.

- **Reading (the advisor's middle case)**: restricting the oracle to the
  deployable decision space removes ~40–50% of the headroom on TextVQA
  (~35–65% on MME), but a family-level headroom of **+4 to +7pp remains,
  CIs excluding 0 on every fresh TextVQA set (n ≥ 250)**; on the n≈200
  Gate-0 sets the dep CIs touch 0. Roughly half of the structural
  opportunity is crop granularity (which box), which the paper already
  shows is not question-predictable (VLM proposals ≈ grid, +0.8pp); the
  other half is family choice, which IS in the selectors' action space and
  remains unrealized: at ½× strict the best deployable cell (F0/ridge, §24)
  reaches .749 on the hidden set against a dep-oracle .777 and full-oracle
  .809 — i.e. the selector closes most of the *family-level* gap and
  none of the crop-granularity gap.
- Framing consequence: keep "opportunity–realization gap" but quantify it
  in two layers — total (11-arm) headroom ≈ +8–10pp, of which ≈ +4–7pp is
  reachable by a family-level controller and ≈ +3–5pp requires region
  selection. The oracle–policy gap that a family-level controller could in
  principle close is ~3pp at ½×, not ~8pp.
- Cross-check: the deterministic-arm full oracle reproduces the published
  Gate-0 headroom (+8.9 / +8.2 vs +9.8 / +9.8 with VLM proposals included).

> **Addendum 2026-09-21 — human semantic adjudication REOPENED (user decision,
> reversing 09-20).** Prereg `paper/PREREG_HUMAN_ADJUDICATION.md` fixed before
> any verdict; blinded package built: 300 items = 200 strict-0 /
> format-tolerant-positive disagreements of the prospective 600 set
> (stratified by dominant family) + 50 both-correct + 50 both-wrong
> controls; mapping sha `1384f99672bc8d65`. Three independent annotators,
> majority label, Fleiss' κ / Krippendorff's α; key numbers = strict
> false-negative share and lenient false-positive share among disagreements.
> `scripts/human_adjudication_analysis.py` runs once after the last export.
> Results will be §26.

## 26. Human semantic adjudication of scorer disagreements — COMPLETE (1 pass; 2026-09-21)

Prereg `paper/PREREG_HUMAN_ADJUDICATION.md`; package
`scripts/make_adjudication_package.py` / `make_adjudication_csv.py` (300
items: 200 strict-0 / format-tolerant-positive disagreements of the
prospective 600 set, stratified by dominant family, + 50 both-correct + 50
both-wrong controls; mapping sha `1384f99672bc8d65`); three independent
annotators (`data/human_adjudication/annotator{1,2,3}.csv`, md5
`d1ced72c…` / `24f29c79…` / `82d06afc…`); analysis
`scripts/human_adjudication_analysis.py` run once → `paper/human_adjudication.{json,md}`.
Process note: the first file drop contained a duplicated file (annotator1
== annotator3 byte-for-byte); the analysis was NOT run on it, the user
supplied the correct third file, and the analysis ran once on three
distinct files.

- **Agreement**: raw pairwise .906 (mean), Cohen's κ .70 / .88 / .72, **Fleiss'
  κ .766, Krippendorff's α .767**; majority label on 292/300 (unanimous 258),
  ambiguous-majority 7, no-majority 1.
- **Key numbers (n=197 disagreement items with a C/I majority)**:
  **P(human correct | strict=0, lenient>0) = .807 [.746, .856]** (strict
  false-negative share); **P(human incorrect | …) = .193 [.144, .254]**
  (lenient false-positive share). By dominant family: STOP .857, CROP .721,
  OCR .854, HIGHRES .898. By format-tolerant value: **1.0 → .877 (n=154);
  2/3 → .538 (n=13); 1/3 → .567 (n=30)** — the lenient false positives
  concentrate in partial-hit items (long list-type answers such as "what
  are some of the comics?" and near-miss brand names).
- **Controls**: both-correct 50/50 judged correct by the majority; both-wrong
  45 with a majority: 34 incorrect, **11 judged correct** (synonyms /
  paraphrases neither scorer catches) + 5 ambiguous. Population-weighted
  agreement with the human label over the three strata (1,495 items):
  strict .758 [.699, .812], format-tolerant .856 [.797, .909].
- **Policy contrast under the human-corrected scorer** (600 states; strict
  everywhere except the 197 adjudicated items = 659 rollouts; unannotated
  disagreements strict = lower bound / lenient = upper bound):

| scorer | ¼× sel − mix | ½× sel − mix | 1× sel − mix | oracle ½× |
|---|:--|:--|:--|--:|
| strict | +0.2 [−2.3, +2.7] | +0.9 [−1.6, +3.4] | +0.5 [−2.2, +3.2] | .818 |
| **human (lower bound)** | **+2.5 [+0.2, +4.9]** | **+4.1 [+1.8, +6.5]** | **+3.0 [+0.5, +5.5]** | .864 |
| human (upper bound) | +3.0 [+0.7, +5.3] | +4.8 [+2.6, +7.0] | +3.3 [+0.9, +5.7] | .900 |
| lenient | +3.0 [+0.7, +5.2] | +4.6 [+2.4, +6.7] | +3.0 [+0.7, +5.4] | .897 |

- **Reading (pre-declared template: "mostly correct", with a real minority
  of lenient false positives)**: where the scorers disagree, four in five
  generations are semantically correct — the strict scorer over-penalizes
  answer formatting — while one in five is a format-tolerant false
  positive, concentrated in partial-token matches. The selector's advantage
  over the mixture that strict scoring masks is **real under human
  judgment: +4.1pp [+1.8, +6.5] at ½× even on the conservative bound**, so
  the measurement-sensitivity claim can now be stated as a semantic-
  validity claim rather than "two arbitrary scorers". Practical corollary
  for the paper: report the format-tolerant rule only at full-match (value
  1), where it agrees with humans 88% of the time; partial hits should not
  count. The title's "Evaluation Confounds" is now supportable.

> **Addendum 2026-09-21 (2) — optimistic-bound definition aligned with the main
> paper.** The §26 "upper bound" originally assigned the format-tolerant score
> wherever it flipped the binarized outcome (≥ 1/2). The paper defines the
> optimistic assumption as *full-credit format-tolerant matches only* (1/3 and
> 2/3 partial matches stay incorrect). `human_adjudication_analysis.py` was
> re-run with that rule (primary quantities unchanged): selector − mixture
> under the optimistic bound = **+2.7 [+0.4, +5.0] / +4.2 [+2.0, +6.5] /
> +2.9 [+0.5, +5.4]** at ¼×/½×/1× (was +3.0 / +4.8 / +3.3 under the ≥1/2
> rule, which is kept in the JSON as `human_sens_any` for transparency).
> Pessimistic bound unchanged: +2.5 / +4.1 / +3.0. Main-paper text quoting
> +4.8 [+2.6, +7.0] must be updated to +4.2 [+2.0, +6.5].
