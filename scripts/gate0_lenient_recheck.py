#!/usr/bin/env python
"""Re-verify the Gate 0 policy headroom (RQ1) under format-tolerant scoring and
symmetric cross-replay.

Reviewer gap: the paper shows that answer format / strict scoring can distort
measured perception utility, but never shows that the headline +9.8pp
P_policy on TextVQA survives that distortion.  This script re-runs the frozen
Gate 0 audit (ppu/audit.py, unchanged) on the ORIGINAL TextVQA rollouts
(logs/tv_200.jsonl: 200 states x 24 rollouts, 2 replays) after rescoring every
generation with the format-tolerant scorer from scripts/format_confound_check.py,
and additionally reports the cross-replay oracle in both directions
(choose on R1 -> score on R2, and choose on R2 -> score on R1) averaged per state.

Nothing is re-generated; nothing in audit.py changes.  Strict rescoring is
checked against the logged scores (must be 0 mismatches).

Outputs paper/gate0_lenient_recheck.{json,md}.
"""
from __future__ import annotations
import copy, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppu.audit import load, audit, oracle_cf, best_fixed_cf, _bootstrap_ci  # noqa: E402
from ppu.evaluator import normalize_answer  # noqa: E402
from ppu.replay import paraphrase, _REWRITES  # noqa: E402

LOG = ROOT / "logs/tv_200.jsonl"
N_BOOT = 10000


def vqa_strict(pred, golds):
    p = normalize_answer(pred or "")
    return min(1.0, sum(normalize_answer(g) == p for g in golds) / 3)


def vqa_lenient(pred, golds):
    p = normalize_answer(pred or "")
    toks = p.split()
    def hit(g):
        g = normalize_answer(g)
        return g != "" and (g == p or (toks and g == toks[0]) or f" {g} " in f" {p} ")
    return min(1.0, sum(hit(g) for g in golds) / 3)


recs = [json.loads(l) for l in open(LOG)]
recs = [r for r in recs if r.get("type") == "rollout"]
mism = sum(abs(vqa_strict(r["answer_after"], r["gold"]) - r["score_after"]) > 1e-9
           for r in recs if r.get("parse_ok", True))
assert mism == 0, f"strict rescoring does not reproduce the log: {mism} mismatches"


def rescored(scorer):
    out = []
    for r in recs:
        r2 = dict(r)
        r2["score_after"] = scorer(r["answer_after"], r["gold"]) if r.get("parse_ok", True) else 0.0
        out.append(r2)
    return out


def swap_replays(views):
    sw = {}
    for sid, v in views.items():
        w = copy.deepcopy(v)
        w.r1, w.r2 = v.r2, v.r1
        w.dummy_r1, w.dummy_r2 = v.dummy_r2, v.dummy_r1
        w.full_r1, w.full_r2 = v.full_r2, v.full_r1
        sw[sid] = w
    return sw


def p_vec(views):
    vs = sorted(views.values(), key=lambda v: v.sample_id)
    _, choice, per = best_fixed_cf(vs)
    return {v.sample_id: oracle_cf(v) - per[v.sample_id] for v in vs if v.sample_id in per}, choice


results = {}
for name, scorer in (("strict", vqa_strict), ("lenient", vqa_lenient)):
    views = load(rescored(scorer))
    a12 = audit(views, f"textvqa/{name}/R1->R2", n_boot=N_BOOT)
    a21 = audit(swap_replays(views), f"textvqa/{name}/R2->R1", n_boot=N_BOOT)
    p12, c12 = p_vec(views)
    p21, c21 = p_vec(swap_replays(views))
    ids = sorted(set(p12) & set(p21))
    sym = np.array([(p12[i] + p21[i]) / 2 for i in ids])
    lo, hi = _bootstrap_ci(sym, N_BOOT)
    # verbosity of STOP answers under this log (format exposure)
    stop_r1 = [r for r in recs if r["action"]["type"] == "STOP" and r["replay"] == 1]
    verbose = float(np.mean([len((r["answer_after"] or "").split()) > 1 for r in stop_r1]))
    results[name] = {
        "R1->R2": {"accuracy": a12["accuracy"], "headroom": a12["headroom"],
                   "harm": a12["harm"], "diversity": a12["diversity"],
                   "exclusive_solve": a12["exclusive_solve"], "reliability": a12["reliability"]},
        "R2->R1": {"accuracy": a21["accuracy"], "headroom": a21["headroom"], "harm": a21["harm"]},
        "symmetric": {"P_policy": float(sym.mean()), "P_ci": [lo, hi], "n": len(ids),
                      "oracle_cf": float(np.mean([a12["accuracy"]["oracle_cf"], a21["accuracy"]["oracle_cf"]])),
                      "best_fixed_cf": float(np.mean([a12["accuracy"]["best_fixed_cf"], a21["accuracy"]["best_fixed_cf"]])),
                      "best_fixed_choice": {"R1->R2": c12, "R2->R1": c21}},
        "verbose_stop_share": verbose,
    }

# paraphrase procedure audit: which rule fired for each of the 200 questions
qs = {r["sample_id"]: r["question"] for r in recs}
rule_hits = Counter()
for q in qs.values():
    q = q.strip()
    hit = next((pat.pattern for pat, _ in _REWRITES if pat.match(q)), "fallback: 'Based on the image: …'")
    rule_hits[hit] += 1
para_examples = [(q, paraphrase(q)) for q in list(qs.values())[:5]]
results["paraphrase_audit"] = {"rule_hits": dict(rule_hits), "examples": para_examples,
                               "n_rules": len(_REWRITES), "deterministic": True}

json.dump(results, open(ROOT / "paper/gate0_lenient_recheck.json", "w"), indent=1, default=float)

P = lambda x: f"{100*x:.1f}"
L = ["# RQ1 re-check: policy headroom under format-tolerant scoring and symmetric cross-replay", "",
     "TextVQA Gate 0 rollouts (`logs/tv_200.jsonl`, n=200, 24 rollouts/state, Qwen2.5-VL-7B, original prompt), "
     "re-audited with the frozen `ppu/audit.py`. strict = official vqa_acc as logged (reproduced, 0 mismatches); "
     "lenient = gold is the whole normalised answer, its first token, or a whole-word substring. "
     "Symmetric = per-state mean of (choose R1 → score R2) and (choose R2 → score R1) for both the oracle "
     "and the 2-fold cross-fitted best fixed policy; CI = paired bootstrap (10k).", "",
     "| scorer | direction | coarse STOP | FULL | best fixed (cf) | oracle (cf) | dummy oracle | H_net [CI] | **P_policy [CI]** | DMG |",
     "|---|---|--:|--:|--:|--:|--:|:--:|:--:|--:|"]
for name in ("strict", "lenient"):
    R = results[name]
    for d in ("R1->R2", "R2->R1"):
        a, h = R[d]["accuracy"], R[d]["headroom"]
        L.append(f"| {name} | {d} | {P(a['coarse_stop_r2'])} | {P(a['full_image_r2'])} | {P(a['best_fixed_cf'])} | "
                 f"{P(a['oracle_cf'])} | {P(a['oracle_dummy_cf'])} | {100*h['H_net']:+.1f} [{100*h['H_ci'][0]:+.1f}, {100*h['H_ci'][1]:+.1f}] | "
                 f"**{100*h['P_policy']:+.1f}** [{100*h['P_ci'][0]:+.1f}, {100*h['P_ci'][1]:+.1f}] | {P(R[d]['harm']['dmg_rate'])} |")
    s = R["symmetric"]
    L.append(f"| {name} | **symmetric** | | | {P(s['best_fixed_cf'])} | {P(s['oracle_cf'])} | | | "
             f"**{100*s['P_policy']:+.1f}** [{100*s['P_ci'][0]:+.1f}, {100*s['P_ci'][1]:+.1f}] | |")
L += ["", f"Verbose (multi-word) coarse STOP answers in this log: {100*results['strict']['verbose_stop_share']:.0f}%.", "",
      "## Decisive-state best-action distribution (strict labels, R1→R2)", "",
      "| scorer | n decisive | STOP | CROP | OCR | HIGHRES | exclusive OCR |", "|---|--:|--:|--:|--:|--:|--:|"]
for name in ("strict", "lenient"):
    dv = results[name]["R1->R2"]["diversity"]; ex = results[name]["R1->R2"]["exclusive_solve"]
    L.append(f"| {name} | {dv['n_decisive']} | " + " | ".join(P(dv['distribution'][f]) for f in ("STOP", "CROP", "OCR", "HIGHRES"))
             + f" | {P(ex['OCR'])} |")
pa = results["paraphrase_audit"]
L += ["", "## R2 paraphrase procedure (ppu/replay.py)", "",
      f"Deterministic, rule-based: {pa['n_rules']} regex rewrites applied in order, first match wins; otherwise "
      "the fallback prefix. No LLM, no sampling. Rule usage on these 200 questions:", ""]
for k, c in sorted(pa["rule_hits"].items(), key=lambda kv: -kv[1]):
    L.append(f"- `{k}`: {c}")
L += ["", "Examples:", ""] + [f"- {q} → {p}" for q, p in pa["examples"]]
open(ROOT / "paper/gate0_lenient_recheck.md", "w").write("\n".join(L))
print("\n".join(L))
