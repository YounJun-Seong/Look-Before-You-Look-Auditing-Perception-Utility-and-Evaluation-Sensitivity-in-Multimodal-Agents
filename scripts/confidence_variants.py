#!/usr/bin/env python3
"""Confidence-statistic variants from stored per-generation statistics (post-hoc).

Reviewer question: the paper's confidence is the mean token log-probability of
the whole generated reply, scaffold included -- does the VQAv2 inversion (and
its repair under the one-word prompt) survive a statistic that is less tied to
reply format?

The logs store two per-generation statistics (schema 0.3.0): mean_logprob and
min_logprob. min_logprob is not scaffold-diluted the way the mean is, but it is
*length-sensitive* in the opposite direction (a minimum over more tokens is
stochastically smaller, and verbose replies are exactly the flip-prone ones),
so it is a bracketing check, not a clean answer-span confidence. The clean
variants (answer-span mean/geometric mean, first-answer-token) need a logprob
re-collection pass; see scripts/collect_answer_span_conf.py.

Strict scorer, HELP-vs-rest ROC (AMBIGUOUS excluded), same label pipeline as
scripts/format_confound_check.py. Outputs paper/confidence_variants.{json,md}.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

THR = 0.5
RUNS = [
    ("Qwen2.5-VL-7B", "textvqa", "logs/lockbox_textvqa.jsonl"),
    ("Qwen2.5-VL-7B", "vqav2", "logs/lockbox_vqav2.jsonl"),
    ("Qwen2.5-VL-7B (one-word)", "vqav2", "logs/lockbox_vqav2_qwen_oneword.jsonl"),
    ("Phi-4-MM-5.6B", "textvqa", "logs/lockbox_textvqa_phi4.jsonl"),
    ("Phi-4-MM-5.6B", "vqav2", "logs/lockbox_vqav2_phi4.jsonl"),
    ("FRESH Qwen original", "vqav2", "logs/fresh_vqav2_orig.jsonl"),
    ("FRESH Qwen one-word", "vqav2", "logs/fresh_vqav2_oneword.jsonl"),
    ("FRESH Qwen original", "textvqa", "logs/fresh_textvqa_orig.jsonl"),
    ("FRESH Qwen one-word", "textvqa", "logs/fresh_textvqa_oneword.jsonl"),
]


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0:
        return "HELP"
    if d1 < 0 and d2 < 0:
        return "HARM"
    if d1 == 0 and d2 == 0:
        return "NEUTRAL"
    return "AMBIGUOUS"


def load(log):
    by = defaultdict(dict)
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") == "rollout":
            by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r
    return by


def roc_ci(y, s, rng, n=3000):
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(roc_auc_score(y[i], s[i]))
    return [float(q) for q in np.quantile(v, [0.025, 0.975])]


def main():
    rng = np.random.default_rng(20260828)
    out = []
    for bb, src, log in RUNS:
        if not (ROOT / log).exists():
            print("skip", log)
            continue
        rows = []
        for sid, d in load(log).items():
            need = [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]
            if not all(k in d for k in need):
                continue
            sc = {k: float(d[k]["score_after"]) for k in need}
            cf = [d[("STOP", r)].get("confidence") or {} for r in (1, 2)]
            if any(c.get("mean_logprob") is None or c.get("min_logprob") is None
                   for c in cf):
                continue
            rows.append({
                "label": four_way(sc[("STOP", 1)], sc[("STOP", 2)],
                                  sc[("HIGHRES", 1)], sc[("HIGHRES", 2)]),
                "mean_lp": float(np.mean([c["mean_logprob"] for c in cf])),
                "min_lp": float(np.mean([c["min_logprob"] for c in cf])),
                "verbose": len((d[("STOP", 1)]["answer_after"] or "").split()) > 1,
            })
        use = [r for r in rows if r["label"] != "AMBIGUOUS"]
        y = np.array([r["label"] == "HELP" for r in use], int)
        if not (0 < y.sum() < len(y)):
            continue
        entry = {"backbone": bb, "regime": src, "log": log, "n_eval": len(use),
                 "n_help": int(y.sum()),
                 "labels": dict(Counter(r["label"] for r in rows))}
        for stat in ("mean_lp", "min_lp"):
            s = -np.array([r[stat] for r in use])  # low confidence -> high score
            entry[stat] = {"roc": float(roc_auc_score(y, s)),
                           "ci": roc_ci(y, s, rng)}
        # length-artifact probe: does min_lp still rank HELP within one-word
        # STOP answers, where length cannot explain it?
        ow = [i for i, r in enumerate(use) if not r["verbose"]]
        y1 = y[ow]
        if 0 < y1.sum() < len(y1):
            for stat in ("mean_lp", "min_lp"):
                s1 = -np.array([use[i][stat] for i in ow])
                entry[f"{stat}_oneword_only"] = {
                    "n": len(ow), "n_help": int(y1.sum()),
                    "roc": float(roc_auc_score(y1, s1))}
        out.append(entry)
        print(f"{bb:28s} {src:8s} n={len(use):3d} HELP={y.sum():3d} "
              f"mean {entry['mean_lp']['roc']:.3f} min {entry['min_lp']['roc']:.3f}")

    Path(ROOT / "paper/confidence_variants.json").write_text(json.dumps(out, indent=1))

    lines = [
        "# Confidence-statistic variants (post-hoc, stored statistics only)",
        "",
        "HELP-vs-rest ROC, strict scorer, AMBIGUOUS excluded. `mean_lp` is the",
        "paper's confidence (whole-reply mean token logprob, averaged over both",
        "STOP replays); `min_lp` is the whole-reply minimum token logprob. The",
        "one-word-only columns restrict to states whose STOP answer is a single",
        "word, where reply length cannot drive either statistic.",
        "",
        "| backbone / regime | n (HELP) | mean-lp ROC [CI] | min-lp ROC [CI] | mean-lp ROC, one-word STOP only | min-lp ROC, one-word STOP only |",
        "|---|--:|:--:|:--:|:--:|:--:|",
    ]
    for e in out:
        def f(d):
            return f"{d['roc']:.3f} [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]"

        def g(key):
            d = e.get(key)
            return f"{d['roc']:.3f} (n={d['n']}, H={d['n_help']})" if d else "—"
        lines.append(f"| {e['backbone']} / {e['regime']} | {e['n_eval']} ({e['n_help']}) | "
                     f"{f(e['mean_lp'])} | {f(e['min_lp'])} | "
                     f"{g('mean_lp_oneword_only')} | {g('min_lp_oneword_only')} |")
    lines += ["",
              "Caveat: min_lp over the whole reply is length-sensitive in the opposite",
              "direction from the mean (minima over more tokens are stochastically",
              "smaller, and verbose replies are the flip-prone ones), so agreement",
              "between the two statistics brackets the format sensitivity rather than",
              "eliminating it. Answer-span confidence variants require a logprob",
              "re-collection pass (scripts/collect_answer_span_conf.py)."]
    Path(ROOT / "paper/confidence_variants.md").write_text("\n".join(lines) + "\n")
    print("wrote paper/confidence_variants.{json,md}")


if __name__ == "__main__":
    main()
