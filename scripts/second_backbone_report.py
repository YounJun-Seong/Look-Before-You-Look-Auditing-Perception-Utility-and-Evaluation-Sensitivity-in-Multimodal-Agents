#!/usr/bin/env python3
"""Second-backbone format intervention on fresh VQAv2 (post-hoc).

Is the below-chance confidence regime and its one-word repair a Qwen-specific
prompting pathology? Same fresh VQAv2 states, same views, same protocol, run
with LLaVA-1.6-Mistral-7B (non-Qwen LM and vision stack), original vs one-word
prompt. Compares against the fresh Qwen rows. Strict scorer as logged.

Outputs paper/second_backbone.{json,md}.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

THR = 0.5
RUNS = [
    ("Qwen2.5-VL-7B / original", "logs/fresh_vqav2_orig.jsonl"),
    ("Qwen2.5-VL-7B / one-word", "logs/fresh_vqav2_oneword.jsonl"),
    ("LLaVA-1.6-Mistral-7B / original", "logs/fresh_vqav2_llava_orig.jsonl"),
    ("LLaVA-1.6-Mistral-7B / one-word", "logs/fresh_vqav2_llava_oneword.jsonl"),
    ("LLaVA (coarse 16) / original", "logs/fresh_vqav2_llava_c16_orig.jsonl"),
    ("LLaVA (coarse 16) / one-word", "logs/fresh_vqav2_llava_c16_oneword.jsonl"),
]
YN_Q = re.compile(r"^\s*(is|are|does|do|was|were|can|could|has|have|did)\b", re.I)


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


def main():
    rng = np.random.default_rng(20260828)
    out = []
    for disp, log in RUNS:
        if not (ROOT / log).exists():
            print("missing", log)
            continue
        by = defaultdict(dict)
        n_parse_fail = 0
        for line in open(ROOT / log):
            r = json.loads(line)
            if r.get("type") != "rollout":
                continue
            by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r
            if not r.get("parse_ok", True):
                n_parse_fail += 1
        rows = []
        for sid, d in by.items():
            need = [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]
            if not all(k in d for k in need):
                continue
            sc = {k: float(d[k]["score_after"]) for k in need}
            st = d[("STOP", 1)]
            sa = st["answer_after"] or ""
            ha = d[("HIGHRES", 1)]["answer_after"] or ""
            cf = [d[("STOP", r)].get("confidence") or {} for r in (1, 2)]
            if any(c.get("mean_logprob") is None for c in cf):
                continue
            rows.append({
                "label": four_way(sc[("STOP", 1)], sc[("STOP", 2)],
                                  sc[("HIGHRES", 1)], sc[("HIGHRES", 2)]),
                "stop": (sc[("STOP", 1)] + sc[("STOP", 2)]) / 2,
                "hi": (sc[("HIGHRES", 1)] + sc[("HIGHRES", 2)]) / 2,
                "verbose": len(sa.split()) > 1, "short_hi": len(ha.split()) <= 1,
                "mean_lp": float(np.mean([c["mean_logprob"] for c in cf])),
                "wh_q": not YN_Q.match(st["question"] or ""),
                "stop_is_yn": sa.strip().lower() in ("yes", "no"),
            })
        use = [r for r in rows if r["label"] != "AMBIGUOUS"]
        y = np.array([r["label"] == "HELP" for r in use], int)
        helps = [r for r in rows if r["label"] == "HELP"]
        entry = {
            "run": disp, "log": log, "n_states": len(rows), "n_eval": len(use),
            "parse_fail_rollouts": n_parse_fail,
            "labels": dict(Counter(r["label"] for r in rows)),
            "stop_acc": float(np.mean([r["stop"] for r in rows])),
            "highres_acc": float(np.mean([r["hi"] for r in rows])),
            "verbose_stop_share": float(np.mean([r["verbose"] for r in rows])),
            "help_prev": (len(helps) / len(use)) if use else None,
            "help_format_flip_share": (float(np.mean(
                [r["verbose"] and r["short_hi"] for r in helps])) if helps else None),
        }
        wh = [r for r in rows if r["wh_q"]]
        entry["ynno_collapse_share"] = (float(np.mean([r["stop_is_yn"] for r in wh]))
                                        if wh else None)
        if 0 < y.sum() < len(y):
            s = -np.array([r["mean_lp"] for r in use])
            entry["conf_roc"] = float(roc_auc_score(y, s))
            v = []
            for _ in range(3000):
                i = rng.integers(0, len(y), len(y))
                if 0 < y[i].sum() < len(i):
                    v.append(roc_auc_score(y[i], s[i]))
            entry["conf_roc_ci"] = [float(q) for q in np.quantile(v, [0.025, 0.975])]
        out.append(entry)
        print(json.dumps(entry, indent=None)[:300])

    Path(ROOT / "paper/second_backbone.json").write_text(json.dumps(out, indent=1))
    lines = [
        "# Second backbone: format intervention on fresh VQAv2 (post-hoc)",
        "",
        "Same 250 fresh VQAv2 states, views, decoding, and strict scorer as the fresh",
        "Qwen runs; LLaVA-1.6-Mistral-7B served with the same pixel budgets",
        "(Qwen-token-equivalent). Confidence = whole-reply mean token logprob of STOP",
        "(mean over replays).",
        "",
        "| backbone / prompt | n | STOP | HIGHRES | verbose STOP | yes/no on wh-q | HELP (prev.) | HELP = flip | conf ROC [CI] |",
        "|---|--:|--:|--:|--:|--:|--:|--:|:--:|",
    ]
    for e in out:
        roc = (f"{e['conf_roc']:.3f} [{e['conf_roc_ci'][0]:.2f}, {e['conf_roc_ci'][1]:.2f}]"
               if e.get("conf_roc") is not None else "—")
        ff = ("—" if e["help_format_flip_share"] is None
              else f"{100 * e['help_format_flip_share']:.0f}%")
        hp = "—" if e["help_prev"] is None else \
             f"{e['labels'].get('HELP', 0)} ({100 * e['help_prev']:.1f}%)"
        yn = ("—" if e.get("ynno_collapse_share") is None
              else f"{100 * e['ynno_collapse_share']:.0f}%")
        lines.append(
            f"| {e['run']} | {e['n_states']} | {100 * e['stop_acc']:.1f}% | "
            f"{100 * e['highres_acc']:.1f}% | {100 * e['verbose_stop_share']:.0f}% | "
            f"{yn} | {hp} | {ff} | {roc} |")
    lines += [
        "",
        "## Reading",
        "",
        "- **Qwen's pathology does not replicate on LLaVA.** Original-prompt LLaVA is",
        "  not verbose (6% vs 40%), has no format-flip HELP states, and its confidence",
        "  ROC is on the right side of chance — including at coarse 16, where a real",
        "  perceptual bottleneck exists (STOP 78.0% vs HIGHRES 84.7%, HELP 7.9%).",
        "  The below-chance regime is therefore not a property of the benchmark or",
        "  the budget alone; it is reader x prompt dependent.",
        "- **The repair is reader-dependent too.** The one-word prompt that fixes",
        "  Qwen (ROC .428 -> .897) *breaks* LLaVA: it answers yes/no on 38-40% of",
        "  wh-questions (vs 2% under the original prompt) and STOP accuracy drops",
        "  17pp. The LLaVA one-word rows are a compliance failure of the",
        "  intervention, not a valid utility measurement — report them as such.",
        "- Claim to draw: there is no universal prompt fix; utility validity must be",
        "  audited per reader/prompt configuration. Matches the Discussion's",
        "  prevalence caveat.",
        "",
        "Provenance: post-hoc; fresh VQAv2 set only (no lockbox contact); LLaVA runs",
        "collected 2026-08-28 via scripts/run_llava_format_check.sh and",
        "scripts/run_llava_c16.sh (5-state smoke gates passed; serial, greedy,",
        "seed 1234; logs/serve_llava*.log).",
    ]
    Path(ROOT / "paper/second_backbone.md").write_text("\n".join(lines) + "\n")
    print("wrote paper/second_backbone.{json,md}")


if __name__ == "__main__":
    main()
