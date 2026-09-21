#!/usr/bin/env python
"""Answer-format confound check: does the HELP label / confidence relationship
survive when the scorer tolerates verbose answers?

Trigger: Phi-4-MM (always one-word answers) showed no confidence reversal on
VQAv2 and a STOP accuracy of 76% where Qwen2.5-VL scored 43%.  Qwen's gap is
answer FORMAT — 40% of its coarse VQAv2 answers are multi-word ("No, it's not
cold.") and the official VQA scorer (exact match after normalisation) gives
those 0.  HIGHRES then "helps" mostly by shortening the answer, not by seeing
more.

For every backbone x regime we rescore the SAME logged generations two ways:

    strict   official vqa_acc / ANLS as logged (reproduced bit-exactly)
    lenient  strict OR the gold appears as the first token / as a whole-word
             substring of the normalised prediction

and recompute: STOP/HIGHRES accuracy, 4-way labels, confidence ROC / lift@1x,
share of HELP states that are pure format flips (multi-word STOP -> one-word
HIGHRES), and — for Qwen — the frozen v2.2 gate's ROC under both label sets.

Nothing here modifies any model or log.  Outputs paper/format_confound.{json,md}.
"""
from __future__ import annotations
import json, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppu.evaluator import normalize_answer, anls  # noqa: E402
from ppu.gate_features import extract  # noqa: E402

FR = pickle.load(open(ROOT / "models/gate_frozen_v2.2_pointwise.pkl", "rb"))
DEV_CDF = FR["conf_transform"]["sorted_dev_conf"]
THR = 0.5
RUNS = [
    ("Qwen2.5-VL-7B", "textvqa", "logs/lockbox_textvqa.jsonl"),
    ("Qwen2.5-VL-7B", "vqav2", "logs/lockbox_vqav2.jsonl"),
    ("Phi-4-MM-5.6B", "textvqa", "logs/lockbox_textvqa_phi4.jsonl"),
    ("Phi-4-MM-5.6B", "vqav2", "logs/lockbox_vqav2_phi4.jsonl"),
    ("Phi-4-MM-5.6B (coarse 16)", "vqav2", "logs/lockbox_vqav2_phi4_c16.jsonl"),
    ("Qwen2.5-VL-7B (one-word prompt)", "vqav2", "logs/lockbox_vqav2_qwen_oneword.jsonl"),
    # fresh prospective lockbox (paper/PREREG_FRESH_LOCKBOX.md), 2026-08-23
    ("FRESH Qwen original", "vqav2", "logs/fresh_vqav2_orig.jsonl"),
    ("FRESH Qwen one-word", "vqav2", "logs/fresh_vqav2_oneword.jsonl"),
    ("FRESH Qwen original", "textvqa", "logs/fresh_textvqa_orig.jsonl"),
    ("FRESH Qwen one-word", "textvqa", "logs/fresh_textvqa_oneword.jsonl"),
]


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


def score(metric, pred, golds, lenient):
    if metric == "vqa_acc":
        return (vqa_lenient if lenient else vqa_strict)(pred, golds)
    # TextVQA in this repo is scored with vqa_acc too; keep ANLS path for safety
    return anls(pred or "", tuple(golds))


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0: return "HELP"
    if d1 < 0 and d2 < 0: return "HARM"
    if d1 == 0 and d2 == 0: return "NEUTRAL"
    return "AMBIGUOUS"


def load(log):
    by = defaultdict(dict)
    metric = None
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") == "rollout":
            by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r
    return by


def conf_pct(x):
    return float(1.0 - np.searchsorted(DEV_CDF, x) / len(DEV_CDF))


rng = np.random.default_rng(20260822)
res = {}
for bb, src, log in RUNS:
    if not (ROOT / log).exists():
        continue
    by = load(log)
    out = {}
    mism = 0
    for lenient in (False, True):
        rows = []
        for sid, d in by.items():
            if not all(k in d for k in [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]):
                continue
            metric = "vqa_acc"  # both lockbox regimes are scored with vqa_acc in this repo
            sc = {}
            for (a, r), rec in d.items():
                s = score(metric, rec["answer_after"], rec["gold"], lenient)
                sc[(a, r)] = s
                if not lenient and abs(s - rec["score_after"]) > 1e-9:
                    mism += 1
            st = d[("STOP", 1)]
            sa, ha = (st["answer_after"] or ""), (d[("HIGHRES", 1)]["answer_after"] or "")
            rows.append({"label": four_way(sc[("STOP", 1)], sc[("STOP", 2)], sc[("HIGHRES", 1)], sc[("HIGHRES", 2)]),
                         "stop": (sc[("STOP", 1)] + sc[("STOP", 2)]) / 2,
                         "hi": (sc[("HIGHRES", 1)] + sc[("HIGHRES", 2)]) / 2,
                         "verbose_stop": len(sa.split()) > 1, "short_hi": len(ha.split()) <= 1,
                         "answer": sa, "question": st["question"],
                         "answer_len_tokens": st["cost"]["tokens_out"],
                         "mean_logprob": float(np.mean(
                             [d[("STOP", r)]["confidence"]["mean_logprob"] for r in (1, 2)]))})
        cnt = Counter(r["label"] for r in rows)
        use = [r for r in rows if r["label"] != "AMBIGUOUS"]
        y = np.array([r["label"] == "HELP" for r in use], int)
        s = -np.array([r["mean_logprob"] for r in use])
        prev = y.mean()

        def lift(sc_):
            k = max(1, int(round(prev * len(y))))
            return float((y[np.argsort(-sc_)[:k]].sum() / max(y.sum(), 1)) / (k / len(y)))

        def roc_ci(sc_):
            v = []
            for _ in range(3000):
                i = rng.integers(0, len(y), len(y))
                if 0 < y[i].sum() < len(i): v.append(roc_auc_score(y[i], sc_[i]))
            return [float(q) for q in np.quantile(v, [0.025, 0.975])]

        helps = [r for r in rows if r["label"] == "HELP"]
        entry = {
            "n": len(rows), "labels": dict(cnt), "n_eval": len(use), "prevalence": float(prev),
            "stop_acc": float(np.mean([r["stop"] for r in rows])),
            "highres_acc": float(np.mean([r["hi"] for r in rows])),
            "verbose_stop_share": float(np.mean([r["verbose_stop"] for r in rows])),
            "help_format_flip_share": (float(np.mean([r["verbose_stop"] and r["short_hi"] for r in helps]))
                                       if helps else None),
            "conf_roc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None,
            "conf_roc_ci": roc_ci(s) if 0 < y.sum() < len(y) else None,
            "conf_lift": lift(s) if y.sum() else None,
        }
        if bb.startswith("Qwen"):
            conf = np.array([conf_pct(r["mean_logprob"]) for r in use])
            X = np.array([extract(r, c) for r, c in zip(use, conf)], float)
            g = FR["model"].predict_proba(FR["scaler"].transform(X))[:, 1]
            entry["v22_roc"] = float(roc_auc_score(y, g)) if 0 < y.sum() < len(y) else None
            entry["v22_lift"] = lift(g) if y.sum() else None
            # one-word-STOP subset: is the reversal there at all?
            m = np.array([not r["verbose_stop"] for r in use])
            if 0 < y[m].sum() < m.sum():
                entry["conf_roc_oneword_subset"] = float(roc_auc_score(y[m], s[m]))
                entry["n_oneword_subset"] = int(m.sum()); entry["help_oneword_subset"] = int(y[m].sum())
        out["lenient" if lenient else "strict"] = entry
    out["strict_mismatches_vs_logged"] = mism
    res[f"{bb} / {src}"] = out

json.dump(res, open(ROOT / "paper/format_confound.json", "w"), indent=1)
L = ["# Answer-format confound: strict vs lenient scoring of the same generations", "",
     "strict = official vqa_acc as logged (reproduced with 0 mismatches); lenient = strict OR gold is "
     "the first token / a whole-word substring of the normalised prediction. Labels, confidence ROC and "
     "the frozen v2.2 gate are recomputed on the rescored outcomes; nothing else changes.", "",
     "| backbone / regime | scorer | STOP | HIGHRES | verbose STOP | HELP (prev.) | HELP = format flip | conf ROC [CI] | conf lift | v2.2 ROC | v2.2 lift |",
     "|---|---|--:|--:|--:|--:|--:|:--:|--:|--:|--:|"]
for key, o in res.items():
    for sc in ("strict", "lenient"):
        e = o[sc]
        ff = "—" if e["help_format_flip_share"] is None else f"{100*e['help_format_flip_share']:.0f}%"
        roc = "—" if e["conf_roc"] is None else f"{e['conf_roc']:.3f} [{e['conf_roc_ci'][0]:.2f}, {e['conf_roc_ci'][1]:.2f}]"
        cl = "—" if e["conf_lift"] is None else f"{e['conf_lift']:.2f}×"
        vr = f"{e['v22_roc']:.3f}" if e.get("v22_roc") is not None else "—"
        vl = f"{e['v22_lift']:.2f}×" if e.get("v22_lift") is not None else "—"
        L.append(f"| {key} | {sc} | {100*e['stop_acc']:.1f}% | {100*e['highres_acc']:.1f}% | "
                 f"{100*e['verbose_stop_share']:.0f}% | {e['labels'].get('HELP',0)} ({100*e['prevalence']:.1f}%) | {ff} | "
                 f"{roc} | {cl} | {vr} | {vl} |")
L.append("")
for key, o in res.items():
    e = o["strict"]
    if "conf_roc_oneword_subset" in e:
        L.append(f"- {key}, strict scorer, one-word STOP answers only (n={e['n_oneword_subset']}, "
                 f"HELP={e['help_oneword_subset']}): confidence ROC **{e['conf_roc_oneword_subset']:.3f}**")
open(ROOT / "paper/format_confound.md", "w").write("\n".join(L))
print("\n".join(L))
