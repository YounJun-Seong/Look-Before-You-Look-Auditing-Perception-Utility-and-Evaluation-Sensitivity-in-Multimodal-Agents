#!/usr/bin/env python
"""Stage A1 learnability: confidence and a light learned gate, with and
without the bottlenecked stratum.

Trained on the same states in both configurations; the difference is only
which states are *scored*, so a gate that works purely because coarse=64
blinded the reader on VQAv2-B is visible as a gap between the two columns.

AMBIGUOUS never enters training or evaluation. Positives are HELP; negatives
are NEUTRAL and HARM (a gate that fires on HARM is worse than useless).
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

rows = [json.loads(l) for l in open("data/stage_a1_dataset.jsonl")]
use = [r for r in rows if r["label"] in ("HELP", "NEUTRAL", "HARM")]
y = np.array([1 if r["label"] == "HELP" else 0 for r in use])
X = np.c_[[r["mean_logprob"] for r in use], [r["min_logprob"] for r in use],
          [r["answer_len_tokens"] for r in use], [r["answer_len_chars"] for r in use],
          [r["question_len_chars"] for r in use], [r["budget_ratio"] for r in use]]
conf = -np.array([r["mean_logprob"] for r in use])   # less confident = higher score

oof = np.zeros(len(y))
skf = StratifiedKFold(5, shuffle=True, random_state=0)
for tr, te in skf.split(X, y):
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)
    m.fit(sc.transform(X[tr]), y[tr])
    oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]

rng = np.random.default_rng(0)


def report(mask, title):
    yy = y[mask]
    if yy.sum() < 3:
        print(f"\n{title}: too few positives ({int(yy.sum())})")
        return
    prev = yy.mean()
    print(f"\n{title}   n={mask.sum()}  positives={int(yy.sum())}  prevalence={100*prev:.1f}%")
    print(f"  {'scorer':26s} {'ROC':>6s} {'PR-AUC':>7s} {'/prev':>6s} "
          f"{'lift@1x':>8s} {'@1.5x':>7s} {'@2x':>7s} {'@3x':>7s}")
    for name, s in (("random", None), ("confidence (M0)", conf[mask]),
                    ("logistic (M1)", oof[mask])):
        if s is None:
            print(f"  {'random':26s} {0.5:6.3f} {prev:7.3f} {1.0:5.1f}x "
                  f"{1.0:7.1f}x {1.0:6.1f}x {1.0:6.1f}x {1.0:6.1f}x")
            continue
        lifts = []
        for mult in (1.0, 1.5, 2.0, 3.0):
            k = max(1, int(round(prev * mult * len(yy))))
            rec = yy[np.argsort(-s)[:k]].sum() / yy.sum()
            lifts.append(rec / (k / len(yy)))
        print(f"  {name:26s} {roc_auc_score(yy, s):6.3f} "
              f"{average_precision_score(yy, s):7.3f} "
              f"{average_precision_score(yy, s)/prev:5.1f}x "
              + " ".join(f"{l:6.1f}x" for l in lifts))


bott = np.array([r["bottlenecked"] for r in use])
src = np.array([r["source"] for r in use])
report(np.ones(len(use), bool), "ALL 500 (B included)")
report(~bott, "EXCLUDING bottlenecked VQAv2-B")
report(bott, "bottlenecked VQAv2-B only")
for s in ("textvqa", "vqav2"):
    report(src == s, f"source = {s} (B included)")
report((src == "vqav2") & ~bott, "source = vqav2, B excluded")
