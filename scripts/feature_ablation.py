#!/usr/bin/env python
"""Feature-group ablation for gate v2.2 — DEVELOPMENT SET ONLY.

Explanatory, not selective: v2.2 is frozen and nothing here can change it. Each
variant retrains the same pointwise recipe on the development states with one
feature group zeroed, so the question is "what was each group contributing",
not "what should the model have been".
"""
from __future__ import annotations
import json, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppu.gate_features import FEATURE_NAMES, extract

FR = pickle.load(open("models/gate_frozen_v2.2_pointwise.pkl", "rb"))
DEV, HYPER = FR["conf_transform"]["sorted_dev_conf"], FR["hyper"]
THR = 0.5
GROUPS = {
    "answer surface": ["ans_chars", "ans_tokens", "ans_words", "is_numeric",
                       "is_yesno", "has_hedge"],
    "question surface": ["q_chars", "q_words", "q_is_howmany", "q_is_what", "q_is_isare"],
    "confidence interactions": ["conf_x_yesno", "conf_x_numeric", "conf_x_other",
                                "conf_x_len", "conf_x_hedge"],
}
SETS = [("logs/a1_vqav2_250.jsonl", "vqav2"), ("logs/a1_vqav2_confirm.jsonl", "vqav2"),
        ("logs/a1_textvqa250.jsonl", "textvqa")]


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0: return "HELP"
    if d1 < 0 and d2 < 0: return "HARM"
    if d1 == 0 and d2 == 0: return "NEUTRAL"
    return "AMBIGUOUS"


rows = []
for log, src in SETS:
    by = defaultdict(dict)
    for line in open(log):
        r = json.loads(line)
        if r.get("type") == "rollout":
            by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r
    for sid, d in by.items():
        if not all(k in d for k in [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]):
            continue
        s1, s2 = d[("STOP", 1)]["score_after"], d[("STOP", 2)]["score_after"]
        x1, x2 = d[("HIGHRES", 1)]["score_after"], d[("HIGHRES", 2)]["score_after"]
        lab = four_way(s1, s2, x1, x2)
        if lab == "AMBIGUOUS":
            continue
        st = d[("STOP", 1)]
        rows.append({"source": src, "y": int(lab == "HELP"), "answer": st["answer_after"],
                     "question": st["question"], "answer_len_tokens": st["cost"]["tokens_out"],
                     "mean_logprob": float(np.mean(
                         [d[("STOP", r)]["confidence"]["mean_logprob"] for r in (1, 2)]))})

conf = np.array([1.0 - np.searchsorted(DEV, r["mean_logprob"]) / len(DEV) for r in rows])
X = np.array([extract(r, c) for r, c in zip(rows, conf)], float)
y = np.array([r["y"] for r in rows])
src = np.array([r["source"] for r in rows])


def evaluate(mask_cols):
    Xa = X.copy()
    for j in mask_cols:
        Xa[:, j] = 0.0
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(Xa, y):
        sc = StandardScaler().fit(Xa[tr])
        m = LogisticRegression(**HYPER).fit(sc.transform(Xa[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(Xa[te]))[:, 1]
    out = {}
    for d in ("textvqa", "vqav2"):
        mm = src == d
        yy, ss = y[mm], oof[mm]
        prev = yy.mean()
        k = max(1, int(round(prev * len(yy))))
        out[d] = ((yy[np.argsort(-ss)[:k]].sum() / yy.sum()) / (k / len(yy)),
                  roc_auc_score(yy, ss))
    return out


print(f"development set n={len(rows)}  HELP={int(y.sum())}  "
      f"(textvqa {int((src=='textvqa').sum())}, vqav2 {int((src=='vqav2').sum())})")
print(f"\n{'variant':30s} {'TextVQA lift':>12s} {'ROC':>6s} {'VQAv2 lift':>11s} {'ROC':>6s}")
base = evaluate([])
print(f"{'v2.2 (all features)':30s} {base['textvqa'][0]:11.2f}x {base['textvqa'][1]:6.3f} "
      f"{base['vqav2'][0]:10.2f}x {base['vqav2'][1]:6.3f}")
for gname, feats in GROUPS.items():
    cols = [FEATURE_NAMES.index(f) for f in feats]
    r = evaluate(cols)
    print(f"{'  − ' + gname:30s} {r['textvqa'][0]:11.2f}x {r['textvqa'][1]:6.3f} "
          f"{r['vqav2'][0]:10.2f}x {r['vqav2'][1]:6.3f}")
only = evaluate([j for j, f in enumerate(FEATURE_NAMES)
                 if f not in GROUPS["confidence interactions"]])
print(f"{'  confidence interactions only':30s} {only['textvqa'][0]:11.2f}x {only['textvqa'][1]:6.3f} "
      f"{only['vqav2'][0]:10.2f}x {only['vqav2'][1]:6.3f}")
