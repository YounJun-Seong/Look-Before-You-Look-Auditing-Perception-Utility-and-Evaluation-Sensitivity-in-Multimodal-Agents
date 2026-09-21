#!/usr/bin/env python
"""Diagnostic upper bound: what if the perceptual regime were known?

Not a proposed algorithm. The universal gate failed because the sign of the
confidence-to-utility relation reverses between regimes; this asks how much
performance that failure costs, by letting an oracle pick the right scorer per
regime. If the bound is high, the missing piece is regime identification, not
perception-utility predictability.
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

exec(open("scripts/train_gate_v2.py").read().split("SETS = ")[0].split("THR = ")[1]
     .replace("0.5\n", "", 1) if False else "")
THR = 0.5


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0: return "HELP"
    if d1 < 0 and d2 < 0: return "HARM"
    if d1 == 0 and d2 == 0: return "NEUTRAL"
    return "AMBIGUOUS"


def load(log, manifest, source):
    man = {json.loads(l)["sample_id"]: json.loads(l) for l in open(manifest)}
    by = defaultdict(dict)
    for line in open(log):
        r = json.loads(line)
        if r.get("type") == "rollout":
            by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r
    out = []
    for sid, d in by.items():
        if not all(k in d for k in [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]):
            continue
        s1, s2 = d[("STOP", 1)]["score_after"], d[("STOP", 2)]["score_after"]
        x1, x2 = d[("HIGHRES", 1)]["score_after"], d[("HIGHRES", 2)]["score_after"]
        st = d[("STOP", 1)]
        out.append({"state_id": sid, "source": source, "label": four_way(s1, s2, x1, x2),
                    "answer": st["answer_after"], "question": st["question"],
                    "answer_len_tokens": st["cost"]["tokens_out"],
                    "stop": (s1 + s2) / 2, "hi": (x1 + x2) / 2,
                    "hi_tokens": d[("HIGHRES", 1)]["cost"]["visual_tokens"],
                    "mean_logprob": float(np.mean([d[("STOP", r)]["confidence"]["mean_logprob"] for r in (1, 2)]))})
    return out


rows = (load("logs/a1_vqav2_250.jsonl", "data/manifests/vqav2_a1.jsonl", "vqav2")
        + load("logs/a1_vqav2_confirm.jsonl", "data/manifests/vqav2_confirm.jsonl", "vqav2")
        + load("logs/a1_textvqa250.jsonl", "data/manifests/textvqa_train.jsonl", "textvqa"))
rows = [r for r in rows if r["label"] != "AMBIGUOUS"]
for src in ("vqav2", "textvqa"):                       # confidence rank within source
    sel = [r for r in rows if r["source"] == src]
    lp = np.array([r["mean_logprob"] for r in sel])
    rank = 1 - lp.argsort().argsort() / max(len(lp) - 1, 1)
    for r, c in zip(sel, rank):
        r["conf_rank"] = float(c)
y = np.array([r["label"] == "HELP" for r in rows], int)
X = np.array([extract(r, r["conf_rank"]) for r in rows], float)

# learned gate, out-of-fold (pointwise logistic on the v2.1 feature set)
oof = np.zeros(len(rows))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(X, y):
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5)
    m.fit(sc.transform(X[tr]), y[tr])
    oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
conf = np.array([r["conf_rank"] for r in rows])
src = np.array([r["source"] for r in rows])


def metrics(y, s, prev):
    k = max(1, int(round(prev * len(y))))
    top = np.argsort(-s)[:k]
    return ((y[top].sum() / max(y.sum(), 1)) / (k / len(y)),
            average_precision_score(y, s) / prev, roc_auc_score(y, s))


print(f"{'gate':26s} {'TextVQA lift':>13s} {'VQAv2 lift':>11s} {'TextVQA PR/prev':>16s} {'VQAv2 PR/prev':>14s}")
res = {}
for nm, s in (("confidence", conf), ("universal learned", oof)):
    out = {}
    for d in ("textvqa", "vqav2"):
        m = src == d
        out[d] = metrics(y[m], s[m], y[m].mean())
    res[nm] = out
    print(f"{nm:26s} {out['textvqa'][0]:12.1f}x {out['vqav2'][0]:10.1f}x "
          f"{out['textvqa'][1]:15.1f}x {out['vqav2'][1]:13.1f}x")
best = {"textvqa": res["confidence"]["textvqa"], "vqav2": res["universal learned"]["vqav2"]}
print(f"{'regime-aware (oracle)':26s} {best['textvqa'][0]:12.1f}x {best['vqav2'][0]:10.1f}x "
      f"{best['textvqa'][1]:15.1f}x {best['vqav2'][1]:13.1f}x")

print("\ncost view at a LOOK budget matched to each regime's prevalence")
print(f"{'policy':26s} {'accuracy':>9s} {'extra tokens/state':>19s}")
for nm, s in (("always STOP", None), ("always HIGHRES", None),
              ("confidence gate", conf), ("universal learned", oof),
              ("regime-aware (oracle)", None)):
    acc, tok = 0.0, 0.0
    for d in ("textvqa", "vqav2"):
        m = src == d
        yy = np.array([r["stop"] for r, mm in zip(rows, m) if mm])
        hh = np.array([r["hi"] for r, mm in zip(rows, m) if mm])
        tk = np.array([r["hi_tokens"] for r, mm in zip(rows, m) if mm], float)
        prev = y[m].mean()
        if nm == "always STOP":
            a, t = yy.mean(), 0.0
        elif nm == "always HIGHRES":
            a, t = hh.mean(), tk.mean()
        else:
            ss = (conf[m] if nm == "confidence gate" else
                  oof[m] if nm == "universal learned" else
                  (conf[m] if d == "textvqa" else oof[m]))
            k = max(1, int(round(prev * len(ss))))
            pick = np.zeros(len(ss), bool); pick[np.argsort(-ss)[:k]] = True
            a = np.where(pick, hh, yy).mean(); t = tk[pick].sum() / len(ss)
        acc += a / 2; tok += t / 2
    print(f"{nm:26s} {100*acc:8.1f}% {tok:18.0f}")
