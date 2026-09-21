#!/usr/bin/env python
"""A1.5 step 1-2: pooled VQAv2 500 effect size, and is answer length a shortcut?

The confirmation numbers (ROC 0.654 / lift 1.2x on the new 250) stay as they
are — that set is burned. Pooling only buys a tighter effect-size estimate.

The shortcut question: answer length is the strongest VQAv2 feature. If it
predicts HELP only because answer TYPE (yes-no vs number vs phrase) differs,
it is a format artefact. The test is whether it still predicts within type.
"""
from __future__ import annotations
import json, pickle
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

THR = 0.5
FR = pickle.load(open("models/gate_frozen_v1.pkl", "rb"))


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0: return "HELP"
    if d1 < 0 and d2 < 0: return "HARM"
    if d1 == 0 and d2 == 0: return "NEUTRAL"
    return "AMBIGUOUS"


def states(log, manifest, coarse, tag):
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
        m = man.get(sid, {}); st = d[("STOP", 1)]; meta = m.get("meta") or {}
        out.append({"state_id": sid, "batch": tag, "label": four_way(s1, s2, x1, x2),
                    "stratum": meta.get("stratum", "-"), "qtype": meta.get("qtype", "-"),
                    "answer": st["answer_after"],
                    "mean_logprob": float(np.mean([d[("STOP", r)]["confidence"]["mean_logprob"] for r in (1, 2)])),
                    "min_logprob": float(np.mean([d[("STOP", r)]["confidence"]["min_logprob"] for r in (1, 2)])),
                    "answer_len_tokens": st["cost"]["tokens_out"],
                    "answer_len_chars": len(st["answer_after"]),
                    "question_len_chars": len(st["question"]),
                    "budget_ratio": m.get("native_tokens", 0) / max(st["views"][0]["tokens"], 1)})
    return out


rows = (states("logs/a1_vqav2_250.jsonl", "data/manifests/vqav2_a1.jsonl", 64, "orig")
        + states("logs/a1_vqav2_confirm.jsonl", "data/manifests/vqav2_confirm.jsonl", 64, "confirm"))
use = [r for r in rows if r["label"] != "AMBIGUOUS"]
y = np.array([r["label"] == "HELP" for r in use], int)
conf = -np.array([r["mean_logprob"] for r in use])
X = np.array([[r[f] for f in FR["features"]] for r in use], float)
v1 = FR["model"].predict_proba(FR["scaler"].transform(X))[:, 1]
prev = y.mean()
rng = np.random.default_rng(0)


def boot(y, s, prev, mult=1.0, n=4000):
    vals = []
    idx0 = np.arange(len(y))
    for _ in range(n):
        i = rng.choice(idx0, len(idx0), replace=True)
        yy, ss = y[i], s[i]
        if yy.sum() == 0: continue
        k = max(1, int(round(prev * mult * len(yy))))
        vals.append((yy[np.argsort(-ss)[:k]].sum() / yy.sum()) / (k / len(yy)))
    return np.quantile(vals, [0.025, 0.975])


print(f"POOLED VQAv2 500  n={len(use)}  HELP={int(y.sum())}  prevalence={100*prev:.1f}%")
print(f"  {'scorer':22s} {'ROC':>6s} {'PR/prev':>8s} {'lift@1x':>8s} {'95% CI':>16s}")
for nm, s in (("confidence", conf), ("frozen v1 gate", v1)):
    k = max(1, int(round(prev * len(y))))
    lift = (y[np.argsort(-s)[:k]].sum() / y.sum()) / (k / len(y))
    lo, hi = boot(y, s, prev)
    print(f"  {nm:22s} {roc_auc_score(y, s):6.3f} "
          f"{average_precision_score(y, s)/prev:7.1f}x {lift:7.1f}x  [{lo:.2f}, {hi:.2f}]")
print("  (the confirmation-set numbers stay separate: ROC 0.654 / lift 1.2x)")

print("\nby stratum")
for st in ["A_global_easy", "B_local_detail", "C_multi_relational"]:
    sel = [i for i, r in enumerate(use) if r["stratum"] == st]
    if len(sel) < 20: continue
    yy = y[sel]
    print(f"  {st:22s} n={len(sel):3d} HELP={int(yy.sum()):3d} ({100*yy.mean():4.1f}%)  "
          f"conf ROC {roc_auc_score(yy, conf[sel]):5.3f}  v1 ROC {roc_auc_score(yy, v1[sel]):5.3f}")

print("\nANSWER-FORMAT SHORTCUT TEST")
bins = [(0, 3, "short (<=3 chars)"), (4, 8, "medium (4-8)"), (9, 999, "long (>=9)")]
print("  HELP prevalence by answer length")
for lo, hi, nm in bins:
    sel = [i for i, r in enumerate(use) if lo <= r["answer_len_chars"] <= hi]
    if sel:
        print(f"    {nm:20s} n={len(sel):3d}  HELP {100*y[sel].mean():5.1f}%")
print("  within answer type (VQAv2 annotation), does length still predict?")
for qt in ("yes/no", "number", "other"):
    sel = [i for i, r in enumerate(use) if r["qtype"] == qt]
    if len(sel) < 25: continue
    yy = y[sel]
    ln = np.array([use[i]["answer_len_chars"] for i in sel], float)
    roc = roc_auc_score(yy, ln) if 0 < yy.sum() < len(yy) else float("nan")
    print(f"    {qt:10s} n={len(sel):3d} HELP={int(yy.sum()):3d} ({100*yy.mean():4.1f}%)  "
          f"answer-length ROC {roc:5.3f}  confidence ROC "
          f"{roc_auc_score(yy, conf[sel]) if 0 < yy.sum() < len(yy) else float('nan'):5.3f}")
