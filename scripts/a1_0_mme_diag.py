#!/usr/bin/env python
"""A1-0 step 1: what is behind MME's 0x lift? Free — existing logs only.

Recall@5.9% over 169 states means the top-10 ranking caught no positives, which
is too coarse a statistic to conclude "no signal". Rank-based measures and the
per-domain breakdown say more, and cost nothing.
"""
from __future__ import annotations
import json
import numpy as np
from collections import Counter
from sklearn.metrics import average_precision_score, roc_auc_score

lab = {json.loads(l)["state_id"]: json.loads(l)
       for l in open("data/stage_a_labels_highres.jsonl")}
rows = []
for line in open("data/stage_a_features.jsonl"):
    f = json.loads(line)
    l = lab[f["state_id"]]
    if l["dataset"] != "mme" or l["label"] == "UNCERTAIN":
        continue
    conf = np.mean([f[f"stop_r{i}"]["mean_logprob"] or -9 for i in (1, 2)])
    rows.append({**l, "conf": conf,
                 "need": f["self_assessment"]["needs_detail"]})

y = np.array([1 if r["label"] == "LOOK" else 0 for r in rows])
s = np.array([-r["conf"] for r in rows])          # less confident = higher score
prev = y.mean()
print(f"MME states {len(rows)}  positives {int(y.sum())}  prevalence {100*prev:.1f}%")
print(f"  ROC-AUC          {roc_auc_score(y, s):.3f}   (0.5 = no ranking signal)")
ap = average_precision_score(y, s)
print(f"  PR-AUC           {ap:.3f}  = {ap/prev:.2f}x prevalence")
print(f"  confidence (mean logprob): HELP {np.mean([r['conf'] for r in rows if r['label']=='LOOK']):.3f} "
      f"vs non-HELP {np.mean([r['conf'] for r in rows if r['label']!='LOOK']):.3f}")
# where do the positives sit in the ranking?
order = np.argsort(-s)
ranks = [int(np.where(order == i)[0][0]) + 1 for i in np.where(y == 1)[0]]
print(f"  positive ranks (of {len(rows)}): {sorted(ranks)}")

print("\nper-domain HELP prevalence")
dom = Counter(r["meta_domain"] for r in rows)
pos = Counter(r["meta_domain"] for r in rows if r["label"] == "LOOK")
unc = Counter(json.loads(l)["meta_domain"] for l in open("data/stage_a_labels_highres.jsonl")
              if json.loads(l)["dataset"] == "mme" and json.loads(l)["label"] == "UNCERTAIN")
print(f"  {'domain':26s} {'n':>4s} {'HELP':>5s} {'prev':>7s} {'UNCERTAIN':>10s}")
for d, n in dom.most_common():
    print(f"  {d:26s} {n:4d} {pos[d]:5d} {100*pos[d]/n:6.1f}% {unc[d]:10d}")
