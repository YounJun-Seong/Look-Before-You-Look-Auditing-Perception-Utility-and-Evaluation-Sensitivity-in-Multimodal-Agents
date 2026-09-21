#!/usr/bin/env python
"""The four lines: TextVQA, the confirmation set, HARM veto, perception ladder."""
from __future__ import annotations
import json, pickle
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

THR = 0.5
FR = pickle.load(open("models/gate_frozen_v1.pkl", "rb"))
FEATS, SC, MODEL = FR["features"], FR["scaler"], FR["model"]


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0: return "HELP"
    if d1 < 0 and d2 < 0: return "HARM"
    if d1 == 0 and d2 == 0: return "NEUTRAL"
    return "AMBIGUOUS"


def states(log, manifest, coarse):
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
        m = man.get(sid, {}); st = d[("STOP", 1)]
        out.append({"state_id": sid, "label": four_way(s1, s2, x1, x2),
                    "stratum": (m.get("meta") or {}).get("stratum", "-"),
                    "stop_score": (s1 + s2) / 2, "highres_score": (x1 + x2) / 2,
                    "mean_logprob": float(np.mean([d[("STOP", r)]["confidence"]["mean_logprob"] for r in (1, 2)])),
                    "min_logprob": float(np.mean([d[("STOP", r)]["confidence"]["min_logprob"] for r in (1, 2)])),
                    "answer_len_tokens": st["cost"]["tokens_out"],
                    "answer_len_chars": len(st["answer_after"]),
                    "question_len_chars": len(st["question"]),
                    "budget_ratio": m.get("native_tokens", 0) / max(st["views"][0]["tokens"], 1)})
    return out


def lift_row(name, y, s, prev=None):
    prev = y.mean() if prev is None else prev
    lifts = []
    for mult in (1.0, 1.5, 2.0):
        k = max(1, int(round(prev * mult * len(y))))
        lifts.append((y[np.argsort(-s)[:k]].sum() / max(y.sum(), 1)) / (k / len(y)))
    print(f"  {name:24s} ROC {roc_auc_score(y, s):5.3f}  PR/prev "
          f"{average_precision_score(y, s)/prev:4.1f}x  lift@1x {lifts[0]:4.1f}x "
          f"@1.5x {lifts[1]:4.1f}x @2x {lifts[2]:4.1f}x")


def score_frozen(rows):
    X = np.array([[r[f] for f in FEATS] for r in rows], dtype=float)
    return MODEL.predict_proba(SC.transform(X))[:, 1]


# ---- 1 & 2 -----------------------------------------------------------------
old = [json.loads(l) for l in open("data/stage_a1_dataset.jsonl")]
new = states("logs/a1_vqav2_confirm.jsonl", "data/manifests/vqav2_confirm.jsonl", 64)

print("LINE 1 — TextVQA (cross-validated, from the training pool)")
tv = [r for r in old if r["source"] == "textvqa" and r["label"] != "AMBIGUOUS"]
y = np.array([r["label"] == "HELP" for r in tv], int)
lift_row("confidence", y, -np.array([r["mean_logprob"] for r in tv]))
print(f"    (learned gate CV lift 1.6x / ROC 0.638 — see A1-500 table; the frozen "
      f"model saw these states, so in-sample numbers are not reported)")

print(f"\nLINE 2 — NEW VQAv2 250, CONFIRMATION SET (frozen gate, never refitted)")
c = Counter(r["label"] for r in new)
n = len(new)
print(f"  labels: HELP {c['HELP']} ({100*c['HELP']/n:.1f}%)  HARM {c['HARM']}  "
      f"NEUTRAL {c['NEUTRAL']}  AMBIGUOUS {c['AMBIGUOUS']}  (n={n})")
usable = [r for r in new if r["label"] != "AMBIGUOUS"]
y = np.array([r["label"] == "HELP" for r in usable], int)
if y.sum() >= 3:
    lift_row("frozen confidence", y, -np.array([r["mean_logprob"] for r in usable]))
    lift_row("frozen learned gate", y, score_frozen(usable))
    print(f"  reference — existing VQAv2 250 (CV): confidence 1.0x, learned 1.4x "
          f"(B excluded: 0.8x / 2.0x)")

# ---- 3 ---------------------------------------------------------------------
print("\nLINE 3 — HARM veto (existing A1-500, self-assessment)")
sa = {json.loads(l)["state_id"]: json.loads(l)
      for l in open("data/stage_a1_self_assessment.jsonl")}
lab = {r["state_id"]: r for r in old}
ids = [i for i in sa if i in lab and lab[i]["label"] != "AMBIGUOUS"]
y = np.array([lab[i]["label"] == "HELP" for i in ids], int)
harm = np.array([lab[i]["label"] == "HARM" for i in ids], int)
need = np.array([sa[i]["needs_detail"] for i in ids], int)
conf = -np.array([lab[i]["mean_logprob"] for i in ids])
prev = y.mean()
k = max(1, int(round(prev * len(y))))
cand = np.zeros(len(y), int); cand[np.argsort(-conf)[:k]] = 1
comb = cand & need
for nm, g in (("confidence gate", cand), ("self-assessment", need), ("combined (veto)", comb)):
    print(f"  {nm:20s} LOOK {100*g.mean():5.1f}%  HELP retention {100*(g & (y==1)).sum()/max(y.sum(),1):5.1f}%  "
          f"HARM LOOK rate {100*(g & (harm==1)).sum()/max(harm.sum(),1):5.1f}%")
print(f"  (HARM n={int(harm.sum())}; A0 saw NEED_DETAIL on 0/9 HARM)")

# ---- 4 ---------------------------------------------------------------------
print("\nLINE 4 — VQAv2 perception ladder (existing 250, FULL added)")
full = defaultdict(dict)
for line in open("logs/a1_vqav2_full.jsonl"):
    r = json.loads(line)
    if r.get("type") == "rollout":
        full[r["sample_id"]][r["replay"]] = r["score_after"]
vq = [r for r in old if r["source"] == "vqav2"]
print(f"  {'stratum':22s} {'n':>4s} {'coarse':>7s} {'HIGHRES':>8s} {'FULL':>7s} {'c/HI':>5s} {'HI/FULL':>8s}")
for st in ["A_global_easy", "B_local_detail", "C_multi_relational"]:
    sel = [r for r in vq if r["stratum"] == st and r["state_id"] in full]
    if not sel: continue
    co = np.mean([r["stop_score"] for r in sel]); hi = np.mean([r["highres_score"] for r in sel])
    fu = np.mean([np.mean(list(full[r["state_id"]].values())) for r in sel])
    print(f"  {st:22s} {len(sel):4d} {100*co:6.1f}% {100*hi:7.1f}% {100*fu:6.1f}% "
          f"{co/hi:5.2f} {hi/fu if fu else float('nan'):8.2f}")
