#!/usr/bin/env python
"""A1-0 verdict: concurrency=6 labels vs serial relabel of the same MME states.

Is MME's weak confidence signal a property of the domain, or an artifact of
labels collected under batching noise? Same states, same budgets, same prompts;
only the execution mode differs. Confidence now comes from the serial pass
itself, so the feature and the label are measured together.
"""
from __future__ import annotations
import json
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score

THR = 0.5


def four_way(s1, s2, x1, x2):
    """HELP / HARM / NEUTRAL / AMBIGUOUS from two replays."""
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0:
        return "HELP"
    if d1 < 0 and d2 < 0:
        return "HARM"
    if d1 == 0 and d2 == 0:
        return "NEUTRAL"
    return "AMBIGUOUS"


def load_serial(path):
    by = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if r.get("type") != "rollout":
            continue
        by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r
    out = {}
    for sid, d in by.items():
        need = [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]
        if not all(k in d for k in need):
            continue
        s1, s2 = d[("STOP", 1)]["score_after"], d[("STOP", 2)]["score_after"]
        x1, x2 = d[("HIGHRES", 1)]["score_after"], d[("HIGHRES", 2)]["score_after"]
        conf = np.mean([d[("STOP", r)]["confidence"]["mean_logprob"] for r in (1, 2)])
        out[sid] = {"label": four_way(s1, s2, x1, x2), "conf": float(conf),
                    "scores": (s1, s2, x1, x2)}
    return out


def metrics(y, s, tag):
    prev = y.mean()
    roc = roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")
    ap = average_precision_score(y, s)
    curve = []
    for mult in (1.0, 1.5, 2.0, 3.0):
        k = max(1, int(round(prev * mult * len(y))))
        top = np.argsort(-s)[:k]
        rec = y[top].sum() / max(y.sum(), 1)
        curve.append((100 * k / len(y), 100 * rec, rec / (k / len(y))))
    return {"n": len(y), "pos": int(y.sum()), "prev": prev, "roc": roc,
            "ap": ap, "ap_over_prev": ap / prev, "curve": curve, "tag": tag}


def show(m):
    print(f"  n={m['n']:4d} pos={m['pos']:3d} prev={100*m['prev']:4.1f}%  "
          f"ROC-AUC {m['roc']:.3f}  PR-AUC {m['ap']:.3f} ({m['ap_over_prev']:.2f}x prev)")
    print("     LOOK rate:  " + "  ".join(f"{r:4.1f}%" for r, _, _ in m["curve"]))
    print("     recall:     " + "  ".join(f"{c:4.1f}%" for _, c, _ in m["curve"]))
    print("     lift:       " + "  ".join(f"{l:4.1f}x" for _, _, l in m["curve"]))


old = {json.loads(l)["state_id"]: json.loads(l)
       for l in open("data/stage_a_labels_highres.jsonl")
       if json.loads(l)["dataset"] == "mme"}
oldf = {json.loads(l)["state_id"]: json.loads(l)
        for l in open("data/stage_a_features.jsonl")}
new = load_serial("logs/mme_serial_diag.jsonl")
shared = sorted(set(old) & set(new))
print(f"MME states compared: {len(shared)}\n")

print(f"{'label':12s} {'concurrency=6':>14s} {'serial':>10s}")
co = Counter(old[s]["label"] for s in shared)
cn = Counter(new[s]["label"] for s in shared)
n = len(shared)
for lab_old, lab_new in [("LOOK", "HELP"), ("STOP", "NEUTRAL"), (None, "HARM"),
                         ("UNCERTAIN", "AMBIGUOUS")]:
    a = f"{co[lab_old]:3d} ({100*co[lab_old]/n:4.1f}%)" if lab_old else "  — (merged)"
    print(f"{lab_new:12s} {a:>14s} {cn[lab_new]:3d} ({100*cn[lab_new]/n:4.1f}%)")

changed = sum(1 for s in shared
              if (old[s]["label"] == "LOOK") != (new[s]["label"] == "HELP"))
print(f"\npositive-label changes between passes: {changed}/{n}")

print("\n--- confidence signal (positives = HELP, ambiguous excluded)")
for tag, get_lab, get_conf, keep in [
    ("concurrency=6 (A0 features)",
     lambda s: old[s]["label"], lambda s: np.mean(
         [oldf[s][f"stop_r{i}"]["mean_logprob"] for i in (1, 2)]),
     lambda l: l in ("LOOK", "STOP")),
    ("serial (same pass)",
     lambda s: new[s]["label"], lambda s: new[s]["conf"],
     lambda l: l in ("HELP", "NEUTRAL", "HARM"))]:
    ids = [s for s in shared if keep(get_lab(s))]
    y = np.array([1 if get_lab(s) in ("LOOK", "HELP") else 0 for s in ids])
    sc = np.array([-get_conf(s) for s in ids])
    print(f"  {tag}")
    show(metrics(y, sc, tag))

print("\n--- per domain (serial)")
dom = defaultdict(list)
for s in shared:
    dom[old[s]["meta_domain"]].append(s)
print(f"  {'domain':26s} {'n':>4s} {'HELP':>5s} {'prev':>7s} {'AMBIG':>7s}")
for d, ids in sorted(dom.items(), key=lambda kv: -len(kv[1])):
    h = sum(1 for s in ids if new[s]["label"] == "HELP")
    a = sum(1 for s in ids if new[s]["label"] == "AMBIGUOUS")
    print(f"  {d:26s} {len(ids):4d} {h:5d} {100*h/len(ids):6.1f}% {100*a/len(ids):6.1f}%")
