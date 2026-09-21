#!/usr/bin/env python
"""A1-500 label distribution — the gate before any predictor is fitted.

Four-way, replicate-consistent, per source and per stratum. Also reports the
states whose native view is under the inclusion rule, since --ids collection
bypasses that filter and the reader should know how many slipped in.
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
import numpy as np

THR = 0.5
RUNS = [("TextVQA train", "logs/a1_textvqa250.jsonl",
         "data/manifests/textvqa_train.jsonl", 192),
        ("VQAv2", "logs/a1_vqav2_250.jsonl", "data/manifests/vqav2_a1.jsonl", 64)]


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0: return "HELP"
    if d1 < 0 and d2 < 0: return "HARM"
    if d1 == 0 and d2 == 0: return "NEUTRAL"
    return "AMBIGUOUS"


def states_of(log, manifest, coarse):
    man = {json.loads(l)["sample_id"]: json.loads(l) for l in open(manifest)}
    by = defaultdict(dict)
    for line in open(log):
        r = json.loads(line)
        if r.get("type") != "rollout":
            continue
        by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r
    out = []
    for sid, d in by.items():
        if not all(k in d for k in [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]):
            continue
        s1, s2 = d[("STOP", 1)]["score_after"], d[("STOP", 2)]["score_after"]
        x1, x2 = d[("HIGHRES", 1)]["score_after"], d[("HIGHRES", 2)]["score_after"]
        m = man.get(sid, {})
        nat = m.get("native_tokens", 0)
        out.append({"sid": sid, "label": four_way(s1, s2, x1, x2),
                    "stratum": (m.get("meta") or {}).get("stratum", "-"),
                    "stop": (s1 + s2) / 2, "hi": (x1 + x2) / 2,
                    "eligible": nat >= 4 * coarse, "native": nat,
                    "conf": float(np.mean([d[("STOP", r)]["confidence"]["mean_logprob"]
                                           for r in (1, 2)]))})
    return out


def row(name, sel):
    if not sel:
        return
    n = len(sel)
    c = Counter(s["label"] for s in sel)
    stop = np.mean([s["stop"] for s in sel]); hi = np.mean([s["hi"] for s in sel])
    print(f"{name:22s} {n:4d} | {c['HELP']:4d} ({100*c['HELP']/n:4.1f}%) "
          f"{c['HARM']:4d} ({100*c['HARM']/n:4.1f}%) "
          f"{c['NEUTRAL']:4d} ({100*c['NEUTRAL']/n:4.1f}%) "
          f"{c['AMBIGUOUS']:4d} ({100*c['AMBIGUOUS']/n:4.1f}%) | "
          f"coarse {100*stop:5.1f} HI {100*hi:5.1f} ratio {stop/hi if hi else float('nan'):4.2f}")


allst = []
print(f"{'group':22s} {'n':>4s} | {'HELP':>11s} {'HARM':>11s} {'NEUTRAL':>12s} "
      f"{'AMBIGUOUS':>12s} | coarse/HIGHRES")
for name, log, man, coarse in RUNS:
    try:
        st = states_of(log, man, coarse)
    except FileNotFoundError:
        continue
    for s in st:
        s["source"] = name
    allst += st
    row(name, st)
    if name.startswith("VQAv2"):
        for k in sorted({s["stratum"] for s in st}):
            row("  " + k, [s for s in st if s["stratum"] == k])
row("ALL", allst)

ineli = [s for s in allst if not s["eligible"]]
print(f"\nbelow the inclusion rule (native < 4x coarse): {len(ineli)}/{len(allst)}"
      f"  — HELP among them {sum(1 for s in ineli if s['label']=='HELP')}")
gen = [s for s in allst if s["source"].startswith("VQAv2")]
print(f"general-domain HELP count: {sum(1 for s in gen if s['label']=='HELP')} "
      f"of {len(gen)}  (target >= 15-20)")
