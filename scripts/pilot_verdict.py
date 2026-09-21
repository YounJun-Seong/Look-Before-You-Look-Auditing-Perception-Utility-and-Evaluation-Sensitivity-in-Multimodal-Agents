#!/usr/bin/env python
"""Pilot verdict: is coarse=64 a coarse observation, or induced blindness?

The decisive quantity is not the absolute coarse accuracy — these datasets have
their own difficulty — but coarse relative to FULL on the same states.
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
import numpy as np

THR = 0.5


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0: return "HELP"
    if d1 < 0 and d2 < 0: return "HARM"
    if d1 == 0 and d2 == 0: return "NEUTRAL"
    return "AMBIGUOUS"


rows = [json.loads(l) for l in open("logs/general_pilot60.jsonl") if l.strip()]
rows = [r for r in rows if r.get("type") == "rollout"]
man = {json.loads(l)["sample_id"]: json.loads(l)
       for l in open("data/manifests/general_pilot.jsonl")}

by = defaultdict(dict)
for r in rows:
    by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r

states = []
for sid, d in by.items():
    need = [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]
    if not all(k in d for k in need):
        continue
    s1, s2 = d[("STOP", 1)]["score_after"], d[("STOP", 2)]["score_after"]
    x1, x2 = d[("HIGHRES", 1)]["score_after"], d[("HIGHRES", 2)]["score_after"]
    m = man[sid]
    states.append({"sid": sid, "ds": m["dataset"], "stratum": m["meta"]["stratum"],
                   "stop": (s1 + s2) / 2, "hi": (x1 + x2) / 2,
                   "label": four_way(s1, s2, x1, x2),
                   "coarse_tok": d[("STOP", 1)]["views"][0]["tokens"],
                   "hi_tok": d[("HIGHRES", 1)]["views"][0]["tokens"],
                   "native": m["native_tokens"]})

def block(name, sel):
    if not sel: return
    stop = np.mean([s["stop"] for s in sel]); hi = np.mean([s["hi"] for s in sel])
    c = Counter(s["label"] for s in sel)
    n = len(sel)
    ratio = stop / hi if hi > 0 else float("nan")
    print(f"{name:22s} n={n:3d} | coarse {100*stop:5.1f}%  HIGHRES {100*hi:5.1f}%  "
          f"gap {100*(hi-stop):+5.1f}pp  coarse/HI {ratio:4.2f} | "
          f"HELP {100*c['HELP']/n:4.1f}%  HARM {100*c['HARM']/n:4.1f}%  "
          f"NEUTRAL {100*c['NEUTRAL']/n:4.1f}%  AMBIG {100*c['AMBIGUOUS']/n:4.1f}%")

print(f"budgets: coarse {np.median([s['coarse_tok'] for s in states]):.0f} tok, "
      f"HIGHRES {np.median([s['hi_tok'] for s in states]):.0f} tok, "
      f"native median {np.median([s['native'] for s in states]):.0f} tok "
      f"(ratio native/coarse {np.median([s['native']/s['coarse_tok'] for s in states]):.1f}x)\n")
block("ALL", states)
for d in ("vqav2", "gqa"):
    block(d, [s for s in states if s["ds"] == d])
print()
for st in sorted({s["stratum"] for s in states}):
    block(st, [s for s in states if s["stratum"] == st])
print("\nreference (earlier gates): MME coarse 37.2 / FULL 41.3 (ratio 0.90); "
      "TextVQA coarse 60.0 / FULL 75.5 (ratio 0.79)")
