#!/usr/bin/env python
"""Assemble the Stage A training set from the A1-500 collection.

Every state carries its flags rather than being silently dropped:

    bottlenecked      VQAv2 B_local_detail — coarse/HIGHRES = 0.48, below the
                      0.5 line, so a HELP here may be induced blindness rather
                      than a genuine perception need
    below_inclusion   native < 4x coarse; --ids collection bypasses the filter

Both are kept in training and reported separately at evaluation, so the effect
of the questionable states is visible instead of assumed.
"""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np

THR = 0.5
RUNS = [("textvqa", "logs/a1_textvqa250.jsonl", "data/manifests/textvqa_train.jsonl", 192),
        ("vqav2", "logs/a1_vqav2_250.jsonl", "data/manifests/vqav2_a1.jsonl", 64)]
BOTTLENECKED = {("vqav2", "B_local_detail")}


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0: return "HELP"
    if d1 < 0 and d2 < 0: return "HARM"
    if d1 == 0 and d2 == 0: return "NEUTRAL"
    return "AMBIGUOUS"


rows = []
for src, log, man_path, coarse in RUNS:
    man = {json.loads(l)["sample_id"]: json.loads(l) for l in open(man_path)}
    by = defaultdict(dict)
    for line in open(log):
        r = json.loads(line)
        if r.get("type") == "rollout":
            by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r
    for sid, d in by.items():
        keys = [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]
        if not all(k in d for k in keys):
            continue
        s1, s2 = d[("STOP", 1)]["score_after"], d[("STOP", 2)]["score_after"]
        x1, x2 = d[("HIGHRES", 1)]["score_after"], d[("HIGHRES", 2)]["score_after"]
        m = man.get(sid, {})
        stratum = (m.get("meta") or {}).get("stratum", "-")
        stop1 = d[("STOP", 1)]
        rows.append({
            "state_id": sid, "source": src, "stratum": stratum,
            "label": four_way(s1, s2, x1, x2),
            "bottlenecked": (src, stratum) in BOTTLENECKED,
            "below_inclusion": m.get("native_tokens", 0) < 4 * coarse,
            "stop_score": (s1 + s2) / 2, "highres_score": (x1 + x2) / 2,
            # features visible before paying for HIGHRES
            "mean_logprob": float(np.mean([d[("STOP", r)]["confidence"]["mean_logprob"]
                                           for r in (1, 2)])),
            "min_logprob": float(np.mean([d[("STOP", r)]["confidence"]["min_logprob"]
                                          for r in (1, 2)])),
            "answer_len_tokens": stop1["cost"]["tokens_out"],
            "answer_len_chars": len(stop1["answer_after"]),
            "question_len_chars": len(stop1["question"]),
            "coarse_tokens": stop1["views"][0]["tokens"],
            "native_tokens": m.get("native_tokens", 0),
            "budget_ratio": m.get("native_tokens", 0) / max(stop1["views"][0]["tokens"], 1),
            "coarse_answer": stop1["answer_after"],
            "question": stop1["question"],
        })

with open("data/stage_a1_dataset.jsonl", "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rows)} states -> data/stage_a1_dataset.jsonl")
print(f"  flagged bottlenecked : {sum(r['bottlenecked'] for r in rows)}")
print(f"  flagged below_inclusion: {sum(r['below_inclusion'] for r in rows)}")
