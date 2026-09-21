#!/usr/bin/env python
"""Single-pass lockbox evaluation of the frozen gate v2.2. No refit, no tuning.

Reports the pre-registered endpoints and, separately, fixed LOOK budgets —
the prevalence-matched rate needs the gold prevalence to define, so it is an
analysis metric; a fixed 10-15% budget is what a deployed policy actually has.
"""
from __future__ import annotations
import json, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppu.gate_features import extract

THR = 0.5
FR = pickle.load(open("models/gate_frozen_v2.2_pointwise.pkl", "rb"))
DEV = FR["conf_transform"]["sorted_dev_conf"]
GO = FR["protocol"]["GO"]
SETS = [("textvqa", "logs/lockbox_textvqa.jsonl", "data/manifests/textvqa_lockbox.jsonl"),
        ("vqav2", "logs/lockbox_vqav2.jsonl", "data/manifests/vqav2_lockbox.jsonl")]


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0: return "HELP"
    if d1 < 0 and d2 < 0: return "HARM"
    if d1 == 0 and d2 == 0: return "NEUTRAL"
    return "AMBIGUOUS"


def conf_pct(x):
    return float(1.0 - np.searchsorted(DEV, x) / len(DEV))


def load(log, manifest, source):
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
                    "mean_logprob": float(np.mean(
                        [d[("STOP", r)]["confidence"]["mean_logprob"] for r in (1, 2)]))})
    return out


rng = np.random.default_rng(20260820)
verdicts = {}
for source, log, man in SETS:
    rows = load(log, man, source)
    c = Counter(r["label"] for r in rows)
    use = [r for r in rows if r["label"] != "AMBIGUOUS"]
    y = np.array([r["label"] == "HELP" for r in use], int)
    conf = np.array([conf_pct(r["mean_logprob"]) for r in use])
    X = np.array([extract(r, cp) for r, cp in zip(use, conf)], float)
    v22 = FR["model"].predict_proba(FR["scaler"].transform(X))[:, 1]
    prev = y.mean()
    print(f"\n=== {source.upper()} LOCKBOX  n={len(rows)}  "
          f"HELP {c['HELP']} HARM {c['HARM']} NEUTRAL {c['NEUTRAL']} AMBIG {c['AMBIGUOUS']}"
          f"  | evaluated n={len(use)} prevalence {100*prev:.1f}%")

    def lift(s, mult=1.0):
        k = max(1, int(round(prev * mult * len(y))))
        return (y[np.argsort(-s)[:k]].sum() / max(y.sum(), 1)) / (k / len(y))

    def ci(s):
        vals = []
        for _ in range(4000):
            i = rng.integers(0, len(y), len(y))
            if y[i].sum() == 0: continue
            k = max(1, int(round(prev * len(i))))
            vals.append((y[i][np.argsort(-s[i])[:k]].sum() / y[i].sum()) / (k / len(i)))
        return np.quantile(vals, [0.025, 0.975])

    print(f"  {'scorer':16s} {'ROC':>6s} {'PR/prev':>8s} {'lift@1x':>8s} {'95% CI':>14s}")
    for nm, s in (("confidence", conf), ("v2.2 (frozen)", v22)):
        lo, hi = ci(s)
        print(f"  {nm:16s} {roc_auc_score(y, s):6.3f} "
              f"{average_precision_score(y, s)/prev:7.1f}x {lift(s):7.1f}x  [{lo:.2f}, {hi:.2f}]")

    print(f"  fixed LOOK budgets (v2.2 | confidence)")
    print(f"    {'budget':>7s} {'HELP recall':>22s} {'precision':>20s} "
          f"{'accuracy':>19s} {'tokens/state':>20s}")
    stop = np.array([r["stop"] for r in use]); hi_s = np.array([r["hi"] for r in use])
    tok = np.array([r["hi_tokens"] for r in use], float)
    for b in (0.05, 0.10, 0.15, 0.20):
        cells = []
        for s in (v22, conf):
            k = max(1, int(round(b * len(y))))
            pick = np.zeros(len(y), bool); pick[np.argsort(-s)[:k]] = True
            cells.append((y[pick].sum() / max(y.sum(), 1), y[pick].mean(),
                          np.where(pick, hi_s, stop).mean(), tok[pick].sum() / len(y)))
        print(f"    {100*b:6.0f}% {100*cells[0][0]:9.1f}% |{100*cells[1][0]:9.1f}% "
              f"{100*cells[0][1]:8.1f}% |{100*cells[1][1]:8.1f}% "
              f"{100*cells[0][2]:8.1f}% |{100*cells[1][2]:8.1f}% "
              f"{cells[0][3]:9.0f} |{cells[1][3]:8.0f}")
    print(f"    {'always':>6s}  STOP acc {100*stop.mean():.1f}% / 0 tok   "
          f"HIGHRES acc {100*hi_s.mean():.1f}% / {tok.mean():.0f} tok")

    g = GO[source]
    passed = (lift(v22) >= g["lift@1x"]) and (roc_auc_score(y, v22) >= g["roc"])
    verdicts[source] = {"lift": lift(v22), "roc": roc_auc_score(y, v22),
                        "need": g, "pass": bool(passed),
                        "reversal": roc_auc_score(y, v22) < 0.5}

print("\n=== PRE-REGISTERED VERDICT ===")
for s, v in verdicts.items():
    print(f"  {s:9s} lift {v['lift']:.2f}x (need >={v['need']['lift@1x']}) "
          f"ROC {v['roc']:.3f} (need >={v['need']['roc']}) -> "
          f"{'PASS' if v['pass'] else 'FAIL'}"
          + ("  [CATASTROPHIC REVERSAL]" if v["reversal"] else ""))
allpass = all(v["pass"] for v in verdicts.values())
anyrev = any(v["reversal"] for v in verdicts.values())
print("  CASE 1 (both PASS -> universal gate reproduced)" if allpass and not anyrev else
      "  CASE 3 (reversal or lift ~1x -> universal gate ends here)" if anyrev else
      "  CASE 2 (signal generalizes, budget-critical ranking short)")
