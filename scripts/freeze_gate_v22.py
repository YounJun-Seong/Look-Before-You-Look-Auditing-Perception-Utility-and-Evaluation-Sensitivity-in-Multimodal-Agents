#!/usr/bin/env python
"""Freeze gate v2.2 (pointwise) before the lockbox exists.

Everything that could move the score is serialized, including the confidence
normalization. That transform is fitted on development data ONLY: ranking the
lockbox's own confidences would be transductive preprocessing, and a deployed
agent seeing one state at a time could not compute it anyway.
"""
from __future__ import annotations
import hashlib, json, pickle, subprocess, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppu.gate_features import FEATURE_NAMES, extract

THR = 0.5
HYPER = dict(max_iter=4000, class_weight="balanced", C=0.5, solver="lbfgs",
             random_state=0)
SETS = [("logs/a1_vqav2_250.jsonl", "data/manifests/vqav2_a1.jsonl", "vqav2"),
        ("logs/a1_vqav2_confirm.jsonl", "data/manifests/vqav2_confirm.jsonl", "vqav2"),
        ("logs/a1_textvqa250.jsonl", "data/manifests/textvqa_train.jsonl", "textvqa")]


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0: return "HELP"
    if d1 < 0 and d2 < 0: return "HARM"
    if d1 == 0 and d2 == 0: return "NEUTRAL"
    return "AMBIGUOUS"


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
                    "mean_logprob": float(np.mean(
                        [d[("STOP", r)]["confidence"]["mean_logprob"] for r in (1, 2)]))})
    return out


rows = [r for log, man, src in SETS for r in load(log, man, src)]
rows = [r for r in rows if r["label"] != "AMBIGUOUS"]

# development-fitted confidence transform: pooled, source-agnostic, frozen.
dev_conf = np.sort(np.array([r["mean_logprob"] for r in rows]))


def conf_percentile(x: float) -> float:
    """1 = least confident. Uses the frozen development CDF, never the batch."""
    return float(1.0 - np.searchsorted(dev_conf, x) / len(dev_conf))


X = np.array([extract(r, conf_percentile(r["mean_logprob"])) for r in rows], float)
y = np.array([r["label"] == "HELP" for r in rows], int)
scaler = StandardScaler().fit(X)
model = LogisticRegression(**HYPER).fit(scaler.transform(X), y)

def sha(p):
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()[:12]

blob = {
    "version": "gate_frozen_v2.2_pointwise",
    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "features": FEATURE_NAMES,
    "feature_code_sha1": sha("ppu/gate_features.py"),
    "hyper": HYPER,
    "scaler": scaler, "model": model,
    "conf_transform": {"kind": "development empirical CDF (pooled, source-agnostic)",
                       "sorted_dev_conf": dev_conf,
                       "n": len(dev_conf),
                       "note": "percentile = 1 - searchsorted(dev, x)/n; "
                               "the lockbox never contributes to this transform"},
    "protocol": {
        "positives": "HELP (replicate-consistent)", "negatives": "NEUTRAL + HARM",
        "excluded": "AMBIGUOUS", "score": "predict_proba[:,1]; higher = LOOK",
        "missing_values": "none permitted; a state lacking STOP or HIGHRES rollouts is dropped",
        "refit_on_lockbox": False, "threshold_tuning_on_lockbox": False,
        "evaluation": "single pass; lift@prevalence-matched LOOK rate + fixed "
                      "5/10/15/20% budgets; ROC; PR-AUC/prev; bootstrap CI as "
                      "supporting evidence only",
        "GO": {"textvqa": {"lift@1x": 1.3, "roc": 0.60},
               "vqav2": {"lift@1x": 1.5, "roc": 0.60},
               "and": "no catastrophic reversal in either regime"},
    },
    "train": {"n": len(rows), "positives": int(y.sum()),
              "sources": {s: sum(1 for r in rows if r["source"] == s)
                          for s in {r["source"] for r in rows}},
              "ids_sha1": hashlib.sha1(
                  "\n".join(sorted(r["state_id"] for r in rows)).encode()).hexdigest()[:16]},
    "coefficients": dict(zip(FEATURE_NAMES, model.coef_[0].round(4).tolist())),
    "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip(),
}
Path("models").mkdir(exist_ok=True)
pickle.dump(blob, open("models/gate_frozen_v2.2_pointwise.pkl", "wb"))
meta = {k: v for k, v in blob.items() if k not in ("scaler", "model")}
meta["conf_transform"] = {k: v for k, v in meta["conf_transform"].items()
                          if k != "sorted_dev_conf"}
Path("models/gate_frozen_v2.2_pointwise.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2)[:1400])
print("\ntop coefficients:", ", ".join(
    f"{n}{c:+.2f}" for n, c in sorted(blob["coefficients"].items(),
                                      key=lambda t: -abs(t[1]))[:6]))
