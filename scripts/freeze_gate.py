#!/usr/bin/env python
"""Freeze the Stage A gate BEFORE the confirmation set exists.

The whole point of the new VQAv2 250 is to test whether "confidence fails,
learned gate recovers" reproduces. That test is void if the gate is refitted,
retuned, or re-featurised after seeing the new states, so the model, the
scaler, the feature list and the evaluation protocol are serialized here and
hashed. Anything later must load this file, not rebuild it.
"""
from __future__ import annotations
import hashlib, json, pickle, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = ["mean_logprob", "min_logprob", "answer_len_tokens",
            "answer_len_chars", "question_len_chars", "budget_ratio"]
HYPER = {"max_iter": 3000, "class_weight": "balanced", "C": 0.5}

rows = [json.loads(l) for l in open("data/stage_a1_dataset.jsonl")]
use = [r for r in rows if r["label"] in ("HELP", "NEUTRAL", "HARM")]
y = np.array([1 if r["label"] == "HELP" else 0 for r in use])
X = np.array([[r[f] for f in FEATURES] for r in use], dtype=float)

scaler = StandardScaler().fit(X)
model = LogisticRegression(**HYPER).fit(scaler.transform(X), y)

train_ids = sorted(r["state_id"] for r in use)
blob = {
    "version": "gate_frozen_v1",
    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "features": FEATURES,
    "hyper": HYPER,
    "scaler": scaler,
    "model": model,
    "protocol": {
        "positives": "HELP (replicate-consistent)",
        "negatives": "NEUTRAL + HARM",
        "excluded": "AMBIGUOUS",
        "score": "predict_proba[:,1]; higher = LOOK",
        "evaluation": "lift at LOOK rate matched to that set's positive "
                      "prevalence; also 1.5x/2x/3x; PR-AUC/prevalence; ROC",
        "refit_on_new_data": False,
    },
    "train": {"n": len(use), "positives": int(y.sum()),
              "sources": {s: sum(1 for r in use if r["source"] == s)
                          for s in {r["source"] for r in use}},
              "ids_sha1": hashlib.sha1("\n".join(train_ids).encode()).hexdigest()[:16]},
    "coefficients": dict(zip(FEATURES, model.coef_[0].round(4).tolist())),
}
Path("models").mkdir(exist_ok=True)
with open("models/gate_frozen_v1.pkl", "wb") as fh:
    pickle.dump(blob, fh)
meta = {k: v for k, v in blob.items() if k not in ("scaler", "model")}
Path("models/gate_frozen_v1.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
