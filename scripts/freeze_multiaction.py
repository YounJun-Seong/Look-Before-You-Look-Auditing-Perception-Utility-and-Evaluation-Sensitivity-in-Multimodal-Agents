#!/usr/bin/env python3
"""Train the multi-action value heads on Gate 0 development states and FREEZE.

Implements the policy half of paper/PREREG_MULTIACTION_HELDOUT.md. Must run
(and be committed) before any held-out CROP/OCR outcome is collected.

Writes models/multiaction_frozen_v1.pkl (+ .json manifest).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "multiaction_selector", ROOT / "scripts/multiaction_selector.py")
mas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mas)

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ppu.gate_features import FEATURE_NAMES  # noqa: E402

SEED = 20260828
LAM_TOK = [0.0] + list(np.logspace(-6, -2, 33))
LAM_LAT = [0.0] + list(np.logspace(-5, -0.5, 33))


def main():
    states, dropped = mas.load_states()
    assert dropped == 0 and len(states) == 396, (len(states), dropped)

    conf_sorted = np.sort(np.array([s["mean_logprob"] for s in states]))

    def rank(lp):
        return 1.0 - np.searchsorted(conf_sorted, lp, side="right") / len(conf_sorted)

    X = mas.feats(states, rank)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    heads = {}
    for fam, key in (("CROP", "crop"), ("OCR", "ocr"), ("HIGHRES", "hi")):
        y = np.array([np.mean(s[key] - s["stop"]) for s in states])
        heads[fam] = Ridge(alpha=1.0, random_state=0).fit(Xs, y)
        print(f"{fam}: train ΔScore mean {y.mean():+.4f}, head R2 {heads[fam].score(Xs, y):.3f}")

    # frozen best-fixed baseline: highest mean training score among fixed arms
    fixed_scores = {f: float(np.mean([np.mean(s[k]) for s in states]))
                    for f, k in (("STOP", "stop"), ("CROP", "crop"),
                                 ("OCR", "ocr"), ("HIGHRES", "hi"))}
    best_fixed = max(fixed_scores, key=fixed_scores.get)
    print("fixed-arm training scores:", {k: round(v, 4) for k, v in fixed_scores.items()},
          "-> frozen best fixed:", best_fixed)

    feat_sha = hashlib.sha256((ROOT / "ppu/gate_features.py").read_bytes()).hexdigest()[:16]
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    blob = {"heads": heads, "scaler": scaler, "conf_sorted_mean_logprob": conf_sorted,
            "feature_names": FEATURE_NAMES, "crop_rule": "uniform-grid expectation",
            "lam_tok": LAM_TOK, "lam_lat": LAM_LAT, "best_fixed_frozen": best_fixed,
            "train": "gate0 mme196+textvqa200", "seed": SEED}
    (ROOT / "models").mkdir(exist_ok=True)
    with open(ROOT / "models/multiaction_frozen_v1.pkl", "wb") as fh:
        pickle.dump(blob, fh)
    manifest = {"artifact": "models/multiaction_frozen_v1.pkl",
                "train_states": len(states), "feature_code_sha256_16": feat_sha,
                "git": git, "seed": SEED, "alpha": 1.0,
                "conf_rank": "ECDF of train STOP mean logprob (mean of replays)",
                "best_fixed_frozen": best_fixed,
                "fixed_arm_training_scores": fixed_scores,
                "prereg": "paper/PREREG_MULTIACTION_HELDOUT.md",
                "refit_on_heldout": False}
    (ROOT / "models/multiaction_frozen_v1.json").write_text(json.dumps(manifest, indent=1))
    print("frozen -> models/multiaction_frozen_v1.{pkl,json}")


if __name__ == "__main__":
    main()
