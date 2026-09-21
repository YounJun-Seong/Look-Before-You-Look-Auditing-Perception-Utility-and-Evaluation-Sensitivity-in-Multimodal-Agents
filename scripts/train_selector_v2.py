#!/usr/bin/env python3
"""Budget-aware deployable selector v2 — train on dev (Gate 0) and FREEZE.

Same decision form as the frozen v1 policy (per-family ΔScore value heads,
action = argmax_f dhat_f - lambda * cost_f with STOP value 0, lambda chosen
per budget on dev), with two disclosed upgrades:

  * candidate feature sets: F0 = the 16 frozen v1 features;
    F1 = F0 + [min-logprob rank, self-assessment needs_detail,
    self-assessment confidence rank] (all pre-action, from the existing
    stage-A side collection; the fresh evaluation collects the same signals).
  * candidate head classes: ridge (3 alphas), gradient boosting, random
    forest — the FULL candidate grid is reported in the manifest, and ONE
    winner is selected by 5-fold dev cross-validation of realized policy
    value on TextVQA at the 1/2x budget (primary; 1/4x reported).

Everything is fitted on the 396 Gate 0 dev states only. The winner is
refitted on all dev states and frozen to models/selector_frozen_v2.pkl before
any fresh-evaluation state is collected (paper/PREREG_PROSPECTIVE_SELECTOR.md).
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
    "mas", ROOT / "scripts/multiaction_selector.py")
mas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mas)

from sklearn.ensemble import (GradientBoostingRegressor,  # noqa: E402
                              RandomForestRegressor)
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SEED = 20260831
FAMILIES = ("CROP", "OCR", "HIGHRES")
LAM_TOK = [0.0] + list(np.logspace(-6, -2, 33))
BUDGET_FRACS = (0.25, 0.5, 1.0)
PRIMARY = ("textvqa", 0.5)


def side_features():
    """stage-A extras per state: min-logprob (mean of replays), self-assessment."""
    out = {}
    for line in open(ROOT / "data/stage_a_features.jsonl"):
        r = json.loads(line)
        mins = [r[k]["min_logprob"] for k in ("stop_r1", "stop_r2")
                if r.get(k) and r[k].get("min_logprob") is not None]
        sa = r.get("self_assessment") or {}
        out[r["state_id"]] = {
            "min_logprob": float(np.mean(mins)) if mins else None,
            "needs_detail": 1.0 if sa.get("needs_detail") else 0.0,
            "sa_logprob": sa.get("mean_logprob"),
        }
    return out


def build_X(states, feat_set, ranks):
    X0 = mas.feats(states, ranks["mean"])
    if feat_set == "F0":
        return X0
    extra = np.array([[ranks["min"](s["_min_logprob"]),
                       s["_needs_detail"],
                       ranks["sa"](s["_sa_logprob"])] for s in states])
    return np.hstack([X0, extra])


def make_rank(vals):
    srt = np.sort(np.array(vals, float))
    return lambda v: 1.0 - np.searchsorted(srt, v, side="right") / len(srt)


def head_factory(name):
    if name.startswith("ridge"):
        return Ridge(alpha=float(name.split("_")[1]), random_state=0)
    if name == "gbr":
        return GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                         learning_rate=0.05, random_state=0)
    if name == "rf":
        return RandomForestRegressor(n_estimators=200, max_depth=4,
                                     min_samples_leaf=8, random_state=0)
    raise ValueError(name)


def policy_value(states_eval, dhat, lam):
    choice = {}
    for i, s in enumerate(states_eval):
        vals = {"STOP": 0.0}
        for f in FAMILIES:
            vals[f] = dhat[f][i] - lam * s["cost_vtok"][f]
        choice[s["sid"]] = max(vals, key=vals.get)
    return mas.evaluate(states_eval, choice)


def cv_value(states, feat_set, head_name, ranks, assign):
    """out-of-fold dhat; lambda chosen on the fold-train; value on fold-test
    TextVQA states at each budget."""
    tv = [s for s in states if s["dataset"] == "textvqa"]
    hi_tok = float(np.mean([s["cost_vtok"]["HIGHRES"] for s in tv]))
    X = build_X(states, feat_set, ranks)
    vals = {f: [] for f in BUDGET_FRACS}
    for fold in range(mas.N_FOLDS):
        tr = [i for i, a in enumerate(assign) if a != fold]
        te = [i for i, a in enumerate(assign) if a == fold]
        scaler = StandardScaler().fit(X[tr])
        dh_tr, dh_te = {}, {}
        for f, key in (("CROP", "crop"), ("OCR", "ocr"), ("HIGHRES", "hi")):
            y = np.array([np.mean(states[i][key] - states[i]["stop"]) for i in tr])
            h = head_factory(head_name).fit(scaler.transform(X[tr]), y)
            dh_tr[f] = h.predict(scaler.transform(X[tr]))
            dh_te[f] = h.predict(scaler.transform(X[te]))
        tr_tv = [j for j, i in enumerate(tr) if states[i]["dataset"] == "textvqa"]
        te_tv = [j for j, i in enumerate(te) if states[i]["dataset"] == "textvqa"]
        s_tr = [states[tr[j]] for j in tr_tv]
        s_te = [states[te[j]] for j in te_tv]
        d_tr = {f: dh_tr[f][tr_tv] for f in FAMILIES}
        d_te = {f: dh_te[f][te_tv] for f in FAMILIES}
        for frac in BUDGET_FRACS:
            B = frac * hi_tok
            best_lam, best_v = 0.0, -1.0
            for lam in LAM_TOK:
                ev = policy_value(s_tr, d_tr, lam)
                if ev["vtok"] <= B + 1e-9 and ev["acc"] > best_v:
                    best_lam, best_v = lam, ev["acc"]
            ev = policy_value(s_te, d_te, best_lam)
            vals[frac].append((ev["acc"], ev["vtok"], len(s_te)))
    out = {}
    for frac, rows in vals.items():
        n = sum(r[2] for r in rows)
        out[frac] = {"acc": sum(r[0] * r[2] for r in rows) / n,
                     "vtok": sum(r[1] * r[2] for r in rows) / n}
    return out


def main():
    states, dropped = mas.load_states()
    assert dropped == 0 and len(states) == 396
    side = side_features()
    kept = []
    for s in states:
        e = side.get(s["sid"])
        if not e or e["min_logprob"] is None or e["sa_logprob"] is None:
            continue
        s["_min_logprob"] = e["min_logprob"]
        s["_needs_detail"] = e["needs_detail"]
        s["_sa_logprob"] = e["sa_logprob"]
        kept.append(s)
    print(f"dev states with full side features: {len(kept)}/396")
    states = kept
    ranks = {"mean": make_rank([s["mean_logprob"] for s in states]),
             "min": make_rank([s["_min_logprob"] for s in states]),
             "sa": make_rank([s["_sa_logprob"] for s in states])}
    assign = mas.folds(states)

    grid = [(fs, hn) for fs in ("F0", "F1")
            for hn in ("ridge_0.3", "ridge_1.0", "ridge_3.0", "gbr", "rf")]
    table = []
    for fs, hn in grid:
        v = cv_value(states, fs, hn, ranks, assign)
        table.append({"features": fs, "head": hn,
                      **{f"cv_acc_{f}x": round(v[f]["acc"], 4) for f in BUDGET_FRACS},
                      **{f"cv_vtok_{f}x": round(v[f]["vtok"], 1) for f in BUDGET_FRACS}})
        print(table[-1])
    key = f"cv_acc_{PRIMARY[1]}x"
    tie = "cv_acc_0.25x"
    winner = max(table, key=lambda r: (r[key], r[tie]))
    print("\nWINNER:", winner)

    # refit winner on all dev states and freeze
    X = build_X(states, winner["features"], ranks)
    scaler = StandardScaler().fit(X)
    heads = {}
    for f, k in (("CROP", "crop"), ("OCR", "ocr"), ("HIGHRES", "hi")):
        y = np.array([np.mean(s[k] - s["stop"]) for s in states])
        heads[f] = head_factory(winner["head"]).fit(scaler.transform(X), y)
    blob = {"heads": heads, "scaler": scaler,
            "feature_set": winner["features"], "head_name": winner["head"],
            "conf_sorted": {k: np.sort(np.array(
                [s[{"mean": "mean_logprob", "min": "_min_logprob",
                    "sa": "_sa_logprob"}[k]] for s in states]))
                for k in ("mean", "min", "sa")},
            "lam_tok": LAM_TOK, "budget_fracs": BUDGET_FRACS,
            "train": f"gate0 dev n={len(states)}", "seed": SEED,
            "candidate_table": table}
    with open(ROOT / "models/selector_frozen_v2.pkl", "wb") as fh:
        pickle.dump(blob, fh)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    sha = hashlib.sha256((ROOT / "models/selector_frozen_v2.pkl").read_bytes()
                         ).hexdigest()[:16]
    (ROOT / "models/selector_frozen_v2.json").write_text(json.dumps(
        {"artifact": "models/selector_frozen_v2.pkl", "sha256_16": sha,
         "git": git, "seed": SEED, "n_train": len(states),
         "winner": winner, "candidate_table": table,
         "selection_rule": "max 5-fold CV TextVQA policy value at 1/2x budget"
                           " (tie-break 1/4x); full grid disclosed above"},
        indent=1))
    print("frozen -> models/selector_frozen_v2.pkl", sha)


if __name__ == "__main__":
    main()
