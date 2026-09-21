#!/usr/bin/env python3
"""Stronger deployable controllers — train on the pooled TextVQA logs, FREEZE.

Reviewer question addressed: "is the oracle–policy gap just a weak (ridge)
controller?"  Same inputs as the frozen v2 selector (the 16 F0 features:
answer surface, question surface, STOP-confidence rank), same targets
(per-family replay-mean score increment over STOP: CROP uniform-grid
expectation, OCR, HIGHRES), same decision rule argmax_f [Δ̂_f − λ_B c_f]
with STOP = 0.  Only the head class and the training pool change:

  head classes : ridge | gradient-boosted trees (xgboost) | 2-layer MLP
  training pool: 1,350 TextVQA states (Gate-0 200 + held-out 250 +
                 prospective 600 + v3-prospective 300), all previously
                 analyzed sets — so ridge is ALSO refit on this pool; the
                 capacity comparison is pool-matched.  The dev-only frozen
                 v2 (n=396) is kept as the published reference.

Per class, a small disclosed grid is scored by 5-fold CV policy value at the
½× budget (tie-break ¼×), exactly the v2 selection rule; λ(B) per class is
chosen on the pooled OUT-OF-FOLD predictions (in-sample λ selection would be
meaningless for high-capacity heads).  The nonlinear "winner" (xgb vs mlp)
for the primary endpoint is fixed here by the same CV rule.  Everything is
frozen to models/strong_controllers_frozen.{pkl,json} BEFORE the fresh
evaluation set is collected (paper/PREREG_STRONG_CONTROLLER.md).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".pylib"))
sys.path.insert(0, str(ROOT / "scripts"))
from strong_controller_common import (BUDGET_FRACS, FAMILIES, LAM_TOK,  # noqa: E402
                                      features, load_pool, make_rank,
                                      policy_from_dhat, policy_value, targets)

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

SEED = 20260920
N_FOLDS = 5
GRID = {
    "ridge": [{"alpha": a} for a in (0.3, 1.0, 3.0, 10.0)],
    "xgb": [{"max_depth": d, "n_estimators": n, "learning_rate": 0.05,
             "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5}
            for d in (2, 3) for n in (100, 300)],
    "mlp": [{"hidden_layer_sizes": h, "alpha": a}
            for h in ((64,), (64, 64)) for a in (1e-3, 1e-2)],
}


def make_head(cls, hp):
    if cls == "ridge":
        return Ridge(alpha=hp["alpha"], random_state=0)
    if cls == "xgb":
        return XGBRegressor(random_state=0, n_jobs=4, verbosity=0, **hp)
    if cls == "mlp":
        return MLPRegressor(max_iter=3000, early_stopping=True, validation_fraction=0.15,
                            n_iter_no_change=30, random_state=0, **hp)
    raise ValueError(cls)


def folds(n, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.arange(n) % N_FOLDS
    rng.shuffle(a)
    return a


def oof_predict(cls, hp, X, Y, assign):
    """out-of-fold Δ̂ for every family; scaler fit inside each fold."""
    dh = {f: np.zeros(len(X)) for f in FAMILIES}
    for k in range(N_FOLDS):
        tr, te = np.where(assign != k)[0], np.where(assign == k)[0]
        sc = StandardScaler().fit(X[tr])
        for f in FAMILIES:
            h = make_head(cls, hp).fit(sc.transform(X[tr]), Y[f][tr])
            dh[f][te] = h.predict(sc.transform(X[te]))
    return dh


def lam_for_budget(states, dhat, B):
    best_lam, best_v = 0.0, -1.0
    for lam in LAM_TOK:
        acc, cost = policy_value(states, policy_from_dhat(states, dhat, lam))
        if cost.mean() <= B + 1e-9 and acc.mean() > best_v:
            best_lam, best_v = lam, acc.mean()
    return best_lam, best_v


def cv_policy_value(states, dhat_oof, assign, hi_tok):
    """λ chosen on the fold-train OOF predictions, value on fold-test (nested)."""
    out = {}
    for frac in BUDGET_FRACS:
        B = frac * hi_tok
        accs, costs = [], []
        for k in range(N_FOLDS):
            tr, te = np.where(assign != k)[0], np.where(assign == k)[0]
            s_tr = [states[i] for i in tr]
            d_tr = {f: dhat_oof[f][tr] for f in FAMILIES}
            lam, _ = lam_for_budget(s_tr, d_tr, B)
            s_te = [states[i] for i in te]
            d_te = {f: dhat_oof[f][te] for f in FAMILIES}
            a, c = policy_value(s_te, policy_from_dhat(s_te, d_te, lam))
            accs.append(a); costs.append(c)
        out[frac] = {"acc": float(np.concatenate(accs).mean()),
                     "vtok": float(np.concatenate(costs).mean())}
    return out


def main():
    print("loading pool")
    pool = load_pool()
    n = len(pool)
    rank = make_rank(np.sort([s["mean_logprob"] for s in pool]))
    X = features(pool, rank)
    Y = targets(pool, "strict")
    hi_tok = float(np.mean([s["vtok"]["HIGHRES"] for s in pool]))
    assign = folds(n)
    print(f"pool n={n}, features {X.shape[1]}, always-HIGHRES {hi_tok:.1f} tok")

    table, oofs = [], {}
    for cls, hps in GRID.items():
        for hp in hps:
            dh = oof_predict(cls, hp, X, Y, assign)
            cv = cv_policy_value(pool, dh, assign, hi_tok)
            row = {"class": cls, "hp": hp,
                   **{f"cv_acc_{f}x": round(cv[f]["acc"], 4) for f in BUDGET_FRACS},
                   **{f"cv_vtok_{f}x": round(cv[f]["vtok"], 1) for f in BUDGET_FRACS}}
            table.append(row); oofs[json.dumps(row["hp"], sort_keys=True) + cls] = dh
            print(row, flush=True)
    key, tie = "cv_acc_0.5x", "cv_acc_0.25x"
    winners = {cls: max([r for r in table if r["class"] == cls], key=lambda r: (r[key], r[tie]))
               for cls in GRID}
    nonlinear = max([winners["xgb"], winners["mlp"]], key=lambda r: (r[key], r[tie]))
    print("\nWINNERS:", json.dumps(winners, indent=1)); print("primary nonlinear:", nonlinear["class"])

    # refit winners on the full pool; λ(B) from the pooled OOF predictions
    scaler = StandardScaler().fit(X)
    frozen = {"scaler": scaler, "conf_sorted": np.sort([s["mean_logprob"] for s in pool]),
              "heads": {}, "lam": {}, "hp": {}, "hi_tok": hi_tok, "n_train": n,
              "train_sources": sorted({s["source"] for s in pool}), "seed": SEED}
    lam_json = {}
    for cls, w in winners.items():
        frozen["hp"][cls] = w["hp"]
        frozen["heads"][cls] = {f: make_head(cls, w["hp"]).fit(scaler.transform(X), Y[f])
                                for f in FAMILIES}
        dh = oofs[json.dumps(w["hp"], sort_keys=True) + cls]
        frozen["lam"][cls] = {}
        lam_json[cls] = {}
        for frac in BUDGET_FRACS:
            lam, v = lam_for_budget(pool, dh, frac * hi_tok)
            frozen["lam"][cls][str(frac)] = lam
            lam_json[cls][str(frac)] = {"lam": lam, "oof_acc": round(v, 4)}
    frozen["primary_nonlinear"] = nonlinear["class"]
    pkl = ROOT / "models/strong_controllers_frozen.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(frozen, fh)
    sha = hashlib.sha256(pkl.read_bytes()).hexdigest()[:16]
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    (ROOT / "models/strong_controllers_frozen.json").write_text(json.dumps({
        "artifact": str(pkl.relative_to(ROOT)), "sha256_16": sha, "git": git, "seed": SEED,
        "n_train": n, "train_sources": frozen["train_sources"], "features": "F0 (16)",
        "targets": "strict replay-mean Δ over STOP per family",
        "winners": winners, "primary_nonlinear": nonlinear["class"],
        "frozen_lambda_textvqa": lam_json, "pool_highres_vtok": hi_tok,
        "candidate_table": table,
        "selection_rule": "per class: max 5-fold nested-CV policy value at 1/2x (tie 1/4x); "
                          "primary nonlinear = better of xgb/mlp by the same rule; "
                          "lambda(B) per class on pooled out-of-fold predictions"},
        indent=1))
    print("frozen ->", pkl, sha)


if __name__ == "__main__":
    main()
