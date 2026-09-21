#!/usr/bin/env python3
"""Deployable state-independent randomized mixture — TextVQA (post-hoc).

The strongest fair version of the mixture baseline: weights are fitted on the
DEV data (Gate 0 `logs/tv_200.jsonl` arm means, strict as logged) and applied
unchanged to the held-out fresh-TextVQA states, exactly as a deployed
state-independent policy would be configured. No held-out information enters
the weights. Budget for fitting is B_dev = frac x dev mean always-HIGHRES
tokens; the realized held-out token cost is reported alongside.

Arms are matched dev->held-out by their fixed normalized crop boxes (the 5
grid + 3 fallback constants of ppu/regions.py; dev slot indices are unusable
because VLM proposals interleave). STOP / OCR / HIGHRES match by name.

Paired bootstrap of (frozen multi-action - deployable mixture) per held-out
state, same read-off budgets as the prereg table.

Outputs paper/deployable_mixture_textvqa.{json,md}.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sklearn.linear_model as lm  # noqa: E402

_orig = lm.LogisticRegression.predict_proba


def _patched(self, X):
    if not hasattr(self, "multi_class"):
        self.multi_class = "deprecated"
    return _orig(self, X)


lm.LogisticRegression.predict_proba = _patched

spec = importlib.util.spec_from_file_location(
    "hme", ROOT / "scripts/heldout_multiaction_eval.py")
hme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hme)

BUDGET_FRACS = (0.25, 0.5, 1.0)


def box_key(box):
    return ",".join(f"{float(v):.4f}" for v in box)


def dev_arm_means(log):
    """Gate 0 dev log -> arm name -> (mean score, mean vtok)."""
    per = defaultdict(lambda: defaultdict(list))  # arm -> sid -> [(score,vtok)]
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") != "rollout":
            continue
        a, sid = r["action"], r["sample_id"]
        if a["type"] == "STOP":
            name = "STOP"
        elif a["type"] in ("OCR", "HIGHRES"):
            name = a["type"]
        elif a["type"] == "CROP" and a.get("source") == "grid":
            name = f"CROP@{box_key(a['box'])}"
        else:
            continue
        per[name][sid].append((r["score_after"], r["cost"]["visual_tokens"]))
    out = {}
    for name, d in per.items():
        ys = [np.mean([x[0] for x in v]) for v in d.values()]
        cs = [np.mean([x[1] for x in v]) for v in d.values()]
        out[name] = (float(np.mean(ys)), float(np.mean(cs)), len(d))
    return out


def best_mixture(menu, B):
    names = sorted(menu)
    ybar = {a: menu[a][0] for a in names}
    cbar = {a: menu[a][1] for a in names}
    best = None
    for a in names:
        if cbar[a] <= B + 1e-9:
            if best is None or ybar[a] > best[1]:
                best = ({a: 1.0}, ybar[a])
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lo, hi = sorted((a, b), key=lambda x: cbar[x])
            if not (cbar[lo] <= B + 1e-9 < cbar[hi]):
                continue
            w = (B - cbar[lo]) / (cbar[hi] - cbar[lo])
            val = (1 - w) * ybar[lo] + w * ybar[hi]
            if val > best[1]:
                best = ({lo: 1 - w, hi: w}, val)
    return best


def main():
    dev = dev_arm_means("logs/tv_200.jsonl")
    dev_hi_tok = dev["HIGHRES"][1]

    ds, fresh_log, arms_log = [x for x in hme.DS if x[0] == "textvqa"][0]
    states, _ = hme.load_states(ds, fresh_log, arms_log)
    X = hme.feats_frozen(states)
    dhat = {f: hme.POL["heads"][f].predict(X) for f in hme.FAMILIES}
    sweeps = hme.sweep_multi(states, dhat, "strict")["multiaction_vtok"]
    hi_tok = float(np.mean([s["cost_vtok"]["HIGHRES"] for s in states]))

    # held-out per-state arm vectors keyed like the dev menu
    held = {"STOP": (np.array([float(np.mean(s["strict"]["stop"])) for s in states]),
                     np.zeros(len(states))),
            "HIGHRES": (np.array([float(np.mean(s["strict"]["hi"])) for s in states]),
                        np.array([s["cost_vtok"]["HIGHRES"] for s in states])),
            "OCR": (np.array([float(np.mean(s["strict"]["ocr"])) for s in states]),
                    np.array([s["cost_vtok"]["OCR"] for s in states]))}
    boxes = [box_key(c[0]["action"]["box"]) for c in states[0]["recs"]["crops"]]
    for i, bk in enumerate(boxes):
        assert all(box_key(s["recs"]["crops"][i][0]["action"]["box"]) == bk
                   for s in states), "crop box order differs across states"
        held[f"CROP@{bk}"] = (
            np.array([float(np.mean(s["strict"]["grid"][i])) for s in states]),
            np.array([float(np.mean([r["cost"]["visual_tokens"]
                                     for r in s["recs"]["crops"][i]]))
                      for s in states]))

    menu = {a: dev[a] for a in dev if a in held}
    dropped = sorted(set(dev) - set(menu))

    out, md = [], [
        "# Deployable mixture (dev-fitted) vs frozen selector — TextVQA strict",
        "",
        f"Weights fitted on Gate 0 dev arm means (n={dev['STOP'][2]} states),",
        "applied unchanged to the 250 held-out states. Arms matched by fixed",
        f"crop box; dev-only arms dropped: {dropped or 'none'}.",
        f"Dev always-HIGHRES {dev_hi_tok:.0f} tok; held-out {hi_tok:.0f} tok.", "",
        "| budget | dev-fitted mixture | dev acc (fit) | held-out acc | "
        "held-out vtok | multi acc | multi − mixture [95% CI] |",
        "|---|---|--:|--:|--:|--:|:--|"]
    for frac in BUDGET_FRACS:
        B = frac * dev_hi_tok
        weights, fit_acc = best_mixture(menu, B)
        assert all(a in held for a in weights)
        mix_scores = sum(w * held[a][0] for a, w in weights.items())
        mix_vtok = float(sum(w * held[a][1] for a, w in weights.items()).mean())
        feas = [p for p in sweeps if p["vtok"] <= frac * hi_tok + 1e-9]
        multi = max(feas, key=lambda p: p["acc"])
        d = multi["_scores"] - mix_scores
        ci = hme.boot_ci(d)
        wtxt = " + ".join(f"{w:.2f}·{a}" for a, w in
                          sorted(weights.items(), key=lambda kv: -kv[1]))
        out.append({"budget_frac": frac, "budget_vtok_dev": B,
                    "weights": weights, "dev_fit_acc": float(fit_acc),
                    "heldout_acc": float(mix_scores.mean()),
                    "heldout_vtok": mix_vtok,
                    "multiaction_acc": multi["acc"],
                    "multi_minus_mixture": float(d.mean()), "ci95": ci})
        md.append(f"| {frac:g}× | {wtxt} | {fit_acc:.3f} | {mix_scores.mean():.3f} "
                  f"| {mix_vtok:.0f} | {multi['acc']:.3f} | {d.mean():+.4f} "
                  f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    (ROOT / "paper/deployable_mixture_textvqa.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/deployable_mixture_textvqa.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
