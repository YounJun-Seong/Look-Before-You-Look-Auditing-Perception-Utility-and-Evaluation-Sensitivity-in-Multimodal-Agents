#!/usr/bin/env python3
"""(v) Cross-reader selector transfer — zero inference, disclosed post-hoc.

Is perception need a reader-independent property of the state, or a
reader-specific one? The Qwen-trained frozen selector v2 is applied to the
Phi-4 Gate-0 states (same images/questions/views, Phi-4 outcomes and Phi-4
confidence from the byte-match-verified side collection), in two variants:

  verbatim      Qwen dev conf-ECDF and TextVQA lambda used unchanged
  recalibrated  conf rank computed within Phi-4's own confidence
                distribution (label-free calibration any deployer could do)

against always-STOP, the cross-fitted per-slot mixture, and the family-level
cross-replay oracle, at budgets frac x Phi-4 always-HIGHRES tokens. Strict
scorer (Gate 0 convention). Outputs paper/cross_reader_transfer.{json,md}.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppu.gate_features import extract  # noqa: E402

SEL = pickle.load(open(ROOT / "models/selector_frozen_v2.pkl", "rb"))
SELM = json.loads((ROOT / "models/selector_frozen_v2.json").read_text())
FAMILIES = ("CROP", "OCR", "HIGHRES")
FRACS = (0.25, 0.5, 1.0)
SEED = 20260901
N_BOOT = 10_000
DS = [("mme", "logs/p4_mme200.jsonl", "data/span_conf/p4_mme_stopconf.jsonl"),
      ("textvqa", "logs/p4_textvqa200.jsonl", "data/span_conf/p4_textvqa_stopconf.jsonl")]


def load(log, conf_path):
    by = defaultdict(lambda: {"slots": defaultdict(dict), "fam": {},
                              "vtok": {}, "q": None, "ans": None, "alen": 0})
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") != "rollout" or r.get("dummy"):
            continue
        a, sid, rep = r["action"], r["sample_id"], r["replay"]
        st = by[sid]
        if a["type"] == "STOP":
            key = "STOP"
        elif a["type"] in ("OCR", "HIGHRES"):
            key = a["type"]
        elif a["type"] == "CROP" and a.get("source") == "grid":
            key = f"CROP#{a['slot']}"
        else:
            continue
        st["slots"][key][rep] = float(r["score_after"])
        st["vtok"][key] = float(r["cost"]["visual_tokens"])
        st["q"] = r["question"]
        if key == "STOP" and rep == 1:
            st["ans"] = r["answer_after"] or ""
            st["alen"] = int(r["cost"].get("tokens_out") or 0)
    conf = {}
    for line in open(ROOT / conf_path):
        r = json.loads(line)
        if r.get("type") != "span_conf":
            continue
        lps = [np.mean(r[k]["logprobs"]) for k in ("r1", "r2")
               if r.get(k) and r[k].get("logprobs")]
        if lps:
            conf[r["sample_id"]] = float(np.mean(lps))
    states = []
    for sid, st in sorted(by.items()):
        keys = sorted(st["slots"])
        if sid not in conf or "STOP" not in keys or "OCR" not in keys \
                or "HIGHRES" not in keys:
            continue
        if any(set(st["slots"][k]) != {1, 2} for k in keys):
            continue
        crops = [k for k in keys if k.startswith("CROP#")]
        pair = lambda k: np.array([st["slots"][k][1], st["slots"][k][2]])
        states.append({
            "sid": sid, "question": st["q"], "answer": st["ans"],
            "answer_len_tokens": st["alen"], "mean_logprob": conf[sid],
            "stop": pair("STOP"), "ocr": pair("OCR"), "hi": pair("HIGHRES"),
            "grid": np.stack([pair(k) for k in crops]),
            "crop": np.mean([pair(k) for k in crops], axis=0),
            "cost_vtok": {"STOP": 0.0, "OCR": st["vtok"]["OCR"],
                          "HIGHRES": st["vtok"]["HIGHRES"],
                          "CROP": float(np.mean([st["vtok"][k] for k in crops]))},
            "slot_vtok": [st["vtok"][k] for k in crops]})
    return states


def boot_ci(d):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    m = np.asarray(d)[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def selector_apply(states, frac, variant):
    if variant == "verbatim":
        srt = SEL["conf_sorted"]["mean"]
    else:
        srt = np.sort([s["mean_logprob"] for s in states])
    rank = lambda v: 1.0 - np.searchsorted(srt, v, side="right") / len(srt)
    X = np.array([extract({"answer": s["answer"], "question": s["question"],
                           "answer_len_tokens": s["answer_len_tokens"]},
                          float(rank(s["mean_logprob"]))) for s in states])
    Xs = SEL["scaler"].transform(X)
    dhat = {f: SEL["heads"][f].predict(Xs) for f in FAMILIES}
    lam = SELM["frozen_lambda_textvqa"][str(frac)]["lam"]
    scores, costs, mix = [], [], {}
    for i, s in enumerate(states):
        vals = {"STOP": 0.0}
        for f in FAMILIES:
            vals[f] = dhat[f][i] - lam * s["cost_vtok"][f]
        fam = max(vals, key=vals.get)
        mix[fam] = mix.get(fam, 0) + 1
        key = {"STOP": "stop", "CROP": "crop", "OCR": "ocr", "HIGHRES": "hi"}[fam]
        scores.append(float(np.mean(s[key])))
        costs.append(s["cost_vtok"][fam])
    return (np.array(scores), float(np.mean(costs)),
            {k: round(v / len(states), 3) for k, v in sorted(mix.items())})


def arm_menu(states):
    n_slots = min(len(s["grid"]) for s in states)
    arms = {"STOP": (np.array([float(np.mean(s["stop"])) for s in states]),
                     np.zeros(len(states))),
            "OCR": (np.array([float(np.mean(s["ocr"])) for s in states]),
                    np.array([s["cost_vtok"]["OCR"] for s in states])),
            "HIGHRES": (np.array([float(np.mean(s["hi"])) for s in states]),
                        np.array([s["cost_vtok"]["HIGHRES"] for s in states])),
            "CROP_uniform": (np.array([float(np.mean(s["crop"])) for s in states]),
                             np.array([s["cost_vtok"]["CROP"] for s in states]))}
    for k in range(n_slots):
        arms[f"CROP_slot{k}"] = (
            np.array([float(np.mean(s["grid"][k])) for s in states]),
            np.array([s["slot_vtok"][k] for s in states]))
    return arms


def best_mixture(arms, B):
    names = sorted(arms)
    ybar = {a: float(arms[a][0].mean()) for a in names}
    cbar = {a: float(arms[a][1].mean()) for a in names}
    best = None
    for a in names:
        if cbar[a] <= B + 1e-9 and (best is None or ybar[a] > best[1]):
            best = ({a: 1.0}, ybar[a])
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lo, hi = sorted((a, b), key=lambda x: cbar[x])
            if not (cbar[lo] <= B + 1e-9 < cbar[hi]):
                continue
            w = (B - cbar[lo]) / (cbar[hi] - cbar[lo])
            v = (1 - w) * ybar[lo] + w * ybar[hi]
            if v > best[1]:
                best = ({lo: 1 - w, hi: w}, v)
    return best


def main():
    out = {}
    md = ["# (v) Cross-reader selector transfer — Qwen-trained selector v2 on "
          "Phi-4 states (zero inference, disclosed post-hoc)", "",
          "Same states/views, Phi-4 outcomes + Phi-4 confidence (verified side",
          "collection). verbatim = Qwen ECDF/λ unchanged; recalibrated = conf",
          "rank within Phi-4's own distribution. Strict; paired 10k bootstrap.",
          ""]
    for ds, log, conf_path in DS:
        states = load(log, conf_path)
        hi = float(np.mean([s["cost_vtok"]["HIGHRES"] for s in states]))
        stop = np.array([float(np.mean(s["stop"])) for s in states])
        arms = arm_menu(states)
        md += [f"## {ds}  (n={len(states)}, always-STOP {stop.mean():.3f}, "
               f"always-HIGHRES vtok {hi:.0f})", "",
               "| budget | variant | selector acc (vtok) | mix-CF | "
               "sel − mix-CF [95% CI] | sel − STOP [95% CI] |",
               "|---|---|---|--:|:--|:--|"]
        res = {}
        for frac in FRACS:
            B = frac * hi
            idx = np.arange(len(states))
            mix_cf = np.zeros(len(states))
            for fold in (0, 1):
                tr = idx[idx % 2 == fold]
                te = idx[idx % 2 != fold]
                arms_tr = {a: (arms[a][0][tr], arms[a][1][tr]) for a in arms}
                w_tr, _ = best_mixture(arms_tr, B)
                mix_cf[te] = sum(w * arms[a][0][te] for a, w in w_tr.items())
            for variant in ("verbatim", "recalibrated"):
                s_acc, s_cost, s_mix = selector_apply(states, frac, variant)
                d_m, d_s = s_acc - mix_cf, s_acc - stop
                ci_m, ci_s = boot_ci(d_m), boot_ci(d_s)
                res[f"{frac}/{variant}"] = {
                    "selector_acc": float(s_acc.mean()), "vtok": s_cost,
                    "mix": s_mix, "mix_cf": float(mix_cf.mean()),
                    "d_mix": float(d_m.mean()), "ci_mix": ci_m,
                    "d_stop": float(d_s.mean()), "ci_stop": ci_s}
                md.append(f"| {frac:g}× | {variant} | {s_acc.mean():.3f} "
                          f"({s_cost:.0f}) | {mix_cf.mean():.3f} | "
                          f"{d_m.mean():+.4f} [{ci_m[0]:+.4f}, {ci_m[1]:+.4f}] | "
                          f"{d_s.mean():+.4f} [{ci_s[0]:+.4f}, {ci_s[1]:+.4f}] |")
        md.append("")
        out[ds] = res
    (ROOT / "paper/cross_reader_transfer.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/cross_reader_transfer.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
