#!/usr/bin/env python3
"""Prospective selector-vs-mixture evaluation — single preregistered pass.

Implements the analysis half of paper/PREREG_PROSPECTIVE_SELECTOR.md on
logs/prospective_textvqa.jsonl. Everything decision-relevant is frozen:
selector v2 (heads/scaler/conf-ECDF/λ per budget) and the mixture weights.
Baselines (random, fixed arms, best-fixed-at-B, in-sample confidence
read-off) and the oracle (Lagrangian + exact MCKP DP) are context.

Outputs paper/prospective_eval.{json,md}.
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
from ppu.evaluator import normalize_answer  # noqa: E402
from ppu.gate_features import extract  # noqa: E402

SEED = 20260831
N_BOOT = 10_000
FRACS = (0.25, 0.5, 1.0)
FAMILIES = ("CROP", "OCR", "HIGHRES")

SEL = pickle.load(open(ROOT / "models/selector_frozen_v2.pkl", "rb"))
SELM = json.loads((ROOT / "models/selector_frozen_v2.json").read_text())
MIX = json.loads((ROOT / "models/mixture_frozen_textvqa.json").read_text())


def lenient(pred, golds):
    p = normalize_answer(pred or "")
    toks = p.split()

    def hit(g):
        g = normalize_answer(g)
        return g != "" and (g == p or (toks and g == toks[0]) or f" {g} " in f" {p} ")
    return min(1.0, sum(hit(g) for g in golds) / 3)


def box_key(box):
    return ",".join(f"{float(v):.4f}" for v in box)


def load_states(log):
    by = defaultdict(lambda: {"arms": defaultdict(dict), "vtok": {}, "conf": {},
                              "gold": None, "question": None, "answer": None,
                              "alen": 0, "ans": defaultdict(dict)})
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") != "rollout":
            continue
        st = by[r["sample_id"]]
        a, rep = r["action"], r["replay"]
        if a["type"] == "STOP":
            key = "STOP"
        elif a["type"] in ("OCR", "HIGHRES"):
            key = a["type"]
        else:
            key = f"CROP@{box_key(a['box'])}"
        st["arms"][key][rep] = float(r["score_after"])
        st["ans"][key][rep] = r["answer_after"] or ""
        st["vtok"][key] = float(r["cost"]["visual_tokens"])
        st["gold"] = r["gold"]
        st["question"] = r["question"]
        if a["type"] == "STOP":
            st["conf"][rep] = (r.get("confidence") or {}).get("mean_logprob")
            if rep == 1:
                st["answer"] = r["answer_after"] or ""
                st["alen"] = int(r["cost"].get("tokens_out") or 0)
    states = []
    for sid, st in sorted(by.items()):
        keys = sorted(st["arms"])
        assert all(set(st["arms"][k]) == {1, 2} for k in keys), sid
        crops = [k for k in keys if k.startswith("CROP@")]
        st.update(sid=sid, keys=keys, crops=crops,
                  mean_logprob=float(np.mean([st["conf"][1], st["conf"][2]])))
        states.append(st)
    return states


def precompute(states):
    """score cache st["S"][scorer][key] -> np.array([r1, r2]); one pass."""
    for st in states:
        st["S"] = {"strict": {}, "lenient": {}}
        for k in st["keys"]:
            st["S"]["strict"][k] = np.array([st["arms"][k][r] for r in (1, 2)])
            st["S"]["lenient"][k] = np.array(
                [lenient(st["ans"][k][r], st["gold"]) for r in (1, 2)])


def scores(st, key, scorer):
    return st["S"][scorer][key]


def fam_score(st, fam, scorer):
    if fam == "STOP":
        return scores(st, "STOP", scorer)
    if fam == "CROP":  # deployable uniform-grid expectation
        return np.mean([scores(st, k, scorer) for k in st["crops"]], axis=0)
    return scores(st, fam, scorer)


def fam_cost(st, fam):
    if fam == "STOP":
        return 0.0
    if fam == "CROP":
        return float(np.mean([st["vtok"][k] for k in st["crops"]]))
    return st["vtok"][fam]


def boot_ci(d, seed=SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def selector_scores(states, frac, scorer):
    rank = lambda v, s=SEL["conf_sorted"]["mean"]: \
        1.0 - np.searchsorted(s, v, side="right") / len(s)
    X = np.array([extract({"answer": st["answer"], "question": st["question"],
                           "answer_len_tokens": st["alen"]},
                          float(rank(st["mean_logprob"]))) for st in states])
    Xs = SEL["scaler"].transform(X)
    dhat = {f: SEL["heads"][f].predict(Xs) for f in FAMILIES}
    lam = SELM["frozen_lambda_textvqa"][str(frac)]["lam"]
    out_s, out_c, mix = [], [], defaultdict(int)
    for i, st in enumerate(states):
        vals = {"STOP": 0.0}
        for f in FAMILIES:
            vals[f] = dhat[f][i] - lam * fam_cost(st, f)
        fam = max(vals, key=vals.get)
        mix[fam] += 1
        out_s.append(float(np.mean(fam_score(st, fam, scorer))))
        out_c.append(fam_cost(st, fam))
    return (np.array(out_s), float(np.mean(out_c)),
            {k: round(v / len(states), 3) for k, v in sorted(mix.items())})


def mixture_scores(states, frac, scorer):
    w = MIX["weights"][str(frac)]["w"]
    out_s, out_c = [], []
    for st in states:
        s = sum(wt * float(np.mean(scores(st, a, scorer))) for a, wt in w.items())
        c = sum(wt * (0.0 if a == "STOP" else st["vtok"][a]) for a, wt in w.items())
        out_s.append(s)
        out_c.append(c)
    return np.array(out_s), float(np.mean(out_c))


def lagrangian_oracle(states, B, scorer):
    keys = states[0]["keys"]
    assert all(st["keys"] == keys for st in states)
    Y = np.array([[scores(st, k, scorer) for k in keys] for st in states])
    C = np.array([[0.0 if k == "STOP" else st["vtok"][k] for k in keys]
                  for st in states])
    best = None
    for lam in [0.0] + list(np.logspace(-7, -0.5, 121)):
        acc = cost = 0.0
        for sel, ev in ((0, 1), (1, 0)):
            idx = np.argmax(Y[:, :, sel] - lam * C, axis=1)
            acc += Y[np.arange(len(states)), idx, ev].mean()
            cost += C[np.arange(len(states)), idx].mean()
        acc, cost = acc / 2, cost / 2
        if cost <= B + 1e-9 and (best is None or acc > best):
            best = float(acc)
    return best


def main():
    states = load_states("logs/prospective_textvqa.jsonl")
    precompute(states)
    n = len(states)
    hi = float(np.mean([st["vtok"]["HIGHRES"] for st in states]))
    out = {"prereg": "paper/PREREG_PROSPECTIVE_SELECTOR.md", "n_states": n,
           "highres_vtok": hi, "results": {}}
    md = ["# Prospective selector-vs-mixture evaluation (preregistered, 1 pass)",
          "",
          f"n = {n} fresh TextVQA states; always-HIGHRES {hi:.0f} tok. Frozen",
          "selector v2 + frozen dev-fitted mixture; paired 10k bootstrap.", ""]
    for scorer in ("strict", "lenient"):
        stop = np.array([float(np.mean(fam_score(st, "STOP", scorer)))
                         for st in states])
        hi_s = np.array([float(np.mean(fam_score(st, "HIGHRES", scorer)))
                         for st in states])
        md += [f"## {scorer}", "",
               f"always-STOP {stop.mean():.3f} · always-HIGHRES {hi_s.mean():.3f} · "
               f"always-CROP {np.mean([np.mean(fam_score(st,'CROP',scorer)) for st in states]):.3f} · "
               f"always-OCR {np.mean([np.mean(fam_score(st,'OCR',scorer)) for st in states]):.3f}",
               "",
               "| budget | selector acc (vtok) | mixture acc (vtok) | "
               "sel − mix [95% CI] | random | best-fixed-at-B | oracle (Lagr.) |",
               "|---|---|---|:--|--:|--:|--:|"]
        res = {}
        for frac in FRACS:
            B = frac * hi
            sel_s, sel_c, sel_mix = selector_scores(states, frac, scorer)
            mix_s, mix_c = mixture_scores(states, frac, scorer)
            d = sel_s - mix_s
            ci = boot_ci(d)
            p_look = min(1.0, B / hi)
            rand = float((1 - p_look) * stop.mean() + p_look * hi_s.mean())
            bf = float(hi_s.mean()) if hi <= B + 1e-9 else float(stop.mean())
            orc = lagrangian_oracle(states, B, scorer)
            res[str(frac)] = {
                "budget_vtok": B, "selector_acc": float(sel_s.mean()),
                "selector_vtok": sel_c, "selector_mix": sel_mix,
                "mixture_acc": float(mix_s.mean()), "mixture_vtok": mix_c,
                "sel_minus_mix": float(d.mean()), "ci95": ci,
                "random": rand, "best_fixed_at_B": bf, "oracle_lagrangian": orc}
            md.append(
                f"| {frac:g}× | {sel_s.mean():.3f} ({sel_c:.0f}) | "
                f"{mix_s.mean():.3f} ({mix_c:.0f}) | {d.mean():+.4f} "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {rand:.3f} | {bf:.3f} | {orc:.3f} |")
        md += ["", f"selector action mix: " + json.dumps(
            {f: res[str(f)]["selector_mix"] for f in FRACS}), ""]
        out["results"][scorer] = res
    (ROOT / "paper/prospective_eval.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/prospective_eval.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
