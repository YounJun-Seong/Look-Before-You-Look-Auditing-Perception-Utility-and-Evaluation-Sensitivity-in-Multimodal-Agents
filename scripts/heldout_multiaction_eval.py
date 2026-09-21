#!/usr/bin/env python3
"""Frozen multi-action policy — single held-out evaluation.

Implements the evaluation half of paper/PREREG_MULTIACTION_HELDOUT.md. Run
ONCE per the prereg, after the held-out CROP/OCR arms are collected. Every
learned component (value heads, scaler, conf ECDF, v2.2 gate) is loaded from
its frozen artifact; nothing is fitted here.

Data per state: STOP/HIGHRES outcomes and STOP confidence from the 2026-08-23
fresh logs (canonical); CROP/OCR outcomes from logs/heldout_{ds}_arms.jsonl;
the re-issued STOP in the arms log is only a determinism check.

Outputs paper/heldout_multiaction.{json,md}.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ppu.evaluator import normalize_answer  # noqa: E402
from ppu.gate_features import extract  # noqa: E402

SEED = 20260828
THR = 0.5
N_BOOT = 10_000
FAMILIES = ("CROP", "OCR", "HIGHRES")
DS = [("vqav2", "logs/fresh_vqav2_orig.jsonl", "logs/heldout_vqav2_arms.jsonl"),
      ("textvqa", "logs/fresh_textvqa_orig.jsonl", "logs/heldout_textvqa_arms.jsonl")]

POL = pickle.load(open(ROOT / "models/multiaction_frozen_v1.pkl", "rb"))
GATE = pickle.load(open(ROOT / "models/gate_frozen_v2.2_pointwise.pkl", "rb"))
DEV_CDF = GATE["conf_transform"]["sorted_dev_conf"]


def vqa_strict(pred, golds):
    p = normalize_answer(pred or "")
    return min(1.0, sum(normalize_answer(g) == p for g in golds) / 3)


def vqa_lenient(pred, golds):
    p = normalize_answer(pred or "")
    toks = p.split()

    def hit(g):
        g = normalize_answer(g)
        return g != "" and (g == p or (toks and g == toks[0]) or f" {g} " in f" {p} ")
    return min(1.0, sum(hit(g) for g in golds) / 3)


def load_log(log):
    by = defaultdict(dict)
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") == "rollout":
            by[r["sample_id"]][(r["action"]["key"], r["replay"])] = r
    return by


def load_states(ds, fresh_log, arms_log):
    fresh, arms = load_log(fresh_log), load_log(arms_log)
    states, det_mismatch = [], 0
    for sid, fd in fresh.items():
        need_f = [("STOP#0", 1), ("STOP#0", 2)]
        hi_keys = sorted({k for k, _ in fd if k.startswith("HIGHRES")})
        if sid not in arms or not hi_keys or not all(k in fd for k in need_f):
            continue
        ad = arms[sid]
        crop_keys = sorted({k for k, _ in ad if k.startswith("CROP")})
        if not crop_keys or ("OCR#0", 1) not in ad or ("OCR#0", 2) not in ad:
            continue
        for rep in (1, 2):  # determinism check on the re-issued STOP
            if (ad.get(("STOP#0", rep), {}).get("answer_after")
                    != fd[("STOP#0", rep)]["answer_after"]):
                det_mismatch += 1
        st = fd[("STOP#0", 1)]
        cf = [fd[("STOP#0", r)].get("confidence") or {} for r in (1, 2)]
        if any(c.get("mean_logprob") is None for c in cf):
            continue

        def pair(d, key, scorer=None):
            recs = [d[(key, 1)], d[(key, 2)]]
            if scorer is None:
                return np.array([float(r["score_after"]) for r in recs])
            return np.array([scorer(r["answer_after"], r["gold"]) for r in recs])

        row = {
            "sid": sid, "question": st["question"],
            "answer": st["answer_after"] or "",
            "answer_len_tokens": int(st["cost"].get("tokens_out") or 0),
            "mean_logprob": float(np.mean([c["mean_logprob"] for c in cf])),
            "gold": st["gold"],
            "recs": {"stop": [fd[("STOP#0", r)] for r in (1, 2)],
                     "hi": [fd[(hi_keys[0], r)] for r in (1, 2)],
                     "ocr": [ad[("OCR#0", r)] for r in (1, 2)],
                     "crops": [[ad[(k, r)] for r in (1, 2)] for k in crop_keys]},
        }
        for scname, scorer in (("strict", None),
                               ("lenient", vqa_lenient if ds == "vqav2" else None)):
            if scname == "lenient" and scorer is None:
                continue
            sc = {"stop": pair(fd, "STOP#0", scorer), "hi": pair(fd, hi_keys[0], scorer),
                  "ocr": pair(ad, "OCR#0", scorer),
                  "grid": np.stack([pair(ad, k, scorer) for k in crop_keys])}
            sc["crop"] = sc["grid"].mean(axis=0)
            row[scname] = sc
        cost_v, cost_l = {}, {}
        for fam, recs in (("STOP", row["recs"]["stop"]), ("HIGHRES", row["recs"]["hi"]),
                          ("OCR", row["recs"]["ocr"])):
            cost_v[fam] = float(np.mean([r["cost"]["visual_tokens"] for r in recs]))
            cost_l[fam] = float(np.mean([r["cost"]["latency_s"]
                                         + r["cost"].get("ocr_latency_s", 0) for r in recs]))
        cost_v["STOP"] = 0.0
        cost_l["STOP"] = 0.0
        crop_recs = [r for kk in row["recs"]["crops"] for r in kk]
        cost_v["CROP"] = float(np.mean([r["cost"]["visual_tokens"] for r in crop_recs]))
        cost_l["CROP"] = float(np.mean([r["cost"]["latency_s"] for r in crop_recs]))
        row["cost_vtok"], row["cost_lat"] = cost_v, cost_l
        states.append(row)
    return states, det_mismatch


def feats_frozen(states):
    srt = POL["conf_sorted_mean_logprob"]

    def rank(lp):
        return 1.0 - np.searchsorted(srt, lp, side="right") / len(srt)
    X = np.array([extract({"answer": s["answer"], "question": s["question"],
                           "answer_len_tokens": s["answer_len_tokens"]},
                          float(rank(s["mean_logprob"]))) for s in states], float)
    return POL["scaler"].transform(X)


def v22_scores(states):
    def conf_pct(x):
        return float(1.0 - np.searchsorted(DEV_CDF, x) / len(DEV_CDF))
    X = np.array([extract({"answer": s["answer"], "question": s["question"],
                           "answer_len_tokens": s["answer_len_tokens"]},
                          conf_pct(s["mean_logprob"])) for s in states], float)
    return GATE["model"].predict_proba(GATE["scaler"].transform(X))[:, 1]


def evaluate(states, choice, scorer):
    scores, vt, lt, mix = [], [], [], defaultdict(int)
    for s in states:
        fam = choice[s["sid"]]
        mix[fam] += 1
        key = {"STOP": "stop", "OCR": "ocr", "HIGHRES": "hi", "CROP": "crop"}[fam]
        scores.append(float(np.mean(s[scorer][key])))
        vt.append(s["cost_vtok"][fam])
        lt.append(s["cost_lat"][fam])
    n = len(states)
    return {"acc": float(np.mean(scores)), "vtok": float(np.mean(vt)),
            "lat": float(np.mean(lt)), "look_rate": 1.0 - mix["STOP"] / n,
            "mix": {k: round(v / n, 4) for k, v in sorted(mix.items())},
            "_scores": np.array(scores)}


def oracle_row(states, scorer):
    scores, vt, lt, mix = [], [], [], defaultdict(int)
    for s in states:
        arms = [("STOP", s[scorer]["stop"], 0.0, 0.0),
                ("OCR", s[scorer]["ocr"], s["cost_vtok"]["OCR"], s["cost_lat"]["OCR"]),
                ("HIGHRES", s[scorer]["hi"], s["cost_vtok"]["HIGHRES"], s["cost_lat"]["HIGHRES"])]
        arms += [("CROP", s[scorer]["grid"][i], s["cost_vtok"]["CROP"], s["cost_lat"]["CROP"])
                 for i in range(len(s[scorer]["grid"]))]
        picked = []
        for sel, ev in ((0, 1), (1, 0)):
            i = int(np.argmax([a[1][sel] for a in arms]))
            picked.append((arms[i][1][ev], arms[i][2], arms[i][3], arms[i][0]))
        scores.append(np.mean([p[0] for p in picked]))
        vt.append(np.mean([p[1] for p in picked]))
        lt.append(np.mean([p[2] for p in picked]))
        for p in picked:
            mix[p[3]] += 0.5
    n = len(states)
    return {"acc": float(np.mean(scores)), "vtok": float(np.mean(vt)),
            "lat": float(np.mean(lt)), "look_rate": 1.0 - mix["STOP"] / n,
            "mix": {k: round(v / n, 4) for k, v in sorted(mix.items())},
            "_scores": np.array(scores)}


def boot_ci(delta, seed=SEED, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), (n_boot, len(delta)))
    m = delta[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def sweep_threshold(states, score, scorer):
    out = []
    for tau in list(np.unique(np.quantile(score, np.linspace(0, 1, 41)))) + [np.inf]:
        choice = {s["sid"]: ("HIGHRES" if score[i] >= tau else "STOP")
                  for i, s in enumerate(states)}
        ev = evaluate(states, choice, scorer)
        ev["tau"] = float(tau)
        out.append(ev)
    return out


def sweep_multi(states, dhat, scorer):
    out = {}
    for axis, lams in (("vtok", POL["lam_tok"]), ("lat", POL["lam_lat"])):
        pts = []
        for lam in lams:
            choice = {}
            for i, s in enumerate(states):
                vals = {"STOP": 0.0}
                for f in FAMILIES:
                    vals[f] = dhat[f][i] - lam * s["cost_vtok" if axis == "vtok" else "cost_lat"][f]
                choice[s["sid"]] = max(vals, key=vals.get)
            ev = evaluate(states, choice, scorer)
            ev["lambda"] = float(lam)
            pts.append(ev)
        out[f"multiaction_{axis}"] = pts
    return out


def main():
    report = {"prereg": "paper/PREREG_MULTIACTION_HELDOUT.md", "seed": SEED,
              "policy": "models/multiaction_frozen_v1.pkl", "datasets": {}}
    for ds, fresh_log, arms_log in DS:
        if not (ROOT / arms_log).exists():
            print("missing", arms_log, "— skip")
            continue
        states, det_mm = load_states(ds, fresh_log, arms_log)
        print(f"== {ds}: {len(states)} states, STOP determinism mismatches {det_mm}")
        X = feats_frozen(states)
        dhat = {f: POL["heads"][f].predict(X) for f in FAMILIES}
        conf = np.array([1.0 - np.searchsorted(POL["conf_sorted_mean_logprob"],
                                               s["mean_logprob"], side="right")
                         / len(POL["conf_sorted_mean_logprob"]) for s in states])
        p22 = v22_scores(states)

        entry = {"n_states": len(states), "stop_determinism_mismatch": det_mm,
                 "scorers": {}}
        scorers = ["strict"] + (["lenient"] if ds == "vqav2" else [])
        for scorer in scorers:
            rows = {}
            for fam in ("STOP", "HIGHRES", "OCR", "CROP"):
                rows[f"always_{fam}"] = evaluate(states, {s["sid"]: fam for s in states}, scorer)
            bf = POL["best_fixed_frozen"]
            rows["best_fixed_frozen"] = dict(rows[f"always_{bf}"], arm=bf)
            rows["oracle_cross_replay"] = oracle_row(states, scorer)
            # random allocation: expectation of Look-with-HIGHRES at rate q
            stop_m = float(np.mean([np.mean(s[scorer]["stop"]) for s in states]))
            hi_m = float(np.mean([np.mean(s[scorer]["hi"]) for s in states]))
            hi_v = float(np.mean([s["cost_vtok"]["HIGHRES"] for s in states]))
            hi_l = float(np.mean([s["cost_lat"]["HIGHRES"] for s in states]))
            random_line = [{"look_rate": q, "acc": (1 - q) * stop_m + q * hi_m,
                            "vtok": q * hi_v, "lat": q * hi_l}
                           for q in np.linspace(0, 1, 21)]

            sweeps = {"confidence": sweep_threshold(states, conf, scorer),
                      "v22_gate": sweep_threshold(states, p22, scorer),
                      "random_allocation": random_line}
            sweeps.update(sweep_multi(states, dhat, scorer))

            stop_sc = rows["always_STOP"]["_scores"]
            for key in ("oracle_cross_replay",):
                rows[key]["delta_vs_stop"] = float(rows[key]["_scores"].mean() - stop_sc.mean())
                rows[key]["delta_ci95"] = boot_ci(rows[key]["_scores"] - stop_sc)
            lam0 = sweeps["multiaction_vtok"][0]
            lam0["delta_vs_stop"] = float(lam0["_scores"].mean() - stop_sc.mean())
            lam0["delta_ci95"] = boot_ci(lam0["_scores"] - stop_sc)
            lam0["delta_vs_bestfixed"] = float(
                lam0["_scores"].mean() - rows["best_fixed_frozen"]["_scores"].mean())
            lam0["delta_vs_bestfixed_ci95"] = boot_ci(
                lam0["_scores"] - rows["best_fixed_frozen"]["_scores"])

            # pre-declared matched-budget read-offs vs best-fixed-at-B
            hi_tok = rows["always_HIGHRES"]["vtok"]
            budgets = []
            for frac in (0.25, 0.5, 1.0):
                B = hi_tok * frac
                pt = {"budget_frac": frac, "budget_vtok": round(B, 1)}
                for name, sw in (("confidence", sweeps["confidence"]),
                                 ("v22_gate", sweeps["v22_gate"]),
                                 ("multiaction", sweeps["multiaction_vtok"])):
                    ok = [p for p in sw if p["vtok"] <= B + 1e-9]
                    best = max(ok, key=lambda p: p["acc"]) if ok else None
                    pt[name] = ({k: best[k] for k in ("acc", "vtok", "lat", "look_rate", "mix")}
                                if best else None)
                    if best is not None:
                        pt[f"_scores_{name}"] = best["_scores"]
                # best fixed under budget B: the frozen fixed arm if affordable, else STOP
                bf_row = (rows["best_fixed_frozen"] if rows["best_fixed_frozen"]["vtok"] <= B + 1e-9
                          else rows["always_STOP"])
                pt["best_fixed_at_B"] = {k: bf_row[k] for k in ("acc", "vtok", "lat")}
                if pt.get("_scores_multiaction") is not None:
                    d = pt["_scores_multiaction"] - bf_row["_scores"]
                    pt["multi_minus_bestfixed"] = {"delta": float(d.mean()), "ci95": boot_ci(d)}
                    for other in ("confidence", "v22_gate"):
                        if pt.get(f"_scores_{other}") is not None:
                            d = pt["_scores_multiaction"] - pt[f"_scores_{other}"]
                            pt[f"multi_minus_{other}"] = {"delta": float(d.mean()),
                                                          "ci95": boot_ci(d)}
                for k in list(pt):
                    if k.startswith("_scores_"):
                        del pt[k]
                budgets.append(pt)

            for r in rows.values():
                r.pop("_scores", None)
            for sw in sweeps.values():
                for p in sw:
                    p.pop("_scores", None)
            entry["scorers"][scorer] = {"rows": rows, "sweeps": sweeps,
                                        "budget_table": budgets}
        report["datasets"][ds] = entry

    Path(ROOT / "paper/heldout_multiaction.json").write_text(json.dumps(report, indent=1))
    print("wrote paper/heldout_multiaction.json (md summary: build after inspection,"
          " numbers frozen in json)")


if __name__ == "__main__":
    main()
