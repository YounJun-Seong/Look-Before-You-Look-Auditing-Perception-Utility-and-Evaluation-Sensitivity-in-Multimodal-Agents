#!/usr/bin/env python3
"""Human semantic adjudication — analysis (paper/PREREG_HUMAN_ADJUDICATION.md).
Run ONCE after the last verdict export.

  python scripts/human_adjudication_analysis.py \
      --verdicts data/human_adjudication/annotator1.csv annotator2.csv annotator3.csv [--tag human_adjudication]

Verdict files may be the filled annotatorN.csv (comma, UTF-8/BOM) or a TSV.

Verdict TSV columns: item_id, verdict (C/I/A), note. Three independent
annotators; final label = majority vote over {C, I, A} (no majority → the
item is unresolved and excluded; a majority of A → ambiguous, excluded).
Agreement: raw pairwise, mean pairwise Cohen's κ, Fleiss' κ, Krippendorff's
α (nominal). Key numbers on the 200 disagreement items:
P(human correct | strict = 0, lenient > 0) (strict false-negative share) and
P(human incorrect | ...) (lenient false-positive share), Wilson CIs.
Outputs paper/<tag>.{json,md}.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from ppu.evaluator import normalize_answer  # noqa: E402
from strong_controller_common import (FAMILIES, fam_cost, fam_score, features,  # noqa: E402
                                      lenient, load_states, make_rank,
                                      policy_from_dhat, precompute, scores)

SEED = 20260921
N_BOOT = 10_000
LOG = "logs/prospective_textvqa.jsonl"
V2 = pickle.load(open(ROOT / "models/selector_frozen_v2.pkl", "rb"))
V2M = json.loads((ROOT / "models/selector_frozen_v2.json").read_text())
MIX = json.loads((ROOT / "models/mixture_frozen_textvqa.json").read_text())


def read_verdicts(path):
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        delim = "\t" if str(path).endswith(".tsv") else ","
        for row in csv.DictReader(fh, delimiter=delim):
            v = (row.get("verdict") or "").strip().upper()
            if v in ("C", "I", "A"):
                out[row["item_id"].strip()] = v
    return out


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def boot_ci(d, seed=SEED, n=N_BOOT):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d))); m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", nargs="+", required=True)
    ap.add_argument("--tag", default="human_adjudication")
    args = ap.parse_args()
    items = [json.loads(l) for l in open(ROOT / "data/human_adjudication/items.jsonl")]
    by_id = {it["item_id"]: it for it in items}
    pop = json.loads((ROOT / "data/human_adjudication/population_strata.json").read_text())
    V = [read_verdicts(v) for v in args.verdicts]
    names = [Path(v).stem.replace("verdicts_", "") for v in args.verdicts]
    ids = [it["item_id"] for it in items]
    both = [i for i in ids if all(i in v for v in V)]
    out = {"n_items": len(ids), "n_annotators": len(V), "annotators": names,
           "n_annotated_all": len(both), "n_per_annotator": [len(v) for v in V]}

    # 1. agreement
    def fleiss(mat):  # mat: items x categories counts
        mat = np.asarray(mat, float); n = mat.sum(1)[0]
        if len(mat) == 0 or n < 2:
            return float("nan")
        P = ((mat * (mat - 1)).sum(1) / (n * (n - 1))).mean()
        pj = mat.sum(0) / mat.sum(); Pe = (pj ** 2).sum()
        return float((P - Pe) / (1 - Pe)) if Pe < 1 else float("nan")

    def kripp_alpha(mat):  # nominal, from the coincidence matrix
        mat = np.asarray(mat, float); m = mat.sum(1)
        keep = m > 1; mat, m = mat[keep], m[keep]
        if len(mat) == 0:
            return float("nan")
        # observed disagreement: sum over units of (pairs of differing values) / (m_u - 1)
        Do = float(sum(((mat[u] * (m[u] - mat[u])).sum() / (m[u] - 1)) for u in range(len(mat))))
        nc = mat.sum(0); n = nc.sum()
        De = float((nc[:, None] * nc[None, :]).sum() - (nc * nc).sum()) / (n - 1)
        return float(1 - Do / De) if De > 0 else float("nan")  # Do, De both scaled by n

    cats = ("C", "I", "A")
    M3 = [[sum(v[i] == c for v in V) for c in cats] for i in both]
    pair_raw, pair_k = [], []
    for a in range(len(V)):
        for b in range(a + 1, len(V)):
            xa, xb = [V[a][i] for i in both], [V[b][i] for i in both]
            pair_raw.append(float(np.mean([x == y for x, y in zip(xa, xb)]))); pair_k.append(kappa(xa, xb))
    out["iaa"] = {"raw_pairwise_mean": float(np.mean(pair_raw)) if pair_raw else None,
                  "cohen_kappa_pairwise": pair_k, "cohen_kappa_pairwise_mean": float(np.mean(pair_k)) if pair_k else None,
                  "fleiss_kappa_3way": fleiss(M3), "krippendorff_alpha_nominal": kripp_alpha(M3),
                  "verdict_counts": [dict(Counter(v[i] for i in both)) for v in V]}

    # 2. human label = majority
    yH, unresolved, n_amb = {}, [], 0
    for i in both:
        c = Counter(v[i] for v in V); top, k = c.most_common(1)[0]
        if k * 2 > len(V):
            if top == "A":
                n_amb += 1
            else:
                yH[i] = 1.0 if top == "C" else 0.0
        else:
            unresolved.append(i)
    out["human_label"] = {"n_labeled": len(yH), "n_ambiguous_majority": n_amb, "n_unresolved_no_majority": len(unresolved),
                          "n_unanimous": sum(len(set(v[i] for v in V)) == 1 for i in both)}
    A, B, T = V[0], V[1], (V[2] if len(V) > 2 else {})

    # 3. scorer validity
    strata = defaultdict(list)
    for i in yH:
        strata[by_id[i]["stratum"]].append(i)
    dis = strata["disagree"]
    agree = {}
    for sc in ("strict", "lenient"):
        agree[sc] = {st: float(np.mean([float(by_id[i][sc]) == yH[i] for i in strata[st]])) if strata[st] else None
                     for st in ("disagree", "both_correct", "both_wrong")}
    k_c = sum(yH[i] == 1.0 for i in dis)
    n_dis_all = [i for i in both if by_id[i]["stratum"] == "disagree"]
    out["key_numbers"] = {
        "n_disagreement_items_annotated": len(n_dis_all), "n_with_majority_C_or_I": len(dis),
        "P_human_correct_given_strict0_lenientpos": k_c / len(dis) if dis else None,
        "P_human_correct_ci": wilson(k_c, len(dis)),
        "P_human_incorrect_given_strict0_lenientpos": 1 - k_c / len(dis) if dis else None,
        "P_human_incorrect_ci": wilson(len(dis) - k_c, len(dis)),
        "n_ambiguous_or_unresolved_in_disagreement": len(n_dis_all) - len(dis),
        "by_dominant_family": {}, "by_lenient_value": {}}
    for f in ("STOP", "CROP", "OCR", "HIGHRES"):
        sub = [i for i in dis if by_id[i]["dominant_family"] == f]
        out["key_numbers"]["by_dominant_family"][f] = {"n": len(sub), "P_human_correct": float(np.mean([yH[i] for i in sub])) if sub else None}
    for lv in (0.33, 0.67, 1.0):
        sub = [i for i in dis if abs(by_id[i]["l_raw"] - lv) < 0.05]
        out["key_numbers"]["by_lenient_value"][str(lv)] = {"n": len(sub), "P_human_correct": float(np.mean([yH[i] for i in sub])) if sub else None}
    # population-weighted agreement: disagreement stratum has all items (weight = own count),
    # agreement strata weighted by population size using control rates
    Npop = {"disagree": pop["disagree"], "both_correct": pop["both_correct"], "both_wrong": pop["both_wrong"]}
    # note: 'other' items (strict fractional, or strict>0 & lenient==0) are outside the three strata and excluded
    def pop_agree(sc, boot=None):
        rng = np.random.default_rng(seed=SEED) if boot else None
        vals = []
        for _ in range(boot or 1):
            tot = 0.0
            for st, n_st in Npop.items():
                idx = strata[st]
                if not idx:
                    continue
                pick = idx if rng is None else [idx[j] for j in rng.integers(0, len(idx), len(idx))]
                tot += n_st * np.mean([float(by_id[i][sc]) == yH[i] for i in pick])
            vals.append(tot / sum(Npop.values()))
        return vals
    out["scorer_validity"] = {
        "disagreement_items": {"n": len(dis), "human_correct": int(k_c),
                               "lenient_agreement": k_c / len(dis) if dis else None,
                               "strict_agreement": 1 - k_c / len(dis) if dis else None,
                               "wilson_ci_lenient": wilson(k_c, len(dis))},
        "per_stratum_agreement": agree,
        "population_agreement": {sc: {"est": pop_agree(sc)[0], "ci": [float(np.percentile(v, q)) for v, q in
                                                                        [(np.array(pop_agree(sc, 2000)), 2.5), (np.array(pop_agree(sc, 2000)), 97.5)]]}
                                 for sc in ("strict", "lenient")},
        "population_sizes": Npop}

    # 4-5. human-corrected scorer on the 600-state log + policy contrasts
    states, dropped = load_states([LOG]); assert dropped == 0
    precompute(states)
    override = {(by_id[i]["sample_id"], by_id[i]["answer_norm"]): yH[i] for i in yH if by_id[i]["stratum"] == "disagree"}
    for st in states:
        for variant in ("human", "human_sens", "human_sens_any"):
            st["S"][variant] = {}
            for k in st["keys"]:
                vals = []
                for r in (1, 2):
                    key = (st["sid"], normalize_answer(st["ans"][k][r]))
                    if key in override:
                        vals.append(override[key])
                    elif variant == "human_sens" and st["S"]["strict"][k][r - 1] == 0.0 and st["S"]["lenient"][k][r - 1] >= 1.0:
                        vals.append(1.0)  # unadjudicated disagreement: only format-tolerant FULL-credit matches count as correct
                    elif variant == "human_sens_any" and st["S"]["strict"][k][r - 1] == 0.0 and st["S"]["lenient"][k][r - 1] >= 0.5:
                        vals.append(st["S"]["lenient"][k][r - 1])  # earlier variant: any binarized flip (>= 1/2), reported for transparency
                    else:
                        vals.append(st["S"]["strict"][k][r - 1])
                st["S"][variant][k] = np.array(vals)
    n_over = sum(1 for st in states for k in st["keys"] for r in (1, 2)
                 if (st["sid"], normalize_answer(st["ans"][k][r])) in override)
    out["rescoring"] = {"n_rollouts_overridden": n_over, "n_items_overridden": len(override)}
    hi_tok = float(np.mean([st["vtok"]["HIGHRES"] for st in states]))
    rank = make_rank(V2["conf_sorted"]["mean"]); Xz = V2["scaler"].transform(features(states, rank))
    dhat = {f: V2["heads"][f].predict(Xz) for f in FAMILIES}

    def lagr(sc, B):
        keys = states[0]["keys"]
        Y = np.array([[scores(st, k, sc) for k in keys] for st in states])
        C = np.array([[0.0 if k == "STOP" else st["vtok"][k] for k in keys] for st in states]); best = None
        for lam in [0.0] + list(np.logspace(-7, -0.5, 121)):
            acc = cost = 0.0
            for sel, ev in ((0, 1), (1, 0)):
                idx = np.argmax(Y[:, :, sel] - lam * C, axis=1)
                acc += Y[np.arange(len(states)), idx, ev].mean(); cost += C[np.arange(len(states)), idx].mean()
            if cost / 2 <= B + 1e-9 and (best is None or acc / 2 > best):
                best = float(acc / 2)
        return best
    out["policy"] = {}
    for sc in ("strict", "lenient", "human", "human_sens", "human_sens_any"):
        out["policy"][sc] = {"always": {f: float(np.mean([np.mean(fam_score(st, f, sc)) for st in states])) for f in ("STOP",) + FAMILIES}}
        for frac in (0.25, 0.5, 1.0):
            fams = policy_from_dhat(states, dhat, V2M["frozen_lambda_textvqa"][str(frac)]["lam"])
            sel = np.array([float(np.mean(fam_score(st, f, sc))) for st, f in zip(states, fams)])
            w = MIX["weights"][str(frac)]["w"]
            mix = np.array([sum(wt * float(np.mean(scores(st, a, sc))) for a, wt in w.items()) for st in states])
            d = sel - mix
            out["policy"][sc][str(frac)] = {"selector": float(sel.mean()), "mixture": float(mix.mean()),
                                            "delta": float(d.mean()), "ci": boot_ci(d), "oracle": lagr(sc, frac * hi_tok)}

    # 6. examples
    ex_ok = [i for i in dis if yH[i] == 1.0][:10]
    ex_fp = [i for i in dis if yH[i] == 0.0]
    def ex(i):
        it = by_id[i]; return {"item_id": i, "question": it["question"], "gold": dict(Counter(x.lower().strip() for x in it["gold"])),
                              "answer": it["answer_raw"], "verdicts": [v.get(i) for v in V], "A": A.get(i), "B": B.get(i), "third": T.get(i)}
    out["examples"] = {"strict_wrong_lenient_right_human_correct": [ex(i) for i in ex_ok],
                       "lenient_false_positives_human_incorrect": [ex(i) for i in ex_fp]}

    # markdown
    sv = out["scorer_validity"]; ia = out["iaa"]; hl = out["human_label"]
    md = [f"# Human semantic adjudication (prereg: PREREG_HUMAN_ADJUDICATION.md)", "",
          f"Items {out['n_items']}; annotators {out['annotators']}; annotated by all: {out['n_annotated_all']}.", "",
          "## Inter-annotator agreement", "",
          f"- raw pairwise agreement (mean) {ia['raw_pairwise_mean']:.3f}; pairwise Cohen's κ {[round(k, 3) for k in ia['cohen_kappa_pairwise']]} (mean {ia['cohen_kappa_pairwise_mean']:.3f})",
          f"- Fleiss' κ {ia['fleiss_kappa_3way']:.3f}; Krippendorff's α (nominal) {ia['krippendorff_alpha_nominal']:.3f}; verdict counts per annotator {ia['verdict_counts']}",
          f"- majority label on {hl['n_labeled']} items (unanimous {hl['n_unanimous']}); ambiguous-majority {hl['n_ambiguous_majority']}, no-majority {hl['n_unresolved_no_majority']}", "",
          "## Key numbers (disagreement items: strict = 0, format-tolerant > 0)", "",
          f"- **P(human correct | strict=0, lenient>0) = {out['key_numbers']['P_human_correct_given_strict0_lenientpos']:.3f} "
          f"[{out['key_numbers']['P_human_correct_ci'][0]:.3f}, {out['key_numbers']['P_human_correct_ci'][1]:.3f}]** (strict false-negative share; n={out['key_numbers']['n_with_majority_C_or_I']})",
          f"- **P(human incorrect | strict=0, lenient>0) = {out['key_numbers']['P_human_incorrect_given_strict0_lenientpos']:.3f} "
          f"[{out['key_numbers']['P_human_incorrect_ci'][0]:.3f}, {out['key_numbers']['P_human_incorrect_ci'][1]:.3f}]** (lenient false-positive share)",
          f"- ambiguous / no-majority among the annotated disagreement items: {out['key_numbers']['n_ambiguous_or_unresolved_in_disagreement']}",
          "- by dominant family: " + "; ".join(f"{f} n={v['n']} P(correct)={'—' if v['P_human_correct'] is None else round(v['P_human_correct'], 3)}" for f, v in out['key_numbers']['by_dominant_family'].items()),
          "- by lenient value: " + "; ".join(f"{lv} n={v['n']} P(correct)={'—' if v['P_human_correct'] is None else round(v['P_human_correct'], 3)}" for lv, v in out['key_numbers']['by_lenient_value'].items()), "",
          "## Scorer validity against the human label", "",
          f"- Disagreement items (n={sv['disagreement_items']['n']}): human says correct {sv['disagreement_items']['human_correct']} → "
          f"**lenient agreement {sv['disagreement_items']['lenient_agreement']:.3f} [{sv['disagreement_items']['wilson_ci_lenient'][0]:.3f}, {sv['disagreement_items']['wilson_ci_lenient'][1]:.3f}], "
          f"strict agreement {sv['disagreement_items']['strict_agreement']:.3f}** (lenient here = format-tolerant score > 0)",
          "", "| stratum | n | strict agrees w/ human | lenient agrees w/ human |", "|---|--:|--:|--:|"]
    for st in ("disagree", "both_correct", "both_wrong"):
        n_st = len(strata[st]); a_s, a_l = agree["strict"][st], agree["lenient"][st]
        md.append(f"| {st} | {n_st} | {'—' if a_s is None else f'{a_s:.3f}'} | {'—' if a_l is None else f'{a_l:.3f}'} |")
    pa = sv["population_agreement"]
    md += ["", f"- Population-weighted agreement over all {sum(Npop.values())} unique items: strict **{pa['strict']['est']:.3f}** [{pa['strict']['ci'][0]:.3f}, {pa['strict']['ci'][1]:.3f}], "
           f"lenient **{pa['lenient']['est']:.3f}** [{pa['lenient']['ci'][0]:.3f}, {pa['lenient']['ci'][1]:.3f}]", "",
           "## Policy contrast on the 600-state prospective set under each scorer", "",
           f"Human-corrected scorer overrides {out['rescoring']['n_rollouts_overridden']} rollouts ({out['rescoring']['n_items_overridden']} adjudicated items); "
           "human_sens = unadjudicated disagreements scored lenient.", "",
           "| scorer | always-STOP | always-HIGHRES | ¼× sel − mix [CI] | ½× sel − mix [CI] | 1× sel − mix [CI] | oracle ½× |", "|---|--:|--:|:--|:--|:--|--:|"]
    for sc in ("strict", "lenient", "human", "human_sens", "human_sens_any"):
        P = out["policy"][sc]; f = lambda fr: f"{P[str(fr)]['delta']:+.4f} [{P[str(fr)]['ci'][0]:+.4f}, {P[str(fr)]['ci'][1]:+.4f}]"
        md.append(f"| {sc} | {P['always']['STOP']:.3f} | {P['always']['HIGHRES']:.3f} | {f(0.25)} | {f(0.5)} | {f(1.0)} | {P['0.5']['oracle']:.3f} |")
    md += ["", "## Examples", "", "Strict wrong / lenient right / human correct (up to 10):", ""]
    for e in out["examples"]["strict_wrong_lenient_right_human_correct"]:
        md.append(f"- Q: {e['question']} | gold: {e['gold']} | answer: **{e['answer']}** (verdicts {e['verdicts']})")
    md += ["", f"Lenient false positives (human incorrect): {len(out['examples']['lenient_false_positives_human_incorrect'])}", ""]
    for e in out["examples"]["lenient_false_positives_human_incorrect"]:
        md.append(f"- Q: {e['question']} | gold: {e['gold']} | answer: **{e['answer']}** (verdicts {e['verdicts']})")
    (ROOT / f"paper/{args.tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (ROOT / f"paper/{args.tag}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
