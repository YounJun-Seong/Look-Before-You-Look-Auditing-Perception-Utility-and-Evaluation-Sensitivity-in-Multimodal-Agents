#!/usr/bin/env python3
"""Cross-reader opportunity agreement CONDITIONAL on baseline capability
(Qwen vs Phi-4; post-hoc, zero inference; addendum 2026-09-20).

Motivation: pooled TextVQA help agreement is low (Jaccard .19, kappa -.05)
but the readers' STOP accuracies differ hugely (Qwen .60 vs Phi-4 .26 strict),
so disagreement could be a capability artifact rather than reader-specific
perception need. Here the Gate-0 states are stratified by whether each
reader's STOP (mean over both replays) is correct, and agreement is
recomputed inside each stratum:

  both STOP wrong / both STOP correct / mixed (exactly one correct)

Per state and reader, deterministic arms only (grid CROPs, OCR, HIGHRES):
  help   = some additional arm strictly beats STOP (mean over replays)
  PPU    = max_a y(a) - y(STOP)      (best-arm opportunity, signed)
  PPU_f  = mean_{a in family} y(a) - y(STOP)  per family (CROP uniform, OCR, HIGHRES)
Reported per stratum: n, P(help) per reader, Jaccard, Cohen's kappa,
Pearson/Spearman corr of PPU across readers, per-family PPU correlation,
best-family agreement given both-help. 10k state-paired bootstrap CIs
(seed 20260920) for Jaccard and Pearson. TextVQA under strict AND the frozen
lenient rule (Phi-4's strict STOP accuracy is largely a format artifact);
MME is MCQ so lenient == strict and only strict is reported.
Outputs paper/cross_reader_conditional.{json,md}.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppu.evaluator import normalize_answer  # noqa: E402

SEED = 20260920
N_BOOT = 10_000
EPS = 1e-9
PAIRS = [("mme", "logs/g0_mme200.jsonl", "logs/p4_mme200.jsonl", ("strict",)),
         ("textvqa", "logs/tv_200.jsonl", "logs/p4_textvqa200.jsonl",
          ("strict", "lenient"))]
FAMS = ("CROP", "OCR", "HIGHRES")


def lenient(pred, golds):  # frozen rule of scripts/p4_lenient_rescore.py
    p = normalize_answer(pred or "")
    toks = p.split()

    def hit(g):
        g = normalize_answer(g)
        return g != "" and (g == p or (toks and g == toks[0]) or f" {g} " in f" {p} ")
    return min(1.0, sum(hit(g) for g in golds) / 3)


def det_fam(a):
    if a["type"] == "CROP" and a.get("source") == "grid":
        return "CROP"
    if a["type"] in ("OCR", "HIGHRES"):
        return a["type"]
    return None


def load(log, scorer):
    stop = defaultdict(list)
    arms = defaultdict(lambda: defaultdict(list))
    fam_of = {}
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") != "rollout" or r.get("dummy"):
            continue
        a, sid = r["action"], r["sample_id"]
        sc = (lenient(r["answer_after"], r["gold"]) if scorer == "lenient"
              else float(r["score_after"]))
        if a["type"] == "STOP":
            stop[sid].append(sc)
            continue
        f = det_fam(a)
        if f:
            key = a.get("key") or f"{a['type']}#{a.get('slot', 0)}"
            arms[sid][key].append(sc)
            fam_of[key] = f
    out = {}
    for sid in stop:
        if sid not in arms:
            continue
        s = float(np.mean(stop[sid]))
        per = {k: float(np.mean(v)) for k, v in arms[sid].items()}
        best_key = max(sorted(per), key=lambda k: per[k])
        fam_delta = {}
        for f in FAMS:
            vals = [v for k, v in per.items() if fam_of[k] == f]
            fam_delta[f] = float(np.mean(vals) - s) if vals else None
        out[sid] = {"stop": s, "stop_ok": s >= 0.5, "ppu": per[best_key] - s,
                    "help": per[best_key] > s + EPS,
                    "best_family": fam_of[best_key], "fam_delta": fam_delta}
    return out


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def jaccard(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = np.sum(a | b)
    return float(np.sum(a & b) / u) if u else float("nan")


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    from scipy.stats import rankdata
    return pearson(rankdata(x), rankdata(y))


def boot(fn, *cols, n_boot=N_BOOT):
    cols = [np.asarray(c) for c in cols]
    n = len(cols[0])
    if n < 3:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(fn(*[c[idx] for c in cols]))
    vals = np.array(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def stratum_stats(q, p, sids):
    n = len(sids)
    if n == 0:
        return {"n": 0}
    qh = np.array([q[s]["help"] for s in sids])
    ph = np.array([p[s]["help"] for s in sids])
    qu = np.array([q[s]["ppu"] for s in sids])
    pu = np.array([p[s]["ppu"] for s in sids])
    both = int(np.sum(qh & ph))
    fam_pairs = [(q[s]["best_family"], p[s]["best_family"])
                 for s in sids if q[s]["help"] and p[s]["help"]]
    row = {
        "n": n, "both": both, "qwen_only": int(np.sum(qh & ~ph)),
        "phi4_only": int(np.sum(~qh & ph)), "neither": int(np.sum(~qh & ~ph)),
        "p_help_qwen": float(qh.mean()), "p_help_phi4": float(ph.mean()),
        "p_phi_given_qwen": float(both / qh.sum()) if qh.sum() else None,
        "p_qwen_given_phi": float(both / ph.sum()) if ph.sum() else None,
        "jaccard": jaccard(qh, ph), "jaccard_ci": boot(jaccard, qh, ph),
        "kappa": kappa(qh, ph), "kappa_ci": boot(kappa, qh, ph),
        "ppu_mean_qwen": float(qu.mean()), "ppu_mean_phi4": float(pu.mean()),
        "ppu_pearson": pearson(qu, pu), "ppu_pearson_ci": boot(pearson, qu, pu),
        "ppu_spearman": spearman(qu, pu),
        "best_family_agreement_given_both": (
            float(np.mean([a == b for a, b in fam_pairs])) if fam_pairs else None),
        "n_both_help": len(fam_pairs),
        "family_ppu_pearson": {},
    }
    for f in FAMS:
        xs = [(q[s]["fam_delta"][f], p[s]["fam_delta"][f]) for s in sids
              if q[s]["fam_delta"][f] is not None and p[s]["fam_delta"][f] is not None]
        if xs:
            x, y = zip(*xs)
            row["family_ppu_pearson"][f] = {"r": pearson(x, y), "n": len(xs)}
    return row


def fmt(v, d=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def fmt_ci(v, ci, d=2):
    return f"{fmt(v, d)} [{fmt(ci[0], d)}, {fmt(ci[1], d)}]"


def main():
    out = {}
    md = ["# Cross-reader opportunity agreement, conditional on baseline capability",
          "", "Qwen2.5-VL-7B vs Phi-4-multimodal on the identical Gate-0 states and views.",
          "help = some deterministic additional arm strictly beats STOP (mean over both",
          "replays); PPU = best-arm gain over STOP. Strata by whether each reader's STOP",
          "(mean over replays) is correct (>= 0.5). 10k state-paired bootstrap CIs.", ""]
    for ds, qlog, plog, scorers in PAIRS:
        for scorer in scorers:
            q, p = load(qlog, scorer), load(plog, scorer)
            sids = sorted(set(q) & set(p))
            strata = {
                "pooled": sids,
                "both_stop_wrong": [s for s in sids if not q[s]["stop_ok"] and not p[s]["stop_ok"]],
                "both_stop_correct": [s for s in sids if q[s]["stop_ok"] and p[s]["stop_ok"]],
                "mixed": [s for s in sids if q[s]["stop_ok"] != p[s]["stop_ok"]],
                "qwen_correct_phi_wrong": [s for s in sids if q[s]["stop_ok"] and not p[s]["stop_ok"]],
                "qwen_wrong_phi_correct": [s for s in sids if not q[s]["stop_ok"] and p[s]["stop_ok"]],
            }
            res = {k: stratum_stats(q, p, v) for k, v in strata.items()}
            res["stop_acc_qwen"] = float(np.mean([q[s]["stop"] for s in sids]))
            res["stop_acc_phi4"] = float(np.mean([p[s]["stop"] for s in sids]))
            out[f"{ds}/{scorer}"] = res
            md += [f"## {ds} — {scorer} scorer  (n={len(sids)}; STOP acc Qwen "
                   f"{res['stop_acc_qwen']:.3f}, Phi-4 {res['stop_acc_phi4']:.3f})", "",
                   "| stratum | n | P(help) Q / P4 | Both | Q only | P4 only | Neither | "
                   "Jaccard [CI] | κ [CI] | corr(PPU) Pearson [CI] | Spearman | "
                   "best-family agree (n both) |",
                   "|---|--:|:--|--:|--:|--:|--:|:--|:--|:--|--:|:--|"]
            for k in ("pooled", "both_stop_wrong", "both_stop_correct", "mixed",
                      "qwen_correct_phi_wrong", "qwen_wrong_phi_correct"):
                r = res[k]
                if r["n"] == 0:
                    md.append(f"| {k} | 0 | | | | | | | | | | |")
                    continue
                md.append(
                    f"| {k} | {r['n']} | {r['p_help_qwen']:.2f} / {r['p_help_phi4']:.2f} | "
                    f"{r['both']} | {r['qwen_only']} | {r['phi4_only']} | {r['neither']} | "
                    f"{fmt_ci(r['jaccard'], r['jaccard_ci'])} | "
                    f"{fmt_ci(r['kappa'], r['kappa_ci'])} | "
                    f"{fmt_ci(r['ppu_pearson'], r['ppu_pearson_ci'])} | "
                    f"{fmt(r['ppu_spearman'])} | "
                    f"{fmt(r['best_family_agreement_given_both'])} ({r['n_both_help']}) |")
            md += ["", "Per-family PPU correlation (Pearson, family-mean gain over STOP):", "",
                   "| stratum | " + " | ".join(FAMS) + " |", "|---|" + "--:|" * len(FAMS)]
            for k in ("pooled", "both_stop_wrong", "both_stop_correct", "mixed"):
                r = res[k]
                if r["n"] == 0:
                    continue
                md.append(f"| {k} | " + " | ".join(
                    fmt(r["family_ppu_pearson"].get(f, {}).get("r")) for f in FAMS) + " |")
            md.append("")
    (ROOT / "paper/cross_reader_conditional.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/cross_reader_conditional.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
