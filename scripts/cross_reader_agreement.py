#!/usr/bin/env python3
"""Cross-reader opportunity agreement (post-hoc, zero inference).

Same Gate-0 states, same views, two readers (Qwen logs g0_mme200/tv_200,
Phi-4 logs p4_mme200/p4_textvqa200). Per state and reader, on deterministic
arms only (grid CROPs, OCR, HIGHRES; scores = mean over the two replays):

  help(reader)  = some additional arm strictly beats STOP
  best family   = family of the highest-scoring additional arm (ties: by
                  fixed family order CROP < OCR < HIGHRES after sort key)

Reports the Both / Qwen-only / Phi-4-only / Neither decomposition,
P(Phi help | Qwen help) and vice versa, Jaccard, Cohen's kappa, and — among
states where both readers have help — best-family agreement.

Outputs paper/cross_reader_agreement.{json,md}.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("mme", "logs/g0_mme200.jsonl", "logs/p4_mme200.jsonl"),
         ("textvqa", "logs/tv_200.jsonl", "logs/p4_textvqa200.jsonl")]
EPS = 1e-9


def det_arm(a):
    if a["type"] == "CROP" and a.get("source") == "grid":
        return "CROP"
    if a["type"] in ("OCR", "HIGHRES"):
        return a["type"]
    return None


def load(log):
    stop = defaultdict(list)          # sid -> [scores]
    arms = defaultdict(lambda: defaultdict(list))  # sid -> key -> [scores]
    fam = {}
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") != "rollout":
            continue
        a, sid = r["action"], r["sample_id"]
        if a["type"] == "STOP":
            stop[sid].append(r["score_after"])
            continue
        f = det_arm(a)
        if f:
            key = a.get("key") or f"{a['type']}#{a.get('slot', 0)}"
            arms[sid][key].append(r["score_after"])
            fam[key] = f
    out = {}
    for sid in stop:
        if sid not in arms:
            continue
        s = sum(stop[sid]) / len(stop[sid])
        best_key, best = None, -1.0
        for k in sorted(arms[sid]):
            v = sum(arms[sid][k]) / len(arms[sid][k])
            if v > best + EPS:
                best_key, best = k, v
        out[sid] = {"stop": s, "best": best, "help": best > s + EPS,
                    "best_family": fam[best_key]}
    return out


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    out = {}
    md = ["# Cross-reader opportunity agreement (Qwen vs Phi-4, post-hoc)", "",
          "help = some deterministic additional arm strictly beats STOP",
          "(mean over both replays; grid CROPs, OCR, HIGHRES). Same states,",
          "same views, strict scores as logged.", ""]
    for ds, qlog, plog in PAIRS:
        q, p = load(qlog), load(plog)
        sids = sorted(set(q) & set(p))
        qh = [q[s]["help"] for s in sids]
        ph = [p[s]["help"] for s in sids]
        both = sum(a and b for a, b in zip(qh, ph))
        qonly = sum(a and not b for a, b in zip(qh, ph))
        ponly = sum(b and not a for a, b in zip(qh, ph))
        neither = sum(not a and not b for a, b in zip(qh, ph))
        jac = both / (both + qonly + ponly) if both + qonly + ponly else 0.0
        fam_pairs = [(q[s]["best_family"], p[s]["best_family"])
                     for s, a, b in zip(sids, qh, ph) if a and b]
        fam_agree = (sum(x == y for x, y in fam_pairs) / len(fam_pairs)
                     if fam_pairs else None)
        conf = Counter(fam_pairs)
        row = {"n": len(sids), "both": both, "qwen_only": qonly,
               "phi4_only": ponly, "neither": neither,
               "p_help_qwen": sum(qh) / len(sids),
               "p_help_phi4": sum(ph) / len(sids),
               "p_phi_given_qwen": both / sum(qh) if sum(qh) else None,
               "p_qwen_given_phi": both / sum(ph) if sum(ph) else None,
               "jaccard": jac, "cohens_kappa": kappa(qh, ph),
               "best_family_agreement_given_both": fam_agree,
               "family_confusion": {f"{a}->{b}": c for (a, b), c in
                                    sorted(conf.items())}}
        out[ds] = row
        n = len(sids)
        md += [f"## {ds}  (n={n})", "",
               f"| Both | Qwen only | Phi-4 only | Neither |", "|--:|--:|--:|--:|",
               f"| {both} ({100*both/n:.0f}%) | {qonly} ({100*qonly/n:.0f}%) | "
               f"{ponly} ({100*ponly/n:.0f}%) | {neither} ({100*neither/n:.0f}%) |", "",
               f"- P(help): Qwen {row['p_help_qwen']:.2f}, Phi-4 {row['p_help_phi4']:.2f}",
               f"- P(Phi-4 help | Qwen help) = {row['p_phi_given_qwen']:.2f}; "
               f"P(Qwen help | Phi-4 help) = {row['p_qwen_given_phi']:.2f}",
               f"- Jaccard {jac:.2f}; Cohen's kappa {row['cohens_kappa']:.2f}",
               f"- Best-family agreement given both-help: "
               f"{'—' if fam_agree is None else f'{fam_agree:.2f}'} "
               f"(n={len(fam_pairs)})", ""]
    (ROOT / "paper/cross_reader_agreement.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/cross_reader_agreement.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
