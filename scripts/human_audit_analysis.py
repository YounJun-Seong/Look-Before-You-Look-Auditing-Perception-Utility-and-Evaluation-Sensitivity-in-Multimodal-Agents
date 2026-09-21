#!/usr/bin/env python3
"""Blinded human-semantic utility audit — analysis (see PREREG_HUMAN_AUDIT.md).

Run ONCE after the sheets are fully annotated:

    python3 scripts/human_audit_analysis.py \
        [--sheet2-vqav2 second_annotator.tsv --sheet2-textvqa ...]

Consumes paper/human_audit_sheet_{ds}.tsv (verdict column: C/P/I/A), rebuilds
four-way PPU labels from human scores with the paper's own rule, and
re-evaluates the FROZEN logged confidence against the human-defined target.
Confidence is never recomputed. Outputs paper/human_audit_results.{json,md}.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 20260828
THR = 0.5
LOGS = {"vqav2": "logs/fresh_vqav2_orig.jsonl",
        "textvqa": "logs/fresh_textvqa_orig.jsonl"}
VALID = {"C", "P", "I", "A"}


def norm(a: str) -> str:
    return " ".join((a or "").lower().split())


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0:
        return "HELP"
    if d1 < 0 and d2 < 0:
        return "HARM"
    if d1 == 0 and d2 == 0:
        return "NEUTRAL"
    return "AMBIGUOUS"


def read_sheet(path):
    verdicts = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            v = (row["verdict"] or "").strip().upper()
            if v:
                if v not in VALID:
                    raise SystemExit(f"invalid verdict {v!r} for {row['item_id']}")
                verdicts[row["item_id"]] = v
    return verdicts


def load_log(log):
    by = defaultdict(dict)
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") == "rollout":
            by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r
    return by


def build(ds, verdicts, partial_as):
    items = [json.loads(l) for l in open(ROOT / f"data/human_audit/items_{ds}.jsonl")]
    vmap = {}  # (sample_id, answer_norm) -> verdict
    missing = 0
    for it in items:
        v = verdicts.get(it["item_id"])
        if v is None:
            missing += 1
            continue
        vmap[(it["sample_id"], it["answer_norm"])] = v
    if missing:
        print(f"  WARNING {ds}: {missing} items without verdict")

    def y_h(sid, answer):
        v = vmap.get((sid, norm(answer)))
        if v is None or v == "A":
            return None
        return {"C": 1.0, "I": 0.0, "P": partial_as}[v]

    rows = []
    for sid, d in load_log(LOGS[ds]).items():
        need = [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]
        if not all(k in d for k in need):
            continue
        strict = {k: float(d[k]["score_after"]) for k in need}
        human = {k: y_h(sid, d[k]["answer_after"]) for k in need}
        st = d[("STOP", 1)]
        cf = [d[("STOP", r)].get("confidence") or {} for r in (1, 2)]
        rows.append({
            "sid": sid,
            "strict_label": four_way(*[strict[k] for k in need]),
            "human_label": ("AMBIGUOUS" if any(v is None for v in human.values())
                            else four_way(*[human[k] for k in need])),
            "human": human, "strict": strict,
            "flip": (len((st["answer_after"] or "").split()) > 1
                     and len((d[("HIGHRES", 1)]["answer_after"] or "").split()) <= 1),
            "mean_lp": (float(np.mean([c["mean_logprob"] for c in cf]))
                        if all(c.get("mean_logprob") is not None for c in cf) else None),
        })
    return rows


def auc_ci(y, s, rng, n=3000):
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(roc_auc_score(y[i], s[i]))
    return [float(q) for q in np.quantile(v, [0.025, 0.975])]


def analyse(ds, rows, rng):
    out = {"n_states": len(rows)}
    trans = Counter((r["strict_label"], r["human_label"]) for r in rows)
    out["transition"] = {f"{a}->{b}": c for (a, b), c in sorted(trans.items())}

    flips = [r for r in rows if r["strict_label"] == "HELP" and r["flip"]]
    out["strict_flip_help"] = {"n": len(flips),
                               "human_labels": dict(Counter(r["human_label"] for r in flips))}
    nonflip_help = [r for r in rows if r["strict_label"] == "HELP" and not r["flip"]]
    out["strict_nonflip_help"] = {"n": len(nonflip_help),
                                  "human_labels": dict(Counter(r["human_label"] for r in nonflip_help))}

    for target in ("strict", "human"):
        use = [r for r in rows if r[f"{target}_label"] != "AMBIGUOUS"
               and r["mean_lp"] is not None]
        y = np.array([r[f"{target}_label"] == "HELP" for r in use], int)
        if 0 < y.sum() < len(y):
            s = -np.array([r["mean_lp"] for r in use])
            out[f"conf_auc_{target}"] = {"n": len(use), "n_help": int(y.sum()),
                                         "auc": float(roc_auc_score(y, s)),
                                         "ci": auc_ci(y, s, rng)}
    hu = [r for r in rows if all(v is not None for v in r["human"].values())]
    out["human_acc"] = {
        "n": len(hu),
        "stop": float(np.mean([(r["human"][("STOP", 1)] + r["human"][("STOP", 2)]) / 2 for r in hu])),
        "highres": float(np.mean([(r["human"][("HIGHRES", 1)] + r["human"][("HIGHRES", 2)]) / 2 for r in hu])),
    }
    out["strict_acc"] = {
        "stop": float(np.mean([(r["strict"][("STOP", 1)] + r["strict"][("STOP", 2)]) / 2 for r in rows])),
        "highres": float(np.mean([(r["strict"][("HIGHRES", 1)] + r["strict"][("HIGHRES", 2)]) / 2 for r in rows])),
    }
    return out


def kappa(v1, v2):
    keys = sorted(set(v1) & set(v2))
    if not keys:
        return None, 0
    a = [v1[k] for k in keys]
    b = [v2[k] for k in keys]
    po = float(np.mean([x == y for x, y in zip(a, b)]))
    cats = sorted(set(a) | set(b))
    pe = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else None, len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet2-vqav2", default="")
    ap.add_argument("--sheet2-textvqa", default="")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    report = {"prereg": "paper/PREREG_HUMAN_AUDIT.md", "seed": SEED,
              "partial_mapping_primary": 0.0, "datasets": {}}
    for ds in ("vqav2", "textvqa"):
        sheet = ROOT / f"paper/human_audit_sheet_{ds}.tsv"
        verdicts = read_sheet(sheet)
        print(f"{ds}: {len(verdicts)} verdicts")
        entry = {"n_verdicts": len(verdicts),
                 "verdict_counts": dict(Counter(verdicts.values()))}
        second = getattr(args, f"sheet2_{ds}")
        if second:
            v2 = read_sheet(second)
            k, n = kappa(verdicts, v2)
            entry["cohens_kappa"] = {"kappa": k, "n_overlap": n}
        for name, pmap in (("primary_P0", 0.0), ("sensitivity_P1", 1.0)):
            rows = build(ds, verdicts, pmap)
            entry[name] = analyse(ds, rows, rng)
        report["datasets"][ds] = entry

    Path(ROOT / "paper/human_audit_results.json").write_text(json.dumps(report, indent=1))

    lines = ["# Blinded human-semantic utility audit — results", "",
             "Analysis exactly as pre-registered (PREREG_HUMAN_AUDIT.md); primary",
             "mapping P→0, sensitivity P→1. Confidence is the logged whole-reply",
             "mean logprob — never recomputed.", ""]
    for ds, e in report["datasets"].items():
        p = e["primary_P0"]
        lines += [f"## {ds} (n={p['n_states']} states; verdicts {e['verdict_counts']})", ""]
        if "cohens_kappa" in e:
            lines += [f"Cohen's κ = {e['cohens_kappa']['kappa']:.3f} "
                      f"(n={e['cohens_kappa']['n_overlap']})", ""]
        lines += ["Strict → human label transitions: " + json.dumps(p["transition"]), "",
                  f"Format-flip strict-HELP (n={p['strict_flip_help']['n']}): "
                  f"human labels {p['strict_flip_help']['human_labels']}",
                  f"Non-flip strict-HELP (n={p['strict_nonflip_help']['n']}): "
                  f"human labels {p['strict_nonflip_help']['human_labels']}", ""]
        for t in ("strict", "human"):
            d = p.get(f"conf_auc_{t}")
            if d:
                lines.append(f"Frozen confidence AUC vs U_{t}: {d['auc']:.3f} "
                             f"[{d['ci'][0]:.2f}, {d['ci'][1]:.2f}] "
                             f"(n={d['n']}, HELP={d['n_help']})")
        lines += ["", f"Accuracy (strict): STOP {100 * p['strict_acc']['stop']:.1f}% / "
                  f"HIGHRES {100 * p['strict_acc']['highres']:.1f}%",
                  f"Accuracy (human, P→0): STOP {100 * p['human_acc']['stop']:.1f}% / "
                  f"HIGHRES {100 * p['human_acc']['highres']:.1f}%", ""]
        s = e["sensitivity_P1"]
        d0, d1 = p.get("conf_auc_human"), s.get("conf_auc_human")
        if d0 and d1:
            lines += [f"Sensitivity P→1: AUC vs U_human {d1['auc']:.3f} "
                      f"[{d1['ci'][0]:.2f}, {d1['ci'][1]:.2f}]", ""]
    Path(ROOT / "paper/human_audit_results.md").write_text("\n".join(lines) + "\n")
    print("wrote paper/human_audit_results.{json,md}")


if __name__ == "__main__":
    main()
