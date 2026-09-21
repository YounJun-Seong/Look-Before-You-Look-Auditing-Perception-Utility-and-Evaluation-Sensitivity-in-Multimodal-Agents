#!/usr/bin/env python3
"""Second-reader replication table (paper/PREREG_SECOND_READER.md, output 1-2).

Compares the frozen-audit numbers of the Qwen Gate 0 runs with the Phi-4
replication runs. Every number comes from `ppu.audit` JSON output produced by
scripts/analyze_gate1.py with identical settings — no reader-specific logic.

Usage (after the Phi-4 audits exist):
    python scripts/second_reader_analysis.py \
        --qwen mme=paper/gate0_mme_audit.json textvqa=paper/gate0_textvqa_audit.json \
        --phi4 mme=paper/p4_mme_audit.json textvqa=paper/p4_textvqa_audit.json

Writes paper/second_reader_replication.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = ("STOP", "CROP", "OCR", "HIGHRES")


def pooled(path):
    d = json.loads(Path(ROOT / path).read_text())
    return d["pooled"] if "pooled" in d else d


def row(p):
    acc, hr, harm, div = p["accuracy"], p["headroom"], p["harm"], p["diversity"]
    return {
        "n": p.get("n"),
        "stop": acc["coarse_stop_r2"],
        "full": acc.get("full_image_r2"),
        "best_fixed": acc["best_fixed_cf"],
        "oracle": acc["oracle_cf"],
        "P_policy": hr["P_policy"], "P_ci": hr["P_ci"],
        "H_net": hr["H_net"], "H_ci": hr["H_ci"],
        "dmg": harm["dmg_rate"], "harm_by_family": harm["by_family"],
        "n_decisive": div["n_decisive"],
        "stop_optimal": div["distribution"].get("STOP"),
        "dist": div["distribution"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", nargs="+", required=True)
    ap.add_argument("--phi4", nargs="+", required=True)
    ap.add_argument("--out", default="second_reader_replication")
    args = ap.parse_args()
    qwen = {k: row(pooled(v)) for k, v in (kv.split("=", 1) for kv in args.qwen)}
    phi4 = {k: row(pooled(v)) for k, v in (kv.split("=", 1) for kv in args.phi4)}

    P = lambda v: "—" if v is None else f"{100*v:.1f}"
    CI = lambda c: f"[{100*c[0]:+.1f}, {100*c[1]:+.1f}]"

    md = ["# Second-reader replication — Qwen2.5-VL-7B vs Phi-4-multimodal",
          "",
          "Frozen `ppu.audit` on identical states, views and replay procedure",
          "(prereg: paper/PREREG_SECOND_READER.md). All numbers %, strict scorer.",
          ""]
    for ds in qwen:
        q, p = qwen[ds], phi4[ds]
        md += [f"## {ds}  (n = {q['n']} / {p['n']})", "",
               "| metric | Qwen | Phi-4 |", "|---|--:|--:|",
               f"| coarse STOP accuracy | {P(q['stop'])} | {P(p['stop'])} |",
               f"| FULL accuracy | {P(q['full'])} | {P(p['full'])} |",
               f"| best fixed (cross-fitted) | {P(q['best_fixed'])} | {P(p['best_fixed'])} |",
               f"| cross-replay oracle (cf) | {P(q['oracle'])} | {P(p['oracle'])} |",
               f"| **P_policy (headroom)** | **{100*q['P_policy']:+.1f}** {CI(q['P_ci'])} "
               f"| **{100*p['P_policy']:+.1f}** {CI(p['P_ci'])} |",
               f"| H_net | {100*q['H_net']:+.1f} {CI(q['H_ci'])} "
               f"| {100*p['H_net']:+.1f} {CI(p['H_ci'])} |",
               f"| any-action DMG | {P(q['dmg'])} | {P(p['dmg'])} |",
               f"| P(harm\\|STOP✓): CROP | {P(q['harm_by_family'].get('CROP'))} "
               f"| {P(p['harm_by_family'].get('CROP'))} |",
               f"| P(harm\\|STOP✓): OCR | {P(q['harm_by_family'].get('OCR'))} "
               f"| {P(p['harm_by_family'].get('OCR'))} |",
               f"| P(harm\\|STOP✓): HIGHRES | {P(q['harm_by_family'].get('HIGHRES'))} "
               f"| {P(p['harm_by_family'].get('HIGHRES'))} |",
               f"| STOP-optimal among decisive | {P(q['stop_optimal'])} "
               f"(n={q['n_decisive']}) | {P(p['stop_optimal'])} (n={p['n_decisive']}) |",
               "",
               "Best-action family distribution among decisive states (oracle picks):",
               "",
               "| reader | " + " | ".join(FAMILIES) + " |",
               "|---|" + "--:|" * len(FAMILIES),
               "| Qwen | " + " | ".join(P(q["dist"].get(f)) for f in FAMILIES) + " |",
               "| Phi-4 | " + " | ".join(P(p["dist"].get(f)) for f in FAMILIES) + " |",
               ""]
    # pooled headroom across datasets (unweighted mean of per-dataset P_policy)
    qp = sum(qwen[d]["P_policy"] for d in qwen) / len(qwen)
    pp = sum(phi4[d]["P_policy"] for d in phi4) / len(phi4)
    md += [f"Pooled (unweighted mean) P_policy: Qwen {100*qp:+.1f}pp, "
           f"Phi-4 {100*pp:+.1f}pp.", ""]
    (ROOT / f"paper/{args.out}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
