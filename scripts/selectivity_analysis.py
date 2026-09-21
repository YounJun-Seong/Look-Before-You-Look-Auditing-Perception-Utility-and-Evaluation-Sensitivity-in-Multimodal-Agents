#!/usr/bin/env python
"""When does question-conditioned perception beat question-agnostic perception?

The dummy arm was designed as a luck floor. On TextVQA it beat the structured
oracle, and the reason turned out not to be a broken control: a random local
zoom on a text-dense image really does acquire task-relevant evidence. So the
arm is re-read as three separate layers instead of one (user, 2026-08-17):

    Control 1  stochasticity floor      NOOP + re-encodes: no new spatial evidence
    Control 2  question-agnostic        random crops: new evidence, no question
    Structured question-conditioned     grid + VLM-proposed crops, OCR, HIGHRES

The sharpest contrast is cardinality-matched: 8 question-conditioned crops vs 8
question-agnostic crops, chosen on R1 and scored on R2 exactly like every other
oracle in the audit. Everything here is computed from existing logs.

    python scripts/selectivity_analysis.py --out paper/selectivity_analysis.md
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ppu.audit import THRESHOLD, SampleView, load
from ppu.datasets import load_manifest

RUNS = [
    ("MME-RealWorld-Lite", "logs/g0_mme200.jsonl",
     "data/manifests/mme_realworld_lite.jsonl", "data/ocr_cache.json"),
    ("TextVQA", "logs/tv_200.jsonl",
     "data/manifests/textvqa_val.jsonl", "data/ocr_cache_textvqa.json"),
]


def _oracle(v: SampleView, keys: list[str], r1: dict, r2: dict) -> float:
    """Choose on R1, score on R2, ties by key for determinism."""
    usable = [k for k in keys if k in r1 and k in r2]
    if not usable:
        return float("nan")
    best = min(usable, key=lambda k: (-r1[k], k))
    return r2[best]


def layers(v: SampleView) -> dict[str, float]:
    """The three layers plus the single-observation references."""
    crops = sorted(k for k in v.r1 if k.startswith("CROP#"))
    rnd = sorted(k for k in v.dummy_r1 if k.startswith("CROP_RANDOM#"))
    stoch = sorted(k for k in v.dummy_r1
                   if k.startswith(("NOOP#", "REENCODE#")))
    out = {
        "stop": v.r2.get("STOP#0", float("nan")),
        "full": v.full_r2 if v.full_r2 is not None else float("nan"),
        "structured_crops": _oracle(v, crops, v.r1, v.r2),
        "agnostic_crops": _oracle(v, rnd, v.dummy_r1, v.dummy_r2),
        "stochasticity": _oracle(v, stoch, v.dummy_r1, v.dummy_r2),
        "n_structured": len(crops),
        "n_agnostic": len(rnd),
    }
    return out


def feature_table(rows: list[dict], key: str, bins: int = 3) -> list[tuple]:
    """Win/loss of question-conditioned crops over question-agnostic ones,
    tabulated across quantile bins of a sample feature."""
    vals = [r[key] for r in rows if r[key] is not None]
    if len(set(vals)) < bins:
        return []
    edges = [np.quantile(vals, q) for q in np.linspace(0, 1, bins + 1)[1:-1]]

    def which(x):
        i = 0
        for e in edges:
            if x > e:
                i += 1
        return i

    out = []
    for b in range(bins):
        sel = [r for r in rows if r[key] is not None and which(r[key]) == b]
        if not sel:
            continue
        d = [r["delta"] for r in sel]
        lo = min(r[key] for r in sel)
        hi = max(r[key] for r in sel)
        out.append((f"{lo:.0f}–{hi:.0f}", len(sel), 100 * st.mean(d),
                    sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/selectivity_analysis.md")
    args = ap.parse_args()

    md = ["# When does selective perception pay off?", "",
          "Question-conditioned vs question-agnostic perception, computed from the "
          "Gate 0 logs. The dummy arm is read as two distinct layers, not one "
          "luck floor.", ""]
    all_rows: list[dict] = []

    for name, log, manifest, ocr_path in RUNS:
        if not Path(log).exists():
            continue
        recs = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
        views = load([r for r in recs if r.get("type") == "rollout"])
        meta = {s.sample_id: s for s in load_manifest(manifest)}
        ocr = json.loads(Path(ocr_path).read_text()) if Path(ocr_path).exists() else {}

        rows = []
        for sid, v in views.items():
            if not (v.dummy_r1 and v.dummy_r2):
                continue  # the contrast needs both arms
            L = layers(v)
            if any(np.isnan(L[k]) for k in ("structured_crops", "agnostic_crops")):
                continue
            s = meta.get(sid)
            lines = ocr.get(s.image_path, {}).get("lines", []) if s else []
            rows.append({
                "dataset": name, "sample_id": sid,
                "stop": L["stop"], "full": L["full"],
                "structured": L["structured_crops"],
                "agnostic": L["agnostic_crops"],
                "stoch": L["stochasticity"],
                "delta": (L["structured_crops"] >= THRESHOLD)
                         - (L["agnostic_crops"] >= THRESHOLD),
                "native_tokens": s.native_tokens if s else None,
                "ocr_lines": len(lines),
                "ocr_area": 100 * sum((b[2] - b[0]) * (b[3] - b[1])
                                      for _, b, _ in lines) if lines else 0.0,
                "coarse_right": int(L["stop"] >= THRESHOLD),
                "domain": (s.meta.get("subtask") or s.meta.get("category") or "-")
                          if s else "-",
            })
        all_rows.extend(rows)

        n = len(rows)
        mean = lambda k: 100 * st.mean([r[k] for r in rows])
        md += [f"## {name} (n = {n} with both arms)", "",
               "| layer | what it buys | accuracy |", "|---|---|--:|",
               f"| STOP (coarse only) | nothing | {mean('stop'):.1f} |",
               f"| Control 1 — stochasticity floor | re-ask, re-encode "
               f"(no new spatial evidence) | {mean('stoch'):.1f} |",
               f"| Control 2 — question-agnostic | 8 random crops | "
               f"{mean('agnostic'):.1f} |",
               f"| Structured | 8 question-conditioned crops | "
               f"{mean('structured'):.1f} |",
               f"| FULL | the whole image at native budget | {mean('full'):.1f} |",
               "",
               f"**Selectivity advantage** (structured − question-agnostic): "
               f"**{mean('structured') - mean('agnostic'):+.1f}pp** "
               f"— wins {sum(1 for r in rows if r['delta'] > 0)}, "
               f"losses {sum(1 for r in rows if r['delta'] < 0)}, "
               f"ties {sum(1 for r in rows if r['delta'] == 0)}", ""]

        if name.startswith("MME"):
            md += ["| domain | n | selectivity advantage (pp) | wins | losses |",
                   "|---|--:|--:|--:|--:|"]
            by_dom = defaultdict(list)
            for r in rows:
                by_dom[r["domain"]].append(r)
            for dom, rs in sorted(by_dom.items(), key=lambda kv: -len(kv[1])):
                d = [r["delta"] for r in rs]
                md.append(f"| {dom} | {len(rs)} | {100 * st.mean(d):+.1f} | "
                          f"{sum(1 for x in d if x > 0)} | {sum(1 for x in d if x < 0)} |")
            md.append("")

    # --- what predicts the advantage, pooled across datasets ----------------
    md += ["## What predicts the selectivity advantage", "",
           "Per-sample outcome of question-conditioned crops minus "
           "question-agnostic crops (+1 / 0 / -1), binned by sample features.", ""]
    for key, label in [("native_tokens", "native resolution (visual tokens)"),
                       ("ocr_lines", "text density (OCR lines)"),
                       ("ocr_area", "text coverage (% of image area)")]:
        tbl = feature_table(all_rows, key)
        if not tbl:
            continue
        md += [f"### by {label}", "",
               "| bin | n | advantage (pp) | wins | losses |", "|---|--:|--:|--:|--:|"]
        for rng, n, adv, w, l in tbl:
            md.append(f"| {rng} | {n} | {adv:+.1f} | {w} | {l} |")
        md.append("")

    # coarse-correct split: selection should matter most where coarse fails
    md += ["### by whether the coarse view already answered it", "",
           "| coarse | n | advantage (pp) | wins | losses |", "|---|--:|--:|--:|--:|"]
    for flag, lab in ((1, "already right"), (0, "wrong")):
        sel = [r for r in all_rows if r["coarse_right"] == flag]
        if sel:
            d = [r["delta"] for r in sel]
            md.append(f"| {lab} | {len(sel)} | {100 * st.mean(d):+.1f} | "
                      f"{sum(1 for x in d if x > 0)} | {sum(1 for x in d if x < 0)} |")
    md.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    json.dump(all_rows, open(str(out.with_suffix(".json")), "w"), indent=1)
    print("\n".join(md))
    print(f"\n-> {out}  ({len(all_rows)} samples with both arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
