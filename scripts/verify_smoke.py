#!/usr/bin/env python
"""Instrumentation checks for the 5-sample GPU smoke.

Accuracy is irrelevant here. The only question is whether the measurement
apparatus is telling the truth: budgets, distinctness of observations, replay
symmetry, gold-letter tracking under permutation, and log completeness.

Any FAIL means the ladder stops — a 200-sample campaign on a lying pipeline is
worse than no campaign, because the numbers look publishable.

    python scripts/verify_smoke.py logs/g0_smoke5.jsonl \
        --manifest data/manifests/mme_realworld_lite.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppu.audit import load
from ppu.datasets import load_manifest

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((ok, name, detail))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--coarse-tokens", type=int, default=256)
    ap.add_argument("--crop-tokens", type=int, default=256)
    ap.add_argument("--highres-tokens", type=int, default=1280)
    ap.add_argument("--min-gap", type=float, default=0.20,
                    help="minimum FULL-over-HIGHRES token gap (fraction)")
    args = ap.parse_args()

    rows, notes = [], []
    with open(args.log, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            (rows if obj.get("type") == "rollout" else notes).append(obj)
    if not rows:
        print("no rollout rows", file=sys.stderr)
        return 2

    samples = {s.sample_id: s for s in load_manifest(args.manifest)}
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sample[r["sample_id"]].append(r)

    def views_of(r):
        return r.get("views", [])

    # 1. coarse budget actually applied
    coarse = [v for r in rows for v in views_of(r) if v["label"].startswith("coarse")]
    check("coarse view respects the budget",
          bool(coarse) and all(0.75 * args.coarse_tokens <= v["tokens"] <= args.coarse_tokens
                               for v in coarse),
          f"n={len(coarse)} tokens={sorted({v['tokens'] for v in coarse})}")

    # 2. CROP adds exactly the crop budget, on top of a retained coarse view
    crops = [r for r in rows if r["action"]["type"] in ("CROP", "CROP_RANDOM")]
    ok = all(len(views_of(r)) == 2
             and views_of(r)[0]["label"].startswith("coarse")
             and views_of(r)[1]["tokens"] <= args.crop_tokens
             and r["cost"]["visual_tokens"] == views_of(r)[1]["tokens"]
             for r in crops)
    check("CROP = coarse + zoom, charged at the zoom's tokens", ok and bool(crops),
          f"n={len(crops)}")

    # 3. HIGHRES and FULL are different real inputs
    pairs = 0
    ok = True
    gaps: list[float] = []
    for sid, rs in by_sample.items():
        hi = [r for r in rs if r["action"]["type"] == "HIGHRES"]
        fu = [r for r in rs if r["action"]["type"] == "FULL"]
        if not hi or not fu:
            continue
        pairs += 1
        h, f = views_of(hi[0])[0], views_of(fu[0])[0]
        # Distinct pixels is not enough: if HIGHRES saturates the native
        # resolution it becomes observationally the same action as FULL, and
        # the perceptual ladder collapses to two rungs.
        gap = (f["tokens"] - h["tokens"]) / max(h["tokens"], 1)
        gaps.append(gap)
        ok &= (h["sha"] != f["sha"]) and (f["tokens"] > h["tokens"] > args.coarse_tokens)
        ok &= gap >= args.min_gap
    check("coarse < HIGHRES < FULL with a real token gap", ok and pairs > 0,
          f"{pairs} samples, FULL-over-HIGHRES gap min={min(gaps or [0]):.0%} "
          f"median={sorted(gaps)[len(gaps) // 2]:.0%} (required >= {args.min_gap:.0%})")

    # 4. FULL never enters the candidate set / oracle
    views = load(rows)
    stray = {k for v in views.values() for k in v.keys if k.startswith("FULL")}
    sizes = {len(v.keys) for v in views.values()}
    check("FULL is a reference row, not a candidate",
          not stray and sizes == {11}, f"candidate set sizes={sizes} stray={stray}")

    # 5. the eight crops are eight different images
    ok, n = True, 0
    for sid, rs in by_sample.items():
        cr = [r for r in rs if r["action"]["type"] == "CROP" and r["replay"] == 1]
        if not cr:
            continue
        n += 1
        boxes = {tuple(r["action"]["box"]) for r in cr}
        shas = {views_of(r)[1]["sha"] for r in cr}
        ok &= len(cr) == 8 and len(boxes) == 8 and len(shas) == 8
    check("8 CROP candidates are 8 distinct crops", ok and n > 0, f"{n} samples")

    # 6. dummy pool holds 11 distinct draws
    dummy_samples = {r["sample_id"] for r in rows if r.get("dummy")}
    ok = True
    for sid in dummy_samples:
        rs = [r for r in by_sample[sid] if r.get("dummy") and r["replay"] == 1]
        keys = {r["action"]["key"] for r in rs}
        shas = {tuple(v["sha"] for v in views_of(r)) for r in rs}
        ok &= len(rs) == 11 and len(keys) == 11 and len(shas) == 11
    check("dummy pool: 11 distinct draws" if dummy_samples
          else "dummy pool: not run in this smoke (skipped)",
          ok, f"{len(dummy_samples)} samples")

    # 7. OCR runs once per image, then serves from cache
    done = [n for n in notes if n.get("level") == "done"]
    n_img = len({r["image_path"] for r in rows})
    if done:
        d = done[-1]
        check("OCR engine called once per image",
              d.get("ocr_engine_calls") == n_img,
              f"engine_calls={d.get('ocr_engine_calls')} images={n_img} "
              f"cache_hits={d.get('ocr_cache_hits')}")
    else:
        check("OCR engine called once per image", False, "no 'done' note in log")

    # 8. replays differ only in what is asked
    ok, n = True, 0
    for sid, rs in by_sample.items():
        per_action = defaultdict(dict)
        for r in rs:
            per_action[r["action"]["key"]][r["replay"]] = r
        for key, d in per_action.items():
            if 1 in d and 2 in d:
                n += 1
                same_pixels = [v["sha"] for v in views_of(d[1])] == \
                              [v["sha"] for v in views_of(d[2])]
                ok &= same_pixels
    check("R1 and R2 see identical pixels", ok and n > 0, f"{n} action pairs")

    mcq = [r for r in rows if samples.get(r["sample_id"]) and
           samples[r["sample_id"]].metric == "mcq"]
    changed = sum(1 for sid, rs in by_sample.items()
                  for key in {r["action"]["key"] for r in rs}
                  if len({r["question_served"] for r in rs
                          if r["action"]["key"] == key}) > 1)
    # Binding for every dataset: MCQ permutes options, open-ended paraphrases.
    # A zero here means R2 is a byte-identical re-run and controls nothing.
    check("R2 changes the question rendering", changed > 0,
          f"{changed} actions whose served question differs across replays")

    # 9. permuted gold letter points at the same option text
    ok, n = True, 0
    for r in rows:
        s = samples.get(r["sample_id"])
        if not s or s.metric != "mcq":
            continue
        n += 1
        gold_letter = r["gold"][0].strip().upper()[:1]
        # option text under the *served* rendering
        served_lines = [ln for ln in r["question_served"].splitlines()
                        if len(ln) > 2 and ln[1] == "." and ln[0] in "ABCDE"]
        table = {ln[0]: ln[2:].strip() for ln in served_lines}
        orig_letter = s.answers[0].strip().upper()[:1]
        orig_text = s.choices["ABCDE".index(orig_letter)]
        ok &= table.get(gold_letter, "") == orig_text
    if n:
        check("gold letter follows the permutation", ok, f"{n} MCQ rows")
    else:
        check("gold letter follows the permutation: open-ended set (skipped)",
              True, "no MCQ rows")

    # 10. log completeness
    required = ("answer_before", "answer_after", "score_before", "score_after",
                "cost", "views", "decode", "config_hash", "raw_generation",
                "model_id", "prompt_version")
    missing = {f for r in rows for f in required if f not in r}
    bad_cost = [r for r in rows
                if not {"visual_tokens", "latency_s", "text_tokens_in", "tokens_out"}
                <= set(r["cost"]) or r["cost"]["latency_s"] <= 0]
    check("every rollout logs answers, scores, tokens, latency",
          not missing and not bad_cost,
          f"missing={missing or '-'} bad_cost_rows={len(bad_cost)}")

    errors = [r for r in rows if r.get("error")]
    parse_fail = [r for r in rows if not r["parse_ok"]]
    check("no transport errors", not errors, f"{len(errors)} rows with error")
    check("parse failure rate under 2%",
          len(parse_fail) / len(rows) < 0.02,
          f"{len(parse_fail)}/{len(rows)}")

    print(f"\n=== smoke verification: {len(rows)} rollouts, "
          f"{len(by_sample)} samples ===")
    for ok_, name, detail in RESULTS:
        print(f"[{'PASS' if ok_ else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    failed = [r for r in RESULTS if not r[0]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("LADDER STOPS HERE — fix before scaling to 20.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
