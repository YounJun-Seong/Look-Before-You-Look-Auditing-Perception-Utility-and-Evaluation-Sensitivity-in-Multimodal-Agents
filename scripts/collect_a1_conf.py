#!/usr/bin/env python
"""Regenerate a1-arm generations (CROP-center, OCR) with logprobs for the
100-state sequential dev set — post-a1 confidence features for the
cost-aware sequential controller v2.

Same regeneration discipline as collect_answer_span_conf.py: identical
prompt path, greedy seed 1234, serial; byte-match against the logged Gate 0
generation is recorded per row. ~100 states x 2 arms x 2 replays = 400 calls.

    .venv/bin/python scripts/collect_a1_conf.py \
        --out data/span_conf/seqdev_a1_conf.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppu import prompts, replay as replay_mod
from ppu.actions import ActionExecutor, view_labels
from ppu.datasets import load_manifest
from ppu.ocr import CachedOCR, PrecomputedOCR, RapidOCREngine
from ppu.regions import GRID_BOXES
from ppu.schema import ActionSpec

CENTER = GRID_BOXES[0]
ARMS = {"center": ActionSpec("CROP", slot=0, box=CENTER, source="grid"),
        "ocr": ActionSpec("OCR")}
CENTER_KEY = "0.2500,0.2500,0.7500,0.7500"


def call(client, model, system, user_text, views, max_tokens=128):
    content = [{"type": "image_url", "image_url": {"url": v.to_data_url("JPEG")}}
               for v in views]
    content.append({"type": "text", "text": user_text})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": content}],
        temperature=0.0, top_p=1.0, seed=1234, max_tokens=max_tokens,
        logprobs=True, top_logprobs=1)
    ch = resp.choices[0]
    lps = [float(t.logprob) for t in (ch.logprobs.content or [])] \
        if ch.logprobs else []
    return ch.message.content or "", lps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-log", default="logs/mr_textvqa.jsonl")
    ap.add_argument("--g0-log", default="logs/tv_200.jsonl")
    ap.add_argument("--manifest", default="data/manifests/textvqa_val.jsonl")
    ap.add_argument("--coarse", type=int, default=192)
    ap.add_argument("--ocr-cache", default="data/ocr_cache_textvqa.json")
    ap.add_argument("--out", default="data/span_conf/seqdev_a1_conf.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=180,
                    max_retries=2)

    ids = sorted({json.loads(l)["sample_id"] for l in open(args.ids_log)
                  if '"rollout"' in l})
    logged = defaultdict(dict)  # sid -> (arm, rep) -> raw_generation
    for line in open(args.g0_log):
        r = json.loads(line)
        if r.get("type") != "rollout" or r["sample_id"] not in set(ids):
            continue
        a = r["action"]
        if a["type"] == "OCR":
            arm = "ocr"
        elif a["type"] == "CROP" and a.get("source") == "grid" and \
                ",".join(f"{v:.4f}" for v in a["box"]) == CENTER_KEY:
            arm = "center"
        else:
            continue
        logged[r["sample_id"]][(arm, r["replay"])] = r.get("raw_generation") or ""

    samples = {s.sample_id: s for s in load_manifest(args.manifest)}
    ocr = CachedOCR(PrecomputedOCR(args.ocr_cache, fallback=RapidOCREngine()))
    ex = ActionExecutor(coarse_tokens=args.coarse, ocr_engine=ocr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0, mismatch, total = time.time(), 0, 0
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "run_meta", "ids_log": args.ids_log,
                             "g0_log": args.g0_log, "coarse": args.coarse,
                             "n_states": len(ids)}) + "\n")
        for i, sid in enumerate(ids, 1):
            sample = samples[sid]
            row = {"type": "a1_conf", "sample_id": sid}
            for arm, spec_ in ARMS.items():
                obs = ex.observe(sample, spec_)
                for rep in (1, 2):
                    served = replay_mod.serve(sample, rep)
                    scoring = replay_mod.scoring_sample(sample, served)
                    q = prompts.render_question(scoring, served.choices or None,
                                                text=served.text)
                    text = prompts.build_user_text(scoring, q,
                                                   view_labels(obs, spec_),
                                                   obs.aux_text)
                    gen, lps = call(client, args.model, prompts.SYSTEM, text,
                                    obs.views)
                    old = logged[sid].get((arm, rep), "")
                    same = gen.strip() == old.strip()
                    total += 1
                    mismatch += int(not same)
                    row[f"{arm}_r{rep}"] = {
                        "mean_logprob": float(sum(lps) / len(lps)) if lps else None,
                        "min_logprob": float(min(lps)) if lps else None,
                        "n_tokens": len(lps), "matches_logged": same}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                r_ = (time.time() - t0) / i
                print(f"  {i}/{len(ids)} eta {r_ * (len(ids) - i) / 60:.0f} min",
                      flush=True)
    print(f"{len(ids)} states -> {args.out} ({(time.time() - t0) / 60:.1f} min); "
          f"generation mismatches vs log: {mismatch}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
