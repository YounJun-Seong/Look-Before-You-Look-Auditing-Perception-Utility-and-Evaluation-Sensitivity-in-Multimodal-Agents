#!/usr/bin/env python
"""Held-out CROP/OCR arm collection for the frozen multi-action evaluation.

Implements the data half of paper/PREREG_MULTIACTION_HELDOUT.md: for every
fresh state, both replays, collect STOP (re-issued as a determinism check
against the 2026-08-23 fresh logs), the deterministic CROP candidate set
(grid ∪ fallback, K ≤ 8, budget 256), and OCR (served from the pre-pass
cache; recorded latency is the engine's real cost). Serial, greedy, seed 1234.

    .venv/bin/python scripts/collect_heldout_arms.py \
        --manifest data/manifests/vqav2_fresh.jsonl --ids data/vqav2_fresh_ids.txt \
        --coarse 64 --run-id heldout_vqav2_arms
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppu.actions import ActionExecutor
from ppu.datasets import load_manifest
from ppu.logger import RolloutLogger
from ppu.ocr import CachedOCR, PrecomputedOCR, RapidOCREngine
from ppu.regions import build_candidates
from ppu.runner import RolloutEngine
from ppu.schema import ActionSpec
from ppu.vlm import DEFAULT_DECODE, OpenAICompatVLM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--coarse", type=int, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--ocr-cache", default="data/ocr_cache.json")
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N states")
    args = ap.parse_args()

    samples = {s.sample_id: s for s in load_manifest(args.manifest)}
    want = [l.strip() for l in open(args.ids) if l.strip()]
    chosen = [samples[i] for i in want if i in samples]
    if args.limit:
        chosen = chosen[: args.limit]

    vlm = OpenAICompatVLM(model_id=args.model, base_url=args.base_url,
                          decode=dict(DEFAULT_DECODE), logprobs=True)
    ocr = CachedOCR(PrecomputedOCR(args.ocr_cache, fallback=RapidOCREngine()))
    ex = ActionExecutor(coarse_tokens=args.coarse, ocr_engine=ocr)
    engine = RolloutEngine(vlm=vlm, executor=ex, cache=None,
                           run_id=args.run_id, concurrency=1)

    log = f"logs/{args.run_id}.jsonl"
    t0 = time.time()
    with RolloutLogger(log, args.run_id,
                       {"purpose": "held-out CROP/OCR arms for frozen multi-action eval",
                        "prereg": "paper/PREREG_MULTIACTION_HELDOUT.md",
                        "n_states": len(chosen), "coarse": args.coarse,
                        "crop_budget": 256, "concurrency": 1, "logprobs": True,
                        "ocr_cache": args.ocr_cache}) as lg:
        n_gen = 0
        for i, s in enumerate(chosen, 1):
            crops = build_candidates([])  # deterministic grid ∪ fallback
            for rep in (1, 2):
                stop_rec = engine.run_action(s, ActionSpec("STOP"), rep)
                lg.log(stop_rec)
                before = (stop_rec.answer_after, stop_rec.score_after)
                for spec in crops + [ActionSpec("OCR")]:
                    lg.log(engine.run_action(s, spec, rep, before=before))
                n_gen += 2 + len(crops)
            if i % 20 == 0:
                r = (time.time() - t0) / i
                print(f"  {i}/{len(chosen)}  eta {r * (len(chosen) - i) / 60:.0f} min",
                      flush=True)
                lg.note(level="progress", done=i, total=len(chosen))
    print(f"{len(chosen)} states, {n_gen} inferences -> {log} "
          f"({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
