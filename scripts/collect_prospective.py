#!/usr/bin/env python
"""Prospective TextVQA collection: all deterministic arms on fresh states.

Data half of paper/PREREG_PROSPECTIVE_SELECTOR.md. For each of the 600 fresh
states, both replays: STOP (with logprobs, feeding the frozen selector's
confidence feature), the deterministic grid∪fallback CROP candidates (K ≤ 8),
OCR (from the pre-pass cache), and HIGHRES (512-tok TextVQA convention).
Serial, greedy, seed 1234 — identical decode path to every prior collection.

    .venv/bin/python scripts/collect_prospective.py \
        --manifest data/manifests/textvqa_prospective.jsonl \
        --ids data/textvqa_prospective_ids.txt --run-id prospective_textvqa
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
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--coarse", type=int, default=192)
    ap.add_argument("--highres", type=int, default=512)
    ap.add_argument("--ocr-cache", default="data/ocr_cache.json")
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N states")
    ap.add_argument("--replays", default="1,2",
                    help="comma list; replays 3-5 use ppu.replay_multi (frozen 1-2 untouched)")
    ap.add_argument("--purpose", default="prospective selector-vs-mixture evaluation")
    ap.add_argument("--prereg", default="paper/PREREG_PROSPECTIVE_SELECTOR.md")
    args = ap.parse_args()
    replays = tuple(int(x) for x in args.replays.split(","))
    if any(r > 2 for r in replays):
        import ppu.runner as _runner
        from ppu import replay_multi
        _runner.replay_mod = replay_multi  # serve()/scoring_sample() delegate 1-2 verbatim

    samples = {s.sample_id: s for s in load_manifest(args.manifest)}
    want = [l.strip() for l in open(args.ids) if l.strip()]
    chosen = [samples[i] for i in want if i in samples]
    if args.limit:
        chosen = chosen[: args.limit]

    vlm = OpenAICompatVLM(model_id=args.model, base_url=args.base_url,
                          decode=dict(DEFAULT_DECODE), logprobs=True)
    ocr = CachedOCR(PrecomputedOCR(args.ocr_cache, fallback=RapidOCREngine()))
    ex = ActionExecutor(coarse_tokens=args.coarse, highres_tokens=args.highres,
                        ocr_engine=ocr)
    engine = RolloutEngine(vlm=vlm, executor=ex, cache=None,
                           run_id=args.run_id, concurrency=1)

    log = f"logs/{args.run_id}.jsonl"
    t0 = time.time()
    with RolloutLogger(log, args.run_id,
                       {"purpose": args.purpose, "prereg": args.prereg,
                        "replays": list(replays),
                        "n_states": len(chosen), "coarse": args.coarse,
                        "highres": args.highres, "crop_budget": 256,
                        "logprobs": True, "ocr_cache": args.ocr_cache}) as lg:
        n_gen = 0
        for i, s in enumerate(chosen, 1):
            crops = build_candidates([])
            for rep in replays:
                stop_rec = engine.run_action(s, ActionSpec("STOP"), rep)
                lg.log(stop_rec)
                before = (stop_rec.answer_after, stop_rec.score_after)
                for spec in crops + [ActionSpec("OCR"), ActionSpec("HIGHRES")]:
                    lg.log(engine.run_action(s, spec, rep, before=before))
                n_gen += 2 + len(crops) + 1
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
