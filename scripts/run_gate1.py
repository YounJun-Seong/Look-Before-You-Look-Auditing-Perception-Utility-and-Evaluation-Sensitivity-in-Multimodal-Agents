#!/usr/bin/env python
"""Brute-force rollout campaign (Gate 0 / Gate 1).

Executes every candidate action on every drawn sample. No policy, no
predictor: this script exists to produce the log that the audit reads.

    # Gate 0 smoke, 20 samples, one replay, no dummy arm
    python scripts/run_gate1.py --manifest data/manifests/synthetic.jsonl \
        --n 20 --replays 1 --run-id g0_smoke

    # Gate 1 cell: 150 samples, both replays, dummy arm on the first half
    python scripts/run_gate1.py --manifest data/manifests/textvqa.jsonl \
        --n 150 --seed 20260816 --run-id g1_textvqa --dummy-fraction 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppu.actions import ActionExecutor
from ppu.cache import GenerationCache
from ppu.datasets import (EVAL_ONLY, burn, burned_ids, draw, draw_stratified,
                          load_manifest)
from ppu.logger import RolloutLogger
from ppu.ocr import CachedOCR, PrecomputedOCR, RapidOCREngine
from ppu.runner import RolloutEngine, run_campaign
from ppu.vlm import DEFAULT_DECODE, OpenAICompatVLM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--run-id", default=time.strftime("run_%Y%m%d_%H%M%S"))
    ap.add_argument("--replays", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--dummy-fraction", type=float, default=0.0,
                    help="fraction of drawn samples that also get the luck-floor arm")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the full-image reference row")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--coarse-tokens", type=int, default=256)
    ap.add_argument("--highres-tokens", type=int, default=1280)
    ap.add_argument("--full-tokens", type=int, default=2560,
                    help="FULL reference budget; <=0 = uncapped (native)")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--cache", default="data/generation_cache.sqlite")
    ap.add_argument("--burn-list", default="data/burned_ids.jsonl")
    ap.add_argument("--exclude-burned", action="store_true",
                    help="draw only samples never used in a previous gate")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="requests in flight per sample (<= server max-num-seqs)")
    ap.add_argument("--ocr-cache", default="",
                    help="OCR pre-pass file (scripts/ocr_prepass.py)")
    ap.add_argument("--stratify-by", default="",
                    help="meta key to stratify the draw on (e.g. category)")
    args = ap.parse_args()

    samples = load_manifest(args.manifest)
    exclude = burned_ids(args.burn_list) if args.exclude_burned else None
    if args.stratify_by:
        drawn = draw_stratified(samples, args.n, seed=args.seed,
                                key=args.stratify_by, exclude=exclude,
                                coarse_tokens=args.coarse_tokens)
    else:
        drawn = draw(samples, args.n, seed=args.seed, exclude=exclude,
                     coarse_tokens=args.coarse_tokens)
    if not drawn:
        print("no eligible samples (inclusion rule filtered everything)",
              file=sys.stderr)
        return 2

    # Burn eval-only ids BEFORE running: a campaign that crashes halfway still
    # consumed those samples (SPEC §5.1).
    if any(s.dataset in EVAL_ONLY for s in drawn):
        k = burn(args.burn_list, drawn, args.run_id)
        print(f"burned {k} eval-only sample ids -> {args.burn_list}")

    vlm = OpenAICompatVLM(model_id=args.model, base_url=args.base_url,
                          decode=dict(DEFAULT_DECODE))
    ocr = (CachedOCR(PrecomputedOCR(args.ocr_cache, fallback=RapidOCREngine()))
           if args.ocr_cache else CachedOCR(RapidOCREngine()))
    executor = ActionExecutor(coarse_tokens=args.coarse_tokens,
                              highres_tokens=args.highres_tokens,
                              full_tokens=args.full_tokens,
                              ocr_engine=ocr)
    engine = RolloutEngine(vlm=vlm, executor=executor,
                           cache=GenerationCache(args.cache), run_id=args.run_id,
                           concurrency=args.concurrency)

    log_path = Path(args.log_dir) / f"{args.run_id}.jsonl"
    meta = {"manifest": args.manifest, "n_drawn": len(drawn), "seed": args.seed,
            "model": args.model, "replays": args.replays,
            "coarse_tokens": args.coarse_tokens, "highres_tokens": args.highres_tokens,
            "full_tokens": args.full_tokens,
            "dummy_fraction": args.dummy_fraction,
            "stratify_by": args.stratify_by,
            "concurrency": args.concurrency,
            "image_format": vlm.image_format,
            "ocr_cache": args.ocr_cache,
            "datasets": sorted({s.dataset for s in drawn}),
            "strata": dict(Counter(str(s.meta.get(args.stratify_by, "_none"))
                                   for s in drawn)) if args.stratify_by else {}}
    print(meta)

    # First real inference time is logged so odd latencies can later be read
    # against server start-up rather than guessed at.
    first_iso = [""]
    _orig_generate = vlm.generate

    def _timed_generate(*a, **kw):
        if not first_iso[0]:
            first_iso[0] = time.strftime("%Y-%m-%dT%H:%M:%S")
        return _orig_generate(*a, **kw)

    vlm.generate = _timed_generate

    t0 = time.time()
    with RolloutLogger(log_path, args.run_id, meta) as lg:
        n = run_campaign(engine, drawn, lg, replays=tuple(args.replays),
                         reference=not args.no_reference)
        if args.dummy_fraction > 0:
            k = int(round(len(drawn) * args.dummy_fraction))
            n += run_campaign(engine, drawn[:k], lg, replays=tuple(args.replays),
                              dummy=True, reference=False)
        lg.note(level="done", rollouts=n,
                minutes=round((time.time() - t0) / 60, 1),
                ocr_engine_calls=ocr.misses, ocr_cache_hits=ocr.hits,
                gen_cache_hits=engine.cache.hits if engine.cache else 0,
                gen_cache_misses=engine.cache.misses if engine.cache else 0,
                first_inference_iso=first_iso[0])

    print(f"{n} rollouts -> {log_path}  ({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
