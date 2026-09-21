#!/usr/bin/env python
"""Minimal Stage A state collector: STOP and HIGHRES only, both replays.

Gate 0 ran 24 rollouts per state because it had to compare eleven actions.
Stage A needs one comparison, so a state costs 4 inferences and no OCR:

    STOP R1 | STOP R2 | HIGHRES R1 | HIGHRES R2

Always serial (batching moved strict labels on 3/20 states) and always with
logprobs on, so the teacher label and the confidence feature come out of the
same pass. The generation cache is deliberately NOT used — this script exists
to produce fresh outcomes, and a cache hit would silently drop the logprobs.

    python scripts/collect_states.py --manifest ... --ids logs/mme_ids.txt \
        --coarse 256 --highres 1280 --run-id mme_serial
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppu.actions import ActionExecutor
from ppu.datasets import draw, draw_stratified, load_manifest
from ppu.logger import RolloutLogger
from ppu.runner import RolloutEngine
from ppu.schema import ActionSpec
from ppu.vlm import DEFAULT_DECODE, OpenAICompatVLM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ids", default="", help="file with one sample_id per line")
    ap.add_argument("--n", type=int, default=0, help="draw this many instead")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--stratify-by", default="")
    ap.add_argument("--exclude-ids", default="", help="ids to keep out of the draw")
    ap.add_argument("--coarse", type=int, default=256)
    ap.add_argument("--highres", type=int, default=1280)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--actions", default="STOP,HIGHRES",
                    help="comma-separated action types to execute per replay")
    ap.add_argument("--full-tokens", type=int, default=0,
                    help="FULL budget; <=0 = uncapped (native)")
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

    samples = {s.sample_id: s for s in load_manifest(args.manifest)}
    if args.ids:
        want = [l.strip() for l in open(args.ids) if l.strip()]
        chosen = [samples[i] for i in want if i in samples]
    else:
        excl = set()
        if args.exclude_ids:
            excl = {l.strip() for l in open(args.exclude_ids) if l.strip()}
        pool = list(samples.values())
        chosen = (draw_stratified(pool, args.n, args.seed, args.stratify_by,
                                  exclude=excl, coarse_tokens=args.coarse)
                  if args.stratify_by else
                  draw(pool, args.n, args.seed, exclude=excl,
                       coarse_tokens=args.coarse))
    if not chosen:
        print("no states selected", file=sys.stderr)
        return 2

    vlm = OpenAICompatVLM(model_id=args.model, base_url=args.base_url,
                          decode=dict(DEFAULT_DECODE), logprobs=True)
    ex = ActionExecutor(coarse_tokens=args.coarse, highres_tokens=args.highres,
                        full_tokens=args.full_tokens)
    engine = RolloutEngine(vlm=vlm, executor=ex, cache=None,
                           run_id=args.run_id, concurrency=1)

    log = f"logs/{args.run_id}.jsonl"
    t0 = time.time()
    with RolloutLogger(log, args.run_id,
                       {"purpose": "Stage A state collection (STOP vs HIGHRES)",
                        "n_states": len(chosen), "coarse": args.coarse,
                        "highres": args.highres, "concurrency": 1,
                        "logprobs": True, "cache": None}) as lg:
        acts = [a.strip() for a in args.actions.split(",") if a.strip()]
        for i, s in enumerate(chosen, 1):
            for rep in (1, 2):
                before = None
                for a in acts:
                    rec = engine.run_action(s, ActionSpec(a), rep, before=before)
                    lg.log(rec)
                    if a == "STOP":
                        before = (rec.answer_after, rec.score_after)
            if i % 20 == 0:
                r = (time.time() - t0) / i
                print(f"  {i}/{len(chosen)}  eta {r*(len(chosen)-i)/60:.0f} min", flush=True)
                lg.note(level="progress", done=i, total=len(chosen))
    print(f"{len(chosen)} states, {2*len(acts)*len(chosen)} inferences -> {log} "
          f"({(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
