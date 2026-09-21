#!/usr/bin/env python
"""Multi-replay robustness collection (paper/PREREG_MULTI_REPLAY.md).

Re-runs the deterministic arm set (STOP, grid CROPs, OCR, HIGHRES) of an
existing Gate 0 brute-force log under NEW replays 3-5 (ppu/replay_multi.py),
for an outcome-blind 100-state subset. Replays 1-2 are NOT recollected — the
Gate 0 log already holds them; analysis joins on (sample_id, action key).

    .venv/bin/python scripts/collect_multi_replay.py \
        --g0-log logs/g0_mme200.jsonl --manifest data/manifests/mme_realworld_lite.jsonl \
        --coarse 256 --highres 1280 --full 2560 --ocr-cache data/ocr_cache.json \
        --run-id mr_mme
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ppu.runner as runner_mod
from ppu import replay_multi
from ppu.actions import ActionExecutor
from ppu.datasets import load_manifest
from ppu.logger import RolloutLogger
from ppu.ocr import CachedOCR, PrecomputedOCR, RapidOCREngine
from ppu.runner import RolloutEngine
from ppu.schema import ActionSpec
from ppu.vlm import DEFAULT_DECODE, OpenAICompatVLM

# replays 3-5 live in replay_multi; 1-2 delegate to the frozen module
runner_mod.replay_mod = replay_multi

SUBSET_SEED = 20260830


def arm_specs(g0_log):
    """Deterministic arm list per state, reconstructed from the Gate 0 log."""
    arms = defaultdict(dict)
    for line in open(g0_log):
        r = json.loads(line)
        if r.get("type") != "rollout" or r["replay"] != 1:
            continue
        a = r["action"]
        if a["type"] == "CROP" and a.get("source") == "grid":
            spec = ActionSpec("CROP", slot=a["slot"], box=tuple(a["box"]),
                              source="grid")
        elif a["type"] in ("OCR", "HIGHRES"):
            spec = ActionSpec(a["type"])
        else:
            continue
        arms[r["sample_id"]][a["type"] + "#" + str(a.get("slot", 0)) +
                             "#" + str(a.get("box"))] = spec
    return {sid: list(d.values()) for sid, d in arms.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-log", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--coarse", type=int, required=True)
    ap.add_argument("--highres", type=int, required=True)
    ap.add_argument("--full", type=int, required=True)
    ap.add_argument("--ocr-cache", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--replays", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N states")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    args = ap.parse_args()

    arms = arm_specs(args.g0_log)
    # outcome-blind subset: sorted ids, fixed shuffle, first n
    ids = sorted(arms)
    random.Random(SUBSET_SEED).shuffle(ids)
    ids = ids[: args.n]
    if args.limit:
        ids = ids[: args.limit]
    samples = {s.sample_id: s for s in load_manifest(args.manifest)}
    chosen = [samples[i] for i in ids if i in samples]

    vlm = OpenAICompatVLM(model_id=args.model, base_url=args.base_url,
                          decode=dict(DEFAULT_DECODE))
    ocr = CachedOCR(PrecomputedOCR(args.ocr_cache, fallback=RapidOCREngine()))
    ex = ActionExecutor(coarse_tokens=args.coarse, highres_tokens=args.highres,
                        full_tokens=args.full, ocr_engine=ocr)
    engine = RolloutEngine(vlm=vlm, executor=ex, cache=None,
                           run_id=args.run_id, concurrency=1)

    log = f"logs/{args.run_id}.jsonl"
    t0 = time.time()
    with RolloutLogger(log, args.run_id,
                       {"purpose": "multi-replay robustness (replays 3-5)",
                        "prereg": "paper/PREREG_MULTI_REPLAY.md",
                        "g0_log": args.g0_log, "subset_seed": SUBSET_SEED,
                        "n_states": len(chosen), "replays": args.replays,
                        "coarse": args.coarse, "highres": args.highres}) as lg:
        for i, s in enumerate(chosen, 1):
            for rep in args.replays:
                stop_rec = engine.run_action(s, ActionSpec("STOP"), rep)
                lg.log(stop_rec)
                before = (stop_rec.answer_after, stop_rec.score_after)
                for spec in arms[s.sample_id]:
                    lg.log(engine.run_action(s, spec, rep, before=before))
            if i % 10 == 0:
                r = (time.time() - t0) / i
                print(f"  {i}/{len(chosen)}  eta {r * (len(chosen) - i) / 60:.0f} min",
                      flush=True)
                lg.note(level="progress", done=i, total=len(chosen))
    print(f"{len(chosen)} states x replays {args.replays} -> {log} "
          f"({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
