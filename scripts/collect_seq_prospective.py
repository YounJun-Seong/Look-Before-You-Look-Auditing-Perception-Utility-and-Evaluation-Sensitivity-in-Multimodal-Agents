#!/usr/bin/env python
"""Sequential prospective collection (paper/PREREG_SEQ_PROSPECTIVE.md).

Per fresh state, both replays: one-step arms STOP / CROP-center / CROP-TR /
OCR / HIGHRES plus the four enumerated two-step trajectories SEQ#0-3 of
scripts/collect_sequential.py (whose SequentialExecutor and boxes are reused
verbatim). Serial, greedy seed 1234, logprobs on. Needs PPU_MM_LIMIT=3.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "cs", ROOT / "scripts/collect_sequential.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)

from ppu.datasets import load_manifest  # noqa: E402
from ppu.logger import RolloutLogger  # noqa: E402
from ppu.ocr import CachedOCR, PrecomputedOCR, RapidOCREngine  # noqa: E402
from ppu.runner import RolloutEngine  # noqa: E402
from ppu.schema import ActionSpec  # noqa: E402
from ppu.vlm import DEFAULT_DECODE, OpenAICompatVLM  # noqa: E402

ONE_STEP = [ActionSpec("STOP"),
            ActionSpec("CROP", slot=0, box=cs.CENTER, source="grid"),
            ActionSpec("CROP", slot=2, box=cs.TOPRIGHT, source="grid"),
            ActionSpec("OCR"), ActionSpec("HIGHRES")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/textvqa_seq_prospective.jsonl")
    ap.add_argument("--ids", default="data/textvqa_seq_prospective_ids.txt")
    ap.add_argument("--run-id", default="seq_prospective")
    ap.add_argument("--coarse", type=int, default=192)
    ap.add_argument("--highres", type=int, default=512)
    ap.add_argument("--ocr-cache", default="data/ocr_cache.json")
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    samples = {s.sample_id: s for s in load_manifest(args.manifest)}
    want = [l.strip() for l in open(args.ids) if l.strip()]
    chosen = [samples[i] for i in want if i in samples]
    if args.limit:
        chosen = chosen[: args.limit]

    vlm = OpenAICompatVLM(model_id=args.model, base_url=args.base_url,
                          decode=dict(DEFAULT_DECODE), logprobs=True)
    ocr = CachedOCR(PrecomputedOCR(args.ocr_cache, fallback=RapidOCREngine()))
    ex = cs.SequentialExecutor(coarse_tokens=args.coarse,
                               highres_tokens=args.highres, ocr_engine=ocr)
    engine = RolloutEngine(vlm=vlm, executor=ex, cache=None,
                           run_id=args.run_id, concurrency=1)

    log = f"logs/{args.run_id}.jsonl"
    t0 = time.time()
    with RolloutLogger(log, args.run_id,
                       {"purpose": "sequential prospective controller test",
                        "prereg": "paper/PREREG_SEQ_PROSPECTIVE.md",
                        "n_states": len(chosen),
                        "coarse": args.coarse, "highres": args.highres}) as lg:
        for i, s in enumerate(chosen, 1):
            for rep in (1, 2):
                for spec_ in ONE_STEP:
                    lg.log(engine.run_action(s, spec_, rep))
                for slot in cs.TRAJ:
                    lg.log(engine.run_action(s, ActionSpec("SEQ", slot=slot), rep))
            if i % 10 == 0:
                r = (time.time() - t0) / i
                print(f"  {i}/{len(chosen)}  eta {r * (len(chosen) - i) / 60:.0f} min",
                      flush=True)
                lg.note(level="progress", done=i, total=len(chosen))
    print(f"{len(chosen)} states x 9 rollouts x 2 replays -> {log} "
          f"({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
