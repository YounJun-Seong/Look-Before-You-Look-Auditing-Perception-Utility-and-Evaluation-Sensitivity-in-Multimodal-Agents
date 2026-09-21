#!/usr/bin/env python
"""Two-step sequential perception collection (paper/PREREG_SEQUENTIAL.md).

Minimal sequential extension of the one-step audit: after a first observation
action a1, is a second acquisition a2 still worth its cost, and does its value
depend on what a1 revealed? Trajectories are enumerated brute-force (the
sequential DECISION analysis is post-hoc on this tree), so each trajectory is
one VLM call with the accumulated evidence:

  slot 0: CROP-center -> CROP-top-right   (views: coarse + 2 crops)
  slot 1: CROP-center -> HIGHRES          (views: highres + center crop)
  slot 2: OCR         -> CROP-top-right   (views: coarse + crop, + OCR text)
  slot 3: OCR         -> HIGHRES          (views: highres, + OCR text)

Boxes are the frozen grid constants (GRID_BOXES[0]=center, [2]=top-right).
One-step references (STOP, each single arm) already exist in the Gate 0 log
for these states and are NOT recollected. States: the 100-state outcome-blind
multi-replay subset (ids read from logs/mr_textvqa.jsonl). Replays 1-2,
frozen replay module, greedy seed 1234.

Requires the server with PPU_MM_LIMIT=3.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ppu.schema as _schema

# additive: register the composite trajectory action type for this experiment
# without editing the frozen schema module on disk
_schema.ALL_ACTION_TYPES = frozenset(_schema.ALL_ACTION_TYPES) | {"SEQ"}

from ppu.actions import ActionExecutor, Observation  # noqa: E402
from ppu.datasets import load_manifest
from ppu.logger import RolloutLogger
from ppu.ocr import CachedOCR, PrecomputedOCR, RapidOCREngine
from ppu.regions import GRID_BOXES
from ppu.runner import RolloutEngine
from ppu.schema import ActionSpec, Cost
from ppu.vlm import DEFAULT_DECODE, OpenAICompatVLM

CENTER, TOPRIGHT = GRID_BOXES[0], GRID_BOXES[2]
TRAJ = {
    0: (ActionSpec("CROP", slot=0, box=CENTER, source="grid"),
        ActionSpec("CROP", slot=2, box=TOPRIGHT, source="grid")),
    1: (ActionSpec("CROP", slot=0, box=CENTER, source="grid"),
        ActionSpec("HIGHRES")),
    2: (ActionSpec("OCR"), ActionSpec("CROP", slot=2, box=TOPRIGHT, source="grid")),
    3: (ActionSpec("OCR"), ActionSpec("HIGHRES")),
}


class SequentialExecutor(ActionExecutor):
    def observe(self, sample, spec):
        if spec.type != "SEQ":
            return super().observe(sample, spec)
        a1, a2 = TRAJ[spec.slot]
        o1 = super().observe(sample, a1)
        o2 = super().observe(sample, a2)
        base = self.coarse(sample)
        views = [base]
        aux = ""
        if a1.type == "CROP":
            views.append(o1.views[-1])
        elif a1.type == "OCR":
            aux = o1.aux_text
        if a2.type == "CROP":
            views.append(o2.views[-1])
        elif a2.type == "HIGHRES":
            views[0] = o2.views[0]  # highres replaces the coarse overview
        cost = Cost(
            visual_tokens=o1.cost.visual_tokens + o2.cost.visual_tokens,
            tool_calls=o1.cost.tool_calls + o2.cost.tool_calls,
            ocr_latency_s=o1.cost.ocr_latency_s + o2.cost.ocr_latency_s)
        return Observation(views, aux, cost,
                           f"seq{spec.slot}:{o1.summary}->{o2.summary}",
                           ocr=o1.ocr or o2.ocr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/textvqa_val.jsonl")
    ap.add_argument("--ids-log", default="logs/mr_textvqa.jsonl",
                    help="log whose sample_ids define the state subset")
    ap.add_argument("--run-id", default="seq_textvqa")
    ap.add_argument("--coarse", type=int, default=192)
    ap.add_argument("--highres", type=int, default=512)
    ap.add_argument("--ocr-cache", default="data/ocr_cache_textvqa.json")
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ids = sorted({json.loads(l)["sample_id"] for l in open(args.ids_log)
                  if '"rollout"' in l})
    samples = {s.sample_id: s for s in load_manifest(args.manifest)}
    chosen = [samples[i] for i in ids if i in samples]
    if args.limit:
        chosen = chosen[: args.limit]

    vlm = OpenAICompatVLM(model_id=args.model, base_url=args.base_url,
                          decode=dict(DEFAULT_DECODE), logprobs=True)
    ocr = CachedOCR(PrecomputedOCR(args.ocr_cache, fallback=RapidOCREngine()))
    ex = SequentialExecutor(coarse_tokens=args.coarse,
                            highres_tokens=args.highres, ocr_engine=ocr)
    engine = RolloutEngine(vlm=vlm, executor=ex, cache=None,
                           run_id=args.run_id, concurrency=1)

    log = f"logs/{args.run_id}.jsonl"
    t0 = time.time()
    with RolloutLogger(log, args.run_id,
                       {"purpose": "two-step sequential perception audit",
                        "prereg": "paper/PREREG_SEQUENTIAL.md",
                        "n_states": len(chosen),
                        "trajectories": {k: f"{v[0].type}#{v[0].slot}->{v[1].type}"
                                         for k, v in TRAJ.items()},
                        "coarse": args.coarse, "highres": args.highres}) as lg:
        for i, s in enumerate(chosen, 1):
            for rep in (1, 2):
                for slot in TRAJ:
                    lg.log(engine.run_action(s, ActionSpec("SEQ", slot=slot), rep))
            if i % 10 == 0:
                r = (time.time() - t0) / i
                print(f"  {i}/{len(chosen)}  eta {r * (len(chosen) - i) / 60:.0f} min",
                      flush=True)
                lg.note(level="progress", done=i, total=len(chosen))
    print(f"{len(chosen)} states x {len(TRAJ)} trajectories x 2 replays -> {log} "
          f"({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
