#!/usr/bin/env python
"""Execute the question-conditioned crops that the original parser threw away.

The Gate 0 campaigns charged for a proposal call on every sample but kept the
boxes only 30% of the time, so "question-conditioned selection" was never
really in the pool. The proposal *generations* are still in the campaign cache,
so no proposal call is repeated here: they are re-parsed with the mixed
coordinate reader and only the resulting crops are executed.

Per sample, with k = number of valid proposals (1..3), no grid backfill:

    VLM-k             the proposals themselves
    MatchedRandom-k   same area and aspect ratio, position randomized

Grid-k, FULL, STOP and the plain random crops already exist in the Gate 0 log,
so the comparison joins against it. Samples with k = 0 are excluded from the
selectivity analysis and reported as a proposal-failure rate.

    python scripts/run_selectivity.py --dataset mme --n 0
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppu import regions
from ppu.actions import ActionExecutor
from ppu.cache import GenerationCache
from ppu.datasets import load_manifest
from ppu.logger import RolloutLogger
from ppu.ocr import CachedOCR, PrecomputedOCR, RapidOCREngine
from ppu.runner import RolloutEngine, run_campaign
from ppu.schema import ActionSpec
from ppu.vlm import DEFAULT_DECODE, OpenAICompatVLM

CONFIG = {
    "mme": dict(log="logs/g0_mme200.jsonl",
                manifest="data/manifests/mme_realworld_lite.jsonl",
                cache="data/cache_g0.sqlite", ocr="data/ocr_cache.json",
                coarse=256, highres=1280, full=2560, run_id="sel_mme"),
    "textvqa": dict(log="logs/tv_200.jsonl",
                    manifest="data/manifests/textvqa_val.jsonl",
                    cache="data/cache_tv200.sqlite", ocr="data/ocr_cache_textvqa.json",
                    coarse=192, highres=512, full=0, run_id="sel_textvqa"),
}


def cached_proposals(cache_path: str) -> dict[str, tuple[str, int, int]]:
    """sample_id -> (generation text, coarse view width, height) from the cache."""
    out = {}
    con = sqlite3.connect(cache_path)
    for payload, text in con.execute("SELECT payload, text FROM generations"):
        p = json.loads(payload)
        if p.get("purpose") != "proposal":
            continue
        view = (p.get("views") or [""])[0]  # "coarse@256:WxH:tokens"
        try:
            wh = view.split(":")[1]
            w, h = (int(x) for x in wh.split("x"))
        except Exception:  # noqa: BLE001
            w = h = 0
        out[p["sample_id"]] = (text, w, h)
    con.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(CONFIG), required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = every sample in the log")
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()
    cfg = CONFIG[args.dataset]

    samples = {s.sample_id: s for s in load_manifest(cfg["manifest"])}
    drawn = []
    with open(cfg["log"], encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("type") == "rollout" and o["sample_id"] not in {s.sample_id for s in drawn}:
                s = samples.get(o["sample_id"])
                if s:
                    drawn.append(s)
    props = cached_proposals(cfg["cache"])

    plan, stats = [], {"k0": 0, "k1": 0, "k2": 0, "k3": 0, "no_generation": 0}
    trace_out = []
    for s in drawn:
        gen = props.get(s.sample_id)
        if gen is None:
            stats["no_generation"] += 1
            continue
        text, w, h = gen
        boxes, trace = regions.parse_proposals_mixed(text, w or 1, h or 1, limit=3)
        trace_out.append({"sample_id": s.sample_id, "view": [w, h],
                          "raw_generation": text, "trace": trace,
                          "boxes": boxes})
        stats[f"k{len(boxes)}"] += 1
        if not boxes:
            continue
        matched = regions.size_matched_random(s.sample_id, boxes)
        specs = [ActionSpec("CROP", slot=10 + i, box=b, source="vlm_proposal")
                 for i, b in enumerate(boxes)]
        specs += [ActionSpec("CROP_RANDOM", slot=20 + i, box=b, source="random")
                  for i, b in enumerate(matched)]
        plan.append((s, specs))
        if args.n and len(plan) >= args.n:
            break

    Path("logs").mkdir(exist_ok=True)
    Path(f"logs/{cfg['run_id']}_proposals.json").write_text(
        json.dumps({"stats": stats, "samples": trace_out}, indent=1))
    total = sum(stats[f"k{i}"] for i in range(4))
    print(f"{args.dataset}: {total} samples with a cached proposal call; "
          f"valid-box counts {stats}")
    print(f"  proposal failure rate (k=0): {100 * stats['k0'] / max(total, 1):.1f}%")
    print(f"  executing {sum(len(sp) for _, sp in plan)} crops x 2 replays "
          f"over {len(plan)} samples")
    if not plan:
        return 0

    vlm = OpenAICompatVLM(model_id=args.model, base_url=args.base_url,
                          decode=dict(DEFAULT_DECODE))
    ocr = CachedOCR(PrecomputedOCR(cfg["ocr"], fallback=RapidOCREngine()))
    ex = ActionExecutor(coarse_tokens=cfg["coarse"], highres_tokens=cfg["highres"],
                        full_tokens=cfg["full"], ocr_engine=ocr)
    engine = RolloutEngine(vlm=vlm, executor=ex,
                           cache=GenerationCache(f"data/cache_{cfg['run_id']}.sqlite"),
                           run_id=cfg["run_id"], concurrency=1)

    log_path = f"logs/{cfg['run_id']}.jsonl"
    t0 = time.time()
    with RolloutLogger(log_path, cfg["run_id"],
                       {"purpose": "selectivity: recovered VLM proposals vs "
                                   "size-matched random", "budgets": cfg,
                        "proposal_stats": stats}) as lg:
        for i, (s, specs) in enumerate(plan, 1):
            for r in (1, 2):
                for spec in specs:
                    lg.log(engine.run_action(s, spec, r))
            if i % 20 == 0:
                rate = (time.time() - t0) / i
                lg.note(level="progress", done=i, total=len(plan),
                        eta_min=round(rate * (len(plan) - i) / 60, 1))
                print(f"  {i}/{len(plan)}  eta {rate * (len(plan) - i) / 60:.0f} min",
                      flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
