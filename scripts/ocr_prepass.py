#!/usr/bin/env python
"""Read every image once, in parallel, before the campaign starts.

OCR on MME-RealWorld images runs 18-31s each single-threaded; doing it inside
the rollout loop leaves the GPU idle for hours. This pass fans the same engine
across CPU workers (the machine has 124 cores) and stores the result, so the
campaign only looks things up. The stored latency is the engine's real cost —
the cost column must keep reporting what the tool actually costs to run.

    python scripts/ocr_prepass.py --manifest data/manifests/mme_realworld_lite.jsonl \
        --out data/ocr_cache.json --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppu.datasets import draw, draw_stratified, load_manifest

_ENGINE = None


def _read_one(image_path: str):
    global _ENGINE
    if _ENGINE is None:
        from ppu.ocr import RapidOCREngine

        _ENGINE = RapidOCREngine()
        # On many-core hosts onnxruntime defaults to ncore intra-op threads
        # PER WORKER (124-core box: 6 workers -> ~744 threads -> ~100s/img
        # thrashing, 2026-08-31). Cap via env; default 4 is ~0.6s/img.
        n = int(os.environ.get("PPU_OCR_THREADS", "4"))
        if n > 0:
            from rapidocr_onnxruntime import RapidOCR

            _ENGINE._engine = RapidOCR(intra_op_num_threads=n,
                                       inter_op_num_threads=1)
    try:
        res = _ENGINE.read(image_path)
        return image_path, {"lines": [[l.text, list(l.box), l.conf] for l in res.lines],
                            "latency_s": res.latency_s}, ""
    except Exception as exc:  # noqa: BLE001 - one bad image must not kill the pass
        return image_path, None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="data/ocr_cache.json")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n", type=int, default=0, help="0 = every image in the manifest")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--stratify-by", default="")
    args = ap.parse_args()

    samples = load_manifest(args.manifest)
    if args.n:
        samples = (draw_stratified(samples, args.n, args.seed, args.stratify_by)
                   if args.stratify_by else draw(samples, args.n, args.seed))
    out_path = Path(args.out)
    table = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = sorted({s.image_path for s in samples} - set(table))
    print(f"{len(todo)} images to read ({len(table)} already cached), "
          f"{args.workers} workers", flush=True)
    if not todo:
        return 0

    t0 = time.time()
    errors = 0
    with Pool(args.workers) as pool:
        for i, (path, entry, err) in enumerate(pool.imap_unordered(_read_one, todo), 1):
            if entry is None:
                errors += 1
                print(f"  ERROR {path}: {err}", file=sys.stderr, flush=True)
            else:
                table[path] = entry
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)}  {rate:.1f}s/img  "
                      f"eta {rate * (len(todo) - i) / 60:.1f} min", flush=True)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(table))

    out_path.write_text(json.dumps(table))
    lat = [e["latency_s"] for e in table.values()]
    print(f"done: {len(table)} images, {errors} errors, "
          f"engine mean {sum(lat) / max(len(lat), 1):.1f}s/img, "
          f"wall {(time.time() - t0) / 60:.1f} min -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
