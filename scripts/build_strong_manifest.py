#!/usr/bin/env python
"""Draw the fresh TextVQA evaluation set for the stronger-controller test
(outcome-blind; paper/PREREG_STRONG_CONTROLLER.md).

Pool = textvqa_val manifest minus every sample_id that appears in ANY
existing log (logs/*.jsonl, all runs, all readers, smokes) or TextVQA
manifest. Draw = ppu.datasets.draw (fixed seed, coarse-192 eligibility).
Writes data/manifests/textvqa_strong.jsonl + data/textvqa_strong_ids.txt.
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppu.datasets import draw, load_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260920
N = 400

used = set()
for p in glob.glob(str(ROOT / "logs/*.jsonl")):
    for line in open(p):
        try:
            sid = json.loads(line).get("sample_id")
        except json.JSONDecodeError:
            continue
        if sid:
            used.add(sid)
for p in glob.glob(str(ROOT / "data/manifests/textvqa_*.jsonl")):
    if p.endswith(("textvqa_val.jsonl", "textvqa_train.jsonl")):
        continue
    for line in open(p):
        used.add(json.loads(line).get("sample_id"))
for line in open(ROOT / "data/burned_ids.jsonl"):
    try:
        used.add(json.loads(line).get("sample_id"))
    except json.JSONDecodeError:
        pass

samples = load_manifest(str(ROOT / "data/manifests/textvqa_val.jsonl"))
val_ids = {s.sample_id for s in samples}
chosen = draw(samples, N, seed=SEED, exclude=used, coarse_tokens=192)
assert len(chosen) == N, len(chosen)
assert not {s.sample_id for s in chosen} & used

by_id = {}
for line in open(ROOT / "data/manifests/textvqa_val.jsonl"):
    r = json.loads(line)
    by_id[r["sample_id"]] = line
out = ROOT / "data/manifests/textvqa_strong.jsonl"
with open(out, "w") as fh:
    for s in chosen:
        fh.write(by_id[s.sample_id])
(ROOT / "data/textvqa_strong_ids.txt").write_text(
    "\n".join(s.sample_id for s in chosen) + "\n")
sha = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
print(f"{N} states -> {out.name}  sha256[:16]={sha}  "
      f"(excluded {len(used & val_ids)} used val ids of {len(val_ids)})")
