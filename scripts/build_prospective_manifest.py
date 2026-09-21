#!/usr/bin/env python
"""Draw the 600-state prospective TextVQA evaluation set (outcome-blind).

Pool = textvqa_val manifest minus every sample_id that appears in ANY
existing textvqa log or manifest (Gate 0 tv_200, fresh/lockbox sets, smokes,
multi-replay, Phi-4 runs). Draw = ppu.datasets.draw (fixed seed, same
coarse-192 eligibility rule as every TextVQA run). Writes
data/manifests/textvqa_prospective.jsonl + data/textvqa_prospective_ids.txt
and prints the manifest sha256 for the prereg.
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
SEED = 20260831
N = 600

used = set()
for p in glob.glob(str(ROOT / "logs/*textvqa*.jsonl")) + [str(ROOT / "logs/tv_200.jsonl")]:
    for line in open(p):
        try:
            sid = json.loads(line).get("sample_id")
        except json.JSONDecodeError:
            continue
        if sid:
            used.add(sid)
for line in open(ROOT / "data/manifests/textvqa_fresh.jsonl"):
    used.add(json.loads(line).get("sample_id"))

samples = load_manifest(str(ROOT / "data/manifests/textvqa_val.jsonl"))
chosen = draw(samples, N, seed=SEED, exclude=used, coarse_tokens=192)
assert len(chosen) == N, len(chosen)
assert not {s.sample_id for s in chosen} & used

by_id = {}
for line in open(ROOT / "data/manifests/textvqa_val.jsonl"):
    r = json.loads(line)
    by_id[r["sample_id"]] = line
out = ROOT / "data/manifests/textvqa_prospective.jsonl"
with open(out, "w") as fh:
    for s in chosen:
        fh.write(by_id[s.sample_id])
(ROOT / "data/textvqa_prospective_ids.txt").write_text(
    "\n".join(s.sample_id for s in chosen) + "\n")
sha = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
print(f"{N} states -> {out.name}  sha256[:16]={sha}  (excluded {len(used)} used ids)")
