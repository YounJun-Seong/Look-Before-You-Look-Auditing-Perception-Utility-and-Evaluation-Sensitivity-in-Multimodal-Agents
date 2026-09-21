"""Dataset adapters: everything becomes a `Sample`.

Runtime never touches raw dataset formats. `scripts/prepare_data.py` converts
each source into a manifest (JSONL of Sample fields) once; the runner only
reads manifests. That keeps the download/extraction mess out of the experiment
loop and makes the exact sample set a committed artifact rather than the
output of whatever the loader did that day.

Selection rules that are part of the pre-registration (SPEC §3.1, §5.2):
  - inclusion: native_tokens >= 4 * coarse budget, else CROP/HIGHRES cannot
    add information and the sample would dilute the action-diversity result
  - sampling: fixed seed, no cherry-picking, per-dataset quotas
  - eval-only datasets record every drawn id in the burn list
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from .schema import Sample

METRIC_BY_DATASET = {
    "textvqa": "vqa_acc",
    "docvqa": "anls",
    "infographicvqa": "anls",
    "gqa": "exact",
    "mme_realworld": "mcq",
    "synthetic": "exact",
}

# Datasets held out for the final blind test; any sample drawn from these in an
# exploratory gate must be burned (SPEC §5.1).
EVAL_ONLY = ("mme_realworld", "3dsrbench", "visualneedle", "vtc_bench")


def load_manifest(path: str | Path) -> list[Sample]:
    samples: list[Sample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(Sample(
                sample_id=d["sample_id"],
                dataset=d["dataset"],
                image_path=d["image_path"],
                question=d["question"],
                answers=tuple(d["answers"]),
                metric=d.get("metric") or METRIC_BY_DATASET[d["dataset"]],
                choices=tuple(d.get("choices", ())),
                native_hw=tuple(d.get("native_hw", (0, 0))),
                native_tokens=int(d.get("native_tokens", 0)),
                meta=d.get("meta", {}),
            ))
    return samples


def write_manifest(path: str | Path, samples: Iterable[Sample]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
            n += 1
    return n


def passes_inclusion(sample: Sample, coarse_tokens: int = 256, ratio: float = 4.0) -> bool:
    """SPEC §3.1 inclusion rule. Reported as a rate in Gate 0 (G0-7)."""
    return sample.native_tokens >= ratio * coarse_tokens


def draw(samples: list[Sample], n: int, seed: int, exclude: Optional[set[str]] = None,
         coarse_tokens: int = 256) -> list[Sample]:
    """Fixed-seed draw from the eligible pool, sorted for determinism."""
    exclude = exclude or set()
    pool = [s for s in samples
            if s.sample_id not in exclude and passes_inclusion(s, coarse_tokens)]
    pool.sort(key=lambda s: s.sample_id)
    if len(pool) <= n:
        return pool
    rng = random.Random(seed)
    return sorted(rng.sample(pool, n), key=lambda s: s.sample_id)


def draw_stratified(samples: list[Sample], n: int, seed: int, key: str = "category",
                    exclude: Optional[set[str]] = None,
                    coarse_tokens: int = 256) -> list[Sample]:
    """Equal-ish quota per stratum, then a fixed-seed draw inside each.

    Gate 0 does not ask "what is the average accuracy on this benchmark"; it
    asks whether different perception regimes need different actions. A plain
    random draw lets one domain (remote sensing, OCR-CC) dominate by luck and
    the action-diversity number becomes a statement about that domain.

    Quota rule, fully determined by (n, strata sizes): base = n // S for every
    stratum, remainder handed out one at a time to the largest strata (ties by
    name). Strata that cannot fill their quota give the shortfall back, which
    is redistributed the same way.
    """
    exclude = exclude or set()
    pool = [s for s in samples
            if s.sample_id not in exclude and passes_inclusion(s, coarse_tokens)]
    if not pool:
        return []
    groups: dict[str, list[Sample]] = {}
    for s in pool:
        groups.setdefault(str(s.meta.get(key, "_none")), []).append(s)
    for g in groups.values():
        g.sort(key=lambda s: s.sample_id)

    names = sorted(groups, key=lambda k: (-len(groups[k]), k))
    quota = {k: min(n // len(names), len(groups[k])) for k in names}
    # Hand out the remainder, largest strata first, repeating until nothing
    # more can be placed (a stratum never exceeds its available samples).
    while sum(quota.values()) < n:
        placed = False
        for k in names:
            if sum(quota.values()) >= n:
                break
            if quota[k] < len(groups[k]):
                quota[k] += 1
                placed = True
        if not placed:
            break

    rng = random.Random(seed)
    out: list[Sample] = []
    for k in sorted(names):
        g = groups[k]
        take = quota[k]
        out.extend(g if take >= len(g) else rng.sample(g, take))
    return sorted(out, key=lambda s: s.sample_id)


def burn(path: str | Path, samples: Iterable[Sample], run_id: str) -> int:
    """Append eval-only ids to the burn list. Called before the rollouts run,
    not after: a crashed campaign still consumed the samples."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for s in samples:
            if s.dataset in EVAL_ONLY:
                fh.write(json.dumps(
                    {"sample_id": s.sample_id, "dataset": s.dataset, "run_id": run_id},
                    ensure_ascii=False) + "\n")
                n += 1
    return n


def burned_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.add(json.loads(line)["sample_id"])
    return out
