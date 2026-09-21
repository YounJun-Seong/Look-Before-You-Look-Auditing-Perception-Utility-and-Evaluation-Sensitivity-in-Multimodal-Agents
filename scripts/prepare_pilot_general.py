#!/usr/bin/env python
"""Pilot manifest for the general-vision sources (VQAv2 / GQA).

Only the images actually sampled are downloaded — COCO and Visual Genome serve
them individually, so a 30-state pilot costs ~20MB instead of a 33GB archive.

Stratification is decided from the QUESTION only, never from any outcome:
    A global/easy      yes-no, scene-level attributes, large objects
    B local/detail     counting, colour/attribute of a specific object, text
    C multi/relational relations, comparisons, compositional chains
"""
from __future__ import annotations
import argparse, json, random, re, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from ppu.datasets import write_manifest
from ppu.imaging import native_tokens
from ppu.schema import Sample

COCO_URL = "http://images.cocodataset.org/train2014/COCO_train2014_{:012d}.jpg"
VG_URLS = ["https://cs.stanford.edu/people/rak248/VG_100K/{}.jpg",
           "https://cs.stanford.edu/people/rak248/VG_100K_2/{}.jpg"]

_LOCAL = re.compile(r"\b(how many|what color|colour|number|small|tiny|letter|word|"
                    r"digit|sign says|written|brand|logo|time|clock)\b", re.I)
_REL = re.compile(r"\b(left of|right of|behind|in front of|next to|near|between|"
                  r"above|below|on top of|compare|same|different|both|either)\b", re.I)
_GLOBAL = re.compile(r"^(is|are|does|do|was|were|can|has|have)\b|"
                     r"\b(weather|indoor|outdoor|scene|room|place|season|daytime)\b", re.I)


def stratum(q: str) -> str:
    if _LOCAL.search(q):
        return "B_local_detail"
    if _REL.search(q):
        return "C_multi_relational"
    if _GLOBAL.search(q):
        return "A_global_easy"
    return "C_multi_relational"


def fetch(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        Image.open(dest).verify()
        return True
    except Exception:
        dest.unlink(missing_ok=True)
        return False


def finish(sid, ds, path, q, answers, meta):
    with Image.open(path) as im:
        w, h = im.size
    return Sample(sample_id=sid, dataset=ds, image_path=str(path), question=q,
                  answers=tuple(answers), metric="vqa_acc" if ds == "vqav2" else "exact",
                  native_hw=(h, w), native_tokens=native_tokens(str(path)), meta=meta)


def pick(cands, n, seed):
    """Even split across the three strata, deterministic."""
    rng = random.Random(seed)
    by = {}
    for c in cands:
        by.setdefault(c["stratum"], []).append(c)
    for v in by.values():
        v.sort(key=lambda c: c["sid"])
    names = sorted(by)
    per = max(1, n // max(len(names), 1))
    picked = {s: rng.sample(by[s], min(per, len(by[s]))) for s in names}
    # Round-robin, so a caller that stops early still gets every stratum —
    # concatenating them would have handed back one stratum only.
    out = []
    for i in range(max(len(v) for v in picked.values())):
        for s in names:
            if i < len(picked[s]):
                out.append(picked[s][i])
    rest = [c for c in cands if c not in out]
    rng.shuffle(rest)
    return (out + rest)[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vqav2", type=int, default=15)
    ap.add_argument("--vqa-strata", default="",
                    help="e.g. A_global_easy=100,B_local_detail=100,C_multi_relational=50")
    ap.add_argument("--exclude", default="", help="manifest whose ids to skip")
    ap.add_argument("--n-gqa", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--out", default="data/manifests/general_pilot.jsonl")
    args = ap.parse_args()
    samples = []

    # --- VQAv2 -------------------------------------------------------------
    qs = json.load(open("data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json"))
    ann = {a["question_id"]: a for a in
           json.load(open("data/vqav2/v2_mscoco_train2014_annotations.json"))["annotations"]}
    skip = set()
    if args.exclude and Path(args.exclude).exists():
        skip = {json.loads(l)["sample_id"] for l in open(args.exclude)}
    cands = [{"sid": f"vqav2/train/{q['question_id']}", "q": q, "stratum": stratum(q["question"])}
             for q in qs["questions"][:200000]
             if f"vqav2/train/{q['question_id']}" not in skip]

    if args.vqa_strata:
        # Explicit per-stratum quotas: the deployment distribution is kept but
        # the weak relational regime must not dominate the training set.
        quotas = dict(kv.split("=") for kv in args.vqa_strata.split(","))
        rng = random.Random(args.seed)
        plan = []
        for st_name, cnt in quotas.items():
            pool = sorted([c for c in cands if c["stratum"] == st_name],
                          key=lambda c: c["sid"])
            plan += rng.sample(pool, min(int(cnt) * 3, len(pool)))
        order, want = plan, sum(int(v) for v in quotas.values())
        per_st = {k: 0 for k in quotas}
    else:
        order, want, per_st, quotas = pick(cands, args.n_vqav2 * 3, args.seed), args.n_vqav2, None, None

    got = 0
    for c in order:
        if got >= want:
            break
        if per_st is not None:
            if per_st[c["stratum"]] >= int(quotas[c["stratum"]]):
                continue
        q = c["q"]
        p = Path(f"data/vqav2/images/{q['image_id']}.jpg")
        if not fetch(COCO_URL.format(q["image_id"]), p):
            continue
        a = ann.get(q["question_id"])
        if not a:
            continue
        samples.append(finish(c["sid"], "vqav2", p, q["question"],
                              [x["answer"] for x in a["answers"]],
                              {"stratum": c["stratum"], "qtype": a.get("answer_type")}))
        got += 1
        if per_st is not None:
            per_st[c["stratum"]] += 1

    # --- GQA ---------------------------------------------------------------
    gq = Path("data/gqa/train_balanced_questions.json")
    if gq.exists():
        data = json.load(open(gq))
        cands = [{"sid": f"gqa/train/{k}", "q": v, "stratum": stratum(v["question"])}
                 for k, v in list(data.items())[:60000]]
        got = 0
        for c in pick(cands, args.n_gqa * 4, args.seed):
            if got >= args.n_gqa:
                break
            v = c["q"]
            p = Path(f"data/gqa/images/{v['imageId']}.jpg")
            if not any(fetch(u.format(v["imageId"]), p) for u in VG_URLS):
                continue
            samples.append(finish(c["sid"], "gqa", p, v["question"], [v["answer"]],
                                  {"stratum": c["stratum"],
                                   "types": v.get("types", {})}))
            got += 1
    else:
        print("GQA questions not downloaded yet — VQAv2 only", file=sys.stderr)

    n = write_manifest(args.out, samples)
    print(f"wrote {n} states -> {args.out}")
    from collections import Counter
    print("  by dataset:", dict(Counter(s.dataset for s in samples)))
    print("  by stratum:", dict(Counter(s.meta["stratum"] for s in samples)))
    tok = [s.native_tokens for s in samples]
    if tok:
        tok.sort()
        print(f"  native tokens: min {tok[0]} median {tok[len(tok)//2]} max {tok[-1]}")
        for c in (64, 96, 192):
            print(f"    eligible at coarse={c} (>=4x): "
                  f"{100*sum(1 for t in tok if t >= 4*c)/len(tok):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
