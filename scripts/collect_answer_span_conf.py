#!/usr/bin/env python
"""Re-run STOP requests with per-token logprobs to build answer-span confidence.

The paper's confidence is the mean token logprob of the whole generated reply,
scaffold included. Reviewer question: does the VQAv2 inversion survive
confidence statistics computed only on the *answer span*? The original
campaigns stored per-generation summaries only, so this pass re-issues the
exact STOP requests (same prompt path, greedy, seed 1234, serial) and stores
the full token/logprob stream. Span statistics and ROCs are computed offline
by scripts/answer_span_analysis.py.

One run per source log, e.g.:

    python scripts/collect_answer_span_conf.py \
        --log logs/lockbox_vqav2.jsonl --manifest data/manifests/vqav2_lockbox.jsonl \
        --coarse 64 --out data/span_conf/lockbox_vqav2.jsonl

For one-word runs set PPU_PROMPT_VARIANT=oneword (module-level in ppu.prompts,
so it must be in the environment before Python starts).

Post-hoc instrumentation: outcomes/labels still come from the original logs;
this pass only adds token-level detail to the STOP generations and verifies
the regenerated text matches the logged one.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppu import prompts, replay as replay_mod
from ppu.actions import ActionExecutor, view_labels
from ppu.datasets import load_manifest
from ppu.schema import ActionSpec


def call(client, model, system, user_text, views, max_tokens=128):
    content = [{"type": "image_url", "image_url": {"url": v.to_data_url("JPEG")}}
               for v in views]
    content.append({"type": "text", "text": user_text})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": content}],
        temperature=0.0, top_p=1.0, seed=1234, max_tokens=max_tokens,
        logprobs=True, top_logprobs=1,
    )
    ch = resp.choices[0]
    toks, lps = [], []
    if ch.logprobs and ch.logprobs.content:
        toks = [t.token for t in ch.logprobs.content]
        lps = [float(t.logprob) for t in ch.logprobs.content]
    return ch.message.content or "", toks, lps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="source campaign log (labels/answers)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--coarse", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=180, max_retries=2)

    logged = defaultdict(dict)
    for line in open(args.log):
        r = json.loads(line)
        if r.get("type") == "rollout" and r["action"]["type"] == "STOP":
            logged[r["sample_id"]][r["replay"]] = r
    sids = [s for s, d in logged.items() if {1, 2} <= set(d)]
    samples = {s.sample_id: s for s in load_manifest(args.manifest)}
    ex = ActionExecutor(coarse_tokens=args.coarse)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0, mismatch = time.time(), 0
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "run_meta", "source_log": args.log,
                             "manifest": args.manifest, "coarse": args.coarse,
                             "model": args.model,
                             "prompt_version": prompts.PROMPT_VERSION,
                             "n_states": len(sids)}) + "\n")
        for i, sid in enumerate(sorted(sids), 1):
            sample = samples[sid]
            obs = ex.observe(sample, ActionSpec("STOP"))
            row = {"type": "span_conf", "sample_id": sid}
            for rep in (1, 2):
                served = replay_mod.serve(sample, rep)
                scoring = replay_mod.scoring_sample(sample, served)
                q = prompts.render_question(scoring, served.choices or None,
                                            text=served.text)
                text = prompts.build_user_text(scoring, q,
                                               view_labels(obs, ActionSpec("STOP")))
                gen, toks, lps = call(client, args.model, prompts.SYSTEM, text, obs.views)
                old = logged[sid][rep]
                same = gen.strip() == (old.get("raw_generation") or "").strip()
                if not same:
                    mismatch += 1
                row[f"r{rep}"] = {"generation": gen, "tokens": toks, "logprobs": lps,
                                  "matches_logged": same,
                                  "logged_answer": old.get("answer_after"),
                                  "logged_score": old.get("score_after")}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0:
                r_ = (time.time() - t0) / i
                print(f"  {i}/{len(sids)} eta {r_ * (len(sids) - i) / 60:.0f} min",
                      flush=True)
    print(f"{len(sids)} states -> {args.out} ({(time.time() - t0) / 60:.1f} min); "
          f"generation mismatches vs log: {mismatch}/{2 * len(sids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
