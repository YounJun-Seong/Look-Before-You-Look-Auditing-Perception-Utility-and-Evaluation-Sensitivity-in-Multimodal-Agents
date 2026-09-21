#!/usr/bin/env python
"""Ask the reader whether its own coarse view suffices — the anti-harm veto.

Confidence already comes free with every STOP rollout, so this collects only
the self-assessment, one call per state on the coarse view it actually saw.
The combined rule was fixed before these results existed: a LOOK candidate is
vetoed when the reader answers SUFFICIENT.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppu import prompts, replay as replay_mod
from ppu.actions import ActionExecutor, view_labels
from ppu.datasets import load_manifest
from ppu.schema import ActionSpec

SOURCES = {"textvqa": ("data/manifests/textvqa_train.jsonl", 192),
           "vqav2": ("data/manifests/vqav2_a1.jsonl", 64)}
SELF_ASSESS = ("Before answering, judge your own view. Can you answer the question "
               "reliably from this low-resolution overview, or would you need a "
               "higher-resolution look at the image?\n"
               "Reply with exactly one word: SUFFICIENT or NEED_DETAIL.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/stage_a1_dataset.jsonl")
    ap.add_argument("--out", default="data/stage_a1_self_assessment.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=180, max_retries=2)
    states = [json.loads(l) for l in open(args.dataset)]
    mans = {k: {s.sample_id: s for s in load_manifest(v[0])} for k, v in SOURCES.items()
            if Path(v[0]).exists()}
    execs = {k: ActionExecutor(coarse_tokens=v[1]) for k, v in SOURCES.items()}

    t0, n_need = time.time(), 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, st in enumerate(states, 1):
            src = st["source"]
            sample = mans[src].get(st["state_id"])
            if sample is None:
                continue
            obs = execs[src].observe(sample, ActionSpec("STOP"))
            served = replay_mod.serve(sample, 1)
            scoring = replay_mod.scoring_sample(sample, served)
            q = prompts.render_question(scoring, served.choices or None, text=served.text)
            content = [{"type": "image_url",
                        "image_url": {"url": v.to_data_url("JPEG")}} for v in obs.views]
            content.append({"type": "text", "text": f"{q}\n\n{SELF_ASSESS}"})
            r = client.chat.completions.create(
                model=args.model, temperature=0.0, top_p=1.0, seed=1234, max_tokens=8,
                logprobs=True, top_logprobs=1,
                messages=[{"role": "system", "content": prompts.SYSTEM},
                          {"role": "user", "content": content}])
            ch = r.choices[0]
            gen = ch.message.content or ""
            lps = [t.logprob for t in ch.logprobs.content] if ch.logprobs and ch.logprobs.content else []
            up = gen.upper()
            need = ("NEED_DETAIL" in up) or ("NEED" in up and "SUFFICIENT" not in up)
            n_need += need
            fh.write(json.dumps({"state_id": st["state_id"], "source": src,
                                 "label": st["label"], "generation": gen,
                                 "needs_detail": bool(need),
                                 "mean_logprob": (sum(lps)/len(lps)) if lps else None},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(states)}  eta "
                      f"{(time.time()-t0)/i*(len(states)-i)/60:.0f} min", flush=True)
    print(f"{len(states)} states -> {args.out} ({(time.time()-t0)/60:.1f} min); "
          f"NEED_DETAIL {n_need} ({100*n_need/max(len(states),1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
