#!/usr/bin/env python
"""Stage A0: collect the two baseline signals a gate could use for free.

    1. confidence  — re-run the STOP request with logprobs on. The answer is
       already part of the state, so its token probabilities cost nothing extra.
    2. self-assessment — ask the reader itself whether the coarse view suffices.
       This is the baseline that decides whether a *learned* gate is needed at
       all: if the model already knows when it needs to look, Stage A is a
       prompt, not a model.

The STOP request is rebuilt through the same prompt/executor path as the
campaign, so the confidence belongs to the answer that was actually logged —
the script reports the answer-match rate as a check on that.

    python scripts/collect_gate_features.py --out data/stage_a_features.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppu import prompts, replay as replay_mod
from ppu.actions import ActionExecutor, view_labels
from ppu.datasets import load_manifest
from ppu.schema import ActionSpec

BUDGETS = {"mme": 256, "textvqa": 192}
MANIFESTS = {"mme": "data/manifests/mme_realworld_lite.jsonl",
             "textvqa": "data/manifests/textvqa_val.jsonl"}

SELF_ASSESS = (
    "Before answering, judge your own view. Can you answer the question "
    "reliably from this low-resolution overview, or would you need a "
    "higher-resolution look at the image?\n"
    "Reply with exactly one word: SUFFICIENT or NEED_DETAIL."
)


def call(client, model, system, user_text, views, logprobs=True, max_tokens=128):
    content = [{"type": "image_url", "image_url": {"url": v.to_data_url("JPEG")}}
               for v in views]
    content.append({"type": "text", "text": user_text})
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": content}],
        temperature=0.0, top_p=1.0, seed=1234, max_tokens=max_tokens,
        logprobs=logprobs, top_logprobs=1 if logprobs else None,
    )
    ch = resp.choices[0]
    lps = []
    if logprobs and ch.logprobs and ch.logprobs.content:
        lps = [t.logprob for t in ch.logprobs.content]
    return ch.message.content or "", lps


def summarize(lps: list[float]) -> dict:
    if not lps:
        return {"mean_logprob": None, "min_logprob": None, "n_tokens": 0}
    return {"mean_logprob": sum(lps) / len(lps), "min_logprob": min(lps),
            "n_tokens": len(lps)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/stage_a_labels_highres.jsonl")
    ap.add_argument("--out", default="data/stage_a_features.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8100/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=180, max_retries=2)
    states = [json.loads(l) for l in open(args.labels, encoding="utf-8") if l.strip()]
    manifests = {k: {s.sample_id: s for s in load_manifest(v)}
                 for k, v in MANIFESTS.items() if Path(v).exists()}
    execs = {k: ActionExecutor(coarse_tokens=b) for k, b in BUDGETS.items()}

    out, t0, mismatch = [], time.time(), 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, st in enumerate(states, 1):
            ds = st["dataset"]
            sample = manifests[ds][st["state_id"]]
            ex = execs[ds]
            obs = ex.observe(sample, ActionSpec("STOP"))
            row = {"state_id": st["state_id"], "dataset": ds, "label": st["label"]}

            for rep in (1, 2):
                served = replay_mod.serve(sample, rep)
                scoring = replay_mod.scoring_sample(sample, served)
                q = prompts.render_question(scoring, served.choices or None,
                                            text=served.text)
                text = prompts.build_user_text(scoring, q,
                                               view_labels(obs, ActionSpec("STOP")))
                gen, lps = call(client, args.model, prompts.SYSTEM, text, obs.views)
                row[f"stop_r{rep}"] = {"generation": gen, **summarize(lps)}

            # the confidence must belong to the answer that was logged
            if row["stop_r1"]["generation"].strip().endswith(
                    st["coarse_answer"].strip()) is False:
                mismatch += 1

            served = replay_mod.serve(sample, 1)
            scoring = replay_mod.scoring_sample(sample, served)
            q = prompts.render_question(scoring, served.choices or None,
                                        text=served.text)
            sa_text = f"{q}\n\n{SELF_ASSESS}"
            gen, lps = call(client, args.model, prompts.SYSTEM, sa_text, obs.views,
                            max_tokens=8)
            up = gen.upper()
            row["self_assessment"] = {
                "generation": gen,
                "needs_detail": ("NEED_DETAIL" in up)
                                or ("NEED" in up and "SUFFICIENT" not in up),
                **summarize(lps),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            out.append(row)
            if i % 25 == 0:
                r = (time.time() - t0) / i
                print(f"  {i}/{len(states)}  eta {r * (len(states) - i) / 60:.0f} min",
                      flush=True)

    nd = sum(1 for r in out if r["self_assessment"]["needs_detail"])
    print(f"\n{len(out)} states -> {args.out}  ({(time.time() - t0) / 60:.1f} min)")
    print(f"  answer-text mismatch vs logged STOP: {mismatch}/{len(out)}")
    print(f"  self-assessment says NEED_DETAIL: {nd} ({100 * nd / max(len(out), 1):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
