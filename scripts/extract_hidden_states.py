#!/usr/bin/env python
"""Pre-action hidden-state extraction for selector v3 (deployable feature).

For each state, builds EXACTLY the STOP-time input the reader sees before any
action is paid for — system prompt, coarse view at the dataset's token
budget, replay-1 question — runs one HF forward pass of the same frozen
reader (Qwen2.5-VL-7B-Instruct, bf16), and stores the mean-pooled last-layer
hidden state (float16). This is information available to a deployed
controller at decision time; no action outcome is touched.

    .venv? no — needs transformers: run with the evconflict venv.
    python scripts/extract_hidden_states.py \
        --manifest data/manifests/textvqa_val.jsonl \
        --ids data/textvqa_v3_prospective_ids.txt --coarse 192 \
        --out data/hidden/v3_prospective.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ppu import imaging, prompts, replay as replay_mod  # noqa: E402
from ppu.datasets import load_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--coarse", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to(args.device)
    model.eval()

    samples = {s.sample_id: s for s in load_manifest(args.manifest)}
    want = [l.strip() for l in open(args.ids) if l.strip()]
    out_ids, vecs = [], []
    import io
    from PIL import Image
    for i, sid in enumerate(want, 1):
        s = samples.get(sid)
        if s is None:
            continue
        served = replay_mod.serve(s, 1)
        scoring = replay_mod.scoring_sample(s, served)
        q = prompts.render_question(scoring, served.choices or None,
                                    text=served.text)
        view = imaging.coarse_view(s.image_path, args.coarse)
        text = prompts.build_user_text(scoring, q,
                                       ["low-resolution overview of the whole image"])
        # JPEG round-trip (quality 92) to match the served transport exactly
        buf = io.BytesIO()
        view.image.save(buf, format="JPEG", quality=92)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        msgs = [{"role": "system", "content": prompts.SYSTEM},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": text}]}]
        chat = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        inputs = proc(text=[chat], images=[img], return_tensors="pt").to(args.device)
        with torch.no_grad():
            h = model(**inputs, output_hidden_states=True).hidden_states[-1]
        vecs.append(h[0].mean(dim=0).to(torch.float16).cpu().numpy())
        out_ids.append(sid)
        if i % 50 == 0:
            print(f"  {i}/{len(want)}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(out_ids),
                        H=np.stack(vecs))
    print(f"{len(out_ids)} states -> {args.out}  (dim {vecs[0].shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
