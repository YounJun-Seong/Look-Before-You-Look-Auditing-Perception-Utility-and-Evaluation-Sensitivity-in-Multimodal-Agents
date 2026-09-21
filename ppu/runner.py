"""Brute-force Gate 1 runner: execute every candidate action on every sample.

No predictor, no policy, no sequential state — Gate 1 asks what the *outcomes*
of the actions are, and any selection logic here would contaminate that. One
step from s0, 11 real actions (STOP, CROP x8, OCR, HIGHRES), optionally 11
cardinality-matched dummies, each under two replays.

Order of operations per (sample, replay):
    1. STOP -> answer_before / score_before for every other action of that replay
    2. the remaining actions, each logged with its own answer_after
STOP is therefore both a candidate action and the reference point, which is
exactly its status in the audit.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable, Optional

from . import prompts, regions, replay as replay_mod
from .actions import ActionExecutor, view_labels
from .cache import GenerationCache
from .evaluator import evaluate
from .logger import RolloutLogger
from .schema import ActionSpec, RolloutRecord, Sample, config_hash
from .vlm import Generation, VLMClient


@dataclass
class RolloutEngine:
    vlm: VLMClient
    executor: ActionExecutor
    cache: Optional[GenerationCache] = None
    prompt_version: str = prompts.PROMPT_VERSION
    run_id: str = "dev"
    concurrency: int = 1  # requests in flight per sample; 1 = strictly serial
    _proposal_memo: dict = field(default_factory=dict)

    # --- low level ---------------------------------------------------------

    def _generate(self, system: str, user_text: str, views, extra_key: dict) -> Generation:
        payload = {
            "model": self.vlm.model_id,
            "prompt_version": self.prompt_version,
            "decode": self.vlm.decode,
            "system": system,
            "user_text": user_text,
            "views": [f"{v.label}:{v.size[0]}x{v.size[1]}:{v.tokens}" for v in views],
            "image_format": getattr(self.vlm, "image_format", "PNG"),
            "logprobs": getattr(self.vlm, "logprobs", False),
            **extra_key,
        }
        if self.cache is not None:
            key = self.cache.make_key(payload)
            hit = self.cache.get(key)
            if hit is not None:
                return Generation(hit.text, hit.prompt_tokens, hit.completion_tokens,
                                  hit.latency_s)
        gen = self.vlm.generate(system, user_text, views)
        if self.cache is not None and not gen.error:
            self.cache.put(key, self.vlm.model_id, payload, gen)
        return gen

    # --- candidate generation ---------------------------------------------

    def candidates(self, sample: Sample) -> tuple[list[ActionSpec], Generation]:
        """Grid + VLM-proposed crops. The proposal call is a real cost and is
        charged to every arm that considers CROP (SPEC §3.4)."""
        if sample.sample_id in self._proposal_memo:
            return self._proposal_memo[sample.sample_id]
        view = self.executor.coarse(sample)
        text = prompts.build_proposal_text(sample.question)
        gen = self._generate(prompts.SYSTEM, text, [view],
                             {"purpose": "proposal", "sample_id": sample.sample_id})
        boxes = regions.parse_proposals(gen.text, limit=3)
        specs = regions.build_candidates(boxes)
        self._proposal_memo[sample.sample_id] = (specs, gen)
        return specs, gen

    def action_set(self, sample: Sample, dummy: bool = False) -> list[ActionSpec]:
        crops, _ = self.candidates(sample)
        if not dummy:
            return ([ActionSpec("STOP")] + list(crops)
                    + [ActionSpec("OCR"), ActionSpec("HIGHRES")])
        rnd = regions.random_boxes(sample.sample_id, crops, n=8)
        return ([ActionSpec("NOOP")] + rnd
                + [ActionSpec("REENCODE", slot=0), ActionSpec("REENCODE", slot=1)])

    # --- one rollout -------------------------------------------------------

    def run_action(self, sample: Sample, spec: ActionSpec, replay: int,
                   before: Optional[tuple[str, float]] = None) -> RolloutRecord:
        served = replay_mod.serve(sample, replay)
        score_sample = replay_mod.scoring_sample(sample, served)
        question_served = prompts.render_question(score_sample, served.choices or None,
                                                  text=served.text)

        t0 = time.time()
        obs = self.executor.observe(sample, spec)
        user_text = prompts.build_user_text(
            score_sample, question_served, view_labels(obs, spec), obs.aux_text
        )
        gen = self._generate(prompts.SYSTEM, user_text, obs.views,
                             {"purpose": "answer", "sample_id": sample.sample_id,
                              "action": spec.key, "replay": replay})
        scored = evaluate(score_sample, gen.text)

        cost = obs.cost
        cost.text_tokens_in = gen.prompt_tokens
        cost.tokens_out = gen.completion_tokens
        cost.latency_s = time.time() - t0

        ans_before, sc_before = before if before is not None else (scored.answer, scored.score)
        return RolloutRecord(
            run_id=self.run_id,
            replay=replay,
            dummy=spec.is_dummy,
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            image_path=sample.image_path,
            native_hw=sample.native_hw,
            native_tokens=sample.native_tokens,
            question=sample.question,
            question_served=question_served,
            gold=tuple(score_sample.answers),
            state_id="s0",
            history=(),
            action={"type": spec.type, "family": spec.family, "slot": spec.slot,
                    "box": list(spec.box) if spec.box else None, "source": spec.source,
                    "key": spec.key},
            obs_summary=obs.summary,
            views=[{"label": v.label, "tokens": v.tokens,
                    "w": v.size[0], "h": v.size[1], "sha": v.sha}
                   for v in obs.views],
            answer_before=ans_before,
            answer_after=scored.answer,
            score_before=sc_before,
            score_after=scored.score,
            parse_ok=scored.parse_ok,
            cost=cost.__dict__,
            model_id=self.vlm.model_id,
            prompt_version=self.prompt_version,
            decode=dict(self.vlm.decode),
            config_hash=config_hash({"model": self.vlm.model_id,
                                     "prompt_version": self.prompt_version,
                                     "decode": self.vlm.decode,
                                     "coarse": self.executor.coarse_tokens,
                                     "highres": self.executor.highres_tokens,
                                     "crop": self.executor.crop_tokens}),
            raw_generation=gen.text,
            confidence={"mean_logprob": gen.mean_logprob,
                        "min_logprob": gen.min_logprob},
            error=gen.error,
        )

    def run_sample(self, sample: Sample, replays: Iterable[int] = (1, 2),
                   dummy: bool = False, reference: bool = True) -> list[RolloutRecord]:
        records: list[RolloutRecord] = []
        for r in replays:
            specs = self.action_set(sample, dummy=dummy)
            stop_spec = specs[0]
            # STOP first: it defines answer_before for every other action of
            # this replay, so it can never run concurrently with them.
            stop_rec = self.run_action(sample, stop_spec, r)
            records.append(stop_rec)
            before = (stop_rec.answer_after, stop_rec.score_after)

            rest = list(specs[1:])
            if reference and not dummy:
                # Full-information row: not a candidate, never in the oracle.
                rest.append(ActionSpec("FULL"))
            if self.concurrency > 1:
                # Order of the returned records stays fixed regardless of
                # completion order — the log must not depend on scheduling.
                with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                    futures = [pool.submit(self.run_action, sample, spec, r, before)
                               for spec in rest]
                    records.extend(f.result() for f in futures)
            else:
                records.extend(self.run_action(sample, spec, r, before=before)
                               for spec in rest)
        return records


def run_campaign(engine: RolloutEngine, samples: list[Sample], logger: RolloutLogger,
                 replays: Iterable[int] = (1, 2), dummy: bool = False,
                 reference: bool = True, progress_every: int = 10) -> int:
    n = 0
    t0 = time.time()
    for i, sample in enumerate(samples, 1):
        try:
            for rec in engine.run_sample(sample, replays=replays, dummy=dummy,
                                         reference=reference):
                logger.log(rec)
                n += 1
        except Exception as exc:  # noqa: BLE001 - one bad sample must not kill a campaign
            logger.note(level="error", sample_id=sample.sample_id,
                        error=f"{type(exc).__name__}: {exc}")
        if progress_every and i % progress_every == 0:
            rate = (time.time() - t0) / i
            logger.note(level="progress", done=i, total=len(samples),
                        rollouts=n, sec_per_sample=round(rate, 2),
                        eta_min=round(rate * (len(samples) - i) / 60, 1))
    return n
