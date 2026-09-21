"""Frozen VLM client.

Invariants (SPEC §4), all of them load-bearing for the audit:

1. The reader is never trained and never varies mid-campaign: one model id,
   one prompt version, one decode config, all logged per rollout.
2. Decoding is greedy with a fixed seed. The two replays differ by what is
   *asked* (permuted choices / paraphrase), never by sampling temperature.
3. Prefix caching MUST be off on the server. Gate 1 issues ~11 near-identical
   requests per image, which is exactly the regime where evconflict measured
   prefix caching silently changing outputs (2026-08-12).
4. Images are sent pre-resized (see imaging.py) so the visual budget is set
   here, not by server-side defaults.

`StubVLM` exists so the pipeline, the runner and the cache can be tested
without a GPU; it is deterministic and never reaches the network.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .imaging import View

DEFAULT_DECODE = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 128,
    "seed": 1234,
}


@dataclass
class Generation:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str = ""
    # Answer-token confidence. Free at generation time and legitimately visible
    # to a gate: the STOP answer is part of the state, so its probabilities
    # cost nothing extra. None when logprobs were not requested.
    mean_logprob: Optional[float] = None
    min_logprob: Optional[float] = None


class VLMClient(Protocol):
    model_id: str
    decode: dict

    def generate(self, system: str, user_text: str, views: list[View]) -> Generation: ...


@dataclass
class OpenAICompatVLM:
    """Talks to a vLLM server started with scripts/serve.sh.

    Retries are bounded and never change the request: a retried call must be
    the same experiment, otherwise a transient 5xx would quietly become a
    different observation.
    """

    model_id: str
    base_url: str = "http://localhost:8100/v1"
    api_key: str = "EMPTY"
    decode: dict = field(default_factory=lambda: dict(DEFAULT_DECODE))
    image_format: str = "JPEG"
    logprobs: bool = False
    timeout_s: float = 180.0
    max_retries: int = 3
    _client: object = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                                  timeout=self.timeout_s, max_retries=0)
        return self._client

    def generate(self, system: str, user_text: str, views: list[View]) -> Generation:
        client = self._ensure()
        content: list[dict] = []
        for v in views:
            content.append({"type": "image_url",
                            "image_url": {"url": v.to_data_url(self.image_format)}})
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": content}]

        last_err = ""
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                extra = {"logprobs": True, "top_logprobs": 1} if self.logprobs else {}
                resp = client.chat.completions.create(
                    model=self.model_id, messages=messages, **self.decode, **extra
                )
                usage = getattr(resp, "usage", None)
                ch = resp.choices[0]
                lps = []
                if self.logprobs and getattr(ch, "logprobs", None) and ch.logprobs.content:
                    lps = [t.logprob for t in ch.logprobs.content]
                return Generation(
                    text=ch.message.content or "",
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_s=time.time() - t0,
                    mean_logprob=(sum(lps) / len(lps)) if lps else None,
                    min_logprob=min(lps) if lps else None,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced in the record
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(2.0 * (attempt + 1))
        return Generation(text="", latency_s=0.0, error=last_err)


@dataclass
class StubVLM:
    """Deterministic offline stand-in used by the tests.

    Answers by hashing (question, views, aux text), so different observations
    deterministically produce different answers — enough to exercise scoring,
    caching, and the audit metrics end to end.
    """

    model_id: str = "stub-vlm"
    decode: dict = field(default_factory=lambda: dict(DEFAULT_DECODE))
    answer_table: dict = field(default_factory=dict)  # digest -> answer text
    calls: int = 0

    def generate(self, system: str, user_text: str, views: list[View]) -> Generation:
        self.calls += 1
        digest = hashlib.sha1(
            (user_text + "|" + "|".join(f"{v.label}:{v.tokens}" for v in views)).encode()
        ).hexdigest()
        answer = self.answer_table.get(digest)
        if answer is None:
            answer = self.answer_table.get(digest[:8], digest[:6])
        return Generation(text=f"Reasoning.\nAnswer: {answer}",
                          prompt_tokens=len(user_text) // 4,
                          completion_tokens=8, latency_s=0.001)

    def key_for(self, user_text: str, views: list[View]) -> str:
        return hashlib.sha1(
            (user_text + "|" + "|".join(f"{v.label}:{v.tokens}" for v in views)).encode()
        ).hexdigest()
