"""Prompt templates, versioned.

One template for every arm. If an action needed its own instructions, the
comparison between actions would partly measure prompt engineering, so the
only thing that varies across actions is the *observation block* they append.

PROMPT_VERSION is logged with every rollout; bumping it invalidates the cache.
"""

from __future__ import annotations

import os

from .schema import Sample

# PPU_PROMPT_VARIANT=oneword swaps the open-ended instruction for a one-word
# constraint (2026-08-22 format-confound follow-up). The version string changes
# with it so logs and caches never mix the two.
PROMPT_VARIANT = os.environ.get("PPU_PROMPT_VARIANT", "")
PROMPT_VERSION = "p0.2-oneword" if PROMPT_VARIANT == "oneword" else "p0.1"

SYSTEM = (
    "You are answering questions about images. You may be shown a low-resolution "
    "overview, additional zoomed-in views, or text extracted from the image. "
    "Use only what you are shown. Answer as briefly as possible."
)

_OPEN_INSTRUCTION = (
    "Answer with a single word or number. For yes/no questions answer exactly "
    "'yes' or 'no'. Do not write a sentence. "
    "End your reply with a line of the form:\nAnswer: <your answer>"
) if PROMPT_VARIANT == "oneword" else (
    "Answer the question with a short phrase (a few words at most). "
    "End your reply with a line of the form:\nAnswer: <your answer>"
)

_MCQ_INSTRUCTION = (
    "Answer with the letter of the correct option. "
    "End your reply with a line of the form:\nAnswer: <letter>"
)


def render_question(sample: Sample, choices: tuple[str, ...] | None = None,
                    text: str | None = None) -> str:
    """Question text as served.

    `text` is the replay's rendering (R2 paraphrases open-ended questions); it
    must be honoured, or R2 silently becomes a byte-identical re-run of R1 and
    the strict label and cross-fitted oracle lose their control. MCQ carries its
    replay in `choices` instead, so `text` equals the original there.
    """
    q = text if text is not None else sample.question
    if sample.metric != "mcq":
        return q
    opts = choices if choices is not None else sample.choices
    lines = [q]
    for letter, opt in zip("ABCDE", opts):
        lines.append(f"{letter}. {opt}")
    return "\n".join(lines)


def build_user_text(sample: Sample, question_served: str, view_labels: list[str],
                    aux_text: str = "") -> str:
    """Assemble the user turn.

    View labels are described in words (e.g. "zoomed view of region
    x0,y0,x1,y1") so the reader knows what it is looking at; the coordinates
    are already part of the action spec, so nothing here leaks the *content*
    of an observation the agent has not paid for.
    """
    parts = []
    if view_labels:
        parts.append("Views provided, in order: " + "; ".join(view_labels) + ".")
    if aux_text:
        parts.append("Text extracted from the image by an OCR tool:\n" + aux_text)
    parts.append(question_served)
    parts.append(_MCQ_INSTRUCTION if sample.metric == "mcq" else _OPEN_INSTRUCTION)
    return "\n\n".join(parts)


# --- crop proposal (charged to every arm that considers CROP, SPEC §3.4) ----

PROPOSAL_INSTRUCTION = (
    "Look at the image and the question. Name up to 3 regions you would need to "
    "inspect at higher resolution to answer it. Reply with one region per line, "
    "each as four numbers between 0 and 1: x0,y0,x1,y1 (left, top, right, bottom "
    "as fractions of the image). No other text."
)


def build_proposal_text(question: str) -> str:
    return f"Question: {question}\n\n{PROPOSAL_INSTRUCTION}"
