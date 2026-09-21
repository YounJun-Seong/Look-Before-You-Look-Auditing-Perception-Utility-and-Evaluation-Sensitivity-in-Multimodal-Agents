"""Visual-budget accounting and view rendering.

The whole project rests on one claim: the reader starts with a *cheap* view and
must pay to see more. That claim is only meaningful if "how much was paid" is
exact, so budgets are enforced here in the same units the model bills in —
Qwen2.5-VL visual tokens, where one token covers a 28x28 pixel block after
smart-resize (14px patches, 2x2 merge).

    tokens = (h_bar / 28) * (w_bar / 28),  h_bar, w_bar = smart_resize(h, w)

so a `max_pixels` of N * 28 * 28 caps the view at N visual tokens. SPEC §3.1:
coarse = 256 tokens, HIGHRES = 1280 tokens, CROP = 256 tokens.

We resize the image ourselves and send the resized bytes rather than passing
min_pixels/max_pixels to the server, because the budget must be identical
across every backend and every replay, and must be *auditable from the log*.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import io
import math
from dataclasses import dataclass
from typing import Optional

from PIL import Image

FACTOR = 28
MIN_PIXELS = 4 * FACTOR * FACTOR  # 4 tokens: below this a crop is unreadable
Box = tuple[float, float, float, float]


def _round_by_factor(x: float, factor: int = FACTOR) -> int:
    return max(factor, int(round(x / factor)) * factor)


def _floor_by_factor(x: float, factor: int = FACTOR) -> int:
    return max(factor, int(math.floor(x / factor)) * factor)


def _ceil_by_factor(x: float, factor: int = FACTOR) -> int:
    return max(factor, int(math.ceil(x / factor)) * factor)


def smart_resize(
    height: int,
    width: int,
    factor: int = FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = 256 * FACTOR * FACTOR,
) -> tuple[int, int]:
    """Qwen2-VL smart resize: keep aspect ratio, snap to `factor`, clamp area.

    Mirrors qwen_vl_utils.smart_resize so our token accounting matches what the
    server would have computed; reimplemented to avoid a dependency on the
    inference stack inside the (deliberately light) PPU venv.
    """
    if height < 1 or width < 1:
        raise ValueError(f"degenerate image size {height}x{width}")
    ratio = max(height, width) / min(height, width)
    if ratio > 200:
        raise ValueError(f"aspect ratio {ratio:.1f} beyond supported range")

    h_bar = _round_by_factor(height, factor)
    w_bar = _round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def tokens_for(height: int, width: int) -> int:
    """Visual tokens an already-sized view costs."""
    return (height // FACTOR) * (width // FACTOR)


def native_tokens(path: str) -> int:
    """Tokens the *unrestricted* image would cost. Used by the inclusion rule
    (SPEC §3.1): a sample only enters Gate 1 if its native view is at least 4x
    the coarse budget, otherwise CROP/HIGHRES cannot possibly add information.
    """
    with Image.open(path) as im:
        w, h = im.size
    h_bar, w_bar = smart_resize(h, w, max_pixels=10**9)
    return tokens_for(h_bar, w_bar)


@dataclass
class View:
    """A rendered observation plus its exact price."""

    image: Image.Image
    tokens: int
    label: str
    box: Optional[Box] = None

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size  # (w, h)

    @property
    def sha(self) -> str:
        """Hash of the rendered pixels.

        Lets the audit prove that eight CROP candidates are eight different
        images and that the dummy pool holds no accidental duplicates —
        distinct coordinates do not guarantee distinct observations.
        """
        return hashlib.sha1(self.image.tobytes()).hexdigest()[:12]

    def to_data_url(self, fmt: str = "JPEG", quality: int = 92) -> str:
        """Encode for transport.

        JPEG by default: PNG encoding of a multi-megapixel view costs seconds
        per rollout and dominates the campaign. The format is part of the
        generation cache key (runner._generate), so switching it invalidates
        cached answers instead of silently mixing two encodings.
        """
        buf = io.BytesIO()
        if fmt.upper() == "JPEG":
            self.image.save(buf, format="JPEG", quality=quality, subsampling=0)
        else:
            self.image.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
        return f"data:{mime};base64,{b64}"


def _render(im: Image.Image, budget_tokens: int, label: str, box: Optional[Box]) -> View:
    w, h = im.size
    h_bar, w_bar = smart_resize(h, w, max_pixels=budget_tokens * FACTOR * FACTOR)
    out = im.convert("RGB").resize((w_bar, h_bar), Image.BICUBIC)
    return View(image=out, tokens=tokens_for(h_bar, w_bar), label=label, box=box)


# Views are rendered repeatedly (once per replay, again for the dummy arm), and
# resizing a multi-megapixel source is not free. Small cache, bounded memory.
@functools.lru_cache(maxsize=48)
def coarse_view(path: str, budget_tokens: int = 256) -> View:
    with Image.open(path) as im:
        return _render(im, budget_tokens, f"coarse@{budget_tokens}", None)


@functools.lru_cache(maxsize=16)
def highres_view(path: str, budget_tokens: int = 1280) -> View:
    with Image.open(path) as im:
        return _render(im, budget_tokens, f"highres@{budget_tokens}", None)


@functools.lru_cache(maxsize=16)
def reencode_view(path: str, budget_tokens: int = 256, variant: int = 0) -> View:
    """Dummy action: same budget as coarse, cosmetically different encoding.

    Measures how much of the oracle's gain is the reader simply being asked
    again with a slightly different image (SPEC §6.3). Two variants exist so
    the dummy pool holds 11 *distinct* draws — a repeated identical dummy would
    contribute nothing to the max and would understate the luck floor.
    """
    with Image.open(path) as im:
        w, h = im.size
        h_bar, w_bar = smart_resize(h, w, max_pixels=budget_tokens * FACTOR * FACTOR)
        if variant == 0:
            out = im.convert("RGB").resize((w_bar, h_bar), Image.LANCZOS)
        else:
            out = im.convert("RGB").resize((w_bar, h_bar), Image.BILINEAR)
            buf = io.BytesIO()
            out.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            out = Image.open(buf).convert("RGB")
    return View(out, tokens_for(h_bar, w_bar), f"reencode{variant}@{budget_tokens}", None)


@functools.lru_cache(maxsize=8)
def full_view(path: str, cap_tokens: int = 2560) -> View:
    """Full-information reference: the native view, capped so a gigapixel image
    cannot blow up the server. The realized token count is logged, so rows that
    hit the cap are identifiable in analysis."""
    with Image.open(path) as im:
        return _render(im, cap_tokens, f"full@{cap_tokens}", None)


@functools.lru_cache(maxsize=32)
def crop_view(path: str, box: Box, budget_tokens: int = 256) -> View:
    """Crop `box` (normalized) from the ORIGINAL image, then re-encode at budget.

    Cropping from the original — not from the coarse view — is what makes CROP
    an information-acquiring action rather than an upsampling artifact.
    """
    x0, y0, x1, y1 = box
    with Image.open(path) as im:
        w, h = im.size
        px = (
            max(0, int(round(x0 * w))),
            max(0, int(round(y0 * h))),
            min(w, int(round(x1 * w))),
            min(h, int(round(y1 * h))),
        )
        if px[2] - px[0] < 8 or px[3] - px[1] < 8:
            raise ValueError(f"crop box {box} degenerate on {w}x{h} image")
        region = im.crop(px)
        return _render(region, budget_tokens, f"crop@{budget_tokens}", box)
