"""Stage 1-2 tests: the pipeline must be provably deterministic and honest
before any GPU time is spent on it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppu import imaging, regions, replay
from ppu.actions import ActionExecutor
from ppu.cache import GenerationCache
from ppu.datasets import draw, load_manifest, passes_inclusion, write_manifest
from ppu.evaluator import anls, evaluate, extract_answer, normalize_answer
from ppu.logger import RolloutLogger, read_rollouts
from ppu.ocr import OCRLine, StubOCR
from ppu.runner import RolloutEngine, run_campaign
from ppu.schema import ActionSpec, Sample
from ppu.vlm import StubVLM


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def image_path(tmp_path: Path) -> str:
    im = Image.new("RGB", (1400, 1000), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([100, 100, 400, 300], fill="red")
    d.text((900, 800), "SALE 30%", fill="black")
    p = tmp_path / "img.png"
    im.save(p)
    return str(p)


@pytest.fixture
def sample(image_path: str) -> Sample:
    return Sample(
        sample_id="synthetic/0001", dataset="synthetic", image_path=image_path,
        question="What is the price?", answers=("14.99",), metric="exact",
        native_hw=(1000, 1400), native_tokens=imaging.native_tokens(image_path),
    )


@pytest.fixture
def mcq_sample(image_path: str) -> Sample:
    return Sample(
        sample_id="mme/0001", dataset="mme_realworld", image_path=image_path,
        question="What colour is the box?", answers=("B",), metric="mcq",
        choices=("blue", "red", "green", "black", "white"),
        native_hw=(1000, 1400), native_tokens=imaging.native_tokens(image_path),
    )


# --- budgets ----------------------------------------------------------------

def test_budgets_are_respected(image_path: str):
    coarse = imaging.coarse_view(image_path, 256)
    high = imaging.highres_view(image_path, 1280)
    crop = imaging.crop_view(image_path, (0.0, 0.0, 0.3, 0.3), 256)
    assert coarse.tokens <= 256 and high.tokens <= 1280 and crop.tokens <= 256
    # The budget must actually bind, otherwise "paying to see more" is fiction.
    assert coarse.tokens > 200 and high.tokens > 1000
    assert high.tokens > coarse.tokens * 3


def test_token_formula_matches_smart_resize():
    h, w = imaging.smart_resize(1000, 1400, max_pixels=256 * 28 * 28)
    assert h % 28 == 0 and w % 28 == 0
    assert imaging.tokens_for(h, w) <= 256


def test_crop_reads_from_original_not_coarse(image_path: str):
    """A crop must carry more detail than the same region of the coarse view."""
    coarse = imaging.coarse_view(image_path, 256)
    crop = imaging.crop_view(image_path, (0.0, 0.0, 0.25, 0.25), 256)
    coarse_region_px = (coarse.size[0] * 0.25) * (coarse.size[1] * 0.25)
    assert crop.size[0] * crop.size[1] > coarse_region_px


def test_inclusion_rule(sample: Sample):
    assert passes_inclusion(sample, 256)
    tiny = Sample(sample_id="x", dataset="synthetic", image_path=sample.image_path,
                  question="q", answers=("a",), metric="exact", native_tokens=300)
    assert not passes_inclusion(tiny, 256)


# --- candidate regions ------------------------------------------------------

def test_candidate_set_is_always_eight():
    for proposals in ([], [(0.1, 0.1, 0.2, 0.2)], regions.GRID_BOXES[:3]):
        specs = regions.build_candidates(list(proposals))
        assert len(specs) == regions.K_CANDIDATES
        assert len({s.box for s in specs}) == regions.K_CANDIDATES
        assert [s.slot for s in specs] == list(range(8))


def test_proposal_parsing_rejects_garbage():
    assert regions.parse_proposals("0.1,0.2,0.5,0.6\nnonsense\n2,3,4,5") == \
        [(0.1, 0.2, 0.5, 0.6)]
    assert regions.parse_proposals("") == []
    # too small to be a useful crop -> dropped, not repaired
    assert regions.parse_proposals("0.10,0.10,0.11,0.11") == []


def test_random_boxes_match_scale_and_are_deterministic():
    specs = regions.build_candidates([])
    a = regions.random_boxes("s1", specs)
    b = regions.random_boxes("s1", specs)
    assert [x.box for x in a] == [x.box for x in b]
    assert len(a) == 8
    area = lambda bx: (bx[2] - bx[0]) * (bx[3] - bx[1])
    assert abs(sum(area(x.box) for x in a) - sum(area(x.box) for x in specs)) < 0.2


# --- scoring ----------------------------------------------------------------

def test_normalization_and_metrics():
    assert normalize_answer("The Two Dogs.") == "2 dogs"
    assert anls("14.99", ("14.99",)) == 1.0
    assert anls("wildly different", ("14.99",)) == 0.0


def test_answer_extraction_forms(sample: Sample):
    assert extract_answer("blah\nAnswer: 14.99", sample)[0] == "14.99"
    assert extract_answer("<answer>14.99</answer>", sample)[0] == "14.99"
    assert extract_answer("14.99", sample)[0] == "14.99"
    assert extract_answer("", sample) == ("", False)


def test_mcq_scoring_follows_permutation(mcq_sample: Sample):
    served = replay.serve(mcq_sample, 2)
    scored_sample = replay.scoring_sample(mcq_sample, served)
    # gold text is still "red", but its letter moved
    assert served.choices[("ABCDE".index(served.gold_letter))] == "red"
    assert evaluate(scored_sample, f"Answer: {served.gold_letter}").score == 1.0
    if served.gold_letter != "B":
        assert evaluate(scored_sample, "Answer: B").score == 0.0


def test_replay_two_changes_the_question(sample: Sample):
    r1 = replay.serve(sample, 1)
    r2 = replay.serve(sample, 2)
    assert r1.text != r2.text and sample.question.rstrip("?").lower() in r2.text.lower()


# --- actions ----------------------------------------------------------------

def test_action_costs_and_views(sample: Sample):
    ocr = StubOCR(table={sample.image_path: [OCRLine("SALE 30%", (0.6, 0.7, 0.9, 0.8))]})
    ex = ActionExecutor(ocr_engine=ocr)

    stop = ex.observe(sample, ActionSpec("STOP"))
    assert len(stop.views) == 1 and stop.cost.visual_tokens == 0

    crop = ex.observe(sample, ActionSpec("CROP", slot=0, box=(0.1, 0.1, 0.5, 0.5)))
    assert len(crop.views) == 2  # context kept alongside the zoom
    assert crop.cost.visual_tokens > 0

    high = ex.observe(sample, ActionSpec("HIGHRES"))
    assert len(high.views) == 1 and high.cost.visual_tokens > 900

    o = ex.observe(sample, ActionSpec("OCR"))
    assert "SALE 30%" in o.aux_text and o.cost.tool_calls == 1
    assert o.cost.visual_tokens == 0  # OCR buys text, not visual tokens


def test_ocr_action_requires_engine(sample: Sample):
    with pytest.raises(RuntimeError):
        ActionExecutor().observe(sample, ActionSpec("OCR"))


# --- runner end to end ------------------------------------------------------

def _engine(tmp_path: Path, sample: Sample, answers: dict | None = None):
    vlm = StubVLM()
    ocr = StubOCR(table={sample.image_path: [OCRLine("14.99", (0.6, 0.7, 0.9, 0.8))]})
    cache = GenerationCache(tmp_path / "cache.sqlite")
    return RolloutEngine(vlm=vlm, executor=ActionExecutor(ocr_engine=ocr),
                         cache=cache, run_id="test"), vlm, cache


def test_action_set_cardinality(tmp_path: Path, sample: Sample):
    engine, _, _ = _engine(tmp_path, sample)
    real = engine.action_set(sample)
    dummy = engine.action_set(sample, dummy=True)
    assert len(real) == 11 and len(dummy) == 11
    assert [a.type for a in real].count("CROP") == 8
    assert real[0].type == "STOP" and dummy[0].type == "NOOP"
    # the luck floor needs 11 distinct draws, not repeats
    assert len({a.key for a in dummy}) == 11


def test_campaign_writes_complete_records(tmp_path: Path, sample: Sample):
    engine, vlm, _ = _engine(tmp_path, sample)
    log_path = tmp_path / "rollouts.jsonl"
    with RolloutLogger(log_path, "test", {"note": "unit"}) as lg:
        n = run_campaign(engine, [sample], lg, replays=(1, 2))
    assert n == 24  # (11 candidate actions + 1 FULL reference) x 2 replays
    recs = read_rollouts(log_path)
    assert len(recs) == 24
    assert sum(r["action"]["type"] == "FULL" for r in recs) == 2
    for r in recs:
        assert r["state_id"] == "s0" and r["raw_generation"]
        assert r["model_id"] == "stub-vlm" and r["prompt_version"]
        assert "visual_tokens" in r["cost"] and r["decode"]["temperature"] == 0.0
    # every non-STOP action carries the STOP result of its own replay
    by_replay = {}
    for r in recs:
        if r["action"]["type"] == "STOP":
            by_replay[r["replay"]] = r["score_after"]
    for r in recs:
        if r["action"]["type"] != "STOP":
            assert r["score_before"] == by_replay[r["replay"]]


def test_cache_prevents_repeat_generations(tmp_path: Path, sample: Sample):
    engine, vlm, cache = _engine(tmp_path, sample)
    with RolloutLogger(tmp_path / "a.jsonl", "t1") as lg:
        run_campaign(engine, [sample], lg, replays=(1,))
    first = vlm.calls
    engine._proposal_memo.clear()
    with RolloutLogger(tmp_path / "b.jsonl", "t2") as lg:
        run_campaign(engine, [sample], lg, replays=(1,))
    assert vlm.calls == first  # second pass fully served from cache
    assert cache.hits > 0


def test_cache_key_tracks_prompt_version(tmp_path: Path, sample: Sample):
    engine, vlm, _ = _engine(tmp_path, sample)
    engine.run_action(sample, ActionSpec("STOP"), 1)
    calls = vlm.calls
    engine.prompt_version = "p0.2"
    engine.run_action(sample, ActionSpec("STOP"), 1)
    assert vlm.calls == calls + 1  # a prompt edit must never reuse old answers


def test_manifest_roundtrip_and_draw(tmp_path: Path, sample: Sample):
    samples = [Sample(**{**sample.__dict__, "sample_id": f"synthetic/{i:04d}"})
               for i in range(20)]
    p = tmp_path / "m.jsonl"
    assert write_manifest(p, samples) == 20
    back = load_manifest(p)
    assert [s.sample_id for s in back] == [s.sample_id for s in samples]
    a = draw(back, 5, seed=13)
    b = draw(back, 5, seed=13)
    assert [s.sample_id for s in a] == [s.sample_id for s in b] and len(a) == 5


def test_stratified_draw_balances_strata(sample: Sample):
    """One domain must not win the draw by luck (Gate 0 asks about regimes)."""
    from ppu.datasets import draw_stratified

    samples = []
    for cat, k in (("remote_sensing", 60), ("ocr", 40), ("diagram", 8)):
        for i in range(k):
            samples.append(Sample(**{**sample.__dict__,
                                     "sample_id": f"mme/{cat}/{i:03d}",
                                     "meta": {"category": cat}}))
    got = draw_stratified(samples, 30, seed=7, key="category")
    counts = {}
    for s in got:
        counts[s.meta["category"]] = counts.get(s.meta["category"], 0) + 1
    assert len(got) == 30
    assert counts["diagram"] == 8  # small stratum gives everything it has
    # its shortfall is redistributed, not dropped
    assert counts["remote_sensing"] + counts["ocr"] == 22
    assert abs(counts["remote_sensing"] - counts["ocr"]) <= 1
    assert [s.sample_id for s in draw_stratified(samples, 30, seed=7, key="category")] \
        == [s.sample_id for s in got]


def test_concurrent_rollouts_match_serial_order(tmp_path: Path, sample: Sample):
    """Concurrency is a speed knob; it must not change what is logged."""
    serial_engine, _, _ = _engine(tmp_path / "a", sample)
    par_engine, _, _ = _engine(tmp_path / "b", sample)
    par_engine.concurrency = 4
    serial = serial_engine.run_sample(sample, replays=(1,))
    par = par_engine.run_sample(sample, replays=(1,))
    assert [r.action["key"] for r in serial] == [r.action["key"] for r in par]
    assert [r.answer_after for r in serial] == [r.answer_after for r in par]
    assert [r.score_before for r in serial] == [r.score_before for r in par]


def test_open_ended_replay_two_is_actually_served(tmp_path: Path, sample: Sample):
    """The R2 paraphrase must reach the model, not just the ServedQuestion.

    Regression: render_question ignored the replay text for open-ended sets, so
    R2 was a byte-identical re-run of R1 and the strict label had no control.
    """
    engine, _, _ = _engine(tmp_path, sample)
    r1 = engine.run_action(sample, ActionSpec("STOP"), 1)
    r2 = engine.run_action(sample, ActionSpec("STOP"), 2)
    assert r1.question_served != r2.question_served
    assert sample.question in r1.question_served


def test_mcq_replay_rendering_unchanged_by_the_fix(tmp_path: Path, mcq_sample: Sample):
    """MCQ carries its replay in the option order; the text stays the original."""
    engine, _, _ = _engine(tmp_path, mcq_sample)
    r1 = engine.run_action(mcq_sample, ActionSpec("STOP"), 1)
    r2 = engine.run_action(mcq_sample, ActionSpec("STOP"), 2)
    assert r1.question_served.splitlines()[0] == mcq_sample.question
    assert r2.question_served.splitlines()[0] == mcq_sample.question
    assert r1.question_served != r2.question_served  # options permuted


def test_mixed_coordinate_proposals_are_recovered():
    """Qwen2.5-VL emits normalized and pixel coords in the same line."""
    boxes, trace = regions.parse_proposals_mixed(
        "0.25,147,0.38,236 pedestrian\n0.52,0.14,0.81,0.67\n", 980, 490)
    assert boxes[0] == (0.25, 0.3, 0.38, 0.4816)  # y divided by view height
    assert boxes[1] == (0.52, 0.14, 0.81, 0.67)   # already normalized
    assert trace[0]["modes"] == ["norm", "pixel", "norm", "pixel"]


def test_mixed_parser_rejects_bad_boxes():
    boxes, trace = regions.parse_proposals_mixed("0.0,0.0,1.0,1.0", 100, 100)
    assert boxes == [] and trace[0]["dropped"] == "near_full_image"
    boxes, _ = regions.parse_proposals_mixed("5000,5000,6000,6000", 100, 100)
    assert boxes == []  # out of range after normalization


def test_size_matched_random_preserves_shape():
    src = [(0.1, 0.1, 0.3, 0.5), (0.4, 0.4, 0.45, 0.45)]
    got = regions.size_matched_random("s1", src)
    for a, b in zip(src, got):
        assert abs((a[2] - a[0]) - (b[2] - b[0])) < 1e-6
        assert abs((a[3] - a[1]) - (b[3] - b[1])) < 1e-6
    assert got == regions.size_matched_random("s1", src)  # deterministic
