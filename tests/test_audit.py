"""Audit tests: the controls must actually control.

These are the tests that matter most in the whole repo. If the cross-fitted
oracle or the luck floor silently mis-computes, Gate 1 returns a confident
number that is wrong, and the project turns on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppu.audit import (SampleView, audit, best_action_family, best_fixed_cf,
                       by_dataset, load, oracle_cf, oracle_naive, verdict)


def _rec(sid, dataset, replay, action_type, key, score, dummy=False, family=None):
    return {
        "type": "rollout", "sample_id": sid, "dataset": dataset, "replay": replay,
        "dummy": dummy, "parse_ok": True, "score_after": score,
        "action": {"type": action_type, "family": family or action_type,
                   "key": key, "slot": 0, "box": None, "source": "none"},
    }


def _view(sid="s1", dataset="d", r1=None, r2=None, dummy_r1=None, dummy_r2=None):
    v = SampleView(sid, dataset)
    for k, s in (r1 or {}).items():
        v.r1[k] = s
        v.family[k] = k.split("#")[0].replace("CROP_RANDOM", "CROP")
    for k, s in (r2 or {}).items():
        v.r2[k] = s
        v.family.setdefault(k, k.split("#")[0])
    for k, s in (dummy_r1 or {}).items():
        v.dummy_r1[k] = s
        v.dummy_family[k] = "CROP" if k.startswith("CROP") else "STOP"
    for k, s in (dummy_r2 or {}).items():
        v.dummy_r2[k] = s
    return v


def test_cross_fitted_oracle_discounts_lucky_hits():
    """An action that only looks good on R1 must not be credited on R2."""
    v = _view(r1={"STOP#0": 0.0, "CROP#1#a": 1.0, "OCR#0": 0.0},
              r2={"STOP#0": 0.0, "CROP#1#a": 0.0, "OCR#0": 0.0})
    assert oracle_naive(v) == 1.0
    assert oracle_cf(v) == 0.0  # chosen on R1, scored on R2


def test_cost_tiebreak_prefers_the_cheaper_action():
    v = _view(r1={"STOP#0": 1.0, "HIGHRES#0": 1.0},
              r2={"STOP#0": 1.0, "HIGHRES#0": 1.0})
    assert best_action_family(v) is None or best_action_family(v) == "STOP"
    v2 = _view(r1={"STOP#0": 0.0, "OCR#0": 1.0, "HIGHRES#0": 1.0},
               r2={"STOP#0": 0.0, "OCR#0": 1.0, "HIGHRES#0": 1.0})
    assert best_action_family(v2) == "OCR"


def test_luck_floor_is_subtracted():
    """11 uninformative draws raise a max too; H must be net of that."""
    views = {}
    for i in range(20):
        lucky_real = 1.0 if i % 2 == 0 else 0.0
        lucky_dummy = 1.0 if i % 2 == 0 else 0.0  # dummies are just as 'lucky'
        views[f"s{i}"] = _view(
            f"s{i}",
            r1={"STOP#0": 0.0, "CROP#1#a": lucky_real},
            r2={"STOP#0": 0.0, "CROP#1#a": lucky_real},
            dummy_r1={"NOOP#0": 0.0, "CROP_RANDOM#1#b": lucky_dummy},
            dummy_r2={"NOOP#0": 0.0, "CROP_RANDOM#1#b": lucky_dummy},
        )
    out = audit(views, n_boot=200)
    assert out["accuracy"]["oracle_cf"] == 0.5
    assert out["accuracy"]["oracle_dummy_cf"] == 0.5
    assert out["headroom"]["H_net"] == 0.0  # entirely explained by the floor
    assert out["headroom"]["oracle_minus_base"] == 0.5  # naive view would claim +50pp


def test_policy_headroom_is_zero_when_one_fixed_action_wins_everywhere():
    views = {}
    for i in range(20):
        views[f"s{i}"] = _view(
            f"s{i}",
            r1={"STOP#0": 0.0, "OCR#0": 1.0, "HIGHRES#0": 0.0},
            r2={"STOP#0": 0.0, "OCR#0": 1.0, "HIGHRES#0": 0.0},
        )
    out = audit(views, n_boot=200)
    assert out["accuracy"]["best_fixed_cf"] == 1.0
    assert abs(out["headroom"]["P_policy"]) < 1e-9  # nothing to learn
    assert out["headroom"]["H_net"] == 1.0  # but the actions do help


def test_policy_headroom_appears_when_the_winner_alternates():
    views = {}
    for i in range(20):
        ocr_wins = i % 2 == 0
        views[f"s{i}"] = _view(
            f"s{i}",
            r1={"STOP#0": 0.0, "OCR#0": float(ocr_wins),
                "HIGHRES#0": float(not ocr_wins)},
            r2={"STOP#0": 0.0, "OCR#0": float(ocr_wins),
                "HIGHRES#0": float(not ocr_wins)},
        )
    out = audit(views, n_boot=200)
    # No fixed policy can beat ~50% here. Cross-fitting lands slightly under it
    # because the two folds are complementary: whichever action wins the
    # training half is, by construction, the minority one in the test half.
    assert 0.35 <= out["accuracy"]["best_fixed_cf"] <= 0.55
    assert out["headroom"]["P_policy"] >= 0.45  # the whole gain is state-dependent
    d = out["diversity"]["distribution"]
    assert d["OCR"] == 0.5 and d["HIGHRES"] == 0.5


def test_exclusive_solve_and_harm():
    views = {
        # only OCR recovers it
        "s0": _view("s0", r1={"STOP#0": 0.0, "OCR#0": 1.0, "HIGHRES#0": 0.0,
                              "CROP#0#a": 0.0},
                    r2={"STOP#0": 0.0, "OCR#0": 1.0, "HIGHRES#0": 0.0,
                        "CROP#0#a": 0.0}),
        # STOP was right, HIGHRES breaks it
        "s1": _view("s1", r1={"STOP#0": 1.0, "OCR#0": 1.0, "HIGHRES#0": 0.0,
                              "CROP#0#a": 1.0},
                    r2={"STOP#0": 1.0, "OCR#0": 1.0, "HIGHRES#0": 0.0,
                        "CROP#0#a": 1.0}),
    }
    out = audit(views, n_boot=200)
    assert out["exclusive_solve"]["OCR"] == 0.5
    assert out["harm"]["dmg_rate"] == 0.5
    assert out["harm"]["by_family"]["HIGHRES"] == 1.0
    assert out["harm"]["by_family"]["OCR"] == 0.0
    assert out["recovery_overlap"]["OCR"] == 1


def test_strict_labels_require_both_replays():
    v = _view(r1={"STOP#0": 0.0, "OCR#0": 1.0}, r2={"STOP#0": 0.0, "OCR#0": 0.0})
    assert v.strict("OCR#0") == 0
    assert v.solved_families() == set()


def test_verdict_thresholds():
    strong = {"headroom": {"H_net": 0.12, "H_ci": [0.07, 0.17],
                           "P_policy": 0.08, "P_ci": [0.03, 0.13]},
              "harm": {"dmg_rate": 0.10}}
    per_ds = [{"diversity": {"max_share": 0.4, "n_families_over_10pct": 3}}] * 3
    assert verdict(strong, per_ds)["verdict"] == "STRONG-GO"

    weak = {"headroom": {"H_net": 0.02, "H_ci": [0.0, 0.05],
                         "P_policy": 0.01, "P_ci": [-0.01, 0.03]},
            "harm": {"dmg_rate": 0.02}}
    flat = [{"diversity": {"max_share": 0.9, "n_families_over_10pct": 1}}] * 3
    assert verdict(weak, flat)["verdict"] == "STOP"

    crop_only = {"headroom": {"H_net": 0.09, "H_ci": [0.05, 0.13],
                              "P_policy": 0.01, "P_ci": [-0.01, 0.03]},
                 "harm": {"dmg_rate": 0.05}}
    # "only CROP is ever useful" must not pass, however large the oracle gain
    assert verdict(crop_only, flat)["verdict"] in ("CONDITIONAL", "STOP")


def test_load_from_records_and_group_by_dataset():
    recs = []
    for ds in ("textvqa", "gqa"):
        for i in range(4):
            sid = f"{ds}/{i}"
            for r in (1, 2):
                recs += [
                    _rec(sid, ds, r, "STOP", "STOP#0", 0.0),
                    _rec(sid, ds, r, "OCR", "OCR#0", 1.0 if ds == "textvqa" else 0.0),
                    _rec(sid, ds, r, "HIGHRES", "HIGHRES#0",
                         0.0 if ds == "textvqa" else 1.0),
                    _rec(sid, ds, r, "FULL", "FULL#0", 1.0),
                ]
    views = load(recs)
    assert len(views) == 8
    per_ds = by_dataset(views, n_boot=100)
    assert [d["label"] for d in per_ds] == ["gqa", "textvqa"]
    # FULL is a reference row, never a candidate: it must not enter the oracle
    for d in per_ds:
        assert d["accuracy"]["full_image_r2"] == 1.0
        assert d["accuracy"]["oracle_cf"] == 1.0
        assert d["diversity"]["counts"]["STOP"] == 0


def test_bottleneck_subset_isolates_budget_limited_samples():
    """coarse wrong + full right = the budget was the binding constraint."""
    from ppu.audit import bottleneck

    def with_full(v, f1, f2):
        v.full_r1, v.full_r2 = f1, f2
        return v

    views = {
        # budget-limited, only OCR recovers it
        "s0": with_full(_view("s0", r1={"STOP#0": 0.0, "OCR#0": 1.0, "HIGHRES#0": 0.0},
                              r2={"STOP#0": 0.0, "OCR#0": 1.0, "HIGHRES#0": 0.0}), 1.0, 1.0),
        # budget-limited, only HIGHRES recovers it
        "s1": with_full(_view("s1", r1={"STOP#0": 0.0, "OCR#0": 0.0, "HIGHRES#0": 1.0},
                              r2={"STOP#0": 0.0, "OCR#0": 0.0, "HIGHRES#0": 1.0}), 1.0, 1.0),
        # hopeless at any budget -> must be excluded from the subset
        "s2": with_full(_view("s2", r1={"STOP#0": 0.0, "OCR#0": 0.0, "HIGHRES#0": 0.0},
                              r2={"STOP#0": 0.0, "OCR#0": 0.0, "HIGHRES#0": 0.0}), 0.0, 0.0),
        # already right on coarse -> not a bottleneck sample
        "s3": with_full(_view("s3", r1={"STOP#0": 1.0, "OCR#0": 1.0, "HIGHRES#0": 1.0},
                              r2={"STOP#0": 1.0, "OCR#0": 1.0, "HIGHRES#0": 1.0}), 1.0, 1.0),
    }
    b = bottleneck(views)
    assert b["n"] == 2 and b["recovery_rate"] == 1.0
    assert b["exclusive_solve"]["OCR"] == 0.5
    assert b["exclusive_solve"]["HIGHRES"] == 0.5
    assert b["best_family"]["OCR"] == 0.5


def test_bottleneck_needs_full_rows():
    from ppu.audit import bottleneck

    assert bottleneck({"s0": _view("s0", r1={"STOP#0": 0.0}, r2={"STOP#0": 0.0})})["n"] == 0
