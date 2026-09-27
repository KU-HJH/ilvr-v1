"""Pre-registered Vision-margin verdict rules and the exact one-sided McNemar test."""

import math

import pytest

from stage2.vision_margin_decision import decide_arm, mcnemar_one_sided_decrease


def test_mcnemar_exact_one_sided_decrease_matches_binomial_tail():
    """Catches a two-sided, normal-approximation, or reversed-direction test."""
    ce = [True] * 10 + [False] * 5 + [True] * 5
    arm = [False] * 10 + [True] * 5 + [True] * 5  # b=10 CE-only correct, c=5 arm-only correct
    result = mcnemar_one_sided_decrease(ce, arm)
    expected = sum(math.comb(15, k) for k in range(10, 16)) / 2 ** 15
    assert (result["b"], result["c"]) == (10, 5)
    assert result["p_value"] == pytest.approx(expected)
    assert mcnemar_one_sided_decrease(ce, ce)["p_value"] == 1.0


def _probe(sf, sm, gt0):
    return {"S_F": sf, "S_M": sm, "G_mean": sf - sm, "G_gt_0": gt0}


def test_decide_arm_applies_all_thresholds_against_step0_and_ce():
    """Catches any threshold, delta base, or combination rule drifting from pre-registration."""
    step0, ce = _probe(-1.20, -1.20, 0.50), _probe(-1.17, -1.17, 0.50)
    passing = decide_arm(_probe(-1.10, -1.18, 0.65), step0=step0, ce=ce,
                         accuracy=0.52, p_value=0.4)
    assert {k: v["verdict"] for k, v in passing["criteria"].items()} == {
        "a": "PASS", "b": "PASS", "c": "PASS", "d": "PASS"}
    assert passing["mechanism"] == "PASS" and passing["no_regression"] == "PASS"

    flat_sf = decide_arm(_probe(-1.19, -1.27, 0.65), step0=step0, ce=ce,
                         accuracy=0.50, p_value=0.2)
    assert flat_sf["criteria"]["a"]["verdict"] == "FAIL"   # dS_F=+0.01 < 0.05
    assert flat_sf["criteria"]["b"]["verdict"] == "FAIL"   # dS_M=-0.07 ok, but dS_F < 0.4*dG
    assert flat_sf["criteria"]["d"]["verdict"] == "FAIL"   # accuracy below 0.51
    assert flat_sf["mechanism"] == "FAIL"

    missing_eval = decide_arm(_probe(-1.10, -1.18, 0.65), step0=step0, ce=ce,
                              accuracy=None, p_value=None)
    assert missing_eval["criteria"]["d"]["verdict"] == "undecided"
    assert missing_eval["no_regression"] == "undecided"


def test_criterion_a_uses_the_005_nat_floor_independently_of_ce():
    """Catches the absolute +0.05 nat S_F floor being weakened or dropped."""
    step0 = ce = _probe(-1.20, -1.20, 0.50)
    below = decide_arm(_probe(-1.16, -1.20, 0.65), step0=step0, ce=ce, accuracy=0.52, p_value=0.5)
    above = decide_arm(_probe(-1.14, -1.20, 0.65), step0=step0, ce=ce, accuracy=0.52, p_value=0.5)
    assert below["criteria"]["a"]["verdict"] == "FAIL"   # dS_F=+0.04
    assert above["criteria"]["a"]["verdict"] == "PASS"   # dS_F=+0.06
