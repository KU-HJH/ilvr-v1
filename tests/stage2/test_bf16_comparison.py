"""Approved CUDA likelihood budget; strict CPU equivalence remains unchanged."""
import pytest

from . import packed_m_helpers


def compare(full, packed):
    implementation = getattr(packed_m_helpers, "compare_cuda_bf16_likelihoods", None)
    assert callable(implementation), "CUDA BF16 likelihood budget is not implemented"
    return implementation(full, packed)


def test_inclusive_absolute_limits_allow_boundary_without_relative_tolerance():
    result = compare([0.0, 0.0], [-0.10, 0.0])
    assert result["passed"]
    assert result["block_abs_error"] == 0.05
    assert result["token_max_abs_error"] == 0.10


def test_individual_token_failure_is_not_hidden_by_cancelling_mean():
    result = compare([-1.0, -1.0], [-1.11, -0.89])
    assert not result["passed"]
    assert result["block_passed"]
    assert not result["tokens_passed"]
    assert result["failed_token_indices"] == [0, 1]


def test_block_failure_is_not_hidden_by_each_token_passing():
    result = compare([-1.0, -1.0], [-1.06, -1.06])
    assert not result["passed"]
    assert not result["block_passed"]
    assert result["tokens_passed"]


def test_large_likelihood_magnitude_does_not_increase_absolute_budget():
    assert not compare([-100.0], [-100.06])["passed"]


@pytest.mark.parametrize("full,packed", [([], []), ([-1.0], [-1.0, -1.0]),
                                        ([-1.0], [float("nan")]),
                                        ([float("inf")], [-1.0])])
def test_missing_misaligned_or_nonfinite_values_fail_closed(full, packed):
    with pytest.raises(ValueError):
        compare(full, packed)
