import pytest

from evaluation.schema import ContractError, RunMetadata, canonical_json, run_fingerprint


def test_fingerprint_ignores_mapping_order_but_changes_with_generation():
    a = RunMetadata("synthetic:model", "synthetic:template", {"temperature": 0, "max_new_tokens": 8})
    b = RunMetadata("synthetic:model", "synthetic:template", {"max_new_tokens": 8, "temperature": 0})
    c = RunMetadata("synthetic:model", "synthetic:template", {"max_new_tokens": 9, "temperature": 0})
    assert run_fingerprint(a) == run_fingerprint(b)
    assert run_fingerprint(a) != run_fingerprint(c)
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_identity_is_rejected(value):
    with pytest.raises(ContractError):
        canonical_json({"temperature": value})


def test_non_string_canonical_json_keys_are_rejected():
    with pytest.raises(ContractError, match="keys must be strings"):
        canonical_json({1: "value"})
