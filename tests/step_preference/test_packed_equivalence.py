import math
from collections import UserDict
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from step_preference.equivalence import (
    EquivalenceResult,
    _to_device,
    compare_gradients,
    compare_scalar_metrics,
    relative_l2_error,
    run_tiny_fp32_equivalence,
    run_tiny_multistep_fp32_equivalence,
    write_evidence_exclusive,
)
from scripts.verify_packed_step_preference import _diagnostic_section, _section


@pytest.mark.parametrize("packing,cap",[(False,None),(True,4608)])
def test_tiny_equivalence_constructs_valid_config_with_exact_packed_cap(packing,cap):
    """Catches equivalence rejecting its packed config before comparing either graph."""
    from step_preference.equivalence import _config
    config=_config(packing=packing)
    assert config.packing_enabled is packing
    assert config.max_packed_sequence_length == cap


def test_relative_l2_error_uses_float64_reference_norm():
    """Catches elementwise or low-precision error replacing the required L2 ratio."""
    reference = torch.tensor([3.0, 4.0], dtype=torch.float16)
    candidate = torch.tensor([6.0, 8.0], dtype=torch.float16)
    assert relative_l2_error(reference, candidate) == pytest.approx(1.0)


def test_comparisons_fail_closed_on_missing_nonfinite_and_zero_reference_values():
    """Catches absent/nonfinite gradients or zero-reference drift being accepted."""
    scalar = compare_scalar_metrics(
        {"loss/total": 1.0, "loss/ce": 2.0},
        {"loss/total": float("nan")},
        epsilon=1e-5,
    )
    assert scalar.passed is False
    assert set(scalar.errors) == {"loss/total", "loss/ce"}

    gradients = compare_gradients(
        {
            "zero": torch.zeros(2),
            "missing": torch.ones(1),
            "nonfinite": torch.ones(1),
        },
        {
            "zero": torch.tensor([0.0, 2e-4]),
            "nonfinite": torch.tensor([float("inf")]),
        },
        epsilon=1e-4,
    )
    assert gradients.passed is False
    assert set(gradients.errors) == {"zero", "missing", "nonfinite"}
    assert math.isinf(gradients.relative_l2["zero"])


def test_evidence_writer_refuses_to_overwrite(tmp_path):
    """Catches a retry replacing the identity-bound runtime witness."""
    path = tmp_path / "equivalence.json"
    write_evidence_exclusive(path, {"schema_version": 1, "status": "pass"})
    with pytest.raises(ValueError, match="already exists"):
        write_evidence_exclusive(path, {"schema_version": 1, "status": "fail"})


def test_device_transfer_handles_mapping_wrappers_from_the_real_collator():
    """Catches BatchFeature-like mappings leaving tensors on the CPU."""
    wrapped = UserDict({"input_ids": torch.tensor([[1, 2]])})
    moved = _to_device(wrapped, torch.device("meta"))
    assert isinstance(moved, dict)
    assert moved["input_ids"].device.type == "meta"


def test_equivalence_cli_refuses_existing_output_before_loading(tmp_path):
    """Catches the production CLI loading artifacts before enforcing no-overwrite."""
    output = tmp_path / "existing.json"
    output.write_text("preserve me\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts/verify_packed_step_preference.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint_path", "missing",
            "--data_path", "missing",
            "--image_root", "missing",
            "--negative_path", "missing",
            "--negative_manifest", "missing",
            "--device", "cuda:0",
            "--output_path", str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "already exists" in completed.stderr
    assert output.read_text(encoding="utf-8") == "preserve me\n"


def test_production_evidence_uses_declared_projection_names():
    """Catches production gradient evidence shortening the schema's parameter keys."""
    result = EquivalenceResult(
        status="pass",
        loss_errors={},
        gradients={
            "model.layers.0.self_attn.q_proj.weight": 0.1,
            "model.layers.0.self_attn.o_proj.weight": 0.2,
            "model.layers.0.mlp.down_proj.weight": 0.3,
        },
        trainable_parameters=(),
        epsilon_loss=1e-3,
        epsilon_grad=1e-2,
        errors=(),
    )
    assert set(_section(result, production=True)["gradient_relative_l2"]) == {
        "q_proj",
        "o_proj",
        "down_proj",
    }


def test_bf16_diagnostic_evidence_has_no_fitted_acceptance_threshold():
    """Catches observed BF16 drift being relabeled as a passing equivalence gate."""
    result = EquivalenceResult(
        status="pass",
        loss_errors={"loss/total": 0.0012},
        gradients={
            "model.layers.0.self_attn.q_proj.weight": 0.05,
            "model.layers.0.self_attn.o_proj.weight": 0.06,
            "model.layers.0.mlp.down_proj.weight": 0.04,
        },
        trainable_parameters=(),
        epsilon_loss=1.0,
        epsilon_grad=1.0,
        errors=(),
    )
    section = _diagnostic_section(result)
    assert set(section) == {"loss_errors", "gradient_relative_l2"}
    assert section["gradient_relative_l2"] == {
        "q_proj": 0.05,
        "o_proj": 0.06,
        "down_proj": 0.04,
    }


def test_tiny_fp32_naive_and_packed_match_all_trainable_gradients():
    """Catches packed loss or autograd drift against the independent naive graph."""
    assert torch.cuda.is_available(), "packed equivalence requires CUDA"
    result = run_tiny_fp32_equivalence(device="cuda:0")
    assert result.status == "pass", result.errors
    assert result.max_loss_error <= 1e-5
    assert result.max_gradient_relative_l2 <= 1e-4
    assert set(result.gradients) == set(result.trainable_parameters)


def test_multistep_ragged_fp32_tree_matches_all_trainable_gradients():
    """Catches topology drift hidden by the one-step equivalence fixture."""
    assert torch.cuda.is_available(), "packed equivalence requires CUDA"
    result = run_tiny_multistep_fp32_equivalence(device="cuda:0")
    assert result.status == "pass", result.errors
    assert result.max_loss_error <= 1e-5
    assert result.max_gradient_relative_l2 <= 1e-4
    assert set(result.gradients) == set(result.trainable_parameters)
