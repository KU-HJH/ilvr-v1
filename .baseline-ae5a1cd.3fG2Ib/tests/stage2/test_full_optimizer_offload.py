"""Full-only opt-in must not weaken existing GPU/PEFT offload boundaries."""
import json
import ast
import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import deepspeed_runtime as runtime


NAME = "deepspeed_zero3_stage2_full_optimizer_offload.json"


def test_direct_main_preflight_preserves_explicit_full_scope(offload_config):
    path, _ = offload_config
    source = Path(runtime.__file__).with_name("main.py")
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_preflight_direct_main")
    namespace = dict(argparse=argparse, Path=Path,
                     validate_generated_artifact_paths=lambda args: None,
                     bind_deepspeed_runtime_profile=runtime.bind_deepspeed_runtime_profile)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    profile = namespace["_preflight_direct_main"]([
        "--stage", "stage2", "--tuning_method", "full",
        "--deepspeed_config", str(path), "--log_file", "logs/test/train.log"])
    assert profile.zero_stage == 3


@pytest.fixture
def offload_config(tmp_path, monkeypatch):
    canonical = tmp_path / "deepspeed_zero3.json"
    payload = json.loads(runtime.canonical_deepspeed_config_path(3).read_text())
    payload["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": False}
    path = tmp_path / NAME
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(runtime, "canonical_deepspeed_config_path", lambda stage: canonical)
    return path, payload


def test_dedicated_full_config_loads(offload_config):
    path, _ = offload_config
    profile = runtime.load_deepspeed_runtime_profile(path, world_size=2)
    assert profile.enabled and profile.zero_stage == 3 and profile.config_path == path


def test_cpu_optimizer_does_not_leak_to_other_paths(offload_config):
    path, payload = offload_config
    other = path.with_name("deepspeed_zero3.json")
    other.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported for canonical GPU runs"):
        runtime.load_deepspeed_runtime_profile(other, world_size=2)


@pytest.mark.parametrize("field,value", [
    ("stage", 2), ("offload_param", {"device": "cpu"}),
    ("offload_optimizer", {"device": "nvme"}),
])
def test_full_policy_rejects_other_offloads(offload_config, field, value):
    path, payload = offload_config
    payload["zero_optimization"][field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Full ZeRO-3 optimizer offload"):
        runtime.load_deepspeed_runtime_profile(path, world_size=2)


@pytest.mark.parametrize("optimizer", [{"type": "AdamW"}, {"type": "Adam"},
    {"type": "AdamW", "params": {"torch_adam": True}}])
def test_full_offload_preserves_trainer_optimizer_groups(offload_config, optimizer):
    path, payload = offload_config
    payload["optimizer"] = optimizer
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Full ZeRO-3 optimizer offload"):
        runtime.load_deepspeed_runtime_profile(path, world_size=2)


@pytest.mark.parametrize("stage,method", [("stage1", "full"), ("stage2", "dora"),
    ("stage2", "lora"), ("stage2", None)])
def test_binding_rejects_non_full_consumers(offload_config, stage, method):
    path, _ = offload_config
    args = SimpleNamespace(stage=stage, tuning_method=method, deepspeed_config=str(path))
    with pytest.raises(ValueError, match="requires explicit Stage 2 full tuning"):
        runtime.bind_deepspeed_runtime_profile(args, world_size=2)


def test_binding_accepts_full_and_rechecks_cached_scope(offload_config):
    path, _ = offload_config
    args = SimpleNamespace(stage="stage2", tuning_method="full", deepspeed_config=str(path))
    profile = runtime.bind_deepspeed_runtime_profile(args, world_size=2)
    assert profile.zero_stage == 3
    args.tuning_method = "dora"
    with pytest.raises(ValueError, match="requires explicit Stage 2 full tuning"):
        runtime.bind_deepspeed_runtime_profile(args, world_size=2)


def test_shipped_config_selects_cpu_optimizer_without_parameter_offload():
    path = runtime.canonical_deepspeed_config_path(3).with_name(NAME)
    assert path.is_file(), "approved Full optimizer-offload configuration is absent"
    profile = runtime.load_deepspeed_runtime_profile(path, world_size=2)
    payload = json.loads(path.read_text())
    assert profile.zero_stage == 3
    assert payload["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert payload["zero_optimization"]["offload_param"]["device"] == "none"
    assert "optimizer" not in payload, "DS-owned optimizer bypasses Trainer no-decay groups"


def test_full_offload_requires_cpu_optimizer_mapping(offload_config):
    path, payload = offload_config
    payload["zero_force_ds_cpu_optimizer"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Full ZeRO-3 optimizer offload"):
        runtime.load_deepspeed_runtime_profile(path, world_size=2)


def test_installed_cpu_adam_mapping_preserves_no_decay_group():
    import torch
    from accelerate.utils.deepspeed import map_pytorch_optim_to_deepspeed

    weight = torch.nn.Parameter(torch.ones(2))
    bias = torch.nn.Parameter(torch.ones(2))
    original = torch.optim.AdamW([
        {"params": [weight], "weight_decay": 0.01},
        {"params": [bias], "weight_decay": 0.0}], lr=1e-3)
    mapped = map_pytorch_optim_to_deepspeed(original)
    assert type(mapped).__name__ == "DeepSpeedCPUAdam"
    assert mapped.adam_w_mode is True
    assert [group["weight_decay"] for group in mapped.param_groups] == [0.01, 0.0]
    assert mapped.param_groups[0]["params"][0] is weight
    assert mapped.param_groups[1]["params"][0] is bias
    weight.grad = torch.zeros_like(weight)
    bias.grad = torch.zeros_like(bias)
    mapped.step()
    torch.testing.assert_close(weight, torch.full_like(weight, 1.0 - 1e-3 * 0.01))
    torch.testing.assert_close(bias, torch.ones_like(bias), rtol=0, atol=0)
