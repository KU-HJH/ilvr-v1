"""Contracts for the bounded test-only premise witness and tensor reference."""
from importlib import import_module, util
from copy import deepcopy

import pytest
import torch


def probe_api():
    name = "tests.stage1.vision_gain_zero_probe"
    assert util.find_spec(name) is not None, "missing bounded vision-gain runtime probe API"
    return import_module(name)


def records():
    return [dict(schema=1, kind="stage1-vision-gain-runtime-probe", commit="a" * 40,
                 packages={"torch": "2.7.1+cu128", "deepspeed": "0.18.7", "accelerate": "1.10.1", "trl": "0.15.2"},
                 global_rank=rank, world_size=4, zero_stage=1, steps=2, alpha=0.001,
                 aux_grad_isolated=True, moments_base_only=True, master_delta_matches=True,
                 second_step_retains_aux=True, reload_retains_aux=True,
                 aux_movement_max=0.002, master_delta_residual=0.0,
                 moment_residual=0.0, reload_residual=0.0, sync_residual=0.0,
                 peak_memory_bytes=1024, valid_blocks_per_microbatch=rank,
                 microbatches_per_step=[4, 2]) for rank in range(4)]


def test_probe_rejects_missing_runtime_evidence():
    api = probe_api()
    assert api.validate_rank_records(records(), commit="a" * 40, zero_stage=1)["steps"] == 2
    corruptions = [lambda r: r[0].pop("aux_grad_isolated"),
                   lambda r: r[1].update(moments_base_only=False),
                   lambda r: r[2].update(master_delta_residual=float("nan")),
                   lambda r: r[3].update(commit="b" * 40),
                   lambda r: r[3].update(global_rank=2),
                   lambda r: r[2].update(aux_movement_max=0.0),
                   lambda r: r[2].update(master_delta_residual=0.1)]
    for corrupt in corruptions:
        changed = deepcopy(records()); corrupt(changed)
        with pytest.raises(ValueError):
            api.validate_rank_records(changed, commit="a" * 40, zero_stage=1)


def test_probe_projection_zero_blocks_and_overflow_skip():
    api = probe_api()
    g, h = torch.tensor([1., -2.]), torch.tensor([0., 1.])
    torch.testing.assert_close(api.project_direction(g, h, 1), torch.tensor([1., 0.]), rtol=1e-5, atol=1e-7)
    assert api.project_direction(g, h, 0) is None
    assert api.displacement_for_step(g, applied=False) is None
    with pytest.raises(FloatingPointError):
        api.project_direction(torch.tensor([float("nan")]), torch.ones(1), 1)


def test_probe_manual_adamw_reference_and_nonzero_auxiliary_displacement():
    api = probe_api()
    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.3]))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    for gradient in [torch.tensor([0.4, -0.2]), torch.tensor([-0.1, 0.7])]:
        old = parameter.detach().clone()
        state = optimizer.state[parameter]
        expected, m, v = api.adamw_reference(old, gradient, state.get("exp_avg", torch.zeros(2)),
                                             state.get("exp_avg_sq", torch.zeros(2)), len(state) and int(state["step"]) or 0,
                                             optimizer.param_groups[0])
        parameter.grad = gradient; optimizer.step()
        torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(state["exp_avg"], m, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(state["exp_avg_sq"], v, rtol=1e-6, atol=1e-7)
    moved = parameter.detach().clone() + 0.001 * torch.tensor([1., -2.])
    assert (moved - parameter).abs().max() > 100 * torch.finfo(torch.float32).eps


def test_tiny_probe_actual_score_gradients_leave_previous_base_gradient_untouched():
    api = probe_api()
    model, batch, targets = api.cpu_score_fixture()
    from stage1.equivalence import stage1_checkpoint_context_fn
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        "use_reentrant": False, "context_fn": stage1_checkpoint_context_fn})
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    # A previous microbatch's real base backward must survive auxiliary differentiation.
    with api.activate_stage1_stochastic_context(api._context(model, batch, 0)):
        model(**api.score_inputs(model, batch, targets, masked=False)).logits.float().square().mean().backward()
    before = [None if p.grad is None else p.grad.clone() for p in parameters]
    assert any(value is not None and value.abs().sum() > 0 for value in before)
    clean = api.score_gradients(model, batch, targets, block_count=1, masked=False, optimizer_step=0)
    masked = api.score_gradients(model, batch, targets, block_count=1, masked=True, optimizer_step=0)
    assert any((a - b).abs().max() > 1e-7 for a, b in zip(clean, masked))
    for p, old in zip(parameters, before):
        if old is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, old, rtol=0, atol=0)


def test_probe_training_arguments_pin_frozen_teacher_and_common_profile(tmp_path):
    api = probe_api()
    from argparse import Namespace
    from types import SimpleNamespace
    resolver = getattr(api, "probe_training_args", None)
    assert callable(resolver), "missing consumed probe training-profile resolver"
    args = Namespace(phase="initial", deepspeed=tmp_path / "ds.json")
    parsed = resolver(args, SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"), tmp_path / "data")
    assert (parsed.per_device_train_batch_size, parsed.gradient_accumulation_steps,
            parsed.max_steps, parsed.learning_rate, parsed.warm_up_steps, parsed.seed,
            parsed.ce_weight, parsed.ema_tau) == (1, 4, 2, 5e-5, 10, 42, 0.1, 1.0)
    assert parsed.gradient_checkpointing is True
    assert (parsed.tuning_method, parsed.lora_r, parsed.lora_alpha, parsed.lora_dropout,
            parsed.latent_size, parsed.attention_implementation) == ("dora", 32, 64, 0.05, 8, "sdpa")
    resumed = resolver(Namespace(phase="resume", deepspeed=args.deepspeed),
                       SimpleNamespace(checkpoint_dir=tmp_path / "checkpoints"), tmp_path / "data")
    assert resumed.max_steps == parsed.max_steps == 2
    assert resumed.resume_from_checkpoint_path == str(tmp_path / "checkpoints/checkpoint-1")
