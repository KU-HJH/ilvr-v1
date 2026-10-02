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
                 aux_movement_max=0.002, initial_aux_movement_max=0.002, successful_mutations=2,
                 master_delta_residual=0.0,
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


@pytest.fixture
def resumed_probe(tmp_path, monkeypatch):
    """Real CPU AdamW; only distributed/master access and Trainer setup are replaced."""
    from types import SimpleNamespace
    import deepspeed.utils
    api = probe_api()
    parameter = torch.nn.Parameter(torch.tensor([0.201, -0.302]))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    trainer = api.ProbeTrainer.__new__(api.ProbeTrainer)
    trainer.model = torch.nn.ParameterList([parameter])
    zero = SimpleNamespace(optimizer=optimizer, update_lp_params=lambda: None)
    trainer.model_wrapped = SimpleNamespace(optimizer=zero)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path))
    trainer.state = SimpleNamespace(global_step=1)
    trainer.lr_scheduler = SimpleNamespace(state_dict=lambda: {"last_epoch": 1})
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.probe_phase = "resume"
    trainer.probe_base_witness = []
    trainer._probe_expected_base_partitions = []
    trainer._probe_base_entry_checked = False
    trainer._probe_verified_masters = None
    trainer.probe_stats = {key: value for key, value in records()[0].items()
                           if key in (*api.BOOLEANS, *api.RESIDUALS, "aux_movement_max",
                                      "initial_aux_movement_max", "successful_mutations")}
    trainer.probe_stats.update(second_step_retains_aux=False, reload_retains_aux=False,
                               successful_mutations=1)
    monkeypatch.setattr(api.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(api.dist, "all_reduce", lambda tensor, op: tensor.mul_(4))
    monkeypatch.setattr(deepspeed.utils, "safe_get_full_fp32_param", lambda p: p.detach().clone())
    def set_master(p, value):
        with torch.no_grad():
            p.copy_(value)
    monkeypatch.setattr(deepspeed.utils, "safe_set_full_fp32_param", set_master)
    monkeypatch.setattr(api.CustomTrainerStage1, "_install_deepspeed_step_adapter", lambda self: None)
    monkeypatch.setattr(api.CustomTrainerStage1, "_finish_optimizer_step", lambda self, model, successful: None)
    path = trainer._reference_path(); path.parent.mkdir(parents=True)
    torch.save(dict(masters=[parameter.detach().clone()], optimizer=api._snapshot(optimizer.state_dict()),
                    stats=trainer.probe_stats), path)
    trainer._install_deepspeed_step_adapter()
    trainer.probe_clean, trainer.probe_masked = torch.tensor([6., -12.]), torch.zeros(2)
    trainer.probe_count, trainer.probe_micros = 6, 2
    return api, trainer, parameter, optimizer


def test_reload_does_not_claim_unperformed_second_update(resumed_probe):
    _, trainer, _, _ = resumed_probe
    assert trainer.probe_stats["reload_retains_aux"] is True
    assert trainer.probe_stats["second_step_retains_aux"] is False


def test_second_optimizer_entry_rejects_lost_initial_displacement(resumed_probe):
    _, _, parameter, optimizer = resumed_probe
    with torch.no_grad():
        parameter.sub_(torch.tensor([0.001, -0.002]))
    parameter.grad = torch.tensor([0.4, -0.2])
    with pytest.raises(AssertionError):
        optimizer.step()


def test_successful_second_base_plus_auxiliary_result_sets_retention(resumed_probe):
    api, trainer, parameter, optimizer = resumed_probe
    old = parameter.detach().clone()
    gradient = torch.tensor([0.4, -0.2])
    expected, _, _ = api.adamw_reference(old, gradient, torch.zeros(2), torch.zeros(2),
                                        0, optimizer.param_groups[0])
    parameter.grad = gradient; optimizer.step()
    trainer._finish_optimizer_step(trainer.model_wrapped, successful=True)
    torch.testing.assert_close(parameter, expected + torch.tensor([0.001, -0.002]), rtol=1e-6, atol=1e-7)
    assert trainer.probe_stats["second_step_retains_aux"] is True
    assert trainer.probe_stats["successful_mutations"] == 2


@pytest.mark.parametrize("phase,global_step,micros", [("initial", 0, 4), ("resume", 1, 2)])
def test_skipped_base_update_cannot_complete_probe(resumed_probe, phase, global_step, micros):
    _, trainer, _, _ = resumed_probe
    trainer.probe_phase = phase
    trainer.state.global_step = global_step
    trainer.probe_micros = micros
    with pytest.raises(AssertionError, match="successful optimizer update"):
        trainer._finish_optimizer_step(trainer.model_wrapped, successful=False)


@pytest.mark.parametrize("bad", [{"successful_mutations": 1}, {"initial_aux_movement_max": 0.0}])
def test_rank_witness_requires_initial_and_resume_mutations(bad):
    api = probe_api()
    changed = records(); changed[0].update(bad)
    with pytest.raises(ValueError):
        api.validate_rank_records(changed, commit="a" * 40, zero_stage=1)


@pytest.mark.parametrize("mutation", ["hook", "reentry", "callback", "registered_callback"])
def test_reducer_snapshot_detects_installed_hook_state_mutation(mutation):
    from types import SimpleNamespace
    from deepspeed.runtime.base_optimizer import BackwardHookStateManager
    api = probe_api()
    snapshot = getattr(api, "reducer_snapshot", None)
    assert callable(snapshot), "missing complete reducer-state snapshot"
    state = BackwardHookStateManager()
    state.enter_backward(); state.exit_backward()
    zero = SimpleNamespace(_backward_hook_state=state)
    before = snapshot(zero)
    if mutation == "hook":
        state.update_hook_state_and_maybe_run_epilogue(3)
    elif mutation == "reentry":
        state.reenter_backward_if_needed()
    elif mutation == "callback":
        # Queue a real installed-manager callback during a CPU autograd graph task.
        state.enter_backward()
        before = snapshot(zero)
        p = torch.tensor(2., requires_grad=True)
        p.register_hook(lambda grad: (state.update_hook_state_and_maybe_run_epilogue(1), grad)[1])
        p.square().backward()
        assert state.post_backward_callback_queued
        assert state.post_backward_callback_graph_task_id is not None
    else:
        state.register_grad_acc_post_hook(lambda: None)
    with pytest.raises(AssertionError):
        api._assert_same(before, snapshot(zero))


def test_finish_rejects_mutation_after_verified_base_result(resumed_probe):
    _, trainer, parameter, optimizer = resumed_probe
    parameter.grad = torch.tensor([0.4, -0.2]); optimizer.step()
    with torch.no_grad():
        parameter.sub_(torch.tensor([0.001, -0.002]))
    with pytest.raises(AssertionError):
        trainer._finish_optimizer_step(trainer.model_wrapped, successful=True)


def test_initial_result_is_nonzero_and_persisted_before_reload(resumed_probe):
    api, trainer, parameter, optimizer = resumed_probe
    trainer.probe_phase = "initial"; trainer.state.global_step = 0
    trainer.probe_stats.update(successful_mutations=0, initial_aux_movement_max=0.0,
                               reload_retains_aux=False)
    trainer.probe_micros = 4
    parameter.grad = torch.tensor([0.4, -0.2]); optimizer.step()
    trainer._finish_optimizer_step(trainer.model_wrapped, successful=True)
    trainer.save_probe_reference()
    saved = torch.load(trainer._reference_path(), weights_only=True)
    torch.testing.assert_close(saved["masters"][0], parameter, rtol=0, atol=0)
    assert saved["stats"]["successful_mutations"] == 1
    assert saved["stats"]["initial_aux_movement_max"] > 100 * torch.finfo(torch.float32).eps
    assert saved["stats"]["second_step_retains_aux"] is False


def test_initial_zero_movement_cannot_use_prior_cumulative_movement(resumed_probe):
    _, trainer, parameter, optimizer = resumed_probe
    trainer.probe_phase = "initial"; trainer.state.global_step = 0
    trainer.probe_stats.update(successful_mutations=0, initial_aux_movement_max=0.0)
    trainer.probe_clean.zero_(); trainer.probe_micros = 4
    parameter.grad = torch.tensor([0.4, -0.2]); optimizer.step()
    with pytest.raises(AssertionError):
        trainer._finish_optimizer_step(trainer.model_wrapped, successful=True)



def test_runtime_comparator_uses_independent_same_bf16_manual_reference(monkeypatch):
    from types import SimpleNamespace
    from dataclasses import replace
    from deepspeed.runtime.base_optimizer import BackwardHookStateManager
    from stage1.equivalence import stage1_checkpoint_context_fn
    api = probe_api()
    torch.manual_seed(42)
    model, batch, targets = api.cpu_score_fixture()
    model.bfloat16()
    targets = replace(targets, latent_values=targets.latent_values.bfloat16())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        "use_reentrant": False, "context_fn": stage1_checkpoint_context_fn})
    trainer = api.ProbeTrainer.__new__(api.ProbeTrainer)
    trainer.model = model
    # This mirrors __init__ at FIX_BASE; corrected reference is cast to the runtime dtype.
    trainer.reference_model = deepcopy(model).float()
    trainer.state = SimpleNamespace(global_step=0)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.target_builder = SimpleNamespace(build=lambda teacher, inputs, metadata: targets)
    trainer.ema_teacher = SimpleNamespace(model=model)
    trainer._prepare_inputs = lambda inputs: inputs
    trainer.probe_clean = trainer.probe_masked = None
    trainer.probe_count = trainer.probe_micros = 0
    trainer.probe_stats = dict(score_residual=0.0, gradient_relative_l2=0.0)
    zero = SimpleNamespace(_backward_hook_state=BackwardHookStateManager())
    monkeypatch.setattr(api.dist, "get_rank", lambda: 2)
    original_fork = torch.random.fork_rng
    monkeypatch.setattr(torch.random, "fork_rng", lambda devices, **kwargs: original_fork(devices=[], **kwargs))
    monkeypatch.setattr(api.CustomTrainerStage1, "training_step",
                        lambda self, model, inputs, num_items_in_batch: torch.tensor(0.0))
    # Prior real base backward must also survive runtime comparison and manual reference.
    with api.activate_stage1_stochastic_context(api._context(model, batch, 0)):
        model(**api.score_inputs(model, batch, targets, masked=False)).logits.float().square().mean().backward()
    old_grads = [None if p.grad is None else p.grad.clone() for p in trainer._parameters()]
    result = trainer.training_step(SimpleNamespace(optimizer=zero), batch)
    assert result == 0
    assert trainer.probe_count == 2 and trainer.probe_micros == 1
    assert trainer.probe_stats["gradient_relative_l2"] <= .01
    assert trainer.probe_clean.dtype == trainer.probe_masked.dtype == torch.float32
    assert trainer.probe_clean.abs().max() > 0
    assert trainer.probe_masked.abs().max() > 0
    for p, before in zip(trainer._parameters(), old_grads):
        if before is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, before, rtol=0, atol=0)



@pytest.mark.parametrize("block_count,masked", [(0, False), (0, True), (1, False), (3, False), (1, True), (3, True)])
def test_manual_reference_is_independent_and_handles_unused_gradient_and_block_sums(monkeypatch, block_count, masked):
    from dataclasses import replace
    from stage1.equivalence import stage1_checkpoint_context_fn
    api = probe_api()
    torch.manual_seed(42)
    model, batch, targets = api.cpu_score_fixture()
    model.bfloat16()
    model.register_parameter("probe_unused", torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16)))
    targets = replace(targets, latent_values=targets.latent_values.bfloat16())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        "use_reentrant": False, "context_fn": stage1_checkpoint_context_fn})
    expected_scores = []
    expected = api.score_gradients(model, batch, targets, block_count=block_count, masked=masked,
                                   optimizer_step=0, scores=expected_scores)
    def forbidden(*args, **kwargs):
        raise AssertionError("manual reference called runtime score_gradients")
    monkeypatch.setattr(api, "score_gradients", forbidden)
    reference_scores = []
    actual = api.reference_score_gradients(deepcopy(model), batch, targets, block_count=block_count,
                                           masked=masked, optimizer_step=0, scores=reference_scores)
    torch.testing.assert_close(torch.tensor(reference_scores), torch.tensor(expected_scores), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(api._flatten(actual), api._flatten(expected), rtol=1e-2, atol=1e-5)
    unused_index = [name for name,p in model.named_parameters() if p.requires_grad].index("probe_unused")
    assert torch.equal(actual[unused_index], torch.zeros(2))
    assert all(gradient.dtype == torch.float32 for gradient in actual)
    if block_count == 0:
        assert all(not torch.count_nonzero(gradient) for gradient in actual)
    else:
        assert api._flatten(actual).abs().max() > 0
