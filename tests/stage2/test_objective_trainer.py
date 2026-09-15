"""Actual tiny-Qwen and native Trainer objective contracts (CPU only)."""

from copy import deepcopy
from pathlib import Path
import json
import tempfile

import pytest
import torch
from transformers import TrainingArguments
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds

from stage2.objective import Stage2ObjectiveConfig
from stage2.objective_windows import TextVocabulary
from stage2.reference import FrozenReference, STAGE2_OBJECTIVE_FILENAME
from stage2.trainer import Stage2Trainer
from .helpers import tiny_stage2_qwen


@pytest.fixture(autouse=True)
def objective_precision_environment(monkeypatch):
    # TrainingArguments preserves inherited mixed precision when both flags are
    # false. Each oracle case starts FP32 and enables BF16 explicitly as needed.
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "no")


def batch(*, valid=True, two=False):
    ids = torch.tensor([[7, 8, 91, 90, 90, 92, 9, 10] +
                        ([91, 90, 90, 92, 11, 12] if two else [])])
    labels = ids.clone()
    labels[ids == 90] = -100
    if not valid:
        labels[:] = -100
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                stage2_text_segments=[[(0, 2), (6, 8), (12, 14)] if two else [(0, 2), (6, 8)]])


def runtime(config, model):
    from stage2.objective_runtime import Stage2ObjectiveRuntime
    reference = FrozenReference(deepcopy(model).requires_grad_(False).eval(), {"source": "original-stage1"}) if config.lambda_dyn else None
    vocabulary = TextVocabulary(tuple(i for i in range(128) if i not in (80, 81, 82, 90, 91, 92)), {}, 128)
    return Stage2ObjectiveRuntime(config, LatentTokenIds(90, 91, 92), 2, vocabulary, reference)


def make_trainer(model, config=None, objective_runtime=None, *, accumulation=2, **kwargs):
    root = Path("checkpoints") / "stage2-objective-dev"
    root.mkdir(exist_ok=True)
    directory = tempfile.mkdtemp(prefix="trainer-", dir=root)
    settings = dict(use_cpu=True, report_to=[], disable_tqdm=True,
                    per_device_train_batch_size=1, gradient_accumulation_steps=accumulation,
                    remove_unused_columns=False, save_strategy="no")
    settings.update(kwargs)
    args = TrainingArguments(directory, **settings)
    return Stage2Trainer(model=model, args=args, stage2_objective_config=config,
                         stage2_objective_runtime=objective_runtime)


@pytest.mark.parametrize("dyn,vis", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_actual_qwen_modes_forward_graphs_and_logs(dyn, vis):
    """Catches wrong branch routing, detached F, or graph-bearing M/reference."""
    torch.manual_seed(41)
    model = tiny_stage2_qwen().eval()
    config = Stage2ObjectiveConfig(True, dyn, vis, "original" if dyn else None)
    rt = runtime(config, model) if config.active else None
    trainer = make_trainer(model, config, rt)
    observed = []
    handles = [model.register_forward_hook(lambda m, a, kw, out: observed.append(("F" if torch.is_grad_enabled() else "M", out.logits.requires_grad, set(kw))), with_kwargs=True)]
    if dyn:
        handles.append(rt.reference.model.register_forward_hook(lambda m, a, out: observed.append(("ref", out.logits.requires_grad, set()))))
    loss, output = trainer.compute_loss(model, batch(two=True), return_outputs=True)
    assert torch.isfinite(loss)
    assert [x[0] for x in observed].count("F") == 1
    assert [x[0] for x in observed].count("M") == 2 * vis
    assert [x[0] for x in observed].count("ref") == dyn
    assert all(graph == (branch == "F") for branch, graph, _ in observed)
    assert not any("stage2_text_segments" in keys or "pixel_values_latent" in keys for _, _, keys in observed)
    loss.backward()
    assert model.model.embed_tokens.weight.grad.abs().sum() > 0
    trainer.create_optimizer()
    if dyn:
        reference_parameters = list(rt.reference.model.parameters())
        assert all(p.grad is None and not p.requires_grad for p in reference_parameters)
        optimizer_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
        assert not optimizer_ids.intersection(map(id, reference_parameters))
    trainer.log({})
    logs = trainer.state.log_history[-1]
    assert logs["stage2/objective/loss_ce"] == pytest.approx(output.loss.item())
    if dyn:
        assert logs["stage2/objective/loss_dyn"] == pytest.approx(-0.5, abs=1e-6)
        assert logs["stage2/objective/r_pre"] == pytest.approx(1.0)
    else:
        assert "stage2/objective/loss_dyn" not in logs
    assert ("stage2/objective/S_M" in logs) == bool(vis)
    assert logs["stage2/objective/model_forwards_F"] == 1
    assert logs["stage2/objective/decoder_forwards_F"] == 5
    for handle in handles:
        handle.remove()


def test_ce_only_bypasses_window_parser_and_preserves_legacy_loss():
    """Catches auxiliary parsing or CE reduction changes when both weights are zero."""
    model = tiny_stage2_qwen().eval()
    legacy = make_trainer(model)
    enabled = make_trainer(model, Stage2ObjectiveConfig(True))
    inputs = batch()
    inputs.pop("stage2_text_segments")
    assert torch.equal(legacy.compute_loss(model, inputs), enabled.compute_loss(model, inputs))
    inputs["stage2_text_segments"] = "unused invalid metadata"
    assert torch.isfinite(enabled.compute_loss(model, inputs))
    inputs["labels"][:] = -100
    loss = enabled.compute_loss(model, inputs)
    assert loss.item() == 0 and loss.requires_grad
    for ids in ([91, 90, 90, 92, 9], [10, 91, 90, 90, 92, 9]):
        tokens = torch.tensor([ids])
        labels = tokens.masked_fill(tokens == 90, -100)
        example = dict(input_ids=tokens, attention_mask=torch.ones_like(tokens), labels=labels)
        assert torch.equal(legacy.compute_loss(model, example), enabled.compute_loss(model, example))


def test_actual_trainer_logs_packed_m_runtime_counters_once():
    """Catches Trainer dropping packed M's literal model/decoder counter meanings."""
    from .packed_m_helpers import (
        TEXT_VOCABULARY,
        TOKENS,
        make_packed_inputs,
        make_packed_model,
    )
    from stage2.objective_runtime import Stage2ObjectiveRuntime

    model = make_packed_model()
    config = Stage2ObjectiveConfig(
        enable_stage2_objective=True,
        lambda_vis=1.0,
        stage2_m_execution="packed_suffix",
        post_window=2,
    )
    objective_runtime = Stage2ObjectiveRuntime(
        config, TOKENS, 2, TEXT_VOCABULARY, reference=None
    )
    trainer = make_trainer(model, config, objective_runtime, accumulation=1)
    inputs = make_packed_inputs(blocks=2)
    selection = objective_runtime.select(inputs)
    loss = trainer.compute_loss(model, inputs)
    assert torch.isfinite(loss)
    trainer.log({})
    logs = trainer.state.log_history[-1]
    assert logs["stage2/objective/model_forwards_M"] == 0
    assert logs["stage2/objective/decoder_forwards_M"] == 1
    assert logs["stage2/objective/m_branches"] == len(selection.blocks)
    assert logs["stage2/objective/m_query_rows"] == sum(
        max(block.post_positions) - block.end_position + 1
        for block in selection.blocks
    )


def test_empty_valid_set_skips_auxiliary_and_malformed_markers_raise():
    model = tiny_stage2_qwen()
    config = Stage2ObjectiveConfig(True, 1, 1, "original")
    trainer = make_trainer(model, config, runtime(config, model))
    loss = trainer.compute_loss(model, batch(valid=False))
    assert loss.item() == 0 and loss.requires_grad
    bad = batch()
    bad["input_ids"][0, 5] = 7
    with pytest.raises(ValueError, match="malformed"):
        trainer.compute_loss(model, bad)


def test_objective_identity_saved_for_full_model_and_rejects_off_resume():
    model = tiny_stage2_qwen()
    trainer = make_trainer(model, Stage2ObjectiveConfig(True))
    trainer.save_model()
    assert (Path(trainer.args.output_dir) / STAGE2_OBJECTIVE_FILENAME).is_file()
    identity = json.loads((Path(trainer.args.output_dir) / STAGE2_OBJECTIVE_FILENAME).read_text())
    assert identity["reference_identity"] is None and identity["vocabulary_identity"] is None
    off = make_trainer(model)
    with pytest.raises(ValueError, match="objective"):
        off.train(resume_from_checkpoint=trainer.args.output_dir)


@pytest.mark.parametrize("change", [dict(stage="stage1"), dict(deepspeed_config={"zero_optimization": {"stage": 1}}),
                                     dict(deepspeed_config={"zero_optimization": {"stage": 2}}),
                                     dict(deepspeed_config={"zero_optimization": {"stage": 3}}),
                                     dict(attention_dropout=0.1), dict(use_sliding_window=True)])
def test_active_invalid_execution_rejected_before_loading(change):
    from main import route_training
    from types import SimpleNamespace
    values = dict(stage="stage2", enable_stage2_objective=True, lambda_vis=1.0)
    values.update(change)
    with pytest.raises(ValueError):
        route_training(SimpleNamespace(**values), stage1_handler=lambda _: pytest.fail("loaded"),
                       stage2_handler=lambda _: pytest.fail("loaded"))


def test_actual_trainer_rejects_accumulation_plugin():
    config = Stage2ObjectiveConfig(True, 0, 1)
    model = tiny_stage2_qwen()
    with pytest.raises(ValueError, match="accumulation"):
        make_trainer(model, config, runtime(config, model),
                     accelerator_config={"gradient_accumulation_kwargs": {"sync_each_batch": True}})


@pytest.mark.parametrize("source", ["args", "environment", "student"])
def test_active_objective_rejects_fp16(source, monkeypatch):
    from types import SimpleNamespace
    from stage2.objective_runtime import objective_config_from_args, validate_objective_student
    if source == "student":
        with pytest.raises(ValueError, match="FP16"):
            validate_objective_student(tiny_stage2_qwen().half())
    else:
        if source == "environment":
            monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp16")
        with pytest.raises(ValueError, match="FP16"):
            objective_config_from_args(SimpleNamespace(enable_stage2_objective=True,
                lambda_vis=1.0, fp16=source == "args"))


@pytest.mark.parametrize("enabled,weight", [(False, 1.0), (True, 0.0)])
def test_inactive_objective_preserves_fp16_configuration(enabled, weight, monkeypatch):
    from types import SimpleNamespace
    from stage2.objective_runtime import objective_config_from_args
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp16")
    assert not objective_config_from_args(SimpleNamespace(enable_stage2_objective=enabled,
        lambda_vis=weight, fp16=True)).active


@pytest.mark.parametrize("branch", ["F_entropy", "reference_entropy", "F_likelihood", "M_likelihood"])
def test_numerical_failure_reports_sample_block_and_selected_positions(branch):
    """Catches loss of context when the actual selected logit rows are nonfinite."""
    dyn = "entropy" in branch
    config = Stage2ObjectiveConfig(True, float(dyn), float(not dyn), "original" if dyn else None)
    model = tiny_stage2_qwen().eval()
    rt = runtime(config, model)
    inputs = batch(two=True)
    selection = rt.select(inputs)
    bad = selection.blocks[1]
    target = rt.reference.model if branch.startswith("reference") else model
    def corrupt(module, args, kwargs, output):
        masked = "stage2_ablation" in kwargs
        if (branch.startswith("M_") and masked) or (not branch.startswith("M_") and not masked):
            output.logits[bad.batch_index, bad.post_positions[0], 7] = float("nan")
    handle = target.register_forward_hook(corrupt, with_kwargs=True)
    try:
        with pytest.raises(ValueError) as caught:
            rt.forward(model, inputs, selection)
    finally:
        handle.remove()
    message = str(caught.value)
    assert f"batch_index={bad.batch_index}" in message
    assert f"block_index={bad.block_index}" in message
    assert f"pre_positions={bad.pre_positions}" in message
    assert f"post_positions={bad.post_positions}" in message
    assert isinstance(caught.value.__cause__, ValueError)
    assert "nonfinite" in str(caught.value.__cause__)


def test_active_rejects_fsdp_environment_before_loading(monkeypatch):
    from main import route_training
    from types import SimpleNamespace
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    with pytest.raises(ValueError, match="FSDP"):
        route_training(SimpleNamespace(stage="stage2", enable_stage2_objective=True, lambda_vis=1.0),
                       stage2_handler=lambda _: pytest.fail("loaded before plugin validation"))


@pytest.mark.parametrize("weight", [0, 1])
def test_enabled_nonfinite_ce_is_reported_even_without_valid_blocks(weight):
    model = tiny_stage2_qwen()
    config = Stage2ObjectiveConfig(True, 0, weight)
    trainer = make_trainer(model, config, runtime(config, model) if weight else None)
    with torch.no_grad():
        model.lm_head.weight.fill_(float("nan"))
    inputs = batch()
    inputs["labels"][0, 6:] = -100
    with pytest.raises(ValueError, match="nonfinite.*CE"):
        trainer.compute_loss(model, inputs)


def test_native_checkpoint_resume_keeps_original_reference():
    """Catches missing per-checkpoint identity or replacing reference on resume."""
    initial = tiny_stage2_qwen()
    config = Stage2ObjectiveConfig(True, 0.5, 0.5, "original")
    model = deepcopy(initial)
    first = make_trainer(model, config, runtime(config, initial), accumulation=1,
                         max_steps=1, save_strategy="steps", save_steps=1, dataloader_pin_memory=False)
    first.train_dataset = [batch()]
    first.data_collator = lambda examples: examples[0]
    first.train()
    saved = Path(first.args.output_dir) / "checkpoint-1"
    assert (saved / STAGE2_OBJECTIVE_FILENAME).exists()
    resumed_model = deepcopy(initial)
    rt = runtime(config, initial)
    before = {name: p.clone() for name, p in rt.reference.model.named_parameters()}
    resumed = make_trainer(resumed_model, config, rt, accumulation=1, max_steps=2,
                           dataloader_pin_memory=False)
    resumed.train_dataset = [batch()]
    resumed.data_collator = lambda examples: examples[0]
    resumed.train(resume_from_checkpoint=str(saved))
    assert resumed.state.global_step == 2
    assert any(not torch.equal(p, before[name]) for name, p in resumed_model.named_parameters())
    assert all(torch.equal(p, before[name]) for name, p in rt.reference.model.named_parameters())


def test_accelerate_prepared_student_and_independent_reference_share_autocast():
    """Catches relying only on the student's prepared forward autocast wrapper."""
    model = tiny_stage2_qwen()
    config = Stage2ObjectiveConfig(True, 1, 1, "original")
    rt = runtime(config, model)
    trainer = make_trainer(model, config, rt, bf16=True)
    prepared = trainer.accelerator.prepare_model(model)
    assert trainer.accelerator.native_amp
    observed = {"student": [], "reference": []}
    model.lm_head.register_forward_hook(lambda m, a, out: observed["student"].append(out.dtype))
    rt.reference.model.lm_head.register_forward_hook(lambda m, a, out: observed["reference"].append(out.dtype))
    loss = trainer.compute_loss(prepared, batch())
    loss.backward()
    assert observed == {"student": [torch.bfloat16, torch.bfloat16], "reference": [torch.bfloat16]}
    assert not torch.is_autocast_enabled("cpu")


def test_logging_pools_block_sums_across_unequal_microbatches():
    """Catches last-batch logging and means of unequal per-microbatch means."""
    model = tiny_stage2_qwen().eval()
    config = Stage2ObjectiveConfig(True, 0, 0.8)
    trainer = make_trainer(model, config, runtime(config, model))
    first, second = batch(two=True), batch(two=True)
    second["labels"][0, 12:] = -100
    ce_values, vis_values = [], []
    for inputs, positions in [(first, [(5, 2, 5, 9), (11, 8, 11, 11)]), (second, [(5, 2, 5, 9)])]:
        clean = {key: value for key, value in inputs.items() if key != "stage2_text_segments"}
        with torch.no_grad():
            f = model(**clean)
            ce_values.append(f.loss.item())
            for post, start, end, target in positions:
                m = model(**{key: value for key, value in clean.items() if key != "labels"}, stage2_ablation=(start, end))
                gain = f.logits[0, post].float().log_softmax(-1)[target] - m.logits[0, post].float().log_softmax(-1)[target]
                vis_values.append(torch.logaddexp(torch.zeros_like(gain), -gain).item())
        trainer.compute_loss(model, inputs)
    trainer.log({})
    logged = trainer.state.log_history[-1]
    assert logged.get("stage2/objective/loss_ce") == pytest.approx(sum(ce_values) / 2)
    assert logged.get("stage2/objective/loss_vis") == pytest.approx(sum(vis_values) / 3)
    assert logged.get("stage2/objective/loss_total") == pytest.approx(sum(ce_values) / 2 + 0.8 * sum(vis_values) / 3)
    assert logged.get("stage2/objective/valid_blocks") == 3
    assert logged.get("stage2/objective/model_forwards_F") == 2


def test_exclusion_counts_are_unique_and_low_entropy_counts_are_token_occurrences():
    model = tiny_stage2_qwen().eval()
    config = Stage2ObjectiveConfig(True, 1, 0, "original", entropy_eps=10, pre_window=2, post_window=2)
    rt = runtime(config, model)
    trainer = make_trainer(model, config, rt)
    trainer.compute_loss(model, batch())
    trainer.log({})
    assert trainer.state.log_history[-1]["stage2/objective/low_reference_entropy"] == 4
    tokens = torch.tensor([[91, 90, 90, 92, 9]])
    trainer.compute_loss(model, dict(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                     labels=torch.full_like(tokens, -100), stage2_text_segments=[[(0, 0), (4, 5)]]))
    trainer.log({})
    logs = trainer.state.log_history[-1]
    assert logs["stage2/objective/excluded_blocks"] == 1
    assert logs["stage2/objective/excluded_empty_pre"] == 1
    assert logs["stage2/objective/excluded_empty_post"] == 1
    assert "stage2/objective/model_forwards_reference" not in logs


@pytest.mark.parametrize("dyn", [0, 1])
def test_main_builds_objective_with_real_original_stage1_reference(monkeypatch, dyn):
    """Catches missing main wiring or auxiliary vocabulary/reference work in CE-only."""
    import main
    from types import SimpleNamespace
    from .test_routing import stage2_args, _stub_stage2_data_pipeline
    from .test_objective_reference import _write_full_stage1, _dependencies
    from tests.stage1.test_checkpointing import TinyStage1Tokenizer

    root = Path("checkpoints/stage2-objective-dev")
    root.mkdir(exist_ok=True)
    source = _write_full_stage1(Path(tempfile.mkdtemp(prefix="main-", dir=root)))
    monkeypatch.setattr(TinyStage1Tokenizer, "__call__", lambda self, *a, **kw: {"input_ids": torch.tensor([[31, 32]])}, raising=False)
    if dyn:
        monkeypatch.setattr(TinyStage1Tokenizer, "get_vocab", lambda self: {**{f"text-{i}": i for i in range(125)}, **self._ids}, raising=False)
        monkeypatch.setattr(TinyStage1Tokenizer, "all_special_ids", [125, 126, 127], raising=False)
    else:
        monkeypatch.setattr("stage2.objective_windows.build_text_vocabulary", lambda *a: pytest.fail("unused vocabulary built"))
    _stub_stage2_data_pipeline(monkeypatch, SimpleNamespace(prepared=[], filtered=None))
    args = stage2_args(stage="stage2", load_model_path=str(source), latent_size=2,
                       enable_stage2_objective=True, lambda_dyn=dyn, lambda_vis=0,
                       reference_model_path=str(source) if dyn else None, attention_implementation="eager",
                       gradient_checkpointing=False)
    components = main.build_stage2_training_components(args, source_dependencies=_dependencies())
    assert components.data_collator.enable_stage2_objective == bool(dyn)
    if not dyn:
        assert components.objective_runtime is None
        return
    rt = components.objective_runtime
    model = components.model
    moves = []
    original_to = rt.reference.model.to

    def observe_move(*args, **kwargs):
        moves.append(next(model.parameters()).device)
        return original_to(*args, **kwargs)

    monkeypatch.setattr(rt.reference.model, "to", observe_move)
    trainer = make_trainer(model, components.objective_config, rt)
    assert moves == [trainer.accelerator.device]
    before = {name: p.clone() for name, p in rt.reference.model.named_parameters()}
    example = batch()
    for old, new in [(90, 125), (91, 126), (92, 127)]:
        example["input_ids"][example["input_ids"] == old] = new
        example["labels"][example["labels"] == old] = new
    trainer.compute_loss(model, example).backward()
    trainer.create_optimizer()
    trainer.optimizer.step()
    for name, p in rt.reference.model.named_parameters():
        assert torch.equal(p, before[name])
    assert len(moves) == 1
    trainer.save_model()
    rt.reference.identity["source"] = "different-original"
    with pytest.raises(ValueError, match="reference identity mismatch"):
        trainer.train(resume_from_checkpoint=trainer.args.output_dir)


@pytest.mark.parametrize("backend", ["flash_attention_2", "other"])
def test_active_invalid_backend_rejected_before_loading(backend):
    from main import route_training
    from types import SimpleNamespace
    args = SimpleNamespace(stage="stage2", enable_stage2_objective=True,
                           lambda_dyn=0.0, lambda_vis=1.0, attention_implementation=backend)
    with pytest.raises(ValueError, match="backend"):
        route_training(args, stage2_handler=lambda _: pytest.fail("loaded before validation"))


@pytest.mark.parametrize("kind", ["reentrant", "custom"])
def test_active_rejects_unvalidated_actual_checkpoint_function(kind):
    """Catches manually enabled/custom checkpoint functions bypassing args checks."""
    model = tiny_stage2_qwen()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": kind == "reentrant"})
    if kind == "custom":
        model.model._gradient_checkpointing_func = lambda fn, *args: fn(*args)
    config = Stage2ObjectiveConfig(True, 0, 1)
    with pytest.raises(ValueError, match="checkpoint"):
        make_trainer(model, config, runtime(config, model))


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("bf16", [False, True])
def test_actual_runtime_dora_rng_latent_gradients_and_checkpoint_equivalence(backend, bf16):
    """Catches runtime replay corrupting F RNG/latent gradients, including recompute."""
    from .test_objective_ablation import _dora_model
    from transformers.models.qwen2_5_vl import stage2_latent_rollout
    from stage2.rng import capture_rng_state, replay_rng_state
    from unittest.mock import patch

    torch.manual_seed(93)
    initial = _dora_model(attn_implementation=backend)
    reference_state = capture_rng_state()
    snapshots = []
    for checkpointed in (False, True):
        model = deepcopy(initial)
        if checkpointed:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        config = Stage2ObjectiveConfig(True, 0.6, 0.8, "original")
        rt = runtime(config, model)
        trainer = make_trainer(model, config, rt)
        generated = []
        original = stage2_latent_rollout._replace_embedding

        def retain(embeds, position, value):
            if torch.is_grad_enabled():
                value.retain_grad()
                generated.append(value)
            return original(embeds, position, value)

        with replay_rng_state(reference_state), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            with patch.object(stage2_latent_rollout, "_replace_embedding", side_effect=retain):
                loss = trainer.compute_loss(model, batch(two=True))
            after_runtime = capture_rng_state()
        with replay_rng_state(reference_state), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            only_f = model(**{k: v for k, v in batch(two=True).items() if k != "stage2_text_segments"})
            after_f = capture_rng_state()
        assert torch.equal(after_runtime.cpu_state, after_f.cpu_state)
        loss.backward()
        assert generated and all(value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0 for value in generated)
        snapshots.append((loss.detach(), {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}))
    torch.testing.assert_close(snapshots[0][0], snapshots[1][0], atol=1e-5, rtol=1e-5)
    assert snapshots[0][1].keys() == snapshots[1][1].keys()
    for name in snapshots[0][1]:
        torch.testing.assert_close(snapshots[0][1][name], snapshots[1][1][name], atol=1e-5, rtol=1e-4, msg=name)
