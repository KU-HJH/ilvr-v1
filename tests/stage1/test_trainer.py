from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from accelerate.utils import DeepSpeedEngineWrapper, DistributedType
from peft_common import resolve_qwen2_5_vl_model
from stage1.peft import build_stage1_tuned_model
from stage1.types import (
    Stage1BatchMetadata,
    Stage1LatentSpan,
    Stage1SampleMetadata,
    Stage1TeacherTargets,
)
from tests.stage1.test_peft import (
    LATENT_TOKEN_IDS,
    assert_finite_nonzero_gradient,
    assert_finite_nonzero_tensor,
    tiny_qwen as tiny_qwen_fixture,
    tuning,
)
from trainer import CustomTrainerStage1
from stage1.trainer import _Stage1OptimizerCallback


class CountingTargetBuilder:
    def __init__(self, latent_values, latent_mask, selected_counts):
        self.latent_values = latent_values
        self.latent_mask = latent_mask
        self.selected_counts = selected_counts
        self.teacher_language_calls = 0
        self.helper_visual_calls = 0
        self.return_empty_targets = False

    def build(self, teacher, inputs, metadata):
        del teacher
        self.teacher_language_calls += 1
        self.helper_visual_calls += 1
        inputs.pop("pixel_values_latent")
        inputs.pop("image_grid_thw_latent")
        if self.return_empty_targets:
            values = self.latent_values.new_empty((0, self.latent_values.shape[-1]))
            mask = torch.zeros_like(self.latent_mask)
            counts = tuple(
                tuple(0 for _ in sample.latent_spans)
                for sample in metadata.samples
            )
        else:
            values = self.latent_values
            mask = self.latent_mask
            counts = self.selected_counts
        return Stage1TeacherTargets(
            latent_values=values,
            latent_mask=mask,
            selected_counts=counts,
            diagnostics={
                "latent_count": torch.tensor(values.shape[0]),
                "latent_norm_mean": (
                    values.float().norm(dim=-1).mean()
                    if values.shape[0]
                    else torch.tensor(0.0)
                ),
            },
        )


class CountingStudent(nn.Module):
    def __init__(self, hidden_size=1):
        super().__init__()
        self.language_scale = nn.Parameter(torch.tensor(2.0))
        self.visual = nn.Linear(1, hidden_size, bias=False)
        self.visual.requires_grad_(False)
        self.hidden_size = hidden_size
        self.student_forward_count = 0
        self.last_forward_kwargs = None
        self.last_inputs_embeds = None

    @property
    def device(self):
        return self.language_scale.device

    @property
    def dtype(self):
        return self.language_scale.dtype

    def forward(self, **kwargs):
        self.student_forward_count += 1
        if self.student_forward_count > 1:
            raise AssertionError("student was called once per sample")
        self.last_forward_kwargs = dict(kwargs)
        forbidden = {"pixel_values_latent", "image_grid_thw_latent"} & kwargs.keys()
        if forbidden:
            raise AssertionError(f"helper images reached student: {sorted(forbidden)}")
        input_ids = kwargs["input_ids"]
        mask = kwargs["image_out_mask"]
        latent_values = kwargs["latent_hidden_states"]
        inputs_embeds = torch.zeros(
            (*input_ids.shape, self.hidden_size),
            dtype=self.language_scale.dtype,
            device=self.language_scale.device,
        )
        inputs_embeds[mask] = latent_values.to(inputs_embeds)
        hidden_states = torch.zeros_like(inputs_embeds) + self.language_scale * 0.0
        for row, column in mask.nonzero(as_tuple=False).tolist():
            if column:
                hidden_states[row, column - 1] = -inputs_embeds[row, column]
        self.last_inputs_embeds = inputs_embeds
        logits = torch.zeros(
            (*input_ids.shape, 4),
            dtype=self.language_scale.dtype,
            device=self.language_scale.device,
        )
        return SimpleNamespace(
            loss=self.language_scale.square(),
            logits=logits,
            hidden_states=hidden_states,
            inputs_embeds=inputs_embeds,
        )


class RecordingEma:
    def __init__(self, model=None, events=None):
        self.model = model if model is not None else nn.Identity()
        self.events = events
        self.update_count = 0
        self.values_at_update = []

    def update(self, student):
        if self.events is not None:
            self.events.append("ema.update")
        self.update_count += 1
        parameter = next(parameter for parameter in student.parameters() if parameter.requires_grad)
        self.values_at_update.append(parameter.detach().clone())


class FakeZero3Optimizer:
    """Small ZeRO-3 norm witness matching DeepSpeed 0.18's step prelude."""

    def __init__(self, norm_groups, events, *, loss_scale=2.0):
        self.norm_groups = norm_groups
        self.events = events
        self.loss_scale = loss_scale

    def _get_norm_groups(self):
        self.events.append("optimizer.norm_groups")
        return self.norm_groups


class FakeDeepSpeedEngine(nn.Module):
    """Engine call order consumed by Accelerate 1.10's wrapper."""

    def __init__(self, module, *, norm_groups, step_applied=True):
        super().__init__()
        self.module = module
        self.events = []
        self.optimizer = FakeZero3Optimizer(norm_groups, self.events)
        self._boundary = False
        self._step_applied = False
        self.next_step_applied = step_applied
        self.mutation_count = 0

    def set_gradient_accumulation_boundary(self, is_boundary):
        self._boundary = bool(is_boundary)
        self.events.append(f"engine.boundary:{self._boundary}")

    def backward(self, loss, **kwargs):
        del loss, kwargs
        self.events.append("engine.backward")

    def is_gradient_accumulation_boundary(self):
        return self._boundary

    def step(self):
        self.events.append("engine.step")
        self._step_applied = bool(self.next_step_applied)
        if self._step_applied:
            self.mutation_count += 1

    def was_step_applied(self):
        self.events.append("engine.was_step_applied")
        return self._step_applied


def metadata_for(mask):
    samples = []
    for row in range(mask.shape[0]):
        positions = mask[row].nonzero(as_tuple=False).view(-1).tolist()
        samples.append(
            Stage1SampleMetadata(
                source_index=row,
                source_id=f"sample-{row}",
                original_grid_range=(row, row + 1),
                helper_grid_range=(row, row + 1),
                latent_spans=(
                    Stage1LatentSpan(
                        start=max(0, positions[0] - 1),
                        end=positions[-1] + 1,
                        latent_positions=tuple(positions),
                    ),
                ),
                helper_to_block=(0,),
                unpadded_length=mask.shape[1],
            )
        )
    return Stage1BatchMetadata(
        samples=tuple(samples),
        batch_size=mask.shape[0],
        sequence_length=mask.shape[1],
    )


def clone_batch(batch):
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else deepcopy(value)
        for key, value in batch.items()
    }


def make_batch(mask):
    batch_size, width = mask.shape
    return {
        "input_ids": torch.arange(batch_size * width).view(batch_size, width) % 4,
        "attention_mask": torch.ones((batch_size, width), dtype=torch.long),
        "labels": torch.arange(batch_size * width).view(batch_size, width) % 4,
        "pixel_values": torch.zeros((batch_size, 1)),
        "image_grid_thw": torch.ones((batch_size, 3), dtype=torch.long),
        "pixel_values_latent": torch.ones((batch_size, 1)),
        "image_grid_thw_latent": torch.ones((batch_size, 3), dtype=torch.long),
        "stage1_batch_metadata": metadata_for(mask),
    }


def make_trainer(model, target_builder, *, sim_weight=1.0, ce_weight=1.0, diagnostics=False):
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    trainer.model = model
    trainer.target_builder = target_builder
    trainer.ema_teacher = RecordingEma()
    trainer.sim_weight = float(sim_weight)
    trainer.ce_weight = float(ce_weight)
    trainer.collect_stage1_diagnostics = diagnostics
    trainer._latest_stage1_diagnostics = None
    trainer._successful_optimizer_steps = 0
    trainer._ema_execution_device = None
    trainer._deepspeed_step_adapter = None
    trainer.model_wrapped = model
    trainer.accelerator = SimpleNamespace(
        optimizer_step_was_skipped=False,
        distributed_type=DistributedType.NO,
        device=model.device,
    )
    return trainer


@pytest.fixture
def batch_b2():
    mask = torch.tensor(
        [
            [False, True, False, False, False],
            [False, True, False, True, False],
        ]
    )
    return make_batch(mask)


@pytest.fixture
def stage1_trainer(batch_b2):
    mask = batch_b2["stage1_batch_metadata"]
    del mask
    latent_mask = torch.tensor(
        [
            [False, True, False, False, False],
            [False, True, False, True, False],
        ]
    )
    values = torch.tensor([[1.0], [2.0], [3.0]])
    builder = CountingTargetBuilder(values, latent_mask, ((1,), (2,)))
    return make_trainer(CountingStudent(), builder)


def test_compatibility_import_points_to_authoritative_stage1_module():
    """Catches the row-zero legacy trainer remaining authoritative."""
    assert CustomTrainerStage1.__module__ == "stage1.trainer"


def test_trainer_runs_one_student_forward_for_batched_examples(stage1_trainer, batch_b2):
    """Catches simulating a batch by invoking the student once per sample."""
    loss = stage1_trainer.compute_loss(stage1_trainer.model, clone_batch(batch_b2))

    assert torch.isfinite(loss)
    assert stage1_trainer.model.student_forward_count == 1
    assert stage1_trainer.target_builder.teacher_language_calls == 1
    assert stage1_trainer.target_builder.helper_visual_calls == 1


def test_helper_images_never_reach_student(stage1_trainer, batch_b2):
    """Catches entering Qwen's legacy helper-image B != 1 branch."""
    stage1_trainer.compute_loss(stage1_trainer.model, clone_batch(batch_b2))

    forwarded = stage1_trainer.model.last_forward_kwargs
    assert "pixel_values_latent" not in forwarded
    assert "image_grid_thw_latent" not in forwarded
    assert forwarded["image_out_mask"].shape == batch_b2["input_ids"].shape


def test_variable_latent_counts_keep_row_major_target_ownership(stage1_trainer, batch_b2):
    """Catches packing every selected target into batch row zero."""
    stage1_trainer.compute_loss(stage1_trainer.model, clone_batch(batch_b2))

    mask = stage1_trainer.model.last_forward_kwargs["image_out_mask"]
    torch.testing.assert_close(
        stage1_trainer.model.last_inputs_embeds[mask],
        torch.tensor([[1.0], [2.0], [3.0]]),
    )


def test_missing_teacher_target_is_not_a_fallback(stage1_trainer, batch_b2):
    """Catches a valid latent block silently falling back to helper encoding."""
    stage1_trainer.target_builder.return_empty_targets = True

    with pytest.raises(RuntimeError, match="valid latent block.*teacher target"):
        stage1_trainer.compute_loss(stage1_trainer.model, clone_batch(batch_b2))
    assert stage1_trainer.model.student_forward_count == 0


def test_b1_ce_alignment_and_total_loss_match_legacy_golden():
    """Catches B=1 noise/detach drift with non-collinear 2-D targets."""
    mask = torch.tensor([[False, True, False, True, False]])
    batch = make_batch(mask)
    teacher_values = torch.tensor(
        [[2.0, 1.0], [1.0, 3.0]], requires_grad=True
    )
    builder = CountingTargetBuilder(
        teacher_values, mask, ((2,),)
    )
    trainer = make_trainer(
        CountingStudent(hidden_size=2),
        builder,
        sim_weight=0.25,
        ce_weight=1.5,
        diagnostics=True,
    )
    seed = 123
    torch.manual_seed(seed)
    expected_target = teacher_values.detach() + 0.01 * torch.randn_like(
        teacher_values.detach()
    )
    expected_predicted = -teacher_values.detach()
    expected_alignment = 1.0 - F.cosine_similarity(
        expected_target, expected_predicted, dim=-1
    ).mean()
    expected_loss = 1.5 * torch.tensor(4.0) + 0.25 * expected_alignment

    torch.manual_seed(seed)
    loss = trainer.compute_loss(trainer.model, batch)
    loss.backward()

    torch.testing.assert_close(loss, expected_loss)
    assert trainer._latest_stage1_diagnostics["ce_loss"].item() == pytest.approx(4.0)
    torch.testing.assert_close(
        trainer._latest_stage1_diagnostics["alignment_loss"],
        expected_alignment,
    )
    assert trainer._latest_stage1_diagnostics["latent_count"].item() == 2
    assert teacher_values.grad is None


def make_qwen_trainer(method):
    model = build_stage1_tuned_model(
        tiny_qwen_fixture.__wrapped__(), tuning(method), LATENT_TOKEN_IDS
    )
    if method == "dora":
        qwen = resolve_qwen2_5_vl_model(model)
        with torch.no_grad():
            qwen.model.embed_tokens.token_adapter.trainable_tokens_delta["default"].add_(0.05)
            for name, parameter in model.named_parameters():
                if "lora_B" in name.split("."):
                    parameter.fill_(0.05)
    input_ids = torch.tensor(
        [
            [1, 125, 4, 126, 5, 127, 126, 6],
            [2, 125, 126, 3, 127, 126, 7, 8],
        ]
    )
    mask = torch.tensor(
        [
            [False, False, False, False, False, False, True, False],
            [False, False, True, False, False, True, False, False],
        ]
    )
    batch = make_batch(mask)
    batch["input_ids"] = input_ids
    batch["attention_mask"] = torch.ones_like(input_ids)
    batch["labels"] = input_ids.clone()
    batch["pixel_values"] = None
    batch["image_grid_thw"] = None
    values = torch.tensor(
        [[0.25] * 8, [0.50] * 8, [0.75] * 8], dtype=torch.float32
    )
    trainer = make_trainer(
        model,
        CountingTargetBuilder(values, mask, ((1,), (2,))),
        sim_weight=0.7,
    )
    return trainer, batch


def test_full_trainer_loss_has_finite_language_gradients_and_no_visual_gradients():
    """Catches trainer-side detachment or visual-gradient leakage in full mode."""
    trainer, batch = make_qwen_trainer("full")
    qwen = resolve_qwen2_5_vl_model(trainer.model)

    trainer.compute_loss(trainer.model, batch).backward()

    for parameter in (
        qwen.model.embed_tokens.weight,
        qwen.model.layers[0].self_attn.q_proj.weight,
        qwen.model.layers[0].mlp.down_proj.weight,
        qwen.model.norm.weight,
        qwen.lm_head.weight,
    ):
        assert_finite_nonzero_gradient(parameter)
    assert all(parameter.grad is None for parameter in qwen.visual.parameters())


def test_dora_trainer_loss_reaches_direction_magnitude_and_exact_token_rows_only():
    """Catches missing adapter/row gradients or leakage into frozen Qwen state."""
    trainer, batch = make_qwen_trainer("dora")
    qwen = resolve_qwen2_5_vl_model(trainer.model)

    trainer.compute_loss(trainer.model, batch).backward()

    adapter_parameters = {
        name: parameter
        for name, parameter in trainer.model.named_parameters()
        if {"lora_A", "lora_B", "lora_magnitude_vector"} & set(name.split("."))
    }
    for component in ("lora_A", "lora_B", "lora_magnitude_vector"):
        selected = [
            parameter
            for name, parameter in adapter_parameters.items()
            if component in name.split(".")
        ]
        assert selected
        for parameter in selected:
            assert_finite_nonzero_gradient(parameter)

    token_adapter = qwen.model.embed_tokens.token_adapter
    input_delta = token_adapter.trainable_tokens_delta["default"]
    output_delta = qwen.lm_head.output_delta
    for row in range(3):
        assert_finite_nonzero_tensor(input_delta.grad[row])
        assert_finite_nonzero_tensor(output_delta.grad[row])

    adapter_ids = {id(parameter) for parameter in adapter_parameters.values()}
    frozen_decoder = [
        parameter
        for parameter in qwen.model.parameters()
        if id(parameter) not in adapter_ids | {id(input_delta)}
    ]
    assert frozen_decoder
    assert all(parameter.grad is None for parameter in frozen_decoder)
    assert token_adapter.base_layer.weight.grad is None
    assert qwen.lm_head.base_head.weight.grad is None
    assert all(parameter.grad is None for parameter in qwen.visual.parameters())


def test_optimizer_step_validates_gradients_then_updates_ema_after_success():
    """Catches EMA updates before or without a successful optimizer mutation."""
    mask = torch.tensor([[False, True, False]])
    trainer = make_trainer(
        CountingStudent(),
        CountingTargetBuilder(torch.tensor([[1.0]]), mask, ((1,),)),
    )
    trainer.ema_teacher = RecordingEma()
    batch = make_batch(mask)
    loss = trainer.compute_loss(trainer.model, batch)
    loss.backward()
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)

    trainer.optimizer_step(optimizer=optimizer)

    assert trainer._successful_optimizer_steps == 1
    assert trainer.ema_teacher.update_count == 1
    torch.testing.assert_close(
        trainer.ema_teacher.values_at_update[0],
        trainer.model.language_scale.detach(),
    )


def test_nonfinite_gradient_prevents_optimizer_and_ema_step():
    """Catches nonfinite training state being counted as a successful step."""
    mask = torch.tensor([[False, True, False]])
    trainer = make_trainer(
        CountingStudent(),
        CountingTargetBuilder(torch.tensor([[1.0]]), mask, ((1,),)),
    )
    trainer.ema_teacher = RecordingEma()
    trainer.model.language_scale.grad = torch.tensor(float("inf"))
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    before = trainer.model.language_scale.detach().clone()

    with pytest.raises(FloatingPointError, match="gradient"):
        trainer.optimizer_step(optimizer=optimizer)

    torch.testing.assert_close(trainer.model.language_scale.detach(), before)
    assert trainer._successful_optimizer_steps == 0
    assert trainer.ema_teacher.update_count == 0


def make_deepspeed_trainer(*, norm_groups, step_applied=True):
    mask = torch.tensor([[False, True, False]])
    trainer = make_trainer(
        CountingStudent(),
        CountingTargetBuilder(torch.tensor([[1.0]]), mask, ((1,),)),
        diagnostics=True,
    )
    engine = FakeDeepSpeedEngine(
        trainer.model,
        norm_groups=norm_groups,
        step_applied=step_applied,
    )
    trainer.ema_teacher = RecordingEma(events=engine.events)
    trainer._attach_deepspeed_step_adapter(engine)
    return trainer, engine, DeepSpeedEngineWrapper(engine)


def test_deepspeed_delayed_prepare_places_ema_on_rank_execution_device():
    """Models Trainer's CPU __init__ followed by post-prepare placement."""
    mask = torch.tensor([[False, True, False]])
    trainer = make_trainer(
        CountingStudent(),
        CountingTargetBuilder(torch.tensor([[1.0]]), mask, ((1,),)),
    )
    trainer.ema_teacher = RecordingEma(nn.Linear(1, 1))
    assert next(trainer.ema_teacher.model.parameters()).device.type == "cpu"
    trainer.model_wrapped = SimpleNamespace(device=torch.device("meta"))
    trainer.accelerator.device = torch.device("meta")

    _Stage1OptimizerCallback(trainer).on_train_begin(
        None, None, SimpleNamespace()
    )

    assert next(trainer.ema_teacher.model.parameters()).device.type == "meta"
    assert trainer._ema_execution_device == torch.device("meta")


def test_teacher_device_mismatch_fails_before_target_construction():
    """Catches silently building CPU EMA targets for a prepared rank batch."""
    mask = torch.tensor([[False, True, False]])
    builder = CountingTargetBuilder(torch.tensor([[1.0]]), mask, ((1,),))
    trainer = make_trainer(CountingStudent(), builder)
    trainer.ema_teacher = RecordingEma(nn.Linear(1, 1))
    batch = make_batch(mask)
    batch["input_ids"] = batch["input_ids"].to("meta")

    with pytest.raises(RuntimeError, match="EMA teacher.*device"):
        trainer.compute_loss(trainer.model, batch)

    assert builder.teacher_language_calls == 0


def test_accelerate_deepspeed_nonboundary_microstep_does_not_validate_or_step():
    trainer, engine, wrapper = make_deepspeed_trainer(
        norm_groups=[torch.tensor(3.0)]
    )

    wrapper.backward(torch.tensor(1.0), sync_gradients=False)

    assert engine.events == ["engine.boundary:False", "engine.backward"]
    assert engine.mutation_count == 0
    assert trainer.ema_teacher.update_count == 0


def test_accelerate_deepspeed_boundary_validates_before_step_and_updates_once():
    trainer, engine, wrapper = make_deepspeed_trainer(
        norm_groups=[torch.tensor(3.0), torch.tensor(4.0)]
    )

    wrapper.backward(torch.tensor(1.0), sync_gradients=True)

    assert engine.events == [
        "engine.boundary:True",
        "engine.backward",
        "optimizer.norm_groups",
        "engine.step",
        "engine.was_step_applied",
        "ema.update",
    ]
    assert engine.mutation_count == 1
    assert trainer._successful_optimizer_steps == 1
    assert trainer.ema_teacher.update_count == 1
    torch.testing.assert_close(
        trainer._latest_stage1_diagnostics["gradient_norm"],
        torch.tensor(2.5),
    )

    callback = _Stage1OptimizerCallback(trainer)
    control = SimpleNamespace()
    callback.on_pre_optimizer_step(None, None, control, model=trainer.model)
    callback.on_optimizer_step(None, None, control, model=trainer.model)
    assert trainer.ema_teacher.update_count == 1


def test_accelerate_deepspeed_overflow_witness_prevents_ema_update():
    trainer, engine, wrapper = make_deepspeed_trainer(
        norm_groups=[torch.tensor(2.0)], step_applied=False
    )

    wrapper.backward(torch.tensor(1.0), sync_gradients=True)

    assert engine.events[-2:] == ["engine.step", "engine.was_step_applied"]
    assert engine.mutation_count == 0
    assert trainer._successful_optimizer_steps == 0
    assert trainer.ema_teacher.update_count == 0


@pytest.mark.parametrize(
    "norm_groups",
    [[], [torch.tensor(-1.0)], [torch.tensor(float("nan"))]],
    ids=["missing", "deepspeed-nonfinite-sentinel", "nan"],
)
def test_accelerate_deepspeed_invalid_gradient_evidence_prevents_engine_step(
    norm_groups,
):
    trainer, engine, wrapper = make_deepspeed_trainer(norm_groups=norm_groups)

    with pytest.raises(FloatingPointError, match="gradient"):
        wrapper.backward(torch.tensor(1.0), sync_gradients=True)

    assert "engine.step" not in engine.events
    assert engine.mutation_count == 0
    assert trainer.ema_teacher.update_count == 0


def test_accelerate_deepspeed_missing_loss_scale_prevents_engine_step():
    trainer, engine, wrapper = make_deepspeed_trainer(
        norm_groups=[torch.tensor(2.0)]
    )
    engine.optimizer.loss_scale = None

    with pytest.raises(FloatingPointError, match="loss scale"):
        wrapper.backward(torch.tensor(1.0), sync_gradients=True)

    assert "engine.step" not in engine.events
    assert trainer.ema_teacher.update_count == 0


def test_unsupported_deepspeed_prepare_lifecycle_fails_closed():
    mask = torch.tensor([[False, True, False]])
    trainer = make_trainer(
        CountingStudent(),
        CountingTargetBuilder(torch.tensor([[1.0]]), mask, ((1,),)),
    )
    trainer.accelerator.distributed_type = DistributedType.DEEPSPEED
    trainer.accelerator.deepspeed_engine_wrapped = SimpleNamespace(engine=object())

    with pytest.raises(RuntimeError, match="unsupported DeepSpeed"):
        trainer._install_deepspeed_step_adapter()


def test_nonfinite_loss_raises_before_backward_or_step():
    """Catches NaN/Inf loss entering gradient accumulation."""
    mask = torch.tensor([[False, True, False]])
    student = CountingStudent()
    student.language_scale.data.fill_(float("inf"))
    trainer = make_trainer(
        student,
        CountingTargetBuilder(torch.tensor([[1.0]]), mask, ((1,),)),
    )

    with pytest.raises(FloatingPointError, match="loss is nonfinite"):
        trainer.compute_loss(trainer.model, make_batch(mask))
