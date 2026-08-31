from __future__ import annotations

import json
import math
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from trl import SFTTrainer

from accelerate.utils import DeepSpeedEngineWrapper, DistributedType
from deepspeed_runtime import (
    DEEPSPEED_RUNTIME_FILENAME,
    DeepSpeedRuntimeProfile,
    canonical_deepspeed_config_path,
    load_deepspeed_runtime_profile,
    write_deepspeed_runtime_sidecar,
)
from peft_common import resolve_qwen2_5_vl_model
from stage1.ema import Stage1EmaTeacher
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
from stage1.trainer import (
    STAGE1_EMA_TEACHER_STATE_FILENAME,
    _DeepSpeedStepAdapter,
    _Stage1OptimizerCallback,
)
import stage1.trainer as stage1_trainer_module


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
    def __init__(self, hidden_size=1, prediction_matrix=None):
        super().__init__()
        self.language_scale = nn.Parameter(torch.tensor(2.0))
        self.visual = nn.Linear(1, hidden_size, bias=False)
        self.visual.requires_grad_(False)
        self.hidden_size = hidden_size
        if prediction_matrix is None:
            prediction_matrix = -torch.eye(hidden_size)
        prediction_matrix = torch.as_tensor(prediction_matrix, dtype=torch.float32)
        if tuple(prediction_matrix.shape) != (hidden_size, hidden_size):
            raise ValueError("prediction_matrix must have shape [H, H]")
        self.register_buffer("prediction_matrix", prediction_matrix)
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
                hidden_states[row, column - 1] = (
                    inputs_embeds[row, column] @ self.prediction_matrix
                )
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


class UnequalTargetLengthStudent(nn.Module):
    """Causal-LM probe whose two samples have different target lengths."""

    def __init__(self):
        super().__init__()
        self.language_scale = nn.Parameter(torch.tensor(1.0))

    @property
    def device(self):
        return self.language_scale.device

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        labels = kwargs["labels"]
        feature = input_ids.to(dtype=self.language_scale.dtype)
        logits = torch.stack(
            (torch.zeros_like(feature), self.language_scale * feature), dim=-1
        )
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, 2),
            labels[:, 1:].reshape(-1),
        )
        return SimpleNamespace(loss=loss, logits=logits)


class RecordingEma:
    def __init__(self, model=None, events=None, decay=0.999):
        self.model = model if model is not None else nn.Identity()
        self.events = events
        self.decay = float(decay)
        self.update_count = 0
        self.values_at_update = []

    def update(self, student):
        if self.events is not None:
            self.events.append("ema.update")
        self.update_count += 1
        parameter = next(parameter for parameter in student.parameters() if parameter.requires_grad)
        self.values_at_update.append(parameter.detach().clone())


class EmaCheckpointProbeStudent(nn.Module):
    """Tiny full/PEFT topology whose forward exposes its EMA-selected value."""

    def __init__(self, method):
        super().__init__()
        self.visual = nn.Linear(1, 1, bias=False)
        self.visual.requires_grad_(False)
        if method == "full":
            self.language_weight = nn.Parameter(torch.tensor(1.0))
        else:
            self.decoder = nn.Module()
            self.decoder.lora_A = nn.Parameter(torch.tensor(1.0))
        self.method = method

    @property
    def ema_value(self):
        if self.method == "full":
            return self.language_weight
        return self.decoder.lora_A

    def forward(self, inputs):
        return inputs * self.ema_value


class FakeZero3Optimizer:
    """Small ZeRO-3 norm witness matching DeepSpeed 0.18's step prelude."""

    def __init__(self, norm_groups, events, *, loss_scale=2.0):
        self.norm_groups = norm_groups
        self.events = events
        self.loss_scale = loss_scale

    def _get_norm_groups(self):
        self.events.append("optimizer.norm_groups")
        return self.norm_groups


class FakeZero12Optimizer:
    def __init__(self, *, partition_gradients: bool):
        self.partition_gradients = partition_gradients
        self.loss_scale = 2.0
        self.calls = 0

    def scaled_global_norm(self):
        self.calls += 1
        return torch.tensor(12.0)


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
    trainer._stage1_pending_ema_checkpoint = None
    trainer.model_wrapped = model
    trainer.accelerator = SimpleNamespace(
        optimizer_step_was_skipped=False,
        distributed_type=DistributedType.NO,
        device=model.device,
    )
    return trainer


def configure_checkpoint_runtime(trainer, output_dir):
    trainer.args = SimpleNamespace(output_dir=str(output_dir), should_save=True)
    trainer.deepspeed_runtime_profile = DeepSpeedRuntimeProfile(
        enabled=False,
        zero_stage=None,
        config_path=None,
        config_digest=None,
        world_size=1,
    )
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda value, reduction: value,
        wait_for_everyone=lambda: None,
    )


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


def test_ce_gives_equal_sample_weight_despite_unequal_target_token_lengths():
    """Catches reducing CE over all valid tokens across the batch."""
    mask = torch.tensor(
        [
            [False, True, False, False],
            [False, True, False, False],
        ]
    )
    batch = make_batch(mask)
    batch["input_ids"] = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    batch["labels"] = torch.tensor(
        [
            [-100, 0, -100, -100],
            [-100, 0, 0, 0],
        ]
    )
    trainer = make_trainer(
        UnequalTargetLengthStudent(),
        CountingTargetBuilder(
            torch.tensor([[1.0], [1.0]]), mask, ((1,), (1,))
        ),
        sim_weight=0.0,
    )

    loss = trainer.compute_loss(trainer.model, batch)
    loss.backward()

    expected_loss = (math.log1p(math.exp(1.0)) + math.log(2.0)) / 2.0
    expected_gradient = (1.0 / (1.0 + math.exp(-1.0))) / 2.0
    assert loss.item() == pytest.approx(expected_loss)
    assert trainer.model.language_scale.grad.item() == pytest.approx(
        expected_gradient
    )


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
    prediction_matrix = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    builder = CountingTargetBuilder(
        teacher_values, mask, ((2,),)
    )
    trainer = make_trainer(
        CountingStudent(
            hidden_size=2,
            prediction_matrix=prediction_matrix,
        ),
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
    expected_predicted = teacher_values.detach() @ prediction_matrix
    noisy_alignment = 1.0 - F.cosine_similarity(
        expected_target, expected_predicted, dim=-1
    ).mean()
    no_noise_alignment = 1.0 - F.cosine_similarity(
        teacher_values.detach(), expected_predicted, dim=-1
    ).mean()
    noisy_loss = 1.5 * torch.tensor(4.0) + 0.25 * noisy_alignment
    no_noise_loss = 1.5 * torch.tensor(4.0) + 0.25 * no_noise_alignment
    assert abs((noisy_alignment - no_noise_alignment).item()) > 5e-4
    assert abs((noisy_loss - no_noise_loss).item()) > 1e-4

    torch.manual_seed(seed)
    loss = trainer.compute_loss(trainer.model, batch)
    loss.backward()

    torch.testing.assert_close(loss, noisy_loss, rtol=0.0, atol=1e-7)
    assert trainer._latest_stage1_diagnostics["ce_loss"].item() == pytest.approx(4.0)
    torch.testing.assert_close(
        trainer._latest_stage1_diagnostics["alignment_loss"],
        noisy_alignment,
        rtol=0.0,
        atol=1e-7,
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


def test_successful_optimizer_step_skips_frozen_teacher_update(monkeypatch):
    """Catches frozen teachers entering the per-step update call path."""
    mask = torch.tensor([[False, True, False]])
    trainer = make_trainer(
        CountingStudent(),
        CountingTargetBuilder(torch.tensor([[1.0]]), mask, ((1,),)),
    )
    trainer.ema_teacher = Stage1EmaTeacher(
        trainer.model, decay=1.0, method="full"
    )
    monkeypatch.setattr(
        trainer.ema_teacher,
        "update",
        lambda _: pytest.fail("frozen EMA teacher entered update"),
    )

    trainer._finish_optimizer_step(trainer.model, successful=True)

    assert trainer._successful_optimizer_steps == 1


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
    trainer.deepspeed_runtime_profile = DeepSpeedRuntimeProfile(
        True, 3, None, None, 2
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


@pytest.mark.parametrize("stage,partitioned", [(1, False), (2, True)])
def test_zero12_norm_uses_partition_aware_scaled_norm(stage, partitioned):
    optimizer = FakeZero12Optimizer(partition_gradients=partitioned)

    norm = stage1_trainer_module.collect_deepspeed_pre_step_norm(
        optimizer, zero_stage=stage
    )

    assert norm.item() == pytest.approx(6.0)
    assert optimizer.calls == 1


def test_ema_updates_once_only_when_deepspeed_applies_step(
    stage1_trainer, monkeypatch
):
    applied = iter((False, True))
    engine = SimpleNamespace(
        is_gradient_accumulation_boundary=lambda: True,
        was_step_applied=lambda: next(applied),
    )
    finishes = []
    monkeypatch.setattr(
        stage1_trainer, "_validate_deepspeed_gradient_norm", lambda _: None
    )
    monkeypatch.setattr(
        stage1_trainer,
        "_finish_optimizer_step",
        lambda model, *, successful: finishes.append(successful),
    )
    adapter = _DeepSpeedStepAdapter(stage1_trainer, engine, lambda: None)

    adapter()
    adapter()

    assert finishes == [False, True]


@pytest.mark.parametrize("method", ["full", "dora"])
def test_numbered_checkpoint_resume_matches_uninterrupted_ema_targets(
    tmp_path, method
):
    """Catches rebuilding resumed helper targets from the student snapshot."""

    uninterrupted_student = EmaCheckpointProbeStudent(method)
    interrupted_student = deepcopy(uninterrupted_student)
    uninterrupted_ema = Stage1EmaTeacher(
        uninterrupted_student, decay=0.5, method=method
    )
    interrupted_ema = Stage1EmaTeacher(
        interrupted_student, decay=0.5, method=method
    )
    with torch.no_grad():
        uninterrupted_student.ema_value.fill_(3.0)
        interrupted_student.ema_value.fill_(3.0)
    uninterrupted_ema.update(uninterrupted_student)
    interrupted_ema.update(interrupted_student)

    checkpoint = tmp_path / "checkpoint-1"
    writer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    writer.ema_teacher = interrupted_ema
    writer._save_stage1_ema_checkpoint(checkpoint)

    resumed_student = deepcopy(interrupted_student)
    resumed_ema = Stage1EmaTeacher(resumed_student, decay=0.5, method=method)
    resumed = CustomTrainerStage1.__new__(CustomTrainerStage1)
    resumed.ema_teacher = resumed_ema
    resumed._load_stage1_ema_checkpoint(checkpoint)

    with torch.no_grad():
        uninterrupted_student.ema_value.fill_(5.0)
        resumed_student.ema_value.fill_(5.0)
    uninterrupted_ema.update(uninterrupted_student)
    resumed_ema.update(resumed_student)

    probe = torch.tensor([2.0])
    torch.testing.assert_close(
        resumed_ema.model(probe), uninterrupted_ema.model(probe), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        resumed_ema.model.ema_value,
        uninterrupted_ema.model.ema_value,
        rtol=0.0,
        atol=0.0,
    )


def test_numbered_checkpoint_saves_ema_after_parent_checkpoint_on_writing_rank(
    tmp_path, monkeypatch
):
    """Catches racing EMA publication ahead of Trainer/DeepSpeed state."""

    checkpoint = tmp_path / "checkpoint-7"
    parent_marker = checkpoint / "deepspeed-parent-save-complete"

    def parent_save(trainer, model, trial):
        del trainer, model, trial
        checkpoint.mkdir()
        parent_marker.write_text("complete\n", encoding="utf-8")

    monkeypatch.setattr(SFTTrainer, "_save_checkpoint", parent_save)
    student = EmaCheckpointProbeStudent("full")
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    trainer.ema_teacher = Stage1EmaTeacher(student, decay=0.5, method="full")
    trainer.deepspeed_runtime_profile = load_deepspeed_runtime_profile(
        canonical_deepspeed_config_path(2), world_size=2
    )
    trainer.args = SimpleNamespace(should_save=True)
    trainer.state = SimpleNamespace(global_step=7)
    trainer._get_output_dir = lambda trial: str(tmp_path)
    barriers = []
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda value, reduction: value,
        wait_for_everyone=lambda: barriers.append("barrier")
    )

    trainer._save_checkpoint(student, trial=None)

    assert parent_marker.is_file()
    assert (checkpoint / STAGE1_EMA_TEACHER_STATE_FILENAME).is_file()
    payload = json.loads(
        (checkpoint / DEEPSPEED_RUNTIME_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["zero_stage"] == 2
    assert payload["world_size"] == 2
    assert barriers == ["barrier"]


def test_train_publishes_runtime_identity_at_output_root(tmp_path, monkeypatch):
    """Catches a run starting without its immutable runtime manifest."""
    parent_calls = []

    def parent_train(trainer, *args, **kwargs):
        del trainer, args, kwargs
        parent_calls.append("train")
        return "trained"

    monkeypatch.setattr(SFTTrainer, "train", parent_train)
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path), should_save=True)
    trainer.deepspeed_runtime_profile = load_deepspeed_runtime_profile(
        canonical_deepspeed_config_path(2), world_size=2
    )
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda value, reduction: value,
        wait_for_everyone=lambda: None,
    )

    result = trainer.train()

    payload = json.loads(
        (tmp_path / DEEPSPEED_RUNTIME_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["zero_stage"] == 2
    assert result == "trained"
    assert parent_calls == ["train"]


def test_stage1_resume_rejects_runtime_mismatch_before_parent_train(
    tmp_path, monkeypatch
):
    """Catches incompatible runtime state reaching Trainer restore."""
    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()
    write_deepspeed_runtime_sidecar(
        checkpoint,
        load_deepspeed_runtime_profile(
            canonical_deepspeed_config_path(1), world_size=2
        ),
    )
    parent_calls = []

    def parent_train(trainer, *args, **kwargs):
        del trainer, args, kwargs
        parent_calls.append("train")

    monkeypatch.setattr(SFTTrainer, "train", parent_train)
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path), should_save=True)
    trainer.deepspeed_runtime_profile = load_deepspeed_runtime_profile(
        canonical_deepspeed_config_path(2), world_size=2
    )
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda value, reduction: value,
        wait_for_everyone=lambda: None,
    )
    trainer._stage1_ema_checkpoint_path = lambda path: checkpoint / "ema.pt"

    with pytest.raises(ValueError, match="zero_stage"):
        trainer.train(resume_from_checkpoint=str(checkpoint))

    assert parent_calls == []


def test_numbered_checkpoint_nonwriting_rank_never_materializes_ema_state(
    tmp_path, monkeypatch
):
    """Catches duplicate per-rank EMA serialization or gathering."""

    checkpoint = tmp_path / "checkpoint-7"

    def parent_save(trainer, model, trial):
        del trainer, model, trial
        checkpoint.mkdir()

    monkeypatch.setattr(SFTTrainer, "_save_checkpoint", parent_save)
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    trainer.ema_teacher = SimpleNamespace(
        state_dict=lambda: pytest.fail("non-writing rank serialized EMA")
    )
    trainer.args = SimpleNamespace(should_save=False)
    trainer.state = SimpleNamespace(global_step=7)
    trainer._get_output_dir = lambda trial: str(tmp_path)
    barriers = []
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda value, reduction: value,
        wait_for_everyone=lambda: barriers.append("barrier")
    )

    trainer._save_checkpoint(nn.Identity(), trial=None)

    assert not (checkpoint / STAGE1_EMA_TEACHER_STATE_FILENAME).exists()
    assert barriers == ["barrier"]


def test_ema_save_failure_preserves_previous_checkpoint_before_rotation(
    tmp_path, monkeypatch
):
    previous = tmp_path / "checkpoint-6"
    current = tmp_path / "checkpoint-7"
    previous.mkdir()
    (previous / STAGE1_EMA_TEACHER_STATE_FILENAME).write_bytes(b"complete")
    rotations = []

    def parent_save(trainer, model, trial):
        del model, trial
        current.mkdir()
        trainer._rotate_checkpoints(use_mtime=False, output_dir=str(tmp_path))

    def parent_rotate(trainer, *, use_mtime, output_dir):
        del trainer, use_mtime, output_dir
        rotations.append("rotate")
        (previous / STAGE1_EMA_TEACHER_STATE_FILENAME).unlink()
        previous.rmdir()

    monkeypatch.setattr(SFTTrainer, "_save_checkpoint", parent_save)
    monkeypatch.setattr(SFTTrainer, "_rotate_checkpoints", parent_rotate)
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    trainer.args = SimpleNamespace(should_save=True)
    trainer.state = SimpleNamespace(global_step=7)
    trainer._get_output_dir = lambda trial: str(tmp_path)
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda value, reduction: value,
        wait_for_everyone=lambda: None,
    )
    trainer._save_stage1_ema_checkpoint = lambda path: (_ for _ in ()).throw(
        OSError("EMA disk failure")
    )

    with pytest.raises(OSError, match="EMA disk failure"):
        trainer._save_checkpoint(nn.Identity(), trial=None)

    assert previous.is_dir()
    assert rotations == []


def test_ema_save_failure_is_collectively_reported_before_writer_raises(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint-7"
    events = []

    def parent_save(trainer, model, trial):
        del trainer, model, trial
        checkpoint.mkdir()

    def save_ema(path):
        del path
        events.append("save")
        raise OSError("EMA disk failure")

    def reduce_failure(value, *, reduction):
        events.append(("reduce", int(value.item()), reduction))
        return value

    monkeypatch.setattr(SFTTrainer, "_save_checkpoint", parent_save)
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    trainer.args = SimpleNamespace(should_save=True)
    trainer.state = SimpleNamespace(global_step=7)
    trainer._get_output_dir = lambda trial: str(tmp_path)
    trainer._save_stage1_ema_checkpoint = save_ema
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=reduce_failure,
        wait_for_everyone=lambda: pytest.fail("failed save reached barrier"),
    )

    with pytest.raises(OSError, match="EMA disk failure"):
        trainer._save_checkpoint(nn.Identity(), trial=None)

    assert events == ["save", ("reduce", 1, "sum")]


def test_remote_ema_save_failure_makes_nonwriting_rank_fail_without_barrier(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint-7"
    reductions = []

    def parent_save(trainer, model, trial):
        del trainer, model, trial
        checkpoint.mkdir()

    def reduce_remote_failure(value, *, reduction):
        reductions.append((int(value.item()), reduction))
        return value.new_tensor(1)

    monkeypatch.setattr(SFTTrainer, "_save_checkpoint", parent_save)
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    trainer.args = SimpleNamespace(should_save=False)
    trainer.state = SimpleNamespace(global_step=7)
    trainer._get_output_dir = lambda trial: str(tmp_path)
    trainer._save_stage1_ema_checkpoint = lambda path: pytest.fail(
        "non-writing rank serialized EMA"
    )
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=reduce_remote_failure,
        wait_for_everyone=lambda: pytest.fail("failed save reached barrier"),
    )

    with pytest.raises(RuntimeError, match="another process"):
        trainer._save_checkpoint(nn.Identity(), trial=None)

    assert reductions == [(0, "sum")]


def test_checkpoint_rotation_runs_after_ema_publication_and_rank_sync(
    tmp_path, monkeypatch
):
    current = tmp_path / "checkpoint-7"
    rotations = []
    barriers = []

    def parent_save(trainer, model, trial):
        del model, trial
        current.mkdir()
        trainer._rotate_checkpoints(use_mtime=False, output_dir=str(tmp_path))

    def parent_rotate(trainer, *, use_mtime, output_dir):
        del trainer, use_mtime, output_dir
        rotations.append(
            (
                (current / STAGE1_EMA_TEACHER_STATE_FILENAME).is_file(),
                tuple(barriers),
            )
        )

    monkeypatch.setattr(SFTTrainer, "_save_checkpoint", parent_save)
    monkeypatch.setattr(SFTTrainer, "_rotate_checkpoints", parent_rotate)
    student = EmaCheckpointProbeStudent("full")
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    trainer.ema_teacher = Stage1EmaTeacher(student, decay=0.5, method="full")
    trainer.deepspeed_runtime_profile = DeepSpeedRuntimeProfile(
        False, None, None, None, 1
    )
    trainer.args = SimpleNamespace(should_save=True)
    trainer.state = SimpleNamespace(global_step=7)
    trainer._get_output_dir = lambda trial: str(tmp_path)
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda value, reduction: value,
        wait_for_everyone=lambda: barriers.append("barrier")
    )

    trainer._save_checkpoint(student, trial=None)

    assert rotations == [(True, ("barrier",))]


def test_explicit_stage1_resume_requires_ema_sidecar_before_parent_train(
    tmp_path, monkeypatch
):
    """Catches silently resuming with a teacher copied from the student."""

    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()
    parent_calls = []

    def parent_train(trainer, *args, **kwargs):
        del trainer, args, kwargs
        parent_calls.append("train")

    monkeypatch.setattr(SFTTrainer, "train", parent_train)
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    configure_checkpoint_runtime(trainer, tmp_path)
    write_deepspeed_runtime_sidecar(
        checkpoint, trainer.deepspeed_runtime_profile
    )

    with pytest.raises(RuntimeError, match=r"missing EMA teacher state.*checkpoint-3"):
        trainer.train(resume_from_checkpoint=str(checkpoint))

    assert parent_calls == []


def test_stage1_resume_rejects_symlinked_ema_sidecar(tmp_path):
    student = EmaCheckpointProbeStudent("full")
    external = tmp_path / "external"
    writer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    writer.ema_teacher = Stage1EmaTeacher(student, decay=0.5, method="full")
    writer._save_stage1_ema_checkpoint(external)
    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()
    (checkpoint / STAGE1_EMA_TEACHER_STATE_FILENAME).symlink_to(
        external / STAGE1_EMA_TEACHER_STATE_FILENAME
    )
    resumed = CustomTrainerStage1.__new__(CustomTrainerStage1)
    resumed.ema_teacher = Stage1EmaTeacher(student, decay=0.5, method="full")

    with pytest.raises(RuntimeError, match="EMA teacher state.*symlink"):
        resumed._load_stage1_ema_checkpoint(checkpoint)


def test_stage1_resume_discovery_requires_ema_on_latest_checkpoint(
    tmp_path, monkeypatch
):
    """Catches checking a stale checkpoint instead of Trainer's latest choice."""

    older = tmp_path / "checkpoint-2"
    latest = tmp_path / "checkpoint-11"
    older.mkdir()
    latest.mkdir()
    (older / STAGE1_EMA_TEACHER_STATE_FILENAME).write_bytes(b"older")
    monkeypatch.setattr(
        SFTTrainer,
        "train",
        lambda *args, **kwargs: pytest.fail("parent train accepted incomplete resume"),
    )
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    configure_checkpoint_runtime(trainer, tmp_path)
    write_deepspeed_runtime_sidecar(latest, trainer.deepspeed_runtime_profile)

    with pytest.raises(RuntimeError, match=r"missing EMA teacher state.*checkpoint-11"):
        trainer.train(resume_from_checkpoint=True)


def test_stage1_resume_discovery_forwards_one_resolved_path_to_parent(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint-11"
    student = EmaCheckpointProbeStudent("full")
    writer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    writer.ema_teacher = Stage1EmaTeacher(student, decay=0.5, method="full")
    writer._save_stage1_ema_checkpoint(checkpoint)
    parent_calls = []

    def parent_train(trainer, *args, **kwargs):
        del args
        parent_calls.append(kwargs["resume_from_checkpoint"])
        assert trainer._stage1_pending_ema_checkpoint == checkpoint
        return "resumed"

    monkeypatch.setattr(SFTTrainer, "train", parent_train)
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    configure_checkpoint_runtime(trainer, tmp_path)
    write_deepspeed_runtime_sidecar(
        checkpoint, trainer.deepspeed_runtime_profile
    )

    result = trainer.train(resume_from_checkpoint=True)

    assert result == "resumed"
    assert parent_calls == [str(checkpoint)]


def test_stage1_deprecated_model_path_precedence_binds_the_same_ema_checkpoint(
    tmp_path, monkeypatch
):
    student = EmaCheckpointProbeStudent("full")
    writer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    writer.ema_teacher = Stage1EmaTeacher(student, decay=0.5, method="full")
    explicit = tmp_path / "checkpoint-3"
    deprecated = tmp_path / "checkpoint-7"
    writer._save_stage1_ema_checkpoint(explicit)
    writer._save_stage1_ema_checkpoint(deprecated)
    parent_calls = []

    def parent_train(trainer, *args, **kwargs):
        del args
        parent_calls.append(
            (kwargs["resume_from_checkpoint"], kwargs["model_path"])
        )
        assert trainer._stage1_pending_ema_checkpoint == deprecated
        return "resumed"

    monkeypatch.setattr(SFTTrainer, "train", parent_train)
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    configure_checkpoint_runtime(trainer, tmp_path)
    write_deepspeed_runtime_sidecar(
        deprecated, trainer.deepspeed_runtime_profile
    )

    result = trainer.train(
        resume_from_checkpoint=str(explicit), model_path=str(deprecated)
    )

    assert result == "resumed"
    assert parent_calls == [(str(deprecated), str(deprecated))]


def test_stage1_deprecated_model_path_true_forwards_the_resolved_checkpoint_once(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint-11"
    student = EmaCheckpointProbeStudent("full")
    writer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    writer.ema_teacher = Stage1EmaTeacher(student, decay=0.5, method="full")
    writer._save_stage1_ema_checkpoint(checkpoint)
    parent_calls = []

    def parent_train(trainer, *args, **kwargs):
        del args
        parent_calls.append(
            (kwargs["resume_from_checkpoint"], kwargs["model_path"])
        )
        assert trainer._stage1_pending_ema_checkpoint == checkpoint
        return "resumed"

    monkeypatch.setattr(SFTTrainer, "train", parent_train)
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    configure_checkpoint_runtime(trainer, tmp_path)
    write_deepspeed_runtime_sidecar(
        checkpoint, trainer.deepspeed_runtime_profile
    )

    result = trainer.train(model_path=True)

    assert result == "resumed"
    assert parent_calls == [(str(checkpoint), str(checkpoint))]


def test_fresh_stage1_train_does_not_require_ema_sidecar(tmp_path, monkeypatch):
    parent_calls = []

    def parent_train(trainer, *args, **kwargs):
        del trainer, args
        parent_calls.append(kwargs.get("resume_from_checkpoint"))
        return "fresh-result"

    monkeypatch.setattr(SFTTrainer, "train", parent_train)
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    configure_checkpoint_runtime(trainer, tmp_path)

    result = trainer.train(resume_from_checkpoint=None)

    assert result == "fresh-result"
    assert parent_calls == [None]


def test_training_begin_restores_pending_ema_after_teacher_device_preparation(
    tmp_path,
):
    """Catches loading too early or building the first target before restore."""

    student = EmaCheckpointProbeStudent("full")
    saved_ema = Stage1EmaTeacher(student, decay=0.5, method="full")
    with torch.no_grad():
        student.ema_value.fill_(3.0)
    saved_ema.update(student)
    checkpoint = tmp_path / "checkpoint-1"
    writer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    writer.ema_teacher = saved_ema
    writer._save_stage1_ema_checkpoint(checkpoint)

    resumed_student = deepcopy(student)
    resumed = CustomTrainerStage1.__new__(CustomTrainerStage1)
    resumed.model = resumed_student
    resumed.model_wrapped = resumed_student
    resumed.accelerator = SimpleNamespace(
        device=torch.device("cpu"), distributed_type=DistributedType.NO
    )
    resumed.ema_teacher = Stage1EmaTeacher(
        resumed_student, decay=0.5, method="full"
    )
    resumed._ema_execution_device = None
    resumed._deepspeed_step_adapter = None
    resumed._stage1_pending_ema_checkpoint = checkpoint

    _Stage1OptimizerCallback(resumed).on_train_begin(
        SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    )

    assert resumed._ema_execution_device == torch.device("cpu")
    torch.testing.assert_close(
        resumed.ema_teacher.model(torch.tensor([2.0])),
        torch.tensor([4.0]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("schema", "schema version"),
        ("method", "method mismatch"),
        ("decay", "decay mismatch"),
        ("names", "parameter names mismatch"),
        ("shapes", "parameter shapes mismatch"),
        ("schema_type", "schema version"),
        ("decay_type", "decay"),
        ("names_type", "parameter names"),
        ("shapes_type", "parameter shapes"),
    ],
)
def test_stage1_ema_checkpoint_rejects_incompatible_schema_and_topology(
    tmp_path, corruption, message
):
    student = EmaCheckpointProbeStudent("full")
    writer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    writer.ema_teacher = Stage1EmaTeacher(student, decay=0.5, method="full")
    checkpoint = tmp_path / "checkpoint-1"
    writer._save_stage1_ema_checkpoint(checkpoint)
    path = checkpoint / STAGE1_EMA_TEACHER_STATE_FILENAME
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if corruption == "schema":
        payload["schema_version"] = 2
    elif corruption == "method":
        payload["state"]["method"] = "dora"
    elif corruption == "decay":
        payload["state"]["decay"] = 0.75
    elif corruption == "names":
        extra_name = "decoder.extra.weight"
        payload["state"]["parameter_names"] += (extra_name,)
        payload["state"]["parameter_shapes"] += ((extra_name, ()),)
        payload["state"]["parameters"][extra_name] = torch.tensor(0.0)
    elif corruption == "shapes":
        name, _ = payload["state"]["parameter_shapes"][0]
        payload["state"]["parameter_shapes"] = ((name, (17,)),)
    elif corruption == "schema_type":
        payload["schema_version"] = True
    elif corruption == "decay_type":
        payload["state"]["decay"] = "0.5"
    elif corruption == "names_type":
        payload["state"]["parameter_names"] = list(
            payload["state"]["parameter_names"]
        )
    else:
        payload["state"]["parameter_shapes"] = tuple(
            (name, list(shape))
            for name, shape in payload["state"]["parameter_shapes"]
        )
    torch.save(payload, path)
    resumed = CustomTrainerStage1.__new__(CustomTrainerStage1)
    resumed.ema_teacher = Stage1EmaTeacher(student, decay=0.5, method="full")

    with pytest.raises(RuntimeError, match=message):
        resumed._load_stage1_ema_checkpoint(checkpoint)


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


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
def test_deepspeed_prepare_accepts_optimizer_for_bound_zero_stage(
    monkeypatch, zero_stage
):
    import deepspeed.runtime.engine as engine_module
    import deepspeed.runtime.zero.stage_1_and_2 as zero12_module
    import deepspeed.runtime.zero.stage3 as zero3_module

    mask = torch.tensor([[False, True, False]])
    trainer = make_trainer(
        CountingStudent(),
        CountingTargetBuilder(torch.tensor([[1.0]]), mask, ((1,),)),
    )
    if zero_stage == 3:
        optimizer = FakeZero3Optimizer([torch.tensor(1.0)], [])
    else:
        optimizer = FakeZero12Optimizer(
            partition_gradients=zero_stage == 2
        )
    engine = SimpleNamespace(
        optimizer=optimizer,
        step=lambda: None,
        is_gradient_accumulation_boundary=lambda: True,
        was_step_applied=lambda: True,
    )
    monkeypatch.setattr(engine_module, "DeepSpeedEngine", type(engine))
    monkeypatch.setattr(
        zero12_module, "DeepSpeedZeroOptimizer", FakeZero12Optimizer
    )
    monkeypatch.setattr(
        zero3_module, "DeepSpeedZeroOptimizer_Stage3", FakeZero3Optimizer
    )
    monkeypatch.setattr(
        stage1_trainer_module.importlib_metadata,
        "version",
        lambda package: {
            "accelerate": "1.10.1",
            "deepspeed": "0.18.0",
        }[package],
    )
    trainer.deepspeed_runtime_profile = DeepSpeedRuntimeProfile(
        True, zero_stage, None, None, 2
    )
    trainer.model_wrapped = engine
    trainer.accelerator.distributed_type = DistributedType.DEEPSPEED
    trainer.accelerator.deepspeed_engine_wrapped = DeepSpeedEngineWrapper(engine)

    trainer._install_deepspeed_step_adapter()

    assert trainer._deepspeed_step_adapter.engine is engine


def test_deepspeed_prepare_rejects_optimizer_mismatched_to_bound_stage(monkeypatch):
    import deepspeed.runtime.engine as engine_module
    import deepspeed.runtime.zero.stage_1_and_2 as zero12_module
    import deepspeed.runtime.zero.stage3 as zero3_module

    mask = torch.tensor([[False, True, False]])
    trainer = make_trainer(
        CountingStudent(),
        CountingTargetBuilder(torch.tensor([[1.0]]), mask, ((1,),)),
    )
    optimizer = FakeZero3Optimizer([torch.tensor(1.0)], [])
    engine = SimpleNamespace(
        optimizer=optimizer,
        step=lambda: None,
        is_gradient_accumulation_boundary=lambda: True,
        was_step_applied=lambda: True,
    )
    monkeypatch.setattr(engine_module, "DeepSpeedEngine", type(engine))
    monkeypatch.setattr(
        zero12_module, "DeepSpeedZeroOptimizer", FakeZero12Optimizer
    )
    monkeypatch.setattr(
        zero3_module, "DeepSpeedZeroOptimizer_Stage3", FakeZero3Optimizer
    )
    monkeypatch.setattr(
        stage1_trainer_module.importlib_metadata,
        "version",
        lambda package: {
            "accelerate": "1.10.1",
            "deepspeed": "0.18.0",
        }[package],
    )
    trainer.deepspeed_runtime_profile = DeepSpeedRuntimeProfile(
        True, 2, None, None, 2
    )
    trainer.model_wrapped = engine
    trainer.accelerator.distributed_type = DistributedType.DEEPSPEED
    trainer.accelerator.deepspeed_engine_wrapped = DeepSpeedEngineWrapper(engine)

    with pytest.raises(RuntimeError, match="ZeRO-2 optimizer"):
        trainer._install_deepspeed_step_adapter()

    assert trainer._deepspeed_step_adapter is None


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
