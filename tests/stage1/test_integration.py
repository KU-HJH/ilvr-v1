from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.state import AcceleratorState, GradientState
from safetensors.torch import load_file
from torch import nn
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers import TrainerCallback, TrainingArguments
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

from peft_common import resolve_qwen2_5_vl_model
from stage1.checkpointing import (
    Stage1ArtifactKind,
    classify_stage1_artifact,
    load_stage1_peft_artifact,
)
from stage1.composition import (
    DEFAULT_STAGE1_DEPENDENCIES,
    LATENT_TOKEN_STRINGS,
    build_stage1_training_components,
)
from stage1.targets import Stage1TeacherTargets
from stage1.token_rows import find_sparse_output_token_rows
from stage1.trainer import CustomTrainerStage1
from stage1.types import (
    Stage1BatchMetadata,
    Stage1LatentSpan,
    Stage1SampleMetadata,
)
from stage2.peft import Stage2TuningConfig, load_stage1_dora_for_fresh_stage2
from stage2.trainer import Stage2Trainer
from utils_deepseed import parse_args

from tests.stage1.test_checkpointing import (
    BASE_VOCAB_SIZE,
    tiny_original_qwen,
)


@pytest.fixture(autouse=True)
def isolated_accelerate_state(monkeypatch):
    """Keep CPU lifecycle tests independent of distributed process globals."""
    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)
    unset_hf_deepspeed_config()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    yield
    unset_hf_deepspeed_config()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


class TinyIntegrationTokenizer:
    unk_token = "<unk>"
    unk_token_id = 0
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1
    padding_side = "right"

    def __init__(self, *, include_latent_tokens: bool = False):
        self.vocab = {f"token-{index}": index for index in range(BASE_VOCAB_SIZE)}
        if include_latent_tokens:
            for token in LATENT_TOKEN_STRINGS:
                self.vocab[token] = len(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def add_special_tokens(self, payload):
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id):
        token_id = int(token_id)
        for token, candidate in self.vocab.items():
            if candidate == token_id:
                return token
        return self.unk_token

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text(
            json.dumps({"vocab": self.vocab}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, directory):
        payload = json.loads(
            (Path(directory) / "tokenizer.json").read_text(encoding="utf-8")
        )
        tokenizer = cls()
        tokenizer.vocab = {key: int(value) for key, value in payload["vocab"].items()}
        return tokenizer


class TinyIntegrationProcessor:
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or TinyIntegrationTokenizer()

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "preprocessor_config.json").write_text(
            '{"processor_class":"TinyIntegrationProcessor"}\n',
            encoding="utf-8",
        )
        self.tokenizer.save_pretrained(directory)

    @classmethod
    def from_pretrained(cls, directory):
        return cls(TinyIntegrationTokenizer.from_pretrained(directory))


class CountingFrozenVisual(nn.Module):
    """Cheap visual boundary: one call and one embedding per image token."""

    def __init__(self, hidden_size=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("anchor", torch.zeros(()))
        self.calls = 0

    @property
    def dtype(self):
        return self.anchor.dtype

    def forward(self, pixel_values, *, grid_thw):
        self.calls += 1
        token_count = int(grid_thw.prod(dim=-1).sum().item())
        return torch.zeros(
            (token_count, self.hidden_size),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )


class TinyLifecycleTargetBuilder:
    def __init__(self):
        self.helper_visual_calls = 0
        self.batch_sizes = []

    def build(self, teacher, inputs, metadata):
        del teacher
        self.batch_sizes.append(metadata.batch_size)
        self.helper_visual_calls += 1
        helper_pixels = inputs.pop("pixel_values_latent")
        inputs.pop("image_grid_thw_latent")
        with torch.no_grad():
            _ = helper_pixels.float().sum()

        mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
        for row, sample in enumerate(metadata.samples):
            for span in sample.latent_spans:
                mask[row, list(span.latent_positions)] = True
        values = torch.arange(
            1,
            int(mask.sum().item()) * 8 + 1,
            device=inputs["input_ids"].device,
            dtype=torch.float32,
        ).reshape(-1, 8) / 100.0
        return Stage1TeacherTargets(
            latent_values=values,
            latent_mask=mask,
            selected_counts=tuple((2,) for _ in metadata.samples),
            diagnostics={"latent_count": torch.tensor(values.shape[0])},
        )


class TinyLifecycleCollator:
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, rows):
        batch_size = len(rows)
        pad = self.token_ids.pad_token_id
        start = self.token_ids.start_token_id
        end = self.token_ids.end_token_id
        row = [80, 80, 80, 80, 1, start, pad, pad, end, 2, 3]
        input_ids = torch.tensor([row] * batch_size, dtype=torch.long)
        samples = tuple(
            Stage1SampleMetadata(
                source_index=int(source_index),
                source_id=f"tiny-{source_index}",
                original_grid_range=(index, index + 1),
                helper_grid_range=(index, index + 1),
                latent_spans=(
                    Stage1LatentSpan(start=5, end=8, latent_positions=(6, 7)),
                ),
                helper_to_block=(0,),
                unpadded_length=len(row),
            )
            for index, source_index in enumerate(rows)
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
            "pixel_values": torch.ones((batch_size * 4, 8)),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * batch_size),
            "pixel_values_latent": torch.ones((batch_size, 8)),
            "image_grid_thw_latent": torch.ones((batch_size, 3), dtype=torch.long),
            "stage1_batch_metadata": Stage1BatchMetadata(
                samples=samples,
                batch_size=batch_size,
                sequence_length=len(row),
            ),
        }


class ResumeStateWitness(TrainerCallback):
    def __init__(self):
        self.initial_global_step = None
        self.optimizer_state_restored = False
        self.scheduler_state_restored = False

    def on_train_begin(self, args, state, control, **kwargs):
        del args, control
        self.initial_global_step = state.global_step
        optimizer = kwargs["optimizer"]
        scheduler = kwargs["lr_scheduler"]
        self.optimizer_state_restored = bool(optimizer.state)
        self.scheduler_state_restored = scheduler.state_dict()["last_epoch"] > 0


@dataclass
class TinyStage1Run:
    components: object
    checkpoint_1: Path
    student_batch_sizes: list[int]
    original_visual_calls: int
    helper_visual_calls: int
    micro_batches: int
    losses: list[float]


@dataclass
class TinyResumedRun:
    components: object
    initial_global_step: int
    final_global_step: int
    optimizer_state_restored: bool
    scheduler_state_restored: bool


def test_stage1_ema_treats_unindexed_cuda_as_the_current_rank_device(monkeypatch):
    """Catches `cuda` versus `cuda:rank` spelling blocking real training."""
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    trainer.model_wrapped = SimpleNamespace(device=torch.device("cuda"))
    trainer.accelerator = SimpleNamespace(device=torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    assert trainer._execution_device() == torch.device("cuda:0")


def _tiny_model(config):
    torch.manual_seed(29)
    model = Qwen2_5_VLForConditionalGeneration(config)
    model.visual = CountingFrozenVisual(config.hidden_size)
    return model


def _tiny_dependencies(target_builders):
    def load_processor(source, **kwargs):
        del kwargs
        source = Path(source)
        if source.is_dir() and (source / "tokenizer.json").is_file():
            return TinyIntegrationProcessor.from_pretrained(source)
        return TinyIntegrationProcessor()

    def load_config(source, **kwargs):
        del kwargs
        source = Path(source)
        if source.is_dir() and (source / "config.json").is_file():
            return Qwen2_5_VLConfig.from_pretrained(source)
        return deepcopy(tiny_original_qwen().config)

    def load_model(source, *, config, **kwargs):
        del kwargs
        source = Path(source)
        if source.is_dir() and (source / "model.safetensors").is_file():
            model = _tiny_model(config)
            model.load_state_dict(load_file(source / "model.safetensors"), strict=True)
            return model
        return _tiny_model(config)

    def collator_factory(processor, token_ids, latent_size):
        del processor
        assert latent_size == 2
        return TinyLifecycleCollator(token_ids)

    def trainer_factory(**kwargs):
        target_builder = TinyLifecycleTargetBuilder()
        target_builders.append(target_builder)
        return CustomTrainerStage1(
            **kwargs,
            target_builder=target_builder,
        )

    return replace(
        DEFAULT_STAGE1_DEPENDENCIES,
        load_processor=load_processor,
        load_config=load_config,
        load_model=load_model,
        prepare_dataset=lambda data_path, image_root: list(range(8)),
        collator_factory=collator_factory,
        trainer_factory=trainer_factory,
    )


def _stage1_args(
    tmp_path,
    *,
    method,
    output_dir,
    batch_size,
    gradient_accumulation_steps,
    max_steps,
    save_steps,
    resume=None,
    legacy_lora=False,
):
    data_path = tmp_path / "TRAIN.jsonl"
    data_path.write_text("{}\n", encoding="utf-8")
    image_root = tmp_path / "images"
    image_root.mkdir(exist_ok=True)
    argv = [
        "--stage", "stage1",
        "--model", "tiny/original-qwen",
        "--data_path", str(data_path),
        "--image_root", str(image_root),
        "--save_model_path", str(output_dir),
        "--latent_size", "2",
        "--per_device_train_batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--max_steps", str(max_steps),
        "--save_strategy", "steps",
        "--save_steps", str(save_steps),
        "--attention_implementation", "eager",
        "--lora_r", "4",
        "--lora_alpha", "8",
        "--lora_dropout", "0",
        "--sim_weight", "0.5",
        "--seed", "47",
        "--use_cpu",
    ]
    if legacy_lora:
        argv.append("--use_lora")
    else:
        argv.extend(("--tuning_method", method))
    if resume is not None:
        argv.extend(("--resume_from_checkpoint_path", str(resume)))
    return parse_args(argv)


def tiny_stage1_run(
    tmp_path,
    *,
    method,
    batch_size,
    gradient_accumulation_steps,
    max_steps,
    save_steps,
    output_dir,
    legacy_lora=False,
):
    target_builders = []
    args = _stage1_args(
        tmp_path,
        method=method,
        output_dir=output_dir,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        save_steps=save_steps,
        legacy_lora=legacy_lora,
    )
    components = build_stage1_training_components(
        args, dependencies=_tiny_dependencies(target_builders)
    )
    assert type(components.trainer) is CustomTrainerStage1
    components.trainer.args.save_total_limit = 2
    result = components.trainer.train()
    builder = target_builders[0]
    visual = resolve_qwen2_5_vl_model(components.model).visual
    checkpoint = Path(output_dir) / "checkpoint-1"
    return TinyStage1Run(
        components=components,
        checkpoint_1=checkpoint,
        student_batch_sizes=list(builder.batch_sizes),
        original_visual_calls=visual.calls,
        helper_visual_calls=builder.helper_visual_calls,
        micro_batches=len(builder.batch_sizes),
        losses=[float(result.metrics["train_loss"])],
    )


def tiny_stage1_resume(tmp_path, run, *, method, max_steps):
    target_builders = []
    args = _stage1_args(
        tmp_path,
        method=method,
        output_dir=run.checkpoint_1.parent,
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=max_steps,
        save_steps=10,
        resume=run.checkpoint_1,
    )
    components = build_stage1_training_components(
        args, dependencies=_tiny_dependencies(target_builders)
    )
    witness = ResumeStateWitness()
    components.trainer.add_callback(witness)
    components.trainer.train(resume_from_checkpoint=str(run.checkpoint_1))
    return TinyResumedRun(
        components=components,
        initial_global_step=witness.initial_global_step,
        final_global_step=components.trainer.state.global_step,
        optimizer_state_restored=witness.optimizer_state_restored,
        scheduler_state_restored=witness.scheduler_state_restored,
    )


@pytest.mark.parametrize("method", ["full", "dora"])
def test_tiny_stage1_b2_train_save_resume_end_to_end(tmp_path, method):
    run = tiny_stage1_run(
        tmp_path,
        method=method,
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=2,
        save_steps=1,
        output_dir=tmp_path / method,
    )

    assert run.student_batch_sizes == [2, 2, 2, 2]
    assert run.original_visual_calls == run.micro_batches
    assert run.helper_visual_calls == run.micro_batches
    assert torch.isfinite(torch.tensor(run.losses)).all()
    assert run.checkpoint_1.is_dir()

    resumed = tiny_stage1_resume(tmp_path, run, method=method, max_steps=3)

    assert resumed.initial_global_step == 1
    assert resumed.final_global_step == 3
    assert resumed.optimizer_state_restored
    assert resumed.scheduler_state_restored


def test_fresh_stage2_handoff_resets_training_state(tmp_path):
    run = tiny_stage1_run(
        tmp_path,
        method="dora",
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=1,
        save_steps=1,
        output_dir=tmp_path / "stage1-dora",
    )
    processor = TinyIntegrationProcessor.from_pretrained(run.checkpoint_1)
    expected = load_stage1_peft_artifact(
        tiny_original_qwen(), processor, run.checkpoint_1, is_trainable=True
    )
    stage2 = load_stage1_dora_for_fresh_stage2(
        tiny_original_qwen(),
        processor,
        run.checkpoint_1,
        Stage2TuningConfig(
            method="dora",
            r=4,
            alpha=8,
            dropout=0.0,
            target_modules=(
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ),
        ),
    )
    expected_rows = find_sparse_output_token_rows(expected)
    actual_rows = find_sparse_output_token_rows(stage2)
    expected_trainable = {
        name: parameter.detach().clone()
        for name, parameter in expected.named_parameters()
        if parameter.requires_grad
    }
    actual_trainable = {
        name: parameter.detach().clone()
        for name, parameter in stage2.named_parameters()
        if parameter.requires_grad
    }
    trainer = Stage2Trainer(
        model=stage2,
        args=TrainingArguments(
            output_dir=str(tmp_path / "stage2"), report_to=[], use_cpu=True
        ),
    )

    assert actual_trainable.keys() == expected_trainable.keys()
    for name in expected_trainable:
        torch.testing.assert_close(actual_trainable[name], expected_trainable[name])
    torch.testing.assert_close(actual_rows.output_delta, expected_rows.output_delta)
    assert trainer.state.global_step == 0
    assert trainer.optimizer is None


def test_legacy_stage1_b1_loss_fixture_remains_finite(tmp_path):
    run = tiny_stage1_run(
        tmp_path,
        method="full",
        batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        save_steps=1,
        output_dir=tmp_path / "legacy-b1",
    )

    assert run.student_batch_sizes == [1]
    assert run.losses == pytest.approx([5.19940710067749], rel=0, abs=1e-6)


def test_legacy_use_lora_one_step_saves_adapter_checkpoint(tmp_path):
    run = tiny_stage1_run(
        tmp_path,
        method="lora",
        batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        save_steps=1,
        output_dir=tmp_path / "legacy-lora",
        legacy_lora=True,
    )

    assert classify_stage1_artifact(run.checkpoint_1) is Stage1ArtifactKind.PEFT
    assert run.components.model.peft_config["default"].use_dora is False
    assert (run.checkpoint_1 / "adapter_model.safetensors").is_file()
    assert not (run.checkpoint_1 / "model.safetensors").exists()
