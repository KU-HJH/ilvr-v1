from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.state import AcceleratorState, GradientState, PartialState
from safetensors.torch import load_file
from torch import nn
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers import TrainerCallback, TrainingArguments
import transformers.integrations.deepspeed as transformers_deepspeed
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
from stage1.targets import Stage1TeacherTargetBuilder
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
def release_accelerate_state_after_lifecycle_test():
    """Release state owned here without hiding pollution from earlier tests."""
    assert "ACCELERATE_USE_DEEPSPEED" not in os.environ
    assert not AcceleratorState._shared_state
    assert not PartialState._shared_state
    assert not GradientState._shared_state
    assert transformers_deepspeed._hf_deepspeed_config_weak_ref is None
    try:
        yield
    finally:
        os.environ.pop("ACCELERATE_USE_DEEPSPEED", None)
        unset_hf_deepspeed_config()
        AcceleratorState._reset_state(reset_partial_state=True)
        PartialState._reset_state()
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
        self.text_calls = []
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

    def __call__(self, text, *, return_tensors):
        assert return_tensors == "pt"
        self.text_calls.append(text)
        marker_ids = {
            "<|im_start|>assistant": 1,
            "<|im_start|>assistant\n": 1,
            "<|vision_start|>": 81,
            "<|vision_end|>": 82,
        }
        token_id = marker_ids.get(text, self.convert_tokens_to_ids(text))
        return {"input_ids": torch.tensor([[token_id]], dtype=torch.long)}

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

    def __call__(self, *args, **kwargs):
        raise AssertionError(
            "multimodal processor must not receive Stage 1 text-only target calls"
        )

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

    spatial_merge_size = 1

    def __init__(self, hidden_size=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("anchor", torch.zeros(()))
        self.calls = 0
        self.original_calls = 0
        self.helper_calls = 0
        self.original_batch_sizes = []

    @property
    def dtype(self):
        return self.anchor.dtype

    def forward(self, pixel_values, *, grid_thw):
        self.calls += 1
        tags = pixel_values[:, 0]
        if bool(torch.all(tags == 0).item()):
            self.original_calls += 1
            self.original_batch_sizes.append(int(grid_thw.shape[0]))
        elif bool(torch.all(tags == 1).item()):
            self.helper_calls += 1
        else:
            raise AssertionError("tiny visual inputs must be all original or all helper")
        token_count = int(grid_thw.prod(dim=-1).sum().item())
        if token_count != pixel_values.shape[0]:
            raise AssertionError("tiny visual rows must match the declared grid")
        return pixel_values[:, : self.hidden_size]


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
            "pixel_values": torch.cat(
                (
                    torch.zeros((batch_size * 4, 1)),
                    torch.ones((batch_size * 4, 7)),
                ),
                dim=1,
            ),
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


def _tiny_dependencies(*, tie_word_embeddings=False):
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
            config = Qwen2_5_VLConfig.from_pretrained(source)
        else:
            config = deepcopy(tiny_original_qwen().config)
        config.tie_word_embeddings = tie_word_embeddings
        return config

    def load_model(source, *, config, **kwargs):
        del kwargs
        source = Path(source)
        if source.is_dir() and (source / "model.safetensors").is_file():
            model = _tiny_model(config)
            incompatible = model.load_state_dict(
                load_file(source / "model.safetensors"), strict=False
            )
            expected_missing = (
                {"lm_head.weight"}
                if config.tie_word_embeddings is True
                else set()
            )
            if (
                set(incompatible.missing_keys) != expected_missing
                or incompatible.unexpected_keys
            ):
                raise RuntimeError(
                    "tiny checkpoint topology mismatch: "
                    f"missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            model.tie_weights()
            return model
        return _tiny_model(config)

    def collator_factory(processor, token_ids, latent_size):
        del processor
        assert latent_size == 2
        return TinyLifecycleCollator(token_ids)

    return replace(
        DEFAULT_STAGE1_DEPENDENCIES,
        load_processor=load_processor,
        load_config=load_config,
        load_model=load_model,
        prepare_dataset=lambda data_path, image_root: list(range(8)),
        collator_factory=collator_factory,
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
    tie_word_embeddings=False,
):
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
        args,
        dependencies=_tiny_dependencies(
            tie_word_embeddings=tie_word_embeddings
        ),
    )
    assert type(components.trainer) is CustomTrainerStage1
    assert type(components.trainer.target_builder) is Stage1TeacherTargetBuilder
    assert (
        components.trainer.target_builder.tokenizer
        is components.processor.tokenizer
    )
    assert components.processor.tokenizer.text_calls == [
        "<|im_start|>assistant",
        "<|im_start|>assistant\n",
        "<|latent_pad|>",
        "<|latent_start|>",
        "<|latent_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    components.trainer.args.save_total_limit = 2
    result = components.trainer.train()
    visual = resolve_qwen2_5_vl_model(components.model).visual
    teacher_visual = resolve_qwen2_5_vl_model(
        components.trainer.ema_teacher.model
    ).visual
    checkpoint = Path(output_dir) / "checkpoint-1"
    return TinyStage1Run(
        components=components,
        checkpoint_1=checkpoint,
        student_batch_sizes=list(visual.original_batch_sizes),
        original_visual_calls=visual.original_calls,
        helper_visual_calls=teacher_visual.helper_calls,
        micro_batches=visual.original_calls,
        losses=[float(result.metrics["train_loss"])],
    )


def tiny_stage1_resume(
    tmp_path,
    run,
    *,
    method,
    max_steps,
    source_tie_word_embeddings=None,
):
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
    tied = (
        resolve_qwen2_5_vl_model(
            run.components.model
        ).config.tie_word_embeddings
        if source_tie_word_embeddings is None
        else source_tie_word_embeddings
    )
    components = build_stage1_training_components(
        args, dependencies=_tiny_dependencies(tie_word_embeddings=tied)
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


def test_tiny_tied_qwen_full_b2_train_save_resume_end_to_end(tmp_path):
    run = tiny_stage1_run(
        tmp_path,
        method="full",
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=2,
        save_steps=1,
        output_dir=tmp_path / "full-tied",
        tie_word_embeddings=True,
    )
    qwen = resolve_qwen2_5_vl_model(run.components.model)

    assert qwen.config.tie_word_embeddings is True
    assert qwen.lm_head.weight is qwen.model.embed_tokens.weight
    assert run.student_batch_sizes == [2, 2, 2, 2]
    assert run.checkpoint_1.is_dir()

    resumed = tiny_stage1_resume(tmp_path, run, method="full", max_steps=3)
    resumed_qwen = resolve_qwen2_5_vl_model(resumed.components.model)

    assert resumed.initial_global_step == 1
    assert resumed.final_global_step == 3
    assert resumed.optimizer_state_restored
    assert resumed.scheduler_state_restored
    assert resumed_qwen.lm_head.weight is resumed_qwen.model.embed_tokens.weight


def test_tiny_tied_qwen_dora_b2_train_save_resume_end_to_end(tmp_path):
    run = tiny_stage1_run(
        tmp_path,
        method="dora",
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=2,
        save_steps=1,
        output_dir=tmp_path / "dora-tied",
        tie_word_embeddings=True,
    )
    qwen = resolve_qwen2_5_vl_model(run.components.model)
    teacher_qwen = resolve_qwen2_5_vl_model(
        run.components.trainer.ema_teacher.model
    )

    assert qwen.config.tie_word_embeddings is False
    assert teacher_qwen.config.tie_word_embeddings is False
    assert qwen.lm_head.base_head.weight is not (
        qwen.model.embed_tokens.token_adapter.base_layer.weight
    )
    assert teacher_qwen.lm_head.base_head.weight is not (
        teacher_qwen.model.embed_tokens.token_adapter.base_layer.weight
    )
    assert run.checkpoint_1.is_dir()

    resumed = tiny_stage1_resume(
        tmp_path,
        run,
        method="dora",
        max_steps=3,
        source_tie_word_embeddings=True,
    )
    resumed_qwen = resolve_qwen2_5_vl_model(resumed.components.model)

    assert resumed.initial_global_step == 1
    assert resumed.final_global_step == 3
    assert resumed.optimizer_state_restored
    assert resumed.scheduler_state_restored
    assert resumed_qwen.config.tie_word_embeddings is False
    assert resumed_qwen.lm_head.base_head.weight is not (
        resumed_qwen.model.embed_tokens.token_adapter.base_layer.weight
    )


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
        tiny_original_qwen(),
        processor,
        run.checkpoint_1,
        requested_base_model_reference="tiny/original-qwen",
        is_trainable=True,
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
        requested_base_model_reference="tiny/original-qwen",
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
    assert run.losses == pytest.approx([5.593050479888916], rel=0, abs=1e-6)


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
