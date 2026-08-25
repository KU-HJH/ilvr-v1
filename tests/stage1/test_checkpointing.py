from __future__ import annotations

import json
import random
import shutil
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import peft
import pytest
import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from peft_common import capture_qwen2_5_vl_architecture, resolve_qwen2_5_vl_model
from stage1.checkpointing import (
    STAGE1_PEFT_METADATA_FILENAME,
    STAGE1_PEFT_SCHEMA_VERSION,
    STAGE1_REPOSITORY_FORMAT_VERSION,
    Stage1ArtifactKind,
    Stage1PeftPreflightWitness,
    Stage1PeftMetadata,
    classify_stage1_artifact,
    load_stage1_peft_artifact,
    load_stage1_peft_metadata,
    preflight_stage1_peft_artifact,
    resolve_stage1_resume,
    save_stage1_peft_artifact,
)
from stage1.peft import build_stage1_tuned_model
from stage1.token_rows import find_sparse_output_token_rows
from stage1.trainer import CustomTrainerStage1
from stage1.types import Stage1TuningConfig


BASE_VOCAB_SIZE = 125
RESIZED_VOCAB_SIZE = 128
LATENT_TOKEN_STRINGS = (
    "<|latent_pad|>",
    "<|latent_start|>",
    "<|latent_end|>",
)
LATENT_TOKEN_IDS = (125, 126, 127)


class TinyStage1Tokenizer:
    unk_token = "<unk>"
    unk_token_id = 0

    def __init__(self, token_ids=LATENT_TOKEN_IDS, *, size=RESIZED_VOCAB_SIZE):
        self._ids = dict(zip(LATENT_TOKEN_STRINGS, token_ids))
        self._tokens = {value: key for key, value in self._ids.items()}
        self._size = size

    def __len__(self):
        return self._size

    def convert_tokens_to_ids(self, token):
        return self._ids.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id):
        return self._tokens.get(int(token_id), self.unk_token)

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text(
            json.dumps({"ids": self._ids, "size": self._size}, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, directory):
        payload = json.loads(
            (Path(directory) / "tokenizer.json").read_text(encoding="utf-8")
        )
        token_ids = tuple(payload["ids"][token] for token in LATENT_TOKEN_STRINGS)
        return cls(token_ids, size=payload["size"])


class TinyStage1Processor:
    model_input_names = ["input_ids"]

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or TinyStage1Tokenizer()

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "preprocessor_config.json").write_text(
            '{"processor_class":"TinyStage1Processor"}\n', encoding="utf-8"
        )
        self.tokenizer.save_pretrained(directory)

    @classmethod
    def from_pretrained(cls, directory):
        return cls(TinyStage1Tokenizer.from_pretrained(directory))


def stage1_tuning(method="dora"):
    return Stage1TuningConfig(
        method=method,
        rank=4,
        alpha=8,
        dropout=0.0,
        target_suffixes=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )


def tiny_original_qwen():
    torch.manual_seed(29)
    config = Qwen2_5_VLConfig(
        vocab_size=BASE_VOCAB_SIZE,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 2,
            "out_hidden_size": 8,
            "fullatt_block_indexes": [0],
        },
        image_token_id=80,
        vision_start_token_id=81,
        video_token_id=82,
        stage="stage1",
        latent_size=2,
        attn_implementation="eager",
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 0]},
    )
    config._name_or_path = "tiny/original-qwen"
    return Qwen2_5_VLForConditionalGeneration(config)


def _resolved_names(tuning):
    owners = {
        "q_proj": "self_attn",
        "k_proj": "self_attn",
        "v_proj": "self_attn",
        "o_proj": "self_attn",
        "gate_proj": "mlp",
        "up_proj": "mlp",
        "down_proj": "mlp",
    }
    return tuple(
        f"model.layers.0.{owners[suffix]}.{suffix}"
        for suffix in tuning.target_suffixes
    )


def sample_metadata(base_model=None, tuning=None):
    base_model = base_model or tiny_original_qwen()
    tuning = tuning or stage1_tuning()
    return Stage1PeftMetadata(
        schema_version=STAGE1_PEFT_SCHEMA_VERSION,
        stage="stage1",
        base_model_reference="tiny/original-qwen",
        base_model_architecture=capture_qwen2_5_vl_architecture(base_model.config),
        base_vocab_size=BASE_VOCAB_SIZE,
        resized_vocab_size=RESIZED_VOCAB_SIZE,
        latent_token_ids=LATENT_TOKEN_IDS,
        latent_token_strings=LATENT_TOKEN_STRINGS,
        latent_size=2,
        ce_weight=1.0,
        sim_weight=0.5,
        coverage_p=0.9,
        image_pool_k=8,
        tuning_method=tuning.method,
        rank=tuning.rank,
        alpha=tuning.alpha,
        dropout=tuning.dropout,
        requested_target_suffixes=tuning.target_suffixes,
        resolved_target_names=_resolved_names(tuning),
        input_row_format="peft_trainable_tokens_v1",
        output_row_format="ilvr_sparse_output_rows_v1",
        output_row_shape=(3, 8),
        peft_version=peft.__version__,
        repository_format_version=STAGE1_REPOSITORY_FORMAT_VERSION,
    )


@dataclass(frozen=True)
class Stage1DoraBundle:
    model: nn.Module
    processor: TinyStage1Processor
    metadata: Stage1PeftMetadata


@pytest.fixture
def stage1_dora_bundle():
    base = tiny_original_qwen()
    metadata = sample_metadata(base)
    base.resize_token_embeddings(RESIZED_VOCAB_SIZE)
    base.config.latent_token_id = LATENT_TOKEN_IDS[0]
    base.config.latent_start_id = LATENT_TOKEN_IDS[1]
    base.config.latent_end_id = LATENT_TOKEN_IDS[2]
    model = build_stage1_tuned_model(base, stage1_tuning(), LATENT_TOKEN_IDS)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.fill_(0.01 * (len(name) % 7 + 1))
    return Stage1DoraBundle(
        model=model,
        processor=TinyStage1Processor(),
        metadata=metadata,
    )


@pytest.fixture
def saved_dora_artifact(tmp_path, stage1_dora_bundle):
    save_stage1_peft_artifact(
        stage1_dora_bundle.model,
        stage1_dora_bundle.processor,
        tmp_path,
        stage1_dora_bundle.metadata,
    )
    return tmp_path


def _write_metadata(directory, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / STAGE1_PEFT_METADATA_FILENAME).write_text(
        json.dumps(metadata.to_dict(), sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_fake_peft(directory, metadata, *, weight="adapter_model.safetensors"):
    directory.mkdir(parents=True, exist_ok=True)
    for filename in (
        "adapter_config.json",
        weight,
        "ilvr_token_rows.safetensors",
    ):
        (directory / filename).write_bytes(b"fixture")
    _write_metadata(directory, metadata)


def _write_full_checkpoint(directory, *, complete=True):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.safetensors").write_bytes(b"fixture")
    if complete:
        _write_standard_trainer_state(directory)
    else:
        (directory / "trainer_state.json").write_text("{}\n", encoding="utf-8")


def _write_full_index(directory, *, binary=False, shard_names=None, weight_map=None):
    directory.mkdir(parents=True, exist_ok=True)
    if shard_names is None:
        stem = "pytorch_model" if binary else "model"
        suffix = "bin" if binary else "safetensors"
        shard_names = [
            f"{stem}-00001-of-00002.{suffix}",
            f"{stem}-00002-of-00002.{suffix}",
        ]
    for name in shard_names:
        (directory / name).write_bytes(b"fixture")
    if weight_map is None:
        weight_map = {
            "model.embed_tokens.weight": shard_names[0],
            "lm_head.weight": shard_names[-1],
        }
    index_name = (
        "pytorch_model.bin.index.json"
        if binary
        else "model.safetensors.index.json"
    )
    (directory / index_name).write_text(
        json.dumps({"metadata": {"total_size": 14}, "weight_map": weight_map})
        + "\n",
        encoding="utf-8",
    )


def _write_standard_trainer_state(directory, *, rank_count=None):
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
        (directory / filename).write_bytes(b"fixture")
    if rank_count is None:
        (directory / "rng_state.pth").write_bytes(b"fixture")
    else:
        for rank in range(rank_count):
            (directory / f"rng_state_{rank}.pth").write_bytes(b"fixture")


def _write_deepspeed_trainer_state(directory, *, step, replicated_model=False):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "trainer_state.json").write_text(
        json.dumps({"global_step": step}) + "\n", encoding="utf-8"
    )
    (directory / "rng_state_0.pth").write_bytes(b"fixture")
    (directory / "rng_state_1.pth").write_bytes(b"fixture")
    (directory / "latest").write_text(
        f"global_step{step}", encoding="utf-8"
    )
    tag = directory / f"global_step{step}"
    tag.mkdir()
    if replicated_model:
        (tag / "mp_rank_00_model_states.pt").write_bytes(b"fixture")
    else:
        for rank in (0, 1):
            (tag / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt").write_bytes(
                b"fixture"
            )
    for rank in (0, 1):
        (tag / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").write_bytes(
            b"fixture"
        )


def test_metadata_schema_v1_round_trips_exact_fields(tmp_path):
    metadata = sample_metadata()

    _write_metadata(tmp_path, metadata)

    assert load_stage1_peft_metadata(tmp_path) == metadata
    assert set(metadata.to_dict()) == {
        "schema_version",
        "stage",
        "base_model_reference",
        "base_model_architecture",
        "base_vocab_size",
        "resized_vocab_size",
        "latent_token_ids",
        "latent_token_strings",
        "latent_size",
        "ce_weight",
        "sim_weight",
        "coverage_p",
        "image_pool_k",
        "tuning_method",
        "rank",
        "alpha",
        "dropout",
        "requested_target_suffixes",
        "resolved_target_names",
        "input_row_format",
        "output_row_format",
        "output_row_shape",
        "peft_version",
        "repository_format_version",
    }


@pytest.mark.parametrize("mutation", ["schema", "missing", "extra"])
def test_metadata_fails_closed_on_unknown_or_partial_schema(tmp_path, mutation):
    payload = sample_metadata().to_dict()
    if mutation == "schema":
        payload["schema_version"] = 999
    elif mutation == "missing":
        payload.pop("coverage_p")
    else:
        payload["future_field"] = True
    (tmp_path / STAGE1_PEFT_METADATA_FILENAME).write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="schema|fields"):
        load_stage1_peft_metadata(tmp_path)


def test_metadata_rejects_full_tuning_as_an_adapter_schema():
    payload = sample_metadata().to_dict()
    payload["tuning_method"] = "full"

    with pytest.raises(ValueError, match="LoRA or DoRA"):
        Stage1PeftMetadata.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("stage", True),
        ("base_model_architecture", True),
        ("base_vocab_size", True),
        ("resized_vocab_size", False),
        ("latent_size", False),
        ("image_pool_k", True),
        ("rank", False),
        ("alpha", True),
        ("repository_format_version", False),
        ("ce_weight", True),
        ("sim_weight", False),
        ("coverage_p", True),
        ("dropout", False),
        ("base_model_reference", 1),
        ("latent_token_ids", [125, 126, True]),
        ("latent_token_strings", ["<|latent_pad|>", True, "<|latent_end|>"]),
        ("tuning_method", True),
        ("requested_target_suffixes", ["q_proj", True]),
        ("resolved_target_names", ["model.layers.0.self_attn.q_proj", True]),
        ("input_row_format", True),
        ("output_row_format", False),
        ("output_row_shape", [3, True]),
        ("peft_version", 1),
        ("latent_token_ids", (125, 126, 127)),
    ],
)
def test_metadata_rejects_json_scalar_type_confusion(field, value):
    payload = sample_metadata().to_dict()
    payload[field] = value

    with pytest.raises(ValueError):
        Stage1PeftMetadata.from_dict(payload)


def test_dora_artifact_is_adapter_only_and_complete(
    tmp_path, stage1_dora_bundle
):
    save_stage1_peft_artifact(
        stage1_dora_bundle.model,
        stage1_dora_bundle.processor,
        tmp_path,
        stage1_dora_bundle.metadata,
    )

    assert classify_stage1_artifact(tmp_path) is Stage1ArtifactKind.PEFT
    assert (tmp_path / "adapter_config.json").is_file()
    assert (tmp_path / "adapter_model.safetensors").is_file()
    assert (tmp_path / "ilvr_token_rows.safetensors").is_file()
    assert (tmp_path / STAGE1_PEFT_METADATA_FILENAME).is_file()
    assert (tmp_path / "tokenizer.json").is_file()
    assert (tmp_path / "preprocessor_config.json").is_file()
    assert not (tmp_path / "model.safetensors").exists()
    assert not (tmp_path / "pytorch_model.bin").exists()


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("adapter_model.safetensors", "incomplete"),
        ("ilvr_token_rows.safetensors", "token-row"),
        (STAGE1_PEFT_METADATA_FILENAME, "metadata"),
    ],
)
def test_incomplete_adapter_fails_before_peft_load(
    saved_dora_artifact, missing, message, monkeypatch
):
    (saved_dora_artifact / missing).unlink()
    calls = 0

    def reject_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("large/base adapter load must not run")

    monkeypatch.setattr(PeftModel, "from_pretrained", reject_load)
    base = tiny_original_qwen()
    with pytest.raises(ValueError, match=message):
        load_stage1_peft_artifact(
            base,
            TinyStage1Processor.from_pretrained(saved_dora_artifact),
            saved_dora_artifact,
            is_trainable=True,
        )
    assert calls == 0
    assert base.get_input_embeddings().num_embeddings == BASE_VOCAB_SIZE


def test_classifier_rejects_mixed_full_and_peft_markers_before_metadata(
    tmp_path,
):
    _write_fake_peft(tmp_path, sample_metadata())
    (tmp_path / "model.safetensors").write_bytes(b"fixture")

    with pytest.raises(ValueError, match="mixed.*full.*PEFT"):
        classify_stage1_artifact(tmp_path)


def test_classifier_rejects_legacy_binary_adapter(tmp_path):
    _write_fake_peft(tmp_path, sample_metadata(), weight="adapter_model.bin")

    with pytest.raises(ValueError, match="legacy binary.*safetensors"):
        classify_stage1_artifact(tmp_path)


def test_classifier_rejects_unknown_model_weight_marker(tmp_path):
    _write_fake_peft(tmp_path, sample_metadata())
    (tmp_path / "unowned_weights.safetensors").write_bytes(b"fixture")

    with pytest.raises(ValueError, match="unknown.*weight"):
        classify_stage1_artifact(tmp_path)


def test_classifier_accepts_exact_rank_qualified_rng_state_marker(tmp_path):
    _write_full_checkpoint(tmp_path)
    (tmp_path / "rng_state_7.pth").write_bytes(b"fixture")

    assert classify_stage1_artifact(tmp_path) is Stage1ArtifactKind.FULL


@pytest.mark.parametrize(
    "filename",
    [
        "rng_state_.pth",
        "rng_state_-1.pth",
        "rng_state_01.pth",
        "rng_state_1_copy.pth",
        "prefix_rng_state_1.pth",
    ],
)
def test_classifier_rejects_inexact_rank_rng_state_marker(tmp_path, filename):
    _write_full_checkpoint(tmp_path)
    (tmp_path / filename).write_bytes(b"fixture")

    with pytest.raises(ValueError, match="unknown.*weight"):
        classify_stage1_artifact(tmp_path)


def test_classifier_accepts_full_final_and_numbered_checkpoint(tmp_path):
    final = tmp_path / "final"
    numbered = tmp_path / "checkpoint-7"
    _write_full_index(final)
    _write_full_checkpoint(numbered)

    assert classify_stage1_artifact(final) is Stage1ArtifactKind.FULL
    assert classify_stage1_artifact(numbered) is Stage1ArtifactKind.FULL


@pytest.mark.parametrize("filename", ["model.safetensors", "pytorch_model.bin"])
def test_classifier_accepts_exactly_one_standalone_full_family(tmp_path, filename):
    (tmp_path / filename).write_bytes(b"fixture")

    assert classify_stage1_artifact(tmp_path) is Stage1ArtifactKind.FULL


@pytest.mark.parametrize("filename", ["model.safetensors", "pytorch_model.bin"])
@pytest.mark.parametrize("target_location", ["inside", "outside"])
def test_classifier_rejects_symlink_standalone_full_weight(
    tmp_path, filename, target_location
):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    target = (
        artifact / "payload"
        if target_location == "inside"
        else tmp_path / "outside-payload"
    )
    target.write_bytes(b"fixture")
    (artifact / filename).symlink_to(target)

    with pytest.raises(ValueError, match="full|standalone|symlink|regular"):
        classify_stage1_artifact(artifact)


@pytest.mark.parametrize("binary", [False, True])
def test_classifier_accepts_one_complete_sharded_full_family(tmp_path, binary):
    _write_full_index(tmp_path, binary=binary)

    assert classify_stage1_artifact(tmp_path) is Stage1ArtifactKind.FULL


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_index",
        "malformed_index",
        "missing_shard",
        "orphan_shard",
        "escaping_reference",
        "empty_reference",
        "mixed_standalone",
        "mixed_binary_family",
    ],
)
def test_classifier_rejects_incoherent_full_serialization_family(
    tmp_path, mutation
):
    _write_full_index(tmp_path)
    index = tmp_path / "model.safetensors.index.json"
    if mutation == "empty_index":
        index.write_text("{}\n", encoding="utf-8")
    elif mutation == "malformed_index":
        index.write_text("{\n", encoding="utf-8")
    elif mutation == "missing_shard":
        (tmp_path / "model-00002-of-00002.safetensors").unlink()
    elif mutation == "orphan_shard":
        (tmp_path / "model-00003-of-00003.safetensors").write_bytes(b"fixture")
    elif mutation == "escaping_reference":
        _write_full_index(
            tmp_path,
            weight_map={"lm_head.weight": "../model-00001-of-00002.safetensors"},
        )
    elif mutation == "empty_reference":
        _write_full_index(tmp_path, weight_map={"lm_head.weight": ""})
    elif mutation == "mixed_standalone":
        (tmp_path / "model.safetensors").write_bytes(b"fixture")
    else:
        (tmp_path / "pytorch_model.bin").write_bytes(b"fixture")

    with pytest.raises(ValueError, match="full|index|shard|serialization|mixed"):
        classify_stage1_artifact(tmp_path)


def test_classifier_accepts_peft_final_and_numbered_checkpoint(
    saved_dora_artifact, tmp_path
):
    numbered = tmp_path / "numbered" / "checkpoint-4"
    shutil.copytree(saved_dora_artifact, numbered)
    (numbered / "trainer_state.json").write_text("{}\n", encoding="utf-8")

    assert classify_stage1_artifact(saved_dora_artifact) is Stage1ArtifactKind.PEFT
    assert classify_stage1_artifact(numbered) is Stage1ArtifactKind.PEFT


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("architecture", "architecture mismatch.*hidden_act"),
        ("token_id", "latent token.*mismatch"),
        ("rank", "rank mismatch"),
        ("resolved_order", "resolved target.*mismatch"),
        ("output_shape", "output_row_shape"),
        ("row_format", "output_row_format"),
        ("repository_schema", "repository format"),
        ("peft_version", "PEFT version"),
    ],
)
def test_load_validates_architecture_token_shape_tuning_and_schema_before_peft(
    saved_dora_artifact, mutation, message, monkeypatch
):
    base = tiny_original_qwen()
    processor = TinyStage1Processor.from_pretrained(saved_dora_artifact)
    payload = load_stage1_peft_metadata(saved_dora_artifact).to_dict()
    if mutation == "architecture":
        base.config.hidden_act = "gelu"
    elif mutation == "token_id":
        processor = TinyStage1Processor(
            TinyStage1Tokenizer((124, 126, 127), size=RESIZED_VOCAB_SIZE)
        )
    elif mutation == "rank":
        payload["rank"] = 5
    elif mutation == "resolved_order":
        payload["resolved_target_names"] = list(
            reversed(payload["resolved_target_names"])
        )
    elif mutation == "output_shape":
        payload["output_row_shape"] = [3, 9]
    elif mutation == "row_format":
        payload["output_row_format"] = "future_rows_v2"
    elif mutation == "repository_schema":
        payload["repository_format_version"] = 999
    else:
        payload["peft_version"] = "0.0.0"
    if mutation not in ("architecture", "token_id"):
        (saved_dora_artifact / STAGE1_PEFT_METADATA_FILENAME).write_text(
            json.dumps(payload) + "\n", encoding="utf-8"
        )

    calls = 0

    def reject_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("PEFT load must not run")

    monkeypatch.setattr(PeftModel, "from_pretrained", reject_load)
    with pytest.raises(ValueError, match=message):
        load_stage1_peft_artifact(
            base,
            processor,
            saved_dora_artifact,
            is_trainable=True,
        )
    assert calls == 0
    assert base.get_input_embeddings().num_embeddings == BASE_VOCAB_SIZE


def test_load_rejects_sparse_row_tensor_shape_before_peft(
    saved_dora_artifact, monkeypatch
):
    save_file(
        {
            "token_ids": torch.tensor(LATENT_TOKEN_IDS, dtype=torch.int64),
            "output_delta": torch.zeros((3, 7), dtype=torch.float32),
        },
        saved_dora_artifact / "ilvr_token_rows.safetensors",
        metadata={"format_version": "1", "training_dtype": "float32"},
    )
    calls = 0

    def reject_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("PEFT load must not run")

    monkeypatch.setattr(PeftModel, "from_pretrained", reject_load)
    with pytest.raises(ValueError, match="sparse output_delta shape mismatch"):
        load_stage1_peft_artifact(
            tiny_original_qwen(),
            TinyStage1Processor.from_pretrained(saved_dora_artifact),
            saved_dora_artifact,
            is_trainable=True,
        )
    assert calls == 0


def test_load_rejects_extra_adapter_tensor_before_peft(
    saved_dora_artifact, monkeypatch
):
    tensors = load_file(saved_dora_artifact / "adapter_model.safetensors")
    tensors["base_model.model.rogue.weight"] = torch.ones(1)
    save_file(tensors, saved_dora_artifact / "adapter_model.safetensors")
    calls = 0

    def reject_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("PEFT load must not run")

    monkeypatch.setattr(PeftModel, "from_pretrained", reject_load)
    with pytest.raises(ValueError, match="unexpected adapter tensor"):
        load_stage1_peft_artifact(
            tiny_original_qwen(),
            TinyStage1Processor.from_pretrained(saved_dora_artifact),
            saved_dora_artifact,
            is_trainable=True,
        )
    assert calls == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("architecture", "architecture mismatch.*hidden_act"),
        ("adapter_tensor", "adapter target shape mismatch"),
        ("sparse_rows", "sparse output_delta shape mismatch"),
    ],
)
def test_preflight_rejects_corrupt_artifact_before_any_large_model_load(
    saved_dora_artifact, mutation, message, monkeypatch
):
    """Catches malformed small artifacts reaching base-weight loading."""
    if mutation == "architecture":
        payload = load_stage1_peft_metadata(saved_dora_artifact).to_dict()
        payload["base_model_architecture"]["decoder"]["hidden_act"] = "gelu"
        (saved_dora_artifact / STAGE1_PEFT_METADATA_FILENAME).write_text(
            json.dumps(payload) + "\n", encoding="utf-8"
        )
    elif mutation == "adapter_tensor":
        path = saved_dora_artifact / "adapter_model.safetensors"
        tensors = load_file(path)
        key = next(name for name in tensors if name.endswith("lora_A.weight"))
        tensors[key] = tensors[key][:, :-1].contiguous()
        save_file(tensors, path)
    else:
        save_file(
            {
                "token_ids": torch.tensor(LATENT_TOKEN_IDS, dtype=torch.int64),
                "output_delta": torch.zeros((3, 7), dtype=torch.float32),
            },
            saved_dora_artifact / "ilvr_token_rows.safetensors",
            metadata={"format_version": "1", "training_dtype": "float32"},
        )

    large_loads = 0

    def reject_large_load(*args, **kwargs):
        nonlocal large_loads
        large_loads += 1
        raise AssertionError("large base weights must not load during preflight")

    monkeypatch.setattr(
        Qwen2_5_VLForConditionalGeneration,
        "from_pretrained",
        reject_large_load,
    )
    base_config = deepcopy(tiny_original_qwen().config)

    with pytest.raises(ValueError, match=message):
        preflight_stage1_peft_artifact(
            TinyStage1Processor.from_pretrained(saved_dora_artifact),
            base_config,
            saved_dora_artifact,
        )

    assert large_loads == 0


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_peft_loader_reuses_preflight_witness_without_revalidating_files(
    saved_dora_artifact, monkeypatch
):
    """Catches the loader discarding its completed small-artifact witness."""
    processor = TinyStage1Processor.from_pretrained(saved_dora_artifact)
    base = tiny_original_qwen()
    witness = preflight_stage1_peft_artifact(
        processor,
        deepcopy(base.config),
        saved_dora_artifact,
    )

    assert isinstance(witness, Stage1PeftPreflightWitness)
    monkeypatch.setattr(
        "stage1.checkpointing.classify_stage1_artifact",
        lambda *args, **kwargs: pytest.fail("artifact was reclassified"),
    )
    monkeypatch.setattr(
        "stage1.checkpointing.load_stage1_peft_metadata",
        lambda *args, **kwargs: pytest.fail("metadata was reread"),
    )
    monkeypatch.setattr(
        "stage1.checkpointing._validate_adapter_tensors",
        lambda *args, **kwargs: pytest.fail("adapter tensors were revalidated"),
    )
    monkeypatch.setattr(
        "stage1.checkpointing._validate_sparse_artifact",
        lambda *args, **kwargs: pytest.fail("sparse rows were revalidated"),
    )

    restored = load_stage1_peft_artifact(
        base,
        processor,
        saved_dora_artifact,
        is_trainable=True,
        preflight=witness,
    )

    assert isinstance(restored, PeftModel)


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_processor_and_adapter_rows_round_trip_without_reassigning_tokens(
    saved_dora_artifact, stage1_dora_bundle
):
    restored_processor = TinyStage1Processor.from_pretrained(saved_dora_artifact)
    before_ids = tuple(
        restored_processor.tokenizer.convert_tokens_to_ids(token)
        for token in LATENT_TOKEN_STRINGS
    )

    restored = load_stage1_peft_artifact(
        tiny_original_qwen(),
        restored_processor,
        saved_dora_artifact,
        is_trainable=True,
    )

    assert before_ids == LATENT_TOKEN_IDS
    assert tuple(
        restored_processor.tokenizer.convert_tokens_to_ids(token)
        for token in LATENT_TOKEN_STRINGS
    ) == LATENT_TOKEN_IDS
    source_state = {
        name: value.detach().cpu()
        for name, value in stage1_dora_bundle.model.state_dict().items()
        if "lora_" in name or "trainable_tokens_delta" in name
    }
    restored_state = {
        name: value.detach().cpu()
        for name, value in restored.state_dict().items()
        if "lora_" in name or "trainable_tokens_delta" in name
    }
    assert restored_state.keys() == source_state.keys()
    for name in source_state:
        assert torch.equal(restored_state[name], source_state[name]), name
    assert torch.equal(
        find_sparse_output_token_rows(restored).output_delta.detach().cpu(),
        find_sparse_output_token_rows(
            stage1_dora_bundle.model
        ).output_delta.detach().cpu(),
    )
    assert resolve_qwen2_5_vl_model(
        restored
    ).model.embed_tokens.token_adapter.token_indices["default"] == list(
        LATENT_TOKEN_IDS
    )


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_adapter_restore_preserves_effective_latent_output_rows_across_resize_rng(
    tmp_path,
):
    source_base = tiny_original_qwen()
    metadata = sample_metadata(source_base)
    torch.manual_seed(101)
    source_base.resize_token_embeddings(RESIZED_VOCAB_SIZE)
    source_base.config.latent_token_id = LATENT_TOKEN_IDS[0]
    source_base.config.latent_start_id = LATENT_TOKEN_IDS[1]
    source_base.config.latent_end_id = LATENT_TOKEN_IDS[2]
    torch.manual_seed(303)
    source = build_stage1_tuned_model(
        source_base, stage1_tuning(), LATENT_TOKEN_IDS
    )
    source_rows = find_sparse_output_token_rows(source)
    assert torch.count_nonzero(source_rows.output_delta) == 0
    save_stage1_peft_artifact(
        source, TinyStage1Processor(), tmp_path, metadata
    )

    restore_base = tiny_original_qwen()
    torch.manual_seed(202)
    restored = load_stage1_peft_artifact(
        restore_base,
        TinyStage1Processor.from_pretrained(tmp_path),
        tmp_path,
        is_trainable=True,
    )
    restored_rows = find_sparse_output_token_rows(restored)
    token_index = torch.tensor(LATENT_TOKEN_IDS)
    source_base_rows = source_rows.base_head.weight[token_index]
    restored_base_rows = restored_rows.base_head.weight[token_index]
    source_effective = source_base_rows + source_rows.output_delta
    restored_effective = restored_base_rows + restored_rows.output_delta
    hidden = torch.tensor(
        [[0.5, -0.25, 1.0, 0.75, -1.0, 0.125, 0.25, -0.5]]
    )

    torch.testing.assert_close(source_base_rows, restored_base_rows, rtol=0, atol=0)
    torch.testing.assert_close(source_effective, restored_effective, rtol=0, atol=0)
    torch.testing.assert_close(
        source_rows(hidden)[..., token_index],
        restored_rows(hidden)[..., token_index],
        rtol=0,
        atol=0,
    )


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_load_honors_is_trainable_false(saved_dora_artifact):
    restored = load_stage1_peft_artifact(
        tiny_original_qwen(),
        TinyStage1Processor.from_pretrained(saved_dora_artifact),
        saved_dora_artifact,
        is_trainable=False,
    )

    assert not any(parameter.requires_grad for parameter in restored.parameters())


def test_resolve_explicit_numbered_resume_wins(tmp_path):
    output = tmp_path / "output"
    explicit = tmp_path / "other" / "checkpoint-2"
    _write_full_checkpoint(output / "checkpoint-10")
    _write_full_checkpoint(explicit)

    decision = resolve_stage1_resume(output, explicit)

    assert decision.checkpoint_path == explicit
    assert decision.artifact_kind is Stage1ArtifactKind.FULL
    assert decision.explicit is True


def test_explicit_full_resume_precedence_rejects_incomplete_selected_path(
    tmp_path,
):
    output = tmp_path / "output"
    explicit = tmp_path / "other" / "checkpoint-2"
    _write_full_checkpoint(output / "checkpoint-10")
    _write_full_checkpoint(explicit, complete=False)

    with pytest.raises(ValueError, match="complete|optimizer|scheduler|RNG"):
        resolve_stage1_resume(output, explicit)


def test_explicit_full_resume_retains_normal_trainer_path_semantics(tmp_path):
    explicit = tmp_path / "named-full-trainer-save"
    _write_full_checkpoint(explicit)

    decision = resolve_stage1_resume(tmp_path / "output", explicit)

    assert decision.checkpoint_path == explicit
    assert decision.artifact_kind is Stage1ArtifactKind.FULL
    assert decision.explicit is True


def test_auto_resume_selects_latest_valid_immediate_numbered_checkpoint_only(
    tmp_path,
):
    output = tmp_path / "output"
    _write_full_checkpoint(output / "checkpoint-2")
    _write_full_checkpoint(output / "checkpoint-11")
    _write_full_checkpoint(output / "checkpoint-not-a-number")
    _write_full_checkpoint(output / "nested" / "checkpoint-99")
    _write_full_checkpoint(tmp_path / "sibling" / "checkpoint-100")
    invalid_latest = output / "checkpoint-12"
    invalid_latest.mkdir(parents=True)
    (invalid_latest / "adapter_config.json").write_text("{}\n")

    decision = resolve_stage1_resume(output, None)

    assert decision.checkpoint_path == output / "checkpoint-11"
    assert decision.explicit is False


def test_auto_resume_skips_incomplete_full_checkpoint(tmp_path):
    output = tmp_path / "output"
    valid = output / "checkpoint-2"
    _write_full_checkpoint(valid)
    _write_full_checkpoint(output / "checkpoint-11", complete=False)

    decision = resolve_stage1_resume(output, None)

    assert decision.checkpoint_path == valid


def test_auto_resume_never_selects_final_adapter_directory(
    saved_dora_artifact,
):
    decision = resolve_stage1_resume(saved_dora_artifact, None)

    assert decision.checkpoint_path is None
    assert decision.artifact_kind is None


def test_explicit_peft_resume_requires_a_numbered_trainer_checkpoint(
    saved_dora_artifact,
):
    with pytest.raises(ValueError, match="numbered.*checkpoint"):
        resolve_stage1_resume(saved_dora_artifact.parent, saved_dora_artifact)


def _copy_numbered_peft(saved_artifact, output, step):
    checkpoint = output / f"checkpoint-{step}"
    shutil.copytree(saved_artifact, checkpoint)
    return checkpoint


def test_explicit_peft_resume_accepts_complete_standard_state(
    tmp_path, saved_dora_artifact
):
    output = tmp_path / "output"
    checkpoint = _copy_numbered_peft(saved_dora_artifact, output, 3)
    _write_standard_trainer_state(checkpoint, rank_count=2)

    decision = resolve_stage1_resume(output, checkpoint)

    assert decision.checkpoint_path == checkpoint
    assert decision.artifact_kind is Stage1ArtifactKind.PEFT


@pytest.mark.parametrize(
    "missing", ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]
)
def test_peft_resume_rejects_incomplete_standard_trainer_state(
    tmp_path, saved_dora_artifact, missing
):
    output = tmp_path / "output"
    checkpoint = _copy_numbered_peft(saved_dora_artifact, output, 4)
    _write_standard_trainer_state(checkpoint)
    (checkpoint / missing).unlink()

    with pytest.raises(ValueError, match="complete|optimizer|scheduler|RNG"):
        resolve_stage1_resume(output, checkpoint)


@pytest.mark.parametrize("layout", ["mixed", "rank_gap"])
def test_peft_resume_rejects_ambiguous_or_incomplete_rng_layout(
    tmp_path, saved_dora_artifact, layout
):
    output = tmp_path / "output"
    checkpoint = _copy_numbered_peft(saved_dora_artifact, output, 5)
    _write_standard_trainer_state(checkpoint)
    if layout == "mixed":
        (checkpoint / "rng_state_0.pth").write_bytes(b"fixture")
    else:
        (checkpoint / "rng_state.pth").unlink()
        (checkpoint / "rng_state_0.pth").write_bytes(b"fixture")
        (checkpoint / "rng_state_2.pth").write_bytes(b"fixture")

    with pytest.raises(ValueError, match="RNG|complete"):
        resolve_stage1_resume(output, checkpoint)


def test_peft_resume_accepts_explicitly_validated_deepspeed_layout(
    tmp_path, saved_dora_artifact
):
    output = tmp_path / "output"
    checkpoint = _copy_numbered_peft(saved_dora_artifact, output, 6)
    _write_deepspeed_trainer_state(checkpoint, step=6)

    decision = resolve_stage1_resume(output, checkpoint)

    assert decision.checkpoint_path == checkpoint
    assert decision.artifact_kind is Stage1ArtifactKind.PEFT


def test_peft_resume_accepts_mp_replicated_deepspeed_layout(
    tmp_path, saved_dora_artifact
):
    output = tmp_path / "output"
    checkpoint = _copy_numbered_peft(saved_dora_artifact, output, 6)
    _write_deepspeed_trainer_state(
        checkpoint, step=6, replicated_model=True
    )

    decision = resolve_stage1_resume(output, checkpoint)

    assert decision.checkpoint_path == checkpoint
    assert decision.artifact_kind is Stage1ArtifactKind.PEFT


def test_peft_resume_accepts_zero3_deepspeed_with_external_scheduler(
    tmp_path, saved_dora_artifact
):
    output = tmp_path / "output"
    checkpoint = _copy_numbered_peft(saved_dora_artifact, output, 6)
    _write_deepspeed_trainer_state(checkpoint, step=6)
    (checkpoint / "scheduler.pt").write_bytes(b"fixture")

    decision = resolve_stage1_resume(output, checkpoint)

    assert decision.checkpoint_path == checkpoint
    assert decision.artifact_kind is Stage1ArtifactKind.PEFT


def test_full_resume_accepts_mp_replicated_deepspeed_with_external_scheduler(
    tmp_path,
):
    output = tmp_path / "output"
    checkpoint = output / "checkpoint-6"
    _write_full_checkpoint(checkpoint, complete=False)
    _write_deepspeed_trainer_state(
        checkpoint, step=6, replicated_model=True
    )
    (checkpoint / "scheduler.pt").write_bytes(b"fixture")

    decision = resolve_stage1_resume(output, checkpoint)

    assert decision.checkpoint_path == checkpoint
    assert decision.artifact_kind is Stage1ArtifactKind.FULL


@pytest.mark.parametrize(
    ("artifact_kind", "scheduler_entry"),
    [("peft", "symlink"), ("full", "directory")],
)
def test_deepspeed_resume_rejects_unsafe_external_scheduler(
    tmp_path, saved_dora_artifact, artifact_kind, scheduler_entry
):
    output = tmp_path / "output"
    if artifact_kind == "peft":
        checkpoint = _copy_numbered_peft(saved_dora_artifact, output, 6)
    else:
        checkpoint = output / "checkpoint-6"
        _write_full_checkpoint(checkpoint, complete=False)
    _write_deepspeed_trainer_state(checkpoint, step=6)
    scheduler = checkpoint / "scheduler.pt"
    if scheduler_entry == "symlink":
        target = checkpoint / "scheduler-payload"
        target.write_bytes(b"fixture")
        scheduler.symlink_to(target)
    else:
        scheduler.mkdir()

    with pytest.raises(ValueError, match="scheduler|complete|mixed|DeepSpeed"):
        resolve_stage1_resume(output, checkpoint)


@pytest.mark.parametrize(
    "mutation",
    [
        "rank0_only",
        "missing_model",
        "missing_optimizer",
        "extra_dp_rank",
        "extra_mp_rank",
        "gapped_rng",
        "tag_step_mismatch",
        "mixed_optimizer",
        "missing_tag",
        "escaping_tag",
    ],
)
def test_peft_resume_rejects_incomplete_deepspeed_layout(
    tmp_path, saved_dora_artifact, mutation
):
    output = tmp_path / "output"
    checkpoint = _copy_numbered_peft(saved_dora_artifact, output, 7)
    _write_deepspeed_trainer_state(checkpoint, step=7)
    tag = checkpoint / "global_step7"
    if mutation == "rank0_only":
        (tag / "zero_pp_rank_1_mp_rank_00_model_states.pt").unlink()
        (tag / "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt").unlink()
    elif mutation == "missing_model":
        (tag / "zero_pp_rank_1_mp_rank_00_model_states.pt").unlink()
    elif mutation == "missing_optimizer":
        (tag / "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt").unlink()
    elif mutation == "extra_dp_rank":
        (tag / "zero_pp_rank_2_mp_rank_00_model_states.pt").write_bytes(b"fixture")
        (tag / "bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt").write_bytes(
            b"fixture"
        )
    elif mutation == "extra_mp_rank":
        (tag / "zero_pp_rank_0_mp_rank_01_model_states.pt").write_bytes(b"fixture")
        (tag / "bf16_zero_pp_rank_0_mp_rank_01_optim_states.pt").write_bytes(
            b"fixture"
        )
    elif mutation == "gapped_rng":
        (checkpoint / "rng_state_1.pth").unlink()
        (checkpoint / "rng_state_2.pth").write_bytes(b"fixture")
    elif mutation == "tag_step_mismatch":
        (checkpoint / "trainer_state.json").write_text(
            '{"global_step":8}\n', encoding="utf-8"
        )
    elif mutation == "mixed_optimizer":
        (checkpoint / "optimizer.pt").write_bytes(b"fixture")
    elif mutation == "missing_tag":
        shutil.rmtree(tag)
    else:
        (checkpoint / "latest").write_text("../outside\n", encoding="utf-8")

    with pytest.raises(ValueError, match="DeepSpeed|complete|mixed"):
        resolve_stage1_resume(output, checkpoint)


@pytest.mark.parametrize("location", ["sibling", "nested"])
def test_explicit_peft_resume_requires_immediate_same_output_checkpoint(
    tmp_path, saved_dora_artifact, location
):
    output = tmp_path / "output"
    if location == "sibling":
        checkpoint = _copy_numbered_peft(saved_dora_artifact, tmp_path / "other", 8)
    else:
        checkpoint = _copy_numbered_peft(saved_dora_artifact, output / "nested", 8)
    _write_standard_trainer_state(checkpoint)

    with pytest.raises(ValueError, match="same output|immediate|outside"):
        resolve_stage1_resume(output, checkpoint)


def test_peft_resume_rejects_symlink_checkpoint_escape(
    tmp_path, saved_dora_artifact
):
    output = tmp_path / "output"
    output.mkdir()
    target = _copy_numbered_peft(saved_dora_artifact, tmp_path / "outside", 9)
    _write_standard_trainer_state(target)
    link = output / "checkpoint-9"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|same output|outside"):
        resolve_stage1_resume(output, link)


def test_auto_resume_skips_symlink_checkpoint_escape(
    tmp_path, saved_dora_artifact
):
    output = tmp_path / "output"
    valid = _copy_numbered_peft(saved_dora_artifact, output, 2)
    _write_standard_trainer_state(valid)
    target = _copy_numbered_peft(saved_dora_artifact, tmp_path / "outside", 10)
    _write_standard_trainer_state(target)
    (output / "checkpoint-10").symlink_to(target, target_is_directory=True)

    decision = resolve_stage1_resume(output, None)

    assert decision.checkpoint_path == valid


class CheckpointLifecycleTrainer(CustomTrainerStage1):
    """Use the real Stage 1 save override with ordinary Trainer state ownership."""

    log = Trainer.log

    def __init__(
        self,
        *,
        stage1_peft_metadata=None,
        stage1_runtime_metadata=None,
        **kwargs,
    ):
        self.stage1_peft_metadata = stage1_peft_metadata
        runtime = stage1_runtime_metadata or stage1_peft_metadata
        if runtime is not None:
            self.ce_weight = runtime.ce_weight
            self.sim_weight = runtime.sim_weight
            self.coverage_p = runtime.coverage_p
            self.image_pool_k = runtime.image_pool_k
        self.collect_stage1_diagnostics = False
        self._latest_stage1_diagnostics = None
        Trainer.__init__(self, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        del num_items_in_batch
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


class StopAfterFirstStep(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if state.global_step == 1:
            control.should_training_stop = True
        return control


def _capture_rng_state():
    numpy_state = np.random.get_state()
    return (
        random.getstate(),
        (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        torch.random.get_rng_state().clone(),
    )


def _assert_rng_state_equal(actual, expected):
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0]
    assert np.array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    assert torch.equal(actual[2], expected[2])


def _assert_nested_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _capture_trainable_state(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _capture_logits(model):
    model.eval()
    batch = _tiny_trainer_batch()
    with torch.no_grad():
        return model(
            input_ids=batch["input_ids"].unsqueeze(0),
            attention_mask=batch["attention_mask"].unsqueeze(0),
        ).logits.detach().cpu()


def _trainer_args(output_dir, *, save_strategy):
    return TrainingArguments(
        output_dir=str(output_dir),
        max_steps=2,
        per_device_train_batch_size=1,
        save_strategy=save_strategy,
        save_steps=1,
        report_to=[],
        disable_tqdm=True,
        use_cpu=True,
        seed=47,
        data_seed=47,
    )


def _tiny_trainer_batch():
    return {
        "input_ids": torch.tensor([1, 125, 4, 126, 5, 127, 6]),
        "attention_mask": torch.ones(7, dtype=torch.long),
        "labels": torch.tensor([1, 125, 4, 126, 5, 127, 6]),
    }


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_stage1_trainer_final_peft_save_is_adapter_only(
    tmp_path, stage1_dora_bundle
):
    trainer = CheckpointLifecycleTrainer(
        model=stage1_dora_bundle.model,
        args=TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True),
        processing_class=stage1_dora_bundle.processor,
        stage1_peft_metadata=stage1_dora_bundle.metadata,
    )

    trainer.save_model(tmp_path)

    assert classify_stage1_artifact(tmp_path) is Stage1ArtifactKind.PEFT
    assert (tmp_path / "tokenizer.json").is_file()
    assert not (tmp_path / "model.safetensors").exists()


def test_stage1_peft_save_uses_logical_zero_shape_before_collective(
    tmp_path, stage1_dora_bundle, monkeypatch
):
    rows = find_sparse_output_token_rows(stage1_dora_bundle.model)
    rows.output_delta.ds_id = 17
    rows.output_delta.ds_shape = torch.Size((3, rows.in_features))
    rows.output_delta.data = torch.empty(0)
    gathered = []

    @contextmanager
    def collective_boundary(parameter):
        gathered.append(parameter)
        raise RuntimeError("collective boundary reached")
        yield

    monkeypatch.setattr(
        "stage1.token_rows._gathered_output_delta", collective_boundary
    )

    with pytest.raises(RuntimeError, match="collective boundary reached"):
        save_stage1_peft_artifact(
            stage1_dora_bundle.model,
            stage1_dora_bundle.processor,
            tmp_path,
            stage1_dora_bundle.metadata,
        )
    assert gathered == [rows.output_delta]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ce_weight", 2.0),
        ("sim_weight", 2.0),
        ("coverage_p", 0.5),
        ("image_pool_k", 4),
    ],
)
def test_stage1_trainer_rejects_mismatched_semantic_metadata_before_save(
    tmp_path, stage1_dora_bundle, field, value
):
    mismatched = replace(stage1_dora_bundle.metadata, **{field: value})
    trainer = CheckpointLifecycleTrainer(
        model=stage1_dora_bundle.model,
        args=TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True),
        processing_class=stage1_dora_bundle.processor,
        stage1_peft_metadata=mismatched,
        stage1_runtime_metadata=stage1_dora_bundle.metadata,
    )

    with pytest.raises(ValueError, match=field):
        trainer.save_model(tmp_path)
    assert not (tmp_path / "adapter_model.safetensors").exists()


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_stage1_trainer_numbered_peft_checkpoint_keeps_resume_state(
    tmp_path, stage1_dora_bundle
):
    initial = stage1_dora_bundle.model
    uninterrupted = CheckpointLifecycleTrainer(
        model=deepcopy(initial),
        args=_trainer_args(tmp_path / "uninterrupted", save_strategy="no"),
        train_dataset=[_tiny_trainer_batch()],
        data_collator=default_data_collator,
        processing_class=stage1_dora_bundle.processor,
        stage1_peft_metadata=stage1_dora_bundle.metadata,
    )
    uninterrupted.train()
    expected_model = _capture_trainable_state(uninterrupted.model)
    expected_optimizer = deepcopy(uninterrupted.optimizer.state_dict())
    expected_scheduler = deepcopy(uninterrupted.lr_scheduler.state_dict())
    expected_rng = _capture_rng_state()
    expected_logits = _capture_logits(uninterrupted.model)

    output = tmp_path / "resumed"
    interrupted = CheckpointLifecycleTrainer(
        model=deepcopy(initial),
        args=_trainer_args(output, save_strategy="steps"),
        train_dataset=[_tiny_trainer_batch()],
        data_collator=default_data_collator,
        processing_class=stage1_dora_bundle.processor,
        stage1_peft_metadata=stage1_dora_bundle.metadata,
        callbacks=[StopAfterFirstStep()],
    )
    interrupted.train()

    checkpoint = output / "checkpoint-1"
    decision = resolve_stage1_resume(output, None)
    assert decision.checkpoint_path == checkpoint
    assert decision.artifact_kind is Stage1ArtifactKind.PEFT
    for filename in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "ilvr_token_rows.safetensors",
        STAGE1_PEFT_METADATA_FILENAME,
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        assert (checkpoint / filename).is_file(), filename
    assert not (checkpoint / "model.safetensors").exists()

    restored = load_stage1_peft_artifact(
        tiny_original_qwen(),
        TinyStage1Processor.from_pretrained(checkpoint),
        checkpoint,
        is_trainable=True,
    )
    resumed = CheckpointLifecycleTrainer(
        model=restored,
        args=_trainer_args(output, save_strategy="no"),
        train_dataset=[_tiny_trainer_batch()],
        data_collator=default_data_collator,
        processing_class=TinyStage1Processor.from_pretrained(checkpoint),
        stage1_peft_metadata=stage1_dora_bundle.metadata,
    )

    resumed.train(resume_from_checkpoint=str(checkpoint))

    assert resumed.state.global_step == 2
    _assert_nested_state_equal(_capture_trainable_state(resumed.model), expected_model)
    _assert_nested_state_equal(resumed.optimizer.state_dict(), expected_optimizer)
    _assert_nested_state_equal(resumed.lr_scheduler.state_dict(), expected_scheduler)
    _assert_rng_state_equal(_capture_rng_state(), expected_rng)
    torch.testing.assert_close(
        _capture_logits(resumed.model), expected_logits, rtol=0, atol=0
    )


def test_stage1_trainer_full_save_retains_normal_trainer_behavior(tmp_path):
    base = tiny_original_qwen()
    base.resize_token_embeddings(RESIZED_VOCAB_SIZE)
    model = build_stage1_tuned_model(base, stage1_tuning("full"), LATENT_TOKEN_IDS)
    trainer = CheckpointLifecycleTrainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True),
        processing_class=TinyStage1Processor(),
    )

    trainer.save_model(tmp_path)

    assert classify_stage1_artifact(tmp_path) is Stage1ArtifactKind.FULL
    assert (tmp_path / "model.safetensors").is_file()
    assert not (tmp_path / STAGE1_PEFT_METADATA_FILENAME).exists()


def test_stage1_trainer_full_numbered_checkpoint_resume_matches_uninterrupted(
    tmp_path,
):
    base = tiny_original_qwen()
    base.resize_token_embeddings(RESIZED_VOCAB_SIZE)
    initial = build_stage1_tuned_model(
        base, stage1_tuning("full"), LATENT_TOKEN_IDS
    )
    processor = TinyStage1Processor()
    dataset = [_tiny_trainer_batch()]
    uninterrupted = CheckpointLifecycleTrainer(
        model=deepcopy(initial),
        args=_trainer_args(tmp_path / "full-uninterrupted", save_strategy="no"),
        train_dataset=dataset,
        data_collator=default_data_collator,
        processing_class=processor,
    )
    uninterrupted.train()
    expected_model = _capture_trainable_state(uninterrupted.model)
    expected_optimizer = deepcopy(uninterrupted.optimizer.state_dict())
    expected_scheduler = deepcopy(uninterrupted.lr_scheduler.state_dict())
    expected_rng = _capture_rng_state()
    expected_logits = _capture_logits(uninterrupted.model)

    output = tmp_path / "full-resumed"
    interrupted = CheckpointLifecycleTrainer(
        model=deepcopy(initial),
        args=_trainer_args(output, save_strategy="steps"),
        train_dataset=dataset,
        data_collator=default_data_collator,
        processing_class=processor,
        callbacks=[StopAfterFirstStep()],
    )
    interrupted.train()
    checkpoint = output / "checkpoint-1"
    decision = resolve_stage1_resume(output, None)
    assert decision.checkpoint_path == checkpoint
    assert decision.artifact_kind is Stage1ArtifactKind.FULL
    for filename in (
        "model.safetensors",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        assert (checkpoint / filename).is_file(), filename

    resumed = CheckpointLifecycleTrainer(
        model=deepcopy(initial),
        args=_trainer_args(output, save_strategy="no"),
        train_dataset=dataset,
        data_collator=default_data_collator,
        processing_class=processor,
    )
    resumed.train(resume_from_checkpoint=str(checkpoint))

    assert resumed.state.global_step == 2
    _assert_nested_state_equal(_capture_trainable_state(resumed.model), expected_model)
    _assert_nested_state_equal(resumed.optimizer.state_dict(), expected_optimizer)
    _assert_nested_state_equal(resumed.lr_scheduler.state_dict(), expected_scheduler)
    _assert_rng_state_equal(_capture_rng_state(), expected_rng)
    torch.testing.assert_close(
        _capture_logits(resumed.model), expected_logits, rtol=0, atol=0
    )


def test_stage1_trainer_keeps_trainer_checkpoint_ownership():
    assert CustomTrainerStage1._save_checkpoint is Trainer._save_checkpoint
