import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from accelerate.state import AcceleratorState, GradientState
from peft import LoraConfig
from torch import nn
from transformers import TrainingArguments, default_data_collator
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

from .helpers import tiny_stage2_qwen
from deepspeed_runtime import (
    DEEPSPEED_RUNTIME_FILENAME,
    canonical_deepspeed_config_path,
    load_deepspeed_runtime_profile,
    write_deepspeed_runtime_sidecar,
)
from src.peft_common import capture_qwen2_5_vl_architecture
from stage2.checkpointing import (
    STAGE2_PEFT_METADATA_FILENAME,
    STAGE2_PEFT_SCHEMA_VERSION,
    Stage2PeftMetadata,
    _capture_base_model_architecture as capture_stage2_base_model_architecture,
    build_stage2_peft_metadata,
    classify_stage2_artifact,
    load_stage2_peft_metadata,
    migrate_stage2_metadata,
    resolve_stage2_load_decision,
    save_stage2_peft_metadata,
    validate_stage2_peft_compatibility,
)
from stage2.peft import (
    Stage2TuningConfig,
    build_stage2_tuned_model,
    resolve_stage2_target_module_names,
)
from stage2.trainer import Stage2Trainer
from stage2.sampler import (
    STAGE2_SAMPLER_FILENAME,
    LatentBlockBalancedSampler,
    write_stage2_sampler_sidecar,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds


def runtime_profile(zero_stage=2, *, world_size=1):
    return load_deepspeed_runtime_profile(
        canonical_deepspeed_config_path(zero_stage), world_size=world_size
    )


def sample_metadata() -> Stage2PeftMetadata:
    return Stage2PeftMetadata(
        schema_version=STAGE2_PEFT_SCHEMA_VERSION,
        stage="stage2",
        tuning_method="dora",
        base_model_reference="/models/stage1",
        base_model_architecture={
            "architectures": [],
            "decoder": {
                "attention_dropout": 0.0,
                "head_dim": 4,
                "hidden_act": "silu",
                "hidden_size": 8,
                "intermediate_size": 16,
                "max_position_embeddings": 32768,
                "max_window_layers": 80,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "num_key_value_heads": 1,
                "rope_scaling": {
                    "mrope_section": [1, 1, 0],
                    "rope_type": "default",
                    "type": "default",
                },
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-05,
                "sliding_window": 4096,
                "tie_word_embeddings": False,
                "use_sliding_window": False,
                "vocab_size": 128,
            },
            "model_type": "qwen2_5_vl",
            "multimodal_token_ids": {
                "image_token_id": 80,
                "video_token_id": 82,
                "vision_end_token_id": None,
                "vision_start_token_id": 81,
            },
            "vision": {
                "depth": 1,
                "fullatt_block_indexes": [0],
                "hidden_act": "silu",
                "hidden_size": 8,
                "in_channels": 3,
                "intermediate_size": 16,
                "num_heads": 2,
                "out_hidden_size": 8,
                "patch_size": 2,
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
                "tokens_per_second": 4,
                "window_size": 2,
            },
        },
        tokenizer_base_vocab_size=125,
        pretrained_model_vocab_size=125,
        resized_vocab_size=128,
        latent_size=8,
        latent_start_token_id=91,
        latent_pad_token_id=90,
        latent_end_token_id=92,
        supervise_latent_boundaries=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        requested_target_modules=("q_proj", "v_proj"),
        resolved_target_module_names=(
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
        ),
        peft_version="0.17.1",
        token_row_source="base",
        stage1_adapter_reference=None,
    )


def complete_v2_metadata() -> dict[str, object]:
    payload = complete_v3_metadata()
    payload["schema_version"] = 2
    payload.pop("token_row_source", None)
    payload.pop("stage1_adapter_reference", None)
    return payload


def complete_v3_metadata() -> dict[str, object]:
    payload = sample_metadata().to_dict()
    payload["schema_version"] = 3
    payload.pop("tokenizer_base_vocab_size")
    payload.pop("pretrained_model_vocab_size")
    payload.pop("resized_vocab_size")
    return payload


def test_complete_v2_metadata_migrates_in_memory_to_base_rows():
    """Catches removing strict compatibility for complete unreleased v2 artifacts."""
    metadata = migrate_stage2_metadata(complete_v2_metadata())

    assert metadata.schema_version == STAGE2_PEFT_SCHEMA_VERSION == 4
    assert metadata.token_row_source == "base"
    assert metadata.stage1_adapter_reference is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("v1", "schema v1"),
        ("partial", "complete v2"),
        ("extra", "complete v2"),
        ("bool_schema", "schema_version"),
    ],
)
def test_inexact_legacy_stage2_metadata_fails_closed(mutation, message):
    """Catches permissive migration of v1, partial, extra, or type-invalid v2."""
    payload = complete_v2_metadata()
    if mutation == "v1":
        payload = {"schema_version": 1}
    elif mutation == "partial":
        payload.pop("base_model_architecture")
    elif mutation == "extra":
        payload["token_row_source"] = "base"
    elif mutation == "bool_schema":
        payload["schema_version"] = True

    with pytest.raises(ValueError, match=message):
        migrate_stage2_metadata(payload)


def test_complete_v3_metadata_migrates_with_explicit_unpadded_provenance():
    metadata = migrate_stage2_metadata(complete_v3_metadata())

    assert metadata.schema_version == 4
    assert metadata.token_row_source == "base"
    assert metadata.tokenizer_base_vocab_size == 125
    assert metadata.pretrained_model_vocab_size == 125
    assert metadata.resized_vocab_size == 128


@pytest.mark.parametrize(
    "base_model_reference",
    ["authenticated-unpadded-stage1", "authenticated-padded-stage1"],
)
def test_sparse_v3_metadata_without_vocab_provenance_fails_closed(
    base_model_reference,
):
    """Catches inventing a pretrained vocabulary for a legacy sparse artifact."""

    payload = complete_v3_metadata()
    payload.update(
        token_row_source="sparse_adapter",
        stage1_adapter_reference="relocated-stage1-adapter",
        base_model_reference=base_model_reference,
        latent_pad_token_id=125,
        latent_start_token_id=126,
        latent_end_token_id=127,
    )

    with pytest.raises(
        ValueError,
        match="sparse.*v3.*cannot be loaded or automatically migrated.*fresh Stage 2 run",
    ):
        migrate_stage2_metadata(payload)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_inexact_legacy_stage2_v3_metadata_fails_closed(mutation):
    payload = complete_v3_metadata()
    if mutation == "missing":
        payload.pop("base_model_architecture")
    else:
        payload["pretrained_model_vocab_size"] = 132

    with pytest.raises(ValueError, match="complete v3"):
        migrate_stage2_metadata(payload)


@pytest.mark.parametrize("pretrained_vocab_size", [125, 132])
def test_schema_v4_accepts_exact_or_padded_pretrained_vocab(pretrained_vocab_size):
    payload = sample_metadata().to_dict()
    payload["pretrained_model_vocab_size"] = pretrained_vocab_size

    metadata = Stage2PeftMetadata.from_dict(payload)

    assert metadata.tokenizer_base_vocab_size == 125
    assert metadata.pretrained_model_vocab_size == pretrained_vocab_size
    assert metadata.resized_vocab_size == 128


def test_schema_v4_rejects_pretrained_vocab_smaller_than_tokenizer_base():
    payload = sample_metadata().to_dict()
    payload["pretrained_model_vocab_size"] = 124

    with pytest.raises(ValueError, match="pretrained_model_vocab_size.*base"):
        Stage2PeftMetadata.from_dict(payload)


def test_sparse_metadata_builder_requires_complete_stage1_vocab_provenance():
    model = tiny_stage2_qwen()
    model.config.supervise_latent_boundaries = True
    tuning = Stage2TuningConfig(
        method="dora",
        r=2,
        alpha=4,
        dropout=0.0,
        target_modules=("q_proj", "v_proj"),
    )

    with pytest.raises(ValueError, match="complete Stage 1 vocabulary provenance"):
        build_stage2_peft_metadata(
            tuning=tuning,
            base_model_reference="stage1/tiny",
            model_config=model.config,
            token_ids=LatentTokenIds(90, 91, 92),
            resolved_target_names=resolve_stage2_target_module_names(
                model, tuning.target_modules
            ),
            token_row_source="sparse_adapter",
            stage1_adapter_reference="stage1/adapter",
        )


def test_sparse_metadata_builder_requires_source_architecture_provenance():
    model = tiny_stage2_qwen()
    model.config.supervise_latent_boundaries = True
    tuning = Stage2TuningConfig(
        method="dora",
        r=2,
        alpha=4,
        dropout=0.0,
        target_modules=("q_proj", "v_proj"),
    )

    with pytest.raises(ValueError, match="source architecture provenance"):
        build_stage2_peft_metadata(
            tuning=tuning,
            base_model_reference="stage1/tiny",
            model_config=model.config,
            token_ids=LatentTokenIds(90, 91, 92),
            resolved_target_names=resolve_stage2_target_module_names(
                model, tuning.target_modules
            ),
            token_row_source="sparse_adapter",
            stage1_adapter_reference="stage1/adapter",
            tokenizer_base_vocab_size=90,
            pretrained_model_vocab_size=96,
            resized_vocab_size=93,
        )


def test_new_stage2_metadata_writes_schema_v4_vocab_provenance_fields(tmp_path):
    """Catches new Stage 2 sidecars losing padded original provenance."""
    metadata = sample_metadata()
    save_stage2_peft_metadata(tmp_path, metadata)
    payload = json.loads(
        (tmp_path / STAGE2_PEFT_METADATA_FILENAME).read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == 4
    assert payload["token_row_source"] == "base"
    assert payload["stage1_adapter_reference"] is None
    assert payload["tokenizer_base_vocab_size"] == 125
    assert payload["pretrained_model_vocab_size"] == 125
    assert payload["resized_vocab_size"] == 128


@pytest.mark.parametrize("missing", [True, False])
def test_sparse_stage2_artifact_requires_its_own_exact_token_rows(tmp_path, missing):
    """Catches sparse Stage 2 checkpoints depending on the Stage 1 directory."""
    payload = sample_metadata().to_dict()
    payload.update(
        token_row_source="sparse_adapter",
        stage1_adapter_reference="/deleted/stage1/provenance-only",
        latent_pad_token_id=125,
        latent_start_token_id=126,
        latent_end_token_id=127,
    )
    (tmp_path / STAGE2_PEFT_METADATA_FILENAME).write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )
    (tmp_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").touch()
    if not missing:
        save_file(
            {
                "token_ids": torch.tensor([125, 126, 127], dtype=torch.int64),
                "output_delta": torch.zeros((3, 8), dtype=torch.float32),
            },
            tmp_path / "ilvr_token_rows.safetensors",
            metadata={"format_version": "1", "training_dtype": "float32"},
        )

    if missing:
        with pytest.raises(ValueError, match="token-row"):
            classify_stage2_artifact(tmp_path)
    else:
        assert classify_stage2_artifact(tmp_path).value == "peft"


@pytest.mark.parametrize("mutation", ["token_ids", "output_shape"])
def test_sparse_stage2_artifact_rejects_mismatched_payload(tmp_path, mutation):
    """Catches structurally present but incompatible sparse row payloads."""
    payload = sample_metadata().to_dict()
    payload.update(
        token_row_source="sparse_adapter",
        stage1_adapter_reference="/provenance-only/stage1",
        latent_pad_token_id=125,
        latent_start_token_id=126,
        latent_end_token_id=127,
    )
    (tmp_path / STAGE2_PEFT_METADATA_FILENAME).write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )
    (tmp_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").touch()
    token_ids = (
        [125, 99, 127]
        if mutation == "token_ids"
        else [125, 126, 127]
    )
    output_shape = (3, 7) if mutation == "output_shape" else (3, 8)
    save_file(
        {
            "token_ids": torch.tensor(token_ids, dtype=torch.int64),
            "output_delta": torch.zeros(output_shape, dtype=torch.float32),
        },
        tmp_path / "ilvr_token_rows.safetensors",
        metadata={"format_version": "1", "training_dtype": "float32"},
    )

    with pytest.raises(ValueError, match="sparse.*(token|shape)"):
        classify_stage2_artifact(tmp_path)


def _write_structural_stage2_peft(directory, *, sparse=True):
    directory.mkdir(parents=True, exist_ok=True)
    metadata = replace(
        sample_metadata(),
        token_row_source="sparse_adapter" if sparse else "base",
        latent_pad_token_id=125 if sparse else 90,
        latent_start_token_id=126 if sparse else 91,
        latent_end_token_id=127 if sparse else 92,
        stage1_adapter_reference=(
            "/deleted/stage1/provenance-only" if sparse else None
        ),
    )
    save_stage2_peft_metadata(directory, metadata)
    (directory / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"adapter")
    if sparse:
        save_file(
            {
                "token_ids": torch.tensor([125, 126, 127], dtype=torch.int64),
                "output_delta": torch.zeros((3, 8), dtype=torch.float32),
            },
            directory / "ilvr_token_rows.safetensors",
            metadata={"format_version": "1", "training_dtype": "float32"},
        )
    return metadata


def _write_stage2_full_index(directory, *, binary=False):
    directory.mkdir(parents=True, exist_ok=True)
    stem = "pytorch_model" if binary else "model"
    suffix = "bin" if binary else "safetensors"
    shards = [
        f"{stem}-00001-of-00002.{suffix}",
        f"{stem}-00002-of-00002.{suffix}",
    ]
    for shard in shards:
        (directory / shard).write_bytes(b"weight")
    index_name = (
        "pytorch_model.bin.index.json"
        if binary
        else "model.safetensors.index.json"
    )
    (directory / index_name).write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "model.embed_tokens.weight": shards[0],
                    "lm_head.weight": shards[1],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return index_name, shards


@pytest.mark.parametrize(
    "filename",
    [
        STAGE2_PEFT_METADATA_FILENAME,
        "adapter_config.json",
        "adapter_model.safetensors",
        "ilvr_token_rows.safetensors",
    ],
)
@pytest.mark.parametrize("target_location", ["inside", "outside"])
def test_stage2_classifier_rejects_symlinked_required_peft_file(
    tmp_path, filename, target_location
):
    """Catches a required PEFT payload escaping regular-file ownership."""
    artifact = tmp_path / "artifact"
    _write_structural_stage2_peft(artifact)
    payload_dir = (
        artifact / "payloads"
        if target_location == "inside"
        else tmp_path / "outside-payloads"
    )
    payload_dir.mkdir()
    source = artifact / filename
    target = payload_dir / filename
    source.replace(target)
    source.symlink_to(target)

    with pytest.raises(ValueError, match="regular|symlink|invalid"):
        classify_stage2_artifact(artifact)


@pytest.mark.parametrize("filename", ["model.safetensors", "pytorch_model.bin"])
def test_stage2_classifier_accepts_one_standalone_full_family(tmp_path, filename):
    """Catches the strict classifier accidentally dropping valid full saves."""
    (tmp_path / filename).write_bytes(b"weight")

    assert classify_stage2_artifact(tmp_path).value == "full"


@pytest.mark.parametrize("binary", [False, True])
def test_stage2_classifier_accepts_one_complete_sharded_full_family(
    tmp_path, binary
):
    """Catches a complete indexed full checkpoint being treated as incomplete."""
    _write_stage2_full_index(tmp_path, binary=binary)

    assert classify_stage2_artifact(tmp_path).value == "full"


@pytest.mark.parametrize("target_location", ["inside", "outside"])
@pytest.mark.parametrize("component", ["standalone", "index", "shard"])
def test_stage2_classifier_rejects_symlinked_full_weight_family(
    tmp_path, target_location, component
):
    """Catches full-model files being accepted through symlink indirection."""
    artifact = tmp_path / "artifact"
    if component == "standalone":
        artifact.mkdir()
        filename = "model.safetensors"
        (artifact / filename).write_bytes(b"weight")
    else:
        index_name, shards = _write_stage2_full_index(artifact)
        filename = index_name if component == "index" else shards[0]
    payload_dir = (
        artifact / "payloads"
        if target_location == "inside"
        else tmp_path / "outside-payloads"
    )
    payload_dir.mkdir()
    source = artifact / filename
    target = payload_dir / filename
    source.replace(target)
    source.symlink_to(target)

    with pytest.raises(ValueError, match="full|regular|symlink|invalid"):
        classify_stage2_artifact(artifact)


@pytest.mark.parametrize(
    "mutation",
    [
        "malformed_index",
        "missing_shard",
        "orphan_shard",
        "mixed_standalone",
        "mixed_binary_family",
    ],
)
def test_stage2_classifier_rejects_incoherent_full_serialization(
    tmp_path, mutation
):
    """Catches incomplete or mixed sharded weight families reaching loading."""
    index_name, shards = _write_stage2_full_index(tmp_path)
    if mutation == "malformed_index":
        (tmp_path / index_name).write_text("{\n", encoding="utf-8")
    elif mutation == "missing_shard":
        (tmp_path / shards[-1]).unlink()
    elif mutation == "orphan_shard":
        (tmp_path / "model-00003-of-00003.safetensors").write_bytes(b"weight")
    elif mutation == "mixed_standalone":
        (tmp_path / "model.safetensors").write_bytes(b"weight")
    else:
        (tmp_path / "pytorch_model.bin").write_bytes(b"weight")

    with pytest.raises(ValueError, match="full|shard|serialization|index"):
        classify_stage2_artifact(tmp_path)


def test_stage2_classifier_rejects_orphan_full_shard_beside_peft(tmp_path):
    """Catches an orphan full shard evading PEFT/full mixed-layout detection."""
    _write_structural_stage2_peft(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"weight")

    with pytest.raises(ValueError, match="mixed.*full.*PEFT|mixed.*PEFT.*full"):
        classify_stage2_artifact(tmp_path)


def test_stage2_classifier_rejects_unknown_weight_like_filename(tmp_path):
    """Catches unowned model bytes being silently ignored during classification."""
    (tmp_path / "model.safetensors").write_bytes(b"weight")
    (tmp_path / "unowned_weights.safetensors").write_bytes(b"weight")

    with pytest.raises(ValueError, match="unknown.*weight"):
        classify_stage2_artifact(tmp_path)


def write_mutated_metadata(directory, metadata, mutation):
    payload = metadata.to_dict()
    if mutation == "bad_schema":
        payload["schema_version"] = 999
    elif mutation == "bad_stage":
        payload["stage"] = "stage1"
    elif mutation == "missing_field":
        payload.pop("latent_size")
    elif mutation == "unknown_architecture_field":
        payload["base_model_architecture"]["decoder"]["unknown"] = 1
    directory.mkdir(parents=True, exist_ok=True)
    (directory / STAGE2_PEFT_METADATA_FILENAME).write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )


def test_stage2_peft_metadata_round_trips(tmp_path):
    save_stage2_peft_metadata(tmp_path, sample_metadata())

    assert load_stage2_peft_metadata(tmp_path) == sample_metadata()
    assert (tmp_path / STAGE2_PEFT_METADATA_FILENAME).read_text(
        encoding="utf-8"
    ).endswith("\n")


def test_stage2_architecture_snapshot_is_unchanged():
    """Catches a shared Qwen snapshot drifting from Stage 2's artifact contract."""
    config = tiny_stage2_qwen().config
    expected = {
        "architectures": [],
        "decoder": {
            "attention_dropout": 0.0,
            "head_dim": 4,
            "hidden_act": "silu",
            "hidden_size": 8,
            "intermediate_size": 16,
            "max_position_embeddings": 32768,
            "max_window_layers": 80,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "mrope_section": [1, 1, 0],
                "rope_type": "default",
                "type": "default",
            },
            "rope_theta": 1000000.0,
            "sliding_window": 4096,
            "tie_word_embeddings": False,
            "use_sliding_window": False,
            "vocab_size": 128,
        },
        "model_type": "qwen2_5_vl",
        "multimodal_token_ids": {
            "image_token_id": 80,
            "video_token_id": 82,
            "vision_end_token_id": None,
            "vision_start_token_id": 81,
        },
        "vision": {
            "depth": 1,
            "fullatt_block_indexes": [0],
            "hidden_act": "silu",
            "hidden_size": 8,
            "in_channels": 3,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": 8,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "tokens_per_second": 4,
            "window_size": 2,
        },
    }

    shared = capture_qwen2_5_vl_architecture(config)
    legacy = capture_stage2_base_model_architecture(config)

    assert shared == expected
    assert shared == legacy


@pytest.mark.parametrize(
    "mutation",
    ["bad_schema", "bad_stage", "missing_field", "unknown_architecture_field"],
)
def test_stage2_peft_metadata_fails_closed(tmp_path, mutation):
    write_mutated_metadata(tmp_path, sample_metadata(), mutation)

    with pytest.raises(ValueError):
        load_stage2_peft_metadata(tmp_path)


def test_unreleased_schema_v1_without_architecture_snapshot_fails_closed(tmp_path):
    payload = sample_metadata().to_dict()
    payload["schema_version"] = 1
    payload.pop("base_model_architecture")
    (tmp_path / STAGE2_PEFT_METADATA_FILENAME).write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unsupported.*schema"):
        load_stage2_peft_metadata(tmp_path)


def test_architecture_snapshot_rejects_non_json_rope_number():
    payload = sample_metadata().to_dict()
    payload["base_model_architecture"]["decoder"]["rope_theta"] = float("nan")

    with pytest.raises(ValueError, match="rope_theta.*finite"):
        Stage2PeftMetadata.from_dict(payload)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("decoder", "hidden_act", False, "hidden_act.*non-empty string"),
        ("decoder", "rms_norm_eps", float("inf"), "rms_norm_eps.*finite"),
        ("decoder", "rms_norm_eps", True, "rms_norm_eps.*finite"),
        (
            "decoder",
            "attention_dropout",
            float("nan"),
            "attention_dropout.*finite",
        ),
        ("decoder", "attention_dropout", False, "attention_dropout.*finite"),
        (
            "decoder",
            "tie_word_embeddings",
            0,
            "tie_word_embeddings.*boolean",
        ),
        ("vision", "hidden_act", 1, "hidden_act.*non-empty string"),
    ],
)
def test_architecture_snapshot_strictly_validates_semantic_fields(
    section, field, value, message
):
    source = tiny_stage2_qwen()
    source.config.supervise_latent_boundaries = True
    tuning = Stage2TuningConfig(
        method="lora",
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=("q_proj",),
    )
    metadata = build_stage2_peft_metadata(
        tuning=tuning,
        base_model_reference="stage1",
        model_config=source.config,
        token_ids=LatentTokenIds(90, 91, 92),
        resolved_target_names=resolve_stage2_target_module_names(
            source, tuning.target_modules
        ),
    )
    payload = metadata.to_dict()
    payload["base_model_architecture"][section][field] = value

    with pytest.raises(ValueError, match=message):
        Stage2PeftMetadata.from_dict(payload)


@pytest.mark.parametrize("operation", ["to_dict", "save"])
def test_invalid_direct_metadata_cannot_be_serialized(tmp_path, operation):
    invalid = replace(sample_metadata(), schema_version=999)

    with pytest.raises(ValueError, match="unsupported.*schema"):
        if operation == "to_dict":
            invalid.to_dict()
        else:
            save_stage2_peft_metadata(tmp_path, invalid)

    assert not (tmp_path / STAGE2_PEFT_METADATA_FILENAME).exists()


def semantic_args():
    return {
        "latent_size": 8,
        "supervise_latent_boundaries": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": "q_proj,v_proj",
    }


def lora_metadata():
    values = sample_metadata().to_dict()
    values["tuning_method"] = "lora"
    return Stage2PeftMetadata.from_dict(values)


def fake_stage_config(path):
    stage = "stage1" if "stage1" in str(path) else "stage2"
    return SimpleNamespace(
        stage=stage, latent_size=8, supervise_latent_boundaries=True
    )


def fake_metadata_loader(path):
    raise AssertionError(f"fresh training must not read PEFT metadata: {path}")


def test_fresh_peft_uses_stage1_base_and_no_trainer_resume():
    decision = resolve_stage2_load_decision(
        load_model_path="stage1",
        resume_from_checkpoint_path=None,
        load_config=fake_stage_config,
        load_peft_metadata=fake_metadata_loader,
        tuning_method="lora",
        tuning_method_explicit=True,
        **semantic_args(),
    )

    assert decision.base_model_source == "stage1"
    assert decision.processor_source == "stage1"
    assert decision.adapter_source is None
    assert decision.trainer_resume_path is None
    assert decision.tuning.method == "lora"
    assert decision.peft_metadata is None


def test_peft_resume_keeps_explicit_base_and_loads_adapter_checkpoint():
    decision = resolve_stage2_load_decision(
        load_model_path="relocated-stage1",
        resume_from_checkpoint_path="stage2/checkpoint-9",
        load_config=fake_stage_config,
        load_peft_metadata=lambda path: lora_metadata(),
        tuning_method=None,
        tuning_method_explicit=False,
        **semantic_args(),
    )

    assert decision.base_model_source == "relocated-stage1"
    assert decision.processor_source == "stage2/checkpoint-9"
    assert decision.adapter_source == "stage2/checkpoint-9"
    assert decision.trainer_resume_path == "stage2/checkpoint-9"
    assert decision.tuning.method == "lora"
    assert decision.peft_metadata == lora_metadata()


def test_base_token_row_resume_still_requires_tagged_full_stage1():
    """Catches the sparse original-base exception leaking into v2/base resumes."""
    migrated_v2 = migrate_stage2_metadata(complete_v2_metadata())

    with pytest.raises(ValueError, match="requires a stage1 checkpoint"):
        resolve_stage2_load_decision(
            load_model_path="tag-free/original-qwen",
            resume_from_checkpoint_path="stage2/checkpoint-9",
            load_config=lambda unused: SimpleNamespace(stage=None),
            load_peft_metadata=lambda unused: migrated_v2,
            tuning_method=None,
            tuning_method_explicit=False,
            **semantic_args(),
        )


@pytest.mark.parametrize(
    "legacy_payload",
    [complete_v2_metadata, complete_v3_metadata],
    ids=["v2-base-rows", "v3-base-rows"],
)
def test_complete_legacy_base_row_metadata_resolves_resume(legacy_payload):
    """Catches rejecting the unambiguous base-row v2/v3 compatibility path."""

    migrated = migrate_stage2_metadata(legacy_payload())
    decision = resolve_stage2_load_decision(
        load_model_path="relocated-stage1",
        resume_from_checkpoint_path="stage2/checkpoint-9",
        load_config=lambda unused: SimpleNamespace(stage="stage1"),
        load_peft_metadata=lambda unused: migrated,
        tuning_method=None,
        tuning_method_explicit=False,
        **semantic_args(),
    )

    assert decision.base_model_source == "relocated-stage1"
    assert decision.adapter_source == "stage2/checkpoint-9"
    assert decision.peft_metadata.token_row_source == "base"


def test_peft_sidecar_classifies_resume_without_loading_adapter_as_full_config():
    def load_only_explicit_base(path):
        if path != "relocated-stage1":
            raise AssertionError("PEFT checkpoint must not be loaded as a full config")
        return SimpleNamespace(stage="stage1")

    decision = resolve_stage2_load_decision(
        load_model_path="relocated-stage1",
        resume_from_checkpoint_path="adapter/checkpoint-9",
        load_config=load_only_explicit_base,
        load_peft_metadata=lambda _: lora_metadata(),
        tuning_method=None,
        tuning_method_explicit=False,
        **semantic_args(),
    )

    assert decision.adapter_source == "adapter/checkpoint-9"


def test_full_resume_keeps_the_complete_stage2_checkpoint_as_every_source():
    def missing_sidecar(_):
        raise FileNotFoundError

    decision = resolve_stage2_load_decision(
        load_model_path="stage1",
        resume_from_checkpoint_path="stage2/checkpoint-11",
        load_config=fake_stage_config,
        load_peft_metadata=missing_sidecar,
        tuning_method="full",
        tuning_method_explicit=True,
        **semantic_args(),
    )

    assert decision.base_model_source == "stage2/checkpoint-11"
    assert decision.processor_source == "stage2/checkpoint-11"
    assert decision.adapter_source is None
    assert decision.trainer_resume_path == "stage2/checkpoint-11"
    assert decision.tuning.method == "full"


def test_incomplete_adapter_resume_fails_before_full_config_load(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    def reject_full_config_load(_):
        raise AssertionError("adapter-shaped resume must not load a full config")

    with pytest.raises(ValueError, match="incomplete Stage 2 PEFT artifact"):
        resolve_stage2_load_decision(
            load_model_path="stage1",
            resume_from_checkpoint_path=str(tmp_path),
            load_config=reject_full_config_load,
            tuning_method=None,
            tuning_method_explicit=False,
            **semantic_args(),
        )


def test_mixed_full_and_peft_resume_fails_before_metadata_or_config_load(tmp_path):
    for marker in (
        STAGE2_PEFT_METADATA_FILENAME,
        "adapter_config.json",
        "adapter_model.safetensors",
        "model.safetensors",
    ):
        (tmp_path / marker).touch()

    def reject_load(_):
        raise AssertionError("mixed artifact must fail before parsing")

    with pytest.raises(ValueError, match="mixed Stage 2 full and PEFT artifact"):
        resolve_stage2_load_decision(
            load_model_path="stage1",
            resume_from_checkpoint_path=str(tmp_path),
            load_config=reject_load,
            load_peft_metadata=reject_load,
            tuning_method=None,
            tuning_method_explicit=False,
            **semantic_args(),
        )


@pytest.mark.parametrize(
    ("tuning_method", "tuning_method_explicit", "metadata_loader", "message"),
    [
        (
            "lora",
            True,
            lambda _: (_ for _ in ()).throw(FileNotFoundError()),
            "missing.*metadata",
        ),
        ("dora", True, lambda _: lora_metadata(), "cannot switch.*lora.*dora"),
        ("full", True, lambda _: lora_metadata(), "full.*PEFT"),
    ],
)
def test_resume_rejects_incompatible_checkpoint_selection(
    tuning_method, tuning_method_explicit, metadata_loader, message
):
    with pytest.raises(ValueError, match=message):
        resolve_stage2_load_decision(
            load_model_path="stage1",
            resume_from_checkpoint_path="stage2/checkpoint-9",
            load_config=fake_stage_config,
            load_peft_metadata=metadata_loader,
            tuning_method=tuning_method,
            tuning_method_explicit=tuning_method_explicit,
            **semantic_args(),
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"lora_r": 4}, "rank.*checkpoint=16.*requested=4"),
        ({"lora_alpha": 7}, "alpha.*checkpoint=32.*requested=7"),
        ({"lora_dropout": 0.25}, "dropout.*checkpoint=0.05.*requested=0.25"),
        ({"lora_target_modules": "q_proj"}, "target_modules.*checkpoint"),
    ],
)
def test_explicit_peft_resume_rejects_changed_adapter_settings(override, message):
    values = semantic_args()
    values.update(override)

    with pytest.raises(ValueError, match=message):
        resolve_stage2_load_decision(
            load_model_path="stage1",
            resume_from_checkpoint_path="stage2/checkpoint-9",
            load_config=fake_stage_config,
            load_peft_metadata=lambda _: lora_metadata(),
            tuning_method="lora",
            tuning_method_explicit=True,
            **values,
        )


def test_peft_resume_rejects_changed_latent_settings():
    values = semantic_args()
    values["latent_size"] = 4

    with pytest.raises(ValueError, match="latent_size.*checkpoint=8.*requested=4"):
        resolve_stage2_load_decision(
            load_model_path="stage1",
            resume_from_checkpoint_path="stage2/checkpoint-9",
            load_config=fake_stage_config,
            load_peft_metadata=lambda _: lora_metadata(),
            tuning_method=None,
            tuning_method_explicit=False,
            **values,
        )


def test_peft_resume_rejects_a_stage1_adapter_as_its_base():
    def stage1_adapter_config(path):
        if path == "stage1-adapter":
            return SimpleNamespace(stage="stage1", peft_type="LORA")
        return fake_stage_config(path)

    with pytest.raises(ValueError, match="Stage 1 PEFT-only base"):
        resolve_stage2_load_decision(
            load_model_path="stage1-adapter",
            resume_from_checkpoint_path="stage2/checkpoint-9",
            load_config=stage1_adapter_config,
            load_peft_metadata=lambda _: lora_metadata(),
            tuning_method=None,
            tuning_method_explicit=False,
            **semantic_args(),
        )


def test_fresh_training_rejects_a_real_stage1_peft_config_directory(tmp_path):
    LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["model.layers.0.self_attn.q_proj"],
    ).save_pretrained(tmp_path)

    with pytest.raises(ValueError, match="Stage 1 PEFT-only base"):
        resolve_stage2_load_decision(
            load_model_path=str(tmp_path),
            resume_from_checkpoint_path=None,
            load_config=lambda _: SimpleNamespace(stage="stage1"),
            load_peft_metadata=fake_metadata_loader,
            tuning_method="lora",
            tuning_method_explicit=True,
            **semantic_args(),
        )


def test_fresh_local_peft_only_stage1_fails_before_full_config_load(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    def reject_full_config_load(_):
        raise AssertionError("PEFT-only Stage 1 source must not load a full config")

    with pytest.raises(ValueError, match="Stage 1 PEFT-only base is not supported"):
        resolve_stage2_load_decision(
            load_model_path=str(tmp_path),
            resume_from_checkpoint_path=None,
            load_config=reject_full_config_load,
            load_peft_metadata=fake_metadata_loader,
            tuning_method="lora",
            tuning_method_explicit=True,
            **semantic_args(),
        )


class TinyTokenizer:
    unk_token_id = 0
    unk_token = "<unk>"

    def __init__(self, *, start_token_id=91):
        self.ids = {
            "<|latent_pad|>": 90,
            "<|latent_start|>": start_token_id,
            "<|latent_end|>": 92,
        }

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, self.unk_token_id)


def write_tiny_adapter(
    directory, *, method="lora", target_modules=("q_proj", "v_proj")
):
    tuning = Stage2TuningConfig(
        method=method,
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=target_modules,
    )
    source = tiny_stage2_qwen()
    source.config.supervise_latent_boundaries = True
    resolved = resolve_stage2_target_module_names(source, tuning.target_modules)
    adapter = build_stage2_tuned_model(source, tuning)
    with pytest.warns(UserWarning, match="Could not find a config file"):
        adapter.save_pretrained(directory)
    metadata = build_stage2_peft_metadata(
        tuning=tuning,
        base_model_reference="/original/stage1",
        model_config=source.config,
        token_ids=LatentTokenIds(90, 91, 92),
        resolved_target_names=resolved,
    )
    return metadata, adapter.peft_config["default"]


def relocated_base():
    base = tiny_stage2_qwen()
    base.config.supervise_latent_boundaries = True
    base.config.stage = "stage1"
    base.config._name_or_path = "/relocated/stage1"
    return base


def test_relocated_stage1_base_with_identical_structure_is_compatible(
    tmp_path, caplog
):
    metadata, adapter_config = write_tiny_adapter(tmp_path)

    validate_stage2_peft_compatibility(
        relocated_base(),
        TinyTokenizer(),
        metadata,
        adapter_config,
        tmp_path,
    )

    assert "relocated" in caplog.text


def test_relocated_stage1_base_accepts_compact_saved_target_selectors(tmp_path):
    """Catches compact adapter selectors being compared to resolved paths."""
    metadata, original_adapter_config = write_tiny_adapter(tmp_path)
    adapter_config = deepcopy(original_adapter_config)
    adapter_config.target_modules = {"q_proj", "v_proj"}

    validate_stage2_peft_compatibility(
        relocated_base(), TinyTokenizer(), metadata, adapter_config, tmp_path
    )


@pytest.mark.parametrize(
    "target_modules",
    ["q_proj", (), ("q_proj", "q_proj"), ("visual.q_proj",), 1],
)
def test_stage2_peft_compatibility_rejects_malformed_adapter_target_selectors(
    tmp_path, target_modules
):
    """Catches malformed adapter selectors bypassing resolved-target checks."""
    metadata, original_adapter_config = write_tiny_adapter(tmp_path)
    adapter_config = deepcopy(original_adapter_config)
    adapter_config.target_modules = target_modules

    with pytest.raises(ValueError, match="adapter.*target"):
        validate_stage2_peft_compatibility(
            relocated_base(), TinyTokenizer(), metadata, adapter_config, tmp_path
        )


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda config: setattr(config, "rope_theta", 500000.0), "rope_theta"),
        (lambda config: setattr(config, "hidden_act", "gelu"), "hidden_act"),
        (
            lambda config: setattr(config, "rms_norm_eps", 1e-6),
            "rms_norm_eps",
        ),
        (
            lambda config: setattr(config, "attention_dropout", 0.25),
            "attention_dropout",
        ),
        (
            lambda config: setattr(config, "tie_word_embeddings", True),
            "tie_word_embeddings",
        ),
        (
            lambda config: setattr(config.vision_config, "window_size", 4),
            "vision.window_size",
        ),
        (
            lambda config: setattr(config.vision_config, "hidden_act", "gelu"),
            "vision.hidden_act",
        ),
    ],
)
def test_relocated_same_shape_base_rejects_semantic_architecture_change(
    tmp_path, mutate, field
):
    metadata, adapter_config = write_tiny_adapter(tmp_path)
    base = relocated_base()
    mutate(base.config)

    with pytest.raises(ValueError, match=rf"base architecture mismatch.*{field}"):
        validate_stage2_peft_compatibility(
            base, TinyTokenizer(), metadata, adapter_config, tmp_path
        )


def test_relocated_stage1_base_accepts_dora_magnitude_tensor_naming(tmp_path):
    metadata, adapter_config = write_tiny_adapter(tmp_path, method="dora")

    validate_stage2_peft_compatibility(
        relocated_base(), TinyTokenizer(), metadata, adapter_config, tmp_path
    )


def test_relocated_stage1_base_accepts_all_default_projection_shapes(tmp_path):
    metadata, adapter_config = write_tiny_adapter(
        tmp_path,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ),
    )

    validate_stage2_peft_compatibility(
        relocated_base(), TinyTokenizer(), metadata, adapter_config, tmp_path
    )


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("use_rslora", True, "use_rslora"),
        ("rank_pattern", {"model.layers.0.self_attn.q_proj": 2}, "rank_pattern"),
        ("alpha_pattern", {"model.layers.0.self_attn.q_proj": 4}, "alpha_pattern"),
        ("task_type", "SEQ_CLS", "task_type"),
        ("bias", "all", "bias"),
    ],
)
def test_stage2_peft_compatibility_rejects_unsupported_lora_variants(
    tmp_path, attribute, value, message
):
    metadata, original_adapter_config = write_tiny_adapter(tmp_path)
    adapter_config = deepcopy(original_adapter_config)
    setattr(adapter_config, attribute, value)

    with pytest.raises(ValueError, match=message):
        validate_stage2_peft_compatibility(
            relocated_base(), TinyTokenizer(), metadata, adapter_config, tmp_path
        )


def test_stage2_peft_compatibility_rejects_added_decoder_target_topology(tmp_path):
    metadata, adapter_config = write_tiny_adapter(tmp_path)
    base = relocated_base()
    base.model.layers.append(deepcopy(base.model.layers[0]))
    base.config.num_hidden_layers = 2

    with pytest.raises(
        ValueError, match="base architecture mismatch.*num_hidden_layers"
    ):
        validate_stage2_peft_compatibility(
            base, TinyTokenizer(), metadata, adapter_config, tmp_path
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("token_id", "latent_start_token_id mismatch"),
        ("latent_boundaries", "supervise_latent_boundaries mismatch"),
        ("hidden_size", "base architecture mismatch.*hidden_size"),
        ("missing_target", "missing resolved target"),
        ("nonlinear_target", "resolved target is not linear"),
        ("target_shape", "target shape mismatch"),
        ("method", "method.*mismatch"),
        ("rank", "rank.*mismatch"),
        ("alpha", "alpha.*mismatch"),
        ("dropout", "dropout.*mismatch"),
        ("use_dora", "use_dora.*mismatch"),
        ("requested_targets", "requested targets"),
        ("resolved_targets", "resolved targets"),
    ],
)
def test_stage2_peft_compatibility_fails_closed(
    tmp_path, mutation, message
):
    metadata, original_adapter_config = write_tiny_adapter(tmp_path)
    base = relocated_base()
    tokenizer = TinyTokenizer()
    adapter_config = deepcopy(original_adapter_config)

    if mutation == "token_id":
        tokenizer = TinyTokenizer(start_token_id=99)
    elif mutation == "latent_boundaries":
        base.config.supervise_latent_boundaries = False
    elif mutation == "hidden_size":
        base.config.hidden_size *= 2
        base.config.num_attention_heads *= 2
    elif mutation == "missing_target":
        delattr(base.model.layers[0].self_attn, "q_proj")
    elif mutation == "nonlinear_target":
        base.model.layers[0].self_attn.q_proj = nn.ReLU()
    elif mutation == "target_shape":
        base.model.layers[0].self_attn.q_proj = nn.Linear(8, 7)
    elif mutation == "method":
        values = metadata.to_dict()
        values["tuning_method"] = "dora"
        metadata = Stage2PeftMetadata.from_dict(values)
    elif mutation == "rank":
        adapter_config.r += 1
    elif mutation == "alpha":
        adapter_config.lora_alpha += 1
    elif mutation == "dropout":
        adapter_config.lora_dropout = 0.25
    elif mutation == "use_dora":
        adapter_config.use_dora = True
    elif mutation == "requested_targets":
        values = metadata.to_dict()
        values["requested_target_modules"] = ["q_proj"]
        metadata = Stage2PeftMetadata.from_dict(values)
    elif mutation == "resolved_targets":
        values = metadata.to_dict()
        values["resolved_target_module_names"] = [
            "model.layers.0.self_attn.q_proj"
        ]
        metadata = Stage2PeftMetadata.from_dict(values)

    with pytest.raises(ValueError, match=message):
        validate_stage2_peft_compatibility(
            base, tokenizer, metadata, adapter_config, tmp_path
        )


def tiny_trainer_metadata(model, tuning):
    model.config.supervise_latent_boundaries = True
    return build_stage2_peft_metadata(
        tuning=tuning,
        base_model_reference="stage1/tiny-base",
        model_config=model.config,
        token_ids=LatentTokenIds(90, 91, 92),
        resolved_target_names=resolve_stage2_target_module_names(
            model, tuning.target_modules
        ),
    )


def tiny_trainer_batch():
    return {
        "input_ids": torch.tensor([1, 91, 90, 90, 92, 2]),
        "attention_mask": torch.ones(6, dtype=torch.long),
        "labels": torch.tensor([-100, 91, -100, -100, 92, 2]),
    }


class TinySavingProcessor:
    model_input_names = ["input_ids"]

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tiny_processor.txt").write_text(
            "latent_pad=90\nlatent_start=91\nlatent_end=92\n",
            encoding="utf-8",
        )


@pytest.fixture
def isolated_trainer_deepspeed_state(monkeypatch):
    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)
    unset_hf_deepspeed_config()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    yield
    unset_hf_deepspeed_config()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("method", ["full", "lora", "dora"])
def test_stage2_trainer_accepts_every_supported_method_profile(
    zero_stage, method, monkeypatch
):
    """Catches Stage 2 dropping a supported tuning/runtime combination."""
    monkeypatch.setattr(
        "stage2.trainer.Trainer.__init__", lambda trainer, *args, **kwargs: None
    )
    metadata = None if method == "full" else SimpleNamespace(tuning_method=method)
    profile = runtime_profile(zero_stage, world_size=2)

    trainer = Stage2Trainer(
        stage2_peft_metadata=metadata,
        deepspeed_runtime_profile=profile,
    )

    assert trainer.deepspeed_runtime_profile is profile
    if method == "full":
        assert trainer.stage2_peft_metadata is None
    else:
        assert trainer.stage2_peft_metadata.tuning_method == method


def test_stage2_trainer_direct_non_deepspeed_constructor_derives_profile(
    monkeypatch,
):
    """Catches the identity contract breaking legacy CPU construction."""
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(
        "stage2.trainer.Trainer.__init__", lambda trainer, *args, **kwargs: None
    )

    trainer = Stage2Trainer(args=SimpleNamespace(deepspeed=None))

    assert trainer.deepspeed_runtime_profile.enabled is False
    assert trainer.deepspeed_runtime_profile.zero_stage is None
    assert trainer.deepspeed_runtime_profile.config_digest is None
    assert trainer.deepspeed_runtime_profile.world_size == 4


def test_stage2_trainer_deepspeed_constructor_requires_explicit_profile(
    monkeypatch,
):
    """Catches an active DeepSpeed trainer being mislabeled as disabled."""
    monkeypatch.setattr(
        "stage2.trainer.Trainer.__init__",
        lambda trainer, *args, **kwargs: pytest.fail(
            "DeepSpeed construction reached parent Trainer without identity"
        ),
    )

    with pytest.raises(TypeError, match="deepspeed_runtime_profile"):
        Stage2Trainer(
            args=SimpleNamespace(
                deepspeed=str(canonical_deepspeed_config_path(2))
            )
        )


def test_stage2_numbered_checkpoint_writes_runtime_identity(tmp_path, monkeypatch):
    """Catches Stage 2 numbered checkpoints missing their resume identity."""
    checkpoint = tmp_path / "checkpoint-3"

    def parent_save(trainer, model, trial):
        del trainer, model, trial
        checkpoint.mkdir()

    monkeypatch.setattr("stage2.trainer.Trainer._save_checkpoint", parent_save)
    trainer = object.__new__(Stage2Trainer)
    trainer.args = SimpleNamespace(
        output_dir=str(tmp_path), should_save=True, world_size=2, seed=42
    )
    trainer.state = SimpleNamespace(global_step=3)
    trainer.model = object()
    trainer.train_dataset = [
        SimpleNamespace(latent_block_count=count) for count in (1, 1, 1, 2, 2)
    ]
    trainer.stage2_sampler_identity = LatentBlockBalancedSampler(
        trainer.train_dataset, num_replicas=2, seed=42
    ).identity
    trainer.deepspeed_runtime_profile = runtime_profile(2, world_size=2)
    trainer._get_output_dir = lambda trial: str(tmp_path)
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda value, reduction: value,
        wait_for_everyone=lambda: None,
    )

    trainer._save_checkpoint(trainer.model, trial=None)

    payload = json.loads(
        (checkpoint / DEEPSPEED_RUNTIME_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["zero_stage"] == 2
    assert payload["world_size"] == 2
    sampler_payload = json.loads(
        (checkpoint / STAGE2_SAMPLER_FILENAME).read_text(encoding="utf-8")
    )
    assert sampler_payload["global_step"] == 3
    assert sampler_payload["bucket_sizes"] == {"1": 3, "2": 2}
    assert sampler_payload["padded_samples"] == 1


def test_stage2_resume_rejects_sampler_identity_before_parent_train(
    tmp_path, monkeypatch
):
    """Catches changed distributed data geometry reaching Trainer restore."""
    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()
    write_deepspeed_runtime_sidecar(
        checkpoint, runtime_profile(2, world_size=2)
    )
    checkpoint_dataset = [
        SimpleNamespace(latent_block_count=count) for count in (1, 1, 2, 2)
    ]
    checkpoint_identity = LatentBlockBalancedSampler(
        checkpoint_dataset, num_replicas=2, seed=42
    ).identity
    write_stage2_sampler_sidecar(checkpoint, checkpoint_identity, global_step=3)
    parent_calls = []

    def parent_train(trainer, *args, **kwargs):
        del trainer, args, kwargs
        parent_calls.append("train")

    monkeypatch.setattr("stage2.trainer.Trainer.train", parent_train)
    trainer = object.__new__(Stage2Trainer)
    trainer.args = SimpleNamespace(
        output_dir=str(tmp_path),
        should_save=True,
        world_size=2,
        seed=42,
    )
    trainer.train_dataset = [
        SimpleNamespace(latent_block_count=count) for count in (1, 1, 1, 2, 2)
    ]
    trainer.stage2_sampler_identity = None
    trainer.deepspeed_runtime_profile = runtime_profile(2, world_size=2)
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda value, reduction: value,
        wait_for_everyone=lambda: None,
    )

    with pytest.raises(ValueError, match="dataset_size"):
        trainer.train(resume_from_checkpoint=str(checkpoint))

    assert parent_calls == []


def test_stage2_resume_reuses_validated_sampler_for_parent_dataloader(
    tmp_path, monkeypatch
):
    """Catches resume validation scanning every distributed sample twice."""

    class CountingDataset:
        def __init__(self, counts):
            self.examples = [
                SimpleNamespace(latent_block_count=count) for count in counts
            ]
            self.item_reads = 0

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, index):
            self.item_reads += 1
            return self.examples[index]

    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()
    profile = runtime_profile(2, world_size=2)
    write_deepspeed_runtime_sidecar(checkpoint, profile)
    counts = (1, 1, 1, 2, 2)
    checkpoint_identity = LatentBlockBalancedSampler(
        [SimpleNamespace(latent_block_count=count) for count in counts],
        num_replicas=2,
        seed=42,
    ).identity
    write_stage2_sampler_sidecar(checkpoint, checkpoint_identity, global_step=3)
    dataset = CountingDataset(counts)
    parent_samplers = []

    def parent_train(trainer, *args, **kwargs):
        del args, kwargs
        sampler = trainer._get_train_sampler()
        parent_samplers.append(sampler)
        return sampler

    monkeypatch.setattr("stage2.trainer.Trainer.train", parent_train)
    trainer = object.__new__(Stage2Trainer)
    trainer.args = SimpleNamespace(
        output_dir=str(tmp_path),
        should_save=True,
        world_size=2,
        seed=42,
    )
    trainer.train_dataset = dataset
    trainer.stage2_sampler_identity = None
    trainer.deepspeed_runtime_profile = profile
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda value, reduction: value,
        wait_for_everyone=lambda: None,
    )

    sampler = trainer.train(resume_from_checkpoint=str(checkpoint))

    assert parent_samplers == [sampler]
    assert sampler.dataset is dataset
    assert dataset.item_reads == len(dataset)


def test_stage2_resume_rejects_runtime_before_parent_train(tmp_path, monkeypatch):
    """Catches DeepSpeed initialization preceding resume identity validation."""
    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()
    write_deepspeed_runtime_sidecar(
        checkpoint, runtime_profile(1, world_size=2)
    )
    parent_calls = []

    def parent_train(trainer, *args, **kwargs):
        del trainer, args, kwargs
        parent_calls.append("train")

    monkeypatch.setattr("stage2.trainer.Trainer.train", parent_train)
    trainer = object.__new__(Stage2Trainer)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path), should_save=True)
    trainer.deepspeed_runtime_profile = runtime_profile(2, world_size=2)
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda value, reduction: value,
        wait_for_everyone=lambda: None,
    )

    with pytest.raises(ValueError, match="zero_stage"):
        trainer.train(resume_from_checkpoint=str(checkpoint))

    assert parent_calls == []


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_stage2_trainer_final_peft_save_is_adapter_only_with_sidecar(
    tmp_path, isolated_trainer_deepspeed_state
):
    """Catches final PEFT saving dropping metadata or copying full base weights."""
    tuning = Stage2TuningConfig(
        method="lora", r=4, alpha=8, dropout=0.0, target_modules=("q_proj",)
    )
    base = tiny_stage2_qwen()
    metadata = tiny_trainer_metadata(base, tuning)
    model = build_stage2_tuned_model(base, tuning)
    trainer = Stage2Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True),
        stage2_peft_metadata=metadata,
        deepspeed_runtime_profile=runtime_profile(),
    )

    trainer.save_model(tmp_path)

    assert (tmp_path / "adapter_config.json").is_file()
    assert (tmp_path / "adapter_model.safetensors").is_file()
    assert (tmp_path / STAGE2_PEFT_METADATA_FILENAME).is_file()
    assert not (tmp_path / "model.safetensors").exists()
    assert not (tmp_path / "pytorch_model.bin").exists()
    assert load_stage2_peft_metadata(tmp_path) == metadata


def test_stage2_trainer_peft_save_excludes_resized_base_embeddings(
    tmp_path, isolated_trainer_deepspeed_state
):
    """Catches PEFT auto-saving resized base embeddings with the adapter."""
    source = tmp_path / "original-base"
    source.mkdir()
    base = tiny_stage2_qwen()
    source_config = deepcopy(base.config)
    source_config.vocab_size -= 3
    source_config.save_pretrained(source)
    tuning = Stage2TuningConfig(
        method="dora", r=4, alpha=8, dropout=0.0, target_modules=("q_proj",)
    )
    metadata = tiny_trainer_metadata(base, tuning)
    model = build_stage2_tuned_model(base, tuning)
    model.peft_config["default"].base_model_name_or_path = str(source)
    trainer = Stage2Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path), report_to=[], use_cpu=True
        ),
        stage2_peft_metadata=metadata,
        deepspeed_runtime_profile=runtime_profile(),
    )

    trainer.save_model(tmp_path)

    saved_keys = set(load_file(tmp_path / "adapter_model.safetensors"))
    assert saved_keys
    assert all("lora_" in key for key in saved_keys)


@pytest.mark.parametrize("explicit_output", [False, True])
def test_stage2_sparse_deepspeed_save_has_all_rank_collective_participation(
    tmp_path, monkeypatch, explicit_output
):
    """Catches a non-writing ZeRO rank skipping the sparse-row collective."""
    final_output = tmp_path / "final"
    actual_output = tmp_path / "checkpoint-1" if explicit_output else final_output
    metadata = replace(
        sample_metadata(),
        token_row_source="sparse_adapter",
        stage1_adapter_reference="/deleted/stage1/provenance-only",
        latent_pad_token_id=125,
        latent_start_token_id=126,
        latent_end_token_id=127,
    )
    events = []
    temporary_targets = []
    active_rank = None

    def framework_save(trainer, output_dir=None, _internal_call=False):
        target = Path(output_dir or trainer.args.output_dir)
        events.append((active_rank, "framework", target))
        if trainer.args.should_save:
            target.mkdir(parents=True, exist_ok=True)
            (target / "processor.json").write_text("{}\n", encoding="utf-8")

    def sparse_save(model, output_dir):
        target = Path(output_dir)
        events.append((active_rank, "sparse", target))
        target.mkdir(parents=True, exist_ok=True)
        (target / "ilvr_token_rows.safetensors").write_bytes(b"rows")
        if active_rank == 1:
            temporary_targets.append(target)

    def metadata_save(output_dir, saved_metadata):
        target = Path(output_dir)
        events.append((active_rank, "metadata", target))
        assert saved_metadata == metadata
        target.mkdir(parents=True, exist_ok=True)
        (target / STAGE2_PEFT_METADATA_FILENAME).write_text(
            "{}\n", encoding="utf-8"
        )

    monkeypatch.setattr("stage2.trainer.Trainer.save_model", framework_save)
    monkeypatch.setattr("stage2.trainer.save_sparse_token_rows", sparse_save)
    monkeypatch.setattr("stage2.trainer.save_stage2_peft_metadata", metadata_save)
    monkeypatch.setattr(
        "stage2.trainer.sparse_token_rows_are_partitioned", lambda model: True
    )

    trainer = object.__new__(Stage2Trainer)
    trainer.model = object()
    trainer.stage2_peft_metadata = metadata
    trainer.is_deepspeed_enabled = True
    for rank, should_save in ((1, False), (0, True)):
        active_rank = rank
        trainer.args = SimpleNamespace(
            output_dir=str(final_output), should_save=should_save
        )
        trainer.save_model(actual_output if explicit_output else None)

    assert [(rank, action) for rank, action, _ in events] == [
        (1, "framework"),
        (1, "sparse"),
        (0, "framework"),
        (0, "sparse"),
        (0, "metadata"),
    ]
    assert temporary_targets
    assert all(target != actual_output for target in temporary_targets)
    assert all(not target.exists() for target in temporary_targets)
    assert (actual_output / "ilvr_token_rows.safetensors").is_file()
    assert (actual_output / STAGE2_PEFT_METADATA_FILENAME).is_file()
    assert (actual_output / "processor.json").is_file()


def test_stage2_sparse_replicated_save_is_rank_zero_only(tmp_path, monkeypatch):
    """Catches ZeRO-1/2 replicated sparse rows entering a gather collective."""
    metadata = replace(sample_metadata(), token_row_source="sparse_adapter")
    saved_ranks = []
    active_rank = None

    monkeypatch.setattr(
        "stage2.trainer.Trainer.save_model",
        lambda trainer, output_dir=None, _internal_call=False: None,
    )
    monkeypatch.setattr(
        "stage2.trainer.sparse_token_rows_are_partitioned", lambda model: False
    )
    monkeypatch.setattr(
        "stage2.trainer.save_sparse_token_rows",
        lambda model, output_dir: saved_ranks.append(active_rank),
    )
    monkeypatch.setattr(
        "stage2.trainer.save_stage2_peft_metadata",
        lambda output_dir, saved_metadata: None,
    )
    trainer = object.__new__(Stage2Trainer)
    trainer.model = object()
    trainer.stage2_peft_metadata = metadata
    for rank, should_save in ((1, False), (0, True)):
        active_rank = rank
        trainer.args = SimpleNamespace(
            output_dir=str(tmp_path), should_save=should_save
        )
        trainer.save_model()

    assert saved_ranks == [0]


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_stage2_trainer_numbered_peft_checkpoint_keeps_resume_state(
    tmp_path, isolated_trainer_deepspeed_state
):
    """Catches sidecar saving replacing Trainer's ordinary checkpoint machinery."""
    tuning = Stage2TuningConfig(
        method="lora", r=4, alpha=8, dropout=0.0, target_modules=("q_proj",)
    )
    base = tiny_stage2_qwen()
    metadata = tiny_trainer_metadata(base, tuning)
    model = build_stage2_tuned_model(base, tuning)
    trainer = Stage2Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            max_steps=1,
            per_device_train_batch_size=1,
            save_strategy="steps",
            save_steps=1,
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
        ),
        train_dataset=[tiny_trainer_batch()],
        data_collator=default_data_collator,
        processing_class=TinySavingProcessor(),
        stage2_peft_metadata=metadata,
        deepspeed_runtime_profile=runtime_profile(),
    )

    trainer.train()

    checkpoint = tmp_path / "checkpoint-1"
    for filename in (
        "adapter_config.json",
        "adapter_model.safetensors",
        STAGE2_PEFT_METADATA_FILENAME,
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        assert (checkpoint / filename).is_file(), filename
    assert not (checkpoint / "model.safetensors").exists()
    assert not (checkpoint / "pytorch_model.bin").exists()
    assert (checkpoint / "tiny_processor.txt").read_text(encoding="utf-8") == (
        "latent_pad=90\nlatent_start=91\nlatent_end=92\n"
    )

    resumed_model = build_stage2_tuned_model(
        tiny_stage2_qwen(),
        tuning,
        adapter_path=checkpoint,
        is_trainable=True,
    )
    resumed_trainer = Stage2Trainer(
        model=resumed_model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            max_steps=2,
            per_device_train_batch_size=1,
            save_strategy="steps",
            save_steps=1,
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
        ),
        train_dataset=[tiny_trainer_batch()],
        data_collator=default_data_collator,
        processing_class=TinySavingProcessor(),
        stage2_peft_metadata=metadata,
        deepspeed_runtime_profile=runtime_profile(),
    )

    resumed_trainer.train(resume_from_checkpoint=str(checkpoint))

    assert resumed_trainer.state.global_step == 2
    resumed_checkpoint = tmp_path / "checkpoint-2"
    assert (resumed_checkpoint / "tiny_processor.txt").is_file()
    assert (resumed_checkpoint / STAGE2_PEFT_METADATA_FILENAME).is_file()


def test_stage2_trainer_full_save_writes_no_peft_sidecar(
    tmp_path, isolated_trainer_deepspeed_state
):
    """Catches full tuning being mislabeled as an adapter checkpoint."""
    trainer = Stage2Trainer(
        model=tiny_stage2_qwen(),
        args=TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True),
        deepspeed_runtime_profile=runtime_profile(),
    )

    trainer.save_model(tmp_path)

    assert not (tmp_path / STAGE2_PEFT_METADATA_FILENAME).exists()
