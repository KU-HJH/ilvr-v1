"""Explicit preparation of a relocated historical Stage 1 initializer."""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from stage1.checkpointing import Stage1PeftMetadata
from tests.stage1.test_checkpointing import sample_metadata


@pytest.fixture
def artifact_root():
    root = Path("checkpoints/test-original-stage1-relocation")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        yield Path(temporary)


def make_fixture(root):
    source = root / "original-stage1"
    source.mkdir()
    old_base = root / "original-base"
    new_base = root / "local-base"
    for base in (old_base, new_base):
        base.mkdir()
        (base / "config.json").write_text('{"model_type":"qwen2_5_vl"}')
        (base / "model.safetensors").write_bytes(b"base-model-weights")
    payload = sample_metadata().to_dict()
    payload.update(
        schema_version=3,
        base_model_reference="/oldhost/models/original-base",
        ema_setting="frozen",
        teacher_target_mode="sample_isolated",
        training_implementation_version=1,
    )
    (source / "stage1_peft_metadata.json").write_text(json.dumps(payload) + "\n")
    (source / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": payload["base_model_reference"]}) + "\n"
    )
    (source / "adapter_model.safetensors").write_bytes(b"original-adapter-weights")
    (source / "ilvr_token_rows.safetensors").write_bytes(b"original-sparse-rows")
    (source / "tokenizer.json").write_bytes(b"original-tokenizer")
    return source, old_base, new_base, root / "fresh-initializer"


def test_preparation_preserves_weights_and_original_provenance(artifact_root):
    from stage1.relocation import prepare_original_stage1_initializer

    source, old_base, new_base, destination = make_fixture(artifact_root)
    manifest = prepare_original_stage1_initializer(
        source, old_base, new_base, destination
    )
    assert destination.is_dir()
    for name in ("adapter_model.safetensors", "ilvr_token_rows.safetensors", "tokenizer.json"):
        assert (destination / name).read_bytes() == (source / name).read_bytes()
    metadata = Stage1PeftMetadata.from_dict(
        json.loads((destination / "stage1_peft_metadata.json").read_text())
    )
    assert metadata.schema_version == 3
    assert metadata.base_model_reference == str(new_base.resolve())
    assert (metadata.ema_setting, metadata.teacher_target_mode,
            metadata.training_implementation_version) == ("frozen", "sample_isolated", 1)
    adapter_config = json.loads((destination / "adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"] == str(new_base.resolve())
    original = json.loads((source / "stage1_peft_metadata.json").read_text())
    assert original["base_model_reference"] == "/oldhost/models/original-base"
    assert manifest["original_base_model_reference"] == original["base_model_reference"]
    assert manifest["source_metadata_sha256"] == hashlib.sha256(
        (source / "stage1_peft_metadata.json").read_bytes()
    ).hexdigest()
    assert json.loads((destination / "original_stage1_provenance.json").read_text()) == manifest


def test_preparation_rejects_different_base_weights_before_writing(artifact_root):
    from stage1.relocation import prepare_original_stage1_initializer

    source, old_base, new_base, destination = make_fixture(artifact_root)
    (new_base / "model.safetensors").write_bytes(b"different-weights")
    with pytest.raises(ValueError, match="base model contents differ"):
        prepare_original_stage1_initializer(source, old_base, new_base, destination)
    assert not destination.exists()


def test_preparation_rejects_adapter_base_mismatch_and_existing_destination(artifact_root):
    from stage1.relocation import prepare_original_stage1_initializer

    source, old_base, new_base, destination = make_fixture(artifact_root)
    config = json.loads((source / "adapter_config.json").read_text())
    config["base_model_name_or_path"] = "another/base"
    (source / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="adapter base reference mismatch"):
        prepare_original_stage1_initializer(source, old_base, new_base, destination)
    assert not destination.exists()
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("untouched")
    with pytest.raises(FileExistsError):
        prepare_original_stage1_initializer(source, old_base, new_base, destination)
    assert sentinel.read_text() == "untouched"


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_relocated_tiny_dora_loads_for_fresh_stage2_and_persists_reference(artifact_root):
    from stage1.relocation import prepare_original_stage1_initializer
    from stage2.checkpointing import (
        build_stage2_peft_metadata,
        load_stage2_peft_metadata,
        resolve_stage2_load_decision,
        save_stage2_peft_metadata,
    )
    from stage2.peft import load_stage1_dora_for_fresh_stage2
    from tests.stage1.test_checkpointing import tiny_original_qwen
    from tests.stage2.test_stage1_checkpoint_handoff import (
        stage2_dora_tuning,
        write_stage1_dora_artifact,
    )
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds

    source = artifact_root / "source-adapter"
    old_base = artifact_root / "original-qwen"
    new_base = artifact_root / "local-qwen"
    for base in (old_base, new_base):
        base.mkdir()
        (base / "config.json").write_bytes(b"same-config")
        (base / "model.safetensors").write_bytes(b"same-base-weights")
    processor, metadata = write_stage1_dora_artifact(source)
    payload = metadata.to_dict()
    payload.update(schema_version=3, ema_setting="frozen",
                   teacher_target_mode="sample_isolated", training_implementation_version=1)
    (source / "stage1_peft_metadata.json").write_text(json.dumps(payload) + "\n")
    destination = artifact_root / "fresh-stage1"
    prepare_original_stage1_initializer(source, old_base, new_base, destination)
    tuning = stage2_dora_tuning()
    decision = resolve_stage2_load_decision(
        load_model_path=str(new_base.resolve()),
        stage1_adapter_path=str(destination.resolve()),
        resume_from_checkpoint_path=None,
        load_config=lambda _: tiny_original_qwen().config,
        latent_size=2,
        supervise_latent_boundaries=True,
        tuning_method="dora",
        tuning_method_explicit=True,
        lora_r=tuning.r,
        lora_alpha=tuning.alpha,
        lora_dropout=tuning.dropout,
        lora_target_modules=tuning.target_modules,
    )
    assert decision.trainer_resume_path is None
    base = tiny_original_qwen()
    base.config._name_or_path = str(new_base.resolve())
    base.name_or_path = str(new_base.resolve())
    model = load_stage1_dora_for_fresh_stage2(
        base, processor, destination, tuning,
        requested_base_model_reference=str(new_base.resolve()),
    )
    assert model.peft_config["default"].base_model_name_or_path == str(new_base.resolve())
    from peft_common import resolve_qwen2_5_vl_model

    qwen = resolve_qwen2_5_vl_model(model)
    qwen.config.supervise_latent_boundaries = True
    stage2_metadata = build_stage2_peft_metadata(
        tuning=tuning,
        base_model_reference=str(new_base.resolve()),
        model_config=qwen.config,
        token_ids=LatentTokenIds(*metadata.latent_token_ids),
        resolved_target_names=metadata.resolved_target_names,
        token_row_source="sparse_adapter",
        stage1_adapter_reference=str(destination.resolve()),
        tokenizer_base_vocab_size=metadata.base_vocab_size,
        pretrained_model_vocab_size=metadata.pretrained_model_vocab_size,
        resized_vocab_size=metadata.resized_vocab_size,
        source_base_model_architecture=metadata.base_model_architecture,
    )
    saved = artifact_root / "stage2-sidecar"
    save_stage2_peft_metadata(saved, stage2_metadata)
    assert load_stage2_peft_metadata(saved).stage1_adapter_reference == str(destination.resolve())
