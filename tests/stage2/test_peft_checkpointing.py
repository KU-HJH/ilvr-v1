import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from peft import LoraConfig
from torch import nn

from helpers import tiny_stage2_qwen
from stage2.checkpointing import (
    STAGE2_PEFT_METADATA_FILENAME,
    Stage2PeftMetadata,
    build_stage2_peft_metadata,
    load_stage2_peft_metadata,
    resolve_stage2_load_decision,
    save_stage2_peft_metadata,
    validate_stage2_peft_compatibility,
)
from stage2.peft import (
    Stage2TuningConfig,
    build_stage2_tuned_model,
    resolve_stage2_target_module_names,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds


def sample_metadata() -> Stage2PeftMetadata:
    return Stage2PeftMetadata(
        schema_version=1,
        stage="stage2",
        tuning_method="dora",
        base_model_reference="/models/stage1",
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
    )


def write_mutated_metadata(directory, metadata, mutation):
    payload = metadata.to_dict()
    if mutation == "bad_schema":
        payload["schema_version"] = 999
    elif mutation == "bad_stage":
        payload["stage"] = "stage1"
    elif mutation == "missing_field":
        payload.pop("latent_size")
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


@pytest.mark.parametrize("mutation", ["bad_schema", "bad_stage", "missing_field"])
def test_stage2_peft_metadata_fails_closed(tmp_path, mutation):
    write_mutated_metadata(tmp_path, sample_metadata(), mutation)

    with pytest.raises(ValueError):
        load_stage2_peft_metadata(tmp_path)


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


def write_tiny_adapter(directory, *, method="lora"):
    tuning = Stage2TuningConfig(
        method=method,
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=("q_proj", "v_proj"),
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


def test_relocated_stage1_base_accepts_dora_magnitude_tensor_naming(tmp_path):
    metadata, adapter_config = write_tiny_adapter(tmp_path, method="dora")

    validate_stage2_peft_compatibility(
        relocated_base(), TinyTokenizer(), metadata, adapter_config, tmp_path
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("token_id", "latent_start_token_id mismatch"),
        ("hidden_size", "hidden size"),
        ("missing_target", "missing resolved target"),
        ("target_shape", "target shape mismatch"),
        ("method", "method.*mismatch"),
        ("rank", "rank.*mismatch"),
        ("alpha", "alpha.*mismatch"),
        ("dropout", "dropout.*mismatch"),
        ("use_dora", "use_dora.*mismatch"),
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
    elif mutation == "hidden_size":
        base.config.hidden_size += 1
    elif mutation == "missing_target":
        delattr(base.model.layers[0].self_attn, "q_proj")
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

    with pytest.raises(ValueError, match=message):
        validate_stage2_peft_compatibility(
            base, tokenizer, metadata, adapter_config, tmp_path
        )
