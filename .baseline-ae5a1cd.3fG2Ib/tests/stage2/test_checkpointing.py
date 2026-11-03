from types import SimpleNamespace

import pytest

from stage2.checkpointing import (
    Stage2ArtifactInspection,
    Stage2ArtifactKind,
    Stage2LoadDecision,
    configure_stage2_metadata,
    resolve_latent_token_ids,
    resolve_stage2b_initialization_decision,
    resolve_stage2_load_decision,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds


class FakeTokenizer:
    unk_token_id = 0
    unk_token = "<unk>"

    def __init__(self, ids):
        self.ids = ids

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, self.unk_token_id)


def no_stage2_peft_metadata(_):
    """Model the full-checkpoint branch without touching a fake filesystem path."""
    raise FileNotFoundError


def test_resolve_latent_ids_requires_tokenizer_config_parity():
    tokenizer = FakeTokenizer(
        {
            "<|latent_pad|>": 10,
            "<|latent_start|>": 11,
            "<|latent_end|>": 12,
        }
    )
    config = SimpleNamespace(latent_token_id=10, latent_start_id=11, latent_end_id=99)
    with pytest.raises(ValueError, match="latent_end_token_id.*tokenizer=12.*config=99"):
        resolve_latent_token_ids(tokenizer, config)


def test_fresh_stage2_resets_trainer_state():
    decision = resolve_stage2_load_decision(
        load_model_path="stage1/model",
        resume_from_checkpoint_path=None,
        load_config=lambda _: SimpleNamespace(stage="stage1"),
        latent_size=4,
        supervise_latent_boundaries=False,
    )
    assert decision.model_source == "stage1/model"
    assert decision.processor_source == "stage1/model"
    assert decision.trainer_resume_path is None
    assert decision.latent_size == 4
    assert decision.supervise_latent_boundaries is False


def test_stage2b_initialization_reuses_resume_resolution_but_clears_trainer_state():
    """Catches Stage 2A optimizer/Trainer state crossing the fresh objective boundary."""
    decision = resolve_stage2b_initialization_decision(
        load_model_path="stage1/original",
        stage2a_checkpoint_path="stage2a/checkpoint-20",
        load_config=lambda _: SimpleNamespace(
            stage="stage2", latent_size=4, supervise_latent_boundaries=False
        ),
        load_peft_metadata=no_stage2_peft_metadata,
        latent_size=4,
        supervise_latent_boundaries=False,
    )

    assert decision == Stage2LoadDecision(
        base_model_source="stage2a/checkpoint-20",
        processor_source="stage2a/checkpoint-20",
        adapter_source=None,
        trainer_resume_path=None,
        tuning=decision.tuning,
        latent_size=4,
        supervise_latent_boundaries=False,
    )


def test_stage2b_full_stage2a_initializes_fresh_dora_without_weakening_resume():
    """Catches treating fresh Stage 2B DoRA as an ordinary PEFT resume."""
    stage2a_checkpoint = "stage2a/checkpoint-20"
    stage2a_config = SimpleNamespace(
        stage="stage2", latent_size=4, supervise_latent_boundaries=True
    )
    decision_kwargs = {
        "load_model_path": "stage1/original",
        "load_config": lambda source: stage2a_config,
        "load_peft_metadata": no_stage2_peft_metadata,
        "latent_size": 4,
        "supervise_latent_boundaries": True,
        "tuning_method": "dora",
        "tuning_method_explicit": True,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_target_modules": "q_proj,k_proj,v_proj",
    }

    with pytest.raises(
        ValueError, match="requested PEFT resume is missing Stage 2 PEFT metadata"
    ):
        resolve_stage2_load_decision(
            resume_from_checkpoint_path=stage2a_checkpoint,
            **decision_kwargs,
        )

    decision = resolve_stage2b_initialization_decision(
        stage2a_checkpoint_path=stage2a_checkpoint,
        inspect_artifact=lambda _: Stage2ArtifactInspection(Stage2ArtifactKind.FULL),
        **decision_kwargs,
    )

    assert decision.base_model_source == stage2a_checkpoint
    assert decision.processor_source == stage2a_checkpoint
    assert decision.adapter_source is None
    assert decision.trainer_resume_path is None
    assert decision.peft_metadata is None
    assert decision.tuning.method == "dora"
    assert decision.tuning.r == 32
    assert decision.tuning.alpha == 64
    assert decision.tuning.dropout == 0.05
    assert decision.tuning.target_modules == ("q_proj", "k_proj", "v_proj")
    assert decision.latent_size == 4
    assert decision.supervise_latent_boundaries is True


def test_stage2b_peft_initialization_rejects_a_relocated_compatible_base_before_load(
    monkeypatch, tmp_path
):
    """Catches loading base B while the negative artifact binds PEFT base A."""
    import stage2.checkpointing as checkpointing

    stage2a_adapter = tmp_path / "stage2a-adapter"
    recorded_base = tmp_path / "base-a"
    requested_base = tmp_path / "base-b"
    for directory in (stage2a_adapter, recorded_base, requested_base):
        directory.mkdir()
    snapshot = object()
    inspection = Stage2ArtifactInspection(
        Stage2ArtifactKind.PEFT,
        peft_metadata_snapshot=snapshot,
    )
    monkeypatch.setattr(
        checkpointing,
        "decode_stage2_peft_metadata_snapshot",
        lambda value: SimpleNamespace(base_model_reference=str(recorded_base))
        if value is snapshot
        else None,
    )
    config_loads = []

    with pytest.raises(ValueError, match="exact Stage 2A PEFT base root"):
        resolve_stage2b_initialization_decision(
            load_model_path=str(requested_base),
            stage2a_checkpoint_path=str(stage2a_adapter),
            load_config=lambda source: config_loads.append(source),
            inspect_artifact=lambda path: inspection,
            latent_size=4,
            supervise_latent_boundaries=False,
        )

    assert config_loads == []


def test_resume_requires_stage2_and_uses_single_source_for_all_state():
    decision = resolve_stage2_load_decision(
        load_model_path="stage1/model",
        resume_from_checkpoint_path="stage2/checkpoint-10",
        load_config=lambda _: SimpleNamespace(
            stage="stage2", latent_size=4, supervise_latent_boundaries=False
        ),
        load_peft_metadata=no_stage2_peft_metadata,
        latent_size=4,
        supervise_latent_boundaries=False,
    )
    assert decision == Stage2LoadDecision(
        base_model_source="stage2/checkpoint-10",
        processor_source="stage2/checkpoint-10",
        adapter_source=None,
        trainer_resume_path="stage2/checkpoint-10",
        tuning=decision.tuning,
        latent_size=4,
        supervise_latent_boundaries=False,
    )


def test_resume_rejects_stage1_source():
    with pytest.raises(ValueError, match="expected stage2 checkpoint for resume"):
        resolve_stage2_load_decision(
            load_model_path="stage1/model",
            resume_from_checkpoint_path="stage1/checkpoint-10",
            load_config=lambda _: SimpleNamespace(stage="stage1"),
            load_peft_metadata=no_stage2_peft_metadata,
            latent_size=4,
            supervise_latent_boundaries=False,
        )


@pytest.mark.parametrize(
    ("checkpoint_values", "requested_values", "message"),
    [
        (
            {"latent_size": 2, "supervise_latent_boundaries": True},
            {"latent_size": 4, "supervise_latent_boundaries": True},
            "latent_size.*checkpoint=2.*requested=4",
        ),
        (
            {"latent_size": 4, "supervise_latent_boundaries": False},
            {"latent_size": 4, "supervise_latent_boundaries": True},
            "supervise_latent_boundaries.*checkpoint=False.*requested=True",
        ),
    ],
)
def test_resume_rejects_semantic_setting_changes(
    checkpoint_values, requested_values, message
):
    """Catches resuming optimizer/global state under a changed recurrence or loss."""
    config = SimpleNamespace(stage="stage2", **checkpoint_values)

    with pytest.raises(ValueError, match=message):
        resolve_stage2_load_decision(
            load_model_path="stage1/model",
            resume_from_checkpoint_path="stage2/checkpoint-10",
            load_config=lambda _: config,
            load_peft_metadata=no_stage2_peft_metadata,
            **requested_values,
        )


def test_resolve_latent_ids_validates_all_canonical_and_legacy_aliases():
    tokenizer = FakeTokenizer(
        {
            "<|latent_pad|>": 10,
            "<|latent_start|>": 11,
            "<|latent_end|>": 12,
        }
    )
    config = SimpleNamespace(
        latent_pad_token_id=10,
        latent_token_id=10,
        latent_start_token_id=11,
        latent_start_id=11,
        latent_end_token_id=12,
        latent_end_id=12,
    )
    assert resolve_latent_token_ids(tokenizer, config) == LatentTokenIds(10, 11, 12)


def test_configure_stage2_metadata_writes_canonical_and_legacy_aliases():
    config = SimpleNamespace()
    configure_stage2_metadata(
        config,
        LatentTokenIds(10, 11, 12),
        latent_size=4,
        supervise_latent_boundaries=True,
    )
    assert config.stage == "stage2"
    assert config.latent_size == 4
    assert config.supervise_latent_boundaries is True
    assert config.latent_pad_token_id == config.latent_token_id == 10
    assert config.latent_start_token_id == config.latent_start_id == 11
    assert config.latent_end_token_id == config.latent_end_id == 12
