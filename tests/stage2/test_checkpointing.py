from types import SimpleNamespace

import pytest

from stage2.checkpointing import (
    Stage2LoadDecision,
    configure_stage2_metadata,
    resolve_latent_token_ids,
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


def test_resume_requires_stage2_and_uses_single_source_for_all_state():
    decision = resolve_stage2_load_decision(
        load_model_path="stage1/model",
        resume_from_checkpoint_path="stage2/checkpoint-10",
        load_config=lambda _: SimpleNamespace(
            stage="stage2", latent_size=4, supervise_latent_boundaries=False
        ),
        latent_size=4,
        supervise_latent_boundaries=False,
    )
    assert decision == Stage2LoadDecision(
        model_source="stage2/checkpoint-10",
        processor_source="stage2/checkpoint-10",
        trainer_resume_path="stage2/checkpoint-10",
        latent_size=4,
        supervise_latent_boundaries=False,
    )


def test_resume_rejects_stage1_source():
    with pytest.raises(ValueError, match="expected stage2 checkpoint for resume"):
        resolve_stage2_load_decision(
            load_model_path="stage1/model",
            resume_from_checkpoint_path="stage1/checkpoint-10",
            load_config=lambda _: SimpleNamespace(stage="stage1"),
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
