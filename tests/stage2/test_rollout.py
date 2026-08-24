import pytest
import torch

from tests.stage2.helpers import ToyCausalLanguageModel
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentRolloutEngine,
    LatentSpan,
    LatentTokenIds,
    parse_latent_spans,
)


TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)


def test_rollout_uses_hidden_before_each_latent_slot():
    ids = torch.tensor([1, 91, 90, 90, 92, 2])
    spans = parse_latent_spans(ids, token_ids=TOKENS, latent_size=2)
    embeds = torch.arange(12, dtype=torch.float32).view(1, 6, 2)
    positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6)
    lm = ToyCausalLanguageModel(hidden_size=2)

    result = LatentRolloutEngine().rollout(
        language_model=lm,
        base_inputs_embeds=embeds,
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        full_position_ids=positions,
        latent_spans=spans,
    )

    assert result.prefix_lengths == (2, 3)
    assert len(result.latent_states) == 2
    assert torch.allclose(
        result.latent_states[0],
        lm.calls[0][0][:, :2].cumsum(1)[:, -1] * lm.scale + 1,
    )


def test_rollout_rejects_local_batch_larger_than_one():
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        LatentRolloutEngine().rollout(
            language_model=ToyCausalLanguageModel(2),
            base_inputs_embeds=torch.zeros(2, 4, 2),
            attention_mask=torch.ones(2, 4),
            full_position_ids=torch.zeros(3, 2, 4, dtype=torch.long),
            latent_spans=(LatentSpan(0, (1, 2), 3),),
        )


def test_second_block_keeps_first_block_context():
    ids = torch.tensor([1, 91, 90, 90, 92, 2, 91, 90, 90, 92, 3])
    spans = parse_latent_spans(ids, token_ids=TOKENS, latent_size=2)
    positions = torch.arange(ids.numel()).view(1, 1, -1).expand(3, 1, -1)
    base = torch.arange(ids.numel() * 2, dtype=torch.float32).view(1, -1, 2)
    changed = base.clone()
    changed[:, 0] += 100
    first = LatentRolloutEngine().rollout(
        language_model=ToyCausalLanguageModel(2),
        base_inputs_embeds=base,
        attention_mask=torch.ones(1, ids.numel()),
        full_position_ids=positions,
        latent_spans=spans,
    )
    second = LatentRolloutEngine().rollout(
        language_model=ToyCausalLanguageModel(2),
        base_inputs_embeds=changed,
        attention_mask=torch.ones(1, ids.numel()),
        full_position_ids=positions,
        latent_spans=spans,
    )

    assert not torch.allclose(first.latent_states[2], second.latent_states[2])
