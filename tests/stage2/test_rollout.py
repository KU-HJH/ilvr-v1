import pytest
import torch
from types import SimpleNamespace
from torch import nn

from tests.stage2.helpers import ToyCausalLanguageModel
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentRolloutEngine,
    LatentSpan,
    LatentTokenIds,
    parse_latent_spans,
)


TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)


def run_toy_rollout_with_pad_embedding(pad_embedding: torch.Tensor):
    ids = torch.tensor([[1, 91, 90, 90, 92, 2]])
    embedding = nn.Embedding(128, 4)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(128 * 4, dtype=torch.float32).view(128, 4))
        embedding.weight[TOKENS.pad_token_id].copy_(pad_embedding)
    lm = ToyCausalLanguageModel(4)
    spans = parse_latent_spans(ids[0], token_ids=TOKENS, latent_size=2)
    positions = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
    result = LatentRolloutEngine().rollout(
        language_model=lm,
        base_inputs_embeds=embedding(ids),
        attention_mask=torch.ones_like(ids),
        full_position_ids=positions,
        latent_spans=spans,
    )
    final_logits = lm(
        inputs_embeds=result.final_inputs_embeds,
        attention_mask=torch.ones_like(ids),
        position_ids=positions,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    return SimpleNamespace(latent_states=result.latent_states, final_logits=final_logits)


def run_toy_rollout(*, collect_diagnostics: bool = False):
    ids = torch.tensor([[1, 91, 90, 90, 92, 2]])
    spans = parse_latent_spans(ids[0], token_ids=TOKENS, latent_size=2)
    positions = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
    return LatentRolloutEngine().rollout(
        language_model=ToyCausalLanguageModel(4),
        base_inputs_embeds=nn.Embedding(128, 4)(ids),
        attention_mask=torch.ones_like(ids),
        full_position_ids=positions,
        latent_spans=spans,
        collect_diagnostics=collect_diagnostics,
    )


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


def test_rollout_rejects_mismatched_attention_mask():
    with pytest.raises(ValueError, match="attention_mask"):
        LatentRolloutEngine().rollout(
            language_model=ToyCausalLanguageModel(2),
            base_inputs_embeds=torch.zeros(1, 4, 2),
            attention_mask=torch.ones(1, 3),
            full_position_ids=torch.zeros(3, 1, 4, dtype=torch.long),
            latent_spans=(),
        )


def test_rollout_rejects_malformed_full_position_ids():
    with pytest.raises(ValueError, match="full_position_ids"):
        LatentRolloutEngine().rollout(
            language_model=ToyCausalLanguageModel(2),
            base_inputs_embeds=torch.zeros(1, 4, 2),
            attention_mask=torch.ones(1, 4),
            full_position_ids=torch.zeros(1, 4, dtype=torch.long),
            latent_spans=(),
        )


def test_rollout_rejects_mismatched_full_position_ids():
    with pytest.raises(ValueError, match="full_position_ids"):
        LatentRolloutEngine().rollout(
            language_model=ToyCausalLanguageModel(2),
            base_inputs_embeds=torch.zeros(1, 4, 2),
            attention_mask=torch.ones(1, 4),
            full_position_ids=torch.zeros(3, 1, 3, dtype=torch.long),
            latent_spans=(),
        )


def test_second_block_keeps_first_block_context():
    ids = torch.tensor([1, 91, 90, 90, 92, 2, 91, 90, 90, 92, 3])
    spans = parse_latent_spans(ids, token_ids=TOKENS, latent_size=2)
    positions = torch.arange(ids.numel()).view(1, 1, -1).expand(3, 1, -1)
    base = torch.arange(ids.numel() * 2, dtype=torch.float32).view(1, -1, 2)
    changed = base.clone()
    changed[:, 0] += 100
    first_lm = ToyCausalLanguageModel(2)
    second_lm = ToyCausalLanguageModel(2)
    first = LatentRolloutEngine().rollout(
        language_model=first_lm,
        base_inputs_embeds=base,
        attention_mask=torch.ones(1, ids.numel()),
        full_position_ids=positions,
        latent_spans=spans,
    )
    second = LatentRolloutEngine().rollout(
        language_model=second_lm,
        base_inputs_embeds=changed,
        attention_mask=torch.ones(1, ids.numel()),
        full_position_ids=positions,
        latent_spans=spans,
    )

    assert first.prefix_lengths == (2, 3, 7, 8)
    assert len(first.latent_states) == 4
    assert len(first_lm.calls) == 4
    assert [call[0].shape[1] for call in first_lm.calls] == [2, 3, 7, 8]
    assert torch.allclose(
        torch.cat(first.latent_states),
        torch.tensor([[2.0, 3.0], [4.0, 5.5], [25.0, 28.75], [38.5, 44.125]]),
    )
    assert not torch.allclose(first.latent_states[2], second.latent_states[2])


def test_downstream_loss_backpropagates_through_every_latent_state():
    embedding = nn.Embedding(128, 4)
    lm = ToyCausalLanguageModel(4)
    ids = torch.tensor([[1, 91, 90, 90, 92, 2]])
    spans = parse_latent_spans(ids[0], token_ids=TOKENS, latent_size=2)
    positions = torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
    result = LatentRolloutEngine().rollout(
        language_model=lm,
        base_inputs_embeds=embedding(ids),
        attention_mask=torch.ones_like(ids),
        full_position_ids=positions,
        latent_spans=spans,
        retain_latent_grads=True,
    )
    final_hidden = lm(
        inputs_embeds=result.final_inputs_embeds,
        attention_mask=torch.ones_like(ids),
        position_ids=positions,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    final_hidden[:, -1].square().mean().backward()

    assert all(state.grad is not None and torch.isfinite(state.grad).all() for state in result.latent_states)
    assert embedding.weight.grad is not None
    assert lm.scale.grad is not None and lm.scale.grad.abs() > 0


def test_latent_pad_embedding_does_not_define_continuous_latents():
    first = run_toy_rollout_with_pad_embedding(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    second = run_toy_rollout_with_pad_embedding(torch.tensor([1e6, -1e6, 5e5, -5e5]))

    assert all(torch.allclose(a, b) for a, b in zip(first.latent_states, second.latent_states))
    assert torch.allclose(first.final_logits, second.final_logits)


def test_diagnostics_are_observability_only():
    result = run_toy_rollout(collect_diagnostics=True)

    assert result.diagnostics.block_count == 1
    assert result.diagnostics.latent_l2_norms.requires_grad is False
    assert result.diagnostics.adjacent_cosine_similarities.requires_grad is False


def test_zero_state_diagnostics_have_empty_metrics():
    base_inputs_embeds = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
    result = LatentRolloutEngine().rollout(
        language_model=ToyCausalLanguageModel(4),
        base_inputs_embeds=base_inputs_embeds,
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        full_position_ids=torch.arange(4).view(1, 1, 4).expand(3, 1, 4),
        latent_spans=(),
        collect_diagnostics=True,
    )

    assert result.diagnostics.latent_l2_norms.shape == (1, 0)
    assert result.diagnostics.adjacent_cosine_similarities.shape == (1, 0)
