import copy
import functools
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from helpers import ToyCausalLanguageModel
from stage2.inference import (
    Stage2GenerationConfig,
    Stage2InferenceController,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentRolloutEngine,
    LatentSpan,
    LatentTokenIds,
    parse_latent_spans,
)


TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)


class ScriptedHead(nn.Module):
    def __init__(self, token_ids, vocab_size=128):
        super().__init__()
        self.token_ids = list(token_ids)
        self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, hidden):
        token_id = self.token_ids[min(self.calls, len(self.token_ids) - 1)]
        self.calls += 1
        logits = hidden.new_full((hidden.shape[0], self.vocab_size), -1e9)
        logits[:, token_id] = 0
        return logits


class ScriptedStage2Model(nn.Module):
    def __init__(self, *, emitted_ids, latent_size):
        super().__init__()
        self.config = SimpleNamespace(latent_size=latent_size)
        self.model = ToyCausalLanguageModel(4)
        self.lm_head = ScriptedHead(emitted_ids)
        self.stage2_rollout_engine = LatentRolloutEngine()
        self.visual_calls = 0
        self.rope_inputs = []

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def prepare_stage2_base_inputs_embeds(
        self, *, input_ids, pixel_values, image_grid_thw
    ):
        self.visual_calls += int(pixel_values is not None)
        return self.get_input_embeddings()(input_ids)

    def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
        self.rope_inputs.append(
            (
                input_ids.clone(),
                None if image_grid_thw is None else image_grid_thw.clone(),
                None if attention_mask is None else attention_mask.clone(),
            )
        )
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        return positions, torch.zeros(input_ids.shape[0], 1, device=input_ids.device)


def scripted_model(*, latent_size, emitted_ids):
    return ScriptedStage2Model(emitted_ids=emitted_ids, latent_size=latent_size)


def test_inference_forces_exactly_k_latents_and_supports_multiple_blocks():
    model = scripted_model(
        latent_size=2,
        emitted_ids=[4, TOKENS.start_token_id, 5, TOKENS.start_token_id, 3],
    )
    result = Stage2InferenceController().generate(
        model=model,
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=16, eos_token_id=3),
        pixel_values=torch.ones(1, 3),
        image_grid_thw=torch.tensor([[1, 1, 1]]),
        return_latent_states=True,
    )

    spans = parse_latent_spans(result.sequences[0], token_ids=TOKENS, latent_size=2)
    assert [len(span.latent_positions) for span in spans] == [2, 2]
    assert len(result.latent_states_by_block) == 2
    assert all(len(block) == 2 for block in result.latent_states_by_block)
    assert model.lm_head.calls == 5
    assert model.visual_calls == 1


def test_started_block_finishes_past_nominal_budget_without_vocabulary_sampling():
    model = scripted_model(latent_size=2, emitted_ids=[TOKENS.start_token_id])

    result = Stage2InferenceController().generate(
        model=model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=1),
    )

    assert result.sequences[0, -4:].tolist() == [91, 90, 90, 92]
    assert model.lm_head.calls == 1


def test_eos_is_checked_only_after_a_completed_latent_block():
    model = scripted_model(latent_size=2, emitted_ids=[TOKENS.start_token_id, 3])

    result = Stage2InferenceController().generate(
        model=model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=5, eos_token_id=3),
    )

    assert result.sequences[0, -5:].tolist() == [91, 90, 90, 92, 3]
    assert model.lm_head.calls == 2


def test_latent_block_rejects_hard_sequence_cap_before_rollout():
    model = scripted_model(latent_size=2, emitted_ids=[TOKENS.start_token_id])

    with pytest.raises(ValueError, match="latent block would exceed max_sequence_length"):
        Stage2InferenceController().generate(
            model=model,
            input_ids=torch.tensor([[1]]),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            token_ids=TOKENS,
            generation_config=Stage2GenerationConfig(
                max_new_tokens=1,
                max_sequence_length=4,
            ),
        )

    assert len(model.model.calls) == 1


def test_ordinary_text_rejects_hard_sequence_cap():
    model = scripted_model(latent_size=2, emitted_ids=[4])

    with pytest.raises(ValueError, match="max_sequence_length"):
        Stage2InferenceController().generate(
            model=model,
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            token_ids=TOKENS,
            generation_config=Stage2GenerationConfig(
                max_new_tokens=1,
                max_sequence_length=2,
            ),
        )


def test_inference_rejects_batch_greater_than_one():
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        Stage2InferenceController().generate(
            model=scripted_model(latent_size=2, emitted_ids=[4]),
            input_ids=torch.ones(2, 2, dtype=torch.long),
            attention_mask=torch.ones(2, 2, dtype=torch.long),
            token_ids=TOKENS,
            generation_config=Stage2GenerationConfig(),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": -1},
        {"max_sequence_length": 0},
        {"temperature": -0.1},
        {"top_p": 0.0},
        {"top_p": 1.1},
    ],
)
def test_generation_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        Stage2GenerationConfig(**kwargs)


def test_inference_rejects_non_positive_latent_size():
    with pytest.raises(ValueError, match="latent_size must be positive"):
        Stage2InferenceController().generate(
            model=scripted_model(latent_size=0, emitted_ids=[4]),
            input_ids=torch.tensor([[1]]),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            token_ids=TOKENS,
            generation_config=Stage2GenerationConfig(max_new_tokens=1),
        )


def test_training_and_inference_share_latent_states_and_full_mrope_positions():
    training_model = scripted_model(latent_size=2, emitted_ids=[TOKENS.start_token_id])
    inference_model = copy.deepcopy(training_model)
    scaffold = torch.tensor([[1, 91, 90, 90, 92]])
    mask = torch.ones_like(scaffold)
    image_grid_thw = torch.tensor([[1, 1, 1]])
    positions, _ = training_model.get_rope_index(
        scaffold,
        image_grid_thw=image_grid_thw,
        attention_mask=mask,
    )
    training = training_model.stage2_rollout_engine.rollout(
        language_model=functools.partial(training_model.model, generate_mode=True),
        base_inputs_embeds=training_model.prepare_stage2_base_inputs_embeds(
            input_ids=scaffold,
            pixel_values=None,
            image_grid_thw=image_grid_thw,
        ),
        attention_mask=mask,
        full_position_ids=positions,
        latent_spans=(LatentSpan(1, (2, 3), 4),),
    ).latent_states

    inference = Stage2InferenceController().generate(
        model=inference_model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=1),
        image_grid_thw=image_grid_thw,
        return_latent_states=True,
    )

    assert len(training) == len(inference.latent_states_by_block[0]) == 2
    for train_state, inference_state in zip(training, inference.latent_states_by_block[0]):
        assert torch.allclose(train_state, inference_state, atol=1e-6, rtol=1e-5)
    full_ids, full_grid, full_mask = inference_model.rope_inputs[-1]
    assert torch.equal(full_ids, scaffold)
    assert torch.equal(full_grid, image_grid_thw)
    assert torch.equal(full_mask, mask)


def test_latent_states_are_returned_only_when_requested():
    model = scripted_model(latent_size=2, emitted_ids=[TOKENS.start_token_id])

    result = Stage2InferenceController().generate(
        model=model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=1),
    )

    assert result.latent_states_by_block == ()


def test_returned_inference_latent_states_do_not_retain_gradients():
    model = scripted_model(latent_size=2, emitted_ids=[TOKENS.start_token_id])

    result = Stage2InferenceController().generate(
        model=model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=1),
        return_latent_states=True,
    )

    assert all(
        state.requires_grad is False
        for block in result.latent_states_by_block
        for state in block
    )
