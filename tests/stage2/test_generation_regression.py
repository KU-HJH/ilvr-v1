import inspect

import pytest
import torch
from torch import nn

from helpers import tiny_stage2_qwen
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.cache_utils import DynamicCache
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerationMixin


def tiny_ordinary_qwen():
    model = tiny_stage2_qwen()
    model.config.stage = None
    model.config.use_cache = True
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    model.eval()
    return model


class DeterministicVisualStub(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.register_buffer("feature", torch.arange(hidden_size, dtype=torch.float32))
        self.calls = 0

    @property
    def dtype(self):
        return self.feature.dtype

    def forward(self, pixel_values, grid_thw):
        self.calls += 1
        return self.feature.unsqueeze(0).expand(grid_thw.shape[0], -1)


def test_generic_sample_exposes_only_the_standard_sampling_contract():
    parameters = inspect.signature(GenerationMixin._sample).parameters

    assert "tokenizer" not in parameters
    assert not hasattr(GenerationMixin, "_sample_backup")


def test_generic_decoder_output_has_only_standard_generation_state():
    assert "latent_hidden_states" not in GenerateDecoderOnlyOutput.__dataclass_fields__


@pytest.mark.parametrize("return_dict_in_generate", [False, True])
def test_tiny_text_model_generates_with_standard_arguments_and_output(return_dict_in_generate):
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_positions=8,
            n_ctx=8,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=15,
            pad_token_id=0,
        )
    )
    model.eval()

    output = model.generate(
        torch.tensor([[1, 2]]),
        max_new_tokens=2,
        do_sample=False,
        return_dict_in_generate=return_dict_in_generate,
    )

    if return_dict_in_generate:
        assert isinstance(output, GenerateDecoderOnlyOutput)
        assert output.sequences.shape == (1, 4)
        assert "latent_hidden_states" not in output
    else:
        assert type(output) is torch.Tensor
        assert output.shape == (1, 4)


def test_local_qwen_ordinary_greedy_generate_returns_tensor():
    """Catches local Qwen bypassing or breaking the generic greedy sampler."""
    torch.manual_seed(31)
    model = tiny_ordinary_qwen()

    output = model.generate(
        torch.tensor([[1, 2]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        max_new_tokens=2,
        do_sample=False,
        return_dict_in_generate=False,
    )

    assert type(output) is torch.Tensor
    assert output.shape == (1, 4)


def test_local_qwen_ordinary_seeded_sampling_returns_structured_output():
    """Catches local Qwen stochastic sampling or structured output regressions."""
    torch.manual_seed(37)
    model = tiny_ordinary_qwen()
    inputs = torch.tensor([[1, 2]])
    attention_mask = torch.ones_like(inputs)

    torch.manual_seed(41)
    first = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=2,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    model.rope_deltas = None
    torch.manual_seed(41)
    second = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=2,
        do_sample=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    assert isinstance(first, GenerateDecoderOnlyOutput)
    assert first.sequences.shape == (1, 4)
    assert len(first.scores) == 2
    assert torch.equal(first.sequences, second.sequences)


def test_local_qwen_multimodal_prefill_uses_cached_decode_without_reencoding_visual():
    """Catches cached local-Qwen decode replaying multimodal pixels or losing cache."""
    torch.manual_seed(43)
    model = tiny_ordinary_qwen()
    visual = DeterministicVisualStub(model.config.hidden_size)
    model.visual = visual
    calls = []
    original_forward = model.forward

    def recording_forward(*args, **kwargs):
        past = kwargs.get("past_key_values")
        calls.append(
            {
                "cache_length": None if past is None else past.get_seq_length(),
                "has_pixels": kwargs.get("pixel_values") is not None,
            }
        )
        return original_forward(*args, **kwargs)

    model.forward = recording_forward
    input_ids = torch.tensor([[model.config.vision_start_token_id, model.config.image_token_id, 2]])

    output = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=torch.ones(1, 3),
        image_grid_thw=torch.tensor([[1, 1, 1]]),
        max_new_tokens=2,
        do_sample=False,
        return_dict_in_generate=True,
    )

    assert isinstance(output, GenerateDecoderOnlyOutput)
    assert output.sequences.shape == (1, 5)
    assert visual.calls == 1
    assert calls[0] == {"cache_length": 0, "has_pixels": True}
    assert any(call["cache_length"] > 0 and not call["has_pixels"] for call in calls[1:])


def test_qwen_generation_preparation_preserves_prefill_multimodal_inputs():
    model = tiny_stage2_qwen()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    pixel_values_videos = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    image_grid_thw = torch.tensor([[1, 1, 1], [1, 1, 1]])
    video_grid_thw = torch.tensor([[1, 1, 1], [1, 1, 1]])
    second_per_grid_ts = torch.tensor([0.5, 1.0])
    cache_position = torch.arange(input_ids.shape[1])

    model_inputs = model.prepare_inputs_for_generation(
        input_ids,
        attention_mask=attention_mask,
        cache_position=cache_position,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
    )

    assert torch.equal(model_inputs["input_ids"], input_ids)
    assert torch.equal(model_inputs["attention_mask"], attention_mask)
    assert torch.equal(model_inputs["cache_position"], cache_position)
    assert model_inputs["position_ids"] is None
    assert torch.equal(model_inputs["pixel_values"], pixel_values)
    assert torch.equal(model_inputs["pixel_values_videos"], pixel_values_videos)
    assert torch.equal(model_inputs["image_grid_thw"], image_grid_thw)
    assert torch.equal(model_inputs["video_grid_thw"], video_grid_thw)
    assert torch.equal(model_inputs["second_per_grid_ts"], second_per_grid_ts)
    assert model_inputs["use_cache"] is True
    assert model_inputs["generate_mode"] is True
    assert not {
        "latent_hidden_states",
        "latent_start_id",
        "latent_end_id",
        "latent_pad_id",
    }.intersection(model_inputs)


def test_qwen_generation_preparation_slices_cached_tokens_and_suppresses_pixels():
    model = tiny_stage2_qwen()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    pixel_values_videos = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    image_grid_thw = torch.tensor([[1, 1, 1], [1, 1, 1]])
    video_grid_thw = torch.tensor([[1, 1, 1], [1, 1, 1]])
    second_per_grid_ts = torch.tensor([0.5, 1.0])
    cache_position = torch.tensor([2])
    past_key_values = DynamicCache()

    model_inputs = model.prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        cache_position=cache_position,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
    )

    assert model_inputs["past_key_values"] is past_key_values
    assert torch.equal(model_inputs["input_ids"], torch.tensor([[3]]))
    assert torch.equal(model_inputs["attention_mask"], attention_mask)
    assert torch.equal(model_inputs["cache_position"], cache_position)
    assert model_inputs["position_ids"] is None
    assert model_inputs["pixel_values"] is None
    assert model_inputs["pixel_values_videos"] is None
    assert torch.equal(model_inputs["image_grid_thw"], image_grid_thw)
    assert torch.equal(model_inputs["video_grid_thw"], video_grid_thw)
    assert torch.equal(model_inputs["second_per_grid_ts"], second_per_grid_ts)
    assert model_inputs["use_cache"] is True
    assert model_inputs["generate_mode"] is True
    assert not {
        "latent_hidden_states",
        "latent_start_id",
        "latent_end_id",
        "latent_pad_id",
    }.intersection(model_inputs)
