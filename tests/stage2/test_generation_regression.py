import inspect

import pytest
import torch

from helpers import tiny_stage2_qwen
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.cache_utils import DynamicCache
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerationMixin


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
