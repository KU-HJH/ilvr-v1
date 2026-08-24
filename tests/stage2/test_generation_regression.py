import inspect

import torch

from helpers import tiny_stage2_qwen
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerationMixin


def test_generic_sample_exposes_only_the_standard_sampling_contract():
    parameters = inspect.signature(GenerationMixin._sample).parameters

    assert "tokenizer" not in parameters
    assert not hasattr(GenerationMixin, "_sample_backup")


def test_generic_decoder_output_has_only_standard_generation_state():
    assert "latent_hidden_states" not in GenerateDecoderOnlyOutput.__dataclass_fields__


def test_tiny_text_model_generates_with_standard_arguments_and_output():
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
        return_dict_in_generate=True,
    )

    assert isinstance(output, GenerateDecoderOnlyOutput)
    assert output.sequences.shape == (1, 4)
    assert "latent_hidden_states" not in output


def test_qwen_generation_preparation_sets_only_model_specific_generation_mode():
    model = tiny_stage2_qwen()
    input_ids = torch.tensor([[1, 2]])

    model_inputs = model.prepare_inputs_for_generation(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        cache_position=torch.arange(input_ids.shape[1]),
    )

    assert model_inputs["generate_mode"] is True
    assert not {
        "latent_hidden_states",
        "latent_start_id",
        "latent_end_id",
        "latent_pad_id",
    }.intersection(model_inputs)
