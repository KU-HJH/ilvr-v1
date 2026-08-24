import pytest
import torch

from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from .helpers import ToyCausalLanguageModel


def tiny_stage2_qwen():
    config = Qwen2_5_VLConfig(
        vocab_size=128,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 2,
            "out_hidden_size": 8,
            "fullatt_block_indexes": [0],
        },
        image_token_id=80,
        vision_start_token_id=81,
        video_token_id=82,
        stage="stage2",
        latent_size=2,
        latent_pad_token_id=90,
        latent_start_token_id=91,
        latent_end_token_id=92,
        attn_implementation="eager",
    )
    return Qwen2_5_VLForConditionalGeneration(config)


def stage2_scaffold():
    input_ids = torch.tensor([[1, 81, 80, 80, 80, 80, 2, 91, 90, 90, 92, 3]])
    attention_mask = torch.ones_like(input_ids)
    image_grid_thw = torch.tensor([[1, 2, 2]])
    return input_ids, attention_mask, image_grid_thw


def test_stage2_forward_slices_one_full_mrope_tensor(monkeypatch):
    model = tiny_stage2_qwen()
    recorder = ToyCausalLanguageModel(8)
    model.model = recorder
    input_ids, attention_mask, image_grid_thw = stage2_scaffold()
    expected_positions, _ = model.get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    rope_calls = 0
    original_get_rope_index = model.get_rope_index

    def record_get_rope_index(*args, **kwargs):
        nonlocal rope_calls
        rope_calls += 1
        return original_get_rope_index(*args, **kwargs)

    monkeypatch.setattr(model, "get_rope_index", record_get_rope_index)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        labels=input_ids.masked_fill(input_ids == 90, -100),
        return_stage2_latent_states=True,
    )

    assert len(outputs.stage2_latent_states) == 2
    assert rope_calls == 1
    assert len(recorder.calls) == 3
    assert torch.equal(recorder.calls[0][2], expected_positions[..., :8])
    assert torch.equal(recorder.calls[1][2], expected_positions[..., :9])
    assert torch.equal(recorder.calls[2][2], expected_positions)
    assert outputs.logits.shape == (1, input_ids.shape[1], model.config.vocab_size)
    assert outputs.loss.ndim == 0


def test_stage2_outputs_do_not_retain_latents_or_diagnostic_graphs_by_default():
    model = tiny_stage2_qwen()
    model.model = ToyCausalLanguageModel(8)
    input_ids, attention_mask, image_grid_thw = stage2_scaffold()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        collect_stage2_diagnostics=True,
    )

    assert outputs.stage2_latent_states is None
    assert torch.equal(outputs.stage2_diagnostics["block_count"], torch.tensor(1))
    assert outputs.stage2_diagnostics["latent_l2_norms"].requires_grad is False
    assert outputs.stage2_diagnostics["adjacent_cosine_similarities"].requires_grad is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("pixel_values_latent", torch.zeros(1, 1)),
        ("image_grid_thw_latent", torch.tensor([[1, 1, 1]])),
        ("image_out_mask", torch.ones(1, 12, dtype=torch.bool)),
        ("latent_hidden_states", torch.zeros(2, 8)),
    ],
)
def test_stage2_forward_rejects_legacy_helper_inputs(name, value):
    model = tiny_stage2_qwen()
    input_ids, attention_mask, image_grid_thw = stage2_scaffold()

    with pytest.raises(ValueError, match="helper inputs"):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            **{name: value},
        )
