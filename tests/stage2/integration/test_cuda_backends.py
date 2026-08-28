import importlib.util

import pytest
import torch

from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration


def _tiny_stage2_model(attn_implementation: str):
    config = Qwen2_5_VLConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "window_size": 2,
            "out_hidden_size": 32,
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
        use_cache=False,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
    )
    return Qwen2_5_VLForConditionalGeneration(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_stage2_eager_and_flash_attention_have_matching_latent_semantics():
    """Catches a backend switch that changes Stage 2 latent count or values."""
    if importlib.util.find_spec("flash_attn") is None:
        pytest.skip("flash_attn is unavailable")

    torch.manual_seed(17)
    eager = _tiny_stage2_model("eager")
    flash = _tiny_stage2_model("flash_attention_2")
    flash.load_state_dict(eager.state_dict())
    eager = eager.to(device="cuda", dtype=torch.bfloat16).eval()
    flash = flash.to(device="cuda", dtype=torch.bfloat16).eval()
    input_ids = torch.tensor([[1, 91, 90, 90, 92, 2, 3]], device="cuda")
    labels = input_ids.masked_fill(input_ids == 90, -100)

    with torch.no_grad():
        eager_outputs = eager(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
            return_stage2_latent_states=True,
        )
        flash_outputs = flash(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels,
            return_stage2_latent_states=True,
        )

    assert torch.isfinite(eager_outputs.loss)
    assert torch.isfinite(flash_outputs.loss)
    assert len(eager_outputs.stage2_latent_states) == len(
        flash_outputs.stage2_latent_states
    ) == 2
    for eager_state, flash_state in zip(
        eager_outputs.stage2_latent_states,
        flash_outputs.stage2_latent_states,
        strict=True,
    ):
        assert torch.allclose(eager_state, flash_state, atol=2e-2, rtol=2e-2)
    assert torch.allclose(eager_outputs.logits, flash_outputs.logits, atol=2e-2, rtol=2e-2)
    assert eager_outputs.past_key_values is None
    assert flash_outputs.past_key_values is None
