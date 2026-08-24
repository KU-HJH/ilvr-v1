from types import SimpleNamespace

import torch
from torch import nn

from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration


class ToyCausalLanguageModel(nn.Module):
    """Deterministic causal model fixture for continuous latent rollouts."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(128, hidden_size)
        self.calls = []

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        position_ids,
        use_cache,
        return_dict,
        generate_mode=False,
    ):
        assert use_cache is False
        self.calls.append((inputs_embeds, attention_mask.clone(), position_ids.clone()))
        position_term = position_ids[0].unsqueeze(-1).to(inputs_embeds.dtype)
        hidden = torch.cumsum(inputs_embeds, dim=1) * self.scale + position_term
        return SimpleNamespace(last_hidden_state=hidden)


def tiny_stage2_qwen(*, attn_implementation="eager"):
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
        attn_implementation=attn_implementation,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 0]},
    )
    return Qwen2_5_VLForConditionalGeneration(config)
