from types import SimpleNamespace

import torch
from torch import nn


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
