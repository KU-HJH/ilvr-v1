"""Literal original utils_deepseed.py template; see original-provenance.json."""
import torch
from transformers import LogitsProcessor

class LatentTemplateLogitsProcessor(LogitsProcessor):
    def __init__(self, latent_start_id: int, latent_end_id: int, latent_pad_id: int, K: int):
        self.latent_start_id = int(latent_start_id)
        self.latent_end_id   = int(latent_end_id)
        self.latent_pad_id   = int(latent_pad_id)
        self.K               = int(K)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        B = input_ids.size(0)
        for b in range(B):
            seq = input_ids[b]
            start_positions = (seq == self.latent_start_id).nonzero().squeeze(-1)
            end_positions   = (seq == self.latent_end_id).nonzero().squeeze(-1)

            def _last_pos(t):
                return int(t[-1].item()) if t.numel() > 0 else -1

            last_s = _last_pos(start_positions)
            last_e = _last_pos(end_positions)
            inside = (last_s != -1) and (last_e < last_s)

            if not inside:
                scores[b, self.latent_pad_id] = -float("inf")
                scores[b, self.latent_end_id] = -float("inf")
                continue

            i = last_s + 1
            n_pad = 0
            L = seq.size(0)
            while i < L and int(seq[i].item()) == self.latent_pad_id:
                n_pad += 1
                i += 1

            if n_pad < self.K:
                scores[b, :] = -float("inf")
                scores[b, self.latent_pad_id] = 0.0
            else:
                scores[b, :] = -float("inf")
                scores[b, self.latent_end_id] = 0.0
        return scores
