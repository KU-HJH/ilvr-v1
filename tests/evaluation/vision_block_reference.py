"""Independent edge oracle; shares decoder weights/math, not replay planning."""
import torch

from evaluation.vision_block_metrics import DEFAULT_PROFILE, IMAGE_PROFILE


@torch.inference_mode()
def reference_logits(qwen, state, site, condition, *, analysis_profile=DEFAULT_PROFILE):
    columns = list(range(site.end, site.target_positions[-1]))
    hidden = state.embeds[:, columns].clone()
    positions = state.positions[:, :, columns]
    if condition in ("boundary_mask", "full_mask"):
        hidden[:, 0] = 0
    root_length = state.embeds.shape[1]
    mask = torch.full((1, 1, len(columns), root_length + len(columns)),
                      float("-inf"), dtype=hidden.dtype, device=hidden.device)
    for query in range(len(columns)):
        for key in range(root_length + len(columns)):
            if key < root_length:
                allowed = key < site.end
                if condition in ("body_mask", "full_mask") and site.body_start <= key < site.body_end:
                    allowed = False
                if condition in ("boundary_mask", "full_mask") and key == site.start:
                    allowed = False
                if analysis_profile == IMAGE_PROFILE:
                    if condition in ("image_mask", "image_body_mask") and key in state.image_positions:
                        allowed = False
                    if condition == "image_body_mask" and site.body_start <= key < site.body_end:
                        allowed = False
            else:
                allowed = key - root_length <= query
            if allowed:
                mask[0, 0, query, key] = 0
    rotary = qwen.model.rotary_emb(hidden, positions)
    for index, layer in enumerate(qwen.model.layers):
        hidden = layer(hidden, attention_mask=mask, position_ids=positions,
                       position_embeddings=rotary, past_key_value=None,
                       use_cache=False, output_attentions=False, generate_mode=True,
                       stage2_prefix_key_value=state.prefix_kv[index])[0]
    return qwen.lm_head(qwen.model.norm(hidden)).float()
