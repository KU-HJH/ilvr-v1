"""One clean EMA decoder call with bounded vocabulary-row consumption."""
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from transformers.models.qwen2_5_vl.stage1_prefix_conditioning import Stage1ReferenceRequest

from .prefix_mask_config import PrefixConditioningConfig
from .prefix_mask_metadata import HelperSnapshot
from .prefix_mask_plan import reverse_kl
from .vision_gain_execution import capture_gain_rng, _restore_rng


def score_clean_reference_once(teacher: nn.Module, model_inputs: Mapping[str, object],
                               snapshot: HelperSnapshot, student_hidden: Tensor,
                               student_lm_head: nn.Module, *,
                               config: PrefixConditioningConfig,
                               consume_reference=None) -> Tensor:
    """Consume fixed GT queries in order; retain only their detached scalar KL."""
    keys = snapshot.gt_queries
    if student_hidden.ndim != 3:
        raise ValueError('student hidden must have shape [B,T,H]')
    request = Stage1ReferenceRequest(keys, config.logit_chunk_rows, lambda *_: None)
    request.validate_positions(student_hidden.shape[:2])
    if not keys:
        return student_hidden.new_empty((0,), dtype=torch.float32)
    ids, positions = model_inputs.get('input_ids'), model_inputs.get('position_ids')
    if not isinstance(ids, Tensor) or tuple(ids.shape) != tuple(student_hidden.shape[:2]):
        raise ValueError('original input_ids must match student hidden positions')
    if not isinstance(positions, Tensor) or tuple(positions.shape) != (3, *ids.shape):
        raise ValueError('explicit original position_ids must have shape [3,B,T]')
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError('position_ids must be integer coordinates')
    if snapshot.latent_values.requires_grad or snapshot.latent_mask.requires_grad:
        raise ValueError('snapshot latent targets must be detached')
    result = student_hidden.new_empty((len(keys),), dtype=torch.float32)
    consumed = 0

    def consume(chunk_keys, reference_logits):
        nonlocal consumed
        count = len(chunk_keys)
        if not 0 < count <= config.logit_chunk_rows or chunk_keys != keys[consumed:consumed + count]:
            raise ValueError('reference query chunks must match snapshot order exactly')
        rows = torch.stack([student_hidden[b, p].detach() for b, p in chunk_keys])
        student_logits = student_lm_head(rows)
        values = reverse_kl(student_logits, reference_logits, temperature=config.temperature)
        if values.shape != (count,):
            raise ValueError('reference logit rows must match requested queries')
        result[consumed:consumed + count] = values.to(result.device)
        if consume_reference is not None:
            consume_reference(chunk_keys, reference_logits)
        consumed += count

    inputs = dict(model_inputs)
    for name in ('labels', 'inputs_embeds', 'past_key_values', 'cache_position',
                 'stage1_vision_gain_request', 'stage1_vision_gain_packed_request',
                 'pixel_values_latent', 'image_grid_thw_latent'):
        inputs.pop(name, None)
    inputs.update(latent_hidden_states=snapshot.latent_values, image_out_mask=snapshot.latent_mask,
                  use_cache=False, generate_mode=False, return_dict=True,
                  output_attentions=False, output_hidden_states=False,
                  stage1_reference_request=Stage1ReferenceRequest(keys, config.logit_chunk_rows, consume))
    modules = tuple(teacher.modules())
    modes = tuple((module, module.training) for module in modules)
    ropes = tuple((module, module.rope_deltas,
                   module.rope_deltas.detach().clone() if isinstance(module.rope_deltas, Tensor) else None)
                  for module in modules if 'rope_deltas' in vars(module))
    rng = capture_gain_rng(teacher)
    try:
        for module, _ in modes:
            module.training = False
        with torch.no_grad():
            teacher(**inputs)
        if consumed != len(keys):
            raise ValueError('reference did not provide all requested queries')
    finally:
        for module, mode in modes:
            module.training = mode
        with torch.no_grad():
            for module, original, saved in ropes:
                if saved is not None:
                    original.copy_(saved)
                module.rope_deltas = original
        _restore_rng(rng)
    return result.detach()
