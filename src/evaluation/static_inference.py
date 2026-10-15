"""Temporary, per-generation reuse of frozen PEFT inference constants."""
from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F


@contextmanager
def static_inference_cache(model):
    """Reuse supported PEFT constants without changing weights or global classes.

    Only standard linear DoRA and non-renormalizing trainable embeddings are
    optimized. Other PEFT layers retain their installed implementation.
    Parameters/adapters must remain fixed for the duration of this eval call.
    """
    from peft.tuners.lora.dora import DoraLinearLayer
    from peft.tuners.lora.layer import Linear as LoraLinear
    from peft.tuners.trainable_tokens.layer import TrainableTokensLayer
    from peft.utils.integrations import dequantize_module_weight
    from peft.utils.other import transpose

    if torch.is_grad_enabled() or any(module.training for module in model.modules()):
        raise ValueError("static inference cache requires eval mode and no grad")
    replacements = []
    caches = []

    def replace(module, name, function):
        replacements.append((module, name, name in module.__dict__, module.__dict__.get(name)))
        setattr(module, name, MethodType(function, module))

    def dora_forward():
        cache = {}
        caches.append(cache)

        def forward(module, x, *, lora_A, lora_B, scaling, base_layer, base_result=None):
            # Dtype is part of the constant: autocast or wrappers can feed a
            # different dtype on the first decode than on prompt prefill.
            if x.dtype not in cache:
                x_eye = torch.eye(lora_A.weight.shape[1], device=lora_A.weight.device, dtype=x.dtype)
                lora_weight = lora_B(lora_A(x_eye)).T
                weight = dequantize_module_weight(base_layer).to(x.dtype)
                weight_norm = module.get_weight_norm(weight, lora_weight.detach(), scaling).detach()
                cache[x.dtype] = ((module.weight / weight_norm).view(1, -1), weight)
            mag_norm_scale, weight = cache[x.dtype]
            lora_result = lora_B(lora_A(x))
            if base_result is not None:
                bias = base_layer.bias
                if bias is not None:
                    base_result = base_result - bias
            else:
                base_result = F.linear(x, transpose(weight, module.fan_in_fan_out))
            return (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scaling
        return forward

    def merged_weights(original):
        cache = {}
        caches.append(cache)

        def merged(module, active_adapters):
            key = tuple(active_adapters)
            if key not in cache:
                cache[key] = original(active_adapters)
            return cache[key]
        return merged

    try:
        for module in model.modules():
            if isinstance(module, LoraLinear) and type(module.get_base_layer()) is torch.nn.Linear:
                for adapter, dora in module.lora_magnitude_vector.items():
                    if (type(dora) is DoraLinearLayer
                            and getattr(dora.forward, "__func__", None) is DoraLinearLayer.forward
                            and type(module.lora_A[adapter]) is torch.nn.Linear
                            and type(module.lora_B[adapter]) is torch.nn.Linear):
                        replace(dora, 'forward', dora_forward())
            elif (isinstance(module, TrainableTokensLayer)
                  and isinstance(module.base_layer, torch.nn.Embedding)
                  and module.base_layer.max_norm is None):
                replace(module, 'get_merged_weights', merged_weights(module.get_merged_weights))
        yield
    finally:
        for module, name, had_override, original in reversed(replacements):
            if had_override:
                setattr(module, name, original)
            else:
                delattr(module, name)
        for cache in caches:
            cache.clear()


class IncrementalMRoPE:
    """Keep original prompt coordinates, then append ordinary text coordinates."""
    def __init__(self, model, input_ids, attention_mask, image_grid_thw):
        self.model = model
        self.image_grid_thw = image_grid_thw
        self.prompt_length = input_ids.shape[1]
        self.positions, _ = model.get_rope_index(
            input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        if image_grid_thw is None:
            # The original text-only routine's delta includes masked slots at
            # position 1; that delta is wrong for the one-valid-token case.
            self.next_position = attention_mask.long().sum(-1).view(1, -1, 1)
        else:
            valid = self.positions[:, 0, attention_mask[0] == 1]
            self.next_position = (valid.max() + 1).view(1, 1, 1)
        self.structural_ids = {
            getattr(model.config, name, None)
            for name in ('image_token_id', 'video_token_id', 'vision_start_token_id')
        }
        self.fallback = False

    def appended(self, token_id):
        # A generated vision marker can make Qwen reinterpret the whole prefix
        # (or reject a malformed grid). Retain that routine for this sample.
        if token_id in self.structural_ids:
            self.fallback = True

    def get(self, sequences, mask, cached_length):
        if self.fallback:
            positions, _ = self.model.get_rope_index(
                sequences, image_grid_thw=self.image_grid_thw, attention_mask=mask)
            return positions[..., cached_length:]
        if cached_length == 0:
            return self.positions
        offset = sequences.shape[1] - self.prompt_length - 1
        return (self.next_position + offset).expand(3, 1, 1)
