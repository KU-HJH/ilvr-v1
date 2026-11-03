"""Reuse detached DoRA norms across one Stage 2 forward/backward.

PEFT's identity-matrix path supports sharded parameters but is expensive for
the many prefix calls in latent rollout. Ordinary, unquantized Linear layers
with full parameters can form the same delta directly. Only detached norm
vectors survive between calls; adapter activations and gradients are unchanged.
"""

from contextlib import contextmanager
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import module as module_hooks
from peft.tuners.lora.dora import DoraLinearLayer
from peft.tuners.lora.layer import Linear


def _plain_linear(module):
    # Calling .weight directly would bypass custom forward/sharding hooks.
    return (
        type(module) is nn.Linear
        and "forward" not in module.__dict__
        and not module._forward_hooks
        and not module._forward_pre_hooks
        and not hasattr(module, "_hf_hook")
        and all(type(p) is nn.Parameter and not hasattr(p, "ds_id")
                and not getattr(p, "_is_flat_param", False)
                for p in module.parameters())
    )


def _scoped_forward(cache):
    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None):
        weight = base_layer.weight.to(x.dtype)
        parameters = (base_layer.weight, lora_A.weight, lora_B.weight)
        key = (id(self), x.dtype, x.device, scaling,
               torch.is_autocast_enabled(x.device.type),
               torch.get_autocast_dtype(x.device.type),
               tuple((id(p), p._version) for p in parameters))
        if key not in cache:
            # PEFT detaches this norm. Do not even construct the discarded
            # autograd graph, including during non-reentrant checkpoint replay.
            # Do not seed autocast's parameter cache with no-grad casts: the
            # subsequent trainable A/B projections need differentiable casts.
            with torch.no_grad(), torch.autocast(
                x.device.type, enabled=torch.is_autocast_enabled(x.device.type),
                dtype=torch.get_autocast_dtype(x.device.type), cache_enabled=False,
            ):
                delta = lora_B.weight @ lora_A.weight
                cache[key] = self.get_weight_norm(weight, delta, scaling).detach()
        mag_norm_scale = (self.weight / cache[key]).view(1, -1)
        lora_result = lora_B(lora_A(x))
        if base_result is None:
            # Training dropout means this is NOT the original base result:
            # PEFT passes the dropped input and requires another projection.
            base_result = F.linear(x, weight)
        elif base_layer.bias is not None:
            base_result = base_result - base_layer.bias
        return (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scaling

    return forward


@contextmanager
def dora_microbatch_scope(model, *, enabled=True):
    """Enclose forward AND backward; release all norms before the next batch.

    Stage2Trainer disables this scope for FSDP/ZeRO-3. Extra guards here leave
    sharded, quantized, customized, or nested scopes on their existing path.
    Methods are changed only on this model and restored even on exceptions;
    installed PEFT, state_dicts and inference behavior are never modified.
    """
    if (not enabled or module_hooks._global_forward_hooks
            or module_hooks._global_forward_pre_hooks):
        yield
        return
    modules = list(model.modules())
    # DataParallel shallow-copies bound instance methods to its replicas;
    # a patched forward would still refer to the original device's magnitude.
    if any(isinstance(m, nn.DataParallel)
           or getattr(m, "_is_fsdp_managed_module", False)
           or hasattr(m, "_fsdp_wrapped_module") for m in modules):
        yield
        return
    cache, patched = {}, []
    forward = _scoped_forward(cache)
    try:
        for layer in modules:
            if type(layer) is not Linear or layer.fan_in_fan_out:
                continue
            base = layer.get_base_layer()
            if not _plain_linear(base):
                continue
            for name, magnitude in layer.lora_magnitude_vector.items():
                if (type(magnitude) is not DoraLinearLayer
                        or "forward" in magnitude.__dict__):
                    continue
                a, b = layer.lora_A[name], layer.lora_B[name]
                if not (_plain_linear(a) and _plain_linear(b)) or a.bias is not None or b.bias is not None:
                    continue
                magnitude.forward = MethodType(forward, magnitude)
                patched.append(magnitude)
        yield
    finally:
        for magnitude in patched:
            del magnitude.forward
        cache.clear()
