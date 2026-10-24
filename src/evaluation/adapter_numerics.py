"""Temporary token-only FP64 arithmetic for the installed standard PEFT DoRA path.

Parameters and the two-dimensional identity reconstruction retain native arithmetic.
This observes operands at Python linear boundaries, not internal CUDA kernels.
"""
from contextlib import contextmanager, ExitStack
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

ADAPTER_TOKEN_FP64_POLICIES = ('adapter_token_fp64', 'adapter_token_fp64_math_sdpa')


class AdapterNumericsError(ValueError):
    """Unsupported adapter execution, call role, binding, or scope nesting."""


# Only live bindings in this process; no model state is changed or persisted.
_bindings = {}
_intervals = []


def _native_binding(module, function):
    forward = module.forward
    binding = _bindings.get(module)
    return (binding is not None and forward is binding['wrapper']) or (
        getattr(forward, '__func__', None) is function and
        getattr(forward, '__self__', None) is module)


def _owners(owner):
    # A resolved Qwen owns vision and decoder; only decoder adapters are in scope.
    decoder = getattr(getattr(owner, 'model', None), 'layers', None)
    roots = list(decoder) if decoder is not None else [owner]
    seen = set()
    for root in roots:
        for module in root.modules():
            if hasattr(module, 'lora_A') and module not in seen:
                seen.add(module)
                yield module


def _validated(owner, require_adapters):
    # Imports stay lazy so historical numerical policy imports do not load PEFT.
    from peft.tuners.lora.layer import Linear
    from peft.tuners.lora.dora import DoraLinearLayer
    from peft.tuners.lora.variants import DoraLinearVariant
    found = []
    for module in _owners(owner):
        if type(module) is not Linear:
            raise AdapterNumericsError('unsupported adapter owner type')
        active = module.active_adapters
        if (module.training or module.disable_adapters
                or module.merged or len(active) != 1 or module.fan_in_fan_out
                or not _native_binding(module, Linear.forward)):
            raise AdapterNumericsError('requires standard eval, enabled, unmerged single-active Linear DoRA')
        adapter = active[0]
        if (not module.use_dora.get(adapter) or type(module.get_base_layer()) is not nn.Linear
                or not _native_binding(module.get_base_layer(), nn.Linear.forward)
                or module.ephemeral_gpu_offload
                or type(module.lora_variant.get(adapter)) is not DoraLinearVariant
                or module.lora_variant[adapter].forward is not DoraLinearVariant.forward):
            raise AdapterNumericsError('unsupported DoRA owner or variant')
        a, b, dora = (getattr(module, attr)[adapter] for attr in
                      ('lora_A', 'lora_B', 'lora_magnitude_vector'))
        if (type(a) is not nn.Linear or type(b) is not nn.Linear
                or type(dora) is not DoraLinearLayer or dora.training
                or not _native_binding(dora, DoraLinearLayer.forward)
                or any(type(m) is not nn.Linear or m.training
                       or not _native_binding(m, nn.Linear.forward)
                       or any(p.dtype != torch.float32 for p in m.parameters()) for m in (a, b))):
            raise AdapterNumericsError('unsupported DoRA A/B/magnitude binding or FP32 parameters')
        found.append((module, a, b, dora))
    if require_adapters and not found:
        raise AdapterNumericsError('no supported decoder DoRA adapters')
    return found


def validate_adapter_token_fp64(owner: nn.Module, *, require_adapters: bool = True) -> int:
    return len(_validated(owner, require_adapters))


def temporary_fp64_token_linear(linear: nn.Linear, operand: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if (type(linear) is not nn.Linear or operand.ndim != 3 or operand.dtype != torch.float32
            or any(p.dtype != torch.float32 for p in linear.parameters())):
        raise AdapterNumericsError('token arithmetic requires 3D FP32 input and standard FP32 Linear')
    x, weight = operand.to(torch.float64), linear.weight.to(torch.float64)
    bias = None if linear.bias is None else linear.bias.to(torch.float64)
    output = F.linear(x, weight, bias)
    returned = output.to(torch.float32)
    return returned, dict(input=str(x.dtype), weight=str(weight.dtype),
        bias=None if bias is None else str(bias.dtype), linear_output=str(output.dtype),
        returned=str(returned.dtype))


class AdapterTokenEvidence:
    def __init__(self, owners):
        self._owners = frozenset(owners)
        self._token = self._normalization = 0
        self._actual = None

    def snapshot(self) -> dict:
        return dict(requested_dtype='torch.float64', dora_modules=len(self._owners),
            wrapped_linear_modules=2*len(self._owners), token_calls=self._token,
            normalization_calls=self._normalization,
            actual_operand_dtypes=None if self._actual is None else dict(self._actual))


def _record(owner, actual):
    for evidence in _intervals:
        if owner in evidence._owners:
            if actual is None:
                evidence._normalization += 1
            else:
                evidence._token += 1
                evidence._actual = actual


def _install(stack, module, forward):
    present, original = 'forward' in module.__dict__, module.__dict__.get('forward')
    wrapper = MethodType(forward, module)
    module.forward = wrapper
    _bindings[module] = dict(wrapper=wrapper)
    def restore():
        if present:
            module.forward = original
        else:
            del module.forward
        del _bindings[module]
    stack.callback(restore)


@contextmanager
def adapter_token_fp64_scope(owner: nn.Module, *, policy: str,
                             require_adapters: bool = True):
    if policy not in ADAPTER_TOKEN_FP64_POLICIES:
        raise AdapterNumericsError('unknown adapter numerical policy')
    targets = _validated(owner, require_adapters)
    owners = frozenset(row[0] for row in targets)
    if _intervals and (policy != _intervals[-1]._policy or
                      not owners.issubset(_intervals[-1]._owners)):
        raise AdapterNumericsError('incompatible adapter scope nesting')
    evidence = AdapterTokenEvidence(owners)
    evidence._policy = policy
    with ExitStack() as stack:
        if not _intervals:
            for projection, a, b, dora in targets:
                state = dict(call=None)
                original_dora, original_a, original_b = dora.forward, a.forward, b.forward
                original_projection = projection.forward
                def owner_forward(_self, x, *args, _original=original_projection, **kwargs):
                    _validated(_self, True)
                    if (torch.is_grad_enabled() or torch.is_autocast_enabled(x.device.type)
                            or args or kwargs):
                        raise AdapterNumericsError('unsupported owner execution or mixed-adapter call')
                    return _original(x)
                _install(stack, projection, owner_forward)
                def enter(_self, x, *args, _p=projection, _a=a, _b=b, _state=state,
                          _original=original_dora, **kwargs):
                    if (_state['call'] is not None or _p.training or _self.training
                            or torch.is_grad_enabled() or torch.is_autocast_enabled(x.device.type)
                            or x.ndim != 3 or x.dtype != torch.float32 or args
                            or kwargs.get('lora_A') is not _a or kwargs.get('lora_B') is not _b):
                        raise AdapterNumericsError('unsupported or reentrant DoRA token call')
                    _state['call'] = dict(index=0, token=x, previous=None)
                    try:
                        result = _original(x, **kwargs)
                        if _state['call']['index'] != 4:
                            raise AdapterNumericsError('incomplete DoRA A/B roles')
                        return result
                    finally:
                        _state['call'] = None
                _install(stack, dora, enter)
                for linear, label, original in ((a, 'A', original_a), (b, 'B', original_b)):
                    def run(_self, x, *args, _p=projection, _a=a, _state=state,
                            _label=label, _original=original, **kwargs):
                        call = _state['call']
                        if call is None or args or kwargs:
                            raise AdapterNumericsError('A/B called outside supported DoRA roles')
                        index = call['index']
                        expected = ('A', 'B', 'A', 'B')
                        if index >= 4 or _label != expected[index]:
                            raise AdapterNumericsError('unexpected DoRA A/B call order')
                        valid = (x.ndim == 2 and x.shape == (_a.in_features, _a.in_features)
                                 and x.dtype == torch.float32 and torch.equal(x, torch.eye(
                                     _a.in_features, device=x.device, dtype=x.dtype))) if index == 0 else (
                            x is call['token'] if index == 2 else x is call['previous'])
                        if not valid:
                            raise AdapterNumericsError('unexpected DoRA identity/token operand linkage')
                        if index < 2:
                            output = _original(x)
                            _record(_p, None)
                        else:
                            output, actual = temporary_fp64_token_linear(_self, x)
                            _record(_p, actual)
                        call['previous'] = output
                        call['index'] += 1
                        return output
                    _install(stack, linear, run)
        _intervals.append(evidence)
        try:
            yield evidence
        finally:
            _intervals.pop()
