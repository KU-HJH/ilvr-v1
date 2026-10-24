"""Explicit, scoped numerical controls for Stage 1 continuous latent-OFF evaluation.

These are observed PyTorch API settings, not evidence of a CUDA kernel identity.
The historical policy deliberately performs no state mutation.
"""
from contextlib import contextmanager
import os

import torch

NUMERICAL_POLICIES = ('historical', 'tf32_off', 'full_accumulation', 'math_sdpa')
_SDPA = dict(math='math', flash='flash', efficient='mem_efficient', cudnn='cudnn')


def _policy(policy):
    if type(policy) is not str or policy not in NUMERICAL_POLICIES:
        raise ValueError(f'unknown numerical policy: {policy!r}')
    return policy


def validate_numerical_policy(policy: str, *, protocol: str, stage: str,
                              disable_latent_generation: bool) -> str:
    policy = _policy(policy)
    if policy != 'historical' and not (
            protocol == 'continuous' and stage == 'stage1'
            and disable_latent_generation is True):
        raise ValueError('numerical policy requires explicit continuous Stage 1 with latent generation disabled')
    return policy


def observe_numerical_policy(policy: str, model=None) -> dict:
    """Read active controls and actual first-Q base/active-adapter parameters."""
    _policy(policy)
    parameters = {}
    if model is not None:
        q = model.model.layers[0].self_attn.q_proj
        active = getattr(q, 'active_adapters', ())
        if isinstance(active, str):
            active = (active,)
        for name, value in q.named_parameters():
            if not hasattr(q, 'base_layer') or name.startswith('base_layer.') or any(
                    name.startswith(f'{kind}.{adapter}.')
                    for adapter in active for kind in ('lora_A', 'lora_B', 'lora_magnitude_vector')):
                parameters[name] = str(value.dtype)
    return dict(policy=policy, float32_matmul_precision=torch.get_float32_matmul_precision(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                sdpa_enabled={key: getattr(torch.backends.cuda, name+'_sdp_enabled')()
                              for key, name in _SDPA.items()},
                parameter_dtypes=parameters,
                environment_controls={key: os.environ.get(key) for key in
                    ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')})


@contextmanager
def numerical_policy_scope(policy: str):
    policy = _policy(policy)
    if policy == 'historical':
        yield
        return
    for key in ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'):
        if os.environ.get(key) == '1':
            raise ValueError(f'{key}=1 conflicts with requested TF32-off numerical policy')
    previous = observe_numerical_policy(policy)
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        if policy in ('full_accumulation', 'math_sdpa'):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if policy == 'math_sdpa':
            for key, name in _SDPA.items():
                getattr(torch.backends.cuda, 'enable_'+name+'_sdp')(key == 'math')
        if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != 'highest':
            raise ValueError('requested TF32-off PyTorch controls did not take effect')
        yield
    finally:
        # allow_tf32 aliases float32 precision; restoring only that flag loses medium.
        torch.backends.cuda.matmul.allow_tf32 = previous['matmul_allow_tf32']
        torch.set_float32_matmul_precision(previous['float32_matmul_precision'])
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous['bf16_reduced_precision_reduction']
        for key, name in _SDPA.items():
            getattr(torch.backends.cuda, 'enable_'+name+'_sdp')(previous['sdpa_enabled'][key])
