"""Real precision-state contracts; no CUDA execution is required."""
import importlib

import pytest
import torch


def numerics():
    try:
        return importlib.import_module('evaluation.numerics')
    except ModuleNotFoundError:
        pytest.fail('shared numerical policy is not implemented')


def state():
    return (torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            tuple(getattr(torch.backends.cuda, name+'_sdp_enabled')() for name in
                  ('math', 'flash', 'mem_efficient', 'cudnn')))


@pytest.fixture(autouse=True)
def restore_precision():
    incoming = state()
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.set_float32_matmul_precision(incoming[0])
        torch.backends.cuda.matmul.allow_tf32 = incoming[1]
        # allow_tf32 and precision share state; restore precision last (medium matters).
        torch.set_float32_matmul_precision(incoming[0])
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = incoming[2]
        for name, enabled in zip(('math', 'flash', 'mem_efficient', 'cudnn'), incoming[3]):
            getattr(torch.backends.cuda, 'enable_'+name+'_sdp')(enabled)


def test_historical_policy_leaves_precision_unchanged(monkeypatch):
    n = numerics()
    torch.set_float32_matmul_precision('medium')
    before = state()
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE', '1')
    with n.numerical_policy_scope('historical'):
        assert state() == before
    assert state() == before


@pytest.mark.parametrize('policy', ['tf32_off', 'full_accumulation', 'math_sdpa'])
def test_policy_applies_expected_controls_and_restores_medium(policy):
    n = numerics()
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    before = state()
    with n.numerical_policy_scope(policy):
        observed = n.observe_numerical_policy(policy)
        assert observed['policy'] == policy
        assert observed['float32_matmul_precision'] == 'highest'
        assert observed['matmul_allow_tf32'] is False
        assert observed['bf16_reduced_precision_reduction'] is (policy == 'tf32_off')
        assert observed['parameter_dtypes'] == {}
        if policy == 'math_sdpa':
            assert observed['sdpa_enabled'] == dict(math=True, flash=False, efficient=False, cudnn=False)
    assert state() == before


@pytest.mark.parametrize('policy', ['tf32_off', 'full_accumulation', 'math_sdpa'])
def test_policy_restores_after_exception(policy):
    n = numerics()
    torch.set_float32_matmul_precision('medium')
    before = state()
    with pytest.raises(RuntimeError, match='body'):
        with n.numerical_policy_scope(policy):
            assert torch.get_float32_matmul_precision() == 'highest'
            if policy != 'tf32_off':
                assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is False
            if policy == 'math_sdpa':
                assert n.observe_numerical_policy(policy)['sdpa_enabled'] == dict(math=True, flash=False, efficient=False, cudnn=False)
            raise RuntimeError('body')
    assert state() == before


def test_policy_rejects_unsupported_execution_and_conflicting_override(monkeypatch):
    n = numerics()
    for protocol, stage, disable in [('original', 'stage1', True),
                                      ('continuous', 'stage2', True),
                                      ('continuous', 'stage1', False),
                                      ('auto', 'stage1', True)]:
        with pytest.raises(ValueError):
            n.validate_numerical_policy('tf32_off', protocol=protocol, stage=stage,
                                        disable_latent_generation=disable)
    for invalid in [True, None, 1, 'unknown']:
        with pytest.raises((TypeError, ValueError)):
            n.validate_numerical_policy(invalid, protocol='continuous', stage='stage1',
                                        disable_latent_generation=True)
        with pytest.raises((TypeError, ValueError)):
            with n.numerical_policy_scope(invalid):
                pytest.fail('invalid policy entered')
    assert n.validate_numerical_policy('historical', protocol='original', stage='stage1',
                                       disable_latent_generation=False) == 'historical'
    before = state()
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE', '1')
    with pytest.raises(ValueError, match='NVIDIA_TF32_OVERRIDE'):
        with n.numerical_policy_scope('tf32_off'):
            pytest.fail('forced override entered')
    assert state() == before


def test_parameter_dtype_evidence_uses_actual_base_and_active_adapter_tensors():
    # Saved configs cannot substitute for the actual mixed-dtype Q weights.
    n = numerics()
    from types import SimpleNamespace
    q = torch.nn.Module()
    q.base_layer = torch.nn.Linear(2, 2).to(torch.bfloat16)
    q.lora_A = torch.nn.ModuleDict({'active': torch.nn.Linear(2, 1), 'inactive': torch.nn.Linear(2, 1)})
    q.lora_B = torch.nn.ModuleDict({'active': torch.nn.Linear(1, 2)})
    q.active_adapters = ['active']
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(q_proj=q))]))
    observed = n.observe_numerical_policy('historical', model)
    assert observed['parameter_dtypes']['base_layer.weight'] == 'torch.bfloat16'
    assert observed['parameter_dtypes']['lora_A.active.weight'] == 'torch.float32'
    assert observed['parameter_dtypes']['lora_B.active.weight'] == 'torch.float32'
    assert not any('inactive' in key for key in observed['parameter_dtypes'])


@pytest.mark.parametrize('exception', [False, True])
def test_cudnn_tf32_remains_untouched_and_environment_controls_are_reported(exception, monkeypatch):
    n = numerics()
    torch.backends.cudnn.allow_tf32 = True
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE', '0')
    monkeypatch.delenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', raising=False)
    try:
        with n.numerical_policy_scope('tf32_off'):
            assert torch.backends.cudnn.allow_tf32 is True
            assert n.observe_numerical_policy('tf32_off')['environment_controls'] == {
                'NVIDIA_TF32_OVERRIDE': '0', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': None}
            if exception:
                raise RuntimeError('body')
    except RuntimeError:
        assert exception
    assert torch.backends.cudnn.allow_tf32 is True


def test_pytorch_cublas_external_tf32_override_is_rejected(monkeypatch):
    n = numerics()
    monkeypatch.setenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', '1')
    with pytest.raises(ValueError, match='TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'):
        with n.numerical_policy_scope('full_accumulation'):
            pass


def test_environment_observations_read_actual_process_controls(monkeypatch):
    monkeypatch.setenv('NVIDIA_TF32_OVERRIDE', '0')
    monkeypatch.delenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', raising=False)
    assert numerics().observe_numerical_policy('historical')['environment_controls'] == {
        'NVIDIA_TF32_OVERRIDE': '0', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': None}
