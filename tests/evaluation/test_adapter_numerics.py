"""Real installed DoRA witnesses for scoped, role-specific adapter arithmetic."""
from types import MethodType
import importlib

import pytest
import torch
from torch import nn


def api():
    try:
        return importlib.import_module('evaluation.adapter_numerics')
    except ModuleNotFoundError:
        pytest.fail('token adapter arithmetic interface is missing')


def tiny_dora():
    from peft.tuners.lora.layer import Linear
    torch.manual_seed(13)
    layer = Linear(nn.Linear(4, 4), 'default', r=2, lora_alpha=4, use_dora=True).eval()
    with torch.no_grad():
        layer.lora_B['default'].weight.copy_(torch.arange(8).reshape(4, 2)/10)
    return layer


def bindings(module):
    return [(m, 'forward' in m.__dict__, m.__dict__.get('forward')) for m in module.modules()]


def assert_bindings(saved):
    for module, present, value in saved:
        assert ('forward' in module.__dict__) is present
        if present:
            assert module.__dict__['forward'] is value


def test_token_fp64_uses_real_operands_and_preserves_norm_path(monkeypatch):
    # Catches casting Parameters, protecting identity reconstruction, or fabricated dtype evidence.
    a = api()
    linear = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.5, -2.], [.5, 1.]]))
        result, actual = a.temporary_fp64_token_linear(linear, torch.tensor([[[2., 3.]]]))
    assert torch.equal(result, torch.tensor([[[-3., 4.]]]))
    assert result.dtype == torch.float32
    assert actual == dict(input='torch.float64', weight='torch.float64', bias=None,
                          linear_output='torch.float64', returned='torch.float32')
    projection = tiny_dora()
    norm_expected = projection.lora_A['default'](torch.eye(4)).detach()
    original = {name: (p.clone(), p.data_ptr(), p.dtype) for name, p in projection.named_parameters()}
    delegated, identity = [], []
    native = torch.nn.functional.linear
    def observe(x, w, bias=None):
        out = native(x, w, bias)
        delegated.append((x.ndim, x.dtype, w.dtype, out.dtype))
        return out
    monkeypatch.setattr(torch.nn.functional, 'linear', observe)
    handles = [projection.lora_A['default'].register_forward_hook(
        lambda _m, args, out: identity.append(out.clone()) if args[0].ndim == 2 else None)]
    try:
        with torch.inference_mode(), a.adapter_token_fp64_scope(projection, policy='adapter_token_fp64') as evidence:
            projection(torch.ones(1, 3, 4))
            observed = evidence.snapshot()
        assert observed['token_calls'] == 2
        assert observed['normalization_calls'] == 2
        assert observed['actual_operand_dtypes']['input'] == 'torch.float64'
        assert len(identity) == 1 and identity[0].dtype == torch.float32
        assert torch.equal(identity[0], norm_expected)
        assert delegated == [(3, torch.float32, torch.float32, torch.float32),
            (2, torch.float32, torch.float32, torch.float32), (2, torch.float32, torch.float32, torch.float32),
            (3, torch.float64, torch.float64, torch.float64), (3, torch.float64, torch.float64, torch.float64)]
        for name, p in projection.named_parameters():
            value, pointer, dtype = original[name]
            assert torch.equal(p, value) and p.data_ptr() == pointer and p.dtype == dtype
    finally:
        for h in handles:
            h.remove()


def test_scope_restores_exact_forward_presence_on_success_and_exception():
    a, projection = api(), tiny_dora()
    projection.lora_A['default'].forward = projection.lora_A['default'].forward
    saved = bindings(projection)
    for fails in (False, True):
        try:
            with torch.inference_mode(), a.adapter_token_fp64_scope(projection, policy='adapter_token_fp64'):
                projection(torch.ones(1, 1, 4))
                if fails:
                    raise RuntimeError('body')
        except RuntimeError as error:
            assert fails and str(error) == 'body'
        assert_bindings(saved)
    # A real invalid token operand must be fatal and still restore wrappers.
    with pytest.raises(a.AdapterNumericsError):
        with torch.inference_mode(), a.adapter_token_fp64_scope(projection, policy='adapter_token_fp64'):
            projection(torch.ones(2, 4))
    assert_bindings(saved)


def test_scope_nested_subset_reuses_bindings_and_interval_counts():
    a = api()
    first, second = tiny_dora(), tiny_dora()
    root = nn.ModuleList([first, second])
    saved = bindings(root)
    with torch.inference_mode(), a.adapter_token_fp64_scope(root, policy='adapter_token_fp64') as outer:
        first(torch.ones(1, 1, 4))
        wrapper = first.lora_A['default'].forward
        with a.adapter_token_fp64_scope(first, policy='adapter_token_fp64') as inner:
            assert first.lora_A['default'].forward is wrapper
            first(torch.ones(1, 1, 4))
        closed = inner.snapshot()
        first(torch.ones(1, 1, 4))
        assert inner.snapshot() == closed
        assert closed['token_calls'] == 2
        assert outer.snapshot()['token_calls'] == 6
        with pytest.raises(a.AdapterNumericsError):
            with a.adapter_token_fp64_scope(first, policy='adapter_token_fp64_math_sdpa'):
                pass
    assert_bindings(saved)


def test_scope_rejects_unknown_roles_bindings_and_unsupported_models():
    a = api()
    with pytest.raises(a.AdapterNumericsError):
        a.validate_adapter_token_fp64(nn.Linear(4, 4))
    for change in ('training', 'custom', 'disabled', 'half', 'plain_lora'):
        projection = tiny_dora()
        if change == 'training':
            projection.train()
        elif change == 'custom':
            projection.lora_A['default'].forward = MethodType(lambda self, x: x, projection.lora_A['default'])
        elif change == 'disabled':
            projection.enable_adapters(False)
        elif change == 'half':
            projection.lora_A['default'].half()
        else:
            projection.use_dora['default'] = False
        saved = bindings(projection)
        with pytest.raises(a.AdapterNumericsError):
            with a.adapter_token_fp64_scope(projection, policy='adapter_token_fp64'):
                pass
        assert_bindings(saved)
    projection = tiny_dora()
    with a.adapter_token_fp64_scope(projection, policy='adapter_token_fp64'):
        with pytest.raises(a.AdapterNumericsError):
            projection.lora_A['default'](torch.ones(1, 1, 4))
        with pytest.raises(a.AdapterNumericsError):
            projection(torch.ones(1, 1, 4))  # Grad enabled is unsupported.


@pytest.mark.parametrize('change', ['disabled', 'training', 'merged'])
def test_scope_revalidates_actual_owner_before_bypassed_dora(change):
    # A state change after entry must fail before the native owner can bypass DoRA.
    a, projection = api(), tiny_dora()
    saved = bindings(projection)
    with torch.inference_mode(), a.adapter_token_fp64_scope(projection, policy='adapter_token_fp64'):
        if change == 'disabled':
            projection.enable_adapters(False)
        elif change == 'training':
            projection.train()
        else:
            projection.merge()
        with pytest.raises(a.AdapterNumericsError):
            projection(torch.ones(1, 1, 4))
    assert_bindings(saved)


def test_scope_rejects_identity_value_or_a_output_linkage_changes():
    a, projection = api(), tiny_dora()
    saved = bindings(projection)
    for position in ('identity', 'A_output'):
        linear = projection.lora_A['default']
        if position == 'identity':
            handle = linear.register_forward_pre_hook(lambda _m, args:
                (torch.zeros_like(args[0]),) if args[0].ndim == 2 else None)
        else:
            handle = linear.register_forward_hook(lambda _m, _args, output: output.clone())
        try:
            with pytest.raises(a.AdapterNumericsError):
                with torch.inference_mode(), a.adapter_token_fp64_scope(projection, policy='adapter_token_fp64'):
                    projection(torch.ones(1, 1, 4))
        finally:
            handle.remove()
        assert_bindings(saved)


def test_unknown_adapter_owner_fails_explicitly_before_binding_changes():
    a = api()
    unknown = nn.Module()
    unknown.lora_A = nn.ModuleDict({'default': nn.Linear(4, 2)})
    saved = bindings(unknown)
    with pytest.raises(a.AdapterNumericsError):
        with a.adapter_token_fp64_scope(unknown, policy='adapter_token_fp64'):
            pass
    assert_bindings(saved)


@pytest.mark.parametrize('target', ['owner', 'base', 'A', 'B', 'dora', 'variant'])
def test_scope_rejects_custom_forward_at_every_supported_binding(target):
    a, projection = api(), tiny_dora()
    objects = dict(owner=projection, base=projection.base_layer,
        A=projection.lora_A['default'], B=projection.lora_B['default'],
        dora=projection.lora_magnitude_vector['default'], variant=projection.lora_variant['default'])
    obj = objects[target]
    obj.forward = lambda *args, **kwargs: args[0]
    saved = bindings(projection)
    with pytest.raises(a.AdapterNumericsError):
        with a.adapter_token_fp64_scope(projection, policy='adapter_token_fp64'):
            pass
    assert_bindings(saved)


def test_primitive_bias_and_generic_partial_replay_preserve_parameters():
    a = api()
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.5, -2.], [.5, 1.]]))
        linear.bias.copy_(torch.tensor([1., -1.]))
        before = [(p.clone(), p.data_ptr(), p.dtype) for p in linear.parameters()]
        out, actual = a.temporary_fp64_token_linear(linear, torch.tensor([[[2., 3.]]]))
        assert torch.equal(out, torch.tensor([[[-2., 3.]]]))
        assert actual['bias'] == 'torch.float64'
        with a.adapter_token_fp64_scope(linear, policy='adapter_token_fp64', require_adapters=False) as evidence:
            assert evidence.snapshot()['token_calls'] == evidence.snapshot()['dora_modules'] == 0
        assert all(torch.equal(p, old) and p.data_ptr() == pointer and p.dtype == dtype
            for p, (old, pointer, dtype) in zip(linear.parameters(), before))


def test_resolved_decoder_discovery_excludes_vision_and_rejects_autocast():
    from tests.evaluation.test_off_kv_cache import adapter_model
    a, model = api(), adapter_model()
    unknown = nn.Module()
    unknown.lora_A = nn.ModuleDict({'default': nn.Linear(4, 2)})
    model.visual.unknown_adapter = unknown
    assert a.validate_adapter_token_fp64(model) == 1
    projection = model.model.layers[0].self_attn.q_proj
    with torch.inference_mode(), a.adapter_token_fp64_scope(model, policy='adapter_token_fp64') as evidence:
        projection(torch.ones(1, 1, 8))
        assert evidence.snapshot()['token_calls'] == 2
        with torch.autocast('cpu', dtype=torch.bfloat16), pytest.raises(a.AdapterNumericsError):
            projection(torch.ones(1, 1, 8))
