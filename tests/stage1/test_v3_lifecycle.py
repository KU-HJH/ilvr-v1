"""Real checkpoint/vmap traversals exercise the finite stochastic lifecycle."""
import importlib
from contextlib import nullcontext

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from stage1.equivalence import (
    SampleStableStage1Dropout, Stage1StochasticContext, activate_stage1_stochastic_context,
)

SITE = 'decoder.layers.0.lora_dropout'
UNIT = ('decoder', 0)
GRAPHS = ('F', 'gain_clean', 'packed_replay')


def api():
    return importlib.import_module('stage1.v3_lifecycle')


def context(*, auxiliary=True, rows=((17, 4),)):
    spec = Stage1StochasticContext(42, 3, rows, (SITE,))
    inventory = {(graph, UNIT): (SITE,) for graph in (GRAPHS if auxiliary else ('F',))}
    return api().V3Lifecycle(spec, inventory, has_auxiliary=auxiliary)


def originals(life, *, omit=None):
    for graph in GRAPHS if life.has_auxiliary else ('F',):
        with life.graph(graph):
            with life.checkpoint_contexts(graph, UNIT)[0]:
                if graph != omit:
                    life.record(SITE, origin='packed' if graph == 'packed_replay' else 'native')


def finish_roles(life):
    for role in ('joint_r_kl', 'rv', 'ordinary') if life.has_auxiliary else ('ordinary',):
        with life.backward(role):
            pass


def cycle(dtype, p, checkpoint_mode, *, fail_rv=False, omit_packed=False, zero_kl=False):
    life = context()
    observations, vjp_arities = [], []

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(9, dtype=dtype).reshape(3, 3) / 13)
            self.register_buffer('counter', torch.zeros((), dtype=dtype))
            self.dropout = SampleStableStage1Dropout(SITE, p)

        def forward(self, x):
            graph, unit, phase, role = life.frame
            observations.append((graph, phase, role, self.weight))
            self.counter.add_(1)
            if fail_rv and phase == 'recompute' and role == 'rv':
                raise RuntimeError('injected RV recompute failure')
            y = x @ self.weight
            if graph == 'packed_replay':
                if omit_packed:
                    return y
                raw = api().packed_dropout_mask(SITE, p,
                    sample_indices=torch.zeros(4, dtype=torch.long),
                    token_positions=torch.arange(4), hidden_size=3, device=y.device)
                return y * (raw.to(y.dtype) * (1 / (1 - p))).unsqueeze(0)
            return self.dropout(y)

    layer = Layer()
    live = layer.weight
    alias = (live.detach().clone() + 2).requires_grad_()
    live_before = live.detach().clone()
    x = torch.linspace(.1, .9, 12, dtype=dtype).reshape(1, 4, 3)

    def forward(graph, value, parameter):
        managers = life.checkpoint_contexts(graph, UNIT)
        # Capture both alias and mutable-buffer bindings; every recompute enters
        # functional_call again after the original call's scope has ended.
        bindings = {'weight': parameter}
        buffers = {'counter': layer.counter.detach().clone()}

        def call(inputs):
            return torch.func.functional_call(layer, (bindings, buffers), (inputs,))

        with life.graph(graph):
            if checkpoint_mode == 'off':
                with managers[0]:
                    return call(value)
            with set_checkpoint_early_stop(checkpoint_mode == 'early'):
                return checkpoint(call, value, use_reentrant=False, context_fn=lambda: managers)

    before = api().current_v3_lifecycle()
    try:
        with life.activate():
            f = forward('F', x, live)
            clean = forward('gain_clean', x, alias)
            packed = forward('packed_replay', clean, alias)
            r, kl, rv = packed.sum(), packed.square().mean(), packed.sin().mean()
            if zero_kl:
                kl = kl * 0

            def joint(seed):
                vjp_arities.append(2)
                return torch.autograd.grad((r, kl), alias, grad_outputs=tuple(seed), retain_graph=True)[0]

            with life.backward('joint_r_kl'):
                torch.vmap(joint, randomness='same')(torch.eye(2, dtype=dtype))
            with life.backward('rv'):
                vjp_arities.append(1)
                torch.autograd.grad(rv, alias)
            with life.backward('ordinary'):
                f.square().sum().backward()
    finally:
        assert api().current_v3_lifecycle() is before
        assert life.frame == (None, None, None, None)
        assert layer.weight is live and torch.equal(live, live_before)
        assert layer.counter.item() == 0
        assert all(weight is (live if graph == 'F' else alias) for graph, _, _, weight in observations)
    return life, observations, vjp_arities


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32])
@pytest.mark.parametrize('p', [0., .05])
@pytest.mark.parametrize('mode', ['off', 'early', 'full'])
def test_v3_lifecycle_reuses_checkpoint_managers_for_two_vjps(dtype, p, mode):
    life, observations, arities = cycle(dtype, p, mode)
    assert life.state == 'CLOSED'
    assert arities == [2, 1]  # Independent of labels emitted by the recorder.
    assert [(g, phase, role) for g, phase, role, _ in observations if phase == 'original'] == [
        ('F', 'original', None), ('gain_clean', 'original', None), ('packed_replay', 'original', None)]
    assert all(role in ('joint_r_kl', 'rv') for graph, phase, role, _ in observations
               if phase == 'recompute' and graph != 'F')


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA autograd')
@pytest.mark.parametrize('mode', ['early', 'full'])
@pytest.mark.parametrize('fail_rv', [False, True])
def test_cuda_checkpoint_vjps_keep_lifecycle_context_and_restore_dispatch(mode, fail_rv):
    # Without scoped dispatch, CUDA recompute loses the activating ContextVars.
    with torch.autograd.set_multithreading_enabled(True):
        try:
            with torch.device('cuda'):
                if fail_rv:
                    with pytest.raises(RuntimeError, match='injected RV recompute failure'):
                        cycle(torch.float32, .05, mode, fail_rv=True)
                else:
                    life, observations, arities = cycle(torch.float32, .05, mode)
                    assert life.state == 'CLOSED'
                    assert arities == [2, 1]
                    assert {(g, phase, role) for g, phase, role, _ in observations
                            if phase == 'recompute'} == {
                        ('F', 'recompute', 'ordinary'),
                        ('gain_clean', 'recompute', 'joint_r_kl'),
                        ('packed_replay', 'recompute', 'joint_r_kl'),
                        ('gain_clean', 'recompute', 'rv'),
                        ('packed_replay', 'recompute', 'rv'),
                    }
        finally:
            assert torch.autograd.is_multithreading_enabled()


@pytest.mark.parametrize('previous', [False, True])
@pytest.mark.parametrize('failure', [False, True])
def test_activation_serializes_backward_and_restores_previous_dispatch(previous, failure):
    # Catches an unscoped global disable and restoration that always enables.
    life = context(auxiliary=False)
    with torch.autograd.set_multithreading_enabled(previous):
        try:
            with pytest.raises(RuntimeError, match='injected activation failure') if failure else nullcontext():
                with life.activate():
                    assert not torch.autograd.is_multithreading_enabled()
                    if failure:
                        raise RuntimeError('injected activation failure')
                    originals(life)
                    finish_roles(life)
        finally:
            assert torch.autograd.is_multithreading_enabled() is previous
    assert life.state == ('FAILED' if failure else 'CLOSED')
    assert api().current_v3_lifecycle() is None


def test_zero_kl_still_has_both_auxiliary_traversals():
    assert cycle(torch.float32, .05, 'early', zero_kl=True)[2] == [2, 1]


@pytest.mark.parametrize('mode', ['early', 'full'])
def test_exceptional_recompute_restores_distinct_live_alias_and_mutable_buffer(mode):
    with pytest.raises(RuntimeError, match='injected RV recompute failure'):
        cycle(torch.float64, .05, mode, fail_rv=True)


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32, torch.bfloat16])
def test_native_and_reordered_packed_masks_match_ordinary_seed_semantics(dtype):
    x = torch.ones(2, 5, 16, dtype=dtype)
    dropout = SampleStableStage1Dropout(SITE, .05)
    spec = Stage1StochasticContext(42, 3, ((17, 4), (19, 5)), (SITE,))
    with activate_stage1_stochastic_context(spec):
        ordinary = dropout(x)
    life = context(rows=((17, 4), (19, 5)))
    samples, tokens = torch.tensor([1, 0, 1, 0]), torch.tensor([4, 2, 4, 0])
    rng = torch.get_rng_state()
    with life.activate():
        for graph in ('F', 'gain_clean'):
            with life.graph(graph), life.checkpoint_contexts(graph, UNIT)[0]:
                assert torch.equal(dropout(x), ordinary)
        with life.graph('packed_replay'), life.checkpoint_contexts('packed_replay', UNIT)[0]:
            raw = api().packed_dropout_mask(SITE, .05, sample_indices=samples,
                token_positions=tokens, hidden_size=16, device=x.device)
            assert raw.dtype == torch.float32
            assert torch.equal(raw.to(dtype) * (1 / .95), ordinary[samples, tokens])
        finish_roles(life)
    assert torch.equal(torch.get_rng_state(), rng)


def test_no_block_shortcut_closes_with_only_ordinary_role():
    life = context(auxiliary=False)
    with life.activate():
        originals(life)
        finish_roles(life)
    assert life.state == 'CLOSED'


@pytest.mark.parametrize('missing', ['F', 'gain_clean', 'packed_replay'])
def test_missing_original_site_is_not_inferred_from_observations(missing):
    life = context()
    with pytest.raises(RuntimeError, match='missing.*original'):
        with life.activate():
            originals(life, omit=missing)
            finish_roles(life)
    assert life.state == 'FAILED'
    assert api().current_v3_lifecycle() is None


@pytest.mark.parametrize('failure', ['original', 'recompute', 'vjp', 'wrong_graph', 'wrong_unit', 'nested', 'unknown_site', 'wrong_origin'])
def test_duplicate_and_misbound_work_is_rejected(failure):
    life = context()
    with pytest.raises(RuntimeError, match='duplicate|wrong|unregistered|nested'):
        with life.activate():
            if failure in ('original', 'nested', 'unknown_site', 'wrong_origin', 'wrong_unit'):
                with life.graph('F'):
                    managers = life.checkpoint_contexts('F', ('vision', 99) if failure == 'wrong_unit' else UNIT)
                    with managers[0]:
                        if failure == 'nested':
                            with managers[0]:
                                pass
                        life.record('unknown' if failure == 'unknown_site' else SITE,
                                    origin='packed' if failure == 'wrong_origin' else 'native')
                        if failure == 'original':
                            life.record(SITE, origin='native')
            else:
                originals(life)
                with life.backward('joint_r_kl'):
                    managers = life.checkpoint_contexts('F' if failure == 'wrong_graph' else 'gain_clean', UNIT)
                    with managers[1]:
                        life.record(SITE, origin='native')
                    if failure == 'recompute':
                        with managers[1]:
                            life.record(SITE, origin='native')
                with life.backward('rv'):
                    pass
                if failure == 'vjp':
                    with life.backward('rv'):
                        pass
    assert life.state == 'FAILED'
    assert api().current_v3_lifecycle() is None


def test_stale_closure_and_nested_activation_are_rejected():
    life = context(auxiliary=False)
    with life.activate():
        originals(life)
        saved = life.checkpoint_contexts('F', UNIT)[1]
        finish_roles(life)
    with pytest.raises(RuntimeError, match='stale'):
        with saved:
            pass
    with pytest.raises(RuntimeError, match='stale'):
        with life.activate():
            pass
    with pytest.raises(RuntimeError, match='nested'):
        with context().activate(), context().activate():
            pass
