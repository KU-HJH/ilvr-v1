"""Conditional tail KL: independent arithmetic, gradients and runtime boundaries."""
from collections import defaultdict
from dataclasses import replace
import math

import pytest
import torch

from stage1.v5_config import Stage1V5Config, resolve_v5_config
from stage1.v5_objective import compute_v5_loss_sums
from step_preference.topk_entropy_loss import CandidateSupport, build_entropy_support
from tests.stage1.v5.test_objective import fixture, metadata


def tail(*args, **kwargs):
    from step_preference import topk_entropy_loss
    function = getattr(topk_entropy_loss, 'entropy_support_tail_kl', None)
    assert callable(function), 'conditional tail KL implementation missing'
    return function(*args, **kwargs)


@pytest.mark.parametrize('temperature', [1., 2.])
@pytest.mark.parametrize('head', [10., 10000.])
def test_tail_kl_and_gradient_ignore_head_mass(temperature, head):
    """First nail: p=(.75,.25), q=(.25,.75), whatever the excluded head mass."""
    clean = torch.tensor([[head, head + 1, temperature * math.log(3), 0.]], requires_grad=True)
    off = torch.tensor([[head + 8, head - 1, 0., temperature * math.log(3)]], requires_grad=True)
    support = build_entropy_support(clean, off, torch.tensor([0]),
        k=2, mode='sync_topk', include_gt=False)
    loss = tail(clean, off, support, temperature=temperature)
    assert loss.item() == pytest.approx(.5 * math.log(3) * temperature**2)
    loss.sum().backward()
    assert clean.grad is None
    torch.testing.assert_close(off.grad, torch.tensor([[0., 0., -.5*temperature, .5*temperature]]))


def test_empty_and_singleton_tail_are_connected_zero_without_nan():
    """Catches all-masked softmax NaNs and padded ID0 overwriting membership."""
    clean = torch.tensor([[2., 1., 0.], [2., 1., 0.]], requires_grad=True)
    off = torch.tensor([[0., 1., 2.], [0., 1., 2.]], requires_grad=True)
    support = CandidateSupport(torch.tensor([[0, 1, 2], [0, 1, 0]]),
        torch.tensor([[True, True, True], [True, True, False]]), torch.tensor([False, True]))
    loss = tail(clean, off, support)
    torch.testing.assert_close(loss, torch.zeros(2), rtol=0, atol=0)
    loss.sum().backward()
    torch.testing.assert_close(off.grad, torch.zeros_like(off), rtol=0, atol=0)
    assert clean.grad is None


@pytest.mark.parametrize('mode,supports', [
    ('independent', [[0, 1], [1, 2, 3], [0, 2, 1]]),
    ('sync_topk', [[2, 0], [0, 2, 3], [1, 3]]),
])
@pytest.mark.parametrize('chunk', [1, 128])
def test_tail_objective_reuses_ragged_support_and_block_reduction(mode, supports, chunk):
    """Catches different support, wrong denominator, or late-bound chunk masks."""
    clean, ref, off, text = fixture()
    config = Stage1V5Config(lambda_tail_kl=.7, preference_topk_k=2,
        preference_topk_mode=mode, topk_include_gt=True, loss_chunk_positions=chunk)
    result = compute_v5_loss_sums(clean, ref, off, text, metadata(), config)
    tokens = []
    for row, head in enumerate(supports):
        ids = [i for i in range(4) if i not in head]
        p = clean[row, ids].detach().log_softmax(-1)
        q = off[row, ids].log_softmax(-1)
        tokens.append((p.exp() * (p-q)).sum())
    expected = (tokens[0]+tokens[1])/2 + tokens[2]
    torch.testing.assert_close(result.tail_kl_sum, expected)
    assert result.original_steps == 3  # Empty original block stays in denominator.
    torch.testing.assert_close(torch.autograd.grad(result.tail_kl_sum, off, retain_graph=True)[0],
        torch.autograd.grad(expected, off)[0])
    assert torch.autograd.grad(result.tail_kl_sum, clean, allow_unused=True)[0] is None


def test_tail_uses_clean_token_selection_and_production_diagnostics_can_stay_off():
    clean, ref, off, text = fixture()
    config = Stage1V5Config(lambda_tail_kl=1., preference_topk_k=2,
        selection='clean', preference_entropy_top_percent=20, preference_topk_mode='sync_topk')
    result = compute_v5_loss_sums(clean, ref, off, text, metadata(), config,
                                 validate=False, collect_diagnostics=False)
    # Clean selected positions 0 and2, with tail IDs[1,3] and[0,2].
    expected = []
    for row, ids in [(0, [1, 3]), (2, [0, 2])]:
        p = clean[row, ids].detach().log_softmax(-1)
        q = off[row, ids].log_softmax(-1)
        expected.append((p.exp() * (p-q)).sum())
    torch.testing.assert_close(result.tail_kl_sum, sum(expected))
    assert result.diagnostics == {}


@pytest.mark.parametrize('options', [
    {'lambda_tail_kl': -1}, {'lambda_tail_kl': float('nan')},
    {'tail_kl_temperature': 0}, {'tail_kl_temperature': float('inf')},
    {'lambda_tail_kl': 1}, {'lambda_tail_kl': 1, 'preference_topk_k': 1},
])
def test_invalid_tail_config_rejected(options):
    with pytest.raises(ValueError):
        resolve_v5_config(options)


def test_tail_default_is_off_and_has_independent_temperature():
    config = resolve_v5_config({})
    assert config.lambda_tail_kl == 0
    assert config.tail_kl_temperature == 1
    assert config.temperature == 2  # Existing Text-off temperature unchanged.
    assert resolve_v5_config({'lambda_tail_kl': .7, 'preference_topk_k': 2}).lambda_topk_entropy == 0


@pytest.mark.parametrize('mode', ['attention_mask', 'cross_helper_reverse_topk_independent',
                                  'cross_helper_reverse_topk_cumulative'])
def test_tail_contract_roundtrip_and_drift_rejection(mode):
    from stage1.v5_resume import build_v5_contract, validate_v5_resume
    from stage1.v5_reference import V5ReferenceIdentity
    reference = V5ReferenceIdentity(1, 'a'*64, 'b'*64, 'c'*64, ('lora_A',))
    config = Stage1V5Config(lambda_tail_kl=.7, tail_kl_temperature=2., preference_topk_k=2, voff_mode=mode)
    saved = build_v5_contract(config, reference, ce_weight=.1, sim_weight=.1,
        schedule={'dil_denominator': 'actual_window_eligible_dil_tokens'}, execution={})
    assert saved['schema_version'] == 5
    assert validate_v5_resume(saved, {}) == config
    for name, value in [('lambda_tail_kl', 0.), ('tail_kl_temperature', 1.)]:
        with pytest.raises(ValueError, match='conflict'):
            validate_v5_resume(saved, {name: value})
    saved['config'].pop('lambda_tail_kl')
    with pytest.raises(ValueError):
        validate_v5_resume(saved, {})


@pytest.mark.parametrize('version', [2, 3, 4])
def test_historical_contract_keeps_its_field_set_and_tail_off(version):
    from stage1.v5_resume import build_v5_contract, validate_v5_resume
    from stage1.v5_reference import V5ReferenceIdentity
    kwargs = {'selection': 'clean', 'preference_entropy_top_percent':20} if version == 3 else {}
    if version == 4:
        kwargs['voff_mode'] = 'cross_helper_reverse_topk_independent'
    config = Stage1V5Config(**kwargs)
    reference = V5ReferenceIdentity(1, 'a'*64, 'b'*64, 'c'*64, ('lora_A',))
    saved = build_v5_contract(config, reference, ce_weight=.1, sim_weight=.1,
        schedule={'dil_denominator': 'actual_window_selected_dil_tokens' if version == 3
                  else 'actual_window_eligible_dil_tokens'}, execution={})
    assert saved['schema_version'] == version
    assert 'lambda_tail_kl' not in saved['config']
    assert 'tail_kl_temperature' not in saved['config']
    assert validate_v5_resume(saved, {}).lambda_tail_kl == 0
    with pytest.raises(ValueError):
        validate_v5_resume(saved, {'lambda_tail_kl': .1})


@pytest.mark.parametrize('entropy', [0., .1])
def test_real_v5_backward_logs_all_weighted_terms_without_diagnostics(tmp_path, monkeypatch, entropy):
    """Catches missing tail-only replay, extra forwards, or debug-only metric wiring."""
    from transformers import Trainer
    from tests.stage1.v5.test_trainer import integrated_fixture
    trainer, inputs, calls, _, backs, _ = integrated_fixture(tmp_path, debug=False)
    trainer.v5_config = replace(trainer.v5_config, lambda_tail_kl=.7,
        lambda_topk_entropy=entropy, lambda_topk_mass=.05 if entropy else 0.)
    trainer._latest_stage1_diagnostics = None
    trainer._metrics = defaultdict(list)
    # Fixture lacks a real callback/history owner; capture only the final sink,
    # retaining native Stage1 and installed SFTTrainer log transformations.
    rows = []
    monkeypatch.setattr(Trainer, 'log', lambda self, logs, start_time=None: rows.append(dict(logs)))
    batches, _ = trainer.get_batch_samples(iter([inputs]), 4)
    loss = trainer.training_step(trainer.model, batches[0])
    assert calls == ['helper', 'F', 'reference', 'packed']
    assert len(backs) == 1
    assert trainer._latest_stage1_diagnostics is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.model.parameters())
    trainer.state.global_step = 1
    trainer.log({'loss': loss.item()})
    row = rows[-1]
    names = ('ce', 'alignment', 'dil', 'topk_entropy', 'topk_mass', 'text_kl', 'tail_kl')
    weighted = sum(row[f'stage1/weighted_{name}_loss'] for name in names)
    assert weighted == pytest.approx(loss.item(), rel=1e-5)
    assert row['stage1/tail_kl_loss'] > 0
    assert row['stage1/weighted_tail_kl_loss'] == pytest.approx(.7*row['stage1/tail_kl_loss'])
    assert row['stage1/total_loss'] == pytest.approx(loss.item())


def test_tail_runtime_identity_binds_kernel_and_objective_sources(tmp_path):
    """Catches accepting a resume after only its tail math source changes."""
    import hashlib
    import inspect
    from copy import deepcopy
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from deepspeed_runtime import DeepSpeedRuntimeProfile
    from peft_common import resolve_qwen2_5_vl_model
    from stage1.types import DEFAULT_STAGE1_TARGET_SUFFIXES
    from stage1.v5_reference import V5ReferenceIdentity
    from step_preference import topk_entropy_loss
    from tests.stage1.v5.test_trainer import integrated_fixture
    from pathlib import Path
    trainer, *_ = integrated_fixture(tmp_path, debug=False)
    trainer.v5_config = replace(trainer.v5_config, lambda_tail_kl=.7)
    trainer.v5_reference_identity = V5ReferenceIdentity(1, 'a'*64, 'b'*64, 'c'*64, ('weight',))
    trainer.stage1_peft_metadata = SimpleNamespace(base_model_reference='original/tiny',
        requested_target_suffixes=DEFAULT_STAGE1_TARGET_SUFFIXES)
    trainer.coverage_p = .9; trainer.image_pool_k = 2; trainer.helper_group_L = 2
    trainer.textual_states = 'previous'
    resolve_qwen2_5_vl_model(trainer.model).config.compress_strategy = 'average'
    trainer.args.bf16 = True; trainer.args.max_seq_length = 32768; trainer.args.max_steps = 3
    trainer.deepspeed_runtime_profile = DeepSpeedRuntimeProfile(True, 1, None, 'a'*64, 4)
    trainer.train_dataset = list(range(10))
    saved = trainer._build_v5_runtime_contract(DataLoader(trainer.train_dataset, batch_size=1))
    for key, function in [('tail_kl', topk_entropy_loss.entropy_support_tail_kl),
                          ('v5_objective', compute_v5_loss_sums)]:
        assert saved['execution']['source_sha256'][key] == hashlib.sha256(
            Path(inspect.getsourcefile(function)).read_bytes()).hexdigest()
    changed = deepcopy(saved)
    changed['execution']['source_sha256']['tail_kl'] = 'd'*64
    assert trainer._effective_v5_contract(changed) != trainer._effective_v5_contract(saved)
