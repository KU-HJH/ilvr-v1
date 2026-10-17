"""Actual CPU v5 parent-Trainer witnesses, without training or GPU claims.

Breaks caught: rebuilding/detaching shared targets or clean prefix, skipping
adaptive preparation, changing configured-GAS tail denominators, leaking history
between samples, retaining attention hooks, and losing saved query semantics.
"""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
import torch
import torch.nn.functional as F

from stage1.query import Stage1QueryConfig, query_execution_config
from tests.stage1.v5.test_replay import digest_state
from tests.stage1.v5.test_trainer import integrated_fixture


def hook_inventory(model):
    return tuple((name, tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
                 for name, module in model.named_modules())


@pytest.mark.parametrize('mode', ['legacy', 'adaptive'])
@pytest.mark.parametrize('empty_slots', [(False,), (False, True), (False, False, False, False)])
def test_real_parent_step_shared_snapshot_gradients_and_literal_window_denominator(
        tmp_path, monkeypatch, mode, empty_slots):
    """Missing history calls, target sharing or GAS scaling must fail here."""
    import transformers.models.qwen2_5_vl.stage1_v5_replay as replay
    from utils_deepseed import parse_args
    parser_flags = ['--stage1_variant', 'v5', '--tuning_method', 'dora',
        '--attention_implementation', 'sdpa', '--per_device_train_batch_size', '1']
    if mode == 'adaptive': parser_flags += ['--query_mode', 'adaptive']
    args = parse_args(parser_flags)
    assert args.stage1_query_config.mode == mode
    trainer, batch, calls, forwards, backwards, _ = integrated_fixture(
        tmp_path, history_config=args.stage1_query_config)
    frozen = trainer.ema_teacher.model
    before, hooks = digest_state(frozen), hook_inventory(frozen)
    steps = []
    for empty in empty_slots:
        metadata = batch['stage1_v5_metadata']
        if empty:
            metadata = replace(metadata, blocks=tuple(replace(block,
                target_positions=(), target_ids=()) for block in metadata.blocks))
        steps.append(dict(batch, stage1_v5_metadata=metadata))
    batches, _ = trainer.get_batch_samples(iter(steps), 4)
    assert all(b['stage1_v5_window'].global_original_steps == 2 * len(steps) for b in batches)
    assert all(b['stage1_v5_window'].actual_microbatches == len(steps) for b in batches)
    traces, clean_outputs, build_rng = [], [], []
    real_replay, real_build = replay.run_v5_packed_replay, trainer.target_builder.build

    def observed_replay(decoder, head, trace, *args, **kwargs):
        # This is the actual attached clean trace consumed by packed OFF, not a
        # separately manufactured embedding or a replacement compute_loss.
        traces.append(trace)
        return real_replay(decoder, head, trace, *args, **kwargs)

    def observed_build(*args, **kwargs):
        rng = torch.random.get_rng_state().clone()
        targets = real_build(*args, **kwargs)
        build_rng.append(torch.equal(rng, torch.random.get_rng_state()))
        return targets

    monkeypatch.setattr(replay, 'run_v5_packed_replay', observed_replay)
    monkeypatch.setattr(trainer.target_builder, 'build', observed_build)
    handle = trainer.model.register_forward_hook(lambda _m, _a, out: clean_outputs.append(out))
    try:
        for index, prepared in enumerate(batches):
            first_call, first_forward, first_trace = len(calls), len(forwards), len(traces)
            loss = trainer.training_step(trainer.model, prepared)
            targets = trainer.target_builder.targets
            diagnostic = trainer._latest_stage1_diagnostics
            # Independent literal reduction: every original block counts, even
            # empty auxiliaries; the parent retains /4 on a one/two-slot tail.
            auxiliary_scale = 4 / (2 * len(steps))
            assert float(diagnostic['auxiliary_scale']) == auxiliary_scale
            expected = (.1 * diagnostic['ce_loss'] + .1 * diagnostic['alignment_loss']
                + diagnostic['dil_scale'] * diagnostic['dil_raw_sum']
                + auxiliary_scale * .1 * diagnostic['topk_entropy_raw_sum']
                + .01 * diagnostic['text_kl_raw']) / 4
            torch.testing.assert_close(loss, expected)
            # Native per-token CE is unchanged, independently evaluated from
            # the actual F logits and original next-token labels.
            logits = clean_outputs[-1].logits
            ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, 128),
                                 batch['labels'][:, 1:].reshape(-1))
            torch.testing.assert_close(diagnostic['ce_loss'], ce.detach())
            expected_calls = ['helper'] * (3 if mode == 'adaptive' else 1) + ['F']
            if not empty_slots[index]:
                expected_calls += ['reference', 'packed']
            assert calls[first_call:] == expected_calls
            if mode == 'adaptive' and not empty_slots[index]:
                assert len(calls[first_call:]) == 2 + 4  # M=2, all three downstream contexts
            f = forwards[first_forward + (0 if mode == 'adaptive' else 1)]
            # Adaptive prefixes call the decoder directly; legacy helper calls
            # the wrapper once. Both then pass their detached snapshot to F.
            assert f['latent_hidden_states'].data_ptr() == targets.latent_values.data_ptr()
            assert not targets.latent_values.requires_grad
            if not empty_slots[index]:
                reference = forwards[-1]
                assert reference['latent_hidden_states'].data_ptr() == targets.latent_values.data_ptr()
                assert len(traces) == first_trace + 1
                trace = traces[-1]
                torch.testing.assert_close(trace.inputs_embeds[targets.latent_mask],
                                           targets.latent_values, rtol=0, atol=0)
                assert trace.inputs_embeds.requires_grad
                assert all(k.requires_grad and v.requires_grad for k, v in trace.key_values)
            else:
                assert len(traces) == first_trace
                assert float(diagnostic['dil_raw_sum']) == float(diagnostic['topk_entropy_raw_sum']) == 0
            assert trainer._v5_step_stack is None and backwards[-1].state == 'CLOSED'
            assert hook_inventory(frozen) == hooks
    finally:
        handle.remove()
    gradients = {name: parameter.grad for name, parameter in trainer.model.named_parameters()
                 if parameter.requires_grad}
    assert any('lora_B' in name and grad is not None and bool(grad.abs().sum() > 0)
               for name, grad in gradients.items())
    assert all(grad is not None and bool(torch.isfinite(grad).all()) for grad in gradients.values())
    trainer.optimizer.step()
    assert digest_state(frozen) == before
    assert all(not p.requires_grad and p.grad is None for p in frozen.parameters())
    assert all(not module.training for module in frozen.modules())
    assert build_rng == [True] * len(steps)
    assert len(backwards) == len(steps)
    print(json.dumps({'mode': mode, 'empty_slots': empty_slots, 'calls': calls,
        'original_blocks': 2 * len(steps), 'auxiliary_scale': auxiliary_scale,
        'parent_denominator': 4, 'reference_unchanged': True, 'builder_rng_unchanged': build_rng}))


def test_repeated_actual_builds_reset_sample_history_preserve_rng_and_remove_hooks(tmp_path):
    """Retained per-sample ledger/text state or attention hooks breaks A/B/A."""
    trainer, batch, *_ = integrated_fixture(tmp_path,
        history_config=Stage1QueryConfig('adaptive', .55, 'all'))
    teacher, builder = trainer.ema_teacher.model, trainer.target_builder
    hooks, frozen = hook_inventory(teacher), digest_state(teacher)
    metadata = batch['stage1_batch_metadata']
    changed = deepcopy(batch)
    changed['input_ids'][0, 11:13] = torch.tensor([44, 45])
    changed['pixel_values_latent'].add_(10)
    rng = torch.random.get_rng_state().clone()
    first = builder.build(teacher, dict(batch), metadata)
    other = builder.build(teacher, dict(changed), metadata)
    last = builder.build(teacher, dict(batch), metadata)
    torch.testing.assert_close(first.latent_values, last.latent_values, rtol=0, atol=0)
    assert torch.equal(first.latent_mask, last.latent_mask)
    assert first.selected_counts == last.selected_counts
    assert first.selected_candidate_indices == last.selected_candidate_indices
    assert not torch.equal(first.latent_values, other.latent_values)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert hook_inventory(teacher) == hooks and digest_state(teacher) == frozen


def test_saved_parser_query_builds_same_next_targets_with_unchanged_reference(tmp_path):
    """Dropping saved gamma/layers/mode or inheriting stale defaults fails here."""
    from stage1.targets import Stage1TeacherTargetBuilder
    from stage1.v5_resume import load_v5_contract
    from tests.stage1.v5.test_resume import contract
    from tests.stage1.history_fixtures import HistoryTokenizer
    from utils_deepseed import parse_args
    config = Stage1QueryConfig('adaptive', .55, 'all')
    trainer, batch, *_ = integrated_fixture(tmp_path, history_config=config)
    saved = contract()
    saved['execution']['query_config'] = query_execution_config(config, (0, 1))
    saved['execution'].update(tuning_method='dora', per_device_train_batch_size=1,
                              attention_implementation='sdpa')
    checkpoint = tmp_path / 'stage1_v5.json'
    checkpoint.write_text(json.dumps(saved))
    args = parse_args(['--resume_from_checkpoint_path', str(tmp_path)])
    assert load_v5_contract(tmp_path) == saved
    assert args.stage1_query_config == config
    next_batch = deepcopy(batch)
    next_batch['input_ids'][0, 17:19] = torch.tensor([42, 43])
    metadata = next_batch['stage1_batch_metadata']
    before = digest_state(trainer.ema_teacher.model)
    expected = trainer.target_builder.build(trainer.ema_teacher.model, dict(next_batch), metadata)
    resumed = Stage1TeacherTargetBuilder(HistoryTokenizer(), image_pool_k=2, helper_group_L=2,
        query_config=args.stage1_query_config)
    actual = resumed.build(trainer.ema_teacher.model, dict(next_batch), metadata)
    torch.testing.assert_close(actual.latent_values, expected.latent_values, rtol=0, atol=0)
    assert torch.equal(actual.latent_mask, expected.latent_mask)
    assert actual.selected_counts == expected.selected_counts
    assert actual.selected_candidate_indices == expected.selected_candidate_indices
    assert digest_state(trainer.ema_teacher.model) == before
    with pytest.raises(ValueError, match='conflict'):
        parse_args(['--resume_from_checkpoint_path', str(tmp_path), '--query_mode', 'legacy'])
