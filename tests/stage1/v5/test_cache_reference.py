"""Reference replay must use its own unchanged output head and every query row."""
from dataclasses import replace

import pytest
import torch

from peft_common import resolve_qwen2_5_vl_model
from tests.stage1.test_run_cache import make_cache
from tests.stage1.v5.test_trainer import integrated_fixture


def setup_reference(tmp_path, *, helper='targets'):
    trainer, batch, calls, _, _, values = integrated_fixture(tmp_path, debug=False)
    trainer.run_cache = make_cache(tmp_path, helper=helper)
    trainer.v5_config = replace(trainer.v5_config, loss_chunk_positions=3)
    mask = torch.zeros_like(batch['input_ids'], dtype=torch.bool)
    for sample in batch['stage1_batch_metadata'].samples:
        for span in sample.latent_spans:
            mask[0, list(span.latent_positions)] = True
    model_inputs = {key: batch[key] for key in ('input_ids', 'attention_mask', 'position_ids')}
    model_inputs.update(latent_hidden_states=values, image_out_mask=mask)
    return trainer, batch, calls, model_inputs


@pytest.mark.parametrize('autocast', [False, True])
def test_reference_replay_uses_original_head_and_chunk_geometry(tmp_path, autocast):
    trainer, batch, calls, model_inputs = setup_reference(tmp_path)
    keys = batch['stage1_v5_metadata'].query_keys
    owner = resolve_qwen2_5_vl_model(trainer.ema_teacher.model)
    seen = []
    def observe(module, args):
        seen.append((id(module), tuple(args[0].shape), args[0].dtype,
                     torch.is_autocast_enabled('cpu'), torch.is_grad_enabled()))
    handle = owner.lm_head.register_forward_pre_hook(observe)
    try:
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
            live = trainer._stage1_reference_logits(model_inputs, keys, batch['stage1_batch_metadata'])
            trainer.run_cache.phase.value = 1
            replay = trainer._stage1_reference_logits(model_inputs, keys, batch['stage1_batch_metadata'])
    finally:
        handle.remove()
    assert torch.equal(live, replay)
    assert calls == ['reference']
    assert [entry[1] for entry in seen[:3]] == [(3, 8), (3, 8), (2, 8)]
    assert seen[:3] == seen[3:]
    assert all(entry[-2:] == (autocast, False) for entry in seen)


def test_all_query_positions_survive_dynamic_entropy_selection(tmp_path):
    trainer, batch, calls, _ = setup_reference(tmp_path)
    trainer.v5_config = replace(trainer.v5_config, selection='clean', preference_entropy_top_percent=20)
    batches, _ = trainer.get_batch_samples(iter([batch]), 4)
    assert batches[0]['stage1_v5_window'].global_dil_tokens == 2
    trainer.training_step(trainer.model, batches[0])
    trainer.run_cache.phase.value = 1
    record = trainer.run_cache.read('reference', 7)
    assert record['positions'] == (9, 10, 11, 19, 20, 21, 22, 23)
    assert record['hidden'].shape == (8, 8)
    trainer.model.zero_grad()
    calls.clear()
    batches, _ = trainer.get_batch_samples(iter([batch]), 4)
    trainer.training_step(trainer.model, batches[0])
    assert calls == ['helper', 'F', 'packed']


def test_reference_eligibility_includes_injected_latents(tmp_path):
    trainer, batch, _, model_inputs = setup_reference(tmp_path, helper='vision')
    with pytest.raises(ValueError, match='invariant|fixed'):
        trainer._stage1_reference_logits(model_inputs, batch['stage1_v5_metadata'].query_keys,
                                         batch['stage1_batch_metadata'])


def test_empty_query_skips_reference_as_before(tmp_path):
    trainer, batch, calls, _, _, _ = integrated_fixture(tmp_path, empty=True, debug=False)
    trainer.run_cache = make_cache(tmp_path)
    batches, _ = trainer.get_batch_samples(iter([batch]), 4)
    trainer.training_step(trainer.model, batches[0])
    assert calls == ['helper', 'F']
    assert not list((trainer.run_cache.plan.root / 'reference').iterdir())


def test_reference_rope_and_rng_state_match(tmp_path):
    trainer, batch, _, model_inputs = setup_reference(tmp_path)
    model_inputs.pop('position_ids')
    owner = resolve_qwen2_5_vl_model(trainer.ema_teacher.model)
    keys = batch['stage1_v5_metadata'].query_keys
    live = trainer._stage1_reference_logits(model_inputs, keys, batch['stage1_batch_metadata'])
    rope = owner.rope_deltas.clone()
    owner.rope_deltas.fill_(999)
    trainer.run_cache.phase.value = 1
    rng = torch.random.get_rng_state()
    replay = trainer._stage1_reference_logits(model_inputs, keys, batch['stage1_batch_metadata'])
    assert torch.equal(live, replay)
    assert torch.equal(rope, owner.rope_deltas)
    assert torch.equal(rng, torch.random.get_rng_state())
