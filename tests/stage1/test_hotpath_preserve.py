"""Exact output and runtime-operation witnesses for metadata-only optimizations."""
from types import SimpleNamespace
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from stage1.targets import Stage1TeacherTargetBuilder
from stage1.types import Stage1LatentSpan
from .test_targets import LiteralTokenizer


@pytest.fixture(scope='module')
def local_processor():
    from transformers import AutoProcessor
    from stage1.composition import _add_fresh_latent_tokens
    path = Path(__file__).resolve().parents[2] / 'models/Qwen2.5-VL-3B-Instruct'
    if not path.is_dir():
        pytest.skip('local processor files unavailable; no downloads or model weights required')
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    _add_fresh_latent_tokens(processor.tokenizer)
    return processor


def image_example(helpers):
    from PIL import Image
    from stage1.data import PreparedStage1Example
    originals = (Image.new('RGB', (56, 84), 'blue'), Image.new('RGB', (112, 56), 'green'))
    help_images = tuple(Image.new('RGB', size, color) for size, color in
                        (((84, 140), 'red'), ((168, 56), 'yellow'))[:helpers])
    content = [dict(type='text', text='reasoning ')]
    for image in help_images:
        content.extend((dict(type='image', image=image), dict(type='text', text=' next step ')))
    return PreparedStage1Example(17, '17', [
        dict(role='user', content=[*(dict(type='image', image=i) for i in originals),
                                  dict(type='text', text='What changes?')]),
        dict(role='assistant', content=content)], originals, help_images)


def image_collator(processor, v3=True):
    from stage1.data import Stage1DataCollator
    from stage1.composition import _tokenizer_ids
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
    return Stage1DataCollator(processor, LatentTokenIds(*_tokenizer_ids(processor.tokenizer)),
                              latent_size=8, v3_enabled=v3)


@pytest.mark.parametrize('helpers', [0, 2])
def test_existing_qwen_full_pixels_equal_separate_role_outputs(local_processor, helpers):
    # Premise witness runs existing processor methods only, before changing dispatch.
    collator = image_collator(local_processor)
    sample = image_example(helpers)
    rng = torch.get_rng_state().clone()
    full = collator._process_full_conversations((sample,))
    original = collator._process_original_images((sample,))
    helper = collator._process_helper_images((sample,))
    n = len(sample.original_image_paths)
    pixels = int(full['image_grid_thw'][:n].prod(-1).sum())
    for key, boundary in (('pixel_values', pixels), ('image_grid_thw', n)):
        torch.testing.assert_close(full[key][:boundary], original[key], rtol=0, atol=0)
        if helpers:
            torch.testing.assert_close(full[key][boundary:], helper[key], rtol=0, atol=0)
        else:
            assert helper[key].numel() == 0
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('helpers', [0, 2])
def test_v3_reuses_processor_pixels_with_exact_legacy_batch(local_processor, helpers):
    legacy = image_collator(local_processor, v3=False)([image_example(helpers)])
    collator = image_collator(local_processor)
    image_type = type(local_processor.image_processor)
    original_preprocess = image_type._preprocess
    calls = []
    def observed(self, image, **kwargs):
        calls.append(image.size)
        return original_preprocess(self, image, **kwargs)
    with patch.object(image_type, '_preprocess', observed):
        actual = collator([image_example(helpers)])
    for key in legacy:
        if key == 'stage1_batch_metadata':
            assert replace(actual[key], v3=None) == legacy[key]
        else:
            torch.testing.assert_close(actual[key], legacy[key], rtol=0, atol=0)
    assert len(calls) == 2 + helpers


@pytest.mark.parametrize('mode', ['previous', 'all'])
def test_token_search_keeps_marker_rules_without_tensor_scalar_loops(mode):
    # Pattern priority differs from earliest/longest match; do not merge the rules.
    builder = Stage1TeacherTargetBuilder(LiteralTokenizer(), textual_states=mode)
    builder._assistant_patterns = (torch.tensor([50, 51]), torch.tensor([50]))
    ids = torch.tensor([1, 50, 7, 50, 51, 8, 91, 90, 92, 9, 97, 10, 91, 90, 92])
    sample = SimpleNamespace(source_index=1, latent_spans=(
        Stage1LatentSpan(6, 8, (7,)), Stage1LatentSpan(12, 14, (13,))))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        if mode == 'all':
            start, end = builder._assistant_bounds(ids, sample)
            assert (start, end) == (1, 2)
        else:
            start = end = builder._assistant_start(ids, sample)
            assert start == 3
        spans = builder._assistant_text_spans(ids, sample, end)
    assert spans == (
        [2, 3, 4, 5] if mode == 'all' else [3, 4, 5], [9, 11])
    operations = {e.key: e.count for e in profile.key_averages()}
    assert operations.get('aten::equal', 0) == 0
    assert operations.get('aten::_local_scalar_dense', 0) == 0


def test_marker_at_zero_still_ends_search_of_that_pattern():
    builder = Stage1TeacherTargetBuilder(LiteralTokenizer())
    ids = torch.tensor([50, 1, 50])
    sample = SimpleNamespace(source_index=7)
    for search in (builder._assistant_start, builder._assistant_bounds):
        with pytest.raises(ValueError, match='assistant marker'):
            search(ids, sample)


def test_bounds_keep_longest_pattern_at_same_start():
    builder = Stage1TeacherTargetBuilder(LiteralTokenizer())
    builder._assistant_patterns = (torch.tensor([50]), torch.tensor([50, 51]))
    assert builder._assistant_bounds(torch.tensor([1, 50, 51, 2]),
                                     SimpleNamespace(source_index=7)) == (1, 3)


def test_static_optimizer_metadata_reused_but_updates_match_fresh_validation(monkeypatch):
    from copy import deepcopy
    from stage1 import v3_zero1_update as api
    from stage1.v3_update import V3GainController
    from .test_v3_optimizer_integration import harness, accumulate
    original_version, original_arange = api.version, torch.arange
    versions, allocations = [], []
    def version(name):
        versions.append(name)
        return original_version(name)
    def arange(*a, **kw):
        allocations.append(a)
        return original_arange(*a, **kw)
    def trajectory(fresh):
        zero, engine, controller, trainer, _ = harness(monkeypatch, weighting='token_entropy')
        for i, lr in enumerate((.03, .007, 0.)):
            if fresh:
                controller = V3GainController(zero.all_parameters, controller.config)
                trainer.v3_gain_controller = controller
            zero.optimizer.param_groups[0]['lr'] = lr
            zero.reset_gradients()
            accumulate(controller, [[1.+i, -2.]], [[1., 1.]])
            engine.step()
        return zero.single_partition_of_fp32_groups[0].detach().clone(), deepcopy(zero.optimizer.state_dict())
    with patch.object(api, 'version', version), patch.object(torch, 'arange', arange):
        master, state = trajectory(False)
    assert versions == ['deepspeed', 'torch']
    assert len(allocations) == 1
    reference, reference_state = trajectory(True)
    assert torch.equal(master, reference)
    for key, value in state['state'][0].items():
        assert torch.equal(value, reference_state['state'][0][key])
    assert state['param_groups'] == reference_state['param_groups']


@pytest.mark.parametrize('damage', ['mapping', 'nan', 'lr', 'padding'])
def test_optimizer_cache_does_not_hide_mutated_mapping_or_dynamic_state(monkeypatch, damage):
    from .test_v3_optimizer_integration import harness, accumulate
    zero, engine, controller, _, _ = harness(monkeypatch, groups=[[[.5]]])
    accumulate(controller, [[1.]], [[1.]])
    engine.step()
    zero.reset_gradients()
    if damage == 'mapping':
        zero.all_parameters[0]._hp_mapping.hp_fragment_address.start += 1
    elif damage == 'nan':
        zero.optimizer.state[zero.single_partition_of_fp32_groups[0]]['exp_avg'].fill_(float('nan'))
    elif damage == 'lr':
        zero.optimizer.param_groups[0]['lr'] = float('nan')
    else:
        zero.averaged_gradients[0][-1].fill_(1.)
    accumulate(controller, [[1.]], [[1.]])
    with pytest.raises(RuntimeError, match='profile/mapping'):
        engine.step()


def test_v3_builds_layer_inventory_once_per_microstep(tmp_path):
    from .v3_helpers import product_v3_trainer
    from stage1.v3_config import Stage1V3Config
    from peft_common import resolve_qwen2_5_vl_model
    trainer, batch = product_v3_trainer(tmp_path, checkpointed=True,
        config=Stage1V3Config(text_selector='entropy'))
    layer = resolve_qwen2_5_vl_model(trainer.model).model.layers[0]
    visits = []
    original = layer.modules
    def observed():
        visits.append(1)
        return original()
    with patch.object(layer, 'modules', observed):
        trainer.training_step(trainer.model, batch)
    assert len(visits) == 1


def test_packed_coordinates_extracted_once_with_exact_checkpoint_masks_and_gradients(tmp_path):
    from stage1 import v3_gradients as api
    from stage1.v3_config import Stage1V3Config
    from .v3_helpers import product_v3_trainer
    original_mask, original_list = api.packed_dropout_mask, torch.Tensor.tolist
    def run(clone_coordinates):
        torch.manual_seed(17)
        trainer, batch = product_v3_trainer(tmp_path, checkpointed=True, image=True,
            config=Stage1V3Config(text_selector='entropy', gain_weighting='token_entropy'))
        masks, extractions, active, logits = [], [], set(), []
        from peft_common import resolve_qwen2_5_vl_model
        head = resolve_qwen2_5_vl_model(trainer.model).lm_head
        hook = head.register_forward_hook(lambda module, args, output: logits.append(output.detach().clone()))
        def tolist(tensor):
            if id(tensor) in active:
                extractions.append(1)
            return original_list(tensor)
        def mask(*a, **kw):
            if clone_coordinates:
                kw['sample_indices'] = kw['sample_indices'].clone()
                kw['token_positions'] = kw['token_positions'].clone()
            active.update((id(kw['sample_indices']), id(kw['token_positions'])))
            try:
                result = original_mask(*a, **kw)
                masks.append(result.detach().clone())
                return result
            finally:
                active.clear()
        with patch.object(api, 'packed_dropout_mask', mask), patch.object(torch.Tensor, 'tolist', tolist):
            try:
                loss = trainer.training_step(trainer.model, batch)
            finally:
                hook.remove()
        grads = [p.grad.clone() for p in trainer.model.parameters() if p.requires_grad and p.grad is not None]
        return loss, masks, grads, torch.get_rng_state(), len(extractions), logits
    actual = run(False)
    reference = run(True)
    for observed, expected in zip(actual[:4], reference[:4]):
        if isinstance(observed, list):
            assert len(observed) == len(expected)
            for a, b in zip(observed, expected):
                assert torch.equal(a, b)
        else:
            assert torch.equal(observed, expected)
    assert actual[4] == 2
    assert reference[4] > 2  # Forced fresh coordinates retain the former extraction behavior.
    assert len(actual[5]) == len(reference[5]) and actual[5]
    for a, b in zip(actual[5], reference[5]):
        assert torch.equal(a, b)


def test_packed_callback_mutation_cannot_poison_next_site_coordinates():
    from transformers.models.qwen2_5_vl.stage1_v3_trace import _PACKED, packed_dropout_output
    samples, positions = torch.zeros((1, 3), dtype=torch.long), torch.arange(3).view(1, -1)
    valid = torch.ones_like(samples, dtype=torch.bool)
    seen = []
    def callback(site, p, *, sample_indices, token_positions, hidden_size, device):
        seen.append(token_positions.tolist())
        token_positions.add_(10)
        return torch.ones((3, hidden_size), device=device)
    token = _PACKED.set((callback, samples, positions, valid, {}))
    try:
        packed_dropout_output('a', .05, torch.ones((1, 3, 2)))
        packed_dropout_output('b', .05, torch.ones((1, 3, 2)))
        positions.add_(1)
        packed_dropout_output('c', .05, torch.ones((1, 3, 2)))
    finally:
        _PACKED.reset(token)
    assert seen == [[0, 1, 2], [0, 1, 2], [1, 2, 3]]


def test_disabled_target_diagnostics_keep_exact_targets_and_selection():
    from .test_targets import make_case, GOLDEN_MULTIBLOCK
    reference_case, actual_case = make_case((GOLDEN_MULTIBLOCK,)), make_case((GOLDEN_MULTIBLOCK,))
    reference = Stage1TeacherTargetBuilder(LiteralTokenizer()).build(
        reference_case.teacher, reference_case.inputs, reference_case.metadata)
    builder = Stage1TeacherTargetBuilder(LiteralTokenizer(), collect_diagnostics=False)
    with patch.object(builder, '_diagnostics', side_effect=AssertionError('unused norm computation')):
        actual = builder.build(actual_case.teacher, actual_case.inputs, actual_case.metadata)
    assert torch.equal(actual.latent_values, reference.latent_values)
    assert torch.equal(actual.latent_mask, reference.latent_mask)
    assert actual.selected_counts == reference.selected_counts
    assert actual.selected_candidate_indices == reference.selected_candidate_indices
    assert actual.selection_margins == reference.selection_margins
    assert actual.diagnostics == {} and reference.diagnostics


def test_normal_v3_skips_event_records_but_observer_keeps_evidence(tmp_path):
    from .v3_helpers import product_v3_trainer
    from .v3_runtime_acceptance import observe_microstep
    from stage1.v3_lifecycle import current_v3_lifecycle
    from stage1.v3_config import Stage1V3Config
    def run(observe):
        torch.manual_seed(17)
        trainer, batch = product_v3_trainer(tmp_path, checkpointed=True,
            config=Stage1V3Config(text_selector='entropy'))
        lives = []
        original = trainer.accelerator.backward
        def backward(*a, **kw):
            lives.append(current_v3_lifecycle())
            return original(*a, **kw)
        from contextlib import nullcontext
        with patch.object(trainer.accelerator, 'backward', backward):
            with observe_microstep(trainer) if observe else nullcontext():
                loss = trainer.training_step(trainer.model, batch)
        assert lives[0].state == 'CLOSED'
        return loss, lives[0].events, [p.grad.clone() for p in trainer.model.parameters()
                                     if p.requires_grad and p.grad is not None]
    normal, diagnostic = run(False), run(True)
    assert torch.equal(normal[0], diagnostic[0])
    for a, b in zip(normal[2], diagnostic[2]):
        assert torch.equal(a, b)
    assert not normal[1] and diagnostic[1]
