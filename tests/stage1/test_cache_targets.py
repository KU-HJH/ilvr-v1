"""Fixed targets replay exactly; EMA may reuse vision but never stale queries."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from stage1.ema import Stage1EmaTeacher
from stage1.targets import Stage1TeacherTargetBuilder
from tests.stage1.test_run_cache import make_cache
from tests.stage1.test_targets import (
    CompleteQwenTeacher, LiteralTokenizer, GOLDEN_MULTIBLOCK, ONE_BLOCK,
    TIED_CANDIDATES, ZERO_HELPERS, make_case,
)


class StatefulTeacher(CompleteQwenTeacher):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Identity()
        self.rope_deltas = None

    def forward(self, **kwargs):
        result = super().forward(**kwargs)
        self.rope_deltas = torch.tensor([[kwargs['input_ids'].shape[1]]])
        return result


def builder(cache=None, *, diagnostics=False, coverage=.9, k=8):
    return Stage1TeacherTargetBuilder(LiteralTokenizer(), helper_group_L=256,
        coverage_p=coverage, image_pool_k=k, validate=diagnostics,
        collect_diagnostics=diagnostics, run_cache=cache)


def inputs(case, *, replay=False, cached=True):
    result = dict(case.inputs)
    if cached:
        result['stage1_cache_sources'] = tuple(s.source_index for s in case.metadata.samples)
    if replay:
        result.pop('pixel_values_latent')
    return result


def with_donor_ids(case):
    case.metadata = replace(case.metadata, samples=tuple(replace(sample,
        helper_image_ids=tuple(f'{sample.source_index}-helper-{i}' for i in sample.helper_to_block),
        original_image_ids=(f'{sample.source_index}-original',)) for sample in case.metadata.samples))
    return case


def test_adaptive_empty_sample_does_not_restore_an_unexecuted_forward(tmp_path):
    from tests.stage1.history_fixtures import make_history_case
    from tests.stage1.test_history_targets import builder as adaptive_builder
    from peft_common import resolve_qwen2_5_vl_model
    case = make_history_case()
    x = {key: value.clone() for key, value in case.inputs.items()}
    for key in ('input_ids', 'attention_mask'):
        x[key] = x[key][:, :13]
    for key in ('pixel_values_latent', 'image_grid_thw_latent'):
        x[key] = x[key][:0]
    sample = replace(case.metadata.samples[0], latent_spans=(), helper_to_block=(),
        helper_grid_range=(0, 0), helper_pixel_range=(0, 0), unpadded_length=13)
    metadata = replace(case.metadata, samples=(sample,), sequence_length=13)
    cache = make_cache(tmp_path)
    target_builder = adaptive_builder(case, run_cache=cache)
    x['stage1_cache_sources'] = (sample.source_index,)
    owner = resolve_qwen2_5_vl_model(case.teacher)
    owner.rope_deltas = torch.tensor([[3]])
    target_builder.build(case.teacher, dict(x), metadata)
    cache.phase.value = 1
    owner.rope_deltas = torch.tensor([[9]])
    target_builder.build(case.teacher, dict(x), metadata)
    assert owner.rope_deltas.tolist() == [[9]]


def assert_targets_equal(left, right):
    assert torch.equal(left.latent_values, right.latent_values)
    assert left.latent_values.dtype == right.latent_values.dtype
    assert torch.equal(left.latent_mask, right.latent_mask)
    assert left.selected_counts == right.selected_counts
    assert left.selected_candidate_indices == right.selected_candidate_indices
    assert left.selection_margins == right.selection_margins
    assert left.diagnostics.keys() == right.diagnostics.keys()
    for name in left.diagnostics:
        assert torch.equal(left.diagnostics[name], right.diagnostics[name])
    assert len(left.rejected_blocks) == len(right.rejected_blocks)
    for a, b in zip(left.rejected_blocks, right.rejected_blocks):
        assert torch.equal(a.values, b.values)
        assert (a.batch_index, a.block_index, a.positions, a.donor_kind,
                a.donor_image_indices, a.donor_candidate_indices, a.query_provenance) == (
                b.batch_index, b.block_index, b.positions, b.donor_kind,
                b.donor_image_indices, b.donor_candidate_indices, b.query_provenance)


@pytest.mark.parametrize('mode', ['attention_mask', 'cross_helper_reverse_topk_independent',
                                'cross_helper_reverse_topk_cumulative'])
def test_fixed_targets_and_reverse_donors_roundtrip(tmp_path, mode):
    cache = make_cache(tmp_path)
    producer = builder(cache)
    case = with_donor_ids(make_case((GOLDEN_MULTIBLOCK, TIED_CANDIDATES)))
    teacher = StatefulTeacher()
    first = producer.build(teacher, inputs(case), case.metadata, voff_mode=mode)
    calls = (teacher.language_forward_count, teacher.helper_visual_call_count)
    cache.phase.value = 1
    second = producer.build(teacher, inputs(case, replay=True), case.metadata, voff_mode=mode)
    assert_targets_equal(first, second)
    assert (teacher.language_forward_count, teacher.helper_visual_call_count) == calls
    if mode != 'attention_mask':
        assert {b.donor_kind for b in second.rejected_blocks} == {'helper', 'original'}
    # Reassemble reversed source order with different padding/batch offsets.
    shuffled = with_donor_ids(make_case((TIED_CANDIDATES, GOLDEN_MULTIBLOCK)))
    expected = builder().build(StatefulTeacher(), inputs(shuffled, cached=False),
                               shuffled.metadata, voff_mode=mode)
    actual = producer.build(teacher, inputs(shuffled, replay=True), shuffled.metadata, voff_mode=mode)
    assert_targets_equal(expected, actual)


@pytest.mark.parametrize('sample', [ONE_BLOCK, ZERO_HELPERS])
def test_debug_off_metadata_stays_absent(tmp_path, sample):
    cache = make_cache(tmp_path)
    producer = builder(cache)
    case = make_case((sample,))
    first = producer.build(StatefulTeacher(), inputs(case), case.metadata)
    assert first.diagnostics == {}
    cache.phase.value = 1
    second = producer.build(StatefulTeacher(), inputs(case, replay=True), case.metadata)
    assert_targets_equal(first, second)
    assert all(not indices for row in second.selected_candidate_indices for indices in row)


def test_ema_reuses_vision_but_queries_updated_teacher(tmp_path):
    cache = make_cache(tmp_path, helper='vision', reference=False)
    producer = builder(cache, coverage=.01, k=1)
    case = make_case((ONE_BLOCK,))
    student = StatefulTeacher()
    student.embedding.requires_grad_(True)
    ema = Stage1EmaTeacher(student, decay=0., method='full')
    first = producer.build(ema.model, inputs(case), case.metadata)
    with torch.no_grad():
        student.embedding.weight.mul_(-10.)
    ema.update(student)
    cache.phase.value = 1
    second = producer.build(ema.model, inputs(case, replay=True), case.metadata)
    assert ema.model.helper_visual_call_count == 1
    assert ema.model.language_forward_count == 2
    assert not torch.equal(first.latent_values, second.latent_values)
    expected = builder(coverage=.01, k=1).build(ema.model, inputs(case, cached=False), case.metadata)
    assert_targets_equal(second, expected)


def test_original_image_vision_stays_live_for_ema(tmp_path):
    cache = make_cache(tmp_path, helper='vision', reference=False)
    producer = builder(cache)
    case = make_case((ONE_BLOCK,))
    teacher = StatefulTeacher()
    producer.build(teacher, inputs(case), case.metadata)
    cache.phase.value = 1
    producer.build(teacher, inputs(case, replay=True), case.metadata)
    assert teacher.original_visual_call_count == 2
    assert teacher.helper_visual_call_count == 1


def test_target_replay_preserves_teacher_forward_state(tmp_path):
    cache = make_cache(tmp_path)
    producer = builder(cache)
    case = make_case((ONE_BLOCK,))
    teacher = StatefulTeacher()
    producer.build(teacher, inputs(case), case.metadata)
    original = teacher.rope_deltas.clone()
    teacher.rope_deltas.fill_(999)
    cache.phase.value = 1
    rng = torch.random.get_rng_state()
    producer.build(teacher, inputs(case, replay=True), case.metadata)
    assert torch.equal(original, teacher.rope_deltas)
    assert torch.equal(rng, torch.random.get_rng_state())
