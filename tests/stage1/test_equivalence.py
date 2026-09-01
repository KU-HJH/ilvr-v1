from dataclasses import replace
from itertools import islice

import pytest
from accelerate.data_loader import BatchSamplerShard

from stage1.data import PreparedStage1Example
from stage1.equivalence import (
    Stage1CanonicalBatchSampler,
    Stage1CanonicalDataset,
    Stage1SampleOccurrence,
    expected_stage1_assignment,
    materialize_stage1_schedule,
    validate_stage1_schedule,
)


@pytest.fixture
def rows_3450():
    return tuple(
        PreparedStage1Example(
            source_index=index,
            source_id=str(index),
            conversation=[],
            original_image_paths=(),
            helper_image_paths=(),
        )
        for index in range(3450)
    )


def test_canonical_view_pads_3450_rows_to_3456_without_losing_provenance(rows_3450):
    """Catches padding that changes source-row provenance or omits final slots."""
    view = Stage1CanonicalDataset(rows_3450)

    assert len(view) == 3456
    assert [view.source_index_for_slot(index) for index in range(3448, 3456)] == [
        3448,
        3449,
        0,
        1,
        2,
        3,
        4,
        5,
    ]


@pytest.mark.parametrize(("micro_batch", "ga"), ((1, 4), (2, 2)))
def test_schedule_has_exact_216_updates_and_closed_rank_grouping(
    rows_3450, micro_batch, ga
):
    """Catches a schedule whose rank-local accumulation no longer consumes four rows."""
    schedule = materialize_stage1_schedule(
        rows_3450,
        rank=2,
        world_size=4,
        seed=42,
        epochs=3,
        micro_batch_size=micro_batch,
        gradient_accumulation_steps=ga,
    )

    assert schedule.updates_per_epoch == 216
    assert schedule.total_updates == 648
    first = schedule.update(0)
    assert [item.permutation_offset for item in first.samples] == [2, 6, 10, 14]
    expected_grouping = (
        [(0, 0), (1, 0), (2, 0), (3, 0)]
        if micro_batch == 1
        else [(0, 0), (0, 1), (1, 0), (1, 1)]
    )
    assert [
        (item.rank_local_microbatch_ordinal, item.row_ordinal_within_microbatch)
        for item in first.samples
    ] == expected_grouping


def test_canonical_dataset_propagates_scheduled_occurrence(rows_3450):
    """Catches canonical indices being reduced to source indices before collation."""
    schedule = materialize_stage1_schedule(
        rows_3450,
        rank=2,
        world_size=4,
        seed=42,
        epochs=3,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
    )
    occurrence = schedule.update(0).samples[0]
    view = Stage1CanonicalDataset(rows_3450)

    example = view[replace(schedule.scheduled_index(occurrence), occurrence=occurrence)]

    assert example.source_index == occurrence.source_index
    assert example.occurrence == occurrence


def test_schedule_validation_rejects_source_provenance_drift(rows_3450):
    """Catches a schedule whose slot identity is retained but source provenance changes."""
    schedule = materialize_stage1_schedule(
        rows_3450,
        rank=2,
        world_size=4,
        seed=42,
        epochs=3,
        micro_batch_size=1,
        gradient_accumulation_steps=4,
    )
    bad_schedule = replace(
        schedule,
        source_indices=(999, *schedule.source_indices[1:]),
    )

    with pytest.raises(ValueError, match="source_index"):
        validate_stage1_schedule(bad_schedule)


@pytest.mark.parametrize("start_global_step", (0, 3))
def test_schedule_validation_rejects_truncated_full_or_resumed_schedule(
    rows_3450, start_global_step
):
    """Catches validation accepting a partial canonical horizon as complete."""
    schedule = materialize_stage1_schedule(
        rows_3450,
        rank=2,
        world_size=4,
        seed=42,
        epochs=3,
        micro_batch_size=1,
        gradient_accumulation_steps=4,
        start_global_step=start_global_step,
    )

    with pytest.raises(ValueError, match="exactly"):
        validate_stage1_schedule(replace(schedule, updates=schedule.updates[:-1]))


def test_canonical_schedule_rejects_noncanonical_seed_and_source_rows(rows_3450):
    """Catches mutable seed or source-row ordering from redefining the fixed protocol."""
    with pytest.raises(ValueError, match="seed=42"):
        materialize_stage1_schedule(
            rows_3450,
            rank=2,
            world_size=4,
            seed=99,
            epochs=3,
            micro_batch_size=1,
            gradient_accumulation_steps=4,
        )
    with pytest.raises(ValueError, match="exactly 3450"):
        Stage1CanonicalDataset(rows_3450[:-1])
    reordered = list(rows_3450)
    reordered[0] = replace(reordered[0], source_index=1)
    with pytest.raises(ValueError, match="zero-based source_index"):
        Stage1CanonicalDataset(reordered)
    with pytest.raises(ValueError, match="invalid epoch"):
        expected_stage1_assignment(
            epoch=3,
            update_in_epoch=0,
            rank=0,
            micro_batch_size=1,
        )


@pytest.mark.parametrize(
    "values",
    (
        (0, 0, 0, 0, 0, 0, 3, 3, 0),
        (0, 3450, 0, 3450, 1, 0, 0, 0, 0),
    ),
)
def test_occurrence_rejects_offset_ordinal_and_source_slot_mismatches(values):
    """Catches occurrence identity that is locally shaped but not protocol-canonical."""
    with pytest.raises(ValueError, match="canonical"):
        Stage1SampleOccurrence(*values)


@pytest.mark.parametrize(("micro_batch", "ga"), ((1, 4), (2, 2)))
def test_accelerate_shard_selects_each_rank_global_batches_once(
    rows_3450, micro_batch, ga
):
    """Catches an Accelerate shard changing the prescribed rank-local row order."""
    sampler = Stage1CanonicalBatchSampler(
        Stage1CanonicalDataset(rows_3450),
        world_size=4,
        seed=42,
        epochs=3,
        micro_batch_size=micro_batch,
        gradient_accumulation_steps=ga,
    )

    rank_two = BatchSamplerShard(
        sampler,
        num_processes=4,
        process_index=2,
        split_batches=False,
        even_batches=False,
    )
    selected = list(islice(rank_two, ga))

    assert rank_two.split_batches is False
    assert rank_two.even_batches is False
    assert [
        index.occurrence.permutation_offset for batch in selected for index in batch
    ] == [2, 6, 10, 14]
