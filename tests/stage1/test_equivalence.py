from dataclasses import replace

import pytest

from stage1.data import PreparedStage1Example
from stage1.equivalence import (
    Stage1CanonicalDataset,
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
        epochs=1,
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
        epochs=1,
        micro_batch_size=1,
        gradient_accumulation_steps=4,
    )
    occurrence = schedule.update(0).samples[0]
    bad_update = replace(
        schedule.update(0),
        samples=(replace(occurrence, source_index=999), *schedule.update(0).samples[1:]),
    )
    bad_schedule = replace(schedule, updates=(bad_update, *schedule.updates[1:]))

    with pytest.raises(ValueError, match="source_index"):
        validate_stage1_schedule(bad_schedule)
