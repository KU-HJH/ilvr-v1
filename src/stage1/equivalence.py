"""Canonical sample identity and global batching for the Stage 1 comparison."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator, Sequence

import torch

from stage1.data import PreparedStage1Example
from stage1.types import STAGE1_CANONICAL_VIEW_LENGTH, Stage1SampleOccurrence


@dataclass(frozen=True)
class Stage1ScheduledIndex:
    """Dataset index that preserves an immutable schedule occurrence."""

    occurrence: Stage1SampleOccurrence


class Stage1CanonicalDataset:
    """A fixed 3,456-slot view which cyclically pads the source rows."""

    def __init__(
        self,
        source_rows: Sequence[PreparedStage1Example],
        *,
        padded_length: int = STAGE1_CANONICAL_VIEW_LENGTH,
    ) -> None:
        self.source_rows = tuple(source_rows)
        self.padded_length = padded_length
        if not self.source_rows:
            raise ValueError("Stage 1 canonical dataset requires at least one source row")
        if padded_length != STAGE1_CANONICAL_VIEW_LENGTH:
            raise ValueError("Stage 1 canonical dataset padded_length must be 3456")
        if len(self.source_rows) > padded_length:
            raise ValueError("Stage 1 canonical dataset source rows exceed 3456 slots")

    def __len__(self) -> int:
        return self.padded_length

    def source_index_for_slot(self, slot: int) -> int:
        if type(slot) is not int or not 0 <= slot < len(self):
            raise IndexError(f"canonical slot must be in [0, {len(self)})")
        return self.source_rows[slot % len(self.source_rows)].source_index

    def __getitem__(self, index: Stage1ScheduledIndex) -> PreparedStage1Example:
        if not isinstance(index, Stage1ScheduledIndex):
            raise TypeError("Stage 1 canonical dataset requires a scheduled index")
        occurrence = index.occurrence
        if self.source_index_for_slot(occurrence.view_slot) != occurrence.source_index:
            raise ValueError("scheduled occurrence source_index does not match canonical slot")
        return replace(
            self.source_rows[occurrence.view_slot % len(self.source_rows)],
            occurrence=occurrence,
        )


@dataclass(frozen=True)
class Stage1ScheduleUpdate:
    samples: tuple[Stage1SampleOccurrence, ...]


@dataclass(frozen=True)
class Stage1MaterializedSchedule:
    """Rank-local update view used to assert the controlled comparison schedule."""

    updates: tuple[Stage1ScheduleUpdate, ...]
    updates_per_epoch: int
    total_updates: int
    rank: int
    seed: int
    micro_batch_size: int
    start_global_step: int
    source_indices: tuple[int, ...]

    def update(self, global_update: int) -> Stage1ScheduleUpdate:
        return self.updates[global_update]

    @staticmethod
    def scheduled_index(occurrence: Stage1SampleOccurrence) -> Stage1ScheduledIndex:
        return Stage1ScheduledIndex(occurrence)


@dataclass(frozen=True)
class Stage1ScheduleSummary:
    updates_per_epoch: int
    total_updates: int
    occurrences_per_rank_update: int


def _validate_schedule_parameters(
    *,
    world_size: int,
    epochs: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    start_global_step: int = 0,
) -> int:
    if world_size != 4:
        raise ValueError("Stage 1 canonical schedule requires world_size=4")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("Stage 1 canonical schedule requires positive integer epochs")
    if type(micro_batch_size) is not int or micro_batch_size not in (1, 2):
        raise ValueError("Stage 1 canonical schedule micro_batch_size must be 1 or 2")
    if type(gradient_accumulation_steps) is not int or gradient_accumulation_steps <= 0:
        raise ValueError("Stage 1 canonical schedule requires positive gradient accumulation")
    if micro_batch_size * gradient_accumulation_steps != 4:
        raise ValueError("Stage 1 canonical schedule requires B * GA = 4")
    updates_per_epoch = STAGE1_CANONICAL_VIEW_LENGTH // (4 * micro_batch_size * gradient_accumulation_steps)
    if type(start_global_step) is not int or not 0 <= start_global_step <= epochs * updates_per_epoch:
        raise ValueError("Stage 1 canonical schedule has an invalid start_global_step")
    return updates_per_epoch


def _epoch_permutation(seed: int, epoch: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    return torch.randperm(STAGE1_CANONICAL_VIEW_LENGTH, generator=generator).tolist()


def _occurrence(
    *,
    permutation: Sequence[int],
    epoch: int,
    update_in_epoch: int,
    rank: int,
    micro_batch_size: int,
    rank_local_sample_ordinal: int,
) -> Stage1SampleOccurrence:
    if micro_batch_size == 1:
        microbatch = rank_local_sample_ordinal
        row = 0
    else:
        microbatch, row = divmod(rank_local_sample_ordinal, 2)
    permutation_offset = 16 * update_in_epoch + 4 * rank_local_sample_ordinal + rank
    view_slot = permutation[permutation_offset]
    return Stage1SampleOccurrence(
        epoch=epoch,
        view_slot=view_slot,
        permutation_offset=permutation_offset,
        canonical_slot_id=epoch * STAGE1_CANONICAL_VIEW_LENGTH + view_slot,
        source_index=view_slot % 3450,
        rank=rank,
        rank_local_sample_ordinal=rank_local_sample_ordinal,
        rank_local_microbatch_ordinal=microbatch,
        row_ordinal_within_microbatch=row,
    )


def expected_stage1_assignment(
    *, epoch: int, update_in_epoch: int, rank: int, micro_batch_size: int
) -> tuple[Stage1SampleOccurrence, ...]:
    """Return the seed-42 canonical assignment for one rank and update."""
    _validate_schedule_parameters(
        world_size=4,
        epochs=1,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=4 // micro_batch_size,
    )
    if not 0 <= rank < 4 or not 0 <= update_in_epoch < 216:
        raise ValueError("Stage 1 expected assignment has an invalid rank or update")
    permutation = _epoch_permutation(42, epoch)
    return tuple(
        _occurrence(
            permutation=permutation,
            epoch=epoch,
            update_in_epoch=update_in_epoch,
            rank=rank,
            micro_batch_size=micro_batch_size,
            rank_local_sample_ordinal=ordinal,
        )
        for ordinal in range(4)
    )


class Stage1CanonicalBatchSampler:
    """Global batch sampler whose batches are sharded by Accelerate, not pre-sharded."""

    def __init__(
        self,
        dataset: Stage1CanonicalDataset,
        *,
        world_size: int,
        seed: int,
        epochs: int,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        start_global_step: int = 0,
    ) -> None:
        self.dataset = dataset
        self.world_size = world_size
        self.seed = seed
        self.epochs = epochs
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.start_global_step = start_global_step
        self.updates_per_epoch = _validate_schedule_parameters(
            world_size=world_size,
            epochs=epochs,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            start_global_step=start_global_step,
        )
        self.batch_size = micro_batch_size

    def __len__(self) -> int:
        remaining_updates = self.epochs * self.updates_per_epoch - self.start_global_step
        return remaining_updates * self.gradient_accumulation_steps * self.world_size

    def __iter__(self) -> Iterator[list[Stage1ScheduledIndex]]:
        for epoch in range(self.epochs):
            permutation = _epoch_permutation(self.seed, epoch)
            for update_in_epoch in range(self.updates_per_epoch):
                global_update = epoch * self.updates_per_epoch + update_in_epoch
                if global_update < self.start_global_step:
                    continue
                for microbatch in range(self.gradient_accumulation_steps):
                    for rank in range(self.world_size):
                        first_ordinal = microbatch * self.micro_batch_size
                        batch = []
                        for row in range(self.micro_batch_size):
                            occurrence = _occurrence(
                                permutation=permutation,
                                epoch=epoch,
                                update_in_epoch=update_in_epoch,
                                rank=rank,
                                micro_batch_size=self.micro_batch_size,
                                rank_local_sample_ordinal=first_ordinal + row,
                            )
                            batch.append(
                                Stage1ScheduledIndex(
                                    replace(
                                        occurrence,
                                        source_index=self.dataset.source_index_for_slot(
                                            occurrence.view_slot
                                        ),
                                    )
                                )
                            )
                        yield batch


def materialize_stage1_schedule(
    source_rows: Sequence[PreparedStage1Example],
    *,
    rank: int,
    world_size: int,
    seed: int,
    epochs: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    start_global_step: int = 0,
) -> Stage1MaterializedSchedule:
    """Materialize the rank-local occurrence identity for each optimizer update."""
    if not 0 <= rank < world_size:
        raise ValueError("Stage 1 materialized schedule has an invalid rank")
    dataset = Stage1CanonicalDataset(source_rows)
    sampler = Stage1CanonicalBatchSampler(
        dataset,
        world_size=world_size,
        seed=seed,
        epochs=epochs,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        start_global_step=start_global_step,
    )
    batches = tuple(sampler)
    rank_batches = batches[rank::world_size]
    updates = tuple(
        Stage1ScheduleUpdate(
            samples=tuple(
                index.occurrence
                for batch in rank_batches[offset : offset + gradient_accumulation_steps]
                for index in batch
            )
        )
        for offset in range(0, len(rank_batches), gradient_accumulation_steps)
    )
    return Stage1MaterializedSchedule(
        updates=updates,
        updates_per_epoch=sampler.updates_per_epoch,
        total_updates=epochs * sampler.updates_per_epoch,
        rank=rank,
        seed=seed,
        micro_batch_size=micro_batch_size,
        start_global_step=start_global_step,
        source_indices=tuple(
            dataset.source_index_for_slot(slot) for slot in range(len(dataset))
        ),
    )


def validate_stage1_schedule(schedule: Stage1MaterializedSchedule) -> Stage1ScheduleSummary:
    """Validate that a materialized rank-local schedule has complete unique identity."""
    if len(schedule.updates) > schedule.total_updates:
        raise ValueError("Stage 1 schedule has more updates than its declared total")
    seen = set()
    for offset, update in enumerate(schedule.updates):
        if len(update.samples) != 4:
            raise ValueError("Stage 1 schedule update must contain four rank-local samples")
        global_update = schedule.start_global_step + offset
        epoch, update_in_epoch = divmod(global_update, schedule.updates_per_epoch)
        permutation = _epoch_permutation(schedule.seed, epoch)
        for sample_ordinal, occurrence in enumerate(update.samples):
            expected = _occurrence(
                permutation=permutation,
                epoch=epoch,
                update_in_epoch=update_in_epoch,
                rank=schedule.rank,
                micro_batch_size=schedule.micro_batch_size,
                rank_local_sample_ordinal=sample_ordinal,
            )
            expected = replace(
                expected,
                source_index=schedule.source_indices[expected.view_slot],
            )
            if occurrence != expected:
                raise ValueError(
                    "Stage 1 schedule occurrence does not match exact slot/rank/grouping/source_index assignment"
                )
            if occurrence.canonical_slot_id in seen:
                raise ValueError("Stage 1 schedule repeats a canonical_slot_id")
            seen.add(occurrence.canonical_slot_id)
    return Stage1ScheduleSummary(
        updates_per_epoch=schedule.updates_per_epoch,
        total_updates=schedule.total_updates,
        occurrences_per_rank_update=4,
    )
