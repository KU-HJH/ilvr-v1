"""Canonical sample identity and global batching for the Stage 1 comparison."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Literal, Sequence

import torch

from stage1.data import PreparedStage1Example
from stage1.types import (
    STAGE1_CANONICAL_SOURCE_ROWS,
    STAGE1_CANONICAL_VIEW_LENGTH,
    Stage1SampleOccurrence,
)


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
        if padded_length != STAGE1_CANONICAL_VIEW_LENGTH:
            raise ValueError("Stage 1 canonical dataset padded_length must be 3456")
        if len(self.source_rows) != STAGE1_CANONICAL_SOURCE_ROWS:
            raise ValueError("Stage 1 canonical dataset requires exactly 3450 source rows")
        if any(
            row.source_index != index for index, row in enumerate(self.source_rows)
        ):
            raise ValueError(
                "Stage 1 canonical dataset requires zero-based source_index ordering"
            )

    def __len__(self) -> int:
        return self.padded_length

    def source_index_for_slot(self, slot: int) -> int:
        if type(slot) is not int or not 0 <= slot < len(self):
            raise IndexError(f"canonical slot must be in [0, {len(self)})")
        return self.source_rows[slot % STAGE1_CANONICAL_SOURCE_ROWS].source_index

    def __getitem__(self, index: Stage1ScheduledIndex) -> PreparedStage1Example:
        if not isinstance(index, Stage1ScheduledIndex):
            raise TypeError("Stage 1 canonical dataset requires a scheduled index")
        occurrence = index.occurrence
        if self.source_index_for_slot(occurrence.view_slot) != occurrence.source_index:
            raise ValueError("scheduled occurrence source_index does not match canonical slot")
        return replace(
            self.source_rows[occurrence.view_slot % STAGE1_CANONICAL_SOURCE_ROWS],
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
    seed: int = 42,
    start_global_step: int = 0,
) -> int:
    if world_size != 4:
        raise ValueError("Stage 1 canonical schedule requires world_size=4")
    if type(seed) is not int or seed != 42:
        raise ValueError("Stage 1 canonical schedule requires seed=42")
    if epochs != 3:
        raise ValueError("Stage 1 canonical schedule requires exactly 3 epochs")
    if type(micro_batch_size) is not int or micro_batch_size not in (1, 2):
        raise ValueError("Stage 1 canonical schedule micro_batch_size must be 1 or 2")
    if type(gradient_accumulation_steps) is not int or gradient_accumulation_steps <= 0:
        raise ValueError("Stage 1 canonical schedule requires positive gradient accumulation")
    if micro_batch_size * gradient_accumulation_steps != 4:
        raise ValueError("Stage 1 canonical schedule requires B * GA = 4")
    updates_per_epoch = STAGE1_CANONICAL_VIEW_LENGTH // (
        4 * micro_batch_size * gradient_accumulation_steps
    )
    if type(start_global_step) is not int or not 0 <= start_global_step <= 648:
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
        source_index=view_slot % STAGE1_CANONICAL_SOURCE_ROWS,
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
        epochs=3,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=4 // micro_batch_size,
    )
    if not 0 <= epoch < 3 or not 0 <= rank < 4 or not 0 <= update_in_epoch < 216:
        raise ValueError("Stage 1 expected assignment has an invalid epoch, rank, or update")
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
            seed=seed,
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
    if schedule.updates_per_epoch != 216 or schedule.total_updates != 648:
        raise ValueError("Stage 1 schedule has noncanonical declared update counts")
    if schedule.seed != 42 or not 0 <= schedule.rank < 4:
        raise ValueError("Stage 1 schedule has noncanonical seed or rank")
    if schedule.micro_batch_size not in (1, 2):
        raise ValueError("Stage 1 schedule has noncanonical microbatch size")
    if not 0 <= schedule.start_global_step <= schedule.total_updates:
        raise ValueError("Stage 1 schedule has an invalid start_global_step")
    expected_source_indices = tuple(
        range(STAGE1_CANONICAL_SOURCE_ROWS)
    ) + tuple(range(6))
    if schedule.source_indices != expected_source_indices:
        raise ValueError("Stage 1 schedule has noncanonical source_index mapping")
    expected_update_count = schedule.total_updates - schedule.start_global_step
    if len(schedule.updates) != expected_update_count:
        raise ValueError(
            "Stage 1 schedule must contain exactly total_updates - start_global_step updates"
        )
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


CROSS_ARM_V1_POINTERS = (
    "/batch/per_device_train_batch_size",
    "/batch/gradient_accumulation_steps",
    "/batch/rank_local_microbatch_grouping",
    "/artifacts/run_id",
    "/artifacts/checkpoints",
    "/artifacts/results",
    "/artifacts/logs",
)
SAME_ARM_V1_POINTERS = (
    "/artifacts/run_id",
    "/artifacts/checkpoints",
    "/artifacts/results",
    "/artifacts/logs",
    "/diagnostics/bounded_optimizer_steps",
)
MANIFEST_PROFILES = {
    "cross_arm_v1": frozenset(CROSS_ARM_V1_POINTERS),
    "same_arm_v1": frozenset(SAME_ARM_V1_POINTERS),
}

_MANIFEST_SCHEMA: dict[str, object] = {
    "schema_version": "schema_version",
    "model": {
        "model_path": "path",
        "processor_path": "path",
        "model_revision": "str",
        "processor_revision": "str",
        "model_files_sha256": "sha256",
        "processor_files_sha256": "sha256",
        "config_sha256": "sha256",
    },
    "code": {
        "git_commit": "git_commit",
        "runtime_source_digest": "sha256",
        "software_versions": {
            "python": "str",
            "pytorch": "str",
            "transformers": "str",
            "peft": "str",
            "deepspeed": "str",
            "cuda": "str",
            "flash_attention": "str",
        },
    },
    "data": {
        "train_jsonl_path": "path",
        "train_sha256": "sha256",
        "test_jsonl_path": "path",
        "test_sha256": "sha256",
        "image_root": "path",
        "image_files_sha256": "sha256",
        "source_rows": "int",
        "canonical_epoch_slots": "int",
        "padding_rule": "str",
        "ordering_rule": "str",
    },
    "batch": {
        "world_size": "int",
        "per_device_train_batch_size": "int",
        "gradient_accumulation_steps": "int",
        "effective_global_batch_size": "int",
        "rank_local_microbatch_grouping": ("list", ("list", "int")),
    },
    "schedule": {
        "epochs": "int",
        "seed": "int",
        "permutation_rule": "str",
        "optimizer_steps_per_epoch": "int",
        "total_optimizer_steps": "int",
        "logging_steps": "int",
        "checkpoint_steps": ("list", "int"),
    },
    "tuning": {
        "method": "str",
        "dora": {
            "rank": "int",
            "alpha": "number",
            "target_modules": ("list", "str"),
            "dropout": "number",
        },
        "losses": {"ce_weight": "number", "similarity_weight": "number"},
        "targets": {
            "coverage_probability": "number",
            "image_pool_size": "int",
            "helper_group_size": "int",
            "latent_size": "int",
            "alignment_noise": "number",
        },
        "maximum_sequence_length": "int",
    },
    "optimizer": {
        "name": "str",
        "learning_rate": "number",
        "betas": ("list", "number"),
        "epsilon": "number",
        "weight_decay": "number",
        "max_grad_norm": "number",
    },
    "scheduler": {
        "name": "str",
        "warmup_optimizer_steps": "int",
        "total_optimizer_steps": "int",
    },
    "ema": {"tau": "number", "initialization": "str", "update_rule": "str"},
    "precision": {
        "training_dtype": "str",
        "loss_reduction_dtype": "str",
        "attention": "str",
        "gradient_checkpointing": "bool",
        "gradient_checkpointing_use_reentrant": "bool",
    },
    "stochastic": {
        "rng_schema": "str",
        "site_allowlist": ("list", "str"),
        "dropout_schema": "str",
        "noise_schema": "str",
    },
    "deepspeed": {
        "source_digest": "sha256",
        "settings": {
            "bf16": {"enabled": "bool"},
            "zero_optimization": {
                "stage": "int",
                "overlap_comm": "bool",
                "contiguous_gradients": "bool",
                "reduce_bucket_size": "int",
            },
            "gradient_clipping": "number",
            "steps_per_print": "int",
            "wall_clock_breakdown": "bool",
        },
    },
    "evaluation": {
        "dataset_path": "path",
        "dataset_sha256": "sha256",
        "shard_count": "int",
        "max_new_tokens": "int",
        "temperature": "number",
        "top_p": "number",
        "scorer_identity": "str",
    },
    "diagnostics": {
        "bounded_optimizer_steps": "bounded_steps",
        "evidence_schema": "str",
    },
    "artifacts": {
        "run_id": "str",
        "checkpoints": "path",
        "results": "path",
        "logs": "path",
    },
    "initialization": {
        "trainable_parameters": (
            "list",
            {"name": "str", "dtype": "str", "shape": ("list", "int"), "sha256": "sha256"},
        ),
        "ema_parameters": (
            "list",
            {"name": "str", "dtype": "str", "shape": ("list", "int"), "sha256": "sha256"},
        ),
    },
}
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")


def _validate_json_value(value: object, *, location: str = "$") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{location} must contain only finite JSON numbers")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, location=f"{location}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{location} JSON object keys must be strings")
            _validate_json_value(item, location=f"{location}/{key}")
        return
    raise TypeError(f"{location} contains unsupported JSON type {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Encode one finite JSON value using the experiment's canonical rules."""
    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor metadata and contiguous little-endian CPU bytes."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("canonical tensor hash requires a torch.Tensor")
    if tensor.layout is not torch.strided:
        raise ValueError("canonical tensor hash requires a strided tensor")
    if tensor.is_quantized:
        raise ValueError("canonical tensor hash does not support quantized tensors")
    if tensor.device.type == "meta":
        raise ValueError("canonical tensor hash requires materialized tensor values")
    cpu_tensor = tensor.detach().cpu().contiguous()
    if (cpu_tensor.is_floating_point() or cpu_tensor.is_complex()) and not bool(
        torch.isfinite(cpu_tensor).all()
    ):
        raise ValueError("canonical tensor hash requires finite tensor values")
    payload = cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    if sys.byteorder == "big" and cpu_tensor.element_size() > 1:
        component_size = (
            cpu_tensor.element_size() // 2
            if cpu_tensor.is_complex()
            else cpu_tensor.element_size()
        )
        payload = b"".join(
            payload[offset : offset + component_size][::-1]
            for offset in range(0, len(payload), component_size)
        )
    metadata = canonical_json_bytes(
        {"dtype": str(cpu_tensor.dtype), "shape": list(cpu_tensor.shape)}
    )
    return hashlib.sha256(metadata + payload).hexdigest()


def _git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("runtime sources must belong to a Git repository") from error
    return completed.stdout.strip()


def runtime_source_sha256(paths: Sequence[str | Path]) -> str:
    """Hash sorted repository-relative runtime source identities and bytes."""
    if not paths:
        raise ValueError("runtime source paths must not be empty")
    files: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.update(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix in {".py", ".sh"}
            )
        elif path.is_file():
            files.add(path.resolve())
        else:
            raise ValueError(f"runtime source path is not a file or directory: {path}")
    if not files:
        raise ValueError("runtime source paths contain no Python or shell files")
    roots = {
        Path(_git_output("-C", str(path.parent), "rev-parse", "--show-toplevel"))
        .resolve()
        for path in files
    }
    if len(roots) != 1:
        raise ValueError("runtime sources must belong to one Git repository")
    repository = roots.pop()
    identities = sorted((path.relative_to(repository).as_posix(), path) for path in files)
    relative_paths = [identity for identity, _ in identities]
    dirty = _git_output(
        "-C",
        str(repository),
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        *relative_paths,
    )
    if dirty:
        raise ValueError(f"dirty tracked runtime source: {dirty.splitlines()[0]}")
    digest = hashlib.sha256()
    for identity, path in identities:
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_manifest(value: Mapping[str, object] | str | bytes) -> dict[str, object]:
    if isinstance(value, (str, bytes)):
        try:
            text = value.decode("utf-8") if isinstance(value, bytes) else value
            decoded = json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid resolved manifest JSON") from error
    elif isinstance(value, Mapping):
        decoded = copy.deepcopy(dict(value))
    else:
        raise TypeError("resolved manifest must be a mapping or JSON string")
    if type(decoded) is not dict:
        raise ValueError("resolved manifest must be a JSON object")
    return decoded


def _validate_scalar(value: object, kind: str, location: str) -> None:
    if kind == "schema_version":
        if value != "stage1-resolved-manifest-v1" or type(value) is not str:
            raise ValueError(f"{location} has unsupported schema_version")
    elif kind == "str":
        if type(value) is not str or not value:
            raise TypeError(f"{location} must have non-empty string type")
    elif kind == "path":
        if type(value) is not str or not value:
            raise TypeError(f"{location} must have repository-relative path string type")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{location} must be a repository-relative logical path")
    elif kind == "sha256":
        if type(value) is not str or _HEX_64.fullmatch(value) is None:
            raise ValueError(f"{location} must be a lowercase SHA-256 string")
    elif kind == "git_commit":
        if type(value) is not str or _GIT_COMMIT.fullmatch(value) is None:
            raise ValueError(f"{location} must be a lowercase Git commit digest")
    elif kind == "bool":
        if type(value) is not bool:
            raise TypeError(f"{location} must have boolean type")
    elif kind == "int":
        if type(value) is not int:
            raise TypeError(f"{location} must have integer type")
    elif kind == "number":
        if type(value) not in (int, float):
            raise TypeError(f"{location} must have numeric type")
        if not math.isfinite(value):
            raise ValueError(f"{location} must be finite")
    elif kind == "bounded_steps":
        if value is not None and (type(value) is not int or value != 1):
            raise ValueError(f"{location} bounded optimizer steps must be null or integer 1")
    else:  # pragma: no cover - internal schema construction error
        raise RuntimeError(f"unknown manifest schema descriptor: {kind}")


def _validate_schema(value: object, schema: object, location: str) -> None:
    if isinstance(schema, str):
        _validate_scalar(value, schema, location)
        return
    if isinstance(schema, tuple):
        _, item_schema = schema
        if type(value) is not list:
            raise TypeError(f"{location} must have array type")
        for index, item in enumerate(value):
            _validate_schema(item, item_schema, f"{location}/{index}")
        return
    if not isinstance(schema, dict):  # pragma: no cover - internal schema construction error
        raise RuntimeError(f"invalid manifest schema at {location}")
    if type(value) is not dict:
        raise TypeError(f"{location} must have object type")
    required = set(schema)
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing:
        raise ValueError(f"{location} missing required keys: {missing}")
    if extra:
        raise ValueError(f"{location} has extra keys: {extra}")
    for key, child_schema in schema.items():
        _validate_schema(value[key], child_schema, f"{location}/{key}")


def validate_resolved_manifest(payload: Mapping[str, object] | str | bytes) -> dict[str, object]:
    """Decode and validate one complete Stage 1 resolved manifest."""
    decoded = _decode_manifest(payload)
    _validate_json_value(decoded)
    _validate_schema(decoded, _MANIFEST_SCHEMA, "$")
    batch = decoded["batch"]
    assert isinstance(batch, dict)
    microbatch = batch["per_device_train_batch_size"]
    accumulation = batch["gradient_accumulation_steps"]
    expected_groupings = {
        1: [[0, 0], [1, 0], [2, 0], [3, 0]],
        2: [[0, 0], [0, 1], [1, 0], [1, 1]],
    }
    if microbatch not in expected_groupings:
        raise ValueError("manifest batch has unsupported microbatch grouping arm")
    if accumulation != 4 // microbatch:
        raise ValueError("manifest batch has invalid gradient accumulation for grouping arm")
    if batch["rank_local_microbatch_grouping"] != expected_groupings[microbatch]:
        raise ValueError("manifest batch has invalid rank-local microbatch grouping")
    initialization = decoded["initialization"]
    assert isinstance(initialization, dict)
    for collection_name in ("trainable_parameters", "ema_parameters"):
        identities = initialization[collection_name]
        assert isinstance(identities, list)
        names = [identity["name"] for identity in identities]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError(
                f"manifest initialization {collection_name} must use unique sorted names"
            )
    return decoded


def _replace_existing_json_pointer(
    payload: dict[str, object], pointer: str, replacement: object
) -> None:
    parts = pointer.removeprefix("/").split("/")
    target: object = payload
    for part in parts[:-1]:
        if type(target) is not dict or part not in target:
            raise ValueError(f"manifest profile pointer is absent: {pointer}")
        target = target[part]
    if type(target) is not dict or parts[-1] not in target:
        raise ValueError(f"manifest profile pointer is absent: {pointer}")
    target[parts[-1]] = replacement


def normalized_manifest_bytes(
    payload: Mapping[str, object] | str | bytes, profile: str
) -> bytes:
    if profile not in MANIFEST_PROFILES:
        raise ValueError(f"unknown manifest profile: {profile}")
    normalized = copy.deepcopy(validate_resolved_manifest(payload))
    for pointer in MANIFEST_PROFILES[profile]:
        _replace_existing_json_pointer(normalized, pointer, "<approved-difference>")
    return canonical_json_bytes(normalized)


@dataclass(frozen=True)
class Stage1ResolvedManifest:
    """An immutable canonical representation of a validated resolved manifest."""

    _canonical_bytes: bytes

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, object] | str | bytes
    ) -> Stage1ResolvedManifest:
        return cls(canonical_json_bytes(validate_resolved_manifest(payload)))

    def to_mapping(self) -> dict[str, object]:
        decoded = json.loads(self._canonical_bytes)
        assert isinstance(decoded, dict)
        return decoded

    @property
    def digest(self) -> str:
        return hashlib.sha256(self._canonical_bytes).hexdigest()

    def normalized_digest(self, profile: str) -> str:
        return hashlib.sha256(
            normalized_manifest_bytes(self.to_mapping(), profile)
        ).hexdigest()


def _repository_path(repository: Path, path: str | Path) -> tuple[Path, str]:
    repository = repository.absolute()
    candidate = Path(path)
    candidate = candidate.absolute() if candidate.is_absolute() else (repository / candidate).absolute()
    try:
        logical = candidate.relative_to(repository).as_posix()
    except ValueError as error:
        raise ValueError(f"runtime input must be repository-relative: {path}") from error
    return candidate, logical


def _file_tree_sha256(path: Path) -> str:
    if path.is_file():
        files = [(path.name, path)]
    elif path.is_dir():
        files = sorted(
            (candidate.relative_to(path).as_posix(), candidate)
            for candidate in path.rglob("*")
            if candidate.is_file()
        )
    else:
        raise ValueError(f"identity path is not a file or directory: {path}")
    if not files:
        raise ValueError(f"identity path contains no files: {path}")
    digest = hashlib.sha256()
    for identity, candidate in files:
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
    return digest.hexdigest()


def _referenced_files_sha256(
    repository: Path, paths: Sequence[str | Path]
) -> str:
    if not paths:
        raise ValueError("referenced image paths must not be empty")
    identities: list[tuple[str, Path]] = []
    for path in paths:
        candidate, logical = _repository_path(repository, path)
        if not candidate.is_file():
            raise ValueError(f"referenced image is not a file: {path}")
        identities.append((logical, candidate))
    if len({logical for logical, _ in identities}) != len(identities):
        raise ValueError("referenced image paths must be unique")
    digest = hashlib.sha256()
    for logical, candidate in sorted(identities):
        digest.update(logical.encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
    return digest.hexdigest()


def _fixed_runtime_source_files(
    repository: Path, *, recorded_git_commit: str, qwen_source_path: str | Path
) -> list[Path]:
    repository = repository.absolute()
    git_root = Path(
        _git_output("-C", str(repository), "rev-parse", "--show-toplevel")
    ).absolute()
    if git_root != repository:
        raise ValueError("repository_root must be the Git worktree root")
    head = _git_output("-C", str(repository), "rev-parse", "HEAD")
    if head != recorded_git_commit:
        raise ValueError(
            f"recorded git_commit does not equal repository HEAD: "
            f"recorded={recorded_git_commit}, HEAD={head}"
        )
    qwen_path, qwen_logical = _repository_path(repository, qwen_source_path)
    fixed_entries = {
        "eval.py",
        "run_stage1_training.sh",
        "run_multinode_evaluation.sh",
    }
    tree_arguments = [
        "-C",
        str(repository),
        "ls-tree",
        "-r",
        "--name-only",
        recorded_git_commit,
        "--",
        "src",
        *sorted(fixed_entries),
        qwen_logical,
    ]
    recorded_candidates = set(_git_output(*tree_arguments).splitlines())

    def belongs_to_closure(logical: str) -> bool:
        if logical in fixed_entries:
            return True
        if logical.startswith("src/") and Path(logical).suffix in {".py", ".sh"}:
            return True
        if qwen_path.is_file():
            return logical == qwen_logical
        return logical.startswith(f"{qwen_logical.rstrip('/')}/") and Path(logical).suffix in {
            ".py",
            ".sh",
        }

    recorded = {logical for logical in recorded_candidates if belongs_to_closure(logical)}
    if not fixed_entries <= recorded:
        missing = sorted(fixed_entries - recorded)
        raise ValueError(f"recorded commit lacks fixed runtime source files: {missing}")
    current: set[str] = set()
    source_root = repository / "src"
    if source_root.is_dir():
        current.update(
            candidate.relative_to(repository).as_posix()
            for candidate in source_root.rglob("*")
            if candidate.is_file() and candidate.suffix in {".py", ".sh"}
        )
    for logical in fixed_entries:
        if (repository / logical).is_file():
            current.add(logical)
    if qwen_path.is_file():
        current.add(qwen_logical)
    elif qwen_path.is_dir():
        current.update(
            candidate.relative_to(repository).as_posix()
            for candidate in qwen_path.rglob("*")
            if candidate.is_file() and candidate.suffix in {".py", ".sh"}
        )
    if current != recorded:
        missing = sorted(recorded - current)
        extra = sorted(current - recorded)
        raise ValueError(
            f"fixed runtime source closure mismatch: missing={missing}, extra={extra}"
        )
    dirty = _git_output(
        "-C",
        str(repository),
        "diff",
        "--name-only",
        recorded_git_commit,
        "--",
        *sorted(recorded),
    )
    if dirty:
        raise ValueError(f"dirty or deleted tracked runtime source: {dirty.splitlines()[0]}")
    return [repository / logical for logical in sorted(recorded)]


def _resolved_software_versions() -> dict[str, str]:
    def distribution_version(*names: str) -> str:
        for name in names:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                continue
        return "unavailable"

    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "pytorch": str(torch.__version__),
        "transformers": distribution_version("transformers"),
        "peft": distribution_version("peft"),
        "deepspeed": distribution_version("deepspeed"),
        "cuda": str(torch.version.cuda or "unavailable"),
        "flash_attention": distribution_version("flash_attn", "flash-attn"),
    }


def _load_resolved_deepspeed_settings(
    path: Path,
    *,
    world_size: int,
    microbatch: int,
    gradient_accumulation_steps: int,
) -> dict[str, object]:
    try:
        source = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid DeepSpeed source JSON: {path}") from error
    required = {
        "train_micro_batch_size_per_gpu",
        "train_batch_size",
        "gradient_accumulation_steps",
        "bf16",
        "zero_optimization",
        "gradient_clipping",
        "steps_per_print",
        "wall_clock_breakdown",
    }
    if type(source) is not dict or set(source) != required:
        raise ValueError("DeepSpeed source must use the exact resolved settings schema")
    expected_batch = {
        "train_micro_batch_size_per_gpu": microbatch,
        "train_batch_size": world_size * microbatch * gradient_accumulation_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    for key, expected in expected_batch.items():
        actual = source[key]
        if actual != "auto" and (type(actual) is not int or actual != expected):
            raise ValueError(f"DeepSpeed {key} does not resolve to the manifest batch")
    return {
        "bf16": copy.deepcopy(source["bf16"]),
        "zero_optimization": copy.deepcopy(source["zero_optimization"]),
        "gradient_clipping": source["gradient_clipping"],
        "steps_per_print": source["steps_per_print"],
        "wall_clock_breakdown": source["wall_clock_breakdown"],
    }


def _tensor_identities(parameters: Mapping[str, torch.Tensor]) -> list[dict[str, object]]:
    if not isinstance(parameters, Mapping):
        raise TypeError("parameter identities require a name-to-tensor mapping")
    identities = []
    for name in sorted(parameters):
        tensor = parameters[name]
        if type(name) is not str or not name:
            raise TypeError("parameter identity names must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"parameter identity is not a tensor: {name}")
        identities.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": canonical_tensor_sha256(tensor),
            }
        )
    return identities


def materialize_stage1_resolved_manifest(
    *,
    semantic_values: Mapping[str, object],
    repository_root: str | Path,
    git_commit: str,
    qwen_source_path: str | Path,
    model_path: str | Path,
    processor_path: str | Path,
    model_config_path: str | Path,
    model_revision: str,
    processor_revision: str,
    train_jsonl_path: str | Path,
    test_jsonl_path: str | Path,
    image_root: str | Path,
    referenced_image_paths: Sequence[str | Path],
    deepspeed_config_path: str | Path,
    world_size: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    trainable_parameters: Mapping[str, torch.Tensor],
    ema_parameters: Mapping[str, torch.Tensor],
) -> Stage1ResolvedManifest:
    """Derive immutable identities while validating caller-resolved semantics.

    ``semantic_values`` owns non-identity hyperparameters and artifact destinations.
    This boundary always replaces identity, batch, DeepSpeed, and initialization data
    with values derived from the concrete runtime inputs supplied separately.
    """
    repository = Path(repository_root).absolute()
    runtime_sources = _fixed_runtime_source_files(
        repository,
        recorded_git_commit=git_commit,
        qwen_source_path=qwen_source_path,
    )
    model, model_logical = _repository_path(repository, model_path)
    processor, processor_logical = _repository_path(repository, processor_path)
    config, _ = _repository_path(repository, model_config_path)
    train_jsonl, train_logical = _repository_path(repository, train_jsonl_path)
    test_jsonl, test_logical = _repository_path(repository, test_jsonl_path)
    images, images_logical = _repository_path(repository, image_root)
    deepspeed, _ = _repository_path(repository, deepspeed_config_path)
    for name, path in (
        ("model config", config),
        ("train JSONL", train_jsonl),
        ("test JSONL", test_jsonl),
        ("DeepSpeed config", deepspeed),
    ):
        if not path.is_file():
            raise ValueError(f"{name} is not a file: {path}")
    if not images.is_dir():
        raise ValueError(f"image root is not a directory: {images}")
    if type(world_size) is not int or world_size != 4:
        raise ValueError("manifest materializer requires world_size=4")
    settings = _load_resolved_deepspeed_settings(
        deepspeed,
        world_size=world_size,
        microbatch=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    groupings = {
        1: [[0, 0], [1, 0], [2, 0], [3, 0]],
        2: [[0, 0], [0, 1], [1, 0], [1, 1]],
    }
    if per_device_train_batch_size not in groupings:
        raise ValueError("manifest materializer requires microbatch 1 or 2")
    if gradient_accumulation_steps != 4 // per_device_train_batch_size:
        raise ValueError("manifest materializer requires B * GA = 4")
    payload = _decode_manifest(semantic_values)
    payload["schema_version"] = "stage1-resolved-manifest-v1"
    payload["model"] = {
        "model_path": model_logical,
        "processor_path": processor_logical,
        "model_revision": model_revision,
        "processor_revision": processor_revision,
        "model_files_sha256": _file_tree_sha256(model),
        "processor_files_sha256": _file_tree_sha256(processor),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    payload["code"] = {
        "git_commit": git_commit,
        "runtime_source_digest": runtime_source_sha256(runtime_sources),
        "software_versions": _resolved_software_versions(),
    }
    data = payload.get("data")
    if type(data) is not dict:
        raise TypeError("semantic_values/data must be an object")
    data.update(
        {
            "train_jsonl_path": train_logical,
            "train_sha256": hashlib.sha256(train_jsonl.read_bytes()).hexdigest(),
            "test_jsonl_path": test_logical,
            "test_sha256": hashlib.sha256(test_jsonl.read_bytes()).hexdigest(),
            "image_root": images_logical,
            "image_files_sha256": _referenced_files_sha256(
                repository, referenced_image_paths
            ),
        }
    )
    payload["batch"] = {
        "world_size": world_size,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_global_batch_size": (
            world_size * per_device_train_batch_size * gradient_accumulation_steps
        ),
        "rank_local_microbatch_grouping": groupings[per_device_train_batch_size],
    }
    payload["deepspeed"] = {
        "source_digest": hashlib.sha256(deepspeed.read_bytes()).hexdigest(),
        "settings": settings,
    }
    evaluation = payload.get("evaluation")
    if type(evaluation) is not dict:
        raise TypeError("semantic_values/evaluation must be an object")
    evaluation["dataset_path"] = test_logical
    evaluation["dataset_sha256"] = data["test_sha256"]
    payload["initialization"] = {
        "trainable_parameters": _tensor_identities(trainable_parameters),
        "ema_parameters": _tensor_identities(ema_parameters),
    }
    return Stage1ResolvedManifest.from_mapping(payload)


@dataclass(frozen=True)
class Stage1ManifestComparison:
    profile: Literal["cross_arm_v1", "same_arm_v1"]
    equal: bool
    left_digest: str
    right_digest: str
    left_normalized_digest: str
    right_normalized_digest: str


def compare_stage1_manifests(
    left: Stage1ResolvedManifest | Mapping[str, object] | str | bytes,
    right: Stage1ResolvedManifest | Mapping[str, object] | str | bytes,
    *,
    profile: Literal["cross_arm_v1", "same_arm_v1"],
) -> Stage1ManifestComparison:
    """Require two manifests to differ only at one profile's approved pointers."""
    if profile not in MANIFEST_PROFILES:
        raise ValueError(f"unknown manifest profile: {profile}")
    left_manifest = (
        left if isinstance(left, Stage1ResolvedManifest) else Stage1ResolvedManifest.from_mapping(left)
    )
    right_manifest = (
        right if isinstance(right, Stage1ResolvedManifest) else Stage1ResolvedManifest.from_mapping(right)
    )
    left_normalized = normalized_manifest_bytes(left_manifest.to_mapping(), profile)
    right_normalized = normalized_manifest_bytes(right_manifest.to_mapping(), profile)
    left_normalized_digest = hashlib.sha256(left_normalized).hexdigest()
    right_normalized_digest = hashlib.sha256(right_normalized).hexdigest()
    if left_normalized != right_normalized:
        raise ValueError(f"Stage 1 manifests contain non-whitelisted differences for {profile}")
    return Stage1ManifestComparison(
        profile=profile,
        equal=True,
        left_digest=left_manifest.digest,
        right_digest=right_manifest.digest,
        left_normalized_digest=left_normalized_digest,
        right_normalized_digest=right_normalized_digest,
    )
