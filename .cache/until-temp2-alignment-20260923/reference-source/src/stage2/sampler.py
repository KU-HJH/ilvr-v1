"""Distributed Stage 2 sampling with aligned latent decoder work."""

import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Sampler

from step_preference.negatives import StepNegativeIndex


STAGE2_SAMPLER_FILENAME = "stage2-sampler.json"
STAGE2_SAMPLER_SCHEMA_VERSION = 1
_PREFERENCE_SIGNATURE_BUCKET_SIZES_FIELD = (
    "preference_signature_bucket_sizes"
)
_STAGE2_SAMPLER_IDENTITY_FIELDS = frozenset(
    {
        "schema_version",
        "seed",
        "world_size",
        "dataset_size",
        "bucket_sizes",
        "padded_samples",
    }
)
_STAGE2_PREFERENCE_SAMPLER_IDENTITY_FIELDS = (
    _STAGE2_SAMPLER_IDENTITY_FIELDS
    | {_PREFERENCE_SIGNATURE_BUCKET_SIZES_FIELD}
)
_STAGE2_SAMPLER_SIDECAR_FIELD_SETS = (
    _STAGE2_SAMPLER_IDENTITY_FIELDS | {"global_step"},
    _STAGE2_PREFERENCE_SAMPLER_IDENTITY_FIELDS | {"global_step"},
)
_MAX_STAGE2_SAMPLER_BYTES = 256 * 1024


@dataclass(frozen=True)
class Stage2SamplerIdentity:
    schema_version: int
    seed: int
    world_size: int
    dataset_size: int
    bucket_sizes: tuple[tuple[int, int], ...]
    padded_samples: int
    preference_signature_bucket_sizes: tuple[tuple[str, int], ...] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "world_size": self.world_size,
            "dataset_size": self.dataset_size,
            "bucket_sizes": {
                str(key): value for key, value in self.bucket_sizes
            },
            "padded_samples": self.padded_samples,
        }
        if self.preference_signature_bucket_sizes is not None:
            payload[_PREFERENCE_SIGNATURE_BUCKET_SIZES_FIELD] = {
                key: value
                for key, value in self.preference_signature_bucket_sizes
            }
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> "Stage2SamplerIdentity":
        if not isinstance(payload, dict) or set(payload) not in (
            _STAGE2_SAMPLER_IDENTITY_FIELDS,
            _STAGE2_PREFERENCE_SAMPLER_IDENTITY_FIELDS,
        ):
            raise ValueError("invalid Stage 2 sampler identity schema")
        schema_version = payload["schema_version"]
        seed = payload["seed"]
        world_size = payload["world_size"]
        dataset_size = payload["dataset_size"]
        bucket_sizes = payload["bucket_sizes"]
        padded_samples = payload["padded_samples"]
        preference_bucket_sizes = payload.get(
            _PREFERENCE_SIGNATURE_BUCKET_SIZES_FIELD
        )
        has_preference_bucket_sizes = (
            _PREFERENCE_SIGNATURE_BUCKET_SIZES_FIELD in payload
        )
        if (
            type(schema_version) is not int
            or schema_version != STAGE2_SAMPLER_SCHEMA_VERSION
        ):
            raise ValueError("unsupported Stage 2 sampler identity schema")
        if type(seed) is not int:
            raise ValueError("invalid Stage 2 sampler identity seed")
        if type(world_size) is not int or world_size < 2:
            raise ValueError("invalid Stage 2 sampler identity world_size")
        if type(dataset_size) is not int or dataset_size < 1:
            raise ValueError("invalid Stage 2 sampler identity dataset_size")
        if type(padded_samples) is not int or padded_samples < 0:
            raise ValueError("invalid Stage 2 sampler identity padded_samples")
        if not isinstance(bucket_sizes, dict) or not bucket_sizes:
            raise ValueError("invalid Stage 2 sampler identity bucket_sizes")
        restored_buckets = []
        for raw_key, size in bucket_sizes.items():
            if (
                type(raw_key) is not str
                or not raw_key.isdecimal()
                or str(int(raw_key)) != raw_key
                or type(size) is not int
                or size < 1
            ):
                raise ValueError("invalid Stage 2 sampler identity bucket_sizes")
            key = int(raw_key)
            if key < 1 or key > 6:
                raise ValueError("invalid Stage 2 sampler identity bucket_sizes")
            restored_buckets.append((key, size))
        restored_buckets.sort()
        if sum(size for _, size in restored_buckets) != dataset_size:
            raise ValueError(
                "Stage 2 sampler bucket_sizes do not match dataset_size"
            )
        restored_preference_buckets = None
        if has_preference_bucket_sizes:
            if not isinstance(preference_bucket_sizes, dict):
                raise ValueError(
                    "invalid Stage 2 sampler preference signature bucket_sizes"
                )
            restored_preference_buckets = []
            aggregate_sizes: Counter[int] = Counter()
            for raw_key, size in preference_bucket_sizes.items():
                block_count, _signature = _decode_preference_bucket_key(
                    raw_key
                )
                if type(size) is not int or size < 1:
                    raise ValueError(
                        "invalid Stage 2 sampler preference signature bucket_sizes"
                    )
                restored_preference_buckets.append((raw_key, size))
                aggregate_sizes[block_count] += size
            restored_preference_buckets.sort()
            if tuple(sorted(aggregate_sizes.items())) != tuple(
                restored_buckets
            ):
                raise ValueError(
                    "preference signature bucket_sizes do not match bucket_sizes"
                )
            expected_padding = sum(
                (-size) % world_size
                for _, size in restored_preference_buckets
            )
        else:
            expected_padding = sum(
                (-size) % world_size for _, size in restored_buckets
            )
        if padded_samples != expected_padding:
            raise ValueError(
                "Stage 2 sampler padded_samples do not match bucket_sizes"
            )
        return cls(
            schema_version=schema_version,
            seed=seed,
            world_size=world_size,
            dataset_size=dataset_size,
            bucket_sizes=tuple(restored_buckets),
            padded_samples=padded_samples,
            preference_signature_bucket_sizes=(
                None
                if restored_preference_buckets is None
                else tuple(restored_preference_buckets)
            ),
        )


def _encode_preference_bucket_key(
    block_count: int,
    signature: tuple[tuple[bool, bool], ...],
) -> str:
    bits = "".join(
        f"{int(visual_valid)}{int(text_valid)}"
        for visual_valid, text_valid in signature
    )
    return f"{block_count}:{bits}"


def _decode_preference_bucket_key(
    raw_key: object,
) -> tuple[int, tuple[tuple[bool, bool], ...]]:
    if not isinstance(raw_key, str) or raw_key.count(":") != 1:
        raise ValueError(
            "invalid Stage 2 sampler preference signature bucket_sizes"
        )
    raw_count, bits = raw_key.split(":")
    if (
        not raw_count.isdecimal()
        or str(int(raw_count)) != raw_count
        or int(raw_count) < 1
        or int(raw_count) > 6
        or len(bits) != 2 * int(raw_count)
        or set(bits) - {"0", "1"}
    ):
        raise ValueError(
            "invalid Stage 2 sampler preference signature bucket_sizes"
        )
    signature = tuple(
        (bits[offset] == "1", bits[offset + 1] == "1")
        for offset in range(0, len(bits), 2)
    )
    return int(raw_count), signature


def _preference_forward_signature(
    item,
    *,
    block_count: int,
    step_negative_index: StepNegativeIndex,
) -> tuple[tuple[bool, bool], ...]:
    sample_id = getattr(item, "sample_id", None)
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(
            "Stage 2B sampler dataset items must expose a non-empty sample_id"
        )
    signature = []
    for step_id in range(block_count):
        negative = step_negative_index.get(sample_id, step_id)
        signature.append(
            (
                negative.valid_visual_negative,
                negative.valid_textual_negative,
            )
        )
    return tuple(signature)


def _bucket_indices(
    dataset,
    step_negative_index: StepNegativeIndex | None = None,
    preference_signatures: tuple[tuple[tuple[bool, bool], ...], ...] | None = None,
    observed_lengths: list[int | None] | None = None,
) -> dict[object, list[int]]:
    if len(dataset) == 0:
        raise ValueError("balanced sampler cannot sample an empty dataset")
    if step_negative_index is not None and not isinstance(
        step_negative_index, StepNegativeIndex
    ):
        raise TypeError("step_negative_index must be a StepNegativeIndex")
    if step_negative_index is not None and preference_signatures is not None:
        raise ValueError("preference signatures and negative index are mutually exclusive")
    if preference_signatures is not None and len(preference_signatures) != len(dataset):
        raise ValueError("preference signatures must match the dataset length")
    buckets: dict[object, list[int]] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        if observed_lengths is not None:
            observed_lengths.append(getattr(item, "sequence_length", None))
        try:
            block_count = item.latent_block_count
        except AttributeError as error:
            raise ValueError(
                f"dataset item {index} must expose integer latent_block_count"
            ) from error
        if type(block_count) is not int:
            raise ValueError(
                f"dataset item {index} latent_block_count must be an integer"
            )
        if block_count < 1 or block_count > 6:
            raise ValueError(
                f"dataset item {index} latent_block_count must be between 1 and 6"
            )
        if preference_signatures is not None:
            signature = preference_signatures[index]
            if (
                len(signature) != block_count
                or any(
                    not isinstance(pair, tuple)
                    or len(pair) != 2
                    or any(type(value) is not bool for value in pair)
                    for pair in signature
                )
            ):
                raise ValueError(
                    f"dataset item {index} has an invalid preference forward signature"
                )
            bucket_key = (block_count, signature)
        elif step_negative_index is None:
            bucket_key = block_count
        else:
            bucket_key = (
                block_count,
                _preference_forward_signature(
                    item,
                    block_count=block_count,
                    step_negative_index=step_negative_index,
                ),
            )
        buckets.setdefault(bucket_key, []).append(index)
    return buckets


class LatentBlockBalancedSampler(Sampler[int]):
    """Yield groups with aligned latent and optional preference forward work."""

    def __init__(
        self,
        dataset,
        *,
        num_replicas: int,
        seed: int,
        step_negative_index: StepNegativeIndex | None = None,
        preference_signatures: tuple[tuple[tuple[bool, bool], ...], ...] | None = None,
        sequence_lengths: tuple[int, ...] | None = None,
    ):
        if type(num_replicas) is not int or num_replicas < 2:
            raise ValueError("balanced sampler requires at least two replicas")
        if type(seed) is not int:
            raise ValueError("balanced sampler seed must be an integer")
        if sequence_lengths is not None and (
            len(sequence_lengths) != len(dataset)
            or any(type(length) is not int or length < 1 for length in sequence_lengths)
        ):
            raise ValueError(
                "sequence lengths must be positive integers matching the dataset length"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.seed = seed
        self.epoch = 0
        self.step_negative_index = step_negative_index
        self.preference_signatures = preference_signatures
        observed_lengths: list[int | None] = []
        self._buckets = _bucket_indices(
            dataset,
            step_negative_index,
            preference_signatures,
            observed_lengths,
        )
        self.sequence_lengths = (
            sequence_lengths
            if sequence_lengths is not None
            else (
                tuple(observed_lengths)
                if observed_lengths
                and all(type(length) is int for length in observed_lengths)
                else None
            )
        )
        padded = sum(
            (-len(indices)) % num_replicas
            for indices in self._buckets.values()
        )
        latent_bucket_sizes: Counter[int] = Counter()
        for key, indices in self._buckets.items():
            block_count = key if isinstance(key, int) else key[0]
            latent_bucket_sizes[block_count] += len(indices)
        preference_bucket_sizes = (
            None
            if step_negative_index is None and preference_signatures is None
            else tuple(
                (
                    _encode_preference_bucket_key(block_count, signature),
                    len(indices),
                )
                for (block_count, signature), indices in sorted(
                    self._buckets.items()
                )
            )
        )
        self.identity = Stage2SamplerIdentity(
            schema_version=STAGE2_SAMPLER_SCHEMA_VERSION,
            seed=seed,
            world_size=num_replicas,
            dataset_size=len(dataset),
            bucket_sizes=tuple(
                sorted(latent_bucket_sizes.items())
            ),
            padded_samples=padded,
            preference_signature_bucket_sizes=preference_bucket_sizes,
        )

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        groups = []
        for original in (
            indices for _, indices in sorted(self._buckets.items())
        ):
            order = torch.randperm(
                len(original), generator=generator
            ).tolist()
            indices = [original[position] for position in order]
            if self.sequence_lengths is not None:
                indices.sort(key=self.sequence_lengths.__getitem__)
            needed = (-len(indices)) % self.num_replicas
            base = tuple(indices)
            if self.sequence_lengths is None:
                indices.extend(
                    base[position % len(base)] for position in range(needed)
                )
            else:
                indices.extend(base[-1] for _ in range(needed))
            groups.extend(
                indices[offset : offset + self.num_replicas]
                for offset in range(0, len(indices), self.num_replicas)
            )
        group_order = torch.randperm(
            len(groups), generator=generator
        ).tolist()
        return iter(
            index for position in group_order for index in groups[position]
        )

    def __len__(self) -> int:
        return self.identity.dataset_size + self.identity.padded_samples

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch


def _sampler_sidecar_payload(
    identity: Stage2SamplerIdentity, *, global_step: int
) -> dict[str, object]:
    if not isinstance(identity, Stage2SamplerIdentity):
        raise TypeError("identity must be a Stage2SamplerIdentity")
    if type(global_step) is not int or global_step < 0:
        raise ValueError("global_step must be non-negative")
    return {**identity.to_dict(), "global_step": global_step}


def _read_stage2_sampler_payload(directory: Path) -> dict[str, object]:
    path = Path(directory) / STAGE2_SAMPLER_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing Stage 2 sampler identity: {path}")
    try:
        if path.stat().st_size > _MAX_STAGE2_SAMPLER_BYTES:
            raise ValueError(f"invalid Stage 2 sampler identity: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Stage 2 sampler identity: {path}") from error
    if not isinstance(payload, dict) or set(payload) not in (
        _STAGE2_SAMPLER_SIDECAR_FIELD_SETS
    ):
        raise ValueError(f"invalid Stage 2 sampler identity schema: {path}")
    global_step = payload["global_step"]
    if type(global_step) is not int or global_step < 0:
        raise ValueError("invalid Stage 2 sampler identity global_step")
    Stage2SamplerIdentity.from_dict(
        {key: value for key, value in payload.items() if key != "global_step"}
    )
    return payload


def read_stage2_sampler_sidecar(directory: Path) -> Stage2SamplerIdentity:
    """Strictly read an immutable Stage 2 sampler identity sidecar."""

    payload = _read_stage2_sampler_payload(Path(directory))
    return Stage2SamplerIdentity.from_dict(
        {key: value for key, value in payload.items() if key != "global_step"}
    )


def write_stage2_sampler_sidecar(
    directory: Path,
    identity: Stage2SamplerIdentity,
    *,
    global_step: int,
) -> Path:
    """Atomically publish one immutable sampler identity in ``directory``."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / STAGE2_SAMPLER_FILENAME
    payload = _sampler_sidecar_payload(identity, global_step=global_step)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    if path.exists() or path.is_symlink():
        if _read_stage2_sampler_payload(destination) == payload:
            return path
        raise FileExistsError(
            f"refusing to replace different Stage 2 sampler identity: {path}"
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{STAGE2_SAMPLER_FILENAME}.",
            suffix=".tmp",
            dir=destination,
            delete=False,
        ) as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if _read_stage2_sampler_payload(destination) != payload:
                raise FileExistsError(
                    "refusing to replace different Stage 2 sampler identity: "
                    f"{path}"
                ) from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def validate_stage2_sampler_resume_identity(
    checkpoint: Path, expected: Stage2SamplerIdentity
) -> None:
    """Reject resume when distributed sample ordering cannot be reconstructed."""

    if not isinstance(expected, Stage2SamplerIdentity):
        raise TypeError("expected must be a Stage2SamplerIdentity")
    actual = read_stage2_sampler_sidecar(Path(checkpoint))
    for field in (
        "seed",
        "world_size",
        "dataset_size",
        "bucket_sizes",
        "padded_samples",
        "preference_signature_bucket_sizes",
    ):
        checkpoint_value = getattr(actual, field)
        current_value = getattr(expected, field)
        if checkpoint_value != current_value:
            raise ValueError(
                f"Stage 2 sampler resume identity mismatch for {field}: "
                f"checkpoint={checkpoint_value!r}, current={current_value!r}"
            )
