"""Distributed Stage 2 sampling with aligned latent decoder work."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler
from accelerate.data_loader import SeedableRandomSampler


STAGE2_SAMPLER_FILENAME = "stage2-sampler.json"
STAGE2_SAMPLER_SCHEMA_VERSION = 1
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
_STAGE2_SAMPLER_SIDECAR_FIELDS = (
    _STAGE2_SAMPLER_IDENTITY_FIELDS | {"global_step"}
)
_MAX_STAGE2_SAMPLER_BYTES = 16 * 1024


@dataclass(frozen=True)
class Stage2SamplerIdentity:
    schema_version: int
    seed: int
    world_size: int
    dataset_size: int
    bucket_sizes: tuple[tuple[int, int], ...]
    padded_samples: int
    policy: str = "latent_balanced"
    gradient_accumulation_steps: int | None = None

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
        if self.policy == "reference_random":
            payload.update(policy=self.policy, gradient_accumulation_steps=self.gradient_accumulation_steps,
                           update_accounting="ceil", padding_weight=0)
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> "Stage2SamplerIdentity":
        reference = isinstance(payload, dict) and payload.get("schema_version") == 2
        fields = _STAGE2_SAMPLER_IDENTITY_FIELDS | (
            {"policy", "gradient_accumulation_steps", "update_accounting", "padding_weight"} if reference else set()
        )
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("invalid Stage 2 sampler identity schema")
        schema_version = payload["schema_version"]
        seed = payload["seed"]
        world_size = payload["world_size"]
        dataset_size = payload["dataset_size"]
        bucket_sizes = payload["bucket_sizes"]
        padded_samples = payload["padded_samples"]
        if (
            type(schema_version) is not int
            or schema_version != (2 if reference else STAGE2_SAMPLER_SCHEMA_VERSION)
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
        if reference:
            gas = payload["gradient_accumulation_steps"]
            if (payload["policy"] != "reference_random" or payload["update_accounting"] != "ceil"
                    or type(payload["padding_weight"]) is not int or payload["padding_weight"] != 0
                    or type(gas) is not int or gas < 1 or world_size != 4
                    or bucket_sizes != {} or padded_samples != (-dataset_size) % world_size):
                raise ValueError("invalid reference_random sampler geometry or policy")
            return cls(schema_version, seed, world_size, dataset_size, (), padded_samples,
                       "reference_random", gas)
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
        )


def _bucket_indices(dataset) -> dict[int, list[int]]:
    if len(dataset) == 0:
        raise ValueError("balanced sampler cannot sample an empty dataset")
    buckets: dict[int, list[int]] = {}
    for index in range(len(dataset)):
        item = dataset[index]
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
        buckets.setdefault(block_count, []).append(index)
    return buckets


class LatentBlockBalancedSampler(Sampler[int]):
    """Yield a global stream whose replica-sized groups share one block count."""

    def __init__(self, dataset, *, num_replicas: int, seed: int):
        if type(num_replicas) is not int or num_replicas < 2:
            raise ValueError("balanced sampler requires at least two replicas")
        if type(seed) is not int:
            raise ValueError("balanced sampler seed must be an integer")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.seed = seed
        self.epoch = 0
        self._buckets = _bucket_indices(dataset)
        padded = sum(
            (-len(indices)) % num_replicas
            for indices in self._buckets.values()
        )
        self.identity = Stage2SamplerIdentity(
            schema_version=STAGE2_SAMPLER_SCHEMA_VERSION,
            seed=seed,
            world_size=num_replicas,
            dataset_size=len(dataset),
            bucket_sizes=tuple(
                (key, len(value))
                for key, value in sorted(self._buckets.items())
            ),
            padded_samples=padded,
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
            needed = (-len(indices)) % self.num_replicas
            base = tuple(indices)
            indices.extend(
                base[position % len(base)] for position in range(needed)
            )
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



@dataclass(frozen=True)
class ReferenceSample:
    example: object
    index: int
    weight: float


class ReferenceRandomDataset(Dataset):
    """Keep real indices intact; -1 is a valid-forward, zero-weight slot."""

    def __init__(self, dataset):
        if not len(dataset):
            raise ValueError("reference_random cannot sample an empty dataset")
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index == -1:
            return ReferenceSample(self.dataset[0], -1, 0.)
        if index < 0 or index >= len(self.dataset):
            raise IndexError(index)
        return ReferenceSample(self.dataset[index], index, 1.)


class ReferenceRandomSampler(Sampler[int]):
    """Source permutation followed only by explicit zero-contribution slots.

    Accelerate shards this global stream with even_batches=False. Each rank has
    the same number of microsteps; no bucket grouping or repeated real exposure.
    """

    def __init__(self, dataset, *, num_replicas, seed, gradient_accumulation_steps):
        if num_replicas != 4 or type(seed) is not int:
            raise ValueError("reference_random requires four ranks and an integer seed")
        if type(gradient_accumulation_steps) is not int or gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not len(dataset):
            raise ValueError("reference_random cannot sample an empty dataset")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.seed = seed
        self.epoch = 0
        self.identity = Stage2SamplerIdentity(2, seed, num_replicas, len(dataset), (),
            (-len(dataset)) % num_replicas, "reference_random", gradient_accumulation_steps)

    def __len__(self):
        return self.identity.dataset_size + self.identity.padded_samples

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __iter__(self):
        source = SeedableRandomSampler(self.dataset, data_seed=self.seed)
        source.set_epoch(self.epoch)
        yield from source
        yield from [-1] * self.identity.padded_samples


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
    if not isinstance(payload, dict) or "global_step" not in payload:
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
        "policy",
        "gradient_accumulation_steps",
        "seed",
        "world_size",
        "dataset_size",
        "bucket_sizes",
        "padded_samples",
    ):
        checkpoint_value = getattr(actual, field)
        current_value = getattr(expected, field)
        if checkpoint_value != current_value:
            raise ValueError(
                f"Stage 2 sampler resume identity mismatch for {field}: "
                f"checkpoint={checkpoint_value!r}, current={current_value!r}"
            )


def select_stage2_sampler_policy(requested, *, profile, config, resume):
    """Resolve fresh V-off reproduction versus immutable saved sampling semantics."""
    if requested not in ("auto", "latent_balanced", "reference_random"):
        raise ValueError("unknown Stage 2 sampler policy")
    if resume is not None and profile.world_size > 1:
        saved = read_stage2_sampler_sidecar(Path(resume)).policy
        if requested != "auto" and requested != saved:
            raise ValueError(f"Stage 2 sampler resume policy mismatch: checkpoint={saved}, current={requested}")
        return saved
    if requested != "auto":
        return requested
    if (profile.world_size == 4 and profile.enabled and profile.zero_stage == 2
            and config is not None and config.is_context_preference
            and config.visual_only_context_preference and config.prefix_mode == "clean_prefix"):
        return "reference_random"
    return "latent_balanced"
