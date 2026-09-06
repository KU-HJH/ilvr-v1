"""Immutable DeepSpeed runtime identity established before accelerator setup."""

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


DEEPSPEED_RUNTIME_FILENAME = "deepspeed-runtime.json"
DEEPSPEED_RUNTIME_SCHEMA_VERSION = 1
STAGE2_FULL_OPTIMIZER_OFFLOAD_CONFIG = "deepspeed_zero3_stage2_full_optimizer_offload.json"
_DEEPSPEED_RUNTIME_FIELDS = frozenset(
    {
        "schema_version",
        "enabled",
        "zero_stage",
        "config_path",
        "config_digest",
        "world_size",
    }
)
_MAX_DEEPSPEED_RUNTIME_BYTES = 16 * 1024


@dataclass(frozen=True)
class DeepSpeedRuntimeProfile:
    enabled: bool
    zero_stage: int | None
    config_path: Path | None
    config_digest: str | None
    world_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "zero_stage": self.zero_stage,
            "config_path": None if self.config_path is None else str(self.config_path),
            "config_digest": self.config_digest,
            "world_size": self.world_size,
        }


def _deepspeed_runtime_payload(
    profile: DeepSpeedRuntimeProfile,
) -> dict[str, object]:
    if not isinstance(profile, DeepSpeedRuntimeProfile):
        raise TypeError("profile must be a DeepSpeedRuntimeProfile")
    return {
        "schema_version": DEEPSPEED_RUNTIME_SCHEMA_VERSION,
        **profile.to_dict(),
    }


def write_deepspeed_runtime_sidecar(
    directory: Path, profile: DeepSpeedRuntimeProfile
) -> Path:
    """Atomically publish one immutable runtime identity in ``directory``."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / DEEPSPEED_RUNTIME_FILENAME
    payload = _deepspeed_runtime_payload(profile)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if path.exists() or path.is_symlink():
        if read_deepspeed_runtime_sidecar(destination) == profile:
            return path
        raise FileExistsError(
            f"refusing to replace different runtime identity: {path}"
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{DEEPSPEED_RUNTIME_FILENAME}.",
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
            if read_deepspeed_runtime_sidecar(destination) != profile:
                raise FileExistsError(
                    f"refusing to replace different runtime identity: {path}"
                ) from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def decode_deepspeed_runtime_sidecar(
    raw: bytes, *, path: Path
) -> DeepSpeedRuntimeProfile:
    """Strictly decode bytes already bound to one runtime-sidecar inode."""

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid DeepSpeed runtime identity: {path}") from error
    if not isinstance(payload, dict) or set(payload) != _DEEPSPEED_RUNTIME_FIELDS:
        raise ValueError(f"invalid DeepSpeed runtime identity schema: {path}")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != DEEPSPEED_RUNTIME_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported DeepSpeed runtime identity schema: {path}")

    enabled = payload["enabled"]
    zero_stage = payload["zero_stage"]
    config_path = payload["config_path"]
    config_digest = payload["config_digest"]
    world_size = payload["world_size"]
    if type(enabled) is not bool:
        raise ValueError("invalid DeepSpeed runtime identity enabled")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("invalid DeepSpeed runtime identity world_size")
    if enabled:
        if type(zero_stage) is not int or zero_stage not in (1, 2, 3):
            raise ValueError("invalid DeepSpeed runtime identity zero_stage")
        if (
            type(config_path) is not str
            or not config_path
            or not Path(config_path).is_absolute()
        ):
            raise ValueError("invalid DeepSpeed runtime identity config_path")
        if (
            type(config_digest) is not str
            or len(config_digest) != 64
            or any(character not in "0123456789abcdef" for character in config_digest)
        ):
            raise ValueError("invalid DeepSpeed runtime identity config_digest")
        restored_path = Path(config_path)
    else:
        if zero_stage is not None or config_path is not None or config_digest is not None:
            raise ValueError(
                "invalid DeepSpeed runtime identity enabled/profile combination"
            )
        restored_path = None
    return DeepSpeedRuntimeProfile(
        enabled=enabled,
        zero_stage=zero_stage,
        config_path=restored_path,
        config_digest=config_digest,
        world_size=world_size,
    )


def read_deepspeed_runtime_sidecar(directory: Path) -> DeepSpeedRuntimeProfile:
    """Read and strictly decode a numbered-checkpoint runtime identity."""

    path = Path(directory) / DEEPSPEED_RUNTIME_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing DeepSpeed runtime identity: {path}")
    try:
        if path.stat().st_size > _MAX_DEEPSPEED_RUNTIME_BYTES:
            raise ValueError(f"invalid DeepSpeed runtime identity: {path}")
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"invalid DeepSpeed runtime identity: {path}") from error
    return decode_deepspeed_runtime_sidecar(raw, path=path)


def validate_deepspeed_resume_identity(
    checkpoint: Path, expected: DeepSpeedRuntimeProfile
) -> None:
    """Reject a Trainer resume whose execution identity is incompatible."""

    if not isinstance(expected, DeepSpeedRuntimeProfile):
        raise TypeError("expected must be a DeepSpeedRuntimeProfile")
    actual = read_deepspeed_runtime_sidecar(Path(checkpoint))
    for field in ("enabled", "zero_stage", "config_digest", "world_size"):
        checkpoint_value = getattr(actual, field)
        current_value = getattr(expected, field)
        if checkpoint_value != current_value:
            raise ValueError(
                f"DeepSpeed resume identity mismatch for {field}: "
                f"checkpoint={checkpoint_value!r}, current={current_value!r}"
            )


def validate_supported_deepspeed_policy(
    payload: Mapping[str, object], *, stage2_peft_zero3: bool = False,
    stage2_full_optimizer_offload: bool = False,
) -> None:
    """Reject unsupported precision, batching, and offload policies."""
    if payload.get("bf16") != {"enabled": True}:
        raise ValueError("canonical DeepSpeed configs require bf16 enabled")
    for field in (
        "train_micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "train_batch_size",
    ):
        if payload.get(field) != "auto":
            raise ValueError(f"{field} must be 'auto'")
    zero = payload["zero_optimization"]
    if not isinstance(zero, Mapping):
        raise ValueError("zero_optimization must be a JSON object")
    for field in ("offload_optimizer", "offload_param"):
        value = zero.get(field)
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"{field} must be a JSON object")
    if stage2_full_optimizer_offload:
        if (
            zero.get("stage") != 3
            or zero.get("offload_optimizer") != {"device": "cpu", "pin_memory": False}
            or zero.get("offload_param") != {"device": "none"}
            or "optimizer" in payload
            or payload.get("zero_force_ds_cpu_optimizer", True) is not True
        ):
            raise ValueError(
                "Full ZeRO-3 optimizer offload requires CPU optimizer without pinning, "
                "GPU parameters, and Trainer optimizer groups mapped to CPUAdam"
            )
    elif stage2_peft_zero3:
        if (
            zero.get("stage") != 3
            or zero.get("stage3_prefetch_bucket_size") != 0
            or zero.get("offload_optimizer") != {"device": "none"}
            or zero.get("offload_param")
            != {"device": "cpu", "pin_memory": False}
        ):
            raise ValueError(
                "Stage 2 PEFT ZeRO-3 requires zero prefetch, CPU parameter "
                "offload without pinning, and no optimizer offload"
            )
    else:
        for field in ("offload_optimizer", "offload_param"):
            value = zero.get(field)
            if value is not None and value.get("device", "none") != "none":
                raise ValueError(f"{field} is unsupported for canonical GPU runs")


def load_deepspeed_runtime_profile(
    config_path: str | Path | None, *, world_size: int
) -> DeepSpeedRuntimeProfile:
    """Load and digest one DeepSpeed config exactly once for this runtime."""
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be positive")
    if config_path is None:
        return DeepSpeedRuntimeProfile(False, None, None, None, world_size)
    path = Path(config_path).expanduser().resolve(strict=True)
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("DeepSpeed config must be a JSON object")
    zero = payload.get("zero_optimization")
    if not isinstance(zero, Mapping):
        raise ValueError("zero_optimization must be a JSON object")
    zero_stage = zero.get("stage")
    if type(zero_stage) is not int or zero_stage not in (1, 2, 3):
        raise ValueError("zero_optimization.stage must be 1, 2, or 3")
    mesh_dimensions = {}
    for field in ("data_parallel_size", "sequence_parallel_size"):
        value = payload.get(field, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"{field} must be a positive integer")
        mesh_dimensions[field] = value
    mesh = mesh_dimensions["data_parallel_size"] * mesh_dimensions[
        "sequence_parallel_size"
    ]
    if mesh > 1 and mesh != world_size:
        raise ValueError(f"DeepSpeed mesh requires world size {mesh}, got {world_size}")
    dedicated_stage2_peft_zero3 = path == canonical_deepspeed_config_path(3).with_name(
        "deepspeed_zero3_stage2_peft.json"
    ).resolve(strict=False)
    validate_supported_deepspeed_policy(
        payload, stage2_peft_zero3=dedicated_stage2_peft_zero3,
        stage2_full_optimizer_offload=(path == canonical_deepspeed_config_path(3).with_name(
            STAGE2_FULL_OPTIMIZER_OFFLOAD_CONFIG
        ).resolve(strict=False)),
    )
    return DeepSpeedRuntimeProfile(
        True, zero_stage, path, hashlib.sha256(raw).hexdigest(), world_size
    )


def canonical_deepspeed_config_path(zero_stage: int) -> Path:
    """Return the repository's approved config for a supported ZeRO stage."""
    if zero_stage not in (1, 2, 3):
        raise ValueError("zero_stage must be 1, 2, or 3")
    repository = Path(__file__).resolve().parents[1]
    return repository / "configs" / f"deepspeed_zero{zero_stage}.json"


def bind_deepspeed_runtime_profile(
    args: argparse.Namespace, *, world_size: int | None = None
) -> DeepSpeedRuntimeProfile:
    """Cache the immutable profile on parsed arguments before CUDA initialization."""
    existing = getattr(args, "deepspeed_runtime_profile", None)
    if existing is not None:
        if not isinstance(existing, DeepSpeedRuntimeProfile):
            raise TypeError("args.deepspeed_runtime_profile has an invalid type")
    selected_config = existing.config_path if existing is not None else getattr(args, "deepspeed_config", None)
    if selected_config is not None and Path(selected_config).expanduser().resolve(strict=False) == (
        canonical_deepspeed_config_path(3).with_name(STAGE2_FULL_OPTIMIZER_OFFLOAD_CONFIG).resolve(strict=False)
    ):
        if (getattr(args, "stage", None), getattr(args, "tuning_method", None)) != ("stage2", "full"):
            raise ValueError("CPU optimizer offload requires explicit Stage 2 full tuning")
    if existing is not None:
        return existing
    resolved_world_size = (
        int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else world_size
    )
    profile = load_deepspeed_runtime_profile(
        getattr(args, "deepspeed_config", None), world_size=resolved_world_size
    )
    args.deepspeed_runtime_profile = profile
    return profile


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--world-size", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        profile = load_deepspeed_runtime_profile(
            args.config, world_size=args.world_size
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(profile.to_dict(), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
