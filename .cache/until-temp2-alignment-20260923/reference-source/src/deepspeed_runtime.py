"""Immutable DeepSpeed runtime identity established before accelerator setup."""

import argparse
import functools
import hashlib
import importlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

from packaging.version import InvalidVersion, Version


DEEPSPEED_RUNTIME_FILENAME = "deepspeed-runtime.json"
DEEPSPEED_RUNTIME_SCHEMA_VERSION = 1
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
_MIN_ZERO3_CHECKPOINTING_DEEPSPEED_VERSION = Version("0.18.7")
_ZERO3_FROZEN_CHECKPOINT_PATCH_MARKER = (
    "__ilvr_zero3_frozen_checkpoint_compatibility__"
)
_ZERO3_STAGE2_OUTPUT_PATCH_MARKER = (
    "__ilvr_zero3_stage2_output_compatibility__"
)
_ZERO3_PRESERVE_OUTPUT_IDENTITY_MARKER = (
    "__ilvr_preserve_zero3_output_identity__"
)


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


@dataclass(frozen=True)
class _Zero3FrozenCheckpointAPI:
    coordinator_cls: type
    iter_params: object
    z3_leaf_module: object
    current_graph_task_id: object


@dataclass(frozen=True)
class _Zero3Stage2OutputAPI:
    zero_utils: object
    parameter_offload: object
    apply_to_tensors_only: object


def _load_zero3_stage2_output_api() -> _Zero3Stage2OutputAPI:
    """Load the DeepSpeed output walker that rewrites module return tensors."""
    try:
        zero_utils = importlib.import_module("deepspeed.runtime.zero.utils")
        parameter_offload = importlib.import_module(
            "deepspeed.runtime.zero.parameter_offload"
        )
    except Exception as error:
        raise RuntimeError(
            "DeepSpeed private API for Stage 2 output compatibility could not "
            "be imported"
        ) from error
    apply_to_tensors_only = getattr(zero_utils, "apply_to_tensors_only", None)
    if not callable(apply_to_tensors_only):
        raise RuntimeError(
            "DeepSpeed private API for Stage 2 output compatibility differs: "
            "missing apply_to_tensors_only"
        )
    return _Zero3Stage2OutputAPI(
        zero_utils=zero_utils,
        parameter_offload=parameter_offload,
        apply_to_tensors_only=apply_to_tensors_only,
    )


def install_zero3_stage2_output_compatibility(
    profile: DeepSpeedRuntimeProfile,
) -> bool:
    """Keep marked loss-bearing auxiliary tensors opaque to ZeRO output hooks."""
    if not isinstance(profile, DeepSpeedRuntimeProfile):
        raise TypeError("profile must be a DeepSpeedRuntimeProfile")
    if not (profile.enabled and profile.zero_stage == 3):
        return False
    api = _load_zero3_stage2_output_api()
    original = api.apply_to_tensors_only
    if getattr(original, _ZERO3_STAGE2_OUTPUT_PATCH_MARKER, False):
        return True

    @functools.wraps(original)
    def apply_to_tensors_only(function, value, warning_msg_fn=None):
        if (
            getattr(value, _ZERO3_PRESERVE_OUTPUT_IDENTITY_MARKER, False)
            is True
        ):
            return value
        return original(function, value, warning_msg_fn=warning_msg_fn)

    setattr(apply_to_tensors_only, _ZERO3_STAGE2_OUTPUT_PATCH_MARKER, True)
    api.zero_utils.apply_to_tensors_only = apply_to_tensors_only
    api.parameter_offload.apply_to_tensors_only = apply_to_tensors_only
    return True


def _load_zero3_frozen_checkpoint_api() -> _Zero3FrozenCheckpointAPI:
    """Load and validate the private DeepSpeed/PyTorch API used by the shim."""
    try:
        coordinator_module = importlib.import_module(
            "deepspeed.runtime.zero.partitioned_param_coordinator"
        )
        torch_module = importlib.import_module("torch")
    except Exception as error:
        raise RuntimeError(
            "DeepSpeed private API for ZeRO-3 frozen checkpoint compatibility "
            "could not be imported"
        ) from error

    coordinator_cls = getattr(
        coordinator_module, "PartitionedParameterCoordinator", None
    )
    iter_params = getattr(coordinator_module, "iter_params", None)
    z3_leaf_module = getattr(coordinator_module, "z3_leaf_module", None)
    torch_c = getattr(torch_module, "_C", None)
    current_graph_task_id = getattr(torch_c, "_current_graph_task_id", None)
    required = (
        (
            "PartitionedParameterCoordinator",
            coordinator_cls,
            isinstance(coordinator_cls, type),
        ),
        ("iter_params", iter_params, callable(iter_params)),
        ("z3_leaf_module", z3_leaf_module, callable(z3_leaf_module)),
        (
            "torch._C._current_graph_task_id",
            current_graph_task_id,
            callable(current_graph_task_id),
        ),
    )
    for name, _value, valid in required:
        if not valid:
            raise RuntimeError(
                "DeepSpeed private API for ZeRO-3 frozen checkpoint "
                f"compatibility differs: missing or invalid {name}"
            )
    release_sub_module = getattr(coordinator_cls, "release_sub_module", None)
    if not callable(release_sub_module):
        raise RuntimeError(
            "DeepSpeed private API for ZeRO-3 frozen checkpoint compatibility "
            "differs: missing or invalid "
            "PartitionedParameterCoordinator.release_sub_module"
        )
    return _Zero3FrozenCheckpointAPI(
        coordinator_cls=coordinator_cls,
        iter_params=iter_params,
        z3_leaf_module=z3_leaf_module,
        current_graph_task_id=current_graph_task_id,
    )


def install_zero3_frozen_checkpoint_compatibility(
    profile: DeepSpeedRuntimeProfile,
    *,
    gradient_checkpointing: bool,
) -> bool:
    """Backport frozen-parameter checkpoint recompute handling for ZeRO-3."""
    if not isinstance(profile, DeepSpeedRuntimeProfile):
        raise TypeError("profile must be a DeepSpeedRuntimeProfile")
    if type(gradient_checkpointing) is not bool:
        raise TypeError("gradient_checkpointing must be boolean")
    active = (
        profile.enabled
        and profile.zero_stage == 3
        and gradient_checkpointing
    )
    if not active:
        return False
    try:
        installed_text = importlib_metadata.version("deepspeed")
        installed_version = Version(installed_text)
    except (importlib_metadata.PackageNotFoundError, InvalidVersion) as error:
        raise RuntimeError(
            "DeepSpeed >=0.18.7 is required for ZeRO-3 with "
            "gradient checkpointing, but no valid installed version was found"
        ) from error
    if installed_version < _MIN_ZERO3_CHECKPOINTING_DEEPSPEED_VERSION:
        raise RuntimeError(
            "DeepSpeed >=0.18.7 is required; "
            f"installed {installed_text} cannot safely run ZeRO-3 with "
            "gradient checkpointing"
        )

    api = _load_zero3_frozen_checkpoint_api()
    original = api.coordinator_cls.release_sub_module
    if getattr(original, _ZERO3_FROZEN_CHECKPOINT_PATCH_MARKER, False):
        return True

    @functools.wraps(original)
    def release_sub_module(self, submodule, forward=False):
        if not forward:
            return original(self, submodule, forward=forward)
        try:
            in_graph_task = api.current_graph_task_id() != -1
        except Exception as error:
            raise RuntimeError(
                "PyTorch graph-task API failed during ZeRO-3 frozen "
                "checkpoint compatibility"
            ) from error
        if not in_graph_task:
            return original(self, submodule, forward=forward)

        try:
            recurse = api.z3_leaf_module(submodule)
            params = tuple(api.iter_params(submodule, recurse=recurse))
        except Exception as error:
            raise RuntimeError(
                "DeepSpeed private parameter API failed during ZeRO-3 "
                "frozen checkpoint compatibility"
            ) from error

        restored: list[tuple[object, bool]] = []
        seen: set[int] = set()
        try:
            for param in params:
                if id(param) in seen:
                    continue
                seen.add(id(param))
                if not hasattr(param, "requires_grad") or not hasattr(
                    param, "is_external_param"
                ):
                    raise RuntimeError(
                        "DeepSpeed private parameter API differs during ZeRO-3 "
                        "frozen checkpoint compatibility"
                    )
                if not param.requires_grad:
                    prior = param.is_external_param
                    if type(prior) is not bool:
                        raise RuntimeError(
                            "DeepSpeed private parameter API differs during "
                            "ZeRO-3 frozen checkpoint compatibility"
                        )
                    restored.append((param, prior))
                    param.is_external_param = True
            return original(self, submodule, forward=forward)
        finally:
            for param, prior in restored:
                param.is_external_param = prior

    setattr(
        release_sub_module,
        _ZERO3_FROZEN_CHECKPOINT_PATCH_MARKER,
        True,
    )
    api.coordinator_cls.release_sub_module = release_sub_module
    return True


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


def read_deepspeed_runtime_sidecar(directory: Path) -> DeepSpeedRuntimeProfile:
    """Read and strictly decode a numbered-checkpoint runtime identity."""

    path = Path(directory) / DEEPSPEED_RUNTIME_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing DeepSpeed runtime identity: {path}")
    try:
        if path.stat().st_size > _MAX_DEEPSPEED_RUNTIME_BYTES:
            raise ValueError(f"invalid DeepSpeed runtime identity: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
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


def validate_supported_deepspeed_policy(payload: Mapping[str, object]) -> None:
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
    validate_supported_deepspeed_policy(payload)
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
