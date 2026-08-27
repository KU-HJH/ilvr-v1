"""Immutable DeepSpeed runtime identity established before accelerator setup."""

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


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
    if world_size < 1:
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
    if zero_stage not in (1, 2, 3):
        raise ValueError("zero_optimization.stage must be 1, 2, or 3")
    try:
        mesh = int(payload.get("data_parallel_size", 1)) * int(
            payload.get("sequence_parallel_size", 1)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("DeepSpeed mesh dimensions must be integers") from error
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
