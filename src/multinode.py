"""Pure fixed topology and launch planning for ILVR multi-node runs."""

import argparse
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Literal

from src.deepspeed_runtime import (
    canonical_deepspeed_config_path,
    load_deepspeed_runtime_profile,
)


_REPOSITORY_ROOT = Path("/root/workspace/ILVR")
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,127}$")
_TRAINING_REQUIRED_FREE_BYTES = 100 * 1024**3


@dataclass(frozen=True)
class ClusterSpec:
    """The approved, non-discoverable two-node ILVR topology."""

    remote_host: str
    remote_container: str
    remote_repo_root: Path
    remote_host_repo_root: Path
    master_addr: str
    socket_ifname: str
    remote_rank: int = 0
    local_rank: int = 1
    num_machines: int = 2
    gpus_per_node: int = 2
    num_processes: int = 4

    @classmethod
    def default(cls) -> "ClusterSpec":
        return cls(
            remote_host="korea@10.1.102.41",
            remote_container="llm_kd",
            remote_repo_root=_REPOSITORY_ROOT,
            remote_host_repo_root=Path("/home/korea/Workspace/hjh/workspace/ILVR"),
            master_addr="10.1.102.41",
            socket_ifname="bond1.102",
        )


@dataclass(frozen=True)
class LaunchPlan:
    """The immutable coordinator-controlled identity of one training attempt."""

    stage: Literal["stage1", "stage2"]
    tuning_method: Literal["full", "lora", "dora"]
    zero_stage: Literal[1, 2, 3]
    run_id: str
    main_process_port: int
    world_size: int
    deepspeed_config: Path
    cluster: ClusterSpec
    attempt_id: str = "initial"
    required_free_bytes: int = _TRAINING_REQUIRED_FREE_BYTES

    def __post_init__(self) -> None:
        if self.stage not in ("stage1", "stage2"):
            raise ValueError("stage must be 'stage1' or 'stage2'")
        if self.tuning_method not in ("full", "lora", "dora"):
            raise ValueError("tuning_method must be 'full', 'lora', or 'dora'")
        if type(self.zero_stage) is not int or self.zero_stage not in (1, 2, 3):
            raise ValueError("zero_stage must be 1, 2, or 3")
        _validate_identifier("run_id", self.run_id)
        _validate_identifier("attempt_id", self.attempt_id)
        if (
            type(self.main_process_port) is not int
            or not 1024 <= self.main_process_port <= 65535
        ):
            raise ValueError("main_process_port must be between 1024 and 65535")
        if self.cluster != ClusterSpec.default():
            raise ValueError("cluster must match the fixed ILVR topology")
        if self.world_size != self.cluster.num_processes:
            raise ValueError("world_size must equal the fixed four-process topology")
        if self.required_free_bytes != _TRAINING_REQUIRED_FREE_BYTES:
            raise ValueError("required_free_bytes must be 100 GiB for training")
        expected_config = canonical_deepspeed_config(self.zero_stage)
        if self.deepspeed_config != expected_config:
            raise ValueError("deepspeed_config must be the canonical ZeRO config")


def _validate_identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not _RUN_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must match {_RUN_ID_PATTERN.pattern}")


def canonical_deepspeed_config(zero_stage: int) -> Path:
    """Return the lexical canonical config path for one supported ZeRO stage."""
    if type(zero_stage) is not int or zero_stage not in (1, 2, 3):
        raise ValueError("zero_stage must be 1, 2, or 3")
    approved_path = canonical_deepspeed_config_path(zero_stage)
    expected_name = f"deepspeed_zero{zero_stage}.json"
    if approved_path.name != expected_name:
        raise ValueError("canonical DeepSpeed config selection is inconsistent")
    return _REPOSITORY_ROOT / "configs" / expected_name


def build_launch_plan(
    *,
    stage: Literal["stage1", "stage2"],
    tuning_method: Literal["full", "lora", "dora"],
    zero_stage: Literal[1, 2, 3],
    run_id: str,
    main_process_port: int,
    attempt_id: str = "initial",
) -> LaunchPlan:
    """Validate and build the fixed four-process training launch plan."""
    cluster = ClusterSpec.default()
    config_path = canonical_deepspeed_config(zero_stage)
    plan = LaunchPlan(
        stage=stage,
        tuning_method=tuning_method,
        zero_stage=zero_stage,
        run_id=run_id,
        main_process_port=main_process_port,
        world_size=cluster.num_processes,
        deepspeed_config=config_path,
        cluster=cluster,
        attempt_id=attempt_id,
    )
    profile = load_deepspeed_runtime_profile(
        canonical_deepspeed_config_path(zero_stage), world_size=cluster.num_processes
    )
    if profile.zero_stage != zero_stage:
        raise ValueError(
            "zero_optimization.stage must match the requested zero_stage"
        )
    return plan


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    plan_parser.add_argument(
        "--tuning-method", choices=("full", "lora", "dora"), required=True
    )
    plan_parser.add_argument("--zero-stage", choices=(1, 2, 3), type=int, required=True)
    plan_parser.add_argument("--run-id", required=True)
    plan_parser.add_argument("--attempt-id", default="initial")
    plan_parser.add_argument("--main-process-port", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        plan = build_launch_plan(
            stage=args.stage,
            tuning_method=args.tuning_method,
            zero_stage=args.zero_stage,
            run_id=args.run_id,
            main_process_port=args.main_process_port,
            attempt_id=args.attempt_id,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(_json_ready(plan), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
