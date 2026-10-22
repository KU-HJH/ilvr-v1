"""Bounded Stage 1 ZeRO save/resume diagnostic.

This opt-in harness intentionally uses the same composition and Trainer lifecycle
as Stage 1 while substituting only the established tiny model/data dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from pathlib import Path
from types import SimpleNamespace

_platform_python_version = platform.python_version
if _platform_python_version().endswith("+"):
    platform.python_version = lambda: _platform_python_version().rstrip("+")

import torch
import torch.distributed as dist
from transformers.integrations.deepspeed import unset_hf_deepspeed_config

from artifact_paths import resolve_run_artifact_paths
from deepspeed_runtime import (
    canonical_deepspeed_config_path,
    load_deepspeed_runtime_profile,
)
from stage1.checkpointing import (
    Stage1ArtifactKind,
    classify_stage1_artifact,
    load_stage1_peft_metadata,
)
from stage1.composition import build_stage1_training_components
from tests.stage1.test_integration import _tiny_dependencies
from utils_deepseed import parse_args as parse_training_args

platform.python_version = _platform_python_version


SUPPORTED_DIAGNOSTIC_METHODS = ("full", "lora", "dora")


def _positive_step(value: str) -> int:
    step = int(value)
    if step <= 0:
        raise argparse.ArgumentTypeError("lifecycle steps must be positive")
    return step


class _LifecycleParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if parsed.resumed_steps != parsed.initial_steps + 1:
            self.error("--resumed-steps must equal --initial-steps + 1")
        return parsed


def build_zero_compat_parser() -> argparse.ArgumentParser:
    parser = _LifecycleParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("lifecycle", "initial", "resume"), default="lifecycle"
    )
    parser.add_argument("--zero-stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument(
        "--method", choices=SUPPORTED_DIAGNOSTIC_METHODS, required=True
    )
    parser.add_argument("--deepspeed", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--initial-steps", type=_positive_step, default=1)
    parser.add_argument("--resumed-steps", type=_positive_step, default=2)
    return parser


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _initialize_distributed() -> None:
    world_size = _world_size()
    if world_size == 1 or dist.is_initialized():
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def _collect_records(record: dict[str, object]) -> list[dict[str, object]]:
    if not dist.is_initialized():
        return [record]
    gathered: list[dict[str, object] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, record)
    if any(item is None for item in gathered):
        raise RuntimeError("Stage 1 lifecycle collective returned an empty record")
    return [item for item in gathered if item is not None]


def _machine_local_leader() -> bool:
    if not dist.is_initialized():
        return True
    return int(os.environ["LOCAL_RANK"]) == 0


def _broadcast_rank0(value: str | None) -> str:
    if not dist.is_initialized():
        if type(value) is not str:
            raise ValueError("rank zero did not observe a Stage 1 artifact")
        return value
    values: list[str | None] = [value]
    dist.broadcast_object_list(values, src=0)
    if type(values[0]) is not str:
        raise ValueError("rank zero broadcast an invalid Stage 1 artifact")
    return values[0]


def _observe_stage1_artifact(checkpoint: Path, method: str) -> str:
    observed: str | None = None
    if _rank() == 0:
        observed = classify_stage1_artifact(checkpoint).value
        if observed == "peft":
            if load_stage1_peft_metadata(checkpoint).tuning_method != method:
                raise ValueError("observed Stage 1 PEFT method is inconsistent")
        elif observed != "full" or method != "full":
            raise ValueError("observed Stage 1 artifact kind is inconsistent")
    return _broadcast_rank0(observed)


def _phase_paths(experiment_id: str):
    validated = resolve_run_artifact_paths(experiment_id)
    return SimpleNamespace(
        experiment_id=validated.experiment_id,
        checkpoint_dir=Path("/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/checkpoints") / experiment_id,
        result_dir=Path("/root/workspace/ILVR/.worktrees/stage1-baseline-10ep-ib-20261022/results") / experiment_id,
    )


def _require_fresh_run(paths) -> None:
    collision = any(
        directory.exists()
        for directory in (paths.checkpoint_dir, paths.result_dir)
    )
    if dist.is_initialized():
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        observed = torch.tensor(int(collision), dtype=torch.int32, device=device)
        dist.all_reduce(observed, op=dist.ReduceOp.MAX)
        collision = bool(observed.item())
    if collision:
        raise FileExistsError(
            f"refusing to reuse Stage 1 experiment ID: {paths.experiment_id}"
        )


def _training_args(
    *,
    paths,
    fixture_root: Path,
    method: str,
    deepspeed: Path,
    max_steps: int,
    resume: Path | None = None,
):
    argv = [
        "--stage",
        "stage1",
        "--model",
        "tiny/original-qwen",
        "--data_path",
        str(fixture_root / "TRAIN.jsonl"),
        "--image_root",
        str(fixture_root / "images"),
        "--save_model_path",
        str(paths.checkpoint_dir),
        "--latent_size",
        "2",
        "--per_device_train_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "2",
        "--max_steps",
        str(max_steps),
        "--save_strategy",
        "steps" if resume is None else "no",
        "--save_steps",
        "1",
        "--attention_implementation",
        "eager",
        "--lora_r",
        "4",
        "--lora_alpha",
        "8",
        "--lora_dropout",
        "0",
        "--sim_weight",
        "0.5",
        "--seed",
        "47",
        "--tuning_method",
        method,
        "--deepspeed_config",
        str(deepspeed),
    ]
    if resume is not None:
        argv.extend(("--resume_from_checkpoint_path", str(resume)))
    return parse_training_args(argv)


def _prepare_fixture(paths, *, fresh: bool = True) -> Path:
    fixture_root = paths.checkpoint_dir / "tiny-data"
    if _machine_local_leader():
        if fresh:
            (fixture_root / "images").mkdir(parents=True, exist_ok=False)
            (fixture_root / "TRAIN.jsonl").write_text("{}\n", encoding="utf-8")
        elif (
            not (fixture_root / "images").is_dir()
            or not (fixture_root / "TRAIN.jsonl").is_file()
        ):
            raise FileNotFoundError("Stage 1 resume fixture is not node-local")
    if dist.is_initialized():
        dist.barrier()
    return fixture_root


def _publish_witness(
    paths,
    witness: dict[str, object],
    *,
    filename: str = "witness.json",
    is_writer: bool | None = None,
    writer=None,
    failure_counter=None,
    synchronize=None,
) -> None:
    from multinode_runtime import _canonical_json_bytes, _publish_bytes_no_replace

    selected_writer = _publish_bytes_no_replace if writer is None else writer
    if failure_counter is None:
        def failure_counter(local_failure):
            if not dist.is_initialized():
                return local_failure
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            failed = torch.tensor(local_failure, dtype=torch.int32, device=device)
            dist.all_reduce(failed, op=dist.ReduceOp.SUM)
            return int(failed.item())
    if synchronize is None:
        synchronize = lambda: dist.barrier() if dist.is_initialized() else None
    local_error: Exception | None = None
    machine_writer = _machine_local_leader() if is_writer is None else is_writer
    if machine_writer:
        try:
            selected_writer(
                paths.result_dir / filename,
                _canonical_json_bytes(witness) + b"\n",
            )
        except Exception as error:  # propagated to every launched rank below
            local_error = error
    failed_writers = failure_counter(int(local_error is not None))
    if failed_writers:
        if local_error is not None:
            raise local_error
        raise RuntimeError(
            "Stage 1 witness publication failed on a machine-local leader"
        )
    synchronize()


def _build_resumed_stage1_components(*, args, profile):
    """Start resume construction without the first Trainer's global ZeRO hint."""
    unset_hf_deepspeed_config()
    return build_stage1_training_components(
        args,
        dependencies=_tiny_dependencies(),
        deepspeed_runtime_profile=profile,
    )


def _run_lifecycle(args: argparse.Namespace) -> dict[str, object]:
    _initialize_distributed()
    paths = resolve_run_artifact_paths(args.experiment_id)
    _require_fresh_run(paths)
    profile = load_deepspeed_runtime_profile(
        args.deepspeed, world_size=_world_size()
    )
    expected_config = canonical_deepspeed_config_path(args.zero_stage).resolve()
    if profile.zero_stage != args.zero_stage or profile.config_path != expected_config:
        raise ValueError(
            f"--zero-stage {args.zero_stage} requires {expected_config}"
        )

    fixture_root = _prepare_fixture(paths)
    first_args = _training_args(
        paths=paths,
        fixture_root=fixture_root,
        method=args.method,
        deepspeed=profile.config_path,
        max_steps=args.initial_steps,
    )
    first_args.deepspeed_runtime_profile = profile
    first = build_stage1_training_components(
        first_args,
        dependencies=_tiny_dependencies(),
        deepspeed_runtime_profile=profile,
    )
    first.trainer.collect_stage1_diagnostics = True
    first.trainer.args.save_total_limit = 2
    first_result = first.trainer.train()
    checkpoint = paths.checkpoint_dir / f"checkpoint-{args.initial_steps}"
    if (
        int(first.trainer.state.global_step) != args.initial_steps
        or not checkpoint.is_dir()
    ):
        raise AssertionError(
            f"Stage 1 did not save a complete checkpoint-{args.initial_steps}"
        )
    if first.trainer._successful_optimizer_steps != args.initial_steps:
        raise AssertionError("Stage 1 initial phase applied the wrong number of steps")

    artifact_kind = classify_stage1_artifact(checkpoint)
    adapter_checkpoint_present = (checkpoint / "adapter_model.safetensors").is_file()
    if args.method in ("lora", "dora"):
        required = {
            "adapter_config.json",
            "adapter_model.safetensors",
            "ilvr_token_rows.safetensors",
            "stage1_peft_metadata.json",
        }
        missing = sorted(
            name for name in required if not (checkpoint / name).is_file()
        )
        if missing:
            raise AssertionError(f"Stage 1 PEFT checkpoint is incomplete: {missing}")
        if (checkpoint / "model.safetensors").exists():
            raise AssertionError("Stage 1 PEFT checkpoint contains full model weights")
        if artifact_kind is not Stage1ArtifactKind.PEFT:
            raise AssertionError(
                f"Stage 1 {args.method} checkpoint kind is {artifact_kind.value}"
            )
        if args.method == "lora":
            metadata = load_stage1_peft_metadata(checkpoint)
            if metadata.tuning_method != "lora":
                raise AssertionError("Stage 1 LoRA metadata does not report method=lora")
            peft_config = json.loads(
                (checkpoint / "adapter_config.json").read_text(encoding="utf-8")
            )
            if peft_config.get("use_dora") is not False:
                raise AssertionError("Stage 1 LoRA PEFT config enables DoRA")
    elif artifact_kind is not Stage1ArtifactKind.FULL:
        raise AssertionError(
            f"Stage 1 full checkpoint kind is {artifact_kind.value}"
        )

    observed_norm = first.trainer._representative_gradient_norm
    if observed_norm is None or not math.isfinite(observed_norm):
        raise AssertionError("Stage 1 did not observe a finite gradient norm")

    resumed_args = _training_args(
        paths=paths,
        fixture_root=fixture_root,
        method=args.method,
        deepspeed=profile.config_path,
        max_steps=args.resumed_steps,
        resume=checkpoint,
    )
    resumed_args.deepspeed_runtime_profile = profile
    resumed = _build_resumed_stage1_components(args=resumed_args, profile=profile)
    resumed.trainer.collect_stage1_diagnostics = True
    resumed_result = resumed.trainer.train(resume_from_checkpoint=str(checkpoint))
    if int(resumed.trainer.state.global_step) != args.resumed_steps:
        raise AssertionError(
            "Stage 1 resume did not advance from step "
            f"{args.initial_steps} to step {args.resumed_steps}"
        )
    if resumed.trainer._successful_optimizer_steps != 1:
        raise AssertionError("Stage 1 resumed phase did not apply exactly one step")

    local_record = {
        "rank": _rank(),
        "initial_global_step": args.initial_steps,
        "resumed_global_step": int(resumed.trainer.state.global_step),
        "initial_optimizer_steps": first.trainer._successful_optimizer_steps,
        "resumed_optimizer_steps": resumed.trainer._successful_optimizer_steps,
        "first_loss": float(first_result.training_loss),
        "resumed_loss": float(resumed_result.training_loss),
        "finite_gradient_norm": math.isfinite(observed_norm),
        "representative_gradient_norm": observed_norm,
        "method_artifact": (
            "full" if args.method == "full" else f"peft-{args.method}"
        ),
        "worker_error": None,
    }
    records = _collect_records(local_record)
    if [record["rank"] for record in records] != list(range(len(records))):
        raise AssertionError(f"Stage 1 process ranks are incomplete: {records}")
    if any(
        record["initial_global_step"] != args.initial_steps
        or record["resumed_global_step"] != args.resumed_steps
        or record["initial_optimizer_steps"] != args.initial_steps
        or record["resumed_optimizer_steps"] != 1
        or not math.isfinite(record["first_loss"])
        or not math.isfinite(record["resumed_loss"])
        or not record["finite_gradient_norm"]
        or not math.isfinite(record["representative_gradient_norm"])
        or record["worker_error"] is not None
        for record in records
    ):
        raise AssertionError(f"Stage 1 ranks disagreed on lifecycle: {records}")

    witness = {
        "schema_version": 1,
        "stage": "stage1",
        "zero_stage": profile.zero_stage,
        "config_digest": profile.config_digest,
        "world_size": profile.world_size,
        "method": args.method,
        "artifact_kind": artifact_kind.value,
        "method_artifact": (
            "full" if args.method == "full" else f"peft-{args.method}"
        ),
        "adapter_checkpoint_present": adapter_checkpoint_present,
        "initial_global_step": args.initial_steps,
        "resumed_global_step": args.resumed_steps,
        "finite_gradient_norm": True,
        "representative_gradient_norm": observed_norm,
        "successful_optimizer_steps": args.resumed_steps,
        "finite_loss": True,
        "first_loss": float(first_result.training_loss),
        "resumed_loss": float(resumed_result.training_loss),
        "process_ranks": [record["rank"] for record in records],
        "rank_records": records,
        "worker_errors": [],
        "checkpoint": str(checkpoint),
        "experiment_id": paths.experiment_id,
    }
    _publish_witness(paths, witness)
    return witness


def _phase_profile(args: argparse.Namespace):
    profile = load_deepspeed_runtime_profile(
        args.deepspeed, world_size=_world_size()
    )
    expected_config = canonical_deepspeed_config_path(args.zero_stage).resolve()
    if profile.zero_stage != args.zero_stage or profile.config_path != expected_config:
        raise ValueError(f"--zero-stage {args.zero_stage} requires {expected_config}")
    return profile


def _run_initial_phase(args: argparse.Namespace) -> dict[str, object]:
    from multinode_evidence import build_lifecycle_phase_evidence

    _initialize_distributed()
    paths = _phase_paths(args.experiment_id)
    _require_fresh_run(paths)
    profile = _phase_profile(args)
    fixture_root = _prepare_fixture(paths)
    training_args = _training_args(
        paths=paths, fixture_root=fixture_root, method=args.method,
        deepspeed=profile.config_path, max_steps=args.initial_steps,
    )
    training_args.deepspeed_runtime_profile = profile
    components = build_stage1_training_components(
        training_args, dependencies=_tiny_dependencies(),
        deepspeed_runtime_profile=profile,
    )
    components.trainer.collect_stage1_diagnostics = True
    components.trainer.args.save_total_limit = 2
    result = components.trainer.train()
    checkpoint = paths.checkpoint_dir / f"checkpoint-{args.initial_steps}"
    if (
        int(components.trainer.state.global_step) != args.initial_steps
        or components.trainer._successful_optimizer_steps != args.initial_steps
        or not checkpoint.is_dir()
    ):
        raise AssertionError("Stage 1 initial phase did not produce checkpoint-2")
    gradient = components.trainer._representative_gradient_norm
    if type(gradient) not in (int, float) or not math.isfinite(gradient):
        raise AssertionError("Stage 1 initial phase did not observe a finite gradient")
    artifact_kind = _observe_stage1_artifact(checkpoint, args.method)
    records = _collect_records({
        "rank": _rank(),
        "global_step": int(components.trainer.state.global_step),
        "optimizer_step": components.trainer._successful_optimizer_steps,
        "loss": float(result.training_loss),
        "gradient_norm": float(gradient),
        "worker_error": None,
    })
    witness = build_lifecycle_phase_evidence(
        phase="initial", stage="stage1", method=args.method,
        zero_stage=profile.zero_stage, config_digest=profile.config_digest,
        experiment_id=paths.experiment_id, checkpoint_path=checkpoint,
        artifact_kind=artifact_kind, rank_records=records,
    )
    _publish_witness(paths, witness, filename="initial-witness.json")
    return witness


def _run_resume_phase(args: argparse.Namespace) -> dict[str, object]:
    from multinode_evidence import build_lifecycle_phase_evidence

    _initialize_distributed()
    paths = _phase_paths(args.experiment_id)
    profile = _phase_profile(args)
    checkpoint = paths.checkpoint_dir / f"checkpoint-{args.initial_steps}"
    if not checkpoint.is_dir():
        raise FileNotFoundError("Stage 1 resume checkpoint-2 is not node-local")
    artifact_kind = _observe_stage1_artifact(checkpoint, args.method)
    fixture_root = _prepare_fixture(paths, fresh=False)
    resumed_args = _training_args(
        paths=paths, fixture_root=fixture_root, method=args.method,
        deepspeed=profile.config_path, max_steps=args.resumed_steps,
        resume=checkpoint,
    )
    resumed_args.deepspeed_runtime_profile = profile
    resumed = _build_resumed_stage1_components(args=resumed_args, profile=profile)
    resumed.trainer.collect_stage1_diagnostics = True
    result = resumed.trainer.train(resume_from_checkpoint=str(checkpoint))
    if (
        int(resumed.trainer.state.global_step) != args.resumed_steps
        or resumed.trainer._successful_optimizer_steps != 1
    ):
        raise AssertionError("Stage 1 resume phase did not advance 2-to-3")
    gradient = resumed.trainer._representative_gradient_norm
    if type(gradient) not in (int, float) or not math.isfinite(gradient):
        raise AssertionError("Stage 1 resume phase did not observe a finite gradient")
    records = _collect_records({
        "rank": _rank(),
        "global_step": int(resumed.trainer.state.global_step),
        "optimizer_step": resumed.trainer._successful_optimizer_steps,
        "loss": float(result.training_loss),
        "gradient_norm": float(gradient),
        "worker_error": None,
    })
    witness = build_lifecycle_phase_evidence(
        phase="resume", stage="stage1", method=args.method,
        zero_stage=profile.zero_stage, config_digest=profile.config_digest,
        experiment_id=paths.experiment_id, checkpoint_path=checkpoint,
        artifact_kind=artifact_kind, rank_records=records,
    )
    _publish_witness(paths, witness, filename="resume-witness.json")
    return witness


def run_zero_compat(args: argparse.Namespace) -> dict[str, object]:
    if args.phase == "initial":
        return _run_initial_phase(args)
    if args.phase == "resume":
        return _run_resume_phase(args)
    return _run_lifecycle(args)


def main(argv: list[str] | None = None) -> int:
    args = build_zero_compat_parser().parse_args(argv)
    witness = run_zero_compat(args)
    if _rank() == 0:
        print(json.dumps(witness, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
