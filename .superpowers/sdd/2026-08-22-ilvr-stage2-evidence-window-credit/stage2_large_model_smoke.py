#!/usr/bin/env python3
"""One-optimizer-step witness for ILVR Stage 1 and Stage-2 EW variants.

The script deliberately performs no checkpoint hashing. Precomputed targets
are authenticated against independent consumer witnesses. Per-rank JSON files
contain only small scalar/structural evidence; only the final completion
manifest denotes a successful distributed attempt.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
STAGE2_CONFIG_ROOT = REPOSITORY_ROOT / "configs" / "evidence_window" / "stage2"
CASE_CONFIGS = {
    "baseline": "baseline.json",
    "ew-reference-free": "ew_reference_free.json",
    "ew-precomputed-dpo": "ew_precomputed_dpo.json",
    "relation-kd": "relation_kd.json",
    "hidden-residual-kd": "hidden_residual_kd.json",
    "online-frozen-zero1": "online_frozen_zero1.json",
    "online-frozen-zero2": "online_frozen_zero2.json",
}
CASE_SIGNATURES = {
    "baseline": (frozenset(), "none", frozenset(), "window_attention_block"),
    "ew-reference-free": (
        frozenset({"ew"}),
        "none",
        frozenset({"token_logps"}),
        "window_attention_block",
    ),
    "ew-precomputed-dpo": (
        frozenset({"ew"}),
        "precomputed",
        frozenset({"token_logps"}),
        "window_attention_block",
    ),
    "relation-kd": (
        frozenset({"relation"}),
        "precomputed",
        frozenset({"token_logps"}),
        "window_attention_block",
    ),
    "hidden-residual-kd": (
        frozenset({"residual"}),
        "precomputed",
        frozenset({"token_logps", "hidden_states"}),
        "window_attention_block",
    ),
    "online-frozen-zero1": (
        frozenset({"ew"}),
        "online_frozen",
        frozenset({"token_logps"}),
        "window_attention_block",
    ),
    "online-frozen-zero2": (
        frozenset({"ew"}),
        "online_frozen",
        frozenset({"token_logps"}),
        "window_attention_block",
    ),
}
SINGLE_GPU_CASES = (
    "stage1-baseline",
    "baseline",
    "ew-reference-free",
    "ew-precomputed-dpo",
    "relation-kd",
    "hidden-residual-kd",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real ILVR optimizer step and write a compact JSON witness."
    )
    parser.add_argument(
        "--case",
        required=True,
        choices=("all-single-gpu", "stage1-baseline", *CASE_CONFIGS),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--sample-index", required=True, type=int)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--ew-config")
    parser.add_argument("--deepspeed-config")
    parser.add_argument("--reference-artifact-dir")
    parser.add_argument("--reference-dataset-sha256")
    parser.add_argument("--reference-checkpoint-id")
    parser.add_argument("--reference-checkpoint-witness")
    parser.add_argument("--reference-tokenizer-id")
    parser.add_argument("--reference-parser-version", default="stage2-marker-v1")
    parser.add_argument("--reference-model-config-id")
    parser.add_argument("--output-dir", required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sample_index < 0:
        raise ValueError("sample index must be nonnegative")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.attempt_id) is None:
        raise ValueError(
            "attempt id must be 1-64 path-safe ASCII letters, digits, dot, underscore, or dash"
        )
    if args.case == "all-single-gpu" and args.ew_config is not None:
        parser.error("--ew-config cannot override the all-single-gpu named matrix")
    return args


def _require_inputs(args: argparse.Namespace, cases: Sequence[str]) -> None:
    for name in ("stage1_checkpoint", "data", "image_root"):
        value = Path(getattr(args, name)).expanduser()
        if not value.exists():
            raise FileNotFoundError(f"--{name.replace('_', '-')} does not exist: {value}")
    if not Path(args.data).is_file():
        raise FileNotFoundError(f"--data is not a file: {args.data}")
    if not Path(args.image_root).is_dir():
        raise FileNotFoundError(f"--image-root is not a directory: {args.image_root}")
    if "stage1-baseline" in cases and not Path(args.model).expanduser().exists():
        raise FileNotFoundError(f"--model does not exist for Stage-1 smoke: {args.model}")
    if args.deepspeed_config is not None and not Path(args.deepspeed_config).is_file():
        raise FileNotFoundError(
            f"--deepspeed-config does not exist: {args.deepspeed_config}"
        )


def _read_source_row(path: str, sample_index: int) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == sample_index:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("selected dataset row must be a JSON object")
                return value
    raise IndexError(f"sample index {sample_index} is outside {path}")


def _load_processor_and_model(checkpoint: str, *, stage: str, device_dtype):
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    processor.tokenizer.add_tokens(
        ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"],
        special_tokens=True,
    )
    config = Qwen2_5_VLConfig.from_pretrained(checkpoint, trust_remote_code=True)
    config.stage = stage
    config.compress_strategy = "average"
    config.latent_size = 8
    marker_ids = {
        name: int(processor.tokenizer(text, return_tensors="pt")["input_ids"].item())
        for name, text in {
            "pad": "<|latent_pad|>",
            "start": "<|latent_start|>",
            "end": "<|latent_end|>",
        }.items()
    }
    config.latent_token_id = marker_ids["pad"]
    config.latent_start_id = marker_ids["start"]
    config.latent_end_id = marker_ids["end"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=device_dtype,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    return processor, model, marker_ids


def _apply_smoke_lora(model):
    from peft import LoraConfig, get_peft_model

    # The production SFTTrainer installs this hook before wrapping a model
    # with PEFT when reentrant gradient checkpointing is enabled.  The smoke
    # harness wraps eagerly, so it must preserve the same input-gradient path.
    model.enable_input_require_grads()
    return get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )


def _token_ids(processor, marker_ids):
    from evidence_window.layout import TokenIds

    padding = int(
        processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"].item()
    )
    termination = int(
        processor.tokenizer("<|im_end|>", return_tensors="pt")["input_ids"].item()
    )
    return TokenIds(
        latent_start=marker_ids["start"],
        latent_end=marker_ids["end"],
        latent_pad=marker_ids["pad"],
        padding=padding,
        special=frozenset(int(value) for value in processor.tokenizer.all_special_ids),
        assistant_termination=frozenset({termination}),
    )


def _stage2_config_signature(config) -> tuple[frozenset[str], str, frozenset[str], str]:
    active = frozenset(
        objective
        for objective in ("ew", "relation", "residual")
        if config.effective_weight(objective, 1.0) > 0.0
    )
    return (
        active,
        config.reference.mode,
        config.required_outputs,
        config.loo.intervention,
    )


def _validate_case_config(case: str, config) -> None:
    expected = CASE_SIGNATURES.get(case)
    if expected is None:
        raise ValueError(f"unsupported Stage-2 smoke case: {case}")
    actual = _stage2_config_signature(config)
    if actual != expected:
        raise ValueError(
            f"case {case!r} configuration signature mismatch: "
            f"expected {expected}, found {actual}"
        )


def _validate_case_zero_stage(case: str, zero_stage: int) -> None:
    expected = {
        "online-frozen-zero1": 1,
        "online-frozen-zero2": 2,
    }.get(case)
    if expected is not None and zero_stage != expected:
        raise ValueError(f"case {case!r} requires DeepSpeed ZeRO-{expected}")


def _resolve_stage2_config(args: argparse.Namespace, case: str):
    from evidence_window.stage2_config import (
        Stage2EvidenceWindowConfig,
        Stage2ScheduleConfig,
    )

    path = Path(args.ew_config) if args.ew_config else STAGE2_CONFIG_ROOT / CASE_CONFIGS[case]
    config = Stage2EvidenceWindowConfig.from_json(path)
    _validate_case_config(case, config)
    # A one-step smoke must exercise the named objective instead of remaining
    # inside a production warmup interval.
    config = replace(config, schedule=Stage2ScheduleConfig())
    if config.reference.mode == "precomputed":
        if not args.reference_artifact_dir:
            raise ValueError("precomputed smoke requires --reference-artifact-dir")
        config = replace(
            config,
            reference=replace(config.reference, artifact=args.reference_artifact_dir),
        )
    if config.reference.mode == "online_frozen":
        config = replace(
            config,
            reference=replace(config.reference, checkpoint=args.stage1_checkpoint),
        )
    return config


def _reference_identity_values(args) -> dict[str, str]:
    values = {
        "dataset_sha256": getattr(args, "reference_dataset_sha256", None),
        "checkpoint_id": getattr(args, "reference_checkpoint_id", None),
        "checkpoint_witness": getattr(args, "reference_checkpoint_witness", None),
        "tokenizer_id": getattr(args, "reference_tokenizer_id", None),
        "parser_version": getattr(args, "reference_parser_version", None),
        "model_config_id": getattr(args, "reference_model_config_id", None),
    }
    missing = sorted(name for name, value in values.items() if not value)
    if missing:
        raise ValueError(
            "precomputed smoke requires authenticated identity argument(s): "
            + ", ".join(missing)
        )
    return values


def _build_expected_reference_identity(args, config, tokenizer_special_tokens):
    """Reconstruct the consumer expectation without reading artifact metadata."""
    from evidence_window.artifacts import ArtifactIdentity

    values = _reference_identity_values(args)
    identity = ArtifactIdentity(
        format_version=1,
        dataset_sha256=values["dataset_sha256"],
        checkpoint_id=values["checkpoint_id"],
        checkpoint_witness=values["checkpoint_witness"],
        tokenizer_id=values["tokenizer_id"],
        tokenizer_special_tokens=tuple(int(value) for value in tokenizer_special_tokens),
        parser_version=values["parser_version"],
        contribution_reduction=config.preference.contribution_reduction,
        intervention=config.loo.intervention,
        attention_backend="sdpa",
        max_evidence_edges_per_window=config.loo.max_evidence_edges_per_window,
        model_config_id=values["model_config_id"],
    )
    # Reuse the artifact schema's strict primitive/range validation without
    # consulting the producer manifest.
    return ArtifactIdentity.from_dict(identity.to_dict())


def _read_precomputed_manifest_identity(artifact_dir: str):
    from evidence_window.artifacts import ArtifactIdentity, ReferenceShardMetadata

    directory = Path(artifact_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"reference artifact directory does not exist: {directory}")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"reference artifact manifest does not exist: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if type(manifest) is not dict or set(manifest) != {"format", "identity", "shards"}:
        raise ValueError("reference artifact manifest schema mismatch")
    if manifest["format"] != "ilvr-stage2-reference":
        raise ValueError("reference artifact manifest format mismatch")
    raw_shards = manifest["shards"]
    if type(raw_shards) is not list or not raw_shards:
        raise ValueError("reference artifact manifest shards must be a nonempty list")
    shards = [ReferenceShardMetadata.from_dict(raw) for raw in raw_shards]
    ranks = [shard.rank for shard in shards]
    if ranks != sorted(ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("reference artifact manifest shard ranks are invalid")
    filenames = [shard.filename for shard in shards]
    if len(set(filenames)) != len(filenames):
        raise ValueError("reference artifact manifest shard filenames are duplicated")
    missing_shards = [name for name in filenames if not (directory / name).is_file()]
    if missing_shards:
        raise ValueError(
            "reference artifact manifest names missing shard(s): "
            + ", ".join(missing_shards)
        )
    return ArtifactIdentity.from_dict(manifest["identity"]), tuple(shards)


def _validate_precomputed_preflight(args, config) -> None:
    """Reject independent identity mismatches before loading a processor/model."""
    values = _reference_identity_values(args)
    actual, shards = _read_precomputed_manifest_identity(args.reference_artifact_dir)
    expected = {
        "dataset_sha256": values["dataset_sha256"],
        "checkpoint_id": values["checkpoint_id"],
        "checkpoint_witness": values["checkpoint_witness"],
        "tokenizer_id": values["tokenizer_id"],
        "parser_version": values["parser_version"],
        "contribution_reduction": config.preference.contribution_reduction,
        "intervention": config.loo.intervention,
        "attention_backend": "sdpa",
        "max_evidence_edges_per_window": config.loo.max_evidence_edges_per_window,
        "model_config_id": values["model_config_id"],
    }
    mismatches = {
        name: (value, getattr(actual, name))
        for name, value in expected.items()
        if getattr(actual, name) != value
    }
    if mismatches:
        raise ValueError(f"reference artifact identity mismatch: {mismatches}")
    requested_sample = args.sample_index
    owner_count = sum(
        requested_sample in shard.sample_indices for shard in shards
    )
    if owner_count != 1:
        raise ValueError(
            f"sample index {requested_sample} must appear in exactly one artifact shard; "
            f"found {owner_count}"
        )
    if config.residual.enabled and config.residual.lambda_res > 0.0:
        if any(
            shard.record_count > 0 and not shard.has_hidden_residuals
            for shard in shards
        ):
            raise ValueError(
                "hidden-residual smoke requires hidden residuals in every nonempty artifact shard"
            )


def _build_precomputed_provider(artifact_dir, expected_identity):
    from evidence_window.artifacts import PrecomputedReferenceProvider

    return PrecomputedReferenceProvider(artifact_dir, expected_identity)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_json_no_overwrite(destination: Path, value: Any) -> Path:
    """Atomically expose a prepared temporary file without replacing a name.

    The successful hard link is the logical commit point. Cleanup and the
    post-link directory fsync are best effort because reporting failure after
    the destination is visible would make the attempt's state ambiguous.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    try:
        temporary.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        _fsync_directory(destination.parent)
    except Exception:
        pass
    return destination


def _attempt_claim_value(
    attempt_id: str,
    cases: Sequence[str],
    world_size: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "attempt_id": attempt_id,
        "cases": list(cases),
        "world_size": world_size,
    }


def _claim_attempt_destination(
    output_dir: Path,
    attempt_id: str,
    cases: Sequence[str],
    *,
    world_size: int,
) -> Path:
    if world_size <= 0:
        raise ValueError("world size must be positive")
    if not cases:
        raise ValueError("smoke attempt must select at least one case")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", attempt_id) is None:
        raise ValueError("invalid attempt id")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"smoke attempt destination already exists and cannot be claimed: {output_dir}"
        ) from exc
    _fsync_directory(output_dir.parent)
    return _atomic_publish_json_no_overwrite(
        output_dir / "attempt.json",
        _attempt_claim_value(attempt_id, cases, world_size),
    )


def _validate_attempt_claim(
    output_dir: Path,
    attempt_id: str,
    cases: Sequence[str],
    world_size: int,
) -> None:
    claim_path = output_dir / "attempt.json"
    if not claim_path.is_file():
        raise ValueError("attempt ownership claim is missing")
    with claim_path.open(encoding="utf-8") as handle:
        claim = json.load(handle)
    if claim != _attempt_claim_value(attempt_id, cases, world_size):
        raise ValueError("attempt ownership claim does not match this launch")


def _prepare_attempt(
    args: argparse.Namespace,
    cases: Sequence[str],
    output_dir: Path,
    *,
    world_size: int,
) -> dict[str, Any]:
    from evidence_window.stage2_config import read_deepspeed_zero_stage

    _require_inputs(args, cases)
    _validate_attempt_claim(output_dir, args.attempt_id, cases, world_size)
    _read_source_row(args.data, args.sample_index)
    zero_stage = (
        read_deepspeed_zero_stage(args.deepspeed_config)
        if args.deepspeed_config
        else 0
    )
    configs: dict[str, Any] = {}
    for case in cases:
        if case == "stage1-baseline":
            continue
        config = _resolve_stage2_config(args, case)
        _validate_case_zero_stage(case, zero_stage)
        config.validate_runtime(zero_stage)
        if config.reference.mode == "precomputed":
            _validate_precomputed_preflight(args, config)
        configs[case] = config
    return configs


def _run_collective_phase(name: str, operation):
    """Return local work only after every distributed rank reports success."""
    import torch
    import torch.distributed as distributed

    result = None
    local_error = None
    local_exception = None
    try:
        result = operation()
    except Exception as exc:
        local_exception = exc
        local_error = f"{type(exc).__name__}: {exc}"

    if distributed.is_available() and distributed.is_initialized():
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if distributed.get_backend() == "nccl"
            else torch.device("cpu")
        )
        failed = torch.tensor(int(local_error is not None), dtype=torch.long, device=device)
        distributed.all_reduce(failed, op=distributed.ReduceOp.MAX)
        if failed.item():
            errors: list[str | None] = [None] * distributed.get_world_size()
            distributed.all_gather_object(errors, local_error)
            details = "; ".join(
                f"rank {rank}: {error}"
                for rank, error in enumerate(errors)
                if error is not None
            )
            raise RuntimeError(f"distributed {name} failed: {details}")
    elif local_error is not None:
        assert local_exception is not None
        raise local_exception
    return result


def _initialize_distributed_from_environment(
    *, backend: str | None = None
) -> tuple[int, int]:
    """Initialize launcher-provided distributed state before any attempt work."""
    import torch
    import torch.distributed as distributed

    required_names = (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    )
    launcher_names = required_names + ("LOCAL_WORLD_SIZE",)
    launcher_environment = any(name in os.environ for name in launcher_names)
    if not launcher_environment:
        if distributed.is_available() and distributed.is_initialized():
            if distributed.get_world_size() > 1:
                raise RuntimeError(
                    "a multi-rank initialized process group requires the complete "
                    "distributed launcher environment"
                )
            rank, world_size, local_rank = 0, 1, 0
        else:
            return 0, 1
    else:
        missing = [name for name in required_names if name not in os.environ]
        if missing:
            raise RuntimeError(
                "distributed launcher environment is incomplete; requires "
                + ", ".join(required_names)
                + "; missing "
                + ", ".join(missing)
            )
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            master_port = int(os.environ["MASTER_PORT"])
            local_world_size = (
                int(os.environ["LOCAL_WORLD_SIZE"])
                if "LOCAL_WORLD_SIZE" in os.environ
                else None
            )
        except ValueError as exc:
            raise RuntimeError("distributed launcher rank environment is invalid") from exc
        if (
            world_size <= 0
            or rank < 0
            or rank >= world_size
            or local_rank < 0
            or local_rank >= world_size
        ):
            raise RuntimeError("distributed launcher rank environment is out of range")
        if not os.environ["MASTER_ADDR"].strip():
            raise RuntimeError("distributed launcher MASTER_ADDR is invalid")
        if master_port < 1 or master_port > 65535:
            raise RuntimeError("distributed launcher MASTER_PORT is out of range")
        if local_world_size is not None and (
            local_world_size <= 0
            or local_world_size > world_size
            or local_rank >= local_world_size
        ):
            raise RuntimeError(
                "distributed launcher LOCAL_RANK is out of LOCAL_WORLD_SIZE range"
            )

    cuda_target = bool(torch.cuda.is_available())
    selected_backend = "nccl" if cuda_target else "gloo"
    if backend is not None and str(backend).lower() != selected_backend:
        raise RuntimeError(
            f"requested distributed backend {backend!r} does not match "
            f"the {selected_backend!r} device target"
        )
    if cuda_target:
        device_count = int(torch.cuda.device_count())
        if device_count <= 0 or local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK {local_rank} does not identify an available CUDA device "
                f"among {device_count} visible devices"
            )
        torch.cuda.set_device(local_rank)

    if distributed.is_available() and distributed.is_initialized():
        actual = (distributed.get_rank(), distributed.get_world_size())
        if actual != (rank, world_size):
            raise RuntimeError(
                "initialized process group does not match launcher rank/world size"
            )
        actual_backend = str(distributed.get_backend()).lower().split(".")[-1]
        if actual_backend != selected_backend:
            raise RuntimeError(
                f"initialized process group backend {actual_backend!r} does not match "
                f"requested backend {selected_backend!r}"
            )
        return actual
    if world_size == 1:
        return rank, world_size
    if not distributed.is_available():
        raise RuntimeError(
            "WORLD_SIZE>1 requires torch.distributed before smoke preflight"
        )

    try:
        distributed.init_process_group(
            backend=selected_backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to initialize distributed coordination before smoke preflight"
        ) from exc
    return distributed.get_rank(), distributed.get_world_size()


def _coordinate_attempt_start(
    args: argparse.Namespace,
    cases: Sequence[str],
    output_dir: Path,
    *,
    collective_phase=None,
) -> dict[str, Any]:
    """Claim once, agree collectively, then preflight on every rank."""
    import torch.distributed as distributed

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not distributed.is_available() or not distributed.is_initialized():
            raise RuntimeError(
                "distributed process group must be initialized before attempt coordination"
            )
        rank = distributed.get_rank()
        world_size = distributed.get_world_size()
    phase = _run_collective_phase if collective_phase is None else collective_phase
    phase(
        "attempt ownership claim",
        lambda: (
            _claim_attempt_destination(
                output_dir,
                args.attempt_id,
                cases,
                world_size=world_size,
            )
            if rank == 0
            else None
        ),
    )
    return phase(
        "attempt preflight",
        lambda: _prepare_attempt(args, cases, output_dir, world_size=world_size),
    )


def _training_arguments(output_dir: Path, deepspeed_config: str | None):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(output_dir),
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=8192,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
    )


def _gradient_witness(log_history) -> dict[str, Any]:
    witness = {"seen": False, "finite": True, "nonzero": False}
    for record in log_history:
        if "grad_norm" not in record:
            continue
        witness["seen"] = True
        value = record["grad_norm"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            witness["finite"] = False
            continue
        witness["finite"] = witness["finite"] and math.isfinite(value)
        witness["nonzero"] = witness["nonzero"] or value != 0.0
    return witness


def _forward_witness(model) -> tuple[dict[str, int], Any]:
    witness = {"total": 0, "blocked": 0, "hidden": 0}

    def observe(_module, _args, kwargs):
        witness["total"] += 1
        witness["blocked"] += int(kwargs.get("attention_block_mask") is not None)
        witness["hidden"] += int(bool(kwargs.get("output_hidden_states", False)))

    return witness, model.register_forward_pre_hook(observe, with_kwargs=True)


def _parameter_update_tracker(model):
    import torch

    selected = [
        (parameter, parameter.detach().cpu().clone())
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.numel() > 0
    ]
    if not selected:
        raise RuntimeError("smoke policy exposes no trainable parameter")

    def finish() -> dict[str, Any]:
        after = [parameter.detach().cpu() for parameter, _before in selected]
        changed = sum(
            not torch.equal(current, before)
            for current, (_parameter, before) in zip(after, selected, strict=True)
        )
        return {
            "changed": changed > 0,
            "finite": all(bool(current.isfinite().all().item()) for current in after),
            "changed_parameter_count": changed,
        }

    return finish


def _distributed_max_int(value: int) -> int:
    import torch
    import torch.distributed as distributed

    if not distributed.is_available() or not distributed.is_initialized():
        return value
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if distributed.get_backend() == "nccl"
        else torch.device("cpu")
    )
    tensor = torch.tensor(value, dtype=torch.long, device=device)
    distributed.all_reduce(tensor, op=distributed.ReduceOp.MAX)
    return int(tensor.item())


def _loss_witness(log_history) -> dict[str, float]:
    result: dict[str, float] = {}
    for record in log_history:
        for key, value in record.items():
            if key == "loss" or key.startswith("loss/") or key.startswith("train_loss"):
                if isinstance(value, (int, float)):
                    result[key] = float(value)
    if not result:
        raise RuntimeError("trainer did not publish a finite loss witness")
    if not all(value == value and abs(value) != float("inf") for value in result.values()):
        raise RuntimeError("optimizer smoke produced a non-finite loss")
    return result


def _run_stage2(
    args: argparse.Namespace,
    case: str,
    case_dir: Path,
    config,
) -> dict[str, Any]:
    import torch
    from evidence_window.stage2_config import read_deepspeed_zero_stage
    from evidence_window.stage2_layout import extract_stage2_layout, select_stage2_edges
    from evidence_window.stage2_reference import OnlineFrozenStage2Reference
    from stage2_data import build_stage2_record, collate_fn_stage2
    import task_deepseed
    from trainer_stage2 import CustomTrainerStage2

    task_deepseed.TRAIN_IMAGE_ROOT = args.image_root
    zero_stage = (
        read_deepspeed_zero_stage(args.deepspeed_config)
        if args.deepspeed_config
        else 0
    )
    _validate_case_zero_stage(case, zero_stage)
    config.validate_runtime(zero_stage)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor, model, marker_ids = _load_processor_and_model(
        args.stage1_checkpoint, stage="stage2", device_dtype=dtype
    )
    model = _apply_smoke_lora(model)
    for parameter in model.visual.parameters():
        parameter.requires_grad_(False)

    row = _read_source_row(args.data, args.sample_index)
    record = build_stage2_record(row, args.sample_index)
    collator_args = SimpleNamespace(latent_size=8, latent_ce_ratio=0.0)
    batch = collate_fn_stage2([record], processor, collator_args)
    token_ids = _token_ids(processor, marker_ids)
    layout = extract_stage2_layout(
        batch["input_ids"], batch["attention_mask"], batch["labels"], token_ids
    )
    selected_edges = select_stage2_edges(
        layout, config.loo.max_evidence_edges_per_window
    )
    edge_count = len(selected_edges)
    edge_counts_by_window: dict[tuple[int, int], int] = {}
    for edge in selected_edges:
        key = (edge.batch_index, edge.window_index)
        edge_counts_by_window[key] = edge_counts_by_window.get(key, 0) + 1
    eligible_window_count = sum(
        count >= 2 for count in edge_counts_by_window.values()
    )

    reference_provider = None
    reference_identity = None
    reference_model = None
    if config.reference.mode == "precomputed":
        assert args.reference_artifact_dir is not None
        reference_identity = _build_expected_reference_identity(
            args,
            config,
            processor.tokenizer.all_special_ids,
        )
        reference_provider = _build_precomputed_provider(
            args.reference_artifact_dir,
            reference_identity,
        )
    elif config.reference.mode == "online_frozen":
        _, reference_model, _ = _load_processor_and_model(
            args.stage1_checkpoint, stage="stage2", device_dtype=dtype
        )

    training_args = _training_arguments(case_dir, args.deepspeed_config)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=[record],
        data_collator=lambda records: collate_fn_stage2(records, processor, collator_args),
        processing_class=processor.tokenizer,
        evidence_window_config=config,
        latent_token_ids={
            "start": token_ids.latent_start,
            "end": token_ids.latent_end,
            "pad": token_ids.latent_pad,
            "padding": token_ids.padding,
            "special": tuple(token_ids.special),
            "assistant_termination": tuple(token_ids.assistant_termination),
        },
    )
    if reference_provider is not None:
        trainer_kwargs.update(
            reference_provider=reference_provider,
            reference_identity=reference_identity,
        )
    if reference_model is not None:
        trainer_kwargs.update(
            reference_model_factory=lambda model=reference_model: model,
            deepspeed_zero_stage=zero_stage,
            policy_initialization_checkpoint=args.stage1_checkpoint,
        )
    trainer = CustomTrainerStage2(**trainer_kwargs)
    policy_update = _parameter_update_tracker(trainer.model)
    policy_forwards, policy_hook = _forward_witness(trainer.model)
    global_max_edge_slots = _distributed_max_int(edge_count)
    fixed = getattr(trainer, "_stage2_reference_provider", None)
    reference_forwards = {"total": 0, "blocked": 0, "hidden": 0}
    reference_hook = None
    if isinstance(fixed, OnlineFrozenStage2Reference):
        reference_forwards, reference_hook = _forward_witness(fixed.model)

    trainer.train()
    policy_grad = _gradient_witness(trainer.state.log_history)
    policy_hook.remove()
    if reference_hook is not None:
        reference_hook.remove()
    reference_has_grad = bool(
        fixed is not None
        and hasattr(fixed, "parameters")
        and any(parameter.grad is not None for parameter in fixed.parameters())
    )
    result = {
        "case": case,
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "device": str(next(trainer.model.parameters()).device),
        "dtype": str(next(trainer.model.parameters()).dtype),
        "zero_stage": zero_stage,
        "losses": _loss_witness(trainer.state.log_history),
        "edge_count": edge_count,
        "eligible_window_count": eligible_window_count,
        "global_max_edge_slots": global_max_edge_slots,
        "policy_forwards": policy_forwards,
        "reference_forwards": reference_forwards,
        "policy_gradient": policy_grad,
        "policy_parameter_update": policy_update(),
        "reference_has_gradient": reference_has_grad,
        "disabled_auxiliary_forward_count": (
            max(policy_forwards["total"] - 1, 0) if config.is_disabled else None
        ),
    }
    del trainer, model, reference_model, processor, batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _run_stage1(args: argparse.Namespace, case_dir: Path) -> dict[str, Any]:
    import torch
    from evidence_window.config import EvidenceWindowConfig
    from main import collate_fn_stage1
    import task_deepseed
    from task_deepseed import interleaved_latent_cot_preprocess_function
    from trainer import CustomTrainerStage1

    task_deepseed.TRAIN_IMAGE_ROOT = args.image_root
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor, model, marker_ids = _load_processor_and_model(
        args.model, stage="stage1", device_dtype=dtype
    )
    model = _apply_smoke_lora(model)
    for parameter in model.visual.parameters():
        parameter.requires_grad_(False)
    row = _read_source_row(args.data, args.sample_index)
    record = interleaved_latent_cot_preprocess_function(row)
    collator_args = SimpleNamespace(latent_size=8)
    trainer = CustomTrainerStage1(
        model=model,
        args=_training_arguments(case_dir, args.deepspeed_config),
        train_dataset=[record],
        data_collator=lambda records: collate_fn_stage1(records, processor, collator_args),
        tokenizer=processor.tokenizer,
        evidence_window_config=EvidenceWindowConfig.disabled(),
    )
    policy_update = _parameter_update_tracker(trainer.model)
    policy_forwards, policy_hook = _forward_witness(trainer.model)
    teacher_forwards, teacher_hook = _forward_witness(trainer._ema.teacher)
    teacher_grad = any(parameter.grad is not None for parameter in trainer._ema.teacher.parameters())
    trainer.train()
    policy_grad = _gradient_witness(trainer.state.log_history)
    policy_hook.remove()
    teacher_hook.remove()
    teacher_grad = teacher_grad or any(
        parameter.grad is not None for parameter in trainer._ema.teacher.parameters()
    )
    if teacher_grad:
        raise RuntimeError("Stage-1 EMA teacher received gradients")
    if trainer._ema_update_count != 1:
        raise RuntimeError(
            f"Stage-1 EMA update count mismatch: expected 1, got {trainer._ema_update_count}"
        )
    result = {
        "case": "stage1-baseline",
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "device": str(next(trainer.model.parameters()).device),
        "dtype": str(next(trainer.model.parameters()).dtype),
        "losses": _loss_witness(trainer.state.log_history),
        "edge_count": 0,
        "eligible_window_count": 0,
        "global_max_edge_slots": 0,
        "latent_evidence_count": sum(
            step.get("type") == "latent" for step in row["sequence_plan"]
        ),
        "policy_forwards": policy_forwards,
        "teacher_forwards": teacher_forwards,
        "policy_gradient": policy_grad,
        "policy_parameter_update": policy_update(),
        "teacher_has_gradient": teacher_grad,
        "ema_update_count": trainer._ema_update_count,
        "disabled_auxiliary_forward_count": 0,
    }
    del trainer, model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _write_result(output_dir: Path, result: dict[str, Any]) -> Path:
    attempt_id = result.get("attempt_id")
    if not isinstance(attempt_id, str):
        raise ValueError("rank result requires an attempt id")
    destination = output_dir / _result_filename(
        attempt_id, result["case"], result["rank"]
    )
    return _atomic_publish_json_no_overwrite(destination, result)


CASE_LOSS_KEYS = {
    "ew-reference-free": "loss/ew_pref",
    "ew-precomputed-dpo": "loss/ew_pref",
    "online-frozen-zero1": "loss/ew_pref",
    "online-frozen-zero2": "loss/ew_pref",
    "relation-kd": "loss/relation",
    "hidden-residual-kd": "loss/residual",
}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _nonnegative_int(witness: dict[str, Any], key: str) -> int:
    value = witness.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a nonnegative integer")
    return value


def _validate_forward_ledger(
    witness: dict[str, Any],
    key: str,
    *,
    total: int,
    blocked: int,
    hidden: int,
) -> None:
    ledger = witness.get(key)
    if not isinstance(ledger, dict):
        raise ValueError(f"missing {key.replace('_', ' ')} ledger")
    expected = {"total": total, "blocked": blocked, "hidden": hidden}
    if any(ledger.get(name) != count for name, count in expected.items()):
        raise ValueError(
            f"{key.replace('_', ' ')} mismatch: expected {expected}, found {ledger}"
        )


def _validate_policy_learning_witness(witness: dict[str, Any]) -> None:
    gradient = witness.get("policy_gradient")
    gradient_valid = bool(
        isinstance(gradient, dict)
        and gradient.get("seen") is True
        and gradient.get("finite") is True
        and gradient.get("nonzero") is True
    )
    if gradient_valid:
        return
    update = witness.get("policy_parameter_update")
    update_valid = bool(
        isinstance(update, dict)
        and update.get("changed") is True
        and update.get("finite") is True
    )
    if witness.get("zero_stage") in {1, 2} and update_valid:
        return
    raise ValueError(
        "policy gradient witness must be seen, finite, and nonzero; "
        "a finite parameter update is accepted only for ZeRO-1/2 after gradient clearing"
    )


def _validate_case_witness(witness: dict[str, Any]) -> None:
    """Authenticate the small smoke witness before it becomes a success artifact."""
    case = witness.get("case")
    if case not in {"stage1-baseline", *CASE_CONFIGS}:
        raise ValueError(f"unsupported smoke witness case: {case!r}")
    losses = witness.get("losses")
    if not isinstance(losses, dict) or not losses:
        raise ValueError("loss witness must be a nonempty mapping")
    total_keys = [key for key in ("loss", "train_loss") if key in losses]
    if not total_keys or any(not _finite_number(losses[key]) for key in total_keys):
        raise ValueError("recorded total loss must be finite")
    if any(not _finite_number(value) for value in losses.values()):
        raise ValueError("all recorded losses must be finite")

    named_loss = CASE_LOSS_KEYS.get(case)
    if named_loss is not None and (
        named_loss not in losses or not _finite_number(losses[named_loss])
    ):
        raise ValueError(f"named case requires finite {named_loss}")

    edge_count = _nonnegative_int(witness, "edge_count")
    global_max = _nonnegative_int(witness, "global_max_edge_slots")
    eligible_windows = _nonnegative_int(witness, "eligible_window_count")
    _validate_policy_learning_witness(witness)

    if case == "stage1-baseline":
        latent_count = _nonnegative_int(witness, "latent_evidence_count")
        teacher_count = latent_count + int(latent_count > 0)
        if edge_count != 0 or global_max != 0 or eligible_windows != 0:
            raise ValueError("Stage-1 witness cannot contain Stage-2 evidence edges")
        _validate_forward_ledger(
            witness, "policy_forwards", total=1, blocked=0, hidden=0
        )
        _validate_forward_ledger(
            witness,
            "teacher_forwards",
            total=teacher_count,
            blocked=0,
            hidden=teacher_count,
        )
        if witness.get("teacher_has_gradient") is not False:
            raise ValueError("Stage-1 EMA teacher must have no gradients")
        if witness.get("ema_update_count") != 1:
            raise ValueError("Stage-1 smoke requires exactly one EMA update")
        if witness.get("disabled_auxiliary_forward_count") != 0:
            raise ValueError("Stage-1 smoke must not run Stage-2 auxiliary forwards")
        return

    _validate_case_zero_stage(case, witness.get("zero_stage"))
    if witness.get("reference_has_gradient") is not False:
        raise ValueError("frozen Stage-2 reference must have no gradients")
    if global_max < edge_count:
        raise ValueError("global maximum edge slots cannot be smaller than local edge count")

    if named_loss is None:
        _validate_forward_ledger(
            witness, "policy_forwards", total=1, blocked=0, hidden=0
        )
        _validate_forward_ledger(
            witness, "reference_forwards", total=0, blocked=0, hidden=0
        )
        if witness.get("disabled_auxiliary_forward_count") != 0:
            raise ValueError("baseline smoke unexpectedly ran auxiliary forwards")
        return

    if edge_count <= 0 or global_max <= 0:
        raise ValueError("named auxiliary case requires at least one selected evidence edge")
    if case == "relation-kd" and eligible_windows <= 0:
        raise ValueError("relation case requires an eligible window with multiple evidence edges")

    hidden_calls = 1 + global_max if case == "hidden-residual-kd" else 0
    _validate_forward_ledger(
        witness,
        "policy_forwards",
        total=2 + global_max,
        blocked=global_max,
        hidden=hidden_calls,
    )
    online = case.startswith("online-frozen")
    _validate_forward_ledger(
        witness,
        "reference_forwards",
        total=1 + global_max if online else 0,
        blocked=global_max if online else 0,
        hidden=0,
    )


def _result_filename(attempt_id: str, case: str, rank: int) -> str:
    return f"{attempt_id}-{case}-rank-{rank:05d}.json"


def _expected_result_records(
    cases: Sequence[str], world_size: int, attempt_id: str
) -> list[dict[str, Any]]:
    return [
        {
            "case": case,
            "rank": rank,
            "attempt_id": attempt_id,
            "filename": _result_filename(attempt_id, case, rank),
        }
        for case in cases
        for rank in range(world_size)
    ]


def _validate_local_attempt_results(
    results: Sequence[dict[str, Any]],
    cases: Sequence[str],
    *,
    rank: int,
    world_size: int,
    attempt_id: str,
) -> None:
    if tuple(result.get("case") for result in results) != tuple(cases):
        raise ValueError("local result cases do not match the selected attempt cases")
    for result in results:
        if result.get("rank") != rank or result.get("world_size") != world_size:
            raise ValueError("local result rank/world size does not match the attempt")
        if result.get("attempt_id") != attempt_id:
            raise ValueError("local result attempt id does not match the attempt")
        _validate_case_witness(result)


def _validate_completion_precommit(
    output_dir: Path,
    cases: Sequence[str],
    world_size: int,
    attempt_id: str,
) -> list[dict[str, Any]]:
    _validate_attempt_claim(output_dir, attempt_id, cases, world_size)
    result_files = _expected_result_records(cases, world_size, attempt_id)
    missing = [
        record["filename"]
        for record in result_files
        if not (output_dir / record["filename"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "cannot publish completion manifest before every rank result exists: "
            + ", ".join(missing)
        )
    for record in result_files:
        path = output_dir / record["filename"]
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if (
            result.get("attempt_id") != attempt_id
            or result.get("case") != record["case"]
            or result.get("rank") != record["rank"]
            or result.get("world_size") != world_size
        ):
            raise ValueError(
                f"rank result metadata does not match attempt: {record['filename']}"
            )
        _validate_case_witness(result)
    if (output_dir / "completion.json").exists():
        raise FileExistsError("completion manifest already exists")
    return result_files


def _write_completion_manifest(
    output_dir: Path,
    cases: Sequence[str],
    world_size: int,
    attempt_id: str,
    *,
    validated_result_files: Sequence[dict[str, Any]] | None = None,
) -> Path:
    result_files = (
        _validate_completion_precommit(output_dir, cases, world_size, attempt_id)
        if validated_result_files is None
        else list(validated_result_files)
    )
    if result_files != _expected_result_records(cases, world_size, attempt_id):
        raise ValueError("validated result files do not match the completion attempt")
    manifest = {
        "format_version": 1,
        "status": "complete",
        "attempt_id": attempt_id,
        "cases": list(cases),
        "world_size": world_size,
        "result_files": result_files,
    }
    destination = output_dir / "completion.json"
    return _atomic_publish_json_no_overwrite(destination, manifest)


def _finalize_attempt(
    output_dir: Path,
    results: Sequence[dict[str, Any]],
    cases: Sequence[str],
    *,
    rank: int,
    world_size: int,
    attempt_id: str,
    collective_phase=None,
) -> Path | None:
    phase = _run_collective_phase if collective_phase is None else collective_phase
    phase(
        "witness validation",
        lambda: _validate_local_attempt_results(
            results,
            cases,
            rank=rank,
            world_size=world_size,
            attempt_id=attempt_id,
        ),
    )
    phase(
        "rank result publication",
        lambda: [_write_result(output_dir, result) for result in results],
    )
    result_files = phase(
        "completion precommit",
        lambda: _validate_completion_precommit(
            output_dir, cases, world_size, attempt_id
        ),
    )
    if rank != 0:
        return None
    return _write_completion_manifest(
        output_dir,
        cases,
        world_size,
        attempt_id,
        validated_result_files=result_files,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    cases = SINGLE_GPU_CASES if args.case == "all-single-gpu" else (args.case,)
    rank, world_size = _initialize_distributed_from_environment()
    configs = _coordinate_attempt_start(args, cases, output_dir)
    results = []
    for case in cases:
        case_dir = output_dir / case
        result = (
            _run_stage1(args, case_dir)
            if case == "stage1-baseline"
            else _run_stage2(args, case, case_dir, configs[case])
        )
        result["attempt_id"] = args.attempt_id
        results.append(result)
    manifest_path = _finalize_attempt(
        output_dir,
        results,
        cases,
        rank=rank,
        world_size=world_size,
        attempt_id=args.attempt_id,
    )
    if rank == 0:
        print(manifest_path)


if __name__ == "__main__":
    main()
