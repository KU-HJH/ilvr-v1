"""Precompute fixed Stage-1-initialized targets under Stage-2 semantics.

The checkpoint is identified only by explicit user-provided witnesses. This
command never hashes model weights. Rank zero computes the dataset digest once
unless an authenticated digest is supplied, then broadcasts that small value.
"""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import torch
import torch.distributed as distributed

from evidence_window.artifacts import (
    ArtifactIdentity,
    ReferenceSampleTargets,
    ReferenceShardMetadata,
    publish_reference_manifest,
    read_reference_shard_metadata,
    write_reference_shard,
)
from evidence_window.scores import chunked_causal_target_logps
from evidence_window.stage2_effects import (
    Stage2ConditionOutput,
    Stage2EvidenceEffects,
    compute_stage2_effects,
)
from evidence_window.stage2_layout import (
    Stage2InterleavedLayout,
    extract_stage2_layout,
    select_stage2_edges,
    validate_plan_types,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute immutable ILVR Stage-2 evidence-window reference targets."
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--image-root",
        required=True,
        help="Root used to resolve relative user-image paths in the source dataset.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        help="Precompute only this zero-based source row while preserving its global ID.",
    )
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--reference-checkpoint-id", required=True)
    parser.add_argument("--reference-checkpoint-witness", required=True)
    parser.add_argument("--dataset-sha256-witness")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--model-config-id", required=True)
    parser.add_argument("--parser-version", default="stage2-marker-v1")
    parser.add_argument("--contribution-reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument("--max-evidence-edges-per-window", type=int, default=0)
    parser.add_argument("--include-hidden-residuals", action="store_true")
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--latent-ce-ratio", type=float, default=0.0)
    parser.add_argument("--compress-strategy", choices=["average"], default="average")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.max_evidence_edges_per_window < 0:
        raise ValueError("max evidence edges per window must be nonnegative")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers nonnegative")
    if args.batch_size != 1:
        raise ValueError("Stage-2 reference precompute requires model batch size 1")
    if args.sample_index is not None and args.sample_index < 0:
        raise ValueError("sample index must be nonnegative")
    if args.dataset_sha256_witness is not None:
        _validate_sha256(args.dataset_sha256_witness)
    return args


def sha256_file(path: str | Path) -> str:
    """Hash one dataset file in one sequential producer pass."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_identity(
    args: argparse.Namespace,
    *,
    tokenizer_special_tokens: Sequence[int],
    dataset_sha256: str | None,
) -> ArtifactIdentity:
    """Build scoring identity without inspecting the reference checkpoint tree."""
    witness = sha256_file(args.data_path) if dataset_sha256 is None else dataset_sha256
    _validate_sha256(witness)
    return ArtifactIdentity(
        format_version=1,
        dataset_sha256=witness,
        checkpoint_id=args.reference_checkpoint_id,
        checkpoint_witness=args.reference_checkpoint_witness,
        tokenizer_id=args.tokenizer_id,
        tokenizer_special_tokens=tuple(int(value) for value in tokenizer_special_tokens),
        parser_version=args.parser_version,
        contribution_reduction=args.contribution_reduction,
        intervention="window_attention_block",
        attention_backend="sdpa",
        max_evidence_edges_per_window=args.max_evidence_edges_per_window,
        model_config_id=args.model_config_id,
    )


def reference_rows_from_effects(
    sample_indices: Sequence[int] | torch.Tensor,
    layout: Stage2InterleavedLayout,
    effects: Stage2EvidenceEffects,
) -> dict[int, ReferenceSampleTargets]:
    """Detach one padded effect batch into stable, unpadded source-row records."""
    if isinstance(sample_indices, torch.Tensor):
        if sample_indices.ndim != 1:
            raise ValueError("sample indices must have shape [B]")
        indices = tuple(sample_indices.detach().cpu().tolist())
    else:
        indices = tuple(sample_indices)
    if len(indices) != layout.causal_pair_mask.shape[0]:
        raise ValueError("sample indices must have one entry per effect row")
    if len(set(indices)) != len(indices):
        raise ValueError("a precompute batch cannot contain duplicate sample indices")
    if effects.contributions.shape != layout.causal_pair_mask.shape:
        raise ValueError("effect contributions do not match the current layout")
    if effects.pair_mask.shape != layout.causal_pair_mask.shape:
        raise ValueError("effect pair mask does not match the current layout")

    result: dict[int, ReferenceSampleTargets] = {}
    for batch_index, sample_index in enumerate(indices):
        if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
            raise ValueError("sample indices must be nonnegative integers")
        evidence_count = int(layout.evidence_valid[batch_index].sum().item())
        window_count = int(layout.window_valid[batch_index].sum().item())
        contribution = effects.contributions[
            batch_index, :evidence_count, :window_count
        ].detach().cpu().contiguous()
        pair_mask = effects.pair_mask[
            batch_index, :evidence_count, :window_count
        ].detach().cpu().contiguous()
        hidden = None
        if effects.hidden_residuals is not None:
            hidden = effects.hidden_residuals[
                batch_index, :evidence_count, :window_count
            ].detach().cpu().contiguous()
        result[sample_index] = ReferenceSampleTargets(contribution, pair_mask, hidden)
    return result


_T = TypeVar("_T")


def run_distributed_phase(
    name: str,
    operation: Callable[[], _T],
    *,
    device: torch.device | None = None,
) -> _T:
    """Run local work, then collectively agree whether any rank may continue."""
    result: _T | None = None
    local_error: str | None = None
    try:
        result = operation()
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    if distributed.is_available() and distributed.is_initialized():
        collective_device = _collective_device() if device is None else device
        failed = torch.tensor(
            int(local_error is not None), dtype=torch.long, device=collective_device
        )
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
        raise RuntimeError(f"{name} failed: {local_error}")

    return result  # type: ignore[return-value]


def _distributed_dataset_witness(args: argparse.Namespace, rank: int) -> str:
    if args.dataset_sha256_witness is not None:
        return args.dataset_sha256_witness
    witness = run_distributed_phase(
        "dataset witness",
        lambda: sha256_file(args.data_path) if rank == 0 else None,
    )
    if distributed.is_available() and distributed.is_initialized():
        values = [witness]
        distributed.broadcast_object_list(values, src=0)
        witness = values[0]
    if not isinstance(witness, str):
        raise RuntimeError("rank zero did not publish the dataset SHA256 witness")
    _validate_sha256(witness)
    return witness


def precompute(args: argparse.Namespace) -> None:
    """Execute one fixed-reference pass per selected rank-local source row."""
    _initialize_distributed_if_requested()
    rank = distributed.get_rank() if distributed.is_initialized() else 0
    world_size = distributed.get_world_size() if distributed.is_initialized() else 1
    artifact_dir = Path(args.artifact_dir)
    run_distributed_phase(
        "artifact precheck",
        lambda: _precheck_artifact(artifact_dir) if rank == 0 else None,
    )
    run_distributed_phase("input precheck", lambda: _precheck_inputs(args))
    dataset_witness = _distributed_dataset_witness(args, rank)
    local_records = run_distributed_phase(
        "record preparation",
        lambda: _load_rank_records(args, rank=rank, world_size=world_size),
    )
    identity, _ = run_distributed_phase(
        "shard production",
        lambda: _produce_rank_shard(
            args,
            rank=rank,
            world_size=world_size,
            artifact_dir=artifact_dir,
            dataset_witness=dataset_witness,
            local_records=local_records,
        ),
    )
    run_distributed_phase(
        "manifest publication",
        lambda: _publish_rank_manifest(
            artifact_dir, identity, world_size
        ) if rank == 0 else None,
    )


def _precheck_artifact(artifact_dir: Path) -> None:
    if not os.path.lexists(artifact_dir):
        return
    if (
        artifact_dir.is_symlink()
        or not artifact_dir.is_dir()
        or any(artifact_dir.iterdir())
    ):
        raise FileExistsError(
            "reference artifact destination must be an empty directory; "
            "refusing a nonempty, non-directory, or symlink destination"
        )


def _precheck_inputs(args: argparse.Namespace) -> None:
    image_root = getattr(args, "image_root", None)
    if not image_root or not Path(image_root).is_dir():
        raise ValueError("image root must name an existing directory")


def _selected_row_indices(
    row_count: int,
    *,
    rank: int,
    world_size: int,
    sample_index: int | None = None,
) -> tuple[int, ...]:
    """Return stable global row IDs owned by this rank."""
    if sample_index is not None:
        if sample_index < 0 or sample_index >= row_count:
            raise IndexError(
                f"sample index {sample_index} is outside dataset with {row_count} rows"
            )
        return (sample_index,) if sample_index % world_size == rank else ()
    return tuple(range(rank, row_count, world_size))


def _load_rank_records(
    args: argparse.Namespace, *, rank: int, world_size: int
) -> list[dict]:
    """Load rank-local records after installing the explicit image root."""
    import task_deepseed
    from stage2_data import build_stage2_record
    from utils_deepseed import load_jsonl_dataset

    task_deepseed.TRAIN_IMAGE_ROOT = str(Path(args.image_root).resolve())
    raw = load_jsonl_dataset(args.data_path)
    return [
        build_stage2_record(raw[row_index], row_index)
        for row_index in _selected_row_indices(
            len(raw),
            rank=rank,
            world_size=world_size,
            sample_index=args.sample_index,
        )
    ]


def _prepare_reference_model(model, device: torch.device):
    """Place the frozen scorer on one deterministic execution dtype."""
    execution_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=execution_dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _produce_rank_shard(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    artifact_dir: Path,
    dataset_witness: str,
    local_records: Sequence[dict],
) -> tuple[ArtifactIdentity, ReferenceShardMetadata]:
    """Load one fixed reference and seal this rank's deterministic rows."""

    # Heavy repository dependencies remain local so identity/schema utilities
    # can be tested and inspected on CPU-only hosts.
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    from stage2_data import collate_fn_stage2

    checkpoint = args.reference_checkpoint
    processor = AutoProcessor.from_pretrained(
        checkpoint, cache_dir=args.cache_dir, trust_remote_code=True
    )
    processor.tokenizer.add_tokens(
        ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"],
        special_tokens=True,
    )
    model_config = Qwen2_5_VLConfig.from_pretrained(
        checkpoint, cache_dir=args.cache_dir, trust_remote_code=True
    )
    model_config.stage = "stage2"
    model_config.compress_strategy = args.compress_strategy
    model_config.latent_size = args.latent_size
    latent_pad = _single_token_id(processor, "<|latent_pad|>")
    latent_start = _single_token_id(processor, "<|latent_start|>")
    latent_end = _single_token_id(processor, "<|latent_end|>")
    padding = _single_token_id(processor, "<|endoftext|>")
    termination = _single_token_id(processor, "<|im_end|>")
    model_config.latent_token_id = latent_pad
    model_config.latent_start_id = latent_start
    model_config.latent_end_id = latent_end

    device = torch.device(
        "cuda", int(os.environ.get("LOCAL_RANK", 0))
    ) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        config=model_config,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="flash_attention_2" if device.type == "cuda" else "sdpa",
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    _prepare_reference_model(model, device)

    identity = build_artifact_identity(
        args,
        tokenizer_special_tokens=processor.tokenizer.all_special_ids,
        dataset_sha256=dataset_witness,
    )
    loader = DataLoader(
        local_records,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn_stage2, processor=processor, args=args),
    )
    token_ids = _token_ids(
        processor, latent_start, latent_end, latent_pad, padding, termination
    )
    rank_targets: dict[int, ReferenceSampleTargets] = {}
    with torch.no_grad():
        for batch in loader:
            sample_indices = batch.pop("ew_sample_indices")
            plan_types = batch.pop("ew_plan_types")
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            layout = extract_stage2_layout(
                batch["input_ids"], batch["attention_mask"], batch["labels"], token_ids
            )
            validate_plan_types(layout, plan_types, sample_indices)
            labels = batch.pop("labels")

            def fixed_forward(block_mask, need_hidden):
                output = model(
                    **batch,
                    attention_block_mask=block_mask,
                    attention_backend_override="sdpa",
                    output_hidden_states=need_hidden,
                    return_dict=True,
                    use_cache=False,
                )
                shifted = chunked_causal_target_logps(output.logits, labels)
                token_logps = torch.cat(
                    (shifted.new_zeros((shifted.shape[0], 1)), shifted), dim=1
                )
                hidden = None
                if need_hidden:
                    values = output.hidden_states
                    hidden = values[-1] if isinstance(values, (tuple, list)) else values
                return Stage2ConditionOutput(token_logps, hidden)

            effects = compute_stage2_effects(
                layout,
                select_stage2_edges(layout, args.max_evidence_edges_per_window),
                fixed_forward,
                reduction=args.contribution_reduction,
                need_hidden=args.include_hidden_residuals,
            )
            rows = reference_rows_from_effects(sample_indices, layout, effects)
            overlap = set(rank_targets) & set(rows)
            if overlap:
                raise ValueError(f"duplicate precompute sample indices: {sorted(overlap)}")
            rank_targets.update(rows)

    metadata = write_reference_shard(
        artifact_dir,
        identity,
        rank_targets,
        rank=rank,
        publish_manifest=False,
    )
    return identity, metadata


def _publish_rank_manifest(
    artifact_dir: Path,
    identity: ArtifactIdentity,
    world_size: int,
) -> None:
    metadata = [
        read_reference_shard_metadata(
            artifact_dir / f"shard-rank-{shard_rank:05d}.metadata.json"
        )
        for shard_rank in range(world_size)
    ]
    publish_reference_manifest(artifact_dir, identity, metadata)


def _token_ids(processor, start, end, pad, padding, termination):
    from evidence_window.layout import TokenIds

    return TokenIds(
        latent_start=start,
        latent_end=end,
        latent_pad=pad,
        padding=padding,
        special=frozenset(int(value) for value in processor.tokenizer.all_special_ids),
        assistant_termination=frozenset({termination}),
    )


def _single_token_id(processor, text: str) -> int:
    values = processor.tokenizer(text, return_tensors="pt")["input_ids"]
    if values.numel() != 1:
        raise ValueError(f"{text} must map to exactly one token")
    return int(values.item())


def _initialize_distributed_if_requested() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not distributed.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")


def _collective_device() -> torch.device:
    if distributed.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _validate_sha256(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("dataset SHA256 witness must be 64 lowercase hexadecimal characters")


def main(argv: Sequence[str] | None = None) -> None:
    precompute(parse_args(argv))


if __name__ == "__main__":
    main()
