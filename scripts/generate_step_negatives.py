#!/usr/bin/env python3
"""Generate deterministic step-wise modality negatives from one frozen Stage 2A."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable, Iterable

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from stage2.checkpointing import (
    SPARSE_TOKEN_ROWS_FILENAME,
    STAGE2_PEFT_METADATA_FILENAME,
    Stage2ArtifactKind,
    decode_stage2_peft_metadata_snapshot,
    inspect_stage2_artifact,
    resolve_latent_token_ids,
)
from stage2.data import (
    Stage2DataCollator,
    load_jsonl_examples,
    prepare_stage2_example,
)
from stage2.peft import load_stage2_model_for_inference, resolve_stage2_qwen_model
from step_preference.negatives import (
    CheckpointFile,
    StepNegative,
    filter_negative,
    greedy_generate_step_negative,
    load_negative_store,
    score_step_completion,
    tokenize_step_completion,
    write_negative_artifact,
)
from step_preference.spans import build_assistant_plan, reject_duplicate_sample_ids


_PROCESSOR_FILES = frozenset(
    {
        "added_tokens.json",
        "chat_template.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    }
)
_FULL_CONFIG_FILES = frozenset({"config.json", "generation_config.json"})
_FULL_WEIGHT_FILES = frozenset(
    {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
)


@dataclass(frozen=True)
class LoadedStage2ASnapshot:
    """One eval-only model/processor pair and its exact local artifact identity."""

    model: object
    processor: object
    checkpoint_files: tuple[CheckpointFile, ...]
    tokenizer_config_identity: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate offline greedy visual/textual step negatives"
    )
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--stage2a_checkpoint_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--min_tokens", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--attention_implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def _read_weight_index(checkpoint: Path, name: str) -> tuple[str, ...]:
    path = checkpoint / name
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid full-checkpoint weight index: {path}") from error
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"full-checkpoint weight index has no weight_map: {path}")
    shard_names = set()
    for shard_name in weight_map.values():
        candidate = Path(shard_name) if isinstance(shard_name, str) else Path()
        if (
            not isinstance(shard_name, str)
            or not shard_name
            or candidate.is_absolute()
            or len(candidate.parts) != 1
            or candidate.name != shard_name
        ):
            raise ValueError("weight index contains an unsafe shard path")
        shard_names.add(shard_name)
    return tuple(sorted(shard_names))


def _loader_relative_paths(
    checkpoint: Path, artifact_kind: Stage2ArtifactKind
) -> tuple[str, ...]:
    names = {
        entry.name
        for entry in checkpoint.iterdir()
        if entry.is_file() and not entry.is_symlink()
    }
    processor = names & _PROCESSOR_FILES
    if artifact_kind is Stage2ArtifactKind.PEFT:
        model_names = {
            STAGE2_PEFT_METADATA_FILENAME,
            "adapter_config.json",
            "adapter_model.safetensors",
        }
        if SPARSE_TOKEN_ROWS_FILENAME in names:
            model_names.add(SPARSE_TOKEN_ROWS_FILENAME)
        missing = model_names - names
        if missing:
            raise ValueError(f"Stage 2A PEFT loader files are missing: {sorted(missing)}")
    else:
        model_names = names & (_FULL_CONFIG_FILES | _FULL_WEIGHT_FILES)
        if "config.json" not in model_names:
            raise ValueError("full Stage 2A loader requires config.json")
        for index_name in (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ):
            if index_name in model_names:
                model_names.update(_read_weight_index(checkpoint, index_name))
        missing = model_names - names
        if missing:
            raise ValueError(f"full Stage 2A weight shards are missing: {sorted(missing)}")
        has_weights = bool(
            model_names
            & {
                "model.safetensors",
                "pytorch_model.bin",
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            }
        )
        if not has_weights:
            raise ValueError("full Stage 2A loader requires model weights")
    return tuple(sorted(processor | model_names))


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _checkpoint_records(
    checkpoint: Path, relative_paths: Iterable[str]
) -> tuple[CheckpointFile, ...]:
    records = []
    for relative_path in sorted(relative_paths):
        path = checkpoint / relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint loader input must be a regular file: {path}")
        size, digest = _sha256_file(path)
        records.append(CheckpointFile(relative_path, size, digest))
    return tuple(records)


def _tokenizer_identity(records: Iterable[CheckpointFile]) -> str:
    payload = [
        {
            "relative_path": entry.relative_path,
            "sha256": entry.sha256,
            "size": entry.size,
        }
        for entry in records
        if entry.relative_path in _PROCESSOR_FILES
    ]
    if not payload:
        raise ValueError("Stage 2A checkpoint contains no processor/tokenizer files")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_frozen_stage2a_snapshot(
    *,
    checkpoint_path: str | Path,
    attention_implementation: str,
    device: str,
    inspect_artifact: Callable = inspect_stage2_artifact,
    load_processor: Callable | None = None,
    load_full_model: Callable | None = None,
    load_peft_model: Callable | None = None,
) -> LoadedStage2ASnapshot:
    """Load a local full Stage 2 checkpoint directly, or its validated PEFT form."""
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("Stage 2A checkpoint path must be a regular directory")
    inspection = inspect_artifact(checkpoint)
    relative_paths = _loader_relative_paths(checkpoint, inspection.artifact_kind)
    checkpoint_files = _checkpoint_records(checkpoint, relative_paths)

    if load_processor is None:
        load_processor = lambda source: AutoProcessor.from_pretrained(
            source, local_files_only=True
        )
    processor = load_processor(str(checkpoint))
    dtype = torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16
    model_options = {
        "attn_implementation": attention_implementation,
        "device_map": device,
        "torch_dtype": dtype,
    }
    if inspection.artifact_kind is Stage2ArtifactKind.FULL:
        if load_full_model is None:
            load_full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained
        model = load_full_model(
            str(checkpoint), local_files_only=True, **model_options
        )
    else:
        metadata_snapshot = inspection.peft_metadata_snapshot
        if metadata_snapshot is None:
            raise ValueError("Stage 2A PEFT inspection lost its metadata")
        metadata = decode_stage2_peft_metadata_snapshot(metadata_snapshot)
        if load_peft_model is None:
            load_peft_model = load_stage2_model_for_inference
        loaded = load_peft_model(
            metadata.base_model_reference,
            str(checkpoint),
            processor,
            source_provenance_kwargs={"local_files_only": True},
            model_load_kwargs=model_options,
            artifact_inspection=inspection,
        )
        model = resolve_stage2_qwen_model(loaded)
    model.eval()
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            parameter.requires_grad_(False)

    return LoadedStage2ASnapshot(
        model=model,
        processor=processor,
        checkpoint_files=checkpoint_files,
        tokenizer_config_identity=_tokenizer_identity(checkpoint_files),
    )


def _load_images(paths: tuple[str, ...]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.load()
        images.append(image)
    return images


def _move_model_inputs(batch, device: torch.device) -> dict[str, torch.Tensor]:
    names = (
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    )
    return {
        name: batch[name].to(device)
        for name in names
        if name in batch and batch[name] is not None
    }


def _score_encoding(model, encoded, device: torch.device) -> float:
    return score_step_completion(
        model,
        model_inputs={
            name: value.to(device) for name, value in encoded.model_inputs.items()
        },
        completion_token_positions=encoded.completion_token_positions,
    )


def _encode_candidate(processor, example, images, assistant_prefix, text):
    if not text:
        return None
    return tokenize_step_completion(
        processor,
        conversation=example.conversation,
        images=images,
        assistant_prefix=assistant_prefix,
        completion_text=text,
    )


def generate_negative_rows(
    *,
    model,
    processor,
    examples,
    latent_size: int,
    min_tokens: int,
    max_new_tokens: int,
    attention_implementation: str,
    device: torch.device,
) -> tuple[StepNegative, ...]:
    """Generate and clean-score every reasoning step in selected examples."""
    token_ids = resolve_latent_token_ids(processor.tokenizer, model.config)
    configured_latent_size = getattr(model.config, "latent_size", None)
    if configured_latent_size != latent_size:
        raise ValueError(
            "Stage 2A latent_size mismatch: "
            f"checkpoint={configured_latent_size}, requested={latent_size}"
        )
    collator = Stage2DataCollator(
        processor=processor,
        token_ids=token_ids,
        latent_size=latent_size,
        supervise_latent_boundaries=bool(
            getattr(model.config, "supervise_latent_boundaries", False)
        ),
        step_preference_enabled=True,
    )
    rows: list[StepNegative] = []
    for example in examples:
        batch = collator([example])
        model_inputs = _move_model_inputs(batch, device)
        span_by_id = {span.step_id: span for span in batch["reasoning_step_spans"]}
        plan_by_id = {step.step_id: step for step in example.reasoning_steps}
        if set(span_by_id) != set(plan_by_id):
            raise ValueError("reasoning plan/token step identities do not match")
        assistant_text = example.conversation[1]["content"][0]["text"]
        images = _load_images(example.original_image_paths)
        for step_id in sorted(span_by_id):
            span = span_by_id[step_id]
            plan = plan_by_id[step_id]
            assistant_prefix = assistant_text[: plan.target_text.start]
            chosen_text = assistant_text[plan.target_text.start : plan.target_text.end]
            visual = greedy_generate_step_negative(
                model,
                full_model_inputs=model_inputs,
                step_span=span,
                modality="visual",
                tokenizer=processor.tokenizer,
                max_new_tokens=max_new_tokens,
                attn_implementation=attention_implementation,
            )
            textual = greedy_generate_step_negative(
                model,
                full_model_inputs=model_inputs,
                step_span=span,
                modality="textual",
                tokenizer=processor.tokenizer,
                max_new_tokens=max_new_tokens,
                attn_implementation=attention_implementation,
            )
            chosen_encoding = tokenize_step_completion(
                processor,
                conversation=example.conversation,
                images=images,
                assistant_prefix=assistant_prefix,
                completion_text=chosen_text,
            )
            visual_encoding = _encode_candidate(
                processor, example, images, assistant_prefix, visual.text
            )
            textual_encoding = _encode_candidate(
                processor, example, images, assistant_prefix, textual.text
            )
            valid_visual = filter_negative(
                chosen_text=chosen_text,
                rejected_text=visual.text,
                scoreable_token_ids=(
                    visual_encoding.completion_token_ids if visual_encoding else ()
                ),
                min_tokens=min_tokens,
            )
            valid_textual = filter_negative(
                chosen_text=chosen_text,
                rejected_text=textual.text,
                scoreable_token_ids=(
                    textual_encoding.completion_token_ids if textual_encoding else ()
                ),
                min_tokens=min_tokens,
            )
            chosen_score = _score_encoding(model, chosen_encoding, device)
            visual_score = (
                _score_encoding(model, visual_encoding, device)
                if valid_visual
                else None
            )
            textual_score = (
                _score_encoding(model, textual_encoding, device)
                if valid_textual
                else None
            )
            rows.append(
                StepNegative(
                    sample_id=example.sample_id,
                    step_id=step_id,
                    chosen_text=chosen_text,
                    rejected_visual_text=visual.text,
                    rejected_textual_text=textual.text,
                    valid_visual_negative=valid_visual,
                    valid_textual_negative=valid_textual,
                    reference_chosen_score=chosen_score,
                    reference_visual_rejected_score=visual_score,
                    reference_textual_rejected_score=textual_score,
                )
            )
    return tuple(rows)


def _selected_examples(data_path: Path, image_root: Path, latent_size: int, max_samples):
    samples = load_jsonl_examples(data_path)
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("negative generation selected no source rows")
    for sample in samples:
        sequence_plan = sample.get("sequence_plan")
        if not isinstance(sequence_plan, list):
            raise ValueError("sequence_plan must be a list")
        build_assistant_plan(
            sequence_plan, latent_size, require_reasoning_units=True
        )
    examples = tuple(
        prepare_stage2_example(sample, image_root=image_root, latent_size=latent_size)
        for sample in samples
    )
    reject_duplicate_sample_ids(example.sample_id for example in examples)
    return examples


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if args.latent_size <= 0:
        raise ValueError("latent_size must be positive")
    if args.min_tokens <= 0:
        raise ValueError("min_tokens must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    device = torch.device(args.device)
    data_path = args.data_path.resolve(strict=True)
    image_root = args.image_root.resolve(strict=True)
    checkpoint_path = args.stage2a_checkpoint_path.resolve(strict=True)
    if not data_path.is_file():
        raise ValueError("data_path must be a regular file")
    if not image_root.is_dir():
        raise ValueError("image_root must be a directory")

    source_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    examples = _selected_examples(
        data_path, image_root, args.latent_size, args.max_samples
    )
    snapshot = load_frozen_stage2a_snapshot(
        checkpoint_path=checkpoint_path,
        attention_implementation=args.attention_implementation,
        device=args.device,
    )
    rows = generate_negative_rows(
        model=snapshot.model,
        processor=snapshot.processor,
        examples=examples,
        latent_size=args.latent_size,
        min_tokens=args.min_tokens,
        max_new_tokens=args.max_new_tokens,
        attention_implementation=args.attention_implementation,
        device=device,
    )
    if not rows:
        raise ValueError("selected source rows contain no reasoning steps")
    selected_row_indices = tuple(range(len(examples)))
    generation_settings = {
        "attention_implementation": args.attention_implementation,
        "max_new_tokens": args.max_new_tokens,
        "min_tokens": args.min_tokens,
    }
    write_negative_artifact(
        negative_path=args.output_path,
        manifest_path=args.manifest_path,
        negatives=rows,
        source_path=data_path,
        source_sha256=source_sha256,
        selected_row_indices=selected_row_indices,
        stage2a_checkpoint_path=checkpoint_path,
        checkpoint_files=snapshot.checkpoint_files,
        tokenizer_config_identity=snapshot.tokenizer_config_identity,
        latent_size=args.latent_size,
        generation_settings=generation_settings,
    )
    # One post-publication read is the complete consumer-side validation gate.
    load_negative_store(
        negative_path=args.output_path,
        manifest_path=args.manifest_path,
        source_path=data_path,
        expected_source_sha256=source_sha256,
        expected_selected_row_indices=selected_row_indices,
        expected_stage2a_checkpoint_path=checkpoint_path,
        expected_checkpoint_files=snapshot.checkpoint_files,
        expected_tokenizer_config_identity=snapshot.tokenizer_config_identity,
        expected_latent_size=args.latent_size,
        expected_generation_settings=generation_settings,
        expected_keys=((row.sample_id, row.step_id) for row in rows),
    )
    valid_pairs = sum(
        row.valid_visual_negative + row.valid_textual_negative for row in rows
    )
    if valid_pairs == 0:
        print("no valid visual/textual negative pair survived", file=sys.stderr)
        return 2
    print(f"wrote {len(rows)} step rows with {valid_pairs} valid pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
