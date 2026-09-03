"""Strict offline negative artifacts for Stage 2B preference training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from step_preference.ablation import (
    StepAblationSpec,
    build_additive_step_attention_mask,
)
from step_preference.spans import ReasoningStepSpan, expand_qwen_image_placeholders
from stage2.checkpointing import (
    SPARSE_TOKEN_ROWS_FILENAME,
    STAGE2_PEFT_METADATA_FILENAME,
    Stage2ArtifactKind,
    decode_stage2_peft_metadata_snapshot,
    inspect_stage2_artifact,
)


NEGATIVE_MANIFEST_SCHEMA_VERSION = 2
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
class GreedyStepCompletion:
    """Text and non-control token IDs produced by one ablated greedy branch."""

    text: str
    token_ids: tuple[int, ...]
    stop_reason: str


@dataclass(frozen=True)
class StepCompletionEncoding:
    """One complete rendered scoring sequence and its completion targets."""

    model_inputs: Mapping[str, torch.Tensor]
    completion_token_positions: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    rendered_text: str


_MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
)


def _require_eval_model(model) -> None:
    if getattr(model, "training", None) is not False:
        raise ValueError("frozen reference model must be in eval mode")


def _require_positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _token_id_set(name: str, values) -> frozenset[int]:
    if values is None:
        return frozenset()
    if isinstance(values, int) and not isinstance(values, bool):
        values = (values,)
    try:
        result = frozenset(values)
    except TypeError as error:
        raise ValueError(f"{name} must contain integer token IDs") from error
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in result
    ):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return result


def _model_dtype(model) -> torch.dtype:
    dtype = getattr(model, "dtype", None)
    if isinstance(dtype, torch.dtype) and dtype.is_floating_point:
        return dtype
    parameter = next(model.parameters(), None)
    if parameter is None or not parameter.dtype.is_floating_point:
        raise ValueError("model must expose a floating compute dtype")
    return parameter.dtype


def _one_sample_inputs(model_inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(model_inputs, Mapping):
        raise ValueError("model_inputs must be a mapping")
    if "input_ids" not in model_inputs:
        raise ValueError("model_inputs must contain input_ids")
    inputs = {
        name: model_inputs[name]
        for name in _MODEL_INPUT_KEYS
        if name in model_inputs and model_inputs[name] is not None
    }
    input_ids = inputs["input_ids"]
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
    ):
        raise ValueError("input_ids must have shape [1, sequence_length]")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        inputs["attention_mask"] = torch.ones_like(input_ids)
    elif (
        not isinstance(attention_mask, torch.Tensor)
        or tuple(attention_mask.shape) != tuple(input_ids.shape)
    ):
        raise ValueError("attention_mask must match input_ids")
    return inputs


def _decode_completion(tokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def greedy_generate_step_negative(
    model,
    *,
    full_model_inputs: Mapping[str, torch.Tensor],
    step_span: ReasoningStepSpan,
    modality: str,
    tokenizer,
    max_new_tokens: int,
    attn_implementation: str,
    eos_token_ids: int | Sequence[int] | None = None,
    control_token_ids: Sequence[int] | None = None,
    dummy_token_id: int | None = None,
) -> GreedyStepCompletion:
    """Greedily decode one reasoning step with exactly one modality edge hidden.

    The full validated trajectory is sliced immediately before ``T_k``.  One
    terminal dummy token keeps the Qwen Stage 2 full-sequence validator intact;
    logits are selected from the physical row immediately before that dummy.
    Causality therefore makes the dummy invisible, and it advances by one
    position after every generated token. Because no cache is used, every mask
    continues to block all recomputed target query rows from the original
    ``T_k`` boundary through the current prediction row.
    """
    _require_eval_model(model)
    _require_positive_int("max_new_tokens", max_new_tokens)
    if not isinstance(step_span, ReasoningStepSpan):
        raise ValueError("step_span must be a ReasoningStepSpan")
    if modality not in ("visual", "textual"):
        raise ValueError("modality must be 'visual' or 'textual'")

    inputs = _one_sample_inputs(full_model_inputs)
    full_ids = inputs["input_ids"]
    full_attention = inputs["attention_mask"]
    prefix_end = step_span.target_text_start
    if prefix_end < 1 or prefix_end > full_ids.shape[1]:
        raise ValueError("target step has no scoreable clean-prefix query row")
    source_start, source_end = (
        (step_span.latent_start, step_span.latent_end)
        if modality == "visual"
        else (step_span.prev_text_start, step_span.prev_text_end)
    )
    if source_start < 0 or source_end <= source_start or source_end > prefix_end:
        raise ValueError("ablation source must be a nonempty span in the clean prefix")

    eos_ids = _token_id_set(
        "eos_token_ids",
        eos_token_ids
        if eos_token_ids is not None
        else getattr(tokenizer, "eos_token_id", None),
    )
    controls = _token_id_set(
        "control_token_ids",
        control_token_ids
        if control_token_ids is not None
        else getattr(tokenizer, "all_special_ids", ()),
    ) - eos_ids
    if dummy_token_id is None:
        dummy_token_id = getattr(tokenizer, "pad_token_id", None)
        if dummy_token_id is None:
            dummy_token_id = next(iter(eos_ids), None)
    if (
        isinstance(dummy_token_id, bool)
        or not isinstance(dummy_token_id, int)
        or dummy_token_id < 0
    ):
        raise ValueError("dummy_token_id must be a nonnegative integer")

    prefix_ids = full_ids[:, :prefix_end]
    prefix_attention = full_attention[:, :prefix_end]
    generated: list[int] = []
    stop_reason = "max_new_tokens"
    model_dtype = _model_dtype(model)
    device = full_ids.device
    for _ in range(max_new_tokens):
        suffix = torch.tensor(
            [generated + [dummy_token_id]], dtype=full_ids.dtype, device=device
        )
        current_ids = torch.cat((prefix_ids, suffix), dim=1)
        current_attention = torch.cat(
            (
                prefix_attention,
                torch.ones(
                    (1, suffix.shape[1]),
                    dtype=full_attention.dtype,
                    device=device,
                ),
            ),
            dim=1,
        )
        dummy_position = current_ids.shape[1] - 1
        query_position = dummy_position - 1
        spec = StepAblationSpec(
            modality=modality,
            target_start=prefix_end,
            target_end=dummy_position + 1,
            source_start=source_start,
            source_end=source_end,
        )
        final_mask = build_additive_step_attention_mask(
            current_attention,
            spec,
            dtype=model_dtype,
            device=device,
            attn_implementation=attn_implementation,
        )
        call_inputs = {
            name: value
            for name, value in inputs.items()
            if name not in ("input_ids", "attention_mask")
        }
        call_inputs.update(input_ids=current_ids, attention_mask=current_attention)
        with torch.inference_mode():
            outputs = model(
                **call_inputs,
                use_cache=False,
                stage2_final_attention_mask=final_mask,
                stage2_logit_positions=torch.tensor(
                    [query_position], dtype=torch.long, device=device
                ),
                stage2_selective_logits_only=True,
            )
        logits = getattr(outputs, "stage2_selected_logits", None)
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape[:2]) != (1, 1):
            raise ValueError("Stage 2 model did not return one selected logit row")
        next_token = int(torch.argmax(logits[0, 0].float()).item())
        if next_token in eos_ids:
            stop_reason = "eos"
            break
        if next_token in controls:
            stop_reason = "control"
            break
        generated.append(next_token)

    return GreedyStepCompletion(
        text=_decode_completion(tokenizer, generated),
        token_ids=tuple(generated),
        stop_reason=stop_reason,
    )


def _assistant_text_conversation(conversation, assistant_text: str):
    copied = deepcopy(conversation)
    if (
        not isinstance(copied, list)
        or len(copied) != 2
        or not isinstance(copied[1], dict)
        or copied[1].get("role") != "assistant"
        or not isinstance(copied[1].get("content"), list)
        or len(copied[1]["content"]) != 1
        or copied[1]["content"][0].get("type") != "text"
    ):
        raise ValueError("conversation must contain one final assistant text part")
    copied[1]["content"][0]["text"] = assistant_text
    return copied


def _special_token_ids(tokenizer) -> frozenset[int]:
    values = list(getattr(tokenizer, "all_special_ids", ()) or ())
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        values.extend(
            eos_token_id
            if isinstance(eos_token_id, (list, tuple))
            else [eos_token_id]
        )
    return _token_id_set("tokenizer special token IDs", values)


def tokenize_step_completion(
    processor,
    *,
    conversation,
    images,
    assistant_prefix: str,
    completion_text: str,
) -> StepCompletionEncoding:
    """Render/tokenize clean prefix, completion, and assistant terminator together."""
    if not isinstance(assistant_prefix, str):
        raise ValueError("assistant_prefix must be a string")
    if not isinstance(completion_text, str) or not completion_text:
        raise ValueError("completion_text must be a non-empty string")
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not getattr(tokenizer, "is_fast", False):
        raise ValueError("completion scoring requires a fast tokenizer")

    assistant_text = assistant_prefix + completion_text
    rendered = processor.apply_chat_template(
        _assistant_text_conversation(conversation, assistant_text),
        tokenize=False,
        add_generation_prompt=False,
    )
    batch = processor(
        text=[rendered],
        images=[images],
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    expanded = rendered
    if "image_grid_thw" in batch:
        expanded = expand_qwen_image_placeholders(
            rendered,
            image_grid_thw=batch["image_grid_thw"],
            image_token=processor.image_token,
            merge_size=processor.image_processor.merge_size,
        )
    offset_batch = tokenizer(
        [expanded],
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offset_mapping = offset_batch.pop("offset_mapping")
    for name, value in offset_batch.items():
        if name not in batch or not torch.equal(value, batch[name]):
            raise ValueError(f"offset tokenization does not match processor tensor: {name}")

    assistant_start = expanded.rfind(assistant_text)
    if assistant_start < 0:
        raise ValueError("assistant completion is absent from rendered chat")
    completion_start = assistant_start + len(assistant_prefix)
    completion_end = completion_start + len(completion_text)
    offsets = offset_mapping.tolist()
    input_ids = batch["input_ids"]
    if len(offsets) != 1 or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError("completion tokenization supports batch_size=1 only")
    excluded_ids = _special_token_ids(tokenizer)
    selected: list[int] = []
    for position, pair in enumerate(offsets[0]):
        start, end = (int(pair[0]), int(pair[1]))
        if start == end:
            continue
        if start < completion_end < end:
            raise ValueError("a token crosses the completion/terminator boundary")
        if end > completion_start and start < completion_end:
            token_id = int(input_ids[0, position])
            if token_id not in excluded_ids:
                selected.append(position)
    if not selected:
        raise ValueError("completion text has no scoreable tokens")

    model_inputs = {
        name: value for name, value in batch.items() if name in _MODEL_INPUT_KEYS
    }
    return StepCompletionEncoding(
        model_inputs=model_inputs,
        completion_token_positions=tuple(selected),
        completion_token_ids=tuple(int(input_ids[0, position]) for position in selected),
        rendered_text=expanded,
    )


def score_step_completion(
    model,
    *,
    model_inputs: Mapping[str, torch.Tensor],
    completion_token_positions: Sequence[int],
) -> float:
    """Mean frozen-reference log probability for completion text under clean context."""
    _require_eval_model(model)
    inputs = _one_sample_inputs(model_inputs)
    positions = tuple(completion_token_positions)
    if (
        not positions
        or any(
            isinstance(position, bool) or not isinstance(position, int)
            for position in positions
        )
        or tuple(sorted(set(positions))) != positions
        or positions[0] < 1
        or positions[-1] >= inputs["input_ids"].shape[1]
    ):
        raise ValueError(
            "completion_token_positions must be unique increasing causal positions"
        )
    device = inputs["input_ids"].device
    query_positions = torch.tensor(
        [position - 1 for position in positions], dtype=torch.long, device=device
    )
    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=False,
            stage2_final_attention_mask=None,
            stage2_logit_positions=query_positions,
            stage2_selective_logits_only=True,
        )
        logits = getattr(outputs, "stage2_selected_logits", None)
        if not isinstance(logits, torch.Tensor) or logits.shape[1] != len(positions):
            raise ValueError("Stage 2 model returned the wrong selected scoring logits")
        target_ids = inputs["input_ids"][:, list(positions)].to(logits.device)
        token_log_probs = torch.log_softmax(logits.float(), dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        )
        score = token_log_probs.mean()
    if not bool(torch.isfinite(score)):
        raise ValueError("reference completion score is not finite")
    return float(score.item())


def _canonical_json(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _require_sha256(name: str, value) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_int(name: str, value, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def _require_finite_score(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _freeze_json(value):
    """Return a recursively immutable JSON value with deterministic mapping order."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("generation settings cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        frozen = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("generation setting keys must be strings")
            frozen.append((key, _freeze_json(item)))
        return ("object", tuple(sorted(frozen)))
    if isinstance(value, (list, tuple)):
        return ("array", tuple(_freeze_json(item) for item in value))
    raise ValueError("generation settings must be JSON values")


def _thaw_json(value):
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "object":
        return {key: _thaw_json(item) for key, item in value[1]}
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "array":
        return [_thaw_json(item) for item in value[1]]
    return value


def _normalize_generation_settings(value):
    frozen = _freeze_json(value)
    if not (isinstance(frozen, tuple) and len(frozen) == 2 and frozen[0] == "object"):
        raise ValueError("generation_settings must be a JSON object")
    return frozen


@dataclass(frozen=True)
class CheckpointFile:
    """One exact regular file consumed while loading the frozen Stage 2A model."""

    relative_path: str
    size: int
    sha256: str
    load_root: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.relative_path, str)
            or not self.relative_path
            or Path(self.relative_path).is_absolute()
            or ".." in Path(self.relative_path).parts
        ):
            raise ValueError("checkpoint relative_path must be a safe non-empty relative path")
        if not isinstance(self.load_root, str) or not self.load_root:
            raise ValueError("checkpoint load_root must be a resolved absolute path")
        root = Path(self.load_root)
        if (
            not root.is_absolute()
            or ".." in root.parts
            or str(root.resolve(strict=False)) != self.load_root
        ):
            raise ValueError("checkpoint load_root must be a resolved absolute path")
        _require_int("checkpoint file size", self.size)
        _require_sha256("checkpoint file SHA-256", self.sha256)


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
    checkpoint: Path,
    artifact_kind: Stage2ArtifactKind,
    *,
    include_processor: bool = True,
) -> tuple[str, ...]:
    names = {
        entry.name
        for entry in checkpoint.iterdir()
        if entry.is_file() and not entry.is_symlink()
    }
    processor = names & _PROCESSOR_FILES if include_processor else set()
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
        if not model_names & _FULL_WEIGHT_FILES:
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
        records.append(
            CheckpointFile(relative_path, size, digest, str(checkpoint))
        )
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


def inspect_stage2a_checkpoint_identity(
    checkpoint_path: str | Path,
) -> tuple[tuple[CheckpointFile, ...], str]:
    """Recompute the exact local files used by Stage 2A model/processor loading."""
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("Stage 2A checkpoint path must be a regular directory")
    checkpoint = checkpoint.resolve(strict=True)
    inspection = inspect_stage2_artifact(checkpoint)
    checkpoint_paths = _loader_relative_paths(checkpoint, inspection.artifact_kind)
    records = _checkpoint_records(checkpoint, checkpoint_paths)
    tokenizer_identity = _tokenizer_identity(records)
    if inspection.artifact_kind is Stage2ArtifactKind.PEFT:
        snapshot = inspection.peft_metadata_snapshot
        if snapshot is None:
            raise ValueError("Stage 2A PEFT inspection lost its metadata")
        metadata = decode_stage2_peft_metadata_snapshot(snapshot)
        base = Path(metadata.base_model_reference).expanduser()
        if base.is_symlink() or not base.is_dir():
            raise ValueError("PEFT base-model reference must be a local regular directory")
        base = base.resolve(strict=True)
        base_inspection = inspect_stage2_artifact(base)
        if base_inspection.artifact_kind is not Stage2ArtifactKind.FULL:
            raise ValueError("PEFT base-model reference must identify a full local model")
        base_paths = _loader_relative_paths(
            base, Stage2ArtifactKind.FULL, include_processor=False
        )
        records += _checkpoint_records(base, base_paths)
    return records, tokenizer_identity


@dataclass(frozen=True)
class StepNegative:
    """One chosen completion and its independently valid visual/textual negatives."""

    sample_id: str
    step_id: int
    chosen_text: str
    rejected_visual_text: str | None
    rejected_textual_text: str | None
    valid_visual_negative: bool
    valid_textual_negative: bool
    reference_chosen_score: float
    reference_visual_rejected_score: float | None
    reference_textual_rejected_score: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        _require_int("step_id", self.step_id)
        if not isinstance(self.chosen_text, str) or not self.chosen_text:
            raise ValueError("chosen_text must be a non-empty string")
        _require_finite_score("reference_chosen_score", self.reference_chosen_score)
        self._validate_modality(
            "visual", self.rejected_visual_text, self.valid_visual_negative,
            self.reference_visual_rejected_score,
        )
        self._validate_modality(
            "textual", self.rejected_textual_text, self.valid_textual_negative,
            self.reference_textual_rejected_score,
        )

    @staticmethod
    def _validate_modality(name: str, text, valid, score) -> None:
        if text is not None and not isinstance(text, str):
            raise ValueError(f"rejected_{name}_text must be a string or null")
        if not isinstance(valid, bool):
            raise ValueError(f"valid_{name}_negative must be a boolean")
        if valid:
            if not text:
                raise ValueError(f"valid {name} negative requires non-empty rejected text")
            _require_finite_score(f"reference_{name}_rejected_score", score)
        elif score is not None:
            raise ValueError(f"invalid {name} negative requires a null rejected score")


@dataclass(frozen=True)
class NegativeManifest:
    """Identity of the immutable source/model/configuration that produced a JSONL."""

    schema_version: int
    source_sha256: str
    selected_row_indices: tuple[int, ...]
    stage2a_checkpoint_path: str
    checkpoint_files: tuple[CheckpointFile, ...]
    tokenizer_config_identity: str
    latent_size: int
    generation_settings: object
    negative_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != NEGATIVE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        _require_sha256("source_sha256", self.source_sha256)
        _require_sha256("negative_sha256", self.negative_sha256)
        if not isinstance(self.selected_row_indices, tuple):
            raise ValueError("selected_row_indices must be a tuple")
        previous = -1
        for index in self.selected_row_indices:
            _require_int("selected row index", index)
            if index <= previous:
                raise ValueError("selected_row_indices must be strictly increasing")
            previous = index
        if (
            not isinstance(self.stage2a_checkpoint_path, str)
            or not self.stage2a_checkpoint_path
        ):
            raise ValueError("stage2a_checkpoint_path must be a non-empty string")
        if not isinstance(self.checkpoint_files, tuple) or not all(
            isinstance(entry, CheckpointFile) for entry in self.checkpoint_files
        ):
            raise ValueError("checkpoint_files must be CheckpointFile values")
        file_keys = [
            (entry.load_root, entry.relative_path) for entry in self.checkpoint_files
        ]
        if len(file_keys) != len(set(file_keys)):
            raise ValueError("checkpoint_files contains duplicate load-root paths")
        if self.checkpoint_files and not any(
            entry.load_root == self.stage2a_checkpoint_path
            for entry in self.checkpoint_files
        ):
            raise ValueError("checkpoint_files do not identify the Stage 2A load root")
        if (
            not isinstance(self.tokenizer_config_identity, str)
            or not self.tokenizer_config_identity
        ):
            raise ValueError("tokenizer_config_identity must be a non-empty string")
        _require_int("latent_size", self.latent_size, minimum=1)
        object.__setattr__(
            self, "generation_settings", _normalize_generation_settings(self.generation_settings)
        )

    @property
    def generation_settings_dict(self) -> dict[str, object]:
        """Return a detached JSON-object view for strict consumer validation."""
        return _thaw_json(self.generation_settings)


def filter_negative(
    *,
    chosen_text: str,
    rejected_text: str | None,
    scoreable_token_ids: Sequence[int],
    min_tokens: int,
) -> bool:
    """Whether a generated completion can form a valid preference negative."""
    if not isinstance(chosen_text, str):
        raise ValueError("chosen_text must be a string")
    if rejected_text is None:
        return False
    if not isinstance(rejected_text, str):
        raise ValueError("rejected_text must be a string or null")
    _require_int("min_tokens", min_tokens, minimum=1)
    if rejected_text == chosen_text or not rejected_text.strip():
        return False
    if " ".join(rejected_text.split()) == " ".join(chosen_text.split()):
        return False
    return len(scoreable_token_ids) >= min_tokens


def _negative_payload(row: StepNegative) -> dict:
    return asdict(row)


def _manifest_payload(manifest: NegativeManifest) -> dict:
    payload = asdict(manifest)
    payload["generation_settings"] = _thaw_json(manifest.generation_settings)
    return payload


def _write_exclusively(path: Path, data: bytes, label: str) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as error:
        raise ValueError(f"{label} path already exists: {path}") from error


def write_negative_artifact(
    *,
    negative_path: str | Path,
    manifest_path: str | Path,
    negatives: Iterable[StepNegative],
    source_path: str | Path,
    source_sha256: str,
    selected_row_indices: Sequence[int],
    stage2a_checkpoint_path: str | Path,
    checkpoint_files: Sequence[CheckpointFile],
    tokenizer_config_identity: str,
    latent_size: int,
    generation_settings: Mapping[str, object],
) -> NegativeManifest:
    """Write deterministic JSONL and its sidecar manifest, without overwriting either."""
    source_path = Path(source_path)
    actual_source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual_source_sha256 != source_sha256:
        raise ValueError("source SHA-256 does not match source_path")
    checkpoint_path = Path(stage2a_checkpoint_path).resolve(strict=True)
    rows = tuple(negatives)
    if not all(isinstance(row, StepNegative) for row in rows):
        raise TypeError("negatives must contain StepNegative values")
    _negative_store(rows)
    negative_bytes = b"".join(
        (_canonical_json(_negative_payload(row)) + "\n").encode("utf-8") for row in rows
    )
    manifest = NegativeManifest(
        schema_version=NEGATIVE_MANIFEST_SCHEMA_VERSION,
        source_sha256=source_sha256,
        selected_row_indices=tuple(selected_row_indices),
        stage2a_checkpoint_path=str(checkpoint_path),
        checkpoint_files=tuple(checkpoint_files),
        tokenizer_config_identity=tokenizer_config_identity,
        latent_size=latent_size,
        generation_settings=generation_settings,
        negative_sha256=hashlib.sha256(negative_bytes).hexdigest(),
    )
    negative_path = Path(negative_path)
    manifest_path = Path(manifest_path)
    _write_exclusively(negative_path, negative_bytes, "negative artifact")
    try:
        _write_exclusively(
            manifest_path, (_canonical_json(_manifest_payload(manifest)) + "\n").encode("utf-8"),
            "negative manifest",
        )
    except Exception:
        # Do not leave a manifest that claims a JSONL we did not publish.
        negative_path.unlink(missing_ok=True)
        raise
    return manifest


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _parse_json(raw: bytes, label: str):
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {error}") from error


def _require_exact_fields(payload: Mapping, expected_type, label: str) -> None:
    expected = {field.name for field in fields(expected_type)}
    actual = set(payload)
    if actual != expected:
        raise ValueError(f"{label} has unknown or missing fields")


def _row_from_payload(payload) -> StepNegative:
    if not isinstance(payload, dict):
        raise ValueError("negative JSONL row must be an object")
    _require_exact_fields(payload, StepNegative, "negative JSONL row")
    return StepNegative(**payload)


def _manifest_from_bytes(raw: bytes) -> NegativeManifest:
    payload = _parse_json(raw, "negative manifest")
    if not isinstance(payload, dict):
        raise ValueError("negative manifest must be an object")
    _require_exact_fields(payload, NegativeManifest, "negative manifest")
    if not isinstance(payload["selected_row_indices"], list):
        raise ValueError("selected_row_indices must be an array")
    if not isinstance(payload["checkpoint_files"], list):
        raise ValueError("checkpoint_files must be an array")
    checkpoint_files = []
    for entry in payload["checkpoint_files"]:
        if not isinstance(entry, dict):
            raise ValueError("checkpoint file must be an object")
        _require_exact_fields(entry, CheckpointFile, "checkpoint file")
        checkpoint_files.append(CheckpointFile(**entry))
    return NegativeManifest(
        schema_version=payload["schema_version"],
        source_sha256=payload["source_sha256"],
        selected_row_indices=tuple(payload["selected_row_indices"]),
        stage2a_checkpoint_path=payload["stage2a_checkpoint_path"],
        checkpoint_files=tuple(checkpoint_files),
        tokenizer_config_identity=payload["tokenizer_config_identity"],
        latent_size=payload["latent_size"],
        generation_settings=payload["generation_settings"],
        negative_sha256=payload["negative_sha256"],
    )


def read_negative_manifest(path: str | Path) -> NegativeManifest:
    """Read one strict schema-2 manifest from a regular non-symlink file."""
    return _manifest_from_bytes(_read_regular_file(path, "negative manifest"))


def _read_regular_file(path: str | Path, label: str) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} path must be a regular file: {path}")
    return path.read_bytes()


def _negative_store(rows: Iterable[StepNegative]) -> dict[tuple[str, int], StepNegative]:
    store = {}
    for row in rows:
        key = (row.sample_id, row.step_id)
        if key in store:
            raise ValueError(f"duplicate negative join key: {key}")
        store[key] = row
    return store


def _verify_checkpoint_files(manifest: NegativeManifest) -> None:
    checkpoint_root = Path(manifest.stage2a_checkpoint_path)
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise ValueError("Stage 2A checkpoint path must be a regular directory")
    checked_roots = set()
    for entry in manifest.checkpoint_files:
        load_root = Path(entry.load_root)
        if load_root not in checked_roots:
            if load_root.is_symlink() or not load_root.is_dir():
                raise ValueError(
                    f"checkpoint load root must be a regular directory: {load_root}"
                )
            if str(load_root.resolve(strict=True)) != entry.load_root:
                raise ValueError(f"checkpoint load root is not resolved: {load_root}")
            checked_roots.add(load_root)
        path = load_root / entry.relative_path
        raw = _read_regular_file(path, "checkpoint file")
        if len(raw) != entry.size or hashlib.sha256(raw).hexdigest() != entry.sha256:
            raise ValueError(f"checkpoint file hash or size does not match: {entry.relative_path}")


def load_negative_store(
    *,
    negative_path: str | Path,
    manifest_path: str | Path,
    source_path: str | Path,
    expected_source_sha256: str,
    expected_selected_row_indices: Sequence[int],
    expected_stage2a_checkpoint_path: str | Path,
    expected_checkpoint_files: Sequence[CheckpointFile],
    expected_tokenizer_config_identity: str,
    expected_latent_size: int,
    expected_generation_settings: Mapping[str, object],
    expected_keys: Iterable[tuple[str, int]],
) -> dict[tuple[str, int], StepNegative]:
    """Fail closed unless artifact bytes and every frozen producer identity agree."""
    manifest = _manifest_from_bytes(_read_regular_file(manifest_path, "negative manifest"))
    expected_checkpoint_path = str(Path(expected_stage2a_checkpoint_path).resolve(strict=True))
    current_source_sha256 = hashlib.sha256(
        _read_regular_file(source_path, "source JSONL")
    ).hexdigest()
    if (
        current_source_sha256 != manifest.source_sha256
        or current_source_sha256 != expected_source_sha256
    ):
        raise ValueError("source JSONL SHA-256 does not match")
    if manifest.selected_row_indices != tuple(expected_selected_row_indices):
        raise ValueError("selected row indices do not match")
    if manifest.stage2a_checkpoint_path != expected_checkpoint_path:
        raise ValueError("Stage 2A checkpoint path does not match")
    if manifest.checkpoint_files != tuple(expected_checkpoint_files):
        raise ValueError("checkpoint file records do not match")
    if manifest.tokenizer_config_identity != expected_tokenizer_config_identity:
        raise ValueError("tokenizer/config identity does not match")
    if manifest.latent_size != expected_latent_size:
        raise ValueError("latent size does not match")
    if manifest.generation_settings != _normalize_generation_settings(expected_generation_settings):
        raise ValueError("generation settings do not match")
    _verify_checkpoint_files(manifest)
    negative_bytes = _read_regular_file(negative_path, "negative artifact")
    if hashlib.sha256(negative_bytes).hexdigest() != manifest.negative_sha256:
        raise ValueError("negative artifact SHA-256 does not match")
    rows = []
    for line_number, line in enumerate(negative_bytes.splitlines(), start=1):
        if not line:
            raise ValueError(f"negative JSONL row {line_number} is empty")
        rows.append(_row_from_payload(_parse_json(line, f"negative JSONL row {line_number}")))
    store = _negative_store(rows)
    required_keys = tuple(expected_keys)
    if len(required_keys) != len(set(required_keys)):
        raise ValueError("expected negative join keys are duplicated")
    if set(store) != set(required_keys):
        raise ValueError("missing or unexpected negative join keys")
    return store
