"""Independent fixed Stage 1 reference loading and Stage 2 objective identity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from peft_common import resolve_qwen2_5_vl_model
from stage1.checkpointing import (
    decode_stage1_peft_metadata_snapshot,
    read_stage1_peft_metadata_snapshot,
)
from stage2.checkpointing import (
    configure_stage2_metadata,
    resolve_latent_token_ids,
    resolve_stage2_load_decision,
)
from stage2.attention_backend import (
    load_qwen2_5_vl_with_attention,
    validate_qwen2_5_vl_attention_backends,
)
from stage2.peft import Stage2TuningConfig, load_stage1_dora_for_fresh_stage2
from stage2.rng import capture_rng_state, replay_rng_state


STAGE2_OBJECTIVE_FILENAME = "stage2_objective.json"
_OBJECTIVE_SCHEMA_VERSION = 1
_SOURCE_SELECTOR_NAMES = ("cache_dir", "revision", "token", "local_files_only")
_TOKENIZER_FILENAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.json",
        "chat_template.jinja",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)
_MODEL_CONFIG_FILENAMES = frozenset(
    {
        "adapter_config.json",
        "config.json",
        "generation_config.json",
        "stage1_peft_metadata.json",
    }
)


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be JSON-safe") from error


@dataclass(frozen=True)
class FrozenReference:
    """A separately loaded frozen model and its JSON-safe source identity."""

    model: nn.Module
    identity: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise TypeError("reference model must be a torch module")
        if type(self.identity) is not dict:
            raise TypeError("reference identity must be a dictionary")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("reference model parameters must be frozen")
        if self.model.training:
            raise ValueError("reference model must be in eval mode")
        object.__setattr__(
            self, "identity", _json_copy(self.identity, label="reference identity")
        )


def _is_load_artifact(path: Path) -> bool:
    name = path.name
    if name in _TOKENIZER_FILENAMES or name in _MODEL_CONFIG_FILENAMES:
        return True
    if name == "ilvr_token_rows.safetensors":
        return True
    if name.endswith((".safetensors", ".safetensors.index.json")):
        return name.startswith(("model", "adapter_model"))
    if name.endswith(".bin"):
        return name.startswith(("pytorch_model", "adapter_model"))
    if name == "pytorch_model.bin.index.json":
        return True
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_identity(
    reference_source: Path, base_source: Path
) -> list[dict[str, str | int]]:
    roots = [("reference", reference_source)]
    if base_source != reference_source:
        roots.append(("base", base_source))
    artifacts: list[dict[str, str | int]] = []
    for label, root in roots:
        if not root.is_dir():
            raise ValueError(f"reference source is not a local directory: {root}")
        candidates = {candidate for candidate in root.rglob("*")
                      if candidate.is_file() and _is_load_artifact(candidate)}
        # The local loader follows file links and index-selected shard paths,
        # including arbitrary shard names beneath linked directories.
        # Match this loader's default local checkpoint precedence. A stale,
        # unselected index remains hashed, but its contents are not interpreted.
        for name in ("model.safetensors", "model.safetensors.index.json",
                     "pytorch_model.bin", "pytorch_model.bin.index.json"):
            selected = root / name
            if not selected.is_file():
                continue
            if name.endswith(".index.json"):
                payload = json.loads(selected.read_text(encoding="utf-8"))
                candidates.update(root / shard for shard in payload["weight_map"].values())
            break
        for candidate in sorted(candidates):
            relative = Path(os.path.relpath(candidate, root)).as_posix()
            artifacts.append(
                {
                    "path": relative if len(roots) == 1 else f"{label}/{relative}",
                    "sha256": _sha256(candidate),
                    "size": candidate.stat().st_size,
                }
            )
    if not artifacts:
        raise ValueError(
            "reference source contains no load-relevant artifacts: "
            f"{reference_source}"
        )
    return artifacts


def _source_selectors(args) -> dict[str, str | bool]:
    result = {}
    for name in _SOURCE_SELECTOR_NAMES:
        value = getattr(args, name, None)
        if value is not None:
            result[name] = value
    return result


def _student_placement(student: nn.Module) -> tuple[torch.device, torch.dtype]:
    floating = tuple(
        parameter for parameter in student.parameters() if parameter.is_floating_point()
    )
    if not floating:
        raise ValueError("student has no floating-point parameters")
    devices = {parameter.device for parameter in floating}
    if len(devices) != 1:
        raise ValueError("student parameters must have one current device")
    dtype = getattr(resolve_qwen2_5_vl_model(student), "dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError("student base model dtype is unavailable")
    return next(iter(devices)), dtype


def _output_size(model: nn.Module) -> int:
    qwen = resolve_qwen2_5_vl_model(model)
    value = getattr(qwen.lm_head, "out_features", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("model output vocabulary layout is invalid")
    return value


def _runtime_tokenizer_identity(tokenizer, token_ids, output_size: int) -> dict[str, Any]:
    try:
        size = len(tokenizer)
    except (TypeError, AttributeError) as error:
        raise ValueError("tokenizer must expose its length") from error
    vocabulary_digest = None
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocabulary = get_vocab()
        vocabulary_digest = hashlib.sha256(
            json.dumps(
                vocabulary, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
    return {
        "size": size,
        "output_size": output_size,
        "latent_token_ids": [
            token_ids.pad_token_id,
            token_ids.start_token_id,
            token_ids.end_token_id,
        ],
        "vocabulary_sha256": vocabulary_digest,
    }


def _resolve_reference_decision(args, source_dependencies, reference_source: Path):
    selectors = _source_selectors(args)
    adapter_config = source_dependencies.find_adapter_config(
        str(reference_source), **selectors
    )
    if adapter_config is None:
        return resolve_stage2_load_decision(
            load_model_path=str(reference_source),
            stage1_adapter_path=None,
            resume_from_checkpoint_path=None,
            load_config=source_dependencies.load_config,
            discover_adapter_config=source_dependencies.find_adapter_config,
            source_provenance_kwargs=selectors,
            latent_size=args.latent_size,
            supervise_latent_boundaries=args.supervise_latent_boundaries,
            tuning_method="full",
            tuning_method_explicit=True,
            lora_r=1,
            lora_alpha=1,
            lora_dropout=0.0,
            lora_target_modules=("q_proj",),
        )

    snapshot = read_stage1_peft_metadata_snapshot(reference_source)
    metadata = decode_stage1_peft_metadata_snapshot(snapshot)
    metadata_base_source = Path(metadata.base_model_reference)
    if not metadata_base_source.is_dir():
        raise ValueError(
            "Stage 1 DoRA base_model_reference must be a local directory: "
            f"{metadata.base_model_reference}"
        )
    return resolve_stage2_load_decision(
        load_model_path=metadata.base_model_reference,
        stage1_adapter_path=str(reference_source),
        resume_from_checkpoint_path=None,
        load_config=source_dependencies.load_config,
        discover_adapter_config=source_dependencies.find_adapter_config,
        source_provenance_kwargs=selectors,
        load_stage1_metadata_snapshot=lambda _path: snapshot,
        latent_size=args.latent_size,
        supervise_latent_boundaries=args.supervise_latent_boundaries,
        tuning_method=metadata.tuning_method,
        tuning_method_explicit=True,
        lora_r=metadata.rank,
        lora_alpha=metadata.alpha,
        lora_dropout=metadata.dropout,
        lora_target_modules=metadata.requested_target_suffixes,
    )


def load_frozen_reference(
    args, *, student: nn.Module, tokenizer, source_dependencies
) -> FrozenReference | None:
    """Load the requested original Stage 1 source as an independent reference."""

    if float(getattr(args, "lambda_dyn", 0.0)) == 0.0:
        return None
    raw_source = getattr(args, "reference_model_path", None)
    if not isinstance(raw_source, str) or not raw_source.strip():
        raise ValueError("reference_model_path is required when lambda_dyn is positive")
    reference_source = Path(raw_source).resolve()
    if not reference_source.is_dir():
        raise ValueError(
            f"reference_model_path must be a local Stage 1 directory: {raw_source}"
        )

    device, dtype = _student_placement(student)
    preserved_rng = capture_rng_state()
    with replay_rng_state(preserved_rng):
        decision = _resolve_reference_decision(
            args, source_dependencies, reference_source
        )
        selectors = _source_selectors(args)
        processor = source_dependencies.load_processor(
            decision.processor_source, **selectors
        )
        vision_attention = getattr(args, "vision_attention_implementation", None)
        mixed_attention = (
            vision_attention is not None
            and vision_attention != args.attention_implementation
        )
        if mixed_attention:
            model = load_qwen2_5_vl_with_attention(
                decision.base_model_source,
                attention_implementation=args.attention_implementation,
                vision_attention_implementation=vision_attention,
                config_load_kwargs=selectors,
                model_load_kwargs={**selectors, "torch_dtype": dtype},
                load_config=source_dependencies.load_config,
                load_model=source_dependencies.load_model,
            )
        else:
            model = source_dependencies.load_model(
                decision.base_model_source,
                **selectors,
                attn_implementation=args.attention_implementation,
                torch_dtype=dtype,
            )
        if decision.stage1_adapter_source is not None:
            model = load_stage1_dora_for_fresh_stage2(
                model,
                processor,
                decision.stage1_adapter_source,
                decision.tuning,
                requested_base_model_reference=decision.base_model_source,
            )
        reference_qwen = resolve_qwen2_5_vl_model(model)
        if mixed_attention:
            validate_qwen2_5_vl_attention_backends(
                reference_qwen,
                attention_implementation=args.attention_implementation,
                vision_attention_implementation=vision_attention,
            )
        reference_token_ids = resolve_latent_token_ids(
            processor.tokenizer, reference_qwen.config
        )
        configure_stage2_metadata(
            reference_qwen.config,
            reference_token_ids,
            latent_size=decision.latent_size,
            supervise_latent_boundaries=decision.supervise_latent_boundaries,
        )
        model.to(device=device)

    student_qwen = resolve_qwen2_5_vl_model(student)
    student_token_ids = resolve_latent_token_ids(tokenizer, student_qwen.config)
    student_output_size = _output_size(student)
    reference_output_size = _output_size(model)
    student_tokenizer_identity = _runtime_tokenizer_identity(
        tokenizer, student_token_ids, student_output_size
    )
    reference_tokenizer_identity = _runtime_tokenizer_identity(
        processor.tokenizer, reference_token_ids, reference_output_size
    )
    if student_tokenizer_identity != reference_tokenizer_identity:
        raise ValueError("reference tokenizer or output vocabulary layout mismatch")

    model.requires_grad_(False)
    model.eval()
    base_source = Path(decision.base_model_source).resolve()
    identity = {
        "schema_version": 1,
        "source": str(reference_source),
        "base_model_source": str(base_source),
        "artifact_kind": (
            "stage1_dora"
            if decision.stage1_adapter_source is not None
            else "stage1_full"
        ),
        "artifacts": _artifact_identity(reference_source, base_source),
        "tokenizer": reference_tokenizer_identity,
        "attention_implementation": args.attention_implementation,
        "dtype": str(dtype).removeprefix("torch."),
        "parameter_dtypes": sorted(
            {
                str(parameter.dtype).removeprefix("torch.")
                for parameter in model.parameters()
            }
        ),
    }
    return FrozenReference(model=model, identity=identity)


def _objective_payload(
    config_dict: Mapping[str, Any],
    reference_identity: Mapping[str, Any] | None,
    vocabulary_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(config_dict, Mapping):
        raise TypeError("objective configuration must be a mapping")
    for label, value in (
        ("reference identity", reference_identity),
        ("vocabulary identity", vocabulary_identity),
    ):
        if value is not None and not isinstance(value, Mapping):
            raise TypeError(f"{label} must be a mapping or None")
    return _json_copy(
        {
            "schema_version": _OBJECTIVE_SCHEMA_VERSION,
            "config": dict(config_dict),
            "reference_identity": (
                None if reference_identity is None else dict(reference_identity)
            ),
            "vocabulary_identity": (
                None if vocabulary_identity is None else dict(vocabulary_identity)
            ),
        },
        label="Stage 2 objective identity",
    )


def _objective_config_for_comparison(config_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Treat older objective sidecars as the historical full-replay, score-mean mode."""
    config = dict(config_dict)
    config.setdefault("stage2_m_execution", "full_replay")
    config.setdefault("vision_token_reduction", "score_mean")
    return config


def save_objective_identity(
    path: str | Path,
    config_dict: Mapping[str, Any],
    reference_identity: Mapping[str, Any] | None,
    vocabulary_identity: Mapping[str, Any] | None,
) -> None:
    """Write one canonical Stage 2 objective sidecar into a checkpoint."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    payload = _objective_payload(
        config_dict, reference_identity, vocabulary_identity
    )
    (directory / STAGE2_OBJECTIVE_FILENAME).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def validate_objective_resume(
    path: str | Path,
    config_dict: Mapping[str, Any],
    reference_identity: Mapping[str, Any] | None,
    vocabulary_identity: Mapping[str, Any] | None,
) -> None:
    """Require exact saved objective, reference, and vocabulary identities."""

    sidecar = Path(path) / STAGE2_OBJECTIVE_FILENAME
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"missing Stage 2 objective identity: {sidecar}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Stage 2 objective identity: {sidecar}") from error
    validate_objective_identity_payload(payload, config_dict, reference_identity, vocabulary_identity)


def validate_objective_identity_payload(
    payload: Any,
    config_dict: Mapping[str, Any],
    reference_identity: Mapping[str, Any] | None,
    vocabulary_identity: Mapping[str, Any] | None,
) -> None:
    """Compare a captured sidecar with the runtime using the resume contract."""
    expected = _objective_payload(
        _objective_config_for_comparison(config_dict),
        reference_identity,
        vocabulary_identity,
    )
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise ValueError("invalid Stage 2 objective sidecar schema")
    if payload.get("schema_version") != _OBJECTIVE_SCHEMA_VERSION:
        raise ValueError("unsupported Stage 2 objective sidecar schema")
    for field, label in (
        ("config", "objective configuration mismatch"),
        ("reference_identity", "reference identity mismatch"),
        ("vocabulary_identity", "vocabulary identity mismatch"),
    ):
        observed = payload.get(field)
        if field == "config" and isinstance(observed, Mapping):
            observed = _objective_config_for_comparison(observed)
        if observed != expected[field]:
            raise ValueError(label)
