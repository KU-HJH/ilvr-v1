import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Sequence

import peft
import torch
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors import SafetensorError
from safetensors.torch import load as load_safetensors
from torch import nn

from peft_common import (
    capture_qwen2_5_vl_architecture,
    normalize_decoder_target_suffixes,
    resolve_qwen2_5_vl_model,
    resolve_qwen_decoder_target_names,
)

TuningMethod = Literal["full", "lora", "dora"]
ALLOWED_STAGE2_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def normalize_stage2_target_modules(
    value: str | Sequence[str],
) -> tuple[str, ...]:
    return normalize_decoder_target_suffixes(
        value, stage_label="Stage 2 PEFT target modules"
    )


@dataclass(frozen=True)
class Stage2TuningConfig:
    method: TuningMethod
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.method not in ("full", "lora", "dora"):
            raise ValueError(f"unsupported Stage 2 tuning method: {self.method}")
        if self.r <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        normalize_stage2_target_modules(self.target_modules)

    @property
    def is_peft(self) -> bool:
        return self.method in ("lora", "dora")

    @property
    def use_dora(self) -> bool:
        return self.method == "dora"


@dataclass(frozen=True)
class Stage2ParameterReport:
    total_parameters: int
    trainable_parameters: int
    trainable_names: tuple[str, ...]
    resolved_target_names: tuple[str, ...]


def resolve_stage2_target_module_names(
    model: nn.Module, requested: Sequence[str]
) -> tuple[str, ...]:
    """Return full decoder projection paths, excluding matching visual layers."""
    requested = normalize_stage2_target_modules(requested)
    try:
        resolved = resolve_qwen_decoder_target_names(model, requested)
    except ValueError as error:
        if str(error).startswith(
            ("Requested decoder targets did not match", "Requested targets must be decoder-owned")
        ):
            missing = str(error).split(": ", 1)[1]
            raise ValueError(
                f"Stage 2 decoder targets matched no modules: {missing}"
            ) from error
        raise
    modules = dict(model.named_modules())
    matches = {
        suffix: tuple(
            name
            for name in resolved
            if name.endswith(f".{suffix}")
            and isinstance(modules[name], torch.nn.Linear)
        )
        for suffix in requested
    }
    missing = [suffix for suffix, names in matches.items() if not names]
    if missing:
        raise ValueError(f"Stage 2 decoder targets matched no modules: {missing}")
    return tuple(name for suffix in requested for name in matches[suffix])


def resolve_stage2_qwen_model(model: nn.Module) -> nn.Module:
    """Unwrap one engine wrapper and a PEFT model to the Stage 2 Qwen base."""
    wrapped = getattr(model, "module", model)
    try:
        candidate = resolve_qwen2_5_vl_model(model)
    except TypeError as error:
        raise TypeError("model does not satisfy the Stage 2 Qwen interface") from error
    required = (
        "prepare_stage2_base_inputs_embeds",
        "get_rope_index",
        "stage2_rollout_engine",
        "model",
        "lm_head",
    )
    if (
        any(not hasattr(candidate, attribute) for attribute in required)
        or getattr(getattr(candidate, "config", None), "stage", None) != "stage2"
    ):
        raise TypeError("model does not satisfy the Stage 2 Qwen interface")
    from stage1.token_rows import SparseOutputTokenRows

    sparse_rows = (
        tuple(
            module
            for module in candidate.modules()
            if isinstance(module, SparseOutputTokenRows)
        )
        if isinstance(candidate, nn.Module)
        else ()
    )
    if isinstance(wrapped, PeftModel):
        adapter_names = tuple(wrapped.peft_config)
        active_adapters = getattr(wrapped, "active_adapters", None)
        if active_adapters is None:
            active_adapters = getattr(wrapped, "active_adapter", None)
        if isinstance(active_adapters, str):
            active_adapters = (active_adapters,)
        else:
            active_adapters = tuple(active_adapters or ())
        if (
            len(adapter_names) != 1
            or len(active_adapters) != 1
            or active_adapters[0] != adapter_names[0]
        ):
            raise ValueError(
                "Stage 2 inference requires exactly one active Stage 2 adapter"
            )
        adapter_config = wrapped.peft_config[active_adapters[0]]
        configured_rows = getattr(adapter_config, "trainable_token_indices", None)
        if configured_rows is None:
            if sparse_rows:
                raise ValueError(
                    "legacy base-source Stage 2 inference must not contain "
                    "sparse output-token rows"
                )
            return candidate
        if (
            not isinstance(configured_rows, dict)
            or set(configured_rows) != {"model.embed_tokens"}
            or not isinstance(configured_rows["model.embed_tokens"], list)
        ):
            raise ValueError("invalid Stage 2 selective-token adapter contract")
        configured_ids = tuple(configured_rows["model.embed_tokens"])
        if len(sparse_rows) != 1 or candidate.lm_head is not sparse_rows[0]:
            raise ValueError(
                "Stage 2 inference requires exactly one sparse output-token row wrapper"
            )
        rows = sparse_rows[0]
        token_ids = tuple(rows.token_ids.detach().cpu().tolist())
        if len(token_ids) != 3 or len(configured_ids) != 3:
            raise ValueError("latent token IDs must contain exactly three values")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in configured_ids
        ):
            raise ValueError("latent token IDs must be valid vocabulary integers")
        if len(set(token_ids)) != 3 or len(set(configured_ids)) != 3:
            raise ValueError("latent token IDs must be unique")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id >= rows.out_features
            for token_id in token_ids
        ):
            raise ValueError("latent token IDs must be valid vocabulary integers")
        if token_ids != configured_ids:
            raise ValueError(
                "sparse output-token IDs do not match active adapter contract"
            )
    elif sparse_rows:
        raise ValueError(
            "sparse Stage 2 inference requires exactly one active Stage 2 adapter"
        )
    return candidate


def _stage2_adapter_target_names(
    model: nn.Module, qwen: nn.Module, tuning: Stage2TuningConfig
) -> tuple[str, ...]:
    if not isinstance(model, PeftModel):
        raise ValueError("Stage 2 PEFT tuning requires a PeftModel")

    config = model.peft_config["default"]
    configured_targets = config.target_modules
    if (
        isinstance(configured_targets, str)
        or not isinstance(configured_targets, (list, tuple, set, frozenset))
        or not configured_targets
        or any(
            not isinstance(pattern, str) or not pattern.strip()
            for pattern in configured_targets
        )
        or len(set(configured_targets)) != len(configured_targets)
    ):
        raise ValueError("Stage 2 PEFT adapter target selectors are invalid")
    target_names = tuple(getattr(model.base_model, "targeted_module_names", ()))
    modules = dict(qwen.named_modules())
    unapproved = sorted(
        name
        for name in target_names
        if name.rsplit(".", 1)[-1] not in ALLOWED_STAGE2_TARGET_SUFFIXES
    )
    if unapproved:
        raise ValueError(f"unapproved Stage 2 decoder targets: {unapproved}")

    requested = normalize_stage2_target_modules(tuning.target_modules)
    expected_targets = {
        name
        for name, module in modules.items()
        if name.startswith("model.layers.")
        and name.rsplit(".", 1)[-1] in requested
        and isinstance(getattr(module, "base_layer", module), torch.nn.Linear)
    }
    configured_matches = {
        name
        for name in modules
        if any(
            name == pattern or name.endswith(f".{pattern}")
            for pattern in configured_targets
        )
    }
    unmatched_configured_targets = tuple(
        pattern
        for pattern in configured_targets
        if not any(
            name == pattern or name.endswith(f".{pattern}")
            for name in expected_targets
        )
    )
    if (
        not target_names
        or config.modules_to_save is not None
        or any(
            not name.startswith("model.layers.")
            or name not in modules
            or not hasattr(modules[name], "lora_A")
            for name in target_names
        )
    ):
        raise ValueError("Stage 2 PEFT adapter targets must be decoder LoRA modules")
    if (
        unmatched_configured_targets
        or configured_matches != expected_targets
        or set(target_names) != expected_targets
    ):
        raise ValueError("loaded adapter targets do not match Stage 2 tuning")
    return (
        tuple(configured_targets)
        if set(configured_targets) == expected_targets
        else target_names
    )


def validate_stage2_parameter_boundary(
    model: nn.Module,
    tuning: Stage2TuningConfig,
    *,
    expect_trainable: bool = True,
    allow_sparse_token_rows: bool = False,
) -> Stage2ParameterReport:
    """Validate that Stage 2 training owns only the permitted parameters."""
    qwen = resolve_stage2_qwen_model(model)
    if allow_sparse_token_rows:
        if tuning.method != "dora":
            raise ValueError("sparse Stage 2 token rows require DoRA")
        from stage1.token_rows import find_sparse_output_token_rows

        rows = find_sparse_output_token_rows(model)
        if expect_trainable:
            from stage1.peft import validate_stage1_trainable_boundary
            from stage1.types import Stage1TuningConfig

            validate_stage1_trainable_boundary(
                model,
                Stage1TuningConfig(
                    method="dora",
                    rank=tuning.r,
                    alpha=tuning.alpha,
                    dropout=tuning.dropout,
                    target_suffixes=tuning.target_modules,
                ),
                tuple(rows.token_ids.detach().cpu().tolist()),
            )
    for label, module in (
        ("visual", qwen.visual),
        ("embedding", qwen.model.embed_tokens),
        ("lm_head", qwen.lm_head),
    ):
        for name, parameter in module.named_parameters():
            if parameter.requires_grad and (
                label == "visual"
                or (tuning.is_peft and not allow_sparse_token_rows)
            ):
                raise ValueError(f"{label} parameter is trainable: {name}")

    if tuning.is_peft:
        resolved_target_names = _stage2_adapter_target_names(model, qwen, tuning)
        config = model.peft_config["default"]
        if config.use_dora is not tuning.use_dora:
            raise ValueError("Stage 2 PEFT adapter method does not match tuning")
    else:
        resolved_target_names = resolve_stage2_target_module_names(
            qwen, tuning.target_modules
        )

    parameters = tuple(model.named_parameters())
    trainable = tuple(name for name, parameter in parameters if parameter.requires_grad)
    if not expect_trainable:
        if trainable:
            raise ValueError("Stage 2 inference model has trainable parameters")
    elif tuning.is_peft:
        if not trainable:
            raise ValueError("Stage 2 PEFT adapter has no trainable parameters")
        unexpected = [
            name
            for name in trainable
            if "lora_" not in name
            and not (
                allow_sparse_token_rows
                and (
                    "trainable_tokens_delta" in name
                    or name.endswith("output_delta")
                )
            )
        ]
        if unexpected:
            raise ValueError(
                f"non-adapter parameter is trainable: {unexpected[0]}"
            )
    else:
        if not trainable:
            raise ValueError("Stage 2 full tuning has no trainable language parameters")

    return Stage2ParameterReport(
        total_parameters=sum(parameter.numel() for _, parameter in parameters),
        trainable_parameters=sum(
            parameter.numel() for _, parameter in parameters if parameter.requires_grad
        ),
        trainable_names=trainable,
        resolved_target_names=resolved_target_names,
    )


def load_stage1_dora_for_fresh_stage2(
    base_model: nn.Module,
    processor,
    stage1_adapter_path: str | Path,
    stage2_tuning: Stage2TuningConfig,
    *,
    requested_base_model_reference: str,
    preflight=None,
) -> nn.Module:
    """Continue one validated Stage 1 DoRA adapter with fresh Stage 2 state."""

    from stage1.checkpointing import (
        Stage1PeftMetadataSnapshot,
        decode_stage1_peft_metadata_snapshot,
        load_stage1_peft_artifact,
        preflight_stage1_peft_artifact,
    )

    qwen = resolve_qwen2_5_vl_model(base_model)
    if preflight is None:
        preflight = preflight_stage1_peft_artifact(
            processor,
            qwen.config,
            Path(stage1_adapter_path),
            requested_base_model_reference=requested_base_model_reference,
        )
    metadata = decode_stage1_peft_metadata_snapshot(
        Stage1PeftMetadataSnapshot(
            artifact_dir=preflight.artifact_dir,
            metadata_json=preflight.metadata_json,
        )
    )
    if metadata.tuning_method != "dora":
        raise ValueError("fresh Stage 2 continuation requires a Stage 1 DoRA adapter")
    if stage2_tuning.method == "full":
        raise ValueError("Stage 1 DoRA cannot initialize Stage 2 full tuning")
    if stage2_tuning.method == "lora":
        raise ValueError("Stage 1 DoRA cannot initialize a new Stage 2 LoRA adapter")
    for label, stage1_value, stage2_value in (
        ("rank", metadata.rank, stage2_tuning.r),
        ("alpha", metadata.alpha, stage2_tuning.alpha),
        ("dropout", metadata.dropout, stage2_tuning.dropout),
        (
            "requested target modules",
            metadata.requested_target_suffixes,
            stage2_tuning.target_modules,
        ),
    ):
        if stage1_value != stage2_value:
            raise ValueError(
                f"Stage 1/Stage 2 {label} mismatch: "
                f"stage1={stage1_value}, stage2={stage2_value}"
            )
    current_resolved = resolve_stage2_target_module_names(
        qwen, stage2_tuning.target_modules
    )
    if current_resolved != metadata.resolved_target_names:
        raise ValueError(
            "Stage 1/Stage 2 resolved target mismatch: "
            f"stage1={metadata.resolved_target_names}, stage2={current_resolved}"
        )

    restored = load_stage1_peft_artifact(
        base_model,
        processor,
        Path(stage1_adapter_path),
        requested_base_model_reference=requested_base_model_reference,
        is_trainable=True,
        preflight=preflight,
    )
    resolve_qwen2_5_vl_model(restored).config.stage = "stage2"
    validate_stage2_parameter_boundary(
        restored,
        stage2_tuning,
        allow_sparse_token_rows=True,
    )
    return restored


def build_stage2_tuned_model(
    model: nn.Module,
    tuning: Stage2TuningConfig,
    adapter_path: str | Path | None = None,
    is_trainable: bool = True,
    *,
    token_row_source: str = "base",
    latent_token_ids: Sequence[int] | None = None,
) -> nn.Module:
    """Freeze vision and construct or load a decoder-only Stage 2 adapter."""
    if token_row_source not in ("base", "sparse_adapter"):
        raise ValueError("token_row_source must be base or sparse_adapter")
    if token_row_source == "sparse_adapter":
        if adapter_path is None or latent_token_ids is None:
            raise ValueError(
                "sparse Stage 2 token rows require an adapter and latent token IDs"
            )
        if not is_trainable:
            raise ValueError(
                "sparse Stage 2 inference loading is not supported by this trainer API"
            )
        from stage1.token_rows import initialize_frozen_output_token_rows

        initialize_frozen_output_token_rows(model, latent_token_ids)
    elif latent_token_ids is not None:
        raise ValueError("base token rows do not accept sparse latent token IDs")
    model.visual.requires_grad_(False)
    if tuning.method == "full":
        if adapter_path is not None:
            raise ValueError("full tuning cannot load a PEFT adapter")
        return model

    resolved = resolve_stage2_target_module_names(model, tuning.target_modules)
    if adapter_path is None:
        config = LoraConfig(
            r=tuning.r,
            lora_alpha=tuning.alpha,
            lora_dropout=tuning.dropout,
            target_modules=list(resolved),
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=tuning.use_dora,
            modules_to_save=None,
        )
        wrapped = get_peft_model(model, config)
    else:
        wrapped = PeftModel.from_pretrained(
            model, str(adapter_path), is_trainable=is_trainable
        )
    if token_row_source == "sparse_adapter":
        from stage1.token_rows import (
            attach_sparse_output_token_rows,
            load_sparse_token_rows,
        )

        attach_sparse_output_token_rows(wrapped, latent_token_ids)
        load_sparse_token_rows(wrapped, Path(adapter_path), latent_token_ids)
    if not is_trainable:
        wrapped.requires_grad_(False)
    validate_stage2_parameter_boundary(
        wrapped,
        tuning,
        expect_trainable=is_trainable,
        allow_sparse_token_rows=token_row_source == "sparse_adapter",
    )
    return wrapped


@dataclass(frozen=True)
class Stage2InferencePreflightWitness:
    """Canonical source-bound immutable Stage 2 inference payloads."""

    artifact_dir: Path
    base_model_source: str
    source_selector_contract: tuple[tuple[str, str | bool | None], ...]
    source_architecture_json: bytes
    processor_vocab_size: int | None
    processor_contract: tuple[tuple[str, int], ...]
    metadata_json: bytes
    adapter_config_json: bytes
    adapter_safetensors: bytes
    sparse_safetensors: bytes | None
    adapter_tensor_contract: tuple[tuple[str, tuple[int, ...]], ...]
    execution_profile: str = "legacy"


def _canonical_json(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Stage 2 inference provenance is not canonical JSON"
        ) from error


def _source_architecture_json(config) -> bytes:
    return _canonical_json(capture_qwen2_5_vl_architecture(config))


def _canonical_source_reference(source: str | Path) -> str:
    candidate = Path(source)
    if candidate.exists():
        return str(candidate.resolve())
    return str(source)


def _source_selector_contract(
    source_kwargs: Mapping[str, object],
) -> tuple[tuple[str, str | bool | None], ...]:
    normalized = []
    for name, value in sorted(source_kwargs.items()):
        if isinstance(value, Path):
            value = str(value)
        if value is not None and type(value) not in (str, bool):
            raise ValueError(
                f"invalid Stage 2 source provenance selector {name}: {value!r}"
            )
        normalized.append((name, value))
    return tuple(normalized)


def _decode_stage2_adapter_config(payload: bytes) -> PeftConfig:
    if type(payload) is not bytes:
        raise TypeError("Stage 2 adapter config payload must be bytes")
    try:
        values = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid Stage 2 adapter config bytes") from error
    if not isinstance(values, Mapping):
        raise ValueError("Stage 2 adapter config must be a JSON object")
    values = dict(values)
    values.pop("runtime_config", None)
    try:
        checked = PeftConfig.check_kwargs(**values)
        return PeftConfig.from_peft_type(**checked)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid Stage 2 adapter config values") from error


def _decode_stage2_inference_witness(witness):
    from .checkpointing import (
        Stage2PeftMetadataSnapshot,
        decode_stage2_peft_metadata_snapshot,
    )

    metadata = decode_stage2_peft_metadata_snapshot(
        Stage2PeftMetadataSnapshot(
            artifact_dir=witness.artifact_dir,
            metadata_json=witness.metadata_json,
        )
    )
    adapter_config = _decode_stage2_adapter_config(witness.adapter_config_json)
    if metadata.peft_version != peft.__version__:
        raise ValueError(
            "Stage 2 PEFT version mismatch: "
            f"metadata={metadata.peft_version}, installed={peft.__version__}"
        )
    return metadata, adapter_config


def _preflight_stage2_inference_artifact(
    base_model_path,
    stage2_artifact_path,
    processor,
    *,
    source_kwargs,
    discover_adapter_config,
    load_config,
    artifact_inspection=None,
):
    from accelerate import init_empty_weights
    from transformers import Qwen2_5_VLForConditionalGeneration

    from .checkpointing import (
        Stage2ArtifactInspection,
        Stage2ArtifactKind,
        configure_stage2_metadata,
        decode_stage2_peft_metadata_snapshot,
        inspect_stage2_artifact,
        resolve_latent_token_ids,
        validate_stage2_adapter_tensor_contract,
        validate_stage2_peft_compatibility,
        validate_stage2_sparse_original_base_config,
        validate_stage2_sparse_processor,
    )

    if artifact_inspection is None:
        inspection = inspect_stage2_artifact(stage2_artifact_path)
    else:
        if type(artifact_inspection) is not Stage2ArtifactInspection:
            raise TypeError(
                "artifact_inspection must be a Stage2ArtifactInspection"
            )
        inspection = artifact_inspection
    if inspection.artifact_kind is not Stage2ArtifactKind.PEFT:
        raise ValueError("Stage 2 inference adapter path must be a PEFT artifact")
    metadata_snapshot = inspection.peft_metadata_snapshot
    if metadata_snapshot is None:
        raise ValueError("Stage 2 PEFT classification lost its metadata snapshot")
    requested_artifact = Path(stage2_artifact_path).resolve(strict=False)
    if requested_artifact != metadata_snapshot.artifact_dir:
        raise ValueError("Stage 2 artifact inspection path witness mismatch")
    metadata = decode_stage2_peft_metadata_snapshot(metadata_snapshot)
    if metadata.peft_version != peft.__version__:
        raise ValueError(
            "Stage 2 PEFT version mismatch: "
            f"metadata={metadata.peft_version}, installed={peft.__version__}"
        )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Stage 2 processor must expose its tokenizer")
    artifact = metadata_snapshot.artifact_dir
    # Capture runtime selection before loading the base, alongside other immutable
    # inference witnesses. Older adapter artifacts may omit config.json entirely.
    execution_profile = "legacy"
    saved_config = artifact / "config.json"
    if saved_config.exists() or saved_config.is_symlink():
        from .checkpointing import _read_stage2_regular_payload
        try:
            payload = json.loads(_read_stage2_regular_payload(saved_config,
                label="Stage 2 saved runtime config"))
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("invalid Stage 2 saved runtime config") from error
        if not isinstance(payload, dict):
            raise ValueError("Stage 2 saved runtime config must be an object")
        execution_profile = payload.get("stage2_execution_profile", "legacy")
        if execution_profile not in ("legacy", "latest_external_until"):
            raise ValueError("unknown Stage 2 saved execution profile")
    adapter_config_json = inspection.adapter_config_json
    adapter_safetensors = inspection.adapter_safetensors
    if adapter_config_json is None or adapter_safetensors is None:
        raise ValueError("Stage 2 PEFT inspection lost required payload bytes")
    adapter_config = _decode_stage2_adapter_config(adapter_config_json)
    if metadata.token_row_source == "sparse_adapter":
        token_ids = validate_stage2_sparse_processor(tokenizer, metadata)
        discovered = discover_adapter_config(str(base_model_path), **source_kwargs)
        if discovered is not None:
            raise ValueError(
                "sparse Stage 2 inference requires an adapter-free original "
                f"base model; discovered adapter config at {discovered}"
            )
        source_config = load_config(str(base_model_path), **source_kwargs)
        validate_stage2_sparse_original_base_config(source_config, metadata)
        preflight_config = deepcopy(source_config)
        preflight_config.vocab_size = metadata.resized_vocab_size
        configure_stage2_metadata(
            preflight_config,
            token_ids,
            latent_size=metadata.latent_size,
            supervise_latent_boundaries=metadata.supervise_latent_boundaries,
        )
    else:
        source_config = load_config(str(base_model_path), **source_kwargs)
        token_ids = resolve_latent_token_ids(tokenizer, source_config)
        preflight_config = deepcopy(source_config)
    with init_empty_weights():
        preflight_model = Qwen2_5_VLForConditionalGeneration(preflight_config)
    if preflight_config.tie_word_embeddings is True:
        preflight_model.tie_weights()
    ordered_ids = (
        token_ids.pad_token_id,
        token_ids.start_token_id,
        token_ids.end_token_id,
    )
    if metadata.token_row_source == "sparse_adapter":
        from stage1.token_rows import initialize_frozen_output_token_rows

        initialize_frozen_output_token_rows(preflight_model, ordered_ids)
    validate_stage2_peft_compatibility(
        preflight_model,
        tokenizer,
        metadata,
        adapter_config,
        artifact,
        validate_adapter_tensors=False,
    )
    contract_config = deepcopy(adapter_config)
    contract_config.base_model_name_or_path = preflight_model.__dict__.get(
        "name_or_path", None
    )
    expected_wrapped = get_peft_model(
        preflight_model,
        contract_config,
        adapter_name="default",
    )
    expected_state = get_peft_model_state_dict(
        expected_wrapped,
        adapter_name="default",
        save_embedding_layers=False,
    )
    adapter_tensor_contract = tuple(
        sorted(
            (key, tuple(int(value) for value in tensor.shape))
            for key, tensor in expected_state.items()
        )
    )
    validate_stage2_adapter_tensor_contract(
        adapter_safetensors,
        dict(adapter_tensor_contract),
        metadata,
    )
    sparse_safetensors = inspection.sparse_safetensors
    if (metadata.token_row_source == "sparse_adapter") is not (
        sparse_safetensors is not None
    ):
        raise ValueError("Stage 2 classification sparse payload mismatch")
    return Stage2InferencePreflightWitness(
        artifact_dir=artifact,
        base_model_source=_canonical_source_reference(base_model_path),
        source_selector_contract=_source_selector_contract(source_kwargs),
        source_architecture_json=_source_architecture_json(source_config),
        processor_vocab_size=(
            len(tokenizer)
            if metadata.token_row_source == "sparse_adapter"
            else None
        ),
        processor_contract=tuple(
            zip(
                ("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"),
                ordered_ids,
            )
        ),
        metadata_json=metadata_snapshot.metadata_json,
        adapter_config_json=adapter_config_json,
        adapter_safetensors=adapter_safetensors,
        sparse_safetensors=sparse_safetensors,
        adapter_tensor_contract=adapter_tensor_contract,
        execution_profile=execution_profile,
    )


def load_stage2_model_for_inference(
    base_model_path: str | Path,
    stage2_artifact_path: str | Path,
    processor,
    *,
    source_provenance_kwargs: Mapping[str, object] | None = None,
    model_load_kwargs: Mapping[str, object] | None = None,
    discover_adapter_config=None,
    load_config=None,
    load_model=None,
    artifact_inspection=None,
) -> PeftModel:
    """Load one frozen Stage 2 adapter from its validated original base."""
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
    from transformers.utils.peft_utils import find_adapter_config_file

    from stage1.token_rows import (
        apply_sparse_token_rows_state,
        attach_sparse_output_token_rows,
    )
    from .checkpointing import (
        configure_stage2_metadata,
        prepare_stage2_sparse_token_base,
        resolve_latent_token_ids,
        validate_stage2_peft_compatibility,
    )

    allowed_source_selectors = frozenset(
        {"cache_dir", "revision", "token", "local_files_only"}
    )
    unknown_source_selectors = sorted(
        set(source_provenance_kwargs or {}) - allowed_source_selectors
    )
    if unknown_source_selectors:
        raise ValueError(
            "unsupported Stage 2 source provenance selectors: "
            f"{unknown_source_selectors}"
        )
    source_kwargs = MappingProxyType(dict(source_provenance_kwargs or {}))
    model_kwargs = MappingProxyType(dict(model_load_kwargs or {}))
    misplaced_model_selectors = sorted(
        set(model_kwargs) & allowed_source_selectors
    )
    if misplaced_model_selectors:
        raise ValueError(
            "Stage 2 model options must not contain source provenance selectors: "
            f"{misplaced_model_selectors}"
        )
    _source_selector_contract(source_kwargs)
    discover_adapter_config = discover_adapter_config or find_adapter_config_file
    load_config = load_config or AutoConfig.from_pretrained
    load_model = load_model or Qwen2_5_VLForConditionalGeneration.from_pretrained
    witness = _preflight_stage2_inference_artifact(
        base_model_path,
        stage2_artifact_path,
        processor,
        source_kwargs=source_kwargs,
        discover_adapter_config=discover_adapter_config,
        load_config=load_config,
        artifact_inspection=artifact_inspection,
    )
    base_model = load_model(
        str(base_model_path), **source_kwargs, **model_kwargs
    )
    metadata, adapter_config = _decode_stage2_inference_witness(witness)
    tokenizer = getattr(processor, "tokenizer", None)
    if (
        _canonical_source_reference(base_model_path) != witness.base_model_source
        or _source_selector_contract(source_kwargs)
        != witness.source_selector_contract
    ):
        raise ValueError("Stage 2 inference source witness mismatch")
    if (
        _source_architecture_json(base_model.config)
        != witness.source_architecture_json
    ):
        raise ValueError("Stage 2 inference runtime architecture witness mismatch")
    if metadata.token_row_source == "sparse_adapter":
        token_ids = prepare_stage2_sparse_token_base(
            base_model, tokenizer, metadata
        )
        validate_stage2_peft_compatibility(
            base_model,
            tokenizer,
            metadata,
            adapter_config,
            witness.artifact_dir,
            adapter_safetensors=witness.adapter_safetensors,
            expected_adapter_tensor_shapes=dict(
                witness.adapter_tensor_contract
            ),
        )
    else:
        token_ids = resolve_latent_token_ids(tokenizer, base_model.config)
        validate_stage2_peft_compatibility(
            base_model,
            tokenizer,
            metadata,
            adapter_config,
            witness.artifact_dir,
            adapter_safetensors=witness.adapter_safetensors,
            expected_adapter_tensor_shapes=dict(
                witness.adapter_tensor_contract
            ),
        )
        configure_stage2_metadata(
            base_model.config,
            token_ids,
            latent_size=metadata.latent_size,
            supervise_latent_boundaries=metadata.supervise_latent_boundaries,
        )
    tuning = Stage2TuningConfig(
        method=metadata.tuning_method,
        r=metadata.lora_r,
        alpha=metadata.lora_alpha,
        dropout=metadata.lora_dropout,
        target_modules=metadata.requested_target_modules,
    )
    current_contract = tuple(
        zip(
            ("<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"),
            (
                token_ids.pad_token_id,
                token_ids.start_token_id,
                token_ids.end_token_id,
            ),
        )
    )
    if (
        (
            witness.processor_vocab_size is not None
            and len(tokenizer) != witness.processor_vocab_size
        )
        or current_contract != witness.processor_contract
    ):
        raise ValueError("Stage 2 inference processor witness mismatch")
    adapter_config.inference_mode = True
    # The artifact config was already provenance-validated. Align only this
    # freshly decoded copy with the witnessed relocated runtime source before
    # PEFT performs the same assignment (and otherwise emits a rename warning).
    adapter_config.base_model_name_or_path = base_model.__dict__.get(
        "name_or_path", None
    )
    wrapped = get_peft_model(
        base_model,
        adapter_config,
        adapter_name="default",
    )
    try:
        adapter_state = load_safetensors(witness.adapter_safetensors)
    except SafetensorError as error:
        raise ValueError("invalid witnessed Stage 2 adapter bytes") from error
    load_result = set_peft_model_state_dict(
        wrapped,
        adapter_state,
        adapter_name="default",
    )
    if load_result.unexpected_keys:
        raise ValueError(
            "unexpected Stage 2 adapter state keys after application: "
            f"{sorted(load_result.unexpected_keys)}"
        )
    if metadata.token_row_source == "sparse_adapter":
        ordered_ids = (
            token_ids.pad_token_id,
            token_ids.start_token_id,
            token_ids.end_token_id,
        )
        attach_sparse_output_token_rows(wrapped, ordered_ids)
        apply_sparse_token_rows_state(
            wrapped, witness.sparse_safetensors, ordered_ids
        )
    if witness.execution_profile != "legacy":
        base_model.config.stage2_execution_profile = witness.execution_profile
        base_model.config.vision_config.stage2_execution_profile = witness.execution_profile
        for block in base_model.visual.blocks:
            block.attn.stage2_execution_profile = witness.execution_profile
    wrapped.requires_grad_(False)
    wrapped.eval()
    validate_stage2_parameter_boundary(
        wrapped,
        tuning,
        expect_trainable=False,
        allow_sparse_token_rows=metadata.token_row_source == "sparse_adapter",
    )
    return wrapped
