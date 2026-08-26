"""Stage 1 full, legacy LoRA, and DoRA model composition."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn

from peft_common import resolve_qwen2_5_vl_model, resolve_qwen_decoder_target_names
from stage1.token_rows import (
    SparseOutputTokenRows,
    attach_sparse_output_token_rows,
    initialize_frozen_output_token_rows,
)
from stage1.types import Stage1TuningConfig


_ADAPTER_NAME = "default"
_EMBEDDING_TARGET = "model.embed_tokens"
_ATTENTION_TARGET_SUFFIXES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
_MLP_TARGET_SUFFIXES = frozenset({"gate_proj", "up_proj", "down_proj"})


@dataclass(frozen=True)
class Stage1ParameterReport:
    """Trainability evidence with explicit element-count and row-count units.

    ``visual``, ``base_decoder``, ``decoder_adapters``, and keys ending in
    ``_parameters`` count scalar parameter elements. Keys ending in ``_rows``
    count logical vocabulary rows and therefore are not additive with the
    parameter-element categories.
    """

    trainable_names: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int
    categories: Mapping[str, int]


def _parameter_shape(parameter: nn.Parameter) -> torch.Size:
    return (
        torch.Size(parameter.ds_shape)
        if getattr(parameter, "ds_id", None) is not None
        and hasattr(parameter, "ds_shape")
        else torch.Size(parameter.shape)
    )


def _parameter_elements(parameter: nn.Parameter) -> int:
    return math.prod(_parameter_shape(parameter))


def _named_parameters(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    return tuple(model.named_parameters())


def _parameter_ids(module: nn.Module) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


def _validated_latent_token_ids(
    qwen: nn.Module, latent_token_ids: Sequence[int]
) -> tuple[int, int, int]:
    try:
        values = tuple(latent_token_ids)
    except TypeError as error:
        raise ValueError("latent token IDs must be a sequence") from error
    if len(values) != 3:
        raise ValueError("latent token IDs must contain exactly three values")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("latent token IDs must be integers")
    if len(set(values)) != 3:
        raise ValueError("latent token IDs must be unique")

    embedding = qwen.model.embed_tokens
    token_adapter = getattr(embedding, "token_adapter", None)
    base_embedding = getattr(token_adapter, "base_layer", embedding)
    output_size = getattr(qwen.lm_head, "out_features", None)
    if not isinstance(base_embedding, nn.Embedding):
        raise TypeError("Qwen2.5-VL input embedding is not torch.nn.Embedding")
    if not isinstance(output_size, int):
        raise TypeError("Qwen2.5-VL lm_head does not expose out_features")
    input_size = base_embedding.num_embeddings
    if input_size != output_size:
        raise ValueError(
            "Qwen2.5-VL input and output vocabulary sizes differ: "
            f"input={input_size}, output={output_size}"
        )
    if any(value < 0 or value >= input_size for value in values):
        raise ValueError(
            f"latent token IDs must be in-vocabulary for vocab size {input_size}"
        )
    return values


def _resolve_linear_decoder_targets(
    qwen: nn.Module, requested_suffixes: Sequence[str]
) -> tuple[str, ...]:
    resolved = resolve_qwen_decoder_target_names(qwen, requested_suffixes)
    modules = dict(qwen.named_modules())
    canonical: list[str] = []
    missing: list[str] = []
    for suffix in requested_suffixes:
        if suffix in _ATTENTION_TARGET_SUFFIXES:
            owner_name = "self_attn"
        elif suffix in _MLP_TARGET_SUFFIXES:
            owner_name = "mlp"
        else:  # The shared normalizer normally rejects this before resolution.
            raise ValueError(f"Unsupported decoder target suffix: {suffix}")
        suffix_matches: list[str] = []
        for layer_index, layer in enumerate(qwen.model.layers):
            owner = getattr(layer, owner_name, None)
            projection = getattr(owner, suffix, None)
            path = f"model.layers.{layer_index}.{owner_name}.{suffix}"
            if projection is not None and modules.get(path) is projection:
                suffix_matches.append(path)
        if not suffix_matches:
            missing.append(suffix)
        canonical.extend(suffix_matches)

    canonical_set = set(canonical)
    unrelated = [name for name in resolved if name not in canonical_set]
    if unrelated:
        raise ValueError(
            "Stage 1 PEFT targets must be canonical decoder-layer projections: "
            + ", ".join(unrelated)
        )
    if missing:
        raise ValueError(
            f"Requested canonical decoder targets did not match: {missing}"
        )
    invalid = [
        name
        for name in canonical
        if not isinstance(getattr(modules[name], "base_layer", modules[name]), nn.Linear)
    ]
    if invalid:
        raise ValueError(
            "Stage 1 PEFT targets must be linear decoder modules: "
            + ", ".join(invalid)
        )
    return tuple(canonical)


def _configured_target_patterns_match(
    qwen: nn.Module,
    configured_targets,
    expected_targets: Sequence[str],
) -> bool:
    if (
        isinstance(configured_targets, str)
        or not isinstance(configured_targets, (list, set, tuple))
        or not configured_targets
        or any(
            not isinstance(target, str) or not target
            for target in configured_targets
        )
    ):
        return False
    patterns = tuple(configured_targets)
    if len(patterns) != len(set(patterns)):
        return False
    matched = {
        name
        for name, _ in qwen.named_modules()
        if any(
            name == pattern or name.endswith(f".{pattern}")
            for pattern in patterns
        )
    }
    return matched == set(expected_targets)


def _peft_model(model: nn.Module) -> PeftModel:
    candidate = getattr(model, "module", model)
    if not isinstance(candidate, PeftModel):
        raise ValueError("Stage 1 adapter tuning requires a PeftModel")
    return candidate


def _adapter_config(model: PeftModel):
    if set(model.peft_config) != {_ADAPTER_NAME}:
        raise ValueError("Stage 1 PEFT model must contain exactly one default adapter")
    return model.peft_config[_ADAPTER_NAME]


def _require_adapter_parameters(
    module: nn.Module,
    target_name: str,
    component_name: str,
) -> tuple[nn.Parameter, ...]:
    container = getattr(module, component_name, None)
    if container is None or _ADAPTER_NAME not in container:
        raise ValueError(
            f"Stage 1 adapter target {target_name} is missing {component_name}"
        )
    parameters = tuple(container[_ADAPTER_NAME].parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError(
            f"Stage 1 adapter target {target_name} has inactive {component_name}"
        )
    return parameters


def _adapter_parameter_ids(
    qwen: nn.Module,
    target_names: Sequence[str],
    *,
    use_dora: bool,
) -> set[int]:
    modules = dict(qwen.named_modules())
    allowed: set[int] = set()
    for target_name in target_names:
        module = modules[target_name]
        for component in ("lora_A", "lora_B"):
            allowed.update(
                id(parameter)
                for parameter in _require_adapter_parameters(
                    module, target_name, component
                )
            )

        magnitude = getattr(module, "lora_magnitude_vector", None)
        has_magnitude = magnitude is not None and _ADAPTER_NAME in magnitude
        if use_dora:
            if not has_magnitude:
                raise ValueError(
                    f"Stage 1 adapter target {target_name} is missing DoRA magnitude vector"
                )
            magnitude_parameters = tuple(magnitude[_ADAPTER_NAME].parameters())
            if not magnitude_parameters or any(
                not parameter.requires_grad for parameter in magnitude_parameters
            ):
                raise ValueError(
                    f"Stage 1 adapter target {target_name} has inactive DoRA magnitude vector"
                )
            allowed.update(id(parameter) for parameter in magnitude_parameters)
        elif has_magnitude:
            raise ValueError("legacy Stage 1 LoRA must not contain DoRA magnitude vectors")
    return allowed


def _validate_selective_input_rows(
    qwen: nn.Module,
    config,
    latent_token_ids: tuple[int, int, int],
) -> tuple[nn.Parameter, nn.Embedding]:
    configured = config.trainable_token_indices
    if not isinstance(configured, dict) or set(configured) != {_EMBEDDING_TARGET}:
        raise ValueError(
            "Stage 1 PEFT must configure only model.embed_tokens as selective input rows"
        )
    configured_ids = tuple(configured[_EMBEDDING_TARGET])
    if len(configured_ids) != 3:
        raise ValueError("Stage 1 PEFT requires exactly three selective input rows")
    if configured_ids != latent_token_ids:
        raise ValueError(
            "selective input token IDs do not match latent token IDs: "
            f"configured={configured_ids}, expected={latent_token_ids}"
        )

    embedding = qwen.model.embed_tokens
    token_adapter = getattr(embedding, "token_adapter", None)
    if token_adapter is None:
        raise ValueError("Stage 1 PEFT selective input-row adapter is missing")
    token_indices = getattr(token_adapter, "token_indices", None)
    if not isinstance(token_indices, dict) or set(token_indices) != {_ADAPTER_NAME}:
        raise ValueError("Stage 1 PEFT selective input-row topology is invalid")
    actual_ids = tuple(token_indices[_ADAPTER_NAME])
    if len(actual_ids) != 3:
        raise ValueError("Stage 1 PEFT requires exactly three selective input rows")
    if actual_ids != latent_token_ids:
        raise ValueError(
            "selective input token IDs do not match latent token IDs: "
            f"actual={actual_ids}, expected={latent_token_ids}"
        )

    delta_container = getattr(token_adapter, "trainable_tokens_delta", None)
    if delta_container is None or set(delta_container) != {_ADAPTER_NAME}:
        raise ValueError("Stage 1 PEFT selective input-row parameters are missing")
    delta = delta_container[_ADAPTER_NAME]
    base_embedding = getattr(token_adapter, "base_layer", None)
    if not isinstance(base_embedding, nn.Embedding):
        raise TypeError("Stage 1 PEFT selective input rows must wrap an embedding")
    expected_shape = torch.Size((3, base_embedding.embedding_dim))
    actual_shape = _parameter_shape(delta)
    if actual_shape != expected_shape:
        raise ValueError(
            "Stage 1 PEFT requires exactly three selective input rows with shape "
            f"{tuple(expected_shape)}, got {tuple(actual_shape)}"
        )
    if not delta.requires_grad:
        raise ValueError("Stage 1 PEFT selective input rows are frozen")
    return delta, base_embedding


def _validate_selective_output_rows(
    qwen: nn.Module, latent_token_ids: tuple[int, int, int]
) -> tuple[nn.Parameter, nn.Linear]:
    rows = qwen.lm_head
    if not isinstance(rows, SparseOutputTokenRows):
        raise ValueError("Stage 1 PEFT sparse output-row adapter is missing")
    actual_ids = tuple(rows.token_ids.detach().cpu().tolist())
    if len(actual_ids) != 3:
        raise ValueError("Stage 1 PEFT requires exactly three selective output rows")
    if actual_ids != latent_token_ids:
        raise ValueError(
            "selective output token IDs do not match latent token IDs: "
            f"actual={actual_ids}, expected={latent_token_ids}"
        )
    expected_shape = torch.Size((3, rows.base_head.in_features))
    actual_shape = _parameter_shape(rows.output_delta)
    if actual_shape != expected_shape:
        raise ValueError(
            "Stage 1 PEFT requires exactly three selective output rows with shape "
            f"{tuple(expected_shape)}, got {tuple(actual_shape)}"
        )
    if not rows.output_delta.requires_grad:
        raise ValueError("Stage 1 PEFT selective output rows are frozen")
    return rows.output_delta, rows.base_head


def _report(
    model: nn.Module,
    *,
    categories: Mapping[str, int],
) -> Stage1ParameterReport:
    parameters = _named_parameters(model)
    return Stage1ParameterReport(
        trainable_names=tuple(
            name for name, parameter in parameters if parameter.requires_grad
        ),
        trainable_parameters=sum(
            _parameter_elements(parameter)
            for _, parameter in parameters
            if parameter.requires_grad
        ),
        total_parameters=sum(
            _parameter_elements(parameter) for _, parameter in parameters
        ),
        categories=dict(categories),
    )


def _validate_full_boundary(
    model: nn.Module,
    qwen: nn.Module,
    latent_token_ids: tuple[int, int, int],
) -> Stage1ParameterReport:
    if isinstance(getattr(model, "module", model), PeftModel):
        raise ValueError("Stage 1 full tuning must not contain a PEFT adapter")
    if isinstance(qwen.lm_head, SparseOutputTokenRows) or hasattr(
        qwen.model.embed_tokens, "token_adapter"
    ):
        raise ValueError("Stage 1 full tuning must not contain sparse token-row adapters")

    visual_ids = _parameter_ids(qwen.visual)
    embedding_ids = _parameter_ids(qwen.model.embed_tokens)
    head_ids = _parameter_ids(qwen.lm_head)
    parameters = _named_parameters(model)
    trainable_visual = [
        name
        for name, parameter in parameters
        if id(parameter) in visual_ids and parameter.requires_grad
    ]
    if trainable_visual:
        raise ValueError(f"visual parameter is trainable: {trainable_visual[0]}")
    frozen_language = [
        name
        for name, parameter in parameters
        if id(parameter) not in visual_ids and not parameter.requires_grad
    ]
    if frozen_language:
        raise ValueError(
            f"full tuning language parameter is frozen: {frozen_language[0]}"
        )

    input_embedding = qwen.model.embed_tokens
    output_head = qwen.lm_head
    if not isinstance(input_embedding, nn.Embedding) or not isinstance(
        output_head, nn.Linear
    ):
        raise TypeError("Stage 1 full tuning requires ordinary Qwen embedding and LM head")
    vocab_size = input_embedding.num_embeddings
    latent_count = len(latent_token_ids)
    base_decoder = sum(
        _parameter_elements(parameter)
        for _, parameter in parameters
        if parameter.requires_grad
        and id(parameter) not in visual_ids | embedding_ids | head_ids
    )
    return _report(
        model,
        categories={
            "visual": 0,
            "base_decoder": base_decoder,
            "decoder_adapters": 0,
            "ordinary_embedding_rows": vocab_size - latent_count,
            "ordinary_lm_head_rows": output_head.out_features - latent_count,
            "latent_input_rows": latent_count,
            "latent_output_rows": latent_count,
            "latent_input_parameters": latent_count * input_embedding.embedding_dim,
            "latent_output_parameters": latent_count * output_head.in_features,
        },
    )


def _validate_adapter_boundary(
    model: nn.Module,
    qwen: nn.Module,
    tuning: Stage1TuningConfig,
    latent_token_ids: tuple[int, int, int],
) -> Stage1ParameterReport:
    peft_model = _peft_model(model)
    config = _adapter_config(peft_model)
    if bool(config.use_dora) is not (tuning.method == "dora"):
        raise ValueError("Stage 1 PEFT adapter method does not match tuning")
    if config.bias != "none" or config.modules_to_save is not None:
        raise ValueError("Stage 1 PEFT adapter has unsupported saved base parameters")

    expected_targets = _resolve_linear_decoder_targets(qwen, tuning.target_suffixes)
    configured_targets = config.target_modules
    injected_targets = getattr(
        peft_model.base_model, "targeted_module_names", None
    )
    if (
        not _configured_target_patterns_match(
            qwen, configured_targets, expected_targets
        )
        or not isinstance(injected_targets, list)
        or any(not isinstance(name, str) for name in injected_targets)
        or len(injected_targets) != len(set(injected_targets))
        or set(injected_targets) != set(expected_targets)
    ):
        raise ValueError("Stage 1 PEFT adapter targets do not match decoder tuning")
    adapter_ids = _adapter_parameter_ids(
        qwen, expected_targets, use_dora=tuning.method == "dora"
    )
    input_delta, base_embedding = _validate_selective_input_rows(
        qwen, config, latent_token_ids
    )
    output_delta, base_head = _validate_selective_output_rows(
        qwen, latent_token_ids
    )

    allowed_ids = adapter_ids | {id(input_delta), id(output_delta)}
    visual_ids = _parameter_ids(qwen.visual)
    embedding_ids = _parameter_ids(qwen.model.embed_tokens)
    head_ids = _parameter_ids(qwen.lm_head)
    decoder_ids = _parameter_ids(qwen.model)
    leaks: dict[str, list[str]] = {
        "visual": [],
        "embedding": [],
        "lm_head": [],
        "decoder": [],
    }
    for name, parameter in _named_parameters(model):
        identity = id(parameter)
        if not parameter.requires_grad or identity in allowed_ids:
            continue
        if identity in visual_ids:
            leaks["visual"].append(name)
        elif identity in embedding_ids:
            leaks["embedding"].append(name)
        elif identity in head_ids:
            leaks["lm_head"].append(name)
        elif identity in decoder_ids:
            leaks["decoder"].append(name)
        else:
            leaks["decoder"].append(name)

    if leaks["visual"]:
        raise ValueError(f"visual parameter is trainable: {leaks['visual'][0]}")
    if leaks["embedding"]:
        raise ValueError(
            f"ordinary embedding parameter is trainable: {leaks['embedding'][0]}"
        )
    if leaks["lm_head"]:
        raise ValueError(
            f"ordinary LM-head parameter is trainable: {leaks['lm_head'][0]}"
        )
    if leaks["decoder"]:
        raise ValueError(
            f"base decoder parameter is trainable: {leaks['decoder'][0]}"
        )

    parameters = _named_parameters(model)
    trainable_ids = {
        id(parameter) for _, parameter in parameters if parameter.requires_grad
    }
    missing = allowed_ids - trainable_ids
    if missing:
        raise ValueError("Stage 1 PEFT required adapter parameter is not trainable")
    decoder_adapter_parameters = sum(
        _parameter_elements(parameter)
        for _, parameter in parameters
        if id(parameter) in adapter_ids
    )
    return _report(
        model,
        categories={
            "visual": 0,
            "base_decoder": 0,
            "decoder_adapters": decoder_adapter_parameters,
            "ordinary_embedding_rows": 0,
            "ordinary_lm_head_rows": 0,
            "latent_input_rows": len(latent_token_ids),
            "latent_output_rows": len(latent_token_ids),
            "latent_input_parameters": _parameter_elements(input_delta),
            "latent_output_parameters": _parameter_elements(output_delta),
        },
    )


def validate_stage1_trainable_boundary(
    model: nn.Module,
    tuning: Stage1TuningConfig,
    latent_token_ids: Sequence[int],
) -> Stage1ParameterReport:
    """Fail closed unless trainability matches the selected Stage 1 method."""
    qwen = resolve_qwen2_5_vl_model(model)
    ids = _validated_latent_token_ids(qwen, latent_token_ids)
    if tuning.method == "full":
        return _validate_full_boundary(model, qwen, ids)
    return _validate_adapter_boundary(model, qwen, tuning, ids)


def build_stage1_tuned_model(
    base_model: nn.Module,
    tuning: Stage1TuningConfig,
    latent_token_ids: Sequence[int],
) -> nn.Module:
    """Freeze vision and compose the requested Stage 1 language topology."""
    qwen = resolve_qwen2_5_vl_model(base_model)
    ids = _validated_latent_token_ids(qwen, latent_token_ids)

    if tuning.method == "full":
        qwen.requires_grad_(True)
        qwen.visual.requires_grad_(False)
        validate_stage1_trainable_boundary(base_model, tuning, ids)
        return base_model

    initialize_frozen_output_token_rows(qwen, ids)
    qwen.visual.requires_grad_(False)
    resolved = _resolve_linear_decoder_targets(qwen, tuning.target_suffixes)
    config = LoraConfig(
        r=tuning.rank,
        lora_alpha=tuning.alpha,
        lora_dropout=tuning.dropout,
        target_modules=list(resolved),
        use_dora=tuning.method == "dora",
        trainable_token_indices={_EMBEDDING_TARGET: list(ids)},
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, config)
    attach_sparse_output_token_rows(resolve_qwen2_5_vl_model(model), ids)
    validate_stage1_trainable_boundary(model, tuning, ids)
    return model


__all__ = [
    "Stage1ParameterReport",
    "build_stage1_tuned_model",
    "validate_stage1_trainable_boundary",
]
