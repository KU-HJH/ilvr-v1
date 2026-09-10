#!/usr/bin/env python3
"""Run and compare opt-in CoMT visual causal interventions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable

from evaluation.scoring import parse_gold_answer, score_answer
from evaluation.visual_metrics import aggregate_rows, matched_image_groups, paired_summary


VISUAL_ABLATION_MODES = (
    "baseline",
    "no_input_image",
    "no_latent_execution",
    "latent_image_attention_block",
    "global_image_attention_block",
    "image_replacement",
)
MASKED_MODES = {
    "latent_image_attention_block",
    "global_image_attention_block",
}


def _read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_identity(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert evidence to strict JSON, representing nonfinite diagnostics as null."""
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n"
            )


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def select_loader_module_name(
    model_dir: str | os.PathLike[str], attention_backend: str
) -> str:
    """Select the existing loader that defines the requested runtime bridge."""
    if attention_backend == "sdpa":
        return "eval"
    if attention_backend != "native":
        raise ValueError("attention_backend must be native or sdpa")
    artifact = Path(model_dir)
    if (artifact / "stage1_peft_metadata.json").is_file():
        return "eval"
    config_path = artifact / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("stage") == "stage1":
            return "eval"
    return "src.evaluate_deepseed"


def load_analysis_artifacts(
    *, model_dir: str, base_model_dir: str | None, attention_backend: str
):
    module_name = select_loader_module_name(model_dir, attention_backend)
    module = importlib.import_module(module_name)
    return module.load_evaluation_artifacts(
        model_dir=model_dir,
        base_model_dir=base_model_dir,
        cache_dir=None,
    )


def _source_index(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        index = row.get("index", position)
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("source row index must be an integer")
        if index in result:
            raise ValueError(f"duplicate source index: {index}")
        result[index] = dict(row)
    return result


def _select_samples(
    test_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int] | str | None,
    limit: int | None,
) -> list[tuple[int, dict[str, Any] | None]]:
    source = _source_index(source_rows)
    if isinstance(indices, str):
        indices = _parse_indices(indices)
    if indices is not None:
        selected = [(index, source.get(index)) for index in indices]
    elif test_rows is source_rows or test_rows == source_rows:
        selected = list(source.items())
    else:
        identities: dict[str, list[int]] = {}
        for index, row in source.items():
            identities.setdefault(_value_identity(row), []).append(index)
        selected = []
        for test_row in test_rows:
            explicit = test_row.get("index")
            if isinstance(explicit, int) and not isinstance(explicit, bool):
                selected.append((explicit, source.get(explicit)))
                continue
            matches = identities.get(_value_identity(test_row), [])
            if len(matches) != 1:
                raise ValueError(
                    "each test row must match exactly one source row when --indices is omitted"
                )
            selected.append((matches[0], source[matches[0]]))
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        selected = selected[:limit]
    return selected


def _normalize_question(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _normalize_replacement_map(
    value: Mapping[int | str, int | str] | str | os.PathLike[str] | None,
) -> dict[int, int]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        raw = value
    else:
        text = str(value)
        candidate = Path(text)
        raw = json.loads(candidate.read_text(encoding="utf-8") if candidate.is_file() else text)
    if not isinstance(raw, Mapping):
        raise ValueError("replacement map must be a JSON object")
    result: dict[int, int] = {}
    for recipient, donor in raw.items():
        try:
            recipient_index = int(recipient)
            donor_index = int(donor)
        except (TypeError, ValueError) as error:
            raise ValueError("replacement map keys and values must be integer indices") from error
        if str(recipient_index) != str(recipient) and not isinstance(recipient, int):
            raise ValueError("replacement map keys must be canonical integer indices")
        if isinstance(donor, bool):
            raise ValueError("replacement map values must be integer indices")
        result[recipient_index] = donor_index
    return result


def _replacement_sample(
    recipient_index: int,
    recipient: Mapping[str, Any],
    source: Mapping[int, Mapping[str, Any]],
    replacements: Mapping[int, int],
) -> tuple[dict[str, Any], int, object]:
    if recipient_index not in replacements:
        raise ValueError(f"replacement map is missing recipient index {recipient_index}")
    donor_index = replacements[recipient_index]
    donor = source.get(donor_index)
    if donor is None:
        raise ValueError(f"replacement donor index {donor_index} is missing")
    if _normalize_question(recipient.get("text_input")) != _normalize_question(
        donor.get("text_input")
    ):
        raise ValueError("replacement pair has a normalized question mismatch")
    recipient_gold = parse_gold_answer(recipient.get("original_final_answer"))
    donor_gold = parse_gold_answer(donor.get("original_final_answer"))
    if recipient_gold.status != "parsed" or donor_gold.status != "parsed":
        raise ValueError("replacement pair requires valid parsed gold labels")
    if recipient_gold.normalized == donor_gold.normalized:
        raise ValueError("replacement pair requires differing gold labels")
    sample = dict(recipient)
    sample["image_input"] = donor.get("image_input", [])
    return sample, donor_index, donor.get("original_final_answer")


def _category(sample: Mapping[str, Any]) -> tuple[str, str]:
    explicit = sample.get("category")
    if isinstance(explicit, str) and explicit:
        return explicit, "sample.category"
    images = sample.get("image_input", [])
    if isinstance(images, str):
        images = [images]
    if isinstance(images, list) and images and isinstance(images[0], str):
        parent = Path(images[0]).parent.name
        if parent:
            return parent, "first_image_parent"
    return "unknown", "unavailable"


def _attention_classes(qwen_model: Any) -> tuple[list[str], list[str]]:
    layers = getattr(getattr(qwen_model, "model", None), "layers", None)
    blocks = getattr(getattr(qwen_model, "visual", None), "blocks", None)
    if layers is None or not layers:
        raise TypeError("model must expose every decoder layer")
    if blocks is None or not blocks:
        raise TypeError("model must expose every vision block")
    decoder = []
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise TypeError(f"decoder layer {index} has no self_attn module")
        decoder.append(type(attention).__name__)
    vision = []
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        if attention is None:
            raise TypeError(f"vision block {index} has no attn module")
        vision.append(type(attention).__name__)
    return decoder, vision


def _runtime_metadata(model: Any, qwen_model: Any, attention_backend: str) -> dict[str, Any]:
    decoder, vision = _attention_classes(qwen_model)
    parameters = list(model.parameters()) if hasattr(model, "parameters") else []
    config = qwen_model.config
    vision_config = getattr(config, "vision_config", None)
    return {
        "requested_attention_backend": attention_backend,
        "stage": getattr(config, "stage", None),
        "decoder_attention_classes": decoder,
        "decoder_attention_layer_count": len(decoder),
        "vision_attention_classes": vision,
        "vision_attention_block_count": len(vision),
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
        "configured_torch_dtype": str(getattr(config, "torch_dtype", None)),
        "configured_decoder_attention": getattr(config, "_attn_implementation", None),
        "configured_vision_attention": getattr(vision_config, "_attn_implementation", None),
        "input_tensor_dtypes": {},
    }


def _require_mask_compatible(runtime: Mapping[str, Any], mode: str, backend: str) -> None:
    if mode not in MASKED_MODES:
        return
    classes = runtime["decoder_attention_classes"]
    if backend != "sdpa" or any(name != "Qwen2_5_VLSdpaAttention" for name in classes):
        raise ValueError(
            "masked visual analysis requires Qwen2_5_VLSdpaAttention at every decoder layer"
        )


def _resolve_image_token_id(tokenizer: Any, config: Any) -> int:
    tokenizer_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if tokenizer_id is None or tokenizer_id == unknown_id:
        raise ValueError("tokenizer does not contain <|image_pad|>")
    tokenizer_id = int(tokenizer_id)
    config_id = getattr(config, "image_token_id", None)
    if config_id is None or int(config_id) != tokenizer_id:
        raise ValueError(
            f"image token ID mismatch: tokenizer={tokenizer_id}, config={config_id}"
        )
    return tokenizer_id


def _sync_for_device(torch: Any, device: Any) -> None:
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _default_generation_runner(**context):
    from evaluation.continuous_latent import ContinuousLatentController

    return ContinuousLatentController().generate(
        model=context["qwen_model"],
        input_ids=context["inputs"]["input_ids"],
        attention_mask=context["inputs"]["attention_mask"],
        token_ids=context["token_ids"],
        generation_config=context["generation_config"],
        pixel_values=context["inputs"].get("pixel_values"),
        image_grid_thw=context["inputs"].get("image_grid_thw"),
        intervention=context["intervention"],
    )


def _error_row(index: int, sample: Mapping[str, Any] | None, error: Exception | str) -> dict[str, Any]:
    sample = sample or {}
    gold_value = sample.get("original_final_answer")
    gold = parse_gold_answer(gold_value)
    invalid_reason = (
        gold.detail if gold.status == "invalid_gold" else gold.status if gold.status != "parsed" else None
    )
    category, category_source = _category(sample)
    return {
        "index": index,
        "sample_id": sample.get("id", index),
        "category": category,
        "category_source": category_source,
        "condition": None,
        "replacement_donor_index": None,
        "gold_source": "recipient",
        "prompt_identity": None,
        "prompt_token_length": None,
        "generated_suffix_token_ids": [],
        "raw_output": None,
        "assistant_only": None,
        "prediction": None,
        "extracted_final_answer": None,
        "prediction_parse_status": "inference_error",
        "prediction_parse_detail": None,
        "prediction_normalized": "",
        "gold_final_answer": gold_value,
        "gold_parse_status": gold.status,
        "gold_parse_detail": gold.detail,
        "gold_normalized": gold.normalized,
        "invalid_gold_reason": invalid_reason,
        "match": False,
        "latent_start_positions": {"suffix": [], "absolute": []},
        "latent_end_positions": {"suffix": [], "absolute": []},
        "latent_start_count": None,
        "latent_block_count": None,
        "truncated_latent_block_count": None,
        "has_truncated_latent_block": None,
        "logits_events": [],
        "intervention_forward_records": [],
        "inference_time_sec": None,
        "wall_time_sec": None,
        "stop_reason": None,
        "error": str(error),
    }


def run_analysis(
    *,
    model_dir: str | os.PathLike[str],
    base_model_dir: str | os.PathLike[str] | None,
    test_data_path: str | os.PathLike[str],
    source_data_path: str | os.PathLike[str],
    indices: Sequence[int] | str | None,
    output_dir: str | os.PathLike[str],
    visual_ablation_mode: str,
    attention_backend: str,
    replacement_map: Mapping[int | str, int | str] | str | os.PathLike[str] | None,
    max_new_tokens: int,
    max_sequence_length: int | None,
    repetition_penalty: float,
    temperature: float,
    top_p: float,
    limit: int | None,
    artifact_loader: Callable[..., tuple[Any, Any]] | None = None,
    generation_runner: Callable[..., Any] | None = None,
    image_loader: Callable[[list[str]], Any] | None = None,
) -> dict[str, Any]:
    """Run one fresh intervention condition and serialize complete row evidence."""
    if visual_ablation_mode not in VISUAL_ABLATION_MODES:
        raise ValueError(f"unknown visual ablation mode: {visual_ablation_mode}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    test_path = Path(test_data_path).resolve(strict=True)
    source_path = Path(source_data_path).resolve(strict=True)
    started = time.perf_counter()
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "commit": _git_commit(),
        "command": " ".join(shlex.quote(argument) for argument in sys.argv),
        "model_dir": str(model_dir),
        "base_model_dir": str(base_model_dir) if base_model_dir is not None else None,
        "condition": visual_ablation_mode,
        "test_data": {"path": str(test_path), "sha256": _sha256(test_path)},
        "source_data": {"path": str(source_path), "sha256": _sha256(source_path)},
        "loader_module": select_loader_module_name(model_dir, attention_backend),
    }
    rows: list[dict[str, Any]] = []
    try:
        import torch
        from evaluation.continuous_latent import build_continuous_generation_config
        from evaluation.runtime import prepare_evaluation_input
        from evaluation.visual_interventions import VisualIntervention
        from multinode_evaluation import resolve_evaluation_image_paths
        from peft_common import resolve_qwen2_5_vl_model
        from stage2.checkpointing import resolve_latent_token_ids

        official = importlib.import_module("eval")
        test_rows = _read_jsonl(test_path)
        source_rows = test_rows if test_path == source_path else _read_jsonl(source_path)
        selected = _select_samples(test_rows, source_rows, indices, limit)
        source = _source_index(source_rows)
        replacements = _normalize_replacement_map(replacement_map)
        if visual_ablation_mode == "image_replacement" and not replacements:
            raise ValueError("image_replacement requires --replacement-map")
        if visual_ablation_mode != "image_replacement" and replacements:
            raise ValueError("--replacement-map is valid only for image_replacement")

        loader = artifact_loader or load_analysis_artifacts
        model, processor = loader(
            model_dir=str(model_dir),
            base_model_dir=str(base_model_dir) if base_model_dir is not None else None,
            attention_backend=attention_backend,
        )
        qwen_model = resolve_qwen2_5_vl_model(model)
        runtime = _runtime_metadata(model, qwen_model, attention_backend)
        _require_mask_compatible(runtime, visual_ablation_mode, attention_backend)
        image_token_id = _resolve_image_token_id(processor.tokenizer, qwen_model.config)
        token_ids = resolve_latent_token_ids(processor.tokenizer, qwen_model.config)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "max_sequence_length": max_sequence_length,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
        }
        generation_config = build_continuous_generation_config(
            model_config=qwen_model.config,
            model_generation_config=getattr(qwen_model, "generation_config", None),
            tokenizer=processor.tokenizer,
            generation_kwargs=generation_kwargs,
        )
        metadata["runtime"] = runtime
        metadata["generation"] = {
            **generation_kwargs,
            "eos_token_ids": list(generation_config.eos_token_ids),
            "latent_size": int(qwen_model.config.latent_size),
            "image_token_id": image_token_id,
            "latent_token_ids": {
                "pad": token_ids.pad_token_id,
                "start": token_ids.start_token_id,
                "end": token_ids.end_token_id,
            },
        }
        opener = image_loader or official.open_images
        generate = generation_runner or _default_generation_runner
        effective_device = getattr(model, "device", getattr(qwen_model, "device", None))
        if effective_device is None:
            raise ValueError("unable to determine the evaluation device")
        observed_input_dtypes: dict[str, set[str]] = {}

        for index, original_sample in selected:
            if original_sample is None:
                missing = _error_row(index, None, f"source index {index} is missing")
                missing["condition"] = visual_ablation_mode
                missing["checkpoint"] = str(model_dir)
                rows.append(missing)
                continue
            sample_started = time.perf_counter()
            sample = dict(original_sample)
            donor_index: int | None = None
            gold_source = "recipient"
            gold_value = original_sample.get("original_final_answer")
            try:
                if visual_ablation_mode == "image_replacement":
                    sample, donor_index, gold_value = _replacement_sample(
                        index, original_sample, source, replacements
                    )
                    gold_source = "replacement_donor"
                raw_image_input = sample.get("image_input", [])
                if visual_ablation_mode == "no_input_image":
                    effective_image_input: object = []
                    resolved_paths: list[str] = []
                    images = None
                else:
                    effective_image_input = raw_image_input
                    resolved_paths = resolve_evaluation_image_paths(
                        effective_image_input, source_path.parent
                    )
                    images = opener(resolved_paths) if resolved_paths else None
                prepared = prepare_evaluation_input(processor, sample, images)
                for name, tensor in prepared.tensors.items():
                    observed_input_dtypes.setdefault(name, set()).add(str(tensor.dtype))
                inputs = {name: tensor.to(effective_device) for name, tensor in prepared.tensors.items()}
                if "attention_mask" not in inputs:
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
                sliding_window = (
                    getattr(qwen_model.config, "sliding_window", None)
                    if getattr(qwen_model.config, "use_sliding_window", False)
                    else None
                )
                intervention = None
                if attention_backend == "sdpa" or visual_ablation_mode != "baseline":
                    intervention = VisualIntervention(
                        visual_ablation_mode, image_token_id, sliding_window=sliding_window
                    )
                generation_context = {
                    "model": model,
                    "qwen_model": qwen_model,
                    "processor": processor,
                    "sample": sample,
                    "prepared": prepared,
                    "inputs": inputs,
                    "token_ids": token_ids,
                    "generation_config": generation_config,
                    "intervention": intervention,
                    "allow_latent_generation": True,
                }
                _sync_for_device(torch, effective_device)
                inference_started = time.perf_counter()
                result = generate(**generation_context)
                _sync_for_device(torch, effective_device)
                inference_time = time.perf_counter() - inference_started

                prompt_length = prepared.prompt_token_length
                suffix_ids = [int(token) for token in result.sequences[0, prompt_length:].tolist()]
                start_suffix = [position for position, token in enumerate(suffix_ids) if token == token_ids.start_token_id]
                end_suffix = [position for position, token in enumerate(suffix_ids) if token == token_ids.end_token_id]
                raw_output = processor.batch_decode(result.sequences, skip_special_tokens=False)[0].strip()
                assistant_text = processor.batch_decode(
                    result.sequences[:, prompt_length:], skip_special_tokens=False
                )[0].strip()
                assistant_only = official.extract_assistant_content(assistant_text)
                answer = score_answer(assistant_only, gold_value)
                category, category_source = _category(original_sample)
                completed = int(result.latent_block_count)
                started_blocks = int(result.latent_start_count)
                row = {
                    "index": index,
                    "sample_id": original_sample.get("id", index),
                    "checkpoint": str(model_dir),
                    "condition": visual_ablation_mode,
                    "category": category,
                    "category_source": category_source,
                    "text_input": original_sample.get("text_input", ""),
                    "text_identity": _value_identity(original_sample.get("text_input", "")),
                    "recipient_image_input": original_sample.get("image_input", []),
                    "effective_image_input": effective_image_input,
                    "image_identity": _value_identity(effective_image_input),
                    "replacement_donor_index": donor_index,
                    "gold_source": gold_source,
                    "prompt_identity": prepared.prompt_identity,
                    "prompt_token_length": prompt_length,
                    "generated_suffix_token_ids": suffix_ids,
                    "raw_output": raw_output,
                    "assistant_only": assistant_only,
                    "prediction": assistant_only,
                    "extracted_final_answer": answer.prediction.extracted,
                    "prediction_parse_status": answer.prediction.status,
                    "prediction_parse_detail": answer.prediction.detail,
                    "prediction_normalized": answer.prediction.normalized,
                    "gold_final_answer": gold_value,
                    "gold_parse_status": answer.gold.status,
                    "gold_parse_detail": answer.gold.detail,
                    "gold_normalized": answer.gold.normalized,
                    "invalid_gold_reason": answer.invalid_gold_reason,
                    "match": bool(answer.match),
                    "latent_start_positions": {
                        "suffix": start_suffix,
                        "absolute": [prompt_length + position for position in start_suffix],
                    },
                    "latent_end_positions": {
                        "suffix": end_suffix,
                        "absolute": [prompt_length + position for position in end_suffix],
                    },
                    "latent_start_count": started_blocks,
                    "latent_block_count": completed,
                    "truncated_latent_block_count": max(0, started_blocks - completed),
                    "has_truncated_latent_block": started_blocks > completed,
                    "logits_events": list(getattr(intervention, "events", ())),
                    "intervention_forward_records": list(
                        getattr(intervention, "forward_records", ())
                    ),
                    "inference_time_sec": inference_time,
                    "wall_time_sec": time.perf_counter() - sample_started,
                    "stop_reason": result.stop_reason,
                    "generated_token_count": len(suffix_ids),
                    "total_token_count": prompt_length + len(suffix_ids),
                    "error": None,
                }
                rows.append(_json_safe(row))
            except Exception as error:
                error_sample = dict(original_sample)
                error_sample["original_final_answer"] = gold_value
                failed = _error_row(index, error_sample, error)
                failed["condition"] = visual_ablation_mode
                failed["checkpoint"] = str(model_dir)
                failed["replacement_donor_index"] = donor_index
                failed["gold_source"] = gold_source
                failed["wall_time_sec"] = time.perf_counter() - sample_started
                rows.append(failed)

        runtime["input_tensor_dtypes"] = {
            name: sorted(dtypes) for name, dtypes in sorted(observed_input_dtypes.items())
        }
        metadata["selection"] = {
            "requested_indices": [index for index, _ in selected],
            "requested_count": len(selected),
        }
        metadata["completed_count"] = sum(row.get("error") in (None, "") for row in rows)
        metadata["error_count"] = sum(row.get("error") not in (None, "") for row in rows)
        metadata["status"] = "failed" if metadata["error_count"] else "completed"
        metadata["aggregate"] = aggregate_rows(rows)
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        metadata.setdefault("completed_count", 0)
        metadata.setdefault("error_count", len(rows))
    metadata["wall_time_sec"] = time.perf_counter() - started
    metadata = _json_safe(metadata)
    rows = _json_safe(rows)
    _write_jsonl(output / "predictions.jsonl", rows)
    _write_json(output / "metadata.json", metadata)
    return {"rows": rows, "metadata": metadata}


def _indexed(rows: Iterable[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("each comparison row index must be an integer")
        if index in result:
            raise ValueError(f"duplicate index: {index}")
        result[index] = row
    return result


def baseline_equivalence(
    baseline_rows: Iterable[Mapping[str, Any]],
    analysis_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare raw IDs, prompt identity, and scorer outputs on identical indices."""
    baseline = _indexed(baseline_rows)
    analysis = _indexed(analysis_rows)
    if set(baseline) != set(analysis):
        raise ValueError("baseline equivalence requires the same index set")
    mismatches = {"raw_token_ids": [], "prompt_identity": [], "scoring": []}
    unavailable: list[dict[str, Any]] = []
    scoring_fields = (
        "prediction_normalized",
        "prediction_parse_status",
        "gold_normalized",
        "gold_parse_status",
        "invalid_gold_reason",
        "match",
    )
    for index in sorted(baseline):
        before, after = baseline[index], analysis[index]
        before_reasons = _equivalence_unavailable_reasons(before)
        after_reasons = _equivalence_unavailable_reasons(after)
        if before_reasons or after_reasons:
            unavailable.append(
                {
                    "index": index,
                    "baseline": before_reasons,
                    "analysis": after_reasons,
                }
            )
            continue
        if before.get("generated_suffix_token_ids") != after.get("generated_suffix_token_ids"):
            mismatches["raw_token_ids"].append(index)
        if before.get("prompt_identity") != after.get("prompt_identity"):
            mismatches["prompt_identity"].append(index)
        if any(before.get(field) != after.get(field) for field in scoring_fields):
            mismatches["scoring"].append(index)
    count = len(baseline)
    comparable_count = count - len(unavailable)
    return {
        "paired_count": count,
        "comparable_count": comparable_count,
        "unavailable_count": len(unavailable),
        "exact": not unavailable and not any(mismatches.values()),
        "raw_token_id_match_count": comparable_count - len(mismatches["raw_token_ids"]),
        "prompt_identity_match_count": comparable_count - len(mismatches["prompt_identity"]),
        "scoring_match_count": comparable_count - len(mismatches["scoring"]),
        "mismatched_indices": mismatches,
        "unavailable_evidence": unavailable,
    }


def _equivalence_unavailable_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    inference_error = (
        row.get("error") not in (None, "")
        or row.get("prediction_parse_status") == "inference_error"
    )
    if inference_error:
        reasons.append("inference_error")

    raw_ids = row.get("generated_suffix_token_ids")
    if not isinstance(raw_ids, list) or any(
        not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
        for token_id in raw_ids
    ):
        reasons.append("missing_or_invalid_raw_token_ids")
    prompt_identity = row.get("prompt_identity")
    if not isinstance(prompt_identity, str) or not prompt_identity:
        reasons.append("missing_prompt_identity")

    if not inference_error:
        valid_scoring = (
            isinstance(row.get("prediction_normalized"), str)
            and isinstance(row.get("prediction_parse_status"), str)
            and bool(row["prediction_parse_status"])
            and isinstance(row.get("gold_normalized"), str)
            and isinstance(row.get("gold_parse_status"), str)
            and bool(row["gold_parse_status"])
            and (
                row.get("invalid_gold_reason") is None
                or isinstance(row.get("invalid_gold_reason"), str)
            )
        ) and isinstance(row.get("match"), bool)
        if not valid_scoring:
            reasons.append("missing_or_invalid_scoring_evidence")
    return reasons


def _flatten(prefix: str, value: Any, rows: list[tuple[str, Any]]) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            _flatten(f"{prefix}.{key}" if prefix else str(key), value[key], rows)
    elif isinstance(value, list):
        rows.append((prefix, json.dumps(value, separators=(",", ":"))))
    else:
        rows.append((prefix, value))


def compare_prediction_files(
    baseline_path: str | os.PathLike[str],
    condition_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    baseline = _read_jsonl(baseline_path)
    condition = _read_jsonl(condition_path)
    summary = {
        "baseline": aggregate_rows(baseline),
        "condition": aggregate_rows(condition),
        "paired": paired_summary(baseline, condition),
        "equivalence": baseline_equivalence(baseline, condition),
        "inputs": {
            "baseline_path": str(Path(baseline_path).resolve(strict=True)),
            "baseline_sha256": _sha256(baseline_path),
            "condition_path": str(Path(condition_path).resolve(strict=True)),
            "condition_sha256": _sha256(condition_path),
        },
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "summary.json", summary)
    flat: list[tuple[str, Any]] = []
    _flatten("", summary, flat)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        writer.writerows(flat)
    paired = summary["paired"]
    equivalence = summary["equivalence"]
    markdown = (
        "# Visual causal comparison\n\n"
        f"Paired rows: {paired['paired_count']}  \n"
        f"Accuracy change: {paired['accuracy_delta_pp']} percentage points  \n"
        f"Comparable equivalence rows: {equivalence['comparable_count']}  \n"
        f"Unavailable equivalence evidence: {equivalence['unavailable_count']}  \n"
        f"Raw-token exact rows: {equivalence['raw_token_id_match_count']}/{equivalence['comparable_count']}  \n"
        f"Prompt-identity exact rows: {equivalence['prompt_identity_match_count']}/{equivalence['comparable_count']}  \n"
        f"Scoring exact rows: {equivalence['scoring_match_count']}/{equivalence['comparable_count']}\n\n"
        "The paired inferential statistics exclude invalid gold labels and inference errors. "
        "All comparisons are exploratory and use rows with identical original indices.\n"
    )
    (output / "analysis.md").write_text(markdown, encoding="utf-8")
    return _json_safe(summary)


def create_group_manifest(
    source_path: str | os.PathLike[str], output_path: str | os.PathLike[str]
) -> dict[str, Any]:
    output = Path(output_path)
    data_root = Path("data").resolve()
    resolved_output = output.resolve()
    if not resolved_output.is_relative_to(data_root):
        raise ValueError("group output path must be under data/")
    rows = _read_jsonl(source_path)
    groups = matched_image_groups(rows)
    manifest = {
        "schema_version": 1,
        "source_data_path": str(Path(source_path).resolve(strict=True)),
        "source_data_sha256": _sha256(source_path),
        "groups": groups,
        "indices": sorted({index for group in groups for index in group}),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, manifest)
    return manifest


def _parse_indices(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        indices = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from error
    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("indices must be unique")
    return indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="run one visual intervention condition")
    run.add_argument("--model-dir", required=True)
    run.add_argument("--base-model-dir")
    run.add_argument("--test-data-path", required=True)
    run.add_argument("--source-data-path", required=True)
    run.add_argument("--indices", type=_parse_indices)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--visual-ablation-mode", choices=VISUAL_ABLATION_MODES, required=True)
    run.add_argument("--attention-backend", choices=("native", "sdpa"), required=True)
    run.add_argument("--replacement-map")
    run.add_argument("--max-new-tokens", type=int, default=4096)
    run.add_argument("--max-sequence-length", type=int, default=32768)
    run.add_argument("--repetition-penalty", type=float, default=1.05)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--limit", type=int)

    compare = subcommands.add_parser("compare", help="compare paired prediction JSONL files")
    compare.add_argument("--baseline-jsonl", required=True)
    compare.add_argument("--condition-jsonl", required=True)
    compare.add_argument("--output-dir", required=True)

    groups = subcommands.add_parser("groups", help="create matched-image groups")
    groups.add_argument("--source-data-path", required=True)
    groups.add_argument("--output-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        result = run_analysis(
            model_dir=args.model_dir,
            base_model_dir=args.base_model_dir,
            test_data_path=args.test_data_path,
            source_data_path=args.source_data_path,
            indices=args.indices,
            output_dir=args.output_dir,
            visual_ablation_mode=args.visual_ablation_mode,
            attention_backend=args.attention_backend,
            replacement_map=args.replacement_map,
            max_new_tokens=args.max_new_tokens,
            max_sequence_length=args.max_sequence_length,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            top_p=args.top_p,
            limit=args.limit,
        )
        return 0 if result["metadata"]["status"] == "completed" else 1
    if args.command == "compare":
        compare_prediction_files(args.baseline_jsonl, args.condition_jsonl, args.output_dir)
        return 0
    create_group_manifest(args.source_data_path, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
