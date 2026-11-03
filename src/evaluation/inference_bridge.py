"""Lightweight validation and adapter boundary for VSP generation."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib
import math
from pathlib import Path
import re

from evaluation.schema import ContractError, EvalConfig, ModelInput, PreparedEvaluation
from evaluation.vsp_prepare import COMMON_IMAGES, LEVELS, PROCESSED_PROMPT, prompt_identity


_GENERATION_KEYS = {
    "max_new_tokens",
    "max_sequence_length",
    "repetition_penalty",
    "temperature",
    "top_p",
    "do_sample",
}
_RESULT_KEYS = {
    "assistant_only",
    "generated_suffix_token_ids",
    "prompt_identity",
    "prompt_token_length",
    "raw_output",
    "extracted_final_answer",
    "latent_start_count",
    "latent_block_count",
    "stop_reason",
    "inference_time",
}
_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "chat_template.jinja",
    "adapter_config.json",
    "stage1_peft_metadata.json",
    "stage2_peft_metadata.json",
)
_ADAPTER_MARKERS = (
    "adapter_config.json",
    "stage1_peft_metadata.json",
    "stage2_peft_metadata.json",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _exact_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractError(f"{name} must be a nonnegative integer")
    return value


def _finite_number(value: object, name: str) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ContractError(f"{name} must be a finite number")
    return value


def validate_generation_config(raw: dict) -> None:
    if not isinstance(raw, dict) or set(raw) != _GENERATION_KEYS:
        raise ContractError("unexpected or missing generation config fields")
    _exact_nonnegative_int(raw["max_new_tokens"], "max_new_tokens")
    if type(raw["max_sequence_length"]) is not int or raw["max_sequence_length"] <= 0:
        raise ContractError("max_sequence_length must be a positive integer")
    if _finite_number(raw["repetition_penalty"], "repetition_penalty") <= 0:
        raise ContractError("repetition_penalty must be positive")
    if _finite_number(raw["temperature"], "temperature") < 0:
        raise ContractError("temperature must be nonnegative")
    top_p = _finite_number(raw["top_p"], "top_p")
    if not 0 < top_p <= 1:
        raise ContractError("top_p must be in (0, 1]")
    if type(raw["do_sample"]) is not bool:
        raise ContractError("do_sample must be boolean")
    if raw["do_sample"] != (raw["temperature"] > 0):
        raise ContractError("do_sample must equal temperature > 0")


def _inside_existing_file(root: Path, relative: str | Path, kind: str) -> Path:
    try:
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise ValueError
        candidate = (root / relative_path).resolve()
        candidate.relative_to(root)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{kind} must resolve inside dataset_root") from exc
    if not candidate.is_file():
        raise ContractError(f"{kind} must be a readable regular file")
    return candidate


def validate_vsp_inputs(prepared: PreparedEvaluation, config: EvalConfig) -> None:
    if config.benchmark != "vsp" or config.profile != "vsp_validated_v1":
        raise ContractError("VSP generation requires the validated VSP profile")
    if config.categories != LEVELS:
        raise ContractError("VSP categories must be level3 through level8 in order")
    if config.max_steps != 100:
        raise ContractError("VSP max_steps must equal 100")
    try:
        dataset_root = Path(config.dataset_root).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError("dataset_root must be an existing directory") from exc
    if not dataset_root.is_dir():
        raise ContractError("dataset_root must be an existing directory")

    prompt_path = _inside_existing_file(dataset_root, PROCESSED_PROMPT, "processed prompt")
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ContractError("processed prompt must be readable UTF-8") from exc
    if config.run_metadata.prompt_template_identity != prompt_identity(prompt):
        raise ContractError("processed prompt identity does not match config")

    for sample in prepared.samples:
        if sample.benchmark != "vsp" or sample.subtask not in LEVELS:
            raise ContractError("sample category is not a VSP level")
        if sample.id != f"vsp/{sample.subtask}/{sample.provenance.source_id}":
            raise ContractError("sample ID does not match source identity")
        if sample.provenance.config != sample.subtask:
            raise ContractError("sample category does not match provenance config")
        if sample.target.max_steps != 100:
            raise ContractError("sample max_steps must equal 100")
        if sample.model_input.choices:
            raise ContractError("VSP model input choices must be empty")
        if sample.model_input.text != prompt:
            raise ContractError("sample text does not match processed prompt")
        expected_images = COMMON_IMAGES + (
            f"maps/{sample.subtask}/img/{sample.provenance.source_id}.png",
        )
        if sample.model_input.images != expected_images:
            raise ContractError("sample image order does not match source identity")
        for relative in sample.model_input.images:
            image_path = _inside_existing_file(dataset_root, relative, "image")
            try:
                from PIL import Image

                with Image.open(image_path) as image:
                    if image.format != "PNG":
                        raise ContractError("image must be a readable PNG")
                    image.verify()
                with Image.open(image_path) as image:
                    image.load()
            except ContractError:
                raise
            except Exception as exc:
                raise ContractError("image must be a readable PNG") from exc


def _resolve_storage_path(
    value: Path, repo_root: Path, storage: str, name: str, *, must_exist: bool
) -> Path:
    try:
        repository = Path(repo_root).resolve(strict=True)
        storage_root = (repository / storage).resolve(strict=True)
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = repository / candidate
        candidate = candidate.resolve(strict=must_exist)
        candidate.relative_to(storage_root)
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(f"{name} must resolve inside repo_root/{storage}") from exc
    if must_exist and not candidate.is_dir():
        raise ContractError(f"{name} must be an existing directory")
    return candidate


def validate_runtime_paths(
    model_dir: Path,
    base_model_dir: Path | None,
    cache_dir: Path,
    repo_root: Path,
) -> tuple[Path, Path | None, Path]:
    model = _resolve_storage_path(
        model_dir, repo_root, "checkpoints", "model_dir", must_exist=True
    )
    base = (
        _resolve_storage_path(
            base_model_dir,
            repo_root,
            "checkpoints",
            "base_model_dir",
            must_exist=True,
        )
        if base_model_dir is not None
        else None
    )
    cache = _resolve_storage_path(
        cache_dir, repo_root, "data", "cache_dir", must_exist=False
    )
    is_adapter = any((model / marker).is_file() for marker in _ADAPTER_MARKERS)
    if is_adapter and base is None:
        raise ContractError("base_model_dir is required for a PEFT artifact")
    if not is_adapter and base is not None:
        raise ContractError("base_model_dir is invalid for a full artifact")
    return model, base, cache


def load_runtime():
    return importlib.import_module("eval")


def _validate_legacy_result(result: object) -> dict:
    if not isinstance(result, dict) or set(result) != _RESULT_KEYS:
        raise ContractError("unexpected or missing legacy result fields")
    for field in ("assistant_only", "raw_output", "extracted_final_answer"):
        if not isinstance(result[field], str):
            raise ContractError(f"{field} must be a string")
    suffix = result["generated_suffix_token_ids"]
    if not isinstance(suffix, list) or any(
        type(token_id) is not int or token_id < 0 for token_id in suffix
    ):
        raise ContractError("generated suffix token IDs must be nonnegative integers")
    if not isinstance(result["prompt_identity"], str) or not _SHA256.fullmatch(
        result["prompt_identity"]
    ):
        raise ContractError("prompt_identity must be a SHA-256 string")
    _exact_nonnegative_int(result["prompt_token_length"], "prompt_token_length")
    _exact_nonnegative_int(result["latent_start_count"], "latent_start_count")
    _exact_nonnegative_int(result["latent_block_count"], "latent_block_count")
    if not isinstance(result["stop_reason"], str) or not result["stop_reason"]:
        raise ContractError("stop_reason must be a nonempty string")
    if _finite_number(result["inference_time"], "inference_time") < 0:
        raise ContractError("inference_time must be nonnegative")
    return result


def generate_one(
    model,
    processor,
    model_input: ModelInput,
    dataset_root: Path,
    generation_config: dict,
    *,
    run_one_example,
) -> dict:
    if not isinstance(model_input, ModelInput):
        raise ContractError("model_input must use the ModelInput contract")
    if model_input.choices:
        raise ContractError("model_input choices must be empty")
    validate_generation_config(generation_config)
    try:
        root = Path(dataset_root).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError("dataset_root must be an existing directory") from exc
    if not root.is_dir():
        raise ContractError("dataset_root must be an existing directory")
    paths = [
        str(_inside_existing_file(root, relative, "image"))
        for relative in model_input.images
    ]
    payload = {"text_input": model_input.text, "image_input": paths}
    result = run_one_example(model, processor, payload, generation_config)
    return _validate_legacy_result(result)


def _json_compatible(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(child) for child in value]
    return str(value)


def _to_dict(value: object) -> dict[str, object]:
    method = getattr(value, "to_dict", None)
    if not callable(method):
        return {}
    result = method()
    if not isinstance(result, Mapping):
        raise ContractError("runtime to_dict() must return a mapping")
    return _json_compatible(result)


def _metadata_hashes(model_dir: Path, base_model_dir: Path | None) -> dict[str, str]:
    result = {}
    sources = (("", model_dir), ("base/", base_model_dir))
    for prefix, directory in sources:
        if directory is None:
            continue
        for name in _METADATA_FILES:
            path = directory / name
            if path.is_file():
                try:
                    result[prefix + name] = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError as exc:
                    raise ContractError(
                        f"cannot read runtime metadata file: {prefix + name}"
                    ) from exc
    return result


def runtime_metadata(
    model,
    processor,
    runtime,
    model_dir: Path,
    base_model_dir: Path | None,
    generation_config: dict,
) -> dict:
    try:
        resolved_model_dir = Path(model_dir).resolve(strict=True)
        resolved_base_dir = (
            Path(base_model_dir).resolve(strict=True)
            if base_model_dir is not None
            else None
        )
        qwen_model = runtime.resolve_qwen2_5_vl_model(model)
        qwen_config = qwen_model.config
        token_ids = runtime.resolve_latent_token_ids(processor.tokenizer, qwen_config)
        transformers_module = getattr(runtime, "transformers", None)
        if transformers_module is None:
            transformers_module = importlib.import_module("transformers")
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError("cannot inspect loaded runtime metadata") from exc

    processor_config = _to_dict(processor)
    processor_init_kwargs = getattr(processor, "init_kwargs", None)
    return {
        "model_dir": str(resolved_model_dir),
        "base_model_dir": str(resolved_base_dir) if resolved_base_dir is not None else None,
        "metadata_sha256": _metadata_hashes(resolved_model_dir, resolved_base_dir),
        "qwen_config": {
            "stage": _json_compatible(getattr(qwen_config, "stage", None)),
            "latent_size": _json_compatible(getattr(qwen_config, "latent_size", None)),
        },
        "model_config": _to_dict(qwen_config),
        "model_generation_config": _to_dict(
            getattr(qwen_model, "generation_config", None)
        ),
        "latent_token_ids": {
            "pad": int(token_ids.pad_token_id),
            "start": int(token_ids.start_token_id),
            "end": int(token_ids.end_token_id),
        },
        "processor": {
            "class": f"{type(processor).__module__}.{type(processor).__qualname__}",
            "config": processor_config,
            "init_kwargs": _json_compatible(processor_init_kwargs)
            if isinstance(processor_init_kwargs, Mapping)
            else {},
            "chat_template": _json_compatible(getattr(processor, "chat_template", None)),
            "tokenizer_chat_template": _json_compatible(
                getattr(processor.tokenizer, "chat_template", None)
            ),
        },
        "generation_kwargs": generation_config,
        "transformers": {
            "file": str(getattr(transformers_module, "__file__", "")),
            "version": str(getattr(transformers_module, "__version__", "")),
        },
    }
