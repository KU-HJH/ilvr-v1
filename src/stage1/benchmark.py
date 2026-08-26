"""Small, sealed evidence contracts for Stage 1 probes and timed runs.

This module deliberately audits JSON/text/reference metadata only.  It never
opens an image, model weight, or checkpoint shard and never recursively hashes
an artifact tree.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import stat as stat_module
import sys
import tempfile
import time
from typing import Any, Literal, Mapping, Sequence


_SCHEMA_VERSION = 1
_HEX_DIGEST_LENGTH = 64


def _is_exact_int(value: object) -> bool:
    return type(value) is int


def _require_int(name: str, value: object, *, positive: bool = False) -> int:
    if not _is_exact_int(value):
        raise TypeError(f"{name} must be an integer (booleans are not numeric)")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _require_float(name: str, value: object) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be a float (booleans are not numeric)")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _require_string(name: str, value: object, *, nonempty: bool = True) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if nonempty and not value:
        raise ValueError(f"{name} must not be empty")
    if "\0" in value:
        raise ValueError(f"{name} must not contain an embedded NUL")
    return value


def _require_digest(name: str, value: object) -> str:
    value = _require_string(name, value)
    if len(value) != _HEX_DIGEST_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], kind: str) -> None:
    if type(payload) is not dict:
        raise TypeError(f"{kind} must be a JSON object")
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"invalid {kind} fields: missing={missing}, unexpected={unexpected}"
        )


def _canonical_json_bytes(payload: object) -> bytes:
    def validate(value: object, location: str) -> None:
        if type(value) is str:
            _require_string(location, value, nonempty=False)
            return
        if value is None or type(value) in (bool, int):
            return
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError(f"{location} contains a non-finite float")
            return
        if type(value) is list:
            for index, item in enumerate(value):
                validate(item, f"{location}[{index}]")
            return
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(f"{location} contains a non-string key")
                _require_string(f"{location} key", key, nonempty=False)
                validate(item, f"{location}.{key}")
            return
        raise TypeError(f"{location} is not canonical JSON data: {type(value).__name__}")

    validate(payload, "config")
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def config_digest(config: Mapping[str, object]) -> str:
    """Digest one small canonical config/launch manifest, never its path contents."""

    if type(config) is not dict:
        raise TypeError("config must be a plain mapping")
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


def capacity_candidates(start: int, max_per_gpu: int) -> tuple[int, ...]:
    """Return the approved ascending powers-of-two capacity search."""

    start = _require_int("start", start, positive=True)
    max_per_gpu = _require_int("max_per_gpu", max_per_gpu, positive=True)
    if start != 8:
        raise ValueError("primary capacity probing must start at batch 8")
    if max_per_gpu < start:
        raise ValueError("max_per_gpu must be at least the initial batch 8")
    candidates: list[int] = []
    candidate = start
    while candidate <= max_per_gpu:
        candidates.append(candidate)
        candidate *= 2
    return tuple(candidates)


def fallback_candidates_after_initial_failure(start: int) -> tuple[int, ...]:
    """Describe the explicit fallback order allowed after batch 8 is a confirmed OOM."""

    start = _require_int("start", start, positive=True)
    if start != 8:
        raise ValueError("fallback is allowed only after the initial batch 8 failure")
    return (4, 2, 1)


@dataclass(frozen=True)
class Stage1BenchmarkEvidence:
    status: Literal["success", "oom", "failure"]
    method: Literal["full", "dora"]
    micro_batch: int
    world_size: int
    gradient_accumulation_steps: int
    effective_global_batch: int
    optimizer_steps: int
    samples_seen: int
    model_load_seconds: float
    trainer_seconds: float
    total_seconds: float
    final_save_seconds: float
    samples_per_second: float
    steps_per_second: float
    rank_peak_allocated_bytes: tuple[int, ...]
    rank_peak_reserved_bytes: tuple[int, ...]
    trainable_parameters: int
    total_parameters: int
    checkpoint_bytes: int
    final_loss: float
    representative_gradient_norm: float
    git_commit: str
    dependency_versions: dict[str, str]
    gpu_names: tuple[str, ...]
    config_digest: str
    probe_row_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in {"success", "oom", "failure"}:
            raise ValueError("status must be success, oom, or failure")
        if type(self.method) is not str or self.method not in {"full", "dora"}:
            raise ValueError("method must be full or dora")
        for name in (
            "micro_batch",
            "world_size",
            "gradient_accumulation_steps",
        ):
            _require_int(name, getattr(self, name), positive=True)
        for name in (
            "effective_global_batch",
            "optimizer_steps",
            "samples_seen",
            "trainable_parameters",
            "total_parameters",
            "checkpoint_bytes",
        ):
            _require_int(name, getattr(self, name))
        expected_batch = (
            self.world_size * self.micro_batch * self.gradient_accumulation_steps
        )
        if self.effective_global_batch != expected_batch:
            raise ValueError(
                "effective_global_batch must equal "
                "world_size * micro_batch * gradient_accumulation_steps"
            )
        if self.trainable_parameters > self.total_parameters:
            raise ValueError("trainable_parameters must not exceed total_parameters")
        for name in (
            "model_load_seconds",
            "trainer_seconds",
            "total_seconds",
            "final_save_seconds",
            "samples_per_second",
            "steps_per_second",
            "final_loss",
            "representative_gradient_norm",
        ):
            _require_float(name, getattr(self, name))
        self._validate_rank_int_tuple(
            "rank_peak_allocated_bytes", self.rank_peak_allocated_bytes
        )
        self._validate_rank_int_tuple(
            "rank_peak_reserved_bytes", self.rank_peak_reserved_bytes
        )
        if any(
            allocated > reserved
            for allocated, reserved in zip(
                self.rank_peak_allocated_bytes, self.rank_peak_reserved_bytes
            )
        ):
            raise ValueError("rank peak allocated bytes must not exceed reserved bytes")
        if self.total_seconds < (
            self.model_load_seconds + self.trainer_seconds + self.final_save_seconds
        ):
            raise ValueError("total_seconds must cover load, trainer, and final save")
        if self.status == "success" and self.optimizer_steps == 0:
            raise ValueError("successful evidence requires optimizer steps")
        if type(self.gpu_names) is not tuple or len(self.gpu_names) != self.world_size:
            raise ValueError("gpu_names tuple length must equal world_size")
        for index, value in enumerate(self.gpu_names):
            _require_string(f"gpu_names[{index}]", value)
        if type(self.probe_row_indices) is not tuple:
            raise TypeError("probe_row_indices must be a tuple")
        if len(set(self.probe_row_indices)) != len(self.probe_row_indices):
            raise ValueError("probe_row_indices must be unique")
        for index, value in enumerate(self.probe_row_indices):
            _require_int(f"probe_row_indices[{index}]", value)
        _require_string("git_commit", self.git_commit)
        _require_digest("config_digest", self.config_digest)
        if type(self.dependency_versions) is not dict:
            raise TypeError("dependency_versions must be a plain dict")
        for name, version in self.dependency_versions.items():
            _require_string("dependency_versions key", name)
            _require_string(f"dependency_versions[{name!r}]", version)

    def _validate_rank_int_tuple(self, name: str, values: object) -> None:
        if type(values) is not tuple or len(values) != self.world_size:
            raise ValueError(f"{name} tuple length must equal world_size")
        for index, value in enumerate(values):
            _require_int(f"{name}[{index}]", value)


_EVIDENCE_FIELDS = {field.name for field in fields(Stage1BenchmarkEvidence)}
_VERIFIED_EVIDENCE_CACHE: dict[
    Path, tuple[tuple[int, int, int, int], Stage1BenchmarkEvidence]
] = {}


def _evidence_to_dict(evidence: Stage1BenchmarkEvidence) -> dict[str, object]:
    if type(evidence) is not Stage1BenchmarkEvidence:
        raise TypeError("evidence must be Stage1BenchmarkEvidence")
    payload = asdict(evidence)
    for name in (
        "rank_peak_allocated_bytes",
        "rank_peak_reserved_bytes",
        "gpu_names",
        "probe_row_indices",
    ):
        payload[name] = list(getattr(evidence, name))
    return payload


def _evidence_from_dict(payload: object) -> Stage1BenchmarkEvidence:
    if type(payload) is not dict:
        raise TypeError("benchmark evidence must be a JSON object")
    _require_exact_keys(payload, _EVIDENCE_FIELDS, "benchmark evidence")
    values = dict(payload)
    for name in (
        "rank_peak_allocated_bytes",
        "rank_peak_reserved_bytes",
        "gpu_names",
        "probe_row_indices",
    ):
        if type(values[name]) is not list:
            raise TypeError(f"{name} must be a JSON array")
        values[name] = tuple(values[name])
    if type(values["dependency_versions"]) is not dict:
        raise TypeError("dependency_versions must be a JSON object")
    return Stage1BenchmarkEvidence(**values)


def _atomic_write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json_once(path: Path) -> object:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed JSON evidence at {path}: {error}") from error


def _file_witness(path: Path) -> tuple[int, int, int, int]:
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def write_benchmark_evidence(path: Path, evidence: Stage1BenchmarkEvidence) -> None:
    """Atomically publish evidence, refusing to replace a different sealed result."""

    path = Path(path)
    if path.exists():
        resolved = path.resolve(strict=True)
        witness = _file_witness(resolved)
        cached = _VERIFIED_EVIDENCE_CACHE.get(resolved)
        if cached is not None and cached[0] == witness:
            existing = cached[1]
        else:
            existing = _evidence_from_dict(_read_json_once(resolved))
            _VERIFIED_EVIDENCE_CACHE[resolved] = (witness, existing)
        if existing == evidence:
            return
        raise FileExistsError(f"refusing to replace existing benchmark evidence: {path}")
    _atomic_write_json(path, _evidence_to_dict(evidence))
    _VERIFIED_EVIDENCE_CACHE[path.resolve()] = (_file_witness(path), evidence)


def load_verified_evidence(
    path: Path, expected_config_digest: str
) -> Stage1BenchmarkEvidence | None:
    """Load and strictly validate a small evidence file once per unchanged stat witness."""

    path = Path(path)
    _require_digest("expected_config_digest", expected_config_digest)
    if not path.is_file():
        return None
    resolved = path.resolve(strict=True)
    witness = _file_witness(resolved)
    cached = _VERIFIED_EVIDENCE_CACHE.get(resolved)
    if cached is not None and cached[0] == witness:
        evidence = cached[1]
    else:
        payload = _read_json_once(resolved)
        if type(payload) is dict and payload.get("kind") == "stage1-producer-evidence":
            evidence, _ = _validate_producer_evidence_payload(payload)
        else:
            # Compatibility for Task 13's original public pure-metrics writer.
            # Official CLI/runtime evidence always uses the producer envelope.
            evidence = _evidence_from_dict(payload)
        _VERIFIED_EVIDENCE_CACHE[resolved] = (witness, evidence)
    if evidence.config_digest != expected_config_digest:
        return None
    return evidence


_FULL_WEIGHT_NAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}
_DORA_REQUIRED_NAMES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "ilvr_token_rows.safetensors",
    "stage1_peft_metadata.json",
}
_PROCESSOR_REQUIRED_NAMES = {
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
}
_TOKENIZER_OPTIONAL_NAMES = {
    "special_tokens_map.json", "added_tokens.json", "chat_template.json",
    "merges.txt", "vocab.json", "generation_config.json",
}
_RUNTIME_OUTPUT_NAMES = {
    "train.log", "launch-manifest.json", "producer-evidence.json", "evidence.json",
}


def _is_allowlisted_artifact_name(name: str, method: str) -> bool:
    common = _PROCESSOR_REQUIRED_NAMES | _TOKENIZER_OPTIONAL_NAMES
    if name in common or name.startswith("chat_template."):
        return True
    if method == "full":
        return name == "config.json" or _is_full_weight_name(name)
    return name in _DORA_REQUIRED_NAMES


def _is_full_weight_name(name: str) -> bool:
    return (
        name in _FULL_WEIGHT_NAMES
        or (name.startswith("model-") and name.endswith(".safetensors"))
        or (name.startswith("pytorch_model-") and name.endswith(".bin"))
    )


def _build_artifact_manifest(
    output_dir: Path, *, method: str, final_save: bool
) -> dict[str, object]:
    """Capture a small topology/stat witness; never read or hash weight contents."""

    method = _validated_method(method)
    output_dir = output_dir.resolve()
    if not final_save:
        entries: list[dict[str, object]] = []
        artifact_kind = "none"
        weight_serialization = None
    else:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"final Stage 1 artifact directory is missing: {output_dir}")
        entries = []
        for item in sorted(output_dir.iterdir(), key=lambda path: path.name):
            if method == "dora" and _is_full_weight_name(item.name):
                raise ValueError("DoRA final artifact must not contain full-model weights")
            if method == "full" and item.name in _DORA_REQUIRED_NAMES:
                raise ValueError("full final artifact unexpectedly contains DoRA weights")
            if item.name in _RUNTIME_OUTPUT_NAMES or not _is_allowlisted_artifact_name(item.name, method):
                continue
            if item.is_symlink() or not item.is_file():
                raise ValueError(f"artifact sidecar must be a regular non-symlink: {item}")
            item_stat = item.stat()
            if item_stat.st_size <= 0:
                raise ValueError(f"artifact file must have positive size: {item}")
            entries.append({"name": item.name, "size": item_stat.st_size})
        names = {item["name"] for item in entries}
        full_names = {name for name in names if _is_full_weight_name(name)}
        missing_processor = _PROCESSOR_REQUIRED_NAMES - names
        if missing_processor:
            raise ValueError(
                "final artifact is missing processor/tokenizer files: "
                f"{sorted(missing_processor)}"
            )
        if method == "full":
            if not full_names:
                raise ValueError("full final artifact has no full-model weight topology")
            if "config.json" not in names:
                raise ValueError("full final artifact is missing config.json")
            if names & _DORA_REQUIRED_NAMES:
                raise ValueError("full final artifact unexpectedly contains DoRA weights")
            standalone = names & {"model.safetensors", "pytorch_model.bin"}
            indexes = names & {
                "model.safetensors.index.json", "pytorch_model.bin.index.json"
            }
            if len(standalone) + len(indexes) != 1:
                raise ValueError("full artifact must have exactly one weight serialization")
            if indexes:
                index_name = next(iter(indexes))
                index_path = output_dir / index_name
                if index_path.stat().st_size > 4 * 1024 * 1024:
                    raise ValueError("full shard index is unexpectedly large")
                try:
                    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise ValueError("malformed full shard index") from error
                if type(index_payload) is not dict or type(index_payload.get("weight_map")) is not dict or not index_payload["weight_map"]:
                    raise ValueError("full shard index requires a nonempty weight_map")
                references = set(index_payload["weight_map"].values())
                expected_suffix = ".safetensors" if index_name.startswith("model") else ".bin"
                expected_prefix = "model-" if index_name.startswith("model") else "pytorch_model-"
                for reference in references:
                    if (
                        type(reference) is not str or "\0" in reference
                        or Path(reference).name != reference
                        or not reference.startswith(expected_prefix)
                        or not reference.endswith(expected_suffix)
                    ):
                        raise ValueError(f"invalid full shard reference: {reference!r}")
                    shard = output_dir / reference
                    if shard.is_symlink() or not shard.is_file() or shard.stat().st_size <= 0:
                        raise ValueError(f"referenced full shard is missing or invalid: {reference}")
                actual_shards = {
                    name for name in names
                    if name.startswith(expected_prefix) and name.endswith(expected_suffix)
                }
                if actual_shards != references:
                    raise ValueError("full shard set does not exactly match its index")
                weight_serialization = {
                    "kind": "indexed",
                    "index": index_name,
                    "shards": sorted(references),
                }
            else:
                weight_serialization = {
                    "kind": "standalone",
                    "index": None,
                    "shards": sorted(standalone),
                }
            artifact_kind = "full"
        else:
            missing = _DORA_REQUIRED_NAMES - names
            if missing:
                raise ValueError(f"DoRA final artifact is missing required files: {sorted(missing)}")
            if full_names:
                raise ValueError("DoRA final artifact must not contain full-model weights")
            artifact_kind = "dora"
            weight_serialization = None
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stage1-artifact-topology-witness",
        "method": method,
        "artifact_kind": artifact_kind,
        "output_dir": str(output_dir),
        "final_save": final_save,
        "weight_serialization": weight_serialization,
        "files": entries,
        "checkpoint_bytes": sum(item["size"] for item in entries),
    }
    payload["artifact_manifest_digest"] = config_digest(payload)
    return payload


def _load_producer_run_config(path: Path) -> dict[str, object]:
    return load_stage1_run_config_witness(path).payload()


@dataclass(frozen=True)
class Stage1RunConfigWitness:
    path: Path
    file_witness: tuple[int, int, int, int]
    canonical_payload: bytes
    config_digest: str

    def payload(self) -> dict[str, object]:
        payload = json.loads(self.canonical_payload)
        if type(payload) is not dict:  # pragma: no cover - constructor invariant.
            raise AssertionError("run config witness payload stopped being an object")
        return payload


def load_stage1_run_config_witness(path: Path) -> Stage1RunConfigWitness:
    """Read and validate a launch config exactly once into an immutable witness."""

    resolved = Path(path).resolve(strict=True)
    before = _file_witness(resolved)
    payload = _validate_producer_run_config_payload(_read_json_once(resolved))
    after = _file_witness(resolved)
    if after != before:
        raise RuntimeError("Stage 1 run config changed while it was read")
    canonical = _canonical_json_bytes(payload)
    return Stage1RunConfigWitness(
        path=resolved,
        file_witness=before,
        canonical_payload=canonical,
        config_digest=payload["config_digest"],
    )


def validate_stage1_runtime_args(
    witness: Stage1RunConfigWitness, args: object
) -> dict[str, object]:
    config = witness.payload()
    expected = {
        "method": getattr(args, "tuning_method", None),
        "model": getattr(args, "model", None),
        "source_data_path": str(
            Path(getattr(args, "benchmark_source_data_path", "")).resolve()
        ),
        "data_path": str(Path(getattr(args, "data_path", "")).resolve()),
        "image_root": str(Path(getattr(args, "image_root", "")).resolve()),
        "output_dir": str(Path(getattr(args, "save_model_path", "")).resolve()),
        "micro_batch": getattr(args, "per_device_train_batch_size", None),
        "gradient_accumulation_steps": getattr(
            args, "gradient_accumulation_steps", None
        ),
        "epochs": getattr(args, "epochs", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "seed": getattr(args, "seed", None),
        "deepspeed_config": str(
            Path(getattr(args, "deepspeed_config", "")).resolve()
        ),
        "attention_implementation": getattr(
            args, "attention_implementation", None
        ),
        "save_strategy": getattr(args, "save_strategy", None),
        "final_save": getattr(args, "final_save", None),
        "max_steps": max(0, getattr(args, "max_steps", -1)),
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(
                f"Stage 1 runtime {name} does not match sealed run config"
            )
    if config["deepspeed_config_digest"] != _small_config_digest(
        Path(expected["deepspeed_config"])
    ):
        raise ValueError("Stage 1 runtime DeepSpeed config digest mismatch")
    if _stat_payload(Path(config["source_data_path"])) != config["data_stat"]:
        raise ValueError("Stage 1 runtime source data stat witness is stale")
    audit = _load_bound_audit(
        Path(config["probe_rows"]),
        data_path=Path(config["source_data_path"]),
        image_root=Path(config["image_root"]),
        count=64,
    )
    if (
        audit["audit_digest"] != config["audit_digest"]
        or audit["input_stat"] != config["data_stat"]
        or [row["source_row_index"] for row in audit["selected_rows"]]
        != config["probe_row_indices"]
    ):
        raise ValueError("Stage 1 runtime audit rows do not match sealed run config")
    tuning = config["tuning_config"]
    if config["method"] == "dora":
        runtime_tuning = {
            "use_dora": True,
            "rank": getattr(args, "lora_r", None),
            "alpha": getattr(args, "lora_alpha", None),
            "dropout": getattr(args, "lora_dropout", None),
            "target_modules": getattr(args, "lora_target_modules", None),
        }
        if tuning != runtime_tuning:
            raise ValueError("Stage 1 runtime DoRA tuning does not match sealed config")
    return config


def _validate_producer_run_config_payload(payload: object) -> dict[str, object]:
    if type(payload) is not dict:
        raise TypeError("Stage 1 producer run config must be an object")
    _require_exact_keys(
        payload,
        {
            "schema_version", "kind", "method", "tuning_config", "model",
            "source_data_path", "data_path", "data_stat", "image_root",
            "output_dir", "micro_batch", "world_size",
            "gradient_accumulation_steps", "effective_global_batch", "epochs",
            "learning_rate", "seed", "max_steps", "final_save",
            "deepspeed_config", "deepspeed_config_digest",
            "attention_implementation", "save_strategy", "audit_digest",
            "probe_rows", "probe_row_indices", "maximum_config_digest",
            "prerequisite_chain_digest",
            "prerequisite_chain",
            "config_digest",
        },
        "Stage 1 producer run config",
    )
    if payload.get("schema_version") != _SCHEMA_VERSION or payload.get("kind") not in {
        "stage1-probe-run-manifest", "stage1-timed-launch-manifest",
        "stage1-smoke-run-manifest",
    }:
        raise ValueError("unsupported Stage 1 producer run config schema")
    claimed = payload.get("config_digest")
    _require_digest("config_digest", claimed)
    digest_input = dict(payload)
    digest_input.pop("config_digest")
    if config_digest(digest_input) != claimed:
        raise ValueError("Stage 1 producer run config digest mismatch")
    for name in (
        "method", "model", "source_data_path", "data_path", "image_root",
        "output_dir", "deepspeed_config", "attention_implementation",
        "save_strategy", "probe_rows",
    ):
        _require_string(name, payload.get(name))
    _validated_method(payload.get("method"))
    tuning = payload.get("tuning_config")
    if payload["method"] == "full":
        if tuning is not None:
            raise ValueError("full run config must not contain DoRA tuning")
    elif tuning != {
        "use_dora": True, "rank": 32, "alpha": 64, "dropout": 0.05,
        "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    }:
        raise ValueError("DoRA run config does not match the approved tuning contract")
    for name in (
        "micro_batch", "world_size", "gradient_accumulation_steps",
        "effective_global_batch",
    ):
        _require_int(name, payload.get(name), positive=True)
    if payload["effective_global_batch"] != (
        payload["micro_batch"] * payload["world_size"]
        * payload["gradient_accumulation_steps"]
    ):
        raise ValueError("producer run config effective batch invariant failed")
    if type(payload.get("final_save")) is not bool:
        raise TypeError("producer run config final_save must be boolean")
    if (
        payload.get("epochs") != 3
        or payload.get("learning_rate") != 1e-5
        or payload.get("seed") != 42
        or payload.get("attention_implementation") != "flash_attention_2"
        or payload.get("save_strategy") != "no"
        or payload.get("gradient_accumulation_steps") != 2
        or payload.get("world_size") != 2
    ):
        raise ValueError("producer run config violates approved experiment invariants")
    _require_int("epochs", payload.get("epochs"), positive=True)
    _require_int("seed", payload.get("seed"))
    _require_int("max_steps", payload.get("max_steps"))
    stat_witness = payload.get("data_stat")
    _require_exact_keys(
        stat_witness, {"device", "inode", "size", "mtime_ns"}, "producer data stat witness"
    )
    for name, value in stat_witness.items():
        _require_int(f"data_stat.{name}", value)
    indices = payload.get("probe_row_indices")
    if type(indices) is not list:
        raise TypeError("producer run config probe_row_indices must be an array")
    for index, value in enumerate(indices):
        _require_int(f"probe_row_indices[{index}]", value)
    _require_digest("audit_digest", payload.get("audit_digest"))
    _require_digest("deepspeed_config_digest", payload.get("deepspeed_config_digest"))
    maximum_digest = payload.get("maximum_config_digest")
    if maximum_digest is not None:
        _require_digest("maximum_config_digest", maximum_digest)
    prerequisite_digest = payload.get("prerequisite_chain_digest")
    if payload["kind"] == "stage1-smoke-run-manifest":
        if maximum_digest is not None:
            raise ValueError("smoke evidence cannot depend on a capacity maximum")
        if prerequisite_digest is not None:
            raise ValueError("smoke evidence cannot depend on itself as a prerequisite")
        if payload.get("prerequisite_chain") is not None:
            raise ValueError("smoke evidence cannot name a prerequisite chain")
    else:
        _require_digest("prerequisite_chain_digest", prerequisite_digest)
        _require_string("prerequisite_chain", payload.get("prerequisite_chain"))
        if payload["kind"] == "stage1-probe-run-manifest":
            if maximum_digest is not None or payload["final_save"] is not False or payload["max_steps"] == 0:
                raise ValueError("probe run config has invalid maximum/save/step settings")
        elif (
            maximum_digest is None
            or payload["final_save"] is not True
            or payload["max_steps"] != 0
        ):
            raise ValueError("timed run config has invalid maximum/save/step settings")
    return payload


def _load_prerequisite_chain(path: Path) -> dict[str, object]:
    payload = _read_json_once(path)
    _require_exact_keys(
        payload,
        {"schema_version", "kind", "shared_identity", "shared_identity_digest",
         "producer_digests", "resume_witness_digests", "prerequisite_chain_digest"},
        "Stage 1 smoke prerequisite chain",
    )
    if payload.get("schema_version") != _SCHEMA_VERSION or payload.get("kind") != "stage1-smoke-prerequisite-chain":
        raise ValueError("invalid Stage 1 smoke prerequisite chain")
    identity = payload.get("shared_identity")
    if type(identity) is not dict:
        raise TypeError("smoke shared identity must be an object")
    _require_digest("shared_identity_digest", payload.get("shared_identity_digest"))
    if config_digest(identity) != payload["shared_identity_digest"]:
        raise ValueError("smoke shared identity digest mismatch")
    for name in ("producer_digests", "resume_witness_digests"):
        values = payload.get(name)
        _require_exact_keys(values, {"full", "dora"}, f"smoke {name}")
        for method, digest in values.items():
            _require_digest(f"{name}.{method}", digest)
    claimed = payload.get("prerequisite_chain_digest")
    _require_digest("prerequisite_chain_digest", claimed)
    digest_input = dict(payload)
    digest_input.pop("prerequisite_chain_digest")
    if config_digest(digest_input) != claimed:
        raise ValueError("smoke prerequisite chain digest mismatch")
    return payload


_PREREQUISITE_CONTROL_FIELDS = (
    "model", "source_data_path", "data_stat", "image_root", "world_size",
    "gradient_accumulation_steps", "epochs", "learning_rate", "seed",
    "deepspeed_config", "deepspeed_config_digest", "attention_implementation",
    "save_strategy",
)


def _validate_prerequisite_controls(
    config: Mapping[str, object], prerequisite: Mapping[str, object]
) -> None:
    identity = prerequisite["shared_identity"]
    for name in _PREREQUISITE_CONTROL_FIELDS:
        source_name = (
            "source_data_path"
            if name == "source_data_path" and "source_data_path" in config
            else "data_path" if name == "source_data_path" else name
        )
        if identity.get(name) != config.get(source_name):
            raise ValueError(f"run changes smoke prerequisite field: {name}")


def _small_config_digest(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if resolved.stat().st_size > 1024 * 1024:
        raise ValueError(f"configuration file is unexpectedly large: {resolved}")
    data = resolved.read_bytes()
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed configuration JSON: {resolved}") from error
    if (
        type(payload) is not dict
        or type(payload.get("zero_optimization")) is not dict
        or payload["zero_optimization"].get("stage") != 3
    ):
        raise ValueError("Stage 1 benchmark requires an exact ZeRO-3 configuration")
    return hashlib.sha256(data).hexdigest()


def _prepare_probe_run(args: argparse.Namespace) -> str:
    method = _validated_method(args.method)
    source_data = Path(args.source_data_path).resolve(strict=True)
    train_data = Path(args.data_path).resolve(strict=True)
    image_root = Path(args.image_root).resolve(strict=True)
    audit = _load_bound_audit(
        Path(args.probe_rows), data_path=source_data, image_root=image_root
    )
    micro_batch = _require_int("micro_batch", args.micro_batch, positive=True)
    world_size = _require_int("world_size", args.world_size, positive=True)
    accumulation = _require_int(
        "gradient_accumulation_steps", args.gradient_accumulation, positive=True
    )
    max_steps = _require_int("max_steps", args.max_steps, positive=True)
    deepspeed = Path(args.deepspeed_config).resolve(strict=True)
    prerequisite = _load_prerequisite_chain(Path(args.prerequisite_chain))
    config = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stage1-probe-run-manifest",
        "method": method,
        "tuning_config": None if method == "full" else {
            "use_dora": True, "rank": 32, "alpha": 64, "dropout": 0.05,
            "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        },
        "model": _require_string("model", args.model),
        "source_data_path": str(source_data),
        "data_path": str(train_data),
        "data_stat": audit["input_stat"],
        "image_root": str(image_root),
        "output_dir": str(Path(args.output_dir).resolve()),
        "micro_batch": micro_batch,
        "world_size": world_size,
        "gradient_accumulation_steps": accumulation,
        "effective_global_batch": world_size * micro_batch * accumulation,
        "epochs": 3,
        "learning_rate": 1e-5,
        "seed": 42,
        "max_steps": max_steps,
        "final_save": False,
        "deepspeed_config": str(deepspeed),
        "deepspeed_config_digest": _small_config_digest(deepspeed),
        "attention_implementation": "flash_attention_2",
        "save_strategy": "no",
        "audit_digest": audit["audit_digest"],
        "probe_rows": str(Path(args.probe_rows).resolve()),
        "probe_row_indices": [row["source_row_index"] for row in audit["selected_rows"]],
        "maximum_config_digest": None,
        "prerequisite_chain_digest": prerequisite["prerequisite_chain_digest"],
        "prerequisite_chain": str(Path(args.prerequisite_chain).resolve(strict=True)),
    }
    _validate_prerequisite_controls(config, prerequisite)
    config["config_digest"] = config_digest(config)
    path = Path(args.run_config)
    if path.exists():
        if _read_json_once(path) == config:
            return str(path)
        raise FileExistsError(f"refusing to replace probe run config: {path}")
    _atomic_write_json(path, config)
    return str(path)


def write_stage1_producer_evidence(
    path: Path,
    *,
    run_config_path: Path | None = None,
    run_config_witness: Stage1RunConfigWitness | None = None,
    evidence: Stage1BenchmarkEvidence,
    smoke_resume_witness: Mapping[str, object] | None = None,
    artifact_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Atomically publish metrics plus the producer-authenticated artifact witness."""

    if (run_config_path is None) == (run_config_witness is None):
        raise ValueError("provide exactly one run config path or immutable witness")
    witness = run_config_witness or load_stage1_run_config_witness(run_config_path)
    config = witness.payload()
    source_path = Path(config["source_data_path"])
    if not source_path.is_file() or _stat_payload(source_path) != config["data_stat"]:
        raise ValueError("producer source data stat witness is stale")
    image_root = Path(config["image_root"])
    if not image_root.is_dir() or image_root.resolve(strict=True) != image_root:
        raise ValueError("producer image root must be a resolved existing directory")
    if evidence.config_digest != config["config_digest"]:
        raise ValueError("producer evidence is not bound to its run config")
    if evidence.method != config["method"]:
        raise ValueError("producer evidence method does not match run config")
    smoke_digest = None
    if config["kind"] == "stage1-smoke-run-manifest":
        smoke_resume_witness = _validate_smoke_resume_witness(smoke_resume_witness)
        smoke_digest = config_digest(smoke_resume_witness)
    elif smoke_resume_witness is not None:
        raise ValueError("resume witness is accepted only for smoke evidence")
    artifact = (
        _build_artifact_manifest(
            Path(config["output_dir"]),
            method=evidence.method,
            final_save=config["final_save"],
        )
        if artifact_manifest is None
        else dict(artifact_manifest)
    )
    if evidence.checkpoint_bytes != artifact["checkpoint_bytes"]:
        raise ValueError("producer checkpoint_bytes does not match artifact witness")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stage1-producer-evidence",
        "run_config": config,
        "benchmark_evidence": _evidence_to_dict(evidence),
        "artifact_manifest": artifact,
        "artifact_manifest_digest": artifact["artifact_manifest_digest"],
        "smoke_resume_witness": smoke_resume_witness,
        "smoke_resume_witness_digest": smoke_digest,
    }
    payload["producer_digest"] = config_digest(payload)
    _validate_producer_evidence_payload(payload)
    path = Path(path)
    if path.exists():
        existing = _read_json_once(path)
        if existing == payload:
            return payload
        raise FileExistsError(f"refusing to replace producer evidence: {path}")
    _atomic_write_json(path, payload)
    return payload


def _load_producer_evidence(path: Path) -> tuple[Stage1BenchmarkEvidence, dict[str, object]]:
    return _validate_producer_evidence_payload(_read_json_once(path))


def _validate_producer_evidence_payload(
    payload: object,
) -> tuple[Stage1BenchmarkEvidence, dict[str, object]]:
    if type(payload) is not dict:
        raise TypeError("producer evidence must be an object")
    _require_exact_keys(
        payload,
        {"schema_version", "kind", "run_config", "benchmark_evidence",
         "artifact_manifest", "artifact_manifest_digest", "producer_digest",
         "smoke_resume_witness", "smoke_resume_witness_digest"},
        "Stage 1 producer evidence",
    )
    if payload.get("schema_version") != _SCHEMA_VERSION or payload.get("kind") != "stage1-producer-evidence":
        raise ValueError("invalid Stage 1 producer evidence schema")
    claimed = payload.get("producer_digest")
    _require_digest("producer_digest", claimed)
    digest_input = dict(payload)
    digest_input.pop("producer_digest")
    if config_digest(digest_input) != claimed:
        raise ValueError("producer evidence digest mismatch")
    config = _validate_producer_run_config_payload(payload["run_config"])
    config_claimed = config["config_digest"]
    evidence = _evidence_from_dict(payload["benchmark_evidence"])
    if evidence.config_digest != config_claimed:
        raise ValueError("producer metric/config digest mismatch")
    metric_bindings = {
        "method": evidence.method,
        "micro_batch": evidence.micro_batch,
        "world_size": evidence.world_size,
        "gradient_accumulation_steps": evidence.gradient_accumulation_steps,
        "effective_global_batch": evidence.effective_global_batch,
        "probe_row_indices": list(evidence.probe_row_indices),
    }
    for name, value in metric_bindings.items():
        if config.get(name) != value:
            raise ValueError(f"producer metric/run config mismatch for {name}")
    smoke_witness = payload.get("smoke_resume_witness")
    smoke_digest = payload.get("smoke_resume_witness_digest")
    if config["kind"] == "stage1-smoke-run-manifest":
        validated_smoke = _validate_smoke_resume_witness(smoke_witness)
        _require_digest("smoke_resume_witness_digest", smoke_digest)
        if config_digest(validated_smoke) != smoke_digest:
            raise ValueError("smoke resume witness digest mismatch")
    elif smoke_witness is not None or smoke_digest is not None:
        raise ValueError("non-smoke producer evidence contains a resume witness")
    artifact = payload["artifact_manifest"]
    if type(artifact) is not dict:
        raise TypeError("producer artifact manifest must be an object")
    _require_exact_keys(
        artifact,
        {"schema_version", "kind", "method", "artifact_kind", "output_dir",
         "final_save", "weight_serialization", "files", "checkpoint_bytes",
         "artifact_manifest_digest"},
        "producer artifact manifest",
    )
    if (
        artifact["schema_version"] != _SCHEMA_VERSION
        or artifact["kind"] != "stage1-artifact-topology-witness"
        or artifact["method"] != evidence.method
        or artifact["output_dir"] != config["output_dir"]
        or artifact["final_save"] is not config["final_save"]
    ):
        raise ValueError("producer artifact manifest identity mismatch")
    artifact_claimed = artifact.get("artifact_manifest_digest")
    _require_digest("artifact_manifest_digest", artifact_claimed)
    artifact_input = dict(artifact)
    artifact_input.pop("artifact_manifest_digest")
    if config_digest(artifact_input) != artifact_claimed or payload["artifact_manifest_digest"] != artifact_claimed:
        raise ValueError("producer artifact manifest digest mismatch")
    expected_kind = "none" if config.get("final_save") is False else evidence.method
    if artifact.get("artifact_kind") != expected_kind:
        raise ValueError("producer artifact topology does not match run final-save mode")
    files = artifact.get("files")
    if type(files) is not list:
        raise TypeError("producer artifact files must be an array")
    names: set[str] = set()
    byte_sum = 0
    for index, item in enumerate(files):
        _require_exact_keys(item, {"name", "size"}, f"producer artifact file {index}")
        name = _require_string(f"artifact file {index}.name", item.get("name"))
        if Path(name).name != name or name in names:
            raise ValueError("producer artifact filenames must be unique basenames")
        names.add(name)
        size = _require_int(f"artifact file {index}.size", item.get("size"))
        if expected_kind != "none" and size == 0:
            raise ValueError("producer artifact files must have positive size")
        if expected_kind != "none" and not _is_allowlisted_artifact_name(name, evidence.method):
            raise ValueError("producer artifact manifest contains a non-artifact file")
        byte_sum += size
    if artifact.get("checkpoint_bytes") != byte_sum:
        raise ValueError("producer artifact file sizes do not sum to checkpoint_bytes")
    if expected_kind != "none" and not _PROCESSOR_REQUIRED_NAMES <= names:
        raise ValueError("producer artifact witness is missing processor/tokenizer files")
    serialization = artifact.get("weight_serialization")
    if expected_kind == "full":
        if "config.json" not in names:
            raise ValueError("producer full artifact witness has incomplete model topology")
        _require_exact_keys(
            serialization, {"kind", "index", "shards"},
            "producer full weight serialization",
        )
        serialization_kind = serialization.get("kind")
        shards = serialization.get("shards")
        if type(shards) is not list or not shards or shards != sorted(set(shards)):
            raise ValueError("producer full weight shards must be a sorted unique array")
        for index, name in enumerate(shards):
            _require_string(f"producer full shard {index}", name)
        if serialization_kind == "standalone":
            if serialization.get("index") is not None or len(shards) != 1 or shards[0] not in {
                "model.safetensors", "pytorch_model.bin"
            }:
                raise ValueError("producer standalone full weight witness is invalid")
            expected_weights = set(shards)
        elif serialization_kind == "indexed":
            index_name = _require_string("producer full weight index", serialization.get("index"))
            if index_name not in {
                "model.safetensors.index.json", "pytorch_model.bin.index.json"
            }:
                raise ValueError("producer full weight index witness is invalid")
            expected_prefix = "model-" if index_name.startswith("model") else "pytorch_model-"
            expected_suffix = ".safetensors" if index_name.startswith("model") else ".bin"
            if any(
                Path(name).name != name
                or not name.startswith(expected_prefix)
                or not name.endswith(expected_suffix)
                for name in shards
            ):
                raise ValueError("producer full shard witness is invalid")
            expected_weights = {index_name, *shards}
        else:
            raise ValueError("producer full weight serialization kind is invalid")
        actual_weights = {name for name in names if _is_full_weight_name(name)}
        if actual_weights != expected_weights:
            raise ValueError("producer full weight witness does not match artifact files")
    if expected_kind == "dora":
        if not _DORA_REQUIRED_NAMES <= names or any(_is_full_weight_name(name) for name in names if type(name) is str):
            raise ValueError("producer DoRA artifact topology is invalid")
        if serialization is not None:
            raise ValueError("producer DoRA artifact must not contain full weights")
    if expected_kind == "none" and serialization is not None:
        raise ValueError("no-save producer artifact cannot contain weight serialization")
    if artifact.get("checkpoint_bytes") != evidence.checkpoint_bytes:
        raise ValueError("producer artifact byte count mismatch")
    source = Path(config["source_data_path"])
    if not source.is_file() or _stat_payload(source) != config["data_stat"]:
        raise ValueError("producer evidence source data stat witness is stale")
    image_root = Path(config["image_root"])
    if not image_root.is_dir() or image_root.resolve(strict=True) != image_root:
        raise ValueError("producer evidence image root witness is stale")
    return evidence, payload


def _validate_smoke_resume_witness(payload: object) -> dict[str, object]:
    _require_exact_keys(
        payload,
        {"initial_final_step", "resume_initial_step", "resume_final_step",
         "batch_size_observed", "finite_loss", "checkpoint_validated",
         "resume_artifact_digest"},
        "Stage 1 smoke resume witness",
    )
    for name in (
        "initial_final_step", "resume_initial_step", "resume_final_step",
        "batch_size_observed",
    ):
        _require_int(name, payload.get(name), positive=True)
    if (
        payload["initial_final_step"] != 3
        or payload["resume_initial_step"] != 3
        or payload["resume_final_step"] != 4
        or payload["batch_size_observed"] != 2
        or payload.get("finite_loss") is not True
        or payload.get("checkpoint_validated") is not True
    ):
        raise ValueError("smoke resume witness violates the approved 3-to-4 contract")
    _require_digest("resume_artifact_digest", payload.get("resume_artifact_digest"))
    return dict(payload)


def _stat_payload(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _validate_count(count: object) -> int:
    return _require_int("count", count, positive=True)


def _reference_list(value: object, *, field: str, row_index: int) -> list[str]:
    if type(value) is str:
        references = [value]
    elif type(value) is list:
        references = value
    else:
        raise ValueError(
            f"source_row_index={row_index}: {field} must be a path or list of paths"
        )
    if not references:
        raise ValueError(f"source_row_index={row_index}: {field} must not be empty")
    for reference in references:
        if type(reference) is not str or not reference:
            raise ValueError(
                f"source_row_index={row_index}: {field} contains a malformed reference"
            )
    return references


def _validate_reference(
    reference: str,
    *,
    image_root: Path,
    row_index: int,
    role: str,
) -> None:
    selected = Path(reference)
    if not selected.is_absolute():
        selected = image_root / selected
    try:
        stat = selected.stat()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"source_row_index={row_index}: missing {role} image reference: {reference}"
        ) from error
    if not stat_module.S_ISREG(stat.st_mode):
        raise ValueError(
            f"source_row_index={row_index}: {role} image reference is not a file: "
            f"{reference} (mode={stat.st_mode:o})"
        )


def _score_row(
    row: object, *, source_row_index: int, image_root: Path
) -> tuple[tuple[int, int, int, int], dict[str, object]]:
    if type(row) is not dict:
        raise ValueError(f"source_row_index={source_row_index}: row must be a JSON object")
    if "image_input" not in row:
        raise ValueError(f"source_row_index={source_row_index}: missing image_input")
    originals = _reference_list(
        row["image_input"], field="image_input", row_index=source_row_index
    )
    for reference in originals:
        _validate_reference(
            reference,
            image_root=image_root,
            row_index=source_row_index,
            role="original",
        )
    text_input = row.get("text_input", "")
    if type(text_input) is not str:
        raise ValueError(f"source_row_index={source_row_index}: text_input must be text")
    sequence_plan = row.get("sequence_plan")
    if type(sequence_plan) is not list:
        raise ValueError(
            f"source_row_index={source_row_index}: sequence_plan must be a list"
        )
    text_parts = [text_input]
    helper_count = 0
    latent_block_count = 0
    for step_index, step in enumerate(sequence_plan):
        if type(step) is not dict:
            raise ValueError(
                f"source_row_index={source_row_index}: sequence_plan[{step_index}] "
                "must be an object"
            )
        step_type = step.get("type")
        if step_type == "text":
            content = step.get("content", "")
            if type(content) is not str:
                raise ValueError(
                    f"source_row_index={source_row_index}: sequence_plan[{step_index}] "
                    "text content must be a string"
                )
            text_parts.append(content)
        elif step_type == "latent":
            latent_block_count += 1
            if "helper_image" not in step:
                raise ValueError(
                    f"source_row_index={source_row_index}: sequence_plan[{step_index}] "
                    "is missing helper_image"
                )
            helpers = _reference_list(
                step["helper_image"],
                field=f"sequence_plan[{step_index}].helper_image",
                row_index=source_row_index,
            )
            helper_count += len(helpers)
            for reference in helpers:
                _validate_reference(
                    reference,
                    image_root=image_root,
                    row_index=source_row_index,
                    role="helper",
                )
        else:
            raise ValueError(
                f"source_row_index={source_row_index}: sequence_plan[{step_index}] "
                f"has unsupported type {step_type!r}"
            )
    token_estimate = sum(len(text.split()) for text in text_parts)
    metrics = {
        "source_row_index": source_row_index,
        "token_estimate": token_estimate,
        "original_image_count": len(originals),
        "helper_image_count": helper_count,
        "latent_block_count": latent_block_count,
        "example": row,
    }
    return (
        token_estimate,
        len(originals),
        helper_count,
        latent_block_count,
    ), metrics


def _validate_audit_for_reuse(
    payload: object,
    *,
    data_path: Path,
    image_root: Path,
    count: int,
    witness: dict[str, int],
) -> tuple[int, ...] | None:
    if type(payload) is not dict:
        raise ValueError("high-cost row audit must be a JSON object")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "kind",
            "data_path",
            "image_root",
            "requested_count",
            "row_count",
            "input_stat",
            "selected_rows",
            "audit_digest",
        },
        "high-cost row audit",
    )
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported high-cost row audit schema_version")
    if payload.get("kind") != "stage1-high-cost-row-audit":
        raise ValueError("invalid high-cost row audit kind")
    claimed_digest = payload.get("audit_digest")
    _require_digest("audit_digest", claimed_digest)
    digest_input = dict(payload)
    digest_input.pop("audit_digest")
    if config_digest(digest_input) != claimed_digest:
        raise ValueError("high-cost row audit digest mismatch")
    for name in ("requested_count", "row_count"):
        _require_int(name, payload.get(name))
    stat_witness = payload.get("input_stat")
    _require_exact_keys(
        stat_witness,
        {"device", "inode", "size", "mtime_ns"},
        "high-cost input stat witness",
    )
    for name, value in stat_witness.items():
        _require_int(f"input_stat.{name}", value)
    if (
        payload.get("data_path") != str(data_path)
        or payload.get("image_root") != str(image_root)
        or payload.get("requested_count") != count
        or payload.get("input_stat") != witness
    ):
        return None
    selected = payload.get("selected_rows")
    if type(selected) is not list or len(selected) != min(count, payload.get("row_count", -1)):
        raise ValueError("high-cost row audit has malformed selected_rows")
    indices: list[int] = []
    for item in selected:
        if type(item) is not dict:
            raise ValueError("high-cost selected row must be an object")
        _require_exact_keys(
            item,
            {
                "source_row_index",
                "token_estimate",
                "original_image_count",
                "helper_image_count",
                "latent_block_count",
                "example",
            },
            "high-cost selected row",
        )
        index = item.get("source_row_index")
        _require_int("source_row_index", index)
        for name in (
            "token_estimate",
            "original_image_count",
            "helper_image_count",
            "latent_block_count",
        ):
            _require_int(name, item.get(name))
        if type(item.get("example")) is not dict:
            raise ValueError("high-cost selected row is missing its text/reference example")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError("high-cost row audit contains duplicate source indices")
    return tuple(indices)


def select_high_cost_probe_rows(
    examples: Sequence[Mapping[str, object]] | str | Path,
    count: int,
    audit_path: str | Path,
    *,
    image_root: str | Path | None = None,
) -> tuple[int, ...]:
    """Rank CoMT rows from text/reference metadata and persist a reusable audit."""

    count = _validate_count(count)
    audit_path = Path(audit_path)
    if isinstance(examples, (str, os.PathLike)):
        data_path = Path(examples).resolve(strict=True)
        root = Path(image_root).resolve(strict=True) if image_root else data_path.parent
        witness = _stat_payload(data_path)
        if audit_path.is_file():
            reused = _validate_audit_for_reuse(
                _read_json_once(audit_path),
                data_path=data_path,
                image_root=root,
                count=count,
                witness=witness,
            )
            if reused is not None:
                return reused
        rows: list[dict[str, object]] = []
        with data_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(
                        f"blank CoMT JSONL physical line {line_number} is not allowed"
                    )
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"malformed CoMT JSONL at line {line_number}: {error}"
                    ) from error
                if type(row) is not dict:
                    raise ValueError(
                        f"malformed CoMT JSONL at line {line_number}: row must be an object"
                    )
                rows.append(row)
        if _stat_payload(data_path) != witness:
            raise RuntimeError("CoMT JSONL changed while its metadata audit was running")
    else:
        if type(examples) not in (list, tuple):
            raise TypeError("examples must be a JSONL path or a list/tuple of mappings")
        rows = list(examples)
        root = Path(image_root or ".").resolve(strict=True)
        metadata_bytes = _canonical_json_bytes(rows)
        digest = hashlib.sha256(metadata_bytes).hexdigest()
        data_path = Path(f"in-memory:{digest}")
        witness = {"device": 0, "inode": 0, "size": len(metadata_bytes), "mtime_ns": 0}

    ranked: list[tuple[tuple[int, int, int, int], dict[str, object]]] = []
    for source_row_index, row in enumerate(rows):
        ranked.append(
            _score_row(row, source_row_index=source_row_index, image_root=root)
        )
    ranked.sort(key=lambda item: (*(-value for value in item[0]), item[1]["source_row_index"]))
    selected_rows = [metrics for _, metrics in ranked[:count]]
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stage1-high-cost-row-audit",
        "data_path": str(data_path),
        "image_root": str(root),
        "requested_count": count,
        "row_count": len(rows),
        "input_stat": witness,
        "selected_rows": selected_rows,
    }
    payload["audit_digest"] = config_digest(payload)
    _atomic_write_json(audit_path, payload)
    return tuple(item["source_row_index"] for item in selected_rows)


def _load_bound_audit(
    audit_path: Path, *, data_path: Path, image_root: Path, count: int = 64
) -> dict[str, object]:
    payload = _read_json_once(audit_path)
    witness = _stat_payload(data_path)
    selected = _validate_audit_for_reuse(
        payload,
        data_path=data_path.resolve(strict=True),
        image_root=image_root.resolve(strict=True),
        count=count,
        witness=witness,
    )
    if selected is None:
        raise ValueError("probe row audit does not match source data/image root/stat")
    if not selected:
        raise ValueError("probe row audit has no selected rows")
    return payload


def _atomic_write_probe_jsonl(
    audit_path: Path,
    output_path: Path,
    *,
    data_path: Path,
    image_root: Path,
) -> None:
    payload = _load_bound_audit(
        audit_path, data_path=data_path, image_root=image_root, count=64
    )
    selected = payload["selected_rows"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for item in selected:
                if type(item) is not dict or type(item.get("example")) is not dict:
                    raise ValueError("probe row audit contains malformed selected example")
                handle.write(
                    json.dumps(
                        item["example"], ensure_ascii=False, allow_nan=False, sort_keys=True
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _probe_attempt_path(attempt_dir: Path) -> Path:
    return attempt_dir / "evidence" / "attempt.json"


def _record_probe_attempt(args: argparse.Namespace) -> dict[str, object]:
    method = _validated_method(args.method)
    micro_batch = _require_int("micro_batch", args.micro_batch, positive=True)
    world_size = _require_int("world_size", args.world_size, positive=True)
    accumulation = _require_int(
        "gradient_accumulation_steps", args.gradient_accumulation, positive=True
    )
    process_id = _require_int("process_id", args.process_id, positive=True)
    exit_code = _require_int("exit_code", args.exit_code)
    max_steps = _require_int("max_steps", args.max_steps, positive=True)
    source_data_path = Path(args.source_data_path).resolve(strict=True)
    image_root = Path(args.image_root).resolve(strict=True)
    audit = _load_bound_audit(
        Path(args.probe_rows), data_path=source_data_path, image_root=image_root
    )
    probe_indices = [row["source_row_index"] for row in audit["selected_rows"]]
    if type(args.command) is not list or not args.command:
        raise ValueError("probe attempt requires a recorded command")
    command = args.command[1:] if args.command[0] == "--" else args.command
    if not command:
        raise ValueError("probe attempt command must not be empty")
    if command.count("--benchmark-run-config") != 1:
        raise ValueError("probe argv requires exactly one benchmark run config")
    run_config_index = command.index("--benchmark-run-config")
    if run_config_index + 1 >= len(command):
        raise ValueError("probe benchmark run config flag has no value")
    sealed_run = _load_producer_run_config(Path(command[run_config_index + 1]))
    expected_run = {
        "method": method,
        "micro_batch": micro_batch,
        "world_size": world_size,
        "gradient_accumulation_steps": accumulation,
        "max_steps": max_steps,
        "final_save": False,
        "model": _require_string("model", args.model),
        "source_data_path": str(source_data_path),
        "image_root": str(image_root),
        "output_dir": str(Path(args.output_dir).resolve()),
        "deepspeed_config": str(Path(args.deepspeed_config).resolve()),
        "attention_implementation": _require_string(
            "attention_implementation", args.attention_implementation
        ),
        "audit_digest": audit["audit_digest"],
    }
    for name, value in expected_run.items():
        if sealed_run.get(name) != value:
            raise ValueError(f"probe run config mismatch for {name}")
    launch_config = {
        "method": method,
        "micro_batch": micro_batch,
        "world_size": world_size,
        "gradient_accumulation_steps": accumulation,
        "effective_global_batch": world_size * micro_batch * accumulation,
        "learning_rate": float(args.learning_rate),
        "seed": args.seed,
        "max_steps": max_steps,
        "final_save": False,
        "model": expected_run["model"],
        "tuning_config": sealed_run["tuning_config"],
        "data_path": str(source_data_path),
        "train_data_path": sealed_run["data_path"],
        "data_stat": audit["input_stat"],
        "image_root": str(image_root),
        "deepspeed_config": str(Path(args.deepspeed_config).resolve()),
        "deepspeed_config_digest": sealed_run["deepspeed_config_digest"],
        "attention_implementation": _require_string(
            "attention_implementation", args.attention_implementation
        ),
        "probe_rows": str(Path(args.probe_rows).resolve()),
        "probe_row_indices": probe_indices,
        "audit_digest": audit["audit_digest"],
        "run_config_digest": sealed_run["config_digest"],
        "prerequisite_chain_digest": sealed_run["prerequisite_chain_digest"],
        "prerequisite_chain": sealed_run["prerequisite_chain"],
        "command": command,
    }
    digest = config_digest(launch_config)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stage1-probe-attempt",
        "attempt_id": _require_string("attempt_id", args.attempt_id),
        "recorded_at_ns": time.time_ns(),
        "status": "success" if exit_code == 0 else "requires-classification",
        "method": method,
        "micro_batch": micro_batch,
        "world_size": world_size,
        "gradient_accumulation_steps": accumulation,
        "effective_global_batch": world_size * micro_batch * accumulation,
        "max_steps": max_steps,
        "final_save": False,
        "process_id": process_id,
        "exit_code": exit_code,
        "requires_oom_classification": exit_code != 0,
        "output_dir": str(Path(args.output_dir).resolve()),
        "stdout_path": str(Path(args.stdout_path).resolve()),
        "stderr_path": str(Path(args.stderr_path).resolve()),
        "probe_rows": str(Path(args.probe_rows).resolve()),
        "probe_row_indices": probe_indices,
        "audit_digest": audit["audit_digest"],
        "data_stat": audit["input_stat"],
        "prerequisite_chain_digest": sealed_run["prerequisite_chain_digest"],
        "command": command,
        "launch_config": launch_config,
        "config_digest": digest,
    }
    payload["attempt_digest"] = config_digest(payload)
    attempt_dir = Path(args.attempt_dir)
    attempt_path = _probe_attempt_path(attempt_dir)
    if attempt_path.exists():
        raise FileExistsError(f"probe attempts are append-only: {attempt_path}")
    if len(attempt_dir.parents) >= 4 and attempt_dir.parents[2].name in {
        "attempts",
        "long-smoke",
    }:
        probe_root = attempt_dir.parents[3]
        for previous_path in _attempt_files(probe_root, method):
            previous = _load_probe_attempt(previous_path)
            if previous["process_id"] == process_id:
                raise RuntimeError(
                    f"probe process ID was reused instead of remaining unique: {process_id}"
                )
    _atomic_write_json(attempt_path, payload)
    return payload


def _load_probe_attempt(path: Path) -> dict[str, object]:
    if path.is_dir():
        path = _probe_attempt_path(path)
    payload = _read_json_once(path)
    if type(payload) is not dict or payload.get("kind") != "stage1-probe-attempt":
        raise ValueError(f"invalid Stage 1 probe attempt: {path}")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "kind",
            "attempt_id",
            "recorded_at_ns",
            "status",
            "method",
            "micro_batch",
            "world_size",
            "gradient_accumulation_steps",
            "effective_global_batch",
            "max_steps",
            "final_save",
            "process_id",
            "exit_code",
            "requires_oom_classification",
            "output_dir",
            "stdout_path",
            "stderr_path",
            "probe_rows",
            "probe_row_indices",
            "audit_digest",
            "data_stat",
            "prerequisite_chain_digest",
            "command",
            "launch_config",
            "config_digest",
            "attempt_digest",
        },
        "Stage 1 probe attempt",
    )
    for name in (
        "schema_version",
        "recorded_at_ns",
        "micro_batch",
        "world_size",
        "gradient_accumulation_steps",
        "effective_global_batch",
        "max_steps",
        "process_id",
        "exit_code",
    ):
        _require_int(name, payload.get(name))
    if payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("unsupported Stage 1 probe attempt schema")
    _validated_method(payload.get("method"))
    for name in (
        "attempt_id", "status", "method", "output_dir", "stdout_path",
        "stderr_path", "probe_rows",
    ):
        _require_string(name, payload.get(name))
    if len(path.parents) >= 4 and path.name == "attempt.json":
        attempt_dir = path.parents[1]
        if (
            path.parents[3].name != payload["method"]
            or path.parents[2].name != f"micro-batch-{payload['micro_batch']}"
            or attempt_dir.name != f"attempt-{payload['attempt_id']}"
            or Path(payload["output_dir"]) != (attempt_dir / "output").resolve()
            or Path(payload["stdout_path"]) != (attempt_dir / "logs" / "stdout.log").resolve()
            or Path(payload["stderr_path"]) != (attempt_dir / "logs" / "stderr.log").resolve()
        ):
            raise ValueError("probe attempt path topology does not match sealed identity")
    expected_status = "success" if payload["exit_code"] == 0 else "requires-classification"
    if payload.get("status") != expected_status:
        raise ValueError("probe attempt status does not match exit_code")
    if type(payload.get("requires_oom_classification")) is not bool:
        raise TypeError("requires_oom_classification must be a boolean")
    if payload["requires_oom_classification"] is not (payload["exit_code"] != 0):
        raise ValueError("probe classification requirement does not match exit_code")
    _require_int("process_id", payload["process_id"], positive=True)
    _require_int("recorded_at_ns", payload["recorded_at_ns"], positive=True)
    _require_int("max_steps", payload["max_steps"], positive=True)
    if type(payload.get("final_save")) is not bool:
        raise TypeError("final_save must be a boolean")
    _require_digest("config_digest", payload.get("config_digest"))
    _require_digest("audit_digest", payload.get("audit_digest"))
    if config_digest(payload.get("launch_config")) != payload["config_digest"]:
        raise ValueError("probe attempt config digest mismatch")
    _validate_probe_launch_config(payload["launch_config"])
    _require_digest("attempt_digest", payload.get("attempt_digest"))
    digest_input = dict(payload)
    digest_input.pop("attempt_digest")
    if config_digest(digest_input) != payload["attempt_digest"]:
        raise ValueError("probe attempt digest mismatch")
    if payload["effective_global_batch"] != (
        payload["world_size"]
        * payload["micro_batch"]
        * payload["gradient_accumulation_steps"]
    ):
        raise ValueError("probe attempt effective_global_batch invariant failed")
    if type(payload.get("command")) is not list or not all(
        type(item) is str for item in payload["command"]
    ):
        raise TypeError("probe attempt command must be a string array")
    for name in (
        "method", "micro_batch", "world_size", "gradient_accumulation_steps",
        "effective_global_batch", "max_steps", "final_save", "command",
        "probe_row_indices", "audit_digest", "data_stat",
        "prerequisite_chain_digest",
    ):
        if payload[name] != payload["launch_config"][name]:
            raise ValueError(f"probe attempt duplicate field mismatch: {name}")
    _validate_probe_argv(payload)
    return payload


def _validate_probe_argv(attempt: Mapping[str, object]) -> None:
    command = attempt["command"]
    if type(command) is not list or not command:
        raise TypeError("probe attempt command must be a nonempty string array")
    for index, item in enumerate(command):
        _require_string(f"command[{index}]", item)
    flag_values = {
        "--method": str(attempt["method"]),
        "--model": str(attempt["launch_config"]["model"]),
        "--source-data-path": str(attempt["launch_config"]["data_path"]),
        "--data-path": str(attempt["launch_config"]["train_data_path"]),
        "--image-root": str(attempt["launch_config"]["image_root"]),
        "--output-dir": str(attempt["output_dir"]),
        "--micro-batch": str(attempt["micro_batch"]),
        "--epochs": "3",
        "--learning-rate": str(attempt["launch_config"]["learning_rate"]),
        "--gradient-accumulation": str(attempt["gradient_accumulation_steps"]),
        "--seed": str(attempt["launch_config"]["seed"]),
        "--deepspeed-config": str(attempt["launch_config"]["deepspeed_config"]),
        "--attention-implementation": str(attempt["launch_config"]["attention_implementation"]),
        "--save-strategy": "no",
        "--log-file": str(attempt["stdout_path"]).replace("stdout.log", "train.log"),
        "--max-steps": str(attempt["max_steps"]),
        "--benchmark-run-config": str(
            Path(str(attempt["output_dir"])).parent / "evidence" / "run-config.json"
        ),
        "--producer-evidence": str(
            Path(str(attempt["output_dir"])).parent / "evidence" / "producer-evidence.json"
        ),
    }
    allowed = set(flag_values) | {"--no-final-save"}
    seen: set[str] = set()
    index = 1
    while index < len(command):
        flag = command[index]
        if flag == "--no-final-save":
            if flag in seen:
                raise ValueError("probe argv repeats --no-final-save")
            seen.add(flag)
            index += 1
            continue
        if flag not in allowed or flag == "--no-final-save":
            raise ValueError(f"probe argv contains an unsupported flag: {flag}")
        if flag in seen or index + 1 >= len(command):
            raise ValueError(f"probe argv has a repeated or valueless flag: {flag}")
        value = command[index + 1]
        _require_string(f"argv value for {flag}", value)
        if flag in flag_values:
            expected = flag_values[flag]
            if flag == "--learning-rate":
                if float(value) != float(expected):
                    raise ValueError(f"probe argv value mismatch for {flag}")
            elif value != expected:
                if flag == "--data-path":
                    raise ValueError(
                        "probe --data-path must equal the materialized audited probe input"
                    )
                raise ValueError(f"probe argv value mismatch for {flag}")
        seen.add(flag)
        index += 2
    if seen != allowed:
        raise ValueError(f"probe argv flag set mismatch: missing={sorted(allowed - seen)}")


def _validate_probe_launch_config(payload: object) -> None:
    _require_exact_keys(
        payload,
        {
            "method",
            "micro_batch",
            "world_size",
            "gradient_accumulation_steps",
            "effective_global_batch",
            "learning_rate",
            "seed",
            "max_steps",
            "final_save",
            "model",
            "tuning_config",
            "data_path",
            "train_data_path",
            "data_stat",
            "image_root",
            "deepspeed_config",
            "deepspeed_config_digest",
            "attention_implementation",
            "probe_rows",
            "probe_row_indices",
            "audit_digest",
            "run_config_digest",
            "prerequisite_chain_digest",
            "prerequisite_chain",
            "command",
        },
        "probe launch config",
    )
    _validated_method(payload.get("method"))
    for name in (
        "micro_batch",
        "world_size",
        "gradient_accumulation_steps",
        "effective_global_batch",
        "max_steps",
    ):
        _require_int(name, payload.get(name), positive=True)
    _require_int("seed", payload.get("seed"))
    if type(payload.get("learning_rate")) is not float or not math.isfinite(
        payload["learning_rate"]
    ) or payload["learning_rate"] <= 0:
        raise ValueError("probe learning_rate must be a positive finite float")
    if payload.get("final_save") is not False:
        raise ValueError("probe launch config must disable final save")
    for name in (
        "model",
        "data_path",
        "train_data_path",
        "image_root",
        "deepspeed_config",
        "attention_implementation",
        "probe_rows",
    ):
        _require_string(name, payload.get(name))
    for name in (
        "audit_digest", "deepspeed_config_digest", "run_config_digest",
        "prerequisite_chain_digest",
    ):
        _require_digest(name, payload.get(name))
    _require_string("prerequisite_chain", payload.get("prerequisite_chain"))
    tuning = payload.get("tuning_config")
    if payload["method"] == "full" and tuning is not None:
        raise ValueError("full probe config must not contain DoRA tuning")
    if payload["method"] == "dora" and type(tuning) is not dict:
        raise ValueError("DoRA probe config requires tuning settings")
    stat_witness = payload.get("data_stat")
    _require_exact_keys(
        stat_witness, {"device", "inode", "size", "mtime_ns"}, "probe data stat witness"
    )
    for name, value in stat_witness.items():
        _require_int(f"data_stat.{name}", value)
    indices = payload.get("probe_row_indices")
    if type(indices) is not list or not indices:
        raise ValueError("probe_row_indices must be a nonempty array")
    for index, value in enumerate(indices):
        _require_int(f"probe_row_indices[{index}]", value)
    if type(payload.get("command")) is not list or not payload["command"] or not all(
        type(item) is str for item in payload["command"]
    ):
        raise TypeError("probe launch config command must be a nonempty string array")
    if payload["effective_global_batch"] != (
        payload["micro_batch"]
        * payload["world_size"]
        * payload["gradient_accumulation_steps"]
    ):
        raise ValueError("probe launch config effective batch invariant failed")


def _classification_path(attempt_path: Path) -> Path:
    if attempt_path.is_dir():
        return attempt_path / "evidence" / "failure-classification.json"
    return attempt_path.parent / "failure-classification.json"


def _classify_failure(attempt_path: Path, classification: str) -> None:
    attempt = _load_probe_attempt(attempt_path)
    if attempt["exit_code"] == 0:
        raise ValueError("a successful probe attempt cannot be classified as an OOM")
    if classification != "capacity_oom":
        raise ValueError("only the systematic-debugging result capacity_oom is accepted")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stage1-probe-failure-classification",
        "classification": classification,
        "method": attempt["method"],
        "micro_batch": attempt["micro_batch"],
        "attempt_config_digest": attempt["config_digest"],
        "attempt_digest": attempt["attempt_digest"],
    }
    payload["classification_digest"] = config_digest(payload)
    path = _classification_path(attempt_path)
    if path.exists():
        if _read_json_once(path) != payload:
            raise FileExistsError(f"different failure classification already exists: {path}")
        return
    _atomic_write_json(path, payload)


def _confirmed_capacity_oom(attempt_path: Path) -> bool:
    attempt = _load_probe_attempt(attempt_path)
    path = _classification_path(attempt_path)
    if not path.is_file():
        return False
    classification = _read_json_once(path)
    _require_exact_keys(
        classification,
        {"schema_version", "kind", "classification", "method", "micro_batch",
         "attempt_config_digest", "attempt_digest", "classification_digest"},
        "probe failure classification",
    )
    claimed = classification.get("classification_digest")
    _require_digest("classification_digest", claimed)
    digest_input = dict(classification)
    digest_input.pop("classification_digest")
    if config_digest(digest_input) != claimed:
        raise ValueError("probe failure classification digest mismatch")
    return bool(
        classification.get("schema_version") == _SCHEMA_VERSION
        and classification.get("kind") == "stage1-probe-failure-classification"
        and classification.get("classification") == "capacity_oom"
        and classification.get("method") == attempt["method"]
        and classification.get("micro_batch") == attempt["micro_batch"]
        and classification.get("attempt_config_digest") == attempt["config_digest"]
        and classification.get("attempt_digest") == attempt["attempt_digest"]
    )


def _attempt_files(probe_dir: Path, method: str | None = None) -> list[Path]:
    pattern = "attempts/*/micro-batch-*/attempt-*/evidence/attempt.json"
    paths = list(probe_dir.glob(pattern))
    attempts: list[tuple[int, Path]] = []
    for path in paths:
        attempt = _load_probe_attempt(path)
        if method is None or attempt["method"] == method:
            attempts.append((attempt["recorded_at_ns"], path))
    attempts.sort(key=lambda item: item[0])
    return [path for _, path in attempts]


def _validated_capacity_chain(
    attempts: Sequence[tuple[Path, dict[str, object]]], *, require_success: bool
) -> tuple[tuple[Path, dict[str, object]] | None, tuple[Path, dict[str, object]] | None]:
    if not attempts:
        if require_success:
            raise ValueError("capacity chain contains no attempts")
        return None, None
    batches = [attempt["micro_batch"] for _, attempt in attempts]
    if len(set(batches)) != len(batches):
        raise ValueError("capacity chain contains a duplicate candidate")
    timestamps = [attempt["recorded_at_ns"] for _, attempt in attempts]
    if len(set(timestamps)) != len(timestamps) or timestamps != sorted(timestamps):
        raise ValueError("capacity chain does not have one strict recorded order")
    if batches[0] != 8:
        raise ValueError("capacity chain must start at batch 8")
    fingerprint_names = (
        "method", "world_size", "gradient_accumulation_steps", "learning_rate",
        "seed", "model", "data_path", "train_data_path", "data_stat", "image_root",
        "deepspeed_config", "deepspeed_config_digest", "attention_implementation",
        "probe_rows", "probe_row_indices", "audit_digest", "tuning_config",
        "prerequisite_chain", "prerequisite_chain_digest",
    )
    first_config = attempts[0][1]["launch_config"]
    for _, attempt in attempts[1:]:
        for name in fingerprint_names:
            if attempt["launch_config"][name] != first_config[name]:
                raise ValueError(f"capacity chain changes immutable field: {name}")
    selected = None
    first_failure = None
    if attempts[0][1]["exit_code"] == 0:
        expected = 8
        for index, item in enumerate(attempts):
            _, attempt = item
            if attempt["micro_batch"] != expected:
                raise ValueError("ascending capacity chain skips or reorders a candidate")
            if attempt["exit_code"] != 0:
                first_failure = item
                if index != len(attempts) - 1:
                    raise ValueError("capacity chain launches after its first failure")
            else:
                if first_failure is not None:
                    raise ValueError("capacity chain records success after first failure")
                selected = item
            expected *= 2
    else:
        expected_order = (8, 4, 2, 1)
        for index, item in enumerate(attempts):
            _, attempt = item
            if index >= len(expected_order) or attempt["micro_batch"] != expected_order[index]:
                raise ValueError("fallback capacity chain skips or reorders a candidate")
            if index == 0:
                first_failure = item
            if attempt["exit_code"] == 0:
                selected = item
                if index != len(attempts) - 1:
                    raise ValueError("fallback capacity chain launches after success")
            elif selected is not None:
                raise ValueError("fallback capacity chain launches after success")
    if require_success and selected is None:
        raise ValueError("capacity chain contains no successful candidate")
    return selected, first_failure


def _require_candidate(probe_dir: Path, method: str, candidate: int) -> None:
    method = _validated_method(method)
    candidate = _require_int("candidate", candidate, positive=True)
    attempts = [
        (path, _load_probe_attempt(path)) for path in _attempt_files(probe_dir, method)
    ]
    selected, first_failure = _validated_capacity_chain(attempts, require_success=False)
    if not attempts:
        if candidate != 8:
            raise ValueError("first capacity candidate must be 8")
        return
    last_path, last = attempts[-1]
    if last["exit_code"] != 0:
        if not _confirmed_capacity_oom(last_path):
            raise ValueError("next candidate requires classification of the prior failure")
        if batches := [item[1]["micro_batch"] for item in attempts]:
            expected = {8: 4, 4: 2, 2: 1}.get(batches[-1])
        if batches[0] != 8 or attempts[0][1]["exit_code"] == 0 or candidate != expected:
            raise ValueError("no launch is allowed after the first ascending failure")
        return
    if attempts[0][1]["exit_code"] != 0:
        raise ValueError("fallback search stops at its first success")
    expected = last["micro_batch"] * 2
    if candidate != expected:
        raise ValueError(f"next ascending candidate must be {expected}")


def _require_initial_oom(probe_dir: Path, method: str) -> Path:
    method = _validated_method(method)
    matches = []
    for path in _attempt_files(probe_dir, method):
        attempt = _load_probe_attempt(path)
        if attempt["micro_batch"] == 8 and attempt["exit_code"] != 0:
            matches.append(path)
    if len(matches) != 1 or not _confirmed_capacity_oom(matches[0]):
        raise ValueError(
            "fallback requires exactly one initial batch 8 failure with a confirmed "
            "capacity_oom classification"
        )
    return matches[0]


def _require_fallback(probe_dir: Path, method: str, candidate: int) -> None:
    candidate = _require_int("fallback", candidate, positive=True)
    if candidate not in {4, 2, 1}:
        raise ValueError("fallback must be 4, 2, or 1")
    required_failures = {4: (8,), 2: (8, 4), 1: (8, 4, 2)}[candidate]
    attempts = [
        (path, _load_probe_attempt(path)) for path in _attempt_files(probe_dir, method)
    ]
    for required_batch in required_failures:
        matches = [
            (path, attempt)
            for path, attempt in attempts
            if attempt["micro_batch"] == required_batch
        ]
        if len(matches) != 1:
            raise ValueError(
                f"fallback {candidate} requires exactly one prior batch "
                f"{required_batch} attempt"
            )
        path, attempt = matches[0]
        if attempt["exit_code"] == 0 or not _confirmed_capacity_oom(path):
            raise ValueError(
                f"fallback {candidate} requires batch {required_batch} to be a "
                "confirmed capacity OOM"
            )


def _validated_method(value: object) -> str:
    if type(value) is not str or value not in {"full", "dora"}:
        raise ValueError("method must be full or dora")
    return value


def _load_evidence_without_expected(path: Path) -> Stage1BenchmarkEvidence:
    if not path.is_file():
        raise FileNotFoundError(path)
    evidence, _ = _load_producer_evidence(path)
    return evidence


def _check_evidence(args: argparse.Namespace) -> str:
    path = Path(args.evidence_dir) / "evidence.json"
    if not path.is_file():
        return "missing"
    evidence, seal = _load_producer_evidence(path)
    if seal["run_config"]["kind"] != "stage1-smoke-run-manifest":
        raise ValueError("check-evidence accepts only Task 15 smoke evidence")
    _validate_smoke_resume_witness(seal["smoke_resume_witness"])
    if (
        evidence.status == "success"
        and evidence.method == args.method
        and evidence.micro_batch == args.micro_batch
        and evidence.world_size == args.world_size
    ):
        return "reusable-success"
    return "missing"


def _build_smoke_prerequisite_chain(
    loaded: Sequence[tuple[Stage1BenchmarkEvidence, dict[str, object]]],
) -> dict[str, object]:
    evidences = [item[0] for item in loaded]
    if len(evidences) != 2 or {item.method for item in evidences} != {"full", "dora"}:
        raise ValueError("required evidence must be an exact full/DoRA pair")
    by_method = {item[0].method: item for item in loaded}
    for method, (evidence, seal) in by_method.items():
        if seal["run_config"].get("kind") != "stage1-smoke-run-manifest":
            raise ValueError("required evidence must be Task 15 smoke evidence")
        _validate_smoke_resume_witness(seal.get("smoke_resume_witness"))
        if evidence.micro_batch != 2 or evidence.world_size != 2:
            raise ValueError("smoke prerequisite must use two ranks at micro-batch 2")
    identity_names = (
        "model", "source_data_path", "data_stat", "image_root", "world_size",
        "gradient_accumulation_steps", "epochs", "learning_rate", "seed",
        "deepspeed_config", "deepspeed_config_digest", "attention_implementation",
        "save_strategy", "audit_digest", "probe_row_indices",
    )
    full_config = by_method["full"][1]["run_config"]
    dora_config = by_method["dora"][1]["run_config"]
    shared_identity = {}
    for name in identity_names:
        if full_config.get(name) != dora_config.get(name):
            raise ValueError(f"smoke prerequisite pair changes controlled field: {name}")
        shared_identity[name] = full_config[name]
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stage1-smoke-prerequisite-chain",
        "shared_identity": shared_identity,
        "shared_identity_digest": config_digest(shared_identity),
        "producer_digests": {
            method: by_method[method][1]["producer_digest"]
            for method in ("full", "dora")
        },
        "resume_witness_digests": {
            method: by_method[method][1]["smoke_resume_witness_digest"]
            for method in ("full", "dora")
        },
    }
    payload["prerequisite_chain_digest"] = config_digest(payload)
    return payload


def _require_evidence(paths: Sequence[str]) -> dict[str, object]:
    loaded = []
    for raw_path in paths:
        evidence, seal = _load_producer_evidence(Path(raw_path))
        if evidence.status != "success":
            raise ValueError(f"required evidence is not successful: {raw_path}")
        loaded.append((evidence, seal))
    payload = _build_smoke_prerequisite_chain(loaded)
    if all(Path(path).is_file() for path in paths):
        common = Path(os.path.commonpath([str(Path(path).resolve()) for path in paths]))
        while common.is_file() or common.name in {"full", "dora"}:
            common = common.parent
        chain_path = common / "prerequisite-chain.json"
        if chain_path.exists() and _read_json_once(chain_path) != payload:
            raise FileExistsError(f"different smoke prerequisite chain exists: {chain_path}")
        if not chain_path.exists():
            _atomic_write_json(chain_path, payload)
    return payload


def _summaries(paths: Sequence[str]) -> list[dict[str, object]]:
    summaries = []
    for raw_path in paths:
        evidence = _load_evidence_without_expected(Path(raw_path))
        summaries.append(
            {
                "path": str(Path(raw_path)),
                "status": evidence.status,
                "method": evidence.method,
                "micro_batch": evidence.micro_batch,
                "world_size": evidence.world_size,
                "effective_global_batch": evidence.effective_global_batch,
                "optimizer_steps": evidence.optimizer_steps,
                "final_loss": evidence.final_loss,
            }
        )
    return summaries


def _load_maximum(path: Path) -> dict[str, object]:
    payload = _read_json_once(path)
    if type(payload) is not dict or payload.get("kind") != "stage1-capacity-maximum":
        raise ValueError(f"invalid Stage 1 maximum manifest: {path}")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "kind",
            "status",
            "method",
            "selected_micro_batch",
            "world_size",
            "gradient_accumulation_steps",
            "effective_global_batch",
            "long_smoke_steps",
            "long_smoke_attempt",
            "selected_attempt",
            "first_failure",
            "launch_config",
            "attempt_chain_digest",
            "audit_digest",
            "data_stat",
            "probe_row_indices",
            "prerequisite_chain_digest",
            "long_smoke_producer_digest",
            "rank_peak_allocated_bytes",
            "rank_peak_reserved_bytes",
            "config_digest",
        },
        "Stage 1 maximum manifest",
    )
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported maximum manifest schema")
    if payload.get("status") != "success":
        raise ValueError("maximum manifest is not successful")
    _validated_method(payload.get("method"))
    for name in (
        "selected_micro_batch",
        "world_size",
        "gradient_accumulation_steps",
        "effective_global_batch",
        "long_smoke_steps",
    ):
        _require_int(name, payload.get(name), positive=True)
    if payload["effective_global_batch"] != (
        payload["selected_micro_batch"]
        * payload["world_size"]
        * payload["gradient_accumulation_steps"]
    ):
        raise ValueError("maximum effective_global_batch invariant failed")
    for name in ("long_smoke_attempt", "selected_attempt"):
        _require_string(name, payload.get(name))
    if payload.get("first_failure") is not None and type(payload["first_failure"]) is not dict:
        raise TypeError("maximum first_failure must be null or an object")
    _validate_probe_launch_config(payload.get("launch_config"))
    for name in (
        "attempt_chain_digest", "audit_digest", "prerequisite_chain_digest",
        "long_smoke_producer_digest"
    ):
        _require_digest(name, payload.get(name))
    if payload["audit_digest"] != payload["launch_config"]["audit_digest"]:
        raise ValueError("maximum audit digest is not bound to launch config")
    if payload["data_stat"] != payload["launch_config"]["data_stat"]:
        raise ValueError("maximum data stat is not bound to launch config")
    if payload["probe_row_indices"] != payload["launch_config"]["probe_row_indices"]:
        raise ValueError("maximum probe row indices are not bound to launch config")
    if (
        payload["prerequisite_chain_digest"]
        != payload["launch_config"]["prerequisite_chain_digest"]
    ):
        raise ValueError("maximum prerequisite chain is not bound to launch config")
    for name in ("rank_peak_allocated_bytes", "rank_peak_reserved_bytes"):
        values = payload.get(name)
        if type(values) is not list or len(values) != payload["world_size"]:
            raise ValueError(f"maximum {name} must contain one value per rank")
        for index, value in enumerate(values):
            _require_int(f"{name}[{index}]", value)
    _require_digest("config_digest", payload.get("config_digest"))
    digest_input = dict(payload)
    digest_input.pop("config_digest")
    if config_digest(digest_input) != payload["config_digest"]:
        raise ValueError("maximum config digest mismatch")
    selected_path = Path(payload["selected_attempt"])
    selected = _load_probe_attempt(selected_path)
    long_attempt = _load_probe_attempt(Path(payload["long_smoke_attempt"]))
    if len(selected_path.parents) < 6 or selected_path.parents[4].name != "attempts":
        raise ValueError("maximum selected attempt is outside a probe attempt tree")
    probe_root = selected_path.parents[5]
    chain = [
        (path, _load_probe_attempt(path))
        for path in _attempt_files(probe_root, payload["method"])
    ]
    chain_selected, chain_failure = _validated_capacity_chain(
        chain, require_success=True
    )
    assert chain_selected is not None
    for attempt_path, attempt in chain:
        if attempt["exit_code"] != 0 and not _confirmed_capacity_oom(attempt_path):
            raise ValueError("maximum chain contains an unclassified failure")
    if config_digest(
        {"attempt_digests": [attempt["attempt_digest"] for _, attempt in chain]}
    ) != payload["attempt_chain_digest"]:
        raise ValueError("maximum attempt chain digest mismatch")
    if chain_selected[0].resolve() != selected_path.resolve():
        raise ValueError("maximum is not the last valid success in its ordered chain")
    expected_failure = None
    if chain_failure is not None:
        expected_failure = {
            "micro_batch": chain_failure[1]["micro_batch"],
            "classification": "capacity_oom",
            "attempt_config_digest": chain_failure[1]["config_digest"],
            "attempt_path": str(chain_failure[0]),
        }
    if payload["first_failure"] != expected_failure:
        raise ValueError("maximum first failure does not match its ordered chain")
    if (
        selected["method"] != payload["method"]
        or selected["micro_batch"] != payload["selected_micro_batch"]
        or long_attempt["micro_batch"] != payload["selected_micro_batch"]
        or long_attempt["max_steps"] != payload["long_smoke_steps"]
        or long_attempt["exit_code"] != 0
        or long_attempt["prerequisite_chain_digest"]
        != payload["prerequisite_chain_digest"]
    ):
        raise ValueError("maximum attempt references do not match its selection")
    producer_path = Path(long_attempt["output_dir"]).parent / "evidence" / "producer-evidence.json"
    _, producer = _load_producer_evidence(producer_path)
    if producer["producer_digest"] != payload["long_smoke_producer_digest"]:
        raise ValueError("maximum long-smoke producer digest mismatch")
    evidence = _evidence_from_dict(producer["benchmark_evidence"])
    if (
        list(evidence.rank_peak_allocated_bytes) != payload["rank_peak_allocated_bytes"]
        or list(evidence.rank_peak_reserved_bytes) != payload["rank_peak_reserved_bytes"]
    ):
        raise ValueError("maximum rank memory witness mismatch")
    return payload


def _rewrite_command_flag(command: list[str], flag: str, value: str) -> None:
    if flag not in command:
        command.extend((flag, value))
        return
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"recorded command has no value for {flag}")
    command[index + 1] = value


def _validate_maximum(probe_dir: Path, long_smoke_steps: int) -> dict[str, object]:
    long_smoke_steps = _require_int("long_smoke_steps", long_smoke_steps, positive=True)
    maximum_path = probe_dir / "maximum.json"
    if maximum_path.is_file():
        existing = _load_maximum(maximum_path)
        if existing["long_smoke_steps"] != long_smoke_steps:
            raise ValueError("existing maximum uses a different long-smoke length")
        current = [
            _load_probe_attempt(path)["attempt_digest"]
            for path in _attempt_files(probe_dir, existing["method"])
        ]
        if config_digest({"attempt_digests": current}) != existing["attempt_chain_digest"]:
            raise ValueError("attempt chain changed after maximum was sealed")
        return existing
    attempt_paths = _attempt_files(probe_dir)
    if not attempt_paths:
        raise ValueError("probe directory contains no attempts")
    attempts = [(path, _load_probe_attempt(path)) for path in attempt_paths]
    methods = {attempt["method"] for _, attempt in attempts}
    if len(methods) != 1:
        raise ValueError("probe directory mixes full and DoRA attempts")
    selected_item, first_failure_item = _validated_capacity_chain(
        attempts, require_success=True
    )
    assert selected_item is not None
    failed = [(path, attempt) for path, attempt in attempts if attempt["exit_code"] != 0]
    for path, _ in failed:
        if not _confirmed_capacity_oom(path):
            raise ValueError(f"requires-classification: {path}")
    selected_path, selected = selected_item
    long_root = (
        probe_dir
        / "long-smoke"
        / selected["method"]
        / f"micro-batch-{selected['micro_batch']}"
    )
    existing_long = list(long_root.glob("attempt-*/evidence/attempt.json"))
    if len(existing_long) > 1:
        raise ValueError("selected maximum has repeated long-smoke evidence")
    if existing_long:
        long_attempt = _load_probe_attempt(existing_long[0])
        if long_attempt["max_steps"] != long_smoke_steps or long_attempt["exit_code"] != 0:
            raise ValueError("existing long smoke is not a matching success")
    else:
        attempt_id = f"{time.time_ns()}-{os.getpid()}"
        long_dir = long_root / f"attempt-{attempt_id}"
        output_dir = long_dir / "output"
        logs_dir = long_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=False)
        command = list(selected["command"])
        _rewrite_command_flag(command, "--max-steps", str(long_smoke_steps))
        _rewrite_command_flag(command, "--output-dir", str(output_dir))
        _rewrite_command_flag(command, "--log-file", str(logs_dir / "train.log"))
        command = [item for item in command if item != "--final-save"]
        if "--no-final-save" not in command:
            command.append("--no-final-save")
        run_config_path = long_dir / "evidence" / "run-config.json"
        producer_path = long_dir / "evidence" / "producer-evidence.json"
        _prepare_probe_run(
            argparse.Namespace(
                run_config=str(run_config_path),
                method=selected["method"],
                model=selected["launch_config"]["model"],
                source_data_path=selected["launch_config"]["data_path"],
                data_path=selected["launch_config"]["train_data_path"],
                image_root=selected["launch_config"]["image_root"],
                output_dir=str(output_dir),
                micro_batch=selected["micro_batch"],
                world_size=selected["world_size"],
                gradient_accumulation=selected["gradient_accumulation_steps"],
                max_steps=long_smoke_steps,
                deepspeed_config=selected["launch_config"]["deepspeed_config"],
                probe_rows=selected["probe_rows"],
                prerequisite_chain=selected["launch_config"]["prerequisite_chain"],
            )
        )
        _rewrite_command_flag(command, "--benchmark-run-config", str(run_config_path))
        _rewrite_command_flag(command, "--producer-evidence", str(producer_path))
        stdout_path = logs_dir / "stdout.log"
        stderr_path = logs_dir / "stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle)
            exit_code = process.wait()
        namespace = argparse.Namespace(
            attempt_dir=str(long_dir),
            attempt_id=attempt_id,
            method=selected["method"],
            micro_batch=selected["micro_batch"],
            world_size=selected["world_size"],
            gradient_accumulation=selected["gradient_accumulation_steps"],
            learning_rate=selected["launch_config"]["learning_rate"],
            seed=selected["launch_config"]["seed"],
            model=selected["launch_config"]["model"],
            source_data_path=selected["launch_config"]["data_path"],
            image_root=selected["launch_config"]["image_root"],
            deepspeed_config=selected["launch_config"]["deepspeed_config"],
            attention_implementation=selected["launch_config"][
                "attention_implementation"
            ],
            process_id=process.pid,
            exit_code=exit_code,
            max_steps=long_smoke_steps,
            output_dir=str(output_dir),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            probe_rows=selected["probe_rows"],
            command=["--", *command],
        )
        long_attempt = _record_probe_attempt(namespace)
        if exit_code != 0:
            raise RuntimeError(
                f"long smoke failed and requires-classification: {long_dir}"
            )
    long_producer_path = (
        Path(long_attempt["output_dir"]).parent / "evidence" / "producer-evidence.json"
    )
    if not long_producer_path.is_file():
        raise FileNotFoundError(
            f"long smoke is missing rank-local producer evidence: {long_producer_path}"
        )
    long_evidence, long_seal = _load_producer_evidence(long_producer_path)
    if (
        long_evidence.status != "success"
        or long_evidence.method != selected["method"]
        or long_evidence.micro_batch != selected["micro_batch"]
        or long_evidence.optimizer_steps != long_smoke_steps
        or long_evidence.config_digest != long_attempt["launch_config"]["run_config_digest"]
    ):
        raise ValueError("long-smoke producer evidence does not match selected maximum")
    first_failure = None
    if first_failure_item is not None:
        first_path, first = first_failure_item
        first_failure = {
            "micro_batch": first["micro_batch"],
            "classification": "capacity_oom",
            "attempt_config_digest": first["config_digest"],
            "attempt_path": str(first_path),
        }
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stage1-capacity-maximum",
        "status": "success",
        "method": selected["method"],
        "selected_micro_batch": selected["micro_batch"],
        "world_size": selected["world_size"],
        "gradient_accumulation_steps": selected["gradient_accumulation_steps"],
        "effective_global_batch": selected["effective_global_batch"],
        "long_smoke_steps": long_smoke_steps,
        "long_smoke_attempt": str(_probe_attempt_path(Path(long_attempt["output_dir"]).parent)),
        "selected_attempt": str(selected_path),
        "first_failure": first_failure,
        "launch_config": selected["launch_config"],
        "attempt_chain_digest": config_digest(
            {"attempt_digests": [attempt["attempt_digest"] for _, attempt in attempts]}
        ),
        "audit_digest": selected["audit_digest"],
        "data_stat": selected["data_stat"],
        "probe_row_indices": selected["probe_row_indices"],
        "prerequisite_chain_digest": selected["prerequisite_chain_digest"],
        "long_smoke_producer_digest": long_seal["producer_digest"],
        "rank_peak_allocated_bytes": list(long_evidence.rank_peak_allocated_bytes),
        "rank_peak_reserved_bytes": list(long_evidence.rank_peak_reserved_bytes),
    }
    payload["config_digest"] = config_digest(payload)
    _atomic_write_json(maximum_path, payload)
    return payload


def _require_matching_maximum_controls(
    maxima: Sequence[Mapping[str, object]],
) -> None:
    configs = [item["launch_config"] for item in maxima]
    for name in (
        "model", "data_path", "data_stat", "image_root", "world_size",
        "gradient_accumulation_steps", "learning_rate", "seed",
        "deepspeed_config", "deepspeed_config_digest", "attention_implementation",
        "probe_rows", "probe_row_indices", "audit_digest",
        "prerequisite_chain_digest",
    ):
        if configs[0][name] != configs[1][name]:
            raise ValueError(f"maximum comparison changes controlled field: {name}")


def _compare_maxima(paths: Sequence[str]) -> dict[str, object]:
    maxima = [_load_maximum(Path(path)) for path in paths]
    if {item["method"] for item in maxima} != {"full", "dora"}:
        raise ValueError("maximum comparison requires separated full and DoRA manifests")
    prerequisite_digests = {
        item["prerequisite_chain_digest"] for item in maxima
    }
    if len(prerequisite_digests) != 1:
        raise ValueError("maximum comparison changes the smoke prerequisite chain")
    _require_matching_maximum_controls(maxima)
    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stage1-capacity-comparison",
        "prerequisite_chain_digest": next(iter(prerequisite_digests)),
        "methods": {
            item["method"]: {
                "micro_batch": item["selected_micro_batch"],
                "effective_global_batch": item["effective_global_batch"],
                "long_smoke_steps": item["long_smoke_steps"],
                "first_failure": item["first_failure"],
                "rank_peak_allocated_bytes": item["rank_peak_allocated_bytes"],
                "rank_peak_reserved_bytes": item["rank_peak_reserved_bytes"],
                "maximum_config_digest": item["config_digest"],
                "prerequisite_chain_digest": item["prerequisite_chain_digest"],
            }
            for item in maxima
        },
    }


def _prepare_timed_run(args: argparse.Namespace) -> str:
    method = _validated_method(args.method)
    maximum = _load_maximum(Path(args.maximum))
    if maximum["method"] != method:
        raise ValueError("timed method does not match maximum manifest")
    if args.epochs != 3 or args.learning_rate != 1e-5:
        raise ValueError("timed runs require exactly 3 epochs and learning rate 1e-5")
    if args.gradient_accumulation != 2 or args.world_size != 2:
        raise ValueError("timed runs require gradient accumulation 2 and world size 2")
    source_config = maximum["launch_config"]
    output_dir = Path(args.output_dir).resolve()
    launch_config = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stage1-timed-launch-manifest",
        "method": method,
        "tuning_config": source_config["tuning_config"],
        "model": source_config["model"],
        "source_data_path": source_config["data_path"],
        "data_path": source_config["data_path"],
        "data_stat": source_config["data_stat"],
        "image_root": source_config["image_root"],
        "output_dir": str(output_dir),
        "micro_batch": maximum["selected_micro_batch"],
        "world_size": args.world_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "effective_global_batch": (
            maximum["selected_micro_batch"]
            * args.world_size
            * args.gradient_accumulation
        ),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": 42,
        "max_steps": 0,
        "attention_implementation": "flash_attention_2",
        "deepspeed_config": source_config["deepspeed_config"],
        "deepspeed_config_digest": source_config["deepspeed_config_digest"],
        "save_strategy": "no",
        "final_save": True,
        "audit_digest": source_config["audit_digest"],
        "probe_rows": source_config["probe_rows"],
        "probe_row_indices": source_config["probe_row_indices"],
        "maximum_config_digest": maximum["config_digest"],
        "prerequisite_chain": source_config["prerequisite_chain"],
        "prerequisite_chain_digest": maximum["prerequisite_chain_digest"],
    }
    launch_config["config_digest"] = config_digest(launch_config)
    manifest_path = output_dir / "launch-manifest.json"
    evidence_path = output_dir / "evidence.json"
    if evidence_path.is_file():
        evidence, seal = _load_producer_evidence(evidence_path)
        if (
            evidence.config_digest == launch_config["config_digest"]
            and seal["run_config"] == launch_config
            and evidence.status == "success"
        ):
            return "reusable-success"
        raise ValueError("timed output contains stale or malformed evidence")
    if output_dir.exists():
        entries = list(output_dir.iterdir())
        if entries and entries != [manifest_path]:
            raise FileExistsError(
                f"refusing incomplete or colliding timed output: {output_dir}"
            )
        if manifest_path.is_file():
            if _read_json_once(manifest_path) != launch_config:
                raise ValueError("existing timed launch manifest does not match")
            return str(manifest_path)
    _atomic_write_json(manifest_path, launch_config)
    return str(manifest_path)


def _launcher_args_from_manifest(path: Path) -> list[str]:
    payload = _load_timed_manifest(path)

    result = [
        "--method", str(payload["method"]),
        "--model", str(payload["model"]),
        "--source-data-path", str(payload["source_data_path"]),
        "--data-path", str(payload["data_path"]),
        "--image-root", str(payload["image_root"]),
        "--output-dir", str(payload["output_dir"]),
        "--micro-batch", str(payload["micro_batch"]),
        "--epochs", str(payload["epochs"]),
        "--learning-rate", str(payload["learning_rate"]),
        "--gradient-accumulation", str(payload["gradient_accumulation_steps"]),
        "--seed", str(payload["seed"]),
        "--deepspeed-config", str(payload["deepspeed_config"]),
        "--attention-implementation", str(payload["attention_implementation"]),
        "--save-strategy", str(payload["save_strategy"]),
        "--final-save",
        "--benchmark-run-config", str(path.resolve()),
        "--producer-evidence", str(Path(payload["output_dir"]) / "producer-evidence.json"),
    ]
    return result


def _load_timed_manifest(path: Path) -> dict[str, object]:
    payload = _load_producer_run_config(path)
    if payload["kind"] != "stage1-timed-launch-manifest":
        raise ValueError("launcher manifest has the wrong kind")
    if payload["final_save"] is not True or payload["maximum_config_digest"] is None:
        raise ValueError("timed manifest must bind a maximum and final save")
    return payload


def _validate_timed_run(run_dir: Path) -> str:
    manifest_path = run_dir / "launch-manifest.json"
    manifest = _load_timed_manifest(manifest_path)
    claimed = manifest["config_digest"]
    evidence_path = run_dir / "evidence.json"
    if evidence_path.is_file():
        reusable, reusable_seal = _load_producer_evidence(evidence_path)
        if (
            reusable.config_digest != claimed
            or reusable_seal["run_config"] != manifest
            or reusable.status != "success"
        ):
            raise ValueError("timed evidence is not successful")
        return "reusable-success"
    producer_path = run_dir / "producer-evidence.json"
    if not producer_path.is_file():
        raise FileNotFoundError(
            f"missing matching small producer evidence: {producer_path}"
        )
    producer, producer_seal = _load_producer_evidence(producer_path)
    if producer.config_digest != claimed:
        raise ValueError("producer evidence is not bound to the timed manifest")
    if producer_seal["run_config"] != manifest:
        raise ValueError("producer run config does not exactly match timed manifest")
    if producer.status != "success" or producer.method != manifest["method"]:
        raise ValueError("producer evidence does not match the successful timed run")
    if (
        producer.micro_batch != manifest["micro_batch"]
        or producer.world_size != manifest["world_size"]
        or producer.gradient_accumulation_steps
        != manifest["gradient_accumulation_steps"]
    ):
        raise ValueError("producer evidence batch configuration mismatch")
    expected_samples = 3450 * manifest["epochs"]
    expected_steps = math.ceil(3450 / manifest["effective_global_batch"]) * manifest[
        "epochs"
    ]
    if producer.samples_seen != expected_samples or producer.optimizer_steps != expected_steps:
        raise ValueError("producer evidence sample/optimizer-step invariant mismatch")
    _atomic_write_json(evidence_path, producer_seal)
    return str(evidence_path)


def _compare_runs(paths: Sequence[str], output: Path) -> dict[str, object]:
    loaded = [_load_producer_evidence(Path(path)) for path in paths]
    evidences = [item[0] for item in loaded]
    if {item.method for item in evidences} != {"full", "dora"}:
        raise ValueError("run comparison requires separated full and DoRA evidence")
    if any(item.status != "success" for item in evidences):
        raise ValueError("run comparison requires successful sealed evidence")
    configs = [item[1]["run_config"] for item in loaded]
    for name in (
        "model", "source_data_path", "data_path", "data_stat", "image_root", "world_size",
        "gradient_accumulation_steps", "epochs", "learning_rate", "seed",
        "deepspeed_config", "deepspeed_config_digest", "attention_implementation",
        "save_strategy", "audit_digest", "probe_row_indices",
        "prerequisite_chain_digest",
    ):
        if configs[0][name] != configs[1][name]:
            raise ValueError(f"run comparison changes controlled field: {name}")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stage1-maximum-throughput-comparison",
        "paper_level_reproduction": False,
        "common_batch_control": False,
        "methods": {
            item.method: _evidence_to_dict(item) for item in evidences
        },
    }
    _atomic_write_json(output, payload)
    return payload


def _final_report(args: argparse.Namespace) -> dict[str, object]:
    smoke_paths = [Path(args.smoke_full), Path(args.smoke_dora)]
    timed_paths = [Path(args.timed_full), Path(args.timed_dora)]
    smoke_loaded = [_load_producer_evidence(path) for path in smoke_paths]
    timed_loaded = [_load_producer_evidence(path) for path in timed_paths]
    smoke = [item[0] for item in smoke_loaded]
    timed = [item[0] for item in timed_loaded]
    maximum = [_load_maximum(Path(args.maximum_full)), _load_maximum(Path(args.maximum_dora))]
    for label, values in (("smoke", smoke), ("timed", timed)):
        if {item.method for item in values} != {"full", "dora"}:
            raise ValueError(f"{label} evidence must contain separated full and DoRA methods")
        if any(item.status != "success" for item in values):
            raise ValueError(f"{label} evidence must be successful")
    if {item["method"] for item in maximum} != {"full", "dora"}:
        raise ValueError("maximum manifests must contain separated full and DoRA methods")
    prerequisite = _build_smoke_prerequisite_chain(smoke_loaded)
    prerequisite_digest = prerequisite["prerequisite_chain_digest"]
    if any(
        item["prerequisite_chain_digest"] != prerequisite_digest
        for item in maximum
    ):
        raise ValueError("maximum evidence does not descend from the smoke prerequisite")
    _require_matching_maximum_controls(maximum)
    for item in maximum:
        _validate_prerequisite_controls(item["launch_config"], prerequisite)
    maximum_by_method = {item["method"]: item for item in maximum}
    for item in timed:
        selected = maximum_by_method[item.method]
        if (
            item.micro_batch != selected["selected_micro_batch"]
            or item.effective_global_batch != selected["effective_global_batch"]
        ):
            raise ValueError(
                f"timed {item.method} evidence does not use its validated maximum"
            )
    for path, item, loaded in zip(timed_paths, timed, timed_loaded):
        manifest = _load_timed_manifest(path.parent / "launch-manifest.json")
        if manifest["config_digest"] != item.config_digest:
            raise ValueError(f"timed evidence config digest mismatch: {path}")
        if loaded[1]["run_config"] != manifest:
            raise ValueError(f"timed evidence run config mismatch: {path}")
        selected = maximum_by_method[item.method]
        if loaded[1]["run_config"]["maximum_config_digest"] != selected["config_digest"]:
            raise ValueError(f"timed evidence maximum digest mismatch: {path}")
        if (
            loaded[1]["run_config"]["prerequisite_chain_digest"]
            != prerequisite_digest
        ):
            raise ValueError(f"timed evidence prerequisite chain mismatch: {path}")
        if (
            loaded[1]["run_config"]["audit_digest"] != selected["audit_digest"]
            or loaded[1]["run_config"]["data_stat"] != selected["data_stat"]
            or loaded[1]["run_config"]["probe_row_indices"] != selected["probe_row_indices"]
        ):
            raise ValueError(f"timed evidence audit chain mismatch: {path}")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stage1-final-evidence-report",
        "paper_level_reproduction": False,
        "prerequisite_chain_digest": prerequisite_digest,
        "smoke": {item.method: _evidence_to_dict(item) for item in smoke},
        "maximum": maximum_by_method,
        "timed": {item.method: _evidence_to_dict(item) for item in timed},
    }
    payload["config_digest"] = config_digest(payload)
    _atomic_write_json(Path(args.output), payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    candidates = subparsers.add_parser("capacity-candidates")
    candidates.add_argument("--start", type=int, default=8)
    candidates.add_argument("--max-per-gpu", type=int, required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--data-path", required=True)
    audit.add_argument("--image-root", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--count", type=int, required=True)

    probe_jsonl = subparsers.add_parser("probe-jsonl")
    probe_jsonl.add_argument("--audit", required=True)
    probe_jsonl.add_argument("--data-path", required=True)
    probe_jsonl.add_argument("--image-root", required=True)
    probe_jsonl.add_argument("--output", required=True)

    record = subparsers.add_parser("record-attempt")
    record.add_argument("--attempt-dir", required=True)
    record.add_argument("--attempt-id", required=True)
    record.add_argument("--method", required=True)
    record.add_argument("--micro-batch", type=int, required=True)
    record.add_argument("--world-size", type=int, default=2)
    record.add_argument("--gradient-accumulation", type=int, required=True)
    record.add_argument("--learning-rate", type=float, required=True)
    record.add_argument("--seed", type=int, required=True)
    record.add_argument("--model", required=True)
    record.add_argument("--source-data-path", required=True)
    record.add_argument("--image-root", required=True)
    record.add_argument("--deepspeed-config", required=True)
    record.add_argument("--attention-implementation", required=True)
    record.add_argument("--process-id", type=int, required=True)
    record.add_argument("--exit-code", type=int, required=True)
    record.add_argument("--max-steps", type=int, required=True)
    record.add_argument("--output-dir", required=True)
    record.add_argument("--stdout-path", required=True)
    record.add_argument("--stderr-path", required=True)
    record.add_argument("--probe-rows", required=True)
    record.add_argument("command", nargs=argparse.REMAINDER)

    probe_run = subparsers.add_parser("prepare-probe-run")
    probe_run.add_argument("--run-config", required=True)
    probe_run.add_argument("--method", required=True)
    probe_run.add_argument("--model", required=True)
    probe_run.add_argument("--source-data-path", required=True)
    probe_run.add_argument("--data-path", required=True)
    probe_run.add_argument("--image-root", required=True)
    probe_run.add_argument("--output-dir", required=True)
    probe_run.add_argument("--micro-batch", type=int, required=True)
    probe_run.add_argument("--world-size", type=int, default=2)
    probe_run.add_argument("--gradient-accumulation", type=int, required=True)
    probe_run.add_argument("--max-steps", type=int, required=True)
    probe_run.add_argument("--deepspeed-config", required=True)
    probe_run.add_argument("--probe-rows", required=True)
    probe_run.add_argument("--prerequisite-chain", required=True)

    classify = subparsers.add_parser("classify-failure")
    classify.add_argument("--attempt", required=True)
    classify.add_argument("--classification", choices=("capacity_oom",), required=True)

    require_initial = subparsers.add_parser("require-initial-oom")
    require_initial.add_argument("--probe-dir", required=True)
    require_initial.add_argument("--method", required=True)

    require_fallback = subparsers.add_parser("require-fallback")
    require_fallback.add_argument("--probe-dir", required=True)
    require_fallback.add_argument("--method", required=True)
    require_fallback.add_argument("--candidate", type=int, required=True)

    require_candidate = subparsers.add_parser("require-candidate")
    require_candidate.add_argument("--probe-dir", required=True)
    require_candidate.add_argument("--method", required=True)
    require_candidate.add_argument("--candidate", type=int, required=True)

    check = subparsers.add_parser("check-evidence")
    check.add_argument("--evidence-dir", required=True)
    check.add_argument("--method", choices=("full", "dora"), required=True)
    check.add_argument("--micro-batch", type=int, required=True)
    check.add_argument("--world-size", type=int, required=True)

    require = subparsers.add_parser("require-evidence")
    require.add_argument("paths", nargs="+")

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("paths", nargs="+")

    validate_maximum = subparsers.add_parser("validate-maximum")
    validate_maximum.add_argument("--probe-dir", required=True)
    validate_maximum.add_argument("--long-smoke-steps", type=int, required=True)

    compare_maxima = subparsers.add_parser("compare-maxima")
    compare_maxima.add_argument("paths", nargs=2)

    prepare = subparsers.add_parser("prepare-timed-run")
    prepare.add_argument("--method", choices=("full", "dora"), required=True)
    prepare.add_argument("--maximum", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--epochs", type=int, required=True)
    prepare.add_argument("--learning-rate", type=float, required=True)
    prepare.add_argument("--gradient-accumulation", type=int, required=True)
    prepare.add_argument("--world-size", type=int, required=True)

    launcher_args = subparsers.add_parser("launcher-args")
    launcher_args.add_argument("--manifest", required=True)

    validate_timed = subparsers.add_parser("validate-timed-run")
    validate_timed.add_argument("--run-dir", required=True)

    compare_runs = subparsers.add_parser("compare-runs")
    compare_runs.add_argument("paths", nargs=2)
    compare_runs.add_argument("--output", required=True)

    final = subparsers.add_parser("final-report")
    final.add_argument("--smoke-full", required=True)
    final.add_argument("--smoke-dora", required=True)
    final.add_argument("--maximum-full", required=True)
    final.add_argument("--maximum-dora", required=True)
    final.add_argument("--timed-full", required=True)
    final.add_argument("--timed-dora", required=True)
    final.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    command = args.command_name
    if command == "capacity-candidates":
        print(" ".join(map(str, capacity_candidates(args.start, args.max_per_gpu))))
    elif command == "audit":
        selected = select_high_cost_probe_rows(
            args.data_path,
            args.count,
            args.output,
            image_root=args.image_root,
        )
        print(" ".join(map(str, selected)))
    elif command == "probe-jsonl":
        _atomic_write_probe_jsonl(
            Path(args.audit),
            Path(args.output),
            data_path=Path(args.data_path),
            image_root=Path(args.image_root),
        )
        print(args.output)
    elif command == "record-attempt":
        payload = _record_probe_attempt(args)
        print(_probe_attempt_path(Path(args.attempt_dir)))
        if payload["requires_oom_classification"]:
            print("requires-classification", file=sys.stderr)
    elif command == "prepare-probe-run":
        print(_prepare_probe_run(args))
    elif command == "classify-failure":
        _classify_failure(Path(args.attempt), args.classification)
        print("capacity_oom")
    elif command == "require-initial-oom":
        print(_require_initial_oom(Path(args.probe_dir), args.method))
    elif command == "require-fallback":
        _require_fallback(Path(args.probe_dir), args.method, args.candidate)
        print("confirmed-fallback")
    elif command == "require-candidate":
        _require_candidate(Path(args.probe_dir), args.method, args.candidate)
        print("confirmed-candidate")
    elif command == "check-evidence":
        print(_check_evidence(args))
    elif command == "require-evidence":
        _require_evidence(args.paths)
        print("reusable-success")
    elif command == "summarize":
        print(json.dumps(_summaries(args.paths), sort_keys=True))
    elif command == "validate-maximum":
        payload = _validate_maximum(Path(args.probe_dir), args.long_smoke_steps)
        print(json.dumps(payload, sort_keys=True))
    elif command == "compare-maxima":
        print(json.dumps(_compare_maxima(args.paths), sort_keys=True))
    elif command == "prepare-timed-run":
        print(_prepare_timed_run(args))
    elif command == "launcher-args":
        for item in _launcher_args_from_manifest(Path(args.manifest)):
            sys.stdout.buffer.write(item.encode("utf-8") + b"\0")
    elif command == "validate-timed-run":
        print(_validate_timed_run(Path(args.run_dir)))
    elif command == "compare-runs":
        print(json.dumps(_compare_runs(args.paths, Path(args.output)), sort_keys=True))
    elif command == "final-report":
        print(json.dumps(_final_report(args), sort_keys=True))
    else:  # pragma: no cover - argparse enforces the closed command set.
        raise AssertionError(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Stage1BenchmarkEvidence",
    "capacity_candidates",
    "config_digest",
    "fallback_candidates_after_initial_failure",
    "load_verified_evidence",
    "main",
    "select_high_cost_probe_rows",
    "write_benchmark_evidence",
]
