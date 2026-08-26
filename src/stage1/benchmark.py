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
        if value is None or type(value) in (str, bool, int):
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
        evidence = _evidence_from_dict(_read_json_once(resolved))
        _VERIFIED_EVIDENCE_CACHE[resolved] = (witness, evidence)
    if evidence.config_digest != expected_config_digest:
        return None
    return evidence


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
        },
        "high-cost row audit",
    )
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported high-cost row audit schema_version")
    if payload.get("kind") != "stage1-high-cost-row-audit":
        raise ValueError("invalid high-cost row audit kind")
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
                    continue
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
    _atomic_write_json(audit_path, payload)
    return tuple(item["source_row_index"] for item in selected_rows)


def _atomic_write_probe_jsonl(audit_path: Path, output_path: Path) -> None:
    payload = _read_json_once(audit_path)
    if type(payload) is not dict or payload.get("kind") != "stage1-high-cost-row-audit":
        raise ValueError("probe rows must be a stage1-high-cost-row-audit")
    selected = payload.get("selected_rows")
    if type(selected) is not list or not selected:
        raise ValueError("probe row audit has no selected rows")
    current_data_path = Path(payload.get("data_path", ""))
    if current_data_path.is_absolute():
        if not current_data_path.is_file():
            raise FileNotFoundError(f"audited data path is missing: {current_data_path}")
        if payload.get("input_stat") != _stat_payload(current_data_path):
            raise ValueError("probe row audit input stat witness is stale")
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
    if type(args.command) is not list or not args.command:
        raise ValueError("probe attempt requires a recorded command")
    command = args.command[1:] if args.command[0] == "--" else args.command
    if not command:
        raise ValueError("probe attempt command must not be empty")
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
        "model": _require_string("model", args.model),
        "data_path": str(Path(args.source_data_path).resolve()),
        "image_root": str(Path(args.image_root).resolve()),
        "deepspeed_config": str(Path(args.deepspeed_config).resolve()),
        "attention_implementation": _require_string(
            "attention_implementation", args.attention_implementation
        ),
        "probe_rows": str(Path(args.probe_rows).resolve()),
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
        "command": command,
        "launch_config": launch_config,
        "config_digest": digest,
    }
    attempt_dir = Path(args.attempt_dir)
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
    _atomic_write_json(_probe_attempt_path(attempt_dir), payload)
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
            "command",
            "launch_config",
            "config_digest",
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
    expected_status = "success" if payload["exit_code"] == 0 else "requires-classification"
    if payload.get("status") != expected_status:
        raise ValueError("probe attempt status does not match exit_code")
    if type(payload.get("requires_oom_classification")) is not bool:
        raise TypeError("requires_oom_classification must be a boolean")
    if type(payload.get("final_save")) is not bool:
        raise TypeError("final_save must be a boolean")
    _require_digest("config_digest", payload.get("config_digest"))
    if config_digest(payload.get("launch_config")) != payload["config_digest"]:
        raise ValueError("probe attempt config digest mismatch")
    _validate_probe_launch_config(payload["launch_config"])
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
    return payload


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
            "data_path",
            "image_root",
            "deepspeed_config",
            "attention_implementation",
            "probe_rows",
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
        "image_root",
        "deepspeed_config",
        "attention_implementation",
        "probe_rows",
    ):
        _require_string(name, payload.get(name))
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
    }
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
    return bool(
        type(classification) is dict
        and classification.get("schema_version") == _SCHEMA_VERSION
        and classification.get("kind") == "stage1-probe-failure-classification"
        and classification.get("classification") == "capacity_oom"
        and classification.get("method") == attempt["method"]
        and classification.get("micro_batch") == attempt["micro_batch"]
        and classification.get("attempt_config_digest") == attempt["config_digest"]
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
    return _evidence_from_dict(_read_json_once(path))


def _check_evidence(args: argparse.Namespace) -> str:
    path = Path(args.evidence_dir) / "evidence.json"
    if not path.is_file():
        return "missing"
    evidence = _load_evidence_without_expected(path)
    if (
        evidence.status == "success"
        and evidence.method == args.method
        and evidence.micro_batch == args.micro_batch
        and evidence.world_size == args.world_size
    ):
        return "reusable-success"
    return "missing"


def _require_evidence(paths: Sequence[str]) -> None:
    for raw_path in paths:
        evidence = _load_evidence_without_expected(Path(raw_path))
        if evidence.status != "success":
            raise ValueError(f"required evidence is not successful: {raw_path}")


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
    _require_digest("config_digest", payload.get("config_digest"))
    digest_input = dict(payload)
    digest_input.pop("config_digest")
    if config_digest(digest_input) != payload["config_digest"]:
        raise ValueError("maximum config digest mismatch")
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
        return existing
    attempt_paths = _attempt_files(probe_dir)
    if not attempt_paths:
        raise ValueError("probe directory contains no attempts")
    attempts = [(path, _load_probe_attempt(path)) for path in attempt_paths]
    methods = {attempt["method"] for _, attempt in attempts}
    if len(methods) != 1:
        raise ValueError("probe directory mixes full and DoRA attempts")
    failed = [(path, attempt) for path, attempt in attempts if attempt["exit_code"] != 0]
    for path, _ in failed:
        if not _confirmed_capacity_oom(path):
            raise ValueError(f"requires-classification: {path}")
    successful = [(path, attempt) for path, attempt in attempts if attempt["exit_code"] == 0]
    if not successful:
        raise ValueError("no successful capacity candidate is available")
    selected_path, selected = max(successful, key=lambda item: item[1]["micro_batch"])
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
    first_failure = None
    if failed:
        first_path, first = min(failed, key=lambda item: item[1]["recorded_at_ns"])
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
    }
    payload["config_digest"] = config_digest(payload)
    _atomic_write_json(maximum_path, payload)
    return payload


def _compare_maxima(paths: Sequence[str]) -> dict[str, object]:
    maxima = [_load_maximum(Path(path)) for path in paths]
    if {item["method"] for item in maxima} != {"full", "dora"}:
        raise ValueError("maximum comparison requires separated full and DoRA manifests")
    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stage1-capacity-comparison",
        "methods": {
            item["method"]: {
                "micro_batch": item["selected_micro_batch"],
                "effective_global_batch": item["effective_global_batch"],
                "long_smoke_steps": item["long_smoke_steps"],
                "first_failure": item["first_failure"],
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
        "model": source_config["model"],
        "data_path": source_config["data_path"],
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
        "attention_implementation": "flash_attention_2",
        "deepspeed_config": source_config["deepspeed_config"],
        "save_strategy": "no",
        "final_save": True,
        "maximum_config_digest": maximum["config_digest"],
    }
    launch_config["config_digest"] = config_digest(launch_config)
    manifest_path = output_dir / "launch-manifest.json"
    evidence_path = output_dir / "evidence.json"
    if evidence_path.is_file():
        evidence = load_verified_evidence(evidence_path, launch_config["config_digest"])
        if evidence is not None and evidence.status == "success":
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
    ]
    return result


def _load_timed_manifest(path: Path) -> dict[str, object]:
    payload = _read_json_once(path)
    if type(payload) is not dict or payload.get("kind") != "stage1-timed-launch-manifest":
        raise ValueError("launcher manifest has the wrong kind")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "kind",
            "method",
            "model",
            "data_path",
            "image_root",
            "output_dir",
            "micro_batch",
            "world_size",
            "gradient_accumulation_steps",
            "effective_global_batch",
            "epochs",
            "learning_rate",
            "seed",
            "attention_implementation",
            "deepspeed_config",
            "save_strategy",
            "final_save",
            "maximum_config_digest",
            "config_digest",
        },
        "Stage 1 timed launch manifest",
    )
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported timed launch manifest schema")
    _validated_method(payload.get("method"))
    for name in (
        "micro_batch",
        "world_size",
        "gradient_accumulation_steps",
        "effective_global_batch",
        "epochs",
    ):
        _require_int(name, payload.get(name), positive=True)
    _require_int("seed", payload.get("seed"))
    if type(payload.get("learning_rate")) is not float or not math.isfinite(
        payload["learning_rate"]
    ):
        raise TypeError("learning_rate must be a finite float")
    if payload["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    for name in (
        "model",
        "data_path",
        "image_root",
        "output_dir",
        "attention_implementation",
        "deepspeed_config",
        "save_strategy",
    ):
        _require_string(name, payload.get(name))
    if type(payload.get("final_save")) is not bool:
        raise TypeError("final_save must be a boolean")
    if payload["effective_global_batch"] != (
        payload["micro_batch"]
        * payload["world_size"]
        * payload["gradient_accumulation_steps"]
    ):
        raise ValueError("timed effective_global_batch invariant failed")
    if (
        payload["epochs"] != 3
        or payload["learning_rate"] != 1e-5
        or payload["world_size"] != 2
        or payload["gradient_accumulation_steps"] != 2
        or payload["seed"] != 42
        or payload["attention_implementation"] != "flash_attention_2"
        or payload["save_strategy"] != "no"
        or payload["final_save"] is not True
    ):
        raise ValueError("timed launch manifest violates approved experiment invariants")
    _require_digest("maximum_config_digest", payload.get("maximum_config_digest"))
    claimed = payload.get("config_digest")
    _require_digest("config_digest", claimed)
    digest_input = dict(payload)
    digest_input.pop("config_digest")
    if config_digest(digest_input) != claimed:
        raise ValueError("launcher manifest config digest mismatch")
    return payload


def _validate_timed_run(run_dir: Path) -> str:
    manifest_path = run_dir / "launch-manifest.json"
    manifest = _load_timed_manifest(manifest_path)
    claimed = manifest["config_digest"]
    evidence_path = run_dir / "evidence.json"
    reusable = load_verified_evidence(evidence_path, claimed)
    if reusable is not None:
        if reusable.status != "success":
            raise ValueError("timed evidence is not successful")
        return "reusable-success"
    producer_path = run_dir / "producer-evidence.json"
    producer = load_verified_evidence(producer_path, claimed)
    if producer is None:
        raise FileNotFoundError(
            f"missing matching small producer evidence: {producer_path}"
        )
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
    write_benchmark_evidence(evidence_path, producer)
    return str(evidence_path)


def _compare_runs(paths: Sequence[str], output: Path) -> dict[str, object]:
    evidences = [_load_evidence_without_expected(Path(path)) for path in paths]
    if {item.method for item in evidences} != {"full", "dora"}:
        raise ValueError("run comparison requires separated full and DoRA evidence")
    if any(item.status != "success" for item in evidences):
        raise ValueError("run comparison requires successful sealed evidence")
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
    smoke = [_load_evidence_without_expected(path) for path in smoke_paths]
    timed = [_load_evidence_without_expected(path) for path in timed_paths]
    maximum = [_load_maximum(Path(args.maximum_full)), _load_maximum(Path(args.maximum_dora))]
    for label, values in (("smoke", smoke), ("timed", timed)):
        if {item.method for item in values} != {"full", "dora"}:
            raise ValueError(f"{label} evidence must contain separated full and DoRA methods")
        if any(item.status != "success" for item in values):
            raise ValueError(f"{label} evidence must be successful")
    if {item["method"] for item in maximum} != {"full", "dora"}:
        raise ValueError("maximum manifests must contain separated full and DoRA methods")
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
    for path, item in zip(timed_paths, timed):
        manifest = _load_timed_manifest(path.parent / "launch-manifest.json")
        if manifest["config_digest"] != item.config_digest:
            raise ValueError(f"timed evidence config digest mismatch: {path}")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "stage1-final-evidence-report",
        "paper_level_reproduction": False,
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
        _atomic_write_probe_jsonl(Path(args.audit), Path(args.output))
        print(args.output)
    elif command == "record-attempt":
        payload = _record_probe_attempt(args)
        print(_probe_attempt_path(Path(args.attempt_dir)))
        if payload["requires_oom_classification"]:
            print("requires-classification", file=sys.stderr)
    elif command == "classify-failure":
        _classify_failure(Path(args.attempt), args.classification)
        print("capacity_oom")
    elif command == "require-initial-oom":
        print(_require_initial_oom(Path(args.probe_dir), args.method))
    elif command == "require-fallback":
        _require_fallback(Path(args.probe_dir), args.method, args.candidate)
        print("confirmed-fallback")
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
