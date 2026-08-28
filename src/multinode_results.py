"""Strict final validation for the four-GPU ILVR result set."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType


_METHODS = ("full", "lora", "dora")
_ZERO_STAGES = (1, 2, 3)
_MATRIX_IDENTITIES = tuple(
    (stage, method, zero_stage)
    for stage in ("stage1", "stage2")
    for method in _METHODS
    for zero_stage in _ZERO_STAGES
)
_CELL_IDENTITIES = tuple(
    (method, zero_stage) for method in _METHODS for zero_stage in _ZERO_STAGES
)
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,127}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ValidationReport:
    matrix_cases: int
    evaluation_samples: int
    stage1_recommendations: Mapping[str, int]
    stage2_recommendations: Mapping[str, int]


def _read_json(path: Path, *, description: str) -> dict[str, object]:
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing {description}: {path}")
        size = path.stat().st_size
        if size <= 0 or size > _MAX_JSON_BYTES:
            raise ValueError(f"{description} size is invalid: {path}")
        raw = path.read_bytes()

        def reject_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"{description} contains a duplicate field")
                result[key] = value
            return result

        payload = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {description}: {path}") from error
    if type(payload) is not dict:
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _require_fields(
    payload: Mapping[str, object], required: set[str], *, description: str
) -> None:
    if set(payload) != required:
        raise ValueError(f"{description} fields are invalid")


def _require_id(value: object, *, description: str) -> str:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{description} is invalid")
    return value


def _is_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


_ATTEMPT_FIELDS = {
    "attempt_id",
    "from_step",
    "to_step",
    "exit_code",
    "machine_ranks",
    "process_ranks",
    "checkpoint",
    "method_artifact",
    "finite_loss",
    "finite_gradient",
    "worker_errors",
}


def _validate_case_attempt(
    value: object,
    *,
    attempt_id: str,
    from_step: int,
    to_step: int,
    method: str,
) -> None:
    if type(value) is not dict:
        raise ValueError("matrix case attempt must be an object")
    _require_fields(value, _ATTEMPT_FIELDS, description="matrix case attempt")
    expected_artifact = "full" if method == "full" else f"peft-{method}"
    if (
        type(value["attempt_id"]) is not str
        or value["attempt_id"] != attempt_id
        or type(value["from_step"]) is not int
        or value["from_step"] != from_step
        or type(value["to_step"]) is not int
        or value["to_step"] != to_step
        or type(value["exit_code"]) is not int
        or value["exit_code"] != 0
        or type(value["machine_ranks"]) is not list
        or value["machine_ranks"] != [0, 1]
        or any(type(rank) is not int for rank in value["machine_ranks"])
        or type(value["process_ranks"]) is not list
        or value["process_ranks"] != [0, 1, 2, 3]
        or any(type(rank) is not int for rank in value["process_ranks"])
        or type(value["checkpoint"]) is not str
        or value["checkpoint"] != f"checkpoint-{to_step}"
        or type(value["method_artifact"]) is not str
        or value["method_artifact"] != expected_artifact
        or type(value["worker_errors"]) is not list
        or value["worker_errors"]
    ):
        raise ValueError("matrix case attempt identity or worker evidence is invalid")
    if value["finite_loss"] is not True or value["finite_gradient"] is not True:
        raise ValueError("matrix case finite diagnostics are invalid")


_CASE_FIELDS = {
    "schema_version",
    "kind",
    "parent_run_id",
    "child_run_id",
    "stage",
    "method",
    "zero_stage",
    "world_size",
    "main_process_port",
    "attempts",
}
_COMPLETION_CASE_FIELDS = {
    "stage",
    "method",
    "zero_stage",
    "child_run_id",
    "main_process_port",
    "witness",
    "exit_codes",
}


def _validate_matrix(matrix_root: Path) -> int:
    completion = _read_json(
        matrix_root / "matrix-complete.json", description="matrix completion"
    )
    _require_fields(
        completion,
        {"schema_version", "kind", "run_id", "world_size", "first_port", "cases"},
        description="matrix completion",
    )
    parent_run_id = _require_id(completion["run_id"], description="matrix run ID")
    cases = completion["cases"]
    if (
        type(completion["schema_version"]) is not int
        or completion["schema_version"] != 1
        or completion["kind"] != "multinode-lifecycle-matrix-complete"
        or type(completion["world_size"]) is not int
        or completion["world_size"] != 4
        or type(completion["first_port"]) is not int
        or type(cases) is not list
        or len(cases) != 18
    ):
        raise ValueError("matrix completion identity is invalid")
    first_port = completion["first_port"]
    if not 1024 <= first_port or first_port + 17 > 65535:
        raise ValueError("matrix completion port range is invalid")

    for index, (summary, identity) in enumerate(zip(cases, _MATRIX_IDENTITIES, strict=True)):
        if type(summary) is not dict:
            raise ValueError("matrix completion case must be an object")
        _require_fields(
            summary, _COMPLETION_CASE_FIELDS, description="matrix completion case"
        )
        stage, method, zero_stage = identity
        child_run_id = f"{parent_run_id}-{stage}-{method}-zero{zero_stage}"
        witness_name = f"{child_run_id}.json"
        if (
            type(summary["stage"]) is not str
            or summary["stage"] != stage
            or type(summary["method"]) is not str
            or summary["method"] != method
            or type(summary["zero_stage"]) is not int
            or summary["zero_stage"] != zero_stage
            or type(summary["child_run_id"]) is not str
            or summary["child_run_id"] != child_run_id
            or type(summary["main_process_port"]) is not int
            or summary["main_process_port"] != first_port + index
            or type(summary["witness"]) is not str
            or summary["witness"] != witness_name
            or type(summary["exit_codes"]) is not list
            or summary["exit_codes"] != [0, 0]
            or any(type(code) is not int for code in summary["exit_codes"])
        ):
            raise ValueError("matrix completion case identity is invalid")

        witness = _read_json(
            matrix_root / witness_name, description="matrix case witness"
        )
        _require_fields(witness, _CASE_FIELDS, description="matrix case witness")
        attempts = witness["attempts"]
        if (
            type(witness["schema_version"]) is not int
            or witness["schema_version"] != 1
            or witness["kind"] != "multinode-lifecycle-case"
            or type(witness["parent_run_id"]) is not str
            or witness["parent_run_id"] != parent_run_id
            or type(witness["child_run_id"]) is not str
            or witness["child_run_id"] != child_run_id
            or type(witness["stage"]) is not str
            or witness["stage"] != stage
            or type(witness["method"]) is not str
            or witness["method"] != method
            or type(witness["zero_stage"]) is not int
            or witness["zero_stage"] != zero_stage
            or type(witness["world_size"]) is not int
            or witness["world_size"] != 4
            or type(witness["main_process_port"]) is not int
            or witness["main_process_port"] != first_port + index
            or type(attempts) is not list
            or len(attempts) != 2
        ):
            raise ValueError("matrix case witness identity is invalid")
        _validate_case_attempt(
            attempts[0],
            attempt_id="initial",
            from_step=0,
            to_step=2,
            method=method,
        )
        _validate_case_attempt(
            attempts[1],
            attempt_id="resume",
            from_step=2,
            to_step=3,
            method=method,
        )
    return 18


_SUMMARY_FIELDS = {
    "summary",
    "num_samples",
    "average_inference_time_sec",
    "total_inference_time_sec",
    "accuracy",
    "success",
}
_SIDECAR_FIELDS = {
    "schema_version",
    "result_jsonl",
    "dataset_identity",
    "model_identity",
    "shard_index",
    "num_shards",
    "expected_source_indices",
    "processed_source_indices",
    "worker_exit_status",
    "successful",
}


def _validate_evaluation(evaluation_root: Path) -> int:
    summary = _read_json(
        evaluation_root / "summary.json", description="evaluation summary"
    )
    _require_fields(summary, _SUMMARY_FIELDS, description="evaluation summary")
    if (
        summary["summary"] is not True
        or type(summary["num_samples"]) is not int
        or summary["num_samples"] != 4
        or type(summary["success"]) is not int
        or not 0 <= summary["success"] <= 4
        or not _is_number(summary["average_inference_time_sec"])
        or summary["average_inference_time_sec"] < 0
        or not _is_number(summary["total_inference_time_sec"])
        or summary["total_inference_time_sec"] < 0
        or not _is_number(summary["accuracy"])
        or not 0 <= summary["accuracy"] <= 1
    ):
        raise ValueError("evaluation summary is invalid")

    common_dataset = None
    common_model = None
    observed_indices: list[int] = []
    for shard_index in range(4):
        sidecar = _read_json(
            evaluation_root / f"shard-{shard_index}.jsonl.manifest.json",
            description=f"evaluation sidecar {shard_index}",
        )
        _require_fields(
            sidecar, _SIDECAR_FIELDS, description=f"evaluation sidecar {shard_index}"
        )
        dataset = sidecar["dataset_identity"]
        model = sidecar["model_identity"]
        if (
            type(sidecar["schema_version"]) is not int
            or sidecar["schema_version"] != 1
            or sidecar["result_jsonl"] != f"shard-{shard_index}.jsonl"
            or type(sidecar["shard_index"]) is not int
            or sidecar["shard_index"] != shard_index
            or type(sidecar["num_shards"]) is not int
            or sidecar["num_shards"] != 4
            or type(sidecar["expected_source_indices"]) is not list
            or sidecar["expected_source_indices"] != [shard_index]
            or any(type(value) is not int for value in sidecar["expected_source_indices"])
            or type(sidecar["processed_source_indices"]) is not list
            or sidecar["processed_source_indices"] != [shard_index]
            or any(type(value) is not int for value in sidecar["processed_source_indices"])
            or type(sidecar["worker_exit_status"]) is not int
            or sidecar["worker_exit_status"] != 0
            or sidecar["successful"] is not True
            or type(model) is not str
            or not model.startswith("task6:")
            or _HEX64.fullmatch(model.removeprefix("task6:")) is None
            or type(dataset) is not dict
            or set(dataset) != {"path", "size", "mtime_ns"}
            or type(dataset["path"]) is not str
            or not Path(dataset["path"]).is_absolute()
            or type(dataset["size"]) is not int
            or dataset["size"] < 0
            or type(dataset["mtime_ns"]) is not int
            or dataset["mtime_ns"] < 0
        ):
            raise ValueError(f"evaluation sidecar {shard_index} is invalid")
        if common_dataset is None:
            common_dataset = dataset
            common_model = model
        elif dataset != common_dataset or model != common_model:
            raise ValueError("evaluation sidecar identities disagree")
        observed_indices.extend(sidecar["processed_source_indices"])
    if observed_indices != [0, 1, 2, 3] or len(set(observed_indices)) != 4:
        raise ValueError("evaluation source indices are incomplete")
    return 4


_CAPACITY_FIELDS = {
    "schema_version",
    "kind",
    "run_id",
    "stage",
    "method",
    "zero_stage",
    "attempts",
    "recommended_batch_per_gpu",
}
_CAPACITY_ATTEMPT_FIELDS = {
    "batch_per_gpu",
    "child_run_id",
    "main_process_port",
    "status",
    "evidence_paths",
}
_STAGE2_WITNESS_FIELDS = {
    "schema_version",
    "run_id",
    "attempt_id",
    "stage",
    "method",
    "zero_stage",
    "batch_per_gpu",
    "world_size",
    "machine_ranks",
    "exit_codes",
    "checkpoint_path",
    "finite_loss_evidence",
}
_FINITE_FIELDS = {"status", "path", "sha256", "size", "finite_loss"}


def _validate_capacity_attempt(value: object) -> int:
    if type(value) is not dict:
        raise ValueError("Stage 1 capacity attempt must be an object")
    _require_fields(
        value, _CAPACITY_ATTEMPT_FIELDS, description="Stage 1 capacity attempt"
    )
    evidence = value["evidence_paths"]
    if (
        type(value["batch_per_gpu"]) is not int
        or value["batch_per_gpu"] <= 0
        or type(value["child_run_id"]) is not str
        or _SAFE_ID.fullmatch(value["child_run_id"]) is None
        or type(value["main_process_port"]) is not int
        or not 1024 <= value["main_process_port"] <= 65535
        or type(value["status"]) is not str
        or value["status"] not in {"success", "capacity_oom"}
        or type(evidence) is not list
        or not evidence
    ):
        raise ValueError("Stage 1 capacity attempt is invalid")
    for reference in evidence:
        if (
            type(reference) is not dict
            or set(reference) != {"node", "path"}
            or reference["node"] not in {"remote", "local", "coordinator"}
            or type(reference["path"]) is not str
            or not Path(reference["path"]).is_absolute()
        ):
            raise ValueError("Stage 1 capacity evidence reference is invalid")
    return value["batch_per_gpu"]


def _validate_capacity(capacity_root: Path) -> tuple[Mapping[str, int], Mapping[str, int]]:
    stage1_root = capacity_root / "capacity"
    stage2_root = capacity_root / "stage2-batch1"
    expected_names = {f"{method}-zero{zero_stage}.json" for method, zero_stage in _CELL_IDENTITIES}
    observed_stage1 = (
        {path.name for path in stage1_root.iterdir() if path.is_file() and not path.is_symlink()}
        if stage1_root.is_dir()
        else set()
    )
    observed_stage2 = (
        {path.name for path in stage2_root.iterdir() if path.is_file() and not path.is_symlink()}
        if stage2_root.is_dir()
        else set()
    )
    if observed_stage1 != expected_names:
        raise ValueError("exactly nine Stage 1 capacity cells are required")
    if observed_stage2 != expected_names:
        raise ValueError("exactly nine Stage 2 batch-1 cells are required")

    stage1_recommendations: dict[str, int] = {}
    stage2_recommendations: dict[str, int] = {}
    for method, zero_stage in _CELL_IDENTITIES:
        key = f"{method}-zero{zero_stage}"
        capacity = _read_json(stage1_root / f"{key}.json", description="Stage 1 capacity cell")
        _require_fields(capacity, _CAPACITY_FIELDS, description="Stage 1 capacity cell")
        attempts = capacity["attempts"]
        recommendation = capacity["recommended_batch_per_gpu"]
        if (
            type(capacity["schema_version"]) is not int
            or capacity["schema_version"] != 1
            or capacity["kind"] != "stage1-multinode-capacity"
            or type(capacity["run_id"]) is not str
            or _SAFE_ID.fullmatch(capacity["run_id"]) is None
            or capacity["stage"] != "stage1"
            or type(capacity["method"]) is not str
            or capacity["method"] != method
            or type(capacity["zero_stage"]) is not int
            or capacity["zero_stage"] != zero_stage
            or type(attempts) is not list
            or not attempts
            or type(recommendation) is not int
            or recommendation not in (1, 2, 4, 8)
        ):
            raise ValueError("Stage 1 capacity cell is invalid")
        batches = [_validate_capacity_attempt(attempt) for attempt in attempts]
        if batches[0] != 2 or batches not in ([2], [2, 4], [2, 4, 8], [2, 1]):
            raise ValueError("Stage 1 capacity attempts must begin at batch 2")
        stage1_recommendations[key] = recommendation

        witness = _read_json(
            stage2_root / f"{key}.json", description="Stage 2 batch-1 witness"
        )
        _require_fields(
            witness, _STAGE2_WITNESS_FIELDS, description="Stage 2 batch-1 witness"
        )
        finite = witness["finite_loss_evidence"]
        if (
            type(witness["schema_version"]) is not int
            or witness["schema_version"] != 1
            or type(witness["run_id"]) is not str
            or _SAFE_ID.fullmatch(witness["run_id"]) is None
            or witness["attempt_id"] != "initial"
            or witness["stage"] != "stage2"
            or type(witness["method"]) is not str
            or witness["method"] != method
            or type(witness["zero_stage"]) is not int
            or witness["zero_stage"] != zero_stage
            or type(witness["batch_per_gpu"]) is not int
            or witness["batch_per_gpu"] != 1
            or type(witness["world_size"]) is not int
            or witness["world_size"] != 4
            or type(witness["machine_ranks"]) is not list
            or witness["machine_ranks"] != [0, 1]
            or any(type(rank) is not int for rank in witness["machine_ranks"])
            or type(witness["exit_codes"]) is not dict
            or witness["exit_codes"] != {"0": 0, "1": 0}
            or any(type(code) is not int for code in witness["exit_codes"].values())
            or type(witness["checkpoint_path"]) is not str
            or not Path(witness["checkpoint_path"]).is_absolute()
            or type(finite) is not dict
            or set(finite) != _FINITE_FIELDS
            or finite["status"] != "validated"
            or finite["finite_loss"] is not True
            or type(finite["path"]) is not str
            or not Path(finite["path"]).is_absolute()
            or type(finite["sha256"]) is not str
            or _HEX64.fullmatch(finite["sha256"]) is None
            or type(finite["size"]) is not int
            or finite["size"] <= 0
        ):
            raise ValueError("Stage 2 batch-1 witness is invalid")
        stage2_recommendations[key] = 1
    return MappingProxyType(stage1_recommendations), MappingProxyType(stage2_recommendations)


def validate_results(
    matrix_root: Path, evaluation_root: Path, capacity_root: Path
) -> ValidationReport:
    """Validate only the documented direct Task 10 result paths."""
    matrix = Path(matrix_root)
    evaluation = Path(evaluation_root)
    capacity = Path(capacity_root)
    matrix_cases = _validate_matrix(matrix)
    evaluation_samples = _validate_evaluation(evaluation)
    stage1, stage2 = _validate_capacity(capacity)
    return ValidationReport(
        matrix_cases=matrix_cases,
        evaluation_samples=evaluation_samples,
        stage1_recommendations=stage1,
        stage2_recommendations=stage2,
    )


def _report_payload(report: ValidationReport) -> dict[str, object]:
    return {
        "schema_version": 1,
        "matrix_cases": report.matrix_cases,
        "evaluation_samples": report.evaluation_samples,
        "stage1_recommendations": dict(report.stage1_recommendations),
        "stage2_recommendations": dict(report.stage2_recommendations),
    }


def _recommendation_table(title: str, values: Mapping[str, int]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Method | ZeRO 1 | ZeRO 2 | ZeRO 3 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in _METHODS:
        lines.append(
            f"| {method} | {values[f'{method}-zero1']} | "
            f"{values[f'{method}-zero2']} | {values[f'{method}-zero3']} |"
        )
    return lines


def _report_markdown(report: ValidationReport) -> str:
    lines = [
        "# ILVR Four-GPU Validation Summary",
        "",
        f"Validated matrix cases: {report.matrix_cases}",
        "",
        f"Validated evaluation samples: {report.evaluation_samples}",
        "",
        *_recommendation_table("Stage 1 recommendations", report.stage1_recommendations),
        "",
        *_recommendation_table("Stage 2 recommendations", report.stage2_recommendations),
        "",
    ]
    return "\n".join(lines)


def publish_validation_outputs(
    report: ValidationReport,
    output_manifest: str | Path,
    output_summary: str | Path,
) -> None:
    """Publish matching JSON/Markdown exclusively and roll back partial output."""
    if not isinstance(report, ValidationReport):
        raise TypeError("report must be a ValidationReport")
    manifest = Path(output_manifest)
    summary = Path(output_summary)
    if manifest == summary:
        raise ValueError("output manifest and summary must differ")
    if os.path.lexists(manifest):
        raise FileExistsError(manifest)
    if os.path.lexists(summary):
        raise FileExistsError(summary)
    manifest_bytes = (
        json.dumps(
            _report_payload(report),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    summary_text = _report_markdown(report)
    manifest_created = False
    summary_created = False
    try:
        with manifest.open("x", encoding="utf-8") as stream:
            manifest_created = True
            stream.write(manifest_bytes)
        with summary.open("x", encoding="utf-8") as stream:
            summary_created = True
            stream.write(summary_text)
    except BaseException:
        if summary_created:
            summary.unlink(missing_ok=True)
        if manifest_created:
            manifest.unlink(missing_ok=True)
        raise


def validate_results_main(argv: Sequence[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.multinode validate-results", allow_abbrev=False
    )
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--capacity-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args(argv)
    try:
        report = validate_results(
            Path(args.matrix_root),
            Path(args.evaluation_root),
            Path(args.capacity_root),
        )
        publish_validation_outputs(
            report, Path(args.output_manifest), Path(args.output_summary)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(_report_payload(report), allow_nan=False, sort_keys=True))
    return 0
