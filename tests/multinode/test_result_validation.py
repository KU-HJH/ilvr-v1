from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.multinode_results import publish_validation_outputs, validate_results


METHODS = ("full", "lora", "dora")
ZERO_STAGES = (1, 2, 3)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=True, sort_keys=True) + "\n")
    return path


def write_matrix_witnesses(root: Path) -> Path:
    matrix = root / "matrix"
    cases = []
    port = 29700
    for stage in ("stage1", "stage2"):
        for method in METHODS:
            for zero_stage in ZERO_STAGES:
                child = f"matrix-parent-{stage}-{method}-zero{zero_stage}"
                witness_name = f"{child}.json"
                artifact = "full" if method == "full" else f"peft-{method}"
                _write_json(
                    matrix / witness_name,
                    {
                        "schema_version": 1,
                        "kind": "multinode-lifecycle-case",
                        "parent_run_id": "matrix-parent",
                        "child_run_id": child,
                        "stage": stage,
                        "method": method,
                        "zero_stage": zero_stage,
                        "world_size": 4,
                        "main_process_port": port,
                        "attempts": [
                            {
                                "attempt_id": "initial",
                                "from_step": 0,
                                "to_step": 2,
                                "exit_code": 0,
                                "machine_ranks": [0, 1],
                                "process_ranks": [0, 1, 2, 3],
                                "checkpoint": "checkpoint-2",
                                "method_artifact": artifact,
                                "finite_loss": True,
                                "finite_gradient": True,
                                "worker_errors": [],
                            },
                            {
                                "attempt_id": "resume",
                                "from_step": 2,
                                "to_step": 3,
                                "exit_code": 0,
                                "machine_ranks": [0, 1],
                                "process_ranks": [0, 1, 2, 3],
                                "checkpoint": "checkpoint-3",
                                "method_artifact": artifact,
                                "finite_loss": True,
                                "finite_gradient": True,
                                "worker_errors": [],
                            },
                        ],
                    },
                )
                cases.append(
                    {
                        "stage": stage,
                        "method": method,
                        "zero_stage": zero_stage,
                        "child_run_id": child,
                        "main_process_port": port,
                        "witness": witness_name,
                        "exit_codes": [0, 0],
                    }
                )
                port += 1
    _write_json(
        matrix / "matrix-complete.json",
        {
            "schema_version": 1,
            "kind": "multinode-lifecycle-matrix-complete",
            "run_id": "matrix-parent",
            "world_size": 4,
            "first_port": 29700,
            "cases": cases,
        },
    )
    return matrix


def write_evaluation_witness(root: Path) -> Path:
    evaluation = root / "evaluation"
    _write_json(
        evaluation / "summary.json",
        {
            "summary": True,
            "num_samples": 4,
            "average_inference_time_sec": 0.25,
            "total_inference_time_sec": 1.0,
            "accuracy": 1.0,
            "success": 4,
        },
    )
    for index in range(4):
        _write_json(
            evaluation / f"shard-{index}.jsonl.manifest.json",
            {
                "schema_version": 1,
                "result_jsonl": f"shard-{index}.jsonl",
                "dataset_identity": {
                    "path": "/root/workspace/ILVR/data/test.jsonl",
                    "size": 123,
                    "mtime_ns": 456,
                },
                "model_identity": "task6:" + "a" * 64,
                "shard_index": index,
                "num_shards": 4,
                "expected_source_indices": [index],
                "processed_source_indices": [index],
                "worker_exit_status": 0,
                "successful": True,
            },
        )
    return evaluation


def write_capacity_witnesses(root: Path) -> Path:
    capacity = root / "capacity-root"
    for method in METHODS:
        for zero_stage in ZERO_STAGES:
            key = f"{method}-zero{zero_stage}"
            child = f"capacity-{key}-batch2"
            _write_json(
                capacity / "capacity" / f"{key}.json",
                {
                    "schema_version": 1,
                    "kind": "stage1-multinode-capacity",
                    "run_id": f"capacity-{key}",
                    "stage": "stage1",
                    "method": method,
                    "zero_stage": zero_stage,
                    "attempts": [
                        {
                            "batch_per_gpu": 2,
                            "child_run_id": child,
                            "main_process_port": 31000 + zero_stage,
                            "status": "success",
                            "evidence_paths": [
                                {
                                    "node": "coordinator",
                                    "path": f"/results/{child}/initial-train-witness.json",
                                }
                            ],
                        }
                    ],
                    "recommended_batch_per_gpu": 2,
                },
            )
            stage2_run = f"stage2-batch1-{key}"
            _write_json(
                capacity / "stage2-batch1" / f"{key}.json",
                {
                    "schema_version": 1,
                    "run_id": stage2_run,
                    "attempt_id": "initial",
                    "stage": "stage2",
                    "method": method,
                    "zero_stage": zero_stage,
                    "batch_per_gpu": 1,
                    "world_size": 4,
                    "machine_ranks": [0, 1],
                    "exit_codes": {"0": 0, "1": 0},
                    "checkpoint_path": f"/checkpoints/{stage2_run}",
                    "finite_loss_evidence": {
                        "status": "validated",
                        "path": f"/results/{stage2_run}/initial-producer-evidence.json",
                        "sha256": "b" * 64,
                        "size": 42,
                        "finite_loss": True,
                    },
                },
            )
    return capacity


def _valid_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        write_matrix_witnesses(tmp_path),
        write_evaluation_witness(tmp_path),
        write_capacity_witnesses(tmp_path),
    )


def test_validate_results_requires_matrix_eval_and_all_capacity_cells(tmp_path):
    report = validate_results(*_valid_roots(tmp_path))

    assert report.matrix_cases == 18
    assert report.evaluation_samples == 4
    assert len(report.stage1_recommendations) == 9
    assert set(report.stage2_recommendations.values()) == {1}


def test_validate_results_rejects_missing_capacity_cell(tmp_path):
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    (capacity / "capacity" / "dora-zero3.json").unlink()

    with pytest.raises(ValueError, match="nine Stage 1 capacity cells"):
        validate_results(matrix, evaluation, capacity)


def test_validate_results_ignores_recursive_or_unrelated_json(tmp_path):
    """Catches stale recursive JSON satisfying a missing documented cell."""
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    missing = capacity / "capacity" / "dora-zero3.json"
    payload = json.loads(missing.read_text())
    missing.unlink()
    _write_json(capacity / "capacity" / "stale" / "dora-zero3.json", payload)
    _write_json(capacity / "unrelated.json", payload)

    with pytest.raises(ValueError, match="nine Stage 1 capacity cells"):
        validate_results(matrix, evaluation, capacity)


def test_validate_results_rejects_bool_as_integer_identity(tmp_path):
    """Catches Python bool/int equality accepting a malformed ZeRO identity."""
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    witness = matrix / "matrix-parent-stage1-full-zero1.json"
    payload = json.loads(witness.read_text())
    payload["zero_stage"] = True
    _write_json(witness, payload)

    with pytest.raises(ValueError, match="case witness"):
        validate_results(matrix, evaluation, capacity)


@pytest.mark.parametrize("field", ["finite_loss", "finite_gradient"])
def test_validate_results_rejects_nonfinite_or_missing_diagnostic(tmp_path, field):
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    witness = matrix / "matrix-parent-stage1-full-zero1.json"
    payload = json.loads(witness.read_text())
    payload["attempts"][0][field] = float("nan")
    _write_json(witness, payload)

    with pytest.raises(ValueError, match="finite diagnostics"):
        validate_results(matrix, evaluation, capacity)


def test_publish_validation_outputs_writes_matching_tables_exclusively(tmp_path):
    report = validate_results(*_valid_roots(tmp_path))
    manifest = tmp_path / "report.json"
    summary = tmp_path / "report.md"

    publish_validation_outputs(report, manifest, summary)

    payload = json.loads(manifest.read_text())
    markdown = summary.read_text()
    assert payload["stage1_recommendations"]["dora-zero3"] == 2
    assert payload["stage2_recommendations"]["dora-zero3"] == 1
    assert "| Method | ZeRO 1 | ZeRO 2 | ZeRO 3 |" in markdown
    assert "| dora | 2 | 2 | 2 |" in markdown
    assert "| dora | 1 | 1 | 1 |" in markdown

    with pytest.raises(FileExistsError):
        publish_validation_outputs(report, manifest, tmp_path / "other.md")


def test_publish_validation_outputs_rolls_back_json_if_markdown_publish_fails(
    tmp_path, monkeypatch
):
    """Catches the machine manifest surviving a failed human-report publish."""
    report = validate_results(*_valid_roots(tmp_path))
    manifest = tmp_path / "report.json"
    summary = tmp_path / "report.md"
    original_open = Path.open

    def failing_open(path, *args, **kwargs):
        if path == summary and args and args[0] == "x":
            raise OSError("simulated Markdown publish failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(OSError, match="Markdown"):
        publish_validation_outputs(report, manifest, summary)

    assert not manifest.exists()
    assert not summary.exists()
