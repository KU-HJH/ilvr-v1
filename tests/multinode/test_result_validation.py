from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import src.multinode_results as results_module
from src.multinode_results import publish_validation_outputs, validate_results
from stage2.evidence import build_stage2_producer_evidence


METHODS = ("full", "lora", "dora")
ZERO_STAGES = (1, 2, 3)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=True, sort_keys=True) + "\n")
    return path


def _raw_lifecycle(stage: str, method: str, zero_stage: int, child: str, checkpoint: Path):
    artifact = "full" if method == "full" else f"peft-{method}"
    if stage == "stage1":
        return {
            "schema_version": 1, "stage": stage, "zero_stage": zero_stage,
            "config_digest": "a" * 64, "world_size": 4, "method": method,
            "artifact_kind": "full" if method == "full" else "peft",
            "method_artifact": artifact,
            "adapter_checkpoint_present": method != "full",
            "initial_global_step": 2, "resumed_global_step": 3,
            "finite_gradient_norm": True, "representative_gradient_norm": 1.0,
            "successful_optimizer_steps": 3, "finite_loss": True,
            "first_loss": 0.5, "resumed_loss": 0.25,
            "process_ranks": [0, 1, 2, 3],
            "rank_records": [
                {
                    "rank": rank, "initial_global_step": 2,
                    "resumed_global_step": 3, "initial_optimizer_steps": 2,
                    "resumed_optimizer_steps": 1, "first_loss": 0.5 + rank,
                    "resumed_loss": 0.25 + rank, "finite_gradient_norm": True,
                    "representative_gradient_norm": 1.0 + rank,
                    "method_artifact": artifact, "worker_error": None,
                }
                for rank in range(4)
            ],
            "worker_errors": [], "checkpoint": str(checkpoint),
            "experiment_id": child,
        }
    records = [
        {
            "rank": rank, "first_global_step": 2, "resumed_global_step": 3,
            "optimizer_step": 3, "first_loss": 0.5 + rank,
            "resumed_loss": 0.25 + rank, "gradient_norm": 1.0 + rank,
            "worker_error": None, "first_batches": [1], "resumed_batches": [1],
            "optimizer_step_at_resume": 2, "scheduler_step_at_resume": 2,
            "adapter_parameters_changed": method != "full",
            "visual_trainable_parameters": 0,
            "base_trainable_parameters": 1 if method == "full" else 0,
        }
        for rank in range(4)
    ]
    return {
        "schema_version": 1, "stage": "stage2", "method": method,
        "zero_stage": zero_stage, "config_digest": "a" * 64,
        "artifact_kind": artifact, "tuning_method": method,
        "first_global_step": 2, "resumed_global_step": 3,
        "rank_first_global_steps": [2] * 4, "rank_resumed_global_steps": [3] * 4,
        "rank_first_losses": [r["first_loss"] for r in records],
        "rank_resumed_losses": [r["resumed_loss"] for r in records],
        "rank_records": records, "process_ranks": [0, 1, 2, 3],
        "finite_loss": True, "finite_gradient": True, "worker_errors": [],
        "optimizer_state_restored": True, "scheduler_state_restored": True,
        "optimizer_step_at_resume": 2, "scheduler_step_at_resume": 2,
        "observed_local_batch_sizes": [1] * 8, "world_size": 4,
        "first_loss": 0.5, "resumed_loss": 0.25, "deepspeed": True,
        "deepspeed_config": f"/root/workspace/ILVR/configs/deepspeed_zero{zero_stage}.json",
        "deepspeed_zero_stage": zero_stage,
        "adapter_parameters_changed": method != "full",
        "adapter_checkpoint_present": method != "full", "sidecar_present": method != "full",
        "processor_present": True, "full_model_weights_absent": method != "full",
        "visual_trainable_parameters": 0,
        "base_trainable_parameters": 1 if method == "full" else 0,
        "git_commit": "a" * 40, "torch_version": "2.7.1",
        "transformers_version": "4.50", "peft_version": "0.15",
        "harness_path": "/root/workspace/ILVR/tests/stage2/integration/tiny_stage2_train.py",
        "checkpoint": str(checkpoint), "experiment_id": child,
    }


def write_matrix_witnesses(root: Path) -> Path:
    matrix = root / "matrix"
    cases = []
    port = 29700
    for stage in ("stage1", "stage2"):
        for method in METHODS:
            for zero_stage in ZERO_STAGES:
                child = f"matrix-parent-{stage}-{method}-zero{zero_stage}"
                witness_name = f"{child}.json"
                checkpoint = root / "checkpoints" / child / "checkpoint-2"
                checkpoint.mkdir(parents=True)
                raw_path = root / "raw" / child / "witness.json"
                raw = _write_json(raw_path, _raw_lifecycle(
                    stage, method, zero_stage, child, checkpoint
                )).read_bytes()
                records = json.loads(raw)["rank_records"]
                gradient_key = "representative_gradient_norm" if stage == "stage1" else "gradient_norm"
                artifact = "full" if method == "full" else f"peft-{method}"
                _write_json(matrix / witness_name, {
                    "schema_version": 1, "kind": "multinode-lifecycle-case",
                    "parent_run_id": "matrix-parent", "child_run_id": child,
                    "stage": stage, "method": method, "zero_stage": zero_stage,
                    "world_size": 4, "main_process_port": port,
                    "machine_ranks": [0, 1], "process_ranks": [0, 1, 2, 3],
                    "method_artifact": artifact, "checkpoint": "checkpoint-2",
                    "raw_witness": {"path": str(raw_path),
                                    "sha256": hashlib.sha256(raw).hexdigest(),
                                    "size": len(raw)},
                    "attempts": [
                        {"from_step": 0, "to_step": 2,
                         "rank_losses": [r["first_loss"] for r in records],
                         "rank_gradients": [r[gradient_key] for r in records]},
                        {"from_step": 2, "to_step": 3,
                         "rank_losses": [r["resumed_loss"] for r in records],
                         "rank_gradients": [r[gradient_key] for r in records]},
                    ],
                    "finite_loss": True, "finite_gradient": True, "worker_errors": [],
                    "collected_node_manifests": 2, "exit_codes": {"0": 0, "1": 0},
                })
                cases.append({"stage": stage, "method": method, "zero_stage": zero_stage,
                              "child_run_id": child, "main_process_port": port,
                              "witness": witness_name, "exit_codes": [0, 0]})
                port += 1
    _write_json(matrix / "matrix-complete.json", {
        "schema_version": 1, "kind": "multinode-lifecycle-matrix-complete",
        "run_id": "matrix-parent", "world_size": 4, "first_port": 29700,
        "cases": cases,
    })
    return matrix


def write_evaluation_witness(root: Path) -> Path:
    evaluation = root / "evaluation"
    for index in range(4):
        jsonl = evaluation / f"shard-{index}.jsonl"
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl.write_text(json.dumps({"index": index, "task_name": "tiny",
                                     "match": True, "inference_time_sec": 0.25}) + "\n")
        _write_json(evaluation / f"shard-{index}.jsonl.manifest.json", {
            "schema_version": 1, "result_jsonl": f"shard-{index}.jsonl",
            "dataset_identity": {"path": "/root/workspace/ILVR/data/test.jsonl",
                                 "size": 123, "mtime_ns": 456},
            "model_identity": "task6:" + "a" * 64, "shard_index": index,
            "num_shards": 4, "expected_source_indices": [index],
            "processed_source_indices": [index], "worker_exit_status": 0,
            "successful": True,
        })
    _write_json(evaluation / "summary.json", {
        "summary": True, "num_samples": 4, "average_inference_time_sec": 0.25,
        "total_inference_time_sec": 1.0, "accuracy": 1.0, "success": 4,
    })
    return evaluation


def _write_stage1_child_witness(stage1_root: Path, *, child: str, method: str,
                                zero_stage: int, batch: int) -> Path:
    child_root = stage1_root / child
    producer = _write_json(child_root / "initial-producer-evidence.json", {"finite_loss": True})
    raw = producer.read_bytes()
    return _write_json(child_root / "initial-train-witness.json", {
        "schema_version": 1, "run_id": child, "attempt_id": "initial",
        "stage": "stage1", "method": method, "zero_stage": zero_stage,
        "batch_per_gpu": batch, "world_size": 4, "machine_ranks": [0, 1],
        "exit_codes": {"0": 0, "1": 0},
        "checkpoint_path": f"/root/workspace/ILVR/checkpoints/{child}",
        "finite_loss_evidence": {"status": "validated", "path": str(producer),
                                 "sha256": hashlib.sha256(raw).hexdigest(),
                                 "size": len(raw), "finite_loss": True},
    })


def write_capacity_witnesses(root: Path) -> Path:
    capacity_root = root / "capacity-root"
    stage1_root = capacity_root / "capacity"
    stage2_root = capacity_root / "stage2-batch1"
    for method in METHODS:
        for zero_stage in ZERO_STAGES:
            key = f"{method}-zero{zero_stage}"
            run_id = f"capacity-{key}"
            child = f"{run_id}-batch2"
            child_witness = _write_stage1_child_witness(
                stage1_root, child=child, method=method, zero_stage=zero_stage, batch=2
            )
            _write_json(stage1_root / run_id / "capacity.json", {
                "schema_version": 1, "kind": "stage1-multinode-capacity",
                "run_id": run_id, "stage": "stage1", "method": method,
                "zero_stage": zero_stage,
                "attempts": [{"batch_per_gpu": 2, "child_run_id": child,
                              "main_process_port": 31000 + zero_stage,
                              "status": "success", "evidence_paths": [
                                  {"node": "coordinator", "path": str(child_witness)}]}],
                "recommended_batch_per_gpu": 2,
            })
            stage2_run = f"stage2-batch1-{key}"
            checkpoint_root = root / "checkpoints" / stage2_run
            checkpoint = checkpoint_root / "checkpoint-1"
            checkpoint.mkdir(parents=True)
            (checkpoint / "rank-shard.bin").write_bytes(b"observed")
            records = [{"rank": rank, "global_step": 1, "optimizer_step": 1,
                        "loss": 0.25 + rank, "gradient_norm": 1.0 + rank,
                        "worker_error": None} for rank in range(4)]
            producer = _write_json(
                stage2_root / stage2_run / "initial-producer-evidence.json",
                build_stage2_producer_evidence(
                    run_id=stage2_run, method=method, zero_stage=zero_stage,
                    checkpoint_path=checkpoint,
                    artifact_kind="full" if method == "full" else f"peft-{method}",
                    rank_records=records,
                ),
            )
            raw = producer.read_bytes()
            _write_json(stage2_root / stage2_run / "initial-train-witness.json", {
                "schema_version": 1, "run_id": stage2_run, "attempt_id": "initial",
                "stage": "stage2", "method": method, "zero_stage": zero_stage,
                "batch_per_gpu": 1, "world_size": 4, "machine_ranks": [0, 1],
                "exit_codes": {"0": 0, "1": 0}, "checkpoint_path": str(checkpoint_root),
                "finite_loss_evidence": {
                    "status": "validated", "path": str(producer),
                    "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw),
                    "finite_loss": True, "finite_gradient": True,
                    "process_ranks": [0, 1, 2, 3],
                },
            })
    return capacity_root


def _valid_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (write_matrix_witnesses(tmp_path), write_evaluation_witness(tmp_path),
            write_capacity_witnesses(tmp_path))


def test_validate_results_requires_matrix_eval_and_all_capacity_cells(tmp_path):
    report = validate_results(*_valid_roots(tmp_path))
    assert report.matrix_cases == 18
    assert report.evaluation_samples == 4
    assert len(report.stage1_recommendations) == 9
    assert set(report.stage2_recommendations.values()) == {1}


def test_validate_results_requires_nested_direct_capacity_producer_paths(tmp_path):
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    run = capacity / "capacity" / "capacity-dora-zero3"
    payload = json.loads((run / "capacity.json").read_text())
    (run / "capacity.json").unlink()
    _write_json(capacity / "capacity" / "dora-zero3.json", payload)
    with pytest.raises(ValueError, match="Stage 1 capacity"):
        validate_results(matrix, evaluation, capacity)


def test_validate_results_rejects_recommendation_without_successful_child(tmp_path):
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    report = capacity / "capacity" / "capacity-full-zero1" / "capacity.json"
    payload = json.loads(report.read_text())
    payload["attempts"][0]["status"] = "capacity_oom"
    _write_json(report, payload)
    with pytest.raises(ValueError, match="successful attempted batch"):
        validate_results(matrix, evaluation, capacity)


def test_validate_results_rereads_shard_rows_and_accuracy(tmp_path):
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    (evaluation / "shard-2.jsonl").write_text(
        json.dumps({"index": 2, "task_name": "tiny", "match": False}) + "\n"
    )
    with pytest.raises(ValueError, match="accuracy"):
        validate_results(matrix, evaluation, capacity)


def test_validate_results_rejects_stage2_producer_digest_or_checkpoint_swap(tmp_path):
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    producer = (capacity / "stage2-batch1" / "stage2-batch1-full-zero1"
                / "initial-producer-evidence.json")
    payload = json.loads(producer.read_text())
    payload["ranks"][0]["gradient_norm"] = 9.0
    _write_json(producer, payload)
    with pytest.raises(ValueError, match="digest"):
        validate_results(matrix, evaluation, capacity)


def test_descriptor_bound_reader_rejects_final_component_swap(tmp_path):
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    summary = evaluation / "summary.json"
    replacement = evaluation / "replacement.json"
    replacement.write_bytes(summary.read_bytes())
    summary.unlink()
    summary.symlink_to(replacement.name)
    with pytest.raises(ValueError, match="summary"):
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
    report = validate_results(*_valid_roots(tmp_path))
    manifest = tmp_path / "report.json"
    summary = tmp_path / "report.md"
    original_publish = results_module._publish_bytes_no_replace

    def failing_publish(path, payload):
        if Path(path) == summary:
            raise OSError("simulated Markdown publication failure")
        return original_publish(path, payload)

    monkeypatch.setattr(results_module, "_publish_bytes_no_replace", failing_publish)
    with pytest.raises(OSError, match="Markdown"):
        publish_validation_outputs(report, manifest, summary)
    assert not manifest.exists()
    assert not summary.exists()
