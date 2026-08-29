from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

import pytest

import src.multinode_evidence as lifecycle_evidence_module
import src.multinode_results as results_module
from src.multinode_evidence import build_lifecycle_phase_evidence
from src.multinode_results import (
    ValidationReport,
    publish_validation_outputs,
    validate_results,
)
from stage2.evidence import build_stage2_producer_evidence


METHODS = ("full", "lora", "dora")
ZERO_STAGES = (1, 2, 3)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, allow_nan=True, sort_keys=True) + "\n")
    return path


def _raw_phase(
    stage: str, method: str, zero_stage: int, child: str, checkpoint: Path,
    phase: str,
):
    global_step = 2 if phase == "initial" else 3
    optimizer_step = 2 if phase == "initial" else (1 if stage == "stage1" else 3)
    gradient_base = 1.0 if phase == "initial" else 9.0
    return build_lifecycle_phase_evidence(
        phase=phase, stage=stage, method=method, zero_stage=zero_stage,
        config_digest="a" * 64, experiment_id=child,
        checkpoint_path=checkpoint,
        artifact_kind="full" if method == "full" else "peft",
        rank_records=[{
            "rank": rank, "global_step": global_step,
            "optimizer_step": optimizer_step, "loss": 0.5 + rank,
            "gradient_norm": gradient_base + rank, "worker_error": None,
        } for rank in range(4)],
    )


@pytest.fixture(autouse=True)
def _canonical_fixture_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(
        results_module, "_CANONICAL_RESULT_ROOT", tmp_path / "canonical-results"
    )
    monkeypatch.setattr(
        results_module, "_CANONICAL_CHECKPOINT_ROOT", tmp_path / "canonical-checkpoints"
    )
    monkeypatch.setattr(
        lifecycle_evidence_module,
        "_CANONICAL_CHECKPOINT_ROOT",
        tmp_path / "canonical-checkpoints",
    )


def write_matrix_witnesses(root: Path) -> Path:
    matrix = root / "matrix"
    cases = []
    port = 29700
    for stage in ("stage1", "stage2"):
        for method in METHODS:
            for zero_stage in ZERO_STAGES:
                child = f"matrix-parent-{stage}-{method}-zero{zero_stage}"
                witness_name = f"{child}.json"
                checkpoint = results_module._CANONICAL_CHECKPOINT_ROOT / child / "checkpoint-2"
                checkpoint.mkdir(parents=True)
                (checkpoint / "rank-shard.bin").write_bytes(b"observed")
                raw_root = results_module._CANONICAL_RESULT_ROOT / child
                raw_paths = {}
                raw_bytes = {}
                raw_payloads = {}
                for phase in ("initial", "resume"):
                    raw_paths[phase] = raw_root / f"{phase}-witness.json"
                    raw_bytes[phase] = _write_json(
                        raw_paths[phase],
                        _raw_phase(stage, method, zero_stage, child, checkpoint, phase),
                    ).read_bytes()
                    raw_payloads[phase] = json.loads(raw_bytes[phase])
                artifact = "full" if method == "full" else "peft"
                _write_json(matrix / witness_name, {
                    "schema_version": 1, "kind": "multinode-lifecycle-case",
                    "parent_run_id": "matrix-parent", "child_run_id": child,
                    "stage": stage, "method": method, "zero_stage": zero_stage,
                    "world_size": 4, "main_process_port": port,
                    "machine_ranks": [0, 1], "process_ranks": [0, 1, 2, 3],
                    "artifact_kind": artifact, "checkpoint": "checkpoint-2",
                    "raw_witnesses": {
                        phase: {"path": str(raw_paths[phase]),
                                "sha256": hashlib.sha256(raw_bytes[phase]).hexdigest(),
                                "size": len(raw_bytes[phase])}
                        for phase in ("initial", "resume")
                    },
                    "attempts": [
                        {"from_step": 0, "to_step": 2,
                         "rank_losses": [r["loss"] for r in raw_payloads["initial"]["ranks"]],
                         "rank_gradients": [r["gradient_norm"] for r in raw_payloads["initial"]["ranks"]]},
                        {"from_step": 2, "to_step": 3,
                         "rank_losses": [r["loss"] for r in raw_payloads["resume"]["ranks"]],
                         "rank_gradients": [r["gradient_norm"] for r in raw_payloads["resume"]["ranks"]]},
                    ],
                    "finite_loss": True, "finite_gradient": True, "worker_errors": [],
                    "collected_node_manifests": 2, "redistributed_entries": 7,
                    "exit_codes": {
                        "initial": {"0": 0, "1": 0},
                        "resume": {"0": 0, "1": 0},
                    },
                })
                cases.append({"stage": stage, "method": method, "zero_stage": zero_stage,
                              "child_run_id": child, "main_process_port": port,
                              "witness": witness_name,
                              "exit_codes": {"initial": [0, 0], "resume": [0, 0]}})
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
            checkpoint_root = results_module._CANONICAL_CHECKPOINT_ROOT / stage2_run
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
                    artifact_kind="full" if method == "full" else "peft",
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


@pytest.mark.parametrize(
    "location",
    ("evaluation-root", "capacity-root", "stage1-run", "stage2-run"),
)
def test_validate_results_rejects_unexpected_direct_producer_paths(
    tmp_path, location
):
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    roots = {
        "evaluation-root": evaluation,
        "capacity-root": capacity,
        "stage1-run": capacity / "capacity" / "capacity-full-zero1",
        "stage2-run": capacity / "stage2-batch1" / "stage2-batch1-full-zero1",
    }
    _write_json(roots[location] / "unexpected.json", {"foreign": True})
    with pytest.raises(ValueError, match="evaluation|capacity|Stage 1|Stage 2"):
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


@pytest.mark.parametrize(
    ("target", "replacement"),
    (
        ("completion-zero", True),
        ("case-machine-ranks", [0.0, 1]),
        ("case-attempt-from-step", False),
        ("completion-case-exit-code", False),
        ("case-exit-code", False),
        ("sidecar-source-indices", [False]),
    ),
)
def test_validate_results_rejects_bool_or_float_identity_types(
    tmp_path, target, replacement
):
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    if target == "completion-zero":
        path = matrix / "matrix-complete.json"
        payload = json.loads(path.read_text())
        payload["cases"][0]["zero_stage"] = replacement
    elif target == "case-machine-ranks":
        path = matrix / "matrix-parent-stage1-full-zero1.json"
        payload = json.loads(path.read_text())
        payload["machine_ranks"] = replacement
    elif target == "case-attempt-from-step":
        path = matrix / "matrix-parent-stage1-full-zero1.json"
        payload = json.loads(path.read_text())
        payload["attempts"][0]["from_step"] = replacement
    elif target == "completion-case-exit-code":
        path = matrix / "matrix-complete.json"
        payload = json.loads(path.read_text())
        payload["cases"][0]["exit_codes"]["initial"][0] = replacement
    elif target == "case-exit-code":
        path = matrix / "matrix-parent-stage1-full-zero1.json"
        payload = json.loads(path.read_text())
        payload["exit_codes"]["initial"]["0"] = replacement
    else:
        path = evaluation / "shard-0.jsonl.manifest.json"
        payload = json.loads(path.read_text())
        payload["expected_source_indices"] = replacement
    _write_json(path, payload)
    with pytest.raises(ValueError, match="matrix|evaluation"):
        validate_results(matrix, evaluation, capacity)


def test_validate_results_rejects_noncanonical_matrix_raw_path(tmp_path):
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    case = matrix / "matrix-parent-stage1-full-zero1.json"
    payload = json.loads(case.read_text())
    reference = payload["raw_witnesses"]["initial"]
    source = Path(reference["path"])
    lookalike = tmp_path / "lookalike" / source.parent.name / source.name
    lookalike.parent.mkdir(parents=True)
    lookalike.write_bytes(source.read_bytes())
    reference["path"] = str(lookalike)
    _write_json(case, payload)
    with pytest.raises(ValueError, match="canonical"):
        validate_results(matrix, evaluation, capacity)


def test_validate_results_rejects_noncanonical_stage2_checkpoint_root(tmp_path):
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    run_id = "stage2-batch1-full-zero1"
    run = capacity / "stage2-batch1" / run_id
    witness_path = run / "initial-train-witness.json"
    producer_path = run / "initial-producer-evidence.json"
    lookalike_root = tmp_path / "lookalike-checkpoints" / run_id
    checkpoint = lookalike_root / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "rank-shard.bin").write_bytes(b"foreign")
    records = [{"rank": rank, "global_step": 1, "optimizer_step": 1,
                "loss": 0.25 + rank, "gradient_norm": 1.0 + rank,
                "worker_error": None} for rank in range(4)]
    producer = _write_json(producer_path, build_stage2_producer_evidence(
        run_id=run_id, method="full", zero_stage=1,
        checkpoint_path=checkpoint, artifact_kind="full", rank_records=records,
    ))
    raw = producer.read_bytes()
    witness = json.loads(witness_path.read_text())
    witness["checkpoint_path"] = str(lookalike_root)
    witness["finite_loss_evidence"]["sha256"] = hashlib.sha256(raw).hexdigest()
    witness["finite_loss_evidence"]["size"] = len(raw)
    _write_json(witness_path, witness)
    with pytest.raises(ValueError, match="canonical"):
        validate_results(matrix, evaluation, capacity)


def test_matrix_directory_swap_after_enumeration_uses_pinned_snapshot(
    tmp_path, monkeypatch
):
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    replacement = tmp_path / "replacement-matrix"
    shutil.copytree(matrix, replacement)
    corrupt = replacement / "matrix-parent-stage1-full-zero1.json"
    corrupt.write_text("{}\n", encoding="utf-8")
    original = tmp_path / "original-matrix"
    real_listdir = results_module.os.listdir
    matrix_stat = matrix.stat()
    swapped = False

    def swap_after_listdir(descriptor):
        nonlocal swapped
        names = real_listdir(descriptor)
        metadata = results_module.os.fstat(descriptor)
        if (
            not swapped and metadata.st_dev == matrix_stat.st_dev
            and metadata.st_ino == matrix_stat.st_ino
        ):
            matrix.rename(original)
            replacement.rename(matrix)
            swapped = True
        return names

    monkeypatch.setattr(results_module.os, "listdir", swap_after_listdir)
    report = validate_results(matrix, evaluation, capacity)
    assert report.matrix_cases == 18
    assert swapped is True


def test_validate_results_allows_only_the_fixed_checkpoint_root_symlink(tmp_path):
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    logical = results_module._CANONICAL_CHECKPOINT_ROOT
    physical = tmp_path / "physical-checkpoints"
    logical.rename(physical)
    logical.symlink_to(physical, target_is_directory=True)

    report = validate_results(matrix, evaluation, capacity)
    assert report.matrix_cases == 18


def test_validate_results_pins_checkpoint_root_once_before_symlink_retarget(
    tmp_path, monkeypatch
):
    """Catches later run snapshots following a retargeted logical root."""
    matrix, evaluation, capacity = _valid_roots(tmp_path)
    logical = results_module._CANONICAL_CHECKPOINT_ROOT
    physical_a = tmp_path / "physical-checkpoints-a"
    physical_b = tmp_path / "physical-checkpoints-b"
    logical.rename(physical_a)
    physical_b.mkdir()
    logical.symlink_to(physical_a, target_is_directory=True)
    real_resolve = Path.resolve
    root_resolves = 0

    def resolve_then_retarget(self, strict=False):
        nonlocal root_resolves
        resolved = real_resolve(self, strict=strict)
        if self == logical:
            root_resolves += 1
            if root_resolves == 1:
                logical.unlink()
                logical.symlink_to(physical_b, target_is_directory=True)
        return resolved

    monkeypatch.setattr(Path, "resolve", resolve_then_retarget)
    report = validate_results(matrix, evaluation, capacity)
    assert report.matrix_cases == 18
    assert root_resolves == 1


def test_publish_validation_outputs_writes_matching_tables_exclusively(tmp_path):
    report = validate_results(*_valid_roots(tmp_path))
    bundle = tmp_path / "validation-report"
    publish_validation_outputs(report, bundle)
    manifest = bundle / "validation-manifest.json"
    summary = bundle / "validation-summary.md"
    payload = json.loads(manifest.read_text())
    markdown = summary.read_text()
    assert payload["stage1_recommendations"]["dora-zero3"] == 2
    assert payload["stage2_recommendations"]["dora-zero3"] == 1
    assert "| Method | ZeRO 1 | ZeRO 2 | ZeRO 3 |" in markdown
    assert "| dora | 2 | 2 | 2 |" in markdown
    assert "| dora | 1 | 1 | 1 |" in markdown
    with pytest.raises(FileExistsError):
        publish_validation_outputs(report, bundle)


@pytest.mark.parametrize("failure", ("write", "fsync"))
def test_second_bundle_file_failure_leaves_final_bundle_absent(
    tmp_path, monkeypatch, failure
):
    report = validate_results(*_valid_roots(tmp_path))
    bundle = tmp_path / "validation-report"
    real_write = results_module.os.write
    real_fsync = results_module.os.fsync
    regular_fsyncs = 0

    def fail_summary_write(descriptor, payload):
        if bytes(payload).startswith(b"# ILVR Four-GPU Validation Summary"):
            raise OSError("simulated summary write failure")
        return real_write(descriptor, payload)

    def fail_summary_fsync(descriptor):
        nonlocal regular_fsyncs
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            regular_fsyncs += 1
            if regular_fsyncs == 2:
                raise OSError("simulated summary fsync failure")
        return real_fsync(descriptor)

    if failure == "write":
        monkeypatch.setattr(results_module.os, "write", fail_summary_write)
    else:
        monkeypatch.setattr(results_module.os, "fsync", fail_summary_fsync)
    with pytest.raises(OSError, match=f"summary {failure} failure"):
        publish_validation_outputs(report, bundle)
    assert not bundle.exists()
    assert list(tmp_path.glob(".*validation-report*")) == []


def test_final_bundle_rename_collision_preserves_foreign_bundle(tmp_path):
    report = validate_results(*_valid_roots(tmp_path))
    bundle = tmp_path / "validation-report"
    bundle.mkdir()
    foreign = bundle / "foreign.txt"
    foreign.write_bytes(b"owned by another publisher")

    with pytest.raises(FileExistsError):
        publish_validation_outputs(report, bundle)

    assert foreign.read_bytes() == b"owned by another publisher"
    assert set(bundle.iterdir()) == {foreign}
    assert list(tmp_path.glob(".*validation-report*")) == []


def test_staging_open_failure_removes_only_private_empty_directory(
    tmp_path, monkeypatch
):
    report = validate_results(*_valid_roots(tmp_path))
    bundle = tmp_path / "validation-report"
    real_open = results_module.os.open

    def fail_staging_open(path, flags, *args, **kwargs):
        if isinstance(path, str) and path.endswith(".staging"):
            raise OSError("simulated staging directory open failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(results_module.os, "open", fail_staging_open)
    with pytest.raises(OSError, match="staging directory open failure"):
        publish_validation_outputs(report, bundle)

    assert not bundle.exists()
    assert list(tmp_path.glob(".*validation-report*")) == []


def test_parent_fsync_failure_leaves_complete_bundle_without_rollback(
    tmp_path, monkeypatch
):
    report = validate_results(*_valid_roots(tmp_path))
    bundle = tmp_path / "validation-report"
    real_fsync = results_module.os.fsync
    directory_fsyncs = 0

    def fail_parent_fsync(descriptor):
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 2:
                raise OSError("simulated parent fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(results_module.os, "fsync", fail_parent_fsync)
    with pytest.raises(OSError, match="durability.*ambiguous"):
        publish_validation_outputs(report, bundle)

    assert directory_fsyncs == 2
    assert {path.name for path in bundle.iterdir()} == {
        "validation-manifest.json",
        "validation-summary.md",
    }
    assert json.loads((bundle / "validation-manifest.json").read_text())[
        "matrix_cases"
    ] == 18
    assert "# ILVR Four-GPU Validation Summary" in (
        bundle / "validation-summary.md"
    ).read_text()
    assert list(tmp_path.glob("*.rollback")) == []
    assert list(tmp_path.glob(".*rollback*")) == []
    assert list(tmp_path.glob(".*capture*")) == []


def test_exception_after_successful_rename_never_cleans_final_bundle(
    tmp_path, monkeypatch
):
    report = validate_results(*_valid_roots(tmp_path))
    bundle = tmp_path / "validation-report"
    real_rename = results_module._rename_noreplace

    def rename_then_interrupt(*args):
        real_rename(*args)
        raise KeyboardInterrupt("simulated post-rename interruption")

    monkeypatch.setattr(results_module, "_rename_noreplace", rename_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="post-rename interruption"):
        publish_validation_outputs(report, bundle)

    assert {path.name for path in bundle.iterdir()} == {
        "validation-manifest.json",
        "validation-summary.md",
    }
    assert json.loads((bundle / "validation-manifest.json").read_text())[
        "matrix_cases"
    ] == 18


def test_foreign_independent_output_names_are_not_part_of_bundle_protocol(tmp_path):
    report = validate_results(*_valid_roots(tmp_path))
    foreign_manifest = tmp_path / "validation-manifest.json"
    foreign_summary = tmp_path / "validation-summary.md"
    foreign_manifest.write_bytes(b"foreign manifest")
    foreign_summary.write_bytes(b"foreign summary")
    bundle = tmp_path / "validation-report"

    publish_validation_outputs(report, bundle)

    assert foreign_manifest.read_bytes() == b"foreign manifest"
    assert foreign_summary.read_bytes() == b"foreign summary"
    assert json.loads((bundle / "validation-manifest.json").read_text())[
        "matrix_cases"
    ] == 18
    assert {foreign_manifest.name, foreign_summary.name, bundle.name} <= {
        path.name for path in tmp_path.iterdir()
    }
    assert list(tmp_path.glob(".*validation-report*")) == []


def test_validate_results_cli_publishes_only_output_root_bundle(
    tmp_path, monkeypatch, capsys
):
    report = ValidationReport(
        matrix_cases=18,
        evaluation_samples=4,
        stage1_recommendations={
            f"{method}-zero{zero_stage}": 2
            for method in METHODS
            for zero_stage in ZERO_STAGES
        },
        stage2_recommendations={
            f"{method}-zero{zero_stage}": 1
            for method in METHODS
            for zero_stage in ZERO_STAGES
        },
    )
    monkeypatch.setattr(results_module, "validate_results", lambda *args: report)
    bundle = tmp_path / "validation-report"

    assert results_module.validate_results_main([
        "--matrix-root", str(tmp_path / "matrix"),
        "--evaluation-root", str(tmp_path / "evaluation"),
        "--capacity-root", str(tmp_path / "capacity"),
        "--output-root", str(bundle),
    ]) == 0

    assert json.loads(capsys.readouterr().out)["matrix_cases"] == 18
    assert {path.name for path in bundle.iterdir()} == {
        "validation-manifest.json",
        "validation-summary.md",
    }
