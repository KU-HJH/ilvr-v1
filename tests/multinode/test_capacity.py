import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest

import src.multinode_artifacts as artifacts_module
from src.multinode_capacity import (
    capacity_candidates,
    classify_failure,
    CapacityEvidence,
    ChildExecution,
    SubprocessCapacityRunner,
    collect_attempt_evidence,
    fallback_candidates,
    run_capacity_probe,
)


def test_capacity_parent_is_reserved_locally_then_remotely_before_child(
    monkeypatch,
):
    result_root = Path(
        "/root/workspace/ILVR/results/task12-capacity/capacity"
    )
    parent_run_id = "capacity-parent"
    destination = result_root / parent_run_id
    events = []
    monkeypatch.setattr(
        artifacts_module,
        "reserve_inference_destination",
        lambda path: events.append(("local", Path(path))) or Path(path),
    )

    def control_runner(command, *, capture_output, text, check, timeout):
        events.append(("remote", tuple(command)))
        assert timeout == 30
        remote_argv = shlex.split(command[6])
        assert remote_argv[:7] == [
            "/usr/bin/docker",
            "exec",
            "--workdir",
            "/root/workspace/ILVR",
            "llm_kd",
            "/usr/bin/bash",
            "-lc",
        ]
        assert shlex.split(remote_argv[7]) == [
            "/usr/bin/env",
            "PYTHONPATH=/root/workspace/ILVR:/root/workspace/ILVR/src",
            "/root/workspace/ILVR/venv/ilvr-venv/bin/python",
            "-m",
            "src.multinode",
            "prepare-evaluation-output",
            "--output-dir",
            str(destination),
            "--run-id",
            parent_run_id,
        ]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    runner = SubprocessCapacityRunner(
        parent_run_id=parent_run_id,
        result_root=result_root,
        control_runner=control_runner,
    )

    assert runner.prepare_parent() == destination
    assert [event[0] for event in events] == ["local", "remote"]


def test_capacity_remote_parent_reservation_failure_prevents_child_launch(
    monkeypatch,
):
    launched = []
    monkeypatch.setattr(
        artifacts_module,
        "reserve_inference_destination",
        lambda path: Path(path),
    )
    runner = SubprocessCapacityRunner(
        parent_run_id="capacity-parent",
        result_root=Path(
            "/root/workspace/ILVR/results/task12-capacity/capacity"
        ),
        training_runner=lambda *_args, **_kwargs: launched.append(True),
        control_runner=lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 5, stdout="", stderr="remote reservation refused"
        ),
    )

    with pytest.raises(RuntimeError, match="remote capacity parent.*refused"):
        runner.prepare_parent()
    assert launched == []


def test_capacity_parent_rejects_unsafe_result_root_before_reservation(
    tmp_path, monkeypatch
):
    reserved = []
    monkeypatch.setattr(
        artifacts_module,
        "reserve_inference_destination",
        lambda path: reserved.append(Path(path)),
    )
    runner = SubprocessCapacityRunner(
        parent_run_id="capacity-parent",
        result_root=tmp_path / "outside-results",
        control_runner=lambda *_args, **_kwargs: pytest.fail(
            "unsafe path reached remote reservation"
        ),
    )

    with pytest.raises(ValueError, match="canonical result root"):
        runner.prepare_parent()
    assert reserved == []


def test_primary_candidates_start_at_two_and_increase():
    assert capacity_candidates(start=2, maximum=8) == (2, 4, 8)


def test_batch_one_requires_candidate_two_cuda_oom():
    with pytest.raises(ValueError, match="candidate 2 CUDA capacity OOM"):
        fallback_candidates(classifications={2: "integration_failure"})
    assert fallback_candidates(classifications={2: "capacity_oom"}) == (1,)


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ("NCCL watchdog timed out", "integration_failure"),
        ("torch.OutOfMemoryError: CUDA out of memory", "capacity_oom"),
        (
            "CUDA out of memory followed by rendezvous connection failure",
            "integration_failure",
        ),
        (
            "CUDA out of memory while output path contains checkpoint-2",
            "capacity_oom",
        ),
        ("DeepSpeed CUDA allocator failed allocation", "capacity_oom"),
        ("host allocator failed allocation", "integration_failure"),
    ],
)
def test_only_cuda_capacity_text_without_integration_markers_is_oom(
    evidence, expected
):
    assert classify_failure(evidence) == expected


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (
            "torch.OutOfMemoryError: CUDA out of memory at checkpoint-2",
            "capacity_oom",
        ),
        ("checkpoint write error: CUDA out of memory", "integration_failure"),
        (
            "DeepSpeed CUDA allocator failed allocation at checkpoint-2",
            "capacity_oom",
        ),
        (
            "checkpoint-2\nunrelated error\ntorch.OutOfMemoryError: CUDA out of memory",
            "capacity_oom",
        ),
        (
            "checkpoint write failed\ntorch.OutOfMemoryError: CUDA out of memory",
            "integration_failure",
        ),
        ("CUDA out of memory\nNCCL watchdog timeout", "integration_failure"),
        ("CUDA out of memory\nloss became nonfinite", "integration_failure"),
    ],
)
def test_conditional_integration_subjects_are_line_local_after_oom_exclusion(
    evidence, expected
):
    assert classify_failure(evidence) == expected


def _write_witness(
    result_root: Path,
    *,
    child_run_id: str,
    method: str,
    zero_stage: int,
    batch: int,
) -> Path:
    producer = result_root / child_run_id / "initial-producer-evidence.json"
    producer.parent.mkdir(parents=True, exist_ok=True)
    producer_bytes = b'{"benchmark_evidence":{"finite_loss":true}}'
    producer.write_bytes(producer_bytes)
    import hashlib

    witness = result_root / child_run_id / "initial-train-witness.json"
    witness.parent.mkdir(parents=True, exist_ok=True)
    witness.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": child_run_id,
                "attempt_id": "initial",
                "stage": "stage1",
                "method": method,
                "zero_stage": zero_stage,
                "batch_per_gpu": batch,
                "world_size": 4,
                "machine_ranks": [0, 1],
                "exit_codes": {"0": 0, "1": 0},
                "checkpoint_path": f"/root/workspace/ILVR/checkpoints/{child_run_id}",
                "finite_loss_evidence": {
                    "status": "validated",
                    "path": str(producer),
                    "sha256": hashlib.sha256(producer_bytes).hexdigest(),
                    "size": len(producer_bytes),
                    "finite_loss": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return witness


class FakeCapacityRunner:
    def __init__(self, result_root, outcomes, *, omit_witness=()):
        self.result_root = Path(result_root)
        self.outcomes = dict(outcomes)
        self.omit_witness = set(omit_witness)
        self.commands = []

    def launch(self, command, *, child_run_id, batch_per_gpu):
        self.commands.append(tuple(command))
        outcome = self.outcomes[batch_per_gpu]
        if outcome == "success":
            if batch_per_gpu not in self.omit_witness:
                _write_witness(
                    self.result_root,
                    child_run_id=child_run_id,
                    method="lora",
                    zero_stage=3,
                    batch=batch_per_gpu,
                )
            return ChildExecution(returncode=0, evidence=())
        log_path = (
            Path("/root/workspace/ILVR/logs/zero-stage-compatibility")
            / child_run_id
            / "initial-train.log"
        )
        text = (
            "torch.OutOfMemoryError: CUDA out of memory"
            if outcome == "capacity_oom"
            else "NCCL collective mismatch"
        )
        return ChildExecution(
            returncode=1,
            evidence=(
                CapacityEvidence(node="local", path=log_path, text=text),
            ),
        )


def _run(tmp_path, runner, *, ports=(29631, 29632, 29633, 29634), dry_run=False):
    return run_capacity_probe(
        run_id="capacity-parent",
        zero_stage=3,
        tuning_method="lora",
        ports=ports,
        result_root=tmp_path / "results",
        stage_arguments=("--model", "tiny", "--max-steps", "2"),
        runner=runner,
        python_executable="/repo/venv/ilvr-venv/bin/python",
        dry_run=dry_run,
    )


def _option(command, name):
    return command[command.index(name) + 1]


def test_primary_oom_stops_and_recommends_last_valid_witness(tmp_path):
    result_root = tmp_path / "results"
    runner = FakeCapacityRunner(
        result_root,
        {2: "success", 4: "success", 8: "capacity_oom"},
    )

    payload = _run(tmp_path, runner)

    assert [attempt["batch_per_gpu"] for attempt in payload["attempts"]] == [2, 4, 8]
    assert [attempt["child_run_id"] for attempt in payload["attempts"]] == [
        "capacity-parent-batch2",
        "capacity-parent-batch4",
        "capacity-parent-batch8",
    ]
    assert [attempt["main_process_port"] for attempt in payload["attempts"]] == [
        29631,
        29632,
        29633,
    ]
    assert [attempt["status"] for attempt in payload["attempts"]] == [
        "success",
        "success",
        "capacity_oom",
    ]
    assert payload["recommended_batch_per_gpu"] == 4
    assert [int(_option(command, "--micro-batch")) for command in runner.commands] == [
        2,
        4,
        8,
    ]
    assert all(command[1:4] == ("-m", "src.multinode", "train-stage1") for command in runner.commands)
    published = result_root / "capacity-parent" / "capacity.json"
    assert json.loads(published.read_text(encoding="utf-8")) == payload
    assert "text" not in published.read_text(encoding="utf-8")


def test_candidate_two_oom_runs_batch_one_exactly_once_on_fourth_port(tmp_path):
    result_root = tmp_path / "results"
    runner = FakeCapacityRunner(result_root, {2: "capacity_oom", 1: "success"})

    payload = _run(tmp_path, runner)

    assert [attempt["batch_per_gpu"] for attempt in payload["attempts"]] == [2, 1]
    assert [attempt["main_process_port"] for attempt in payload["attempts"]] == [
        29631,
        29634,
    ]
    assert payload["recommended_batch_per_gpu"] == 1
    assert len(runner.commands) == 2


def test_integration_failure_stops_immediately_without_recommendation(tmp_path):
    result_root = tmp_path / "results"
    runner = FakeCapacityRunner(
        result_root,
        {2: "success", 4: "integration_failure"},
    )

    payload = _run(tmp_path, runner)

    assert [attempt["batch_per_gpu"] for attempt in payload["attempts"]] == [2, 4]
    assert payload["attempts"][-1]["status"] == "integration_failure"
    assert payload["recommended_batch_per_gpu"] is None
    assert len(runner.commands) == 2


def test_zero_exit_without_exact_valid_child_witness_is_integration_failure(tmp_path):
    result_root = tmp_path / "results"
    runner = FakeCapacityRunner(
        result_root,
        {2: "success"},
        omit_witness={2},
    )

    payload = _run(tmp_path, runner)

    assert payload["attempts"][0]["status"] == "integration_failure"
    assert payload["recommended_batch_per_gpu"] is None
    assert len(runner.commands) == 1


def test_zero_exit_without_witness_cannot_use_oom_text_to_unlock_fallback(tmp_path):
    class InvalidSuccessRunner:
        def __init__(self):
            self.commands = []

        def launch(self, command, *, child_run_id, batch_per_gpu):
            self.commands.append(tuple(command))
            return ChildExecution(
                returncode=0,
                evidence=(
                    CapacityEvidence(
                        node="coordinator",
                        path=Path("/root/workspace/ILVR/logs/capacity/child.stderr"),
                        text="torch.OutOfMemoryError: CUDA out of memory",
                    ),
                ),
            )

    runner = InvalidSuccessRunner()
    payload = _run(tmp_path, runner)

    assert [attempt["batch_per_gpu"] for attempt in payload["attempts"]] == [2]
    assert payload["attempts"][0]["status"] == "integration_failure"
    assert payload["recommended_batch_per_gpu"] is None


def test_zero_exit_with_stale_finite_loss_reference_is_integration_failure(tmp_path):
    result_root = tmp_path / "results"
    runner = FakeCapacityRunner(result_root, {2: "success"})

    original_launch = runner.launch

    def launch_with_stale_evidence(command, *, child_run_id, batch_per_gpu):
        execution = original_launch(
            command,
            child_run_id=child_run_id,
            batch_per_gpu=batch_per_gpu,
        )
        producer = result_root / child_run_id / "initial-producer-evidence.json"
        producer.write_text("changed", encoding="utf-8")
        return execution

    runner.launch = launch_with_stale_evidence
    payload = _run(tmp_path, runner)

    assert payload["attempts"][0]["status"] == "integration_failure"
    assert payload["recommended_batch_per_gpu"] is None


def _mutate_witness(payload, case):
    if case == "top-extra":
        payload["unexpected"] = "stale"
    elif case == "top-missing":
        del payload["checkpoint_path"]
    elif case == "schema-float":
        payload["schema_version"] = 1.0
    elif case == "schema-bool":
        payload["schema_version"] = True
    elif case == "zero-bool":
        payload["zero_stage"] = True
    elif case == "batch-float":
        payload["batch_per_gpu"] = 2.0
    elif case == "world-float":
        payload["world_size"] = 4.0
    elif case == "ranks-floats":
        payload["machine_ranks"] = [0.0, 1.0]
    elif case == "ranks-bool":
        payload["machine_ranks"] = [0, True]
    elif case == "ranks-extra":
        payload["machine_ranks"] = [0, 1, 2]
    elif case == "exit-float":
        payload["exit_codes"] = {"0": 0.0, "1": 0}
    elif case == "exit-extra":
        payload["exit_codes"] = {"0": 0, "1": 0, "2": 0}
    elif case == "finite-extra":
        payload["finite_loss_evidence"]["unexpected"] = True
    elif case == "finite-size-float":
        payload["finite_loss_evidence"]["size"] = float(
            payload["finite_loss_evidence"]["size"]
        )
    else:
        raise AssertionError(f"unknown witness mutation: {case}")


@pytest.mark.parametrize(
    "case",
    (
        "top-extra",
        "top-missing",
        "schema-float",
        "schema-bool",
        "zero-bool",
        "batch-float",
        "world-float",
        "ranks-floats",
        "ranks-bool",
        "ranks-extra",
        "exit-float",
        "exit-extra",
        "finite-extra",
        "finite-size-float",
    ),
)
def test_malformed_child_witness_never_authorizes_recommendation(tmp_path, case):
    result_root = tmp_path / "results"
    runner = FakeCapacityRunner(result_root, {2: "success"})
    original_launch = runner.launch

    def launch_with_malformed_witness(command, *, child_run_id, batch_per_gpu):
        execution = original_launch(
            command,
            child_run_id=child_run_id,
            batch_per_gpu=batch_per_gpu,
        )
        witness = result_root / child_run_id / "initial-train-witness.json"
        payload = json.loads(witness.read_text(encoding="utf-8"))
        _mutate_witness(payload, case)
        witness.write_text(json.dumps(payload), encoding="utf-8")
        return execution

    runner.launch = launch_with_malformed_witness
    payload = _run(tmp_path, runner)

    assert [attempt["batch_per_gpu"] for attempt in payload["attempts"]] == [2]
    assert payload["attempts"][0]["status"] == "integration_failure"
    assert payload["recommended_batch_per_gpu"] is None


def test_capacity_report_is_exclusive(tmp_path):
    result_root = tmp_path / "results"
    first = FakeCapacityRunner(result_root, {2: "capacity_oom", 1: "capacity_oom"})
    _run(tmp_path, first)
    second = FakeCapacityRunner(result_root, {2: "capacity_oom", 1: "capacity_oom"})

    with pytest.raises(FileExistsError, match="capacity.json"):
        _run(tmp_path, second)
    assert second.commands == []


def test_dry_run_has_no_launch_or_publication(tmp_path):
    result_root = tmp_path / "results"
    runner = FakeCapacityRunner(result_root, {})

    payload = _run(tmp_path, runner, dry_run=True)

    assert [attempt["batch_per_gpu"] for attempt in payload["attempts"]] == [2, 4, 8]
    assert runner.commands == []
    assert not result_root.exists()


@pytest.mark.parametrize(
    "ports",
    [
        (29631, 29632, 29633),
        (29631, 29631, 29633, 29634),
        (22, 29632, 29633, 29634),
    ],
)
def test_probe_requires_four_distinct_explicit_valid_ports(tmp_path, ports):
    runner = FakeCapacityRunner(tmp_path / "results", {})
    with pytest.raises(ValueError, match="four distinct|between 1024"):
        _run(tmp_path, runner, ports=ports, dry_run=True)


def test_exact_child_log_classification_uses_only_bounded_tail(tmp_path):
    run_id = "capacity-parent-batch2"
    stage_log = tmp_path / "stage-logs" / run_id / "initial-train.log"
    stage_log.parent.mkdir(parents=True)
    stage_log.write_bytes(
        b"NCCL collective mismatch\n"
        + b"x" * (300 * 1024)
        + b"\ntorch.OutOfMemoryError: CUDA out of memory\n"
    )
    status = tmp_path / "node-logs" / run_id / "initial" / "exit-rank1.json"
    status.parent.mkdir(parents=True)
    status.write_text('{"exit_code":1}\n', encoding="utf-8")

    evidence = collect_attempt_evidence(
        run_id=run_id,
        node="local",
        machine_rank=1,
        result_root=tmp_path / "results",
        log_root=tmp_path / "node-logs",
        stage_log_root=tmp_path / "stage-logs",
    )

    assert {item.path for item in evidence} == {stage_log, status}
    assert all(len(item.text.encode("utf-8")) <= 256 * 1024 for item in evidence)
    assert classify_failure("\n".join(item.text for item in evidence)) == "capacity_oom"


def test_subprocess_runner_rejects_remote_evidence_outside_exact_child(tmp_path):
    training_commands = []
    control_commands = []

    def training_runner(command, *, stdout, stderr, text, check):
        training_commands.append(tuple(command))
        stderr.write("torch.OutOfMemoryError: CUDA out of memory\n")
        stderr.flush()
        return subprocess.CompletedProcess(command, 1)

    def control_runner(command, *, capture_output, text, check, timeout):
        control_commands.append(tuple(command))
        assert tuple(command[:6]) == (
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "korea@10.1.102.41",
        )
        remote_argv = shlex.split(command[6])
        assert remote_argv[:5] == [
            "/usr/bin/docker",
            "exec",
            "llm_kd",
            "/usr/bin/bash",
            "-lc",
        ]
        assert shlex.split(remote_argv[5]) == [
            "env",
            "PYTHONPATH=/root/workspace/ILVR:/root/workspace/ILVR/src",
            "/root/workspace/ILVR/venv/ilvr-venv/bin/python",
            "-m",
            "src.multinode",
            "capacity-evidence-show",
            "--run-id",
            "capacity-parent-batch2",
            "--node",
            "remote",
            "--machine-rank",
            "0",
            "--result-root",
            str(tmp_path / "results"),
        ]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                [
                    {
                        "node": "remote",
                        "path": "/root/workspace/ILVR/logs/foreign/train.log",
                        "text": "CUDA out of memory",
                    }
                ]
            ),
            stderr="",
        )

    runner = SubprocessCapacityRunner(
        parent_run_id="capacity-parent",
        result_root=tmp_path / "results",
        training_runner=training_runner,
        control_runner=control_runner,
        log_root=tmp_path / "capacity-logs",
    )
    child = runner.launch(
        ("/repo/python", "-m", "src.multinode", "train-stage1"),
        child_run_id="capacity-parent-batch2",
        batch_per_gpu=2,
    )

    assert len(training_commands) == 1
    assert len(control_commands) == 1
    assert classify_failure("\n".join(item.text for item in child.evidence)) == (
        "integration_failure"
    )
    assert all(item.path.is_absolute() for item in child.evidence)


def test_subprocess_runner_records_child_launch_error_as_integration(tmp_path):
    def training_runner(command, **kwargs):
        raise OSError("executable unavailable")

    def control_runner(*args, **kwargs):
        raise AssertionError("a child that never launched has no node logs to query")

    runner = SubprocessCapacityRunner(
        parent_run_id="capacity-parent",
        result_root=tmp_path / "results",
        training_runner=training_runner,
        control_runner=control_runner,
        log_root=tmp_path / "capacity-logs",
    )

    child = runner.launch(
        ("/missing/python", "-m", "src.multinode", "train-stage1"),
        child_run_id="capacity-parent-batch2",
        batch_per_gpu=2,
    )

    assert child.returncode != 0
    assert classify_failure("\n".join(item.text for item in child.evidence)) == (
        "integration_failure"
    )


def test_wrapper_help_and_dry_run_are_local_and_do_not_publish():
    repository = Path(__file__).resolve().parents[2]
    wrapper = repository / "run_multinode_stage1_batch_probe.sh"
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    help_result = subprocess.run(
        (wrapper, "--help"),
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert help_result.returncode == 0, help_result.stderr

    run_id = "capacity-stage1-dry-run-test"
    parent_result = Path("/root/workspace/ILVR/results") / run_id
    assert not parent_result.exists()
    dry_result = subprocess.run(
        (
            wrapper,
            "--run-id",
            run_id,
            "--zero-stage",
            "3",
            "--tuning-method",
            "lora",
            "--main-process-port",
            "29631",
            "--main-process-port",
            "29632",
            "--main-process-port",
            "29633",
            "--main-process-port",
            "29634",
            "--dry-run",
            "--model",
            "tiny/model",
            "--max-steps",
            "2",
        ),
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert dry_result.returncode == 0, dry_result.stderr
    payload = json.loads(dry_result.stdout)
    assert [attempt["batch_per_gpu"] for attempt in payload["attempts"]] == [2, 4, 8]
    assert payload["fallback"]["batch_per_gpu"] == 1
    assert not parent_result.exists()


def test_task12_capacity_invocation_shape_dry_runs_without_publication():
    repository = Path(__file__).resolve().parents[2]
    wrapper = repository / "run_multinode_stage1_batch_probe.sh"
    run_id = "task12-stage1-capacity-lora-zero3-dry"
    capacity_root = Path("/root/workspace/ILVR/results/task12-dry/capacity")
    parent_result = capacity_root / run_id
    assert not parent_result.exists()

    completed = subprocess.run(
        (
            wrapper,
            "--run-id",
            run_id,
            "--result-root",
            str(capacity_root),
            "--zero-stage",
            "3",
            "--tuning-method",
            "lora",
            "--main-process-port",
            "30032",
            "--main-process-port",
            "30033",
            "--main-process-port",
            "30034",
            "--main-process-port",
            "30035",
            "--dry-run",
            "--model",
            "/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct",
            "--data-path",
            "/root/workspace/ILVR/data/multinode-capacity-subset.jsonl",
            "--image-root",
            "/root/workspace/ILVR/data/images_comt",
            "--max-steps",
            "1",
        ),
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert [
        (attempt["batch_per_gpu"], attempt["main_process_port"])
        for attempt in payload["attempts"]
    ] == [(2, 30032), (4, 30033), (8, 30034)]
    assert payload["fallback"]["main_process_port"] == 30035
    assert not parent_result.exists()


def test_stage1_coordinator_writes_witness_for_run_owned_checkpoint_root(
    tmp_path, monkeypatch, capsys
):
    import src.multinode as multinode_module

    repository = tmp_path / "repository"
    result_root = repository / "results"
    checkpoint_root = repository / "checkpoints"
    child_run_id = "capacity-parent-batch2"
    plan = SimpleNamespace(
        run_id=child_run_id,
        attempt_id="initial",
        stage="stage1",
        tuning_method="lora",
        zero_stage=3,
        world_size=4,
        cluster=SimpleNamespace(remote_repo_root=repository),
    )
    commands = SimpleNamespace(
        command_digests=("a" * 64, "b" * 64),
        finite_loss_evidence_path=(
            result_root / child_run_id / "initial-producer-evidence.json"
        ),
    )
    monkeypatch.setattr(multinode_module, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        multinode_module, "validate_stage1_arguments", lambda values: tuple(values)
    )
    monkeypatch.setattr(multinode_module, "build_launch_plan", lambda **kwargs: plan)
    monkeypatch.setattr(
        multinode_module, "_command_stdout", lambda *args, **kwargs: "c" * 40
    )
    monkeypatch.setattr(
        multinode_module, "build_owned_tmux_commands", lambda *args, **kwargs: commands
    )
    monkeypatch.setattr(
        multinode_module,
        "build_preflight_commands",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        multinode_module, "collect_node_snapshot", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        multinode_module,
        "validate_preflight",
        lambda *args, **kwargs: SimpleNamespace(commit="c" * 40),
    )
    monkeypatch.setattr(
        multinode_module, "SubprocessCoordinatorRunner", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(multinode_module, "start_rank0_then_rank1", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        multinode_module, "monitor_owned_jobs", lambda *args, **kwargs: {0: 0, 1: 0}
    )

    def collect(*args, checkpoint_path, **kwargs):
        checkpoint_path.mkdir(parents=True)
        (checkpoint_path / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(multinode_module, "collect_checkpoint_artifacts", collect)
    monkeypatch.setattr(
        multinode_module,
        "_checkpoint_artifact_identity",
        lambda *args, **kwargs: {"run_id": child_run_id},
    )

    def write_witness(
        launch_plan, *, result_root, checkpoint_path, batch_per_gpu, **kwargs
    ):
        if checkpoint_path.name != launch_plan.run_id:
            raise ValueError("training witness checkpoint must be owned by the run ID")
        witness = Path(result_root) / launch_plan.run_id / "initial-train-witness.json"
        witness.parent.mkdir(parents=True)
        witness.write_text("{}", encoding="utf-8")
        return witness

    monkeypatch.setattr(multinode_module, "write_training_witness", write_witness)

    result = multinode_module._main(
        [
            "train-stage1",
            "--run-id",
            child_run_id,
            "--zero-stage",
            "3",
            "--main-process-port",
            "29631",
            "--result-root",
            str(result_root),
            "--method",
            "lora",
            "--max-steps",
            "2",
            "--micro-batch",
            "2",
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["training_witness"].endswith(
        "initial-train-witness.json"
    )
    assert (checkpoint_root / child_run_id / "checkpoint-2").is_dir()
