import json
import os
import shlex
import stat
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from src.multinode import build_launch_plan
from src.multinode_runtime import (
    OwnershipRecord,
    SubprocessCoordinatorRunner,
    build_owned_tmux_commands,
    canonical_command_digest,
    monitor_owned_jobs,
    serialize_launch_plan,
    start_rank0_then_rank1,
    stop_owned_peer,
    validate_owned_process,
    validate_ownership,
    validate_stage1_arguments,
    validated_stage2_environment,
    write_ownership,
    write_training_witness,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def plan(stage="stage1", method="lora"):
    return build_launch_plan(
        stage=stage,
        tuning_method=method,
        zero_stage=3,
        run_id=f"{stage}-{method}-zero3-four-gpu",
        main_process_port=29631,
    )


def record(*, rank=0, start_ticks=9981, command=("./run_stage1_training.sh",)):
    return OwnershipRecord.create(
        run_id="stage1-lora-zero3-four-gpu",
        attempt_id="initial",
        machine_rank=rank,
        pid=4312 + rank,
        process_start_ticks=start_ticks,
        commit="a" * 40,
        command=command,
        environment={"ILVR_MACHINE_RANK": str(rank)},
    )


class FakeRunner:
    def __init__(self):
        self.events = []
        self.remote_ready = True
        self.observed_start_ticks = 9981
        self.statuses = {0: None, 1: None}
        self.ownership = {
            0: record(rank=0),
            1: record(rank=1),
        }

    def prepare_plan(self, launch_plan):
        assert launch_plan.run_id == "stage1-lora-zero3-four-gpu"

    def start_remote(self, command):
        assert command[:6] == (
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "korea@10.1.102.41",
        )
        self.events.append("remote-tmux-start")

    def rank0_is_ready(self, launch_plan):
        self.events.append("remote-ready")
        return self.remote_ready

    def start_local(self, command):
        assert command[:5] == (
            "tmux",
            "new-session",
            "-d",
            "-s",
            "ilvr-stage1-lora-zero3-four-gpu-initial-rank1",
        )
        self.events.append("local-tmux-start")

    def sleep(self, seconds):
        assert seconds == pytest.approx(0.25)

    def read_ownership(self, machine_rank):
        return self.ownership[machine_rank]

    def process_start_ticks(self, ownership):
        return self.observed_start_ticks if ownership.machine_rank == 0 else 9981

    def command_digest(self, ownership):
        return ownership.command_digest

    def commit(self, ownership):
        return ownership.commit

    def signal(self, ownership):
        self.events.append("signal")

    def poll_exit(self, machine_rank):
        return self.statuses[machine_rank]


@pytest.fixture
def fake_runner():
    return FakeRunner()


def test_ownership_requires_exact_run_pid_start_commit_and_command():
    ownership = OwnershipRecord.create(
        run_id="stage1-lora-zero3-four-gpu",
        attempt_id="initial",
        machine_rank=0,
        pid=4312,
        process_start_ticks=9981,
        commit="a" * 40,
        command=("./run_stage1_training.sh", "--method", "lora"),
    )
    assert validate_ownership(
        ownership,
        expected_run_id=ownership.run_id,
        expected_attempt_id="initial",
        expected_machine_rank=0,
        expected_pid=4312,
        expected_commit="a" * 40,
        expected_command_digest=ownership.command_digest,
        observed_start_ticks=9981,
    )
    with pytest.raises(ValueError, match="start time"):
        validate_ownership(
            ownership,
            expected_run_id=ownership.run_id,
            expected_attempt_id="initial",
            expected_machine_rank=0,
            expected_pid=4312,
            expected_commit="a" * 40,
            expected_command_digest=ownership.command_digest,
            observed_start_ticks=9982,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"run_id": "other-run"}, "run ID"),
        ({"attempt_id": "resume-one"}, "attempt ID"),
        ({"machine_rank": 1}, "machine rank"),
        ({"pid": 9999}, "PID"),
        ({"commit": "b" * 40}, "commit"),
        ({"command_digest": "b" * 64}, "command digest"),
    ],
)
def test_ownership_rejects_every_foreign_identity_field(changes, message):
    ownership = replace(record(), **changes)

    with pytest.raises(ValueError, match=message):
        validate_ownership(
            ownership,
            expected_run_id="stage1-lora-zero3-four-gpu",
            expected_attempt_id="initial",
            expected_machine_rank=0,
            expected_pid=4312,
            expected_commit="a" * 40,
            expected_command_digest=record().command_digest,
            observed_start_ticks=9981,
        )


def test_ownership_rejects_foreign_schema_at_decode_boundary():
    payload = record().to_json()
    payload["schema_version"] = 2

    with pytest.raises(ValueError, match="schema"):
        OwnershipRecord.from_json(payload)


def test_command_digest_binds_argv_boundaries_and_environment_values():
    baseline = canonical_command_digest(
        ("launcher", "literal value;$(touch BAD)"),
        {"ILVR_MACHINE_RANK": "0", "TOKENIZERS_PARALLELISM": "false"},
    )
    assert baseline != canonical_command_digest(
        ("launcher", "literal", "value;$(touch BAD)"),
        {"ILVR_MACHINE_RANK": "0", "TOKENIZERS_PARALLELISM": "false"},
    )
    assert baseline != canonical_command_digest(
        ("launcher", "literal value;$(touch BAD)"),
        {"ILVR_MACHINE_RANK": "1", "TOKENIZERS_PARALLELISM": "false"},
    )


def test_write_ownership_is_exact_mode_0600_and_refuses_replacement(tmp_path):
    ownership = record()
    path = tmp_path / ownership.run_id / ownership.attempt_id / "ownership-rank0.json"

    write_ownership(path, ownership)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text(encoding="utf-8")) == ownership.to_json()
    assert not list(path.parent.glob(".*.tmp"))
    with pytest.raises(FileExistsError):
        write_ownership(path, ownership)


def test_plan_is_attempt_scoped_exclusive_and_digest_bound(tmp_path):
    launch_plan = plan()

    serialized = serialize_launch_plan(launch_plan, log_root=tmp_path)

    assert serialized.path == (
        tmp_path / launch_plan.run_id / "initial" / "launch-plan.json"
    )
    assert serialized.size == serialized.path.stat().st_size
    assert len(serialized.sha256) == 64
    assert stat.S_IMODE(serialized.path.stat().st_mode) == 0o600
    payload = json.loads(serialized.path.read_text(encoding="utf-8"))
    assert payload["run_id"] == launch_plan.run_id
    assert payload["attempt_id"] == "initial"
    with pytest.raises(FileExistsError):
        serialize_launch_plan(launch_plan, log_root=tmp_path)


def test_tmux_commands_use_fixed_sessions_and_quote_shell_metacharacters():
    marker = Path("COORDINATOR_INJECTION")
    launch_plan = plan()
    commands = build_owned_tmux_commands(
        launch_plan,
        commit="a" * 40,
        stage_arguments=("--model", "literal;$(touch COORDINATOR_INJECTION)"),
        stage_environment={"TOKEN": "value with spaces;$(touch ENV_INJECTION)"},
    )

    assert commands.local[:5] == (
        "tmux",
        "new-session",
        "-d",
        "-s",
        f"ilvr-{launch_plan.run_id}-initial-rank1",
    )
    assert commands.local[5:8] == ("--", "bash", "-lc")
    shell_tokens = shlex.split(commands.local[8].split("; exit_status=$?", 1)[0])
    assert "literal;$(touch COORDINATOR_INJECTION)" in shell_tokens
    assert "TOKEN=value with spaces;$(touch ENV_INJECTION)" in shell_tokens
    assert not marker.exists()
    assert commands.remote[:6] == (
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "korea@10.1.102.41",
    )


def test_coordinator_starts_remote_rank_zero_before_local_rank_one(fake_runner):
    start_rank0_then_rank1(plan(), fake_runner, ready_timeout_seconds=30)
    assert fake_runner.events[:3] == [
        "remote-tmux-start",
        "remote-ready",
        "local-tmux-start",
    ]


def test_rank_zero_timeout_prevents_rank_one_start(fake_runner):
    fake_runner.remote_ready = False
    with pytest.raises(TimeoutError, match="rank 0 readiness"):
        start_rank0_then_rank1(
            plan(), fake_runner, ready_timeout_seconds=0.5
        )
    assert "local-tmux-start" not in fake_runner.events


def test_dry_run_starts_neither_node(fake_runner):
    commands = start_rank0_then_rank1(plan(), fake_runner, dry_run=True)
    assert len(commands) == 2
    assert fake_runner.events == []


def test_peer_stop_requires_exact_ownership(fake_runner):
    fake_runner.observed_start_ticks = 9982
    with pytest.raises(ValueError, match="start time"):
        stop_owned_peer(record(start_ticks=9981), fake_runner)
    assert "signal" not in fake_runner.events


def test_monitor_stops_live_peer_only_after_exact_revalidation(fake_runner):
    fake_runner.statuses = {0: 7, 1: None}

    exits = monitor_owned_jobs(
        plan(),
        fake_runner,
        expected_commit="a" * 40,
        expected_command_digests={
            0: fake_runner.ownership[0].command_digest,
            1: fake_runner.ownership[1].command_digest,
        },
    )

    assert exits == {0: 7, 1: -15}
    assert fake_runner.events == ["signal"]


def test_stage1_allowlist_rejects_coordinator_owned_and_unknown_flags():
    assert validate_stage1_arguments(
        ("--method", "lora", "--model", "literal;$(touch BAD)")
    ) == ("--method", "lora", "--model", "literal;$(touch BAD)")
    with pytest.raises(ValueError, match="coordinator-owned"):
        validate_stage1_arguments(("--deepspeed-config", "/tmp/foreign.json"))
    with pytest.raises(ValueError, match="not an approved Stage 1"):
        validate_stage1_arguments(("--unknown-option", "value"))


def test_stage2_forwards_only_allowlisted_environment_and_safe_arguments():
    environment = validated_stage2_environment(
        {
            "STAGE1_CHECKPOINT": "/models/stage1;literal",
            "DATA_PATH": "/data/train.jsonl",
            "IMAGE_ROOT": "/data/images",
            "TUNING_METHOD": "dora",
            "LORA_R": "16",
            "STAGE2_DRY_RUN": "true",
            "ILVR_MACHINE_RANK": "99",
            "UNRELATED_SECRET": "do-not-forward",
        },
        output_dir="/root/workspace/ILVR/checkpoints/stage2-dora-zero3-four-gpu",
        log_file="/root/workspace/ILVR/logs/zero-stage-compatibility/stage2-dora-zero3-four-gpu/train.log",
        deepspeed_config="/root/workspace/ILVR/configs/deepspeed_zero3.json",
    )

    assert environment["STAGE1_CHECKPOINT"] == "/models/stage1;literal"
    assert environment["LORA_R"] == "16"
    assert environment["OUTPUT_DIR"].endswith("stage2-dora-zero3-four-gpu")
    assert "STAGE2_DRY_RUN" not in environment
    assert "ILVR_MACHINE_RANK" not in environment
    assert "UNRELATED_SECRET" not in environment


def test_success_witness_references_real_finite_loss_evidence_exclusively(tmp_path):
    producer = tmp_path / "producer-evidence.json"
    producer.write_text(
        json.dumps({"benchmark_evidence": {"finite_loss": True}}),
        encoding="utf-8",
    )
    launch_plan = plan()
    checkpoint = tmp_path / "checkpoints" / launch_plan.run_id
    checkpoint.mkdir(parents=True)

    witness = write_training_witness(
        launch_plan,
        result_root=tmp_path / "results",
        exit_codes={0: 0, 1: 0},
        checkpoint_path=checkpoint,
        batch_per_gpu=2,
        finite_loss_evidence_path=producer,
    )

    payload = json.loads(witness.read_text(encoding="utf-8"))
    assert payload["stage"] == "stage1"
    assert payload["method"] == "lora"
    assert payload["zero_stage"] == 3
    assert payload["batch_per_gpu"] == 2
    assert payload["world_size"] == 4
    assert payload["machine_ranks"] == [0, 1]
    assert payload["exit_codes"] == {"0": 0, "1": 0}
    assert payload["checkpoint_path"].endswith(launch_plan.run_id)
    assert payload["finite_loss_evidence"]["path"] == str(producer)
    assert payload["finite_loss_evidence"]["finite_loss"] is True
    with pytest.raises(FileExistsError):
        write_training_witness(
            launch_plan,
            result_root=tmp_path / "results",
            exit_codes={0: 0, 1: 0},
            checkpoint_path=checkpoint,
            batch_per_gpu=2,
            finite_loss_evidence_path=producer,
        )


def test_witness_never_invents_missing_finite_loss_evidence(tmp_path):
    producer = tmp_path / "producer-evidence.json"
    producer.write_text(
        json.dumps({"benchmark_evidence": {"finite_loss": False}}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoints/run"
    checkpoint.mkdir(parents=True)

    with pytest.raises(ValueError, match="finite loss"):
        write_training_witness(
            plan(),
            result_root=tmp_path / "results",
            exit_codes={0: 0, 1: 0},
            checkpoint_path=checkpoint,
            batch_per_gpu=2,
            finite_loss_evidence_path=producer,
        )


def test_witness_rejects_a_missing_checkpoint_reference(tmp_path):
    with pytest.raises(ValueError, match="checkpoint"):
        write_training_witness(
            plan(),
            result_root=tmp_path / "results",
            exit_codes={0: 0, 1: 0},
            checkpoint_path=tmp_path / "missing-checkpoint",
            batch_per_gpu=2,
        )


@pytest.mark.parametrize(
    "wrapper", ["run_multinode_stage1_training.sh", "run_multinode_stage2_training.sh"]
)
def test_wrapper_help_is_local_and_side_effect_free(wrapper, tmp_path):
    before = set(tmp_path.iterdir())
    completed = subprocess.run(
        (REPO_ROOT / wrapper, "--help"),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert completed.returncode == 0, completed.stderr
    assert "--run-id" in completed.stdout
    assert set(tmp_path.iterdir()) == before


def test_stage1_wrapper_dry_run_builds_both_nodes_without_remote_calls(tmp_path):
    marker = tmp_path / "DRY_RUN_INJECTION"
    completed = subprocess.run(
        (
            REPO_ROOT / "run_multinode_stage1_training.sh",
            "--run-id",
            "stage1-lora-zero3-four-gpu",
            "--zero-stage",
            "3",
            "--main-process-port",
            "29631",
            "--dry-run",
            "--method",
            "lora",
            "--model",
            f"literal;$(touch {marker})",
            "--data-path",
            "/data/train.jsonl",
            "--image-root",
            "/data/images",
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["plan"]["world_size"] == 4
    assert payload["remote"][:6] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "korea@10.1.102.41",
    ]
    assert payload["local"][:5] == [
        "tmux",
        "new-session",
        "-d",
        "-s",
        "ilvr-stage1-lora-zero3-four-gpu-initial-rank1",
    ]
    assert not marker.exists()


def test_resume_requires_a_fresh_noninitial_attempt_id():
    completed = subprocess.run(
        (
            REPO_ROOT / "run_multinode_stage1_training.sh",
            "--run-id",
            "stage1-lora-zero3-four-gpu",
            "--zero-stage",
            "3",
            "--main-process-port",
            "29631",
            "--dry-run",
            "--method",
            "lora",
            "--model",
            "model",
            "--data-path",
            "/data/train.jsonl",
            "--image-root",
            "/data/images",
            "--resume",
            "/root/workspace/ILVR/checkpoints/stage1-lora-zero3-four-gpu/checkpoint-2",
        ),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert completed.returncode == 2
    assert "fresh --attempt-id" in completed.stderr


def test_node_run_publishes_exact_ownership_before_exec(tmp_path, monkeypatch):
    ownership_path = tmp_path / "logs/run/initial/ownership-rank0.json"
    observed = {}

    def fake_exec(program, command, environment):
        observed["program"] = program
        observed["command"] = tuple(command)
        observed["environment"] = environment
        observed["ownership"] = json.loads(ownership_path.read_text(encoding="utf-8"))
        raise RuntimeError("exec boundary reached")

    monkeypatch.setattr("src.multinode.os.getpid", lambda: 4312)
    monkeypatch.setattr("src.multinode._proc_start_ticks", lambda pid: 9981)
    monkeypatch.setattr("src.multinode.os.execvpe", fake_exec)

    with pytest.raises(RuntimeError, match="exec boundary"):
        from src.multinode import _node_run

        _node_run(
            [
                "--run-id",
                "stage1-lora-zero3-four-gpu",
                "--attempt-id",
                "initial",
                "--machine-rank",
                "0",
                "--commit",
                "a" * 40,
                "--ownership-path",
                str(ownership_path),
                "--env",
                "ILVR_MACHINE_RANK=0",
                "--",
                "/usr/bin/bash",
                "/root/workspace/ILVR/run_stage1_training.sh",
                "--model",
                "literal;$(touch BAD)",
            ]
        )

    assert observed["program"] == "/usr/bin/bash"
    assert observed["command"][-1] == "literal;$(touch BAD)"
    assert observed["environment"]["ILVR_MACHINE_RANK"] == "0"
    assert observed["ownership"]["pid"] == 4312
    assert observed["ownership"]["process_start_ticks"] == 9981
    assert stat.S_IMODE(ownership_path.stat().st_mode) == 0o600


class FakeCommandRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, command, *, input_bytes=None):
        self.calls.append((tuple(command), input_bytes))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_subprocess_runner_transfers_only_attempt_plan_with_size_and_sha(tmp_path):
    launch_plan = plan()
    commands = build_owned_tmux_commands(launch_plan, commit="a" * 40)
    command_runner = FakeCommandRunner()
    runner = SubprocessCoordinatorRunner(
        launch_plan,
        commands=commands,
        commit="a" * 40,
        log_root=tmp_path,
        command_runner=command_runner,
    )

    runner.prepare_plan(launch_plan)

    assert runner.serialized_plan is not None
    assert runner.serialized_plan.path.parent == (
        tmp_path / launch_plan.run_id / launch_plan.attempt_id
    )
    assert len(command_runner.calls) == 2
    mkdir_command, transfer_command = command_runner.calls
    assert mkdir_command[1] is None
    assert transfer_command[1] == runner.serialized_plan.path.read_bytes()
    transfer_display = " ".join(transfer_command[0])
    assert str(runner.serialized_plan.size) in transfer_display
    assert runner.serialized_plan.sha256 in transfer_display
    assert not any("rsync" in token or "scp" in token for token in transfer_command[0])


def test_live_process_validation_reads_exact_proc_argv_env_and_start_ticks(tmp_path):
    command = (
        "/usr/bin/bash",
        "/root/workspace/ILVR/run_stage1_training.sh",
        "--model",
        "literal value;$(touch BAD)",
    )
    environment = {
        "ILVR_MACHINE_RANK": "0",
        "ILVR_REPOSITORY_COMMIT": "a" * 40,
    }
    ownership = OwnershipRecord.create(
        run_id="stage1-lora-zero3-four-gpu",
        attempt_id="initial",
        machine_rank=0,
        pid=4312,
        process_start_ticks=9981,
        commit="a" * 40,
        command=command,
        environment=environment,
    )
    process_directory = tmp_path / "4312"
    process_directory.mkdir()
    trailing_fields = ["S", *("0" for _ in range(18)), "9981", "0"]
    (process_directory / "stat").write_text(
        f"4312 (stage command with spaces) {' '.join(trailing_fields)}\n",
        encoding="utf-8",
    )
    (process_directory / "cmdline").write_bytes(
        b"\0".join(item.encode() for item in command) + b"\0"
    )
    live_environment = {
        **environment,
        "ILVR_OWNED_COMMAND_DIGEST": ownership.command_digest,
    }
    (process_directory / "environ").write_bytes(
        b"\0".join(
            f"{key}={value}".encode() for key, value in live_environment.items()
        )
        + b"\0"
    )

    assert validate_owned_process(
        ownership,
        expected_run_id=ownership.run_id,
        expected_attempt_id="initial",
        expected_machine_rank=0,
        expected_commit="a" * 40,
        expected_command=command,
        expected_environment=environment,
        proc_root=tmp_path,
    )

    # Stage 2 legitimately execs Accelerate in place; the launch-bound digest
    # survives that exec in the owned environment while PID/start ticks stay fixed.
    (process_directory / "cmdline").write_bytes(
        b"/root/workspace/ILVR/venv/ilvr-venv/bin/python\0accelerate\0"
    )
    assert validate_owned_process(
        ownership,
        expected_run_id=ownership.run_id,
        expected_attempt_id="initial",
        expected_machine_rank=0,
        expected_commit="a" * 40,
        expected_command=command,
        expected_environment=environment,
        proc_root=tmp_path,
    )

    live_environment["ILVR_OWNED_COMMAND_DIGEST"] = "b" * 64
    (process_directory / "environ").write_bytes(
        b"\0".join(
            f"{key}={value}".encode() for key, value in live_environment.items()
        )
        + b"\0"
    )
    with pytest.raises(ValueError, match="command digest"):
        validate_owned_process(
            ownership,
            expected_run_id=ownership.run_id,
            expected_attempt_id="initial",
            expected_machine_rank=0,
            expected_commit="a" * 40,
            expected_command=command,
            expected_environment=environment,
            proc_root=tmp_path,
        )
