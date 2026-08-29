import json
import os
import shlex
import signal
import stat
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import pytest

import src.multinode as multinode_module
import src.multinode_runtime as runtime_module
from src.multinode import build_launch_plan
from src.multinode_runtime import (
    OwnershipRecord,
    SubprocessCoordinatorRunner,
    _build_remote_stdin_exec,
    build_remote_exec,
    build_owned_tmux_commands,
    canonical_command_digest,
    monitor_owned_jobs,
    owned_listener_is_ready,
    serialize_launch_plan,
    start_rank0_then_rank1,
    stop_owned_peer,
    validate_resume_path,
    validate_owned_process,
    validate_ownership,
    validate_stage1_arguments,
    validate_stage2_arguments,
    validated_stage2_environment,
    write_ownership,
    write_incomplete_attempt,
    write_training_witness,
)
from stage2.evidence import build_stage2_producer_evidence


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
            "/usr/bin/ssh",
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
            "/usr/bin/tmux",
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

    def cleanup_owned(self, machine_rank, *, timeout_seconds):
        assert timeout_seconds == 30
        self.events.append(f"cleanup-rank{machine_rank}")
        ownership = self.read_ownership(machine_rank)
        stop_owned_peer(ownership, self)
        self.events.append(f"observed-exit-rank{machine_rank}")
        return True


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


def test_publication_rejects_symlink_ancestor_and_racing_replacement(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    ownership = record()

    with pytest.raises((OSError, ValueError), match="symlink|trusted"):
        write_ownership(linked / "run/initial/ownership.json", ownership)
    assert not any(outside.iterdir())

    path = tmp_path / "safe/run/initial/ownership.json"
    outcomes = []

    def publish():
        try:
            write_ownership(path, ownership)
            outcomes.append("created")
        except FileExistsError:
            outcomes.append("exists")

    threads = [threading.Thread(target=publish) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["created", "exists"]
    assert json.loads(path.read_text(encoding="utf-8")) == ownership.to_json()


@pytest.mark.parametrize("stdin_attached", [False, True])
def test_remote_exec_survives_openssh_joining_and_preserves_stdin(
    stdin_attached, tmp_path
):
    marker = tmp_path / "NEVER"
    target = (
        "/usr/bin/python3",
        "-c",
        "import sys; print(sys.argv[1]); print(sys.stdin.buffer.read().hex())",
        f"literal value;$(touch {marker})",
    )
    command = (
        _build_remote_stdin_exec(target)
        if stdin_attached
        else build_remote_exec(target)
    )

    # OpenSSH joins everything after the host into one remote shell command.
    remote_words = command[6:]
    assert len(remote_words) == 1
    docker_argv = shlex.split(" ".join(remote_words))
    assert ("-i" in docker_argv) is stdin_attached
    assert docker_argv[:2] == ["/usr/bin/docker", "exec"]
    assert docker_argv[-3:-1] == ["/usr/bin/bash", "-lc"]
    assert tuple(shlex.split(docker_argv[-1])) == target
    transferred = b"plan-bytes\x00unchanged" if stdin_attached else b""

    # This is a fake Docker boundary: execute only the decoded container target.
    completed = subprocess.run(
        docker_argv[-3:],
        input=transferred,
        capture_output=True,
        check=True,
    )
    lines = completed.stdout.decode().splitlines()
    assert lines == [target[-1], transferred.hex()]
    assert not marker.exists()


def test_tmux_commands_use_fixed_sessions_and_quote_shell_metacharacters():
    marker = Path("COORDINATOR_INJECTION")
    launch_plan = plan()
    commands = build_owned_tmux_commands(
        launch_plan,
        commit="a" * 40,
        stage_arguments=("--model", "literal;$(touch COORDINATOR_INJECTION)"),
    )

    assert commands.local[:5] == (
        "/usr/bin/tmux",
        "new-session",
        "-d",
        "-s",
        f"ilvr-{launch_plan.run_id}-initial-rank1",
    )
    assert commands.local[5:8] == ("--", "/usr/bin/env", "-i")
    assert "literal;$(touch COORDINATOR_INJECTION)" in commands.local
    assert not marker.exists()
    assert commands.remote[:6] == (
        "/usr/bin/ssh",
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
    assert fake_runner.events[-3:] == [
        "cleanup-rank0",
        "signal",
        "observed-exit-rank0",
    ]


def test_local_start_failure_cleans_up_owned_rank_zero(fake_runner):
    def fail_local(_command):
        raise RuntimeError("local tmux failed")

    fake_runner.start_local = fail_local
    with pytest.raises(RuntimeError, match="local tmux failed"):
        start_rank0_then_rank1(plan(), fake_runner)
    assert fake_runner.events[-6:] == [
        "cleanup-rank0",
        "signal",
        "observed-exit-rank0",
        "cleanup-rank1",
        "signal",
        "observed-exit-rank1",
    ]


def test_dry_run_starts_neither_node(fake_runner):
    commands = start_rank0_then_rank1(plan(), fake_runner, dry_run=True)
    assert len(commands) == 2
    assert fake_runner.events == []


def test_peer_stop_requires_exact_ownership(fake_runner):
    fake_runner.observed_start_ticks = 9982
    with pytest.raises(ValueError, match="start time"):
        stop_owned_peer(record(start_ticks=9981), fake_runner)
    assert "signal" not in fake_runner.events


def test_stage2_multinode_forwards_explicit_save_step_for_final_checkpoint():
    assert validate_stage2_arguments(
        ("--max_steps", "1", "--save_steps", "1")
    ) == ("--max_steps", "1", "--save_steps", "1")


@pytest.mark.parametrize("value", ("0", "-1", "1.5", "true", "01"))
def test_stage2_multinode_rejects_noncanonical_save_step(value):
    with pytest.raises(ValueError, match="save_steps.*positive integer"):
        validate_stage2_arguments(("--max_steps", "1", "--save_steps", value))


def test_stage2_multinode_rejects_duplicate_save_step():
    with pytest.raises(ValueError, match="save_steps.*exactly once"):
        validate_stage2_arguments(
            ("--max_steps", "1", "--save_steps", "1", "--save_steps=1")
        )


def _write_stop_ownership(
    control_root: Path,
    *,
    run_id: str,
    attempt_id: str,
    rank: int,
    pid: int,
) -> OwnershipRecord:
    command = ("/root/workspace/ILVR/venv/ilvr-venv/bin/python", "worker")
    environment = {"ILVR_REPOSITORY_COMMIT": "a" * 40}
    ownership = OwnershipRecord.create(
        run_id=run_id,
        attempt_id=attempt_id,
        machine_rank=rank,
        pid=pid,
        process_start_ticks=9000 + pid,
        commit="a" * 40,
        command=command,
        environment=environment,
    )
    attempt = control_root / run_id / attempt_id
    attempt.mkdir(parents=True, mode=0o700)
    (attempt / f"ownership-rank{rank}.json").write_text(
        json.dumps(ownership.to_json()) + "\n",
        encoding="utf-8",
    )
    return ownership


def test_stop_owned_enumerates_multiple_attempts_and_tolerates_exact_exits(
    tmp_path, monkeypatch
):
    control_root = tmp_path / "logs" / "multinode"
    run_id = "stage1-lora-zero3-four-gpu"
    for attempt_id, pid in (("initial", 4301), ("resume-one", 4302), ("finished", 4303)):
        _write_stop_ownership(
            control_root,
            run_id=run_id,
            attempt_id=attempt_id,
            rank=1,
            pid=pid,
        )
    monkeypatch.setattr(runtime_module, "_LOG_ROOT", control_root)

    def stop_process(ownership, **expected):
        assert expected == {
            "expected_run_id": run_id,
            "expected_attempt_id": ownership.attempt_id,
            "expected_machine_rank": 1,
        }
        return ownership.attempt_id != "finished"

    monkeypatch.setattr(
        runtime_module,
        "stop_exact_owned_local_process",
        stop_process,
        raising=False,
    )
    stop_owned_attempts = getattr(runtime_module, "stop_owned_attempts", None)
    assert stop_owned_attempts is not None

    assert stop_owned_attempts(run_id=run_id, machine_rank=1) == {
        "run_id": run_id,
        "machine_rank": 1,
        "stopped_attempts": ["initial", "resume-one"],
        "already_exited_attempts": ["finished"],
    }


def test_stop_owned_rejects_foreign_record_before_signaling_any_attempt(
    tmp_path, monkeypatch
):
    control_root = tmp_path / "logs" / "multinode"
    run_id = "stage1-lora-zero3-four-gpu"
    _write_stop_ownership(
        control_root,
        run_id=run_id,
        attempt_id="initial",
        rank=1,
        pid=4301,
    )
    foreign = _write_stop_ownership(
        control_root,
        run_id="foreign-stage1-run",
        attempt_id="resume-one",
        rank=1,
        pid=4302,
    )
    foreign_path = (
        control_root / run_id / "resume-one" / "ownership-rank1.json"
    )
    foreign_path.parent.mkdir(parents=True, mode=0o700)
    foreign_path.write_text(json.dumps(foreign.to_json()) + "\n", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "_LOG_ROOT", control_root)
    signaled = []
    monkeypatch.setattr(
        runtime_module,
        "stop_exact_owned_local_process",
        lambda ownership, **expected: signaled.append(ownership) or True,
        raising=False,
    )
    stop_owned_attempts = getattr(runtime_module, "stop_owned_attempts", None)
    assert stop_owned_attempts is not None

    with pytest.raises(ValueError, match="run ID"):
        stop_owned_attempts(run_id=run_id, machine_rank=1)
    assert signaled == []


def test_exact_owned_local_stop_rejects_tampered_live_command_before_pidfd_signal(
    monkeypatch
):
    command = ("/root/workspace/ILVR/venv/ilvr-venv/bin/python", "worker")
    environment = {"ILVR_REPOSITORY_COMMIT": "a" * 40}
    ownership = OwnershipRecord.create(
        run_id="stage1-lora-zero3-four-gpu",
        attempt_id="initial",
        machine_rank=1,
        pid=4301,
        process_start_ticks=9981,
        commit="a" * 40,
        command=command,
        environment=environment,
    )
    monkeypatch.setattr(runtime_module.os, "pidfd_open", lambda pid: 91)
    monkeypatch.setattr(runtime_module, "_proc_start_ticks", lambda pid, **kwargs: 9981)

    def nul_fields(path):
        if path.name == "cmdline":
            return (command[0], "foreign-worker")
        return ("ILVR_REPOSITORY_COMMIT=" + "a" * 40,)

    monkeypatch.setattr(runtime_module, "_nul_fields", nul_fields)
    signals = []
    monkeypatch.setattr(
        runtime_module.signal,
        "pidfd_send_signal",
        lambda pidfd, signum: signals.append((pidfd, signum)),
    )
    real_close = runtime_module.os.close
    monkeypatch.setattr(
        runtime_module.os,
        "close",
        lambda descriptor: None if descriptor == 91 else real_close(descriptor),
    )
    stop_process = getattr(runtime_module, "stop_exact_owned_local_process", None)
    assert stop_process is not None

    with pytest.raises(ValueError, match="live ownership command digest"):
        stop_process(
            ownership,
            expected_run_id=ownership.run_id,
            expected_attempt_id=ownership.attempt_id,
            expected_machine_rank=1,
        )
    assert signals == []


def test_stop_owned_cli_has_exact_public_run_and_rank_interface(
    monkeypatch, capsys
):
    run_id = "stage1-lora-zero3-four-gpu"
    summary = {
        "run_id": run_id,
        "machine_rank": 1,
        "stopped_attempts": ["initial", "resume-one"],
        "already_exited_attempts": [],
    }
    monkeypatch.setattr(
        runtime_module,
        "stop_owned_attempts",
        lambda **kwargs: summary,
        raising=False,
    )

    assert multinode_module._main(
        ["stop-owned", "--run-id", run_id, "--machine-rank", "1"]
    ) == 0
    assert json.loads(capsys.readouterr().out) == summary


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
    assert fake_runner.events == [
        "cleanup-rank1",
        "signal",
        "observed-exit-rank1",
    ]


def test_attempt_monitor_has_fixed_24_hour_bound_and_cleans_proven_peers(
    fake_runner, monkeypatch
):
    launch_plan = plan()
    assert launch_plan.attempt_timeout_seconds == 24 * 60 * 60
    clock = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr("src.multinode_runtime.time.monotonic", lambda: next(clock))

    with pytest.raises(TimeoutError, match="24-hour"):
        monitor_owned_jobs(
            launch_plan,
            fake_runner,
            expected_commit="a" * 40,
            expected_command_digests={
                0: fake_runner.ownership[0].command_digest,
                1: fake_runner.ownership[1].command_digest,
            },
            attempt_timeout_seconds=0.5,
        )
    assert fake_runner.events == [
        "cleanup-rank0",
        "signal",
        "observed-exit-rank0",
        "cleanup-rank1",
        "signal",
        "observed-exit-rank1",
    ]


def test_monitor_poll_exception_attempts_independent_cleanup_for_both_nodes(
    fake_runner,
):
    """Catches an ambiguous poll failure leaving either owned GPU job alive."""
    events = []

    def fail_poll(machine_rank):
        events.append(f"poll-rank{machine_rank}")
        raise RuntimeError("poll transport failed")

    def cleanup(machine_rank, *, timeout_seconds):
        assert timeout_seconds == 30
        events.append(f"cleanup-rank{machine_rank}")
        if machine_rank == 0:
            raise OSError("rank0 cleanup transport failed")
        return True

    fake_runner.poll_exit = fail_poll
    fake_runner.cleanup_owned = cleanup
    with pytest.raises(RuntimeError, match="poll transport failed") as caught:
        monitor_owned_jobs(
            plan(), fake_runner, expected_commit="a" * 40,
            expected_command_digests={
                0: fake_runner.ownership[0].command_digest,
                1: fake_runner.ownership[1].command_digest,
            },
        )
    assert events == ["poll-rank0", "cleanup-rank0", "cleanup-rank1"]
    assert any("rank0 cleanup transport failed" in note for note in caught.value.__notes__)


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

    commands = build_owned_tmux_commands(
        plan(stage="stage2", method="dora"),
        commit="a" * 40,
        stage_environment=environment,
    )
    effective = dict(commands.node_specs[0].environment)
    assert effective["PATH"].startswith("/usr/local/sbin:")
    assert effective["HOME"] == "/root"
    assert "UNRELATED_SECRET" not in effective
    assert "STAGE2_DRY_RUN" not in effective


def test_stage2_coordinator_owns_a_fresh_producer_evidence_path():
    """Catches Stage 2 returning to Task 5's intentionally incomplete state."""
    launch_plan = plan(stage="stage2", method="dora")
    producer = Path(
        "/root/workspace/ILVR/results/stage2-dora-zero3-four-gpu/"
        "initial-producer-evidence.json"
    )
    environment = validated_stage2_environment(
        {
            "STAGE1_CHECKPOINT": "/models/stage1",
            "DATA_PATH": "/data/train.jsonl",
            "IMAGE_ROOT": "/data/images",
            "TUNING_METHOD": "dora",
        },
        output_dir=f"/root/workspace/ILVR/checkpoints/{launch_plan.run_id}",
        log_file=(
            "/root/workspace/ILVR/logs/zero-stage-compatibility/"
            f"{launch_plan.run_id}/initial-train.log"
        ),
        deepspeed_config=str(launch_plan.deepspeed_config),
        benchmark_evidence_path=str(producer),
    )

    commands = build_owned_tmux_commands(
        launch_plan,
        commit="a" * 40,
        stage_environment=environment,
    )

    assert commands.finite_loss_evidence_path == producer
    assert dict(commands.node_specs[0].environment)[
        "BENCHMARK_EVIDENCE_PATH"
    ] == str(producer)


def test_stage1_rejects_any_environment_override():
    with pytest.raises(ValueError, match="does not accept"):
        build_owned_tmux_commands(
            plan(),
            commit="a" * 40,
            stage_environment={"UNRELATED_SECRET": "must-not-forward"},
        )


def test_success_witness_references_real_finite_loss_evidence_exclusively(tmp_path):
    producer = tmp_path / "producer-evidence.json"
    producer.write_text(
        json.dumps({"benchmark_evidence": {"finite_loss": True}}),
        encoding="utf-8",
    )
    launch_plan = plan()
    checkpoint = tmp_path / "checkpoints" / launch_plan.run_id
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")

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
    checkpoint = tmp_path / "checkpoints" / plan().run_id
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="finite loss"):
        write_training_witness(
            plan(),
            result_root=tmp_path / "results",
            exit_codes={0: 0, 1: 0},
            checkpoint_path=checkpoint,
            batch_per_gpu=2,
            finite_loss_evidence_path=producer,
        )


def test_stage2_training_witness_revalidates_finite_gradient_and_rank_evidence(
    tmp_path,
):
    """Catches Task 5 accepting an aggregate-only Stage 2 producer file."""
    launch_plan = plan(stage="stage2", method="dora")
    checkpoint_root = tmp_path / "checkpoints" / launch_plan.run_id
    checkpoint = checkpoint_root / "checkpoint-3"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    records = [
        {
            "rank": rank,
            "global_step": 3,
            "optimizer_step": 3,
            "loss": 0.25 + rank,
            "gradient_norm": 1.0 + rank,
            "worker_error": None,
        }
        for rank in range(4)
    ]
    producer = tmp_path / "producer.json"
    producer.write_text(
        json.dumps(
            build_stage2_producer_evidence(
                run_id=launch_plan.run_id,
                method="dora",
                zero_stage=3,
                checkpoint_path=checkpoint,
                artifact_kind="peft",
                rank_records=records,
            )
        ),
        encoding="utf-8",
    )

    witness = write_training_witness(
        launch_plan,
        result_root=tmp_path / "results",
        exit_codes={0: 0, 1: 0},
        checkpoint_path=checkpoint_root,
        batch_per_gpu=1,
        finite_loss_evidence_path=producer,
    )

    reference = json.loads(witness.read_text())["finite_loss_evidence"]
    assert reference["finite_loss"] is True
    assert reference["finite_gradient"] is True
    assert reference["process_ranks"] == [0, 1, 2, 3]


def test_witness_rejects_a_missing_checkpoint_reference(tmp_path):
    with pytest.raises(ValueError, match="checkpoint"):
        write_training_witness(
            plan(),
            result_root=tmp_path / "results",
            exit_codes={0: 0, 1: 0},
            checkpoint_path=tmp_path / "missing-checkpoint",
            batch_per_gpu=2,
        )


def test_missing_evidence_creates_only_an_incomplete_attempt_record(tmp_path):
    launch_plan = plan()
    checkpoint = tmp_path / "checkpoints" / launch_plan.run_id
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="finite loss evidence is required"):
        write_training_witness(
            launch_plan,
            result_root=tmp_path / "results",
            exit_codes={0: 0, 1: 0},
            checkpoint_path=checkpoint,
            batch_per_gpu=2,
        )
    incomplete = write_incomplete_attempt(
        launch_plan,
        result_root=tmp_path / "results",
        reason="finite-loss-evidence-missing",
        exit_codes={0: 0, 1: 0},
    )
    assert incomplete.name == "initial-train-incomplete.json"
    assert not (incomplete.parent / "initial-train-witness.json").exists()
    assert json.loads(incomplete.read_text())["status"] == "incomplete"


def test_success_witness_rejects_empty_checkpoint(tmp_path):
    producer = tmp_path / "producer.json"
    producer.write_text(json.dumps({"finite_loss": True}), encoding="utf-8")
    checkpoint = tmp_path / "checkpoints" / plan().run_id
    checkpoint.mkdir(parents=True)
    with pytest.raises(ValueError, match="nonempty"):
        write_training_witness(
            plan(),
            result_root=tmp_path / "results",
            exit_codes={0: 0, 1: 0},
            checkpoint_path=checkpoint,
            batch_per_gpu=2,
            finite_loss_evidence_path=producer,
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
        "/usr/bin/ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "korea@10.1.102.41",
    ]
    assert payload["local"][:5] == [
        "/usr/bin/tmux",
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


def test_resume_checkpoint_must_belong_to_the_same_run(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    own = checkpoint_root / "current-run/checkpoint-2"
    foreign = checkpoint_root / "other-run/checkpoint-2"
    own.mkdir(parents=True)
    foreign.mkdir(parents=True)
    assert validate_resume_path(
        "current-run", own, checkpoint_root=checkpoint_root
    ) == own.resolve()
    with pytest.raises(ValueError, match="same run"):
        validate_resume_path(
            "current-run", foreign, checkpoint_root=checkpoint_root
        )


def test_wrapper_rejects_different_run_resume_before_any_remote_call():
    completed = subprocess.run(
        (
            REPO_ROOT / "run_multinode_stage1_training.sh",
            "--run-id",
            "stage1-lora-zero3-four-gpu",
            "--attempt-id",
            "resume-step-two",
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
            "/root/workspace/ILVR/checkpoints/a-different-run/checkpoint-2",
        ),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 2
    assert "same run ID" in completed.stderr


def test_node_run_supervises_child_with_pidfd_and_records_exact_exit(
    tmp_path, monkeypatch
):
    ownership_path = tmp_path / "logs/run/initial/ownership-rank0.json"
    exit_path = tmp_path / "logs/run/initial/exit-rank0.json"
    observed = {}
    handlers = {}
    child_pidfd = os.open("/dev/null", os.O_RDONLY)

    class FakeChild:
        pid = 5312

        def wait(self):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return 7

    def fake_popen(command, **kwargs):
        observed["child_command"] = tuple(command)
        observed["child_environment"] = kwargs["env"]
        observed["ownership"] = json.loads(ownership_path.read_text(encoding="utf-8"))
        return FakeChild()

    supervisor_argv = (
        "/root/workspace/ILVR/venv/ilvr-venv/bin/python",
        "-m",
        "src.multinode",
        "node-run",
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
        "--exit-path",
        str(exit_path),
        "--env",
        "ILVR_MACHINE_RANK=0",
        "--",
        "/usr/bin/bash",
        "/root/workspace/ILVR/run_stage1_training.sh",
        "--model",
        "literal;$(touch BAD)",
    )

    monkeypatch.setattr("src.multinode.os.getpid", lambda: 4312)
    monkeypatch.setattr("src.multinode._proc_start_ticks", lambda pid: 9981)
    monkeypatch.setattr("src.multinode.sys.orig_argv", list(supervisor_argv))
    monkeypatch.setattr("src.multinode.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "src.multinode.signal.signal",
        lambda signum, handler: handlers.setdefault(signum, handler),
    )
    monkeypatch.setattr(
        "src.multinode.os.pidfd_open", lambda pid: child_pidfd
    )
    forwarded = []
    monkeypatch.setattr(
        "src.multinode.signal.pidfd_send_signal",
        lambda fd, signum: forwarded.append((fd, signum)),
    )
    monkeypatch.setattr(
        "src.multinode.os.environ", {"ILVR_MACHINE_RANK": "0"}
    )

    from src.multinode import _node_run

    assert _node_run(list(supervisor_argv[4:])) == 7

    assert observed["child_command"][-1] == "literal;$(touch BAD)"
    assert observed["child_environment"] == {"ILVR_MACHINE_RANK": "0"}
    assert observed["ownership"]["pid"] == 4312
    assert observed["ownership"]["process_start_ticks"] == 9981
    assert observed["ownership"]["command_digest"] == canonical_command_digest(
        supervisor_argv, {"ILVR_MACHINE_RANK": "0"}
    )
    assert forwarded == [(child_pidfd, signal.SIGTERM)]
    assert json.loads(exit_path.read_text(encoding="utf-8"))["exit_code"] == 7
    assert stat.S_IMODE(ownership_path.stat().st_mode) == 0o600


class FakeCommandRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, command, *, input_bytes=None, timeout_seconds=None):
        self.calls.append((tuple(command), input_bytes, timeout_seconds))
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
    assert mkdir_command[2] == transfer_command[2] == 30
    assert transfer_command[1] == runner.serialized_plan.path.read_bytes()
    transfer_display = " ".join(transfer_command[0])
    assert str(runner.serialized_plan.size) in transfer_display
    assert runner.serialized_plan.sha256 in transfer_display
    assert not any("rsync" in token or "scp" in token for token in transfer_command[0])


def test_cleanup_does_not_signal_when_live_ownership_cannot_be_proven(tmp_path):
    launch_plan = plan()
    commands = build_owned_tmux_commands(launch_plan, commit="a" * 40)
    spec = commands.node_specs[0]
    ownership = OwnershipRecord.create(
        run_id=launch_plan.run_id,
        attempt_id=launch_plan.attempt_id,
        machine_rank=0,
        pid=4312,
        process_start_ticks=9981,
        commit="a" * 40,
        command=spec.command,
        environment=dict(spec.environment),
    )
    calls = []

    def fake_command(command, *, input_bytes=None, timeout_seconds=None):
        display = " ".join(command)
        calls.append(display)
        if "ownership-show" in display:
            stdout = json.dumps(ownership.to_json())
        elif "owned-process-info" in display:
            stdout = json.dumps(
                {
                    "command": list(spec.command),
                    "environment": dict(spec.environment),
                    "process_start_ticks": 9982,
                }
            )
        else:
            stdout = ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    runner = SubprocessCoordinatorRunner(
        launch_plan,
        commands=commands,
        commit="a" * 40,
        log_root=tmp_path,
        command_runner=fake_command,
    )

    assert not runner.cleanup_owned(0, timeout_seconds=30)
    assert not any("signal-owned" in call for call in calls)


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
    live_environment = dict(environment)
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

    # An environment marker cannot make an arbitrary post-exec argv owned.
    (process_directory / "cmdline").write_bytes(
        b"/root/workspace/ILVR/venv/ilvr-venv/bin/python\0accelerate\0"
    )
    with pytest.raises(ValueError, match="command"):
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

    (process_directory / "cmdline").write_bytes(
        b"\0".join(item.encode() for item in command) + b"\0"
    )
    live_environment["UNAPPROVED_SECRET"] = "secret"
    (process_directory / "environ").write_bytes(
        b"\0".join(
            f"{key}={value}".encode() for key, value in live_environment.items()
        )
        + b"\0"
    )
    with pytest.raises(ValueError, match="environment"):
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


def _write_proc_identity(proc_root, *, pid, parent_pid, start_ticks):
    fields = ["S", str(parent_pid), *("0" for _ in range(17)), str(start_ticks), "0"]
    (proc_root / str(pid) / "stat").write_text(
        f"{pid} (owned child) {' '.join(fields)}\n", encoding="ascii"
    )


def test_owned_listener_rejects_foreign_socket_and_accepts_stable_pinned_descendant(
    tmp_path, monkeypatch
):
    import src.multinode_runtime as runtime

    owner = 4312
    child = 4313
    (tmp_path / "net").mkdir()
    (tmp_path / str(owner) / "task" / str(owner)).mkdir(parents=True)
    (tmp_path / str(owner) / "task" / str(owner) / "children").write_text(
        str(child), encoding="ascii"
    )
    (tmp_path / str(child) / "task" / str(child)).mkdir(parents=True)
    (tmp_path / str(child) / "task" / str(child) / "children").write_text(
        "", encoding="ascii"
    )
    (tmp_path / str(owner) / "fd").mkdir()
    (tmp_path / str(child) / "fd").mkdir()
    _write_proc_identity(tmp_path, pid=owner, parent_pid=1, start_ticks=9981)
    _write_proc_identity(tmp_path, pid=child, parent_pid=owner, start_ticks=9982)
    header = "  sl  local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
    line = "   0: 00000000:73BF 00000000:0000 0A 0:0 0:0 00:0 0 0 44444\n"
    (tmp_path / "net/tcp").write_text(header + line, encoding="ascii")
    (tmp_path / "net/tcp6").write_text(header, encoding="ascii")
    pinned = {}
    liveness = []

    def pin(pid):
        descriptor = os.open("/dev/null", os.O_RDONLY)
        pinned[descriptor] = pid
        return descriptor

    monkeypatch.setattr(runtime.os, "pidfd_open", pin)
    monkeypatch.setattr(
        signal,
        "pidfd_send_signal",
        lambda descriptor, signum: liveness.append((pinned[descriptor], signum)),
    )

    assert not owned_listener_is_ready(
        owner, master_addr="10.1.102.41", port=29631, proc_root=tmp_path
    )
    liveness.clear()
    (tmp_path / str(child) / "fd/7").symlink_to("socket:[44444]")
    assert owned_listener_is_ready(
        owner, master_addr="10.1.102.41", port=29631, proc_root=tmp_path
    )
    assert liveness == [(owner, 0), (child, 0)]


def test_owned_listener_rejects_descendant_pid_reuse_before_fd_verdict(
    tmp_path, monkeypatch
):
    import src.multinode_runtime as runtime

    owner = 4312
    child = 4313
    (tmp_path / "net").mkdir()
    for pid in (owner, child):
        (tmp_path / str(pid) / "task" / str(pid)).mkdir(parents=True)
        (tmp_path / str(pid) / "fd").mkdir()
    (tmp_path / str(owner) / "task" / str(owner) / "children").write_text(
        str(child), encoding="ascii"
    )
    (tmp_path / str(child) / "task" / str(child) / "children").write_text(
        "", encoding="ascii"
    )
    _write_proc_identity(tmp_path, pid=owner, parent_pid=1, start_ticks=9981)
    _write_proc_identity(tmp_path, pid=child, parent_pid=owner, start_ticks=9982)
    header = "  sl  local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
    line = "   0: 00000000:73BF 00000000:0000 0A 0:0 0:0 00:0 0 0 44444\n"
    (tmp_path / "net/tcp").write_text(header + line, encoding="ascii")
    (tmp_path / "net/tcp6").write_text(header, encoding="ascii")
    listener_fd = tmp_path / str(child) / "fd/7"
    listener_fd.symlink_to("socket:[44444]")
    pinned = {}

    def pin(pid):
        descriptor = os.open("/dev/null", os.O_RDONLY)
        pinned[descriptor] = pid
        return descriptor

    real_readlink = runtime.os.readlink

    def reuse_before_fd_verdict(path):
        if Path(path) == listener_fd:
            _write_proc_identity(
                tmp_path, pid=child, parent_pid=9999, start_ticks=19982
            )
        return real_readlink(path)

    monkeypatch.setattr(runtime.os, "pidfd_open", pin)
    monkeypatch.setattr(runtime.os, "readlink", reuse_before_fd_verdict)
    monkeypatch.setattr(signal, "pidfd_send_signal", lambda *_args: None)

    assert not owned_listener_is_ready(
        owner, master_addr="10.1.102.41", port=29631, proc_root=tmp_path
    )


def test_signal_owned_uses_pidfd_and_never_os_kill(tmp_path, monkeypatch):
    from src.multinode import _signal_owned
    import src.multinode_runtime as runtime

    command = ("/usr/bin/bash", "/root/workspace/ILVR/run_stage1_training.sh")
    environment = {"ILVR_MACHINE_RANK": "0", "ILVR_REPOSITORY_COMMIT": "a" * 40}
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
    ownership_path = tmp_path / "logs/run/initial/ownership.json"
    write_ownership(ownership_path, ownership)
    process_directory = tmp_path / "proc/4312"
    process_directory.mkdir(parents=True)
    fields = ["S", *("0" for _ in range(18)), "9981", "0"]
    (process_directory / "stat").write_text(
        f"4312 (bash) {' '.join(fields)}\n", encoding="utf-8"
    )
    (process_directory / "cmdline").write_bytes(
        b"\0".join(part.encode() for part in command) + b"\0"
    )
    (process_directory / "environ").write_bytes(
        b"\0".join(f"{key}={value}".encode() for key, value in environment.items())
        + b"\0"
    )
    events = []
    pidfd = os.open("/dev/null", os.O_RDONLY)
    real_nul_fields = runtime._nul_fields

    def observed_proc_read(path):
        events.append(("proc-read", Path(path).name))
        return real_nul_fields(path)

    monkeypatch.setattr("src.multinode._PROC_ROOT", tmp_path / "proc")
    monkeypatch.setattr("src.multinode.os.pidfd_open", lambda pid: events.append(("open", pid)) or pidfd)
    monkeypatch.setattr(runtime, "_nul_fields", observed_proc_read)
    monkeypatch.setattr(
        "src.multinode.signal.pidfd_send_signal",
        lambda fd, sig: events.append(("signal", fd, sig)),
    )
    monkeypatch.setattr(
        "src.multinode.os.kill",
        lambda *_args: pytest.fail("PID-based signaling is forbidden"),
    )
    arguments = [
        "--ownership-path", str(ownership_path), "--run-id", ownership.run_id,
        "--attempt-id", ownership.attempt_id, "--machine-rank", "0",
        "--pid", str(ownership.pid), "--process-start-ticks", "9981",
        "--commit", ownership.commit, "--command-digest", ownership.command_digest,
    ]
    for name, value in environment.items():
        arguments.extend(("--env", f"{name}={value}"))
    arguments.extend(("--", *command))

    assert _signal_owned(arguments) == 0
    assert events == [
        ("open", 4312),
        ("proc-read", "cmdline"),
        ("proc-read", "environ"),
        ("signal", pidfd, signal.SIGTERM),
    ]
    with pytest.raises(OSError):
        os.fstat(pidfd)
