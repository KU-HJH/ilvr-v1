from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import subprocess

import pytest

from src.multinode import _main
from src.multinode_artifacts import (
    ArtifactEntry,
    NodeArtifactManifest,
    build_evaluation_pull_command,
    build_prepared_inference_pull_command,
    build_prepared_inference_push_command,
    evaluation_result_files_bytes,
    prepare_inference_copy,
    validate_evaluation_sidecars,
)
from src.multinode_evaluation import write_shard_manifest
from src.multinode_runtime import (
    EvaluationPlan,
    EvaluationSubprocessRunner,
    OwnershipRecord,
    build_evaluation_assignments,
    build_evaluation_commands,
    prepare_attempt_directory,
    run_evaluation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _identity(*, stage: str = "stage1") -> dict[str, object]:
    return {
        "stage": stage,
        "tuning_method": "lora",
        "zero_stage": 3,
        "deepspeed_config_digest": "a" * 64,
        "world_size": 4,
        "rank_mapping": {"0": [0, 1], "1": [2, 3]},
        "global_step": 2,
        "model_state_layout": "partitioned",
        "optimizer": True,
        "scheduler": True,
        "rng_ranks": [0, 1, 2, 3],
        "stage_sidecar": f"{stage}_peft_metadata.json",
        "commit": "b" * 40,
        "run_id": "trained-model-run",
    }


def _manifest(*, stage: str = "stage1") -> NodeArtifactManifest:
    names = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "ilvr_token_rows.safetensors",
        f"{stage}_peft_metadata.json",
    )
    return NodeArtifactManifest(
        schema_version=1,
        run_id="trained-model-run",
        node_rank=0,
        process_ranks=(0, 1, 2, 3),
        entries=tuple(ArtifactEntry(Path(name), 1, 0) for name in names),
        runtime_identity=_identity(stage=stage),
        completion_marker=True,
        checkpoint_relative=Path("checkpoint-2"),
    )


def _plan() -> EvaluationPlan:
    return EvaluationPlan(
        run_id="eval-four-gpu",
        attempt_id="initial",
        model_dir=(
            REPO_ROOT
            / "checkpoints/evaluation-staging/trained-model-run/prepared-model"
        ),
        test_data_path=REPO_ROOT / "data/TEST.jsonl",
        task_name="zebra-cot",
        output_dir=REPO_ROOT / "results/eval-four-gpu",
    )


def test_evaluation_assigns_remote_zero_one_and_local_two_three():
    assignments = build_evaluation_assignments(run_id="eval-four-gpu")
    assert [(a.node, a.gpu, a.shard_index) for a in assignments] == [
        ("remote", 0, 0),
        ("remote", 1, 1),
        ("local", 0, 2),
        ("local", 1, 3),
    ]
    assert len({a.session_name for a in assignments}) == 4
    assert len({a.ownership_name for a in assignments}) == 4


def test_local_inference_staging_is_checkpoint_owned():
    plan = _plan()

    assert plan.local_model_dir == (
        REPO_ROOT / "checkpoints/evaluation-staging/eval-four-gpu/local-model"
    )


def test_evaluation_model_artifacts_are_checkpoint_owned():
    plan = EvaluationPlan(
        run_id="eval-four-gpu",
        model_dir=(
            REPO_ROOT
            / "checkpoints/evaluation-staging/trained-model-run/prepared-model"
        ),
        test_data_path=REPO_ROOT / "data/TEST.jsonl",
        task_name="zebra-cot",
        output_dir=REPO_ROOT / "results/eval-four-gpu",
    )

    assert plan.local_model_dir == (
        REPO_ROOT / "checkpoints/evaluation-staging/eval-four-gpu/local-model"
    )
    with pytest.raises(ValueError, match="checkpoint"):
        dataclasses.replace(
            plan,
            model_dir=REPO_ROOT / "results/trained-model-run/prepared-model",
        )


@pytest.mark.parametrize(
    ("stage", "evaluator"),
    [
        ("stage1", REPO_ROOT / "eval.py"),
        ("stage2", REPO_ROOT / "src/evaluate_deepseed.py"),
    ],
)
def test_worker_commands_use_observed_stage_exact_gpu_and_one_model_identity(
    stage, evaluator
):
    commands = build_evaluation_commands(
        _plan(), inference_manifest=_manifest(stage=stage), commit="b" * 40
    )

    assert len(commands.workers) == 4
    assert {dict(worker.node_spec.environment)["CUDA_VISIBLE_DEVICES"]
            for worker in commands.workers} == {"0", "1"}
    assert all(
        str(evaluator) in worker.node_spec.command for worker in commands.workers
    )
    assert [worker.node_spec.command[-1] for worker in commands.workers] == [
        "0", "1", "2", "3"
    ]
    identities = {
        worker.node_spec.command[
            worker.node_spec.command.index("--model-identity") + 1
        ]
        for worker in commands.workers
    }
    assert len(identities) == 1
    model_dirs = [
        worker.node_spec.command[worker.node_spec.command.index("--model_dir") + 1]
        for worker in commands.workers
    ]
    assert model_dirs[:2] == [str(_plan().model_dir)] * 2
    assert model_dirs[2:] == [str(_plan().local_model_dir)] * 2
    assert {
        worker.node_spec.command[
            worker.node_spec.command.index("--max_new_tokens") + 1
        ]
        for worker in commands.workers
    } == {"4096"}
    assert {
        worker.node_spec.command[
            worker.node_spec.command.index("--max_sequence_length") + 1
        ]
        for worker in commands.workers
    } == {"32768"}
    assert {
        worker.node_spec.command[
            worker.node_spec.command.index("--repetition_penalty") + 1
        ]
        for worker in commands.workers
    } == {"1.05"}


def test_worker_commands_forward_external_generation_limit_policy():
    """Losing the coordinator option would silently restore bounded generation."""
    plan = dataclasses.replace(
        _plan(), generation_limit_policy="external_full_latent_block"
    )

    commands = build_evaluation_commands(
        plan, inference_manifest=_manifest(), commit="b" * 40
    )

    assert {
        worker.node_spec.command[
            worker.node_spec.command.index("--generation-limit-policy") + 1
        ]
        for worker in commands.workers
    } == {"external_full_latent_block"}


def test_evaluation_worker_environment_imports_worktree_transformers():
    """Catches the shared venv editable path selecting another checkout."""
    commands = build_evaluation_commands(
        _plan(), inference_manifest=_manifest(), commit="b" * 40
    )
    environment = dict(commands.workers[0].node_spec.environment)

    completed = subprocess.run(
        [
            str(REPO_ROOT / "venv/ilvr-venv/bin/python"),
            "-c",
            (
                "from pathlib import Path; import transformers; "
                "print(Path(transformers.__file__).resolve())"
            ),
        ],
        cwd="/",
        env={**environment,"PYTHONDONTWRITEBYTECODE":"1"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    imported = Path(completed.stdout.strip())
    assert imported.is_relative_to((REPO_ROOT / "transformers/src").resolve())


def test_worker_supervisors_bind_exact_evaluation_shard_identity():
    commands = build_evaluation_commands(
        _plan(), inference_manifest=_manifest(stage="stage2"), commit="b" * 40
    )

    for worker in commands.workers:
        command = worker.node_spec.command
        shard = worker.assignment.shard_index
        option = command.index("--evaluation-shard-index")
        assert command[option + 1] == str(shard)
        assert worker.node_spec.ownership_path.name == (
            f"ownership-eval-shard{shard}.json"
        )
        assert worker.node_spec.exit_path.name == f"exit-eval-shard{shard}.json"


class FakeEvaluationRunner:
    def __init__(
        self,
        *,
        exit_codes: dict[int, int] | None = None,
        invalid_exit_shard: int | None = None,
        launch_timeout_shard: int | None = None,
        poll_timeout_shard: int | None = None,
        cleanup_error_shards: set[int] | None = None,
        valid=True,
    ):
        self.exit_codes = {index: 0 for index in range(4)}
        self.exit_codes.update(exit_codes or {})
        self.invalid_exit_shard = invalid_exit_shard
        self.launch_timeout_shard = launch_timeout_shard
        self.poll_timeout_shard = poll_timeout_shard
        self.cleanup_error_shards = cleanup_error_shards or set()
        self.valid = valid
        self.events: list[str] = []

    def prepare_inference(self, plan):
        self.events.append("prepare-inference")
        return _manifest()

    def repository_commit(self):
        return "b" * 40

    def start_worker(self, worker):
        shard = worker.assignment.shard_index
        self.events.append(f"start-{shard}")
        if shard == self.launch_timeout_shard:
            raise subprocess.TimeoutExpired(worker.launch, timeout=30)

    def poll_worker(self, worker):
        shard = worker.assignment.shard_index
        self.events.append(f"poll-{shard}")
        if shard == self.invalid_exit_shard:
            raise ValueError("invalid owned exit record")
        if shard == self.poll_timeout_shard:
            raise subprocess.TimeoutExpired("exit-show", timeout=30)
        return self.exit_codes[shard]

    def cleanup_worker(self, worker):
        shard = worker.assignment.shard_index
        self.events.append(f"cleanup-{shard}")
        if shard in self.cleanup_error_shards:
            raise RuntimeError(f"cleanup {shard} failed")
        return True

    def pull_remote_results(self, commands):
        self.events.append("pull")

    def validate_sidecars(self, commands):
        self.events.append("sidecars")
        if not self.valid:
            raise ValueError("unsuccessful worker sidecar")

    def merge(self, commands):
        self.events.append("merge")
        return {"summary": True, "num_samples": 4}

    def sleep(self, seconds):
        raise AssertionError("immediate fake exits must not sleep")


def test_inference_manifest_is_prepared_before_any_worker_is_created():
    runner = FakeEvaluationRunner()

    result = run_evaluation(_plan(), runner)

    assert result == {"summary": True, "num_samples": 4}
    assert runner.events[:5] == [
        "prepare-inference", "start-0", "start-1", "start-2", "start-3"
    ]
    assert runner.events[-3:] == ["pull", "sidecars", "merge"]


def test_production_runner_prepares_fresh_local_then_remote_control_namespace_once(
    tmp_path
):
    events = []
    log_root = tmp_path / "logs"

    def local_prepare(**kwargs):
        events.append(("local-prepare", kwargs))
        return prepare_attempt_directory(**kwargs)

    def fake_command(command, *, input_bytes=None, timeout_seconds=None):
        events.append(("remote-prepare", tuple(command)))
        assert "prepare-attempt" in " ".join(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    runner = EvaluationSubprocessRunner(
        _plan(),
        commit="b" * 40,
        log_root=log_root,
        command_runner=fake_command,
        attempt_preparer=local_prepare,
    )

    runner.prepare_control_namespaces()

    assert [event[0] for event in events] == [
        "local-prepare", "remote-prepare"
    ]
    assert (log_root / "eval-four-gpu/initial").is_dir()
    with pytest.raises(RuntimeError, match="already prepared"):
        runner.prepare_control_namespaces()
    assert len(events) == 2


def test_stale_local_control_namespace_blocks_remote_preparation_and_launch(tmp_path):
    log_root = tmp_path / "logs"
    prepare_attempt_directory(
        log_root=log_root,
        run_id="eval-four-gpu",
        attempt_id="initial",
    )
    calls = []

    def fake_command(command, *, input_bytes=None, timeout_seconds=None):
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    runner = EvaluationSubprocessRunner(
        _plan(),
        commit="b" * 40,
        log_root=log_root,
        command_runner=fake_command,
    )

    with pytest.raises(FileExistsError):
        runner.prepare_control_namespaces()
    commands = build_evaluation_commands(
        _plan(), inference_manifest=_manifest(), commit="b" * 40
    )
    with pytest.raises(RuntimeError, match="control namespaces"):
        runner.start_worker(commands.workers[0])
    assert calls == []


def test_stale_remote_control_namespace_blocks_launch_after_local_reservation(
    tmp_path
):
    calls = []

    def fake_command(command, *, input_bytes=None, timeout_seconds=None):
        calls.append(tuple(command))
        return subprocess.CompletedProcess(
            command, 2, stdout="", stderr="stale remote attempt"
        )

    runner = EvaluationSubprocessRunner(
        _plan(),
        commit="b" * 40,
        log_root=tmp_path / "logs",
        command_runner=fake_command,
    )

    with pytest.raises(RuntimeError, match="stale remote attempt"):
        runner.prepare_control_namespaces()
    assert (tmp_path / "logs/eval-four-gpu/initial").is_dir()
    worker = build_evaluation_commands(
        _plan(), inference_manifest=_manifest(), commit="b" * 40
    ).workers[0]
    with pytest.raises(RuntimeError, match="control namespaces"):
        runner.start_worker(worker)
    assert len(calls) == 1


def test_production_inference_preparation_reserves_controls_before_copy(
    tmp_path, monkeypatch
):
    import src.multinode_artifacts as artifacts_module

    events = []

    def fake_command(command, *, input_bytes=None, timeout_seconds=None):
        display = " ".join(command)
        if "prepared-inference-show" in display:
            events.append("manifest")
            stdout = json.dumps(_manifest().to_json())
        elif "prepare-attempt" in display:
            events.append("remote-control")
            stdout = ""
        elif "prepare-evaluation-output" in display:
            events.append("remote-output")
            stdout = ""
        else:
            raise AssertionError(f"unexpected fake command: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    def local_prepare(**kwargs):
        events.append("local-control")
        return Path(kwargs["log_root"]) / kwargs["run_id"] / kwargs["attempt_id"]

    monkeypatch.setattr(
        artifacts_module,
        "reserve_inference_destination",
        lambda path: events.append("reserve-output") or Path(path),
    )
    monkeypatch.setattr(
        artifacts_module,
        "resolve_local_checkpoint_destination",
        lambda path: events.append("resolve-model")
        or Path("/physical-checkpoints/eval-four-gpu/local-model"),
    )
    monkeypatch.setattr(
        artifacts_module,
        "prepare_inference_copy",
        lambda manifest, destination, runner: (
            events.append(f"copy:{destination}") or Path(destination)
        ),
    )
    runner = EvaluationSubprocessRunner(
        _plan(),
        commit="b" * 40,
        log_root=tmp_path / "logs",
        command_runner=fake_command,
        attempt_preparer=local_prepare,
    )

    assert runner.prepare_inference(_plan()) == _manifest()
    assert events == [
        "manifest",
        "local-control",
        "remote-control",
        "reserve-output",
        "resolve-model",
        "copy:/physical-checkpoints/eval-four-gpu/local-model",
        "remote-output",
    ]


def test_production_runner_poll_timeout_is_exposed_to_coordinator_cleanup(tmp_path):
    calls = []

    def fake_command(command, *, input_bytes=None, timeout_seconds=None):
        calls.append(tuple(command))
        display = " ".join(command)
        if "exit-show" in display:
            raise subprocess.TimeoutExpired(command, timeout_seconds)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    runner = EvaluationSubprocessRunner(
        _plan(),
        commit="b" * 40,
        log_root=tmp_path / "logs",
        command_runner=fake_command,
    )
    runner.prepare_control_namespaces()
    worker = build_evaluation_commands(
        _plan(), inference_manifest=_manifest(), commit="b" * 40
    ).workers[0]
    runner.start_worker(worker)

    with pytest.raises(subprocess.TimeoutExpired):
        runner.poll_worker(worker)
    assert any("exit-show" in " ".join(call) for call in calls)


def test_production_runner_transfer_publishes_exact_list_before_fake_rsync(
    tmp_path
):
    calls = []

    def fake_command(command, *, input_bytes=None, timeout_seconds=None):
        calls.append(tuple(command))
        if "--rsync-path" not in command:
            return subprocess.CompletedProcess(
                command,
                12,
                stdout="",
                stderr="host-mounted result is not traversable",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    commands = build_evaluation_commands(
        _plan(), inference_manifest=_manifest(), commit="b" * 40
    )
    files_path = tmp_path / "evaluation-pull.files"
    commands = dataclasses.replace(commands, remote_files_path=files_path)
    runner = EvaluationSubprocessRunner(
        _plan(), commit="b" * 40, command_runner=fake_command
    )

    runner.pull_remote_results(commands)

    assert files_path.read_bytes() == evaluation_result_files_bytes(include_runtime_identity=True)
    assert calls == [commands.remote_pull]
    assert "--secluded-args" in calls[0]
    assert calls[0][calls[0].index("--rsync-path") + 1] == (
        "/usr/bin/docker exec -i llm_kd_rdma /usr/bin/rsync"
    )


def test_production_runner_cleanup_uses_exact_owned_signal_command():
    commands = build_evaluation_commands(
        _plan(), inference_manifest=_manifest(), commit="b" * 40
    )
    worker = commands.workers[0]
    spec = worker.node_spec
    ownership = OwnershipRecord.create(
        run_id=_plan().run_id,
        attempt_id=_plan().attempt_id,
        machine_rank=worker.assignment.machine_rank,
        pid=4312,
        process_start_ticks=9981,
        commit="b" * 40,
        command=spec.command,
        environment=dict(spec.environment),
    )
    calls = []

    def fake_command(command, *, input_bytes=None, timeout_seconds=None):
        calls.append(tuple(command))
        if "ownership-show" in " ".join(command):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(ownership.to_json()),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    runner = EvaluationSubprocessRunner(
        _plan(), commit="b" * 40, command_runner=fake_command
    )

    assert runner.cleanup_worker(worker) is True
    assert len(calls) == 2
    assert "ownership-show" in " ".join(calls[0])
    assert "signal-owned" in " ".join(calls[1])
    assert str(ownership.pid) in calls[1][-1]
    assert "--evaluation-shard-index" in " ".join(calls[1])


def test_production_runner_reports_exact_owned_signal_failure():
    commands = build_evaluation_commands(
        _plan(), inference_manifest=_manifest(), commit="b" * 40
    )
    worker = commands.workers[0]
    ownership = OwnershipRecord.create(
        run_id=_plan().run_id,
        attempt_id=_plan().attempt_id,
        machine_rank=worker.assignment.machine_rank,
        pid=4312,
        process_start_ticks=9981,
        commit="b" * 40,
        command=worker.node_spec.command,
        environment=dict(worker.node_spec.environment),
    )

    def fake_command(command, *, input_bytes=None, timeout_seconds=None):
        if "ownership-show" in " ".join(command):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(ownership.to_json()),
                stderr="",
            )
        return subprocess.CompletedProcess(
            command, 5, stdout="", stderr="pidfd signal failed"
        )

    runner = EvaluationSubprocessRunner(
        _plan(), commit="b" * 40, command_runner=fake_command
    )

    with pytest.raises(RuntimeError, match="signal-owned.*pidfd signal failed"):
        runner.cleanup_worker(worker)


def test_incomplete_worker_set_is_not_merged_and_only_live_peers_are_cleaned():
    runner = FakeEvaluationRunner(exit_codes={1: 9})

    with pytest.raises(RuntimeError, match="all four evaluation workers"):
        run_evaluation(_plan(), runner)

    assert "merge" not in runner.events
    assert "pull" not in runner.events
    assert "cleanup-1" not in runner.events
    assert {event for event in runner.events if event.startswith("cleanup-")} == {
        "cleanup-2", "cleanup-3"
    }


def test_invalid_sidecar_prevents_merge():
    runner = FakeEvaluationRunner(valid=False)

    with pytest.raises(RuntimeError, match="all four evaluation workers"):
        run_evaluation(_plan(), runner)

    assert runner.events[-2:] == ["pull", "sidecars"]
    assert "merge" not in runner.events


def test_invalid_owned_exit_record_stops_only_still_live_exact_peers():
    runner = FakeEvaluationRunner(invalid_exit_shard=1)

    with pytest.raises(ValueError, match="invalid owned exit record"):
        run_evaluation(_plan(), runner)

    assert "merge" not in runner.events
    assert {event for event in runner.events if event.startswith("cleanup-")} == {
        "cleanup-1", "cleanup-2", "cleanup-3"
    }


def test_ambiguous_launch_timeout_cleans_current_and_prior_possibly_live_workers():
    runner = FakeEvaluationRunner(launch_timeout_shard=2)

    with pytest.raises(subprocess.TimeoutExpired):
        run_evaluation(_plan(), runner)

    assert [event for event in runner.events if event.startswith("cleanup-")] == [
        "cleanup-0", "cleanup-1", "cleanup-2"
    ]


def test_poll_timeout_cleans_every_worker_without_a_proven_exit():
    runner = FakeEvaluationRunner(poll_timeout_shard=1)

    with pytest.raises(subprocess.TimeoutExpired):
        run_evaluation(_plan(), runner)

    assert [event for event in runner.events if event.startswith("cleanup-")] == [
        "cleanup-1", "cleanup-2", "cleanup-3"
    ]


def test_cleanup_exception_does_not_mask_launch_timeout_or_skip_later_workers():
    runner = FakeEvaluationRunner(
        launch_timeout_shard=2,
        cleanup_error_shards={0},
    )

    with pytest.raises(subprocess.TimeoutExpired) as captured:
        run_evaluation(_plan(), runner)

    assert [event for event in runner.events if event.startswith("cleanup-")] == [
        "cleanup-0", "cleanup-1", "cleanup-2"
    ]
    assert any(
        "cleanup 0 failed" in note
        for note in getattr(captured.value, "__notes__", ())
    )


def test_remote_result_pull_is_fixed_container_rsync_with_explicit_nul_list():
    output_dir = REPO_ROOT / "results/eval-four-gpu"
    files_path = (
        REPO_ROOT
        / "logs/multinode/eval-four-gpu/initial/evaluation-pull.files"
    )

    payload = evaluation_result_files_bytes()
    command = build_evaluation_pull_command(
        output_dir, files_from=files_path
    )

    assert payload == (
        b"shard-0.jsonl\0shard-0.jsonl.manifest.json\0"
        b"shard-1.jsonl\0shard-1.jsonl.manifest.json\0"
    )
    assert command == (
        "/usr/bin/rsync",
        "--archive",
        "--ignore-existing",
        "--secluded-args",
        "--files-from",
        str(files_path),
        "--from0",
        "--rsh",
        "/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=8",
        "--rsync-path",
        "/usr/bin/docker exec -i llm_kd_rdma /usr/bin/rsync",
        "--",
        f"korea@172.16.11.41:{REPO_ROOT}/results/eval-four-gpu/",
        f"{REPO_ROOT}/results/eval-four-gpu/",
    )
    assert "--recursive" not in command


@pytest.mark.parametrize(
    "unsafe_component",
    (
        "bad`touch-owned`",
        "bad$PATH",
        "bad'quote",
        'bad"quote',
        "bad:host",
        "bad\ncontrol",
    ),
)
def test_unsafe_result_component_is_rejected_before_runner_creation(
    unsafe_component
):
    with pytest.raises(ValueError, match="safe component"):
        EvaluationPlan(
            run_id="eval-four-gpu",
            model_dir=(
                REPO_ROOT
                / "checkpoints/evaluation-staging/trained-model-run/prepared-model"
            ),
            test_data_path=REPO_ROOT / "data/TEST.jsonl",
            task_name="zebra-cot",
            output_dir=(
                REPO_ROOT / "results"
                / unsafe_component
                / "eval-four-gpu"
            ),
        )


def test_prepared_inference_backtick_source_is_rejected_before_rsync():
    with pytest.raises(ValueError, match="safe component"):
        build_prepared_inference_pull_command(
            _manifest(),
            (
                REPO_ROOT
                / "checkpoints/evaluation-staging/bad`touch-owned`/prepared-model"
            ),
            (
                REPO_ROOT
                / "checkpoints/evaluation-staging/eval-four-gpu/local-model"
            ),
            files_from=REPO_ROOT / "logs/evaluation.files",
        )


def test_prepared_inference_pull_has_exact_secluded_rsync_argv():
    files_path = REPO_ROOT / "logs/evaluation.files"

    command = build_prepared_inference_pull_command(
        _manifest(),
        (
            REPO_ROOT
            / "checkpoints/evaluation-staging/trained-model-run/prepared-model"
        ),
        (
            REPO_ROOT
            / "checkpoints/evaluation-staging/eval-four-gpu/local-model"
        ),
        files_from=files_path,
    )

    assert command == (
        "/usr/bin/rsync",
        "--archive",
        "--ignore-existing",
        "--secluded-args",
        "--files-from",
        str(files_path),
        "--from0",
        "--rsh",
        "/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=8",
        "--rsync-path",
        "/usr/bin/docker exec -i llm_kd_rdma /usr/bin/rsync",
        "--",
        f"korea@172.16.11.41:{REPO_ROOT}/checkpoints/evaluation-staging/"
        "trained-model-run/prepared-model/",
        f"{REPO_ROOT}/checkpoints/evaluation-staging/eval-four-gpu/local-model/",
    )


def test_prepared_inference_pull_accepts_resolved_local_checkpoint_destination(
    tmp_path, monkeypatch
):
    import src.multinode_artifacts as artifacts_module

    repository = tmp_path / "repo"
    repository.mkdir()
    storage = tmp_path / "checkpoint-storage"
    storage.mkdir()
    (repository / "checkpoints").symlink_to(storage, target_is_directory=True)
    source = (
        repository
        / "checkpoints"
        / "evaluation-staging"
        / "trained-model-run"
        / "prepared-model"
    )
    physical_destination = (
        storage / "evaluation-staging" / "eval-four-gpu" / "local-model"
    )
    monkeypatch.setattr(artifacts_module, "_LOCAL_REPOSITORY", repository)

    command = build_prepared_inference_pull_command(
        _manifest(),
        source,
        physical_destination,
        files_from=tmp_path / "evaluation.files",
    )

    assert command[-2] == f"korea@172.16.11.41:{source}/"
    assert command[-1] == f"{physical_destination}/"


def test_prepared_inference_push_has_exact_container_rsync_argv():
    files_path = REPO_ROOT / "logs/prepared-push.files"

    command = build_prepared_inference_push_command(
        _manifest(),
        (
            REPO_ROOT
            / "checkpoints/evaluation-staging/trained-model-run/prepared-model"
        ),
        files_from=files_path,
    )

    assert command == (
        "/usr/bin/rsync",
        "--archive",
        "--ignore-existing",
        "--secluded-args",
        "--files-from",
        str(files_path),
        "--from0",
        "--rsh",
        "/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=8",
        "--rsync-path",
        "/usr/bin/docker exec -i llm_kd_rdma /usr/bin/rsync",
        "--",
        f"{REPO_ROOT}/checkpoints/evaluation-staging/trained-model-run/",
        f"korea@172.16.11.41:{REPO_ROOT}/checkpoints/evaluation-staging/"
        "trained-model-run/",
    )


def test_task6_inference_copy_publishes_identity_manifest_for_task8(tmp_path):
    manifest = _manifest()
    destination = tmp_path / "evaluation-model"

    class CopyRunner:
        def validate_inference_manifest(self, observed):
            assert observed == manifest

        def pull_inference_artifacts(self, observed, target):
            for entry in observed.entries:
                path = target / entry.relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x" * entry.byte_size)

    prepare_inference_copy(manifest, destination, CopyRunner())

    published = destination.with_name("evaluation-model.manifest.json")
    assert json.loads(published.read_text(encoding="utf-8")) == manifest.to_json()


def test_incomplete_success_sidecar_is_rejected_before_merge(tmp_path):
    jsonls = []
    manifests = []
    for shard_index in range(4):
        jsonl = tmp_path / f"shard-{shard_index}.jsonl"
        jsonl.write_text(
            json.dumps(
                {"index": shard_index, "task_name": "task", "match": True}
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = Path(f"{jsonl}.manifest.json")
        write_shard_manifest(
            manifest,
            shard_jsonl=jsonl,
            dataset_identity={
                "path": str(REPO_ROOT / "data/TEST.jsonl"),
                "size": 10,
                "mtime_ns": 20,
            },
            model_identity="task6:" + "a" * 64,
            generation_config={
                "max_new_tokens": 4096,
                "max_sequence_length": 32768,
                "repetition_penalty": 1.05,
                "temperature": 0.0,
                "top_p": 1.0,
                "generation_limit_policy": "bounded",
            },
            shard_index=shard_index,
            num_shards=4,
            expected_source_indices=[shard_index],
            processed_source_indices=([] if shard_index == 1 else [shard_index]),
            worker_exit_status=0,
        )
        jsonls.append(jsonl)
        manifests.append(manifest)

    with pytest.raises(ValueError, match="incomplete"):
        validate_evaluation_sidecars(
            manifests,
            jsonls,
            expected_model_identity="task6:" + "a" * 64,
            expected_sample_count=4,
            expected_generation_config={
                "max_new_tokens": 4096,
                "max_sequence_length": 32768,
                "repetition_penalty": 1.05,
                "temperature": 0.0,
                "top_p": 1.0,
                "generation_limit_policy": "bounded",
            },
        )


def test_dry_run_builds_commands_without_constructing_a_live_runner(
    monkeypatch, capsys
):
    import src.multinode_runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "EvaluationSubprocessRunner",
        lambda *args, **kwargs: pytest.fail("dry-run constructed live runner"),
    )

    result = _main(
        [
            "evaluate",
            "--run-id", "eval-four-gpu",
            "--model-dir",
            str(REPO_ROOT / "checkpoints/evaluation-staging/model/prepared-model"),
            "--test-data-path", str(REPO_ROOT / "data/TEST.jsonl"),
            "--task-name", "zebra-cot",
            "--output-dir", str(REPO_ROOT / "results/eval-four-gpu"),
            "--dry-run",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["worker_commands_by_observed_stage"]) == {
        "stage1", "stage2"
    }
    assert payload["remote_result_files"] == [
        "shard-0.jsonl",
        "shard-0.jsonl.manifest.json",
        "shard-1.jsonl",
        "shard-1.jsonl.manifest.json",
        "shard-0.runtime-identity.json",
        "shard-1.runtime-identity.json",
    ]
    for commands in payload["worker_commands_by_observed_stage"].values():
        for command in commands:
            if "--max_sequence_length" in command:
                option = command.index("--max_sequence_length")
                assert command[option + 1] == "32768"
            else:
                assert "--max_sequence_length 32768" in command[-1]
            if "--repetition_penalty" in command:
                option = command.index("--repetition_penalty")
                assert command[option + 1] == "1.05"
            else:
                assert "--repetition_penalty 1.05" in command[-1]


def test_vision_ablation_dry_run_keeps_stage2_commands_and_marks_stage1_unsupported(
    monkeypatch, capsys
):
    import src.multinode_runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "EvaluationSubprocessRunner",
        lambda *args, **kwargs: pytest.fail("dry-run constructed live runner"),
    )

    assert _main(
        [
            "evaluate",
            "--run-id", "eval-four-gpu",
            "--model-dir",
            str(REPO_ROOT / "checkpoints/evaluation-staging/model/prepared-model"),
            "--test-data-path", str(REPO_ROOT / "data/TEST.jsonl"),
            "--task-name", "zebra-cot",
            "--output-dir", str(REPO_ROOT / "results/eval-four-gpu"),
            "--vision-ablation", "all",
            "--dry-run",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert "unsupported" in payload["worker_commands_by_observed_stage"]["stage1"]
    stage2 = payload["worker_commands_by_observed_stage"]["stage2"]
    assert all("--vision-ablation all" in " ".join(command) for command in stage2)
