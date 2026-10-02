from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from src.multinode import build_launch_plan, _training_main
import src.multinode as coordinator
import src.multinode_artifacts as artifacts
import src.multinode_runtime as runtime


def builder():
    api = getattr(runtime, "build_owned_vision_gain_probe_commands", None)
    assert callable(api), "missing fixed owned vision-gain probe builder"
    return api


def test_probe_is_fixed_dora_zero12_four_ranks():
    api = builder()
    for stage in (1, 2):
        plan = build_launch_plan(stage="stage1", tuning_method="dora", zero_stage=stage,
                                 run_id=f"vg-probe-z{stage}", main_process_port=29590)
        commands = api(plan, commit="a" * 40)
        assert [s.machine_rank for s in commands.node_specs] == [0, 1]
        for rank, spec in enumerate(commands.node_specs):
            cmd = spec.command
            assert cmd[cmd.index("--num_machines") + 1] == "2"
            assert cmd[cmd.index("--num_processes") + 1] == "4"
            assert cmd[cmd.index("--machine_rank") + 1] == str(rank)
            assert str(plan.cluster.remote_repo_root / "tests/stage1/vision_gain_zero_probe.py") in cmd
            assert cmd[cmd.index("--phase") + 1] == "initial"
        with pytest.raises((ValueError, TypeError)):
            api(plan, commit="a" * 40, harness_path="evil.py")
        with pytest.raises(ValueError):
            api(replace(plan, tuning_method="full"), commit="a" * 40)
        for changed in (dict(world_size=2), dict(cluster=replace(plan.cluster, num_machines=1)),
                        dict(cluster=replace(plan.cluster, local_rank=0))):
            with pytest.raises(ValueError):
                api(replace(plan, **changed), commit="a" * 40)
    zero3 = build_launch_plan(stage="stage1", tuning_method="dora", zero_stage=3,
                              run_id="vg-probe-z3", main_process_port=29590)
    with pytest.raises(ValueError):
        api(zero3, commit="a" * 40)


def test_fixed_probe_coordinator_dry_run_has_both_attempts(capsys):
    assert _training_main("stage1", ["--vision-gain-runtime-probe", "--run-id", "vg-route",
                                    "--zero-stage", "1", "--main-process-port", "29590", "--dry-run"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["kind"] == "stage1-vision-gain-runtime-probe"
    assert set(record["attempts"]) == {"initial", "resume"}
    assert record["resolved_profile"] == dict(microbatch=1, accumulation=4, global_batch=16,
           learning_rate=5e-5, seed=42, warmup=10, ce_weight=0.1, alpha=0.001, max_steps=2)


def test_probe_resume_transfer_preserves_exclusive_launch_directory(tmp_path, monkeypatch):
    # A transfer through the fresh resume runner must reproduce the real collision:
    # pull_resume_artifacts publishes its files-list before prepare_plan reserves it.
    commit = "a" * 40
    run_id = "vg-resume-directory"
    checkpoint = tmp_path / "checkpoints" / run_id / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    attempt_root = tmp_path / "logs/multinode" / run_id
    entry = artifacts.ArtifactEntry(Path("rng_state_2.pth"), 3, 1)

    def command_runner(command, **kwargs):
        # Substitute only actual process/network execution; retain real files-list
        # publication, transfer metadata checks and exclusive plan serialization.
        if command[0] == "/usr/bin/rsync":
            (checkpoint / entry.relative_path).write_bytes(b"rng")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    class FilesystemRunner(runtime.SubprocessCoordinatorRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, command_runner=command_runner)

        def read_lifecycle_phase_evidence(self, rank, phase):
            assert phase == "initial"
            return json.dumps({
                "kind": "stage1-vision-gain-runtime-probe", "commit": commit,
                "ranks": [{"global_rank": rank, "steps": 1} for rank in range(4)],
            }).encode()

    def redistribute(runner, *, checkpoint_path, checkpoint_identity,
                     requested_identity, destination):
        # Manifest discovery is remote. Exercise the real pull and its side effects
        # with one missing rank-local artifact, the branch seen in the failed run.
        assert checkpoint_path == destination == checkpoint
        assert checkpoint_identity == requested_identity
        manifest = artifacts.NodeArtifactManifest(
            schema_version=1, run_id=run_id, node_rank=0,
            process_ranks=(0, 1, 2, 3), entries=(entry,),
            runtime_identity=checkpoint_identity, checkpoint_relative=Path("checkpoint-1"),
        )
        runner._reserved_transfer_destinations[destination] = destination
        runner.pull_resume_artifacts(manifest, (entry,), destination)

    class ResumeLaunched(Exception):
        pass

    def monitor(plan, runner, **kwargs):
        if plan.attempt_id == "resume":
            raise ResumeLaunched
        return {0: 0, 1: 0}

    monkeypatch.setattr(coordinator, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(coordinator, "_REMOTE_CONTAINER_REPOSITORY_BASE", tmp_path)
    monkeypatch.setattr(artifacts, "_LOCAL_REPOSITORY", tmp_path)
    monkeypatch.setattr(coordinator, "_repository_git_stdout", lambda args: commit)
    monkeypatch.setattr(coordinator, "build_preflight_commands", lambda *args, **kwargs: None)
    monkeypatch.setattr(coordinator, "collect_node_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(coordinator, "validate_preflight", lambda *args: SimpleNamespace(commit=commit))
    monkeypatch.setattr(coordinator, "SubprocessCoordinatorRunner", FilesystemRunner)
    monkeypatch.setattr(coordinator, "collect_checkpoint_artifacts", lambda *args, **kwargs: ())
    monkeypatch.setattr(coordinator, "redistribute_resume_checkpoint", redistribute)
    monkeypatch.setattr(coordinator, "monitor_owned_jobs", monitor)

    with pytest.raises(ResumeLaunched):
        _training_main("stage1", ["--vision-gain-runtime-probe", "--run-id", run_id,
                                 "--zero-stage", "1", "--main-process-port", "29590"])

    assert (attempt_root / "initial/artifact-resume-rank1.files").read_bytes() == b"rng_state_2.pth\0"
    assert not (attempt_root / "resume/artifact-resume-rank1.files").exists()
    assert (checkpoint / entry.relative_path).read_bytes() == b"rng"
    saved = json.loads((attempt_root / "resume/launch-plan.json").read_text())
    assert saved["attempt_id"] == "resume"
    # The correction must retain exclusive ownership, including duplicate refusal.
    plan = build_launch_plan(stage="stage1", tuning_method="dora", zero_stage=1,
                             run_id=run_id, main_process_port=29590, attempt_id="resume")
    with pytest.raises(FileExistsError):
        runtime.serialize_launch_plan(plan, log_root=tmp_path / "logs/multinode")
