import dataclasses
import json
import os
import subprocess
from pathlib import Path

import pytest

from src.multinode_artifacts import (
    ArtifactEntry,
    NodeArtifactManifest,
    build_inference_pull_command,
    build_node_manifest,
    build_pull_command,
    build_push_command,
    prepare_inference_copy,
    prepare_resume_transfer,
    publish_files_list,
    publish_node_manifest,
    select_resume_entries,
    validate_collection,
    validate_resume_identity,
    validate_transfer_destination,
)
from src.multinode_runtime import collect_checkpoint_artifacts
from src.multinode import build_launch_plan
from src.multinode_runtime import SubprocessCoordinatorRunner, build_owned_tmux_commands


def identity(**changes):
    values = {
        "stage": "stage1",
        "tuning_method": "lora",
        "zero_stage": 3,
        "deepspeed_config_digest": "a" * 64,
        "world_size": 4,
        "rank_mapping": {"0": [0, 1], "1": [2, 3]},
        "global_step": 2,
        "optimizer": True,
        "scheduler": True,
        "rng_ranks": [0, 1, 2, 3],
        "stage_sidecar": "stage1_peft_metadata.json",
        "commit": "b" * 40,
        "run_id": "stage1-lora-zero3-four-gpu",
    }
    values.update(changes)
    return values


def entry(rank, *, size=8, node_rank=None):
    owner = rank // 2 if node_rank is None else node_rank
    return ArtifactEntry(
        Path(f"checkpoint-2/global_step2/zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"),
        size,
        owner,
    )


def manifest_for(
    ranks,
    *,
    node_rank=None,
    runtime_identity=None,
    run_id="stage1-lora-zero3-four-gpu",
    completion_marker=True,
    extra_entries=(),
):
    ranks = tuple(ranks)
    node = ranks[0] // 2 if node_rank is None else node_rank
    return NodeArtifactManifest(
        schema_version=1,
        run_id=run_id,
        node_rank=node,
        process_ranks=(0, 1) if node == 0 else (2, 3),
        entries=tuple(entry(rank, node_rank=node) for rank in ranks)
        + tuple(extra_entries),
        runtime_identity=identity() if runtime_identity is None else runtime_identity,
        completion_marker=completion_marker,
    )


def complete_manifests(**changes):
    return (
        manifest_for((0, 1), **changes),
        manifest_for((2, 3), **changes),
    )


def launch_plan():
    return build_launch_plan(
        stage="stage1",
        tuning_method="lora",
        zero_stage=3,
        run_id="stage1-lora-zero3-four-gpu",
        main_process_port=29631,
    )


def test_manifest_records_size_and_rank_without_weight_hash(tmp_path, monkeypatch):
    shard = tmp_path / "global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt"
    shard.parent.mkdir()
    shard.write_bytes(b"rank-two")

    def reject_content_read(*args, **kwargs):
        raise AssertionError("checkpoint contents must never be opened")

    monkeypatch.setattr(Path, "read_bytes", reject_content_read)
    manifest = build_node_manifest(
        tmp_path,
        node_rank=1,
        process_ranks=(2, 3),
        run_id="stage1-lora-zero3-four-gpu",
        runtime_identity=identity(),
    )

    assert manifest.entries == (
        ArtifactEntry(
            relative_path=Path(
                "global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt"
            ),
            byte_size=8,
            node_rank=1,
        ),
    )
    assert manifest.entries[0].process_rank == 2


def test_manifest_rejects_symlink_and_nonregular_file(tmp_path):
    outside = tmp_path.parent / "outside-shard"
    outside.write_bytes(b"outside")
    (tmp_path / "linked.pt").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        build_node_manifest(
            tmp_path,
            node_rank=1,
            process_ranks=(2, 3),
            runtime_identity=identity(),
        )

    (tmp_path / "linked.pt").unlink()
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(ValueError, match="regular"):
        build_node_manifest(
            tmp_path,
            node_rank=1,
            process_ranks=(2, 3),
            runtime_identity=identity(),
        )


def test_manifest_rejects_shard_owned_by_another_node(tmp_path):
    shard = tmp_path / "global_step2/zero_pp_rank_0_mp_rank_00_optim_states.pt"
    shard.parent.mkdir()
    shard.write_bytes(b"foreign")

    with pytest.raises(ValueError, match="assigned process ranks"):
        build_node_manifest(
            tmp_path,
            node_rank=1,
            process_ranks=(2, 3),
            runtime_identity=identity(),
        )


def test_manifest_json_round_trip_preserves_path_and_identity():
    original = manifest_for((2, 3))
    assert NodeArtifactManifest.from_json(original.to_json()) == original

    payload = original.entries[0].to_json()
    payload["relative_path"] = "../escape.pt"
    with pytest.raises(ValueError, match="relative path"):
        ArtifactEntry.from_json(payload)


def test_collection_requires_all_four_process_ranks():
    with pytest.raises(ValueError, match="ranks 0 through 3"):
        validate_collection((manifest_for((0, 1)),), world_size=4)


@pytest.mark.parametrize(
    ("manifests", "message"),
    [
        (
            (
                dataclasses.replace(
                    manifest_for((0, 1)),
                    entries=(ArtifactEntry(Path("../escape.pt"), 8, 0),),
                ),
                manifest_for((2, 3)),
            ),
            "relative path",
        ),
        (
            (
                manifest_for(
                    (0, 1),
                    extra_entries=(ArtifactEntry(Path("trainer_state.json"), 8, 0),),
                ),
                manifest_for(
                    (2, 3),
                    extra_entries=(ArtifactEntry(Path("trainer_state.json"), 9, 1),),
                ),
            ),
            "different sizes",
        ),
        (
            (
                manifest_for((0, 1), run_id="run-a"),
                manifest_for((2, 3), run_id="run-b"),
            ),
            "run ID",
        ),
        (
            (
                manifest_for((0, 1), completion_marker=False),
                manifest_for((2, 3)),
            ),
            "completion marker",
        ),
        (
            (
                manifest_for((0, 1)),
                manifest_for((2, 3), runtime_identity=identity(commit="c" * 40)),
            ),
            "runtime identity",
        ),
    ],
)
def test_collection_rejects_invalid_manifests(manifests, message):
    with pytest.raises(ValueError, match=message):
        validate_collection(manifests, world_size=4)


def test_collection_accepts_identical_small_common_metadata_once():
    common_remote = ArtifactEntry(Path("trainer_state.json"), 31, 0)
    common_local = ArtifactEntry(Path("trainer_state.json"), 31, 1)
    entries = validate_collection(
        (
            manifest_for((0, 1), extra_entries=(common_remote,)),
            manifest_for((2, 3), extra_entries=(common_local,)),
        ),
        world_size=4,
    )
    assert [item.relative_path for item in entries].count(Path("trainer_state.json")) == 1


def test_collection_never_treats_equal_sized_weights_as_common_metadata():
    remote_weight = ArtifactEntry(Path("adapter_model.safetensors"), 31, 0)
    local_weight = ArtifactEntry(Path("adapter_model.safetensors"), 31, 1)
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        validate_collection(
            (
                manifest_for((0, 1), extra_entries=(remote_weight,)),
                manifest_for((2, 3), extra_entries=(local_weight,)),
            ),
            world_size=4,
        )


def test_transfer_refuses_preexisting_destination(tmp_path):
    destination = tmp_path / "global_step2/model.safetensors"
    destination.parent.mkdir()
    destination.write_bytes(b"owned-by-earlier-run")
    with pytest.raises(FileExistsError, match="destination"):
        validate_transfer_destination((destination,))


def test_transfer_allows_only_validated_identical_common_metadata(tmp_path):
    destination = tmp_path / "trainer_state.json"
    destination.write_bytes(b"small")
    validate_transfer_destination(
        (destination,),
        validated_identical_common_metadata={destination: 5},
    )
    with pytest.raises(FileExistsError, match="destination"):
        validate_transfer_destination(
            (destination,),
            validated_identical_common_metadata={destination: 6},
        )
    weight = tmp_path / "adapter_model.safetensors"
    weight.write_bytes(b"small")
    with pytest.raises(FileExistsError, match="destination"):
        validate_transfer_destination(
            (weight,), validated_identical_common_metadata={weight: 5}
        )


def test_push_never_overwrites_and_is_scoped_to_the_run():
    command = build_push_command(manifest_for((2, 3)))
    assert "--archive" in command
    assert "--ignore-existing" in command
    assert "--files-from" in command
    assert "--from0" in command
    assert "--checksum" not in command
    assert command[-2] == "/root/workspace/ILVR/checkpoints/stage1-lora-zero3-four-gpu/"
    assert command[-1] == (
        "korea@10.1.102.41:"
        "/home/korea/Workspace/hjh/workspace/ILVR/checkpoints/"
        "stage1-lora-zero3-four-gpu/"
    )
    assert not any("venv" in argument or "/data/" in argument for argument in command)


def test_pull_selects_only_local_process_ranks_and_common_metadata(tmp_path):
    common = ArtifactEntry(Path("trainer_state.json"), 31, 0)
    combined = dataclasses.replace(
        manifest_for((0, 1), extra_entries=(common,)),
        process_ranks=(0, 1, 2, 3),
        entries=tuple(entry(rank, node_rank=0) for rank in range(4)) + (common,),
    )

    selected = select_resume_entries(combined)
    assert {item.process_rank for item in selected} == {None, 2, 3}
    command = build_pull_command(combined, tmp_path / "checkpoint-copy")
    assert command[-2].startswith("korea@10.1.102.41:")
    assert command[-1] == f"{tmp_path}/checkpoint-copy/"
    assert "--checksum" not in command


@pytest.mark.parametrize(
    "field", ("stage", "tuning_method", "zero_stage", "world_size", "run_id")
)
def test_resume_rejects_identity_change_before_transfer(field):
    replacements = {
        "stage": "stage2",
        "tuning_method": "dora",
        "zero_stage": 2,
        "world_size": 8,
        "run_id": "another-run",
    }
    requested = identity(**{field: replacements[field]})
    with pytest.raises(ValueError, match=field):
        validate_resume_identity(identity(), requested)


@pytest.mark.parametrize(
    "field",
    (
        "deepspeed_config_digest",
        "rank_mapping",
        "global_step",
        "optimizer",
        "scheduler",
        "rng_ranks",
        "stage_sidecar",
        "commit",
    ),
)
def test_resume_rejects_changed_checkpoint_state_field(field):
    changed = {
        "deepspeed_config_digest": "d" * 64,
        "rank_mapping": {"0": [0], "1": [1, 2, 3]},
        "global_step": 3,
        "optimizer": False,
        "scheduler": False,
        "rng_ranks": [0, 1],
        "stage_sidecar": "stage2_peft_metadata.json",
        "commit": "e" * 40,
    }
    with pytest.raises(ValueError, match=field):
        validate_resume_identity(identity(), identity(**{field: changed[field]}))


def test_inference_copy_requires_fresh_nonsymlink_local_destination(tmp_path):
    remote = manifest_for((0, 1))
    destination = tmp_path / "inference-copy"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="fresh inference destination"):
        build_inference_pull_command(remote, destination)

    destination.rmdir()
    destination.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError, match="fresh inference destination"):
        build_inference_pull_command(remote, destination)


def test_inference_pull_transfers_only_manifest_listed_files(tmp_path):
    remote = dataclasses.replace(
        manifest_for((0, 1)),
        entries=(ArtifactEntry(Path("adapter_model.safetensors"), 99, 0),),
    )
    destination = tmp_path / "fresh-inference"
    command = build_inference_pull_command(remote, destination)
    assert "--files-from" in command
    assert "--ignore-existing" in command
    assert command[-1] == f"{destination}/"
    assert not destination.exists()


def test_inference_pull_rejects_untrusted_manifest_path(tmp_path):
    remote = dataclasses.replace(
        manifest_for((0, 1)),
        entries=(ArtifactEntry(Path("../escape.safetensors"), 99, 0),),
    )
    with pytest.raises(ValueError, match="relative path"):
        build_inference_pull_command(remote, tmp_path / "fresh-inference")


def test_small_manifest_and_files_list_publish_exclusively(tmp_path):
    manifest = manifest_for((2, 3))
    manifest_path = tmp_path / "control" / "node-manifest.json"
    files_path = tmp_path / "control" / "push.files"

    publish_node_manifest(manifest_path, manifest)
    publish_files_list(files_path, manifest.entries)

    assert json.loads(manifest_path.read_text()) == manifest.to_json()
    assert files_path.read_bytes().split(b"\0") == [
        b"checkpoint-2/global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt",
        b"checkpoint-2/global_step2/zero_pp_rank_3_mp_rank_00_optim_states.pt",
        b"",
    ]
    with pytest.raises(FileExistsError):
        publish_node_manifest(manifest_path, manifest)
    with pytest.raises(FileExistsError):
        publish_files_list(files_path, manifest.entries)


class FakeArtifactRunner:
    def __init__(self):
        self.events = []
        common_remote = ArtifactEntry(Path("trainer_state.json"), 31, 0)
        common_local = ArtifactEntry(Path("trainer_state.json"), 31, 1)
        self.manifests = {
            0: manifest_for((0, 1), extra_entries=(common_remote,)),
            1: manifest_for((2, 3), extra_entries=(common_local,)),
        }

    def read_artifact_manifest(self, machine_rank, checkpoint_path, runtime_identity):
        self.events.append(f"manifest-rank{machine_rank}")
        assert checkpoint_path.name == "stage1-lora-zero3-four-gpu"
        assert runtime_identity == identity()
        return self.manifests[machine_rank]

    def validate_push_destination(self, manifest, entries):
        self.events.append("destination-precondition")
        assert manifest.node_rank == 1
        assert {item.process_rank for item in entries} == {None, 2, 3}

    def push_artifacts(self, manifest, entries):
        self.events.append("push")


def test_collection_runs_after_both_zero_exits_and_checks_destination_first(tmp_path):
    runner = FakeArtifactRunner()
    checkpoint = tmp_path / "stage1-lora-zero3-four-gpu"

    entries = collect_checkpoint_artifacts(
        runner,
        checkpoint_path=checkpoint,
        runtime_identity=identity(),
        exit_codes={0: 0, 1: 0},
        world_size=4,
    )

    assert {item.process_rank for item in entries if item.process_rank is not None} == {
        0,
        1,
        2,
        3,
    }
    assert runner.events == [
        "manifest-rank0",
        "manifest-rank1",
        "destination-precondition",
        "push",
    ]


def test_failed_node_exit_never_fabricates_collection(tmp_path):
    runner = FakeArtifactRunner()
    with pytest.raises(ValueError, match="zero exits"):
        collect_checkpoint_artifacts(
            runner,
            checkpoint_path=tmp_path / "stage1-lora-zero3-four-gpu",
            runtime_identity=identity(),
            exit_codes={0: 0, 1: 7},
            world_size=4,
        )
    assert runner.events == []


class FakeTransferRunner:
    def __init__(self):
        self.events = []

    def validate_resume_destination(self, entries, destination):
        self.events.append("resume-destination")

    def pull_resume_artifacts(self, manifest, entries, destination):
        self.events.append("resume-pull")

    def validate_inference_manifest(self, manifest):
        self.events.append("inference-manifest")

    def pull_inference_artifacts(self, manifest, destination):
        self.events.append("inference-pull")


def test_resume_validates_identity_before_destination_or_transfer(tmp_path):
    runner = FakeTransferRunner()
    remote = dataclasses.replace(
        manifest_for((0, 1)),
        process_ranks=(0, 1, 2, 3),
        entries=tuple(entry(rank, node_rank=0) for rank in range(4)),
    )
    with pytest.raises(ValueError, match="zero_stage"):
        prepare_resume_transfer(
            identity(),
            identity(zero_stage=2),
            remote,
            tmp_path / "resume",
            runner,
        )
    assert runner.events == []

    prepare_resume_transfer(
        identity(), identity(), remote, tmp_path / "resume", runner
    )
    assert runner.events == ["resume-destination", "resume-pull"]


def test_inference_preparation_validates_manifest_before_one_pull(tmp_path):
    runner = FakeTransferRunner()
    remote = dataclasses.replace(
        manifest_for((0, 1)),
        entries=(ArtifactEntry(Path("adapter_model.safetensors"), 99, 0),),
    )
    destination = tmp_path / "fresh-inference"

    prepare_inference_copy(remote, destination, runner)

    assert runner.events == ["inference-manifest", "inference-pull"]
    assert not destination.exists()


def test_subprocess_runner_uses_bounded_node_json_then_exact_rsync_list(tmp_path):
    plan = launch_plan()
    commands = build_owned_tmux_commands(plan, commit="b" * 40)
    remote_manifest = manifest_for(
        (0, 1),
        extra_entries=(ArtifactEntry(Path("trainer_state.json"), 31, 0),),
    )
    local_manifest = manifest_for(
        (2, 3),
        extra_entries=(ArtifactEntry(Path("trainer_state.json"), 31, 1),),
    )
    calls = []

    def fake_command(command, *, input_bytes=None, timeout_seconds=None):
        argv = tuple(command)
        calls.append((argv, input_bytes, timeout_seconds))
        display = " ".join(argv)
        if "artifact-manifest" in display:
            payload = remote_manifest if argv[0] == "/usr/bin/ssh" else local_manifest
            stdout = json.dumps(payload.to_json())
        else:
            stdout = ""
        return __import__("subprocess").CompletedProcess(
            command, 0, stdout=stdout, stderr=""
        )

    runner = SubprocessCoordinatorRunner(
        plan,
        commands=commands,
        commit="b" * 40,
        log_root=tmp_path / "logs",
        command_runner=fake_command,
    )
    checkpoint = Path("/root/workspace/ILVR/checkpoints") / plan.run_id

    collected = collect_checkpoint_artifacts(
        runner,
        checkpoint_path=checkpoint,
        runtime_identity=identity(),
        exit_codes={0: 0, 1: 0},
        world_size=4,
    )

    assert len(collected) == 5
    displays = [" ".join(call[0]) for call in calls]
    assert sum("artifact-manifest" in display for display in displays) == 2
    destination_call = next(
        call for call in calls if "artifact-destination-check" in " ".join(call[0])
    )
    assert destination_call[0][0] == "/usr/bin/ssh"
    assert destination_call[1] is not None
    rsync = next(call[0] for call in calls if call[0][0] == "/usr/bin/rsync")
    assert "--ignore-existing" in rsync
    assert "--files-from" in rsync
    assert "--checksum" not in rsync
    files_path = Path(rsync[rsync.index("--files-from") + 1])
    assert files_path.read_bytes().split(b"\0") == [
        b"checkpoint-2/global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt",
        b"checkpoint-2/global_step2/zero_pp_rank_3_mp_rank_00_optim_states.pt",
        b"trainer_state.json",
        b"",
    ]
    assert all(call[2] == 30 for call in calls)


def test_artifact_manifest_cli_emits_only_small_json_metadata(tmp_path):
    run_directory = tmp_path / "stage1-lora-zero3-four-gpu"
    shard = run_directory / "global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"rank-two")

    completed = subprocess.run(
        (
            Path(__file__).resolve().parents[2] / "venv/ilvr-venv/bin/python",
            "-m",
            "src.multinode",
            "artifact-manifest",
            "--run-directory",
            str(run_directory),
            "--run-id",
            run_directory.name,
            "--node-rank",
            "1",
            "--process-rank",
            "2",
            "--process-rank",
            "3",
            "--runtime-identity-json",
            json.dumps(identity()),
            "--completion-marker",
        ),
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["entries"] == [
        {
            "relative_path": "global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt",
            "byte_size": 8,
            "node_rank": 1,
            "process_rank": 2,
        }
    ]
    assert "sha256" not in completed.stdout


def test_destination_cli_rejects_ranked_collision_but_allows_identical_common(tmp_path):
    run_directory = tmp_path / "stage1-lora-zero3-four-gpu"
    ranked = run_directory / "global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt"
    common = run_directory / "trainer_state.json"
    ranked.parent.mkdir(parents=True)
    ranked.write_bytes(b"old-rank")
    common.write_bytes(b"small")
    payload = {
        "entries": [
            ArtifactEntry(
                Path("global_step2/zero_pp_rank_2_mp_rank_00_optim_states.pt"),
                8,
                1,
            ).to_json(),
            ArtifactEntry(Path("trainer_state.json"), 5, 1).to_json(),
        ],
        "runtime_identity": identity(),
    }
    command = (
        Path(__file__).resolve().parents[2] / "venv/ilvr-venv/bin/python",
        "-m",
        "src.multinode",
        "artifact-destination-check",
        "--run-directory",
        str(run_directory),
        "--run-id",
        run_directory.name,
    )

    completed = subprocess.run(
        command,
        input=json.dumps(payload),
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 2
    assert "destination" in completed.stderr

    ranked.unlink()
    completed = subprocess.run(
        command,
        input=json.dumps(payload),
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
