"""Strict V2 native identity and ownership contracts; legacy V1 controls."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile

import pytest

from src.multinode_artifacts import (ArtifactEntry, build_node_manifest, validate_collection,
    validate_runtime_identity, validate_resume_identity, observe_checkpoint_identity)
from .native_ddp_helpers import ROOT, ddp_identity
from .test_artifacts import identity as legacy_identity, strict_manifest


def test_ddp_v2_identity_and_legacy_v1_are_distinct():
    assert validate_runtime_identity(ddp_identity()) == ddp_identity()
    old = legacy_identity()
    assert validate_runtime_identity(old) == old
    validate_collection((strict_manifest(0), strict_manifest(1)), world_size=4)
    with pytest.raises(ValueError):
        validate_resume_identity(old, ddp_identity())


@pytest.mark.parametrize("changes", [dict(zero_stage=0), dict(zero_stage=1),
    dict(deepspeed_config_digest="a" * 64), dict(model_state_layout="replicated"),
    dict(stage="stage1"), dict(execution_backend="deepspeed"), dict(objective_enabled=1),
    dict(world_size=True), dict(global_step=True), dict(rng_ranks=[0, 1, 2]),
    dict(optimizer=False), dict(scheduler=False)])
def test_mixed_native_identity_rejected(changes):
    with pytest.raises(ValueError):
        validate_runtime_identity(ddp_identity(**changes))


@pytest.mark.parametrize("field", ["execution_backend", "objective_enabled"])
def test_missing_v2_field_rejected(field):
    value = ddp_identity()
    del value[field]
    with pytest.raises(ValueError):
        validate_runtime_identity(value)


def test_legacy_v1_rejects_v2_fields():
    with pytest.raises(ValueError):
        validate_runtime_identity(legacy_identity(execution_backend="ddp", objective_enabled=False))


@pytest.fixture
def native_tree():
    """Structural fixtures only; actual Trainer serialization is covered separately."""
    root = Path(tempfile.mkdtemp(prefix="native-contract-", dir=ROOT / "checkpoints"))
    for node in (0, 1):
        checkpoint = root / f"node{node}" / "ddp-checkpoint-contract/checkpoint-2"
        checkpoint.mkdir(parents=True)
        for rank in ((0, 1) if node == 0 else (2, 3)):
            (checkpoint / f"rng_state_{rank}.pth").write_bytes(b"rng")
    checkpoint = root / "node0/ddp-checkpoint-contract/checkpoint-2"
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "training_args.bin"):
        (checkpoint / name).write_bytes(b"structural fixture")
    for name, value in {
        "config.json": {"stage": "stage2"}, "preprocessor_config.json": {},
        "tokenizer_config.json": {}, "tokenizer.json": {}, "trainer_state.json": {"global_step": 2},
        "deepspeed-runtime.json": dict(schema_version=1, enabled=False, zero_stage=None,
                config_path=None, config_digest=None, world_size=4),
        "stage2-sampler.json": dict(schema_version=1, seed=42, world_size=4, dataset_size=24,
                bucket_sizes={"1": 24}, padded_samples=0, global_step=2),
        "stage2_objective.json": dict(schema_version=1, config=dict(enable_stage2_objective=True,
                lambda_dyn=0.0, lambda_vis=0.0, stage2_reference_source=None),
                reference_identity=None, vocabulary_identity=None),
    }.items():
        (checkpoint / name).write_text(json.dumps(value))
    return root


def manifests(root, identity=None):
    identity = ddp_identity() if identity is None else identity
    return tuple(build_node_manifest(root / f"node{node}" / "ddp-checkpoint-contract/checkpoint-2",
            node_rank=node, process_ranks=((0, 1) if node == 0 else (2, 3)),
            runtime_identity=identity, checkpoint_root=root / f"node{node}") for node in (0, 1))


def test_native_original_ownership_and_schema(native_tree):
    remote, local = manifests(native_tree)
    assert (remote.schema_version, local.schema_version) == (2, 2)
    assert sorted(str(e.relative_path) for e in local.entries) == ["rng_state_2.pth", "rng_state_3.pth"]
    combined = validate_collection((remote, local), world_size=4)
    assert len([e for e in combined if e.relative_path == Path("optimizer.pt")]) == 1
    assert {e.process_rank for e in combined if e.process_rank is not None} == {0, 1, 2, 3}


@pytest.mark.parametrize("name", ["optimizer.pt", "scheduler.pt", "training_args.bin", "trainer_state.json",
    "deepspeed-runtime.json", "stage2-sampler.json", "stage2_objective.json", "rng_state_0.pth",
    "config.json", "preprocessor_config.json", "tokenizer.json", "model.safetensors"])
def test_native_missing_required_file_rejected(native_tree, name):
    (native_tree / "node0/ddp-checkpoint-contract/checkpoint-2" / name).unlink()
    with pytest.raises(ValueError):
        validate_collection(manifests(native_tree), world_size=4)


@pytest.mark.parametrize("name", ["latest", "global_step2/zero_pp_rank_0_mp_rank_00_optim_states.pt",
    "global_step2/mp_rank_00_model_states.pt", "mp_rank_00_model_states.pt", "nested/rng_state_0.pth"])
def test_native_rejects_deepspeed_or_nested_rng(native_tree, name):
    path = native_tree / "node0/ddp-checkpoint-contract/checkpoint-2" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError):
        validate_collection(manifests(native_tree), world_size=4)


@pytest.mark.parametrize("name", ["optimizer.pt", "stage2_objective.json", "rng_state_0.pth", "nested/rng_state_2.pth"])
def test_original_node1_is_strictly_rng_only(native_tree, name):
    path = native_tree / "node1/ddp-checkpoint-contract/checkpoint-2" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError):
        manifests(native_tree)


@pytest.mark.parametrize("filename,field,value", [("trainer_state.json", "global_step", 3),
    ("stage2-sampler.json", "global_step", 3), ("stage2-sampler.json", "world_size", 2),
    ("deepspeed-runtime.json", "world_size", 2), ("deepspeed-runtime.json", "enabled", True),
    ("deepspeed-runtime.json", "config_path", "stale.json")])
def test_native_observed_identity_conflicts(native_tree, filename, field, value):
    path = native_tree / "node0/ddp-checkpoint-contract/checkpoint-2" / filename
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        manifests(native_tree)


def test_native_schema_mismatch_duplicate_and_identity_conflict(native_tree):
    remote, local = manifests(native_tree)
    for changed in (replace(local, schema_version=1),
                    replace(local, entries=local.entries + (local.entries[0],)),
                    replace(local, runtime_identity=ddp_identity(objective_enabled=False)),
                    replace(local, entries=local.entries + (ArtifactEntry(Path("../escape"), 1, 1),))):
        with pytest.raises(ValueError):
            validate_collection((remote, changed), world_size=4)


def test_native_feature_off_rejects_enabled_sidecar(native_tree):
    with pytest.raises(ValueError):
        manifests(native_tree, ddp_identity(objective_enabled=False))
    (native_tree / "node0/ddp-checkpoint-contract/checkpoint-2/stage2_objective.json").unlink()
    validate_collection(manifests(native_tree, ddp_identity(objective_enabled=False)), world_size=4)


def test_native_resume_rejects_objective_conversion():
    with pytest.raises(ValueError):
        validate_resume_identity(ddp_identity(), ddp_identity(objective_enabled=False))


@pytest.mark.parametrize("name", ["optimizer.pt", "rng_state_0.pth", "stage2-sampler.json"])
def test_native_resume_destination_rejects_partial_replica(native_tree, name):
    from src.multinode_artifacts import native_resume_destination_is_canonical
    checkpoint = native_tree / "node1/ddp-checkpoint-contract/checkpoint-2"
    (checkpoint / name).write_bytes(b"partial")
    with pytest.raises(ValueError):
        native_resume_destination_is_canonical(checkpoint, runtime_identity=ddp_identity(),
                                               checkpoint_root=native_tree / "node1")


@pytest.mark.parametrize("enabled", [False, True])
def test_coordinator_emits_native_identity_from_frozen_plan(enabled):
    from src.multinode import build_launch_plan, _checkpoint_artifact_identity
    from src.stage2.objective import Stage2ObjectiveConfig
    plan = build_launch_plan(stage="stage2", tuning_method="full", backend="ddp",
        run_id="ddp-checkpoint-contract", main_process_port=29717,
        stage2_objective_config=Stage2ObjectiveConfig(True) if enabled else None)
    assert _checkpoint_artifact_identity(plan, commit="b" * 40, global_step=2) == ddp_identity(objective_enabled=enabled)


def test_native_reader_rejects_manifest_schema_identity_mix(native_tree):
    from src.multinode_artifacts import NodeArtifactManifest
    remote, _ = manifests(native_tree)
    assert NodeArtifactManifest.from_json(remote.to_json()) == remote
    payload = remote.to_json()
    payload["schema_version"] = 1
    with pytest.raises(ValueError):
        NodeArtifactManifest.from_json(payload)


def test_native_observer_rejects_conflicting_model_stage(native_tree):
    checkpoint = native_tree / "node0/ddp-checkpoint-contract/checkpoint-2"
    (checkpoint / "config.json").write_text('{"stage":"stage1"}')
    with pytest.raises(ValueError):
        observe_checkpoint_identity(checkpoint, ddp_identity())


def test_native_original_node1_rejects_empty_directory(native_tree):
    (native_tree / "node1/ddp-checkpoint-contract/checkpoint-2/foreign").mkdir()
    with pytest.raises(ValueError):
        manifests(native_tree)


def test_native_resume_cli_selects_only_original_or_complete_views(native_tree, capsys):
    from src.multinode import _artifact_manifest
    from .native_ddp_helpers import FilesystemTransport
    from src.multinode_runtime import collect_checkpoint_artifacts, redistribute_resume_checkpoint
    runner = FilesystemTransport(native_tree)
    args = ["--run-directory", str(runner.path(1)), "--run-id", "ddp-checkpoint-contract",
            "--checkpoint-root", str(runner.root / "node1"), "--node-rank", "1",
            "--process-rank", "2", "--process-rank", "3", "--runtime-identity-json",
            json.dumps(ddp_identity()), "--completion-marker", "--resume-destination"]
    assert _artifact_manifest(args) == 0
    assert json.loads(capsys.readouterr().out)["process_ranks"] == [2, 3]
    collect_checkpoint_artifacts(runner, checkpoint_path=runner.path(0), runtime_identity=ddp_identity(),
                                exit_codes={0: 0, 1: 0}, world_size=4)
    redistribute_resume_checkpoint(runner, checkpoint_path=runner.path(0), checkpoint_identity=ddp_identity(),
                                    requested_identity=ddp_identity(), destination=runner.path(0))
    assert _artifact_manifest(args) == 0
    assert json.loads(capsys.readouterr().out)["process_ranks"] == [0, 1, 2, 3]
    (runner.path(1) / "optimizer.pt").unlink()
    with pytest.raises(SystemExit):
        _artifact_manifest(args)


def test_native_symlink_checkpoint_and_payload_rejected(native_tree):
    root = native_tree / "node0"
    checkpoint = root / "ddp-checkpoint-contract/checkpoint-2"
    outside = native_tree / "outside.json"
    outside.write_text('{}')
    (checkpoint / "config.json").unlink()
    (checkpoint / "config.json").symlink_to(outside)
    with pytest.raises(ValueError):
        manifests(native_tree)


def test_native_peer_agreement_cannot_override_trusted_identity_before_transfer(native_tree):
    from .native_ddp_helpers import FilesystemTransport
    from src.multinode_runtime import collect_checkpoint_artifacts
    runner = FilesystemTransport(native_tree)
    original = runner.read_artifact_manifest

    def untrusted_read(*args, **kwargs):
        return replace(original(*args, **kwargs), runtime_identity=ddp_identity(commit="c" * 40))

    runner.read_artifact_manifest = untrusted_read
    with pytest.raises(ValueError):
        collect_checkpoint_artifacts(runner, checkpoint_path=runner.path(0), runtime_identity=ddp_identity(),
                                    exit_codes={0: 0, 1: 0}, world_size=4)
    assert runner.pushed == []
