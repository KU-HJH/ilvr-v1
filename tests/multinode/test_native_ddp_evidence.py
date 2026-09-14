"""Native producer identity and canonical completion, including real Gloo failures."""
import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from stage2.evidence import build_stage2_producer_evidence, decode_stage2_producer_evidence
from src.multinode import build_launch_plan
from src.multinode_runtime import write_training_witness
from src.stage2.objective import Stage2ObjectiveConfig
from .native_ddp_helpers import ROOT, FilesystemTransport, ddp_identity, launch_workers, result_directory, validate_native_publication


def records():
    return [dict(rank=r, global_step=2, optimizer_step=2, loss=0.5+r,
                 gradient_norm=1.0, worker_error=None) for r in range(4)]


def literal(checkpoint, digest=None):
    return dict(schema_version=2, kind="stage2-producer-evidence", run_id="ddp-checkpoint-contract",
        stage="stage2", method="full", zero_stage=None, execution_backend="ddp",
        objective_identity_sha256=digest, world_size=4, process_ranks=[0, 1, 2, 3],
        global_step=2, optimizer_step=2, checkpoint_path=str(checkpoint), artifact_kind="full",
        finite_loss=True, finite_gradient=True, worker_errors=[], ranks=records())


def decode(payload, checkpoint):
    return decode_stage2_producer_evidence(json.dumps(payload).encode(), expected_run_id="ddp-checkpoint-contract",
        expected_method="full", expected_zero_stage=None, expected_execution_backend="ddp",
        expected_checkpoint=checkpoint, expected_global_step=2)


def test_v2_builder_and_literal_decode():
    checkpoint = Path("/checkpoints/ddp-checkpoint-contract/checkpoint-2")
    actual = build_stage2_producer_evidence(run_id="ddp-checkpoint-contract", method="full", zero_stage=None,
        checkpoint_path=checkpoint, artifact_kind="full", rank_records=records(), execution_backend="ddp",
        objective_identity_sha256="a"*64)
    assert actual == literal(checkpoint, "a"*64)
    assert decode(literal(checkpoint), checkpoint) == literal(checkpoint)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("zero_stage", 0),
    ("execution_backend", "deepspeed"), ("world_size", 2), ("objective_identity_sha256", True),
    ("objective_identity_sha256", "z"*64), ("process_ranks", [0, 1, 2]),
    ("ranks", records()[:3]), ("ranks", [dict(r, loss=float("nan")) for r in records()])])
def test_v2_rejects_unobserved_or_malformed_claims(field, value):
    checkpoint = Path("/checkpoints/ddp-checkpoint-contract/checkpoint-2")
    payload = literal(checkpoint)
    payload[field] = value
    with pytest.raises(ValueError):
        decode(payload, checkpoint)


def test_v1_unchanged():
    checkpoint = Path("/checkpoints/ddp-checkpoint-contract/checkpoint-2")
    expected = literal(checkpoint)
    expected.update(schema_version=1, zero_stage=2)
    del expected["execution_backend"], expected["objective_identity_sha256"]
    assert build_stage2_producer_evidence(run_id="ddp-checkpoint-contract", method="full", zero_stage=2,
        checkpoint_path=checkpoint, artifact_kind="full", rank_records=records()) == expected
    assert decode_stage2_producer_evidence(json.dumps(expected).encode(), expected_run_id="ddp-checkpoint-contract",
        expected_method="full", expected_zero_stage=2, expected_checkpoint=checkpoint, expected_global_step=2) == expected


def witness_case(tmp_path, *, enabled=True, config_change=False, digest=True, step=2, raw_missing=False,
                 digest_mismatch=False, cross_backend=False, unexpected_reference=False):
    root = Path(tempfile.mkdtemp(prefix="native-witness-", dir=ROOT / "checkpoints"))
    checkpoint = root / "ddp-checkpoint-contract"
    (checkpoint / "checkpoint-2").mkdir(parents=True)
    (checkpoint / "checkpoint-2/rng_state_2.pth").write_bytes(b"local RNG only")
    config = Stage2ObjectiveConfig(enable_stage2_objective=enabled)
    plan = build_launch_plan(stage="stage2", tuning_method="full", zero_stage=None, backend="ddp",
        run_id=checkpoint.name, main_process_port=29876, stage2_objective_config=config)
    saved_config = config.to_dict()
    if config_change:
        saved_config["gamma_H"] = 0.5
    raw = json.dumps(dict(schema_version=1, config=saved_config,
                          reference_identity={"unexpected": True} if unexpected_reference else None,
                          vocabulary_identity=None)).encode()+b"\n"
    payload = literal(checkpoint / "checkpoint-2", hashlib.sha256(raw).hexdigest() if digest else None)
    if digest_mismatch:
        raw += b" "
    if cross_backend:
        payload.update(schema_version=1, zero_stage=2)
        del payload["execution_backend"], payload["objective_identity_sha256"]
    evidence = tmp_path / "producer.json"
    evidence.write_text(json.dumps(payload))
    return write_training_witness(plan, result_root=tmp_path, exit_codes={0: 0, 1: 0}, checkpoint_path=checkpoint,
        batch_per_gpu=1, finite_loss_evidence_path=evidence, expected_global_step=step,
        objective_identity_bytes=None if raw_missing else raw)


def test_ce_only_witness_accepts_canonical_bytes_without_local_common_files(tmp_path):
    path = witness_case(tmp_path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 2 and payload["execution_backend"] == "ddp"
    assert payload["zero_stage"] is None and payload["world_size"] == 4


@pytest.mark.parametrize("changes", [dict(config_change=True), dict(digest=False), dict(enabled=False),
    dict(step=3), dict(raw_missing=True), dict(digest_mismatch=True), dict(cross_backend=True),
    dict(unexpected_reference=True)])
def test_witness_rejects_missing_changed_disabled_or_wrong_step(tmp_path, changes):
    with pytest.raises(ValueError):
        witness_case(tmp_path, **changes)
    assert not list(tmp_path.glob("**/*train-witness.json"))


def test_actual_rank_zero_failure_is_collective_and_publishes_nothing():
    root = Path(tempfile.mkdtemp(prefix="native-publish-fail-", dir=ROOT / "checkpoints"))
    launch_workers("full", root, ROOT / "logs/stage2-ddp-dev" / f"{root.name}.log", "evidence_fail")
    results = result_directory(root)
    observed = [json.loads((results / f"publication-{rank}.json").read_text()) for rank in range(4)]
    assert observed == [{"error": "native checkpoint observation failed"}] * 4
    assert not list(results.glob("producer-*.json"))


@pytest.mark.parametrize("phase", ["evidence_ce", "evidence_runtime"])
def test_actual_prepared_native_runtime_publication_and_canonical_witness(phase):
    from src.multinode_runtime import collect_checkpoint_artifacts
    root = Path(tempfile.mkdtemp(prefix=f"native-{phase}-", dir=ROOT / "checkpoints"))
    launch_workers("full", root, ROOT / "logs/stage2-ddp-dev" / f"{root.name}.log", phase)
    runner = FilesystemTransport(root)
    collect_checkpoint_artifacts(runner, checkpoint_path=runner.path(0), runtime_identity=ddp_identity(),
                                 exit_codes={0: 0, 1: 0}, world_size=4)
    sidecar = validate_native_publication(runner, ce_only=phase == "evidence_ce")
    if phase == "evidence_ce":
        assert sidecar["config"]["enable_stage2_objective"] is True
        assert sidecar["config"]["lambda_dyn"] == sidecar["config"]["lambda_vis"] == 0.0
        assert sidecar["reference_identity"] is sidecar["vocabulary_identity"] is None
    else:
        for rank in range(4):
            assert json.loads((result_directory(root) / f"runtime-rejections-{rank}.json").read_text()) == [
                "cross_backend", "world", "rank2_nonfinite"]


def test_remote_runner_reads_fixed_cli_exact_bytes(tmp_path, monkeypatch):
    import shlex
    import sys
    from src.multinode_runtime import SubprocessCoordinatorRunner, _subprocess_command_runner
    checkpoint = tmp_path / "checkpoints/ddp-checkpoint-contract/checkpoint-2"
    checkpoint.mkdir(parents=True)
    raw = b'{"identity":"canonical"}\r\n'
    (checkpoint / "stage2_objective.json").write_bytes(raw)
    plan = build_launch_plan(stage="stage2", tuning_method="full", zero_stage=None, backend="ddp",
        run_id="ddp-checkpoint-contract", main_process_port=29876,
        stage2_objective_config=Stage2ObjectiveConfig(True))
    def command_runner(command, **kwargs):
        # Execute the production CLI at the remote subprocess boundary only.
        inner = shlex.split(command[-1])
        while "stage2-objective-identity-show" not in inner:
            inner = shlex.split(inner[-1])
        start = inner.index("stage2-objective-identity-show")
        assert inner[start:] == ["stage2-objective-identity-show", "--run-id", "ddp-checkpoint-contract",
                                 "--checkpoint-step", "2"]
        local = [sys.executable, "-c", "import sys; from pathlib import Path; "
                 "import src.multinode as m; m._REPOSITORY_ROOT=Path(sys.argv[1]); "
                 "raise SystemExit(m._main(sys.argv[2:]))", str(tmp_path), *inner[start:]]
        return _subprocess_command_runner(local, **kwargs)
    runner = SubprocessCoordinatorRunner(plan, commands=None, commit="b"*40, command_runner=command_runner)
    assert runner.read_stage2_objective_identity(2) == raw
    for bad_step in (True, 0, -1, "../../outside"):
        with pytest.raises(ValueError):
            runner.read_stage2_objective_identity(bad_step)


def test_disabled_native_witness_requires_absent_digest_and_bytes(tmp_path):
    path = witness_case(tmp_path, enabled=False, digest=False, raw_missing=True)
    assert json.loads(path.read_text())["schema_version"] == 2


@pytest.mark.parametrize("bad", [None, "symlink", "oversize", "directory-link", "run-link"])
def test_fixed_objective_reader_validates_path_and_bytes(tmp_path, monkeypatch, capsysbinary, bad):
    import src.multinode as multinode
    root = tmp_path / "checkpoints"
    checkpoint = root / "ddp-checkpoint-contract/checkpoint-2"
    checkpoint.mkdir(parents=True)
    raw = b'{"sentinel":"exact bytes"}\r\n'
    file = checkpoint / "stage2_objective.json"
    file.write_bytes(raw)
    if bad == "symlink":
        file.unlink()
        file.symlink_to(tmp_path / "outside")
    elif bad == "oversize":
        file.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    elif bad == "directory-link":
        checkpoint.rename(root / "saved")
        checkpoint.symlink_to(root / "saved", target_is_directory=True)
    elif bad == "run-link":
        checkpoint.parent.rename(root / "saved")
        checkpoint.parent.symlink_to(root / "saved", target_is_directory=True)
    monkeypatch.setattr(multinode, "_REPOSITORY_ROOT", tmp_path)
    if bad:
        with pytest.raises(SystemExit):
            multinode._main(["stage2-objective-identity-show", "--run-id", "ddp-checkpoint-contract", "--checkpoint-step", "2"])
    else:
        assert multinode._main(["stage2-objective-identity-show", "--run-id", "ddp-checkpoint-contract", "--checkpoint-step", "2"]) == 0
        assert capsysbinary.readouterr().out == raw
