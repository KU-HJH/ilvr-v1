"""Coordinator terminal separation for the fixed no-optimizer diagnostic."""
import json
from pathlib import Path

import pytest

from src.multinode import build_launch_plan
import src.multinode as coordinator
import src.multinode_runtime as runtime
from scripts.probe_dil_full_model import BASE, CASES, COMPARATOR, INITIALIZER, PROFILE


def manifest_fixture(tmp_path, variant="portTorch"):
    repository = tmp_path / "repository"
    for name in ("checkpoints", "results", "data", "logs"):
        (repository / name).mkdir(parents=True)
    value = {"schema": 1, "variant": variant, "base": BASE,
             "initializer": str(repository / "checkpoints" / INITIALIZER),
             "data_path": str(repository / "data/two-cases.jsonl"), "data_sha256": "b"*64,
             "image_root": str(repository / "data"), "cases": list(CASES), "profile": PROFILE, "comparator": COMPARATOR}
    for name in ("source_files", "port_files", "base_files", "initializer_files", "package_files"):
        value[name] = {"fixture.py": "a"*64}
    path = repository / "results/manifest.json"
    path.write_text(json.dumps(value))
    return repository, path, value


def test_fixed_probe_command_uses_owned_four_rank_harness_without_training():
    builder = getattr(runtime, "build_dil_full_model_probe_commands", None)
    assert builder is not None, "fixed coordinator diagnostic command is missing"
    plan = build_launch_plan(stage="stage2", tuning_method="dora", zero_stage=2,
                             run_id="probe-contract", main_process_port=29631)
    manifest = plan.cluster.remote_repo_root / "results/probe-manifest.json"
    commands = builder(plan, commit="a"*40, manifest=manifest, manifest_sha256="b"*64)
    for rank, spec in enumerate(commands.node_specs):
        command = spec.command
        assert command[command.index("--")+1].endswith("/bin/accelerate")
        assert "--num_processes" in command and command[command.index("--num_processes")+1] == "4"
        assert command[command.index("--machine_rank")+1] == str(rank)
        assert command[command.index("--main_process_ip")+1] == "172.16.11.41"
        assert str(plan.cluster.remote_repo_root / "scripts/probe_dil_full_model.py") in command
        assert str(manifest) in command
        assert "run_stage2_training.sh" not in " ".join(command)
    assert commands.finite_loss_evidence_path is None


def test_probe_routes_before_ordinary_training_configuration(monkeypatch):
    dispatched = []
    monkeypatch.setattr(coordinator, "_dil_full_model_probe_main", lambda argv: dispatched.append(argv) or 0, raising=False)
    assert coordinator._training_main("stage2", ["--dil-full-model-probe", "/manifest"]) == 0
    assert dispatched == [["--dil-full-model-probe", "/manifest"]]


@pytest.mark.parametrize("option", ["--max_steps", "--save_steps", "--resume_from_checkpoint_path", "--entrypoint"])
def test_probe_rejects_training_resume_or_arbitrary_entrypoint(option):
    handler = getattr(coordinator, "_dil_full_model_probe_main", None)
    assert handler is not None, "fixed diagnostic coordinator is missing"
    with pytest.raises(SystemExit):
        handler(["--dil-full-model-probe", "/manifest", "--run-id", "probe-contract",
                 "--zero-stage", "2", "--main-process-port", "29631", option, "1"])


def test_probe_dry_run_exposes_actual_identity_checks_and_topology(tmp_path, monkeypatch, capsys):
    repository, manifest, _ = manifest_fixture(tmp_path)
    monkeypatch.setattr(coordinator, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(coordinator, "_remote_host_repository_root", lambda path: path)
    monkeypatch.setattr(coordinator, "_repository_git_stdout", lambda args: "a"*40)
    # The canonical config belongs to the actual checkout; no fixture weights.
    monkeypatch.setattr(coordinator, "selected_deepspeed_config", lambda *args: Path(__file__).resolve().parents[2]/"configs/deepspeed_zero2.json")
    assert coordinator._training_main("stage2", ["--dil-full-model-probe", str(manifest),
        "--run-id", "probe-contract", "--zero-stage", "2", "--main-process-port", "29631", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["kind"] == "dil-full-model-probe" and result["plan"]["world_size"] == 4
    assert result["variant"] == "portTorch"
    assert result["identity_preflight"][1][-2:] == ["--machine-rank", "1"]
    assert result["comparator"] == {"rtol": 1e-5, "atol": 1e-7}


def test_probe_completion_collects_four_records_without_training_witness(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    repository, manifest, _ = manifest_fixture(tmp_path)
    import hashlib
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    from tests.stage2.test_dil_full_model_probe import full_rank_record
    records = [full_rank_record(rank, digest) for rank in range(4)]
    (repository / "results/probe-contract").mkdir()
    monkeypatch.setattr(coordinator, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(coordinator, "_remote_host_repository_root", lambda path: path)
    monkeypatch.setattr(coordinator, "_repository_git_stdout", lambda args: "a"*40)
    monkeypatch.setattr(coordinator, "selected_deepspeed_config", lambda *args: Path(__file__).resolve().parents[2]/"configs/deepspeed_zero2.json")
    monkeypatch.setattr(coordinator, "collect_node_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(coordinator, "validate_preflight", lambda *args: SimpleNamespace(commit="a"*40))
    def identity_preflight(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps({"manifest_sha256": digest, "variant": "portTorch"}), stderr="")
    monkeypatch.setattr(coordinator.subprocess, "run", identity_preflight)
    reads = []
    class Runner:
        def __init__(self, *args, **kwargs):
            pass
        def read_dil_full_model_probe_record(self, rank):
            reads.append(rank)
            return json.dumps(records[rank]).encode()
    monkeypatch.setattr(coordinator, "SubprocessCoordinatorRunner", Runner)
    monkeypatch.setattr(coordinator, "start_rank0_then_rank1", lambda *args, **kwargs: None)
    monkeypatch.setattr(coordinator, "monitor_owned_jobs", lambda *args, **kwargs: {0: 0, 1: 0})
    def forbidden(*args, **kwargs):
        pytest.fail("diagnostic entered ordinary checkpoint/training terminal")
    monkeypatch.setattr(coordinator, "collect_checkpoint_artifacts", forbidden)
    monkeypatch.setattr(coordinator, "write_training_witness", forbidden)
    assert coordinator._training_main("stage2", ["--dil-full-model-probe", str(manifest),
        "--run-id", "probe-contract", "--zero-stage", "2", "--main-process-port", "29631"]) == 0
    assert reads == [0, 1, 2, 3]
    witness = json.loads((repository / "results/probe-contract/initial-probe-witness.json").read_text())
    assert witness["kind"] == "dil-full-model-probe" and len(witness["ranks"]) == 4
    assert not list((repository / "checkpoints").rglob("*"))
