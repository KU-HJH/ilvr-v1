"""Native checkpoint integration across four real CPU Gloo Trainer ranks."""

from pathlib import Path
import tempfile
import pytest

from .native_ddp_helpers import ROOT, FilesystemTransport, ddp_identity, launch_workers, run_native_checkpoint_probe
from src.multinode_runtime import collect_checkpoint_artifacts
from src.multinode_runtime import redistribute_resume_checkpoint


def test_actual_native_save_collection():
    root = Path(tempfile.mkdtemp(prefix="native-save-", dir=ROOT / "checkpoints"))
    log = ROOT / "logs/stage2-ddp-dev" / f"{root.name}.log"
    launch_workers("full", root, log, "save")
    runner = FilesystemTransport(root)
    assert sorted(p.name for p in runner.path(1).iterdir()) == ["rng_state_2.pth", "rng_state_3.pth"]
    assert {"model.safetensors", "optimizer.pt", "scheduler.pt", "training_args.bin",
            "stage2_objective.json", "stage2-sampler.json", "trainer_state.json",
            "deepspeed-runtime.json", "rng_state_0.pth", "rng_state_1.pth"} <= {p.name for p in runner.path(0).iterdir()}
    print(f"ACTUAL_SAVE_ROOT={root}")
    combined = collect_checkpoint_artifacts(runner, checkpoint_path=runner.path(0),
                runtime_identity=ddp_identity(), exit_codes={0: 0, 1: 0}, world_size=4)
    assert runner.pushed == ["rng_state_2.pth", "rng_state_3.pth"]
    assert {str(e.relative_path) for e in combined} == {p.name for p in runner.path(0).iterdir()}
    required = redistribute_resume_checkpoint(runner, checkpoint_path=runner.path(0),
        checkpoint_identity=ddp_identity(), requested_identity=ddp_identity(), destination=runner.path(0))
    pulled = list(runner.pulled)
    assert len(pulled) == len(required) - 2
    redistribute_resume_checkpoint(runner, checkpoint_path=runner.path(0),
        checkpoint_identity=ddp_identity(), requested_identity=ddp_identity(), destination=runner.path(0))
    assert runner.pulled == pulled


@pytest.mark.parametrize("method", ["full", "dora"])
def test_actual_native_save_resume_matches_uninterrupted(method):
    root = Path(tempfile.mkdtemp(prefix=f"native-{method}-", dir=ROOT / "checkpoints"))
    log = ROOT / "logs/stage2-ddp-dev" / f"{root.name}.log"
    observed = run_native_checkpoint_probe(method, output_root=root, log_path=log)
    assert observed["resumed_steps"] == [3, 3, 3, 3]
    assert observed["reference_changed"] == [False, False, False, False]
    assert observed["max_model_delta"] == [0.0, 0.0, 0.0, 0.0]
    assert observed["matches_uninterrupted"] is True


def test_actual_lora_loader_requires_matching_metadata_and_adapter_files():
    from src.multinode_artifacts import validate_loader_artifact
    root = Path(tempfile.mkdtemp(prefix="native-lora-", dir=ROOT / "checkpoints"))
    launch_workers("lora", root, ROOT / "logs/stage2-ddp-dev" / f"{root.name}.log", "save")
    checkpoint = FilesystemTransport(root).path(0)
    validate_loader_artifact(checkpoint, ddp_identity("lora"))
    with pytest.raises(ValueError, match="stage/method"):
        validate_loader_artifact(checkpoint, ddp_identity("dora"))
    for name in ("adapter_config.json", "adapter_model.safetensors", "stage2_peft_metadata.json"):
        path = checkpoint / name
        contents = path.read_bytes()
        path.unlink()
        with pytest.raises(ValueError):
            validate_loader_artifact(checkpoint, ddp_identity("lora"))
        path.write_bytes(contents)
