"""Narrow actual Trainer/DeepSpeed smoke, enabled explicitly on two free GPUs."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from tiny_stage2_train import build_tiny_stage2_parser


def test_harness_routes_shared_smoke_without_save_resume_lifecycle():
    args = build_tiny_stage2_parser().parse_args([
        "--experiment-id", "shared-prefix-test", "--shared-prefix",
        "--tuning-method", "dora", "--zero-stage", "2"])
    assert args.shared_prefix


@pytest.mark.skipif(os.environ.get("RUN_SHARED_PREFIX_DDP") != "1", reason="explicit two-GPU smoke")
def test_two_rank_shared_flex_dora_zero2_accumulation():
    root = Path(__file__).resolve().parents[3]
    run_id = os.environ["SHARED_PREFIX_RUN_ID"]
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                               check=True, text=True, capture_output=True).stdout.strip()
    assert not processes, f"GPUs have existing processes: {processes}"
    from artifact_paths import require_canonical_artifact_path
    log = require_canonical_artifact_path(root / "logs" / run_id / "ddp.log", kind="log")
    log.parent.mkdir(parents=True, exist_ok=False)
    with log.open("x") as handle:
        completed = subprocess.run([str(root / "venv/ilvr-venv/bin/torchrun"),
            "--standalone", "--nproc_per_node=2", str(root / "tests/stage2/integration/tiny_stage2_train.py"),
            "--experiment-id", run_id, "--shared-prefix", "--tuning-method", "dora", "--zero-stage", "2",
            "--deepspeed", str(root / "configs/deepspeed_zero2.json")],
            cwd=root, stdout=handle, stderr=subprocess.STDOUT, timeout=600)
    assert completed.returncode == 0, log.read_text()[-16000:]
    witness = json.loads((root / "results" / run_id / "ddp-witness.json").read_text())
    assert witness["world_size"] == 2
    for rank in witness["ranks"]:
        assert rank["global_step"] == rank["optimizer_step"] == 1
        assert rank["microbatches"] == 2
        assert rank["finite_loss"] and rank["finite_gradients"]
        assert rank["changed_trainable_tensors"] > 0
        assert rank["rollout_calls"] == 2
        assert rank["pre_engine_parameter_dtypes"] == ["torch.float32"]
        assert rank["trained_parameter_dtypes"] == ["torch.bfloat16"]
    assert witness["adapter"] == dict(method="dora", rank=32, alpha=64, dropout=0.05)
    assert witness["zero_stage"] == 2
    assert witness["gradient_accumulation_steps"] == 2
