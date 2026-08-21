import os
from pathlib import Path
import subprocess
import sys


def test_frozen_reference_is_replicated_and_invariant_under_zero2():
    repository = Path(__file__).resolve().parents[2]
    worker = repository / "tests" / "evidence_window" / "zero2_reference_worker.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "src")
    environment["ACCELERATE_USE_DEEPSPEED"] = "true"
    environment.setdefault("OMP_NUM_THREADS", "1")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(worker),
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.count("ZERO2_REFERENCE_OK") == 2
