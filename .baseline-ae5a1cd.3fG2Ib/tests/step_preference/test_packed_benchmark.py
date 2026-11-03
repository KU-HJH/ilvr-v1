from pathlib import Path
import subprocess
import sys

import pytest
import torch

from step_preference.benchmark import (
    Stage2BTrainingStep,
    benchmark_training_path,
    evaluate_benchmark_gate,
)


class _Event:
    def __init__(self, runtime, number):
        self.runtime = runtime
        self.number = number

    def record(self):
        self.runtime.calls.append(("record", self.number))


class _Runtime:
    def __init__(self):
        self.calls = []
        self.events = 0
        self.elapsed = iter(range(10, 20))

    def synchronize(self):
        self.calls.append("sync")

    def memory_allocated(self):
        self.calls.append("allocated")
        return 1_000

    def reset_peak_memory_stats(self):
        self.calls.append("reset_peak")

    def max_memory_allocated(self):
        self.calls.append("max_allocated")
        return 1_250

    def new_event(self):
        event = _Event(self, self.events)
        self.events += 1
        return event

    def elapsed_ms(self, start, end):
        self.calls.append(("elapsed", start.number, end.number))
        return float(next(self.elapsed))


class _Step:
    def __init__(self, calls):
        self.calls = calls

    def zero_grad(self):
        self.calls.append("zero_grad")

    def __call__(self):
        self.calls.append("step")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 benchmark requires CUDA")
def test_training_step_uses_bfloat16_cuda_autocast(monkeypatch):
    """Catches the handcrafted benchmark silently measuring FP32 execution."""
    observed = []

    def fake_compute_loss(_trainer, _model, _batch):
        observed.append(
            (
                torch.is_autocast_enabled("cuda"),
                torch.get_autocast_dtype("cuda"),
            )
        )
        return torch.ones((), device="cuda", requires_grad=True)

    monkeypatch.setattr(
        "step_preference.benchmark.Stage2Trainer.compute_loss", fake_compute_loss
    )

    class Model:
        def zero_grad(self, *, set_to_none):
            assert set_to_none is True

    step = Stage2BTrainingStep(Model(), {}, object())
    step()

    assert observed == [(True, torch.bfloat16)]
    assert step.autocast_dtype == torch.bfloat16


def test_benchmark_protocol_warms_up_then_times_exactly_ten_iterations():
    """Catches compilation, missing synchronization, or stale gradients entering timing."""
    runtime = _Runtime()
    step = _Step(runtime.calls)

    result = benchmark_training_path(step, runtime, warmups=3, measurements=10)

    assert runtime.calls[:8] == [
        "zero_grad", "step",
        "zero_grad", "step",
        "zero_grad", "step",
        "sync", "allocated",
    ]
    assert runtime.calls[8] == "reset_peak"
    assert runtime.calls.count("zero_grad") == 13
    assert runtime.calls.count("step") == 13
    assert runtime.calls.count("sync") == 21
    assert runtime.calls[-1] == "max_allocated"
    assert result.raw_times_ms == tuple(float(value) for value in range(10, 20))
    assert result.median_ms == 14.5
    assert result.iterations_per_second == pytest.approx(1000 / 14.5)
    assert result.baseline_allocated_bytes == 1_000
    assert result.peak_allocated_bytes == 1_250
    assert result.peak_increment_bytes == 250


def test_gate_accepts_time_or_memory_improvement():
    """Catches AND-combining or weakening the declared disjunctive performance gate."""
    assert evaluate_benchmark_gate(
        naive_ms=100, packed_ms=94.9, naive_bytes=1000, packed_bytes=1000
    ).passed
    assert evaluate_benchmark_gate(
        naive_ms=100, packed_ms=100, naive_bytes=1000, packed_bytes=990
    ).passed
    assert not evaluate_benchmark_gate(
        naive_ms=100, packed_ms=96, naive_bytes=1000, packed_bytes=995
    ).passed


def test_benchmark_cli_refuses_existing_output_before_loading(tmp_path):
    """Catches a benchmark retry overwriting an earlier performance witness."""
    output = tmp_path / "existing.json"
    output.write_text("preserve me\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts/benchmark_packed_step_preference.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint_path", "missing",
            "--data_path", "missing",
            "--image_root", "missing",
            "--negative_path", "missing",
            "--negative_manifest", "missing",
            "--device", "cuda:0",
            "--output_path", str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "already exists" in completed.stderr
    assert output.read_text(encoding="utf-8") == "preserve me\n"


@pytest.mark.parametrize(
    "values",
    [
        (0, 1, 1, 1),
        (1, 0, 1, 1),
        (1, 1, 0, 1),
        (1, 1, 1, 0),
    ],
)
def test_gate_rejects_nonpositive_measurements(values):
    """Catches division by zero or meaningless measurements being reported as gains."""
    with pytest.raises(ValueError, match="positive"):
        evaluate_benchmark_gate(
            naive_ms=values[0],
            packed_ms=values[1],
            naive_bytes=values[2],
            packed_bytes=values[3],
        )
