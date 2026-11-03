"""Synchronized training-iteration benchmarks for packed Stage 2B."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch

from stage2.trainer import Stage2Trainer
from step_preference.config import StepPreferenceConfig


@dataclass(frozen=True)
class PathBenchmark:
    raw_times_ms: tuple[float, ...]
    median_ms: float
    iterations_per_second: float
    baseline_allocated_bytes: int
    peak_allocated_bytes: int
    peak_increment_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_times_ms": list(self.raw_times_ms),
            "median_ms": self.median_ms,
            "iterations_per_second": self.iterations_per_second,
            "baseline_allocated_bytes": self.baseline_allocated_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_increment_bytes": self.peak_increment_bytes,
        }


@dataclass(frozen=True)
class BenchmarkGate:
    passed: bool
    time_reduction: float
    memory_reduction: float
    time_threshold: float = 0.05
    memory_threshold: float = 0.01


class CudaBenchmarkRuntime:
    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    def memory_allocated(self):
        return torch.cuda.memory_allocated(self.device)

    def reset_peak_memory_stats(self):
        torch.cuda.reset_peak_memory_stats(self.device)

    def max_memory_allocated(self):
        return torch.cuda.max_memory_allocated(self.device)

    def new_event(self):
        return torch.cuda.Event(enable_timing=True)

    @staticmethod
    def elapsed_ms(start, end):
        return start.elapsed_time(end)


class Stage2BTrainingStep:
    """One complete trainer-routed Stage 2B forward and backward."""

    def __init__(self, model, batch, config: StepPreferenceConfig):
        self.model = model
        self.batch = batch
        self.autocast_dtype = torch.bfloat16
        self.trainer = object.__new__(Stage2Trainer)
        self.trainer.stage2_diagnostics_interval = 0
        self.trainer._stage2_microbatch_index = 0
        self.trainer.step_preference_config = config
        self.trainer._step_preference_metric_totals = {}
        self.trainer._step_preference_metric_batches = 0

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def __call__(self):
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            loss = self.trainer.compute_loss(self.model, self.batch)
        loss.backward()


def _capture_rng():
    return (
        torch.random.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def _restore_rng(state) -> None:
    torch.random.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def benchmark_training_path(
    step_fn,
    runtime,
    *,
    warmups: int = 3,
    measurements: int = 10,
) -> PathBenchmark:
    """Benchmark one path with compilation warmups and synchronized CUDA events."""
    if warmups < 0 or measurements <= 0:
        raise ValueError("benchmark counts must be nonnegative warmups and positive measurements")
    if not callable(step_fn) or not callable(getattr(step_fn, "zero_grad", None)):
        raise TypeError("step_fn must be callable and expose zero_grad")
    rng_state = _capture_rng()
    for _ in range(warmups):
        step_fn.zero_grad()
        _restore_rng(rng_state)
        step_fn()

    runtime.synchronize()
    baseline = int(runtime.memory_allocated())
    runtime.reset_peak_memory_stats()
    raw_times = []
    for _ in range(measurements):
        step_fn.zero_grad()
        _restore_rng(rng_state)
        runtime.synchronize()
        start = runtime.new_event()
        end = runtime.new_event()
        start.record()
        step_fn()
        end.record()
        runtime.synchronize()
        raw_times.append(float(runtime.elapsed_ms(start, end)))
    peak = int(runtime.max_memory_allocated())
    median = float(statistics.median(raw_times))
    if median <= 0 or baseline < 0 or peak < 0:
        raise ValueError("benchmark returned invalid timing or memory values")
    return PathBenchmark(
        raw_times_ms=tuple(raw_times),
        median_ms=median,
        iterations_per_second=1000.0 / median,
        baseline_allocated_bytes=baseline,
        peak_allocated_bytes=peak,
        peak_increment_bytes=max(0, peak - baseline),
    )


def evaluate_benchmark_gate(
    *,
    naive_ms: float,
    packed_ms: float,
    naive_bytes: int,
    packed_bytes: int,
) -> BenchmarkGate:
    values = (naive_ms, packed_ms, naive_bytes, packed_bytes)
    if any(value <= 0 for value in values):
        raise ValueError("benchmark measurements must be positive")
    time_reduction = 1.0 - packed_ms / naive_ms
    memory_reduction = 1.0 - packed_bytes / naive_bytes
    return BenchmarkGate(
        passed=time_reduction >= 0.05 or memory_reduction >= 0.01,
        time_reduction=time_reduction,
        memory_reduction=memory_reduction,
    )
