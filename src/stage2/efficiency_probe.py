"""Focused Stage 2 no-cache runtime evidence and segmented-KV authorization gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import statistics
import time
from typing import Mapping

import torch
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode
from torch.utils.data import DataLoader

from artifact_paths import resolve_run_artifact_paths


EFFICIENCY_SCHEMA_VERSION = 1
SEGMENTED_KV_GATE = 0.30
WARMUP_COUNT = 3
MEASURED_COUNT = 10


def _finite(name: str, value: float, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0 or (not allow_zero and value == 0.0):
        qualifier = "positive" if not allow_zero else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class Stage2RepresentativeExample:
    quantile: str
    raw_index: int
    length: int
    latent_block_count: int
    target_length: float
    target_block_count: float


@dataclass(frozen=True)
class Stage2EfficiencySample:
    step_ms: float
    prefix_decoder_ms: float
    peak_bytes: int
    collation_ms: float = 0.0
    final_decoder_ms: float = 0.0
    visual_ms: float = 0.0
    lm_head_ms: float = 0.0
    shifted_ce_loss_ms: float = 0.0
    forward_loss_ms: float = 0.0
    backward_ms: float = 0.0
    optimizer_ms: float = 0.0
    step_overhead_ms: float = 0.0
    forward_other_ms: float = 0.0
    loss: float = 0.0
    gradient_norm: float = 0.0
    prefix_input_length_sum: int = 0
    decoder_input_lengths: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.decoder_input_lengths, tuple):
            raise ValueError("decoder_input_lengths must be a tuple")


@dataclass(frozen=True)
class Stage2EfficiencyEvidence:
    sample_count: int
    step_ms: float
    prefix_decoder_ms: float
    prefix_decoder_share: float
    peak_bytes: int
    collation_ms: float
    final_decoder_ms: float
    visual_ms: float
    lm_head_ms: float
    shifted_ce_loss_ms: float
    forward_loss_ms: float
    backward_ms: float
    optimizer_ms: float
    step_overhead_ms: float
    forward_other_ms: float
    loss: float
    gradient_norm: float
    prefix_input_length_sum: int
    prefix_threshold_met: bool
    segmented_kv_authorized: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count <= 0
        ):
            raise ValueError("sample_count must be positive")
        step_ms = _finite("step_ms", self.step_ms, allow_zero=False)
        prefix_ms = _finite("prefix_decoder_ms", self.prefix_decoder_ms)
        for name in (
            "prefix_decoder_share",
            "collation_ms",
            "final_decoder_ms",
            "visual_ms",
            "lm_head_ms",
            "shifted_ce_loss_ms",
            "forward_loss_ms",
            "backward_ms",
            "optimizer_ms",
            "step_overhead_ms",
            "forward_other_ms",
            "loss",
            "gradient_norm",
        ):
            _finite(name, getattr(self, name))
        _non_negative_int("peak_bytes", self.peak_bytes)
        _non_negative_int("prefix_input_length_sum", self.prefix_input_length_sum)
        if not math.isclose(
            self.prefix_decoder_share,
            prefix_ms / step_ms,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError("prefix_decoder_share does not match component medians")
        expected_threshold = self.prefix_decoder_share >= SEGMENTED_KV_GATE
        if not isinstance(self.prefix_threshold_met, bool):
            raise ValueError("prefix_threshold_met must be a boolean")
        if self.prefix_threshold_met != expected_threshold:
            raise ValueError(
                "prefix threshold classification does not match the fixed gate"
            )
        if not isinstance(self.segmented_kv_authorized, bool):
            raise ValueError("segmented_kv_authorized must be a boolean")
        if self.segmented_kv_authorized and not self.prefix_threshold_met:
            raise ValueError("segmented KV authorization requires the prefix threshold")


@dataclass(frozen=True)
class Stage2WorkerEvidence:
    worker_count: int
    measured_batches: int
    median_collation_ms: float
    peak_host_rss_bytes: int
    within_host_capacity: bool
    worker_pids: tuple[int, ...] = ()


def _linear_quantile(values: tuple[int, ...], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile for empty values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def select_representative_examples(manifest) -> tuple[Stage2RepresentativeExample, ...]:
    """Select deterministic retained rows closest to joint p50 and p95 targets."""
    lengths = tuple(manifest.lengths)
    block_counts = tuple(manifest.latent_block_counts)
    retained = tuple(manifest.retained_indices)
    if len(lengths) != len(block_counts):
        raise ValueError("manifest length/block cardinality mismatch")
    if not retained:
        raise ValueError("manifest has no retained example")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= len(lengths)
        for index in retained
    ):
        raise ValueError("manifest retained index is invalid")
    retained_lengths = tuple(lengths[index] for index in retained)
    retained_blocks = tuple(block_counts[index] for index in retained)
    selected = []
    for label, probability in (("p50", 0.50), ("p95", 0.95)):
        target_length = _linear_quantile(retained_lengths, probability)
        target_blocks = _linear_quantile(retained_blocks, probability)
        raw_index = min(
            retained,
            key=lambda index: (
                (lengths[index] - target_length) ** 2
                + (block_counts[index] - target_blocks) ** 2,
                index,
            ),
        )
        selected.append(
            Stage2RepresentativeExample(
                quantile=label,
                raw_index=raw_index,
                length=lengths[raw_index],
                latent_block_count=block_counts[raw_index],
                target_length=target_length,
                target_block_count=target_blocks,
            )
        )
    return tuple(selected)


def _validate_sample(sample: Stage2EfficiencySample) -> None:
    if not isinstance(sample, Stage2EfficiencySample):
        raise TypeError("samples must contain Stage2EfficiencySample values")
    _finite("step_ms", sample.step_ms, allow_zero=False)
    for name in (
        "prefix_decoder_ms",
        "collation_ms",
        "final_decoder_ms",
        "visual_ms",
        "lm_head_ms",
        "shifted_ce_loss_ms",
        "forward_loss_ms",
        "backward_ms",
        "optimizer_ms",
        "step_overhead_ms",
        "forward_other_ms",
        "loss",
        "gradient_norm",
    ):
        _finite(name, getattr(sample, name))
    _non_negative_int("peak_bytes", sample.peak_bytes)
    _non_negative_int("prefix_input_length_sum", sample.prefix_input_length_sum)
    for input_length in sample.decoder_input_lengths:
        _non_negative_int("decoder input length", input_length)


def summarize_samples(
    samples: list[Stage2EfficiencySample] | tuple[Stage2EfficiencySample, ...],
) -> Stage2EfficiencyEvidence:
    """Validate measured samples and aggregate every scalar with its median."""
    samples = tuple(samples)
    if not samples:
        raise ValueError("cannot summarize zero samples")
    for sample in samples:
        _validate_sample(sample)

    def median(name: str) -> float:
        return float(statistics.median(getattr(sample, name) for sample in samples))

    step_ms = median("step_ms")
    prefix_decoder_ms = median("prefix_decoder_ms")
    prefix_share = prefix_decoder_ms / step_ms
    return Stage2EfficiencyEvidence(
        sample_count=len(samples),
        step_ms=step_ms,
        prefix_decoder_ms=prefix_decoder_ms,
        prefix_decoder_share=prefix_share,
        peak_bytes=int(statistics.median(sample.peak_bytes for sample in samples)),
        collation_ms=median("collation_ms"),
        final_decoder_ms=median("final_decoder_ms"),
        visual_ms=median("visual_ms"),
        lm_head_ms=median("lm_head_ms"),
        shifted_ce_loss_ms=median("shifted_ce_loss_ms"),
        forward_loss_ms=median("forward_loss_ms"),
        backward_ms=median("backward_ms"),
        optimizer_ms=median("optimizer_ms"),
        step_overhead_ms=median("step_overhead_ms"),
        forward_other_ms=median("forward_other_ms"),
        loss=median("loss"),
        gradient_norm=median("gradient_norm"),
        prefix_input_length_sum=int(
            statistics.median(sample.prefix_input_length_sum for sample in samples)
        ),
        prefix_threshold_met=prefix_share >= SEGMENTED_KV_GATE,
        segmented_kv_authorized=False,
    )


def _representative_b200_metadata_is_valid(metadata: Mapping[str, object]) -> bool:
    return (
        metadata.get("probe_kind") == "stage2_no_cache"
        and "B200" in str(metadata.get("device_name", "")).upper()
        and metadata.get("manifest_validated") is True
        and metadata.get("representative_quantiles") == ["p50", "p95"]
        and metadata.get("warmup_count_per_representative") == WARMUP_COUNT
        and metadata.get("measured_count_per_representative") == MEASURED_COUNT
        and metadata.get("profile_uses_fixed_worker_count") is True
    )


def authorize_segmented_kv(
    evidence: Stage2EfficiencyEvidence, metadata: Mapping[str, object]
) -> Stage2EfficiencyEvidence:
    """Apply the project gate only to validated representative B200 evidence."""
    if not isinstance(evidence, Stage2EfficiencyEvidence):
        raise TypeError("evidence must be Stage2EfficiencyEvidence")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    return replace(
        evidence,
        segmented_kv_authorized=(
            evidence.prefix_threshold_met
            and _representative_b200_metadata_is_valid(metadata)
        ),
    )


def _process_rss_bytes(pid: int) -> int:
    try:
        resident_pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
    except (FileNotFoundError, IndexError, PermissionError, ValueError):
        return 0
    return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))


def _aggregate_process_rss_bytes(pids: tuple[int, ...]) -> int:
    return sum(_process_rss_bytes(pid) for pid in pids)


def _available_host_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("host memory capacity is unavailable")


def benchmark_worker_counts(
    *,
    dataset,
    collate_fn,
    worker_counts: tuple[int, ...] = (0, 2, 4),
    warmup_batches: int = 2,
    measured_batches: int = 20,
    loader_factory=DataLoader,
) -> tuple[Stage2WorkerEvidence, ...]:
    """Time identical sequential Stage 2 batches for each requested worker count."""
    if not dataset:
        raise ValueError("worker benchmark requires a retained example")
    if warmup_batches < 0 or measured_batches <= 0:
        raise ValueError("worker benchmark counts are invalid")
    retained_batch_count = min(len(dataset), measured_batches)
    evidence = []
    for worker_count in worker_counts:
        _non_negative_int("worker_count", worker_count)
        kwargs = {"num_workers": worker_count}
        if worker_count > 0:
            kwargs["prefetch_factor"] = 2
            kwargs["persistent_workers"] = True
        retained_indices = [
            index % len(dataset) for index in range(warmup_batches)
        ] + list(range(retained_batch_count))
        capacity_bytes = _available_host_memory_bytes() + _process_rss_bytes(
            os.getpid()
        )
        loader = loader_factory(
            dataset,
            batch_size=1,
            sampler=retained_indices,
            collate_fn=collate_fn,
            **kwargs,
        )
        durations = []
        iterator = iter(loader)
        worker_pids = tuple(
            worker.pid for worker in getattr(iterator, "_workers", ())
        )
        measured_pids = (os.getpid(), *worker_pids)
        peak_rss = _aggregate_process_rss_bytes(measured_pids)
        try:
            for _ in range(warmup_batches):
                next(iterator)
                peak_rss = max(
                    peak_rss, _aggregate_process_rss_bytes(measured_pids)
                )
            for _ in range(retained_batch_count):
                started = time.perf_counter_ns()
                next(iterator)
                durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
                peak_rss = max(
                    peak_rss, _aggregate_process_rss_bytes(measured_pids)
                )
        finally:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
        evidence.append(
            Stage2WorkerEvidence(
                worker_count=worker_count,
                measured_batches=retained_batch_count,
                median_collation_ms=float(statistics.median(durations)),
                peak_host_rss_bytes=peak_rss,
                within_host_capacity=peak_rss <= capacity_bytes,
                worker_pids=worker_pids,
            )
        )
    return tuple(evidence)


def select_fastest_worker_count(evidence) -> int:
    """Choose the fastest capacity-safe worker count with a stable low-count tie."""
    eligible = tuple(item for item in evidence if item.within_host_capacity)
    if not eligible:
        raise RuntimeError("no worker count stayed within host-memory capacity")
    return min(
        eligible,
        key=lambda item: (item.median_collation_ms, item.worker_count),
    ).worker_count


def build_representative_dataloader(
    *, example, collate_fn, worker_count: int, batch_count: int
) -> DataLoader:
    """Build the fixed-worker loader used by one representative GPU profile."""
    _non_negative_int("worker_count", worker_count)
    if (
        isinstance(batch_count, bool)
        or not isinstance(batch_count, int)
        or batch_count < 1
    ):
        raise ValueError("batch_count must be a positive integer")
    kwargs = {"num_workers": worker_count}
    if worker_count > 0:
        kwargs.update(prefetch_factor=2, persistent_workers=True)
    return DataLoader(
        [example] * batch_count,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        **kwargs,
    )


def _module_input_length(args, kwargs) -> int | None:
    value = kwargs.get("inputs_embeds")
    if value is None and args:
        value = args[0]
    if isinstance(value, torch.Tensor) and value.ndim >= 2:
        return int(value.shape[-2])
    return None


def _gradient_norm_tensor(optimizer) -> torch.Tensor:
    gradients = [
        parameter.grad.detach().float()
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.grad is not None
    ]
    if not gradients:
        raise ValueError("training step produced no gradients")
    return torch.sqrt(sum(gradient.square().sum() for gradient in gradients))


class _CrossEntropyCudaTimer(TorchFunctionMode):
    """Record exact CUDA regions spent inside shifted cross entropy."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is not F.cross_entropy:
            return func(*args, **(kwargs or {}))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = func(*args, **(kwargs or {}))
        end.record()
        self.calls.append((start, end))
        return result


def measure_cuda_training_step(
    *,
    decoder,
    visual,
    lm_head,
    collate_fn,
    loss_fn,
    optimizer,
    scheduler=None,
) -> Stage2EfficiencySample:
    """Measure one real optimizer step and classify decoder calls after one sync."""
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 efficiency measurement requires CUDA")
    device = next(
        (
            parameter.device
            for module in (decoder, visual, lm_head)
            for parameter in module.parameters()
        ),
        torch.device("cuda"),
    )
    if device.type != "cuda":
        raise ValueError("measured Stage 2 modules must be on CUDA")
    calls: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        "decoder": [],
        "visual": [],
        "lm_head": [],
    }
    pending: dict[str, list[torch.cuda.Event]] = {
        "decoder": [],
        "visual": [],
        "lm_head": [],
    }
    decoder_input_lengths: list[int] = []
    handles = []

    def pre_hook(label):
        def record(_module, args, kwargs):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            pending[label].append(event)
            if label == "decoder":
                input_length = _module_input_length(args, kwargs)
                if input_length is None:
                    raise ValueError("decoder hook could not resolve its input length")
                decoder_input_lengths.append(input_length)

        return record

    def post_hook(label):
        def record(_module, _args, _kwargs, _output):
            if not pending[label]:
                raise RuntimeError(f"unpaired {label} timing hook")
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            calls[label].append((pending[label].pop(), end))

        return record

    try:
        for label, module in (
            ("decoder", decoder),
            ("visual", visual),
            ("lm_head", lm_head),
        ):
            handles.append(
                module.register_forward_pre_hook(pre_hook(label), with_kwargs=True)
            )
            handles.append(
                module.register_forward_hook(post_hook(label), with_kwargs=True)
            )

        torch.cuda.reset_peak_memory_stats(device)
        step_started_ns = time.perf_counter_ns()
        collate_started = time.perf_counter_ns()
        batch = collate_fn()
        collation_ms = (time.perf_counter_ns() - collate_started) / 1_000_000.0
        optimizer.zero_grad(set_to_none=True)
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        optimizer_start = torch.cuda.Event(enable_timing=True)
        optimizer_end = torch.cuda.Event(enable_timing=True)
        cross_entropy_timer = _CrossEntropyCudaTimer()
        forward_start.record()
        with cross_entropy_timer:
            loss = loss_fn(batch)
        forward_end.record()
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ValueError("trainer loss path must return one scalar tensor")
        backward_start.record()
        loss.backward()
        backward_end.record()
        optimizer_start.record()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer_end.record()
        torch.cuda.synchronize(device)
        step_ms = (time.perf_counter_ns() - step_started_ns) / 1_000_000.0
        peak_bytes = int(torch.cuda.max_memory_allocated(device))

        loss_value = float(loss.detach().item())
        gradient_norm = _gradient_norm_tensor(optimizer)
        gradient_norm_value = float(gradient_norm.item())
        _finite("loss", loss_value)
        _finite("gradient_norm", gradient_norm_value)
        if any(
            not torch.isfinite(gradient).all().item()
            for group in optimizer.param_groups
            for parameter in group["params"]
            if (gradient := parameter.grad) is not None
        ):
            raise ValueError("training step produced a non-finite gradient")
        if pending["decoder"] or pending["visual"] or pending["lm_head"]:
            raise RuntimeError("timing hooks did not finish")
        decoder_ms = tuple(start.elapsed_time(end) for start, end in calls["decoder"])
        if not decoder_ms:
            raise ValueError("training step did not call the decoder")
        component_ms = {
            label: sum(start.elapsed_time(end) for start, end in pairs)
            for label, pairs in calls.items()
            if label != "decoder"
        }
        shifted_ce_loss_ms = sum(
            start.elapsed_time(end) for start, end in cross_entropy_timer.calls
        )
        if not cross_entropy_timer.calls:
            raise ValueError("training loss path did not call shifted cross entropy")
        forward_loss_ms = forward_start.elapsed_time(forward_end)
        backward_ms = backward_start.elapsed_time(backward_end)
        optimizer_ms = optimizer_start.elapsed_time(optimizer_end)
        forward_components_ms = (
            sum(decoder_ms)
            + component_ms["visual"]
            + component_ms["lm_head"]
            + shifted_ce_loss_ms
        )
        forward_other_ms = max(0.0, forward_loss_ms - forward_components_ms)
        step_overhead_ms = max(
            0.0,
            step_ms
            - collation_ms
            - forward_loss_ms
            - backward_ms
            - optimizer_ms,
        )
        result = Stage2EfficiencySample(
            step_ms=step_ms,
            prefix_decoder_ms=sum(decoder_ms[:-1]),
            final_decoder_ms=decoder_ms[-1],
            visual_ms=component_ms["visual"],
            lm_head_ms=component_ms["lm_head"],
            shifted_ce_loss_ms=shifted_ce_loss_ms,
            forward_loss_ms=forward_loss_ms,
            backward_ms=backward_ms,
            optimizer_ms=optimizer_ms,
            step_overhead_ms=step_overhead_ms,
            forward_other_ms=forward_other_ms,
            collation_ms=collation_ms,
            peak_bytes=peak_bytes,
            loss=loss_value,
            gradient_norm=gradient_norm_value,
            decoder_input_lengths=tuple(decoder_input_lengths),
            prefix_input_length_sum=sum(decoder_input_lengths[:-1]),
        )
        _validate_sample(result)
        return result
    finally:
        for handle in reversed(handles):
            handle.remove()


class _CudaSelfTestModel(torch.nn.Module):
    """Small real-CUDA workload for validating the measurement machinery."""

    def __init__(self) -> None:
        super().__init__()
        self.visual = torch.nn.Linear(16, 16, bias=False)
        self.model = torch.nn.Linear(16, 16, bias=False)
        self.lm_head = torch.nn.Linear(16, 8, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.visual(values)
        self.model(encoded[:, :1])
        self.model(encoded[:, :2])
        final = self.model(encoded)
        logits = self.lm_head(final).float()
        labels = torch.zeros(
            logits.shape[:-1], dtype=torch.long, device=logits.device
        )
        return F.cross_entropy(logits.flatten(0, -2), labels.flatten())


def _artifact_paths(
    *, result_root: Path, log_root: Path, experiment_id: str
) -> tuple[Path, Path]:
    _validate_experiment_id(experiment_id)
    result_path = result_root / experiment_id / "stage2-efficiency.json"
    log_path = log_root / experiment_id / "stage2-efficiency.log"
    for path in (result_path, log_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"immutable probe artifact already exists: {path}")
    return result_path, log_path


def _open_probe_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"stage2.efficiency_probe.{path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, mode="x", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers[:] = [handler]
    return logger


def _close_probe_logger(logger: logging.Logger) -> None:
    for handler in tuple(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _device_metadata() -> dict[str, object]:
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "device_index": device,
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _log_evidence(logger: logging.Logger, label: str, evidence) -> None:
    logger.info("%s %s", label, _canonical_json(asdict(evidence)))


def run_cuda_self_test(
    *,
    experiment_id: str,
    result_root: Path,
    log_root: Path,
) -> Path:
    """Run the fixed 3+10 CUDA protocol against a tiny measured optimizer path."""
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 efficiency self-test requires CUDA")
    result_path, log_path = _artifact_paths(
        result_root=result_root,
        log_root=log_root,
        experiment_id=experiment_id,
    )
    logger = _open_probe_logger(log_path)
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        model = _CudaSelfTestModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def measure() -> Stage2EfficiencySample:
            return measure_cuda_training_step(
                decoder=model.model,
                visual=model.visual,
                lm_head=model.lm_head,
                collate_fn=lambda: torch.ones(1, 3, 16, device=device),
                loss_fn=model,
                optimizer=optimizer,
            )

        for _ in range(WARMUP_COUNT):
            measure()
        samples = tuple(measure() for _ in range(MEASURED_COUNT))
        evidence = summarize_samples(samples)
        metadata = {
            **_device_metadata(),
            "probe_kind": "tiny_cuda_self_test",
            "model_identity": "stage2.efficiency_probe._CudaSelfTestModel",
            "warmup_count": WARMUP_COUNT,
            "measured_count": MEASURED_COUNT,
            "samples": [asdict(sample) for sample in samples],
        }
        for index, sample in enumerate(samples):
            _log_evidence(logger, f"measured_sample={index}", sample)
        _log_evidence(logger, "summary", evidence)
        written = write_efficiency_result(
            result_root=result_root,
            experiment_id=experiment_id,
            evidence=evidence,
            metadata=metadata,
        )
        assert written == result_path
        return written
    finally:
        _close_probe_logger(logger)


def _stage2_probe_namespace(args, *, worker_count: int):
    from types import SimpleNamespace

    return SimpleNamespace(
        load_model_path=args.model_source,
        stage1_adapter_path=args.stage1_adapter,
        resume_from_checkpoint_path=None,
        latent_size=args.latent_size,
        supervise_latent_boundaries=args.supervise_latent_boundaries,
        tuning_method=args.tuning_method,
        tuning_method_explicit=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        cache_dir=args.cache_dir,
        revision=None,
        token=None,
        local_files_only=args.local_files_only,
        stage2_length_manifest=args.stage2_length_manifest,
        stage2_length_manifest_sha256=args.stage2_length_manifest_sha256,
        data_path=args.data_path,
        image_root=args.image_root,
        max_seq_length_train=args.max_sequence_length,
        max_train_samples=None,
        attention_implementation=args.attention_implementation,
        gradient_checkpointing=args.gradient_checkpointing,
        per_device_train_batch_size=1,
        dataloader_num_workers=worker_count,
        save_model_path=str(
            Path("checkpoints/evaluation-staging") / args.experiment_id
        ),
        epochs=1.0,
        gradient_accumulation_steps=1,
        warm_up_steps=0,
        learning_rate=args.learning_rate,
        save_steps=10_000,
        max_steps=-1,
        deepspeed_config=None,
    )


def run_stage2_probe(args) -> Path:
    """Measure retained representatives through the production Stage 2 loss path."""
    from main import build_stage2_training_arguments, build_stage2_training_components
    from stage2.length_manifest import read_stage2_length_manifest
    from stage2.peft import resolve_stage2_qwen_model

    canonical_paths = resolve_run_artifact_paths(args.experiment_id)
    result_root = canonical_paths.result_dir.parent
    log_root = canonical_paths.log_dir.parent
    result_path, log_path = _artifact_paths(
        result_root=result_root,
        log_root=log_root,
        experiment_id=args.experiment_id,
    )
    manifest_path = Path(args.stage2_length_manifest)
    actual_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_sha256 != args.stage2_length_manifest_sha256:
        raise ValueError("Stage 2 length manifest SHA-256 does not match")
    manifest = read_stage2_length_manifest(manifest_path)
    representatives = select_representative_examples(manifest)

    worker_count = args.dataloader_num_workers
    components = build_stage2_training_components(
        _stage2_probe_namespace(args, worker_count=worker_count)
    )
    training_args = build_stage2_training_arguments(
        _stage2_probe_namespace(args, worker_count=worker_count)
    )
    trainer = components.trainer_class(
        model=components.model,
        args=training_args,
        train_dataset=components.train_dataset,
        data_collator=components.data_collator,
        processing_class=components.processor,
        stage2_peft_metadata=components.peft_metadata,
        stage2_diagnostics_interval=0,
    )
    trainer.create_optimizer_and_scheduler(
        num_training_steps=len(representatives) * (WARMUP_COUNT + MEASURED_COUNT)
    )
    qwen = resolve_stage2_qwen_model(trainer.model)
    logger = _open_probe_logger(log_path)
    representative_payloads = []
    all_samples = []
    try:
        for representative in representatives:
            retained_offset = manifest.retained_indices.index(representative.raw_index)
            example = components.train_dataset[retained_offset]
            loader = build_representative_dataloader(
                example=example,
                collate_fn=components.data_collator,
                worker_count=worker_count,
                batch_count=WARMUP_COUNT + MEASURED_COUNT,
            )
            iterator = iter(loader)

            def measure() -> Stage2EfficiencySample:
                return measure_cuda_training_step(
                    decoder=qwen.model,
                    visual=qwen.visual,
                    lm_head=qwen.lm_head,
                    collate_fn=lambda: trainer._prepare_inputs(next(iterator)),
                    loss_fn=lambda batch: trainer.compute_loss(trainer.model, batch),
                    optimizer=trainer.optimizer,
                    scheduler=trainer.lr_scheduler,
                )

            try:
                for _ in range(WARMUP_COUNT):
                    measure()
                samples = tuple(measure() for _ in range(MEASURED_COUNT))
            finally:
                shutdown = getattr(iterator, "_shutdown_workers", None)
                if callable(shutdown):
                    shutdown()
            evidence = summarize_samples(samples)
            all_samples.extend(samples)
            representative_payloads.append(
                {
                    **asdict(representative),
                    "evidence": asdict(evidence),
                    "samples": [asdict(sample) for sample in samples],
                }
            )
            for index, sample in enumerate(samples):
                _log_evidence(
                    logger,
                    f"representative={representative.quantile} sample={index}",
                    sample,
                )
            _log_evidence(logger, f"representative={representative.quantile}", evidence)
        overall = summarize_samples(all_samples)
        metadata = {
            **_device_metadata(),
            "probe_kind": "stage2_no_cache",
            "model_source": str(Path(args.model_source).resolve()),
            "stage1_adapter": (
                None
                if args.stage1_adapter is None
                else str(Path(args.stage1_adapter).resolve())
            ),
            "attention_implementation": args.attention_implementation,
            "gradient_checkpointing": args.gradient_checkpointing,
            "tuning_method": args.tuning_method,
            "warmup_count_per_representative": WARMUP_COUNT,
            "measured_count_per_representative": MEASURED_COUNT,
            "manifest_validated": True,
            "representative_quantiles": [
                representative.quantile for representative in representatives
            ],
            "profile_uses_fixed_worker_count": True,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": actual_sha256,
            "selected_worker_count": worker_count,
            "representatives": representative_payloads,
        }
        overall = authorize_segmented_kv(overall, metadata)
        _log_evidence(logger, "overall", overall)
        written = write_efficiency_result(
            result_root=result_root,
            experiment_id=args.experiment_id,
            evidence=overall,
            metadata=metadata,
        )
        assert written == result_path
        return written
    finally:
        _close_probe_logger(logger)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure the optimized Stage 2 no-cache path and apply its KV gate"
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-source")
    parser.add_argument("--stage1-adapter")
    parser.add_argument("--data-path")
    parser.add_argument("--image-root")
    parser.add_argument("--stage2-length-manifest")
    parser.add_argument("--stage2-length-manifest-sha256")
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument(
        "--supervise-latent-boundaries",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--tuning-method", choices=("full", "lora", "dora"))
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="flash_attention_2",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--dataloader-num-workers", type=int, choices=(0, 2, 4))
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    canonical_paths = resolve_run_artifact_paths(args.experiment_id)
    if args.self_test:
        path = run_cuda_self_test(
            experiment_id=args.experiment_id,
            result_root=canonical_paths.result_dir.parent,
            log_root=canonical_paths.log_dir.parent,
        )
    else:
        required = {
            "--model-source": args.model_source,
            "--data-path": args.data_path,
            "--image-root": args.image_root,
            "--stage2-length-manifest": args.stage2_length_manifest,
            "--stage2-length-manifest-sha256": args.stage2_length_manifest_sha256,
            "--tuning-method": args.tuning_method,
            "--dataloader-num-workers": args.dataloader_num_workers,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit(f"representative probe requires: {', '.join(missing)}")
        path = run_stage2_probe(args)
    print(path)
    return 0


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _validate_experiment_id(experiment_id: str) -> None:
    if (
        not isinstance(experiment_id, str)
        or not experiment_id
        or Path(experiment_id).name != experiment_id
        or experiment_id in {".", ".."}
    ):
        raise ValueError("experiment_id must be one path component")


def write_efficiency_result(
    *,
    result_root: str | Path,
    experiment_id: str,
    evidence: Stage2EfficiencyEvidence,
    metadata: Mapping[str, object],
) -> Path:
    """Publish one canonical schema-v1 result without replacing prior evidence."""
    _validate_experiment_id(experiment_id)
    if not isinstance(evidence, Stage2EfficiencyEvidence):
        raise TypeError("evidence must be Stage2EfficiencyEvidence")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    expected_authorization = authorize_segmented_kv(evidence, metadata)
    if (
        evidence.segmented_kv_authorized
        != expected_authorization.segmented_kv_authorized
    ):
        raise ValueError(
            "segmented KV authorization requires representative B200 metadata"
        )
    result_dir = Path(result_root) / experiment_id
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "stage2-efficiency.json"
    payload = {
        "schema_version": EFFICIENCY_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "prefix_threshold_met": evidence.prefix_threshold_met,
        "segmented_kv_authorized": evidence.segmented_kv_authorized,
        "evidence": asdict(evidence),
        "metadata": dict(metadata),
    }
    serialized = _canonical_json(payload) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return path


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def read_efficiency_result(path: str | Path) -> dict:
    """Read a strict schema-v1 probe result for downstream authorization."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"efficiency result must be a regular file: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid efficiency result JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("efficiency result must be a JSON object")
    expected = {
        "schema_version",
        "experiment_id",
        "prefix_threshold_met",
        "segmented_kv_authorized",
        "evidence",
        "metadata",
    }
    if set(payload) != expected:
        raise ValueError(
            f"efficiency result has unknown fields {sorted(set(payload) - expected)} "
            f"or missing fields {sorted(expected - set(payload))}"
        )
    if payload["schema_version"] != EFFICIENCY_SCHEMA_VERSION:
        raise ValueError("unsupported efficiency result schema_version")
    _validate_experiment_id(payload["experiment_id"])
    for name in ("prefix_threshold_met", "segmented_kv_authorized"):
        if not isinstance(payload[name], bool):
            raise ValueError(f"{name} must be a boolean")
    if not isinstance(payload["evidence"], dict) or not isinstance(
        payload["metadata"], dict
    ):
        raise ValueError("evidence and metadata must be JSON objects")
    evidence_fields = {field.name for field in fields(Stage2EfficiencyEvidence)}
    if set(payload["evidence"]) != evidence_fields:
        raise ValueError("evidence fields do not match schema version 1")
    evidence = Stage2EfficiencyEvidence(**payload["evidence"])
    expected_authorization = authorize_segmented_kv(evidence, payload["metadata"])
    if (
        payload["prefix_threshold_met"] != evidence.prefix_threshold_met
        or payload["segmented_kv_authorized"]
        != expected_authorization.segmented_kv_authorized
        or evidence.segmented_kv_authorized
        != expected_authorization.segmented_kv_authorized
    ):
        raise ValueError("segmented KV authorization does not match project gates")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
