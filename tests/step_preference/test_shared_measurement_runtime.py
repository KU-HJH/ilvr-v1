"""Regression witnesses for standalone measurement setup and failure evidence."""
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest
import torch

from step_preference import equivalence
from step_preference import shared_prefix_measurement as measurement


def _options():
    root = Path(__file__).resolve().parents[2]
    return SimpleNamespace(output_path=root / "results" / f"shared-prefix-runtime-test-{uuid.uuid4().hex}" / "report.json",
        preserve_dropout=False, tiny_only=False, checkpoint_path="missing-shared-measurement-checkpoint",
        data_path="missing", image_root="missing", negative_path="missing", negative_manifest="missing",
        device="cuda:0", tuning_method="full", gradient_checkpointing=False, requested_sample_id=None,
        sidecar_attention_backend="flex", sidecar_query_budget=1024, warmups=0, measurements=1)


def test_actual_missing_checkpoint_writes_failure_evidence():
    """Catches preflight/load failures escaping with only stderr and no JSON."""
    args = _options()
    with pytest.raises((FileNotFoundError, ValueError)):
        measurement.run_shared_measurement_cli(args, benchmark=False, git_commit="test")
    assert args.output_path.exists()
    report = json.loads(args.output_path.read_text())
    assert report["status"] == "fail" and report["errors"]
    assert report["records"] == []
    # A repeated attempt cannot overwrite the first evidence file.
    original = args.output_path.read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        measurement.run_shared_measurement_cli(args, benchmark=False, git_commit="other")
    assert args.output_path.read_bytes() == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual BF16/Flex execution requires CUDA")
def test_production_measurement_casts_both_routes_to_engine_parameter_precision(monkeypatch):
    """Catches fixing tiny only, or casting only the shared candidate."""
    args = _options()
    model, batch = measurement.tiny_measurement_case(steps=1, total_target_tokens=2, device="cuda:0")
    config = equivalence._config(packing=False)
    # The expensive authenticated loader is replaced; actual FP32 model, both
    # Trainer routes, autocast, Flex, backward and report writer still execute.
    context = equivalence.ProductionContext(model, batch, config, equivalence._config(packing=True), {})
    monkeypatch.setattr(equivalence, "load_production_context", lambda **kwargs: context)
    seen = []
    handle = model.register_forward_pre_hook(lambda module, args: seen.append({p.dtype for p in module.parameters()}))
    try:
        assert measurement.run_shared_measurement_cli(args, benchmark=False, git_commit="test") == 0
    finally:
        handle.remove()
    assert len(seen) >= 2 and all(dtypes == {torch.bfloat16} for dtypes in seen)
    report = json.loads(args.output_path.read_text())
    assert report["records"][0]["parameter_dtypes"] == ["torch.bfloat16"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual first benchmark path requires CUDA")
def test_failure_after_measured_path_preserves_completed_record(monkeypatch):
    """Catches a later unavailable path erasing a completed real measurement."""
    args = _options()
    model, batch = measurement.tiny_measurement_case(steps=1, total_target_tokens=2, device="cuda:0")
    model.bfloat16()
    config = equivalence._config(packing=False)
    context = equivalence.ProductionContext(model, batch, config, equivalence._config(packing=True), {})
    monkeypatch.setattr(equivalence, "load_production_context", lambda **kwargs: context)
    real_measure = measurement.measure_paths
    def stop_after_naive(model, batch, configs, runtime, **kwargs):
        if "shared" in configs:
            raise RuntimeError("next benchmark path unavailable")
        return real_measure(model, batch, configs, runtime, **kwargs)
    monkeypatch.setattr(measurement, "measure_paths", stop_after_naive)
    with pytest.raises(RuntimeError, match="next benchmark path unavailable"):
        measurement.run_shared_measurement_cli(args, benchmark=True, git_commit="test")
    report = json.loads(args.output_path.read_text())
    assert report["status"] == "fail"
    assert [r["mode"] for r in report["records"]] == ["naive"]
    assert report["records"][0]["runtime_counters"]["rollout_calls_forward"] == 4
    assert report["records"][0]["median_ms"] > 0
    assert report["errors"] == ["next benchmark path unavailable"]
