"""Observed accounting must catch prefix replay or missing backward work."""
from dataclasses import replace
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from step_preference.equivalence import _config


def measurement():
    from step_preference import equivalence
    assert hasattr(equivalence, "run_shared_prefix_comparison"), "shared comparison is missing"
    return importlib.import_module("step_preference.shared_prefix_measurement")


@pytest.mark.parametrize("checkpoint", [False, True])
def test_counters_observe_suffix_only_and_backward_checkpoint_replay(checkpoint):
    m = measurement()
    model, batch = m.tiny_measurement_case(steps=2, total_target_tokens=8, device="cpu")
    if checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config = replace(_config(packing=False), shared_prefix_enabled=True,
                     sidecar_attention_backend="dense")
    with m.RuntimeCounters(model) as counters:
        step = m.MeasuredTrainingStep(model, batch, config, counters, autocast_dtype=None)
        step()
    got = counters.snapshot()
    assert got["policy_calls_forward"] == 1
    assert got["rollout_calls_forward"] == 1
    assert got["latent_decoder_calls_forward"] == 4
    assert got["decoder_calls_forward"] == 7
    # Two four-token targets, V/T entropy plus one four-token rejection each.
    assert got["sidecar_projected_rows_forward"] == 24
    assert got["sidecar_mlp_rows_forward"] == 24
    assert got["sidecar_projected_rows_backward"] == 24
    assert got["sidecar_lm_head_rows_forward"] == 24
    assert got["sidecar_chunks_forward"] == 2
    assert got["clean_projected_rows_backward"] > 0 if checkpoint else got.get("clean_projected_rows_backward", 0) == 0
    counts = m.batch_accounting(batch)
    assert counts["entropy_scored_tokens"] == 16
    assert counts["rejected_scored_tokens"] == 8


def test_tiny_comparison_checks_all_trainable_gradients():
    m = measurement()
    from step_preference.equivalence import run_shared_prefix_comparison
    model, batch = m.tiny_measurement_case(steps=2, total_target_tokens=8, device="cpu")
    report = run_shared_prefix_comparison(model, batch, naive_config=_config(packing=False),
                                          backend="dense", autocast_dtype=None, strict=True)
    assert report["status"] == "pass"
    assert report["trainable_parameters_compared"] > 20
    assert report["max_gradient_abs"] < 1e-5
    assert report["max_component_abs"] < 1e-5


def test_k_sweep_separates_fixed_target_length_from_growing_length():
    m = measurement()
    cases = m.tiny_sweep_cases(device="cpu", step_counts=(1, 2, 4), target_tokens=8)
    assert [(x["series"], x["steps"], m.batch_accounting(x["batch"])["chosen_target_tokens"])
            for x in cases] == [("fixed", 1, 8), ("fixed", 2, 8), ("fixed", 4, 8),
                               ("growing", 1, 8), ("growing", 2, 16), ("growing", 4, 32)]


@pytest.mark.parametrize("script", ["verify_packed_step_preference", "benchmark_packed_step_preference"])
def test_existing_cli_accepts_shared_backend_budget_and_preserves_legacy_default(script):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{script}.py"
    spec = importlib.util.spec_from_file_location(script, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    common = ["--checkpoint_path", "checkpoint", "--data_path", "data", "--image_root", "images",
              "--negative_path", "negative", "--negative_manifest", "manifest", "--output_path", "report"]
    args = module._parser().parse_args(common+["--shared-prefix", "--sidecar-attention-backend", "dense",
                                              "--sidecar-query-budget", "7"])
    assert args.shared_prefix and args.sidecar_attention_backend == "dense" and args.sidecar_query_budget == 7
    legacy = module._parser().parse_args(common)
    assert not legacy.shared_prefix
    assert legacy.sidecar_query_budget == 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual pinned tensor transport requires CUDA")
def test_actual_dataloader_pin_memory_metadata_remains_valid_for_shared_planner():
    """Catches rejecting lists produced from tuple metadata by Torch pin_memory."""
    from torch.utils.data._utils.pin_memory import pin_memory
    from step_preference.sidecar import build_sidecar_plan
    m = measurement()
    _, batch = m.tiny_measurement_case(steps=2, total_target_tokens=8, device="cpu")
    pinned = pin_memory(batch)
    plan = build_sidecar_plan(clean_input_ids=pinned["input_ids"], attention_mask=pinned["attention_mask"],
        spans=pinned["reasoning_step_spans"], steps=pinned["step_preference"]["steps"])
    assert sum(len(group) for group in plan.groups) == 6
    assert sum(len(b.target_token_ids) for g in plan.groups for b in g) == 24


@pytest.mark.skipif(os.environ.get("RUN_SHARED_TINY_CLI") != "1", reason="explicit CUDA CLI witness")
def test_tiny_cli_runs_bf16_flex_after_strict_fp32():
    """Catches leaving tiny weights FP32 when Flex receives BF16 projected V."""
    import json
    root = Path(__file__).resolve().parents[2]
    run_id = os.environ["SHARED_PREFIX_RUN_ID"]
    output = root / "results" / run_id / "verify.json"
    from artifact_paths import require_canonical_artifact_path
    log = require_canonical_artifact_path(root / "logs" / run_id / "verify.log", kind="log")
    log.parent.mkdir(parents=True, exist_ok=False)
    with log.open("x") as handle:
        completed = subprocess.run([sys.executable, str(root / "scripts/verify_packed_step_preference.py"),
            "--shared-prefix", "--tiny-only", "--sidecar-attention-backend", "flex",
            "--device", "cuda:0", "--output_path", str(output)], stdout=handle, stderr=subprocess.STDOUT)
    assert completed.returncode == 0, log.read_text()
    records = json.loads(output.read_text())["records"]
    assert records[0]["status"] == "pass" and records[0]["dtype"] == "torch.float32"
    assert records[1]["status"] == "diagnostic" and records[1]["dtype"] == "torch.bfloat16"
    assert not records[1]["errors"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="legacy tree Flex requires CUDA")
def test_legacy_measurement_labels_tree_flex_separately_from_clean_sdpa():
    """Catches reporting configured SDPA as the actual legacy tree backend."""
    from step_preference.benchmark import CudaBenchmarkRuntime
    m = measurement()
    model, batch = m.tiny_measurement_case(steps=1, total_target_tokens=2, device="cuda:0")
    model.to(dtype=torch.bfloat16)
    config = replace(_config(packing=True), attention_implementation="sdpa")
    records = m.measure_paths(model, batch, {"legacy_packed": config}, CudaBenchmarkRuntime("cuda:0"),
                              warmups=0, measurements=1)
    assert records[0]["backend"] == "flex"
    assert records[0]["ordinary_language_backend"] == "sdpa"
