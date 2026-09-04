import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from step_preference.equivalence import load_production_context
from step_preference.config import StepPreferenceConfig
from scripts import probe_packed_stage2b_capacity as capacity_probe
from scripts.probe_packed_stage2b_capacity import run_capacity_probe


class _Model(torch.nn.Module):
    def __init__(self, *, checkpointing: bool):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor([1.0]))
        self.config = SimpleNamespace(attention_dropout=0.1)
        self.is_gradient_checkpointing = checkpointing
        self.disable_calls = 0
        self.device = None
        self.training_requested = False

    def gradient_checkpointing_disable(self):
        self.disable_calls += 1
        self.is_gradient_checkpointing = False

    def to(self, device):
        self.device = device
        return self

    def train(self, mode=True):
        self.training_requested = mode
        return super().train(mode)


@pytest.mark.parametrize(
    ("options", "checkpointing", "expected_method", "expected_flag"),
    [
        (
            {"tuning_method": "dora", "gradient_checkpointing": True},
            True,
            "dora",
            "--gradient_checkpointing",
        ),
        ({}, False, "full", "--no-gradient_checkpointing"),
    ],
)
def test_production_context_routes_tuning_and_checkpointing_without_overrides(
    monkeypatch,
    tmp_path,
    options,
    checkpointing,
    expected_method,
    expected_flag,
):
    """Catches the capacity loader silently reverting to full/no-checkpointing."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    checkpoint = tmp_path / "checkpoint"
    image_root = tmp_path / "images"
    checkpoint.mkdir()
    image_root.mkdir()
    data_path = tmp_path / "data.jsonl"
    negative_path = tmp_path / "negatives.jsonl"
    manifest_path = tmp_path / "manifest.json"
    for path in (data_path, negative_path, manifest_path):
        path.write_text("{}\n", encoding="utf-8")

    captured = {}

    def fake_parse_args(argv):
        captured["argv"] = tuple(argv)
        return SimpleNamespace()

    negative = SimpleNamespace(
        valid_visual_negative=True,
        valid_textual_negative=False,
    )
    prepared = SimpleNamespace(
        sample_id="sample-a",
        reasoning_steps=(SimpleNamespace(step_id=0),),
    )
    preflight = SimpleNamespace(
        prepared_examples=(prepared,),
        negative_store={("sample-a", 0): negative},
    )
    model = _Model(checkpointing=checkpointing)
    components = SimpleNamespace(
        model=model,
        train_dataset=(prepared,),
        data_collator=lambda _examples: {
            "sample_ids": ("sample-a",),
            "step_preference": {"steps": ({"span": SimpleNamespace(step_id=0)},)},
        },
        step_preference_config=StepPreferenceConfig(
            enabled=True,
            mode="stage2b",
            negative_path=str(negative_path),
            negative_manifest=str(manifest_path),
            stage2a_checkpoint_path=str(checkpoint),
            attention_implementation="sdpa",
        ),
    )
    monkeypatch.setattr("utils_deepseed.parse_args", fake_parse_args)
    monkeypatch.setattr("main.preflight_stage2b_artifacts", lambda _args: preflight)
    monkeypatch.setattr("main.build_stage2_training_components", lambda _args: components)
    monkeypatch.setattr(
        "step_preference.negatives.read_negative_manifest",
        lambda _path: SimpleNamespace(
            selected_row_indices=(0,), negative_sha256="a" * 64
        ),
    )

    context = load_production_context(
        checkpoint_path=checkpoint,
        data_path=data_path,
        image_root=image_root,
        negative_path=negative_path,
        negative_manifest=manifest_path,
        device="cuda:0",
        **options,
    )

    argv = captured["argv"]
    assert argv[argv.index("--tuning_method") + 1] == expected_method
    assert expected_flag in argv
    assert model.disable_calls == 0
    assert model.is_gradient_checkpointing is checkpointing
    assert context.identity["tuning_method"] == expected_method
    assert context.identity["gradient_checkpointing"] is checkpointing


class _Runtime:
    def __init__(self):
        self.calls = []

    def synchronize(self):
        self.calls.append("synchronize")

    def device_total_bytes(self):
        return 10_000

    def memory_allocated(self):
        return 1_000

    def memory_reserved(self):
        return 1_500

    def reset_peak_memory_stats(self):
        self.calls.append("reset_peak")

    def max_memory_allocated(self):
        return 4_000

    def max_memory_reserved(self):
        return 5_000


def _context():
    model = _Model(checkpointing=True)
    return SimpleNamespace(
        model=model,
        batch={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "sample_ids": ("sample-a",),
            "reasoning_step_spans": (object(), object()),
            "step_preference": {"steps": (object(), object())},
        },
        packed_config=object(),
        identity={
            "checkpoint_path": "/checkpoint",
            "sample_id": "sample-a",
            "negative_sha256": "a" * 64,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "tuning_method": "dora",
            "gradient_checkpointing": True,
        },
    )


def test_capacity_report_passes_only_with_finite_losses_and_adapter_gradient(
    monkeypatch
):
    """Catches a successful report that omits raw objectives or real gradients."""
    context = _context()
    runtime = _Runtime()
    calls = []
    monkeypatch.setattr(
        "scripts.probe_packed_stage2b_capacity.build_packed_tree_plan",
        lambda **_kwargs: SimpleNamespace(
            root_length=3,
            unpadded_length=241,
            padded_length=256,
            branches=(object(), object(), object()),
        ),
    )

    class Step:
        def __init__(self, model, batch, config):
            assert model is context.model
            assert batch is context.batch
            assert config is context.packed_config
            self.trainer = SimpleNamespace(
                _step_preference_metric_totals={
                    "loss/ce": 1.25,
                    "loss/dpo": 0.5,
                    "loss/entropy": 0.125,
                    "loss/total": 1.875,
                },
                _step_preference_metric_batches=1,
            )

        def zero_grad(self):
            calls.append("zero_grad")

        def __call__(self):
            calls.append("step")
            context.model.adapter.grad = torch.tensor([3.0])

    report = run_capacity_probe(
        context,
        runtime,
        source_row_index=7,
        step_factory=Step,
        clock=iter((10.0, 12.5)).__next__,
    )

    assert set(report) == {
        "schema_version",
        "status",
        "identity",
        "sample",
        "losses",
        "gradient_norm",
        "memory",
        "elapsed_seconds",
        "errors",
    }
    assert report["status"] == "pass"
    assert report["sample"] == {
        "source_row_index": 7,
        "sample_id": "sample-a",
        "clean_length": 3,
        "step_count": 2,
        "branch_count": 3,
        "packed_unpadded_length": 241,
        "packed_padded_length": 256,
    }
    assert report["losses"] == {
        "ce": 1.25,
        "dpo": 0.5,
        "entropy": 0.125,
        "total": 1.875,
    }
    assert report["gradient_norm"] == 3.0
    assert report["memory"] == {
        "device_total_bytes": 10_000,
        "baseline_allocated_bytes": 1_000,
        "peak_allocated_bytes": 4_000,
        "baseline_reserved_bytes": 1_500,
        "peak_reserved_bytes": 5_000,
    }
    assert report["elapsed_seconds"] == 2.5
    assert report["errors"] == []
    assert calls == ["zero_grad", "step"]
    assert runtime.calls == ["synchronize", "reset_peak", "synchronize"]


def test_capacity_report_classifies_only_torch_oom_as_oom(monkeypatch):
    """Catches CUDA exhaustion being mislabeled as a generic or passing result."""
    context = _context()
    runtime = _Runtime()
    monkeypatch.setattr(
        "scripts.probe_packed_stage2b_capacity.build_packed_tree_plan",
        lambda **_kwargs: SimpleNamespace(
            root_length=3,
            unpadded_length=128,
            padded_length=128,
            branches=(object(),),
        ),
    )

    class OomStep:
        def __init__(self, *_args):
            self.trainer = SimpleNamespace(
                _step_preference_metric_totals={},
                _step_preference_metric_batches=0,
            )

        def zero_grad(self):
            pass

        def __call__(self):
            raise torch.OutOfMemoryError("capacity exhausted")

    report = run_capacity_probe(
        context,
        runtime,
        source_row_index=0,
        step_factory=OomStep,
        clock=iter((1.0, 1.25)).__next__,
    )

    assert report["status"] == "oom"
    assert report["losses"] == {
        "ce": None,
        "dpo": None,
        "entropy": None,
        "total": None,
    }
    assert report["gradient_norm"] is None
    assert report["elapsed_seconds"] == 0.25
    assert report["errors"] == ["OutOfMemoryError: capacity exhausted"]


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_exit"),
    [
        (torch.OutOfMemoryError("load exhausted"), "oom", 1),
        (RuntimeError("load failed"), "error", 2),
    ],
)
def test_capacity_cli_publishes_exact_failure_status_and_exit_code(
    monkeypatch, tmp_path, error, expected_status, expected_exit
):
    """Catches load-time CUDA OOM or ordinary errors receiving the wrong exit code."""
    output = tmp_path / "capacity.json"
    monkeypatch.setattr(
        capacity_probe.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="deadbeef\n"),
    )
    monkeypatch.setattr(
        capacity_probe,
        "load_production_context",
        lambda **_kwargs: (_ for _ in ()).throw(error),
    )

    exit_code = capacity_probe.main(
        [
            "--checkpoint_path",
            "checkpoint",
            "--data_path",
            "data.jsonl",
            "--image_root",
            "images",
            "--negative_path",
            "negatives.jsonl",
            "--negative_manifest",
            "manifest.json",
            "--output_path",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == expected_exit
    assert report["status"] == expected_status
    assert set(report) == {
        "schema_version",
        "status",
        "identity",
        "sample",
        "losses",
        "gradient_norm",
        "memory",
        "elapsed_seconds",
        "errors",
    }


def test_capacity_cli_publishes_git_identity_failure_as_error(monkeypatch, tmp_path):
    """Catches Git setup failure escaping without JSON or reusing OOM exit 1."""
    output = tmp_path / "capacity.json"

    def fail_git(*_args, **_kwargs):
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(capacity_probe.subprocess, "run", fail_git)

    exit_code = capacity_probe.main(
        [
            "--checkpoint_path",
            "checkpoint",
            "--data_path",
            "data.jsonl",
            "--image_root",
            "images",
            "--negative_path",
            "negatives.jsonl",
            "--negative_manifest",
            "manifest.json",
            "--output_path",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["status"] == "error"
    assert report["identity"]["git_commit"] is None
    assert report["errors"] == [
        "CalledProcessError: Command '['git', 'rev-parse', 'HEAD']' "
        "returned non-zero exit status 128."
    ]
