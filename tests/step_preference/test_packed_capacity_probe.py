import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from step_preference.equivalence import load_production_context
from step_preference.config import StepPreferenceConfig
from scripts import audit_packed_stage2b_lengths as capacity_audit
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
        captured["parsed"] = SimpleNamespace()
        return captured["parsed"]

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
            max_packed_sequence_length=4608,
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
    assert captured["parsed"]._stage2b_capacity_probe is True
    assert argv[argv.index("--attention_implementation") + 1] == "sdpa"
    assert (
        argv[argv.index("--vision_attention_implementation") + 1]
        == "flash_attention_2"
    )
    assert argv[argv.index("--tuning_method") + 1] == expected_method
    assert expected_flag in argv
    assert model.disable_calls == 0
    assert model.is_gradient_checkpointing is checkpointing
    assert context.identity["tuning_method"] == expected_method
    assert context.identity["gradient_checkpointing"] is checkpointing
    assert context.identity["attention_implementation"] == "sdpa"
    assert context.identity["vision_attention_implementation"] == (
        "flash_attention_2"
    )


def test_length_audit_authenticates_and_records_both_attention_backends(
    monkeypatch, tmp_path
):
    """Catches the audit authenticating or reporting only the language backend."""
    checkpoint = tmp_path / "checkpoint"
    images = tmp_path / "images"
    checkpoint.mkdir()
    images.mkdir()
    data = tmp_path / "data.jsonl"
    negatives = tmp_path / "negatives.jsonl"
    manifest = tmp_path / "manifest.json"
    for path in (data, negatives, manifest):
        path.write_text("{}\n", encoding="utf-8")
    captured = {}

    def parse_args(argv):
        captured["argv"] = tuple(argv)
        return SimpleNamespace(
            attention_implementation="sdpa",
            vision_attention_implementation="flash_attention_2",
            latent_size=8,
            max_seq_length_train=8192,
            supervise_latent_boundaries=True,
        )

    monkeypatch.setattr("utils_deepseed.parse_args", parse_args)
    monkeypatch.setattr(
        "main.preflight_stage2b_artifacts",
        lambda _args: SimpleNamespace(prepared_examples=(), negative_store={}),
    )
    monkeypatch.setattr(
        "step_preference.negatives.read_negative_manifest",
        lambda _path: SimpleNamespace(
            negative_sha256="a" * 64,
            assistant_plan_rendering="adjacent-text-space-lf-v1",
        ),
    )
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(tokenizer=object()),
    )
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "stage2.checkpointing.resolve_latent_token_ids",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "stage2.data.build_prepared_example_length_measure",
        lambda _processor: object(),
    )
    monkeypatch.setattr(
        "stage2.data.filter_stage2_examples",
        lambda **_kwargs: ((), object()),
    )
    monkeypatch.setattr(
        "stage2.data.Stage2DataCollator", lambda *_args, **_kwargs: object()
    )

    identity, records = capacity_audit.collect_packed_length_audit(
        SimpleNamespace(
            checkpoint_path=str(checkpoint),
            data_path=str(data),
            image_root=str(images),
            negative_path=str(negatives),
            negative_manifest=str(manifest),
        )
    )

    argv = captured["argv"]
    assert argv[argv.index("--attention_implementation") + 1] == "sdpa"
    assert (
        argv[argv.index("--vision_attention_implementation") + 1]
        == "flash_attention_2"
    )
    assert identity["attention_implementation"] == "sdpa"
    assert identity["vision_attention_implementation"] == "flash_attention_2"
    assert records == ()


class _Runtime:
    def __init__(self):
        self.calls = []

    def synchronize(self):
        self.calls.append("synchronize")

    def device_total_bytes(self):
        return 10_000

    def physical_index(self):
        return 0

    def device_name(self):
        return "NVIDIA H100 NVL"

    def device_uuid(self):
        return "01234567-0123-4567-89ab-0123456789ab"

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
        "gpu",
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
        "peak_reserved_fraction": 0.5,
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
    context_loads = []
    monkeypatch.setattr(capacity_probe, "git_identity", lambda: "a"*40)
    monkeypatch.setattr(capacity_probe, "authenticate_probe_gpu", lambda *a: {}, raising=False)

    def fail_context(**kwargs):
        context_loads.append(kwargs)
        raise error

    monkeypatch.setattr(capacity_probe, "load_production_context", fail_context)

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
            "--gpu-snapshot", str(tmp_path / "gpu.json"),
            "--gpu-snapshot-sha256", "a" * 64,
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == expected_exit
    assert report["status"] == expected_status
    assert context_loads[0]["attention_implementation"] == "sdpa"
    assert context_loads[0]["vision_attention_implementation"] == (
        "flash_attention_2"
    )
    assert report["identity"]["attention_implementation"] == "sdpa"
    assert report["identity"]["vision_attention_implementation"] == (
        "flash_attention_2"
    )
    assert set(report) == {
        "schema_version",
        "status",
        "identity",
        "sample",
        "losses",
        "gradient_norm",
        "memory",
        "gpu",
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
            "--gpu-snapshot", str(tmp_path / "gpu.json"),
            "--gpu-snapshot-sha256", "a" * 64,
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


@pytest.mark.parametrize("peak,status", [(8500, "pass"), (8501, "error")])
def test_capacity_headroom_uses_exact_bytes(monkeypatch, peak, status):
    context, runtime = _context(), _Runtime()
    monkeypatch.setattr(runtime, "max_memory_reserved", lambda: peak)
    monkeypatch.setattr(capacity_probe, "build_packed_tree_plan", lambda **kw: SimpleNamespace(
        root_length=3, unpadded_length=100, padded_length=128, branches=(1,)))
    class Step:
        def __init__(self, *a):
            self.trainer = SimpleNamespace(_step_preference_metric_batches=1,
                _step_preference_metric_totals={k: 1.0 for k in ("loss/ce", "loss/dpo", "loss/entropy", "loss/total")})
        def zero_grad(self): pass
        def __call__(self): context.model.adapter.grad = torch.ones(1)
    report = run_capacity_probe(context, runtime, source_row_index=7, step_factory=Step)
    assert report["status"] == status
    assert report["memory"]["peak_reserved_fraction"] == peak / 10000
    assert report["gpu"]["name"] == "NVIDIA H100 NVL"


def test_capacity_rejects_wrong_requested_sample_before_step(monkeypatch):
    monkeypatch.setattr(capacity_probe, "build_packed_tree_plan", lambda **kw: SimpleNamespace(
        root_length=3, unpadded_length=100, padded_length=128, branches=(1,)))
    report = run_capacity_probe(_context(), _Runtime(), source_row_index=7,
        sample_id="another", step_factory=lambda *a: pytest.fail("step constructed"))
    assert report["status"] == "error"
