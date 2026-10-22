import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import stage1.evaluation_diagnostics as diagnostics
from stage1.evaluation_diagnostics import (
    boundary_target_parameters,
    compare_forced_prefix,
    cosine_similarity,
    derive_execution_log_path,
    independent_no_cache_forced_prefix,
    main,
    parameter_delta_summary,
    rescore_historical_rows,
    symmetric_relative_l2,
    token_rank,
    validated_test_source_rows,
    validate_assistant_cue_scaffold,
)


def test_diagnostic_entrypoint_imports_with_build_marker_outside_pytest_normalization():
    """Catches diagnostics importing datasets with the host's trailing-plus version."""
    repo_root = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{repo_root}:{repo_root / 'src'}:{repo_root / 'transformers/src'}",
    }
    completed = subprocess.run(
        [
            str(repo_root / "venv/ilvr-venv/bin/python"),
            "-m",
            "stage1.evaluation_diagnostics",
            "--help",
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "recurrence-probe" in completed.stdout


def test_diagnostic_entrypoint_leaves_an_extra_build_marker_invalid():
    """Catches normalizing an independently invalid trailing-plus version."""
    repo_root = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{repo_root}:{repo_root / 'src'}:{repo_root / 'transformers/src'}",
    }
    completed = subprocess.run(
        [
            str(repo_root / "venv/ilvr-venv/bin/python"),
            "-c",
            """
import builtins
import platform
from packaging.version import InvalidVersion, Version

platform.python_version = lambda: "3.11.15++"
original = platform.python_version
real_import = builtins.__import__
observed = []

def validate_version_at_heavy_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch":
        value = platform.python_version()
        observed.append(value)
        Version(value)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = validate_version_at_heavy_import
try:
    import stage1.evaluation_diagnostics
except InvalidVersion as error:
    assert str(error) == "Invalid version: '3.11.15+'"
else:
    raise AssertionError("an extra build marker was accepted")
finally:
    builtins.__import__ = real_import

assert observed == ["3.11.15+"]
assert platform.python_version is original
assert platform.python_version() == "3.11.15++"
""",
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_diagnostic_entrypoint_restores_platform_after_heavy_import_failure():
    """Catches a failed heavy import leaking the patched platform callable."""
    repo_root = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{repo_root}:{repo_root / 'src'}:{repo_root / 'transformers/src'}",
    }
    completed = subprocess.run(
        [
            str(repo_root / "venv/ilvr-venv/bin/python"),
            "-c",
            """
import builtins
import platform

platform.python_version = lambda: "3.11.15+"
original = platform.python_version
real_import = builtins.__import__
observed = []

def fail_torch_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch":
        observed.append(platform.python_version())
        assert platform.python_version() == "3.11.15"
        raise RuntimeError("simulated heavy import failure")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = fail_torch_import
try:
    import stage1.evaluation_diagnostics
except RuntimeError as error:
    assert str(error) == "simulated heavy import failure"
else:
    raise AssertionError("the simulated heavy import unexpectedly succeeded")
finally:
    builtins.__import__ = real_import

assert observed == ["3.11.15"]
assert platform.python_version is original
assert platform.python_version().endswith("+")
""",
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_tensor_reducers_report_hand_checked_values_and_reject_nonfinite_inputs():
    """Catches a reducer that hides non-finite evidence or mis-ranks a token."""
    vector = torch.tensor([3.0, 4.0])

    assert symmetric_relative_l2(vector, vector.clone()) == 0.0
    assert cosine_similarity(vector, vector.clone()) == pytest.approx(1.0)
    assert token_rank(torch.tensor([3.0, 1.0, 2.0]), 2) == 2

    with pytest.raises(ValueError, match="finite"):
        symmetric_relative_l2(vector, torch.tensor([math.nan, 4.0]))
    with pytest.raises(ValueError, match="finite"):
        token_rank(torch.tensor([3.0, math.inf, 2.0]), 2)


def test_historical_rescore_joins_unique_indices_and_marks_legacy_limitations():
    """Catches rescoring by row order or pretending old raw-generation evidence exists."""
    rows = [
        {"index": 1, "prediction": "Final answer: 5", "match": False},
        {"index": 0, "prediction": "Final answer: 2", "match": False},
    ]
    dataset = [
        {"original_final_answer": "2"},
        {"original_final_answer": "4"},
    ]

    report = rescore_historical_rows(rows, dataset)

    assert report["source_count"] == 2
    assert report["legacy_raw_id_limitation"] == "unavailable"
    assert report["old_correct"] == 0
    assert report["new_correct"] == 1
    assert report["match_changes"] == [
        {"index": 0, "old_match": False, "new_match": True},
    ]
    assert all(row["evidence_status"] == "legacy_assistant_only" for row in report["rows"])
    assert all(row["generated_suffix_token_ids"] is None for row in report["rows"])
    assert all(row["stop_reason"] is None for row in report["rows"])
    assert all(row["latent_evidence"] is None for row in report["rows"])

    with pytest.raises(ValueError, match="duplicate"):
        rescore_historical_rows(rows + [dict(rows[0])], dataset)
    with pytest.raises(ValueError, match="out of range"):
        rescore_historical_rows([{ "index": 2, "prediction": "2"}], dataset)
    with pytest.raises(ValueError, match="missing"):
        rescore_historical_rows(rows[:1], dataset)


def test_forced_prefix_comparison_requires_exact_discrete_evidence_and_thresholds():
    """Catches a recurrence gate that accepts a scaffold mismatch or a loose tensor delta."""
    reference = {
        "sequences": torch.tensor([[1, 91, 90, 92, 3]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "position_ids": torch.arange(5).view(1, 1, 5).expand(3, 1, 5),
        "latent_states": (torch.tensor([[1.0, 0.0]]),),
        "post_end_logits": torch.tensor([[0.0, 2.0]]),
    }
    result = compare_forced_prefix(reference, reference)

    assert result["pass"] is True
    assert result["discrete_match"] is True
    assert result["latent_state_count"] == 1
    assert result["max_symmetric_relative_l2"] == 0.0
    assert result["min_cosine_similarity"] == pytest.approx(1.0)

    changed = dict(reference, sequences=torch.tensor([[1, 7, 90, 92, 3]]))
    mismatch = compare_forced_prefix(reference, changed)
    assert mismatch["pass"] is False
    assert mismatch["discrete_match"] is False

    drifted = dict(reference, latent_states=(torch.tensor([[1.0, 1e-3]]),))
    with pytest.raises(ValueError, match="threshold"):
        compare_forced_prefix(reference, drifted, require_pass=True)


def test_parameter_delta_summary_reports_trainable_tensors_with_finite_distances():
    """Catches parameter evidence that omits a trainable row or accepts NaN deltas."""
    before = {
        "latent.output_delta": torch.tensor([[1.0, 2.0]]),
        "adapter.lora_B": torch.tensor([[0.0]]),
    }
    after = {
        "latent.output_delta": torch.tensor([[1.0, 2.5]]),
        "adapter.lora_B": torch.tensor([[0.1]]),
    }

    summary = parameter_delta_summary(before, after)

    assert summary["parameter_count"] == 2
    assert summary["parameters"]["latent.output_delta"]["changed_elements"] == 1
    assert summary["parameters"]["adapter.lora_B"]["l2_delta"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="finite"):
        parameter_delta_summary(before, {**after, "adapter.lora_B": torch.tensor([[math.nan]])})


def test_cli_enforces_required_contract_writes_result_and_deterministic_log(tmp_path, monkeypatch):
    """Catches a CLI that emits evidence without the paired logs/<run-id> record."""
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "results" / "unit-run"
    results.mkdir(parents=True)
    historical = tmp_path / "old.jsonl"
    dataset = tmp_path / "TEST.jsonl"
    historical.write_text('{"index": 0, "prediction": "2", "match": false}\n', encoding="utf-8")
    dataset.write_text('{"original_final_answer": "2"}\n', encoding="utf-8")
    output = results / "rescore.json"

    assert main([
        "rescore", "--input-jsonl", str(historical), "--dataset-jsonl", str(dataset),
        "--output-json", str(output),
    ]) == 0
    assert output.is_file()
    assert derive_execution_log_path(output) == tmp_path / "logs" / "unit-run" / "rescore.log"
    assert "status=success" in derive_execution_log_path(output).read_text(encoding="utf-8")

    nested_left = results / "left" / "report.json"
    nested_right = results / "right" / "report.json"
    assert derive_execution_log_path(nested_left) == tmp_path / "logs" / "unit-run" / "left" / "report.log"
    assert derive_execution_log_path(nested_right) == tmp_path / "logs" / "unit-run" / "right" / "report.log"
    assert derive_execution_log_path(nested_left) != derive_execution_log_path(nested_right)

    with pytest.raises(SystemExit) as missing:
        main(["rescore"])
    assert missing.value.code == 2
    with pytest.raises(ValueError, match="results"):
        main([
            "rescore", "--input-jsonl", str(historical), "--dataset-jsonl", str(dataset),
            "--output-json", str(tmp_path / "outside.json"),
        ])


class _CueTokenizer:
    def __call__(self, text, **_kwargs):
        assert text in {"<|im_start|>assistant", "<|im_start|>assistant\n"}
        return {"input_ids": [31, 32]}


class _BoundaryTargets(nn.Module):
    def __init__(self, *, include_rows=True, include_magnitude=True):
        super().__init__()
        self.lm_head = nn.Module()
        if include_rows:
            self.lm_head.output_delta = nn.Parameter(torch.zeros(3, 2))
        self.adapter = nn.Module()
        self.adapter.lora_A = nn.Parameter(torch.zeros(2, 2))
        self.adapter.lora_B = nn.Parameter(torch.zeros(2, 2))
        if include_magnitude:
            self.adapter.lora_magnitude_vector = nn.Parameter(torch.zeros(2))


def test_boundary_validates_test_source_identity_cue_and_all_trainable_targets(tmp_path, monkeypatch):
    """Catches a reordered/non-TEST row or incomplete DoRA evidence being accepted."""
    monkeypatch.chdir(tmp_path)
    dataset_path = tmp_path / "data" / "TEST.jsonl"
    dataset_path.parent.mkdir()
    dataset_path.write_text("\n", encoding="utf-8")
    rows = [{"index": index, "text_input": str(index)} for index in range(8)]

    source_rows = validated_test_source_rows(dataset_path, rows, (0, 1, 2, 7))
    assert [row["source_index"] for row in source_rows] == [0, 1, 2, 7]
    with pytest.raises(ValueError, match="TEST.jsonl"):
        validated_test_source_rows(tmp_path / "train.jsonl", rows, (0, 1, 2, 7))
    alternate = tmp_path / "other" / "TEST.jsonl"
    alternate.parent.mkdir()
    alternate.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        validated_test_source_rows(alternate, rows, (0, 1, 2, 7))
    with pytest.raises(ValueError, match="source index"):
        validated_test_source_rows(dataset_path, [{"index": 1}] + rows[1:], (0, 1, 2, 7))

    validate_assistant_cue_scaffold(
        {"input_ids": torch.tensor([[7, 31, 32]])}, _CueTokenizer()
    )
    with pytest.raises(ValueError, match="assistant cue"):
        validate_assistant_cue_scaffold(
            {"input_ids": torch.tensor([[7, 31]])}, _CueTokenizer()
        )

    token_ids = SimpleNamespace(pad_token_id=90, start_token_id=91, end_token_id=92)
    targets = boundary_target_parameters(_BoundaryTargets(), token_ids)
    assert [row["token_id"] for row in targets["latent_rows"]] == [90, 91, 92]
    assert set(targets["dora_parameters"]) == {
        "adapter.lora_A", "adapter.lora_B", "adapter.lora_magnitude_vector",
    }
    with pytest.raises(ValueError, match="latent rows"):
        boundary_target_parameters(_BoundaryTargets(include_rows=False), token_ids)
    with pytest.raises(ValueError, match="DoRA"):
        boundary_target_parameters(_BoundaryTargets(include_magnitude=False), token_ids)


def test_boundary_reenables_only_sparse_and_dora_measurement_targets():
    """Catches the inference loader leaving every boundary target frozen."""
    model = _BoundaryTargets()
    model.frozen_base = nn.Parameter(torch.zeros(1))
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    diagnostics.enable_boundary_measurement_parameters(model)

    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable == {
        "lm_head.output_delta",
        "adapter.lora_A",
        "adapter.lora_B",
        "adapter.lora_magnitude_vector",
    }


class _ReferenceLanguageModel(nn.Module):
    def forward(self, *, inputs_embeds, **_kwargs):
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(dim=1))


class _ReferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(latent_size=1)
        self.model = _ReferenceLanguageModel()
        self.embed = nn.Embedding(16, 2)
        self.lm_head = nn.Linear(2, 16, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def prepare_stage2_base_inputs_embeds(self, *, input_ids, **_kwargs):
        return self.embed(input_ids)

    def get_rope_index(self, input_ids, **_kwargs):
        positions = torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
        return positions, None


def test_independent_recurrence_builds_forced_latent_scaffold_without_controller():
    """Catches a reference recurrence that fails to append the complete forced scaffold."""
    token_ids = SimpleNamespace(pad_token_id=8, start_token_id=9, end_token_id=10)
    evidence = independent_no_cache_forced_prefix(
        _ReferenceModel(),
        {"input_ids": torch.tensor([[1]]), "attention_mask": torch.ones(1, 1, dtype=torch.long)},
        token_ids,
        (token_ids.start_token_id,),
    )

    assert evidence["sequences"].tolist() == [[1, 9, 8, 10]]
    assert len(evidence["latent_states"]) == 1


class _BoundaryRuntimeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(latent_size=1)
        self.lm_head = nn.Module()
        self.lm_head.output_delta = nn.Parameter(torch.zeros(3, 2))
        self.adapter = nn.Module()
        self.adapter.lora_A = nn.Parameter(torch.zeros(2, 2))
        self.adapter.lora_B = nn.Parameter(torch.zeros(2, 2))
        self.adapter.lora_magnitude_vector = nn.Parameter(torch.zeros(2))
        self.adapter.decoy_lora_A_suffix = nn.Parameter(torch.zeros(2))
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = object()
        self.model = nn.Module()
        self.model.gradient_checkpointing = True
        self.model._gradient_checkpointing_func = object()
        self.gradient_checkpoint_calls = []
        self.leave_language_base_frozen = False

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
        self.gradient_checkpoint_calls.append(dict(gradient_checkpointing_kwargs))
        for module in self.modules():
            if self.leave_language_base_frozen and module is self.model:
                continue
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                module._gradient_checkpointing_func = dict(gradient_checkpointing_kwargs)


class _BoundaryProcessor:
    tokenizer = _CueTokenizer()


def _configure_boundary_runtime(
    tmp_path, monkeypatch, *, fail_after_step=False, leave_language_base_frozen=False,
):
    monkeypatch.chdir(tmp_path)
    dataset_path = tmp_path / "data" / "TEST.jsonl"
    dataset_path.parent.mkdir()
    dataset_path.write_text("\n", encoding="utf-8")
    model = _BoundaryRuntimeModel()
    model.leave_language_base_frozen = leave_language_base_frozen
    if leave_language_base_frozen:
        model.model.gradient_checkpointing = False
    model.eval()
    for index, parameter in enumerate(model.parameters()):
        parameter.data.fill_(float(index + 1))
        parameter.requires_grad_(index % 2 == 0)
        parameter.grad = torch.full_like(parameter, float(index + 1) / 10)
    original_values = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    original_flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    original_gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    token_ids = SimpleNamespace(pad_token_id=90, start_token_id=91, end_token_id=92)
    rows = [{"index": index, "text_input": str(index)} for index in range(8)]
    prepared = {"input_ids": torch.tensor([[31, 32]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}
    observed_starts = []
    observed_checkpointing = []
    observed_language_checkpointing = []
    original_checkpointing = [
        (module, module.gradient_checkpointing, module._gradient_checkpointing_func)
        for module in model.modules() if hasattr(module, "gradient_checkpointing")
    ]

    monkeypatch.setattr(diagnostics, "_load_probe_runtime", lambda *_args, **_kwargs: (model, _BoundaryProcessor()))
    monkeypatch.setattr(diagnostics, "_read_jsonl", lambda _path: rows)
    monkeypatch.setattr(diagnostics, "resolve_latent_token_ids", lambda *_args: token_ids)
    monkeypatch.setattr(diagnostics, "_ordinary_token_id", lambda *_args: 7)
    monkeypatch.setattr(diagnostics, "_prepared_for_sample", lambda *_args: prepared)

    def fake_loss(active_model, *_args):
        observed_starts.append(active_model.lm_head.output_delta.detach().clone())
        observed_language_checkpointing.append(active_model.model.gradient_checkpointing)
        observed_checkpointing.append([
            (module.gradient_checkpointing, module._gradient_checkpointing_func)
            for module in active_model.modules() if hasattr(module, "gradient_checkpointing")
        ])
        loss = sum(parameter.sum() for parameter in active_model.parameters() if parameter.requires_grad)
        logits = torch.zeros(1, 100)
        return loss, {"latent_start": logits, "eos": logits, "per_position_ce": {2: 1.0}}

    class FakeController:
        def generate(self, **_kwargs):
            return SimpleNamespace(stop_reason="eos")

    monkeypatch.setattr(diagnostics, "_manual_boundary_loss", fake_loss)
    monkeypatch.setattr(diagnostics, "ContinuousLatentController", FakeController)
    if fail_after_step:
        def fail_delta_evidence(*_args, **_kwargs):
            raise RuntimeError("injected after optimizer.step")

        monkeypatch.setattr(diagnostics, "_boundary_delta_evidence", fail_delta_evidence)

    return (
        model, dataset_path, original_values, original_flags, original_gradients,
        original_checkpointing, observed_starts, observed_checkpointing,
        observed_language_checkpointing,
    )


def _assert_boundary_runtime_restored(model, values, flags, gradients, checkpointing):
    assert model.training is False
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is flags[name]
        assert torch.equal(parameter, values[name])
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, gradients[name])
    assert [
        (module, module.gradient_checkpointing, module._gradient_checkpointing_func)
        for module in model.modules() if hasattr(module, "gradient_checkpointing")
    ] == checkpointing


def test_boundary_runtime_restores_state_and_starts_each_case_from_common_values(tmp_path, monkeypatch):
    """Catches temporary boundary updates leaking into callers or later cases."""
    (
        model, dataset_path, values, flags, gradients, checkpointing,
        observed_starts, observed_checkpointing, observed_language_checkpointing,
    ) = _configure_boundary_runtime(tmp_path, monkeypatch)

    report = diagnostics.run_boundary_probe(
        model_dir="unused", base_model_dir="unused", dataset_jsonl=str(dataset_path),
        cache_dir="unused", source_indices=(0, 1, 2, 7),
    )

    first = report["cases"][0]
    assert [row["token_id"] for row in first["gradients_before_clipping"]["latent_rows"]] == [90, 91, 92]
    assert set(first["gradients_before_clipping"]["dora_parameters"]) == {
        "adapter.lora_A", "adapter.lora_B", "adapter.lora_magnitude_vector",
    }
    assert len(first["parameter_deltas"]["latent_rows"]) == 3
    assert len(report["budget_stop_observations"]) == 8
    assert len(observed_starts) == 8
    assert all(torch.equal(value, values["lm_head.output_delta"]) for value in observed_starts)
    assert model.gradient_checkpoint_calls == [{"use_reentrant": False}]
    assert all(observed_language_checkpointing)
    assert all(all(enabled for enabled, _func in state) for state in observed_checkpointing)
    assert all(
        all(func == {"use_reentrant": False} for _enabled, func in state)
        for state in observed_checkpointing
    )
    _assert_boundary_runtime_restored(model, values, flags, gradients, checkpointing)


def test_boundary_runtime_restores_state_after_exception_following_optimizer_step(tmp_path, monkeypatch):
    """Catches a post-step diagnostic failure leaking temporary boundary state."""
    (
        model, dataset_path, values, flags, gradients, checkpointing,
        _observed_starts, observed_checkpointing, _observed_language_checkpointing,
    ) = _configure_boundary_runtime(tmp_path, monkeypatch, fail_after_step=True)

    with pytest.raises(RuntimeError, match="injected after optimizer.step"):
        diagnostics.run_boundary_probe(
            model_dir="unused", base_model_dir="unused", dataset_jsonl=str(dataset_path),
            cache_dir="unused", source_indices=(0, 1, 2, 7),
        )

    assert model.gradient_checkpoint_calls == [{"use_reentrant": False}]
    assert len(observed_checkpointing) == 1
    assert all(enabled for enabled, _func in observed_checkpointing[0])
    _assert_boundary_runtime_restored(model, values, flags, gradients, checkpointing)


def test_boundary_rejects_wrapper_checkpoint_enablement_without_language_base(tmp_path, monkeypatch):
    """Catches checkpointing a wrapper while manual forwards still use its disabled base."""
    (
        model, dataset_path, values, flags, gradients, checkpointing,
        observed_starts, _observed_checkpointing, _observed_language_checkpointing,
    ) = _configure_boundary_runtime(
        tmp_path, monkeypatch, leave_language_base_frozen=True
    )

    with pytest.raises(ValueError, match="language base"):
        diagnostics.run_boundary_probe(
            model_dir="unused", base_model_dir="unused", dataset_jsonl=str(dataset_path),
            cache_dir="unused", source_indices=(0, 1, 2, 7),
        )

    assert model.gradient_checkpoint_calls == [{"use_reentrant": False}]
    assert observed_starts == []
    _assert_boundary_runtime_restored(model, values, flags, gradients, checkpointing)
