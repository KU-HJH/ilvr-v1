from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import MethodType
from types import SimpleNamespace

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

import trainer as trainer_module
from evidence_window.config import (
    DiagnosticsConfig,
    EvidenceWindowConfig,
    PreferenceConfig,
    ReferenceConfig,
    RelationConfig,
    ResidualConfig,
)
from evidence_window.engine import EvidenceWindowEngine, ModelConditionOutput
from evidence_window.layout import InterleavedLayout
from evidence_window.metrics import MetricBatch


CONFIG_DIRECTORY = Path("configs/evidence_window")
COMMON = {
    "version": 1,
    "reference": {"mode": "none"},
    "preference": {
        "beta": 0.1,
        "lambda_global": 0.0,
        "lambda_window": 0.0,
        "lambda_ew": 0.0,
    },
    "relation": {
        "lambda_jsd": 0.0,
        "lambda_magnitude": 0.0,
        "temperature": 1.0,
        "huber_delta": 1.0,
    },
    "residual": {
        "lambda_logit": 0.0,
        "metric": "huber",
        "huber_delta": 1.0,
        "vocab_chunk_size": 4096,
        "max_tokens_per_window": 0,
    },
    "loo": {"intervention": "zero_latent", "max_spans_per_example": 0},
    "diagnostics": {
        "global_saturation": False,
        "contribution_threshold": 0.0,
        "matrix_dump_interval": 0,
        "matrix_dump_dir": "evidence_window_matrices",
    },
}


def _variant(**sections):
    value = json.loads(json.dumps(COMMON))
    for section, overrides in sections.items():
        value[section].update(overrides)
    return value


EXPECTED_CONFIGS = {
    "baseline.json": (
        _variant(),
        {"policy": set(), "reference": set(), "ema": set()},
    ),
    "diagnostic_global.json": (
        _variant(
            reference={"mode": "adapter_disabled"},
            diagnostics={"global_saturation": True},
        ),
        {
            "policy": {"full", "all_null"},
            "reference": {"full", "all_null"},
            "ema": set(),
        },
    ),
    "window_preference.json": (
        _variant(
            reference={"mode": "adapter_disabled"},
            preference={"lambda_window": 1.0},
        ),
        {
            "policy": {"full", "all_null"},
            "reference": {"full", "all_null"},
            "ema": set(),
        },
    ),
    "loo_window_preference.json": (
        _variant(
            reference={"mode": "adapter_disabled"},
            preference={"lambda_ew": 1.0},
        ),
        {
            "policy": {"full", "loo"},
            "reference": {"full", "loo"},
            "ema": set(),
        },
    ),
    "relation_kd.json": (
        _variant(relation={"lambda_jsd": 1.0, "lambda_magnitude": 1.0}),
        {
            "policy": {"full", "loo"},
            "reference": set(),
            "ema": {"full", "loo"},
        },
    ),
    "logit_residual.json": (
        _variant(residual={"lambda_logit": 1.0}),
        {
            "policy": {"full", "loo"},
            "reference": set(),
            "ema": {"full", "loo"},
        },
    ),
    "loo_relation_combined.json": (
        _variant(
            reference={"mode": "adapter_disabled"},
            preference={"lambda_ew": 1.0},
            relation={"lambda_jsd": 1.0, "lambda_magnitude": 1.0},
        ),
        {
            "policy": {"full", "loo"},
            "reference": {"full", "loo"},
            "ema": {"full", "loo"},
        },
    ),
    "frozen_reference_zero2.json": (
        _variant(
            reference={"mode": "frozen_copy"},
            preference={"lambda_ew": 1.0},
            diagnostics={"global_saturation": True},
        ),
        {
            "policy": {"full", "all_null", "loo"},
            "reference": {"full", "all_null", "loo"},
            "ema": set(),
        },
    ),
}


def test_shipped_configs_are_explicit_and_activate_only_the_scoped_objectives():
    for name, (expected, conditions) in EXPECTED_CONFIGS.items():
        path = CONFIG_DIRECTORY / name
        raw = json.loads(path.read_text(encoding="utf-8"))
        config = EvidenceWindowConfig.from_json(path)

        assert raw == expected, name
        assert asdict(config) == expected, name
        for role, wanted in conditions.items():
            assert set(config.required_conditions(role)) == wanted, (name, role)

    assert EvidenceWindowConfig.from_json(
        CONFIG_DIRECTORY / "baseline.json"
    ).is_disabled


def _heterogeneous_layout() -> InterleavedLayout:
    evidence = torch.tensor(
        [
            [[1, 0, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0]],
        ],
        dtype=torch.bool,
    )
    windows = torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
        ],
        dtype=torch.bool,
    )
    causal = torch.tensor(
        [
            [[0, 1, 1], [0, 0, 1]],
            [[1, 1, 0], [0, 1, 0]],
        ],
        dtype=torch.bool,
    )
    training = torch.tensor(
        [
            [[0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0]],
        ],
        dtype=torch.bool,
    )
    return InterleavedLayout(
        evidence_token_mask=evidence,
        window_token_mask=windows,
        evidence_valid=evidence.any(dim=-1),
        window_valid=windows.any(dim=-1),
        causal_pair_mask=causal,
        training_pair_mask=training,
        native_predecessor=torch.tensor([[-1, 0, 1], [0, 1, -1]]),
    )


def test_engine_ordinal_metrics_follow_native_predecessors_across_heterogeneous_rows():
    config = EvidenceWindowConfig(
        reference=ReferenceConfig(mode="adapter_disabled"),
        preference=PreferenceConfig(beta=1.0, lambda_window=1.0, lambda_ew=1.0),
    )
    outputs = {
        ("policy", "all_null", None): [[9.0, 18.0, 27.0], [36.0, 45.0, 0.0]],
        ("policy", "loo", 0): [[10.0, 19.0, 28.0], [37.0, 47.0, 0.0]],
        ("policy", "loo", 1): [[10.0, 20.0, 26.0], [40.0, 45.0, 0.0]],
        ("reference", "full", None): [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ("reference", "all_null", None): [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ("reference", "loo", 0): [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ("reference", "loo", 1): [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    }

    def forward(role, condition, latent_values, need_hidden):
        del latent_values, need_hidden
        key = (role, condition.kind, condition.evidence_index)
        return ModelConditionOutput(torch.tensor(outputs[key]))

    result = EvidenceWindowEngine(config).compute(
        _heterogeneous_layout(),
        torch.ones((2, 2, 1)),
        ModelConditionOutput(torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 0.0]])),
        forward,
    )

    expected = {
        "preference/window_margin/0": (6.0, 2.0),
        "preference/window_margin/1": (8.0, 2.0),
        "preference/ew_margin/0": (4.0, 2.0),
        "preference/ew_margin/1": (9.0, 2.0),
    }
    for key, (numerator, count) in expected.items():
        actual_numerator, actual_count = result.metric_sums[key]
        assert actual_numerator.item() == numerator
        assert actual_count.item() == count
    assert "preference/window_margin/2" not in result.metric_sums
    assert "preference/ew_margin/2" not in result.metric_sums


def test_window_ordinal_uses_native_predecessor_after_empty_windows_are_omitted():
    evidence = torch.eye(3, dtype=torch.bool).unsqueeze(0)
    windows = torch.tensor([[[0, 0, 1]]], dtype=torch.bool)
    pairs = torch.tensor([[[False], [False], [True]]])
    layout = InterleavedLayout(
        evidence_token_mask=evidence,
        window_token_mask=windows,
        evidence_valid=torch.ones((1, 3), dtype=torch.bool),
        window_valid=torch.ones((1, 1), dtype=torch.bool),
        causal_pair_mask=torch.ones((1, 3, 1), dtype=torch.bool),
        training_pair_mask=pairs,
        native_predecessor=torch.tensor([[2]]),
    )
    config = EvidenceWindowConfig(
        reference=ReferenceConfig(mode="adapter_disabled"),
        preference=PreferenceConfig(lambda_window=1.0),
    )

    def forward(role, condition, latent_values, need_hidden):
        del latent_values, need_hidden
        if role == "policy":
            assert condition.kind == "all_null"
            return ModelConditionOutput(torch.tensor([[0.0, 0.0, 2.0]]))
        return ModelConditionOutput(torch.zeros((1, 3)))

    result = EvidenceWindowEngine(config).compute(
        layout,
        torch.ones((1, 3, 1)),
        ModelConditionOutput(torch.tensor([[0.0, 0.0, 5.0]])),
        forward,
    )

    numerator, count = result.metric_sums["preference/window_margin/2"]
    assert numerator.item() == 3.0
    assert count.item() == 1.0


def test_logging_reduces_sufficient_statistics_before_weighted_mean(
    tiny_trainer_factory,
    monkeypatch,
):
    trainer, _ = tiny_trainer_factory()
    captured = []

    def add_remote_statistics(numerator, count=None):
        if isinstance(numerator, MetricBatch):
            batch = numerator
        else:
            batch = MetricBatch(numerator, count)
        assert not batch.numerator.requires_grad
        return MetricBatch(batch.numerator + 12.0, batch.count + 3.0)

    monkeypatch.setattr(
        trainer_module, "reduce_sufficient_statistics", add_remote_statistics
    )
    monkeypatch.setattr(
        trainer_module.SFTTrainer,
        "log",
        lambda _self, logs, start_time=None: captured.append(dict(logs)),
    )

    trainer._accumulate_evidence_window_metrics(
        {"loss/ilvr": (torch.tensor(2.0, requires_grad=True), torch.tensor(1.0))},
        ordinal_families=(),
    )
    trainer.log({"loss": 99.0})

    assert captured == [{"loss": 99.0, "loss/ilvr": 3.5}]


def test_evaluation_and_training_statistics_are_logged_separately(
    tiny_trainer_factory,
    monkeypatch,
):
    trainer, _ = tiny_trainer_factory(
        evidence_window_config=EvidenceWindowConfig(
            reference=ReferenceConfig(mode="frozen_copy"),
            preference=PreferenceConfig(lambda_window=1.0),
        ),
        latent_token_ids={
            "start": 90,
            "end": 91,
            "pad": 92,
            "padding": 0,
            "special": (0, 90, 91, 92, 99),
        },
    )
    captured = []
    monkeypatch.setattr(
        trainer_module.SFTTrainer,
        "log",
        lambda _self, logs, start_time=None: captured.append(dict(logs)),
    )

    trainer._accumulate_evidence_window_metrics(
        {"loss/ilvr": (torch.tensor(2.0), torch.tensor(1.0))},
        ordinal_families=(),
        split="eval",
    )
    trainer._accumulate_evidence_window_metrics(
        {"loss/ilvr": (torch.tensor(10.0), torch.tensor(1.0))},
        ordinal_families=(),
        split="train",
    )
    trainer.log({"eval_loss": 1.0})
    trainer.log({"loss": 1.0})

    assert captured[0]["eval_loss/ilvr"] == 2.0
    assert "loss/ilvr" not in captured[0]
    assert captured[1]["loss/ilvr"] == 10.0
    assert "eval_loss/ilvr" not in captured[1]


def test_enabled_logging_uses_stable_bounded_scalar_keys(
    tiny_trainer_factory,
    legacy_latent_case,
    monkeypatch,
):
    config = EvidenceWindowConfig(
        reference=ReferenceConfig(mode="frozen_copy"),
        preference=PreferenceConfig(
            beta=0.7,
            lambda_global=1.0,
            lambda_window=1.0,
            lambda_ew=1.0,
        ),
        relation=RelationConfig(lambda_jsd=1.0, lambda_magnitude=1.0),
        residual=ResidualConfig(lambda_logit=1.0, vocab_chunk_size=31),
        diagnostics=DiagnosticsConfig(global_saturation=True),
    )
    trainer, model = tiny_trainer_factory(
        evidence_window_config=config,
        latent_token_ids={
            "start": 90,
            "end": 91,
            "pad": 92,
            "padding": 0,
            "special": (0, 90, 91, 92, 99),
        },
        sim_weight=0.0,
    )
    captured = []
    monkeypatch.setattr(
        trainer_module.SFTTrainer,
        "log",
        lambda _self, logs, start_time=None: captured.append(dict(logs)),
    )

    trainer.compute_loss(model, dict(legacy_latent_case["inputs"]))
    trainer.log({"loss": 99.0})

    required = {
        "loss/ilvr",
        "loss/global_preference",
        "loss/window_preference",
        "loss/ew_dpo",
        "loss/relation_jsd",
        "loss/relation_magnitude",
        "loss/logit_residual",
        "preference/global_margin",
        "preference/global_outer_weight",
        "preference/window_margin_mean",
        "preference/ew_margin_mean",
        "preference/window_margin/0",
        "preference/window_margin/1",
        "preference/ew_margin/0",
        "preference/ew_margin/1",
        "relation/student/top1_ratio",
        "relation/student/normalized_entropy",
        "relation/student/above_threshold_count",
        "relation/teacher/top1_ratio",
        "relation/teacher/normalized_entropy",
        "relation/teacher/above_threshold_count",
        "loo/selected_span_count",
        "residual/selected_token_count",
    }
    assert len(captured) == 1
    assert set(captured[0]) == {"loss"} | required
    assert len(captured[0]) < 30
    assert all(isinstance(value, float) for value in captured[0].values())

    trainer.log({"loss": 98.0})
    assert captured[1] == {"loss": 98.0}


def test_enabled_ce_fallback_retains_original_ilvr_loss_metric(
    tiny_trainer_factory,
    legacy_no_latent_case,
    monkeypatch,
):
    trainer, model = tiny_trainer_factory(
        evidence_window_config=EvidenceWindowConfig(
            relation=RelationConfig(lambda_magnitude=1.0)
        ),
        latent_token_ids={
            "start": 90,
            "end": 91,
            "pad": 92,
            "padding": 0,
            "special": (0, 90, 91, 92, 99),
        },
    )
    captured = []
    monkeypatch.setattr(
        trainer_module.SFTTrainer,
        "log",
        lambda _self, logs, start_time=None: captured.append(dict(logs)),
    )

    def no_latents(_self, inputs, k, p):
        del k, p
        return None, torch.zeros_like(inputs["input_ids"], dtype=torch.bool)

    trainer._teacher_build_latents = MethodType(no_latents, trainer)
    trainer.compute_loss(model, dict(legacy_no_latent_case["inputs"]))
    trainer.log({"loss": 99.0})

    assert captured[0]["loss/ilvr"] == legacy_no_latent_case[
        "recorded_ce_loss"
    ].item()


def test_baseline_does_not_add_logging_or_collectives(
    tiny_trainer_factory,
    legacy_latent_case,
    monkeypatch,
):
    trainer, model = tiny_trainer_factory(
        evidence_window_config=EvidenceWindowConfig.disabled()
    )
    captured = []
    monkeypatch.setattr(
        trainer_module,
        "reduce_sufficient_statistics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled path reduced evidence-window metrics")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        trainer_module.SFTTrainer,
        "log",
        lambda _self, logs, start_time=None: captured.append(dict(logs)),
    )

    trainer.compute_loss(model, dict(legacy_latent_case["inputs"]))
    trainer.log({"loss": 99.0})

    assert captured == [{"loss": 99.0}]


def test_matrix_dump_interval_is_once_per_rank_step_and_private(
    tiny_trainer_factory,
    legacy_latent_case,
    monkeypatch,
    tmp_path,
):
    dump_dir = tmp_path / "numeric-matrices"
    config = EvidenceWindowConfig(
        relation=RelationConfig(lambda_magnitude=1.0),
        diagnostics=DiagnosticsConfig(
            matrix_dump_interval=2,
            matrix_dump_dir=str(dump_dir),
        ),
    )
    trainer, model = tiny_trainer_factory(
        evidence_window_config=config,
        latent_token_ids={
            "start": 90,
            "end": 91,
            "pad": 92,
            "padding": 0,
            "special": (0, 90, 91, 92, 99),
        },
        sim_weight=0.0,
    )
    trainer.accelerator.process_index = 3
    trainer.state = SimpleNamespace(global_step=1)

    trainer.compute_loss(model, dict(legacy_latent_case["inputs"]))
    assert not dump_dir.exists()

    trainer.state.global_step = 2
    trainer.compute_loss(model, dict(legacy_latent_case["inputs"]))
    trainer.compute_loss(model, dict(legacy_latent_case["inputs"]))
    trainer.state.global_step = 4
    trainer.compute_loss(model, dict(legacy_latent_case["inputs"]))

    path = dump_dir / "rank-00003.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["step"] for record in records] == [2, 4]
    assert [record["observation_id"] for record in records] == ["r3-s2-o0", "r3-s4-o1"]
    for record in records:
        assert set(record) == {
            "observation_id",
            "rank",
            "step",
            "relation",
            "pair_mask",
        }
        assert record["rank"] == 3
        assert isinstance(record["relation"], list)
        assert isinstance(record["pair_mask"], list)
        assert not {
            "input_ids",
            "labels",
            "text",
            "image_path",
            "sample_id",
        } & set(record)


def test_matrix_dump_only_configuration_requests_and_writes_relations(
    tiny_trainer_factory,
    legacy_latent_case,
    tmp_path,
):
    dump_dir = tmp_path / "dump-only"
    config = EvidenceWindowConfig(
        diagnostics=DiagnosticsConfig(
            matrix_dump_interval=1,
            matrix_dump_dir=str(dump_dir),
        )
    )
    assert config.required_conditions("policy") == frozenset({"full", "loo"})
    assert config.required_conditions("ema") == frozenset({"full", "loo"})
    trainer, model = tiny_trainer_factory(
        evidence_window_config=config,
        latent_token_ids={
            "start": 90,
            "end": 91,
            "pad": 92,
            "padding": 0,
            "special": (0, 90, 91, 92, 99),
        },
        sim_weight=0.0,
    )
    trainer.state = SimpleNamespace(global_step=1)

    trainer.compute_loss(model, dict(legacy_latent_case["inputs"]))

    record = json.loads((dump_dir / "rank-00000.jsonl").read_text())
    assert record["step"] == 1
    assert record["relation"]
    assert record["pair_mask"]


def _heterogeneous_logging_worker(rank: int, init_file: str) -> None:
    distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        trainer = object.__new__(trainer_module.CustomTrainerStage1)
        trainer.model = torch.nn.Linear(1, 1, bias=False)
        trainer.evidence_window_config = EvidenceWindowConfig(
            reference=ReferenceConfig(mode="adapter_disabled"),
            preference=PreferenceConfig(lambda_ew=1.0),
        )
        trainer._ew_pending_metrics = {"train": {}, "eval": {}}
        if rank == 0:
            trainer._accumulate_evidence_window_metrics(
                {
                    "loss/ilvr": (torch.tensor(2.0), torch.tensor(1.0)),
                    "preference/ew_margin/2": (
                        torch.tensor(7.0),
                        torch.tensor(1.0),
                    ),
                },
                ordinal_families=("preference/ew_margin",),
                split="train",
            )

        reduced = trainer._reduce_evidence_window_metrics("train")
        assert reduced["loss/ilvr"] == 2.0
        assert reduced["preference/ew_margin/2"] == 7.0
        assert "preference/ew_margin/0" not in reduced
        assert "preference/ew_margin/1" not in reduced
    finally:
        distributed.destroy_process_group()


def test_two_ranks_align_missing_metrics_and_sparse_ordinal_keys(tmp_path):
    init_file = tmp_path / "logging-gloo-init"
    multiprocessing.spawn(
        _heterogeneous_logging_worker,
        args=(str(init_file),),
        nprocs=2,
        join=True,
    )
