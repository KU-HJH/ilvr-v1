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
import evidence_window.engine as engine_module
from evidence_window.config import (
    DiagnosticsConfig,
    EvidenceWindowConfig,
    PreferenceConfig,
    ReferenceConfig,
    RelationConfig,
    ResidualConfig,
)
from evidence_window.engine import (
    EvidenceWindowEngine,
    EvidenceWindowResult,
    ModelConditionOutput,
)
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
    "frozen_reference_zero1.json": (
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


def test_condition_execution_plan_is_canonical_across_local_span_sets():
    builder = getattr(engine_module, "build_condition_execution_plan", None)
    assert callable(builder), "canonical condition planner is missing"
    config = EvidenceWindowConfig(
        reference=ReferenceConfig(mode="adapter_disabled"),
        preference=PreferenceConfig(
            lambda_global=1.0,
            lambda_ew=1.0,
        ),
        relation=RelationConfig(lambda_magnitude=1.0),
        residual=ResidualConfig(lambda_logit=1.0),
    )

    plan = builder(config, (0, 2))

    assert [
        (request.role, request.condition.kind, request.condition.evidence_index, request.need_hidden)
        for request in plan.requests
    ] == [
        ("ema", "full", None, True),
        ("policy", "loo", 0, True),
        ("ema", "loo", 0, True),
        ("policy", "loo", 2, True),
        ("ema", "loo", 2, True),
        ("policy", "all_null", None, False),
        ("reference", "full", None, False),
        ("reference", "all_null", None, False),
        ("reference", "loo", 0, False),
        ("reference", "loo", 2, False),
    ]
    assert plan.evidence_indices == (0, 2)


def test_evidence_layout_padding_preserves_local_masks_and_adds_dummy_ordinals():
    pad_layout = getattr(engine_module, "pad_evidence_layout", None)
    assert callable(pad_layout), "evidence-layout padding is missing"
    layout = _heterogeneous_layout()

    padded = pad_layout(layout, 4)

    assert padded.evidence_token_mask.shape == (2, 4, 3)
    assert padded.causal_pair_mask.shape == (2, 4, 3)
    assert torch.equal(padded.evidence_token_mask[:, :2], layout.evidence_token_mask)
    assert torch.equal(padded.evidence_valid[:, :2], layout.evidence_valid)
    assert torch.equal(padded.causal_pair_mask[:, :2], layout.causal_pair_mask)
    assert not padded.evidence_token_mask[:, 2:].any()
    assert not padded.evidence_valid[:, 2:].any()
    assert not padded.causal_pair_mask[:, 2:].any()
    assert not padded.training_pair_mask[:, 2:].any()


def test_trainer_synchronizes_global_evidence_ordinals_before_condition_plan(
    monkeypatch,
):
    trainer = object.__new__(trainer_module.CustomTrainerStage1)
    trainer.evidence_window_config = EvidenceWindowConfig(
        reference=ReferenceConfig(mode="adapter_disabled"),
        preference=PreferenceConfig(lambda_ew=1.0),
    )
    trainer._ew_engine = EvidenceWindowEngine(trainer.evidence_window_config)
    reductions = []

    monkeypatch.setattr(distributed, "is_available", lambda: True)
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)

    def fake_all_reduce(value, op):
        reductions.append(op)
        if len(reductions) == 1:
            value.fill_(4)
        else:
            value.copy_(torch.tensor([1, 1, 0, 1], device=value.device))

    monkeypatch.setattr(distributed, "all_reduce", fake_all_reduce)

    padded, plan = trainer._synchronize_evidence_window_plan(
        _heterogeneous_layout()
    )

    assert padded.evidence_token_mask.shape[1] == 4
    assert plan.evidence_indices == (0, 1, 3)
    assert reductions == [distributed.ReduceOp.MAX, distributed.ReduceOp.MAX]


def test_residual_dummy_conditions_request_scores_without_hidden_states():
    config = EvidenceWindowConfig(residual=ResidualConfig(lambda_logit=1.0))
    engine = EvidenceWindowEngine(config)
    layout = _distributed_schedule_layout(rank=1)
    layout = engine_module.pad_evidence_layout(layout, 1)
    plan = engine_module.build_condition_execution_plan(config, (0,))
    requests = []

    def forward(role, condition, latent_values, need_hidden):
        del latent_values
        requests.append((role, condition.kind, condition.evidence_index, need_hidden))
        return ModelConditionOutput(torch.zeros((1, 3)), None)

    result = engine.compute(
        layout,
        torch.empty((1, 0, 1)),
        ModelConditionOutput(torch.zeros((1, 3), requires_grad=True), None),
        forward,
        lm_head_weight=lambda role: torch.ones((3, 1)),
        execution_plan=plan,
    )

    assert requests == [
        ("ema", "full", None, False),
        ("policy", "loo", 0, False),
        ("ema", "loo", 0, False),
    ]
    assert result.metric_sums["loss/logit_residual"][1].item() == 0


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
        "relation/student/entropy_valid_count",
        "relation/student/above_threshold_count",
        "relation/teacher/top1_ratio",
        "relation/teacher/normalized_entropy",
        "relation/teacher/entropy_valid_count",
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

    assert captured == [
        {
            "loss": 99.0,
            "loss/ilvr": legacy_no_latent_case["recorded_ce_loss"].item(),
        }
    ]


def test_relation_metrics_expose_entropy_valid_window_count():
    evidence = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool)
    windows = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool)
    causal = torch.tensor([[[True, True], [False, True]]])
    layout = InterleavedLayout(
        evidence_token_mask=evidence,
        window_token_mask=windows,
        evidence_valid=torch.ones((1, 2), dtype=torch.bool),
        window_valid=torch.ones((1, 2), dtype=torch.bool),
        causal_pair_mask=causal,
        training_pair_mask=torch.tensor([[[True, False], [False, True]]]),
        native_predecessor=torch.tensor([[0, 1]]),
    )
    outputs = {
        ("policy", "loo", 0): [[1.0, 2.0]],
        ("policy", "loo", 1): [[2.0, 3.0]],
        ("ema", "full", None): [[3.0, 5.0]],
        ("ema", "loo", 0): [[2.0, 3.0]],
        ("ema", "loo", 1): [[3.0, 4.0]],
    }

    def forward(role, condition, latent_values, need_hidden):
        del latent_values, need_hidden
        key = (role, condition.kind, condition.evidence_index)
        return ModelConditionOutput(torch.tensor(outputs[key]))

    result = EvidenceWindowEngine(
        EvidenceWindowConfig(relation=RelationConfig(lambda_magnitude=1.0))
    ).compute(
        layout,
        torch.ones((1, 2, 1)),
        ModelConditionOutput(torch.tensor([[2.0, 4.0]])),
        forward,
    )

    for role in ("student", "teacher"):
        numerator, count = result.metric_sums[
            f"relation/{role}/entropy_valid_count"
        ]
        assert numerator.item() == 1.0
        assert count.item() == 1.0


def test_startup_log_states_complete_objective_and_diagnostic_switches(
    tiny_trainer_factory,
    caplog,
):
    caplog.set_level("INFO")
    tiny_trainer_factory(
        evidence_window_config=EvidenceWindowConfig(
            reference=ReferenceConfig(mode="frozen_copy"),
            preference=PreferenceConfig(
                beta=0.3,
                lambda_global=0.0,
                lambda_window=2.0,
                lambda_ew=0.5,
            ),
            relation=RelationConfig(
                lambda_jsd=0.0,
                lambda_magnitude=3.0,
                temperature=0.7,
            ),
            diagnostics=DiagnosticsConfig(
                global_saturation=True,
                matrix_dump_interval=7,
            ),
        ),
        latent_token_ids={
            "start": 90,
            "end": 91,
            "pad": 92,
            "padding": 0,
            "special": (0, 90, 91, 92, 99),
        },
    )

    objective_log = next(
        record.getMessage()
        for record in caplog.records
        if "Evidence-window objective:" in record.getMessage()
    )
    for expected in (
        "ce_weight=1.3",
        "sim_weight=0.7",
        "beta=0.3",
        "lambda_global=0.0",
        "lambda_window=2.0",
        "lambda_ew=0.5",
        "lambda_jsd=0.0",
        "lambda_magnitude=3.0",
        "lambda_logit=0.0",
        "global_saturation=true",
        "matrix_dump_interval=7",
    ):
        assert expected in objective_log


def test_startup_log_distinguishes_explicit_zero1_and_zero2_replication(
    tiny_trainer_factory,
    caplog,
    monkeypatch,
):
    expected_partitioning = {
        1: "optimizer states only",
        2: "optimizer states and policy gradients",
    }
    for zero_stage, partitioning in expected_partitioning.items():
        caplog.clear()
        caplog.set_level("INFO")
        monkeypatch.setattr(
            trainer_module.CustomTrainerStage1,
            "_deepspeed_zero_stage",
            lambda _self, stage=zero_stage: stage,
        )
        tiny_trainer_factory(
            evidence_window_config=EvidenceWindowConfig(
                reference=ReferenceConfig(mode="frozen_copy"),
                preference=PreferenceConfig(lambda_ew=1.0),
            ),
            latent_token_ids={
                "start": 90,
                "end": 91,
                "pad": 92,
                "padding": 0,
                "special": (0, 90, 91, 92, 99),
            },
        )

        replication_log = next(
            (
                record.getMessage()
                for record in caplog.records
                if "frozen reference is rank-local replicated" in record.getMessage()
            ),
            None,
        )
        assert replication_log is not None
        assert f"ZeRO-{zero_stage}" in replication_log
        assert partitioning in replication_log
        assert "configured stage is preserved" in replication_log


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


def _objective_gradient_worker(rank: int, init_file: str) -> None:
    distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        trainer = object.__new__(trainer_module.CustomTrainerStage1)
        trainer.evidence_window_config = EvidenceWindowConfig(
            reference=ReferenceConfig(mode="adapter_disabled"),
            preference=PreferenceConfig(lambda_ew=0.5),
            relation=RelationConfig(lambda_magnitude=2.0),
        )
        parameter = torch.tensor(0.25, requires_grad=True)
        auxiliary = None
        if rank == 0:
            targets = torch.tensor([1.0, 2.0, 4.0])
            local_mean = (parameter - targets).square().mean()
            empty_pairs = torch.zeros((1, 0, 0), dtype=torch.bool)
            auxiliary = EvidenceWindowResult(
                total_loss=0.5 * local_mean,
                losses={"ew_dpo": local_mean},
                metric_sums={
                    "loss/ew_dpo": (
                        local_mean.detach() * 3.0,
                        torch.tensor(3.0),
                    )
                },
                ew_margins=None,
                relations={},
                preference_pair_mask=empty_pairs,
                relation_pair_mask=empty_pairs,
            )

        weighted = trainer._compose_globally_weighted_auxiliary_loss(
            parameter * 0.0,
            auxiliary,
        )
        weighted.backward()
        averaged_gradient = parameter.grad.detach().clone()
        distributed.all_reduce(averaged_gradient, op=distributed.ReduceOp.SUM)
        averaged_gradient /= 2.0

        reference_parameter = torch.tensor(0.25, requires_grad=True)
        centralized = 0.5 * (
            reference_parameter - torch.tensor([1.0, 2.0, 4.0])
        ).square().mean()
        centralized.backward()
        assert torch.allclose(
            averaged_gradient,
            reference_parameter.grad,
            atol=1e-7,
            rtol=1e-7,
        )
    finally:
        distributed.destroy_process_group()


def test_two_rank_auxiliary_gradient_matches_global_observation_mean(tmp_path):
    init_file = tmp_path / "objective-gloo-init"
    multiprocessing.spawn(
        _objective_gradient_worker,
        args=(str(init_file),),
        nprocs=2,
        join=True,
    )


def _distributed_schedule_layout(rank: int) -> InterleavedLayout:
    windows = torch.tensor([[[0, 1, 0], [0, 0, 1]]], dtype=torch.bool)
    if rank == 0:
        evidence = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.bool)
        causal = torch.tensor([[[1, 1], [0, 1]]], dtype=torch.bool)
        training = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool)
        valid = torch.ones((1, 2), dtype=torch.bool)
        predecessor = torch.tensor([[0, 1]])
    else:
        evidence = torch.zeros((1, 0, 3), dtype=torch.bool)
        causal = torch.zeros((1, 0, 2), dtype=torch.bool)
        training = torch.zeros_like(causal)
        valid = torch.zeros((1, 0), dtype=torch.bool)
        predecessor = torch.full((1, 2), -1, dtype=torch.long)
    return InterleavedLayout(
        evidence_token_mask=evidence,
        window_token_mask=windows,
        evidence_valid=valid,
        window_valid=torch.ones((1, 2), dtype=torch.bool),
        causal_pair_mask=causal,
        training_pair_mask=training,
        native_predecessor=predecessor,
    )


def _condition_schedule_worker(rank: int, init_file: str) -> None:
    distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        config = EvidenceWindowConfig(
            reference=ReferenceConfig(mode="adapter_disabled"),
            preference=PreferenceConfig(lambda_global=0.2, lambda_ew=0.5),
            relation=RelationConfig(lambda_magnitude=0.7),
        )
        trainer = object.__new__(trainer_module.CustomTrainerStage1)
        trainer.evidence_window_config = config
        trainer._ew_engine = EvidenceWindowEngine(config)
        layout, plan = trainer._synchronize_evidence_window_plan(
            _distributed_schedule_layout(rank)
        )
        parameter = torch.tensor(0.25, requires_grad=True)
        base = torch.tensor([[1.0, 2.0, 3.0]])
        ledger = []

        def forward(role, condition, latent_values, need_hidden):
            assert latent_values.shape[1] == (2 if rank == 0 else 0)
            ledger.append(
                (role, condition.kind, condition.evidence_index, need_hidden)
            )
            marker = torch.tensor(float(len(ledger)))
            distributed.all_reduce(marker, op=distributed.ReduceOp.SUM)
            assert marker.item() == 2.0 * len(ledger)
            if role == "policy":
                if condition.kind == "all_null":
                    scores = 0.5 * parameter * base
                else:
                    assert condition.evidence_index is not None
                    scores = parameter * base - (
                        0.1 * (condition.evidence_index + 1) * parameter * base
                    )
            else:
                scores = torch.zeros_like(base)
            return ModelConditionOutput(scores, None)

        latent_values = (
            torch.ones((1, 2, 1))
            if rank == 0
            else torch.empty((1, 0, 1))
        )
        auxiliary = trainer._ew_engine.compute(
            layout,
            latent_values,
            ModelConditionOutput(parameter * base),
            forward,
            execution_plan=plan,
        )
        expected_ledger = [
            (
                request.role,
                request.condition.kind,
                request.condition.evidence_index,
                request.need_hidden,
            )
            for request in plan.requests
        ]
        gathered_ledgers = [None, None]
        distributed.all_gather_object(gathered_ledgers, ledger)
        assert ledger == expected_ledger
        assert gathered_ledgers == [expected_ledger, expected_ledger]

        reference_gradient = torch.zeros(())
        if rank == 0:
            reference_gradient = torch.autograd.grad(
                auxiliary.total_loss,
                parameter,
                retain_graph=True,
            )[0].detach()
        distributed.broadcast(reference_gradient, src=0)
        weighted = trainer._compose_globally_weighted_auxiliary_loss(
            parameter * 0.0,
            auxiliary,
        )
        weighted.backward()
        averaged_gradient = parameter.grad.detach().clone()
        distributed.all_reduce(averaged_gradient, op=distributed.ReduceOp.SUM)
        averaged_gradient /= 2.0
        assert torch.allclose(
            averaged_gradient,
            reference_gradient,
            atol=1e-7,
            rtol=1e-7,
        )
        if rank == 1:
            assert all(
                count.item() == 0
                for key, (_, count) in auxiliary.metric_sums.items()
                if key.startswith("loss/")
            )
    finally:
        distributed.destroy_process_group()


def test_two_rank_condition_schedule_handles_different_j_and_no_latent_rank(
    tmp_path,
):
    init_file = tmp_path / "condition-schedule-gloo-init"
    multiprocessing.spawn(
        _condition_schedule_worker,
        args=(str(init_file),),
        nprocs=2,
        join=True,
    )
