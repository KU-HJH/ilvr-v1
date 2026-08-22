from dataclasses import replace
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import trainer_stage2 as trainer_stage2_module
from evidence_window.artifacts import ReferenceBatchTargets
from evidence_window.stage2_config import (
    Stage2EvidenceWindowConfig,
    Stage2ReferenceConfig,
    Stage2RelationConfig,
    Stage2ResidualConfig,
    Stage2ScheduleConfig,
)
from evidence_window.stage2_effects import (
    Stage2ConditionOutput,
    Stage2EvidenceEffects,
)
from evidence_window.stage2_engine import (
    Stage2EvidenceWindowEngine,
    Stage2EvidenceWindowResult,
)
from evidence_window.stage2_layout import Stage2InterleavedLayout
from evidence_window.stage2_reference import OnlineFrozenStage2Reference
from trainer_stage2 import CustomTrainerStage2


TOKEN_IDS = {
    "start": 90,
    "end": 91,
    "pad": 92,
    "padding": 0,
    "special": (0, 90, 91, 92, 99),
}


class _TinyTrainerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(stage="stage1", hidden_size=2)
        self.embedding = torch.nn.Embedding(100, 2)
        self.head = torch.nn.Linear(2, 100, bias=False)
        self.calls = []

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        attention_block_mask=None,
        attention_backend_override=None,
        output_hidden_states=False,
        return_dict=True,
        use_cache=None,
    ):
        del return_dict
        self.calls.append(
            {
                "hidden": output_hidden_states,
                "backend": attention_backend_override,
                "block": attention_block_mask,
                "use_cache": use_cache,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=(hidden,))


class _RecordingReferenceProvider:
    def __init__(self, *, fail_when_hidden_required: bool = False) -> None:
        self.calls = []
        self.fail_when_hidden_required = fail_when_hidden_required

    def get_batch(self, sample_indices, layout, *, require_hidden_residuals=False):
        self.calls.append(require_hidden_residuals)
        if require_hidden_residuals and self.fail_when_hidden_required:
            raise ValueError("hidden residual missing for sample 17")
        return ReferenceBatchTargets(
            sample_indices=tuple(sample_indices.tolist()),
            contributions=torch.zeros_like(layout.causal_pair_mask, dtype=torch.float32),
            pair_mask=layout.causal_pair_mask.clone(),
            hidden_residuals=None,
        )


def _trainer_batch():
    return {
        "input_ids": torch.tensor([[90, 92, 91, 5, 6]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, 5, 6]]),
        "ew_sample_indices": torch.tensor([17]),
        "ew_plan_types": (("latent", "text"),),
    }


def _no_edge_trainer_batch():
    return {
        "input_ids": torch.tensor([[5, 6]]),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "labels": torch.tensor([[5, 6]]),
        "ew_sample_indices": torch.tensor([17]),
        "ew_plan_types": (("text",),),
    }


def _make_trainer(monkeypatch, tmp_path, config, provider):
    def fake_sft_init(instance, *args, **kwargs):
        del args
        instance.model = kwargs["model"]
        instance.args = SimpleNamespace(output_dir=str(tmp_path))
        instance.state = SimpleNamespace(global_step=5, max_steps=10)
        instance.accelerator = SimpleNamespace(process_index=0)

    monkeypatch.setattr(trainer_stage2_module.SFTTrainer, "__init__", fake_sft_init)
    model = _TinyTrainerModel()
    trainer = CustomTrainerStage2(
        model=model,
        evidence_window_config=config,
        latent_token_ids=TOKEN_IDS,
        reference_provider=provider,
    )
    return trainer, model


def _one_edge_layout() -> Stage2InterleavedLayout:
    return Stage2InterleavedLayout(
        evidence_token_mask=torch.tensor([[[1, 1, 1, 0, 0]]], dtype=torch.bool),
        window_token_mask=torch.tensor([[[0, 0, 0, 1, 1]]], dtype=torch.bool),
        evidence_valid=torch.tensor([[True]]),
        window_valid=torch.tensor([[True]]),
        causal_pair_mask=torch.tensor([[[True]]]),
    )


def _reference(
    *, hidden_residuals: torch.Tensor | None
) -> ReferenceBatchTargets:
    return ReferenceBatchTargets(
        sample_indices=(17,),
        contributions=torch.zeros((1, 1, 1)),
        pair_mask=torch.ones((1, 1, 1), dtype=torch.bool),
        hidden_residuals=hidden_residuals,
    )


def _residual_config(**changes) -> Stage2EvidenceWindowConfig:
    base = Stage2EvidenceWindowConfig(
        reference=Stage2ReferenceConfig(
            mode="precomputed", artifact="/fixed/reference"
        ),
        residual=Stage2ResidualConfig(
            enabled=True, lambda_res=2.0, huber_delta=1.0
        ),
    )
    return replace(base, **changes)


def test_pooled_hidden_effect_is_masked_mean_full_minus_loo():
    """A padded row or reversed subtraction would corrupt every residual edge."""
    from evidence_window.stage2_effects import pooled_hidden_effect

    full = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [20.0, 20.0]]])
    loo = torch.tensor([[[0.0, 1.0], [1.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, True, False]])

    torch.testing.assert_close(
        pooled_hidden_effect(full, loo, mask), torch.tensor([[1.5, 2.0]])
    )


def test_hidden_residual_huber_averages_width_then_valid_edges_in_fp32():
    """Invalid padded edges and hidden width must not change the objective scale."""
    from evidence_window.losses import stage2_hidden_residual_loss

    student = torch.tensor(
        [[[[1.0, 2.0]], [[100.0, 100.0]]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    teacher = torch.tensor([[[[0.0, 2.0]], [[0.0, 0.0]]]])
    mask = torch.tensor([[[True], [False]]])

    loss = stage2_hidden_residual_loss(student, teacher, mask, 1.0)

    assert loss.dtype is torch.float32
    assert loss.item() == pytest.approx(0.25)


def test_hidden_residual_loss_trains_full_and_loo_policy_but_stops_reference():
    """Detaching either policy branch or updating the fixed target breaks KD semantics."""
    from evidence_window.losses import stage2_hidden_residual_loss

    policy_full = torch.tensor([[[[2.0, 4.0]]]], requires_grad=True)
    policy_loo = torch.tensor([[[[0.0, 1.0]]]], requires_grad=True)
    reference_full = torch.tensor([[[[1.0, 2.0]]]], requires_grad=True)
    reference_loo = torch.tensor([[[[0.5, 1.0]]]], requires_grad=True)
    mask = torch.ones((1, 1, 1), dtype=torch.bool)

    loss = stage2_hidden_residual_loss(
        policy_full - policy_loo,
        reference_full - reference_loo,
        mask,
        1.0,
    )
    loss.backward()

    assert policy_full.grad is not None and torch.count_nonzero(policy_full.grad) == 2
    assert policy_loo.grad is not None and torch.count_nonzero(policy_loo.grad) == 2
    assert reference_full.grad is None
    assert reference_loo.grad is None


def test_hidden_residual_loss_rejects_mismatched_reference_width():
    """Same-architecture KD must fail instead of silently projecting hidden widths."""
    from evidence_window.losses import stage2_hidden_residual_loss

    with pytest.raises(ValueError, match="same shape|hidden width"):
        stage2_hidden_residual_loss(
            torch.zeros((1, 1, 1, 3)),
            torch.zeros((1, 1, 1, 2)),
            torch.ones((1, 1, 1), dtype=torch.bool),
            1.0,
        )


def test_residual_engine_uses_own_schedule_predictor_effect_and_edge_metrics():
    """Residual KD must use predictor rows and its own scheduled valid-edge mean."""
    config = _residual_config(
        schedule=Stage2ScheduleConfig(
            residual_warmup_ratio=0.2,
            residual_ramp_ratio=0.2,
        )
    )
    engine = Stage2EvidenceWindowEngine(config)
    # Targets are at rows 3 and 4, so their causal predictors are rows 2 and 3.
    policy_full = torch.tensor(
        [[[50.0, 50.0], [40.0, 40.0], [2.0, 4.0], [4.0, 8.0], [99.0, 99.0]]],
        requires_grad=True,
    )
    policy_loo = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [1.0, 4.0], [0.0, 0.0]]],
        requires_grad=True,
    )
    calls: list[tuple[torch.Tensor | None, bool]] = []

    def policy_forward(block_mask, need_hidden):
        calls.append((block_mask, need_hidden))
        hidden = policy_full if block_mask is None else policy_loo
        return Stage2ConditionOutput(torch.zeros((1, 5)), hidden if need_hidden else None)

    reference_hidden = torch.tensor([[[[0.5, 1.0]]]], requires_grad=True)
    targets = _reference(hidden_residuals=reference_hidden)
    before = engine.compute(
        _one_edge_layout(), policy_forward, progress=0.2, reference_targets=targets
    )
    assert before.effects.forward_count == 0
    assert calls == []

    result = engine.compute(
        _one_edge_layout(), policy_forward, progress=0.3, reference_targets=targets
    )

    assert [need_hidden for _, need_hidden in calls] == [True, True]
    assert result.effects.hidden_residuals is not None
    # mean([2,4],[4,8]) - mean([1,2],[1,4]) = [2,3]
    torch.testing.assert_close(
        result.effects.hidden_residuals[0, 0, 0], torch.tensor([2.0, 3.0])
    )
    # Huber([2,3] - [.5,1]) = mean([1.0,1.5]) = 1.25; ramp weight is 1.
    assert result.losses["residual"].item() == pytest.approx(1.25)
    assert result.auxiliary_loss.item() == pytest.approx(1.25)
    assert result.metric_sums["loss/residual"][1].item() == 1.0
    assert result.metric_sums["residual/valid_edge_count"][0].item() == 1.0
    assert result.metric_sums["residual/policy_effect_norm"][0].item() == pytest.approx(
        torch.tensor([2.0, 3.0]).norm().item()
    )
    assert result.metric_sums["schedule/residual_weight"][0].item() == pytest.approx(
        1.0
    )

    result.auxiliary_loss.backward()
    assert policy_full.grad is not None
    assert policy_full.grad[0, [2, 3]].abs().sum() > 0
    assert policy_full.grad[0, [0, 1, 4]].abs().sum() == 0
    assert policy_loo.grad is not None
    assert policy_loo.grad[0, [2, 3]].abs().sum() > 0
    assert policy_loo.grad[0, [0, 1, 4]].abs().sum() == 0
    assert reference_hidden.grad is None


def test_engine_preserves_fp32_reference_precision_for_residual_huber():
    """Casting artifact targets to BF16 before FP32 Huber changes the fixed target."""
    from evidence_window.losses import stage2_hidden_residual_loss

    engine = Stage2EvidenceWindowEngine(_residual_config())
    full_hidden = torch.ones((1, 5, 2), dtype=torch.bfloat16, requires_grad=True)
    loo_hidden = torch.zeros((1, 5, 2), dtype=torch.bfloat16, requires_grad=True)

    def policy_forward(block_mask, need_hidden):
        hidden = full_hidden if block_mask is None else loo_hidden
        return Stage2ConditionOutput(
            torch.zeros((1, 5), dtype=torch.bfloat16),
            hidden if need_hidden else None,
        )

    reference_hidden = torch.tensor([[[[0.12345, 0.12345]]]], dtype=torch.float32)
    result = engine.compute(
        _one_edge_layout(),
        policy_forward,
        progress=1.0,
        reference_targets=_reference(hidden_residuals=reference_hidden),
    )
    expected = stage2_hidden_residual_loss(
        result.effects.hidden_residuals,
        reference_hidden,
        result.effects.pair_mask,
        1.0,
    )

    torch.testing.assert_close(result.losses["residual"], expected, rtol=0, atol=0)


def test_inactive_residual_does_not_request_hidden_during_relation_training():
    """Configured residual warmup must not increase memory for relation-only steps."""
    config = _residual_config(
        relation=Stage2RelationConfig(enabled=True, lambda_rel=1.0),
        schedule=Stage2ScheduleConfig(residual_warmup_ratio=0.8),
    )
    engine = Stage2EvidenceWindowEngine(config)
    calls = []

    def policy_forward(block_mask, need_hidden):
        calls.append((block_mask, need_hidden))
        assert need_hidden is False
        value = 0.0 if block_mask is None else -1.0
        return Stage2ConditionOutput(torch.full((1, 5), value), None)

    result = engine.compute(
        _one_edge_layout(),
        policy_forward,
        progress=0.5,
        reference_targets=_reference(hidden_residuals=None),
    )

    assert set(result.losses) == {"relation"}
    assert len(calls) == 2


def test_missing_hidden_reference_fails_before_policy_replay():
    """An incomplete precomputed artifact must fail before policy collectives begin."""
    engine = Stage2EvidenceWindowEngine(_residual_config())
    calls = 0

    def forbidden_forward(_block_mask, _need_hidden):
        nonlocal calls
        calls += 1
        raise AssertionError("policy replay must not start")

    with pytest.raises(ValueError, match="hidden residual.*required"):
        engine.compute(
            _one_edge_layout(),
            forbidden_forward,
            progress=1.0,
            reference_targets=_reference(hidden_residuals=None),
        )

    assert calls == 0


def test_trainer_requests_precomputed_hidden_only_when_residual_schedule_is_active(
    monkeypatch, tmp_path
):
    """Residual warmup must not demand artifacts or hidden activations for relation."""
    config = _residual_config(
        relation=Stage2RelationConfig(enabled=True, lambda_rel=1.0),
        schedule=Stage2ScheduleConfig(residual_warmup_ratio=0.8),
    )
    provider = _RecordingReferenceProvider()
    trainer, model = _make_trainer(monkeypatch, tmp_path, config, provider)

    loss = trainer.compute_loss(model, _trainer_batch())

    assert torch.isfinite(loss)
    assert provider.calls == [False]
    assert [call["hidden"] for call in model.calls] == [False, False, False]


def test_active_precomputed_residual_failure_precedes_policy_auxiliary_replay(
    monkeypatch, tmp_path
):
    """All ranks must discover a missing hidden target before policy SDPA replay."""
    provider = _RecordingReferenceProvider(fail_when_hidden_required=True)
    trainer, model = _make_trainer(
        monkeypatch,
        tmp_path,
        _residual_config(),
        provider,
    )

    with pytest.raises(ValueError, match="reference target lookup.*hidden residual"):
        trainer.compute_loss(model, _trainer_batch())

    assert provider.calls == [True]
    assert len(model.calls) == 1


def test_online_reference_requests_hidden_inside_no_grad():
    """Online residual targets must not retain a second model's activation graph."""
    class Accelerator:
        @staticmethod
        def prepare_model(model, evaluation_mode=False):
            assert evaluation_mode is True
            return model

    model = _TinyTrainerModel()
    reference = OnlineFrozenStage2Reference.create(
        lambda: model,
        Accelerator(),
        2,
        policy_checkpoint="same-stage1",
        reference_checkpoint="same-stage1",
    )

    targets = reference.get_batch(
        [17],
        _one_edge_layout(),
        model_inputs={
            "input_ids": torch.tensor([[90, 92, 91, 5, 6]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
        },
        labels=torch.tensor([[-100, -100, -100, 5, 6]]),
        contribution_reduction="mean",
        max_evidence_edges_per_window=0,
        require_hidden_residuals=True,
    )

    assert targets.hidden_residuals is not None
    assert targets.hidden_residuals.requires_grad is False
    assert [call["hidden"] for call in model.calls] == [True, True]
    assert all(call["grad_enabled"] is False for call in model.calls)


def test_online_residual_with_globally_no_edges_returns_finite_ce_without_aux_replay(
    monkeypatch, tmp_path
):
    """No-edge batches need neither policy nor reference hidden residual replay."""
    def fake_sft_init(instance, *args, **kwargs):
        del args
        instance.model = kwargs["model"]
        instance.args = SimpleNamespace(output_dir=str(tmp_path))
        instance.state = SimpleNamespace(global_step=1, max_steps=1)
        instance.accelerator = SimpleNamespace(
            process_index=0,
            prepare_model=lambda model, evaluation_mode=False: model,
        )

    monkeypatch.setattr(trainer_stage2_module.SFTTrainer, "__init__", fake_sft_init)
    policy = _TinyTrainerModel()
    reference = _TinyTrainerModel()
    config = Stage2EvidenceWindowConfig(
        reference=Stage2ReferenceConfig(
            mode="online_frozen", checkpoint="same-stage1"
        ),
        residual=Stage2ResidualConfig(enabled=True, lambda_res=1.0),
    )
    trainer = CustomTrainerStage2(
        model=policy,
        evidence_window_config=config,
        latent_token_ids=TOKEN_IDS,
        reference_model_factory=lambda: reference,
        deepspeed_zero_stage=2,
        policy_initialization_checkpoint="same-stage1",
    )

    loss = trainer.compute_loss(policy, _no_edge_trainer_batch())

    assert torch.isfinite(loss)
    assert len(policy.calls) == 1
    assert reference.calls == []


def _asymmetric_hidden_target_worker(rank, init_file, result_dir):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )

    class Provider:
        @staticmethod
        def get_batch(sample_indices, layout, *, require_hidden_residuals=False):
            assert require_hidden_residuals is True
            width = 3 if rank == 0 else 2
            return ReferenceBatchTargets(
                sample_indices=tuple(sample_indices.tolist()),
                contributions=torch.zeros_like(
                    layout.causal_pair_mask, dtype=torch.float32
                ),
                pair_mask=layout.causal_pair_mask.clone(),
                hidden_residuals=torch.zeros((*layout.causal_pair_mask.shape, width)),
            )

    def fake_sft_init(instance, *args, **kwargs):
        del args
        instance.model = kwargs["model"]
        instance.args = SimpleNamespace(output_dir=result_dir)
        instance.state = SimpleNamespace(global_step=1, max_steps=1)
        instance.accelerator = SimpleNamespace(process_index=rank)

    trainer_stage2_module.SFTTrainer.__init__ = fake_sft_init
    model = _TinyTrainerModel()
    trainer = CustomTrainerStage2(
        model=model,
        evidence_window_config=_residual_config(),
        latent_token_ids=TOKEN_IDS,
        reference_provider=Provider(),
    )
    try:
        trainer.compute_loss(model, _trainer_batch())
        outcome = "unexpected success"
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}; policy_calls={len(model.calls)}"
    finally:
        Path(result_dir, f"rank-{rank}.txt").write_text(outcome, encoding="utf-8")
        torch.distributed.destroy_process_group()


def test_asymmetric_hidden_target_preflight_fails_collectively_before_policy_replay(
    tmp_path,
):
    """One malformed target must not strand its peer in the engine edge collective."""
    context = multiprocessing.get_context("spawn")
    init_file = tmp_path / "gloo-init"
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    processes = [
        context.Process(
            target=_asymmetric_hidden_target_worker,
            args=(rank, str(init_file), str(result_dir)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(timeout=5)

    assert alive == [], "malformed reference stranded a peer in engine collectives"
    outcomes = [
        (result_dir / f"rank-{rank}.txt").read_text(encoding="utf-8")
        for rank in range(2)
    ]
    assert outcomes[0] == outcomes[1]
    assert "distributed reference target lookup failed" in outcomes[0]
    assert "hidden width" in outcomes[0]
    assert "policy_calls=1" in outcomes[0]


def test_nonresidual_reference_lookup_skips_hidden_edge_and_width_collectives(
    monkeypatch,
):
    """EW/relation lookup must not pay residual-only edge-count or width collectives."""
    reductions = []
    monkeypatch.setattr(trainer_stage2_module.distributed, "is_available", lambda: True)
    monkeypatch.setattr(trainer_stage2_module.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_stage2_module.distributed, "get_world_size", lambda: 1)

    def record_success(value, op):
        reductions.append(op)

    monkeypatch.setattr(
        trainer_stage2_module.distributed,
        "all_reduce",
        record_success,
    )
    monkeypatch.setattr(
        trainer_stage2_module.distributed,
        "all_gather_object",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("non-residual lookup entered hidden-width gather")
        ),
    )
    layout = _one_edge_layout()
    provider = _RecordingReferenceProvider()

    targets = trainer_stage2_module._coordinated_reference_lookup(
        provider,
        torch.tensor([17]),
        layout,
        require_hidden_residuals=False,
        max_evidence_edges_per_window=0,
        policy_hidden_width=2,
    )

    assert targets.sample_indices == (17,)
    assert len(reductions) == 2


def test_zero_edge_rank_matches_remote_residual_replay_slots_with_hidden_graph():
    """A rank without edges must join every remote residual forward/backward slot."""
    layout = Stage2InterleavedLayout(
        evidence_token_mask=torch.zeros((1, 0, 4), dtype=torch.bool),
        window_token_mask=torch.tensor([[[0, 1, 1, 0]]], dtype=torch.bool),
        evidence_valid=torch.zeros((1, 0), dtype=torch.bool),
        window_valid=torch.tensor([[True]]),
        causal_pair_mask=torch.zeros((1, 0, 1), dtype=torch.bool),
    )
    engine = Stage2EvidenceWindowEngine(
        _residual_config(),
        edge_count_coordinator=lambda _count, _device: (2, 2),
    )
    scale = torch.tensor(1.0, requires_grad=True)
    calls = []

    def policy_forward(block_mask, need_hidden):
        calls.append((block_mask, need_hidden))
        return Stage2ConditionOutput(
            scale * torch.ones((1, 4)),
            scale * torch.ones((1, 4, 2)) if need_hidden else None,
        )

    result = engine.compute(
        layout,
        policy_forward,
        progress=1.0,
        reference_targets=ReferenceBatchTargets(
            sample_indices=(17,),
            contributions=torch.zeros((1, 0, 1)),
            pair_mask=torch.zeros((1, 0, 1), dtype=torch.bool),
            hidden_residuals=torch.zeros((1, 0, 1, 2)),
        ),
    )

    assert result.effects.forward_count == 3
    assert [need_hidden for _, need_hidden in calls] == [True, True, True]
    assert calls[0][0] is None
    assert all(not block.any() for block, _ in calls[1:])
    assert set(result.weighted_objectives) == {"residual"}
    assert result.weighted_objectives["residual"].item() == 0.0
    result.auxiliary_loss.backward()
    assert scale.grad is not None and scale.grad.item() == 0.0


def test_residual_objective_uses_valid_edge_population_for_distributed_mean(
    monkeypatch,
):
    """Ranks with zero residual edges must still join the residual count reduction."""
    reductions = []
    monkeypatch.setattr(trainer_stage2_module.distributed, "is_available", lambda: True)
    monkeypatch.setattr(trainer_stage2_module.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_stage2_module.distributed, "get_world_size", lambda: 2)

    def add_remote_edges(value, op):
        reductions.append(op)
        value.add_(3.0)

    monkeypatch.setattr(
        trainer_stage2_module.distributed, "all_reduce", add_remote_edges
    )
    residual = torch.tensor(2.0, requires_grad=True)
    effects = Stage2EvidenceEffects(
        contributions=torch.zeros((1, 1, 1)),
        pair_mask=torch.zeros((1, 1, 1), dtype=torch.bool),
        hidden_residuals=torch.zeros((1, 1, 1, 2)),
        forward_count=2,
    )
    result = Stage2EvidenceWindowResult(
        auxiliary_loss=residual,
        losses={"residual": residual},
        metric_sums={"loss/residual": (torch.tensor(0.0), torch.tensor(0.0))},
        effects=effects,
        weighted_objectives={"residual": residual},
    )

    scaled = CustomTrainerStage2._globally_scaled_auxiliary(torch.tensor(0.0), result)

    assert scaled.item() == 0.0
    assert len(reductions) == 1
    scaled.backward()
    assert residual.grad is not None and residual.grad.item() == 0.0
