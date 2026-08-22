from dataclasses import replace
import importlib
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import trainer_stage2 as trainer_stage2_module
from evidence_window.metrics import softmax_contribution_summaries
from evidence_window.stage2_config import (
    Stage2DiagnosticsConfig,
    Stage2EvidenceWindowConfig,
    Stage2PreferenceConfig,
    Stage2ScheduleConfig,
)
from evidence_window.stage2_effects import Stage2ConditionOutput
from evidence_window.stage2_effects import Stage2EvidenceEffects
from evidence_window.stage2_layout import Stage2InterleavedLayout
from trainer_stage2 import CustomTrainerStage2


TOKEN_IDS = {
    "start": 90,
    "end": 91,
    "pad": 92,
    "padding": 0,
    "special": (0, 90, 91, 92, 99),
}


def _config(**changes) -> Stage2EvidenceWindowConfig:
    base = Stage2EvidenceWindowConfig(
        preference=Stage2PreferenceConfig(enabled=True, lambda_ew=2.0, beta=0.3),
    )
    return replace(base, **changes)


def _one_edge_layout() -> Stage2InterleavedLayout:
    return Stage2InterleavedLayout(
        evidence_token_mask=torch.tensor([[[1, 1, 1, 0, 0]]], dtype=torch.bool),
        window_token_mask=torch.tensor([[[0, 0, 0, 1, 1]]], dtype=torch.bool),
        evidence_valid=torch.tensor([[True]]),
        window_valid=torch.tensor([[True]]),
        causal_pair_mask=torch.tensor([[[True]]]),
    )


def _no_edge_layout() -> Stage2InterleavedLayout:
    return Stage2InterleavedLayout(
        evidence_token_mask=torch.zeros((1, 0, 4), dtype=torch.bool),
        window_token_mask=torch.tensor([[[0, 1, 1, 0]]], dtype=torch.bool),
        evidence_valid=torch.zeros((1, 0), dtype=torch.bool),
        window_valid=torch.tensor([[True]]),
        causal_pair_mask=torch.zeros((1, 0, 1), dtype=torch.bool),
    )


def test_stage2_engine_module_and_public_interface_exist():
    """A missing engine leaves Stage 2 unable to compose exact edge effects."""
    module = importlib.import_module("evidence_window.stage2_engine")

    assert hasattr(module, "Stage2EvidenceWindowEngine")
    assert hasattr(module, "Stage2EvidenceWindowResult")


def test_local_gradient_survives_global_saturation():
    """One dominant edge must not suppress another edge's outer gradient."""
    beta = 1.0
    global_margins = torch.tensor([20.0, 0.0], requires_grad=True)
    global_gradient = torch.autograd.grad(
        -F.logsigmoid(beta * global_margins.sum()), global_margins
    )[0]
    local_margins = torch.tensor([20.0, 0.0], requires_grad=True)
    local_gradient = torch.autograd.grad(
        -F.logsigmoid(beta * local_margins).sum(), local_margins
    )[0]

    assert abs(global_gradient[1].item()) < 1e-7
    assert abs(local_gradient[1].item()) == pytest.approx(beta / 2)


def test_reference_free_formula_and_both_branches_receive_gradients():
    """Reference-free EW is the independent masked edge logistic objective."""
    from evidence_window.losses import stage2_ew_preference_loss

    full = torch.tensor([[[0.4, -0.2]]], requires_grad=True)
    loo = torch.tensor([[[0.1, -0.5]]], requires_grad=True)
    mask = torch.tensor([[[True, True]]])
    contribution = full - loo

    actual = stage2_ew_preference_loss(contribution, mask, beta=0.3)
    expected = -F.logsigmoid(0.3 * contribution[mask]).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert full.grad is not None and torch.count_nonzero(full.grad).item() == 2
    assert loo.grad is not None and torch.count_nonzero(loo.grad).item() == 2


def test_no_edge_loss_and_engine_result_are_finite_zero_without_a_forward():
    """A batch without causal evidence falls back to CE without NaN or replay."""
    from evidence_window.losses import stage2_ew_preference_loss
    from evidence_window.stage2_engine import Stage2EvidenceWindowEngine

    empty = torch.zeros((1, 0, 1), requires_grad=True)
    loss = stage2_ew_preference_loss(empty, torch.zeros_like(empty, dtype=torch.bool), 0.3)
    assert loss.item() == 0.0
    assert torch.isfinite(loss)

    calls = 0

    def forbidden_forward(_block, _hidden):
        nonlocal calls
        calls += 1
        raise AssertionError("no-edge batch entered an auxiliary forward")

    result = Stage2EvidenceWindowEngine(_config()).compute(
        _no_edge_layout(), forbidden_forward, progress=1.0
    )
    assert calls == 0
    assert result.effects.forward_count == 0
    assert result.auxiliary_loss.item() == 0.0
    assert torch.isfinite(result.auxiliary_loss)


def test_engine_applies_schedule_and_keeps_global_margin_diagnostic_only():
    """Changing the global aggregate must not replace the local edge objective."""
    from evidence_window.stage2_engine import Stage2EvidenceWindowEngine

    config = _config(
        schedule=Stage2ScheduleConfig(ew_warmup_ratio=0.2, ew_ramp_ratio=0.2)
    )
    engine = Stage2EvidenceWindowEngine(config)
    full_scale = torch.tensor(4.0, requires_grad=True)
    loo_scale = torch.tensor(2.0, requires_grad=True)

    def policy_forward(block_mask, need_hidden):
        assert need_hidden is False
        scale = full_scale if block_mask is None else loo_scale
        return Stage2ConditionOutput(
            token_logps=scale * torch.ones((1, 5)),
            final_hidden_state=None,
        )

    before = engine.compute(_one_edge_layout(), policy_forward, progress=0.2)
    assert before.effects.forward_count == 0
    assert before.auxiliary_loss.item() == 0.0

    result = engine.compute(_one_edge_layout(), policy_forward, progress=0.3)
    expected_raw = -F.logsigmoid(torch.tensor(0.3 * 2.0))
    torch.testing.assert_close(result.losses["ew_pref"], expected_raw)
    torch.testing.assert_close(result.auxiliary_loss, expected_raw)
    assert "ew/global_margin" in result.metric_sums
    assert "ew/global_outer_coefficient" in result.metric_sums
    result.auxiliary_loss.backward()
    assert full_scale.grad is not None
    assert loo_scale.grad is not None


def test_softmax_concentration_and_raw_threshold_statistics_are_finite():
    """Concentration is descriptive softmax mass, never an entropy reward."""
    relations = torch.tensor(
        [[[2.0, -3.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 9.0, 0.0]]]
    )
    pairs = torch.tensor(
        [[[True, True, False], [True, True, True], [True, False, False]]]
    )

    out = softmax_contribution_summaries(relations, pairs, temperature=1.0, threshold=0.5)

    expected_top1 = torch.softmax(torch.tensor([2.0, 0.0, -1.0]), dim=0).max()
    torch.testing.assert_close(out.top1_ratio[0, 0], expected_top1)
    assert out.above_threshold_count.tolist() == [[1, 1, 0]]
    assert out.entropy_valid.tolist() == [[True, True, False]]
    for value in (out.top1_ratio, out.normalized_entropy):
        assert torch.isfinite(value).all()


class TinyStage2AuxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 8)
        self.head = torch.nn.Linear(8, 128)
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
                "blocked": attention_block_mask is not None,
                "backend": attention_backend_override,
                "hidden": output_hidden_states,
                "use_cache": use_cache,
            }
        )
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        hidden = hidden.cumsum(dim=1)
        if attention_block_mask is not None:
            blocked_queries = attention_block_mask.any(dim=-1).unsqueeze(-1)
            hidden = hidden - blocked_queries.to(hidden.dtype) * self.embedding.weight[92]
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden)


def _make_trainer(monkeypatch, tmp_path, config):
    def fake_sft_init(instance, *args, **kwargs):
        del args
        instance.model = kwargs["model"]
        instance.args = SimpleNamespace(output_dir=str(tmp_path))
        instance.state = SimpleNamespace(global_step=2, max_steps=10)
        instance.accelerator = SimpleNamespace(process_index=0)

    monkeypatch.setattr(trainer_stage2_module.SFTTrainer, "__init__", fake_sft_init)
    model = TinyStage2AuxModel()
    trainer = CustomTrainerStage2(
        model=model,
        evidence_window_config=config,
        latent_token_ids=TOKEN_IDS,
    )
    return trainer, model


def _stage2_batch():
    return {
        "input_ids": torch.tensor([[90, 92, 91, 5, 6]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, 5, 6]]),
        "ew_sample_indices": torch.tensor([17]),
        "ew_plan_types": (("latent", "text"),),
    }


def test_enabled_trainer_runs_ce_then_same_policy_matched_sdpa_and_keeps_metadata_out(monkeypatch, tmp_path):
    """The auxiliary path must use the trainable policy twice and explicit SDPA."""
    trainer, model = _make_trainer(monkeypatch, tmp_path, _config())
    caller_batch = _stage2_batch()

    loss, outputs = trainer.compute_loss(model, caller_batch, return_outputs=True)
    loss.backward()

    assert torch.isfinite(loss)
    assert outputs.loss is not None
    assert model.calls[0] == {
        "blocked": False,
        "backend": None,
        "hidden": False,
        "use_cache": None,
    }
    assert [call["backend"] for call in model.calls[1:]] == ["sdpa", "sdpa"]
    assert [call["blocked"] for call in model.calls[1:]] == [False, True]
    assert all(call["hidden"] is False for call in model.calls)
    assert all(call["use_cache"] is False for call in model.calls[1:])
    assert model.embedding.weight.grad is not None
    assert model.head.weight.grad is not None
    assert "ew_sample_indices" in caller_batch and "ew_plan_types" in caller_batch


def test_trainer_matrix_dump_is_interval_gated_detached_and_opaque(monkeypatch, tmp_path):
    """Repeated loss calls at one dump step must not emit payload-bearing records."""
    dump_dir = tmp_path / "matrices"
    config = _config(
        diagnostics=Stage2DiagnosticsConfig(
            matrix_dump_interval=2,
            matrix_dump_dir=str(dump_dir),
        )
    )
    trainer, model = _make_trainer(monkeypatch, tmp_path, config)

    trainer.compute_loss(model, _stage2_batch())
    trainer.compute_loss(model, _stage2_batch())

    records = (dump_dir / "rank-00000.jsonl").read_text().splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert set(record) == {"observation_id", "rank", "step", "relation", "pair_mask"}
    assert record["observation_id"] == "r0-s2-o0"
    assert "17" not in record["observation_id"]


def test_rank_without_edges_still_joins_global_edge_count_collective(monkeypatch):
    """Skipping an all-reduce on an empty rank deadlocks heterogeneous batches."""
    from evidence_window.stage2_engine import Stage2EvidenceWindowResult

    calls = []

    monkeypatch.setattr(trainer_stage2_module.distributed, "is_available", lambda: True)
    monkeypatch.setattr(trainer_stage2_module.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_stage2_module.distributed, "get_world_size", lambda: 2)

    def add_remote_edges(value, op):
        calls.append(op)
        value.add_(3.0)

    monkeypatch.setattr(trainer_stage2_module.distributed, "all_reduce", add_remote_edges)
    zero = torch.tensor(0.0)
    effects = Stage2EvidenceEffects(
        contributions=torch.zeros((1, 0, 1)),
        pair_mask=torch.zeros((1, 0, 1), dtype=torch.bool),
        hidden_residuals=None,
        forward_count=0,
    )
    result = Stage2EvidenceWindowResult(zero, {"ew_pref": zero}, {}, effects)

    scaled = CustomTrainerStage2._globally_scaled_auxiliary(zero, result)

    assert len(calls) == 1
    assert scaled.item() == 0.0
