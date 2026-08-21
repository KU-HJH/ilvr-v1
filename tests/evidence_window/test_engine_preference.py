from __future__ import annotations

import pytest
import torch

from evidence_window.config import (
    DiagnosticsConfig,
    EvidenceWindowConfig,
    LOOConfig,
    PreferenceConfig,
    ReferenceConfig,
    ResidualConfig,
)
from evidence_window.engine import (
    Condition,
    EvidenceWindowEngine,
    ModelConditionOutput,
    build_condition_execution_plan,
)
from evidence_window.layout import InterleavedLayout


def _layout(
    evidence_token_mask: torch.Tensor,
    window_token_mask: torch.Tensor,
    causal_pair_mask: torch.Tensor,
    training_pair_mask: torch.Tensor,
) -> InterleavedLayout:
    batch_size, evidence_count, _ = evidence_token_mask.shape
    window_count = window_token_mask.shape[1]
    predecessor = torch.full((batch_size, window_count), -1, dtype=torch.long)
    for batch_index, evidence_index, window_index in training_pair_mask.nonzero().tolist():
        predecessor[batch_index, window_index] = evidence_index
    return InterleavedLayout(
        evidence_token_mask=evidence_token_mask,
        window_token_mask=window_token_mask,
        evidence_valid=evidence_token_mask.any(dim=-1),
        window_valid=window_token_mask.any(dim=-1),
        causal_pair_mask=causal_pair_mask,
        training_pair_mask=training_pair_mask,
        native_predecessor=predecessor,
    )


def _ew_config(*, cap: int = 0) -> EvidenceWindowConfig:
    return EvidenceWindowConfig(
        reference=ReferenceConfig(mode="adapter_disabled"),
        preference=PreferenceConfig(beta=1.0, lambda_ew=1.0),
        loo=LOOConfig(max_spans_per_example=cap),
    )


def _scalar_layout() -> InterleavedLayout:
    return _layout(
        torch.ones((1, 1, 1), dtype=torch.bool),
        torch.ones((1, 1, 1), dtype=torch.bool),
        torch.ones((1, 1, 1), dtype=torch.bool),
        torch.ones((1, 1, 1), dtype=torch.bool),
    )


def test_one_loo_forward_scores_all_downstream_windows_and_uses_pair_policies():
    layout = _layout(
        torch.tensor([[[1, 0, 0, 0], [0, 0, 1, 0]]], dtype=torch.bool),
        torch.tensor([[[0, 1, 0, 0], [0, 0, 0, 1]]], dtype=torch.bool),
        torch.tensor([[[1, 1], [0, 1]]], dtype=torch.bool),
        torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool),
    )
    calls: list[tuple[str, str, int | None]] = []
    scores = {
        ("policy", "loo", 0): [0.0, 1.0, 0.0, 2.0],
        ("policy", "loo", 1): [0.0, 4.0, 0.0, 5.0],
        ("reference", "full", None): [0.0, 2.0, 0.0, 4.0],
        ("reference", "loo", 0): [0.0, 1.0, 0.0, 2.0],
        ("reference", "loo", 1): [0.0, 2.0, 0.0, 3.0],
    }

    def forward(role, condition, values, need_hidden):
        del need_hidden
        calls.append((role, condition.kind, condition.evidence_index))
        if condition.kind == "loo":
            expected = torch.ones_like(values)
            expected[:, condition.evidence_index] = 0
            assert torch.equal(values, expected)
        return ModelConditionOutput(torch.tensor([scores[calls[-1]]]))

    result = EvidenceWindowEngine(_ew_config()).compute(
        layout,
        torch.ones((1, 2, 1)),
        ModelConditionOutput(torch.tensor([[0.0, 4.0, 0.0, 8.0]])),
        forward,
    )

    assert calls.count(("policy", "full", None)) == 0
    assert calls.count(("policy", "loo", 0)) == 1
    assert calls.count(("policy", "loo", 1)) == 1
    assert calls.count(("reference", "full", None)) == 1
    assert calls.count(("reference", "loo", 0)) == 1
    assert calls.count(("reference", "loo", 1)) == 1
    assert not any(kind == "all_null" for _, kind, _ in calls)
    assert result.ew_margins.shape[-2:] == layout.training_pair_mask.shape[-2:]
    assert result.ew_margins.tolist() == [[[2.0, 4.0], [0.0, 2.0]]]
    assert torch.equal(result.preference_pair_mask, layout.training_pair_mask)
    assert torch.equal(result.relation_pair_mask, layout.causal_pair_mask)


def test_margin_matches_boxed_reference_adjusted_formula_and_reference_is_detached():
    policy_full = torch.tensor([[10.0]], requires_grad=True)
    policy_loo = torch.tensor([[7.0]], requires_grad=True)
    reference_full = torch.tensor([[8.0]], requires_grad=True)
    reference_loo = torch.tensor([[6.0]], requires_grad=True)

    def forward(role, condition, values, need_hidden):
        del values, need_hidden
        if role == "policy":
            assert condition == Condition("loo", 0)
            return ModelConditionOutput(policy_loo)
        if condition.kind == "full":
            return ModelConditionOutput(reference_full)
        return ModelConditionOutput(reference_loo)

    result = EvidenceWindowEngine(_ew_config()).compute(
        _scalar_layout(),
        torch.ones((1, 1, 1)),
        ModelConditionOutput(policy_full),
        forward,
    )

    assert result.ew_margins.item() == 1.0
    result.total_loss.backward()
    assert policy_full.grad is not None and policy_full.grad.abs().sum() > 0
    assert policy_loo.grad is not None and policy_loo.grad.abs().sum() > 0
    assert reference_full.grad is None
    assert reference_loo.grad is None


def test_direct_window_mask_prefix_cancels_earlier_margin():
    layout = _layout(
        torch.tensor([[[1, 0]]], dtype=torch.bool),
        torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool),
        torch.tensor([[[1, 1]]], dtype=torch.bool),
        torch.tensor([[[0, 1]]], dtype=torch.bool),
    )

    def run(earlier_score: float) -> torch.Tensor:
        def forward(role, condition, values, need_hidden):
            del condition, values, need_hidden
            return ModelConditionOutput(torch.zeros((1, 2)))

        result = EvidenceWindowEngine(_ew_config()).compute(
            layout,
            torch.ones((1, 1, 1)),
            ModelConditionOutput(
                torch.tensor([[earlier_score, 2.0]], requires_grad=True)
            ),
            forward,
        )
        return result.ew_margins

    baseline = run(3.0)
    changed_prefix = run(3000.0)
    assert baseline[0, 0, 1].item() == 2.0
    assert changed_prefix[0, 0, 1].item() == 2.0


def test_all_null_global_and_window_objectives_use_only_needed_conditions():
    config = EvidenceWindowConfig(
        reference=ReferenceConfig(mode="adapter_disabled"),
        preference=PreferenceConfig(
            beta=1.0,
            lambda_global=2.0,
            lambda_window=3.0,
        ),
    )
    calls: list[tuple[str, str, int | None]] = []
    outputs = {
        ("policy", "all_null", None): 7.0,
        ("reference", "full", None): 8.0,
        ("reference", "all_null", None): 6.0,
    }

    def forward(role, condition, values, need_hidden):
        del values, need_hidden
        key = (role, condition.kind, condition.evidence_index)
        calls.append(key)
        return ModelConditionOutput(torch.tensor([[outputs[key]]]))

    result = EvidenceWindowEngine(config).compute(
        _scalar_layout(),
        torch.ones((1, 1, 1)),
        ModelConditionOutput(torch.tensor([[10.0]], requires_grad=True)),
        forward,
    )

    expected = torch.nn.functional.softplus(torch.tensor(-1.0))
    assert torch.allclose(result.losses["global_preference"], expected)
    assert torch.allclose(result.losses["window_preference"], expected)
    assert torch.allclose(result.total_loss, 5.0 * expected)
    assert calls.count(("reference", "full", None)) == 1
    assert set(calls) == set(outputs)
    assert result.ew_margins is None


def test_combined_objectives_reuse_reference_full_exactly_once():
    config = EvidenceWindowConfig(
        reference=ReferenceConfig(mode="adapter_disabled"),
        preference=PreferenceConfig(
            beta=1.0,
            lambda_global=1.0,
            lambda_ew=1.0,
        ),
    )
    calls: list[tuple[str, str, int | None]] = []

    def forward(role, condition, values, need_hidden):
        del values, need_hidden
        calls.append((role, condition.kind, condition.evidence_index))
        return ModelConditionOutput(torch.zeros((1, 1)))

    EvidenceWindowEngine(config).compute(
        _scalar_layout(),
        torch.ones((1, 1, 1)),
        ModelConditionOutput(torch.zeros((1, 1), requires_grad=True)),
        forward,
    )

    assert calls.count(("reference", "full", None)) == 1
    assert set(calls) == {
        ("policy", "all_null", None),
        ("policy", "loo", 0),
        ("reference", "full", None),
        ("reference", "all_null", None),
        ("reference", "loo", 0),
    }


def test_logit_residual_hidden_states_are_requested_only_for_full_and_loo():
    config = EvidenceWindowConfig(
        reference=ReferenceConfig(mode="adapter_disabled"),
        preference=PreferenceConfig(
            beta=1.0,
            lambda_window=1.0,
            lambda_ew=1.0,
        ),
        residual=ResidualConfig(lambda_logit=1.0),
    )
    calls: list[tuple[str, str, int | None, bool]] = []

    def forward(role, condition, values, need_hidden):
        del values
        calls.append((role, condition.kind, condition.evidence_index, need_hidden))
        hidden_states = torch.zeros((1, 2, 2)) if need_hidden else None
        return ModelConditionOutput(torch.zeros((1, 2)), hidden_states)

    full_hidden_states = torch.ones((1, 2, 2), requires_grad=True)
    policy_full = ModelConditionOutput(
        torch.zeros((1, 2), requires_grad=True),
        full_hidden_states,
    )
    EvidenceWindowEngine(config).compute(
        _layout(
            torch.tensor([[[1, 0]]], dtype=torch.bool),
            torch.tensor([[[0, 1]]], dtype=torch.bool),
            torch.ones((1, 1, 1), dtype=torch.bool),
            torch.ones((1, 1, 1), dtype=torch.bool),
        ),
        torch.ones((1, 1, 1)),
        policy_full,
        forward,
        lm_head_weight=lambda role: torch.eye(2),
    )

    assert ("policy", "full", None, True) not in calls
    assert ("policy", "loo", 0, True) in calls
    assert ("policy", "all_null", None, False) in calls
    assert all(
        not need_hidden
        for role, _, _, need_hidden in calls
        if role == "reference"
    )
    assert policy_full.hidden_states is full_hidden_states


def test_loo_cap_masks_unselected_pairs_and_calls_only_selected_conditions():
    identity = torch.eye(3, dtype=torch.bool).unsqueeze(0)
    layout = _layout(
        identity.clone(),
        identity.clone(),
        torch.triu(torch.ones((1, 3, 3), dtype=torch.bool)),
        identity.clone(),
    )
    calls: list[tuple[str, str, int | None]] = []

    def forward(role, condition, values, need_hidden):
        del values, need_hidden
        calls.append((role, condition.kind, condition.evidence_index))
        return ModelConditionOutput(torch.zeros((1, 3)))

    result = EvidenceWindowEngine(_ew_config(cap=2)).compute(
        layout,
        torch.ones((1, 3, 1)),
        ModelConditionOutput(torch.zeros((1, 3), requires_grad=True)),
        forward,
    )

    assert {index for _, kind, index in calls if kind == "loo"} == {0, 2}
    assert result.preference_pair_mask.nonzero().tolist() == [[0, 0, 0], [0, 2, 2]]
    assert result.relation_pair_mask[0, 1].sum() == 0


def test_empty_pair_set_skips_conditions_and_returns_finite_differentiable_zero():
    layout = _layout(
        torch.ones((1, 1, 1), dtype=torch.bool),
        torch.ones((1, 1, 1), dtype=torch.bool),
        torch.zeros((1, 1, 1), dtype=torch.bool),
        torch.zeros((1, 1, 1), dtype=torch.bool),
    )
    policy_full = torch.tensor([[2.0]], requires_grad=True)

    def unexpected_forward(*_args):
        raise AssertionError("empty pair set must not execute a condition")

    result = EvidenceWindowEngine(_ew_config()).compute(
        layout,
        torch.ones((1, 1, 1)),
        ModelConditionOutput(policy_full),
        unexpected_forward,
    )

    assert result.total_loss.requires_grad
    assert torch.isfinite(result.total_loss)
    assert result.total_loss.item() == 0.0
    result.total_loss.backward()
    assert torch.equal(policy_full.grad, torch.zeros_like(policy_full))


def test_dummy_loo_is_graph_free_but_later_locally_used_loo_keeps_gradients():
    layout = _layout(
        torch.tensor([[[0, 0], [1, 0]]], dtype=torch.bool),
        torch.tensor([[[0, 1]]], dtype=torch.bool),
        torch.tensor([[[0], [1]]], dtype=torch.bool),
        torch.tensor([[[0], [1]]], dtype=torch.bool),
    )
    config = _ew_config()
    plan = build_condition_execution_plan(config, (0, 1))
    dummy_parameter = torch.tensor(2.0, requires_grad=True)
    real_parameter = torch.tensor(3.0, requires_grad=True)
    calls = []
    outputs = {}

    def forward(role, condition, values, need_hidden):
        del need_hidden
        key = (role, condition.kind, condition.evidence_index)
        calls.append((*key, torch.is_grad_enabled()))
        if role == "policy":
            parameter = (
                dummy_parameter if condition.evidence_index == 0 else real_parameter
            )
            output = ModelConditionOutput(parameter * torch.ones((1, 2)))
            outputs[key] = output
            if condition.evidence_index == 0:
                assert torch.equal(values, torch.ones((1, 1, 1)))
            else:
                assert not values.any()
            return output
        return ModelConditionOutput(torch.zeros((1, 2)))

    result = EvidenceWindowEngine(config).compute(
        layout,
        torch.ones((1, 1, 1)),
        ModelConditionOutput(torch.zeros((1, 2), requires_grad=True)),
        forward,
        execution_plan=plan,
    )

    assert calls == [
        ("reference", "full", None, False),
        ("policy", "loo", 0, False),
        ("reference", "loo", 0, False),
        ("policy", "loo", 1, True),
        ("reference", "loo", 1, False),
    ]
    assert not outputs[("policy", "loo", 0)].token_logps.requires_grad
    assert outputs[("policy", "loo", 0)].token_logps.grad_fn is None
    assert outputs[("policy", "loo", 0)].hidden_states is None
    assert outputs[("policy", "loo", 1)].token_logps.requires_grad
    result.total_loss.backward()
    assert dummy_parameter.grad is None
    assert real_parameter.grad is not None and real_parameter.grad.abs() > 0


def test_locally_empty_all_null_runs_in_order_without_building_a_graph():
    layout = _layout(
        torch.tensor([[[1, 0]]], dtype=torch.bool),
        torch.tensor([[[0, 1]]], dtype=torch.bool),
        torch.zeros((1, 1, 1), dtype=torch.bool),
        torch.zeros((1, 1, 1), dtype=torch.bool),
    )
    config = EvidenceWindowConfig(
        reference=ReferenceConfig(mode="adapter_disabled"),
        preference=PreferenceConfig(lambda_window=1.0),
    )
    parameter = torch.tensor(2.0, requires_grad=True)
    calls = []
    policy_output = None

    def forward(role, condition, values, need_hidden):
        nonlocal policy_output
        del need_hidden
        assert torch.equal(values, torch.ones((1, 1, 1)))
        calls.append((role, condition.kind, torch.is_grad_enabled()))
        if role == "policy":
            policy_output = ModelConditionOutput(parameter * torch.ones((1, 2)))
            return policy_output
        return ModelConditionOutput(torch.zeros((1, 2)))

    policy_full = torch.zeros((1, 2), requires_grad=True)
    result = EvidenceWindowEngine(config).compute(
        layout,
        torch.ones((1, 1, 1)),
        ModelConditionOutput(policy_full),
        forward,
    )

    assert calls == [
        ("policy", "all_null", False),
        ("reference", "full", False),
        ("reference", "all_null", False),
    ]
    assert policy_output is not None
    assert not policy_output.token_logps.requires_grad
    assert policy_output.token_logps.grad_fn is None
    assert policy_output.hidden_states is None
    result.total_loss.backward()
    assert parameter.grad is None


def test_diagnostic_only_all_null_policy_condition_is_graph_free():
    config = EvidenceWindowConfig(
        reference=ReferenceConfig(mode="adapter_disabled"),
        diagnostics=DiagnosticsConfig(global_saturation=True),
    )
    parameter = torch.tensor(2.0, requires_grad=True)
    policy_output = None
    policy_grad_enabled = None

    def forward(role, condition, values, need_hidden):
        nonlocal policy_output, policy_grad_enabled
        del condition, values, need_hidden
        if role == "policy":
            policy_grad_enabled = torch.is_grad_enabled()
            policy_output = ModelConditionOutput(parameter * torch.ones((1, 1)))
            return policy_output
        return ModelConditionOutput(torch.zeros((1, 1)))

    EvidenceWindowEngine(config).compute(
        _scalar_layout(),
        torch.ones((1, 1, 1)),
        ModelConditionOutput(torch.zeros((1, 1), requires_grad=True)),
        forward,
    )

    assert policy_grad_enabled is False
    assert policy_output is not None
    assert not policy_output.token_logps.requires_grad
    assert policy_output.token_logps.grad_fn is None


def test_nonfinite_enabled_output_raises_with_numeric_layout_only():
    def forward(role, condition, values, need_hidden):
        del role, condition, values, need_hidden
        return ModelConditionOutput(torch.tensor([[float("nan")]]))

    with pytest.raises(
        FloatingPointError,
        match=r"batch=1, evidence=1, windows=1, preference_pairs=1, relation_pairs=1",
    ) as error:
        EvidenceWindowEngine(_ew_config()).compute(
            _scalar_layout(),
            torch.ones((1, 1, 1)),
            ModelConditionOutput(torch.ones((1, 1), requires_grad=True)),
            forward,
        )
    assert "sample" not in str(error.value).lower()
