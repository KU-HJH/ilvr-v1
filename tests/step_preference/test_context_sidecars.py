"""Context-only sidecar planning and grouped replay contracts."""

from dataclasses import replace

import pytest
import torch

import step_preference.sidecar as sidecar
from step_preference.shared_prefix_measurement import RuntimeCounters, tiny_measurement_case


def _context_plan(batch):
    builder = getattr(sidecar, "build_context_sidecar_plan", None)
    assert callable(builder), "context sidecar planner missing"
    return builder(
        clean_input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        spans=batch["reasoning_step_spans"],
    )


@pytest.mark.parametrize("k", [1, 4, 8, 16])
def test_context_group_ignores_legacy_chunk_budget(k):
    """Catches budget-based or per-step decoder calls in context mode."""
    model, batch = tiny_measurement_case(
        steps=k, total_target_tokens=2 * k, device="cpu"
    )
    plan = _context_plan(batch)

    assert plan.context_preference is True
    assert [len(group) for group in plan.groups] == [k, k]
    assert all(
        branch.rejected_model_inputs is None
        for group in plan.groups
        for branch in group
    )

    with RuntimeCounters(model) as counters:
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            stage2_shared_prefix_inputs={
                "plan": plan,
                "backend": "dense",
                "query_budget": 1,
                "checkpoint": True,
            },
        )

    calls = counters.snapshot()
    assert calls["decoder_calls_forward"] - calls["latent_decoder_calls_forward"] == 3
    assert calls["sidecar_chunks_forward"] == 2
    assert len(output.stage2_sidecar_results) == 2 * k
    assert {result["kind"] for result in output.stage2_sidecar_results} == {
        "visual_off",
        "textual_off",
    }
    assert all(
        set(result) == {"name", "kind", "step_id", "mean_logp", "entropies"}
        for result in output.stage2_sidecar_results
    )
    for span, branch in zip(batch["reasoning_step_spans"], plan.groups[0]):
        assert branch.logical_positions == tuple(
            range(span.target_text_start - 1, span.target_text_end - 1)
        )
        assert branch.target_token_ids == tuple(
            batch["input_ids"][0, span.target_text_start : span.target_text_end].tolist()
        )


@pytest.mark.parametrize("group_count", [1, 3])
def test_context_plan_requires_exactly_two_modality_groups(group_count):
    """Catches silently accepting a missing or extra context modality group."""
    model, batch = tiny_measurement_case(steps=1, total_target_tokens=2, device="cpu")
    plan = _context_plan(batch)
    groups = (plan.groups[0],) if group_count == 1 else (*plan.groups, plan.groups[0])

    with pytest.raises(ValueError, match="local chosen branches"):
        model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            stage2_shared_prefix_inputs={"plan": replace(plan, groups=groups)},
        )


def test_context_plan_rejects_swapped_groups_and_rejected_payloads():
    """Catches mislabeled V/T groups or negative replay entering context mode."""
    model, batch = tiny_measurement_case(steps=1, total_target_tokens=2, device="cpu")
    plan = _context_plan(batch)
    invalid_plans = (
        replace(plan, groups=(plan.groups[1], plan.groups[0])),
        replace(
            plan,
            groups=(
                (
                    replace(
                        plan.groups[0][0],
                        rejected_model_inputs={"unexpected": "negative"},
                    ),
                ),
                plan.groups[1],
            ),
        ),
    )

    for invalid in invalid_plans:
        with pytest.raises(ValueError, match="local chosen branches"):
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                stage2_shared_prefix_inputs={"plan": invalid},
            )


def test_legacy_plan_keeps_budget_chunking_and_default_mode():
    """Catches the context fast path changing existing legacy replay semantics."""
    model, batch = tiny_measurement_case(steps=1, total_target_tokens=2, device="cpu")
    plan = sidecar.build_sidecar_plan(
        clean_input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        spans=batch["reasoning_step_spans"],
        steps=batch["step_preference"]["steps"],
    )

    assert plan.context_preference is False
    with RuntimeCounters(model) as counters:
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            stage2_shared_prefix_inputs={"plan": plan, "query_budget": 1},
        )

    assert counters.snapshot()["sidecar_chunks_forward"] == 3
    assert {result["kind"] for result in output.stage2_sidecar_results} == {
        "visual_off",
        "visual_rejected",
        "textual_off",
    }
