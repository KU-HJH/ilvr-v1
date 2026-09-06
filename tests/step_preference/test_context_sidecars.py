"""Context-only sidecar planning and grouped replay contracts."""

import copy
from dataclasses import replace
import os

import pytest
import torch
import torch.nn.functional as F

import step_preference.sidecar as sidecar
from step_preference.equivalence import _trainer
from step_preference.shared_prefix_measurement import RuntimeCounters, tiny_measurement_case
from tests.step_preference.test_context_preference import context_config


@pytest.fixture(autouse=True)
def _isolate_tf32_backend_state():
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def two_step_case():
    """Return two FP32 steps whose target lengths differ (2 versus 5)."""
    model, batch = tiny_measurement_case(
        steps=2, total_target_tokens=4, device="cpu"
    )
    assert model.config.num_hidden_layers == 2
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    assert model.config.attention_dropout == 0
    assert all(
        module.p == 0
        for module in model.modules()
        if isinstance(module, torch.nn.Dropout)
    )
    ids = torch.cat((batch["input_ids"], torch.tensor([[17, 18, 19]])), dim=1)
    spans = batch["reasoning_step_spans"]
    spans = (spans[0], replace(spans[1], target_text_end=ids.shape[1]))
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": ids.clone(),
        "sample_ids": ("context-two-step",),
        "reasoning_step_spans": spans,
        "step_preference": {"steps": tuple({"span": span} for span in spans)},
    }
    # Distinct temporal/height/width coordinates catch accidental 1-D reuse.
    positions = torch.arange(ids.shape[1])[None, None, :]
    batch["position_ids"] = torch.cat((positions, 2 * positions, 3 * positions))
    return model, batch


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_CONTEXT_CUDA") != "1" or not torch.cuda.is_available(),
    reason="explicit context CUDA check",
)
def test_context_cuda_flex_matches_dense_scores_gradients_and_calls(monkeypatch):
    """Catches a dense fallback, non-rectangular Flex, or CUDA parity regression."""
    import transformers.integrations.flex_attention as integration

    base, batch = two_step_case()
    batch = {
        key: (value.to("cuda:0") if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }
    base = base.to("cuda:0")
    records = []
    actual_flex_calls = []
    original = integration.compile_friendly_flex_attention

    def observed(q, k, v, **kwargs):
        assert kwargs.get("kernel_options") == {"FORCE_USE_FLEX_ATTENTION": True}
        actual_flex_calls.append(
            {
                "query_length": q.shape[-2],
                "key_length": k.shape[-2],
                "enable_gqa": kwargs["enable_gqa"],
            }
        )
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(integration, "compile_friendly_flex_attention", observed)
    for backend in ("dense", "flex"):
        model = copy.deepcopy(base)
        config = replace(context_config(), sidecar_attention_backend=backend)
        trainer = _trainer(config)
        with RuntimeCounters(model) as counters:
            loss, output = trainer.compute_loss(model, batch, return_outputs=True)
            before = counters.snapshot()
            counters.phase = "backward"
            loss.backward()
            counts = counters.snapshot()
        assert (
            before["decoder_calls_forward"]
            - before["latent_decoder_calls_forward"]
            == 3
        )
        records.append(
            (
                loss.detach(),
                output.stage2_sidecar_results,
                {
                    name: parameter.grad.clone()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                },
                counts,
            )
        )

    root_length = batch["input_ids"].shape[1]
    assert actual_flex_calls
    assert all(
        call["key_length"] == root_length + call["query_length"]
        and call["enable_gqa"]
        for call in actual_flex_calls
    )
    torch.testing.assert_close(records[0][0], records[1][0], atol=1e-5, rtol=1e-4)
    for left, right in zip(records[0][1], records[1][1]):
        assert left["name"] == right["name"]
        for key in ("mean_logp", "entropies"):
            torch.testing.assert_close(
                left[key], right[key], atol=1e-5, rtol=1e-4
            )
    for name in records[0][2]:
        torch.testing.assert_close(
            records[0][2][name],
            records[1][2][name],
            atol=1e-5,
            rtol=1e-4,
            msg=name,
        )
    print(
        {
            "backend_seen": "actual_cuda_flex",
            "actual_flex_calls": actual_flex_calls,
            "context_cuda_dense": records[0][3],
            "context_cuda_flex": records[1][3],
        }
    )


def literal_context_oracle(model, batch):
    """Score literal full-prefix ablations without production planning/reduction."""
    ids = batch["input_ids"]
    kwargs = {
        key: batch[key]
        for key in ("input_ids", "attention_mask", "labels", "position_ids")
    }
    clean = model(**kwargs, use_cache=False)
    preference_losses, entropy_losses, scores = [], [], {}

    def scored(logits, targets):
        log_probs = logits.float().log_softmax(-1)
        mean_logp = log_probs.gather(-1, targets[:, None]).mean()
        entropies = -(log_probs.exp() * log_probs).sum(-1)
        return mean_logp, entropies

    for span in batch["reasoning_step_spans"]:
        start, end = span.target_text_start, span.target_text_end
        clean_logp, clean_entropy = scored(
            clean.logits[0, start - 1 : end - 1], ids[0, start:end]
        )
        for modality, blocked in (
            ("visual", (span.latent_start, span.latent_end)),
            ("textual", (span.prev_text_start, span.prev_text_end)),
        ):
            length = ids.shape[1]
            visible = torch.ones(
                length, length, dtype=torch.bool, device=ids.device
            ).tril()
            visible[start - 1 : end - 1, blocked[0] : blocked[1]] = False
            mask = torch.zeros(length, length, device=ids.device).masked_fill(
                ~visible, torch.finfo(torch.float32).min
            )[None, None]
            output = model(
                **kwargs,
                stage2_final_attention_mask=mask,
                use_cache=False,
            )
            off_logp, off_entropy = scored(
                output.logits[0, start - 1 : end - 1], ids[0, start:end]
            )
            preference_losses.append(
                -F.logsigmoid(0.1 * (clean_logp - off_logp))
            )
            entropy_losses.append(
                F.relu(0.1 - (off_entropy - clean_entropy)).mean()
            )
            scores[(span.step_id, modality)] = (
                off_logp,
                off_entropy,
                output.logits[0, start - 1 : end - 1],
            )

    total = (
        clean.loss
        + torch.stack(preference_losses).mean()
        + 0.1 * torch.stack(entropy_losses).mean()
    )
    return total, clean, scores


def _context_plan(batch):
    builder = getattr(sidecar, "build_context_sidecar_plan", None)
    assert callable(builder), "context sidecar planner missing"
    return builder(
        clean_input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        spans=batch["reasoning_step_spans"],
    )


@pytest.mark.parametrize("checkpoint", [False, True])
def test_context_matches_literal_loss_and_all_trainable_gradients(checkpoint):
    """Catches planner/replay/reducer errors with an independent full-prefix oracle."""
    model, batch = two_step_case()
    reference = copy.deepcopy(model)
    if checkpoint:
        for current in (model, reference):
            current.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    actual, output = _trainer(context_config()).compute_loss(
        model, batch, return_outputs=True
    )
    expected, clean, scores = literal_context_oracle(reference, batch)
    loss_error = float((actual.detach() - expected.detach()).abs())
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(output.logits, clean.logits, atol=1e-5, rtol=1e-4)
    for result in output.stage2_sidecar_results:
        modality = {
            "visual_off": "visual",
            "textual_off": "textual",
        }[result["kind"]]
        mean_logp, entropies, _ = scores[(result["step_id"], modality)]
        torch.testing.assert_close(
            result["mean_logp"], mean_logp, atol=1e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            result["entropies"], entropies, atol=1e-5, rtol=1e-4
        )

    actual.backward()
    expected.backward()
    max_gradient_error = 0.0
    trainable_count = 0
    for (name, parameter), (other_name, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert name == other_name
        if parameter.requires_grad:
            trainable_count += 1
            assert parameter.grad is not None and reference_parameter.grad is not None, name
            max_gradient_error = max(
                max_gradient_error,
                float((parameter.grad - reference_parameter.grad).abs().max()),
            )
            torch.testing.assert_close(
                parameter.grad,
                reference_parameter.grad,
                atol=1e-5,
                rtol=1e-4,
                msg=name,
            )
    print(
        {
            "checkpoint": checkpoint,
            "loss_abs_error": loss_error,
            "max_trainable_gradient_abs_error": max_gradient_error,
            "trainable_parameters_checked": trainable_count,
        }
    )


def test_context_visibility_has_only_own_ancestors_and_branch_past():
    """Catches access to clean future, a sibling branch, or the blocked context."""
    _, batch = two_step_case()
    plan = sidecar.build_context_sidecar_plan(
        clean_input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        spans=batch["reasoning_step_spans"],
    )

    for group, blocked_ranges in zip(
        plan.groups,
        (((2, 4), (8, 10)), ((0, 1), (5, 7))),
    ):
        expected = torch.zeros(7, 23, dtype=torch.bool)
        for low, high, ancestor_end, blocked in (
            (0, 2, 4, blocked_ranges[0]),
            (2, 7, 10, blocked_ranges[1]),
        ):
            for row in range(low, high):
                expected[row, :ancestor_end] = True
                expected[row, blocked[0] : blocked[1]] = False
                expected[row, 16 + low : 16 + row + 1] = True

        actual = sidecar.sidecar_dense_visibility(16, group, device="cpu")
        assert torch.equal(actual, expected)
        block_mask = sidecar.build_sidecar_flex_block_mask(16, group, "cpu")
        probe = block_mask.mask_mod(
            torch.tensor(0),
            torch.tensor(0),
            torch.arange(7)[:, None],
            torch.arange(23)[None, :],
        )
        assert torch.equal(probe, expected)


def test_runtime_prefix_bank_identity_and_mrope_reuse():
    """Catches copied/detached prefix KV or collapsed one-dimensional positions."""
    model, batch = two_step_case()
    clean_banks, sidecar_banks, sidecar_positions = [], [], []

    def observe(_module, _args, kwargs, output):
        if kwargs.get("stage2_capture_kv"):
            clean_banks.append(output.stage2_key_values)
        if kwargs.get("stage2_prefix_key_values") is not None:
            sidecar_banks.append(kwargs["stage2_prefix_key_values"])
            sidecar_positions.append(kwargs["position_ids"])

    handle = model.model.register_forward_hook(observe, with_kwargs=True)
    try:
        _trainer(context_config()).compute_loss(model, batch)
    finally:
        handle.remove()

    assert len(clean_banks) == 1
    assert len(sidecar_banks) == 2
    for layer, clean_pair in enumerate(clean_banks[0]):
        for index, tensor in enumerate(clean_pair):
            for bank in sidecar_banks:
                torch.testing.assert_close(bank[layer][index], tensor)
                assert bank[layer][index].requires_grad
            assert (
                sidecar_banks[0][layer][index].data_ptr()
                == sidecar_banks[1][layer][index].data_ptr()
            )
    expected_positions = batch["position_ids"][:, :, [4, 5, 10, 11, 12, 13, 14]]
    for positions in sidecar_positions:
        assert torch.equal(positions, expected_positions)


@pytest.mark.parametrize(
    ("modality", "group_index"),
    (("visual", 0), ("textual", 1)),
)
def test_first_prediction_changes_without_propagating_to_clean_or_other_sidecars(
    modality, group_index
):
    """Catches a copied boundary predictor or propagation outside one local branch."""
    model, batch = two_step_case()
    plan = sidecar.build_context_sidecar_plan(
        clean_input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        spans=batch["reasoning_step_spans"],
    )
    first = replace(
        plan.groups[group_index][0], blocked_start=0, blocked_end=0
    )
    changed_group = (first, *plan.groups[group_index][1:])
    changed_groups = list(plan.groups)
    changed_groups[group_index] = changed_group
    altered = replace(plan, groups=tuple(changed_groups))

    outputs, projected = [], []
    for current_plan in (plan, altered):
        head_outputs = []

        def observe_head(_module, _args, output):
            head_outputs.append(output)

        handle = model.lm_head.register_forward_hook(observe_head)
        try:
            outputs.append(
                model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                    use_cache=False,
                    stage2_shared_prefix_inputs={
                        "plan": current_plan,
                        "backend": "dense",
                        "checkpoint": False,
                    },
                )
            )
        finally:
            handle.remove()
        projected.append(head_outputs)

    expected_result_order = [
        "step-0-visual-off",
        "step-1-visual-off",
        "step-0-textual-off",
        "step-1-textual-off",
    ]
    for output, head_outputs in zip(outputs, projected):
        # The observed row counts distinguish the two grouped sidecar projections
        # from the full clean projection and establish their runtime order.
        assert [head.shape[1] for head in head_outputs] == [7, 7, 16]
        assert [
            result["name"] for result in output.stage2_sidecar_results
        ] == expected_result_order
        for observed_group, expected_modality in enumerate(("visual", "textual")):
            offset = 0
            for span in batch["reasoning_step_spans"]:
                target_count = span.target_text_end - span.target_text_start
                logits = head_outputs[observed_group][0, offset : offset + target_count]
                targets = batch["input_ids"][0, span.target_text_start : span.target_text_end]
                log_probs = logits.float().log_softmax(-1)
                result = {
                    item["name"]: item for item in output.stage2_sidecar_results
                }[f"step-{span.step_id}-{expected_modality}-off"]
                torch.testing.assert_close(
                    result["mean_logp"],
                    log_probs.gather(-1, targets[:, None]).mean(),
                )
                torch.testing.assert_close(
                    result["entropies"],
                    -(log_probs.exp() * log_probs).sum(-1),
                )
                offset += target_count

    first_token_change = torch.linalg.vector_norm(
        projected[0][group_index][0, 0] - projected[1][group_index][0, 0]
    )
    assert first_token_change > 1e-8
    torch.testing.assert_close(outputs[0].logits, outputs[1].logits)
    original = {
        result["name"]: result for result in outputs[0].stage2_sidecar_results
    }
    changed = {
        result["name"]: result for result in outputs[1].stage2_sidecar_results
    }
    changed_name = f"step-0-{modality}-off"
    for name in original:
        if name != changed_name:
            torch.testing.assert_close(
                original[name]["mean_logp"], changed[name]["mean_logp"]
            )
            torch.testing.assert_close(
                original[name]["entropies"], changed[name]["entropies"]
            )
    print(
        {
            "unblocked": modality,
            "first_token_logit_l2_change": float(first_token_change),
            "head_rows_in_runtime_order": [
                head.shape[1] for head in projected[0]
            ],
        }
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
