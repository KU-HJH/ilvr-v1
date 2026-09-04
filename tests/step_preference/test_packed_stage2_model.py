import copy
from unittest.mock import patch

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from step_preference.ablation import (
    StepAblationSpec,
    build_additive_step_attention_mask,
)
from step_preference.negatives import StepNegative
from step_preference.packing import build_packed_tree_plan
from step_preference.spans import ReasoningStepSpan


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
from tests.stage2.helpers import tiny_stage2_qwen
from transformers.cache_utils import DynamicCache


def _tiny_packed_qwen():
    template = tiny_stage2_qwen(attn_implementation="eager")
    config = copy.deepcopy(template.config)
    config.hidden_size = 32
    config.intermediate_size = 64
    config.rope_scaling["mrope_section"] = [3, 3, 2]
    return type(template)(config)


def _packed_fixture():
    clean_ids = torch.tensor([[11, 91, 90, 90, 92, 12, 13, 14, 15]])
    rejected_ids = torch.tensor([[11, 91, 90, 90, 92, 12, 23, 24, 15]])
    span = ReasoningStepSpan(0, 0, 1, 2, 4, 6, 8, None)
    negative = StepNegative(
        sample_id="packed-model",
        step_id=0,
        chosen_text="chosen",
        rejected_visual_text="rejected",
        rejected_textual_text=None,
        valid_visual_negative=True,
        valid_textual_negative=False,
        reference_chosen_score=-0.1,
        reference_visual_rejected_score=-0.2,
        reference_textual_rejected_score=None,
    )
    rejected = {
        "model_inputs": {
            "input_ids": rejected_ids,
            "attention_mask": torch.ones_like(rejected_ids),
        },
        "completion_token_positions": (6, 7),
        "completion_token_ids": (23, 24),
        "rendered_text": "fixture",
    }
    plan = build_packed_tree_plan(
        clean_input_ids=clean_ids,
        attention_mask=torch.ones_like(clean_ids),
        reasoning_spans=(span,),
        joined_steps=(
            {
                "span": span,
                "negative": negative,
                "visual_rejected": rejected,
                "textual_rejected": None,
            },
        ),
    )
    return clean_ids, rejected_ids, span, plan


def _latent_end_branch_fixture():
    clean_ids = torch.tensor([[11, 91, 90, 90, 92, 13, 14, 15]])
    rejected_ids = torch.tensor([[11, 91, 90, 90, 92, 23, 24, 15]])
    span = ReasoningStepSpan(0, 0, 1, 2, 4, 5, 7, None)
    negative = StepNegative(
        sample_id="latent-end-branch",
        step_id=0,
        chosen_text="chosen",
        rejected_visual_text="rejected",
        rejected_textual_text=None,
        valid_visual_negative=True,
        valid_textual_negative=False,
        reference_chosen_score=-0.1,
        reference_visual_rejected_score=-0.2,
        reference_textual_rejected_score=None,
    )
    plan = build_packed_tree_plan(
        clean_input_ids=clean_ids,
        attention_mask=torch.ones_like(clean_ids),
        reasoning_spans=(span,),
        joined_steps=(
            {
                "span": span,
                "negative": negative,
                "visual_rejected": {
                    "model_inputs": {
                        "input_ids": rejected_ids,
                        "attention_mask": torch.ones_like(rejected_ids),
                    },
                    "completion_token_positions": (5, 6),
                    "completion_token_ids": (23, 24),
                    "rendered_text": "latent-end rejected",
                },
                "textual_rejected": None,
            },
        ),
    )
    return clean_ids, rejected_ids, plan


def _independent_selected_logits(model, clean_ids, rejected_ids, span, plan):
    attention_mask = torch.ones_like(clean_ids)
    root_positions = torch.tensor(plan.root_selected_positions, device=clean_ids.device)
    with torch.no_grad():
        clean = model(
            input_ids=clean_ids,
            attention_mask=attention_mask,
            labels=clean_ids,
            return_stage2_latent_states=True,
        )
        selected = [clean.logits.index_select(1, root_positions)]
        for branch, modality, source in (
            (plan.branch("step-0-visual-off"), "visual", (span.latent_start, span.latent_end)),
            (plan.branch("step-0-textual-off"), "textual", (span.prev_text_start, span.prev_text_end)),
        ):
            final_mask = build_additive_step_attention_mask(
                attention_mask,
                StepAblationSpec(
                    modality,
                    span.target_text_start,
                    span.target_text_end,
                    source[0],
                    source[1],
                ),
                dtype=model.dtype,
                device=model.device,
                attn_implementation="eager",
            )
            output = model(
                input_ids=clean_ids,
                attention_mask=attention_mask,
                stage2_final_attention_mask=final_mask,
                stage2_logit_positions=torch.tensor(
                    branch.logical_query_positions, device=clean_ids.device
                ),
                stage2_selective_logits_only=True,
            )
            selected.append(output.stage2_selected_logits)

        rejected_branch = plan.branch("step-0-visual-rejected")
        rejected = model(
            input_ids=rejected_ids,
            attention_mask=torch.ones_like(rejected_ids),
            stage2_logit_positions=torch.tensor(
                rejected_branch.logical_query_positions, device=clean_ids.device
            ),
            stage2_selective_logits_only=True,
        )
        selected.append(rejected.stage2_selected_logits)
    return clean, torch.cat(selected, dim=1)


def test_packed_stage2_matches_independent_forwards_with_one_rollout_and_final_decode():
    """Catches packed sidecars drifting from eager forwards or repeating Stage 2 work."""
    assert torch.cuda.is_available(), "packed Stage 2 requires a real CUDA FlexAttention run"
    torch.manual_seed(211)
    model = _tiny_packed_qwen().cuda().float().eval()
    clean_ids, rejected_ids, _span, plan = _packed_fixture()
    clean_ids = clean_ids.cuda()
    rejected_ids = rejected_ids.cuda()
    attention_mask = torch.ones_like(clean_ids)
    labels = clean_ids.clone()
    clean, expected = _independent_selected_logits(
        model, clean_ids, rejected_ids, _span, plan
    )
    packed_inputs = plan.to_model_inputs(torch.device("cuda"))
    selected_positions = plan.selected_positions().cuda()
    packed_final_completions = []

    def count_packed_final(_module, _args, output):
        if output.shape[1] == plan.padded_length:
            packed_final_completions.append(output.shape[1])

    norm_handle = model.model.norm.register_forward_hook(count_packed_final)
    original_rollout = model.stage2_rollout_engine.rollout
    try:
        with patch.object(
            model.stage2_rollout_engine, "rollout", wraps=original_rollout
        ) as rollout:
            output = model(
                input_ids=clean_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                output_hidden_states=True,
                stage2_loss_only=True,
                stage2_packed_inputs=packed_inputs,
                stage2_logit_positions=selected_positions,
                return_stage2_latent_states=True,
            )
    finally:
        norm_handle.remove()

    assert rollout.call_count == 1
    assert packed_final_completions == [plan.padded_length]
    assert output.logits is None
    assert output.inputs_embeds.shape == (1, plan.padded_length, model.config.hidden_size)
    assert output.hidden_states[-1].shape == output.inputs_embeds.shape
    assert output.stage2_selected_logits.shape == (
        1,
        selected_positions.numel(),
        model.config.vocab_size,
    )
    torch.testing.assert_close(
        output.stage2_selected_logits.float(), expected.float(), rtol=2e-4, atol=2e-5
    )
    torch.testing.assert_close(output.loss.float(), clean.loss.float(), rtol=2e-5, atol=2e-6)
    assert len(output.stage2_latent_states) == len(clean.stage2_latent_states)
    for packed_state, clean_state in zip(
        output.stage2_latent_states, clean.stage2_latent_states, strict=True
    ):
        torch.testing.assert_close(packed_state, clean_state, rtol=0, atol=0)

    output.stage2_selected_logits.float().square().mean().backward()
    clean_prefix_grad = model.model.embed_tokens.weight.grad[clean_ids[0, 0]]
    q_proj_grad = model.model.layers[0].self_attn.q_proj.weight.grad
    assert torch.isfinite(clean_prefix_grad).all() and clean_prefix_grad.abs().sum() > 0
    assert q_proj_grad is not None
    assert torch.isfinite(q_proj_grad).all() and q_proj_grad.abs().sum() > 0


def test_packed_stage2_accepts_matching_latent_end_rejected_branch_start():
    """Catches valid target-adjacent rejected branches being rejected as control leakage."""
    assert torch.cuda.is_available(), "packed Stage 2 requires CUDA"
    torch.manual_seed(223)
    model = _tiny_packed_qwen().cuda().float().eval()
    clean_ids, rejected_ids, plan = _latent_end_branch_fixture()
    clean_ids = clean_ids.cuda()
    rejected_ids = rejected_ids.cuda()
    branch = plan.branch("step-0-visual-rejected")
    assert plan.token_ids[branch.packed_start].item() == model.config.latent_end_token_id
    assert plan.logical_position_sources[branch.packed_start].item() == 4

    packed = model(
        input_ids=clean_ids,
        attention_mask=torch.ones_like(clean_ids),
        labels=clean_ids,
        use_cache=False,
        stage2_loss_only=True,
        stage2_packed_inputs=plan.to_model_inputs(torch.device("cuda")),
        stage2_logit_positions=plan.selected_positions().cuda(),
    )
    independent = model(
        input_ids=rejected_ids,
        attention_mask=torch.ones_like(rejected_ids),
        use_cache=False,
        stage2_logit_positions=torch.tensor(
            branch.logical_query_positions, device="cuda"
        ),
        stage2_selective_logits_only=True,
    )
    branch_offset = len(plan.root_selected_positions) + sum(
        len(item.selected_positions)
        for item in plan.branches
        if item.packed_start < branch.packed_start
    )
    actual = packed.stage2_selected_logits[
        :, branch_offset : branch_offset + len(branch.selected_positions)
    ]
    torch.testing.assert_close(
        actual.float(), independent.stage2_selected_logits.float(), rtol=2e-4, atol=2e-5
    )


def test_packed_stage2_rejects_latent_end_with_nonmatching_logical_source():
    """Catches broad latent-end exemptions that can conceal malformed sidecars."""
    assert torch.cuda.is_available(), "packed Stage 2 requires CUDA"
    model = _tiny_packed_qwen().cuda().float().eval()
    clean_ids, _rejected_ids, plan = _latent_end_branch_fixture()
    clean_ids = clean_ids.cuda()
    packed_inputs = plan.to_model_inputs(torch.device("cuda"))
    branch = plan.branch("step-0-visual-rejected")
    packed_inputs["logical_position_sources"] = packed_inputs[
        "logical_position_sources"
    ].clone()
    packed_inputs["logical_position_sources"][branch.packed_start] = 0
    with pytest.raises(ValueError, match="control token"):
        _packed_call(
            model,
            clean_ids,
            packed_inputs,
            plan.selected_positions().cuda(),
        )


@pytest.fixture
def packed_case():
    assert torch.cuda.is_available(), "packed Stage 2 requires a real CUDA FlexAttention run"
    model = _tiny_packed_qwen().cuda().float().eval()
    clean_ids, _rejected_ids, _span, plan = _packed_fixture()
    clean_ids = clean_ids.cuda()
    return model, clean_ids, plan, plan.to_model_inputs(torch.device("cuda"))


def _packed_call(model, clean_ids, packed_inputs, positions, **kwargs):
    return model(
        input_ids=clean_ids,
        attention_mask=torch.ones_like(clean_ids),
        labels=clean_ids,
        use_cache=False,
        stage2_loss_only=True,
        stage2_packed_inputs=packed_inputs,
        stage2_logit_positions=positions,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("not-mapping", "mapping"),
        ("missing-key", "exact keys"),
        ("extra-key", "exact keys"),
        ("short-vector", "same nonzero length"),
        ("matrix-vector", "one-dimensional"),
        ("wrong-long-dtype", "torch.long"),
        ("wrong-bool-dtype", "torch.bool"),
        ("wrong-device", "same CUDA device"),
        ("cpu-block-mask", "CUDA BlockMask"),
        ("negative-logical", "logical_position_sources"),
        ("large-logical", "logical_position_sources"),
        ("small-clean-source", "clean_embedding_sources"),
        ("large-clean-source", "clean_embedding_sources"),
        ("root-length", "root_length"),
        ("mutated-root-token", "clean root"),
        ("mutated-root-source", "clean root"),
        ("mutated-root-logical", "clean root"),
        ("control-direct-token", "control token"),
        ("nonzero-padding", "padding token_ids"),
        ("noncanonical-padding-source", "padding metadata"),
        ("noncanonical-padding-logical", "padding metadata"),
        ("noncontiguous-valid-slots", "contiguous prefix"),
    ),
)
def test_packed_stage2_rejects_malformed_packed_boundaries(
    packed_case, mutation, message
):
    """Catches malformed packed metadata crossing the one-decoder boundary."""
    model, clean_ids, plan, packed_inputs = packed_case
    packed_inputs = dict(packed_inputs)
    if mutation == "not-mapping":
        packed_inputs = []
    elif mutation == "missing-key":
        packed_inputs.pop("valid_slots")
    elif mutation == "extra-key":
        packed_inputs["extra"] = None
    elif mutation == "short-vector":
        packed_inputs["token_ids"] = packed_inputs["token_ids"][:-1]
    elif mutation == "matrix-vector":
        packed_inputs["logical_position_sources"] = packed_inputs[
            "logical_position_sources"
        ].unsqueeze(0)
    elif mutation == "wrong-long-dtype":
        packed_inputs["clean_embedding_sources"] = packed_inputs[
            "clean_embedding_sources"
        ].to(torch.int32)
    elif mutation == "wrong-bool-dtype":
        packed_inputs["valid_slots"] = packed_inputs["valid_slots"].long()
    elif mutation == "wrong-device":
        packed_inputs["token_ids"] = packed_inputs["token_ids"].cpu()
    elif mutation == "cpu-block-mask":
        packed_inputs["block_mask"] = create_block_mask(
            lambda _b, _h, q, kv: q >= kv,
            B=1,
            H=None,
            Q_LEN=plan.padded_length,
            KV_LEN=plan.padded_length,
            device="cpu",
            BLOCK_SIZE=128,
        )
    elif mutation == "negative-logical":
        packed_inputs["logical_position_sources"] = packed_inputs[
            "logical_position_sources"
        ].clone()
        packed_inputs["logical_position_sources"][0] = -1
    elif mutation == "large-logical":
        packed_inputs["logical_position_sources"] = packed_inputs[
            "logical_position_sources"
        ].clone()
        packed_inputs["logical_position_sources"][0] = clean_ids.shape[1]
    elif mutation == "small-clean-source":
        packed_inputs["clean_embedding_sources"] = packed_inputs[
            "clean_embedding_sources"
        ].clone()
        packed_inputs["clean_embedding_sources"][0] = -2
    elif mutation == "large-clean-source":
        packed_inputs["clean_embedding_sources"] = packed_inputs[
            "clean_embedding_sources"
        ].clone()
        packed_inputs["clean_embedding_sources"][0] = clean_ids.shape[1]
    elif mutation == "root-length":
        packed_inputs["root_length"] = clean_ids.shape[1] - 1
    elif mutation == "mutated-root-token":
        packed_inputs["token_ids"] = packed_inputs["token_ids"].clone()
        packed_inputs["token_ids"][0] = int(clean_ids[0, 0]) + 1
    elif mutation == "mutated-root-source":
        packed_inputs["clean_embedding_sources"] = packed_inputs[
            "clean_embedding_sources"
        ].clone()
        packed_inputs["clean_embedding_sources"][0] = 1
    elif mutation == "mutated-root-logical":
        packed_inputs["logical_position_sources"] = packed_inputs[
            "logical_position_sources"
        ].clone()
        packed_inputs["logical_position_sources"][0] = 1
    elif mutation == "control-direct-token":
        direct = torch.nonzero(
            packed_inputs["valid_slots"]
            & (packed_inputs["clean_embedding_sources"] == -1)
            & (torch.arange(plan.padded_length, device=clean_ids.device) >= plan.root_length)
        )[0, 0]
        packed_inputs["token_ids"] = packed_inputs["token_ids"].clone()
        packed_inputs["token_ids"][direct] = model.config.image_token_id
    elif mutation == "nonzero-padding":
        padding = torch.nonzero(~packed_inputs["valid_slots"])[0, 0]
        packed_inputs["token_ids"] = packed_inputs["token_ids"].clone()
        packed_inputs["token_ids"][padding] = 1
    elif mutation == "noncanonical-padding-source":
        padding = torch.nonzero(~packed_inputs["valid_slots"])[0, 0]
        packed_inputs["clean_embedding_sources"] = packed_inputs[
            "clean_embedding_sources"
        ].clone()
        packed_inputs["clean_embedding_sources"][padding] = 0
    elif mutation == "noncanonical-padding-logical":
        padding = torch.nonzero(~packed_inputs["valid_slots"])[0, 0]
        packed_inputs["logical_position_sources"] = packed_inputs[
            "logical_position_sources"
        ].clone()
        packed_inputs["logical_position_sources"][padding] = 1
    elif mutation == "noncontiguous-valid-slots":
        packed_inputs["valid_slots"] = packed_inputs["valid_slots"].clone()
        packed_inputs["valid_slots"][plan.root_length] = False

    with pytest.raises((TypeError, ValueError), match=message):
        _packed_call(
            model,
            clean_ids,
            packed_inputs,
            plan.selected_positions().cuda(),
        )


@pytest.mark.parametrize(
    ("positions", "message"),
    (
        (torch.tensor([], dtype=torch.long), "nonempty"),
        (None, "requires stage2_logit_positions"),
        (torch.tensor([5, 5]), "strictly increasing"),
        (torch.tensor([6, 5]), "strictly increasing"),
        (torch.tensor([128]), "packed sequence"),
        (torch.tensor([127]), "valid packed slots"),
    ),
)
def test_packed_stage2_rejects_invalid_selected_positions(
    packed_case, positions, message
):
    """Catches ambiguous or non-packed selected-logit ordering."""
    model, clean_ids, _plan, packed_inputs = packed_case
    if positions is not None:
        positions = positions.cuda()
    with pytest.raises((TypeError, ValueError), match=message):
        _packed_call(model, clean_ids, packed_inputs, positions)


@pytest.mark.parametrize(
    ("option", "message"),
    (
        ({"use_cache": True}, "use_cache=False"),
        ({"past_key_values": DynamicCache()}, "past_key_values"),
        ({"output_attentions": True}, "output_attentions=False"),
        ({"return_dict": False}, "return_dict=True"),
        ({"stage2_final_attention_mask": torch.zeros(1)}, "cannot be combined"),
    ),
)
def test_packed_stage2_rejects_incompatible_decoder_options(
    packed_case, option, message
):
    """Catches packed Flex dispatch through an unsupported decoder mode."""
    model, clean_ids, plan, packed_inputs = packed_case
    call = dict(option)
    use_cache = call.pop("use_cache", False)
    with pytest.raises((TypeError, ValueError), match=message):
        model(
            input_ids=clean_ids,
            attention_mask=torch.ones_like(clean_ids),
            labels=clean_ids,
            use_cache=use_cache,
            stage2_loss_only=True,
            stage2_packed_inputs=packed_inputs,
            stage2_logit_positions=plan.selected_positions().cuda(),
            **call,
        )


def test_packed_stage2_rejects_active_gradient_checkpointing(packed_case):
    """Catches a BlockMask entering the unspecified checkpoint wrapper."""
    model, clean_ids, plan, packed_inputs = packed_case
    model.gradient_checkpointing_enable()
    with pytest.raises(ValueError, match="gradient checkpointing"):
        _packed_call(
            model,
            clean_ids,
            packed_inputs,
            plan.selected_positions().cuda(),
        )
