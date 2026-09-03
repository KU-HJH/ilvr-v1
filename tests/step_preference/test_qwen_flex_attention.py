import copy
import os

import pytest
import torch

from step_preference.packing import (
    PackedBranch,
    PackedTreePlan,
    build_flex_block_mask,
    dense_visibility,
)
from tests.stage2.helpers import tiny_stage2_qwen


def _tiny_flex_qwen(*, attn_implementation: str):
    template = tiny_stage2_qwen(attn_implementation=attn_implementation)
    config = copy.deepcopy(template.config)
    config.hidden_size = 32
    config.intermediate_size = 64
    config.rope_scaling["mrope_section"] = [3, 3, 2]
    return type(template)(config)


def _one_block_tree_plan() -> PackedTreePlan:
    root_length = 124
    packed_length = 128
    branch_length = packed_length - root_length
    logical_sources = torch.cat(
        (torch.arange(root_length), torch.arange(4, 4 + branch_length))
    )
    return PackedTreePlan(
        root_length=root_length,
        unpadded_length=packed_length,
        padded_length=packed_length,
        token_ids=torch.arange(packed_length),
        clean_embedding_sources=torch.arange(packed_length),
        logical_position_sources=logical_sources,
        branch_ids=torch.cat(
            (torch.zeros(root_length, dtype=torch.long), torch.ones(branch_length, dtype=torch.long))
        ),
        branch_local_positions=torch.cat(
            (torch.arange(root_length), torch.arange(branch_length))
        ),
        clean_ancestor_ends=torch.cat(
            (torch.zeros(root_length, dtype=torch.long), torch.full((branch_length,), 4))
        ),
        blocked_starts=torch.cat(
            (torch.full((root_length,), -1), torch.ones(branch_length, dtype=torch.long))
        ),
        blocked_ends=torch.cat(
            (torch.full((root_length,), -1), torch.full((branch_length,), 2))
        ),
        valid_slots=torch.ones(packed_length, dtype=torch.bool),
        root_selected_positions=(),
        branches=(
            PackedBranch(
                name="copied-sidecar",
                kind="visual_off",
                step_id=0,
                packed_start=root_length,
                packed_end=packed_length,
                clean_ancestor_end=4,
                blocked_start=1,
                blocked_end=2,
                logical_query_positions=(4, 5, 6, 7),
                selected_positions=(124, 125, 126, 127),
                target_token_ids=(0, 0, 0, 0),
            ),
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
def test_qwen_flex_decoder_matches_dense_tree_mask_with_copied_positions():
    """Catches Flex bypassing tree visibility, copied RoPE positions, or decoder layers."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    handles = []
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(101)

        dense_model = _tiny_flex_qwen(attn_implementation="eager").cuda().float().train()
        flex_model = copy.deepcopy(dense_model)
        plan = _one_block_tree_plan()
        visible = dense_visibility(plan).cuda()
        dense_mask = torch.where(
            visible,
            0.0,
            torch.finfo(torch.float32).min,
        )[None, None]
        block_mask = build_flex_block_mask(plan, torch.device("cuda"))
        logical_positions = (
            plan.logical_position_sources.view(1, 1, -1).expand(3, 1, -1).clone().cuda()
        )
        embeds = torch.linspace(
            -0.75,
            0.75,
            steps=plan.padded_length * dense_model.config.hidden_size,
            device="cuda",
            dtype=torch.float32,
        ).view(1, plan.padded_length, dense_model.config.hidden_size)

        received_masks = []

        def record_layer_mask(_module, _args, kwargs):
            received_masks.append(kwargs["attention_mask"])

        handles = [
            layer.register_forward_pre_hook(record_layer_mask, with_kwargs=True)
            for layer in flex_model.model.layers
        ]

        dense_out = dense_model.model(
            inputs_embeds=embeds,
            attention_mask=dense_mask,
            position_ids=logical_positions,
            use_cache=False,
            return_dict=True,
        )
        flex_out = flex_model.model(
            inputs_embeds=embeds.clone(),
            attention_mask=block_mask,
            position_ids=logical_positions,
            use_cache=False,
            return_dict=True,
        )

        torch.testing.assert_close(
            flex_out.last_hidden_state,
            dense_out.last_hidden_state,
            rtol=1e-5,
            atol=1e-5,
        )
        dense_out.last_hidden_state.square().mean().backward()
        flex_out.last_hidden_state.square().mean().backward()

        assert len(received_masks) == len(flex_model.model.layers)
        assert all(mask is block_mask for mask in received_masks)
        for dense_layer, flex_layer in zip(
            dense_model.model.layers, flex_model.model.layers, strict=True
        ):
            dense_parameters = dict(dense_layer.named_parameters())
            flex_parameters = dict(flex_layer.named_parameters())
            for parameter_name in (
                "self_attn.q_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.down_proj.weight",
            ):
                dense_grad = dense_parameters[parameter_name].grad
                flex_grad = flex_parameters[parameter_name].grad
                assert dense_grad is not None and flex_grad is not None
                torch.testing.assert_close(
                    flex_grad.float(), dense_grad.float(), rtol=1e-5, atol=1e-5
                )
    finally:
        for handle in handles:
            handle.remove()
        torch.use_deterministic_algorithms(deterministic, warn_only=deterministic_warn_only)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        if cublas_workspace_config is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
