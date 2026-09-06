"""Differentiable Stage 2 suffix replay and checkpoint-local vocabulary scoring."""

from collections.abc import Mapping

import torch
from torch.utils.checkpoint import checkpoint


def validate_shared_inputs(model, options, input_ids, attention_mask, *, packed,
                           final_mask, return_dict, use_cache, past_key_values,
                           output_attentions):
    from step_preference.sidecar import SidecarBranch, SidecarPlan, chunk_sidecar_branches

    if not isinstance(options, Mapping):
        raise TypeError("stage2_shared_prefix_inputs must be a mapping")
    if set(options) - {"plan", "backend", "query_budget", "checkpoint"}:
        raise ValueError("unknown shared-prefix input options")
    if not isinstance(options.get("plan"), SidecarPlan):
        raise ValueError("shared-prefix inputs require a SidecarPlan")
    plan = options["plan"]
    if plan.context_preference:
        expected_kinds = ("visual_off", "textual_off")
        if not isinstance(plan.groups, tuple) or len(plan.groups) != len(expected_kinds):
            raise ValueError("context preference permits only local chosen branches")
        for group, expected_kind in zip(plan.groups, expected_kinds):
            if not isinstance(group, tuple) or any(
                not isinstance(branch, SidecarBranch)
                or branch.kind != expected_kind
                or branch.rejected_model_inputs is not None
                for branch in group
            ):
                raise ValueError("context preference permits only local chosen branches")
    if options.get("backend", "dense") not in ("dense", "flex"):
        raise ValueError("shared-prefix backend must be dense or flex")
    if options.get("backend") == "flex" and model.training and model.config.attention_dropout:
        raise ValueError("Flex sidecars require zero attention dropout; adapter dropout remains independent")
    if type(options.get("checkpoint", True)) is not bool:
        raise ValueError("shared-prefix checkpoint must be a bool")
    chunk_sidecar_branches((), options.get("query_budget", 1024))
    if packed or final_mask is not None:
        raise ValueError("shared-prefix inputs cannot be combined with packed inputs or a final mask")
    if not return_dict or use_cache or past_key_values is not None or output_attentions:
        raise ValueError("shared-prefix Stage 2 requires return_dict=True, use_cache=False, no cache or attentions")
    if model.config._attn_implementation not in ("eager", "sdpa"):
        raise ValueError("shared-prefix Stage 2 requires eager or sdpa language attention")
    mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
    if mask.shape != input_ids.shape or mask.ndim != 2:
        raise ValueError("shared-prefix attention_mask must match input_ids")
    expected = torch.arange(input_ids.shape[1], device=mask.device) < plan.root_length
    if not torch.equal(mask.bool()[0], expected):
        raise ValueError("shared-prefix root_length must match a right-padded clean input")


def _branch_inputs(model, branch, input_ids, clean_embeds, clean_positions):
    device = clean_embeds.device
    logical = torch.tensor(branch.logical_positions, dtype=torch.long, device=device)
    if branch.rejected_model_inputs is None:
        sources = torch.tensor(branch.clean_embedding_sources, dtype=torch.long, device=device)
        return clean_embeds.index_select(1, sources), clean_positions.index_select(2, logical)

    rejected = branch.rejected_model_inputs
    rejected_ids = rejected["input_ids"].to(device)
    rejected_mask = rejected["attention_mask"].to(device)
    grid = rejected.get("image_grid_thw")
    if grid is not None:
        grid = grid.to(device)
    positions, _ = model.get_rope_index(rejected_ids, image_grid_thw=grid, attention_mask=rejected_mask)
    end = branch.clean_ancestor_end
    if not torch.equal(input_ids[:, :end], rejected_ids[:, :end]):
        raise ValueError("rejected shared ancestors must have matching clean tokens")
    if not torch.equal(clean_positions[:, :, :end], positions[:, :, :end]):
        raise ValueError("rejected shared ancestors must have matching mRoPE positions")
    token_ids = torch.tensor(branch.token_ids, dtype=torch.long, device=device)[None]
    # Only text and the shared latent-end boundary may be replayed directly.
    forbidden = (model.config.image_token_id, model.config.video_token_id,
                 model.config.vision_start_token_id, model.config.latent_pad_token_id,
                 model.config.latent_start_token_id)
    if any(bool((token_ids == token_id).any()) for token_id in forbidden):
        raise ValueError("rejected replay cannot introduce image or unfinished latent inputs")
    return model.model.embed_tokens(token_ids), positions.index_select(2, logical)


def _score_chunk(model, branches, embeds, positions, mask, flat_prefix):
    """Factory binds metadata per chunk, including during delayed backward replay."""
    selected = []
    targets = []
    lengths = []
    offset = 0
    for branch in branches:
        selected.extend(offset + position for position in branch.selected_positions)
        targets.extend(branch.target_token_ids)
        lengths.append(len(branch.target_token_ids))
        offset += len(branch.token_ids)
    selected = torch.tensor(selected, dtype=torch.long, device=embeds.device)
    targets = torch.tensor(targets, dtype=torch.long, device=embeds.device)

    def score(embeds, positions, *prefix_tensors):
        hidden = model.model(
            inputs_embeds=embeds, position_ids=positions, attention_mask=mask,
            stage2_prefix_key_values=tuple(zip(prefix_tensors[::2], prefix_tensors[1::2])),
            use_cache=False, return_dict=True,
        ).last_hidden_state
        logits = model.lm_head(hidden.index_select(1, selected))[0].float()
        log_probs = logits.log_softmax(-1)
        target_log_probs = log_probs.gather(-1, targets[:, None]).squeeze(-1)
        entropies = -(log_probs.exp() * log_probs).sum(-1)
        reduced = []
        start = 0
        for length in lengths:
            reduced.extend((target_log_probs[start:start+length].mean(), entropies[start:start+length]))
            start += length
        return tuple(reduced)

    return score, (embeds, positions, *flat_prefix)


def score_sidecars(model, options, input_ids, clean_embeds, clean_positions, prefix_kv):
    from step_preference.sidecar import (
        chunk_sidecar_branches, sidecar_dense_visibility, build_sidecar_flex_block_mask,
    )

    plan = options["plan"]
    budget = options.get("query_budget", 1024)
    # The final clean decoder may contain right padding; never expose padded K/V.
    flat_prefix = tuple(tensor[:, :, :plan.root_length] for pair in prefix_kv for tensor in pair)
    results = []
    stats = dict(logical_groups=1 + sum(bool(group) for group in plan.groups), chunks=0,
                 oversized_branches=0, query_rows=0, entropy_scored_rows=0,
                 rejected_scored_rows=0)
    for group in plan.groups:
        batches = ((group,) if group else ()) if plan.context_preference else chunk_sidecar_branches(group, budget)
        for branches in batches:
            parts = [_branch_inputs(model, branch, input_ids, clean_embeds, clean_positions) for branch in branches]
            embeds = torch.cat([part[0] for part in parts], dim=1)
            positions = torch.cat([part[1] for part in parts], dim=2)
            if options.get("backend", "dense") == "flex":
                mask = build_sidecar_flex_block_mask(plan.root_length, branches, embeds.device)
            else:
                visible = sidecar_dense_visibility(plan.root_length, branches, device=embeds.device)
                mask = torch.zeros(visible.shape, dtype=embeds.dtype, device=embeds.device)
                mask = mask.masked_fill(~visible, torch.finfo(embeds.dtype).min)[None, None]
            score, args = _score_chunk(model, branches, embeds, positions, mask, flat_prefix)
            values = checkpoint(score, *args, use_reentrant=False, preserve_rng_state=True) if options.get("checkpoint", True) else score(*args)
            for index, branch in enumerate(branches):
                results.append(dict(name=branch.name, kind=branch.kind, step_id=branch.step_id,
                                    mean_logp=values[index*2], entropies=values[index*2+1]))
                key = "rejected_scored_rows" if branch.rejected_model_inputs is not None else "entropy_scored_rows"
                stats[key] += len(branch.target_token_ids)
                stats["oversized_branches"] += len(branch.token_ids) > budget
            stats["chunks"] += 1
            stats["query_rows"] += embeds.shape[1]
    return tuple(results), stats
