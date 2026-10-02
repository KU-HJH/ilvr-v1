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
    shared_option_names = {
        "plan", "backend", "query_budget", "checkpoint", "prefix_mode",
        "branch_mode", "block_original_image", "dil_objective",
    }
    replacement_option_names = {
        "sample_id", "replacement_seed", "replacement_version",
    }
    prefix_mode = options.get("prefix_mode", "clean_prefix")
    allowed_option_names = shared_option_names | (
        replacement_option_names
        if prefix_mode == "evidence_replacement" else set()
    )
    if set(options) - allowed_option_names:
        raise ValueError("unknown shared-prefix input options")
    if not isinstance(options.get("plan"), SidecarPlan):
        raise ValueError("shared-prefix inputs require a SidecarPlan")
    plan = options["plan"]
    dil = options.get("dil_objective")
    if dil is not None:
        import math
        if (not isinstance(dil, Mapping) or set(dil) != {"loss_type", "beta", "k"}
                or dil["loss_type"] not in ("dil_bce_rf_topk", "dil_bce_rf_topk_reverse", "dil_bce_rf_topk_both_upper")
                or type(dil["beta"]) not in (int, float)
                or not math.isfinite(dil["beta"]) or dil["beta"] <= 0
                or type(dil["k"]) is not int
                or not 2 <= dil["k"] < model.get_output_embeddings().weight.shape[0]):
            raise ValueError("invalid DIL objective options")
        if (not plan.context_preference or plan.groups[1]
                or prefix_mode != "clean_prefix"
                or options.get("branch_mode", "independent") != "independent"
                or options.get("block_original_image", False)
                or any(b.zero_first_embedding for b in plan.groups[0])):
            raise ValueError("DIL requires V-only independent clean-prefix branches")
        aligned_positions = []
        for step_id, branch in enumerate(plan.groups[0]):
            predictor_positions = tuple(branch.logical_positions[i] for i in branch.selected_positions)
            if (branch.step_id != step_id or branch.name != f"step-{step_id}-visual-off"
                    or branch.clean_embedding_sources != branch.logical_positions
                    or branch.token_ids != tuple(int(input_ids[0, p]) for p in branch.logical_positions)
                    or any(p < 0 or p+1 >= plan.root_length for p in predictor_positions)
                    or branch.target_token_ids != tuple(int(input_ids[0, p+1]) for p in predictor_positions)):
                raise ValueError("DIL branch identity/target alignment mismatch")
            aligned_positions.extend(predictor_positions)
        if tuple(aligned_positions) != plan.root_selected_positions:
            raise ValueError("DIL root predictor alignment mismatch")
    if prefix_mode not in ("clean_prefix", "v_off_prefix", "v_off_insertion", "evidence_replacement"):
        raise ValueError("unknown preference prefix_mode")
    branch_mode = options.get("branch_mode", "independent")
    if branch_mode not in ("independent", "cumulative"):
        raise ValueError("unknown preference branch_mode")
    if branch_mode == "cumulative" and prefix_mode != "v_off_insertion":
        raise ValueError("cumulative branch_mode requires v_off_insertion")
    if type(options.get("block_original_image", False)) is not bool:
        raise ValueError("block_original_image must be a boolean")
    if options.get("block_original_image", False) and prefix_mode != "v_off_prefix":
        raise ValueError("block_original_image requires v_off_prefix")
    if prefix_mode in ("v_off_prefix", "v_off_insertion", "evidence_replacement") and (
        not plan.context_preference or plan.groups[1]
        or any(branch.zero_first_embedding for branch in plan.groups[0])
    ):
        raise ValueError(f"{prefix_mode} requires V-only context preference with START/END kept")
    if prefix_mode == "evidence_replacement":
        if not isinstance(options.get("sample_id"), str) or not options["sample_id"]:
            raise ValueError("evidence replacement requires a nonempty sample_id")
        if type(options.get("replacement_seed")) is not int:
            raise ValueError("evidence replacement requires an integer replacement_seed")
        if options.get("replacement_version") != "ilvr-evidence-replacement-v1":
            raise ValueError("evidence replacement requires sampling version ilvr-evidence-replacement-v1")
        if options.get("block_original_image", False):
            raise ValueError("evidence replacement preserves original image access")
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
    zero_boundary_mode = any(
        branch.zero_first_embedding
        for group in plan.groups
        for branch in group
    )
    if zero_boundary_mode and plan.groups[1]:
        raise ValueError("zero boundary mode requires an empty textual group")
    if zero_boundary_mode and any(
        not branch.zero_first_embedding for branch in plan.groups[0]
    ):
        raise ValueError(
            "zero boundary mode requires consistently suppressed visual branches"
        )
    for group in plan.groups:
        for branch in group:
            if type(branch.zero_first_embedding) is not bool:
                raise ValueError("zero_first_embedding must be a boolean")
            if not branch.zero_first_embedding:
                continue
            marker_start = branch.blocked_start
            marker_end = branch.blocked_end - 1
            valid_boundary_branch = (
                plan.context_preference
                and branch.kind == "visual_off"
                and 0 <= marker_start < marker_end < plan.root_length
                and branch.clean_ancestor_end == marker_end
                and int(input_ids[0, marker_start])
                == model.config.latent_start_token_id
                and int(input_ids[0, marker_end])
                == model.config.latent_end_token_id
            )
            if not valid_boundary_branch:
                raise ValueError(
                    "zero boundary query requires a marker-inclusive visual branch"
                )
            canonical_latent_block = (
                type(model.config.latent_size) is int
                and model.config.latent_size > 0
                and marker_end - marker_start == model.config.latent_size + 1
                and bool(
                    (
                        input_ids[0, marker_start + 1 : marker_end]
                        == model.config.latent_pad_token_id
                    ).all()
                )
            )
            if not canonical_latent_block:
                raise ValueError(
                    "zero boundary query requires one canonical latent block"
                )
            row_count = len(branch.token_ids)
            replay_positions = tuple(range(marker_end, marker_end + row_count))
            target_positions = tuple(
                range(marker_end + 1, marker_end + row_count + 1)
            )
            canonical_replay = (
                row_count > 0
                and target_positions[-1] < plan.root_length
                and branch.clean_embedding_sources == replay_positions
                and branch.logical_positions == replay_positions
                and branch.selected_positions == tuple(range(row_count))
                and branch.token_ids
                == tuple(int(input_ids[0, position]) for position in replay_positions)
                and branch.target_token_ids
                == tuple(int(input_ids[0, position]) for position in target_positions)
            )
            if not canonical_replay:
                raise ValueError(
                    "zero boundary query requires canonical teacher-forced replay"
                )
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
        embeds = clean_embeds.index_select(1, sources)
        if branch.zero_first_embedding:
            embeds = embeds.clone()
            embeds[:, 0] = 0
        return embeds, clean_positions.index_select(2, logical)

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


def _score_chunk(model, branches, embeds, positions, mask, flat_prefix, *, dil_objective=None, dil_clean_logits=None):
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

    def score(embeds, positions, *tensor_inputs):
        clean_logits = tensor_inputs[0] if dil_objective is not None else None
        prefix_tensors = tensor_inputs[1:] if dil_objective is not None else tensor_inputs
        hidden = model.model(
            inputs_embeds=embeds, position_ids=positions, attention_mask=mask,
            stage2_prefix_key_values=tuple(zip(prefix_tensors[::2], prefix_tensors[1::2])),
            use_cache=False, return_dict=True,
        ).last_hidden_state
        if dil_objective is None:
            return score_selected_hidden(model, hidden, selected, targets, lengths)
        from step_preference.dil_topk_loss import topk_token_terms

        logits = model.lm_head(hidden.index_select(1, selected))[0].float()
        log_probs = logits.log_softmax(-1)
        target_logps = log_probs.gather(-1, targets[:, None]).squeeze(-1)
        entropies = -(log_probs.exp() * log_probs).sum(-1)
        terms = topk_token_terms(clean_logits, logits, targets,
                                beta=dil_objective["beta"], k=dil_objective["k"])
        reduced, start = [], 0
        for length in lengths:
            stop = start + length
            reduced.extend((target_logps[start:stop].mean(), entropies[start:stop],
                            target_logps[start:stop], *(value[start:stop] for value in terms)))
            start = stop
        return tuple(reduced)

    extra = (dil_clean_logits,) if dil_objective is not None else ()
    return score, (embeds, positions, *extra, *flat_prefix)


def score_selected_hidden(model, hidden, selected, targets, lengths):
    """Project only predictor rows and preserve existing per-target reductions."""
    logits = model.lm_head(hidden.index_select(1, selected))[0].float()
    log_probs = logits.log_softmax(-1)
    target_log_probs = log_probs.gather(-1, targets[:, None]).squeeze(-1)
    entropies = -(log_probs.exp() * log_probs).sum(-1)
    reduced, start = [], 0
    for length in lengths:
        reduced.extend((target_log_probs[start:start+length].mean(), entropies[start:start+length]))
        start += length
    return tuple(reduced)


def _evidence_mask(input_ids, embeds, branch, image_token_id, *, block_image, off, backend):
    length = embeds.shape[1]
    hidden_keys = (input_ids[0, :length] == image_token_id) & block_image
    if off:
        hidden_keys = hidden_keys.clone()
        hidden_keys[branch.blocked_start:branch.blocked_end] = True
    if backend == "flex":
        from torch.nn.attention.flex_attention import create_block_mask

        def visible(batch, head, query, key):
            return ((query >= key) & (query < length) & (key < length)
                    & ~hidden_keys[key.clamp(max=length - 1)])

        return create_block_mask(visible, B=1, H=None, Q_LEN=length,
                                 KV_LEN=length, device=str(embeds.device))
    positions = torch.arange(length, device=embeds.device)
    visible = (positions[:, None] >= positions[None, :]) & ~hidden_keys[None, :]
    return torch.zeros((1, 1, length, length), device=embeds.device,
                       dtype=embeds.dtype).masked_fill(~visible, torch.finfo(embeds.dtype).min)


def _evidence_pair(model, branch, on_mask, off_mask, device):
    # Bind each branch's metadata before checkpoint's delayed backward replay.
    selected = torch.tensor(branch.logical_positions, device=device, dtype=torch.long)
    selected = selected[list(branch.selected_positions)]
    targets = torch.tensor(branch.target_token_ids, device=device, dtype=torch.long)

    def score(embeds, positions, mask):
        hidden = model.model(inputs_embeds=embeds, position_ids=positions,
                             attention_mask=mask, use_cache=False,
                             return_dict=True).last_hidden_state
        log_probs = model.lm_head(hidden.index_select(1, selected))[0].float().log_softmax(-1)
        return (log_probs.gather(-1, targets[:, None]).mean(),
                -(log_probs.exp() * log_probs).sum(-1))

    def pair(embeds, positions):
        # Common random numbers: changing visibility must not change dropout draws.
        cpu_rng = torch.get_rng_state()
        devices = [embeds.device.index] if embeds.is_cuda else []
        cuda_rng = torch.cuda.get_rng_state(embeds.device) if devices else None
        on = score(embeds, positions, on_mask)
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_rng)
            if devices:
                torch.cuda.set_rng_state(cuda_rng, embeds.device)
            off = score(embeds, positions, off_mask)
        return (*on, *off)

    return pair


def score_evidence_gain(model, options, input_ids, clean_embeds, clean_positions):
    """Recompute both prefixes; keep clean latent vectors, boundaries, positions and CE.

    Image blocking removes direct original-image keys throughout each replay. It
    does not erase image information from clean-generated latent vectors or text.
    No already-clean decoder K/V is reused in either preference branch.
    """
    results = []
    stats = dict(logical_groups=1, chunks=0, oversized_branches=0, query_rows=0,
                 entropy_scored_rows=0, rejected_scored_rows=0)
    for branch in options["plan"].groups[0]:
        length = branch.logical_positions[-1] + 1
        embeds = clean_embeds[:, :length]
        positions = clean_positions[:, :, :length]
        masks = [_evidence_mask(input_ids, embeds, branch, model.config.image_token_id,
                  block_image=options.get("block_original_image", False), off=off,
                  backend=options.get("backend", "dense")) for off in (False, True)]
        pair = _evidence_pair(model, branch, *masks, embeds.device)
        values = (checkpoint(pair, embeds, positions, use_reentrant=False,
                             preserve_rng_state=True) if options.get("checkpoint", True)
                  else pair(embeds, positions))
        for offset, kind in ((0, "visual_on"), (2, "visual_off")):
            results.append(dict(name=f"{kind}:{branch.step_id}", kind=kind,
                step_id=branch.step_id, mean_logp=values[offset], entropies=values[offset + 1]))
        stats["logical_groups"] += 2
        stats["chunks"] += 1
        stats["query_rows"] += 2 * length
        stats["entropy_scored_rows"] += 2 * len(branch.target_token_ids)
    return tuple(results), stats


def score_sidecars(model, options, input_ids, clean_embeds, clean_positions, prefix_kv, *, dil_clean_logits=None):
    if options.get("prefix_mode", "clean_prefix") == "v_off_prefix":
        return score_evidence_gain(model, options, input_ids, clean_embeds, clean_positions)
    from step_preference.sidecar import (
        chunk_sidecar_branches, sidecar_dense_visibility, build_sidecar_flex_block_mask,
    )

    plan = options["plan"]
    dil = options.get("dil_objective")
    if dil is not None and plan.root_selected_positions:
        if (not isinstance(dil_clean_logits, torch.Tensor) or dil_clean_logits.ndim != 3
                or dil_clean_logits.shape[:2] != (1, len(plan.root_selected_positions))
                or dil_clean_logits.shape[2] != model.get_output_embeddings().weight.shape[0]):
            raise ValueError("DIL clean selected logits alignment mismatch")
        clean_row_by_position = {p: i for i, p in enumerate(plan.root_selected_positions)}
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
            token_ids = [token for branch in branches for token in branch.token_ids]
            replaced_tokens = (model.config.image_token_id, model.config.video_token_id,
                               model.config.latent_pad_token_id)
            if (plan.context_preference
                    and options.get("prefix_mode", "clean_prefix") == "clean_prefix"
                    and not any(token in replaced_tokens for token in token_ids)):
                # One independent lookup per context group preserves the BF16
                # sparse-token backward accumulation used by grouped replay.
                embeds = model.model.embed_tokens(torch.tensor(
                    [token_ids], dtype=torch.long, device=clean_embeds.device))
                logical = torch.tensor(
                    [position for branch in branches for position in branch.logical_positions],
                    dtype=torch.long, device=clean_embeds.device)
                positions = clean_positions.index_select(2, logical)
                offset = 0
                for branch in branches:
                    if branch.zero_first_embedding:
                        embeds[:, offset] = 0
                    offset += len(branch.token_ids)
            else:
                # Generated latent/image embeddings and non-context branches
                # keep their existing clean-source/rejected-input semantics.
                parts = [_branch_inputs(model, branch, input_ids, clean_embeds, clean_positions) for branch in branches]
                embeds = torch.cat([part[0] for part in parts], dim=1)
                positions = torch.cat([part[1] for part in parts], dim=2)
            if options.get("backend", "dense") == "flex":
                mask = build_sidecar_flex_block_mask(plan.root_length, branches, embeds.device)
            else:
                visible = sidecar_dense_visibility(plan.root_length, branches, device=embeds.device)
                mask = torch.zeros(visible.shape, dtype=embeds.dtype, device=embeds.device)
                mask = mask.masked_fill(~visible, torch.finfo(embeds.dtype).min)[None, None]
            clean_chunk = None
            if dil is not None:
                indices = [clean_row_by_position[branch.logical_positions[i]]
                           for branch in branches for i in branch.selected_positions]
                clean_chunk = dil_clean_logits[0].detach().index_select(
                    0, torch.tensor(indices, device=dil_clean_logits.device))
            score, args = _score_chunk(model, branches, embeds, positions, mask, flat_prefix,
                                      dil_objective=dil, dil_clean_logits=clean_chunk)
            values = checkpoint(score, *args, use_reentrant=False, preserve_rng_state=True) if options.get("checkpoint", True) else score(*args)
            for index, branch in enumerate(branches):
                stride = 10 if dil is not None else 2
                result = dict(name=branch.name, kind=branch.kind, step_id=branch.step_id,
                              mean_logp=values[index*stride], entropies=values[index*stride+1])
                if dil is not None:
                    from step_preference.dil_topk_loss import TopkTokenResult
                    result.update(target_logps=values[index*stride+2],
                                  dil_tokens=TopkTokenResult(*values[index*stride+3:index*stride+10]),
                                  target_ids=torch.tensor(branch.target_token_ids, device=embeds.device),
                                  predictor_positions=tuple(branch.logical_positions[i] for i in branch.selected_positions))
                results.append(result)
                key = "rejected_scored_rows" if branch.rejected_model_inputs is not None else "entropy_scored_rows"
                stats[key] += len(branch.target_token_ids)
                stats["oversized_branches"] += len(branch.token_ids) > budget
            stats["chunks"] += 1
            stats["query_rows"] += embeds.shape[1]
    return tuple(results), stats
