"""Native paired full-prefix scoring for same-image evidence replacement."""

import torch
from torch.utils.checkpoint import checkpoint


_VERSION = "ilvr-evidence-replacement-v1"


def _validate_inputs(model, options, input_ids, base_inputs_embeds, clean_inputs_embeds, clean_positions):
    if options.get("prefix_mode") != "evidence_replacement":
        raise ValueError("evidence replacement requires prefix_mode=evidence_replacement")
    if not isinstance(options.get("sample_id"), str) or not options["sample_id"]:
        raise ValueError("evidence replacement sample_id must be a nonempty string")
    if type(options.get("replacement_seed")) is not int:
        raise ValueError("evidence replacement replacement_seed must be an integer")
    if options.get("replacement_version") != _VERSION:
        raise ValueError("evidence replacement replacement_version is unsupported")
    if options.get("backend", "dense") not in ("dense", "flex"):
        raise ValueError("evidence replacement backend must be dense or flex")
    if type(options.get("checkpoint", True)) is not bool:
        raise ValueError("evidence replacement checkpoint must be a bool")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("evidence replacement input_ids must have shape [1, sequence_length]")
    if not isinstance(base_inputs_embeds, torch.Tensor) or not isinstance(clean_inputs_embeds, torch.Tensor):
        raise ValueError("evidence replacement embeddings must be tensors")
    if base_inputs_embeds.shape != clean_inputs_embeds.shape or base_inputs_embeds.ndim != 3:
        raise ValueError("evidence replacement base and clean embeddings must have matching [1, L, H] shapes")
    if base_inputs_embeds.shape[:2] != input_ids.shape:
        raise ValueError("evidence replacement embeddings must match input_ids")
    if not isinstance(clean_positions, torch.Tensor) or clean_positions.shape != (3, 1, input_ids.shape[1]):
        raise ValueError("evidence replacement positions must have shape [3, 1, L]")
    if clean_positions.device != clean_inputs_embeds.device or base_inputs_embeds.device != clean_inputs_embeds.device:
        raise ValueError("evidence replacement tensors must share one device")
    if options.get("backend", "dense") == "flex" and (
        not clean_inputs_embeds.is_cuda
        or (model.training and model.config.attention_dropout)
    ):
        raise ValueError("Flex evidence replacement requires CUDA and zero training attention dropout")


def _native_mask(model, embeds, backend):
    """Use the decoder's ordinary causal mask, including its actual SWA rule."""
    length = embeds.shape[1]
    if backend == "dense":
        return torch.ones((1, length), dtype=torch.long, device=embeds.device)

    from torch.nn.attention.flex_attention import create_block_mask

    # Qwen's uncached 2-D-mask path applies ``sliding_window`` directly.  This
    # intentionally mirrors that path, including configurations whose separate
    # ``use_sliding_window`` flag remains false.
    window = model.config.sliding_window

    def visible(batch, head, query, key):
        causal = (query < length) & (key < length) & (key <= query)
        return causal if window is None else causal & (key > query - window)

    return create_block_mask(
        visible, B=1, H=None, Q_LEN=length, KV_LEN=length,
        device=str(embeds.device),
    )


def _score_prefix(model, embeds, positions, targets, predictors, backend):
    mask = _native_mask(model, embeds, backend)
    hidden = model.model(
        inputs_embeds=embeds,
        position_ids=positions,
        attention_mask=mask,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    selected = torch.tensor(predictors, dtype=torch.long, device=embeds.device)
    target_ids = torch.tensor(targets, dtype=torch.long, device=embeds.device)
    log_probs = model.lm_head(hidden.index_select(1, selected))[0].float().log_softmax(-1)
    target_log_probs = log_probs.gather(-1, target_ids[:, None]).squeeze(-1)
    return target_log_probs.mean(), -(log_probs.exp() * log_probs).sum(-1)


def _paired_score(model, branch, base_inputs_embeds, clean_inputs_embeds, clean_positions, backend):
    """Replay both full prefixes with common dropout draws and no reused K/V."""
    original = clean_inputs_embeds[:, : branch.prefix_length]
    positions = clean_positions[:, :, : branch.prefix_length]
    source = clean_inputs_embeds if branch.source_kind == "donor" else base_inputs_embeds
    indices = torch.tensor(branch.source_positions, dtype=torch.long, device=original.device)
    replacement_values = source.index_select(1, indices).detach()
    replacement = torch.cat((
        original[:, : branch.latent_start],
        replacement_values,
        original[:, branch.latent_end :],
    ), dim=1)

    cpu_rng = torch.get_rng_state()
    devices = [original.device.index] if original.is_cuda else []
    cuda_rng = torch.cuda.get_rng_state(original.device) if devices else None
    original_values = _score_prefix(
        model, original, positions, branch.target_token_ids,
        branch.predictor_positions, backend,
    )
    # The context restores the post-original global state when it exits.  The
    # replacement sees the pre-original state so paired dropout draws agree.
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(cpu_rng)
        if devices:
            torch.cuda.set_rng_state(cuda_rng, original.device)
        replacement_values = _score_prefix(
            model, replacement, positions, branch.target_token_ids,
            branch.predictor_positions, backend,
        )
    return (*original_values, *replacement_values)


def _branch_pair(model, branch, backend):
    """Bind immutable branch metadata for checkpoint's delayed recomputation."""
    def pair(base_inputs_embeds, clean_inputs_embeds, clean_positions):
        return _paired_score(
            model, branch, base_inputs_embeds, clean_inputs_embeds,
            clean_positions, backend,
        )
    return pair


def score_evidence_replacement(
    model, options, input_ids, base_inputs_embeds, clean_inputs_embeds,
    clean_positions, image_grid_thw,
):
    """Score original and detached-replacement prefixes for every scored step."""
    _validate_inputs(
        model, options, input_ids, base_inputs_embeds, clean_inputs_embeds,
        clean_positions,
    )
    from step_preference.replacement import build_replacement_plan

    # The planner owns all metadata validation and private sampling.  It is
    # intentionally built before any checkpointed replay can be recomputed.
    plan = build_replacement_plan(
        input_ids=input_ids,
        sidecar_plan=options["plan"],
        image_token_id=model.config.image_token_id,
        image_count=1,
        latent_size=model.config.latent_size,
        sample_id=options["sample_id"],
        seed=options["replacement_seed"],
    )
    stats = dict(
        logical_groups=2 * len(plan.branches), chunks=0,
        oversized_branches=0, query_rows=0, entropy_scored_rows=0,
        rejected_scored_rows=0, replacement_image_count=0,
        replacement_donor_count=0,
    )
    results = []
    backend = options.get("backend", "dense")
    for branch in plan.branches:
        pair = _branch_pair(model, branch, backend)
        values = (
            checkpoint(
                pair, base_inputs_embeds, clean_inputs_embeds, clean_positions,
                use_reentrant=False, preserve_rng_state=True,
            ) if options.get("checkpoint", True) else pair(
                base_inputs_embeds, clean_inputs_embeds, clean_positions
            )
        )
        for offset, kind in ((0, "visual_original"), (2, "visual_replacement")):
            results.append(dict(
                name=f"{kind}:{branch.step_id}", kind=kind,
                step_id=branch.step_id, mean_logp=values[offset],
                entropies=values[offset + 1], source_kind=branch.source_kind,
                donor_step_id=branch.donor_step_id,
                source_positions=branch.source_positions,
            ))
        stats["chunks"] += 1
        stats["query_rows"] += 2 * branch.prefix_length
        stats["entropy_scored_rows"] += 2 * len(branch.target_token_ids)
        stats[f"replacement_{branch.source_kind}_count"] += 1
    return tuple(results), stats
