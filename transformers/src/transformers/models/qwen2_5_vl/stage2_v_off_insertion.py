"""Image-free OFF decoder execution and differentiable visual insertion branches."""
import torch
from torch.utils.checkpoint import checkpoint

from .stage2_latent_rollout import LatentSpan
from .stage2_shared_prefix import score_selected_hidden


def _causal_mask(embeds, prefix_length, backend):
    length = embeds.shape[1]
    if backend == 'flex':
        from torch.nn.attention.flex_attention import create_block_mask

        def visible(batch, head, query, key):
            return (query < length) & (key < prefix_length + length) & (key <= prefix_length + query)

        return create_block_mask(visible, B=1, H=None, Q_LEN=length,
                                 KV_LEN=prefix_length + length, device=str(embeds.device))
    query = torch.arange(length, device=embeds.device)[:, None] + prefix_length
    keys = torch.arange(prefix_length + length, device=embeds.device)[None, :]
    return torch.zeros((1, 1, length, prefix_length + length), device=embeds.device,
                       dtype=embeds.dtype).masked_fill(keys > query, torch.finfo(embeds.dtype).min)


def _target_tensors(targets, device, offset=0):
    selected = torch.tensor([p - offset for t in targets for p in t.selected_positions], device=device)
    tokens = torch.tensor([i for t in targets for i in t.target_token_ids], device=device)
    return selected, tokens, tuple(len(t.target_token_ids) for t in targets)


def _on_function(model, branch, positions, backend):
    """Bind metadata once per branch for nonreentrant backward recomputation."""
    prefix_length = branch.prefix_length
    selected, tokens, lengths = _target_tensors(branch.targets, positions.device, prefix_length)
    spans = tuple(LatentSpan(start - prefix_length,
                            tuple(p - prefix_length for p in pads), end - prefix_length)
                  for start, pads, end in branch.active_spans)

    def score(embeds, *prefix_tensors):
        prefix = tuple(zip(prefix_tensors[::2], prefix_tensors[1::2]))

        def decoder(*, inputs_embeds, attention_mask, position_ids, **kwargs):
            return model.model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                attention_mask=_causal_mask(inputs_embeds, prefix_length, backend),
                stage2_prefix_key_values=prefix, generate_mode=True, **kwargs)

        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
        rollout = model.stage2_rollout_engine.rollout(language_model=decoder,
            base_inputs_embeds=embeds, attention_mask=mask,
            full_position_ids=positions, latent_spans=spans)
        hidden = decoder(inputs_embeds=rollout.final_inputs_embeds, attention_mask=mask,
            position_ids=positions, use_cache=False, return_dict=True).last_hidden_state
        return score_selected_hidden(model, hidden, selected, tokens, lengths)

    return score


def score_v_off_insertion(model, options, input_ids, base_inputs_embeds, image_grid_thw):
    """Reuse actual OFF KV, raw encoder features and unchanged Stage2 recurrence.

    No clean decoder state is accepted. OFF embeds its own image-free token IDs;
    ON shares only raw input features and differentiable pre-insertion OFF KV.
    """
    from step_preference.insertion import build_insertion_plan

    backend = options.get('backend', 'dense')
    if backend not in ('dense', 'flex'):
        raise ValueError('insertion backend must be dense or flex')
    if backend == 'flex' and (not base_inputs_embeds.is_cuda or
                             (model.training and model.config.attention_dropout)):
        raise ValueError('Flex insertion requires CUDA and zero training attention dropout')
    if options.get('block_original_image', False):
        raise ValueError('v_off_insertion requires block_original_image=False')
    plan = build_insertion_plan(input_ids=input_ids, sidecar_plan=options['plan'],
        image_token_id=model.config.image_token_id,
        vision_start_token_id=model.config.vision_start_token_id,
        vision_end_token_id=model.config.vision_end_token_id,
        latent_start_token_id=model.config.latent_start_token_id,
        latent_pad_token_id=model.config.latent_pad_token_id,
        latent_end_token_id=model.config.latent_end_token_id,
        branch_mode=options.get('branch_mode', 'independent'))
    stats = dict(logical_groups=0, chunks=0, oversized_branches=0, query_rows=0,
                 entropy_scored_rows=0, rejected_scored_rows=0)
    if not plan.branches:
        return (), stats
    if (image_grid_thw is None or image_grid_thw.ndim != 2
            or image_grid_thw.shape != (len(plan.image_blocks), 3)):
        raise ValueError('insertion requires one image grid per original image block')
    merge = model.config.vision_config.spatial_merge_size
    for block, grid in zip(plan.image_blocks, image_grid_thw.tolist()):
        t, h, w = grid
        if min(grid) <= 0 or h % merge or w % merge or t * (h // merge) * (w // merge) != len(block) - 2:
            raise ValueError('original image block/grid token count mismatch')
    device = base_inputs_embeds.device
    off_ids = torch.tensor([plan.off_token_ids], device=device)
    off_mask = torch.ones_like(off_ids)
    off_positions, _ = model.get_rope_index(off_ids, attention_mask=off_mask)
    off_embeds = model.model.embed_tokens(off_ids)
    off = model.model(inputs_embeds=off_embeds, position_ids=off_positions,
        attention_mask=off_mask, stage2_capture_kv=True, use_cache=False,
        return_dict=True, generate_mode=True)
    selected, tokens, lengths = _target_tensors(plan.off_targets, device)
    def off_score(hidden):
        return score_selected_hidden(model, hidden, selected, tokens, lengths)
    use_checkpoint = options.get('checkpoint', True)
    values = (checkpoint(off_score, off.last_hidden_state, use_reentrant=False,
                         preserve_rng_state=True) if use_checkpoint else off_score(off.last_hidden_state))
    results = []

    def append_results(targets, kind, values):
        for i, target in enumerate(targets):
            results.append(dict(name=f'{kind}:{target.step_id}', kind=kind,
                step_id=target.step_id, mean_logp=values[2 * i], entropies=values[2 * i + 1]))
            stats['entropy_scored_rows'] += len(target.target_token_ids)

    append_results(plan.off_targets, 'visual_off', values)
    stats['logical_groups'] = 1
    stats['query_rows'] = len(plan.off_token_ids)
    for branch in plan.branches:
        prefix_length = branch.prefix_length
        ids = torch.tensor([branch.token_ids], device=device)
        grids = image_grid_thw.to(device).repeat(len(branch.active_spans), 1)
        positions, _ = model.get_rope_index(ids, image_grid_thw=grids, attention_mask=torch.ones_like(ids))
        if not torch.equal(positions[:, :, :prefix_length], off_positions[:, :, :prefix_length]):
            raise ValueError('ON positions before the insertion must equal actual OFF positions')
        sources = torch.tensor(branch.source_indices[prefix_length:], device=device)
        embeds = base_inputs_embeds.index_select(1, sources)
        prefix = tuple(tensor[:, :, :prefix_length] for pair in off.stage2_key_values for tensor in pair)
        score = _on_function(model, branch, positions[:, :, prefix_length:], backend)
        values = (checkpoint(score, embeds, *prefix, use_reentrant=False, preserve_rng_state=True)
                  if use_checkpoint else score(embeds, *prefix))
        append_results(branch.targets, 'visual_on', values)
        stats['logical_groups'] += 1
        stats['chunks'] += 1
        stats['query_rows'] += embeds.shape[1]
    return tuple(results), stats
