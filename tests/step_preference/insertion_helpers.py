"""Tiny native fixtures and an independent literal full-sequence oracle."""
from functools import partial

import torch

from step_preference.equivalence import _tiny_model
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentSpan

from tests.step_preference.test_v_off_insertion import two_step_inputs


def native_case(num_hidden_layers=2):
    torch.manual_seed(481)
    config = _tiny_model().config
    config.num_hidden_layers = num_hidden_layers
    config.vision_start_token_id = 78
    config.vision_end_token_id = 79
    config._attn_implementation = 'sdpa'
    model = type(_tiny_model())(config).train()
    model.visual.requires_grad_(False)
    kwargs = two_step_inputs()
    ids = kwargs['input_ids']
    grid = torch.tensor([[1, 1, 1]])
    pixels = torch.linspace(-1, 1, 12).reshape(1, 12)
    base = model.prepare_stage2_base_inputs_embeds(input_ids=ids, pixel_values=pixels, image_grid_thw=grid)
    options = dict(plan=kwargs['sidecar_plan'], prefix_mode='v_off_insertion',
                   branch_mode='independent', backend='dense', checkpoint=False)
    return model, ids, base, grid, options, pixels


def literal_oracle(model, base, grid, mode):
    """No insertion planner or prefix scorer: full decoder recomputation only.

    The source lists, spans and predictors are hand-derived from the brief.
    """
    source_sequences = ([0, 4, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16],
                        [0, 4, 5, 8, 9, 10, 1, 2, 3, 11, 12, 13, 14, 15, 16])
    token_sequences = ([11, 12, 78, 80, 79, 91, 90, 90, 92, 13, 14, 91, 92, 15, 16],
                       [11, 12, 91, 92, 13, 14, 78, 80, 79, 91, 90, 90, 92, 15, 16])
    spans = ((LatentSpan(5, (6, 7), 8),), (LatentSpan(9, (10, 11), 12),))
    selections = (((0, (8, 9), (13, 14)),), ((1, (12, 13), (15, 16)),))
    if mode == 'cumulative':
        source_sequences = ([0, 4, 1, 2, 3, 5, 6, 7, 8, 9, 10, 1, 2, 3, 11, 12, 13, 14, 15, 16],)
        token_sequences = ([11, 12, 78, 80, 79, 91, 90, 90, 92, 13, 14, 78, 80, 79, 91, 90, 90, 92, 15, 16],)
        spans = ((LatentSpan(5, (6, 7), 8), LatentSpan(14, (15, 16), 17)),)
        selections = (((0, (8, 9), (13, 14)), (1, (17, 18), (15, 16))),)
    logits, scores, states = {}, {}, []
    off_ids = torch.tensor([[11, 12, 91, 92, 13, 14, 91, 92, 15, 16]], device=base.device)
    positions, _ = model.get_rope_index(off_ids, attention_mask=torch.ones_like(off_ids))
    hidden = model.model(inputs_embeds=model.model.embed_tokens(off_ids), position_ids=positions,
                         attention_mask=torch.ones_like(off_ids), use_cache=False,
                         return_dict=True, generate_mode=True).last_hidden_state
    for step, selected, target in ((0, (3, 4), (13, 14)), (1, (7, 8), (15, 16))):
        rows = model.lm_head(hidden[:, list(selected)])[0].float()
        logits[('visual_off', step)] = rows
        scores[('visual_off', step)] = rows.log_softmax(-1).gather(-1, torch.tensor(target, device=base.device)[:, None]).mean()
    for sources, tokens, active, targets in zip(source_sequences, token_sequences, spans, selections):
        ids = torch.tensor([tokens], device=base.device)
        embeds = base[:, sources]
        mask = torch.ones_like(ids)
        positions, _ = model.get_rope_index(ids, image_grid_thw=grid.repeat(len(active), 1), attention_mask=mask)
        rollout = model.stage2_rollout_engine.rollout(language_model=partial(model.model, generate_mode=True),
            base_inputs_embeds=embeds, attention_mask=mask, full_position_ids=positions, latent_spans=active)
        states.extend(rollout.latent_states)
        hidden = model.model(inputs_embeds=rollout.final_inputs_embeds, position_ids=positions,
            attention_mask=mask, use_cache=False, return_dict=True, generate_mode=True).last_hidden_state
        for step, selected, target in targets:
            rows = model.lm_head(hidden[:, list(selected)])[0].float()
            logits[('visual_on', step)] = rows
            scores[('visual_on', step)] = rows.log_softmax(-1).gather(-1, torch.tensor(target, device=base.device)[:, None]).mean()
    return scores, logits, states


def preference(scores):
    return torch.stack([-torch.nn.functional.logsigmoid(.1 * (scores[('visual_on', i)] - scores[('visual_off', i)])) for i in (0, 1)]).mean()


def image_case(model, grids):
    """Valid vision inputs with the same literal text and two latent blocks."""
    image_tokens = [token for t, h, w in grids for token in [78, *([80] * (t*h*w)), 79]]
    tokens = [11, *image_tokens, 12, 91, 90, 90, 92, 13, 14, 91, 90, 90, 92, 15, 16]
    ids = torch.tensor([tokens])
    start = len(image_tokens) + 2
    from step_preference.sidecar import build_context_sidecar_plan
    from step_preference.spans import ReasoningStepSpan
    spans = (ReasoningStepSpan(0, 0, start, start+1, start+3, start+4, start+6, None),
             ReasoningStepSpan(1, start+4, start+6, start+7, start+9, start+10, start+12, 0))
    plan = build_context_sidecar_plan(clean_input_ids=ids, attention_mask=torch.ones_like(ids), spans=spans, visual_only=True)
    grid = torch.tensor(grids)
    pixels = torch.linspace(-1, 1, sum(t*h*w for t,h,w in grids)*12).reshape(-1, 12)
    base = model.prepare_stage2_base_inputs_embeds(input_ids=ids, pixel_values=pixels, image_grid_thw=grid)
    return ids, base, grid, plan, pixels
