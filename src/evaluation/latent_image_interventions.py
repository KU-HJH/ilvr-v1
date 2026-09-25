"""Teacher-forced latent recomputation under image-routing interventions."""
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch

from evaluation.latent_image_layout import LATENT, QUESTION, REASONING, latent_query_positions

SOURCES = ("clean", "alt", "none")


@dataclass(frozen=True)
class Condition:
    name: str
    text_source: str = "clean"
    latent_source: str = "clean"
    head_subset: tuple = ()
    random_keys: bool = False
    native: bool = False

    def __post_init__(self):
        if self.text_source not in SOURCES or self.latent_source not in SOURCES:
            raise ValueError(f"unknown image source in {self}")
        if self.head_subset and self.latent_source != "clean":
            raise ValueError("head_subset requires latent_source='clean' as its base")
        if self.native and (self.text_source, self.latent_source) != ("clean", "clean"):
            raise ValueError("native condition cannot route images")


@dataclass(frozen=True)
class PhysicalLayout:
    length: int
    logical_of: tuple
    physical_of: tuple
    clean_image: tuple
    alt_image: tuple
    latent_rows: tuple
    block_rows: tuple


def build_physical_layout(layout, with_alt):
    image = layout.image_positions
    n = len(image) if with_alt else 0
    last = image[-1]
    physical_of = tuple(p if p <= last else p + n for p in range(layout.length))
    logical_of = list(range(last + 1)) + list(image[:n]) + list(range(last + 1, layout.length))
    alt = tuple(range(last + 1, last + 1 + n))
    rows = tuple(physical_of[p] for p in latent_query_positions(layout))
    block_rows = tuple(tuple(physical_of[p] for p in (b.start, *b.pads)) for b in layout.blocks)
    return PhysicalLayout(layout.length + n, tuple(logical_of), physical_of, tuple(image), alt, rows, block_rows)


def build_visibility(phys, condition, random_blocked=None, *, latent_source=None):
    latent_source = latent_source or condition.latent_source
    if not phys.alt_image and "alt" in (condition.text_source, latent_source):
        raise ValueError("alt image source requires a dual-image layout")
    allowed = torch.ones(phys.length, phys.length, dtype=torch.bool).tril()
    clean, alt = list(phys.clean_image), list(phys.alt_image)
    if alt:
        allowed[alt[0]:alt[-1] + 1, clean] = False
    first = (alt[-1] if alt else clean[-1]) + 1
    latent_rows = set(phys.latent_rows)
    for q in range(first, phys.length):
        source = latent_source if q in latent_rows else condition.text_source
        if source in ("clean", "none") and alt:
            allowed[q, alt] = False
        if source in ("alt", "none"):
            allowed[q, clean] = False
    if condition.random_keys:
        if random_blocked is None:
            raise ValueError("random_keys condition requires random_blocked")
        for block, rows in enumerate(phys.block_rows):
            keys = [phys.physical_of[k] for k in random_blocked.get(block, ())]
            if keys:
                allowed[torch.tensor(rows)[:, None], torch.tensor(keys)[None, :]] = False
    return allowed


def random_key_sets(layout, seed=42, source_index=0):
    sets, matched = {}, {}
    n_image = len(layout.image_positions)
    for b in layout.blocks:
        eligible = [k for k in range(b.start)
                    if layout.base_regions[k] in (QUESTION, REASONING, LATENT)]
        count = min(n_image, len(eligible))
        rng = np.random.default_rng([seed, source_index, b.index])
        chosen = sorted(int(k) for k in rng.choice(eligible, size=count, replace=False)) if count else []
        sets[b.index] = tuple(chosen)
        matched[b.index] = count / n_image
    return sets, matched


from stage2.peft import resolve_stage2_qwen_model
from transformers.models.qwen2_5_vl.stage2_latent_rollout import parse_latent_spans, validate_stage2_config


@dataclass
class ConditionResult:
    name: str
    latents: torch.Tensor
    mid_body: torch.Tensor
    final_body: torch.Tensor
    image_mid: torch.Tensor
    image_final: torch.Tensor
    last_hidden: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)


def _additive(allowed, dtype):
    """[P, P] -> [1, 1, P, P]; [H, P, P] -> [1, H, P, P]."""
    mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask[None, None] if allowed.ndim == 2 else mask[None]


@contextmanager
def _head_masks(qwen, base_allowed, none_allowed, head_subset, dtype):
    by_layer = {}
    for layer, head in head_subset:
        by_layer.setdefault(layer, set()).add(head)
    heads = qwen.config.num_attention_heads
    handles = []

    def hook(index):
        chosen = torch.zeros(heads, dtype=torch.bool, device=base_allowed.device)
        chosen[list(by_layer[index])] = True

        def pre(module, args, kwargs):
            incoming = kwargs["attention_mask"]
            p = incoming.shape[-1]
            if not torch.equal(torch.isfinite(incoming[0, 0]), base_allowed[:p, :p]):
                raise ValueError("decoder altered visibility")
            allowed = torch.where(chosen[:, None, None], none_allowed[None, :p, :p], base_allowed[None, :p, :p])
            kwargs["attention_mask"] = _additive(allowed, incoming.dtype)      # [1, H, P, P]
            return args, kwargs
        return pre

    try:
        for index in by_layer:
            handles.append(qwen.model.layers[index].self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _physical_embeds(qwen, trajectory, phys, alt_pixel_values, alt_grid, pixel_override):
    device = qwen.model.embed_tokens.weight.device
    ids = trajectory.input_ids.to(device)
    pixels, grid = (pixel_override if pixel_override is not None
                    else (trajectory.pixel_values, trajectory.image_grid_thw))
    grid = grid.to(device)
    if pixel_override is not None and not torch.equal(grid.cpu(), trajectory.image_grid_thw):
        raise ValueError("override grid differs from trajectory grid")
    base = qwen.prepare_stage2_base_inputs_embeds(input_ids=ids, pixel_values=pixels.to(device), image_grid_thw=grid)
    positions, _ = qwen.get_rope_index(ids, image_grid_thw=grid, attention_mask=trajectory.attention_mask.to(device))
    index = torch.tensor(phys.logical_of, device=device)
    embeds = base.index_select(1, index)
    physical_ids = ids.index_select(1, index)
    if phys.alt_image:
        if alt_grid is None or not torch.equal(alt_grid.cpu(), trajectory.image_grid_thw):
            raise ValueError("alt grid differs from trajectory grid")
        with torch.no_grad():
            alt = qwen.visual(alt_pixel_values.to(device, qwen.visual.dtype), grid_thw=alt_grid.to(device))
        embeds[:, list(phys.alt_image)] = alt.to(embeds.dtype)[None]
    return embeds, positions.index_select(2, index), physical_ids


@torch.inference_mode()
def run_condition(model, trajectory, condition, *, alt_pixel_values=None, alt_grid=None, pixel_override=None,
                  random_blocked=None, mid_layer=18, final_observer=None):
    from evaluation.vision_block_runtime import _scores
    if alt_pixel_values is not None and pixel_override is not None:
        raise ValueError("alt image and pixel override are exclusive")
    qwen = resolve_stage2_qwen_model(model)
    layout = trajectory.layout
    phys = build_physical_layout(layout, with_alt=alt_pixel_values is not None)
    embeds, positions, physical_ids = _physical_embeds(qwen, trajectory, phys, alt_pixel_values, alt_grid, pixel_override)
    device, dtype = embeds.device, embeds.dtype
    token_ids, latent_size = validate_stage2_config(qwen.config)
    spans = parse_latent_spans(physical_ids[0], token_ids=token_ids, latent_size=latent_size)
    if condition.native:
        allowed = None
    else:
        allowed = build_visibility(phys, condition, random_blocked).to(device)
    stack = ExitStack()
    if condition.head_subset:
        none_allowed = build_visibility(phys, condition, random_blocked, latent_source="none").to(device)
        stack.enter_context(_head_masks(qwen, allowed, none_allowed, condition.head_subset, dtype))

    def decoder(**kwargs):
        if allowed is not None:
            p = kwargs["inputs_embeds"].shape[1]
            kwargs["attention_mask"] = _additive(allowed[:p, :p], dtype)
        return qwen.model(generate_mode=True, **kwargs)

    with stack:
        rollout = qwen.stage2_rollout_engine.rollout(
            language_model=decoder, base_inputs_embeds=embeds,
            attention_mask=torch.ones(physical_ids.shape, dtype=torch.long, device=device),
            full_position_ids=positions, latent_spans=spans)
        with (final_observer() if final_observer else nullcontext()):
            final = decoder(inputs_embeds=rollout.final_inputs_embeds,
                            attention_mask=torch.ones(physical_ids.shape, dtype=torch.long, device=device),
                            position_ids=positions, use_cache=False, return_dict=True, output_hidden_states=True)
    hidden = final.last_hidden_state
    n_blocks = len(layout.blocks)
    latents = torch.stack([s[0] for s in rollout.latent_states]).float().reshape(n_blocks, latent_size, -1).cpu()
    pad_rows = [[phys.physical_of[p] for p in b.pads] for b in layout.blocks]
    mid = final.hidden_states[mid_layer][0]
    mid_body = torch.stack([mid[r].float().mean(0) for r in pad_rows]).cpu()
    final_body = torch.stack([hidden[0, r].float().mean(0) for r in pad_rows]).cpu()
    image_rows = list(phys.alt_image if condition.text_source == "alt" else phys.clean_image)
    targets = list(layout.scored_positions)
    predictors = [phys.physical_of[p] - 1 for p in targets]
    token_list = trajectory.input_ids[0].tolist()
    scores = dict(zip(targets, _scores(qwen, hidden, predictors, [token_list[p] for p in targets])))
    controls = {p for group in (layout.pre_text + layout.post_text) for p in group}
    return ConditionResult(
        condition.name, latents, mid_body, final_body,
        mid[image_rows].float().mean(0).cpu(), hidden[0, image_rows].float().mean(0).cpu(),
        {p: hidden[0, phys.physical_of[p]].float().cpu() for p in controls}, scores)
