"""Immutable deletion/insertion maps for image-free OFF preference contexts."""
from dataclasses import dataclass

from step_preference.sidecar import SidecarPlan


@dataclass(frozen=True)
class InsertionTarget:
    step_id: int
    target_token_ids: tuple[int, ...]
    selected_positions: tuple[int, ...]


@dataclass(frozen=True)
class InsertionBranch:
    token_ids: tuple[int, ...]
    source_indices: tuple[int, ...]
    prefix_length: int
    active_spans: tuple[tuple[int, tuple[int, ...], int], ...]
    targets: tuple[InsertionTarget, ...]


@dataclass(frozen=True)
class InsertionPlan:
    off_token_ids: tuple[int, ...]
    off_source_indices: tuple[int, ...]
    off_targets: tuple[InsertionTarget, ...]
    image_blocks: tuple[tuple[int, ...], ...]
    branches: tuple[InsertionBranch, ...]


def build_insertion_plan(*, input_ids, sidecar_plan, image_token_id,
                         vision_start_token_id, vision_end_token_id,
                         latent_start_token_id, latent_pad_token_id,
                         latent_end_token_id, branch_mode):
    """Preserve teacher-forced target IDs while deleting/reinserting visual slots.

    Source indices always refer to raw encoder inputs, never generated latents.
    Span and predictor positions are full-sequence positions; the scorer shifts
    them into the suffix after slicing the actual OFF prefix.
    """
    if branch_mode not in ('independent', 'cumulative'):
        raise ValueError('unknown insertion branch_mode')
    if not isinstance(sidecar_plan, SidecarPlan) or not sidecar_plan.context_preference:
        raise ValueError('insertion requires a context preference SidecarPlan')
    if len(sidecar_plan.groups) != 2 or sidecar_plan.groups[1]:
        raise ValueError('insertion requires visual-only branches')
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError('insertion requires microbatch 1')
    length = sidecar_plan.root_length
    if not 0 < length <= input_ids.shape[1]:
        raise ValueError('invalid insertion root length')
    special = (image_token_id, vision_start_token_id, vision_end_token_id,
               latent_start_token_id, latent_pad_token_id, latent_end_token_id)
    if len(set(special)) != len(special):
        raise ValueError('image and latent token IDs must be distinct')
    ids = tuple(int(v) for v in input_ids[0, :length].tolist())
    removed, images, latent_blocks = set(), [], {}
    cursor = 0
    while cursor < length:
        token = ids[cursor]
        if token in (vision_start_token_id, latent_start_token_id):
            image = token == vision_start_token_id
            pad = image_token_id if image else latent_pad_token_id
            end = vision_end_token_id if image else latent_end_token_id
            stop = cursor + 1
            while stop < length and ids[stop] == pad:
                stop += 1
            if stop == cursor + 1 or stop >= length or ids[stop] != end:
                raise ValueError('malformed image or latent block')
            if image:
                block = tuple(range(cursor, stop + 1))
                images.append(block)
                removed.update(block)
            else:
                latent_blocks[cursor] = stop
                removed.update(range(cursor + 1, stop))
            cursor = stop + 1
        elif token in (image_token_id, vision_end_token_id, latent_pad_token_id, latent_end_token_id):
            raise ValueError('image or latent token outside its canonical block')
        else:
            cursor += 1
    retained = tuple(i for i in range(length) if i not in removed)
    off_ids = tuple(ids[i] for i in retained)
    off_map = {old: new for new, old in enumerate(retained)}
    entries = sidecar_plan.groups[0]
    if entries and not images:
        raise ValueError('visual insertion requires original image blocks')
    targets, active = [], []
    for entry in entries:
        start = entry.blocked_start - 1
        if (entry.kind != 'visual_off' or entry.zero_first_embedding
                or entry.rejected_model_inputs is not None
                or latent_blocks.get(start) != entry.blocked_end):
            raise ValueError('visual branch must identify canonical latent pads with boundaries kept')
        if active and start <= active[-1]:
            raise ValueError('insertion branches must be ordered and distinct')
        positions = tuple(entry.logical_positions[p] + 1 for p in entry.selected_positions)
        if (not positions or any(p not in off_map or p <= entry.blocked_end for p in positions)
                or tuple(ids[p] for p in positions) != entry.target_token_ids
                or any(a >= b for a, b in zip(positions, positions[1:]))):
            raise ValueError('insertion target IDs/positions do not match retained teacher forcing')
        targets.append(positions)
        active.append(start)

    def mapped_target(entry, positions, mapping):
        return InsertionTarget(entry.step_id, entry.target_token_ids,
                               tuple(mapping[p] - 1 for p in positions))

    off_targets = tuple(mapped_target(e, p, off_map) for e, p in zip(entries, targets))
    batches = (tuple(range(len(entries))),) if entries and branch_mode == 'cumulative' else tuple((i,) for i in range(len(entries)))
    branches = []
    image_sources = tuple(i for block in images for i in block)
    for batch in batches:
        selected_starts = {active[i] for i in batch}
        sources, spans, mapping = [], [], {}
        for old in retained:
            if old in selected_starts:
                sources.extend(image_sources)
            mapping[old] = len(sources)
            sources.append(old)
            if old in selected_starts:
                new_start = len(sources) - 1
                end = latent_blocks[old]
                pads = tuple(range(len(sources), len(sources) + end - old - 1))
                sources.extend(range(old + 1, end))
                spans.append((new_start, pads, len(sources)))
        branches.append(InsertionBranch(
            tuple(ids[i] for i in sources), tuple(sources), off_map[active[batch[0]]],
            tuple(spans), tuple(mapped_target(entries[i], targets[i], mapping) for i in batch)))
    return InsertionPlan(off_ids, retained, off_targets, tuple(images), tuple(branches))
