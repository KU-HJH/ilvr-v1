"""Validated, project-independent shared-prefix replay for Stage 2."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask

from .stage2_latent_rollout import Stage2DecoderCallStats
from .stage2_prefix_cache import (
    DifferentiablePrefixState,
    rebuild_stage2_dynamic_cache,
)


@dataclass(frozen=True)
class PackedSidecarBranch:
    branch_key: str
    sample_id: str
    step_id: int
    modality: str
    role: str
    replay_boundary: int
    clean_visible_end: int
    blocked_clean_interval: tuple[int, int] | None
    query_start: int
    query_end: int
    score_offsets: torch.LongTensor
    target_token_ids: torch.LongTensor


@dataclass(frozen=True)
class PackedSidecarLayout:
    modality: str
    clean_length: int
    branches: tuple[PackedSidecarBranch, ...]
    input_token_ids: torch.LongTensor
    position_ids: torch.LongTensor
    clean_attention_mask: torch.Tensor


@dataclass(frozen=True)
class Stage2SidecarLogits:
    by_branch_key: dict[str, torch.FloatTensor]


@dataclass(frozen=True)
class Stage2ComputeStats:
    """Execution counters; the checkpoint witness stays live through backward.

    Avoided tokens count each replay's [0, boundary) prefix; clean tokens
    include any padding actually passed through the clean decoder.
    """

    logical_groups: int
    top_level_policy_calls: int
    clean_decoder_calls: int
    sidecar_decoder_calls: int
    branch_microbatches: int
    clean_prefix_tokens: int
    prefix_tokens_avoided: int
    chosen_entropy_tokens: int
    rejected_dpo_tokens: int
    reference_calls: int
    reference_score_hits: int
    _decoder_call_stats: Stage2DecoderCallStats = field(repr=False)

    @property
    def checkpoint_recompute_calls(self) -> int:
        return self._decoder_call_stats.checkpoint_recompute_calls

    @property
    def latent_recurrence_calls(self) -> int:
        """Original ILVR latent recurrence; excludes the final clean prediction."""
        return self.clean_decoder_calls - 1

    @property
    def prediction_forward_calls(self) -> int:
        """Final clean plus V/T prediction calls, excluding latent recurrence."""
        return self.clean_decoder_calls - self.latent_recurrence_calls + self.sidecar_decoder_calls

    def counters(self) -> dict[str, int]:
        """Snapshot at the consumer's logging point, after any recomputation."""
        return {name: getattr(self, name) for name in (
            "logical_groups", "top_level_policy_calls", "clean_decoder_calls",
            "sidecar_decoder_calls", "checkpoint_recompute_calls", "branch_microbatches",
            "clean_prefix_tokens", "prefix_tokens_avoided", "chosen_entropy_tokens",
            "rejected_dpo_tokens", "reference_calls", "reference_score_hits",
            "latent_recurrence_calls", "prediction_forward_calls",
        )}


def _attribute(value: Any, name: str) -> Any:
    if not hasattr(value, name):
        raise ValueError(f"sidecar request requires {name}")
    return getattr(value, name)


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"sidecar {name} must be a nonnegative integer")
    return value


def _vector(value: Any, name: str, device: torch.device) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or value.dtype != torch.long
        or not value.numel()
        or value.device != device
    ):
        raise ValueError(f"sidecar {name} must be a nonempty long vector on the clean input device")
    return value


def pack_stage2_sidecar_plan(
    plan: Any,
    *,
    clean_input_ids: torch.LongTensor,
    clean_position_ids: torch.LongTensor,
    clean_attention_mask: torch.Tensor,
) -> tuple[PackedSidecarLayout, ...]:
    """Validate the complete structural boundary before executing either group.

    This module deliberately does not import the producer's project types.
    Each query is the replayed boundary or one previous target, and every query
    scores one target. The clean boundary copy is never visible to a replay.
    """
    stage2_debugging = getattr(plan, "stage2_debugging", True)
    if (
        not isinstance(clean_input_ids, torch.Tensor)
        or clean_input_ids.ndim != 2
        or clean_input_ids.shape[0] != 1
        or clean_input_ids.dtype != torch.long
        or not clean_input_ids.numel()
    ):
        raise ValueError("sidecar clean_input_ids must be one nonempty long row")
    device = clean_input_ids.device
    clean_length = clean_input_ids.shape[1]
    if (
        not isinstance(clean_position_ids, torch.Tensor)
        or clean_position_ids.shape != (3, 1, clean_length)
        or clean_position_ids.dtype != torch.long
        or clean_position_ids.device != device
    ):
        raise ValueError("sidecar clean_position_ids must contain three clean MRoPE axes")
    if (
        not isinstance(clean_attention_mask, torch.Tensor)
        or clean_attention_mask.shape != clean_input_ids.shape
        or clean_attention_mask.device != device
        or (stage2_debugging and not bool(((clean_attention_mask == 0) | (clean_attention_mask == 1)).all()))
    ):
        raise ValueError("sidecar clean_attention_mask must be a binary clean row")

    layouts = []
    seen_keys = set()
    sample_ids = set()
    chosen_by_modality = []
    modalities = getattr(plan, "modalities", "visual_text")
    if modalities not in {"visual", "visual_text"}:
        raise ValueError("sidecar modalities must be visual or visual_text")
    if modalities == "visual" and _attribute(plan, "text"):
        raise ValueError("visual-only plan must not contain text branches")
    for modality in (("visual",) if modalities == "visual" else ("visual", "text")):
        requests = _attribute(plan, modality)
        if not isinstance(requests, (tuple, list)) or not requests:
            raise ValueError(f"sidecar plan requires a nonempty {modality} group")
        packed = []
        inputs = []
        positions = []
        chosen = {}
        cursor = 0
        for request in requests:
            sample_id = _attribute(request, "sample_id")
            branch_key = _attribute(request, "branch_key")
            step_id = _integer(_attribute(request, "step_id"), "step_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("sidecar sample_id must be a nonempty string")
            sample_ids.add(sample_id)
            request_modality = _attribute(request, "modality")
            if not isinstance(request_modality, str) or request_modality != modality:
                raise ValueError("sidecar modality disagrees with its packed group")
            role = _attribute(request, "role")
            if not isinstance(role, str) or role not in ("chosen_entropy", "rejected_dpo"):
                raise ValueError("sidecar role must be chosen_entropy or rejected_dpo")
            if not isinstance(branch_key, str) or branch_key != f"{sample_id}:{step_id}:{modality}:{role}":
                raise ValueError("sidecar branch_key must match the stable branch identity")
            if branch_key in seen_keys:
                raise ValueError("sidecar branch_key must be unique")
            seen_keys.add(branch_key)
            boundary = _integer(_attribute(request, "replay_boundary"), "replay_boundary")
            visible_end = _integer(_attribute(request, "clean_visible_end"), "clean_visible_end")
            if boundary >= clean_length or visible_end != boundary:
                raise ValueError("sidecar clean_visible_end must equal an in-bounds replay_boundary")
            if stage2_debugging and clean_attention_mask[0, boundary] != 1:
                raise ValueError("sidecar replay boundary must be an unpadded clean token")
            token_inputs, targets, scores, offsets = (
                _vector(_attribute(request, field), field, device)
                for field in ("input_token_ids", "target_token_ids", "score_offsets", "logical_position_offsets")
            )
            length = token_inputs.numel()
            expected_offsets = torch.arange(length, device=device)
            if (
                targets.numel() != length
                or scores.numel() != length or offsets.numel() != length
                or (stage2_debugging and not torch.equal(scores, expected_offsets))
                or (stage2_debugging and not torch.equal(offsets, expected_offsets))
            ):
                raise ValueError("sidecar targets, scores and logical offsets must cover every query in order")
            if stage2_debugging and (bool((targets < 0).any()) or bool((token_inputs < 0).any())):
                raise ValueError("sidecar token IDs must be nonnegative")
            if stage2_debugging and (token_inputs[0] != clean_input_ids[0, boundary] or not torch.equal(token_inputs[1:], targets[:-1])):
                raise ValueError("sidecar inputs must replay the clean boundary followed by previous targets")
            blocked = _attribute(request, "blocked_clean_interval")
            removed = getattr(request, "removed_clean_interval", None)
            target_start = boundary + 1
            if removed is not None:
                if (modality != "visual" or role != "chosen_entropy" or blocked is not None
                    or not isinstance(removed, tuple) or len(removed) != 2
                    or any(type(value) is not int for value in removed)
                    or removed[0] != boundary + 1
                    or not removed[0] + 2 < removed[1] < clean_length):
                    raise ValueError("invalid full visual removal interval")
                target_start = removed[1]
            if role == "chosen_entropy":
                if removed is None and (
                    not isinstance(blocked, tuple)
                    or len(blocked) != 2
                    or any(type(value) is not int for value in blocked)
                    or not 0 <= blocked[0] < blocked[1] <= boundary
                ):
                    raise ValueError("chosen sidecar requires a nonempty blocked interval before replay")
                if stage2_debugging and not torch.equal(targets, clean_input_ids[0, target_start : target_start + length]):
                    raise ValueError("chosen sidecar targets must match the clean trajectory")
                chosen[(sample_id, step_id)] = (
                    boundary, tuple(targets.tolist()) if stage2_debugging else length, target_start)
            elif blocked is not None:
                raise ValueError("rejected sidecar must retain clean context without an ablation interval")
            packed.append(
                PackedSidecarBranch(
                    branch_key,
                    sample_id,
                    step_id,
                    modality,
                    role,
                    boundary,
                    visible_end,
                    blocked,
                    cursor,
                    cursor + length,
                    scores.clone(),
                    targets.clone(),
                )
            )
            inputs.append(token_inputs)
            positions.append(clean_position_ids[..., boundary : boundary + 1] + offsets)
            cursor += length
        if not chosen or any((branch.sample_id, branch.step_id) not in chosen for branch in packed):
            raise ValueError("sidecar group must include chosen entropy for every requested step")
        if any(branch.replay_boundary != chosen[(branch.sample_id, branch.step_id)][0] for branch in packed):
            raise ValueError("sidecar branches for the same step must share one replay boundary")
        chosen_by_modality.append({key: (value[2], value[1]) for key, value in chosen.items()})
        layouts.append(
            PackedSidecarLayout(
                modality,
                clean_length,
                tuple(packed),
                torch.cat(inputs).unsqueeze(0),
                torch.cat(positions, dim=-1),
                clean_attention_mask.clone(),
            )
        )
    if len(sample_ids) != 1 or any(group != chosen_by_modality[0] for group in chosen_by_modality[1:]):
        raise ValueError("sidecar groups must cover identical chosen steps in one clean sample")
    return tuple(layouts)


def _sidecar_visibility(layout: PackedSidecarLayout, device: torch.device):
    """Capture O(Q + KV) metadata once; the mask callback uses only tensor logic.

    A query sees unpadded, unablated clean keys before its replay boundary, plus
    its own branch up to its physical query index. Clean boundary copies,
    clean future tokens, and every other replay branch are invisible.
    """
    queries = layout.input_token_ids.shape[1]
    clean_length = layout.clean_length
    metadata = [[], [], [], []]
    for branch in layout.branches:
        left, right = branch.blocked_clean_interval or (0, 0)
        for values, value in zip(metadata, (branch.query_start, branch.clean_visible_end, left, right)):
            values.extend([value] * (branch.query_end - branch.query_start))
    branch_start, visible_end, blocked_start, blocked_end = torch.tensor(metadata, device=device).unbind()
    clean_valid = layout.clean_attention_mask[0].to(device=device, dtype=torch.bool)
    # PyTorch 2.7's Flex HOP cannot lower arithmetic on captured dynamic
    # SymInts. Keep bounds as device tensors, just like the branch metadata.
    query_count, clean_count, key_count = torch.tensor(
        [queries, clean_length, clean_length + queries], device=device
    ).unbind()

    def sidecar_visibility(batch_index, head_index, query_index, key_index):
        # Flex kernels may evaluate padded lanes; clamp every metadata lookup.
        query = query_index.clamp(min=0).minimum(query_count - 1)
        clean_key = key_index.clamp(min=0).minimum(clean_count - 1)
        own_causal = (key_index >= clean_count + branch_start[query]) & (key_index <= clean_count + query_index)
        clean_prefix = (key_index < visible_end[query]) & clean_valid[clean_key]
        blocked = (key_index >= blocked_start[query]) & (key_index < blocked_end[query])
        in_bounds = (query_index < query_count) & (key_index < key_count)
        return in_bounds & ((clean_prefix & ~blocked) | own_causal)

    return sidecar_visibility


def build_dense_sidecar_mask(layout: PackedSidecarLayout, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Materialize the common visibility rule as the eager/SDPA additive oracle."""
    queries = layout.input_token_ids.shape[1]
    visible = _sidecar_visibility(layout, device)(
        0, 0, torch.arange(queries, device=device)[:, None],
        torch.arange(layout.clean_length + queries, device=device)[None, :],
    )
    return torch.zeros((1, 1, queries, layout.clean_length + queries), dtype=dtype, device=device).masked_fill_(
        ~visible, torch.finfo(dtype).min
    )


def build_flex_sidecar_block_mask(layout: PackedSidecarLayout, device: torch.device) -> "BlockMask":
    """Build sparse blocks from the same predicate, never a dense additive mask."""
    from torch.nn.attention.flex_attention import create_block_mask

    queries = layout.input_token_ids.shape[1]
    builder = torch.compile(create_block_mask, dynamic=True) if device.type == "cuda" else create_block_mask
    return builder(
        _sidecar_visibility(layout, device), B=1, H=None, Q_LEN=queries,
        KV_LEN=layout.clean_length + queries, device=device,
    )


def execute_stage2_sidecar_groups(
    *,
    layouts: tuple[PackedSidecarLayout, ...],
    prefix_state: DifferentiablePrefixState,
    embed_tokens,
    language_model,
    lm_head,
    clean_decoder_stats: Stage2DecoderCallStats,
    attention_backend: str = "eager",
) -> tuple[Stage2SidecarLogits, Stage2ComputeStats]:
    """Run each modality once using a fresh wrapper over the exact clean tensors."""
    if not torch.is_grad_enabled():
        raise ValueError("sidecar execution requires gradients enabled")
    if not isinstance(prefix_state, DifferentiablePrefixState) or not prefix_state.ordered_tensors:
        raise ValueError("sidecar execution requires a retained differentiable prefix")
    if tuple(layout.modality for layout in layouts) not in (("visual",), ("visual", "text")):
        raise ValueError("sidecar execution requires visual, or visual and text packed layouts")
    by_branch_key = {}
    chosen_tokens = rejected_tokens = avoided_tokens = calls = 0
    for layout in layouts:
        if prefix_state.sequence_length != layout.clean_length:
            raise ValueError("sidecar layout does not match the retained clean prefix length")
        cache = rebuild_stage2_dynamic_cache(prefix_state)
        inputs = embed_tokens(layout.input_token_ids)
        if attention_backend == "flex_attention":
            mask = build_flex_sidecar_block_mask(layout, inputs.device)
        else:
            mask = build_dense_sidecar_mask(layout, dtype=inputs.dtype, device=inputs.device)
        output = language_model(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=layout.position_ids,
            past_key_values=cache,
            cache_position=torch.arange(
                layout.clean_length, layout.clean_length + inputs.shape[1], device=inputs.device
            ),
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        calls += 1
        logits = lm_head(output.last_hidden_state)[0]
        for branch in layout.branches:
            by_branch_key[branch.branch_key] = logits[branch.query_start + branch.score_offsets]
            count = branch.target_token_ids.numel()
            if branch.role == "chosen_entropy":
                chosen_tokens += count
            else:
                rejected_tokens += count
            avoided_tokens += branch.replay_boundary
    return Stage2SidecarLogits(by_branch_key), Stage2ComputeStats(
        logical_groups=1 + calls,
        top_level_policy_calls=1,
        clean_decoder_calls=clean_decoder_stats.clean_forward_calls,
        sidecar_decoder_calls=calls,
        branch_microbatches=calls,
        clean_prefix_tokens=prefix_state.sequence_length,
        prefix_tokens_avoided=avoided_tokens,
        chosen_entropy_tokens=chosen_tokens,
        rejected_dpo_tokens=rejected_tokens,
        reference_calls=0,
        reference_score_hits=0,
        _decoder_call_stats=clean_decoder_stats,
    )
