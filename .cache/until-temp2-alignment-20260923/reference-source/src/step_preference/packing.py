"""Model-neutral construction of shared-prefix sidecar replay requests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Sequence

import torch


if TYPE_CHECKING:
    from stage2.data import StepPreferenceBatchItem


SidecarModality = Literal["visual", "text"]
SidecarRole = Literal["chosen_entropy", "rejected_dpo"]


@dataclass(frozen=True)
class SidecarBranchRequest:
    """One short, isolated causal replay rooted in the clean trajectory."""

    branch_key: str
    sample_id: str
    step_id: int
    modality: SidecarModality
    role: SidecarRole
    replay_boundary: int
    input_token_ids: torch.LongTensor
    target_token_ids: torch.LongTensor
    score_offsets: torch.LongTensor
    logical_position_offsets: torch.LongTensor
    clean_visible_end: int
    blocked_clean_interval: tuple[int, int] | None
    removed_clean_interval: tuple[int, int] | None = None


@dataclass(frozen=True)
class SidecarExecutionPlan:
    """The visual and textual sidecar groups for one clean trajectory."""

    visual: tuple[SidecarBranchRequest, ...]
    text: tuple[SidecarBranchRequest, ...]
    modalities: str = "visual_text"
    stage2_debugging: bool = True


def _require_token_vector(value: object, field_name: str) -> torch.LongTensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"{field_name} must be a one-dimensional tensor")
    if value.dtype != torch.long:
        raise ValueError(f"{field_name} must use torch.long")
    if value.numel() == 0:
        raise ValueError(f"{field_name} must contain at least one target token")
    return value


def _require_clean_input_ids(clean_input_ids: object) -> torch.LongTensor:
    if not isinstance(clean_input_ids, torch.Tensor) or clean_input_ids.ndim != 2:
        raise ValueError("clean_input_ids must be a two-dimensional tensor")
    if clean_input_ids.dtype != torch.long:
        raise ValueError("clean_input_ids must use torch.long")
    if clean_input_ids.shape[0] != 1:
        raise ValueError("sidecar planning requires one clean input row")
    if clean_input_ids.shape[1] == 0:
        raise ValueError("clean_input_ids must not be empty")
    return clean_input_ids


def _blocked_interval(item: StepPreferenceBatchItem, modality: SidecarModality) -> tuple[int, int]:
    span = item.span
    if modality == "visual":
        return span.latent_start, span.latent_end
    return span.prev_text_start, span.prev_text_end


def _validate_span(*, item: StepPreferenceBatchItem, clean_length: int) -> None:
    span = item.span
    if not (
        0
        <= span.prev_text_start
        < span.prev_text_end
        <= span.latent_start
        < span.latent_end
        < span.target_text_start
        < span.target_text_end
        <= clean_length
    ):
        raise ValueError("reasoning step span components must be ordered within clean input")


def _validate_chosen_item(
    *, clean_input_ids: torch.LongTensor, item: StepPreferenceBatchItem,
    stage2_debugging: bool = True,
) -> tuple[torch.LongTensor, torch.LongTensor, int]:
    targets = _require_token_vector(item.chosen_target_ids, "chosen target ids")
    predictors = _require_token_vector(
        item.chosen_predictor_positions, "chosen predictor positions"
    )
    if predictors.numel() != targets.numel():
        raise ValueError("chosen predictor and target length mismatch")
    if stage2_debugging and not torch.equal(
        predictors,
        torch.arange(
            predictors[0],
            predictors[0] + predictors.numel(),
            dtype=predictors.dtype,
            device=predictors.device,
        ),
    ):
        raise ValueError("chosen predictor positions must be contiguous")

    replay_boundary = (int(predictors[0].item()) if stage2_debugging
                       else item.span.target_text_start - 1)
    if not 0 <= replay_boundary < clean_input_ids.shape[1]:
        raise ValueError("replay boundary is out of clean input bounds")
    if predictors.device != clean_input_ids.device or targets.device != clean_input_ids.device:
        raise ValueError("chosen sidecar tensors must share the clean input device")
    _validate_span(item=item, clean_length=clean_input_ids.shape[1])
    if replay_boundary != item.span.target_text_start - 1:
        raise ValueError("replay boundary must predict the first target token")
    if stage2_debugging and int(predictors[-1].item()) != item.span.target_text_end - 2:
        raise ValueError("chosen predictor positions disagree with the target span")
    if stage2_debugging and int(predictors[-1].item()) >= clean_input_ids.shape[1]:
        raise ValueError("chosen predictor positions are out of clean input bounds")
    if stage2_debugging and not torch.equal(
        targets,
        clean_input_ids[0, item.span.target_text_start : item.span.target_text_end],
    ):
        raise ValueError("chosen target ids disagree with the clean trajectory")

    return predictors, targets, replay_boundary


def _validate_blocked_interval(
    *, interval: tuple[int, int], replay_boundary: int, modality: SidecarModality
) -> None:
    start, end = interval
    if not 0 <= start < end <= replay_boundary:
        raise ValueError(
            f"{modality} blocked clean interval must be non-empty and strictly before replay"
        )


def _make_branch(
    *,
    item: StepPreferenceBatchItem,
    modality: SidecarModality,
    role: SidecarRole,
    replay_boundary: int,
    replay_token_id: torch.LongTensor,
    target_token_ids: torch.LongTensor,
    blocked_clean_interval: tuple[int, int] | None,
) -> SidecarBranchRequest:
    input_token_ids = torch.cat((replay_token_id.reshape(1), target_token_ids[:-1])).clone()
    target_token_ids = target_token_ids.clone()
    score_offsets = torch.arange(
        target_token_ids.numel(), device=target_token_ids.device, dtype=torch.long
    )
    branch_key = f"{item.sample_id}:{item.span.step_id}:{modality}:{role}"
    return SidecarBranchRequest(
        branch_key=branch_key,
        sample_id=item.sample_id,
        step_id=item.span.step_id,
        modality=modality,
        role=role,
        replay_boundary=replay_boundary,
        input_token_ids=input_token_ids,
        target_token_ids=target_token_ids,
        score_offsets=score_offsets,
        logical_position_offsets=torch.arange(
            input_token_ids.numel(), device=input_token_ids.device, dtype=torch.long
        ),
        clean_visible_end=replay_boundary,
        blocked_clean_interval=blocked_clean_interval,
    )


def _rejected_branch(
    *,
    clean_input_ids: torch.LongTensor,
    item: StepPreferenceBatchItem,
    modality: SidecarModality,
    replay_boundary: int,
) -> SidecarBranchRequest | None:
    if modality == "visual":
        is_valid = item.negative.valid_visual_negative
        branch_inputs = item.rejected_visual_inputs
        target_ids = item.rejected_visual_target_ids
        expected_ids = item.negative.rejected_visual_token_ids
    else:
        is_valid = item.negative.valid_textual_negative
        branch_inputs = item.rejected_text_inputs
        target_ids = item.rejected_text_target_ids
        expected_ids = item.negative.rejected_textual_token_ids
    if not is_valid:
        return None
    if not isinstance(branch_inputs, dict) or "input_ids" not in branch_inputs:
        raise ValueError(f"valid rejected {modality} branch requires input_ids")
    branch_ids = branch_inputs["input_ids"]
    if (
        not isinstance(branch_ids, torch.Tensor)
        or branch_ids.ndim != 2
        or branch_ids.shape[0] != 1
        or branch_ids.dtype != torch.long
    ):
        raise ValueError(f"rejected {modality} input_ids must be one unbatched long tensor")
    targets = _require_token_vector(target_ids, f"rejected {modality} target ids")
    if branch_ids.device != targets.device:
        raise ValueError(f"rejected {modality} inputs and targets must share a device")
    if replay_boundary + targets.numel() >= branch_ids.shape[1]:
        raise ValueError(f"rejected {modality} inputs do not contain the full target scaffold")
    if tuple(targets.tolist()) != expected_ids:
        raise ValueError(f"rejected {modality} target ids disagree with the negative record")
    target_start = replay_boundary + 1
    target_end = target_start + targets.numel()
    if not torch.equal(branch_ids[0, target_start:target_end], targets):
        raise ValueError(f"rejected {modality} input scaffold disagrees with target ids")
    if not torch.equal(branch_ids[0, replay_boundary], clean_input_ids[0, replay_boundary]):
        raise ValueError(f"rejected {modality} replay boundary disagrees with clean input")
    return _make_branch(
        item=item,
        modality=modality,
        role="rejected_dpo",
        replay_boundary=replay_boundary,
        replay_token_id=clean_input_ids[0, replay_boundary],
        target_token_ids=targets,
        blocked_clean_interval=None,
    )


def build_sidecar_execution_plan(
    *,
    clean_input_ids: torch.LongTensor,
    items: Sequence[StepPreferenceBatchItem],
    include_rejected: bool = True,
    stage2_debugging: bool = True,
) -> SidecarExecutionPlan:
    """Build all local sidecar replays without invoking a model or decoder."""
    clean_input_ids = _require_clean_input_ids(clean_input_ids)
    visual: list[SidecarBranchRequest] = []
    text: list[SidecarBranchRequest] = []
    seen_identities: set[tuple[str, int]] = set()

    for item in items:
        if include_rejected and (
            item.sample_id != item.negative.sample_id or item.span.step_id != item.negative.step_id
        ):
            raise ValueError("step preference item identity disagrees with its negative record")
        identity = (item.sample_id, item.span.step_id)
        if identity in seen_identities:
            raise ValueError("duplicate sidecar branch identity")
        seen_identities.add(identity)
        _, chosen_targets, replay_boundary = _validate_chosen_item(
            clean_input_ids=clean_input_ids, item=item, stage2_debugging=stage2_debugging,
        )

        for modality, group in (("visual", visual), ("text", text)):
            interval = _blocked_interval(item, modality)
            _validate_blocked_interval(
                interval=interval,
                replay_boundary=replay_boundary,
                modality=modality,
            )
            group.append(
                _make_branch(
                    item=item,
                    modality=modality,
                    role="chosen_entropy",
                    replay_boundary=replay_boundary,
                    replay_token_id=clean_input_ids[0, replay_boundary],
                    target_token_ids=chosen_targets,
                    blocked_clean_interval=interval,
                )
            )
            rejected = _rejected_branch(
                clean_input_ids=clean_input_ids,
                item=item,
                modality=modality,
                replay_boundary=replay_boundary,
            ) if include_rejected else None
            if rejected is not None:
                group.append(rejected)

    return SidecarExecutionPlan(visual=tuple(visual), text=tuple(text),
                               stage2_debugging=stage2_debugging)


def build_context_sidecar_execution_plan(*, clean_input_ids, items, ablation="attention_mask",
                                         modalities="visual_text", stage2_debugging=True):
    """Chosen-only V/T plans; never inspect a negative record or branch."""
    if ablation not in {"attention_mask", "remove_full_visual_block"}:
        raise ValueError("ablation must be attention_mask or remove_full_visual_block")
    if modalities not in {"visual", "visual_text"}:
        raise ValueError("modalities must be visual or visual_text")
    plan = build_sidecar_execution_plan(
        clean_input_ids=clean_input_ids, items=items, include_rejected=False,
        stage2_debugging=stage2_debugging,
    )
    if modalities == "visual":
        plan = replace(plan, text=(), modalities="visual")
    if ablation == "attention_mask":
        return plan
    visual = []
    for branch, item in zip(plan.visual, items, strict=True):
        span = item.span
        start, end = span.prev_text_end, span.target_text_start
        if start < 1 or span.latent_start != start + 1 or span.latent_end != end - 1:
            raise ValueError("full visual removal requires complete start/latent/end boundaries")
        boundary = start - 1
        visual.append(replace(
            branch, replay_boundary=boundary, clean_visible_end=boundary,
            input_token_ids=torch.cat((clean_input_ids[0, boundary:boundary + 1],
                                       branch.target_token_ids[:-1])).clone(),
            blocked_clean_interval=None, removed_clean_interval=(start, end),
        ))
    return replace(plan, visual=tuple(visual))
