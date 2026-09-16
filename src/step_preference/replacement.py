"""Deterministic planning for same-image evidence replacement."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

from step_preference.sidecar import SidecarBranch, SidecarPlan


_SAMPLING_VERSION = "ilvr-evidence-replacement-v1"


@dataclass(frozen=True)
class ReplacementBranch:
    step_id: int
    latent_start: int
    latent_end: int
    prefix_length: int
    predictor_positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    source_kind: str
    source_positions: tuple[int, ...]
    donor_step_id: int | None


@dataclass(frozen=True)
class ReplacementPlan:
    root_length: int
    branches: tuple[ReplacementBranch, ...]


def _local_generator(*, seed: int, sample_id: str, step_id: int, kind: str):
    raw = json.dumps(
        [_SAMPLING_VERSION, seed, sample_id, step_id, kind],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    local_seed &= (1 << 63) - 1
    return torch.Generator(device="cpu").manual_seed(local_seed)


def _validate_branch(
    branch: SidecarBranch,
    *,
    expected_step_id: int,
    clean_ids: torch.Tensor,
    root_length: int,
    latent_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(branch, SidecarBranch):
        raise ValueError("sidecar branch must be a SidecarBranch")
    if branch.kind != "visual_off":
        raise ValueError("replacement branches must have kind visual_off")
    if branch.step_id != expected_step_id:
        raise ValueError("sidecar branch step_id must be contiguous and ordered")
    if branch.zero_first_embedding:
        raise ValueError("replacement does not support suppressed latent boundaries")
    if branch.rejected_model_inputs is not None:
        raise ValueError("visual_off branch cannot contain rejected model inputs")
    for field in (
        "token_ids",
        "clean_embedding_sources",
        "logical_positions",
        "selected_positions",
        "target_token_ids",
    ):
        if not isinstance(getattr(branch, field), tuple):
            raise ValueError(
                f"step {branch.step_id} {field} must be a tuple"
            )
    if (
        type(branch.blocked_start) is not int
        or type(branch.blocked_end) is not int
        or branch.blocked_start < 0
        or branch.blocked_end > root_length
        or branch.blocked_end - branch.blocked_start != latent_size
    ):
        raise ValueError(
            f"step {branch.step_id} latent block length must equal latent_size"
        )

    if (
        len(branch.token_ids) != len(branch.logical_positions)
        or any(
            type(position) is not int
            or position < 0
            or position >= root_length
            for position in branch.logical_positions
        )
        or tuple(branch.token_ids)
        != tuple(int(clean_ids[position].item()) for position in branch.logical_positions)
    ):
        raise ValueError(
            f"step {branch.step_id} token_ids must match logical input positions"
        )

    if (
        not branch.selected_positions
        or len(branch.selected_positions) != len(branch.target_token_ids)
        or any(type(position) is not int for position in branch.selected_positions)
        or any(
            position < 0 or position >= len(branch.logical_positions)
            for position in branch.selected_positions
        )
    ):
        raise ValueError(f"step {branch.step_id} predictor selection is invalid")
    predictor_positions = tuple(
        branch.logical_positions[position] for position in branch.selected_positions
    )
    if (
        branch.selected_positions != tuple(range(len(branch.logical_positions)))
        or branch.logical_positions
        != tuple(
            range(branch.logical_positions[0], branch.logical_positions[-1] + 1)
        )
        or any(type(position) is not int for position in predictor_positions)
        or any(
            right <= left
            for left, right in zip(predictor_positions, predictor_positions[1:])
        )
        or predictor_positions[0] < 0
        or predictor_positions[-1] + 1 >= root_length
    ):
        raise ValueError(
            f"step {branch.step_id} predictor positions must be the exact contiguous target range"
        )
    expected_targets = tuple(
        int(clean_ids[position + 1].item()) for position in predictor_positions
    )
    if tuple(branch.target_token_ids) != expected_targets:
        raise ValueError(
            f"step {branch.step_id} target_token_ids do not match input_ids"
        )
    if branch.blocked_end > predictor_positions[0]:
        raise ValueError(
            f"step {branch.step_id} latent block range must precede predictors"
        )
    if tuple(branch.clean_embedding_sources) != tuple(branch.logical_positions):
        raise ValueError(
            f"step {branch.step_id} clean embedding sources must match logical positions"
        )
    return predictor_positions, expected_targets


def build_replacement_plan(
    *,
    input_ids,
    sidecar_plan,
    image_token_id,
    image_count,
    latent_size,
    sample_id,
    seed,
) -> ReplacementPlan:
    """Plan detached latent replacements without consuming process-global RNG."""
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence_length]")
    if not isinstance(sidecar_plan, SidecarPlan):
        raise ValueError("sidecar_plan must be a SidecarPlan")
    if not sidecar_plan.context_preference:
        raise ValueError("sidecar_plan must have context_preference enabled")
    if type(sidecar_plan.root_length) is not int or not (
        0 <= sidecar_plan.root_length <= input_ids.shape[1]
    ):
        raise ValueError("sidecar_plan root_length is outside input_ids")
    if type(latent_size) is not int or latent_size <= 0:
        raise ValueError("latent_size must be a positive integer")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a nonempty string")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(image_token_id) is not int:
        raise ValueError("image_token_id must be an integer")
    if type(image_count) is not int or image_count < 0:
        raise ValueError("image_count must be a nonnegative integer")

    if (
        not isinstance(sidecar_plan.groups, tuple)
        or len(sidecar_plan.groups) != 2
        or any(not isinstance(group, tuple) for group in sidecar_plan.groups)
    ):
        raise ValueError("sidecar_plan groups must contain two branch tuples")
    if not isinstance(sidecar_plan.root_selected_positions, tuple):
        raise ValueError("sidecar_plan root_selected_positions must be a tuple")
    visual_branches, textual_branches = sidecar_plan.groups
    if textual_branches:
        raise ValueError("replacement requires a visual-only context plan")
    if not visual_branches:
        if sidecar_plan.root_selected_positions:
            raise ValueError("empty sidecar branches require empty root predictors")
        return ReplacementPlan(root_length=sidecar_plan.root_length, branches=())
    if image_count != 1:
        raise ValueError("image_count must be exactly one for replacement")

    clean_ids = input_ids[0, : sidecar_plan.root_length]
    image_positions = tuple(
        int(position)
        for position in torch.nonzero(clean_ids == image_token_id, as_tuple=False)
        .flatten()
        .tolist()
    )
    if not image_positions:
        raise ValueError("input_ids must contain image_token_id positions")

    validated: list[tuple[SidecarBranch, tuple[int, ...], tuple[int, ...]]] = []
    all_predictors: list[int] = []
    previous_latent_end: int | None = None
    for expected_step_id, branch in enumerate(visual_branches):
        predictors, targets = _validate_branch(
            branch,
            expected_step_id=expected_step_id,
            clean_ids=clean_ids,
            root_length=sidecar_plan.root_length,
            latent_size=latent_size,
        )
        if previous_latent_end is not None and previous_latent_end > branch.blocked_start:
            raise ValueError("latent blocks must follow absolute causal order")
        validated.append((branch, predictors, targets))
        all_predictors.extend(predictors)
        previous_latent_end = branch.blocked_end
    if tuple(all_predictors) != tuple(sidecar_plan.root_selected_positions):
        raise ValueError("root predictor positions do not match sidecar branches")

    planned: list[ReplacementBranch] = []
    for branch, predictors, targets in validated:
        if branch.step_id == 0:
            source_kind = "image"
            donor_step_id = None
            generator = _local_generator(
                seed=seed,
                sample_id=sample_id,
                step_id=branch.step_id,
                kind=source_kind,
            )
            count = len(image_positions)
            if count >= latent_size:
                selected = torch.randperm(count, generator=generator)[:latent_size]
            else:
                selected = torch.randint(
                    count, (latent_size,), generator=generator
                )
            source_positions = tuple(
                sorted(image_positions[int(index)] for index in selected.tolist())
            )
        else:
            source_kind = "donor"
            generator = _local_generator(
                seed=seed,
                sample_id=sample_id,
                step_id=branch.step_id,
                kind=source_kind,
            )
            donor_index = int(
                torch.randint(branch.step_id, (1,), generator=generator).item()
            )
            donor = validated[donor_index][0]
            if donor.blocked_end > branch.blocked_start:
                raise ValueError("donor latent block must precede current branch")
            donor_step_id = donor.step_id
            source_positions = tuple(range(donor.blocked_start, donor.blocked_end))
        planned.append(
            ReplacementBranch(
                step_id=branch.step_id,
                latent_start=branch.blocked_start,
                latent_end=branch.blocked_end,
                prefix_length=predictors[-1] + 1,
                predictor_positions=predictors,
                target_token_ids=targets,
                source_kind=source_kind,
                source_positions=source_positions,
                donor_step_id=donor_step_id,
            )
        )
    return ReplacementPlan(
        root_length=sidecar_plan.root_length,
        branches=tuple(planned),
    )
