"""Test fixtures for deterministic evidence-replacement planning."""

import torch

from step_preference.sidecar import build_context_sidecar_plan
from step_preference.spans import ReasoningStepSpan


def planner_case():
    ids = torch.tensor(
        [[11, 81, 80, 80, 80, 83, 14, 91, 90, 90, 92, 21, 22,
          91, 90, 90, 92, 23, 24]],
        dtype=torch.long,
    )
    spans = (
        ReasoningStepSpan(0, 0, 7, 8, 10, 11, 13, None),
        ReasoningStepSpan(1, 11, 13, 14, 16, 17, 19, None),
    )
    sidecar_plan = build_context_sidecar_plan(
        clean_input_ids=ids,
        attention_mask=torch.ones_like(ids),
        spans=spans,
        visual_only=True,
        suppress_latent_boundaries=False,
    )
    return ids, sidecar_plan


def three_step_planner_case():
    ids = torch.tensor(
        [[11, 81, 80, 80, 80, 83, 14, 91, 90, 90, 92, 21, 22,
          91, 90, 90, 92, 23, 24, 91, 90, 90, 92, 25, 26]],
        dtype=torch.long,
    )
    spans = (
        ReasoningStepSpan(0, 0, 7, 8, 10, 11, 13, None),
        ReasoningStepSpan(1, 11, 13, 14, 16, 17, 19, None),
        ReasoningStepSpan(2, 17, 19, 20, 22, 23, 25, None),
    )
    sidecar_plan = build_context_sidecar_plan(
        clean_input_ids=ids,
        attention_mask=torch.ones_like(ids),
        spans=spans,
        visual_only=True,
        suppress_latent_boundaries=False,
    )
    return ids, sidecar_plan
