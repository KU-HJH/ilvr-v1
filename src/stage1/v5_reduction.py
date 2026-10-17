"""Token DIL and original-block auxiliary normalization for the actual window."""
from dataclasses import dataclass


@dataclass(frozen=True)
class V5Window:
    global_original_steps: int
    actual_microbatches: int
    configured_gas: int
    world_size: int
    global_dil_tokens: int

    def __post_init__(self):
        values = (self.global_original_steps, self.actual_microbatches,
                  self.configured_gas, self.world_size)
        if (any(type(v) is not int for v in values) or self.global_original_steps < 0
                or type(self.global_dil_tokens) is not int or self.global_dil_tokens < 0
                or min(values[1:]) <= 0 or self.actual_microbatches > self.configured_gas):
            raise ValueError('invalid v5 accumulation window')


def v5_dil_scale(window: V5Window) -> float:
    # Compensate the same native GAS division and rank averaging as auxiliaries,
    # but use the eligible token count instead of original blocks.
    return (window.configured_gas * window.world_size / window.global_dil_tokens
            if window.global_dil_tokens else 0.)


def v5_auxiliary_scale(window: V5Window) -> float:
    # Pinned Trainer divides by configured GAS; Accelerate's DeepSpeed dispatch
    # disables engine GAS scaling. ZeRO averages rank gradients by world size.
    return (window.configured_gas * window.world_size / window.global_original_steps
            if window.global_original_steps else 0.)
