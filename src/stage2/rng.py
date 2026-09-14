"""Scoped RNG snapshots for paired Stage 2 student replays."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class RNGState:
    """Cloned CPU and already-active CUDA generator state for one process."""

    cpu_state: torch.Tensor
    cuda_states: tuple[tuple[int, torch.Tensor], ...]


def capture_rng_state() -> RNGState:
    """Capture RNG state without causing a previously unused CUDA context to start."""
    cpu_state = torch.get_rng_state().clone()
    cuda_states: tuple[tuple[int, torch.Tensor], ...] = ()
    if torch.cuda.is_initialized():
        device = torch.cuda.current_device()
        cuda_states = ((device, torch.cuda.get_rng_state(device).clone()),)
    return RNGState(cpu_state=cpu_state, cuda_states=cuda_states)


def _set_rng_state(state: RNGState) -> None:
    torch.set_rng_state(state.cpu_state.clone())
    for device, cuda_state in state.cuda_states:
        torch.cuda.set_rng_state(cuda_state.clone(), device=device)


@contextmanager
def replay_rng_state(state: RNGState) -> Iterator[None]:
    """Replay a snapshot temporarily and restore the caller's streams on exit."""
    if not isinstance(state, RNGState):
        raise TypeError("state must be an RNGState")
    outer = capture_rng_state()
    requested_devices = tuple(device for device, _ in state.cuda_states)
    outer_devices = tuple(device for device, _ in outer.cuda_states)
    if requested_devices != outer_devices:
        raise RuntimeError(
            "RNGState CUDA devices do not match the currently initialized device set"
        )
    _set_rng_state(state)
    try:
        yield
    finally:
        _set_rng_state(outer)
