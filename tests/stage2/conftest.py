from __future__ import annotations

import os

import pytest
from accelerate.state import AcceleratorState, GradientState, PartialState
from transformers.integrations.deepspeed import unset_hf_deepspeed_config


def _clear_training_process_state() -> None:
    os.environ.pop("ACCELERATE_USE_DEEPSPEED", None)
    unset_hf_deepspeed_config()
    AcceleratorState._reset_state(reset_partial_state=True)
    PartialState._reset_state()
    GradientState._reset_state()


@pytest.fixture(autouse=True)
def isolate_stage2_training_process_state():
    """Prevent one test's Accelerate/DeepSpeed state from leaking to another."""
    _clear_training_process_state()
    try:
        yield
    finally:
        _clear_training_process_state()
