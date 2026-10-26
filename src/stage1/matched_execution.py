"""Opt-in execution compatibility for the native Stage1 baseline control."""
from __future__ import annotations

import json
from pathlib import Path


MATCHED_EXECUTION_FILENAME = 'stage1_matched_execution.json'
_IDENTITY = {
    'schema_version': 1,
    'kind': 'stage1_standard_matched_v5_execution',
    'objective': 'CE + alignment',
    'ce_weight': 1.0,
    'sim_weight': 1.0,
    'epoch_schedule': 'actual_prepared_loader_windows',
    'tail_denominator': 'configured_gas_world_size',
    'resume_supported': False,
}


def validate_matched_standard_execution(args, tuning, profile):
    if (tuning.method != 'dora' or not profile.enabled
            or profile.zero_stage != 1 or profile.world_size != 4
            or args.ce_weight != 1. or args.sim_weight != 1.
            or args.ema_tau != 1. or args.per_device_train_batch_size != 1
            or args.gradient_accumulation_steps != 4 or not args.gradient_checkpointing
            or args.attention_implementation != 'sdpa'
            or args.vision_attention_implementation != 'flash_attention_2'):
        raise ValueError('unsupported matched standard Stage1 execution profile')


def enable_matched_standard_execution(trainer):
    trainer.stage1_match_v5_execution = True
    # The recorded V5 profile disables extra target scans and norm reductions.
    # Set this before the native constructor builds its teacher target builder.
    trainer.stage1_v5_debug = False
    # Reuse the existing parent-loop fix without changing legacy standard runs.
    trainer.stage1_v5_actual_window_schedule = True


def reject_matched_execution_resume(checkpoint, *, enabled):
    if checkpoint is None:
        return
    marker = Path(checkpoint) / MATCHED_EXECUTION_FILENAME
    if enabled or marker.exists() or marker.is_symlink():
        raise ValueError('matched standard Stage1 execution supports fresh runs only')


def write_matched_execution_identity(directory):
    marker = Path(directory) / MATCHED_EXECUTION_FILENAME
    payload = json.dumps(_IDENTITY, sort_keys=True, indent=2) + '\n'
    if marker.is_symlink():
        raise ValueError('matched execution identity must not be a symlink')
    if marker.exists():
        if marker.read_text() != payload:
            raise ValueError('conflicting matched execution identity')
        return
    with marker.open('x') as stream:
        stream.write(payload)
