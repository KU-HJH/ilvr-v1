"""Prepare one retained matched Stage1 checkpoint in a fresh canonical run."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .matched_resume import (
    MATCHED_RESUME_FILENAME, load_matched_resume_contract, sha256_file,
    validate_matched_resume_profile, write_matched_resume_contract,
)


def import_legacy_profile(checkpoint, worker, records, *, source_commit):
    """Bind a trusted, completed legacy checkpoint to its two launch records.

    training_args.bin is the original trusted Trainer pickle, never an external
    download. This function reads source state and does not change source files.
    """
    import torch
    from .matched_resume import capture_matched_resume_profile
    checkpoint = Path(checkpoint)
    if len(records) != 2:
        raise ValueError('legacy resume requires both source node records')
    saved = torch.load(checkpoint/'training_args.bin', map_location='cpu', weights_only=False)
    names = {'epochs': 'num_train_epochs', 'warm_up_steps': 'warmup_steps'}
    for key in ('max_steps', 'per_device_train_batch_size', 'gradient_accumulation_steps',
                'dataloader_num_workers', 'learning_rate', 'seed', 'save_steps',
                'save_total_limit', 'save_strategy', 'gradient_checkpointing',
                'epochs', 'warm_up_steps'):
        if hasattr(worker, key):
            actual = getattr(worker, key)
            expected = getattr(saved, names.get(key, key))
            if actual != expected and actual != getattr(expected, 'value', expected):
                raise ValueError(f'legacy saved Trainer profile mismatch: {key}')
    profile = capture_matched_resume_profile(worker, saved)
    for record in records:
        if record.get('commit') != source_commit:
            raise ValueError('legacy source record commit mismatch')
        if record.get('training_data_sha256') != profile['data_sha256']:
            raise ValueError('legacy source data differs from launch record')
        base = {**record.get('base_metadata_sha256', {}), **record.get('base_weights_sha256', {})}
        if base != profile['base_sha256']:
            raise ValueError('legacy source base differs from launch record')
        for key, value in record.get('profile', {}).items():
            expected = getattr(saved, key, object())
            if value not in (expected, getattr(expected, 'value', expected), str(expected)):
                raise ValueError(f'legacy source saved profile mismatch: {key}')
        for key, value in record.get('execution_args', {}).items():
            if value != getattr(worker, key, object()):
                raise ValueError(f'legacy source worker record mismatch: {key}')
        for key, field in (('ce_entropy_top_percent', 'stage1_ce_entropy_top_percent'),
                           ('ce_weight', 'ce_weight'), ('sim_weight', 'sim_weight')):
            if record.get(key) != getattr(worker, field):
                raise ValueError(f'legacy source objective record mismatch: {key}')
    return profile


def assemble_source_checkpoint(runner, checkpoint, identity):
    """Use real supervisor evidence, preserving already assembled final state."""
    from src.multinode_artifacts import validate_canonical_checkpoint_manifest
    from src.multinode_runtime import collect_checkpoint_artifacts
    exits = {rank: runner.poll_exit(rank) for rank in (0, 1)}
    if exits != {0: 0, 1: 0}:
        raise ValueError('source resume preparation requires successful supervisors on both nodes')
    manifest = runner.read_canonical_artifact_manifest(checkpoint, identity)
    ranks = {entry.process_rank for entry in manifest.entries if entry.process_rank is not None}
    if ranks == {0, 1}:
        return collect_checkpoint_artifacts(runner, checkpoint_path=checkpoint,
            runtime_identity=identity, exit_codes=exits, world_size=4)
    return validate_canonical_checkpoint_manifest(manifest, expected_identity=identity)


def _canonical_child(path, root):
    root = Path(root).resolve(strict=True)
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or path.parent.parent.resolve() != root:
        raise ValueError('resume checkpoint must be an immediate canonical run/checkpoint-N')
    for component in (path.parent, path):
        if component.is_symlink():
            raise ValueError('resume run/checkpoint must not be a symlink')
    return path


def copy_resume_checkpoint(checkpoint, destination, *, profile, source, checkpoint_root):
    """Copy full state once and add the binding only to the fresh destination."""
    from .checkpointing import _validate_complete_trainer_state
    checkpoint = _canonical_child(checkpoint, checkpoint_root)
    destination = _canonical_child(destination, checkpoint_root)
    if destination.parent == checkpoint.parent or destination.parent.exists():
        raise FileExistsError('resume requires a fresh output run')
    if checkpoint.name != destination.name or checkpoint.name != f"checkpoint-{source['checkpoint_step']}":
        raise ValueError('resume checkpoint step mismatch')
    _validate_complete_trainer_state(checkpoint)
    state = json.loads((checkpoint/'trainer_state.json').read_text())
    if state.get('global_step') != source['checkpoint_step'] or state.get('max_steps') != profile['training']['max_steps']:
        raise ValueError('resume Trainer step/budget mismatch')
    original = {}
    for path in checkpoint.rglob('*'):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('resume source must contain only regular files/directories')
        if path.is_file(): original[path.relative_to(checkpoint)] = sha256_file(path)
    if (checkpoint/MATCHED_RESUME_FILENAME).exists():
        validate_matched_resume_profile(load_matched_resume_contract(checkpoint)['profile'], profile)
    destination.parent.mkdir()
    shutil.copytree(checkpoint, destination)
    for relative, expected in original.items():
        if sha256_file(destination/relative) != expected:
            raise ValueError('resume copy content mismatch')
    write_matched_resume_contract(destination, profile, source=source)
    _validate_complete_trainer_state(destination)
    return destination
