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


_COORDINATOR_PAIRS = {'--run-id', '--attempt-id', '--zero-stage', '--main-process-port',
                      '--local-repo-root', '--remote-repo-root', '--remote-host-repo-root',
                      '--result-root'}


def _stage_arguments(command, *, preserve_runtime=False):
    result, index = [], 1
    while index < len(command):
        item = command[index]
        if item in _COORDINATOR_PAIRS or (not preserve_runtime and item in ('--resume', '--stop-after-checkpoint')):
            index += 2
        elif item == '--dry-run':
            index += 1
        else:
            result.append(item)
            index += 1
    return result


def resume_command(command, *, source_root, plan, checkpoint, stop=None):
    """Keep original training flags, remapping only execution/output locations."""
    cluster = plan.cluster
    result = [str(cluster.local_repo_root/'run_multinode_stage1_training.sh'),
        '--run-id', plan.run_id, '--attempt-id', plan.attempt_id,
        '--zero-stage', str(plan.zero_stage), '--main-process-port', str(plan.main_process_port),
        '--local-repo-root', str(cluster.local_repo_root),
        '--remote-repo-root', str(cluster.remote_repo_root),
        '--remote-host-repo-root', str(cluster.remote_host_repo_root)]
    prefix = str(source_root) + '/'
    result.extend(str(cluster.local_repo_root/item[len(prefix):]) if item.startswith(prefix)
                  else item for item in _stage_arguments(command))
    result.extend(('--resume', str(checkpoint)))
    if stop is not None: result.extend(('--stop-after-checkpoint', str(stop)))
    return result


def preparation_main(argv):
    """Assemble/copy one checkpoint; emit an existing-coordinator resume command."""
    import argparse
    import subprocess
    from src.multinode import (
        _REPOSITORY_ROOT, _checkpoint_artifact_identity, build_launch_plan, decode_launch_plan,
    )
    from src.multinode_runtime import (
        SubprocessCoordinatorRunner, build_owned_tmux_commands, redistribute_resume_checkpoint,
        publish_small_file,
    )
    from .matched_resume import _read_json, WORKER_FIELDS, _values
    from utils_deepseed import parse_args
    parser = argparse.ArgumentParser(prog='python -m src.multinode prepare-stage1-resume')
    parser.add_argument('--source-log-dir', type=Path, required=True)
    parser.add_argument('--source-attempt', default='initial')
    parser.add_argument('--checkpoint-step', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--main-process-port', type=int, default=29931)
    parser.add_argument('--remote-repo-root', type=Path, required=True)
    parser.add_argument('--remote-host-repo-root', type=Path, required=True)
    parser.add_argument('--stop-after-checkpoint', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        config = _read_json(args.source_log_dir/'launch-config.json')
        source_root = Path(config['local_runtime'])
        source_commit = config['head']
        if (len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit)
                or subprocess.run(['git','merge-base','--is-ancestor',source_commit,'HEAD'],
                                  cwd=_REPOSITORY_ROOT, check=False).returncode != 0):
            raise ValueError('source code must be a compatible reviewed ancestor of this checkout')
        source_plan = decode_launch_plan((source_root/'logs/multinode'/config['run_id'] /
            args.source_attempt/'launch-plan.json').read_bytes())
        if (source_plan.stage, source_plan.tuning_method, source_plan.zero_stage) != ('stage1','dora',1):
            raise ValueError('source is outside matched Stage1 DoRA ZeRO1 scope')
        if source_plan.run_id != config['run_id'] or source_plan.cluster.local_repo_root != source_root:
            raise ValueError('source launch records disagree')
        plan = build_launch_plan(stage='stage1', tuning_method='dora', zero_stage=1,
            run_id=args.run_id, attempt_id='resume1', main_process_port=args.main_process_port,
            local_repo_root=_REPOSITORY_ROOT, remote_repo_root=args.remote_repo_root,
            remote_host_repo_root=args.remote_host_repo_root)
        destination = _REPOSITORY_ROOT/'checkpoints'/args.run_id/f'checkpoint-{args.checkpoint_step}'
        ready = resume_command(config['command'], source_root=source_root, plan=plan,
                               checkpoint=destination, stop=args.stop_after_checkpoint)
        if args.dry_run:
            print(json.dumps({'command': ready, 'preparation_performed': False}))
            return 0
        commands = build_owned_tmux_commands(source_plan, commit=source_commit,
            stage_arguments=_stage_arguments(config['command'], preserve_runtime=True), result_root=source_root/'results',
            checkpoint_root=source_root/'checkpoints', log_root=source_root/'logs/multinode',
            stage_log_root=source_root/'logs/zero-stage-compatibility')
        from dataclasses import replace
        current_checkpoints = _REPOSITORY_ROOT/'checkpoints'
        if (source_root/'checkpoints').resolve(strict=True) != current_checkpoints.resolve(strict=True):
            raise ValueError('source and reader must share canonical checkpoint storage')
        # Supervisor commands stay bound to the original source. Artifact reads
        # use the reader's logical root, required by existing pull reservations.
        reader_plan = replace(source_plan,
            cluster=replace(source_plan.cluster, local_repo_root=_REPOSITORY_ROOT),
            deepspeed_config=_REPOSITORY_ROOT/'configs'/source_plan.deepspeed_config.name)
        reader = SubprocessCoordinatorRunner(reader_plan, commands=commands, commit=source_commit,
            checkpoint_root=current_checkpoints,
            log_root=_REPOSITORY_ROOT/'logs/stage1-resume-preparation'/plan.run_id/'source')
        source_path = current_checkpoints/config['run_id']/f'checkpoint-{args.checkpoint_step}'
        identity = _checkpoint_artifact_identity(source_plan, commit=source_commit, global_step=args.checkpoint_step)
        assemble_source_checkpoint(reader, source_path, identity)
        redistribute_resume_checkpoint(reader, checkpoint_path=source_path, checkpoint_identity=identity,
                                       requested_identity=identity, destination=source_path)
        if (source_path/MATCHED_RESUME_FILENAME).exists():
            binding = load_matched_resume_contract(source_path)
            profile, origin = binding['profile'], binding['source']
        else:
            workers = [parse_args(_read_json(args.source_log_dir/f'worker-rank{rank}-argv.json'))
                       for rank in (0,1)]
            if _values(workers[0], WORKER_FIELDS) != _values(workers[1], WORKER_FIELDS):
                raise ValueError('source node worker profiles disagree')
            records = [_read_json(args.source_log_dir/f'node-rank{rank}-resolved-final.json')
                       for rank in (0,1)]
            profile = import_legacy_profile(source_path, workers[1], records, source_commit=source_commit)
            origin = {'run_id': config['run_id'], 'checkpoint_step': args.checkpoint_step,
                      'commit': source_commit}
        if args.stop_after_checkpoint is not None and not args.checkpoint_step < args.stop_after_checkpoint < profile['training']['max_steps']:
            raise ValueError('planned resume stop must follow the selected step and precede full budget')
        copy_resume_checkpoint(source_path, destination, profile=profile, source=origin,
                               checkpoint_root=_REPOSITORY_ROOT/'checkpoints')
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=_REPOSITORY_ROOT,text=True).strip()
        writer = SubprocessCoordinatorRunner(plan, commands=None, commit=commit,
            log_root=_REPOSITORY_ROOT/'logs/stage1-resume-preparation'/plan.run_id/'destination')
        writer.publish_resume_copy(destination, _checkpoint_artifact_identity(plan, commit=commit,
                                                                             global_step=args.checkpoint_step))
        output = _REPOSITORY_ROOT/'logs'/plan.run_id
        publish_small_file(output/'resume-command.json', (json.dumps(ready,indent=2)+'\n').encode())
        launch = {'run_id': plan.run_id, 'head': commit, 'local_runtime': str(_REPOSITORY_ROOT),
                  'command': ready, 'source': origin}
        publish_small_file(output/'launch-config.json', (json.dumps(launch,indent=2)+'\n').encode())
        print(json.dumps({'checkpoint': str(destination), 'command': ready, 'source': origin}))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
