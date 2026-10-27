"""Bound matched-standard continuation to its saved objective and execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


MATCHED_RESUME_FILENAME = 'stage1_matched_resume.json'

WORKER_FIELDS = (
    'stage1_variant', 'stage1_match_v5_execution', 'stage1_ce_entropy_top_percent',
    'ce_weight', 'sim_weight', 'ema_tau', 'latent_size', 'coverage_p', 'image_pool_k',
    'helper_group_L', 'textual_states', 'query_mode', 'compress', 'tuning_method',
    'lora_r', 'lora_alpha', 'lora_dropout', 'lora_target_modules',
    'attention_implementation', 'vision_attention_implementation',
    'max_seq_length_train',
)
TRAINING_FIELDS = (
    'max_steps', 'num_train_epochs', 'per_device_train_batch_size',
    'gradient_accumulation_steps', 'dataloader_num_workers', 'learning_rate',
    'warmup_steps', 'warmup_ratio', 'seed', 'data_seed', 'bf16', 'fp16',
    'gradient_checkpointing', 'optim', 'lr_scheduler_type', 'weight_decay',
    'max_grad_norm', 'adam_beta1', 'adam_beta2', 'adam_epsilon',
    'dataloader_drop_last', 'dataloader_pin_memory', 'dataloader_persistent_workers',
    'ignore_data_skip', 'save_steps', 'save_strategy', 'save_total_limit',
    'max_seq_length', 'ddp_find_unused_parameters',
)


def _values(obj, fields):
    from enum import Enum
    return {key: (getattr(obj, key).value if isinstance(getattr(obj, key), Enum)
                  else getattr(obj, key)) for key in fields if hasattr(obj, key)}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def capture_matched_resume_profile(worker, training):
    """Bind effective settings and input bytes, without machine-specific paths."""
    base = Path(worker.model)
    files = sorted({*base.glob('*.safetensors'), *base.glob('*.bin'),
                    *(base / name for name in (
                        'config.json', 'model.safetensors.index.json',
                        'tokenizer_config.json') if (base / name).is_file())})
    if not (base / 'config.json').is_file() or not any(
            p.suffix in ('.safetensors', '.bin') for p in files):
        raise ValueError('matched resume requires original base model content')
    settings = _values(worker, WORKER_FIELDS)
    targets = settings.get('lora_target_modules')
    if isinstance(targets, str):
        settings['lora_target_modules'] = sorted(targets.split(','))
    return {'worker': settings, 'training': _values(training, TRAINING_FIELDS),
            'data_sha256': sha256_file(worker.data_path),
            'base_sha256': {p.name: sha256_file(p) for p in files}}


def write_matched_resume_contract(checkpoint, profile, *, source):
    """Add continuation metadata; never replace a legacy marker or contract."""
    target = Path(checkpoint) / MATCHED_RESUME_FILENAME
    contract = {'schema_version': 1, 'kind': 'stage1_matched_resume',
                'source': source, 'profile': profile}
    if target.exists() or target.is_symlink():
        if _read_json(target) != contract:
            raise ValueError('conflicting existing matched resume contract')
    else:
        with target.open('x') as stream:
            json.dump(contract, stream, sort_keys=True, allow_nan=False, indent=2)
            stream.write('\n')
    load_matched_resume_contract(checkpoint)
    return contract


def validate_matched_resume_profile(saved, requested):
    """Refuse any objective, schedule, execution or input-content drift."""
    def canonical(value):
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))
    if canonical(saved) != canonical(requested):
        raise ValueError('matched resume profile differs from saved execution or inputs')


def validate_matched_trainer_resume(trainer, contract):
    """Check the direct Trainer entrypoint as well as composition's input check."""
    from copy import deepcopy
    profile = getattr(trainer, 'matched_resume_profile', None)
    if profile is None:
        raise ValueError('matched resume requires the verified input profile')
    live = deepcopy(profile)
    live['training'] = _values(trainer.args, TRAINING_FIELDS)
    live['worker'].update({
        'stage1_variant': 'standard', 'stage1_match_v5_execution': True,
        'stage1_ce_entropy_top_percent': trainer.ce_entropy_top_percent,
        'ce_weight': trainer.ce_weight, 'sim_weight': trainer.sim_weight,
        'ema_tau': trainer.ema_teacher.decay,
    })
    for key in ('coverage_p', 'image_pool_k', 'helper_group_L', 'textual_states'):
        if key in live['worker']:
            live['worker'][key] = getattr(trainer, key)
    validate_matched_resume_profile(contract['profile'], live)


def checkpoint_source(checkpoint):
    import subprocess
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    checkpoint = Path(checkpoint)
    return {'run_id': checkpoint.parent.name,
            'checkpoint_step': int(checkpoint.name.removeprefix('checkpoint-')),
            'commit': commit}


def _read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError(f'matched resume requires a regular bounded file: {path}')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate matched resume field: {key}')
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError('nonfinite matched resume data')))


def load_matched_resume_contract(checkpoint):
    """Read the continuation binding without altering a historical v1 marker."""
    from .matched_execution import _IDENTITY, MATCHED_EXECUTION_FILENAME
    checkpoint = Path(checkpoint)
    contract = _read_json(checkpoint / MATCHED_RESUME_FILENAME)
    if (not isinstance(contract, dict) or contract.get('schema_version') != 1
            or contract.get('kind') != 'stage1_matched_resume'):
        raise ValueError('unsupported matched resume contract')
    source = contract.get('source')
    profile = contract.get('profile')
    if not isinstance(source, dict) or not isinstance(profile, dict):
        raise ValueError('matched resume requires source and profile')
    step = source.get('checkpoint_step')
    if type(step) is not int or step <= 0 or checkpoint.name != f'checkpoint-{step}':
        raise ValueError('matched resume checkpoint step mismatch')
    commit = source.get('commit')
    if (not isinstance(commit, str) or len(commit) != 40
            or any(c not in '0123456789abcdef' for c in commit)):
        raise ValueError('matched resume source commit is invalid')
    worker, training = profile.get('worker'), profile.get('training')
    if not isinstance(worker, dict) or not isinstance(training, dict):
        raise ValueError('matched resume requires saved worker/training settings')
    if (worker.get('stage1_variant') != 'standard'
            or worker.get('stage1_match_v5_execution') is not True
            or worker.get('ce_weight') != 1. or worker.get('sim_weight') != 1.
            or worker.get('ema_tau') != 1.):
        raise ValueError('unsupported matched resume objective')
    required = {'max_steps', 'num_train_epochs', 'per_device_train_batch_size',
                'gradient_accumulation_steps', 'dataloader_num_workers',
                'learning_rate', 'warmup_steps', 'seed'}
    if not required <= training.keys() or step > training['max_steps']:
        raise ValueError('matched resume training schedule is incomplete')
    hashes = [profile.get('data_sha256'), *profile.get('base_sha256', {}).values()]
    if len(hashes) < 2 or any(not isinstance(h, str) or len(h) != 64
                             or any(c not in '0123456789abcdef' for c in h)
                             for h in hashes):
        raise ValueError('matched resume input hashes are incomplete')
    marker = _read_json(checkpoint / MATCHED_EXECUTION_FILENAME)
    expected = dict(_IDENTITY)
    percent = worker.get('stage1_ce_entropy_top_percent')
    if percent is not None:
        from .entropy_ce import validate_ce_entropy_percent
        expected['ce_entropy_selection'] = {
            'percent': validate_ce_entropy_percent(percent),
            'source': 'clean_student_logits', 'temperature': 1,
            'candidate_pool': 'supervised_non_special_text',
            'quota': 'sample_local_ceil', 'ties': 'original_position',
            'special_targets': 'preserve', 'reduction': 'retained_label_sample_mean',
        }
    if marker != expected:
        raise ValueError('matched resume objective marker mismatch')
    return contract


from transformers import TrainerCallback


class MatchedCheckpointStop(TrainerCallback):
    """Keep the original schedule; stop only after a complete forced save."""
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            control.should_training_stop = True
        return control
