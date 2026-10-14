"""Stage1-v5 objective and content identity sidecar, with strict resume inheritance."""
from __future__ import annotations

from dataclasses import asdict, fields
import json
import math
from pathlib import Path
from typing import Mapping

from stage1.v5_config import Stage1V5Config, resolve_v5_config, v5_formula_id
from stage1.v5_reference import V5ReferenceIdentity

V5_CONTRACT_FILENAME = 'stage1_v5.json'
_SEMANTICS = {
    'reference': 'immutable_initial_effective_teacher_v1',
    'auxiliary_targets': 'immediate_original_text_exclude_crossing_special_padding_v1',
    'interventions': 'independent_clean_prefix_voff_body_textoff_previous_text_v1',
    'reduction': 'global_actual_window_original_blocks_empty_included_v1',
    'ce_alignment_text_reduction': 'configured_stage1_accumulation_v1',
    'text_target': 'detached_clean_full_vocab_temperature_squared_kl_v1',
    'text_schedule': 'from_first_update_no_kl_warmup_v1',
}


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError('v5 contract must contain finite JSON data') from error


def _execution(execution: Mapping[str, object]) -> None:
    if not isinstance(execution, Mapping):
        raise ValueError('invalid v5 execution profile')
    supported = {'tuning_method': 'dora', 'method': 'dora', 'ema_tau': 1.,
                 'zero_stage': 1, 'per_device_train_batch_size': 1,
                 'microbatch': 1, 'bf16': True}
    for key, expected in supported.items():
        if key in execution and execution[key] != expected:
            raise ValueError(f'unsupported v5 execution {key}')


def build_v5_contract(config, reference: V5ReferenceIdentity, *, ce_weight: float,
                      sim_weight: float, schedule: Mapping[str, object],
                      execution: Mapping[str, object]) -> dict:
    if not isinstance(config, Stage1V5Config):
        raise ValueError('v5 contract requires Stage1V5Config')
    reference = V5ReferenceIdentity(**asdict(reference))
    for weight in (ce_weight, sim_weight):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError('invalid CE/alignment coefficient')
    if not isinstance(schedule, Mapping):
        raise ValueError('invalid v5 sampler schedule')
    _execution(execution)
    return _json_copy({'schema_version': 1, 'variant': 'v5', 'formula_id': v5_formula_id(config),
                       'config': asdict(config), 'coefficients': {'ce_weight': ce_weight, 'sim_weight': sim_weight},
                       'semantics': _SEMANTICS, 'reference': asdict(reference),
                       'schedule': dict(schedule), 'execution': dict(execution)})


def _validate(saved: Mapping[str, object]) -> Stage1V5Config:
    required = {'schema_version', 'variant', 'formula_id', 'config', 'coefficients',
                'semantics', 'reference', 'schedule', 'execution'}
    if not isinstance(saved, Mapping) or set(saved) != required or type(saved['schema_version']) is not int or saved['schema_version'] != 1:
        raise ValueError('invalid or unsupported v5 contract schema')
    if saved['variant'] != 'v5' or saved['semantics'] != _SEMANTICS:
        raise ValueError('saved v5 variant or formula semantics mismatch')
    config = saved['config']
    if not isinstance(config, Mapping) or set(config) != {f.name for f in fields(Stage1V5Config)}:
        raise ValueError('incomplete saved v5 objective')
    try:
        result = Stage1V5Config(**config)
        reference = V5ReferenceIdentity(**saved['reference'])
        rebuilt = build_v5_contract(result, reference, **saved['coefficients'],
                                    schedule=saved['schedule'], execution=saved['execution'])
    except (TypeError, AttributeError) as error:
        raise ValueError('malformed saved v5 objective or identity') from error
    if _json_copy(saved) != rebuilt:
        raise ValueError('saved v5 formula/content contract mismatch')
    return result


def load_v5_contract(checkpoint: Path) -> dict:
    try:
        saved = json.loads((Path(checkpoint) / V5_CONTRACT_FILENAME).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('missing or unreadable Stage1-v5 sidecar; cannot relabel legacy resume') from error
    _validate(saved)
    return saved


def validate_v5_resume(saved: Mapping[str, object], explicit: Mapping[str, object]) -> Stage1V5Config:
    _validate(saved)
    objective = {}
    other = {'stage1_variant': saved['variant'], **saved['coefficients'], **saved['execution']}
    # CLI option spellings normalize once; loss values use the explicit alias map.
    normalized = {}
    for spelling, value in explicit.items():
        name = spelling.removeprefix('--').replace('-', '_')
        if name in normalized and normalized[name] != value:
            raise ValueError(f'conflicting explicit aliases for {name}')
        normalized[name] = value
    for name, value in normalized.items():
        if name in other:
            if value != other[name]:
                raise ValueError(f'explicit {name} conflicts with saved v5 contract')
        else:
            objective[name] = value
    return resolve_v5_config(objective, saved=saved['config'])
