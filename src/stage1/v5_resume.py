"""Stage1-v5 objective and content identity sidecar, with strict resume inheritance."""
from __future__ import annotations

from dataclasses import asdict, fields
import json
import math
from pathlib import Path
from typing import Mapping

from stage1.v5_config import (Stage1V5Config, resolve_v5_config, v5_formula_id,
                              validate_v5_training_config)
from stage1.v5_reference import V5ReferenceIdentity
from stage1.query import Stage1QueryConfig, query_execution_config, resolve_query_config

V5_CONTRACT_FILENAME = 'stage1_v5.json'
_SEMANTICS_V1 = {
    'reference': 'immutable_initial_effective_teacher_v1',
    'auxiliary_targets': 'immediate_original_text_exclude_crossing_special_padding_v1',
    'interventions': 'independent_clean_prefix_voff_body_textoff_previous_text_v1',
    'reduction': 'global_actual_window_original_blocks_empty_included_v1',
    'ce_alignment_text_reduction': 'configured_stage1_accumulation_v1',
    'text_target': 'detached_clean_full_vocab_temperature_squared_kl_v1',
    'text_schedule': 'from_first_update_no_kl_warmup_v1',
}
_SEMANTICS_V2 = {
    **_SEMANTICS_V1,
    'reduction': 'global_actual_window_dil_tokens_aux_original_blocks_v2',
    'dil_sum_units': 'eligible_token_loss_sum',
    'auxiliary_sum_units': 'original_block_token_mean_sum',
}
_SEMANTICS_V3 = {
    **_SEMANTICS_V2,
    'reduction': 'global_actual_window_selected_dil_tokens_aux_original_blocks_v3',
    'dil_sum_units': 'selected_token_loss_sum',
}
_SCHEMA_SEMANTICS = {1: _SEMANTICS_V1, 2: _SEMANTICS_V2, 3: _SEMANTICS_V3}


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError('v5 contract must contain finite JSON data') from error


def _execution(execution: Mapping[str, object]) -> None:
    if not isinstance(execution, Mapping):
        raise ValueError('invalid v5 execution profile')
    if 'query_config' in execution:
        if not isinstance(execution['query_config'], Mapping):
            raise ValueError('saved query config must be an execution mapping')
        resolve_query_config({}, saved=execution['query_config'])
    supported = {'tuning_method': 'dora', 'method': 'dora', 'ema_tau': 1.,
                 'zero_stage': 1, 'per_device_train_batch_size': 1,
                 'microbatch': 1, 'bf16': True}
    for key, expected in supported.items():
        if key in execution and execution[key] != expected:
            raise ValueError(f'unsupported v5 execution {key}')


def build_v5_contract(config, reference: V5ReferenceIdentity, *, ce_weight: float,
                      sim_weight: float, schedule: Mapping[str, object],
                      execution: Mapping[str, object]) -> dict:
    validate_v5_training_config(config)
    return _build_contract(config, reference, ce_weight=ce_weight, sim_weight=sim_weight,
                           schedule=schedule, execution=execution,
                           schema_version=3 if config.selection == 'clean' else 2)


def _build_contract(config, reference, *, ce_weight, sim_weight, schedule, execution,
                    schema_version):
    """Canonical payload reconstruction only; no historical execution engine."""
    if not isinstance(config, Stage1V5Config):
        raise ValueError('v5 contract requires Stage1V5Config')
    reference = V5ReferenceIdentity(**asdict(reference))
    for weight in (ce_weight, sim_weight):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError('invalid CE/alignment coefficient')
    if not isinstance(schedule, Mapping):
        raise ValueError('invalid v5 sampler schedule')
    _execution(execution)
    if schema_version == 2 and config.selection != 'all':
        raise ValueError('schema-2 v5 token mean requires selection=all; clean requires a fresh run')
    if schema_version == 3 and (config.selection != 'clean' or
            schedule.get('dil_denominator') != 'actual_window_selected_dil_tokens'):
        raise ValueError('schema-3 v5 requires clean selection and selected-token schedule denominator')
    return _json_copy({'schema_version': schema_version, 'variant': 'v5', 'formula_id': v5_formula_id(config),
                       'config': asdict(config), 'coefficients': {'ce_weight': ce_weight, 'sim_weight': sim_weight},
                       'semantics': _SCHEMA_SEMANTICS[schema_version], 'reference': asdict(reference),
                       'schedule': dict(schedule), 'execution': dict(execution)})


def _validate(saved: Mapping[str, object]) -> Stage1V5Config:
    required = {'schema_version', 'variant', 'formula_id', 'config', 'coefficients',
                'semantics', 'reference', 'schedule', 'execution'}
    if (not isinstance(saved, Mapping) or set(saved) != required
            or type(saved['schema_version']) is not int
            or saved['schema_version'] not in _SCHEMA_SEMANTICS):
        raise ValueError('invalid or unsupported v5 contract schema')
    version = saved['schema_version']
    if saved['variant'] != 'v5' or saved['semantics'] != _SCHEMA_SEMANTICS[version]:
        raise ValueError('saved v5 variant or formula semantics mismatch')
    config = saved['config']
    if not isinstance(config, Mapping) or set(config) != {f.name for f in fields(Stage1V5Config)}:
        raise ValueError('incomplete saved v5 objective')
    try:
        result = Stage1V5Config(**config)
        if version in (2, 3):
            validate_v5_training_config(result)
        reference = V5ReferenceIdentity(**saved['reference'])
        rebuilt = _build_contract(result, reference, **saved['coefficients'],
            schedule=saved['schedule'], execution=saved['execution'], schema_version=version)
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
    if saved['schema_version'] == 1:
        raise ValueError('historical Stage1-v5 block-mean training requires its original reviewed code; '
                         'token-mean training requires a fresh run')
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


def resolve_v5_query_resume(saved_contract: Mapping[str, object],
                            explicit_query: Mapping[str, object]) -> Stage1QueryConfig:
    """Authenticate the v5 sidecar, then inherit its independent query semantics."""
    _validate(saved_contract)
    execution = saved_contract['execution']
    if 'query_config' not in execution:
        # Only historical contracts predate query metadata. Current source
        # bindings cannot silently lose their query execution contract.
        if {'query', 'targets', 'history_attention', 'history_teacher'} & set(
                execution.get('source_sha256', {})):
            raise ValueError('current v5 contract is missing query_config')
        saved = query_execution_config(Stage1QueryConfig(), ())
    else:
        saved = execution['query_config']
    return resolve_query_config(explicit_query, saved=saved)
