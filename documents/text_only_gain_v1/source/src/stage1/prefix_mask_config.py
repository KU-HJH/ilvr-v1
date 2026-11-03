"""Strict settings and persisted semantics for query-local prefix conditioning."""
from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class PrefixConditioningConfig:
    enabled: bool = False
    temperature: float = 2.0
    epsilon: float = 1e-6
    sigmoid_scale: float = 1.0
    text_rho_min: float = 0.30
    text_rho_max: float = 0.50
    text_max_mask_ratio: float = 0.30
    text_low_attention_threshold: float = 0.10
    text_fallback_mask_ratio: float = 0.10
    groups_per_initial_image: int = 16
    helper_proxy_mass_floor: float = 1e-4
    image_score_epsilon: float = 1e-8
    image_rho_min: float = 0.10
    image_rho_max: float = 0.30
    image_max_mask_ratio: float = 0.25
    logit_chunk_rows: int = 32
    trace_chunk_rows: int = 32
    max_packed_tokens: int | None = None
    max_mask_bytes: int | None = None
    mask_all_original_image_keys: bool = False

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name in {'enabled', 'mask_all_original_image_keys'}:
                if type(value) is not bool:
                    raise ValueError(f'{name} must be boolean')
            elif name in {'groups_per_initial_image', 'logit_chunk_rows', 'trace_chunk_rows',
                          'max_packed_tokens', 'max_mask_bytes'}:
                if value is None and name in {'max_packed_tokens', 'max_mask_bytes'}:
                    continue
                if type(value) is not int or value <= 0:
                    raise ValueError(f'{name} must be a positive integer')
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f'{name} must be finite numeric')
                if name in {'temperature', 'epsilon', 'sigmoid_scale', 'image_score_epsilon'}:
                    if value <= 0:
                        raise ValueError(f'{name} must be positive')
                elif not 0 <= value <= 1:
                    raise ValueError(f'{name} must be in [0, 1]')
        for source in ('text', 'image'):
            if getattr(self, f'{source}_rho_min') > getattr(self, f'{source}_rho_max'):
                raise ValueError(f'{source} rho_min exceeds rho_max')

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> 'PrefixConditioningConfig':
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f'unknown prefix conditioning fields: {sorted(unknown)}')
        return cls(**data)

    def to_metadata(self) -> dict[str, object]:
        settings = asdict(self)
        # Preserve existing checkpoint contracts byte-for-byte when disabled.
        if not self.mask_all_original_image_keys:
            settings.pop('mask_all_original_image_keys')
        return {
            **settings, 'schema_version': 3,
            'execution_profile': 'packed_four_decoder_with_reference',
            'objective': 'conditional_latent_gain',
            'comparison': 'same_prefix_mask_latent_on_vs_off',
            'activation': 'query_local_scored_gt',
            'source_intervention': 'attention_edge',
            'preserve_input_values_and_positions': True,
            'gt_prediction_query': 'target_position_minus_one',
            'text_candidates': 'reasoning_before_current_latent_start',
            'text_fallback_rounding': 'floor_no_minimum_one',
            'text_fallback_order': 'descending_attention_then_original_position',
            'text_low_attention_comparison': 'strict_less_than',
            'cap_rounding': 'floor', 'selection_tie_break': 'original_position',
            'memory_limit_none': 'no_configured_precap_not_memory_assurance',
            'trace': {
                'source': 'existing_gain_clean_forward',
                'rows': 'scored_gt_and_actual_latent_prediction_queries',
                'attention_reduction': 'mean_heads_per_layer_then_layers',
                'detach_selector_statistics': True, 'detach_shared_prefix_kv': False,
                'extra_trace_replay': 'forbidden', 'observer_unsupported': 'error',
                'record_checkpoint_recompute_statistics': False,
            },
            'reference': {
                'provider': 'ema_clean_interleaved', 'execution': 'once_per_valid_microbatch',
                'eval_mode': True, 'no_grad': True, 'persistent_kv_cache': False,
                'rebuild_helper_targets': False, 'same_teacher_latent_snapshot': True,
                'centering': 'per_sample_all_gain_valid_gt_queries',
                'fallback_to_nll_or_fixed_budget': False,
            },
            'image': {
                'trace_method': 'helper_prediction_attention_v1',
                'latent_prediction_query': 'selected_target_position_minus_one',
                'helper_row_aggregation': 'mean_over_actual_selected_positions',
                'grouping': 'balanced_token_order_without_cross_image_groups',
                'online_group_ablation': False, 'online_dependency_anchor': False,
                'online_attribution_backward': False, 'allow_empty_selection': True,
            },
            'replay': {
                'pack_both_conditions': True, 'pair_mask_selection': 'frozen_common_plan',
                'reuse_only_pre_intervention_clean_kv': True, 'detach_shared_prefix_kv': False,
                'preserve_original_position_ids': True, 'preserve_existing_dropout_replay': True,
                'chunked_decoder_fallback': 'forbidden', 'full_replay_oracle_in_training': 'forbidden',
            },
            'forward_budget': {
                'student_base': 1, 'student_gain_clean': 1, 'reference_clean': 1,
                'student_packed_pair': 1, 'selector_extra_decoder': 0,
                'iier_additional_decoder_calls': 1,
                'accounting': 'initial_decoder_per_rank_microbatch',
                'common_helper_target_generation': 'separately_counted',
                'checkpoint_recompute': 'separately_counted', 'on_violation': 'error',
            },
            'objective_integration': {
                'add_reader_ce_to_base_loss': False, 'add_reader_kd_to_base_loss': False,
                'reuse_existing_gain_projection': True, 'update_order': 'optimizer_then_gain_then_ema',
            },
        }


def read_small_json(path, *, max_bytes=16384):
    """Read a bounded regular file without following its final symlink."""
    import json
    import os
    import stat
    from pathlib import Path
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate metadata field: {key}')
            result[key] = value
        return result
    try:
        fd = os.open(Path(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'r', encoding='utf-8') as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
                raise ValueError('metadata must be a bounded regular file')
            raw = handle.read(max_bytes + 1)
            if len(raw.encode('utf-8')) > max_bytes:
                raise ValueError('metadata exceeds size limit')
            value = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(value, dict):
            raise ValueError('metadata must be a JSON object')
        return value
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f'invalid or missing JSON metadata: {path}') from error


def load_prefix_conditioning_config(path: str | Path | None) -> PrefixConditioningConfig:
    if path is None:
        return PrefixConditioningConfig()
    data = read_small_json(path)
    names = {field.name for field in fields(PrefixConditioningConfig)}
    if set(data) not in (names, names - {'mask_all_original_image_keys'}):
        raise ValueError('prefix conditioning JSON must contain exactly all config fields')
    return PrefixConditioningConfig.from_mapping(data)


def validate_prefix_runtime(config, gain, attention):
    if not isinstance(config, PrefixConditioningConfig):
        raise TypeError('prefix conditioning requires PrefixConditioningConfig')
    if not config.enabled:
        return
    if gain.execution != 'packed':
        raise ValueError('prefix conditioning requires explicitly selected packed execution')
    if attention not in {'eager', 'sdpa'}:
        raise ValueError('prefix conditioning requires eager or sdpa attention')



def validate_prefix_model(model, *, sequence_length=None):
    """Reuse packed structural checks before any helper or student execution."""
    from transformers.models.qwen2_5_vl.stage1_vision_gain_packed import validate_packed_gain_model
    modules = tuple(model.modules())
    decoders = [module for module in modules if type(module).__name__ == 'Qwen2_5_VLModel']
    if len(decoders) != 1:
        raise ValueError('prefix conditioning requires exactly one Qwen language decoder')
    decoder = decoders[0]
    heads = [module.lm_head for module in modules
             if getattr(module, 'model', None) is decoder and hasattr(module, 'lm_head')]
    if len(heads) != 1:
        raise ValueError('prefix conditioning requires exactly one decoder output head')
    validate_packed_gain_model(decoder, heads[0])
    window = getattr(decoder.config, 'sliding_window', None)
    if sequence_length is not None and window is not None and sequence_length > window:
        raise ValueError('conditioned gain cannot share an effective sliding-window prefix')
