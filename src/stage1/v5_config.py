"""Opt-in Stage1-v5 objective contracts, independent of Stage2 phase defaults."""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping


LOSS_ALIASES = {'dil-bce-orig': 'dil_bce_orig', 'dil_bce_orig': 'dil_bce_orig'}
V5_OPTION_ALIASES = {
    'stage1_loss_type': 'loss_type',
    'stage1_v5_selection': 'selection',
    'stage1_v5_text_kl_weight': 'text_kl_weight',
    'step_preference_weight': 'lambda_preference',
    'step_preference_beta': 'beta', 'step_dpo_beta': 'beta',
    'step_preference_margin': 'margin',
    'step_preference_topk': 'preference_topk_k',
    'step_preference_lambda_topk_mass': 'lambda_topk_mass',
    'step_preference_entropy_top_percent': 'preference_entropy_top_percent',
    'step_entropy_weight': 'lambda_entropy',
}


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


@dataclass(frozen=True)
class Stage1V5Config:
    loss_type: str = 'dil_bce_orig'
    beta: float = 1.0
    margin: float = 0.0
    lambda_preference: float = 1.0
    lambda_topk_entropy: float = 0.0
    preference_topk_k: int | None = None
    preference_topk_mode: str = 'independent'
    topk_include_gt: bool = False
    lambda_topk_mass: float = 0.0
    lambda_topk_chosen_mass: float = 0.0
    lambda_entropy: float = 0.0
    selection: str = 'all'
    preference_entropy_top_percent: float | None = None
    loss_chunk_positions: int = 128
    text_kl_weight: float = 0.01
    temperature: float = 2.0
    rho_min: float = 0.3
    rho_max: float = 0.5
    text_cap: float = 0.3
    low_attention_mass: float = 0.1
    low_attention_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.loss_type not in LOSS_ALIASES:
            raise ValueError('Stage1-v5 supports only dil-bce-orig/dil_bce_orig')
        object.__setattr__(self, 'loss_type', LOSS_ALIASES[self.loss_type])
        for name, expected in (('beta', 1.), ('margin', 0.), ('lambda_entropy', 0.),
                               ('lambda_topk_chosen_mass', 0.)):
            if _finite(name, getattr(self, name)) != expected:
                raise ValueError(f'Stage1-v5 requires {name}={expected}')
        if _finite('lambda_preference', self.lambda_preference) <= 0:
            raise ValueError('lambda_preference must be positive')
        for name in ('lambda_topk_entropy', 'lambda_topk_mass', 'text_kl_weight'):
            if _finite(name, getattr(self, name)) < 0:
                raise ValueError(f'{name} must be nonnegative')
        if self.preference_topk_k is not None and (
                type(self.preference_topk_k) is not int or self.preference_topk_k <= 0):
            raise ValueError('preference_topk_k must be a positive integer')
        if self.lambda_topk_entropy > 0 and self.preference_topk_k is None:
            raise ValueError('positive entropy requires an explicit valid preference_topk_k')
        if self.lambda_topk_mass > 0 and self.lambda_topk_entropy <= 0:
            raise ValueError('positive mass requires active entropy')
        if self.preference_topk_mode not in ('independent', 'sync_topk'):
            raise ValueError('unsupported preference_topk_mode')
        if type(self.topk_include_gt) is not bool:
            raise ValueError('topk_include_gt must be boolean')
        if self.selection not in ('all', 'clean', 'voff', 'gap', 'intersection'):
            raise ValueError('unsupported Stage1-v5 selection')
        percent = self.preference_entropy_top_percent
        if percent is not None and not 0 < _finite('preference_entropy_top_percent', percent) <= 100:
            raise ValueError('preference_entropy_top_percent must be in (0, 100]')
        if (self.selection == 'all') != (percent is None):
            raise ValueError('selection=all requires no percentage; non-all selection requires a percentage')
        if type(self.loss_chunk_positions) is not int or self.loss_chunk_positions <= 0:
            raise ValueError('loss_chunk_positions must be a positive integer')
        if _finite('temperature', self.temperature) <= 0:
            raise ValueError('temperature must be positive')
        for name in ('rho_min', 'rho_max', 'text_cap', 'low_attention_mass', 'low_attention_fraction'):
            if not 0 <= _finite(name, getattr(self, name)) <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        if self.rho_min > self.rho_max:
            raise ValueError('rho_min must not exceed rho_max')


def _canonical_values(values: Mapping[str, object]) -> dict[str, object]:
    allowed = {field.name for field in fields(Stage1V5Config)}
    resolved = {}
    for spelling, value in values.items():
        name = V5_OPTION_ALIASES.get(spelling, spelling)
        if name not in allowed:
            raise ValueError(f'unknown Stage1-v5 option: {spelling}')
        if name == 'loss_type':
            if not isinstance(value, str) or value not in LOSS_ALIASES:
                raise ValueError('Stage1-v5 supports only dil-bce-orig/dil_bce_orig')
            value = LOSS_ALIASES[value]
        if name in resolved and resolved[name] != value:
            raise ValueError(f'conflicting explicit aliases for {name}')
        resolved[name] = value
    return resolved


def resolve_v5_config(values: Mapping[str, object], *, saved: Mapping[str, object] | None = None) -> Stage1V5Config:
    """Resolve explicit inputs; omitted resume inputs inherit the saved objective."""
    explicit = _canonical_values(values)
    if saved is None:
        if 'lambda_topk_mass' not in explicit:
            explicit['lambda_topk_mass'] = 0.5 * _finite(
                'lambda_topk_entropy', explicit.get('lambda_topk_entropy', 0.0))
        if explicit.get('preference_entropy_top_percent') is not None and 'selection' not in explicit:
            explicit['selection'] = 'clean'
        config = Stage1V5Config(**explicit)
        validate_v5_training_config(config)
        return config
    inherited = _canonical_values(saved)
    saved_config = Stage1V5Config(**inherited)
    for name, value in explicit.items():
        if value != getattr(saved_config, name):
            raise ValueError(f'explicit {name} conflicts with saved Stage1-v5 resume objective')
    return saved_config


def validate_v5_training_config(config: Stage1V5Config) -> None:
    """Support metadata-countable masks while keeping historical selectors readable."""
    if not isinstance(config, Stage1V5Config):
        raise ValueError('v5 training requires Stage1V5Config')
    if config.selection not in ('all', 'clean'):
        raise ValueError('Stage1-v5 token-mean training requires selection=all or clean')


def v5_formula_id(config: Stage1V5Config) -> str:
    if config.lambda_topk_entropy == 0:
        return 'dil_bce_orig_token_ref_full_vocab_v1'
    if config.topk_include_gt:
        return 'dil_bce_orig_token_ref_full_vocab_topk_gt_union_entropy_v1'
    return 'dil_bce_orig_token_ref_full_vocab_topk_entropy_v1'
