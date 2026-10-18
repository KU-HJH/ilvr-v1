"""Explicit evaluation identities; historical unlabelled files remain continuous."""
from __future__ import annotations


def resolve_protocol(value: str, stage: str) -> str:
    if value not in ('auto','original','continuous'):
        raise ValueError(f'unknown evaluation protocol: {value!r}')
    if stage not in ('stage1','stage2'):
        raise ValueError(f'unsupported evaluation stage: {stage!r}')
    resolved = ('original' if stage == 'stage1' else 'continuous') if value == 'auto' else value
    if resolved == 'original' and stage != 'stage1':
        raise ValueError('original evaluation protocol supports Stage 1 only')
    return resolved


def resolve_options(protocol, repetition_penalty, generation_limit_policy):
    if protocol == 'original':
        if generation_limit_policy not in (None,'native'):
            raise ValueError('original evaluation requires native per-token limits')
        return repetition_penalty, 'native'
    if protocol != 'continuous':
        raise ValueError('evaluation protocol must be resolved before options')
    if generation_limit_policy == 'native':
        raise ValueError('native generation limits require original evaluation')
    return (1.05 if repetition_penalty is None else repetition_penalty,
            generation_limit_policy or 'bounded')


def validate_execution_identity(value):
    if not isinstance(value,dict) or not {'evaluation_protocol','source_commit','generation_config','attention_backend'} <= set(value):
        raise ValueError('invalid evaluation execution identity')
    if value['evaluation_protocol'] not in ('original','continuous'):
        raise ValueError('invalid evaluation execution identity protocol')
    expected_source = '13a67ed2ee975d6ca56b4e9eba18d68b352b166b' if value['evaluation_protocol']=='original' else None
    if value['source_commit'] != expected_source:
        raise ValueError('invalid evaluation execution identity source')
    if not isinstance(value['generation_config'],dict):
        raise ValueError('invalid evaluation execution identity config')
    backend=value['attention_backend']
    if not isinstance(backend,dict) or set(backend)!={'language','vision'} or any(
            not isinstance(backend[key],list) or not backend[key] or
            any(not isinstance(name,str) or not name for name in backend[key]) for key in backend):
        raise ValueError('invalid evaluation execution identity backend')
    return value
