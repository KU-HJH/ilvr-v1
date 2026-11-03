"""Small, strict objective identities; publication never reads weight files."""
from dataclasses import asdict, dataclass, fields, is_dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import types
from typing import get_args, get_origin, get_type_hints

from .config import ObjectiveConfig, validate_objective_config
from .execution import DropoutPolicy, ObjectiveRuntime, DROPOUT_POLICY_VERSION
from .reference import ReferenceIdentity

FILENAME = 'stage2_objective_metadata.json'


@dataclass(frozen=True)
class ObjectiveMetadata:
    schema_version: int
    objective: str
    config: ObjectiveConfig
    vocabulary_identity: str
    excluded_tokens: tuple[tuple[int, str, str], ...]
    reference: ReferenceIdentity | None
    dropout: DropoutPolicy
    intervention: str
    reduction: str
    backend: str
    rollout_backend: str
    latent_size: int
    supervise_latent_boundaries: bool


def make_objective_metadata(runtime: ObjectiveRuntime, *, backend: str,
                            latent_size: int, supervise_latent_boundaries: bool) -> ObjectiveMetadata:
    intervention = {
        'latent_body': 'local_latent_kv_v1',
        'full_block_attention_mask': 'local_full_latent_block_kv_v1',
    }[runtime.visual_ablation]
    result = ObjectiveMetadata(1, 'entropy_transition_vision_gain_v1', runtime.config,
        runtime.vocabulary.identity, runtime.vocabulary.excluded,
        runtime.reference.identity if runtime.reference else None, runtime.dropout_policy,
        intervention, 'global_blocks_per_optimizer_update_v1', backend,
        'no_cache', latent_size, supervise_latent_boundaries)
    _validate(result)
    return result


def _decode(value, annotation, name):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is types.UnionType:
        if value is None and type(None) in args:
            return None
        return _decode(value, next(t for t in args if t is not type(None)), name)
    if is_dataclass(annotation):
        expected = {f.name for f in fields(annotation)}
        if type(value) is not dict or set(value) != expected:
            raise ValueError(f'{name}: exact fields required: {sorted(expected)}')
        hints = get_type_hints(annotation)
        return annotation(**{key: _decode(item, hints[key], f'{name}.{key}') for key, item in value.items()})
    if origin is tuple:
        if type(value) is not list:
            raise ValueError(f'{name}: expected array')
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(item, args[0], name) for item in value)
        if len(value) != len(args):
            raise ValueError(f'{name}: incorrect tuple length')
        return tuple(_decode(item, kind, name) for item, kind in zip(value, args))
    if annotation is float:
        if type(value) not in (int, float):
            raise ValueError(f'{name}: expected number')
        return float(value)
    if type(value) is not annotation:
        raise ValueError(f'{name}: expected {annotation.__name__}')
    return value


def _validate(metadata):
    validate_objective_config(metadata.config)
    if not metadata.config.enabled:
        raise ValueError('metadata requires enabled objective')
    for field, expected in [('schema_version', 1), ('objective', 'entropy_transition_vision_gain_v1'),
                            ('reduction', 'global_blocks_per_optimizer_update_v1'),
                            ('rollout_backend', 'no_cache')]:
        if getattr(metadata, field) != expected:
            raise ValueError(f'unsupported {field}: {getattr(metadata, field)!r}')
    if metadata.intervention not in {'local_latent_kv_v1', 'local_full_latent_block_kv_v1'}:
        raise ValueError(f'unsupported intervention: {metadata.intervention!r}')
    if metadata.intervention == 'local_full_latent_block_kv_v1' and metadata.config.lambda_vis <= 0:
        raise ValueError('full-block intervention requires positive lambda_vis')
    if type(metadata.supervise_latent_boundaries) is not bool or type(metadata.latent_size) is not int:
        raise ValueError('latent_size must be integer and supervise_latent_boundaries boolean')
    if metadata.backend not in {'eager', 'sdpa', 'sdpa_vision_flash_attention_2'} or metadata.latent_size < 1:
        raise ValueError('invalid backend or latent_size')
    if metadata.dropout.version != DROPOUT_POLICY_VERSION or metadata.dropout.effective_value != 0:
        raise ValueError('invalid dropout runtime policy')
    if any(not math.isfinite(v) or not 0 <= v < 1 for _, v in metadata.dropout.original_values):
        raise ValueError('invalid original dropout')
    if (metadata.reference is not None) != (metadata.config.lambda_dyn > 0):
        raise ValueError('reference identity must match positive lambda_dyn')
    if metadata.reference and metadata.reference.vocabulary_identity != metadata.vocabulary_identity:
        raise ValueError('reference vocabulary identity mismatch')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate metadata field: {key}')
        result[key] = value
    return result


def save_objective_metadata(directory: Path, metadata: ObjectiveMetadata) -> None:
    payload = json.dumps(asdict(metadata), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    checked = _decode(json.loads(payload), ObjectiveMetadata, 'metadata')
    _validate(checked)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / FILENAME
    if target.is_symlink():
        raise ValueError('symlink objective metadata publication target')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                         prefix='.' + FILENAME + '.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if target.is_symlink():
            raise ValueError('symlink objective metadata publication target')
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_objective_metadata(directory: Path) -> ObjectiveMetadata:
    target = Path(directory) / FILENAME
    if target.is_symlink() or not target.is_file():
        raise ValueError('missing or symlink stage2_objective_metadata.json')
    result = _decode(json.loads(target.read_text(encoding='utf-8'), object_pairs_hook=_pairs),
                     ObjectiveMetadata, 'metadata')
    _validate(result)
    return result


def validate_objective_resume(directory: Path, expected: ObjectiveMetadata | None) -> None:
    target = Path(directory) / FILENAME
    if expected is None:
        if target.exists() or target.is_symlink():
            raise ValueError('unexpected objective metadata while objective is disabled')
        return
    observed = load_objective_metadata(directory)
    def compare(left, right, prefix=''):
        for field in fields(left):
            name = prefix + field.name
            a, b = getattr(left, field.name), getattr(right, field.name)
            if is_dataclass(a) and is_dataclass(b):
                compare(a, b, name + '.')
            elif a != b:
                raise ValueError(f'objective resume {name}: expected {a!r}, observed {b!r}')
    compare(expected, observed)
