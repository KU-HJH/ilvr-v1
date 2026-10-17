"""Immutable initial DoRA reference, composed from original base and sparse state."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile

from safetensors import SafetensorError, safe_open
import torch
from torch import nn

from artifact_paths import require_canonical_artifact_path
from peft_common import resolve_qwen2_5_vl_model
from stage1.ema import Stage1EmaTeacher, _named_parameters, _selected_parameters, _unwrap_model
from stage1.peft import validate_stage1_trainable_boundary
from stage1.types import Stage1TuningConfig

V5_REFERENCE_FILENAME = 'stage1_v5_reference.pt'


@dataclass(frozen=True)
class V5ReferenceIdentity:
    schema_version: int
    base_content_sha256: str
    effective_state_sha256: str
    processor_sha256: str
    selected_parameter_names: tuple[str, ...]
    formula_domain: str = 'teacher_forced_token'

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported v5 reference schema')
        if self.formula_domain != 'teacher_forced_token':
            raise ValueError('unsupported reference formula domain')
        for value in (self.base_content_sha256, self.effective_state_sha256, self.processor_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid reference content SHA256')
        if (not isinstance(self.selected_parameter_names, (tuple, list)) or
                not self.selected_parameter_names or
                any(not isinstance(n, str) or not n for n in self.selected_parameter_names) or
                len(set(self.selected_parameter_names)) != len(self.selected_parameter_names)):
            raise ValueError('invalid selected reference parameter names')
        object.__setattr__(self, 'selected_parameter_names', tuple(self.selected_parameter_names))


def _hash_files(directory: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        name = path.relative_to(directory).as_posix().encode()
        digest.update(len(name).to_bytes(8, 'big'))
        digest.update(name)
        digest.update(path.stat().st_size.to_bytes(8, 'big'))
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def _base_identity(base_path: Path) -> tuple[str, dict[str, tuple[Path, str]]]:
    # Original Qwen safetensors (standalone or sharded), never a trained adapter.
    from stage1.checkpointing import Stage1ArtifactKind, classify_stage1_artifact
    if classify_stage1_artifact(base_path) != Stage1ArtifactKind.FULL:
        raise ValueError('v5 reference requires the original full base artifact')
    files = sorted(base_path.glob('*.safetensors'))
    if not files or not (base_path / 'config.json').is_file():
        raise ValueError('v5 original base requires safetensors and config.json')
    keys = {}
    for path in files:
        try:
            with safe_open(path, framework='pt', device='cpu') as stream:
                for name in stream.keys():
                    if name in keys:
                        raise ValueError('duplicate original-base parameter')
                    keys[name] = (path, name)
        except (OSError, SafetensorError) as error:
            raise ValueError('invalid original-base content') from error
    identity_files = files + [base_path / 'config.json']
    index = base_path / 'model.safetensors.index.json'
    if index.exists():
        identity_files.append(index)
    config = json.loads((base_path / 'config.json').read_text())
    if (config.get('tie_word_embeddings') is True and 'lm_head.weight' not in keys
            and 'model.embed_tokens.weight' in keys):
        # HF omits the tied output alias from physical safetensors. Stage1's
        # existing sparse-output loader intentionally unties this same value.
        keys['lm_head.weight'] = keys['model.embed_tokens.weight']
    return _hash_files(base_path, identity_files), keys


def _processor_identity(processor) -> str:
    # Serialized tokenizer vocabulary/special mapping, image processing and templates.
    with tempfile.TemporaryDirectory(prefix='stage1-v5-processor-') as temporary:
        directory = Path(temporary)
        processor.save_pretrained(directory)
        files = [p for p in directory.rglob('*') if p.is_file()]
        if not files:
            raise ValueError('processor serialization is empty')
        return _hash_files(directory, files)


def _validate_processor_layout(teacher: Stage1EmaTeacher, processor) -> None:
    from stage1.checkpointing import validate_stage1_vocab_alignment
    qwen = resolve_qwen2_5_vl_model(teacher.model)
    tokenizer = processor.tokenizer
    validate_stage1_vocab_alignment(teacher.model, len(tokenizer))
    ids = tuple(qwen.model.embed_tokens.token_adapter.token_indices['default'])
    tokens = ('<|latent_pad|>', '<|latent_start|>', '<|latent_end|>')
    config_ids = tuple(getattr(qwen.config, name, None) for name in
                       ('latent_token_id', 'latent_start_id', 'latent_end_id'))
    if ids != tuple(range(len(tokenizer)-3,len(tokenizer))) or config_ids != ids or tuple(tokenizer.convert_tokens_to_ids(t) for t in tokens) != ids or tuple(
            tokenizer.convert_ids_to_tokens(i) for i in ids) != tokens:
        raise ValueError('processor/reference latent token layout mismatch')


def _validate_adapter_runtime(model: nn.Module, config) -> None:
    from peft.tuners.lora.layer import Linear as LoraLinear
    from peft.tuners.lora.variants import DoraLinearVariant
    from peft.tuners.tuners_utils import BaseTunerLayer

    # The supported Stage1 composer uses ordinary alpha/r scaling and one
    # default DoRA adapter. Runtime dispatch state is not an immutable cache.
    if config.use_rslora or config.rank_pattern or config.alpha_pattern:
        raise ValueError('unsupported runtime adapter scaling configuration')
    if model.active_adapters != ['default']:
        raise ValueError('unsupported runtime adapter activation')
    expected_scale = config.lora_alpha / config.r
    for module in model.modules():
        # Transformers also defines an unrelated active_adapters() method;
        # inspect the PEFT dispatch owners, not that native model method.
        if isinstance(module, BaseTunerLayer) and module.active_adapters != ['default']:
            raise ValueError('unsupported runtime adapter activation')
        if getattr(module, 'disable_adapters', False) is True or getattr(module, 'merged', False) is True:
            raise ValueError('disabled or merged runtime adapters are unsupported')
        if not hasattr(module, 'lora_A'):
            continue
        if (not isinstance(module, LoraLinear) or
                module.r != {'default': config.r} or
                module.lora_alpha != {'default': config.lora_alpha} or
                module.scaling != {'default': expected_scale} or
                module.use_dora != {'default': True} or
                set(module.lora_variant) != {'default'} or
                type(module.lora_variant['default']) is not DoraLinearVariant):
            raise ValueError('runtime adapter state differs from supported DoRA composition')


def _topology(teacher: Stage1EmaTeacher, student: nn.Module) -> None:
    student = _unwrap_model(student)
    if teacher.decay != 1. or teacher.method != 'dora':
        raise ValueError('Stage1-v5 requires frozen EMA1 DoRA')
    if any(m.training for m in teacher.model.modules()) or any(
            p.requires_grad or p.grad is not None for p in teacher.model.parameters()):
        raise ValueError('v5 reference must be frozen and entirely in eval mode')
    selected = _selected_parameters(student, 'dora')
    if tuple(selected) != teacher.parameter_names:
        raise ValueError('reference mutable-effective-state coverage mismatch')
    qwen = resolve_qwen2_5_vl_model(student)
    peft_config = getattr(student, 'peft_config', {})
    if set(peft_config) != {'default'} or not peft_config['default'].use_dora:
        raise ValueError('v5 reference requires single default DoRA adapter')
    config = peft_config['default']
    teacher_config = getattr(teacher.model, 'peft_config', {})
    if set(teacher_config) != {'default'} or teacher_config['default'].to_dict() != config.to_dict():
        raise ValueError('reference/student adapter configuration mismatch')
    targets = tuple(sorted({n.rsplit('.', 1)[-1] for n in config.target_modules}))
    tuning = Stage1TuningConfig(method='dora', rank=config.r, alpha=config.lora_alpha,
                              dropout=config.lora_dropout, target_suffixes=targets)
    ids = tuple(qwen.model.embed_tokens.token_adapter.token_indices['default'])
    validate_stage1_trainable_boundary(student, tuning, ids)
    # Prove teacher topology as well without altering its frozen flags.
    teacher_named = _named_parameters(teacher.model)
    student_named = _named_parameters(student)
    if set(teacher_named) != set(student_named):
        raise ValueError('reference/student topology mismatch')
    student_storages = {(p.device, p.untyped_storage().data_ptr()) for p in student_named.values()}
    for name, parameter in teacher_named.items():
        if parameter.device.type == 'meta' or getattr(parameter, 'ds_id', None) is not None:
            raise ValueError('v5 reference must be replicated, materialized state')
        if (parameter.device, parameter.untyped_storage().data_ptr()) in student_storages:
            raise ValueError('student/reference storage alias')
        if parameter.shape != student_named[name].shape:
            raise ValueError('reference/student shape mismatch')
    _validate_adapter_runtime(student, config)
    _validate_adapter_runtime(teacher.model, config)
    teacher_qwen = resolve_qwen2_5_vl_model(teacher.model)
    if tuple(teacher_qwen.model.embed_tokens.token_adapter.token_indices['default']) != ids or tuple(
            teacher_qwen.lm_head.token_ids.cpu().tolist()) != ids:
        raise ValueError('reference sparse token IDs mismatch')


def _base_name(name: str) -> str:
    if not name.startswith('base_model.model.'):
        raise ValueError('unsupported reference wrapper topology')
    return name[len('base_model.model.'):].replace('.token_adapter.base_layer.', '.').replace(
        '.base_layer.', '.').replace('.base_head.', '.')


def _validate_base_coverage(teacher: Stage1EmaTeacher, keys: dict[str, tuple[Path, str]]) -> None:
    selected = set(teacher.parameter_names)
    covered = set()
    for name, parameter in _named_parameters(teacher.model).items():
        if name in selected:
            continue
        base_name = _base_name(name)
        if base_name not in keys:
            raise ValueError(f'uncovered initialization-dependent parameter: {name}')
        source_path, source_name = keys[base_name]
        with safe_open(source_path, framework='pt', device='cpu') as stream:
            original = stream.get_tensor(source_name).to(dtype=parameter.dtype)
        current = parameter.detach().cpu()
        if base_name in ('model.embed_tokens.weight', 'lm_head.weight'):
            # Stage1 resizes to tokenizer+3, which may shrink source padding.
            # Layout authentication above requires the final three latent rows.
            n = current.shape[0]-3
            if (n<=0 or original.shape[0]<n or current.shape[1:]!=original.shape[1:]
                    or not torch.equal(current[:n], original[:n])):
                raise ValueError(f'reference does not match original base: {name}')
            if base_name == 'lm_head.weight':
                expected = original[:n].float().mean(0, keepdim=True).to(current.dtype).expand(3, -1)
                if not torch.equal(current[n:], expected):
                    raise ValueError('uncovered appended frozen output rows')
            # PEFT replaces latent input rows with saved trainable_tokens_delta;
            # their underlying base rows never enter active policy.
        elif current.shape != original.shape or not torch.equal(current, original):
            raise ValueError(f'reference does not match original base: {name}')
        covered.add(base_name)
    if covered != set(keys):
        raise ValueError('original base parameter coverage mismatch')


def _state_identity(teacher: Stage1EmaTeacher, state: dict) -> str:
    digest = hashlib.sha256()
    config = resolve_qwen2_5_vl_model(teacher.model).config.to_dict()
    for key in ('_name_or_path', 'transformers_version'):
        config.pop(key, None)
    digest.update(json.dumps(config, sort_keys=True, allow_nan=False).encode())
    adapter = teacher.model.peft_config['default'].to_dict()
    adapter.pop('base_model_name_or_path', None)
    # PEFT serializes set-valued target modules through sorted JSON values.
    digest.update(json.dumps(adapter, sort_keys=True, allow_nan=False,
                             default=lambda v: sorted(v) if isinstance(v, set) else str(v)).encode())
    for name, value in state['parameters'].items():
        header = json.dumps([name, str(value.dtype), list(value.shape)]).encode()
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _inspect_v5_reference(teacher, student, processor, base_path):
    _topology(teacher, student)
    _validate_processor_layout(teacher, processor)
    base_digest, keys = _base_identity(Path(base_path))
    _validate_base_coverage(teacher, keys)
    state = teacher.state_dict()
    identity = V5ReferenceIdentity(1, base_digest, _state_identity(teacher, state),
                                   _processor_identity(processor), teacher.parameter_names)
    return identity, state


def inspect_v5_reference(teacher: Stage1EmaTeacher, student: nn.Module, processor, *,
                         base_path: Path) -> V5ReferenceIdentity:
    """Authenticate every rank's base/processor/effective reference without saving."""
    identity, _ = _inspect_v5_reference(teacher, student, processor, base_path)
    return identity


def capture_v5_reference(teacher: Stage1EmaTeacher, student: nn.Module, processor, *,
                         base_path: Path, directory: Path) -> V5ReferenceIdentity:
    """Validate and persist initial selected state; caller coordinates rank publication."""
    directory = require_canonical_artifact_path(directory, kind='checkpoint')
    identity, state = _inspect_v5_reference(teacher, student, processor, base_path)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / V5_REFERENCE_FILENAME
    if destination.is_symlink():
        raise ValueError('immutable initial reference must be a regular file')
    if destination.exists():
        existing = torch.load(destination, map_location='cpu', weights_only=True)
        if (set(existing) != set(state) or
                any(existing[key] != state[key] for key in state if key != 'parameters') or
                _state_identity(teacher, existing) != identity.effective_state_sha256):
            raise ValueError('cannot overwrite immutable initial reference')
    else:
        with destination.open('xb') as stream:
            torch.save(state, stream)
    return identity


def restore_v5_reference(teacher: Stage1EmaTeacher, student: nn.Module, processor, *,
                         base_path: Path, directory: Path, expected: V5ReferenceIdentity) -> None:
    """Restore before any helper/reference call, rejecting all content conflicts."""
    # Revalidate even a caller-provided identity instead of trusting its type alone.
    expected = V5ReferenceIdentity(**vars(expected))
    directory = require_canonical_artifact_path(directory, kind='checkpoint')
    _topology(teacher, student)
    _validate_processor_layout(teacher, processor)
    base_digest, keys = _base_identity(Path(base_path))
    if base_digest != expected.base_content_sha256:
        raise ValueError('original base content identity mismatch')
    if _processor_identity(processor) != expected.processor_sha256:
        raise ValueError('processor content identity mismatch')
    if teacher.parameter_names != expected.selected_parameter_names:
        raise ValueError('selected reference parameter identity mismatch')
    _validate_base_coverage(teacher, keys)
    reference_path = directory / V5_REFERENCE_FILENAME
    if reference_path.is_symlink():
        raise ValueError('immutable initial reference must be a regular file')
    state = torch.load(reference_path, map_location='cpu', weights_only=True)
    if state.get('decay') != 1. or state.get('method') != 'dora':
        raise ValueError('initial reference must retain frozen EMA1 DoRA')
    if _state_identity(teacher, state) != expected.effective_state_sha256:
        raise ValueError('initial reference effective-state identity mismatch')
    teacher.load_state_dict(state)
    if _state_identity(teacher, teacher.state_dict()) != expected.effective_state_sha256:
        raise ValueError('restored reference effective-state identity mismatch')
