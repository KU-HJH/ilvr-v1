"""Independent Stage 1 reference ownership and authenticated source restoration."""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from types import SimpleNamespace

import torch
from torch import nn

from peft_common import capture_qwen2_5_vl_architecture
from stage1.checkpointing import (
    Stage1ArtifactKind, Stage1PeftMetadataSnapshot, Stage1PeftPreflightWitness,
    classify_stage1_artifact, decode_stage1_peft_metadata_snapshot,
    load_stage1_peft_artifact, preflight_stage1_peft_artifact,
    read_stage1_peft_metadata_snapshot, _decode_preflight_witness,
)
from stage2.checkpointing import configure_stage2_metadata
from stage2_objective.vocabulary import TextVocabulary, build_text_vocabulary
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds


def freeze_and_validate_reference(model: nn.Module, student: nn.Module) -> None:
    if any(module is model for module in student.modules()):
        raise ValueError("reference cannot be the student or a student child")
    student_storage = {
        (tensor.device, tensor.untyped_storage().data_ptr())
        for tensor in (*student.parameters(), *student.buffers())
        if tensor.numel() and not tensor.is_meta
    }
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.is_meta:
            raise ValueError("reference contains unrestored meta tensors")
        if tensor.numel() and (tensor.device, tensor.untyped_storage().data_ptr()) in student_storage:
            raise ValueError("reference shares student storage")
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    model.eval()


@dataclass(frozen=True)
class ReferenceIdentity:
    artifact: str
    base_source: str
    kind: str
    weights_digest: str
    metadata_digest: str
    vocabulary_identity: str


@dataclass(frozen=True)
class ReferenceSource:
    identity: ReferenceIdentity
    stage1_preflight: Any | None = field(repr=False)
    # Credentials belong only to transient loader arguments, never provenance.
    load_kwargs: dict[str, Any] = field(repr=False)
    _authentication: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ReferenceLoaders:
    load_config: Callable
    load_processor: Callable
    load_model: Callable


@dataclass(frozen=True)
class FrozenReference:
    model: nn.Module
    identity: ReferenceIdentity


@dataclass(frozen=True)
class FileWitness:
    path: str
    size: int
    mtime_ns: int
    ctime_ns: int
    inode: int
    device: int
    digest: str


@dataclass(frozen=True)
class WeightSourceWitness:
    """Producer-authenticated local weights; later consumers check only stats.

    Share the same registry between student and reference preparation. An existing
    witness must be produced by this authenticator (or an equivalent trusted
    producer), never fabricated from a metadata-only checkpoint/run manifest.
    """

    source: str
    selectors: tuple[tuple[str, Any], ...]
    weights_digest: str
    files: tuple[FileWitness, ...]
    metadata_files: tuple[FileWitness, ...]
    inventory: tuple[str, ...]


@dataclass(frozen=True)
class _ReferenceAuthentication:
    identity: ReferenceIdentity
    dependencies: tuple[WeightSourceWitness, ...]
    architecture_json: bytes
    config_json: bytes = field(repr=False)
    stage1_preflight: Stage1PeftPreflightWitness | None = field(repr=False, compare=False)
    latent_size: int
    token_ids: tuple[int, int, int]
    selectors: tuple[tuple[str, Any], ...]


_SUPPORTED_SELECTORS = {"cache_dir", "revision", "token", "local_files_only"}
_METADATA_NAMES = (
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "preprocessor_config.json", "processor_config.json", "stage1_peft_metadata.json",
    "adapter_config.json",
)


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _selectors(source_kwargs: dict) -> tuple[tuple[str, Any], ...]:
    if set(source_kwargs) - _SUPPORTED_SELECTORS:
        raise ValueError("unsupported reference source selector")
    result = []
    for key, value in sorted(source_kwargs.items()):
        if key == "token":
            continue
        if value is not None and not isinstance(value, (str, bool)):
            raise ValueError("reference source selectors must be strings or booleans")
        result.append((key, value))
    return tuple(result)


def _local_source(source) -> Path:
    path = Path(source).expanduser()
    if not path.is_dir():
        raise ValueError("reference requires an authenticated local source/snapshot; unresolved remote IDs are unsupported")
    return path.resolve()


def _stat(path: Path) -> tuple[int, ...]:
    try:
        value = path.stat()
        if not path.is_file():
            raise ValueError("reference dependency is not a regular file")
        return value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino, value.st_dev
    except OSError as error:
        raise ValueError("reference source dependency changed or disappeared") from error


def _validate_files(files: tuple[FileWitness, ...]) -> None:
    for item in files:
        if _stat(Path(item.path)) != (item.size, item.mtime_ns, item.ctime_ns, item.inode, item.device):
            raise ValueError("reference source dependency changed after authentication")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _authenticate_file(path: Path, registry: dict, *, payload: bytes | None = None) -> FileWitness:
    path = path.absolute()
    key = ("reference-file", str(path.resolve()))
    existing = registry.get(key)
    if existing is not None:
        if not isinstance(existing, FileWitness):
            raise TypeError("invalid authenticated file witness")
        _validate_files((existing,))
        return existing
    before = _stat(path)
    if payload is not None and len(payload) != before[0]:
        raise ValueError("Stage 1 preflight payload no longer matches source size")
    digest = hashlib.sha256(payload).hexdigest() if payload is not None else _hash_file(path)
    if _stat(path) != before:
        raise ValueError("reference source mutated during authentication")
    item = FileWitness(str(path), *before, digest)
    registry[key] = item
    return item


def _validate_weight_witness(witness: WeightSourceWitness, source: str, selectors: tuple) -> None:
    if not isinstance(witness, WeightSourceWitness):
        raise TypeError("a WeightSourceWitness is required; metadata-only evidence is insufficient")
    if witness.source != source or witness.selectors != selectors or not witness.files:
        raise ValueError("weight source witness source/selectors mismatch")
    if witness.weights_digest != _digest([(Path(f.path).name, f.digest) for f in witness.files]):
        raise ValueError("weight source witness digest mismatch")
    if _inventory(Path(source)) != witness.inventory:
        raise ValueError("reference source file inventory changed after authentication")
    _validate_files(witness.files + witness.metadata_files)


def _full_weight_files(path: Path) -> tuple[Path, ...]:
    if classify_stage1_artifact(path) is not Stage1ArtifactKind.FULL:
        raise ValueError("base source must contain full weights")
    for filename in ("model.safetensors", "pytorch_model.bin"):
        if (path / filename).is_file():
            return (path / filename,)
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (path / filename).is_file():
            payload = json.loads((path / filename).read_text())
            return tuple(path / name for name in sorted(set(payload["weight_map"].values())))
    raise ValueError("source has no supported immutable model weights")


def _inventory(path: Path) -> tuple[str, ...]:
    return tuple(sorted(p.name for p in path.iterdir() if p.is_file()))


def _source_witness(path: Path, selectors: tuple, registry: dict,
                    payloads: dict[str, bytes] | None = None) -> WeightSourceWitness:
    key = ("reference-weights", str(path), selectors)
    # A selector change cannot erase this local source's authentication history.
    # Validate every prior witness before discovering files or seeding the cache.
    same_source = tuple(
        (prior_key, witness) for prior_key, witness in registry.items()
        if isinstance(prior_key, tuple) and len(prior_key) == 3
        and prior_key[:2] == key[:2]
    )
    for prior_key, witness in same_source:
        _validate_weight_witness(witness, str(path), prior_key[2])
    for _, witness in same_source:
        for item in witness.files + witness.metadata_files:
            registry.setdefault(("reference-file", str(Path(item.path).resolve())), item)
    existing = registry.get(key)
    if existing is not None:
        return existing
    paths = tuple(path / name for name in sorted(payloads)) if payloads is not None else _full_weight_files(path)
    # Evidence is checked above before any expensive pass. A large unevidenced
    # source requires explicit approval outside this loader, never an implicit hash.
    inventory = _inventory(path)
    if sum(_stat(p)[0] for p in paths
           if ("reference-file", str(p.resolve())) not in registry) > 10_000_000_000:
        raise ValueError("large weight authentication requires an existing producer witness or explicit approval")
    files = tuple(_authenticate_file(p, registry, payload=None if payloads is None else payloads[p.name]) for p in paths)
    metadata_paths = [path / name for name in _METADATA_NAMES if (path / name).is_file()]
    metadata_paths += sorted(path.glob("*.index.json"))
    metadata_files = tuple(_authenticate_file(p, registry) for p in metadata_paths)
    witness = WeightSourceWitness(str(path), selectors,
        _digest([(p.name, f.digest) for p, f in zip(paths, files)]), files, metadata_files, inventory)
    if _inventory(path) != inventory:
        raise ValueError("reference source inventory changed during authentication")
    registry[key] = witness
    return witness


def authenticate_weight_source(path: Path, *, source_kwargs: dict,
                               witnessed_sources: dict) -> WeightSourceWitness:
    """Authenticate local full weights once, or consume a typed producer witness.

    The registry key is ("reference-weights", canonical_source, selectors), where
    selectors is the sorted nonsecret tuple. Individual files are deduplicated
    across selector keys too. No save/publication consumer should call this again.
    """
    return _source_witness(_local_source(path), _selectors(source_kwargs), witnessed_sources)


def _vocabulary(processor, config, *, input_rows=None, output_rows=None) -> TextVocabulary:
    return build_text_vocabulary(processor.tokenizer, config,
        embedding_rows=config.vocab_size if input_rows is None else input_rows,
        output_rows=config.vocab_size if output_rows is None else output_rows)


def _require_vocabulary(actual: TextVocabulary, expected: TextVocabulary) -> None:
    if actual != expected:
        raise ValueError("reference vocabulary identity does not match student")


def _markers(config) -> tuple[int, int, int]:
    pairs = (("latent_token_id", "latent_pad_token_id"),
             ("latent_start_id", "latent_start_token_id"),
             ("latent_end_id", "latent_end_token_id"))
    ids = []
    for short, long in pairs:
        values = [getattr(config, name) for name in (short, long) if getattr(config, name, None) is not None]
        if not values or any(type(value) is not int or value != values[0] for value in values):
            raise ValueError("Stage 1 reference has missing or inconsistent latent marker IDs")
        ids.append(values[0])
    if len(set(ids)) != 3:
        raise ValueError("Stage 1 reference latent marker IDs must be distinct")
    return tuple(ids)


def prepare_reference_source(path: Path, *, processor: Any, vocabulary: TextVocabulary,
                             loaders: ReferenceLoaders, source_kwargs: dict,
                             witnessed_sources: dict,
                             stage1_preflight: Any | None = None) -> ReferenceSource:
    selectors = _selectors(source_kwargs)
    artifact = _local_source(path)
    key = ("reference-source", str(artifact), selectors)
    existing = witnessed_sources.get(key)
    if existing is not None:
        if not isinstance(existing, ReferenceSource):
            raise TypeError("invalid reference source witness")
        _validate_source(existing, processor, vocabulary, source_kwargs)
        if stage1_preflight is not None and stage1_preflight is not existing.stage1_preflight:
            raise ValueError("reference source already has a different Stage 1 preflight witness")
        return _replace_source_kwargs(existing, source_kwargs)
    if (artifact / "stage2_peft_metadata.json").exists() or (artifact / "stage2_objective.json").exists():
        raise ValueError("Stage 2 resume artifact cannot be a fixed Stage 1 reference")
    kind = classify_stage1_artifact(artifact)
    source_processor = loaders.load_processor(str(artifact), **source_kwargs)
    dependencies = []
    if kind is Stage1ArtifactKind.PEFT:
        if stage1_preflight is not None:
            if not isinstance(stage1_preflight, Stage1PeftPreflightWitness):
                raise TypeError("stage1_preflight must be a Stage1PeftPreflightWitness")
            if stage1_preflight.artifact_dir != artifact:
                raise ValueError("Stage 1 preflight artifact path mismatch")
            snapshot = Stage1PeftMetadataSnapshot(artifact, stage1_preflight.metadata_json)
        else:
            snapshot = read_stage1_peft_metadata_snapshot(artifact)
        metadata = decode_stage1_peft_metadata_snapshot(snapshot)
        base_source = metadata.base_model_reference
        base = _local_source(base_source)
        config = loaders.load_config(base_source, **source_kwargs)
        if stage1_preflight is None:
            stage1_preflight = preflight_stage1_peft_artifact(source_processor, config, artifact,
                requested_base_model_reference=base_source, metadata_snapshot=snapshot)
        _decode_preflight_witness(stage1_preflight)  # Small scalar/config consistency only.
        if _json_bytes(capture_qwen2_5_vl_architecture(config)) != stage1_preflight.base_architecture_json:
            raise ValueError("Stage 1 preflight base architecture mismatch")
        if getattr(config, "stage", None) == "stage2":
            raise ValueError("Stage 2 base cannot be used for a fixed Stage 1 reference")
        dependencies.append(authenticate_weight_source(base, source_kwargs=source_kwargs, witnessed_sources=witnessed_sources))
        dependencies.append(_source_witness(artifact, selectors, witnessed_sources, {
            "adapter_model.safetensors": stage1_preflight.adapter_safetensors,
            "ilvr_token_rows.safetensors": stage1_preflight.sparse_safetensors,
        }))
        sidecars = {Path(item.path).name: item.digest for item in dependencies[-1].metadata_files}
        for name, payload in (("stage1_peft_metadata.json", stage1_preflight.metadata_json),
                              ("adapter_config.json", stage1_preflight.adapter_config_json)):
            if sidecars.get(name) != hashlib.sha256(payload).hexdigest():
                raise ValueError("Stage 1 preflight metadata sidecar changed")
        config = deepcopy(config)
        config.vocab_size = metadata.resized_vocab_size
        config.tie_word_embeddings = False  # Sparse output rows own the untied runtime head.
        config.latent_size = metadata.latent_size
        config.latent_token_id, config.latent_start_id, config.latent_end_id = metadata.latent_token_ids
    else:
        if stage1_preflight is not None:
            raise ValueError("full reference cannot use a Stage 1 PEFT witness")
        base_source = str(artifact)
        config = loaders.load_config(str(artifact), **source_kwargs)
        if getattr(config, "stage", None) != "stage1":
            raise ValueError("fixed reference must be a Stage 1 checkpoint, not a Stage 2 resume artifact")
        dependencies.append(authenticate_weight_source(artifact, source_kwargs=source_kwargs, witnessed_sources=witnessed_sources))
    token_ids = _markers(config)
    if type(getattr(config, "latent_size", None)) is not int or config.latent_size < 1:
        raise ValueError("Stage 1 reference requires a positive latent_size")
    _require_vocabulary(_vocabulary(source_processor, config), vocabulary)
    _require_vocabulary(_vocabulary(processor, config), vocabulary)
    architecture_json = _json_bytes(capture_qwen2_5_vl_architecture(config))
    identity = ReferenceIdentity(str(artifact), base_source, kind.value,
        _digest([w.weights_digest for w in dependencies]),
        _digest({"metadata": [(w.source, [(f.path, f.digest) for f in w.metadata_files]) for w in dependencies],
                 "architecture": json.loads(architecture_json), "latent_ids": token_ids,
                 "latent_size": config.latent_size, "selectors": selectors}), vocabulary.identity)
    authentication = _ReferenceAuthentication(identity, tuple(dependencies), architecture_json,
        _json_bytes(config.to_dict()), stage1_preflight, config.latent_size, token_ids, selectors)
    result = ReferenceSource(identity, stage1_preflight, dict(source_kwargs), authentication)
    witnessed_sources[key] = result
    return result


def _replace_source_kwargs(source: ReferenceSource, kwargs: dict) -> ReferenceSource:
    return ReferenceSource(source.identity, source.stage1_preflight, dict(kwargs), source._authentication)


def _validate_source(source, processor, vocabulary, kwargs):
    authentication = source._authentication
    if not isinstance(authentication, _ReferenceAuthentication) or source.identity != authentication.identity:
        raise ValueError("reference source requires intact preparation authentication")
    if _selectors(kwargs) != authentication.selectors:
        raise ValueError("reference source selectors changed after authentication")
    if vocabulary.identity != source.identity.vocabulary_identity:
        raise ValueError("reference vocabulary identity changed")
    if source.stage1_preflight is not authentication.stage1_preflight:
        raise ValueError("reference preflight changed after authentication")
    _require_vocabulary(_vocabulary(processor, SimpleNamespace(**json.loads(authentication.config_json))), vocabulary)
    for dependency in authentication.dependencies:
        _validate_weight_witness(dependency, dependency.source, authentication.selectors)


def align_reference_precision(reference: nn.Module, student: nn.Module, *, device=None) -> None:
    """Match every replicated tensor without rounding unrelated FP32 adapters.

    Full/full and PEFT/PEFT require identical named topology. There is no
    established semantic mapping for a full reference and a PEFT student.
    Validate the entire policy before changing any tensor; preserve identities,
    frozen ownership, and source values whenever no conversion is required.
    """
    placements = {}
    for kind in ("parameters", "buffers"):
        actual = dict(getattr(reference, "named_" + kind)(remove_duplicate=False))
        expected = dict(getattr(student, "named_" + kind)(remove_duplicate=False))
        if actual.keys() != expected.keys():
            raise ValueError(f"reference/student named {kind} topology mismatch")
        actual_aliases, expected_aliases = {}, {}
        for name, target in actual.items():
            source = expected[name]
            if target.shape != source.shape:
                raise ValueError(f"reference/student {kind} shape mismatch: {name}")
            actual_aliases.setdefault(id(target), []).append(name)
            expected_aliases.setdefault(id(source), []).append(name)
            policy = (source.device if device is None else torch.device(device), source.dtype)
            prior = placements.setdefault(id(target), (target, policy))
            if prior[1] != policy:
                raise ValueError(f"reference/student aliased precision policy mismatch: {name}")
        if {tuple(names) for names in actual_aliases.values()} != {
                tuple(names) for names in expected_aliases.values()}:
            raise ValueError(f"reference/student named {kind} alias topology mismatch")
    with torch.no_grad():
        for target, (target_device, target_dtype) in placements.values():
            if target.device != target_device or target.dtype != target_dtype:
                target.data = target.data.to(device=target_device, dtype=target_dtype)


def load_frozen_reference(source: ReferenceSource, *, student: nn.Module,
                          processor: Any, vocabulary: TextVocabulary,
                          loaders: ReferenceLoaders, device: torch.device,
                          dtype: torch.dtype, backend: str, latent_size: int,
                          supervise_latent_boundaries: bool, split_attention=False) -> FrozenReference:
    _validate_source(source, processor, vocabulary, source.load_kwargs)
    authentication = source._authentication
    if backend not in {"eager", "sdpa"}:
        raise ValueError("reference requires eager or sdpa no_cache execution")
    if type(latent_size) is not int or latent_size != authentication.latent_size:
        raise ValueError("reference latent_size differs from Stage 1 source")
    if type(supervise_latent_boundaries) is not bool:
        raise ValueError("supervise_latent_boundaries must be boolean")
    if hasattr(student, "config"):
        expected = json.loads(authentication.architecture_json)
        actual = capture_qwen2_5_vl_architecture(student.config)
        # Runtime objective setup disables dropout after checkpoint restoration.
        expected["decoder"].pop("attention_dropout", None)
        actual["decoder"].pop("attention_dropout", None)
        if actual != expected:
            raise ValueError("reference architecture does not match student")
        if getattr(student.config, "latent_size", None) != latent_size:
            raise ValueError("reference latent_size does not match student")
        if _markers(student.config) != authentication.token_ids:
            raise ValueError("reference latent marker IDs do not match student")
    origin = source.identity.base_source if source.identity.kind == "peft" else source.identity.artifact
    model = loaders.load_model(origin, **source.load_kwargs, attn_implementation=backend,
                               **({'stage2_split_attention': True} if split_attention else {}))
    # Check ownership before resizing, configuration, placement, or PEFT attachment.
    freeze_and_validate_reference(model, student)
    if source.identity.kind == "peft":
        model = load_stage1_peft_artifact(model, processor, Path(source.identity.artifact),
            requested_base_model_reference=source.identity.base_source,
            is_trainable=False, preflight=source.stage1_preflight)
    if getattr(model.config, "stage", None) != "stage1":
        raise ValueError("restored reference must be a Stage 1 model")
    if _json_bytes(capture_qwen2_5_vl_architecture(model.config)) != authentication.architecture_json:
        raise ValueError("restored reference architecture differs from authenticated source")
    if _markers(model.config) != authentication.token_ids or model.config.latent_size != latent_size:
        raise ValueError("restored reference latent configuration differs from source")
    _require_vocabulary(_vocabulary(processor, model.config,
        input_rows=model.get_input_embeddings().weight.shape[0],
        output_rows=model.get_output_embeddings().weight.shape[0]), vocabulary)
    configure_stage2_metadata(model.config, LatentTokenIds(*authentication.token_ids),
        latent_size=latent_size, supervise_latent_boundaries=supervise_latent_boundaries,
        rollout_backend="no_cache")
    model.config.use_cache = False
    # Preserve the source configuration in authentication; disable runtime dropout
    # only after faithful restoration and validation of the original architecture.
    from .execution import apply_objective_dropout_policy
    apply_objective_dropout_policy(model)
    model.config.attention_dropout = 0.0
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if getattr(model, "is_gradient_checkpointing", False):
        raise ValueError("reference gradient checkpointing must be disabled")
    # The scalar dtype argument is retained for call compatibility; mixed PEFT
    # precision is determined by the actual student, never its first tensor.
    align_reference_precision(model, student, device=device)
    freeze_and_validate_reference(model, student)
    # Detect mutation during the restore too; no dependency is rehashed.
    _validate_source(source, processor, vocabulary, source.load_kwargs)
    return FrozenReference(model, source.identity)
