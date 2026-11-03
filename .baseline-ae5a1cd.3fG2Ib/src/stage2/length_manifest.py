"""Strict, immutable metadata for one exact Stage 2 length audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from PIL import Image

from stage2.data import (
    Stage2LengthStats,
    Stage2PreparedExample,
    build_prepared_example_length_measure,
    parse_jsonl_examples,
    prepare_stage2_example,
    summarize_stage2_lengths,
)
from step_preference.spans import (
    ASSISTANT_PLAN_RENDERING_CONCATENATED,
    normalize_assistant_plan_rendering,
)


STAGE2_LENGTH_MANIFEST_SCHEMA_VERSION = 2


def _class_name(value) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _normalize_json(value):
    """Convert configuration values without falling back to unstable repr strings."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("processor identity cannot contain non-finite floats")
        return value
    if isinstance(value, Enum):
        return _normalize_json(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in normalized:
                raise ValueError("processor identity contains colliding mapping keys")
            normalized[normalized_key] = _normalize_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_json(item) for item in value]
        return sorted(normalized, key=_canonical_json)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "class": _class_name(value),
            "fields": _normalize_json(asdict(value)),
        }
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return {
            "class": _class_name(value),
            "value": _normalize_json(to_dict()),
        }
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        public_attributes = {
            key: item for key, item in attributes.items() if not key.startswith("_")
        }
        return {
            "class": _class_name(value),
            "attributes": _normalize_json(public_attributes),
        }
    stable_string = None if type(value).__str__ is object.__str__ else str(value)
    return {"class": _class_name(value), "string": stable_string}


def _stringify_tokens(value):
    if isinstance(value, Mapping):
        return {str(key): _stringify_tokens(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_tokens(item) for item in value]
    if value is None:
        return None
    return str(value)


def _canonical_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _processor_sha256(processor) -> str:
    tokenizer = processor.tokenizer
    image_processor = getattr(processor, "image_processor", None)
    vocabulary = tokenizer.get_vocab()
    vocabulary_items = sorted(
        ((str(token), _normalize_json(token_id)) for token, token_id in vocabulary.items()),
        key=lambda item: item[0],
    )
    payload = {
        "processor_class": _class_name(processor),
        "processor_config": _normalize_json(processor.to_dict()),
        "image_processor_class": (
            _class_name(image_processor) if image_processor is not None else None
        ),
        "image_processor_config": (
            _normalize_json(image_processor.to_dict())
            if image_processor is not None
            else None
        ),
        "tokenizer_class": _class_name(tokenizer),
        "tokenizer_vocabulary": vocabulary_items,
        "special_tokens_map": _stringify_tokens(
            getattr(tokenizer, "special_tokens_map", {})
        ),
        "special_tokens_map_extended": _stringify_tokens(
            getattr(tokenizer, "special_tokens_map_extended", {})
        ),
        "tokenizer_init_kwargs": _normalize_json(getattr(tokenizer, "init_kwargs", {})),
        "processor_chat_template": _normalize_json(
            getattr(processor, "chat_template", None)
        ),
        "tokenizer_chat_template": _normalize_json(
            getattr(tokenizer, "chat_template", None)
        ),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_int(name: str, value, *, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_float(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_sha256(name: str, value) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class Stage2ImageIdentity:
    path: str
    size: int
    mtime_ns: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("image path must be a non-empty string")
        _require_int("image size", self.size, minimum=0)
        _require_int("image mtime_ns", self.mtime_ns)
        _require_int("image width", self.width, minimum=1)
        _require_int("image height", self.height, minimum=1)


def _validate_stats(stats: Stage2LengthStats) -> None:
    if not isinstance(stats, Stage2LengthStats):
        raise ValueError("stats must be Stage2LengthStats")
    for name in ("total", "retained", "filtered", "minimum", "maximum"):
        _require_int(f"stats.{name}", getattr(stats, name), minimum=0)
    for name in ("filtered_percentage", "median", "p90", "p95", "p99"):
        _require_float(f"stats.{name}", getattr(stats, name))
    if not isinstance(stats.latent_block_distribution, dict):
        raise ValueError("stats.latent_block_distribution must be an object")
    for block_count, count in stats.latent_block_distribution.items():
        _require_int("latent block distribution key", block_count, minimum=0)
        _require_int("latent block distribution count", count, minimum=0)


@dataclass(frozen=True)
class Stage2LengthManifest:
    schema_version: int
    data_sha256: str
    processor_sha256: str
    latent_size: int
    supervise_latent_boundaries: bool
    max_sequence_length: int
    assistant_plan_rendering: str
    images: tuple[Stage2ImageIdentity, ...]
    lengths: tuple[int, ...]
    latent_block_counts: tuple[int, ...]
    retained_indices: tuple[int, ...]
    stats: Stage2LengthStats

    def __post_init__(self) -> None:
        _require_int("schema_version", self.schema_version, minimum=1)
        if self.schema_version != STAGE2_LENGTH_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        _require_sha256("data_sha256", self.data_sha256)
        _require_sha256("processor_sha256", self.processor_sha256)
        _require_int("latent_size", self.latent_size, minimum=1)
        if not isinstance(self.supervise_latent_boundaries, bool):
            raise ValueError("supervise_latent_boundaries must be a boolean")
        _require_int("max_sequence_length", self.max_sequence_length, minimum=1)
        normalize_assistant_plan_rendering(self.assistant_plan_rendering)
        for name in ("images", "lengths", "latent_block_counts", "retained_indices"):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(f"{name} must be a tuple")
        if not all(isinstance(image, Stage2ImageIdentity) for image in self.images):
            raise ValueError("images must contain Stage2ImageIdentity values")
        for index, length in enumerate(self.lengths):
            _require_int(f"lengths[{index}]", length, minimum=1)
        for index, count in enumerate(self.latent_block_counts):
            _require_int(f"latent_block_counts[{index}]", count, minimum=0)
        for index, raw_index in enumerate(self.retained_indices):
            _require_int(f"retained_indices[{index}]", raw_index, minimum=0)
        if len(self.lengths) != len(self.latent_block_counts):
            raise ValueError("lengths and latent_block_counts must have the same cardinality")
        expected_indices = tuple(
            index
            for index, length in enumerate(self.lengths)
            if length <= self.max_sequence_length
        )
        if self.retained_indices != expected_indices:
            raise ValueError(
                "retained_indices must exactly match length <= max_sequence_length"
            )
        _validate_stats(self.stats)
        expected_stats = summarize_stage2_lengths(
            lengths=self.lengths,
            latent_block_counts=self.latent_block_counts,
            max_length=self.max_sequence_length,
        )
        if self.stats != expected_stats:
            raise ValueError("stats do not match manifest lengths and latent block counts")


@dataclass(frozen=True)
class Stage2LengthManifestSnapshot:
    """One validated manifest together with the exact rows ranks will consume."""

    manifest: Stage2LengthManifest
    raw_examples: tuple[dict, ...]
    prepared_examples: tuple[Stage2PreparedExample, ...]


def _image_identity(path: str) -> Stage2ImageIdentity:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"image is not a regular file: {resolved}")
    stat_before = resolved.stat()
    with Image.open(resolved) as image:
        width, height = image.size
    stat_after = resolved.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (
        stat_after.st_size,
        stat_after.st_mtime_ns,
    ):
        raise ValueError(f"image changed while recording identity: {resolved}")
    return Stage2ImageIdentity(
        path=str(resolved),
        size=stat_after.st_size,
        mtime_ns=stat_after.st_mtime_ns,
        width=width,
        height=height,
    )


def validate_stage2_length_manifest(
    *,
    path: str | Path,
    expected_sha256: str,
    data_path: str | Path,
    image_root: str | Path,
    processor,
    latent_size: int,
    supervise_latent_boundaries: bool,
    max_sequence_length: int,
    assistant_plan_rendering: str = ASSISTANT_PLAN_RENDERING_CONCATENATED,
) -> Stage2LengthManifestSnapshot:
    """Bind a strict manifest to one parsed dataset snapshot and current images."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"manifest path must be a regular file: {path}")
    manifest_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("Stage 2 length manifest SHA-256 does not match")
    manifest = read_stage2_length_manifest(path)
    if path.read_bytes() != manifest_bytes:
        raise ValueError("Stage 2 length manifest changed while being read")
    data_bytes = Path(data_path).read_bytes()
    if hashlib.sha256(data_bytes).hexdigest() != manifest.data_sha256:
        raise ValueError("Stage 2 length manifest data SHA-256 does not match")
    if _processor_sha256(processor) != manifest.processor_sha256:
        raise ValueError("Stage 2 length manifest processor identity does not match")
    if (
        manifest.latent_size,
        manifest.supervise_latent_boundaries,
        manifest.max_sequence_length,
        manifest.assistant_plan_rendering,
    ) != (
        latent_size,
        supervise_latent_boundaries,
        max_sequence_length,
        assistant_plan_rendering,
    ):
        raise ValueError("Stage 2 length manifest configuration does not match")
    raw_examples = tuple(parse_jsonl_examples(data_bytes.decode("utf-8")))
    if len(manifest.lengths) != len(raw_examples):
        raise ValueError("Stage 2 length manifest cardinality does not match dataset")
    prepared_examples = tuple(
        prepare_stage2_example(
            raw_example,
            image_root=Path(image_root),
            latent_size=latent_size,
            assistant_plan_rendering=assistant_plan_rendering,
        )
        for raw_example in raw_examples
    )
    images = tuple(
        _image_identity(path)
        for prepared in prepared_examples
        for path in prepared.original_image_paths
    )
    if images != manifest.images:
        raise ValueError("Stage 2 length manifest image identities do not match")
    return Stage2LengthManifestSnapshot(
        manifest=manifest,
        raw_examples=raw_examples,
        prepared_examples=prepared_examples,
    )


def build_stage2_length_manifest(
    *,
    data_path: str | Path,
    image_root: str | Path,
    processor,
    processor_source: str,
    latent_size: int,
    supervise_latent_boundaries: bool,
    max_sequence_length: int,
    assistant_plan_rendering: str = ASSISTANT_PLAN_RENDERING_CONCATENATED,
) -> Stage2LengthManifest:
    """Audit each ordered raw record exactly once and bind all length inputs."""
    if not isinstance(processor_source, str) or not processor_source:
        raise ValueError("processor_source must be a non-empty string")
    _require_int("latent_size", latent_size, minimum=1)
    if not isinstance(supervise_latent_boundaries, bool):
        raise ValueError("supervise_latent_boundaries must be a boolean")
    _require_int("max_sequence_length", max_sequence_length, minimum=1)
    assistant_plan_rendering = normalize_assistant_plan_rendering(
        assistant_plan_rendering
    )

    data_path = Path(data_path)
    image_root = Path(image_root)
    data_bytes = data_path.read_bytes()
    data_sha256 = hashlib.sha256(data_bytes).hexdigest()
    raw_examples = parse_jsonl_examples(data_bytes.decode("utf-8"))
    measure_length = build_prepared_example_length_measure(processor)
    lengths = []
    latent_block_counts = []
    images = []
    for raw_example in raw_examples:
        prepared = prepare_stage2_example(
            raw_example,
            image_root=image_root,
            latent_size=latent_size,
            assistant_plan_rendering=assistant_plan_rendering,
        )
        images_before = tuple(
            _image_identity(path) for path in prepared.original_image_paths
        )
        lengths.append(measure_length(prepared))
        images_after = tuple(
            _image_identity(path) for path in prepared.original_image_paths
        )
        if images_after != images_before:
            raise ValueError("image changed during length measurement")
        latent_block_counts.append(prepared.latent_block_count)
        images.extend(images_before)

    lengths_tuple = tuple(lengths)
    latent_block_counts_tuple = tuple(latent_block_counts)
    stats = summarize_stage2_lengths(
        lengths=lengths_tuple,
        latent_block_counts=latent_block_counts_tuple,
        max_length=max_sequence_length,
    )
    retained_indices = tuple(
        index for index, length in enumerate(lengths_tuple) if length <= max_sequence_length
    )
    return Stage2LengthManifest(
        schema_version=STAGE2_LENGTH_MANIFEST_SCHEMA_VERSION,
        data_sha256=data_sha256,
        processor_sha256=_processor_sha256(processor),
        latent_size=latent_size,
        supervise_latent_boundaries=supervise_latent_boundaries,
        max_sequence_length=max_sequence_length,
        assistant_plan_rendering=assistant_plan_rendering,
        images=tuple(images),
        lengths=lengths_tuple,
        latent_block_counts=latent_block_counts_tuple,
        retained_indices=retained_indices,
        stats=stats,
    )


def _manifest_payload(manifest: Stage2LengthManifest) -> dict:
    return asdict(manifest)


def _require_exact_fields(payload: Mapping, expected_type, context: str) -> None:
    expected = {field.name for field in fields(expected_type)}
    actual = set(payload)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise ValueError(
            f"{context} has unknown fields {unknown} or missing fields {missing}"
        )


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _parse_distribution(value) -> dict[int, int]:
    if not isinstance(value, dict):
        raise ValueError("stats.latent_block_distribution must be an object")
    result = {}
    for raw_key, count in value.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as error:
            raise ValueError("latent block distribution keys must be integers") from error
        if str(key) != raw_key:
            raise ValueError("latent block distribution keys must be canonical integers")
        if key in result:
            raise ValueError("duplicate latent block distribution key")
        result[key] = count
    return result


def _manifest_from_payload(payload) -> Stage2LengthManifest:
    if not isinstance(payload, dict):
        raise ValueError("Stage 2 length manifest must be a JSON object")
    _require_exact_fields(payload, Stage2LengthManifest, "manifest")

    raw_images = payload["images"]
    if not isinstance(raw_images, list):
        raise ValueError("images must be an array")
    images = []
    for raw_image in raw_images:
        if not isinstance(raw_image, dict):
            raise ValueError("image identity must be an object")
        _require_exact_fields(raw_image, Stage2ImageIdentity, "image identity")
        images.append(Stage2ImageIdentity(**raw_image))

    raw_stats = payload["stats"]
    if not isinstance(raw_stats, dict):
        raise ValueError("stats must be an object")
    _require_exact_fields(raw_stats, Stage2LengthStats, "stats")
    raw_stats = dict(raw_stats)
    raw_stats["latent_block_distribution"] = _parse_distribution(
        raw_stats["latent_block_distribution"]
    )
    stats = Stage2LengthStats(**raw_stats)

    for name in ("lengths", "latent_block_counts", "retained_indices"):
        if not isinstance(payload[name], list):
            raise ValueError(f"{name} must be an array")
    return Stage2LengthManifest(
        schema_version=payload["schema_version"],
        data_sha256=payload["data_sha256"],
        processor_sha256=payload["processor_sha256"],
        latent_size=payload["latent_size"],
        supervise_latent_boundaries=payload["supervise_latent_boundaries"],
        max_sequence_length=payload["max_sequence_length"],
        assistant_plan_rendering=payload["assistant_plan_rendering"],
        images=tuple(images),
        lengths=tuple(payload["lengths"]),
        latent_block_counts=tuple(payload["latent_block_counts"]),
        retained_indices=tuple(payload["retained_indices"]),
        stats=stats,
    )


def read_stage2_length_manifest(path: str | Path) -> Stage2LengthManifest:
    """Read one strict regular-file manifest without following symlinks."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"manifest path must be a regular file: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Stage 2 length manifest JSON: {error}") from error
    return _manifest_from_payload(payload)


def write_stage2_length_manifest(
    path: str | Path, manifest: Stage2LengthManifest
) -> Path:
    """Publish exclusively, accepting only an identical prior publication."""
    if not isinstance(manifest, Stage2LengthManifest):
        raise TypeError("manifest must be Stage2LengthManifest")
    manifest.__post_init__()
    path = Path(path)
    serialized = _canonical_json(_manifest_payload(manifest)) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError:
        existing = read_stage2_length_manifest(path)
        if existing != manifest:
            raise ValueError(f"path already contains a different Stage 2 length manifest: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    """Build and exclusively publish the one pre-distributed length manifest."""
    import argparse

    from transformers import AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--processor-source", required=True)
    parser.add_argument("--latent-size", required=True, type=int)
    parser.add_argument(
        "--supervise-latent-boundaries",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-sequence-length", required=True, type=int)
    parser.add_argument(
        "--assistant-plan-rendering",
        default=ASSISTANT_PLAN_RENDERING_CONCATENATED,
    )
    parser.add_argument("--manifest-path", required=True)
    args = parser.parse_args(argv)
    processor = AutoProcessor.from_pretrained(args.processor_source)
    manifest = build_stage2_length_manifest(
        data_path=args.data_path,
        image_root=args.image_root,
        processor=processor,
        processor_source=args.processor_source,
        latent_size=args.latent_size,
        supervise_latent_boundaries=args.supervise_latent_boundaries,
        max_sequence_length=args.max_sequence_length,
        assistant_plan_rendering=args.assistant_plan_rendering,
    )
    write_stage2_length_manifest(args.manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
