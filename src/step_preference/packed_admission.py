"""Strict schema-1 packed-admission artifacts and capacity witnesses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence

from step_preference.negatives import CheckpointFile


PACKED_ADMISSION_SCHEMA_VERSION = 1
PACKED_ADMISSION_MANIFEST_KIND = "stage2b-packed-admission"
PACKED_ADMISSION_VERIFICATION_KIND = "stage2b-packed-admission-verification"
_REASONS = frozenset({"retained", "packed_length_exceeds_limit"})
_SHA256_CHARS = frozenset("0123456789abcdef")


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_absolute_path(name: str, value: object) -> str:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{name} must be a non-empty absolute path")
    return value


def _require_tuple(name: str, value: object) -> tuple:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")
    return value


@dataclass(frozen=True)
class PackedAdmissionIdentity:
    original_data_path: str
    original_data_sha256: str
    clean_data_path: str
    clean_data_sha256: str
    source_map_path: str
    source_map_sha256: str
    retained_data_path: str
    retained_data_sha256: str
    clean_length_manifest_path: str
    clean_length_manifest_sha256: str
    image_root_path: str
    checkpoint_path: str
    checkpoint_files: tuple[CheckpointFile, ...]
    tokenizer_config_identity: str
    negative_path: str
    negative_sha256: str
    negative_manifest_path: str
    negative_manifest_sha256: str
    assistant_plan_rendering: str
    attention_implementation: str
    vision_attention_implementation: str
    latent_size: int
    clean_model_limit: int
    clean_prefilter_limit: int
    max_packed_sequence_length: int
    git_commit: str
    tracked_worktree_clean: bool

    def __post_init__(self) -> None:
        for name in (
            "original_data_path", "clean_data_path", "source_map_path",
            "retained_data_path", "clean_length_manifest_path", "image_root_path",
            "checkpoint_path", "negative_path", "negative_manifest_path",
        ):
            _require_absolute_path(name, getattr(self, name))
        for name in (
            "original_data_sha256", "clean_data_sha256", "source_map_sha256",
            "retained_data_sha256", "clean_length_manifest_sha256",
            "tokenizer_config_identity", "negative_sha256", "negative_manifest_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        _require_tuple("checkpoint_files", self.checkpoint_files)
        if not all(isinstance(item, CheckpointFile) for item in self.checkpoint_files):
            raise ValueError("checkpoint_files must contain CheckpointFile values")
        keys = tuple((item.load_root, item.relative_path) for item in self.checkpoint_files)
        if len(keys) != len(set(keys)) or keys != tuple(sorted(keys)):
            raise ValueError("checkpoint_files must be unique and sorted")
        if self.assistant_plan_rendering != "adjacent-text-space-lf-v1":
            raise ValueError("assistant_plan_rendering must be adjacent-text-space-lf-v1")
        if self.attention_implementation != "sdpa":
            raise ValueError("attention_implementation must be sdpa")
        if self.vision_attention_implementation != "flash_attention_2":
            raise ValueError("vision_attention_implementation must be flash_attention_2")
        _require_int("latent_size", self.latent_size, minimum=1)
        _require_int("clean_model_limit", self.clean_model_limit, minimum=1)
        _require_int("clean_prefilter_limit", self.clean_prefilter_limit, minimum=1)
        _require_int("max_packed_sequence_length", self.max_packed_sequence_length, minimum=1)
        if self.clean_model_limit != 8192 or self.clean_prefilter_limit != 4608:
            raise ValueError("clean model and prefilter limits must be 8192 and 4608")
        if self.max_packed_sequence_length != 4608:
            raise ValueError("max_packed_sequence_length must be 4608")
        if not isinstance(self.git_commit, str) or len(self.git_commit) != 40 or any(
            character not in _SHA256_CHARS for character in self.git_commit
        ):
            raise ValueError("git_commit must be a lowercase 40-character commit")
        if type(self.tracked_worktree_clean) is not bool:
            raise ValueError("tracked_worktree_clean must be a boolean")


@dataclass(frozen=True)
class PackedAdmissionMeasurement:
    source_index: int
    sample_id: str
    clean_length: int
    packed_unpadded_length: int
    packed_padded_length: int
    vision_token_count: int
    image_count: int
    reasoning_block_count: int
    branch_count: int
    scored_token_count: int

    def __post_init__(self) -> None:
        _require_int("source_index", self.source_index)
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        for name in ("clean_length", "packed_unpadded_length", "packed_padded_length"):
            _require_int(name, getattr(self, name), minimum=1)
        if not self.packed_padded_length >= self.packed_unpadded_length >= self.clean_length:
            raise ValueError("packed lengths must satisfy padded >= unpadded >= clean")
        if self.packed_padded_length % 128:
            raise ValueError("packed_padded_length must be divisible by 128")
        for name in ("vision_token_count", "image_count"):
            _require_int(name, getattr(self, name))
        for name in ("reasoning_block_count", "branch_count", "scored_token_count"):
            _require_int(name, getattr(self, name), minimum=1)


@dataclass(frozen=True)
class PackedAdmissionRecord(PackedAdmissionMeasurement):
    admission_reason: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.admission_reason not in _REASONS:
            raise ValueError("admission_reason is not allowed")


@dataclass(frozen=True)
class PackedTieClass:
    packed_padded_length: int
    vision_token_count: int
    branch_count: int
    scored_token_count: int
    member_source_indices: tuple[int, ...]
    representative_source_index: int

    def __post_init__(self) -> None:
        _require_int("packed_padded_length", self.packed_padded_length, minimum=1)
        if self.packed_padded_length % 128:
            raise ValueError("packed_padded_length must be divisible by 128")
        for name in ("vision_token_count",):
            _require_int(name, getattr(self, name))
        for name in ("branch_count", "scored_token_count"):
            _require_int(name, getattr(self, name), minimum=1)
        _require_tuple("member_source_indices", self.member_source_indices)
        if len(self.member_source_indices) < 2:
            raise ValueError("tie classes require at least two members")
        for value in self.member_source_indices:
            _require_int("tie member source index", value)
        if tuple(sorted(set(self.member_source_indices)) ) != self.member_source_indices:
            raise ValueError("tie members must be unique and sorted")
        _require_int("representative_source_index", self.representative_source_index)
        if self.representative_source_index != self.member_source_indices[0]:
            raise ValueError("tie representative must be the lowest source index")


@dataclass(frozen=True)
class PackedBoundaryWitnesses:
    tie_classes: tuple[PackedTieClass, ...]
    pareto_source_indices: tuple[int, ...]
    global_max_source_index: int
    final_probe_source_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_tuple("tie_classes", self.tie_classes)
        if not all(isinstance(item, PackedTieClass) for item in self.tie_classes):
            raise ValueError("tie_classes must contain PackedTieClass values")
        axes = tuple(_tie_axes(item) for item in self.tie_classes)
        if axes != tuple(sorted(axes)) or len(axes) != len(set(axes)):
            raise ValueError("tie_classes must be unique and sorted by axes")
        for name in ("pareto_source_indices", "final_probe_source_indices"):
            values = _require_tuple(name, getattr(self, name))
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be unique and sorted")
            for value in values:
                _require_int(name, value)
        _require_int("global_max_source_index", self.global_max_source_index)
        if not self.final_probe_source_indices:
            raise ValueError("final_probe_source_indices must be non-empty")
        if len(self.final_probe_source_indices) > 8:
            raise ValueError("packed capacity witness set exceeds eight rows")
        expected = tuple(sorted(set(self.pareto_source_indices + (self.global_max_source_index,))))
        if self.final_probe_source_indices != expected:
            raise ValueError("final probes must be the Pareto/global maximum union")


@dataclass(frozen=True)
class PackedAdmissionCounts:
    original: int
    clean_admissible: int
    mapped: int
    retained: int
    filtered: int

    def __post_init__(self) -> None:
        for name in ("original", "clean_admissible", "mapped", "retained", "filtered"):
            _require_int(name, getattr(self, name))
        if self.mapped != self.retained + self.filtered:
            raise ValueError("mapped must equal retained plus filtered")
        if not self.mapped <= self.clean_admissible <= self.original:
            raise ValueError("counts must satisfy mapped <= clean_admissible <= original")


@dataclass(frozen=True)
class PackedAdmissionManifest:
    schema_version: int
    kind: str
    identity: PackedAdmissionIdentity
    records: tuple[PackedAdmissionRecord, ...]
    retained_source_indices: tuple[int, ...]
    retained_sample_ids: tuple[str, ...]
    filtered_source_indices: tuple[int, ...]
    filtered_sample_ids: tuple[str, ...]
    boundary_witnesses: PackedBoundaryWitnesses
    counts: PackedAdmissionCounts

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != PACKED_ADMISSION_SCHEMA_VERSION:
            raise ValueError("unsupported packed admission schema_version")
        if self.kind != PACKED_ADMISSION_MANIFEST_KIND:
            raise ValueError("invalid packed admission manifest kind")
        if not isinstance(self.identity, PackedAdmissionIdentity):
            raise ValueError("identity must be a PackedAdmissionIdentity")
        _require_tuple("records", self.records)
        if not self.records or not all(isinstance(item, PackedAdmissionRecord) for item in self.records):
            raise ValueError("records must be non-empty PackedAdmissionRecord values")
        source_indices = tuple(item.source_index for item in self.records)
        sample_ids = tuple(item.sample_id for item in self.records)
        if source_indices != tuple(sorted(source_indices)) or len(source_indices) != len(set(source_indices)):
            raise ValueError("records contain duplicate or unordered source indices")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("records contain duplicate sample IDs")
        retained = tuple(item for item in self.records if item.admission_reason == "retained")
        filtered = tuple(item for item in self.records if item.admission_reason == "packed_length_exceeds_limit")
        for item in retained:
            if item.packed_padded_length > self.identity.max_packed_sequence_length:
                raise ValueError("retained record exceeds the packed length limit")
        for item in filtered:
            if item.packed_padded_length <= self.identity.max_packed_sequence_length:
                raise ValueError("filtered record does not exceed the packed length limit")
        _require_partition("retained", retained, self.retained_source_indices, self.retained_sample_ids)
        _require_partition("filtered", filtered, self.filtered_source_indices, self.filtered_sample_ids)
        if not isinstance(self.boundary_witnesses, PackedBoundaryWitnesses):
            raise ValueError("boundary_witnesses must be PackedBoundaryWitnesses")
        if self.boundary_witnesses != build_boundary_witnesses(retained):
            raise ValueError("boundary_witnesses do not match retained records")
        if not isinstance(self.counts, PackedAdmissionCounts):
            raise ValueError("counts must be PackedAdmissionCounts")
        if (
            self.counts.mapped != len(self.records)
            or self.counts.retained != len(retained)
            or self.counts.filtered != len(filtered)
        ):
            raise ValueError("counts do not match records and partitions")


@dataclass(frozen=True)
class PackedAdmissionVerification:
    schema_version: int
    kind: str
    status: str
    manifest_path: str
    manifest_sha256: str | None
    clean_admissible_sha256: str | None
    packed_retained_sha256: str | None
    negative_sha256: str | None
    negative_manifest_sha256: str | None
    verified_record_count: int
    retained_source_indices: tuple[int, ...]
    filtered_source_indices: tuple[int, ...]
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != PACKED_ADMISSION_SCHEMA_VERSION:
            raise ValueError("unsupported packed admission verification schema_version")
        if self.kind != PACKED_ADMISSION_VERIFICATION_KIND or self.status not in ("pass", "fail"):
            raise ValueError("invalid packed admission verification kind or status")
        _require_absolute_path("manifest_path", self.manifest_path)
        for name in (
            "manifest_sha256", "clean_admissible_sha256", "packed_retained_sha256",
            "negative_sha256", "negative_manifest_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(name, value)
        _require_int("verified_record_count", self.verified_record_count)
        for name in ("retained_source_indices", "filtered_source_indices"):
            values = _require_tuple(name, getattr(self, name))
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be unique and sorted")
            for value in values:
                _require_int(name, value)
        _require_tuple("errors", self.errors)
        if any(not isinstance(error, str) or not error for error in self.errors):
            raise ValueError("errors must contain non-empty strings")
        if self.status == "pass":
            if self.errors or any(
                getattr(self, name) is None
                for name in (
                    "manifest_sha256", "clean_admissible_sha256", "packed_retained_sha256",
                    "negative_sha256", "negative_manifest_sha256",
                )
            ):
                raise ValueError("passing verification requires all hashes and no errors")
        elif not self.errors or self.verified_record_count or self.retained_source_indices or self.filtered_source_indices:
            raise ValueError("failing verification requires one error and empty result fields")


def _require_partition(
    name: str,
    records: tuple[PackedAdmissionRecord, ...],
    source_indices: object,
    sample_ids: object,
) -> None:
    _require_tuple(f"{name}_source_indices", source_indices)
    _require_tuple(f"{name}_sample_ids", sample_ids)
    if source_indices != tuple(item.source_index for item in records) or sample_ids != tuple(item.sample_id for item in records):
        raise ValueError(f"{name} partition does not match records")


def _tie_axes(row: PackedAdmissionRecord | PackedTieClass) -> tuple[int, int, int, int]:
    return (
        row.packed_padded_length,
        row.vision_token_count,
        row.branch_count,
        row.scored_token_count,
    )


def select_packed_admission(
    measurements: Sequence[PackedAdmissionMeasurement], *, max_packed_length: int
) -> tuple[tuple[PackedAdmissionRecord, ...], tuple[PackedAdmissionRecord, ...]]:
    """Classify validated measurements without changing their order or contents."""
    _require_int("max_packed_length", max_packed_length, minimum=1)
    rows = tuple(measurements)
    if not all(type(item) is PackedAdmissionMeasurement for item in rows):
        raise ValueError("measurements must contain PackedAdmissionMeasurement values")
    source_indices = tuple(item.source_index for item in rows)
    sample_ids = tuple(item.sample_id for item in rows)
    if len(source_indices) != len(set(source_indices)) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("measurements must have unique source indices and sample IDs")
    retained, filtered = [], []
    for item in rows:
        reason = "retained" if item.packed_padded_length <= max_packed_length else "packed_length_exceeds_limit"
        target = retained if reason == "retained" else filtered
        target.append(PackedAdmissionRecord(**asdict(item), admission_reason=reason))
    return tuple(retained), tuple(filtered)


def build_boundary_witnesses(records: Sequence[PackedAdmissionRecord]) -> PackedBoundaryWitnesses:
    """Build the exact Pareto representative/global maximum witness union."""
    rows = tuple(records)
    if not rows:
        raise ValueError("capacity witnesses require non-empty retained records")
    if not all(isinstance(item, PackedAdmissionRecord) for item in rows):
        raise ValueError("capacity witnesses require PackedAdmissionRecord values")
    if any(item.admission_reason != "retained" for item in rows):
        raise ValueError("capacity witnesses require retained records only")
    indices = tuple(item.source_index for item in rows)
    if len(indices) != len(set(indices)):
        raise ValueError("capacity witnesses require unique source indices")
    representatives: dict[tuple[int, int, int, int], PackedAdmissionRecord] = {}
    grouped: dict[tuple[int, int, int, int], list[int]] = {}
    for item in sorted(rows, key=lambda value: value.source_index):
        axes = _tie_axes(item)
        representatives.setdefault(axes, item)
        grouped.setdefault(axes, []).append(item.source_index)
    pareto = tuple(
        representative
        for axes, representative in representatives.items()
        if not any(
            all(left >= right for left, right in zip(other_axes, axes))
            and any(left > right for left, right in zip(other_axes, axes))
            for other_axes in representatives
            if other_axes != axes
        )
    )
    maximum = max(
        rows,
        key=lambda item: (
            item.packed_padded_length, item.packed_unpadded_length,
            item.clean_length, item.source_index,
        ),
    )
    selected = tuple(sorted({item.source_index for item in pareto} | {maximum.source_index}))
    if len(selected) > 8:
        raise ValueError("packed capacity witness set exceeds eight rows")
    ties = tuple(
        PackedTieClass(
            packed_padded_length=axes[0], vision_token_count=axes[1],
            branch_count=axes[2], scored_token_count=axes[3],
            member_source_indices=tuple(members), representative_source_index=members[0],
        )
        for axes, members in sorted(grouped.items())
        if len(members) >= 2
    )
    return PackedBoundaryWitnesses(
        tie_classes=ties,
        pareto_source_indices=tuple(sorted(item.source_index for item in pareto)),
        global_max_source_index=maximum.source_index,
        final_probe_source_indices=selected,
    )


def select_capacity_witness_indices(records: Sequence[PackedAdmissionRecord]) -> tuple[int, ...]:
    return build_boundary_witnesses(records).final_probe_source_indices


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _parse_json(raw: bytes, label: str):
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {value}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {error}") from error


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _require_exact_fields(payload: object, expected_type: type, label: str) -> Mapping[str, object]:
    if not isinstance(payload, dict) or set(payload) != {field.name for field in fields(expected_type)}:
        raise ValueError(f"{label} has unknown or missing fields")
    return payload


def _identity_from_payload(payload: object) -> PackedAdmissionIdentity:
    body = _require_exact_fields(payload, PackedAdmissionIdentity, "packed admission identity")
    checkpoint_payload = body["checkpoint_files"]
    if not isinstance(checkpoint_payload, list):
        raise ValueError("checkpoint_files must be an array")
    checkpoint_files = []
    for entry in checkpoint_payload:
        item = _require_exact_fields(entry, CheckpointFile, "checkpoint file")
        checkpoint_files.append(CheckpointFile(**item))
    return PackedAdmissionIdentity(**{**body, "checkpoint_files": tuple(checkpoint_files)})


def _record_from_payload(payload: object) -> PackedAdmissionRecord:
    return PackedAdmissionRecord(**_require_exact_fields(payload, PackedAdmissionRecord, "packed admission record"))


def _witnesses_from_payload(payload: object) -> PackedBoundaryWitnesses:
    body = _require_exact_fields(payload, PackedBoundaryWitnesses, "packed boundary witnesses")
    tie_payload = body["tie_classes"]
    if not isinstance(tie_payload, list):
        raise ValueError("tie_classes must be an array")
    ties = tuple(PackedTieClass(**_require_exact_fields(item, PackedTieClass, "packed tie class")) for item in tie_payload)
    for name in ("pareto_source_indices", "final_probe_source_indices"):
        if not isinstance(body[name], list):
            raise ValueError(f"{name} must be an array")
    return PackedBoundaryWitnesses(
        tie_classes=ties, pareto_source_indices=tuple(body["pareto_source_indices"]),
        global_max_source_index=body["global_max_source_index"],
        final_probe_source_indices=tuple(body["final_probe_source_indices"]),
    )


def _counts_from_payload(payload: object) -> PackedAdmissionCounts:
    return PackedAdmissionCounts(**_require_exact_fields(payload, PackedAdmissionCounts, "packed admission counts"))


def _manifest_from_payload(payload: object) -> PackedAdmissionManifest:
    body = _require_exact_fields(payload, PackedAdmissionManifest, "packed admission manifest")
    for name in ("records", "retained_source_indices", "retained_sample_ids", "filtered_source_indices", "filtered_sample_ids"):
        if not isinstance(body[name], list):
            raise ValueError(f"{name} must be an array")
    return PackedAdmissionManifest(
        schema_version=body["schema_version"], kind=body["kind"],
        identity=_identity_from_payload(body["identity"]),
        records=tuple(_record_from_payload(item) for item in body["records"]),
        retained_source_indices=tuple(body["retained_source_indices"]),
        retained_sample_ids=tuple(body["retained_sample_ids"]),
        filtered_source_indices=tuple(body["filtered_source_indices"]),
        filtered_sample_ids=tuple(body["filtered_sample_ids"]),
        boundary_witnesses=_witnesses_from_payload(body["boundary_witnesses"]),
        counts=_counts_from_payload(body["counts"]),
    )


def _verification_from_payload(payload: object) -> PackedAdmissionVerification:
    body = _require_exact_fields(payload, PackedAdmissionVerification, "packed admission verification")
    for name in ("retained_source_indices", "filtered_source_indices", "errors"):
        if not isinstance(body[name], list):
            raise ValueError(f"{name} must be an array")
    return PackedAdmissionVerification(
        **{
            **body,
            "retained_source_indices": tuple(body["retained_source_indices"]),
            "filtered_source_indices": tuple(body["filtered_source_indices"]),
            "errors": tuple(body["errors"]),
        }
    )


def _read_strict_bytes(path: str | Path, label: str) -> bytes:
    target = Path(path)
    try:
        initial = os.lstat(target)
    except OSError as error:
        raise ValueError(f"{label} path must be a regular file: {target}") from error
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise ValueError(f"{label} path must be a regular file: {target}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as error:
        raise ValueError(f"{label} path could not be opened safely: {target}") from error
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        opened = os.fstat(handle.fileno())
        expected = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
        observed = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        if observed != expected or not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} changed before opening")
        raw = handle.read()
        final = os.fstat(handle.fileno())
        if (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns) != expected:
            raise ValueError(f"{label} changed while reading")
    return raw


def _write_exclusively(path: str | Path, data: bytes, label: str) -> None:
    target = Path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o666)
    except FileExistsError as error:
        raise ValueError(f"{label} path already exists: {target}") from error
    except OSError as error:
        raise ValueError(f"cannot create {label}: {target}") from error
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(data)


def packed_admission_manifest_sha256(manifest: PackedAdmissionManifest) -> str:
    if not isinstance(manifest, PackedAdmissionManifest):
        raise ValueError("manifest must be PackedAdmissionManifest")
    return hashlib.sha256(_canonical_bytes(asdict(manifest))).hexdigest()


def packed_admission_verification_sha256(verification: PackedAdmissionVerification) -> str:
    if not isinstance(verification, PackedAdmissionVerification):
        raise ValueError("verification must be PackedAdmissionVerification")
    return hashlib.sha256(_canonical_bytes(asdict(verification))).hexdigest()


def write_packed_admission_manifest(path: str | Path, manifest: PackedAdmissionManifest) -> str:
    if not isinstance(manifest, PackedAdmissionManifest):
        raise ValueError("manifest must be PackedAdmissionManifest")
    raw = _canonical_bytes(asdict(manifest))
    _write_exclusively(path, raw, "packed admission manifest")
    return hashlib.sha256(raw).hexdigest()


def read_packed_admission_manifest(path: str | Path) -> PackedAdmissionManifest:
    raw = _read_strict_bytes(path, "packed admission manifest")
    manifest = _manifest_from_payload(_parse_json(raw, "packed admission manifest"))
    if raw != _canonical_bytes(asdict(manifest)):
        raise ValueError("packed admission manifest is not canonical JSON")
    return manifest


def write_packed_admission_verification(path: str | Path, verification: PackedAdmissionVerification) -> str:
    if not isinstance(verification, PackedAdmissionVerification):
        raise ValueError("verification must be PackedAdmissionVerification")
    raw = _canonical_bytes(asdict(verification))
    _write_exclusively(path, raw, "packed admission verification")
    return hashlib.sha256(raw).hexdigest()


def read_packed_admission_verification(path: str | Path) -> PackedAdmissionVerification:
    raw = _read_strict_bytes(path, "packed admission verification")
    verification = _verification_from_payload(_parse_json(raw, "packed admission verification"))
    if raw != _canonical_bytes(asdict(verification)):
        raise ValueError("packed admission verification is not canonical JSON")
    return verification
