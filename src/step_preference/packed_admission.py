"""Strict schema-1 packed-admission artifacts and capacity witnesses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence
from types import SimpleNamespace
import subprocess

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
    if (
        not isinstance(value, str)
        or not value
        or not Path(value).is_absolute()
        or str(Path(value).resolve(strict=False)) != value
    ):
        raise ValueError(f"{name} must be a resolved non-empty absolute path")
    return value


def _verification_manifest_path(path: str | Path) -> str:
    """Identify a rejected path even when a symlink loop prevents resolution."""
    try:
        return str(Path(path).resolve(strict=False))
    except (OSError, RuntimeError):
        return os.path.abspath(path)


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
        if self.latent_size != 8:
            raise ValueError("latent_size must be 8")
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
        if self.status == "fail" and self.manifest_sha256 is None:
            if (
                not isinstance(self.manifest_path, str) or not self.manifest_path
                or not Path(self.manifest_path).is_absolute()
                or _verification_manifest_path(self.manifest_path) != self.manifest_path
            ):
                raise ValueError("manifest_path must be a resolved or normalized rejected absolute path")
        else:
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
        else:
            hashes = (
                self.manifest_sha256,
                self.clean_admissible_sha256,
                self.packed_retained_sha256,
                self.negative_sha256,
                self.negative_manifest_sha256,
            )
            saw_null_hash = False
            for value in hashes:
                if value is None:
                    saw_null_hash = True
                elif saw_null_hash:
                    raise ValueError(
                        "failing verification hashes must form an authenticated prefix"
                    )
            if (
                not self.errors
                or self.verified_record_count
                or self.retained_source_indices
                or self.filtered_source_indices
            ):
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
    """Classify unique measurements into source-index-sorted partitions."""
    _require_int("max_packed_length", max_packed_length, minimum=1)
    rows = tuple(measurements)
    if not all(type(item) is PackedAdmissionMeasurement for item in rows):
        raise ValueError("measurements must contain PackedAdmissionMeasurement values")
    source_indices = tuple(item.source_index for item in rows)
    sample_ids = tuple(item.sample_id for item in rows)
    if len(source_indices) != len(set(source_indices)) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("measurements must have unique source indices and sample IDs")
    retained, filtered = [], []
    for item in sorted(rows, key=lambda value: value.source_index):
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
    ties = []
    for item in tie_payload:
        entry = _require_exact_fields(item, PackedTieClass, "packed tie class")
        if not isinstance(entry["member_source_indices"], list):
            raise ValueError("member_source_indices must be an array")
        ties.append(PackedTieClass(**{**entry, "member_source_indices": tuple(entry["member_source_indices"])}))
    for name in ("pareto_source_indices", "final_probe_source_indices"):
        if not isinstance(body[name], list):
            raise ValueError(f"{name} must be an array")
    return PackedBoundaryWitnesses(
        tie_classes=tuple(ties), pareto_source_indices=tuple(body["pareto_source_indices"]),
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


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _authenticate(raw: bytes, expected: str, label: str) -> str:
    _require_sha256(label, expected)
    if _sha(raw) != expected:
        raise ValueError(f"{label} SHA-256 does not match")
    return expected


def _strict_json(path, label, decoder, expected=None):
    raw = _read_strict_bytes(path, label)
    payload = _parse_json(raw, label)
    decoded = decoder(payload)
    if raw != _canonical_bytes(payload):
        raise ValueError(f"{label} is not canonical JSON")
    if expected is not None:
        _authenticate(raw, expected, label)
    return decoded, raw


def _directory(path, label) -> Path:
    target = Path(path)
    if target.is_symlink() or not target.is_dir():
        raise ValueError(f"{label} must be a regular directory")
    return target.resolve(strict=True)


def _git_identity() -> str:
    def run(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True)
        if result.returncode:
            raise ValueError("packed admission requires a clean tracked worktree")
        return result.stdout.strip()
    commit = run("rev-parse", "HEAD")
    run("diff", "--quiet")
    run("diff", "--cached", "--quiet")
    status = run("status", "--porcelain=v1", "--untracked-files=all", "--ignored=no")
    for line in status.splitlines():
        path = line[3:]
        if not (line.startswith("?? ") and "__pycache__" in Path(path).parts and path.endswith(".pyc")):
            raise ValueError("packed admission requires a clean tracked worktree and no untracked executable inputs")
    return commit


@dataclass(frozen=True)
class SubsetSourceMap:
    schema_version: int
    kind: str
    original_source_sha256: str
    subset_sha256: str
    original_source_indices: tuple[int, ...]
    sample_ids: tuple[str, ...]

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1 or self.kind != "stage2b-subset-source-map":
            raise ValueError("invalid subset source map schema or kind")
        _require_sha256("original source", self.original_source_sha256)
        _require_sha256("subset", self.subset_sha256)
        _require_tuple("original_source_indices", self.original_source_indices)
        _require_tuple("sample_ids", self.sample_ids)
        for index in self.original_source_indices:
            _require_int("original source index", index)
        if tuple(sorted(set(self.original_source_indices))) != self.original_source_indices:
            raise ValueError("source map indices must be unique and increasing")
        if (not self.sample_ids or len(self.sample_ids) != len(self.original_source_indices)
                or any(not isinstance(value, str) or not value for value in self.sample_ids)
                or len(set(self.sample_ids)) != len(self.sample_ids)):
            raise ValueError("source map sample IDs must be unique nonempty strings matching indices")


def _source_map_from_payload(payload):
    body = _require_exact_fields(payload, SubsetSourceMap, "subset source map")
    for name in ("original_source_indices", "sample_ids"):
        if not isinstance(body[name], list):
            raise ValueError(f"{name} must be an array")
    return SubsetSourceMap(**{
        **body,
        "original_source_indices": tuple(body["original_source_indices"]),
        "sample_ids": tuple(body["sample_ids"]),
    })


def _source_rows(raw: bytes):
    rows = []
    # Physical original JSONL ordinals, never post-parse or post-filter ordinals.
    for index, line in enumerate(raw.splitlines()):
        value = _parse_json(line, f"source row {index}")
        if not isinstance(value, dict):
            raise ValueError(f"source row {index} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError("source JSONL must be nonempty")
    return tuple(rows)


def _negative_inputs(*, checkpoint_path, data_path, negative_path, negative_manifest):
    from step_preference import negatives
    raw_source = _read_strict_bytes(data_path, "clean source")
    raw_negative = _read_strict_bytes(negative_path, "negative")
    rows = _source_rows(raw_source)
    store = _parse_negative_store(raw_negative)
    manifest, raw_manifest = _strict_json(
        negative_manifest, "negative manifest",
        lambda body: negatives._manifest_from_bytes(_canonical_bytes(body)),
    )
    checkpoint = _directory(checkpoint_path, "checkpoint")
    if manifest.stage2a_checkpoint_path != str(checkpoint):
        raise ValueError("negative checkpoint identity does not match")
    _authenticate(raw_source, manifest.source_sha256, "negative clean source")
    _authenticate(raw_negative, manifest.negative_sha256, "negative")
    for entry in manifest.checkpoint_files:
        root = _directory(entry.load_root, "checkpoint load root")
        if str(root) != entry.load_root:
            raise ValueError("checkpoint load root must be resolved")
        path = root / entry.relative_path
        raw = _read_strict_bytes(path, "checkpoint file")
        if len(raw) != entry.size:
            raise ValueError("checkpoint file size does not match")
        _authenticate(raw, entry.sha256, "checkpoint file")
    inventory, tokenizer_identity = negatives.inspect_stage2a_checkpoint_identity(checkpoint)
    if (tuple(sorted(inventory, key=lambda item: (item.load_root, item.relative_path)))
            != tuple(sorted(manifest.checkpoint_files, key=lambda item: (item.load_root, item.relative_path)))
            or tokenizer_identity != manifest.tokenizer_config_identity):
        raise ValueError("checkpoint inventory or tokenizer identity does not match")
    if manifest.latent_size != 8 or manifest.assistant_plan_rendering != "adjacent-text-space-lf-v1":
        raise ValueError("negative configuration does not match")
    expected_settings = dict(
        attention_implementation="sdpa",
        vision_attention_implementation="flash_attention_2",
        device="cuda:0", max_new_tokens=32, min_tokens=2, torch_dtype="bfloat16",
    )
    if manifest.generation_settings_dict != expected_settings:
        raise ValueError("negative generation settings do not match packed Stage 2B")
    if manifest.selected_row_indices != tuple(range(len(rows))):
        raise ValueError("negative selected row indices must cover the clean source")
    return manifest, store, rows, (raw_source, raw_negative, raw_manifest)


def _parse_negative_store(raw_negative):
    from step_preference import negatives
    negative_rows = []
    for index, line in enumerate(raw_negative.splitlines()):
        payload = _parse_json(line, f"negative row {index}")
        negative_rows.append(negatives._row_from_payload(payload))
        if line + b"\n" != _canonical_bytes(payload):
            raise ValueError("negative rows must be canonical JSON")
    if raw_negative != b"".join(_canonical_bytes(asdict(row)) for row in negative_rows):
        raise ValueError("negative JSONL must end with LF")
    return negatives._negative_store(negative_rows)


def reconstruct_natural_packed_records(
    *, checkpoint_path, data_path, image_root, negative_path, negative_manifest,
    original_source_indices, max_packed_length, processor=None, model_config=None,
) -> tuple[PackedAdmissionMeasurement, ...]:
    """Measure unchanged natural plans through the production collator/planner."""
    from stage2.checkpointing import resolve_latent_token_ids
    from stage2.data import Stage2DataCollator, prepare_stage2_example
    from step_preference.packing import build_packed_tree_plan
    from transformers import AutoConfig, AutoProcessor

    if type(max_packed_length) is not int or max_packed_length != 4608:
        raise ValueError("max_packed_length must be 4608")
    _, store, rows, before = _negative_inputs(
        checkpoint_path=checkpoint_path, data_path=data_path,
        negative_path=negative_path, negative_manifest=negative_manifest,
    )
    images = _directory(image_root, "image root")
    _validate_original_image_inputs(rows, images)
    if len(rows) != len(original_source_indices):
        raise ValueError("source-map cardinality does not match reconstruction")
    prepared = tuple(
        prepare_stage2_example(
            row, image_root=images, latent_size=8, require_reasoning_units=True,
            assistant_plan_rendering="adjacent-text-space-lf-v1",
        )
        for row in rows
    )
    keys = tuple((example.sample_id, step.step_id) for example in prepared for step in example.reasoning_steps)
    if len(keys) != len(set(keys)) or set(store) != set(keys):
        raise ValueError("missing or unexpected negative join keys")
    _git_identity()
    if processor is None:
        processor = AutoProcessor.from_pretrained(str(checkpoint_path), local_files_only=True)
    if model_config is None:
        model_config = AutoConfig.from_pretrained(str(checkpoint_path), local_files_only=True)
    if getattr(model_config, "latent_size", None) != 8 or getattr(model_config, "supervise_latent_boundaries", None) is not True:
        raise ValueError("checkpoint latent configuration does not match")
    token_ids = resolve_latent_token_ids(processor.tokenizer, model_config)
    collator = Stage2DataCollator(
        processor, token_ids, 8, True,
        step_preference_enabled=True, negative_store=store,
    )
    measurements = []
    for source_index, example in zip(original_source_indices, prepared):
        batch = collator([example])
        plan = build_packed_tree_plan(
            clean_input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            reasoning_spans=batch["reasoning_step_spans"],
            joined_steps=batch["step_preference"]["steps"],
        )
        measurements.append(PackedAdmissionMeasurement(
            source_index=source_index, sample_id=example.sample_id,
            clean_length=int(plan.root_length),
            packed_unpadded_length=int(plan.unpadded_length),
            packed_padded_length=int(plan.padded_length),
            vision_token_count=int((batch["input_ids"] == model_config.image_token_id).sum()),
            image_count=len(example.original_image_paths),
            reasoning_block_count=len(batch["reasoning_step_spans"]),
            branch_count=len(plan.branches),
            scored_token_count=int(plan.selected_positions().numel()),
        ))
    after = _negative_inputs(
        checkpoint_path=checkpoint_path, data_path=data_path,
        negative_path=negative_path, negative_manifest=negative_manifest,
    )[3]
    if after != before:
        raise ValueError("negative inputs changed during reconstruction")
    return tuple(measurements)


@dataclass(frozen=True)
class _CleanSourceChain:
    original_source_indices: tuple[int, ...]
    sample_ids: tuple[str, ...]
    original_count: int
    clean_admissible_count: int
    clean_bytes: bytes
    bound_files: tuple[tuple[str, bytes], ...]
    identity_values: Mapping[str, object]
    processor: object
    images: tuple


def validate_clean_source_chain(args) -> _CleanSourceChain:
    """Authenticate mapped original ordinals and the original schema-2 audit."""
    from stage2 import length_manifest
    from step_preference.spans import canonical_sample_id
    from transformers import AutoProcessor
    if type(args.max_packed_sequence_length) is not int or args.max_packed_sequence_length != 4608:
        raise ValueError("max_packed_sequence_length must be 4608")
    original = _read_strict_bytes(args.original_data_path, "original source")
    clean = _read_strict_bytes(args.clean_data_path, "clean source")
    manifest, length_raw = _strict_json(args.clean_length_manifest, "clean length manifest", length_manifest._manifest_from_payload, args.clean_length_manifest_sha256)
    mapping, map_raw = _strict_json(args.source_map_path, "source map", _source_map_from_payload, args.source_map_sha256)
    _authenticate(original, manifest.data_sha256, "clean length original source")
    _authenticate(original, mapping.original_source_sha256, "source map original source")
    _authenticate(clean, mapping.subset_sha256, "source map clean source")
    rows = _source_rows(original)
    if len(rows) != len(manifest.lengths):
        raise ValueError("clean length manifest cardinality does not match original source")
    if (manifest.max_sequence_length, manifest.latent_size, manifest.supervise_latent_boundaries, manifest.assistant_plan_rendering) != (4608, 8, True, "adjacent-text-space-lf-v1"):
        raise ValueError("clean length manifest configuration must be 4608/8/adjacent-text-space-lf-v1")
    retained_indices = set(manifest.retained_indices)
    if any(index not in retained_indices for index in mapping.original_source_indices):
        raise ValueError("source map indices must belong to clean-admissible retained indices")
    lines = original.splitlines()
    expected_clean = b"".join(lines[index] + b"\n" for index in mapping.original_source_indices)
    if clean != expected_clean:
        raise ValueError("clean source bytes do not match mapped original lines")
    if mapping.sample_ids != tuple(canonical_sample_id(rows[index]) for index in mapping.original_source_indices):
        raise ValueError("source map sample IDs do not match original rows")
    images = _directory(args.image_root, "image root")
    _validate_original_image_inputs(rows, images)
    checkpoint = _directory(args.checkpoint_path, "checkpoint")
    negative, _, _, negative_bytes = _negative_inputs(
        checkpoint_path=checkpoint, data_path=args.clean_data_path,
        negative_path=args.negative_path, negative_manifest=args.negative_manifest,
    )
    commit = _git_identity()
    _validate_images(manifest.images)
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    # Existing schema-2 semantics check processor and every original image identity.
    length_manifest.validate_stage2_length_manifest(
        path=args.clean_length_manifest,
        expected_sha256=args.clean_length_manifest_sha256,
        data_path=args.original_data_path, image_root=images, processor=processor,
        latent_size=8, supervise_latent_boundaries=True, max_sequence_length=4608,
        assistant_plan_rendering="adjacent-text-space-lf-v1",
    )
    _validate_images(manifest.images)
    file_inputs = (
        ("original_data", args.original_data_path, original), ("clean_data", args.clean_data_path, clean),
        ("source_map", args.source_map_path, map_raw), ("clean_length_manifest", args.clean_length_manifest, length_raw),
        ("negative", args.negative_path, negative_bytes[1]), ("negative_manifest", args.negative_manifest, negative_bytes[2]),
    )
    values = {}
    bound = []
    for name, path, raw in file_inputs:
        if _read_strict_bytes(path, name) != raw:
            raise ValueError(f"{name} changed during validation")
        resolved = str(Path(path).resolve(strict=True))
        values[f"{name}_path"] = resolved
        values[f"{name}_sha256"] = _sha(raw)
        bound.append((resolved, raw))
    values.update(
        image_root_path=str(images), checkpoint_path=str(checkpoint),
        checkpoint_files=tuple(sorted(
            negative.checkpoint_files, key=lambda entry: (entry.load_root, entry.relative_path)
        )),
        tokenizer_config_identity=negative.tokenizer_config_identity,
        assistant_plan_rendering=manifest.assistant_plan_rendering,
        attention_implementation="sdpa", vision_attention_implementation="flash_attention_2",
        latent_size=8, clean_model_limit=8192, clean_prefilter_limit=4608,
        max_packed_sequence_length=4608, git_commit=commit, tracked_worktree_clean=True,
    )
    return _CleanSourceChain(
        mapping.original_source_indices, mapping.sample_ids, len(rows),
        len(manifest.retained_indices), clean, tuple(bound), values, processor, manifest.images,
    )


def _validate_original_image_inputs(rows, image_root):
    """Check supplied leaves before the existing resolver erases symlink identity."""
    for row in rows:
        raw_images = row.get("image_input")
        if isinstance(raw_images, str):
            raw_images = [raw_images]
        if not isinstance(raw_images, list) or not raw_images:
            raise ValueError("image_input must contain at least one original image")
        for raw_image in raw_images:
            if not isinstance(raw_image, str) or not raw_image:
                raise ValueError("original image path must be a non-empty string")
            candidate = Path(raw_image)
            if (
                candidate.is_absolute() or not candidate.parts
                or any(part in ("", ".", "..") for part in candidate.parts)
                or any(ord(char) < 32 or ord(char) == 127 for char in raw_image)
            ):
                raise ValueError("original image must use a safe relative path")
            if candidate.parts[0] == image_root.name:
                candidate = Path(*candidate.parts[1:])
            if not candidate.parts:
                raise ValueError("original image must use a safe relative path")
            # Parent symlinks remain legal. The production resolver below still
            # enforces containment after resolution; only the leaf is rejected.
            _read_strict_bytes(image_root / candidate, "original image")


def _validate_images(images):
    from stage2.length_manifest import _image_identity
    for image in dict.fromkeys(images):
        _read_strict_bytes(image.path, "original image")
        if _image_identity(image.path) != image:
            raise ValueError("original image identity changed during reconstruction")


def _reconstruct_chain(args, chain):
    rows = reconstruct_natural_packed_records(
        checkpoint_path=args.checkpoint_path, data_path=args.clean_data_path,
        image_root=args.image_root, negative_path=args.negative_path,
        negative_manifest=args.negative_manifest,
        original_source_indices=chain.original_source_indices,
        max_packed_length=args.max_packed_sequence_length, processor=chain.processor,
    )
    if tuple(row.sample_id for row in rows) != chain.sample_ids:
        raise ValueError("reconstructed sample IDs do not match source map")
    return rows


def _check_unchanged(chain):
    for path, raw in chain.bound_files:
        if _read_strict_bytes(path, "bound input") != raw:
            raise ValueError("bound input changed before publication")
    _validate_images(chain.images)
    for entry in chain.identity_values["checkpoint_files"]:
        raw = _read_strict_bytes(Path(entry.load_root) / entry.relative_path, "checkpoint file")
        if len(raw) != entry.size:
            raise ValueError("checkpoint file size changed before publication")
        _authenticate(raw, entry.sha256, "checkpoint file")
    if _git_identity() != chain.identity_values["git_commit"]:
        raise ValueError("git identity changed before publication")


def _retained_bytes(chain, indices):
    chosen = set(indices)
    return b"".join(
        line + b"\n"
        for index, line in zip(chain.original_source_indices, chain.clean_bytes.splitlines())
        if index in chosen
    )


def _preflight_outputs(*paths):
    resolved = tuple(Path(path).resolve(strict=False) for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("output paths must be distinct")
    for path in paths:
        if os.path.lexists(path):
            raise ValueError(f"output path already exists: {path}")


def build_packed_admission(args) -> PackedAdmissionManifest:
    """Publish retained source first, then its exclusive admission manifest."""
    _preflight_outputs(args.retained_output_path, args.manifest_output_path)
    chain = validate_clean_source_chain(args)
    measurements = _reconstruct_chain(args, chain)
    retained, filtered = select_packed_admission(measurements, max_packed_length=args.max_packed_sequence_length)
    witnesses = build_boundary_witnesses(retained)
    indices = tuple(row.source_index for row in retained)
    raw = _retained_bytes(chain, indices)
    identity = PackedAdmissionIdentity(
        **chain.identity_values,
        retained_data_path=str(Path(args.retained_output_path).resolve(strict=False)),
        retained_data_sha256=_sha(raw),
    )
    result = PackedAdmissionManifest(
        schema_version=1, kind=PACKED_ADMISSION_MANIFEST_KIND, identity=identity,
        records=tuple(sorted((*retained, *filtered), key=lambda row: row.source_index)),
        retained_source_indices=indices,
        retained_sample_ids=tuple(row.sample_id for row in retained),
        filtered_source_indices=tuple(row.source_index for row in filtered),
        filtered_sample_ids=tuple(row.sample_id for row in filtered),
        boundary_witnesses=witnesses,
        counts=PackedAdmissionCounts(
            chain.original_count, chain.clean_admissible_count,
            len(measurements), len(retained), len(filtered),
        ),
    )
    _check_unchanged(chain)
    _write_exclusively(args.retained_output_path, raw, "packed retained source")
    _authenticate(_read_strict_bytes(args.retained_output_path, "packed retained source"), identity.retained_data_sha256, "packed retained source")
    # A second-write failure deliberately preserves the immutable orphan source.
    write_packed_admission_manifest(args.manifest_output_path, result)
    return result


def verify_packed_admission(args) -> PackedAdmissionVerification:
    """Recompute from authenticated inputs without producer calls or repairs."""
    _preflight_outputs(args.output_path)
    hashes = dict(
        manifest_sha256=None, clean_admissible_sha256=None,
        packed_retained_sha256=None, negative_sha256=None, negative_manifest_sha256=None,
    )
    errors = ()
    records, retained, filtered = (), (), ()
    try:
        manifest, raw = _strict_json(args.manifest_path, "packed admission manifest", _manifest_from_payload, args.manifest_sha256)
        hashes["manifest_sha256"] = _sha(raw)
        identity = manifest.identity
        for key, path, expected in (
            ("clean_admissible_sha256", identity.clean_data_path, identity.clean_data_sha256),
            ("packed_retained_sha256", identity.retained_data_path, identity.retained_data_sha256),
            ("negative_sha256", identity.negative_path, identity.negative_sha256),
            ("negative_manifest_sha256", identity.negative_manifest_path, identity.negative_manifest_sha256),
        ):
            current = _read_strict_bytes(path, key)
            if key in ("clean_admissible_sha256", "packed_retained_sha256"):
                _source_rows(current)
            elif key == "negative_sha256":
                _parse_negative_store(current)
            if key == "negative_manifest_sha256":
                from step_preference.negatives import _manifest_from_bytes
                _manifest_from_bytes(current)
                if _canonical_bytes(_parse_json(current, key)) != current:
                    raise ValueError("negative manifest is not canonical JSON")
            hashes[key] = _authenticate(current, expected, key)
        source_args = SimpleNamespace(
            original_data_path=identity.original_data_path,
            clean_data_path=identity.clean_data_path,
            clean_length_manifest=identity.clean_length_manifest_path,
            clean_length_manifest_sha256=identity.clean_length_manifest_sha256,
            image_root=identity.image_root_path, checkpoint_path=identity.checkpoint_path,
            source_map_path=identity.source_map_path, source_map_sha256=identity.source_map_sha256,
            negative_path=identity.negative_path, negative_manifest=identity.negative_manifest_path,
            max_packed_sequence_length=identity.max_packed_sequence_length,
        )
        if not identity.tracked_worktree_clean or _git_identity() != identity.git_commit:
            raise ValueError("git identity or tracked worktree does not match")
        chain = validate_clean_source_chain(source_args)
        for key, value in chain.identity_values.items():
            if getattr(identity, key) != value:
                raise ValueError(f"reconstructed identity does not match: {key}")
        measured = _reconstruct_chain(source_args, chain)
        # Recompute classifications and complete expectations from measurements.
        # No manifest record/partition value is an input to this decision.
        retained, filtered = select_packed_admission(measured, max_packed_length=4608)
        records = tuple(sorted((*retained, *filtered), key=lambda row: row.source_index))
        expected_identity = PackedAdmissionIdentity(
            **chain.identity_values,
            retained_data_path=identity.retained_data_path,
            retained_data_sha256=_sha(_retained_bytes(chain, tuple(row.source_index for row in retained))),
        )
        expected = PackedAdmissionManifest(
            1, PACKED_ADMISSION_MANIFEST_KIND, expected_identity, records,
            tuple(row.source_index for row in retained), tuple(row.sample_id for row in retained),
            tuple(row.source_index for row in filtered), tuple(row.sample_id for row in filtered),
            build_boundary_witnesses(retained),
            PackedAdmissionCounts(chain.original_count, chain.clean_admissible_count, len(records), len(retained), len(filtered)),
        )
        if manifest != expected:
            raise ValueError("admission manifest does not match independently reconstructed source records")
        if _read_strict_bytes(identity.retained_data_path, "packed retained source") != _retained_bytes(chain, expected.retained_source_indices):
            raise ValueError("retained source bytes do not match reconstructed partition")
        _check_unchanged(chain)
        if _read_strict_bytes(args.manifest_path, "packed admission manifest") != raw:
            raise ValueError("admission manifest changed during verification")
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors = (str(error) or type(error).__name__,)
    verification = PackedAdmissionVerification(
        schema_version=1, kind=PACKED_ADMISSION_VERIFICATION_KIND,
        status="fail" if errors else "pass",
        manifest_path=_verification_manifest_path(args.manifest_path),
        **hashes,
        verified_record_count=0 if errors else len(records),
        retained_source_indices=() if errors else tuple(row.source_index for row in retained),
        filtered_source_indices=() if errors else tuple(row.source_index for row in filtered),
        errors=errors,
    )
    write_packed_admission_verification(args.output_path, verification)
    return verification


def main(argv=None) -> int:
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    for name in (
        "original-data-path", "clean-data-path", "clean-length-manifest",
        "clean-length-manifest-sha256", "image-root", "checkpoint-path",
        "source-map-path", "source-map-sha256", "negative-path", "negative-manifest",
        "retained-output-path", "manifest-output-path",
    ):
        build.add_argument(f"--{name}", required=True)
    build.add_argument("--max-packed-sequence-length", required=True, type=int)
    verify = commands.add_parser("verify")
    for name in ("manifest-path", "manifest-sha256", "output-path"):
        verify.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            build_packed_admission(args)
        elif verify_packed_admission(args).status != "pass":
            return 2
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
