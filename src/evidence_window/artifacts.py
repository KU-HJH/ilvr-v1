"""Immutable, identity-checked Stage-2 reference contribution artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch

from .stage2_layout import Stage2InterleavedLayout, select_stage2_edges


_ARTIFACT_FORMAT = "ilvr-stage2-reference"


@dataclass(frozen=True)
class ArtifactIdentity:
    """Every input or scoring choice that changes a reference target."""

    format_version: int
    dataset_sha256: str
    checkpoint_id: str
    checkpoint_witness: str
    tokenizer_id: str
    tokenizer_special_tokens: tuple[int, ...]
    parser_version: str
    contribution_reduction: str
    intervention: str
    attention_backend: str
    max_evidence_edges_per_window: int
    model_config_id: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tokenizer_special_tokens"] = list(self.tokenizer_special_tokens)
        return value

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ArtifactIdentity":
        if type(raw) is not dict:
            raise ValueError("artifact identity must be an object")
        expected = {field.name for field in fields(cls)}
        if set(raw) != expected:
            missing = expected - set(raw)
            extra = set(raw) - expected
            detail = ", ".join(sorted(missing | extra))
            raise ValueError(f"artifact identity schema mismatch: {detail}")
        values = dict(raw)
        special = values["tokenizer_special_tokens"]
        if type(special) is not list:
            raise ValueError("tokenizer_special_tokens must be a list")
        values["tokenizer_special_tokens"] = tuple(special)
        identity = cls(**values)
        _validate_identity(identity)
        return identity


@dataclass(frozen=True)
class ReferenceSampleTargets:
    """One source row's unpadded evidence-window reference targets."""

    contributions: torch.Tensor
    pair_mask: torch.Tensor
    hidden_residuals: torch.Tensor | None = None


@dataclass(frozen=True)
class ReferenceBatchTargets:
    """Reference targets padded into one current policy batch layout."""

    sample_indices: tuple[int, ...]
    contributions: torch.Tensor
    pair_mask: torch.Tensor
    hidden_residuals: torch.Tensor | None = None


@dataclass(frozen=True)
class ReferenceShardMetadata:
    """Small shard publication witness; no tensor payload is reread by rank zero."""

    filename: str
    rank: int
    sample_indices: tuple[int, ...]
    record_count: int
    has_hidden_residuals: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "rank": self.rank,
            "sample_indices": list(self.sample_indices),
            "record_count": self.record_count,
            "has_hidden_residuals": self.has_hidden_residuals,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReferenceShardMetadata":
        expected = {
            "filename",
            "rank",
            "sample_indices",
            "record_count",
            "has_hidden_residuals",
        }
        if type(raw) is not dict or set(raw) != expected:
            raise ValueError("reference shard metadata schema mismatch")
        sample_indices = raw["sample_indices"]
        if type(sample_indices) is not list:
            raise ValueError("reference shard sample_indices must be a list")
        metadata = cls(
            filename=raw["filename"],
            rank=raw["rank"],
            sample_indices=tuple(sample_indices),
            record_count=raw["record_count"],
            has_hidden_residuals=raw["has_hidden_residuals"],
        )
        _validate_shard_metadata(metadata)
        return metadata


def write_reference_shard(
    artifact_dir: str | Path,
    identity: ArtifactIdentity,
    targets: Mapping[int, ReferenceSampleTargets],
    *,
    rank: int = 0,
    publish_manifest: bool = True,
) -> ReferenceShardMetadata:
    """Write one rank's primitive/tensor shard exactly once."""
    _validate_identity(identity)
    _require_nonnegative_int(rank, "rank")
    if not isinstance(targets, Mapping):
        raise TypeError("targets must map sample indices to reference targets")

    normalized: list[tuple[int, ReferenceSampleTargets]] = []
    for sample_index, target in targets.items():
        _require_nonnegative_int(sample_index, "sample index")
        _validate_sample_targets(target)
        normalized.append((sample_index, target))
    normalized.sort(key=lambda item: item[0])
    sample_indices = tuple(index for index, _ in normalized)
    if len(set(sample_indices)) != len(sample_indices):
        raise ValueError("duplicate sample index in reference shard")

    residual_flags = {target.hidden_residuals is not None for _, target in normalized}
    if len(residual_flags) > 1:
        raise ValueError("a reference shard cannot mix present and absent hidden residuals")
    has_hidden = residual_flags == {True}

    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"shard-rank-{rank:05d}.pt"
    destination = directory / filename
    records = [
        {
            "sample_index": sample_index,
            "contributions": _sealed_cpu_tensor(target.contributions),
            "pair_mask": _sealed_cpu_tensor(target.pair_mask),
            "hidden_residuals": (
                None
                if target.hidden_residuals is None
                else _sealed_cpu_tensor(target.hidden_residuals)
            ),
        }
        for sample_index, target in normalized
    ]
    payload = {
        "format": _ARTIFACT_FORMAT,
        "identity": identity.to_dict(),
        "rank": rank,
        "sample_indices": list(sample_indices),
        "records": records,
    }
    _write_torch_exclusive(destination, payload)
    metadata = ReferenceShardMetadata(
        filename=filename,
        rank=rank,
        sample_indices=sample_indices,
        record_count=len(records),
        has_hidden_residuals=has_hidden,
    )
    _write_json_exclusive(directory / f"shard-rank-{rank:05d}.metadata.json", metadata.to_dict())
    if publish_manifest:
        publish_reference_manifest(directory, identity, [metadata])
    return metadata


def publish_reference_manifest(
    artifact_dir: str | Path,
    identity: ArtifactIdentity,
    shards: Sequence[ReferenceShardMetadata],
) -> Path:
    """Publish a manifest solely from the producer-returned small metadata."""
    _validate_identity(identity)
    normalized = tuple(shards)
    if not normalized:
        raise ValueError("at least one reference shard is required")
    for shard in normalized:
        _validate_shard_metadata(shard)
    _validate_manifest_shards(normalized)
    owner: dict[int, str] = {}
    for shard in normalized:
        for sample_index in shard.sample_indices:
            if sample_index in owner:
                raise ValueError(
                    f"duplicate reference sample {sample_index} across shards "
                    f"{owner[sample_index]} and {shard.filename}"
                )
            owner[sample_index] = shard.filename
    ordered = sorted(normalized, key=lambda item: item.rank)
    manifest = {
        "format": _ARTIFACT_FORMAT,
        "identity": identity.to_dict(),
        "shards": [shard.to_dict() for shard in ordered],
    }
    destination = Path(artifact_dir) / "manifest.json"
    _write_json_exclusive(destination, manifest)
    return destination


def read_reference_shard_metadata(path: str | Path) -> ReferenceShardMetadata:
    """Read only a producer's small rank metadata sidecar."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return ReferenceShardMetadata.from_dict(raw)


class PrecomputedReferenceProvider:
    """Strict immutable lookup keyed by deterministic dataset source-row index."""

    def __init__(
        self,
        artifact_dir: str | Path,
        expected_identity: ArtifactIdentity,
    ) -> None:
        _validate_identity(expected_identity)
        self.artifact_dir = Path(artifact_dir)
        manifest_path = self.artifact_dir / "manifest.json"
        try:
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
        except FileNotFoundError as exc:
            raise ValueError("missing reference manifest") from exc
        if type(manifest) is not dict or set(manifest) != {"format", "identity", "shards"}:
            raise ValueError("reference manifest schema mismatch")
        if type(manifest["format"]) is not str or manifest["format"] != _ARTIFACT_FORMAT:
            raise ValueError("reference manifest format mismatch")
        actual_identity = ArtifactIdentity.from_dict(manifest["identity"])
        _require_matching_identity(actual_identity, expected_identity)
        raw_shards = manifest["shards"]
        if type(raw_shards) is not list or not raw_shards:
            raise ValueError("reference manifest shards must be a nonempty list")
        metadata = [ReferenceShardMetadata.from_dict(raw) for raw in raw_shards]
        _validate_manifest_shards(metadata)
        if [item.rank for item in metadata] != sorted(item.rank for item in metadata):
            raise ValueError("reference manifest shard order must be rank-sorted")
        self.identity = actual_identity
        self._targets: dict[int, ReferenceSampleTargets] = {}
        for shard in metadata:
            self._load_shard(shard)

    def _load_shard(self, metadata: ReferenceShardMetadata) -> None:
        path = self.artifact_dir / metadata.filename
        if not path.is_file():
            raise ValueError(f"missing shard: {metadata.filename}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"reference shard schema invalid: {metadata.filename}") from exc
        expected_keys = {"format", "identity", "rank", "sample_indices", "records"}
        if type(payload) is not dict or set(payload) != expected_keys:
            raise ValueError(f"reference shard schema invalid: {metadata.filename}")
        if type(payload["format"]) is not str or payload["format"] != _ARTIFACT_FORMAT:
            raise ValueError(f"reference shard format mismatch: {metadata.filename}")
        shard_identity = ArtifactIdentity.from_dict(payload["identity"])
        _require_matching_identity(shard_identity, self.identity)
        _require_nonnegative_int(payload["rank"], "reference shard rank")
        if payload["rank"] != metadata.rank:
            raise ValueError(f"reference shard rank mismatch: {metadata.filename}")
        if type(payload["sample_indices"]) is not list:
            raise ValueError(f"reference shard sample index layout mismatch: {metadata.filename}")
        for sample_index in payload["sample_indices"]:
            _require_nonnegative_int(sample_index, "sample index")
        if payload["sample_indices"] != list(metadata.sample_indices):
            raise ValueError(f"reference shard sample index layout mismatch: {metadata.filename}")
        records = payload["records"]
        if type(records) is not list or len(records) != metadata.record_count:
            raise ValueError(f"reference shard record layout mismatch: {metadata.filename}")
        seen: list[int] = []
        residual_flags: set[bool] = set()
        for record in records:
            expected_record_keys = {
                "sample_index",
                "contributions",
                "pair_mask",
                "hidden_residuals",
            }
            if type(record) is not dict or set(record) != expected_record_keys:
                raise ValueError(f"reference shard record schema invalid: {metadata.filename}")
            sample_index = record["sample_index"]
            _require_nonnegative_int(sample_index, "sample index")
            target = ReferenceSampleTargets(
                record["contributions"],
                record["pair_mask"],
                record["hidden_residuals"],
            )
            _validate_sample_targets(target)
            if sample_index in self._targets:
                raise ValueError(f"duplicate reference sample {sample_index} across shards")
            self._targets[sample_index] = target
            seen.append(sample_index)
            residual_flags.add(target.hidden_residuals is not None)
        if seen != list(metadata.sample_indices):
            raise ValueError(f"reference shard record order mismatch: {metadata.filename}")
        actual_hidden = residual_flags == {True}
        if len(residual_flags) > 1 or actual_hidden != metadata.has_hidden_residuals:
            raise ValueError(f"reference shard residual layout mismatch: {metadata.filename}")

    def get_batch(
        self,
        sample_indices: Sequence[int] | torch.Tensor,
        layout: Stage2InterleavedLayout,
        *,
        require_hidden_residuals: bool = False,
    ) -> ReferenceBatchTargets:
        """Look up and repad rows, validating the exact selected edge layout."""
        if not isinstance(layout, Stage2InterleavedLayout):
            raise TypeError("layout must be a Stage2InterleavedLayout")
        if isinstance(sample_indices, torch.Tensor):
            if sample_indices.ndim != 1:
                raise ValueError("sample indices must have shape [B]")
            requested = tuple(sample_indices.detach().cpu().tolist())
        else:
            requested = tuple(sample_indices)
        batch_size, evidence_count, window_count = layout.causal_pair_mask.shape
        if len(requested) != batch_size:
            raise ValueError("sample indices must have one entry per batch row")
        for sample_index in requested:
            _require_nonnegative_int(sample_index, "sample index")
            if sample_index not in self._targets:
                raise KeyError(f"reference artifact has no sample {sample_index}")

        expected_pair = torch.zeros_like(layout.causal_pair_mask)
        for edge in select_stage2_edges(
            layout, self.identity.max_evidence_edges_per_window
        ):
            expected_pair[edge.batch_index, edge.evidence_index, edge.window_index] = True

        device = layout.causal_pair_mask.device
        contributions = torch.zeros(
            (batch_size, evidence_count, window_count), dtype=torch.float32, device=device
        )
        hidden_size: int | None = None
        for sample_index in requested:
            residual = self._targets[sample_index].hidden_residuals
            if residual is not None:
                if hidden_size is None:
                    hidden_size = residual.shape[-1]
                elif hidden_size != residual.shape[-1]:
                    raise ValueError("reference hidden residual width differs across samples")
            elif require_hidden_residuals:
                raise ValueError(f"hidden residual missing for sample {sample_index}")
        hidden = None
        if hidden_size is not None:
            if any(self._targets[index].hidden_residuals is None for index in requested):
                raise ValueError("reference batch mixes present and absent hidden residuals")
            hidden = torch.zeros(
                (batch_size, evidence_count, window_count, hidden_size),
                dtype=torch.float32,
                device=device,
            )

        for batch_index, sample_index in enumerate(requested):
            target = self._targets[sample_index]
            current_evidence = int(layout.evidence_valid[batch_index].sum().item())
            current_windows = int(layout.window_valid[batch_index].sum().item())
            if target.contributions.shape != (current_evidence, current_windows):
                raise ValueError(
                    f"reference layout shape mismatch for sample {sample_index}: "
                    f"expected {(current_evidence, current_windows)}, "
                    f"found {tuple(target.contributions.shape)}"
                )
            current_pair = expected_pair[
                batch_index, :current_evidence, :current_windows
            ].cpu()
            if not torch.equal(target.pair_mask.cpu(), current_pair):
                raise ValueError(f"pair_mask mismatch for sample {sample_index}")
            contributions[
                batch_index, :current_evidence, :current_windows
            ] = target.contributions.to(device=device, dtype=torch.float32)
            if hidden is not None:
                assert target.hidden_residuals is not None
                hidden[
                    batch_index, :current_evidence, :current_windows
                ] = target.hidden_residuals.to(device=device, dtype=torch.float32)

        return ReferenceBatchTargets(
            sample_indices=requested,
            contributions=contributions,
            pair_mask=expected_pair,
            hidden_residuals=hidden,
        )


def _validate_identity(identity: ArtifactIdentity) -> None:
    if not isinstance(identity, ArtifactIdentity):
        raise TypeError("identity must be an ArtifactIdentity")
    if type(identity.format_version) is not int or identity.format_version != 1:
        raise ValueError("format_version must be integer 1")
    if (
        type(identity.dataset_sha256) is not str
        or len(identity.dataset_sha256) != 64
        or any(character not in "0123456789abcdef" for character in identity.dataset_sha256)
    ):
        raise ValueError("dataset_sha256 must be a lowercase SHA256 witness")
    for name in (
        "checkpoint_id",
        "checkpoint_witness",
        "tokenizer_id",
        "parser_version",
        "model_config_id",
    ):
        value = getattr(identity, name)
        if type(value) is not str or not value:
            raise ValueError(f"{name} must be a nonempty string")
    special = identity.tokenizer_special_tokens
    if (
        not isinstance(special, tuple)
        or not special
        or any(type(value) is not int for value in special)
        or len(set(special)) != len(special)
    ):
        raise ValueError("tokenizer_special_tokens must be unique integers")
    if identity.contribution_reduction not in {"mean", "sum"}:
        raise ValueError("contribution_reduction must be mean or sum")
    if identity.intervention != "window_attention_block":
        raise ValueError("intervention must be window_attention_block")
    if identity.attention_backend != "sdpa":
        raise ValueError("attention_backend must be sdpa")
    _require_nonnegative_int(
        identity.max_evidence_edges_per_window,
        "max_evidence_edges_per_window",
    )


def _require_matching_identity(
    actual: ArtifactIdentity, expected: ArtifactIdentity
) -> None:
    for field in fields(ArtifactIdentity):
        name = field.name
        if getattr(actual, name) != getattr(expected, name):
            raise ValueError(
                f"artifact identity mismatch for {name}: expected "
                f"{getattr(expected, name)!r}, found {getattr(actual, name)!r}"
            )


def _validate_sample_targets(target: ReferenceSampleTargets) -> None:
    if not isinstance(target, ReferenceSampleTargets):
        raise TypeError("reference target must be a ReferenceSampleTargets")
    contributions = target.contributions
    pair_mask = target.pair_mask
    if not torch.is_tensor(contributions) or contributions.ndim != 2:
        raise ValueError("contributions must have shape [J, K]")
    if not contributions.is_floating_point():
        raise ValueError("contributions must be floating point")
    if not torch.isfinite(contributions).all():
        raise ValueError("contributions must be finite")
    if not torch.is_tensor(pair_mask) or pair_mask.shape != contributions.shape:
        raise ValueError("pair_mask must match contributions shape")
    if pair_mask.dtype is not torch.bool:
        raise ValueError("pair_mask must have bool dtype")
    if pair_mask.device != contributions.device:
        raise ValueError("pair_mask and contributions must share a device")
    invalid_values = contributions.masked_select(~pair_mask)
    if invalid_values.numel() and torch.count_nonzero(invalid_values).item():
        raise ValueError("contributions outside pair_mask must be zero")
    hidden = target.hidden_residuals
    if hidden is None:
        return
    if (
        not torch.is_tensor(hidden)
        or hidden.ndim != 3
        or hidden.shape[:2] != contributions.shape
        or not hidden.is_floating_point()
    ):
        raise ValueError("hidden_residuals must have shape [J, K, H]")
    if hidden.device != contributions.device or not torch.isfinite(hidden).all():
        raise ValueError("hidden_residuals must be finite and share the target device")
    if hidden.shape[-1] <= 0:
        raise ValueError("hidden_residuals must have a positive hidden width")
    invalid_hidden = hidden.masked_select(~pair_mask.unsqueeze(-1))
    if invalid_hidden.numel() and torch.count_nonzero(invalid_hidden).item():
        raise ValueError("hidden_residuals outside pair_mask must be zero")


def _validate_shard_metadata(metadata: ReferenceShardMetadata) -> None:
    if not isinstance(metadata, ReferenceShardMetadata):
        raise TypeError("shards must contain ReferenceShardMetadata")
    if type(metadata.filename) is not str or not metadata.filename:
        raise ValueError("reference shard filename must be nonempty")
    if Path(metadata.filename).name != metadata.filename:
        raise ValueError("reference shard filename must be local")
    _require_nonnegative_int(metadata.rank, "reference shard rank")
    if type(metadata.sample_indices) is not tuple:
        raise ValueError("reference shard sample_indices must be a tuple")
    for index in metadata.sample_indices:
        _require_nonnegative_int(index, "sample index")
    if tuple(sorted(metadata.sample_indices)) != metadata.sample_indices:
        raise ValueError("reference shard sample indices must be sorted")
    if len(set(metadata.sample_indices)) != len(metadata.sample_indices):
        raise ValueError("duplicate sample index in reference shard metadata")
    _require_nonnegative_int(metadata.record_count, "reference shard record_count")
    if metadata.record_count != len(metadata.sample_indices):
        raise ValueError("reference shard record count mismatch")
    if type(metadata.has_hidden_residuals) is not bool:
        raise ValueError("reference shard residual flag must be boolean")


def _sealed_cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").contiguous().clone()


def _require_nonnegative_int(value: Any, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    _write_bytes_exclusive(path, encoded)


def _write_torch_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        _fsync_file(temporary)
        _publish_temp_no_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temp_no_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_temp_no_replace(temporary: Path, destination: Path) -> None:
    """Atomically expose one complete file while refusing existing names."""
    os.link(temporary, destination)
    _fsync_directory(destination.parent)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_manifest_shards(
    shards: Sequence[ReferenceShardMetadata],
) -> None:
    ranks = [shard.rank for shard in shards]
    filenames = [shard.filename for shard in shards]
    if len(set(ranks)) != len(ranks) or len(set(filenames)) != len(filenames):
        raise ValueError("duplicate reference shard rank or filename")


__all__ = [
    "ArtifactIdentity",
    "PrecomputedReferenceProvider",
    "ReferenceBatchTargets",
    "ReferenceSampleTargets",
    "ReferenceShardMetadata",
    "publish_reference_manifest",
    "read_reference_shard_metadata",
    "write_reference_shard",
]
