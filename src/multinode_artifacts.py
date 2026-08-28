"""Small metadata manifests for non-shared multi-node checkpoint trees."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Protocol


_MANIFEST_SCHEMA_VERSION = 1
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,127}$")
_RANKED_SHARD_PATTERNS = (
    re.compile(
        r"(?:bf16_)?zero_pp_rank_(0|[1-9][0-9]*)_mp_rank_[0-9]+_"
        r"(?:optim|model)_states\.pt\Z"
    ),
    re.compile(r"rng_state_(0|[1-9][0-9]*)\.pth\Z"),
)
_MANIFEST_IDENTITY_FIELDS = (
    "stage",
    "tuning_method",
    "zero_stage",
    "deepspeed_config_digest",
    "world_size",
    "rank_mapping",
    "commit",
    "run_id",
)
_RESUME_IDENTITY_FIELDS = (
    *_MANIFEST_IDENTITY_FIELDS[:-2],
    "global_step",
    "optimizer",
    "scheduler",
    "rng_ranks",
    "stage_sidecar",
    "commit",
    "run_id",
)
_LOCAL_REPOSITORY = Path("/root/workspace/ILVR")
_REMOTE_HOST_REPOSITORY = Path("/home/korea/Workspace/hjh/workspace/ILVR")
_REMOTE_HOST = "korea@10.1.102.41"
_RSYNC_SSH = "/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=8"
_MAX_COMMON_METADATA_BYTES = 4 * 1024 * 1024
_COMMON_METADATA_SUFFIXES = frozenset({".json", ".txt", ".yaml", ".yml", ".md"})
_COMMON_STATE_NAMES = frozenset({"latest", "scheduler.pt", "scaler.pt", "training_args.bin"})


def _process_rank(path: Path) -> int | None:
    for pattern in _RANKED_SHARD_PATTERNS:
        match = pattern.fullmatch(path.name)
        if match is not None:
            return int(match.group(1))
    return None


def _safe_relative_path(path: Path) -> Path:
    candidate = Path(path)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or any(part in ("", ".", "..") for part in candidate.parts)
        or any("\n" in part or "\r" in part or "\0" in part for part in candidate.parts)
    ):
        raise ValueError(f"artifact relative path is invalid: {candidate}")
    return candidate


def _validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("artifact run ID is invalid")
    return run_id


@dataclass(frozen=True)
class ArtifactEntry:
    relative_path: Path
    byte_size: int
    node_rank: int

    @property
    def process_rank(self) -> int | None:
        """Return ownership encoded by a canonical DeepSpeed shard filename."""
        return _process_rank(self.relative_path)

    @property
    def is_common_metadata(self) -> bool:
        return (
            self.process_rank is None
            and self.byte_size <= _MAX_COMMON_METADATA_BYTES
            and (
                self.relative_path.suffix.lower() in _COMMON_METADATA_SUFFIXES
                or self.relative_path.name in _COMMON_STATE_NAMES
            )
        )

    def to_json(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path.as_posix(),
            "byte_size": self.byte_size,
            "node_rank": self.node_rank,
            "process_rank": self.process_rank,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "ArtifactEntry":
        expected = {"relative_path", "byte_size", "node_rank", "process_rank"}
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("invalid artifact entry fields")
        entry = cls(
            relative_path=_safe_relative_path(Path(str(payload["relative_path"]))),
            byte_size=payload["byte_size"],  # type: ignore[arg-type]
            node_rank=payload["node_rank"],  # type: ignore[arg-type]
        )
        if payload["process_rank"] != entry.process_rank:
            raise ValueError("artifact process-rank ownership mismatch")
        return entry


@dataclass(frozen=True)
class NodeArtifactManifest:
    schema_version: int
    run_id: str
    node_rank: int
    process_ranks: tuple[int, ...]
    entries: tuple[ArtifactEntry, ...]
    runtime_identity: Mapping[str, object]
    completion_marker: bool = True

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "node_rank": self.node_rank,
            "process_ranks": list(self.process_ranks),
            "entries": [entry.to_json() for entry in self.entries],
            "runtime_identity": _json_value(self.runtime_identity),
            "completion_marker": self.completion_marker,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "NodeArtifactManifest":
        expected = {
            "schema_version",
            "run_id",
            "node_rank",
            "process_ranks",
            "entries",
            "runtime_identity",
            "completion_marker",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("invalid node artifact manifest fields")
        ranks = payload["process_ranks"]
        entries = payload["entries"]
        runtime_identity = payload["runtime_identity"]
        if type(ranks) is not list or type(entries) is not list:
            raise ValueError("invalid node artifact manifest sequence fields")
        if type(runtime_identity) is not dict:
            raise ValueError("invalid node artifact runtime identity")
        return cls(
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
            run_id=payload["run_id"],  # type: ignore[arg-type]
            node_rank=payload["node_rank"],  # type: ignore[arg-type]
            process_ranks=tuple(ranks),  # type: ignore[arg-type]
            entries=tuple(ArtifactEntry.from_json(entry) for entry in entries),
            runtime_identity=runtime_identity,
            completion_marker=payload["completion_marker"],  # type: ignore[arg-type]
        )


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _walk_regular_files(root: Path) -> tuple[tuple[Path, os.stat_result], ...]:
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise ValueError(f"artifact directory is unavailable: {root}") from error
    if stat.S_ISLNK(root_metadata.st_mode):
        raise ValueError("artifact directory must not be a symlink")
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("artifact root must be a directory")

    found: list[tuple[Path, os.stat_result]] = []

    def visit(directory: Path, relative_directory: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise ValueError(f"cannot inspect artifact directory: {directory}") from error
        for child in children:
            relative = relative_directory / child.name
            metadata = child.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"artifact tree contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                visit(Path(child.path), relative)
            elif stat.S_ISREG(metadata.st_mode):
                found.append((_safe_relative_path(relative), metadata))
            else:
                raise ValueError(
                    f"artifact tree entries must be regular files: {relative}"
                )

    visit(root, Path())
    return tuple(found)


def build_node_manifest(
    run_directory: str | Path,
    *,
    node_rank: int,
    process_ranks: Sequence[int],
    run_id: str | None = None,
    runtime_identity: Mapping[str, object] | None = None,
    completion_marker: bool = True,
) -> NodeArtifactManifest:
    """Inspect only lstat metadata and describe one completed node-local tree."""
    directory = Path(run_directory)
    if type(node_rank) is not int or node_rank not in (0, 1):
        raise ValueError("node rank must be 0 or 1")
    ranks = tuple(process_ranks)
    expected_ranks = (0, 1) if node_rank == 0 else (2, 3)
    if ranks != expected_ranks:
        raise ValueError("node manifest must claim only its assigned process ranks")
    entries: list[ArtifactEntry] = []
    for relative, metadata in _walk_regular_files(directory):
        item = ArtifactEntry(relative, metadata.st_size, node_rank)
        if item.process_rank is not None and item.process_rank not in ranks:
            raise ValueError(
                f"artifact shard is outside the node's assigned process ranks: {relative}"
            )
        entries.append(item)
    return NodeArtifactManifest(
        schema_version=_MANIFEST_SCHEMA_VERSION,
        run_id=_validate_run_id(run_id or directory.name),
        node_rank=node_rank,
        process_ranks=ranks,
        entries=tuple(entries),
        runtime_identity=dict(runtime_identity or {}),
        completion_marker=completion_marker,
    )


def _validate_identity(
    identity: Mapping[str, object],
    *,
    required_fields: Sequence[str] = _MANIFEST_IDENTITY_FIELDS,
) -> dict[str, object]:
    if not isinstance(identity, Mapping):
        raise ValueError("runtime identity must be a mapping")
    missing = [field for field in required_fields if field not in identity]
    if missing:
        raise ValueError(f"runtime identity is missing fields: {', '.join(missing)}")
    normalized = _json_value(identity)
    assert isinstance(normalized, dict)
    _validate_run_id(normalized["run_id"])
    return normalized


def _validate_manifest(manifest: NodeArtifactManifest, *, world_size: int) -> None:
    if not isinstance(manifest, NodeArtifactManifest):
        raise TypeError("collection members must be NodeArtifactManifest values")
    if manifest.schema_version != _MANIFEST_SCHEMA_VERSION:
        raise ValueError("artifact manifest schema mismatch")
    _validate_run_id(manifest.run_id)
    if manifest.node_rank not in (0, 1):
        raise ValueError("artifact manifest node rank is invalid")
    expected = (0, 1) if manifest.node_rank == 0 else (2, 3)
    if manifest.process_ranks != expected:
        raise ValueError("node manifest process ranks do not match assigned process ranks")
    if manifest.completion_marker is not True:
        raise ValueError("node artifact completion marker is missing")
    identity = _validate_identity(manifest.runtime_identity)
    if identity["run_id"] != manifest.run_id:
        raise ValueError("runtime identity run ID differs from manifest run ID")
    if identity["world_size"] != world_size:
        raise ValueError("runtime identity world_size mismatch")
    seen: set[Path] = set()
    for entry in manifest.entries:
        relative = _safe_relative_path(entry.relative_path)
        if relative in seen:
            raise ValueError("node manifest contains an ambiguous duplicate path")
        seen.add(relative)
        if type(entry.byte_size) is not int or entry.byte_size < 0:
            raise ValueError("artifact byte size must be nonnegative")
        if entry.node_rank != manifest.node_rank:
            raise ValueError("artifact entry node-rank ownership mismatch")
        if entry.process_rank is not None and entry.process_rank not in expected:
            raise ValueError("artifact entry is outside assigned process ranks")


def validate_collection(
    manifests: Sequence[NodeArtifactManifest], *, world_size: int
) -> tuple[ArtifactEntry, ...]:
    """Validate two node manifests and return one unambiguous combined file set."""
    if type(world_size) is not int or world_size != 4:
        raise ValueError("artifact collection requires world_size 4")
    values = tuple(manifests)
    if not values:
        raise ValueError("artifact collection requires ranks 0 through 3")
    for manifest in values:
        _validate_manifest(manifest, world_size=world_size)
    run_ids = {manifest.run_id for manifest in values}
    if len(run_ids) != 1:
        raise ValueError("artifact manifests have different run IDs")
    identities = [_validate_identity(manifest.runtime_identity) for manifest in values]
    if any(candidate != identities[0] for candidate in identities[1:]):
        raise ValueError("artifact manifests have different runtime identity")
    owned_ranks = {
        entry.process_rank
        for manifest in values
        for entry in manifest.entries
        if entry.process_rank is not None
    }
    if owned_ranks != set(range(world_size)):
        raise ValueError("artifact collection requires exactly ranks 0 through 3")
    node_ranks = [manifest.node_rank for manifest in values]
    if sorted(node_ranks) != [0, 1]:
        raise ValueError("artifact collection requires one manifest from each node")

    combined: dict[Path, ArtifactEntry] = {}
    for manifest in sorted(values, key=lambda item: item.node_rank):
        for entry in manifest.entries:
            previous = combined.get(entry.relative_path)
            if previous is None:
                combined[entry.relative_path] = entry
                continue
            if previous.byte_size != entry.byte_size:
                raise ValueError(
                    f"duplicate artifact path has different sizes: {entry.relative_path}"
                )
            if not previous.is_common_metadata or not entry.is_common_metadata:
                raise ValueError(
                    f"ambiguous duplicate artifact path: {entry.relative_path}"
                )
            # Equal-sized common metadata under an identical runtime identity is
            # represented once, retaining the canonical owner's node-0 entry.
    return tuple(combined[path] for path in sorted(combined, key=lambda item: item.as_posix()))


def _lstat_if_present(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _reject_symlink_ancestors(path: Path) -> None:
    candidate = path if path.is_absolute() else path.absolute()
    for ancestor in reversed(candidate.parents):
        metadata = _lstat_if_present(ancestor)
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"destination ancestor is a symlink: {ancestor}")


def validate_transfer_destination(
    destinations: Sequence[str | Path],
    *,
    validated_identical_common_metadata: Mapping[str | Path, int] | None = None,
) -> None:
    """Prove no destination will be overwritten before invoking rsync."""
    allowed = {
        Path(path): size
        for path, size in (validated_identical_common_metadata or {}).items()
    }
    for value in destinations:
        destination = Path(value)
        _reject_symlink_ancestors(destination)
        metadata = _lstat_if_present(destination)
        if metadata is None:
            continue
        expected_size = allowed.get(destination)
        if (
            expected_size is not None
            and expected_size <= _MAX_COMMON_METADATA_BYTES
            and (
                destination.suffix.lower() in _COMMON_METADATA_SUFFIXES
                or destination.name in _COMMON_STATE_NAMES
            )
            and stat.S_ISREG(metadata.st_mode)
            and metadata.st_size == expected_size
        ):
            continue
        raise FileExistsError(f"refusing to overwrite transfer destination: {destination}")


def _files_from_path(manifest: NodeArtifactManifest, label: str) -> Path:
    return (
        _LOCAL_REPOSITORY
        / "logs"
        / "multinode"
        / manifest.run_id
        / f"artifact-{label}-rank{manifest.node_rank}.files"
    )


def _rsync_prefix(files_from: Path) -> tuple[str, ...]:
    return (
        "/usr/bin/rsync",
        "--archive",
        "--ignore-existing",
        "--files-from",
        str(files_from),
        "--from0",
        "--rsh",
        _RSYNC_SSH,
        "--",
    )


def build_push_command(
    manifest: NodeArtifactManifest, *, files_from: str | Path | None = None
) -> tuple[str, ...]:
    """Build a fixed local-ranks-to-canonical-owner rsync argv."""
    _validate_run_id(manifest.run_id)
    source = _LOCAL_REPOSITORY / "checkpoints" / manifest.run_id
    destination = _REMOTE_HOST_REPOSITORY / "checkpoints" / manifest.run_id
    list_path = Path(files_from) if files_from is not None else _files_from_path(manifest, "push")
    return (
        *_rsync_prefix(list_path),
        f"{source}/",
        f"{_REMOTE_HOST}:{destination}/",
    )


def select_resume_entries(
    manifest: NodeArtifactManifest,
) -> tuple[ArtifactEntry, ...]:
    """Select local ranks 2/3 and the small common metadata needed by resume."""
    return tuple(
        entry
        for entry in manifest.entries
        if entry.is_common_metadata or entry.process_rank in (2, 3)
    )


def build_pull_command(
    manifest: NodeArtifactManifest,
    destination: str | Path | None = None,
    *,
    files_from: str | Path | None = None,
) -> tuple[str, ...]:
    """Build a canonical-owner-to-local rsync argv for resume artifacts."""
    _validate_run_id(manifest.run_id)
    source = _REMOTE_HOST_REPOSITORY / "checkpoints" / manifest.run_id
    local_destination = (
        _LOCAL_REPOSITORY / "checkpoints" / manifest.run_id
        if destination is None
        else Path(destination)
    )
    list_path = Path(files_from) if files_from is not None else _files_from_path(manifest, "pull")
    return (
        *_rsync_prefix(list_path),
        f"{_REMOTE_HOST}:{source}/",
        f"{local_destination}/",
    )


def _identity_mapping(value: object) -> dict[str, object]:
    if is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint identity must be a mapping or dataclass")
    return _validate_identity(value, required_fields=_RESUME_IDENTITY_FIELDS)


def validate_resume_identity(checkpoint: object, requested: object) -> None:
    """Reject every checkpoint/runtime identity change before artifact transfer."""
    actual = _identity_mapping(checkpoint)
    expected = _identity_mapping(requested)
    for field in _RESUME_IDENTITY_FIELDS:
        if actual[field] != expected[field]:
            raise ValueError(f"resume identity mismatch for {field}")


def build_inference_pull_command(
    remote_manifest: NodeArtifactManifest,
    destination: str | Path,
    *,
    files_from: str | Path | None = None,
) -> tuple[str, ...]:
    """Build one manifest-limited pull into a fresh local inference directory."""
    target = Path(destination)
    _reject_symlink_ancestors(target)
    if _lstat_if_present(target) is not None:
        raise FileExistsError(f"fresh inference destination already exists: {target}")
    if remote_manifest.completion_marker is not True or not remote_manifest.entries:
        raise ValueError("remote inference manifest is incomplete")
    _validate_run_id(remote_manifest.run_id)
    identity = _validate_identity(remote_manifest.runtime_identity)
    if identity["run_id"] != remote_manifest.run_id:
        raise ValueError("remote inference manifest run ID mismatch")
    for entry in remote_manifest.entries:
        _safe_relative_path(entry.relative_path)
        if type(entry.byte_size) is not int or entry.byte_size < 0:
            raise ValueError("remote inference artifact size is invalid")
        if entry.node_rank != remote_manifest.node_rank:
            raise ValueError("remote inference artifact node ownership mismatch")
    return build_pull_command(remote_manifest, target, files_from=files_from)


def files_list_bytes(entries: Sequence[ArtifactEntry]) -> bytes:
    """Encode the exact validated rsync file set without reading any artifact."""
    paths = sorted({_safe_relative_path(entry.relative_path).as_posix() for entry in entries})
    return b"".join(path.encode("utf-8") + b"\0" for path in paths)


def publish_node_manifest(
    path: str | Path, manifest: NodeArtifactManifest
) -> Path:
    """Exclusively publish one bounded JSON manifest through the shared helper."""
    from src.multinode_runtime import publish_small_file

    payload = (
        json.dumps(
            manifest.to_json(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return publish_small_file(path, payload)


def publish_files_list(
    path: str | Path, entries: Sequence[ArtifactEntry]
) -> Path:
    """Exclusively publish the exact newline-delimited rsync input list."""
    from src.multinode_runtime import publish_small_file

    return publish_small_file(path, files_list_bytes(entries))


class _ResumeTransferRunner(Protocol):
    def validate_resume_destination(
        self, entries: Sequence[ArtifactEntry], destination: Path
    ) -> None: ...

    def pull_resume_artifacts(
        self,
        manifest: NodeArtifactManifest,
        entries: Sequence[ArtifactEntry],
        destination: Path,
    ) -> None: ...


def prepare_resume_transfer(
    checkpoint_identity: object,
    requested_identity: object,
    remote_manifest: NodeArtifactManifest,
    destination: str | Path,
    runner: _ResumeTransferRunner,
) -> tuple[ArtifactEntry, ...]:
    """Validate all resume identity before destination checks or transfer."""
    validate_resume_identity(checkpoint_identity, requested_identity)
    entries = select_resume_entries(remote_manifest)
    target = Path(destination)
    runner.validate_resume_destination(entries, target)
    runner.pull_resume_artifacts(remote_manifest, entries, target)
    return entries


class _InferenceTransferRunner(Protocol):
    def validate_inference_manifest(self, manifest: NodeArtifactManifest) -> None: ...
    def pull_inference_artifacts(
        self, manifest: NodeArtifactManifest, destination: Path
    ) -> None: ...


def prepare_inference_copy(
    remote_manifest: NodeArtifactManifest,
    destination: str | Path,
    runner: _InferenceTransferRunner,
) -> Path:
    """Validate a remote inference manifest and pull it once to a fresh path."""
    target = Path(destination)
    build_inference_pull_command(remote_manifest, target)
    runner.validate_inference_manifest(remote_manifest)
    runner.pull_inference_artifacts(remote_manifest, target)
    return target
