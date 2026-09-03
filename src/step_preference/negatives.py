"""Strict offline negative artifacts for Stage 2B preference training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


NEGATIVE_MANIFEST_SCHEMA_VERSION = 1


def _canonical_json(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _require_sha256(name: str, value) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_int(name: str, value, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def _require_finite_score(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _freeze_json(value):
    """Return a recursively immutable JSON value with deterministic mapping order."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("generation settings cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        frozen = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("generation setting keys must be strings")
            frozen.append((key, _freeze_json(item)))
        return tuple(sorted(frozen))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise ValueError("generation settings must be JSON values")


def _thaw_json(value):
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw_json(item) for key, item in value}
        return [_thaw_json(item) for item in value]
    return value


def _normalize_generation_settings(value) -> tuple[tuple[str, object], ...]:
    frozen = _freeze_json(value)
    if not isinstance(frozen, tuple) or not all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        for item in frozen
    ):
        raise ValueError("generation_settings must be a JSON object")
    return frozen


@dataclass(frozen=True)
class CheckpointFile:
    """One exact regular file consumed while loading the frozen Stage 2A model."""

    relative_path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.relative_path, str)
            or not self.relative_path
            or Path(self.relative_path).is_absolute()
            or ".." in Path(self.relative_path).parts
        ):
            raise ValueError("checkpoint relative_path must be a safe non-empty relative path")
        _require_int("checkpoint file size", self.size)
        _require_sha256("checkpoint file SHA-256", self.sha256)


@dataclass(frozen=True)
class StepNegative:
    """One chosen completion and its independently valid visual/textual negatives."""

    sample_id: str
    step_id: int
    chosen_text: str
    rejected_visual_text: str | None
    rejected_textual_text: str | None
    valid_visual_negative: bool
    valid_textual_negative: bool
    reference_chosen_score: float
    reference_visual_rejected_score: float | None
    reference_textual_rejected_score: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string")
        _require_int("step_id", self.step_id)
        if not isinstance(self.chosen_text, str) or not self.chosen_text:
            raise ValueError("chosen_text must be a non-empty string")
        _require_finite_score("reference_chosen_score", self.reference_chosen_score)
        self._validate_modality(
            "visual", self.rejected_visual_text, self.valid_visual_negative,
            self.reference_visual_rejected_score,
        )
        self._validate_modality(
            "textual", self.rejected_textual_text, self.valid_textual_negative,
            self.reference_textual_rejected_score,
        )

    @staticmethod
    def _validate_modality(name: str, text, valid, score) -> None:
        if text is not None and not isinstance(text, str):
            raise ValueError(f"rejected_{name}_text must be a string or null")
        if not isinstance(valid, bool):
            raise ValueError(f"valid_{name}_negative must be a boolean")
        if valid:
            if not text:
                raise ValueError(f"valid {name} negative requires non-empty rejected text")
            _require_finite_score(f"reference_{name}_rejected_score", score)
        elif score is not None:
            raise ValueError(f"invalid {name} negative requires a null rejected score")


@dataclass(frozen=True)
class NegativeManifest:
    """Identity of the immutable source/model/configuration that produced a JSONL."""

    schema_version: int
    source_sha256: str
    selected_row_indices: tuple[int, ...]
    stage2a_checkpoint_path: str
    checkpoint_files: tuple[CheckpointFile, ...]
    tokenizer_config_identity: str
    latent_size: int
    generation_settings: tuple[tuple[str, object], ...]
    negative_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != NEGATIVE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        _require_sha256("source_sha256", self.source_sha256)
        _require_sha256("negative_sha256", self.negative_sha256)
        if not isinstance(self.selected_row_indices, tuple):
            raise ValueError("selected_row_indices must be a tuple")
        previous = -1
        for index in self.selected_row_indices:
            _require_int("selected row index", index)
            if index <= previous:
                raise ValueError("selected_row_indices must be strictly increasing")
            previous = index
        if not isinstance(self.stage2a_checkpoint_path, str) or not self.stage2a_checkpoint_path:
            raise ValueError("stage2a_checkpoint_path must be a non-empty string")
        if not isinstance(self.checkpoint_files, tuple) or not all(
            isinstance(entry, CheckpointFile) for entry in self.checkpoint_files
        ):
            raise ValueError("checkpoint_files must be CheckpointFile values")
        relative_paths = [entry.relative_path for entry in self.checkpoint_files]
        if len(relative_paths) != len(set(relative_paths)):
            raise ValueError("checkpoint_files contains duplicate relative paths")
        if not isinstance(self.tokenizer_config_identity, str) or not self.tokenizer_config_identity:
            raise ValueError("tokenizer_config_identity must be a non-empty string")
        _require_int("latent_size", self.latent_size, minimum=1)
        if self.generation_settings != _normalize_generation_settings(
            self.generation_settings
        ):
            raise ValueError("generation_settings must be normalized immutable JSON")


def filter_negative(
    *, chosen_text: str, rejected_text: str | None, scoreable_token_ids: Sequence[int], min_tokens: int
) -> bool:
    """Whether a generated completion can form a valid preference negative."""
    if not isinstance(chosen_text, str):
        raise ValueError("chosen_text must be a string")
    if rejected_text is None:
        return False
    if not isinstance(rejected_text, str):
        raise ValueError("rejected_text must be a string or null")
    _require_int("min_tokens", min_tokens, minimum=1)
    if rejected_text == chosen_text or not rejected_text.strip():
        return False
    if " ".join(rejected_text.split()) == " ".join(chosen_text.split()):
        return False
    return len(scoreable_token_ids) >= min_tokens


def _negative_payload(row: StepNegative) -> dict:
    return asdict(row)


def _manifest_payload(manifest: NegativeManifest) -> dict:
    payload = asdict(manifest)
    payload["generation_settings"] = _thaw_json(manifest.generation_settings)
    return payload


def _write_exclusively(path: Path, data: bytes, label: str) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as error:
        raise ValueError(f"{label} path already exists: {path}") from error


def write_negative_artifact(
    *,
    negative_path: str | Path,
    manifest_path: str | Path,
    negatives: Iterable[StepNegative],
    source_path: str | Path,
    source_sha256: str,
    selected_row_indices: Sequence[int],
    stage2a_checkpoint_path: str | Path,
    checkpoint_files: Sequence[CheckpointFile],
    tokenizer_config_identity: str,
    latent_size: int,
    generation_settings: Mapping[str, object],
) -> NegativeManifest:
    """Write deterministic JSONL and its sidecar manifest, without overwriting either."""
    source_path = Path(source_path)
    actual_source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual_source_sha256 != source_sha256:
        raise ValueError("source SHA-256 does not match source_path")
    checkpoint_path = Path(stage2a_checkpoint_path).resolve(strict=True)
    rows = tuple(negatives)
    if not all(isinstance(row, StepNegative) for row in rows):
        raise TypeError("negatives must contain StepNegative values")
    _negative_store(rows)
    negative_bytes = b"".join(
        (_canonical_json(_negative_payload(row)) + "\n").encode("utf-8") for row in rows
    )
    manifest = NegativeManifest(
        schema_version=NEGATIVE_MANIFEST_SCHEMA_VERSION,
        source_sha256=source_sha256,
        selected_row_indices=tuple(selected_row_indices),
        stage2a_checkpoint_path=str(checkpoint_path),
        checkpoint_files=tuple(checkpoint_files),
        tokenizer_config_identity=tokenizer_config_identity,
        latent_size=latent_size,
        generation_settings=_normalize_generation_settings(generation_settings),
        negative_sha256=hashlib.sha256(negative_bytes).hexdigest(),
    )
    negative_path = Path(negative_path)
    manifest_path = Path(manifest_path)
    _write_exclusively(negative_path, negative_bytes, "negative artifact")
    try:
        _write_exclusively(
            manifest_path, (_canonical_json(_manifest_payload(manifest)) + "\n").encode("utf-8"),
            "negative manifest",
        )
    except Exception:
        # Do not leave a manifest that claims a JSONL we did not publish.
        negative_path.unlink(missing_ok=True)
        raise
    return manifest


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
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON: {error}") from error


def _require_exact_fields(payload: Mapping, expected_type, label: str) -> None:
    expected = {field.name for field in fields(expected_type)}
    actual = set(payload)
    if actual != expected:
        raise ValueError(f"{label} has unknown or missing fields")


def _row_from_payload(payload) -> StepNegative:
    if not isinstance(payload, dict):
        raise ValueError("negative JSONL row must be an object")
    _require_exact_fields(payload, StepNegative, "negative JSONL row")
    return StepNegative(**payload)


def _manifest_from_bytes(raw: bytes) -> NegativeManifest:
    payload = _parse_json(raw, "negative manifest")
    if not isinstance(payload, dict):
        raise ValueError("negative manifest must be an object")
    _require_exact_fields(payload, NegativeManifest, "negative manifest")
    if not isinstance(payload["selected_row_indices"], list):
        raise ValueError("selected_row_indices must be an array")
    if not isinstance(payload["checkpoint_files"], list):
        raise ValueError("checkpoint_files must be an array")
    checkpoint_files = []
    for entry in payload["checkpoint_files"]:
        if not isinstance(entry, dict):
            raise ValueError("checkpoint file must be an object")
        _require_exact_fields(entry, CheckpointFile, "checkpoint file")
        checkpoint_files.append(CheckpointFile(**entry))
    return NegativeManifest(
        schema_version=payload["schema_version"],
        source_sha256=payload["source_sha256"],
        selected_row_indices=tuple(payload["selected_row_indices"]),
        stage2a_checkpoint_path=payload["stage2a_checkpoint_path"],
        checkpoint_files=tuple(checkpoint_files),
        tokenizer_config_identity=payload["tokenizer_config_identity"],
        latent_size=payload["latent_size"],
        generation_settings=_normalize_generation_settings(payload["generation_settings"]),
        negative_sha256=payload["negative_sha256"],
    )


def _read_regular_file(path: str | Path, label: str) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} path must be a regular file: {path}")
    return path.read_bytes()


def _negative_store(rows: Iterable[StepNegative]) -> dict[tuple[str, int], StepNegative]:
    store = {}
    for row in rows:
        key = (row.sample_id, row.step_id)
        if key in store:
            raise ValueError(f"duplicate negative join key: {key}")
        store[key] = row
    return store


def _verify_checkpoint_files(manifest: NegativeManifest) -> None:
    checkpoint_root = Path(manifest.stage2a_checkpoint_path)
    if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
        raise ValueError("Stage 2A checkpoint path must be a regular directory")
    for entry in manifest.checkpoint_files:
        path = checkpoint_root / entry.relative_path
        raw = _read_regular_file(path, "checkpoint file")
        if len(raw) != entry.size or hashlib.sha256(raw).hexdigest() != entry.sha256:
            raise ValueError(f"checkpoint file hash or size does not match: {entry.relative_path}")


def load_negative_store(
    *,
    negative_path: str | Path,
    manifest_path: str | Path,
    expected_source_sha256: str,
    expected_selected_row_indices: Sequence[int],
    expected_stage2a_checkpoint_path: str | Path,
    expected_checkpoint_files: Sequence[CheckpointFile],
    expected_tokenizer_config_identity: str,
    expected_latent_size: int,
    expected_generation_settings: Mapping[str, object],
    expected_keys: Iterable[tuple[str, int]],
) -> dict[tuple[str, int], StepNegative]:
    """Fail closed unless artifact bytes and every frozen producer identity agree."""
    manifest = _manifest_from_bytes(_read_regular_file(manifest_path, "negative manifest"))
    expected_checkpoint_path = str(Path(expected_stage2a_checkpoint_path).resolve(strict=True))
    if manifest.source_sha256 != expected_source_sha256:
        raise ValueError("source SHA-256 does not match")
    if manifest.selected_row_indices != tuple(expected_selected_row_indices):
        raise ValueError("selected row indices do not match")
    if manifest.stage2a_checkpoint_path != expected_checkpoint_path:
        raise ValueError("Stage 2A checkpoint path does not match")
    if manifest.checkpoint_files != tuple(expected_checkpoint_files):
        raise ValueError("checkpoint file records do not match")
    if manifest.tokenizer_config_identity != expected_tokenizer_config_identity:
        raise ValueError("tokenizer/config identity does not match")
    if manifest.latent_size != expected_latent_size:
        raise ValueError("latent size does not match")
    if manifest.generation_settings != _normalize_generation_settings(expected_generation_settings):
        raise ValueError("generation settings do not match")
    _verify_checkpoint_files(manifest)
    negative_bytes = _read_regular_file(negative_path, "negative artifact")
    if hashlib.sha256(negative_bytes).hexdigest() != manifest.negative_sha256:
        raise ValueError("negative artifact SHA-256 does not match")
    rows = []
    for line_number, line in enumerate(negative_bytes.splitlines(), start=1):
        if not line:
            raise ValueError(f"negative JSONL row {line_number} is empty")
        rows.append(_row_from_payload(_parse_json(line, f"negative JSONL row {line_number}")))
    store = _negative_store(rows)
    required_keys = tuple(expected_keys)
    if len(required_keys) != len(set(required_keys)):
        raise ValueError("expected negative join keys are duplicated")
    if set(store) != set(required_keys):
        raise ValueError("missing or unexpected negative join keys")
    return store
