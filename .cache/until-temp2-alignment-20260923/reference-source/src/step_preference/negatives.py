"""Validated contracts for offline step-wise negative artifacts."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Protocol, Sequence

from torch import Tensor

from .spans import ReasoningStepSpan, stable_sample_id

from artifact_paths import (
    CANONICAL_REPOSITORY_ROOT,
    EXPERIMENT_ID_PATTERN,
    STEP_PREFERENCE_LOG_ROOT,
    STEP_PREFERENCE_RESULT_ROOT,
)


NEGATIVE_SCHEMA_VERSION = 2


RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "step_id",
        "chosen_text",
        "rejected_visual_text",
        "rejected_textual_text",
        "valid_visual_negative",
        "valid_textual_negative",
        "reference_chosen_mean_logp",
        "reference_rejected_visual_mean_logp",
        "reference_rejected_textual_mean_logp",
        "chosen_token_count",
        "rejected_visual_token_count",
        "rejected_textual_token_count",
        "rejected_visual_token_ids",
        "rejected_textual_token_ids",
    }
)
MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "stage2a_artifact",
        "stage2a_metadata_digest",
        "stage2a_lifecycle_metadata_digest",
        "tokenizer_reference",
        "tokenizer_vocab_size",
        "dataset_reference",
        "generation_settings",
        "record_count",
        "valid_visual_count",
        "valid_textual_count",
        "records_sha256",
    }
)


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_finite_score(value: object, field_name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


def _require_exact_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _require_token_ids(value: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(
        type(token_id) is not int or token_id < 0 for token_id in value
    ):
        raise ValueError(f"{field_name} must be a sequence of non-negative integers")
    return tuple(value)


def _require_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _normalise_text(text: str) -> str:
    return " ".join(text.split())


def validate_negative_text(chosen: str, rejected: str, min_chars: int) -> bool:
    """Return whether a generated rejection is distinct and long enough."""
    if type(min_chars) is not int or min_chars < 0:
        raise ValueError("min_chars must be a non-negative integer")
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return False
    normalized_rejected = _normalise_text(rejected)
    return (
        bool(normalized_rejected)
        and len(normalized_rejected) >= min_chars
        and _normalise_text(chosen) != normalized_rejected
    )


@dataclass(frozen=True)
class StepGenerationExample:
    """One prepared sample with token-stable reasoning-step coordinates."""

    sample_id: str
    model_inputs: Mapping[str, Tensor]
    spans: tuple[ReasoningStepSpan, ...]


class StepNegativeBackend(Protocol):
    """Model-dependent operations used by offline negative orchestration."""

    def chosen_text(
        self, *, example: StepGenerationExample, step_span: ReasoningStepSpan
    ) -> str: ...

    def generate(
        self,
        *,
        example: StepGenerationExample,
        step_span: ReasoningStepSpan,
        mode: Literal["visual", "text"],
        max_new_tokens: int,
        temperature: float,
    ) -> str: ...

    def mean_logp(
        self,
        *,
        example: StepGenerationExample,
        step_span: ReasoningStepSpan,
        text: str,
    ) -> tuple[float, int, tuple[int, ...]]: ...


def generate_step_negative_records(
    examples: Sequence[StepGenerationExample],
    backend: StepNegativeBackend,
    *,
    min_chars: int,
    max_new_tokens: int,
    temperature: float,
    correctness_checker: Callable[[str], bool] | None = None,
) -> tuple["StepNegativeRecord", ...]:
    """Generate and clean-score independent visual/text negatives per step."""
    if type(min_chars) is not int or min_chars < 0:
        raise ValueError("min_chars must be a non-negative integer")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be finite and non-negative")
    seen_ids: set[str] = set()
    for example in examples:
        if example.sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id: {example.sample_id}")
        seen_ids.add(example.sample_id)

    records: list[StepNegativeRecord] = []
    for example in examples:
        for span in example.spans:
            chosen = backend.chosen_text(example=example, step_span=span)
            chosen_logp, chosen_count, _chosen_token_ids = backend.mean_logp(
                example=example, step_span=span, text=chosen
            )
            rejected = {
                mode: backend.generate(
                    example=example,
                    step_span=span,
                    mode=mode,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                for mode in ("visual", "text")
            }
            rejected_scores = {
                mode: backend.mean_logp(
                    example=example, step_span=span, text=text
                )
                if text
                else (0.0, 0, ())
                for mode, text in rejected.items()
            }
            valid = {
                mode: validate_negative_text(chosen, text, min_chars)
                and (
                    correctness_checker is None
                    or not correctness_checker(text)
                )
                for mode, text in rejected.items()
            }
            records.append(
                StepNegativeRecord(
                    schema_version=NEGATIVE_SCHEMA_VERSION,
                    sample_id=example.sample_id,
                    step_id=span.step_id,
                    chosen_text=chosen,
                    rejected_visual_text=rejected["visual"],
                    rejected_textual_text=rejected["text"],
                    valid_visual_negative=valid["visual"],
                    valid_textual_negative=valid["text"],
                    reference_chosen_mean_logp=chosen_logp,
                    reference_rejected_visual_mean_logp=rejected_scores[
                        "visual"
                    ][0],
                    reference_rejected_textual_mean_logp=rejected_scores[
                        "text"
                    ][0],
                    chosen_token_count=chosen_count,
                    rejected_visual_token_count=rejected_scores["visual"][1],
                    rejected_textual_token_count=rejected_scores["text"][1],
                    rejected_visual_token_ids=rejected_scores["visual"][2],
                    rejected_textual_token_ids=rejected_scores["text"][2],
                )
            )
    return tuple(records)


@dataclass(frozen=True)
class StepNegativeRecord:
    schema_version: int
    sample_id: str
    step_id: int
    chosen_text: str
    rejected_visual_text: str
    rejected_textual_text: str
    valid_visual_negative: bool
    valid_textual_negative: bool
    reference_chosen_mean_logp: float
    reference_rejected_visual_mean_logp: float
    reference_rejected_textual_mean_logp: float
    chosen_token_count: int
    rejected_visual_token_count: int
    rejected_textual_token_count: int
    rejected_visual_token_ids: tuple[int, ...]
    rejected_textual_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != NEGATIVE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"negative record schema_version must be {NEGATIVE_SCHEMA_VERSION}"
            )
        _require_non_empty_string(self.sample_id, "sample_id")
        _require_non_negative_int(self.step_id, "step_id")
        _require_non_empty_string(self.chosen_text, "chosen_text")
        _require_exact_bool(self.valid_visual_negative, "valid_visual_negative")
        _require_exact_bool(self.valid_textual_negative, "valid_textual_negative")
        _require_finite_score(
            self.reference_chosen_mean_logp, "reference_chosen_mean_logp"
        )
        _require_finite_score(
            self.reference_rejected_visual_mean_logp,
            "reference_rejected_visual_mean_logp",
        )
        _require_finite_score(
            self.reference_rejected_textual_mean_logp,
            "reference_rejected_textual_mean_logp",
        )
        _require_non_negative_int(self.chosen_token_count, "chosen_token_count")
        if self.chosen_token_count == 0:
            raise ValueError("chosen_token_count must be a positive token count")
        self._validate_rejected_pair(
            self.rejected_visual_text,
            self.valid_visual_negative,
            self.rejected_visual_token_count,
            self.rejected_visual_token_ids,
            "visual",
        )
        self._validate_rejected_pair(
            self.rejected_textual_text,
            self.valid_textual_negative,
            self.rejected_textual_token_count,
            self.rejected_textual_token_ids,
            "textual",
        )

    def _validate_rejected_pair(
        self,
        rejected_text: object,
        is_valid: bool,
        token_count: object,
        token_ids: object,
        modality: str,
    ) -> None:
        if not isinstance(rejected_text, str):
            raise ValueError(f"rejected_{modality}_text must be a string")
        _require_non_negative_int(token_count, f"rejected_{modality}_token_count")
        validated_token_ids = _require_token_ids(
            token_ids, f"rejected_{modality}_token_ids"
        )
        object.__setattr__(
            self, f"rejected_{modality}_token_ids", validated_token_ids
        )
        if len(validated_token_ids) != token_count:
            raise ValueError(
                f"rejected {modality} token IDs disagree with token count"
            )
        normalized_rejected = _normalise_text(rejected_text)
        if not normalized_rejected:
            if is_valid or token_count != 0:
                raise ValueError(
                    f"empty rejected {modality} text requires invalid flag and zero token count"
                )
            return
        if token_count == 0:
            raise ValueError(f"rejected {modality} text requires a positive token count")
        if is_valid and _normalise_text(self.chosen_text) == normalized_rejected:
            raise ValueError(f"valid rejected {modality} text must differ from chosen text")


@dataclass(frozen=True)
class StepNegativeManifest:
    schema_version: int
    stage2a_artifact: str
    stage2a_metadata_digest: str
    stage2a_lifecycle_metadata_digest: str
    tokenizer_reference: str
    tokenizer_vocab_size: int
    dataset_reference: str
    generation_settings: dict[str, object]
    record_count: int
    valid_visual_count: int
    valid_textual_count: int
    records_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != NEGATIVE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"negative manifest schema_version must be {NEGATIVE_SCHEMA_VERSION}"
            )
        _require_non_empty_string(self.stage2a_artifact, "stage2a_artifact")
        _require_non_empty_string(self.stage2a_metadata_digest, "stage2a_metadata_digest")
        _require_sha256(
            self.stage2a_lifecycle_metadata_digest,
            "stage2a_lifecycle_metadata_digest",
        )
        _require_non_empty_string(self.tokenizer_reference, "tokenizer_reference")
        if type(self.tokenizer_vocab_size) is not int or self.tokenizer_vocab_size <= 0:
            raise ValueError("tokenizer_vocab_size must be a positive integer")
        _require_non_empty_string(self.dataset_reference, "dataset_reference")
        if type(self.generation_settings) is not dict:
            raise ValueError("generation_settings must be an object")
        _require_non_negative_int(self.record_count, "record_count")
        _require_non_negative_int(self.valid_visual_count, "valid_visual_count")
        _require_non_negative_int(self.valid_textual_count, "valid_textual_count")
        if (
            self.valid_visual_count > self.record_count
            or self.valid_textual_count > self.record_count
        ):
            raise ValueError("manifest valid counts cannot exceed record count")
        if self.records_sha256 is not None:
            _require_sha256(self.records_sha256, "records_sha256")


@dataclass(frozen=True)
class StepNegativeIndex:
    records: Mapping[tuple[str, int], StepNegativeRecord]
    manifest: StepNegativeManifest | None

    def get(self, sample_id: str, step_id: int) -> StepNegativeRecord:
        try:
            return self.records[(sample_id, step_id)]
        except KeyError as error:
            raise KeyError(f"missing step negative: {sample_id}/{step_id}") from error


@dataclass(frozen=True)
class StepPreferenceOutputPaths:
    result_dir: Path
    log_dir: Path


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON field: {key}")
        payload[key] = value
    return payload


def _read_json_object(path: Path, *, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {description}: {path}") from error
    if type(payload) is not dict:
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _record_from_payload(payload: object, *, line_number: int) -> StepNegativeRecord:
    if type(payload) is not dict or set(payload) != RECORD_FIELDS:
        raise ValueError(
            f"record line {line_number} must contain exactly the "
            f"{len(RECORD_FIELDS)} record fields"
        )
    try:
        return StepNegativeRecord(**payload)
    except TypeError as error:
        raise ValueError(f"invalid record at line {line_number}") from error


def _load_manifest(path: Path) -> StepNegativeManifest:
    payload = _read_json_object(path, description="negative manifest")
    if set(payload) != MANIFEST_FIELDS:
        raise ValueError("negative manifest must contain exactly the manifest fields")
    try:
        manifest = StepNegativeManifest(**payload)
    except TypeError as error:
        raise ValueError("invalid negative manifest") from error
    if manifest.records_sha256 is None:
        raise ValueError("negative manifest records_sha256 must seal the JSONL")
    return manifest


def _validate_complete_sample_ids(complete_sample_ids: Sequence[str] | None) -> None:
    if complete_sample_ids is None:
        return
    seen: set[str] = set()
    for sample_id in complete_sample_ids:
        _require_non_empty_string(sample_id, "complete sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate complete sample ID: {sample_id}")
        seen.add(sample_id)


def _validate_complete_record_keys(
    complete_record_keys: Sequence[tuple[str, int]] | None,
) -> tuple[tuple[str, int], ...] | None:
    if complete_record_keys is None:
        return None
    validated: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for key in complete_record_keys:
        if not isinstance(key, (list, tuple)) or len(key) != 2:
            raise ValueError("complete record key must be (sample_id, step_id)")
        sample_id = _require_non_empty_string(key[0], "complete sample_id")
        step_id = _require_non_negative_int(key[1], "complete step_id")
        normalized = (sample_id, step_id)
        if normalized in seen:
            raise ValueError(
                f"duplicate complete step key: {sample_id}/{step_id}"
            )
        seen.add(normalized)
        validated.append(normalized)
    return tuple(validated)


def load_step_negative_index(
    jsonl_path: Path,
    manifest_path: Path | None = None,
    complete_sample_ids: Sequence[str] | None = None,
    complete_record_keys: Sequence[tuple[str, int]] | None = None,
) -> StepNegativeIndex:
    """Load one strict negative artifact without reading model artifacts."""
    _validate_complete_sample_ids(complete_sample_ids)
    expected_keys = _validate_complete_record_keys(complete_record_keys)
    path = Path(jsonl_path)
    records: dict[tuple[str, int], StepNegativeRecord] = {}
    records_digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                records_digest.update(raw_line)
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError(
                        f"invalid UTF-8 record at line {line_number}"
                    ) from error
                if not line.strip():
                    raise ValueError(f"record line {line_number} must not be blank")
                try:
                    payload = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON record at line {line_number}") from error
                record = _record_from_payload(payload, line_number=line_number)
                key = (record.sample_id, record.step_id)
                if key in records:
                    raise ValueError(f"duplicate step negative record: {record.sample_id}/{record.step_id}")
                records[key] = record
    except OSError as error:
        raise ValueError(f"cannot read negative JSONL: {path}") from error

    if complete_sample_ids is not None:
        complete_ids = set(complete_sample_ids)
        record_ids = {sample_id for sample_id, _step_id in records}
        missing = sorted(complete_ids - record_ids)
        extra = sorted(record_ids - complete_ids)
        if missing or extra:
            raise ValueError(
                "negative artifact sample coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )

    if expected_keys is not None:
        actual_keys = tuple(records)
        expected_set = set(expected_keys)
        actual_set = set(actual_keys)
        missing = [key for key in expected_keys if key not in actual_set]
        extra = [key for key in actual_keys if key not in expected_set]
        if missing or extra:
            raise ValueError(
                "negative artifact step coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )
        if actual_keys != expected_keys:
            raise ValueError(
                "negative artifact ordered step coverage mismatch: "
                f"expected={expected_keys}, actual={actual_keys}"
            )

    manifest = _load_manifest(Path(manifest_path)) if manifest_path is not None else None
    if manifest is not None:
        valid_visual_count = sum(record.valid_visual_negative for record in records.values())
        valid_textual_count = sum(record.valid_textual_negative for record in records.values())
        if manifest.record_count != len(records):
            raise ValueError("manifest record count does not match negative JSONL")
        if manifest.valid_visual_count != valid_visual_count:
            raise ValueError("manifest valid visual count does not match negative JSONL")
        if manifest.valid_textual_count != valid_textual_count:
            raise ValueError("manifest valid textual count does not match negative JSONL")
        if manifest.records_sha256 != records_digest.hexdigest():
            raise ValueError("negative JSONL digest does not match manifest")
    return StepNegativeIndex(records=MappingProxyType(records), manifest=manifest)


def expected_step_negative_keys(
    samples: Sequence[Mapping[str, object]],
) -> tuple[tuple[str, int], ...]:
    """Derive the exact ordered artifact key contract without tokenizing data."""
    keys: list[tuple[str, int]] = []
    seen_sample_ids: set[str] = set()
    for sample in samples:
        sample_id = stable_sample_id(sample)
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate selected sample ID: {sample_id}")
        seen_sample_ids.add(sample_id)
        sequence_plan = sample.get("sequence_plan")
        if not isinstance(sequence_plan, list):
            raise ValueError("sequence_plan must be a list")
        step_id = 0
        for entry in sequence_plan:
            if not isinstance(entry, Mapping):
                raise ValueError("sequence_plan steps must be objects")
            entry_type = entry.get("type")
            if entry_type == "latent":
                keys.append((sample_id, step_id))
                step_id += 1
            elif entry_type != "text":
                raise ValueError(
                    f"unsupported sequence_plan step type: {entry_type}"
                )
    return tuple(keys)


def validate_step_negative_dataset_provenance(
    manifest: StepNegativeManifest, dataset_path: str | Path
) -> None:
    """Bind the artifact to the selected dataset before any source loading."""
    if not isinstance(manifest, StepNegativeManifest):
        raise TypeError("manifest must be StepNegativeManifest")
    try:
        requested_dataset = Path(dataset_path).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"negative artifact dataset does not exist: {dataset_path}") from error
    manifest_dataset = Path(manifest.dataset_reference).resolve(strict=False)
    if manifest_dataset != requested_dataset:
        raise ValueError(
            "negative artifact dataset identity mismatch: "
            f"manifest={manifest_dataset}, requested={requested_dataset}"
        )


def _normalize_reference(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return str(path.resolve(strict=False))
    return value


def validate_step_negative_runtime_provenance(
    manifest: StepNegativeManifest,
    *,
    tokenizer_reference: str,
    tokenizer_vocab_size: int,
    attention_implementation: str,
    ablation: str,
    reference: str,
    records: Sequence[StepNegativeRecord] = (),
) -> None:
    """Validate tokenizer and fixed generation-method identity pre-model-load."""
    if not isinstance(manifest, StepNegativeManifest):
        raise TypeError("manifest must be StepNegativeManifest")
    actual_tokenizer_reference = _require_non_empty_string(
        tokenizer_reference, "tokenizer_reference"
    )
    if _normalize_reference(manifest.tokenizer_reference) != _normalize_reference(
        actual_tokenizer_reference
    ):
        raise ValueError("negative artifact tokenizer identity mismatch")
    if (
        type(tokenizer_vocab_size) is not int
        or tokenizer_vocab_size != manifest.tokenizer_vocab_size
    ):
        raise ValueError("negative artifact tokenizer vocabulary mismatch")
    expected_settings = {
        "attention_implementation": attention_implementation,
        "ablation": ablation,
        "reference_context": "clean",
        "reference": reference,
    }
    for name, expected in expected_settings.items():
        actual = manifest.generation_settings.get(name)
        if actual != expected:
            raise ValueError(
                f"negative artifact generation {name} mismatch: "
                f"manifest={actual!r}, requested={expected!r}"
            )
    for record in records:
        for token_ids in (
            record.rejected_visual_token_ids,
            record.rejected_textual_token_ids,
        ):
            if any(token_id >= tokenizer_vocab_size for token_id in token_ids):
                raise ValueError(
                    "negative artifact rejected token ID exceeds tokenizer vocabulary"
                )


def resolve_step_preference_output_paths(experiment_id: str) -> StepPreferenceOutputPaths:
    """Return canonical shared output paths without creating directories."""
    if not isinstance(experiment_id, str) or not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError("experiment ID must be one safe directory name")
    canonical_root = CANONICAL_REPOSITORY_ROOT.resolve(strict=False)
    result_root = STEP_PREFERENCE_RESULT_ROOT.resolve(strict=False)
    log_root = STEP_PREFERENCE_LOG_ROOT.resolve(strict=False)
    if (
        result_root == canonical_root
        or log_root == canonical_root
        or not result_root.is_relative_to(canonical_root)
        or not log_root.is_relative_to(canonical_root)
    ):
        raise ValueError("experiment output root must remain below canonical repository")
    result_dir = (result_root / experiment_id).resolve(strict=False)
    log_dir = (log_root / experiment_id).resolve(strict=False)
    if not result_dir.is_relative_to(result_root) or not log_dir.is_relative_to(log_root):
        raise ValueError("experiment output path escapes its canonical root")
    return StepPreferenceOutputPaths(result_dir=result_dir, log_dir=log_dir)
