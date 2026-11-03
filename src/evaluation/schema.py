from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
import hashlib
import json


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class ModelInput:
    text: str
    images: tuple[str, ...]
    choices: tuple[str, ...]


@dataclass(frozen=True)
class VspTarget:
    map_desc: tuple[tuple[int | str, ...], ...]
    max_steps: int


@dataclass(frozen=True)
class McqTarget:
    index: int
    label: str
    content: str


@dataclass(frozen=True)
class EmmaTarget:
    question_type: str
    answer: str
    index: int | None
    content: str
    category: str
    task: str


@dataclass(frozen=True)
class ZebraTarget:
    answer: str


@dataclass(frozen=True)
class Provenance:
    dataset_id: str
    revision: str
    config: str
    split: str
    source_id: str
    asset_sha256: dict[str, str]


@dataclass(frozen=True)
class Sample:
    schema_version: int
    id: str
    benchmark: str
    subtask: str
    model_input: ModelInput
    target: VspTarget | McqTarget | EmmaTarget | ZebraTarget
    provenance: Provenance


@dataclass(frozen=True)
class RunMetadata:
    model_identity: str
    prompt_template_identity: str
    generation_config: dict[str, Any]


@dataclass(frozen=True)
class Prediction:
    id: str
    manifest_sha256: str
    run_fingerprint: str
    response: str | None
    status: Literal["ok", "infrastructure_error"]
    error: str | None = None
    termination_reason: str | None = None
    prompt_identity: str | None = None


@dataclass(frozen=True)
class EvalConfig:
    schema_version: int
    benchmark: str
    profile: str
    categories: tuple[str, ...]
    dataset_root: Path
    max_steps: int | None
    run_metadata: RunMetadata


@dataclass(frozen=True)
class ParsedAnswer:
    value: str
    status: Literal["parsed", "empty", "invalid_format"]
    method: Literal[
        "raw",
        "box",
        "lang",
        "marker",
        "boxed",
        "parenthesized",
        "choice_content",
        "upper_label",
        "lower_label",
    ]
    error: str | None = None


@dataclass(frozen=True)
class Simulation:
    success: bool
    position: tuple[int, int]
    steps: int
    reward: float
    terminated: bool
    truncated: bool
    reason: str


@dataclass(frozen=True)
class Score:
    id: str
    category: str
    benchmark: str
    profile: str
    scorer_version: str
    status: str
    parsed_answer: str | None
    extraction_method: str | None
    correctness: bool | None
    simulation: Simulation | None
    error: str | None


@dataclass(frozen=True)
class PreparedEvaluation:
    samples: tuple[Sample, ...]
    predictions: dict[str, Prediction]
    manifest_sha256: str
    run_fingerprint: str


def canonical_json(value: object) -> bytes:
    def check(item: object) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ContractError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)

    check(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError("Non-canonical JSON value") from exc


def run_fingerprint(run: RunMetadata) -> str:
    return hashlib.sha256(canonical_json(asdict(run))).hexdigest()
