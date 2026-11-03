from dataclasses import asdict
import hashlib
import json
from pathlib import Path, PurePosixPath

from PIL import Image

from evaluation.benchmarks.comt import CATEGORIES, LABELS, validate_target
from evaluation.io import exact_keys, read_jsonl_bytes, unique_object, reject_constant
from evaluation.schema import (
    ContractError,
    McqTarget,
    ModelInput,
    Provenance,
    Sample,
    canonical_json,
)


SOURCE_REVISION = "bde314be94942a7b4ab66581e7f7c4801b84e686"
SOURCE_SHA256 = "15935e366f3b9998d3fd18661f563d9cd9a248d33a534e38a73d9b5c65310133"
SELECTION_SHA256 = "6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618"
DATASET_ID = "czh-up/CoMT"
DATASET_PROFILE = "ilvr-test400-corrected-v1"
SOURCE_KEYS = (
    "id",
    "question",
    "option",
    "image",
    "rationale",
    "answer",
    "type",
    "annotations",
)
SELECTION_KEYS = (
    "text_input",
    "image_input",
    "sequence_plan",
    "original_final_answer",
)
REPAIRS = {
    "creation-12125": {
        "old": ["17", "", "18", "", "22", "", "34", ""],
        "new": ["17", "18", "22", "34"],
        "label": "B",
        "content": "18",
    },
    "creation-12164": {
        "old": ["10cm", "12cm", "16", "", "24", ""],
        "new": ["10cm", "12cm", "16", "24"],
        "label": "D",
        "content": "24",
    },
}


def _inside_data(path: Path, data_root: Path, kind: str) -> Path:
    try:
        resolved = Path(path).resolve()
        resolved.relative_to(data_root)
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(f"{kind} path must resolve inside repo_root/data") from exc
    return resolved


def _read_bytes(path: Path, kind: str) -> bytes:
    try:
        return path.read_bytes()
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(f"cannot read {kind}") from exc


def _source_file(source_root: Path, relative: str) -> Path:
    try:
        path = (source_root / relative).resolve(strict=True)
        path.relative_to(source_root)
    except (OSError, ValueError) as exc:
        raise ContractError("source file must stay inside source root") from exc
    if not path.is_file():
        raise ContractError("source file must stay inside source root")
    return path


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, kind: str) -> dict:
    payload = _read_bytes(path, kind)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except ContractError:
        raise
    except (UnicodeError, TypeError, ValueError) as exc:
        raise ContractError(f"invalid {kind} JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"invalid {kind} JSON")
    return value


def _required_string(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ContractError(f"{name} must be a string")
    return value


def _validate_source_rows(rows: list[dict]) -> tuple[dict[tuple[str, str], dict], dict[str, int]]:
    by_image: dict[tuple[str, str], dict] = {}
    source_lines: dict[str, int] = {}
    for line, row in enumerate(rows, 1):
        exact_keys(row, SOURCE_KEYS)
        source_id = _required_string(row["id"], "source id")
        if source_id in source_lines:
            raise ContractError("duplicate source id")
        category = _required_string(row["type"], "source category")
        if category not in CATEGORIES:
            raise ContractError("unknown source category")
        _required_string(row["question"], "source question", allow_empty=True)
        _required_string(row["rationale"], "source rationale", allow_empty=True)
        label = _required_string(row["answer"], "source answer")
        if label not in LABELS:
            raise ContractError("source answer must be A through D")
        options = row["option"]
        if not isinstance(options, list) or any(
            type(option) not in (str, int) for option in options
        ):
            raise ContractError("source options must contain only strings or integers")
        image = row["image"]
        if not isinstance(image, dict) or "IMAGE0" not in image:
            raise ContractError("source image requires IMAGE0")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in image.items()
        ):
            raise ContractError("source image tokens must be strings")
        token = _required_string(image["IMAGE0"], "source IMAGE0 token")
        if token != Path(token).name or token in (".", ".."):
            raise ContractError("source IMAGE0 token must be a filename token")
        join_key = (category, token)
        if join_key in by_image:
            raise ContractError("ambiguous source category/IMAGE0 token")
        source_lines[source_id] = line
        by_image[join_key] = row
    return by_image, source_lines


def _selection_image(row: dict) -> tuple[str, str, str]:
    exact_keys(row, SELECTION_KEYS)
    _required_string(row["text_input"], "selection text", allow_empty=True)
    _required_string(
        row["original_final_answer"], "selection gold", allow_empty=True
    )
    if not isinstance(row["sequence_plan"], list):
        raise ContractError("selection sequence_plan must be a list")
    images = row["image_input"]
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
        raise ContractError("selection requires exactly one IMAGE0 image")
    relative = PurePosixPath(images[0])
    if (
        relative.is_absolute()
        or len(relative.parts) != 3
        or relative.parts[0] != "images_comt"
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise ContractError("selection image path must be images_comt/<category>/<file>")
    category = relative.parts[1]
    if category not in CATEGORIES:
        raise ContractError("selection image has unknown category")
    filename = relative.parts[2]
    path = Path(filename)
    if path.suffix.lower() != ".png" or not path.stem:
        raise ContractError("selection image filename must be a PNG")
    return category, path.stem, relative.as_posix()


def _verify_image(
    image_root: Path, category: str, token: str, selected_relative: str, data_root: Path
) -> tuple[str, str]:
    directory = image_root / category
    try:
        candidates = [
            path for path in directory.iterdir() if path.is_file() and path.stem == token
        ]
    except OSError as exc:
        raise ContractError("cannot inspect selected image") from exc
    if len(candidates) != 1:
        raise ContractError("selected image token must resolve to exactly one image")
    image = candidates[0]
    expected_relative = image.relative_to(data_root).as_posix()
    if selected_relative != expected_relative:
        raise ContractError("selection image path does not match source category/IMAGE0")
    try:
        resolved = image.resolve(strict=True)
        resolved.relative_to(image_root)
        resolved.relative_to(data_root)
    except (OSError, ValueError) as exc:
        raise ContractError("selected image path must stay inside image root") from exc
    try:
        with Image.open(resolved) as opened:
            opened.verify()
        with Image.open(resolved) as opened:
            opened.load()
    except Exception as exc:
        raise ContractError("selected image is not a valid decodable image") from exc
    payload = _read_bytes(resolved, "selected image")
    return expected_relative, _digest(payload)


def _source_text(question: str, options: list[object]) -> str:
    return question.replace("[IMAGE0]", "").strip() + " option:" + str(options)


def _data_relative(path: Path, data_root: Path) -> str:
    return (Path("data") / path.relative_to(data_root)).as_posix()


def prepare_comt(
    source_root: Path,
    selection_manifest: Path,
    image_root: Path,
    output: Path,
    *,
    repo_root: Path,
) -> dict:
    repository = Path(repo_root).resolve()
    data_root = (repository / "data").resolve()
    source_root = _inside_data(source_root, data_root, "source root")
    selection_manifest = _inside_data(
        selection_manifest, data_root, "selection manifest"
    )
    image_root = _inside_data(image_root, data_root, "image root")
    output = _inside_data(output, data_root, "output")
    if output.exists():
        raise ContractError("dataset output already exists")
    if not source_root.is_dir():
        raise ContractError("source root must be an existing directory")
    if not selection_manifest.is_file():
        raise ContractError("selection manifest must be an existing file")
    if image_root != data_root / "images_comt" or not image_root.is_dir():
        raise ContractError("image root must be repo_root/data/images_comt")

    metadata_path = _source_file(source_root, "dataset-info.json")
    metadata = _read_json(metadata_path, "dataset metadata")
    if metadata.get("id") != DATASET_ID:
        raise ContractError("source dataset identity is not canonical czh-up/CoMT")
    if metadata.get("sha") != SOURCE_REVISION:
        raise ContractError("source dataset revision is not pinned")

    source_path = _source_file(source_root, "comt/data.jsonl")
    source_payload = _read_bytes(source_path, "CoMT source JSONL")
    if _digest(source_payload) != SOURCE_SHA256:
        raise ContractError("CoMT source SHA-256 mismatch")
    selection_payload = _read_bytes(selection_manifest, "selection JSONL")
    if _digest(selection_payload) != SELECTION_SHA256:
        raise ContractError("CoMT selection SHA-256 mismatch")
    source_rows = read_jsonl_bytes(source_payload)
    selection_rows = read_jsonl_bytes(selection_payload)
    if len(selection_rows) != 400:
        raise ContractError("CoMT selection must contain exactly 400 rows")
    by_image, source_lines = _validate_source_rows(source_rows)

    category_counts = {category: 0 for category in CATEGORIES}
    selected_ids: set[str] = set()
    prepared_rows: list[dict] = []
    links: list[dict] = []
    correction_records: list[dict] = []
    for selection_line, selection in enumerate(selection_rows, 1):
        category, token, selected_image = _selection_image(selection)
        source = by_image.get((category, token))
        if source is None:
            raise ContractError("selection image has no matching source row")
        source_id = source["id"]
        if source_id in selected_ids:
            raise ContractError("duplicate source row in selection")
        selected_ids.add(source_id)
        category_counts[category] += 1

        original_options = source["option"]
        expected_text = _source_text(source["question"], original_options)
        if selection["text_input"] != expected_text:
            raise ContractError("selection text does not match source question/options")
        if "[IMAGE" in source["question"].replace("[IMAGE0]", ""):
            raise ContractError("selected source question contains a helper IMAGE placeholder")
        label = source["answer"]
        label_index = LABELS.index(label)
        if label_index >= len(original_options):
            raise ContractError("source label has no corresponding option")
        legacy_content = str(original_options[label_index])
        if selection["original_final_answer"] != legacy_content:
            raise ContractError("selection legacy gold does not match source label content")

        repair = REPAIRS.get(source_id)
        if repair is not None:
            if (
                category != "creation"
                or original_options != repair["old"]
                or label != repair["label"]
            ):
                raise ContractError("pinned correction source row does not match policy")
            choices = list(repair["new"])
            correction_records.append(
                {
                    "source_id": source_id,
                    "old": list(repair["old"]),
                    "new": list(repair["new"]),
                    "label": label,
                    "content": repair["content"],
                    "evidence": {
                        "kind": "pinned_source_rationale",
                        "source_id": source_id,
                        "source_sha256": SOURCE_SHA256,
                    },
                }
            )
        else:
            choices = [str(option) for option in original_options]
        if len(choices) != 4 or any(not choice.strip() for choice in choices):
            raise ContractError("selected source row must have four nonempty choices")
        target = McqTarget(label_index, label, choices[label_index])
        model_text = (
            _source_text(source["question"], choices)
            if repair is not None
            else selection["text_input"]
        )
        image_name, image_hash = _verify_image(
            image_root, category, token, selected_image, data_root
        )
        model_input = ModelInput(model_text, (image_name,), tuple(choices))
        validate_target(target, model_input)
        sample = Sample(
            1,
            f"comt/{category}/{source_id}",
            "comt",
            category,
            model_input,
            target,
            Provenance(
                DATASET_ID,
                SOURCE_REVISION,
                "default",
                DATASET_PROFILE,
                source_id,
                {image_name: image_hash},
            ),
        )
        prepared_rows.append(asdict(sample))
        links.append(
            {
                "manifest_index": selection_line - 1,
                "selection_line": selection_line,
                "source_line": source_lines[source_id],
                "source_id": source_id,
                "category": category,
                "image_token": token,
            }
        )

    if category_counts != {category: 100 for category in CATEGORIES}:
        raise ContractError("CoMT selection must contain exactly 100 rows per category")
    if {record["source_id"] for record in correction_records} != set(REPAIRS):
        raise ContractError("CoMT selection must contain both pinned corrections exactly once")

    manifest_payload = b"".join(
        canonical_json(row) + b"\n" for row in prepared_rows
    )
    manifest_hash = _digest(manifest_payload)
    summary = {
        "N_expected": 400,
        "categories": category_counts,
        "correction_count": 2,
        "manifest_sha256": manifest_hash,
        "dataset_root": "data",
        "dataset_id": DATASET_ID,
        "revision": SOURCE_REVISION,
        "config": "default",
        "dataset_profile": DATASET_PROFILE,
        "source_sha256": SOURCE_SHA256,
        "selection_sha256": SELECTION_SHA256,
        "protocol_kind": "modified",
        "paper_protocol_verified": False,
        "ood_verified": False,
    }
    source_record = {
        "schema_version": 1,
        **summary,
        "source": {
            "path": _data_relative(source_path, data_root),
            "sha256": SOURCE_SHA256,
        },
        "selection": {
            "path": _data_relative(selection_manifest, data_root),
            "sha256": SELECTION_SHA256,
        },
        "limitations": [
            "paper protocol is not verified",
            "out-of-distribution status is not verified",
        ],
    }
    selection_record = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "revision": SOURCE_REVISION,
        "source": {
            "path": _data_relative(source_path, data_root),
            "sha256": SOURCE_SHA256,
        },
        "selection": {
            "path": _data_relative(selection_manifest, data_root),
            "sha256": SELECTION_SHA256,
        },
        "rows": links,
    }
    corrections_record = {
        "schema_version": 1,
        "dataset_profile": DATASET_PROFILE,
        "source_sha256": SOURCE_SHA256,
        "corrections": correction_records,
    }

    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.jsonl").write_bytes(manifest_payload)
    (output / "source-selection.json").write_bytes(
        canonical_json(selection_record) + b"\n"
    )
    (output / "corrections.json").write_bytes(
        canonical_json(corrections_record) + b"\n"
    )
    (output / "SOURCE.json").write_bytes(canonical_json(source_record) + b"\n")
    return summary
