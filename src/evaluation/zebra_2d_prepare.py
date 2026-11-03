"""Prepare the pinned 400-row Zebra-CoT 2D source projection."""
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re

from evaluation.benchmarks.zebra_2d import CATEGORIES, validate_target
from evaluation.schema import ContractError, ModelInput, Provenance, Sample, ZebraTarget, canonical_json


DATASET_ID = "multimodal-reasoning-lab/Zebra-CoT"
SOURCE_REVISION = "f6d1defc169d19180a69f517925ed6cbb06a1c97"
DATASET_PROFILE = "zebra-2d400-v1"
SELECTION_SHA256 = "92e776e6b7e7e646c7f3b23c05c87b0993af44821faecac789a0e29ef2737927"
RECORDS_SHA256 = "1a533ce69f426cce9397251df976cf4331118f530b342713747f2ebdae3d4f25"
CONFIGS = {"Jigsaw": "2D Visual Reasoning - Visual Jigsaw",
           "Search": "2D Visual Reasoning - Visual Search"}
SELECTION_KEYS = {"task", "config", "split", "global_row", "shard_path", "shard_row", "selection_hash"}
RECORD_KEYS = SELECTION_KEYS | {"Question", "Final Answer", "problem_image_1"}
IMAGE_KEYS = {"path", "source_path", "sha256", "bytes", "format", "width", "height", "mode"}
SHA256 = re.compile(r"[0-9a-f]{64}")


def _inside(path: Path, parent: Path, message: str) -> Path:
    try:
        resolved = Path(path).resolve()
        resolved.relative_to(parent)
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(message) from exc
    return resolved


def _source_file(source: Path, relative: str) -> Path:
    try:
        path = (source / relative).resolve(strict=True)
        path.relative_to(source)
    except (OSError, ValueError) as exc:
        raise ContractError("source file must stay inside source root") from exc
    if not path.is_file():
        raise ContractError("source file must stay inside source root")
    return path


def _manifest_relative(value: str) -> str:
    relative = _nonempty(value, "image path")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in ("", ".", "..") for part in relative.split("/"))
    ):
        raise ContractError("image path must be a clean manifest-relative path")
    return relative


def _nonempty(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a nonempty string")
    return value


def _row_number(value, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        raise ContractError(f"{field} must be a valid integer")
    return value


def _validate_selection(row: object) -> tuple[str, int]:
    if not isinstance(row, dict) or set(row) != SELECTION_KEYS:
        raise ContractError("unexpected or missing selection fields")
    task = _nonempty(row["task"], "task")
    if task not in CATEGORIES or row["config"] != CONFIGS[task] or row["split"] != "train":
        raise ContractError("wrong task, config, or split")
    _nonempty(row["config"], "config")
    _nonempty(row["split"], "split")
    global_row = _row_number(row["global_row"], "global_row")
    _nonempty(row["shard_path"], "shard_path")
    _row_number(row["shard_row"], "shard_row")
    if not isinstance(row["selection_hash"], str) or not SHA256.fullmatch(row["selection_hash"]):
        raise ContractError("selection_hash must be lowercase SHA-256")
    return task, global_row


def prepare_zebra_2d(source_root: Path, output: Path, *, repo_root: Path) -> dict:
    data_root = (Path(repo_root).resolve() / "data").resolve()
    source = _inside(source_root, data_root, "source path must resolve inside repo_root/data")
    destination = _inside(output, data_root, "output path must resolve inside repo_root/data")
    if destination.exists():
        raise ContractError("output already exists")
    selection_path = _source_file(source, "selected-ids.json")
    records_path = _source_file(source, "source-records.jsonl")
    selection_payload = selection_path.read_bytes()
    records_payload = records_path.read_bytes()
    if hashlib.sha256(selection_payload).hexdigest() != SELECTION_SHA256:
        raise ContractError("selected-ids SHA-256 mismatch")
    if hashlib.sha256(records_payload).hexdigest() != RECORDS_SHA256:
        raise ContractError("source-records SHA-256 mismatch")
    try:
        selected = json.loads(selection_payload)
        records = [json.loads(line) for line in records_payload.splitlines()]
    except (UnicodeError, ValueError) as exc:
        raise ContractError("invalid source JSON") from exc
    if not isinstance(selected, list) or len(selected) != 400 or len(records) != 400:
        raise ContractError("wrong source population")

    selected_by_id = {}
    counts = {task: 0 for task in CATEGORIES}
    for row in selected:
        key = _validate_selection(row)
        if key in selected_by_id:
            raise ContractError("duplicate selected ID")
        selected_by_id[key] = row

    samples = []
    audits = []
    assets = []
    seen = set()
    for raw in records:
        if not isinstance(raw, dict) or set(raw) != RECORD_KEYS:
            raise ContractError("unexpected or missing source record fields")
        selected_fields = {key: raw[key] for key in SELECTION_KEYS}
        key = _validate_selection(selected_fields)
        if key in seen:
            raise ContractError("duplicate source ID")
        seen.add(key)
        if selected_by_id.get(key) != selected_fields:
            raise ContractError("selected ID/source record mismatch")
        task, global_row = key
        question = _nonempty(raw["Question"], "Question")
        answer = _nonempty(raw["Final Answer"], "Final Answer")
        image = raw["problem_image_1"]
        if not isinstance(image, dict) or set(image) != IMAGE_KEYS:
            raise ContractError("unexpected or missing image fields")
        relative = _manifest_relative(image["path"])
        if image["source_path"] is not None and not isinstance(image["source_path"], str):
            raise ContractError("source_path must be a string or null")
        digest = image["sha256"]
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ContractError("image sha256 must be lowercase SHA-256")
        for field in ("bytes", "width", "height"):
            _row_number(image[field], field, positive=True)
        for field in ("format", "mode"):
            _nonempty(image[field], field)
        path = _source_file(source, relative)
        payload = path.read_bytes()
        if len(payload) != image["bytes"] or hashlib.sha256(payload).hexdigest() != digest:
            raise ContractError("image byte/SHA-256 mismatch")
        from PIL import Image
        try:
            with Image.open(BytesIO(payload)) as opened:
                facts = (opened.format, opened.width, opened.height, opened.mode)
                opened.load()
        except Exception as exc:
            raise ContractError("corrupt image") from exc
        if facts != (image["format"], image["width"], image["height"], image["mode"]):
            raise ContractError("image metadata mismatch")
        source_id = f"{global_row:05d}"
        target = ZebraTarget(answer)
        model_input = ModelInput(question, (relative,), ())
        validate_target(target, model_input, task)
        sample = Sample(1, f"zebra_2d/{task}/{source_id}", "zebra_2d", task,
            model_input, target, Provenance(DATASET_ID, SOURCE_REVISION, raw["config"],
            "train", source_id, {relative: digest}))
        samples.append(sample)
        assets.append((relative, payload))
        counts[task] += 1
        audits.append({**raw, "source_id": source_id})
    if seen != set(selected_by_id) or counts != {"Jigsaw": 200, "Search": 200}:
        raise ContractError("wrong source correspondence or task population")

    manifest = b"".join(canonical_json(asdict(sample)) + b"\n" for sample in samples)
    summary = {"N_expected": 400, "categories": counts, "image_count": 400,
        "excluded_rows": 0, "dataset_id": DATASET_ID, "revision": SOURCE_REVISION,
        "dataset_profile": DATASET_PROFILE, "split": "train",
        "dataset_root": (Path("data") / destination.relative_to(data_root)).as_posix(),
        "selection_sha256": SELECTION_SHA256, "records_sha256": RECORDS_SHA256,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "protocol_kind": "modified", "paper_protocol_verified": False, "ood_verified": "unknown"}
    destination.mkdir(parents=True, exist_ok=False)
    for relative, payload in assets:
        target_path = _inside(destination / relative, destination, "destination image must stay inside output")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(payload)
    (destination / "manifest.jsonl").write_bytes(manifest)
    (destination / "source-selection.json").write_bytes(canonical_json({"schema_version": 1, "rows": audits}) + b"\n")
    (destination / "SOURCE.json").write_bytes(canonical_json({"schema_version": 1, **summary,
        "source_root": source.relative_to(data_root).as_posix()}) + b"\n")
    return summary
