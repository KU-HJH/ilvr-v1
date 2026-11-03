"""Prepare the pinned 400-row EMMA-mini evaluation source."""
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePath
import re

from evaluation.benchmarks.emma import CATEGORIES, validate_target
from evaluation.schema import ContractError, EmmaTarget, ModelInput, Provenance, Sample, canonical_json


DATASET_ID = "luckychao/EMMA-mini"
SOURCE_REVISION = "6b9ae9a74733bb57f0b741213f8d0ec9ebae067a"
DATASET_PROFILE = "emma-mini400-v1"
PARQUET_FILES = {
    "Chemistry": {"bytes": 8503247, "sha256": "b5849a452f8114d114f9d355e64a4c89c4c87f6a759d30a7be6d16650996872b"},
    "Coding": {"bytes": 25725653, "sha256": "f161007210600bf6f254d0cff8f5007bdb386529cea350ed7880f1c1305ddec0"},
    "Math": {"bytes": 7317246, "sha256": "929bee1405aaf98ab25ba082ef5dd9a67537977286ac108a757bdc7bb3110490"},
    "Physics": {"bytes": 9067825, "sha256": "8663a7e70267110f16be9609923d97e8be903a5315b6a16397f78d212ee676c5"},
}
SOURCE_KEYS = ("pid", "question", "options", "answer", "image_1", "image_2", "image_3",
               "image_4", "image_5", "solution", "subject", "task", "category", "source", "type", "context")
EXPECTED_TYPES = {"Chemistry": {"multiple_choice": 78, "open_ended": 22},
                  "Coding": {"multiple_choice": 100, "open_ended": 0},
                  "Math": {"multiple_choice": 63, "open_ended": 37},
                  "Physics": {"multiple_choice": 100, "open_ended": 0}}
TOKEN = re.compile(r"<image_(\d+)>")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _inside(path: Path, parent: Path, message: str) -> Path:
    try:
        resolved = Path(path).resolve()
        resolved.relative_to(parent)
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(message) from exc
    return resolved


def _file(source: Path, relative: str) -> Path:
    try:
        result = (source / relative).resolve(strict=True)
        result.relative_to(source)
    except (OSError, ValueError) as exc:
        raise ContractError("source file must stay inside source root") from exc
    if not result.is_file():
        raise ContractError("source file must stay inside source root")
    return result


def _string(value, field: str, *, empty=False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ContractError(f"{field} must be a string")
    return value


def _image(cell, pid: str, slot: int) -> tuple[bytes, str]:
    if not isinstance(cell, dict) or set(cell) != {"bytes", "path"} or not isinstance(cell["bytes"], bytes) or not cell["bytes"]:
        raise ContractError("invalid image cell")
    from PIL import Image
    try:
        with Image.open(BytesIO(cell["bytes"])) as image:
            fmt = image.format
            image.verify()
    except Exception as exc:
        raise ContractError("corrupt image bytes") from exc
    extension = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}.get(fmt)
    if extension is None:
        raise ContractError("unsupported image format")
    return cell["bytes"], f"images/{pid}/image_{slot}.{extension}"


def prepare_emma(source_root: Path, output: Path, *, repo_root: Path) -> dict:
    data_root = (Path(repo_root).resolve() / "data").resolve()
    source = _inside(source_root, data_root, "source path must resolve inside repo_root/data")
    destination = _inside(output, data_root, "output path must resolve inside repo_root/data")
    if destination.exists():
        raise ContractError("output already exists")
    metadata_path = _file(source, "EMMA-mini-dataset-info.json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ContractError("invalid source metadata") from exc
    if not isinstance(metadata, dict) or metadata.get("id") != DATASET_ID or metadata.get("sha") != SOURCE_REVISION:
        raise ContractError("source identity or revision mismatch")

    import pyarrow.parquet as pq
    source_files = {}
    source_rows = []
    for subject in CATEGORIES:
        relative = f"EMMA-mini/{subject}/test-00000-of-00001.parquet"
        path = _file(source, relative)
        payload = path.read_bytes()
        actual = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        if actual != PARQUET_FILES[subject]:
            raise ContractError(f"{subject} parquet byte/SHA-256 mismatch")
        try:
            rows = pq.read_table(path).to_pylist()
        except Exception as exc:
            raise ContractError(f"invalid {subject} parquet") from exc
        source_files[subject] = {"path": relative, **actual}
        source_rows.append((subject, rows))

    samples = []
    audits = []
    assets = []
    ids = set()
    type_counts = {subject: {"multiple_choice": 0, "open_ended": 0} for subject in CATEGORIES}
    for expected_subject, rows in source_rows:
        if len(rows) != 100:
            raise ContractError("wrong subject population")
        for source_row, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != set(SOURCE_KEYS):
                raise ContractError("unexpected or missing fields")
            pid = _string(row["pid"], "pid")
            if not SAFE_ID.fullmatch(pid) or PurePath(pid).name != pid:
                raise ContractError("unsafe pid")
            if pid in ids:
                raise ContractError("duplicate pid")
            ids.add(pid)
            subject = _string(row["subject"], "subject")
            if subject != expected_subject:
                raise ContractError("unknown or mismatched subject")
            question = _string(row["question"], "question")
            context = _string(row["context"], "context", empty=True)
            answer = _string(row["answer"], "answer")
            category = _string(row["category"], "category", empty=True)
            task = _string(row["task"], "task", empty=True)
            _string(row["source"], "source", empty=True)
            if row["solution"] is not None and not isinstance(row["solution"], str):
                raise ContractError("solution must be a string or null")
            raw_kind = _string(row["type"], "type").lower()
            if raw_kind not in ("multiple choice", "open-ended"):
                raise ContractError("unknown question type")
            kind = "multiple_choice" if raw_kind == "multiple choice" else "open_ended"
            options = row["options"]
            if kind == "multiple_choice":
                if not isinstance(options, list) or not 2 <= len(options) <= 8 or any(not isinstance(x, str) or not x for x in options):
                    raise ContractError("invalid multiple choice options")
                choices = tuple(options)
                index = ord(answer.upper()) - ord("A") if len(answer) == 1 else -1
                if index not in range(len(choices)):
                    raise ContractError("invalid multiple choice answer")
                content = choices[index]
            else:
                if options not in (None, []):
                    raise ContractError("open-ended options must be null or empty")
                choices = (); index = None; content = answer
            type_counts[subject][kind] += 1
            slots = []
            asset_paths = []
            asset_hashes = {}
            for slot in range(1, 6):
                cell = row[f"image_{slot}"]
                if cell is None:
                    continue
                payload, asset_path = _image(cell, pid, slot)
                slots.append(slot); asset_paths.append(asset_path)
                asset_hashes[asset_path] = hashlib.sha256(payload).hexdigest()
                assets.append((asset_path, payload))
            referenced = {int(value) for value in TOKEN.findall("\n".join([question, context, *choices]))}
            if not referenced <= set(slots):
                raise ContractError("missing referenced image slot")
            text = context + "\n\n" + question if context else question
            model_input = ModelInput(text, tuple(asset_paths), choices)
            target = EmmaTarget(kind, answer, index, content, category, task)
            validate_target(target, model_input)
            sample = Sample(1, pid, "emma", subject, model_input, target,
                Provenance(DATASET_ID, SOURCE_REVISION, subject, "test", pid, asset_hashes))
            samples.append(sample)
            audits.append({"source_id": pid, "subject": subject, "source_row": source_row,
                "question": question, "context": context, "options": options, "answer": answer,
                "type": row["type"], "category": category, "task": task, "source": row["source"],
                "solution": row["solution"], "image_slots": slots})
    if type_counts != EXPECTED_TYPES:
        raise ContractError("wrong subject/type population")
    if len(assets) != 614:
        raise ContractError("wrong image population")

    manifest = b"".join(canonical_json(asdict(sample)) + b"\n" for sample in samples)
    summary = {"N_expected": 400, "categories": {subject: 100 for subject in CATEGORIES},
        "question_types": {"multiple_choice": 341, "open_ended": 59}, "image_count": 614,
        "excluded_rows": 0, "dataset_id": DATASET_ID, "revision": SOURCE_REVISION,
        "dataset_profile": DATASET_PROFILE, "split": "test",
        "dataset_root": (Path("data") / destination.relative_to(data_root)).as_posix(), "source_files": source_files,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(), "protocol_kind": "modified",
        "paper_protocol_verified": False, "ood_verified": "unknown"}
    destination.mkdir(parents=True, exist_ok=False)
    for relative, payload in assets:
        path = destination / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
    (destination / "manifest.jsonl").write_bytes(manifest)
    (destination / "source-selection.json").write_bytes(canonical_json({"schema_version": 1, "rows": audits}) + b"\n")
    source_relative = str(source.relative_to(data_root))
    (destination / "SOURCE.json").write_bytes(canonical_json({"schema_version": 1, **summary, "source_root": source_relative}) + b"\n")
    return summary
