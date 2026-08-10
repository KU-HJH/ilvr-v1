"""Validation helpers for the CoMT JSONL dataset and referenced images."""

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


class DatasetValidationError(ValueError):
    pass


class UnsafeArchiveError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationReport:
    train_records: int
    test_records: int
    unique_images: int
    missing_images: int
    corrupt_images: int


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or member.isdev():
                raise UnsafeArchiveError(f"Unsafe archive member: {member.name}")
            if member.issym():
                link_target = (target.parent / member.linkname).resolve()
                if not link_target.is_relative_to(root):
                    raise UnsafeArchiveError(f"Unsafe archive link: {member.name}")
            if member.islnk():
                link_target = (root / member.linkname).resolve()
                if not link_target.is_relative_to(root):
                    raise UnsafeArchiveError(f"Unsafe archive link: {member.name}")
        archive.extractall(destination, members=members, filter="data")


def locate_image_root(extraction_root: Path, references: set[str]) -> Path:
    if any(Path(reference).is_absolute() for reference in references):
        raise DatasetValidationError("Image references must be relative for archive root discovery")
    candidates = [extraction_root]
    candidates.extend(
        path for path in extraction_root.iterdir() if path.is_dir()
    )
    matches = []
    for candidate in candidates:
        if all(
            (candidate / reference).resolve().is_relative_to(candidate.resolve())
            and (candidate / reference).is_file()
            for reference in references
        ):
            matches.append(candidate)
    if len(matches) != 1:
        checked = ", ".join(str(path) for path in candidates)
        raise DatasetValidationError(
            f"Unable to locate unique image root; candidates checked: {checked}"
        )
    return matches[0]


def validate_dataset(
    train_path: Path,
    test_path: Path,
    image_root: Path,
    verify_images: bool = True,
) -> ValidationReport:
    train_records, train_images = _parse_split(train_path, require_plan=True)
    test_records, test_images = _parse_split(test_path, require_plan=False)
    if train_records == 0 or test_records == 0:
        raise DatasetValidationError("TRAIN.jsonl and TEST.jsonl must both be non-empty")
    references = train_images | test_images
    missing, corrupt = _check_images(references, image_root, verify_images)
    if missing or corrupt:
        raise DatasetValidationError(
            f"Referenced image validation failed: missing={missing}, corrupt={corrupt}"
        )
    return ValidationReport(train_records, test_records, len(references), 0, 0)


def _parse_split(path: Path, require_plan: bool) -> tuple[int, set[str]]:
    images: set[str] = set()
    records = 0
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise DatasetValidationError(f"Unable to read {path.name}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"{path.name}:{line_number}: malformed JSON") from exc
            if not isinstance(record, dict):
                raise DatasetValidationError(f"{path.name}:{line_number}: record must be an object")
            _require_string(record, "text_input", path, line_number)
            image_input = record.get("image_input")
            if isinstance(image_input, str):
                images.add(image_input)
            elif isinstance(image_input, list) and all(isinstance(item, str) for item in image_input):
                images.update(image_input)
            else:
                raise DatasetValidationError(
                    f"{path.name}:{line_number}: image_input must be a string or list of strings"
                )
            if require_plan:
                plan = record.get("sequence_plan")
                if not isinstance(plan, list):
                    raise DatasetValidationError(
                        f"{path.name}:{line_number}: sequence_plan must be a list"
                    )
                for step in plan:
                    if not isinstance(step, dict):
                        raise DatasetValidationError(
                            f"{path.name}:{line_number}: sequence steps must be objects"
                        )
                    step_type = step.get("type")
                    if step_type not in {"text", "latent"}:
                        raise DatasetValidationError(
                            f"{path.name}:{line_number}: Unknown sequence step type {step_type!r}"
                        )
                    if step_type == "text":
                        if not isinstance(step.get("content"), str):
                            raise DatasetValidationError(
                                f"{path.name}:{line_number}: text sequence step requires content"
                            )
                    elif not isinstance(step.get("helper_image"), str):
                        raise DatasetValidationError(
                            f"{path.name}:{line_number}: latent sequence step requires helper_image"
                        )
                    else:
                        images.add(step["helper_image"])
            records += 1
    return records, images


def _require_string(record: dict, field: str, path: Path, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise DatasetValidationError(f"{path.name}:{line_number}: {field} must be a string")
    return value


def _check_images(references: set[str], image_root: Path, verify_images: bool) -> tuple[int, int]:
    missing = 0
    corrupt = 0
    for reference in references:
        image_path = Path(reference)
        if not image_path.is_absolute():
            image_path = image_root / image_path
        if not image_path.is_file():
            missing += 1
            continue
        if verify_images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError):
                corrupt += 1
    return missing, corrupt
