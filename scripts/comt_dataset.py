"""Validation helpers for the CoMT JSONL dataset and referenced images."""

import hashlib
import json
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, UnidentifiedImageError


SOURCE_REPO = "shuai22/comt"
SOURCE_FILES = ("TRAIN.jsonl", "TEST.jsonl", "comt.tar.gz")


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_revision(repo_id: str, requested_revision: str | None) -> str:
    info = HfApi().dataset_info(repo_id, revision=requested_revision)
    return info.sha


def download_sources(repo_id: str, revision: str, staging_dir: Path) -> dict[str, Path]:
    download_dir = staging_dir / "downloads"
    return {
        filename: Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                local_dir=download_dir,
            )
        )
        for filename in SOURCE_FILES
    }


def prepare_dataset(
    data_dir: Path,
    repo_id: str = SOURCE_REPO,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    data_dir = Path(data_dir)
    final_paths = {
        "TRAIN.jsonl": data_dir / "TRAIN.jsonl",
        "TEST.jsonl": data_dir / "TEST.jsonl",
        "images_comt": data_dir / "images_comt",
        "comt_manifest.json": data_dir / "comt_manifest.json",
    }
    existing = [path.exists() for path in final_paths.values()]
    if all(existing):
        validate_dataset(
            final_paths["TRAIN.jsonl"], final_paths["TEST.jsonl"], final_paths["images_comt"]
        )
        return final_paths["comt_manifest.json"]
    if any(existing) and not force:
        raise FileExistsError("incomplete dataset exists; use force=True to replace final artifacts")

    staging_dir = data_dir / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    resolved_revision = resolve_revision(repo_id, revision)
    sources = download_sources(repo_id, resolved_revision, staging_dir)
    extracted_dir = staging_dir / "extracted"
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    safe_extract_tar(sources["comt.tar.gz"], extracted_dir)

    train_path = sources["TRAIN.jsonl"]
    test_path = sources["TEST.jsonl"]
    _, train_references = _parse_split(train_path, require_plan=True)
    _, test_references = _parse_split(test_path, require_plan=False)
    image_root = locate_image_root(extracted_dir, train_references | test_references)

    ready_dir = staging_dir / "ready"
    if ready_dir.exists():
        shutil.rmtree(ready_dir)
    ready_dir.mkdir()
    ready_train = ready_dir / "TRAIN.jsonl"
    ready_test = ready_dir / "TEST.jsonl"
    ready_images = ready_dir / "images_comt"
    shutil.copy2(train_path, ready_train)
    shutil.copy2(test_path, ready_test)
    shutil.move(str(image_root), str(ready_images))
    report = validate_dataset(ready_train, ready_test, ready_images)

    manifest_path = ready_dir / "comt_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": {"repository": repo_id, "revision": resolved_revision},
                "files": {
                    "TRAIN.jsonl": {"sha256": sha256_file(ready_train)},
                    "TEST.jsonl": {"sha256": sha256_file(ready_test)},
                    "comt.tar.gz": {"sha256": sha256_file(sources["comt.tar.gz"])},
                },
                "records": {"train": report.train_records, "test": report.test_records},
                "images": {
                    "unique": report.unique_images,
                    "missing": report.missing_images,
                    "corrupt": report.corrupt_images,
                },
                "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if force:
        for path in final_paths.values():
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    if any(path.exists() for path in final_paths.values()):
        raise FileExistsError("final dataset artifacts appeared before promotion")
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, final_path in final_paths.items():
        shutil.move(str(ready_dir / name), str(final_path))
    return final_paths["comt_manifest.json"]


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
