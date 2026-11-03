"""Prepare the pinned Visual-Spatial-Planning maze data for ILVR evaluation."""

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess

from evaluation.benchmarks.vsp import normalize_map
from evaluation.schema import (
    ContractError,
    ModelInput,
    Provenance,
    Sample,
    VspTarget,
    canonical_json,
)


LEVELS = tuple(f"level{n}" for n in range(3, 9))
COMMON_IMAGES = (
    "prompt-visual-images/system-figure-1.png",
    "prompt-visual-images/system-figure-2.png",
)
SOURCE_COMMIT = "6d74d7bb46ae71bf41aa4d3d874a4f31fc5f0366"
TASK_ROOT = Path("VSP-main/maze/task-main")
SOURCE_PROMPT = Path("prompt-text/prompt-text.txt")
PROCESSED_PROMPT = Path("prompt-text/ilvr-three-images-v1.txt")
SYMBOLS = {"_": 0, "#": -1, "@": 1, "*": 2}


def _table_cells(line: str) -> list[str]:
    if not line.startswith("|") or not line.endswith("|"):
        raise ContractError("malformed map table row")
    return [cell.strip() for cell in line[1:-1].split("|")]


def _coordinate(match: re.Match[str], size: int) -> tuple[int, int]:
    coordinate = (int(match.group(1)), int(match.group(2)))
    if not (1 <= coordinate[0] <= size and 1 <= coordinate[1] <= size):
        raise ContractError("coordinate is outside map bounds")
    return coordinate


def parse_map_pair(
    table_text: str, description: str, size: int
) -> tuple[tuple[int, ...], ...]:
    if type(size) is not int or size <= 0:
        raise ContractError("map size must be a positive integer")
    lines = table_text.splitlines()
    if len(lines) != size + 1:
        raise ContractError("map table dimensions do not match level")
    header = _table_cells(lines[0])
    if header != ["", *(f"Col {column}" for column in range(1, size + 1))]:
        raise ContractError("map table has malformed column labels")

    grid = []
    for number, line in enumerate(lines[1:], 1):
        cells = _table_cells(line)
        if len(cells) != size + 1 or cells[0] != f"Row {number}":
            raise ContractError("map table has malformed row labels")
        try:
            grid.append(tuple(SYMBOLS[cell] for cell in cells[1:]))
        except KeyError as exc:
            raise ContractError("map table has an invalid symbol") from exc
    result = tuple(grid)
    normalize_map(VspTarget(result, 100))

    detail_lines = description.splitlines()
    if len(detail_lines) != 4:
        raise ContractError("malformed map description")
    dimension = re.fullmatch(r"This is a ([1-9][0-9]*)x([1-9][0-9]*) map\.", detail_lines[0])
    player = re.fullmatch(
        r"The player is at: row ([1-9][0-9]*), column ([1-9][0-9]*);",
        detail_lines[1],
    )
    goal = re.fullmatch(
        r"The goal is at: Row ([1-9][0-9]*), Column ([1-9][0-9]*)\.",
        detail_lines[3],
    )
    if not dimension or int(dimension.group(1)) != size or int(dimension.group(2)) != size:
        raise ContractError("description dimensions do not match level")
    if not player or not goal:
        raise ContractError("malformed player or goal coordinate")
    player_coordinate = _coordinate(player, size)
    goal_coordinate = _coordinate(goal, size)

    holes_line = detail_lines[2]
    if holes_line == "There is no holes in this map;":
        holes: list[tuple[int, int]] = []
    else:
        prefix = "The hole(s) are at: "
        if not holes_line.startswith(prefix) or not holes_line.endswith(";"):
            raise ContractError("malformed hole coordinates")
        payload = holes_line[len(prefix) : -1]
        parts = payload.split("; ")
        holes = []
        for part in parts:
            match = re.fullmatch(r"Row ([1-9][0-9]*), Column ([1-9][0-9]*)", part)
            if not match:
                raise ContractError("malformed hole coordinates")
            holes.append(_coordinate(match, size))

    coordinates = [player_coordinate, goal_coordinate, *holes]
    if len(set(coordinates)) != len(coordinates):
        raise ContractError("map description contains duplicate coordinates")
    described = [[0 for _ in range(size)] for _ in range(size)]
    for row, column in holes:
        described[row - 1][column - 1] = -1
    described[player_coordinate[0] - 1][player_coordinate[1] - 1] = 1
    described[goal_coordinate[0] - 1][goal_coordinate[1] - 1] = 2
    if result != tuple(tuple(row) for row in described):
        raise ContractError("map table and description disagree")
    return result


def build_prompt(source_text: str) -> str:
    intro = (
        "The three images are provided in this order: Image 1 (legend), "
        "Image 2 (worked example), Image 3 (maze to solve).\n\n"
    )
    for marker in ("<IMAGE-1>", "<IMAGE-2>", "<TEST-IMAGE>"):
        if source_text.count(marker) != 1:
            raise ContractError("expected one of each source image marker")
    return (
        intro
        + source_text.replace("<IMAGE-1>", "Image 1 (legend)")
        .replace("<IMAGE-2>", "Image 2 (worked example)")
        .replace("<TEST-IMAGE>", "Image 3 (maze to solve)")
    )


def prompt_identity(prompt: str) -> str:
    return "vsp-ilvr-three-images-v1:" + hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()


def _run_git(source_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("cannot verify source Git checkout") from exc
    return completed.stdout.strip()


def _source_file(source_root: Path, relative: str | Path) -> Path:
    try:
        root = source_root.resolve(strict=True)
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError("source path must stay inside source checkout") from exc
    if not candidate.is_file():
        raise ContractError("source path must be a regular file")
    return candidate


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ContractError("cannot read source file") from exc


def verify_source_reference(source_root: Path, repo_root: Path) -> dict:
    try:
        source_root = Path(source_root).resolve(strict=True)
        repository = Path(repo_root).resolve(strict=True)
        lock = json.loads(
            (repository / "references.lock.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError("cannot read reference lock") from exc
    if not isinstance(lock, dict) or not isinstance(lock.get("repositories"), list):
        raise ContractError("reference lock is malformed")
    entries = [
        entry
        for entry in lock.get("repositories", [])
        if isinstance(entry, dict) and entry.get("name") == "Visual-Spatial-Planning"
    ]
    if len(entries) != 1:
        raise ContractError("reference lock must contain one VSP repository")
    entry = entries[0]
    if entry.get("commit") != SOURCE_COMMIT:
        raise ContractError("VSP source commit is not pinned")
    if (
        Path(_run_git(source_root, "rev-parse", "--show-toplevel")).resolve()
        != source_root
    ):
        raise ContractError("source root is not the Git checkout root")
    if (
        _run_git(source_root, "remote", "get-url", "origin")
        != entry.get("upstream_url")
    ):
        raise ContractError("source origin does not match reference lock")
    if _run_git(source_root, "rev-parse", "HEAD") != SOURCE_COMMIT:
        raise ContractError("source HEAD does not match reference lock")
    if _run_git(source_root, "status", "--porcelain", "--untracked-files=all"):
        raise ContractError("source checkout is not clean")
    for field in ("license_files", "evidence_files"):
        records = entry.get(field)
        if not isinstance(records, list):
            raise ContractError("reference lock contains malformed file records")
        for record in records:
            if not isinstance(record, dict) or not {"path", "sha256"} <= set(record):
                raise ContractError("reference lock contains malformed file record")
            if _digest(_source_file(source_root, record["path"])) != record["sha256"]:
                raise ContractError("source evidence SHA-256 mismatch")
    return {
        key: entry[key]
        for key in ("name", "upstream_url", "commit", "license_files", "evidence_files")
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ContractError("cannot read source text") from exc


def _verify_png(path: Path) -> None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ContractError("source image is not a valid image") from exc


def _exact_files(
    directory: Path, expected: set[str], source_root: Path
) -> dict[str, Path]:
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        raise ContractError("cannot read source map directory") from exc
    if {entry.name for entry in entries} != expected:
        raise ContractError("source map filenames do not match expected IDs")
    checked = {}
    for entry in entries:
        try:
            resolved = entry.resolve(strict=True)
            resolved.relative_to(source_root)
        except (OSError, ValueError) as exc:
            raise ContractError("source path must stay inside source checkout") from exc
        if not resolved.is_file():
            raise ContractError("source map filenames do not match expected IDs")
        checked[entry.name] = resolved
    return checked


def _validate_source_identity(source_identity: dict) -> None:
    required = {"name", "upstream_url", "commit", "license_files", "evidence_files"}
    if not isinstance(source_identity, dict) or not required <= set(source_identity):
        raise ContractError("invalid source identity")
    if not all(
        isinstance(source_identity[key], str) and source_identity[key]
        for key in ("name", "upstream_url", "commit")
    ):
        raise ContractError("invalid source identity")
    if (
        not isinstance(source_identity["license_files"], list)
        or not source_identity["license_files"]
    ):
        raise ContractError("source identity requires a license")
    if not isinstance(source_identity["evidence_files"], list):
        raise ContractError("invalid source evidence")


def prepare_vsp(
    source_root: Path, dataset_root: Path, *, source_identity: dict
) -> dict:
    _validate_source_identity(source_identity)
    source_root = Path(source_root).resolve()
    dataset_root = Path(dataset_root)
    if dataset_root.exists():
        raise ContractError("dataset output already exists")
    try:
        task_root = (source_root / TASK_ROOT).resolve(strict=True)
        task_root.relative_to(source_root)
    except (OSError, ValueError) as exc:
        raise ContractError("source task root must stay inside source checkout") from exc
    if not task_root.is_dir():
        raise ContractError("source task root must be a directory")

    prompt_path = _source_file(task_root, SOURCE_PROMPT)
    prompt_source = _read_text(prompt_path)
    prompt = build_prompt(prompt_source)
    common_sources = {
        relative: _source_file(task_root, relative) for relative in COMMON_IMAGES
    }
    common_hashes = {}
    for relative, image in common_sources.items():
        _verify_png(image)
        common_hashes[relative] = _digest(image)

    license_records = source_identity["license_files"]
    for record in license_records:
        if not isinstance(record, dict) or not {"path", "sha256"} <= set(record):
            raise ContractError("invalid source license record")
        license_path = _source_file(source_root, record["path"])
        if _digest(license_path) != record["sha256"]:
            raise ContractError("source license SHA-256 mismatch")
    if len(license_records) != 1:
        raise ContractError("VSP preparation requires exactly one license file")
    license_source = _source_file(source_root, license_records[0]["path"])

    prepared_rows: list[
        tuple[str, int, tuple[tuple[int, ...], ...], Path, Path, Path, str]
    ] = []
    expected_text = {f"{index}.txt" for index in range(100)}
    expected_images = {f"{index}.png" for index in range(100)}
    for level in LEVELS:
        size = int(level.removeprefix("level"))
        level_root = task_root / "maps" / level
        tables = _exact_files(level_root / "table", expected_text, source_root)
        descriptions = _exact_files(
            level_root / "pure_text", expected_text, source_root
        )
        images = _exact_files(level_root / "img", expected_images, source_root)
        for index in range(100):
            table = tables[f"{index}.txt"]
            description = descriptions[f"{index}.txt"]
            image = images[f"{index}.png"]
            grid = parse_map_pair(_read_text(table), _read_text(description), size)
            _verify_png(image)
            prepared_rows.append(
                (level, index, grid, table, description, image, _digest(image))
            )

    dataset_root.mkdir(parents=True, exist_ok=False)
    (dataset_root / SOURCE_PROMPT).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prompt_path, dataset_root / SOURCE_PROMPT)
    (dataset_root / PROCESSED_PROMPT).write_text(prompt, encoding="utf-8")
    for relative, source in common_sources.items():
        destination = dataset_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(license_source, dataset_root / "LICENSE")

    samples = []
    smoke = []
    for level, index, grid, table, description, image, image_hash in prepared_rows:
        base = Path("maps") / level
        for kind, source in (
            ("table", table),
            ("pure_text", description),
            ("img", image),
        ):
            destination = dataset_root / base / kind / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        target_image = (base / "img" / f"{index}.png").as_posix()
        asset_hashes = {**common_hashes, target_image: image_hash}
        model_input = ModelInput(prompt, (*COMMON_IMAGES, target_image), ())
        sample = Sample(
            1,
            f"vsp/{level}/{index}",
            "vsp",
            level,
            model_input,
            VspTarget(grid, 100),
            Provenance(
                source_identity["name"],
                source_identity["commit"],
                level,
                "repository-maps",
                str(index),
                asset_hashes,
            ),
        )
        row = asdict(sample)
        samples.append(row)
        if index == 0:
            smoke.append(row)

    source_record = {
        "schema_version": 1,
        "repository": {
            key: source_identity[key] for key in ("name", "upstream_url", "commit")
        },
        "source_paths": {
            "task_root": TASK_ROOT.as_posix(),
            "prompt": SOURCE_PROMPT.as_posix(),
            "processed_prompt": PROCESSED_PROMPT.as_posix(),
            "common_images": list(COMMON_IMAGES),
            "maps": "maps",
        },
        "license_files": source_identity["license_files"],
        "evidence_files": source_identity["evidence_files"],
        "protocol_kind": "modified",
    }
    (dataset_root / "SOURCE.json").write_bytes(canonical_json(source_record) + b"\n")
    manifest_payload = b"".join(canonical_json(row) + b"\n" for row in samples)
    smoke_payload = b"".join(canonical_json(row) + b"\n" for row in smoke)
    (dataset_root / "manifest.jsonl").write_bytes(manifest_payload)
    (dataset_root / "smoke-manifest.jsonl").write_bytes(smoke_payload)
    return {
        "N_expected": 600,
        "categories": {level: 100 for level in LEVELS},
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "smoke_manifest_sha256": hashlib.sha256(smoke_payload).hexdigest(),
        "prompt_template_identity": prompt_identity(prompt),
        "protocol_kind": "modified",
        "source_commit": source_identity["commit"],
    }
