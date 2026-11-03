from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import hashlib
import json
import uuid

from PIL import Image

from evaluation.schema import RunMetadata, canonical_json


SOURCE_PROMPT = "Legend: <IMAGE-1>\nExample: <IMAGE-2>\nSolve: <TEST-IMAGE>\n"
GENERATION_CONFIG = {
    "max_new_tokens": 8,
    "max_sequence_length": 32768,
    "repetition_penalty": 1.05,
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
}


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (3, 3), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _manifest_row(index: int, prompt: str, asset_hashes: dict[str, str]) -> dict:
    target_name = f"maps/level3/img/{index}.png"
    images = [
        "prompt-visual-images/system-figure-1.png",
        "prompt-visual-images/system-figure-2.png",
        target_name,
    ]
    return {
        "schema_version": 1,
        "id": f"vsp/level3/{index}",
        "benchmark": "vsp",
        "subtask": "level3",
        "model_input": {"text": prompt, "images": images, "choices": []},
        "target": {
            "map_desc": [[1, 2, 0], [0, 0, 0], [0, 0, 0]],
            "max_steps": 100,
        },
        "provenance": {
            "dataset_id": "synthetic:vsp-fixture",
            "revision": "fixture-v1",
            "config": "level3",
            "split": "repository-maps",
            "source_id": str(index),
            "asset_sha256": {name: asset_hashes[name] for name in images},
        },
    }


def make_vsp_case(sample_count: int = 2) -> dict:
    from evaluation.vsp_prepare import LEVELS, build_prompt, prompt_identity

    if not 1 <= sample_count <= 100:
        raise ValueError("sample_count must be between 1 and 100")
    root = Path(__file__).resolve().parents[2]
    run_id = "vsp-test-" + uuid.uuid4().hex
    data_root = root / "data/evaluation" / run_id
    result_root = root / "results/evaluation" / run_id
    staging_root = root / "checkpoints/evaluation-staging" / run_id
    data_root.mkdir(parents=True, exist_ok=False)
    result_root.mkdir(parents=True, exist_ok=False)

    prompt = build_prompt(SOURCE_PROMPT)
    _write(data_root / "prompt-text/ilvr-three-images-v1.txt", prompt.encode())
    assets = {
        "prompt-visual-images/system-figure-1.png": _png_bytes((255, 0, 0)),
        "prompt-visual-images/system-figure-2.png": _png_bytes((0, 255, 0)),
    }
    for index in range(sample_count):
        assets[f"maps/level3/img/{index}.png"] = _png_bytes((0, 0, 255))
    for relative, payload in assets.items():
        _write(data_root / relative, payload)
    hashes = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in assets.items()
    }
    samples = [_manifest_row(index, prompt, hashes) for index in range(sample_count)]
    manifest = data_root / "manifest.jsonl"
    manifest.write_bytes(b"".join(canonical_json(row) + b"\n" for row in samples))

    run = RunMetadata(
        "synthetic:model", prompt_identity(prompt), GENERATION_CONFIG.copy()
    )
    config = {
        "schema_version": 1,
        "benchmark": "vsp",
        "profile": "vsp_validated_v1",
        "categories": list(LEVELS),
        "dataset_root": str(data_root),
        "max_steps": 100,
        "run_metadata": asdict(run),
    }
    model_dir = staging_root / "model"
    base_model_dir = staging_root / "base-model"
    for path, payload in (
        (model_dir / "config.json", {"model_type": "synthetic-vsp-fixture"}),
        (base_model_dir / "config.json", {"model_type": "synthetic-vsp-base-fixture"}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "root": root,
        "data_root": data_root,
        "result_root": result_root,
        "run_id": run_id,
        "manifest": manifest,
        "samples": samples,
        "config": config,
        "model_dir": model_dir,
        "base_model_dir": base_model_dir,
    }


def write_source_fixture(source_root: Path) -> dict:
    task_root = source_root / "VSP-main/maze/task-main"
    _write(task_root / "prompt-text/prompt-text.txt", SOURCE_PROMPT.encode())
    _write(
        task_root / "prompt-visual-images/system-figure-1.png",
        _png_bytes((255, 0, 0)),
    )
    _write(
        task_root / "prompt-visual-images/system-figure-2.png",
        _png_bytes((0, 255, 0)),
    )
    _write(source_root / "LICENSE", b"Synthetic MIT fixture license\n")
    for size in range(3, 9):
        header = (
            "| | "
            + " | ".join(f"Col {column}" for column in range(1, size + 1))
            + " |"
        )
        rows = []
        for row in range(1, size + 1):
            cells = ["_" for _ in range(size)]
            if row == 1:
                cells[0:2] = ["@", "*"]
            rows.append(f"| Row {row} | " + " | ".join(cells) + " |")
        table = "\n".join([header, *rows])
        description = (
            f"This is a {size}x{size} map.\n"
            "The player is at: row 1, column 1;\n"
            "There is no holes in this map;\n"
            "The goal is at: Row 1, Column 2."
        )
        image = _png_bytes((0, 0, min(255, size * 30)))
        for index in range(100):
            base = task_root / "maps" / f"level{size}"
            _write(base / "table" / f"{index}.txt", table.encode())
            _write(base / "pure_text" / f"{index}.txt", description.encode())
            _write(base / "img" / f"{index}.png", image)
    return {
        "name": "synthetic:vsp-fixture",
        "upstream_url": "synthetic:vsp-fixture",
        "commit": "fixture-v1",
        "license_files": [
            {
                "path": "LICENSE",
                "sha256": hashlib.sha256(b"Synthetic MIT fixture license\n").hexdigest(),
            }
        ],
        "evidence_files": [],
    }
