from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import uuid

from PIL import Image


CATEGORIES = ("creation", "deletion", "update", "selection")
SOURCE_REVISION = "bde314be94942a7b4ab66581e7f7c4801b84e686"


def _png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), (17, 34, 51)).save(buffer, format="PNG")
    return buffer.getvalue()


PNG = _png()


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_comt_source() -> dict:
    worktree = Path(__file__).resolve().parents[2]
    fixture_parent = worktree / "data/evaluation" / (
        "comt-prepare-fixture-" + uuid.uuid4().hex
    )
    root = fixture_parent / "repo"
    root.mkdir(parents=True, exist_ok=False)
    fixture_data = fixture_parent / "repo-data"
    fixture_data.mkdir()
    os.symlink(fixture_data, root / "data", target_is_directory=True)
    source_root = root / "data/evaluation/comt-source"
    selection_manifest = root / "data/TEST.jsonl"
    image_root = root / "data/images_comt"
    output = root / "data/evaluation/comt-prepared"
    (source_root / "comt").mkdir(parents=True, exist_ok=False)
    selection_manifest.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=False)
    for storage in ("results", "logs", "checkpoints"):
        os.symlink(worktree / storage, root / storage, target_is_directory=True)

    source_rows = []
    selection_rows = []
    selected_source_ids = []
    labels = "ABCD"
    for category_index, category in enumerate(CATEGORIES):
        (image_root / category).mkdir()
        for index in range(100):
            if category == "creation" and index == 57:
                source_id = "creation-12125"
                token = "12125"
                options = ["17", "", "18", "", "22", "", "34", ""]
                label = "B"
            elif category == "creation" and index == 93:
                source_id = "creation-12164"
                token = "12164"
                options = ["10cm", "12cm", "16", "", "24", ""]
                label = "D"
            else:
                token = f"{category_index + 1}{index:04d}"
                source_id = f"{category}-{token}"
                options = [f"{source_id}-A", f"{source_id}-B", f"{source_id}-C", f"{source_id}-D"]
                label = labels[index % 4]
            question = f"Synthetic question for {source_id} [IMAGE0]"
            image = {
                "IMAGE0": token,
                "IMAGE1": f"HELPER_IMAGE_SENTINEL_{source_id}",
            }
            source_rows.append(
                {
                    "id": source_id,
                    "question": question,
                    "option": options,
                    "image": image,
                    "rationale": f"RATIONALE_SENTINEL_{source_id}",
                    "answer": label,
                    "type": category,
                    "annotations": f"ANNOTATIONS_SENTINEL_{source_id}",
                }
            )
            image_path = image_root / category / f"{token}.png"
            image_path.write_bytes(PNG)
            text = question.replace("[IMAGE0]", "").strip() + " option:" + str(options)
            selection_rows.append(
                {
                    "text_input": text,
                    "image_input": [f"images_comt/{category}/{token}.png"],
                    "sequence_plan": [
                        {"type": "text", "content": f"PLAN_SENTINEL_{source_id}"},
                        {
                            "type": "latent",
                            "helper_image": f"HELPER_PLAN_SENTINEL_{source_id}",
                        },
                    ],
                    "original_final_answer": str(options[labels.index(label)]),
                }
            )
            selected_source_ids.append(source_id)

    source_path = source_root / "comt/data.jsonl"
    _jsonl(source_path, list(reversed(source_rows)))
    _jsonl(selection_manifest, selection_rows)
    (source_root / "dataset-info.json").write_text(
        json.dumps({"id": "czh-up/CoMT", "sha": SOURCE_REVISION}),
        encoding="utf-8",
    )
    return {
        "root": root,
        "source_root": source_root,
        "selection_manifest": selection_manifest,
        "image_root": image_root,
        "output": output,
        "source_rows": source_rows,
        "selection_rows": selection_rows,
        "selected_source_ids": selected_source_ids,
        "source_sha256": _sha256(source_path),
        "selection_sha256": _sha256(selection_manifest),
    }


def pin_fixture(monkeypatch, case: dict) -> None:
    from evaluation import comt_prepare

    monkeypatch.setattr(comt_prepare, "SOURCE_SHA256", case["source_sha256"])
    monkeypatch.setattr(comt_prepare, "SELECTION_SHA256", case["selection_sha256"])
