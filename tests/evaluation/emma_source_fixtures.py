"""Synthetic pinned-shape EMMA source data for preparation tests."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import uuid

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


SUBJECTS = ("Chemistry", "Coding", "Math", "Physics")


def _image_bytes(fmt: str) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), (17, 34, 51)).save(buffer, format=fmt)
    return buffer.getvalue()


def write_subject(case: dict, subject: str) -> None:
    path = case["source"] / "EMMA-mini" / subject / "test-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(case["rows"][subject]), path)


def rewrite_all(case: dict) -> None:
    for subject in SUBJECTS:
        write_subject(case, subject)


def make_source() -> dict:
    workspace = Path(__file__).resolve().parents[2]
    run_id = "p5-emma-source-" + uuid.uuid4().hex
    base = workspace / "data/evaluation" / run_id
    root = base / "repo"
    root.mkdir(parents=True)
    (base / "repo-data/evaluation").mkdir(parents=True)
    (root / "data").symlink_to(base / "repo-data", target_is_directory=True)
    for name in ("results", "logs", "checkpoints"):
        (root / name).symlink_to(workspace / name, target_is_directory=True)
    source = root / "data/evaluation/source"
    source.mkdir()
    (source / "EMMA-mini-dataset-info.json").write_text(json.dumps({
        "id": "luckychao/EMMA-mini",
        "sha": "6b9ae9a74733bb57f0b741213f8d0ec9ebae067a",
    }))
    blobs = (_image_bytes("PNG"), _image_bytes("JPEG"), _image_bytes("WEBP"))
    rows = {subject: [] for subject in SUBJECTS}
    flat = []
    for subject in SUBJECTS:
        open_count = 22 if subject == "Chemistry" else 37 if subject == "Math" else 0
        prefix = subject[:3].lower()
        for index in range(100):
            pid = f"{prefix}_{index}"
            is_open = index >= 100 - open_count
            choice_count = (2, 3, 4, 5, 7, 8)[index % 6]
            options = None if is_open and index % 2 else [] if is_open else [f"choice {n}" for n in range(choice_count)]
            answer = f"open answer {pid}" if is_open else chr(ord("A") + index % choice_count)
            if subject == "Physics" and index < 21:
                answer = answer.lower()
            row = {
                "pid": pid, "question": f"Question {pid}", "options": options,
                "answer": answer, "image_1": None, "image_2": None, "image_3": None,
                "image_4": None, "image_5": None, "solution": "private solution",
                "subject": subject, "task": f"task-{subject}", "category": f"category-{subject}",
                "source": "synthetic", "type": "open-ended" if is_open else "multiple choice",
                "context": f"Context {pid}" if index % 2 else "",
            }
            rows[subject].append(row); flat.append(row)
    counts = [5] * 25 + [4] * 26 + [2] * 36 + [1] * 313
    phy6 = rows["Physics"][6]
    swap = next(i for i, row in enumerate(flat) if counts[i] == 2)
    phy_index = flat.index(phy6)
    counts[swap], counts[phy_index] = counts[phy_index], counts[swap]
    for position, (row, count) in enumerate(zip(flat, counts)):
        slots = [2, 3] if row["pid"] == "phy_6" else list(range(1, count + 1))
        for slot in slots:
            payload = blobs[0] if row["pid"] == "phy_6" else blobs[(position + slot) % len(blobs)]
            row[f"image_{slot}"] = {"bytes": payload, "path": "ignored"}
        if slots:
            row["question"] += " " + "".join(f"<image_{slot}>" for slot in slots)
    rows["Chemistry"][0]["options"][0] = "x" * 513
    case = {"root": root, "source": source, "output": root / "data/evaluation" / run_id,
            "rows": rows, "assets": blobs, "run_id": run_id}
    rewrite_all(case)
    return case


def pin_source(monkeypatch, case: dict) -> None:
    from evaluation import emma_prepare
    facts = {}
    for subject in SUBJECTS:
        path = case["source"] / "EMMA-mini" / subject / "test-00000-of-00001.parquet"
        payload = path.read_bytes()
        facts[subject] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(emma_prepare, "PARQUET_FILES", facts)
