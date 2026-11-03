import base64
import json
from pathlib import Path
import shutil
import uuid

import pytest


BROKEN_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/EH//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/EH//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/EH//2Q=="
)

JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDi6KKK+ZP3E//Z"
)


def write_source(source: Path, selections: list[dict], records: list[dict]) -> None:
    (source / "selected-ids.json").write_text(
        json.dumps(selections, separators=(",", ":")), encoding="utf-8"
    )
    (source / "source-records.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )


def source_fixture(repo_root: Path, owner_id: str) -> Path:
    import hashlib

    source = repo_root / "data/evaluation" / f"zebra-source-fixture-{owner_id}-{uuid.uuid4().hex}"
    (source / "images").mkdir(parents=True)
    selections = []
    records = []
    configs = {
        "Jigsaw": "2D Visual Reasoning - Visual Jigsaw",
        "Search": "2D Visual Reasoning - Visual Search",
    }
    image_hash = hashlib.sha256(JPEG).hexdigest()
    for task, config in configs.items():
        for global_row in range(200):
            relative = f"images/{task.lower()}-{global_row:05d}.jpg"
            (source / relative).write_bytes(JPEG)
            selected = {
                "task": task,
                "config": config,
                "split": "train",
                "global_row": global_row,
                "shard_path": f"{config}/train-00000-of-00001.parquet",
                "shard_row": global_row,
                "selection_hash": hashlib.sha256(f"{task}:{global_row}".encode()).hexdigest(),
            }
            selections.append(selected)
            records.append({
                **selected,
                "Question": f"Synthetic {task} question {global_row} <image_start>[problem_image_1]<image_end>",
                "Final Answer": "A" if task == "Jigsaw" else f"item {global_row}",
                "problem_image_1": {
                    "path": relative,
                    "source_path": "problem_image_1.jpg",
                    "sha256": image_hash,
                    "bytes": len(JPEG),
                    "format": "JPEG",
                    "width": 1,
                    "height": 1,
                    "mode": "RGB",
                },
            })
    write_source(source, selections, records)
    return source


class SourceFactory:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.owner_id = uuid.uuid4().hex
        self.paths: list[Path] = []

    def __call__(self) -> Path:
        source = source_fixture(self.repo_root, self.owner_id)
        self.track(source)
        return source

    def track(self, *paths: Path) -> None:
        for path in map(Path, paths):
            if not any(self.owner_id in part for part in path.parts):
                raise ValueError(f"cleanup path is not owned by this fixture: {path}")
            self.paths.append(path)

    def cleanup(self) -> None:
        for path in reversed(self.paths):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()


@pytest.fixture
def zebra_source_factory():
    repo_root = Path(__file__).resolve().parents[2]
    factory = SourceFactory(repo_root)
    yield factory
    factory.cleanup()
