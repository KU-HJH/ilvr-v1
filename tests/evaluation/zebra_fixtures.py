"""Independent synthetic Zebra 2D fixtures."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import uuid

import pytest

from evaluation.schema import (
    ModelInput,
    Provenance,
    RunMetadata,
    Sample,
    ZebraTarget,
    run_fingerprint,
)


def sample(subtask="Search", answer="knife", source_id="00000") -> Sample:
    image = f"images/{source_id}.jpg"
    return Sample(
        1,
        f"zebra_2d/{subtask}/{source_id}",
        "zebra_2d",
        subtask,
        ModelInput("Synthetic question <image_start>[problem_image_1]<image_end>", (image,), ()),
        ZebraTarget(answer),
        Provenance("synthetic:zebra", "fixture", subtask, "train", source_id, {image: "0" * 64}),
    )


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def make_case(owner, subtasks=("Jigsaw", "Search")) -> dict:
    root = owner.root
    run_id = f"zebra-fixture-{owner.owner_id}-{uuid.uuid4().hex}"
    data = root / "data/evaluation" / run_id
    results = root / "results/evaluation" / run_id
    (data / "images").mkdir(parents=True)
    results.mkdir(parents=True)
    owner.track(data, results)
    answers = {"Jigsaw": "B", "Search": "knife"}
    responses = {"Jigsaw": "B", "Search": "Knife."}
    samples = []
    for index, subtask in enumerate(subtasks):
        source_id = f"{index:05d}"
        row = json.loads(json.dumps(asdict(sample(subtask, answers[subtask], source_id))))
        image_name = row["model_input"]["images"][0]
        payload = f"synthetic-{subtask}".encode()
        (data / image_name).write_bytes(payload)
        row["provenance"]["asset_sha256"][image_name] = hashlib.sha256(payload).hexdigest()
        samples.append(row)
    manifest = data / "manifest.jsonl"
    write_rows(manifest, samples)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    metadata = RunMetadata("external:model", "zebra-2d400-v1", {})
    prediction_rows = [
        dict(id=row["id"], manifest_sha256=manifest_hash,
             run_fingerprint=run_fingerprint(metadata), response=responses[row["subtask"]], status="ok")
        for row in samples
    ]
    predictions = results / "predictions.jsonl"
    write_rows(predictions, prediction_rows)
    config = dict(schema_version=1, benchmark="zebra_2d", profile="zebra_2d_rules_v1",
                  categories=list(dict.fromkeys(subtasks)), dataset_root=str(data),
                  run_metadata=asdict(metadata))
    return dict(root=root, data=data, results=results, manifest=manifest, predictions=predictions,
                samples=samples, prediction_rows=prediction_rows, config=config)


class CaseFactory:
    def __init__(self, root: Path):
        self.root = root
        self.owner_id = uuid.uuid4().hex
        self.paths: list[Path] = []

    def __call__(self, subtasks=("Jigsaw", "Search")) -> dict:
        return make_case(self, subtasks)

    def track(self, *paths: Path) -> None:
        for path in map(Path, paths):
            if not any(self.owner_id in part for part in path.parts):
                raise ValueError(f"cleanup path is not owned by this fixture: {path}")
            self.paths.append(path)

    def cleanup(self) -> None:
        import shutil
        for path in reversed(self.paths):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()


@pytest.fixture
def zebra_case_factory():
    factory = CaseFactory(Path(__file__).resolve().parents[2])
    yield factory
    factory.cleanup()
