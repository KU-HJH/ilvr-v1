from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import uuid

from evaluation.schema import RunMetadata, run_fingerprint


CATEGORIES = ("creation", "deletion", "update", "selection")


def make_comt_case(sample_count=4):
    repo_root = Path(__file__).resolve().parents[2]
    run_id = "p3-comt-test-" + uuid.uuid4().hex
    data_root = repo_root / "data/evaluation" / run_id
    result_root = repo_root / "results/evaluation" / run_id
    data_root.mkdir(parents=True, exist_ok=False)
    result_root.mkdir(parents=True, exist_ok=False)
    run = RunMetadata(
        "synthetic:comt-model", "synthetic:comt-template", {"max_new_tokens": 8}
    )
    samples = []
    for index in range(sample_count):
        category = CATEGORIES[index % len(CATEGORIES)]
        source_id = f"s{index}"
        samples.append(
            dict(
                schema_version=1,
                id=f"comt/{category}/{source_id}",
                benchmark="comt",
                subtask=category,
                model_input=dict(
                    text="Synthetic multiple-choice question",
                    images=[],
                    choices=["5", "10", "15", "18"],
                ),
                target=dict(index=1, label="B", content="10"),
                provenance=dict(
                    dataset_id="synthetic:comt-contract",
                    revision="v1",
                    config="tiny",
                    split="fixture",
                    source_id=source_id,
                    asset_sha256={},
                ),
            )
        )
    manifest = data_root / "manifest.jsonl"
    save_rows(manifest, samples)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    predictions = [
        dict(
            id=row["id"],
            manifest_sha256=manifest_hash,
            run_fingerprint=run_fingerprint(run),
            response="B",
            status="ok",
        )
        for row in samples
    ]
    prediction_path = result_root / "predictions.jsonl"
    save_rows(prediction_path, predictions)
    config = dict(
        schema_version=1,
        benchmark="comt",
        profile="comt_reference_rules_v1",
        categories=list(CATEGORIES),
        dataset_root=str(data_root),
        run_metadata=asdict(run),
    )
    return dict(
        root=repo_root,
        run_id=run_id,
        data_root=data_root,
        result_root=result_root,
        manifest=manifest,
        prediction_path=prediction_path,
        samples=samples,
        predictions=predictions,
        config=config,
    )


def save_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def save_manifest(case):
    save_rows(case["manifest"], case["samples"])
    digest = hashlib.sha256(case["manifest"].read_bytes()).hexdigest()
    for row in case["predictions"]:
        row["manifest_sha256"] = digest
    save_rows(case["prediction_path"], case["predictions"])
