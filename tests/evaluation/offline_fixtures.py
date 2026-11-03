from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import uuid

from evaluation.schema import RunMetadata, run_fingerprint


def make_case(sample_count=2):
    repo_root = Path(__file__).resolve().parents[2]
    run_id = "p1-test-" + uuid.uuid4().hex
    data_root = repo_root / "data/evaluation" / run_id
    result_root = repo_root / "results/evaluation" / run_id
    data_root.mkdir(parents=True, exist_ok=False)
    result_root.mkdir(parents=True, exist_ok=False)
    run = RunMetadata(
        "synthetic:model", "synthetic:template", {"max_new_tokens": 8}
    )
    samples = []
    for index in range(sample_count):
        source_id = f"s{index}"
        samples.append(
            dict(
                schema_version=1,
                id=f"vsp/planning/{source_id}",
                benchmark="vsp",
                subtask="planning",
                model_input=dict(
                    text="Synthetic question", images=[], choices=[]
                ),
                target=dict(map_desc=[[1, 2], [0, 0]], max_steps=100),
                provenance=dict(
                    dataset_id="synthetic:vsp-contract",
                    revision="v1",
                    config="tiny",
                    split="fixture",
                    source_id=source_id,
                    asset_sha256={},
                ),
            )
        )
    manifest = data_root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in samples))
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    predictions = [
        dict(
            id=row["id"],
            manifest_sha256=manifest_hash,
            run_fingerprint=run_fingerprint(run),
            response="R",
            status="ok",
        )
        for row in samples
    ]
    prediction_path = result_root / "predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(row) + "\n" for row in predictions)
    )
    config = dict(
        schema_version=1,
        benchmark="vsp",
        profile="vsp_validated_v1",
        categories=["planning", "planning-empty"],
        dataset_root=str(data_root),
        max_steps=100,
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
