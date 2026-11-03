import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import uuid

import pytest
import yaml

from comt_fixtures import make_comt_case
from comt_source_fixtures import make_comt_source, pin_fixture
from evaluation.io import parse_config


ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = ROOT / "configs/eval/comt_reference_rules_v1.yaml"
PIPELINE_FILES = (
    "scripts/__init__.py",
    "scripts/prepare_eval.py",
    "scripts/score_eval.py",
    "src/evaluation/__init__.py",
    "src/evaluation/benchmarks/__init__.py",
    "src/evaluation/benchmarks/comt.py",
    "src/evaluation/benchmarks/vsp.py",
    "src/evaluation/comt_prepare.py",
    "src/evaluation/io.py",
    "src/evaluation/registry.py",
    "src/evaluation/report.py",
    "src/evaluation/schema.py",
    "src/evaluation/vsp_prepare.py",
)
EXPECTED_MODULE_ORIGINS = {
    "evaluation",
    "evaluation.benchmarks",
    "evaluation.benchmarks.comt",
    "evaluation.benchmarks.vsp",
    "evaluation.comt_prepare",
    "evaluation.io",
    "evaluation.registry",
    "evaluation.report",
    "evaluation.schema",
    "evaluation.vsp_prepare",
    "scripts",
    "scripts.prepare_eval",
    "scripts.score_eval",
}
FIXTURE_IMAGE_SHA256 = "6dfd8f8a62305636eb28fea315599cdbeb813549ae4757252f862d18bfa53c53"
FIXTURE_MANIFEST_SHA256 = (
    "408c18046dc1862a1a70a3b854078e79508cc7aabf0d5c8f05c8472d13155773"
)
FIXTURE_SOURCE_SHA256 = (
    "5841178818018bfedb3c079d2951f80e6556ec94ff078d0366d8250741822677"
)
FIXTURE_SELECTION_SHA256 = (
    "5bd3db4d6c2d96c293813ed4940cbbce572d6529a7391fdf16f7487175b42140"
)
FIXTURE_RUN_METADATA = {
    "model_identity": "external:model",
    "prompt_template_identity": "ilvr-test400-corrected-v1",
    "generation_config": {},
}
FIXTURE_RUN_FINGERPRINT = (
    "94c9845a4b5dfb658a00d5e4ef81a0695a9b79d5bfe9d81bc9b7b20853ecb424"
)


def _expected_source_ids() -> list[str]:
    expected = []
    for category_index, category in enumerate(
        ("creation", "deletion", "update", "selection"), 1
    ):
        for index in range(100):
            if category == "creation" and index == 57:
                expected.append("creation-12125")
            elif category == "creation" and index == 93:
                expected.append("creation-12164")
            else:
                expected.append(f"{category}-{category_index}{index:04d}")
    return expected


def _make_literal_b_fixture() -> dict:
    case = make_comt_source()
    for source, selection in zip(case["source_rows"], case["selection_rows"]):
        if source["id"] != "creation-12164":
            source["answer"] = "B"
        answer_index = "ABCD".index(source["answer"])
        selection["original_final_answer"] = str(source["option"][answer_index])
    source_path = case["source_root"] / "comt/data.jsonl"
    source_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in reversed(case["source_rows"])
        ),
        encoding="utf-8",
    )
    case["selection_manifest"].write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in case["selection_rows"]
        ),
        encoding="utf-8",
    )
    case["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    case["selection_sha256"] = hashlib.sha256(
        case["selection_manifest"].read_bytes()
    ).hexdigest()
    return case


def _prepare_args(case: dict, output: Path) -> list[str]:
    return [
        "--benchmark",
        "comt",
        "--profile",
        "comt_reference_rules_v1",
        "--source-root",
        str(case["source_root"]),
        "--selection-manifest",
        str(case["selection_manifest"]),
        "--image-root",
        str(case["image_root"]),
        "--output-dir",
        str(output),
    ]


def test_comt_prepare_cli_routes_without_vsp_source_verification(monkeypatch):
    import scripts.prepare_eval as cli

    case = make_comt_source()
    pin_fixture(monkeypatch, case)
    output = case["root"] / "data/evaluation" / (
        "comt-cli-" + uuid.uuid4().hex
    )
    monkeypatch.setattr(
        cli,
        "verify_source_reference",
        lambda *_: pytest.fail("CoMT preparation entered VSP source verification"),
    )

    assert cli.main(_prepare_args(case, output), repo_root=case["root"]) == 0
    result = case["root"] / "results/evaluation" / output.name / "prepare.json"
    summary = json.loads(result.read_text(encoding="utf-8"))
    assert summary["N_expected"] == 400
    assert summary["categories"] == {
        "creation": 100,
        "deletion": 100,
        "selection": 100,
        "update": 100,
    }
    assert summary["correction_count"] == 2
    assert (output / "manifest.jsonl").is_file()
    assert "complete expected=400" in (
        case["root"] / "logs/evaluation" / output.name / "prepare.log"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("missing", ["selection-manifest", "image-root"])
def test_comt_prepare_cli_requires_comt_paths(missing):
    from scripts.prepare_eval import main

    case = make_comt_source()
    output = case["root"] / "data/evaluation" / (
        "comt-missing-" + missing + "-" + uuid.uuid4().hex
    )
    args = _prepare_args(case, output)
    position = args.index("--" + missing)
    del args[position : position + 2]

    assert main(args, repo_root=case["root"]) == 2
    assert not output.exists()
    assert not (case["root"] / "results/evaluation" / output.name).exists()
    assert not (case["root"] / "logs/evaluation" / output.name).exists()


def test_vsp_prepare_rejects_comt_only_paths_before_source_verification(monkeypatch):
    import scripts.prepare_eval as cli

    case = make_comt_source()
    output = case["root"] / "data/evaluation" / (
        "vsp-extra-" + uuid.uuid4().hex
    )
    calls = []
    monkeypatch.setattr(cli, "verify_source_reference", lambda *_: calls.append(True))
    args = _prepare_args(case, output)
    args[args.index("comt")] = "vsp"
    args[args.index("comt_reference_rules_v1")] = "vsp_validated_v1"

    assert cli.main(args, repo_root=case["root"]) == 2
    assert calls == []
    assert not output.exists()


def test_prepare_cli_rejects_unknown_registered_inputs_before_output(monkeypatch):
    import scripts.prepare_eval as cli

    case = make_comt_source()
    output = case["root"] / "data/evaluation" / (
        "comt-unknown-" + uuid.uuid4().hex
    )
    args = _prepare_args(case, output)
    args[args.index("comt_reference_rules_v1")] = "unknown"
    monkeypatch.setattr(
        cli,
        "verify_source_reference",
        lambda *_: pytest.fail("unknown inputs reached source verification"),
    )

    assert cli.main(args, repo_root=case["root"]) == 2
    assert not output.exists()
    assert not (case["root"] / "logs/evaluation" / output.name).exists()


def test_real_recipe_is_the_offline_comt_contract():
    raw = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    config = parse_config(raw, ROOT)

    assert raw == {
        "schema_version": 1,
        "benchmark": "comt",
        "profile": "comt_reference_rules_v1",
        "categories": ["creation", "deletion", "update", "selection"],
        "dataset_root": "data",
        "run_metadata": {
            "model_identity": "external:model",
            "prompt_template_identity": "ilvr-test400-corrected-v1",
            "generation_config": {},
        },
    }
    assert config.max_steps is None
    assert config.dataset_root == (ROOT / "data").resolve()


def test_run_eval_rejects_valid_comt_before_runtime_loading(monkeypatch):
    import scripts.run_eval as cli

    case = make_comt_case(1)
    case["config"]["run_metadata"]["generation_config"] = {
        "max_new_tokens": 8,
        "max_sequence_length": 32768,
        "repetition_penalty": 1.05,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
    }
    config_path = case["data_root"] / "run-config.yaml"
    config_path.write_text(yaml.safe_dump(case["config"]), encoding="utf-8")
    output = case["root"] / "results/evaluation" / (
        "comt-run-rejection-" + uuid.uuid4().hex
    )
    calls = []
    monkeypatch.setattr(cli, "load_runtime", lambda: calls.append(True))

    assert cli.main(
        [
            "--manifest",
            str(case["manifest"]),
            "--config",
            str(config_path),
            "--model-dir",
            "checkpoints/evaluation-staging/not-needed-for-comt-rejection",
            "--model-identity",
            "synthetic:comt-model",
            "--output-dir",
            str(output),
        ],
        repo_root=case["root"],
    ) == 2
    assert calls == []
    assert not output.exists()


CHILD = textwrap.dedent(
    r"""
    from dataclasses import asdict
    import hashlib
    import json
    from pathlib import Path
    import sys
    import yaml

    root = Path(sys.argv[1]).resolve()
    source_root = Path(sys.argv[2])
    selection = Path(sys.argv[3])
    image_root = Path(sys.argv[4])
    output = Path(sys.argv[5])
    source_hash = sys.argv[6]
    selection_hash = sys.argv[7]
    config_path = Path(sys.argv[8])

    blocked = {"torch", "transformers", "eval", "Reference"}
    from evaluation import comt_prepare
    comt_prepare.SOURCE_SHA256 = source_hash
    comt_prepare.SELECTION_SHA256 = selection_hash
    import scripts.prepare_eval as prepare_cli
    import scripts.score_eval as score_cli
    from evaluation.schema import Prediction, RunMetadata, run_fingerprint

    prepare_exit = prepare_cli.main(
        [
            "--benchmark", "comt",
            "--profile", "comt_reference_rules_v1",
            "--source-root", str(source_root),
            "--selection-manifest", str(selection),
            "--image-root", str(image_root),
            "--output-dir", str(output),
        ],
        repo_root=root,
    )
    manifest = output / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    raw_config = yaml.safe_load(config_path.read_text())
    run = RunMetadata(**raw_config["run_metadata"])
    predictions = []
    for row in rows:
        response = "B"
        if row["provenance"]["source_id"] == "creation-12125":
            response = "B"
        elif row["provenance"]["source_id"] == "creation-12164":
            response = "D"
        predictions.append(
            asdict(
                Prediction(
                    row["id"], manifest_hash, run_fingerprint(run), response, "ok"
                )
            )
        )
    prediction_path = root / "results/evaluation" / (output.name + "-responses.jsonl")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_bytes(
        b"".join(canonical + b"\n" for canonical in (
            __import__("evaluation.schema", fromlist=["canonical_json"]).canonical_json(row)
            for row in predictions
        ))
    )
    score_output = root / "results/evaluation" / (output.name + "-score")
    score_exit = score_cli.main(
        [
            "--manifest", str(manifest),
            "--predictions", str(prediction_path),
            "--config", str(config_path),
            "--output-dir", str(score_output),
        ],
        repo_root=root,
    )
    incomplete_predictions = root / "results/evaluation" / (
        output.name + "-responses-incomplete.jsonl"
    )
    incomplete_predictions.write_bytes(
        b"".join(canonical + b"\n" for canonical in (
            __import__("evaluation.schema", fromlist=["canonical_json"]).canonical_json(row)
            for row in predictions[:-1]
        ))
    )
    incomplete_output = root / "results/evaluation" / (output.name + "-score-incomplete")
    incomplete_exit = score_cli.main(
        [
            "--manifest", str(manifest),
            "--predictions", str(incomplete_predictions),
            "--config", str(config_path),
            "--output-dir", str(incomplete_output),
        ],
        repo_root=root,
    )
    module_origins = {
        name: str(Path(module.__file__).resolve())
        for name, module in sorted(sys.modules.items())
        if (
            name == "evaluation"
            or name.startswith("evaluation.")
            or name in {"scripts", "scripts.prepare_eval", "scripts.score_eval"}
        )
        and getattr(module, "__file__", None)
    }
    evidence = {
        "exits": [prepare_exit, score_exit],
        "incomplete_exit": incomplete_exit,
        "summary": json.loads((score_output / "summary.json").read_text()),
        "incomplete": json.loads((incomplete_output / "summary.json").read_text()),
        "prepare": json.loads(
            (root / "results/evaluation" / output.name / "prepare.json").read_text()
        ),
        "rows": rows,
        "predictions": predictions,
        "manifest_hash": manifest_hash,
        "run_metadata": raw_config["run_metadata"],
        "module_origins": module_origins,
        "heavy_modules_loaded": sorted(name for name in blocked if name in sys.modules),
        "paths": {
            "manifest": str(manifest),
            "score": str(score_output),
            "incomplete_score": str(incomplete_output),
        },
    }
    evidence_path = root / "results/evaluation" / (output.name + "-evidence.json")
    evidence_path.write_text(json.dumps(evidence) + "\n")
    print(json.dumps({"evidence_path": str(evidence_path)}))
    """
)


def test_copied_root_real_prepare_and_score_complete_400_then_incomplete():
    case = _make_literal_b_fixture()
    copied_root = case["root"]
    for relative in PIPELINE_FILES:
        destination = copied_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    config_path = copied_root / "configs/eval/comt_reference_rules_v1.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REAL_CONFIG, config_path)
    output = copied_root / "data/evaluation" / ("comt-pipeline-" + uuid.uuid4().hex)
    log_dir = ROOT / "logs/evaluation" / (output.name + "-subprocess")
    log_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = "src:."
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD,
            str(copied_root),
            str(case["source_root"]),
            str(case["selection_manifest"]),
            str(case["image_root"]),
            str(output),
            case["source_sha256"],
            case["selection_sha256"],
            str(config_path),
        ],
        cwd=copied_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    (log_dir / "copied-root.log").write_text(
        "stdout:\n" + completed.stdout + "\nstderr:\n" + completed.stderr,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    evidence_path = Path(json.loads(completed.stdout.splitlines()[-1])["evidence_path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    summary = evidence["summary"]
    incomplete = evidence["incomplete"]
    expected_source_ids = _expected_source_ids()
    expected_ids = [
        f"comt/{source_id.split('-', 1)[0]}/{source_id}"
        for source_id in expected_source_ids
    ]
    expected_category = {
        "accuracy": 1.0,
        "complete": True,
        "correct": 100,
        "count": 100,
        "observed": 100,
    }

    assert evidence["exits"] == [0, 0]
    assert summary == {
        "N_expected": 400,
        "accuracy": 1.0,
        "benchmark": "comt",
        "categories": {
            category: expected_category
            for category in ("creation", "deletion", "update", "selection")
        },
        "complete": True,
        "correct": 400,
        "coverage": 1.0,
        "manifest_sha256": FIXTURE_MANIFEST_SHA256,
        "observed": 400,
        "profile": "comt_reference_rules_v1",
        "protocol_kind": "modified",
        "run_fingerprint": FIXTURE_RUN_FINGERPRINT,
        "schema_version": 1,
        "scorer_version": "1",
        "status_counts": {"parsed": 400},
        "verification": {
            "cpu_tests": "not_recorded_by_scoring_run",
            "full_benchmark": "not_run",
            "gpu_smoke": "not_run",
            "paper_protocol_verified": False,
            "upstream_environment": "not_run",
        },
    }
    assert evidence["incomplete_exit"] == 1
    assert incomplete["N_expected"] == 400
    assert incomplete["complete"] is False
    assert incomplete["accuracy"] is None
    assert evidence["manifest_hash"] == FIXTURE_MANIFEST_SHA256
    assert evidence["run_metadata"] == FIXTURE_RUN_METADATA
    assert evidence["prepare"] == {
        "N_expected": 400,
        "categories": {
            "creation": 100,
            "deletion": 100,
            "selection": 100,
            "update": 100,
        },
        "config": "default",
        "correction_count": 2,
        "dataset_id": "czh-up/CoMT",
        "dataset_profile": "ilvr-test400-corrected-v1",
        "dataset_root": "data",
        "manifest_sha256": FIXTURE_MANIFEST_SHA256,
        "ood_verified": False,
        "paper_protocol_verified": False,
        "protocol_kind": "modified",
        "revision": "bde314be94942a7b4ab66581e7f7c4801b84e686",
        "selection_sha256": FIXTURE_SELECTION_SHA256,
        "source_sha256": FIXTURE_SOURCE_SHA256,
    }
    assert [row["id"] for row in evidence["rows"]] == expected_ids
    assert [row["provenance"]["source_id"] for row in evidence["rows"]] == (
        expected_source_ids
    )
    for row, source_id in zip(evidence["rows"], expected_source_ids):
        category, image_token = source_id.split("-", 1)
        image = f"images_comt/{category}/{image_token}.png"
        assert row["provenance"] == {
            "asset_sha256": {image: FIXTURE_IMAGE_SHA256},
            "config": "default",
            "dataset_id": "czh-up/CoMT",
            "revision": "bde314be94942a7b4ab66581e7f7c4801b84e686",
            "source_id": source_id,
            "split": "ilvr-test400-corrected-v1",
        }
    assert [row["id"] for row in evidence["predictions"]] == expected_ids
    assert {
        row["manifest_sha256"] for row in evidence["predictions"]
    } == {FIXTURE_MANIFEST_SHA256}
    assert {row["run_fingerprint"] for row in evidence["predictions"]} == {
        FIXTURE_RUN_FINGERPRINT
    }
    by_source = {row["provenance"]["source_id"]: row for row in evidence["rows"]}
    assert by_source["creation-12125"]["target"] == {
        "index": 1,
        "label": "B",
        "content": "18",
    }
    assert by_source["creation-12164"]["target"] == {
        "index": 3,
        "label": "D",
        "content": "24",
    }
    assert evidence["heavy_modules_loaded"] == []
    assert set(evidence["module_origins"]) == EXPECTED_MODULE_ORIGINS
    assert all(
        Path(origin).is_relative_to(copied_root.resolve())
        for origin in evidence["module_origins"].values()
    )
