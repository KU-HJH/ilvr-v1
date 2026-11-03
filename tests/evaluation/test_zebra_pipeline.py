import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import yaml

from zebra_fixtures import write_rows, zebra_case_factory
from tests.evaluation.zebra_source_fixtures import zebra_source_factory


def test_prepare_cli_dispatches_zebra_and_reader_accepts_manifest(monkeypatch, zebra_source_factory):
    import hashlib
    from evaluation import zebra_2d_prepare as module
    from evaluation.io import parse_config, read_manifest
    from scripts import prepare_eval

    root = Path(__file__).resolve().parents[2]
    source = zebra_source_factory()
    monkeypatch.setattr(module, "SELECTION_SHA256", hashlib.sha256((source / "selected-ids.json").read_bytes()).hexdigest())
    monkeypatch.setattr(module, "RECORDS_SHA256", hashlib.sha256((source / "source-records.jsonl").read_bytes()).hexdigest())
    output = root / "data/evaluation" / (source.name + "-cli-prepared")
    zebra_source_factory.track(
        output,
        root / "results/evaluation" / output.name,
        root / "logs/evaluation" / output.name,
    )
    args = ["--benchmark", "zebra_2d", "--profile", "zebra_2d_rules_v1",
            "--source-root", str(source), "--output-dir", str(output)]
    assert prepare_eval.main(args, repo_root=root) == 0
    config = parse_config({"schema_version": 1, "benchmark": "zebra_2d", "profile": "zebra_2d_rules_v1",
        "categories": ["Jigsaw", "Search"], "dataset_root": str(output.relative_to(root)),
        "run_metadata": {"model_identity": "synthetic:zebra", "prompt_template_identity": "synthetic:question",
                         "generation_config": {}}}, root)
    assert len(read_manifest(output / "manifest.jsonl", config, root).samples) == 400

    rejected_source = zebra_source_factory()
    rejected_output = root / "data/evaluation" / (rejected_source.name + "-rejected")
    zebra_source_factory.track(
        rejected_output,
        root / "results/evaluation" / rejected_output.name,
        root / "logs/evaluation" / rejected_output.name,
    )
    rejected_args = ["--benchmark", "zebra_2d", "--profile", "zebra_2d_rules_v1",
        "--source-root", str(rejected_source), "--output-dir", str(rejected_output),
        "--image-root", str(rejected_source)]
    assert prepare_eval.main(rejected_args, repo_root=root) == 2
    assert not rejected_output.exists()


def test_prepare_cli_reports_zebra_source_error(monkeypatch, zebra_source_factory):
    import hashlib
    from evaluation import zebra_2d_prepare as module
    from scripts import prepare_eval

    root = Path(__file__).resolve().parents[2]
    source = zebra_source_factory()
    monkeypatch.setattr(module, "SELECTION_SHA256", hashlib.sha256((source / "selected-ids.json").read_bytes()).hexdigest())
    monkeypatch.setattr(module, "RECORDS_SHA256", "0" * 64)
    output = root / "data/evaluation" / (source.name + "-invalid")
    zebra_source_factory.track(
        output,
        root / "results/evaluation" / output.name,
        root / "logs/evaluation" / output.name,
    )
    assert prepare_eval.main(["--benchmark", "zebra_2d", "--profile", "zebra_2d_rules_v1",
        "--source-root", str(source), "--output-dir", str(output)], repo_root=root) == 2
    assert not output.exists()


def test_score_cli_reports_complete_missing_infrastructure_and_invalid_cases(zebra_case_factory):
    from scripts import score_eval
    case = zebra_case_factory(("Jigsaw", "Search"))
    config = case["data"] / "config.yaml"; config.write_text(yaml.safe_dump(case["config"]))
    common = ["--manifest", str(case["manifest"]), "--predictions", str(case["predictions"]),
              "--config", str(config), "--output-dir"]
    complete = case["root"] / "results/evaluation" / f"zebra-score-{zebra_case_factory.owner_id}-{uuid.uuid4().hex}"
    zebra_case_factory.track(complete)
    assert score_eval.main(common + [str(complete)], repo_root=case["root"]) == 0
    summary = json.loads((complete / "summary.json").read_text())
    assert (summary["benchmark"], summary["correct"], summary["coverage"]) == ("zebra_2d", 2, 1.0)
    assert summary["zebra_2d"] == {"comparison_kind": "conservative_rules", "ood_verified": "unknown",
        "paper_sample_identity": "unverified", "paper_judge_equivalent": False}
    scores = [json.loads(line) for line in (complete / "scores.jsonl").read_text().splitlines()]
    assert [row["parsed_answer"] for row in scores] == ["B", "Knife."]

    case["prediction_rows"][0]["response"] = "Option B"
    case["prediction_rows"][1].update(status="infrastructure_error", error="worker", response=None)
    write_rows(case["predictions"], case["prediction_rows"])
    partial = case["root"] / "results/evaluation" / f"zebra-partial-{zebra_case_factory.owner_id}-{uuid.uuid4().hex}"
    zebra_case_factory.track(partial)
    assert score_eval.main(common + [str(partial)], repo_root=case["root"]) == 1
    summary = json.loads((partial / "summary.json").read_text())
    assert summary["status_counts"] == {"infrastructure_error": 1, "invalid_format": 1}
    assert (summary["observed"], summary["accuracy"]) == (1, None)

    write_rows(case["predictions"], [])
    missing = case["root"] / "results/evaluation" / f"zebra-missing-{zebra_case_factory.owner_id}-{uuid.uuid4().hex}"
    zebra_case_factory.track(missing)
    assert score_eval.main(common + [str(missing)], repo_root=case["root"]) == 1
    assert json.loads((missing / "summary.json").read_text())["status_counts"] == {"missing": 2}


def test_absent_configured_category_is_reported(zebra_case_factory):
    from evaluation.io import parse_config, read_inputs
    from evaluation.report import evaluate
    case = zebra_case_factory(("Jigsaw",))
    case["config"]["categories"] = ["Jigsaw", "Search"]
    prepared = read_inputs(case["manifest"], case["predictions"], parse_config(case["config"], case["root"]), case["root"])
    summary = evaluate(prepared, parse_config(case["config"], case["root"]))[1]
    assert summary["categories"]["Search"] == {"count": 0, "observed": 0, "correct": 0, "complete": True, "accuracy": None}


def test_score_path_in_isolated_copy_blocks_model_and_reference_imports(zebra_case_factory):
    workspace = Path(__file__).resolve().parents[2]
    case = zebra_case_factory(("Search",))
    isolated = workspace / "data/evaluation" / f"zebra-isolated-{zebra_case_factory.owner_id}-{uuid.uuid4().hex}"
    isolated.mkdir(parents=True)
    zebra_case_factory.track(isolated)
    shutil.copytree(workspace / "src", isolated / "src")
    shutil.copytree(workspace / "scripts", isolated / "scripts")
    data = isolated / "data/evaluation/case"; shutil.copytree(case["data"], data)
    inputs = isolated / "results/evaluation/inputs"; inputs.mkdir(parents=True)
    shutil.copy2(case["predictions"], inputs / "predictions.jsonl")
    config = dict(case["config"], dataset_root="data/evaluation/case")
    (data / "config.yaml").write_text(yaml.safe_dump(config))
    child = """
import importlib.abc, sys
blocked={'torch','transformers','Reference'}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked: raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from scripts import score_eval
raise SystemExit(score_eval.main())
"""
    output = isolated / "results/evaluation/score"
    completed = subprocess.run([str(workspace / "venv/ilvr-venv/bin/python"), "-c", child,
        "--manifest", str(data / "manifest.jsonl"), "--predictions", str(inputs / "predictions.jsonl"),
        "--config", str(data / "config.yaml"), "--output-dir", str(output)], cwd=isolated,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src:."}, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads((output / "summary.json").read_text())["benchmark"] == "zebra_2d"
