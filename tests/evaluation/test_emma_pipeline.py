import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import yaml

from emma_fixtures import make_case, write_rows
from emma_source_fixtures import make_source, pin_source
from evaluation.schema import RunMetadata, canonical_json, run_fingerprint


def _prepare_args(case):
    return ["--benchmark", "emma", "--profile", "emma_safe_rules_v1",
            "--source-root", str(case["source"]), "--output-dir", str(case["output"])]


def test_prepare_cli_dispatches_emma_and_rejects_invalid_source(monkeypatch):
    from scripts import prepare_eval
    case = make_source(); pin_source(monkeypatch, case)
    assert prepare_eval.main(_prepare_args(case), repo_root=case["root"]) == 0
    summary = json.loads((case["root"] / "results/evaluation" / case["output"].name / "prepare.json").read_text())
    assert summary["dataset_id"] == "luckychao/EMMA-mini" and summary["N_expected"] == 400
    prepared_rows = [json.loads(line) for line in (case["output"] / "manifest.jsonl").read_text().splitlines()]
    open_row = next(row for row in prepared_rows if row["id"] == "che_78")
    assert open_row["target"]["question_type"] == "open_ended"
    one_manifest = case["output"] / "open-manifest.jsonl"
    one_manifest.write_bytes(canonical_json(open_row) + b"\n")
    digest = hashlib.sha256(one_manifest.read_bytes()).hexdigest()
    metadata = RunMetadata("synthetic:emma-open", "synthetic:question", {})
    score_config = case["output"] / "open-config.yaml"
    score_config.write_text(yaml.safe_dump({"schema_version": 1, "benchmark": "emma", "profile": "emma_safe_rules_v1",
        "categories": ["Chemistry"], "dataset_root": str(case["output"]), "run_metadata": {
            "model_identity": metadata.model_identity, "prompt_template_identity": metadata.prompt_template_identity,
            "generation_config": metadata.generation_config}}))
    predictions_dir = case["root"] / "results/evaluation" / ("p5-emma-open-input-" + uuid.uuid4().hex)
    predictions_dir.mkdir()
    predictions = predictions_dir / "predictions.jsonl"
    predictions.write_bytes(canonical_json({"id": "che_78", "manifest_sha256": digest,
        "run_fingerprint": run_fingerprint(metadata), "response": "open answer che_78", "status": "ok"}) + b"\n")
    score_output = case["root"] / "results/evaluation" / ("p5-emma-open-score-" + uuid.uuid4().hex)
    from scripts import score_eval
    assert score_eval.main(["--manifest", str(one_manifest), "--predictions", str(predictions),
        "--config", str(score_config), "--output-dir", str(score_output)], repo_root=case["root"]) == 0
    scored = json.loads((score_output / "summary.json").read_text())
    assert scored["emma"]["question_types"]["open_ended"] == {
        "count": 1, "observed": 1, "correct": 1, "complete": True, "accuracy": 1.0}
    rejected_path = make_source()
    assert prepare_eval.main(_prepare_args(rejected_path) + ["--image-root", str(rejected_path["source"])], repo_root=rejected_path["root"]) == 2
    assert not rejected_path["output"].exists()
    invalid = make_source(); pin_source(monkeypatch, invalid)
    invalid["source"].joinpath("EMMA-mini-dataset-info.json").write_text("{}")
    assert prepare_eval.main(_prepare_args(invalid), repo_root=invalid["root"]) == 2
    assert not invalid["output"].exists()


def test_emma_score_cli_reports_correct_wrong_and_missing_then_completes():
    from scripts import score_eval
    case = make_case(3)
    case["prediction_rows"][1]["response"] = "A"
    write_rows(case["predictions"], case["prediction_rows"][:2])
    config = case["data"] / "config.yaml"; config.write_text(yaml.safe_dump(case["config"]))
    partial = case["root"] / "results/evaluation" / ("p5-emma-partial-" + uuid.uuid4().hex)
    common = ["--manifest", str(case["manifest"]), "--predictions", str(case["predictions"]),
              "--config", str(config), "--output-dir"]
    assert score_eval.main(common + [str(partial)], repo_root=case["root"]) == 1
    summary = json.loads((partial / "summary.json").read_text())
    assert (summary["benchmark"], summary["profile"], summary["observed"], summary["correct"]) == ("emma", "emma_safe_rules_v1", 2, 1)
    assert summary["status_counts"] == {"missing": 1, "parsed": 2}
    assert summary["emma"]["comparison_diagnostics"] == {"math_limit_exceeded": 0, "non_finite_value": 0, "unsupported_expression": 0}
    case["prediction_rows"][2]["response"] = "B"; write_rows(case["predictions"], case["prediction_rows"])
    complete = case["root"] / "results/evaluation" / ("p5-emma-complete-" + uuid.uuid4().hex)
    assert score_eval.main(common + [str(complete)], repo_root=case["root"]) == 0
    assert json.loads((complete / "summary.json").read_text())["complete"] is True


def test_run_eval_rejects_emma_before_runtime_load(monkeypatch):
    from scripts import run_eval
    case = make_case(1)
    case["config"]["run_metadata"]["generation_config"] = {"max_new_tokens": 8, "max_sequence_length": 32768,
        "repetition_penalty": 1.05, "temperature": 0.0, "top_p": 1.0, "do_sample": False}
    config = case["data"] / "config.yaml"; config.write_text(yaml.safe_dump(case["config"]))
    calls = []; monkeypatch.setattr(run_eval, "load_runtime", lambda: calls.append(True))
    output = case["root"] / "results/evaluation" / ("p5-emma-run-reject-" + uuid.uuid4().hex)
    result = run_eval.main(["--manifest", str(case["manifest"]), "--config", str(config),
        "--model-dir", "checkpoints/not-needed", "--model-identity", "synthetic:emma",
        "--output-dir", str(output)], repo_root=case["root"])
    assert result == 2 and not calls and not output.exists()


def test_score_path_in_isolated_copy_blocks_model_and_reference_imports():
    workspace = Path(__file__).resolve().parents[2]
    case = make_case(1)
    isolated = workspace / "data/evaluation" / ("p5-emma-isolated-" + uuid.uuid4().hex)
    isolated.mkdir(parents=True)
    shutil.copytree(workspace / "src", isolated / "src")
    shutil.copytree(workspace / "scripts", isolated / "scripts")
    data = isolated / "data/evaluation/case"; data.mkdir(parents=True)
    shutil.copy2(case["manifest"], data / "manifest.jsonl")
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
    assert json.loads((output / "summary.json").read_text())["benchmark"] == "emma"
