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

from offline_fixtures import make_case, save_manifest, save_rows
from scripts.score_eval import main


def invoke(case, output_name=None):
    config_path = case["data_root"] / "config.yaml"
    config_path.write_text(yaml.safe_dump(case["config"]))
    out = case["root"] / "results/evaluation" / (
        output_name or ("p1-score-" + uuid.uuid4().hex)
    )
    code = main(
        [
            "--manifest", str(case["manifest"]),
            "--predictions", str(case["prediction_path"]),
            "--config", str(config_path),
            "--output-dir", str(out),
        ],
        repo_root=case["root"],
    )
    return code, out


def test_complete_cli_and_no_overwrite():
    case = make_case(2)
    code, out = invoke(case)
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["N_expected"] == 2 and summary["accuracy"] == 1.0
    before = (out / "summary.json").read_bytes()
    code, _ = invoke(case, out.name)
    assert code == 2 and (out / "summary.json").read_bytes() == before


def test_empty_predictions_cli_is_incomplete():
    case = make_case(2)
    save_rows(case["prediction_path"], [])
    code, out = invoke(case)
    assert code == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete"] is False and summary["accuracy"] is None


def test_preflight_error_writes_no_scores():
    case = make_case(1)
    case["config"]["profile"] = "unknown"
    code, out = invoke(case)
    assert code == 2 and not out.exists()


@pytest.mark.parametrize("kind", ["missing_image", "bad_hash", "bad_config", "fingerprint"])
def test_cli_preflight_failures(kind):
    case = make_case(1)
    if kind in ("missing_image", "bad_hash"):
        row = case["samples"][0]
        row["model_input"]["images"] = ["image.bin"]
        row["provenance"]["asset_sha256"] = {"image.bin": "0" * 64}
        if kind == "bad_hash":
            (case["data_root"] / "image.bin").write_bytes(b"fixture-image")
        save_manifest(case)
    if kind == "bad_config":
        case["config"] = []
    if kind == "fingerprint":
        case["predictions"][0]["manifest_sha256"] = "0" * 64
        save_rows(case["prediction_path"], case["predictions"])
    code, out = invoke(case)
    assert code == 2 and not out.exists()


def test_argument_error_and_outside_output():
    case = make_case(1)
    args = [
        "--manifest", str(case["manifest"]),
        "--predictions", str(case["prediction_path"]),
        "--config", "unused.yaml",
        "--output-dir", str(case["data_root"] / "forbidden-output"),
    ]
    assert main(args, repo_root=case["root"]) == 2
    assert not (case["data_root"] / "forbidden-output").exists()
    with pytest.raises(SystemExit) as exc:
        main(args + ["--unknown"], repo_root=case["root"])
    assert exc.value.code == 2


def test_report_serialization_and_log():
    case = make_case(2)
    code, out = invoke(case)
    assert code == 0
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    assert [row["id"] for row in rows] == [row["id"] for row in case["samples"]]
    def reject_nonfinite(value):
        raise AssertionError(value)
    json.loads((out / "summary.json").read_text(), parse_constant=reject_nonfinite)
    assert (case["root"] / "logs/evaluation" / out.name / "score.log").is_file()


def test_output_failure_is_infrastructure_error(monkeypatch):
    import scripts.score_eval as cli
    case = make_case(1)
    def fail_write(out, scores, summary):
        out.mkdir()
        raise OSError("synthetic output failure")
    monkeypatch.setattr(cli, "write_reports", fail_write)
    code, out = invoke(case)
    assert code == 1 and out.is_dir()
    assert not (out / "summary.json").exists()
    assert "synthetic output failure" in (
        case["root"] / "logs/evaluation" / out.name / "score.log"
    ).read_text()


def test_copied_workspace_cli_without_reference_or_heavy_imports():
    case = make_case(2)
    workspace = case["result_root"] / "workspace"
    relative_files = [
        "src/evaluation/__init__.py", "src/evaluation/schema.py", "src/evaluation/io.py",
        "src/evaluation/registry.py", "src/evaluation/report.py",
        "src/evaluation/benchmarks/__init__.py", "src/evaluation/benchmarks/vsp.py",
        "scripts/__init__.py", "scripts/score_eval.py",
    ]
    source_root = case["root"]
    for relative in relative_files:
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)
    inventory = {
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    assert inventory == set(relative_files)
    (workspace / "data").symlink_to(case["root"] / "data", target_is_directory=True)
    (workspace / "results").symlink_to(case["root"] / "results", target_is_directory=True)
    (workspace / "logs").symlink_to(case["root"] / "logs", target_is_directory=True)
    forbidden = ("eval.py", "src/evaluate_deepseed.py", "src/task_deepseed.py",
                 "src/evaluation/scoring.py", "src/evaluation/runtime.py",
                 "src/evaluation/continuous_latent.py", "Reference")
    inventory = {path.relative_to(workspace).as_posix() for path in workspace.rglob("*")
                 if not path.is_symlink()}
    assert not any(item in inventory for item in forbidden)
    config_path = case["data_root"] / "config.yaml"
    config_path.write_text(yaml.safe_dump(case["config"]))
    out = case["root"] / "results/evaluation" / ("p1-subprocess-" + uuid.uuid4().hex)
    child = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        import importlib.abc
        blocked = {'torch', 'transformers', 'mathruler', 'MathRuler', 'gym', 'gymnasium',
                   'Reference', 'evaluation.scoring', 'evaluation.runtime',
                   'evaluation.continuous_latent'}
        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked or fullname.split('.')[0] in blocked:
                    raise AssertionError('heavy or legacy import: ' + fullname)
                return None
        sys.meta_path.insert(0, Blocker())
        assert not any(name in sys.modules for name in blocked)
        import evaluation, evaluation.io, evaluation.registry, evaluation.report
        import scripts.score_eval
        assert Path(evaluation.__file__).resolve() == Path.cwd() / 'src/evaluation/__init__.py'
        assert Path(scripts.score_eval.__file__).resolve() == Path.cwd() / 'scripts/score_eval.py'
        copied_root = Path.cwd().resolve()
        module_paths = {}
        for name, module in sorted(sys.modules.items()):
            if name == 'scripts.score_eval' or name == 'evaluation' or name.startswith('evaluation.'):
                path = getattr(module, '__file__', None)
                assert path is not None, name
                resolved = Path(path).resolve()
                resolved.relative_to(copied_root)
                module_paths[name] = str(resolved)
        assert 'scripts.score_eval' in module_paths
        assert 'evaluation' in module_paths
        print('module_paths=' + repr(module_paths))
        raise SystemExit(scripts.score_eval.main())
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [str(case["root"] / "venv/ilvr-venv/bin/python"), "-c", child,
         "--manifest", str(Path("data/evaluation") / case["run_id"] / "manifest.jsonl"),
         "--predictions", str(Path("results/evaluation") / case["run_id"] / "predictions.jsonl"),
         "--config", str(Path("data/evaluation") / case["run_id"] / "config.yaml"),
         "--output-dir", str(Path("results/evaluation") / out.name)],
        cwd=workspace, env=env, text=True, capture_output=True, timeout=60,
    )
    child_log = case["root"] / "logs/evaluation" / out.name / "copied-workspace.log"
    child_log.write_text("stdout:\n" + proc.stdout + "\nstderr:\n" + proc.stderr)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["N_expected"] == 2
    assert summary["correct"] == 2
    assert summary["accuracy"] == 1.0
    assert summary["complete"] is True
    assert child_log.is_file()
