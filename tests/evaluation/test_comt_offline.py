import copy
import json
import os
import subprocess
import sys
import textwrap

import pytest

from comt_fixtures import CATEGORIES, make_comt_case, save_rows


def run(case):
    from evaluation.io import parse_config, read_inputs
    from evaluation.report import evaluate

    config = parse_config(case["config"], case["root"])
    prepared = read_inputs(
        case["manifest"], case["prediction_path"], config, case["root"]
    )
    return evaluate(prepared, config)


def test_real_read_inputs_and_evaluate_four_categories():
    case = make_comt_case()
    scores, summary = run(case)
    assert [score.id for score in scores] == [row["id"] for row in case["samples"]]
    assert [score.correctness for score in scores] == [True, True, True, True]
    assert all(score.simulation is None for score in scores)
    assert summary["N_expected"] == 4
    assert summary["accuracy"] == 1.0
    assert summary["complete"] is True
    assert summary["benchmark"] == "comt"
    assert summary["profile"] == "comt_reference_rules_v1"
    assert summary["categories"] == {
        category: dict(count=1, observed=1, correct=1, complete=True, accuracy=1.0)
        for category in CATEGORIES
    }


@pytest.mark.parametrize("kind", ["unknown", "duplicate", "manifest", "fingerprint"])
def test_comt_prediction_join_errors_fail_closed(kind):
    from evaluation.schema import ContractError

    case = make_comt_case(2)
    rows = copy.deepcopy(case["predictions"])
    if kind == "unknown":
        rows[0]["id"] = "comt/creation/not-in-manifest"
    elif kind == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif kind == "manifest":
        rows[0]["manifest_sha256"] = "0" * 64
    else:
        rows[0]["run_fingerprint"] = "0" * 64
    save_rows(case["prediction_path"], rows)
    with pytest.raises(ContractError):
        run(case)


def test_missing_and_infrastructure_rows_keep_denominator_and_null_accuracy():
    case = make_comt_case()
    rows = copy.deepcopy(case["predictions"][:3])
    rows[1].update(status="infrastructure_error", error="worker failed", response="B")
    save_rows(case["prediction_path"], rows)
    scores, summary = run(case)
    assert [row.status for row in scores] == [
        "parsed",
        "infrastructure_error",
        "parsed",
        "missing",
    ]
    assert [row.correctness for row in scores] == [True, None, True, None]
    assert summary["N_expected"] == 4
    assert summary["observed"] == 2
    assert summary["coverage"] == 0.5
    assert summary["complete"] is False
    assert summary["accuracy"] is None


def test_configured_empty_category_has_zero_count_and_null_accuracy():
    case = make_comt_case(3)
    _, summary = run(case)
    assert summary["categories"]["selection"] == dict(
        count=0, observed=0, correct=0, complete=True, accuracy=None
    )


def test_invalid_and_empty_responses_are_observed_model_failures():
    case = make_comt_case(2)
    case["predictions"][0]["response"] = ""
    case["predictions"][1]["response"] = "no usable choice"
    save_rows(case["prediction_path"], case["predictions"])
    scores, summary = run(case)
    assert [row.status for row in scores] == ["empty", "invalid_format"]
    assert [row.correctness for row in scores] == [False, False]
    assert summary["complete"] is True
    assert summary["accuracy"] == 0.0


def test_comt_import_and_scoring_path_avoids_heavy_or_reference_modules():
    case = make_comt_case(1)
    config_path = case["data_root"] / "config.json"
    config_path.write_text(json.dumps(case["config"]))
    log_dir = case["root"] / "logs/evaluation" / case["run_id"]
    log_dir.mkdir(parents=True, exist_ok=False)
    log_path = log_dir / "import-guard.log"
    child = textwrap.dedent(
        """
        import importlib.abc
        import json
        from pathlib import Path
        import sys

        blocked = {
            'torch', 'transformers', 'eval', 'Reference', 'evaluation.scoring',
            'evaluation.runtime', 'evaluation.continuous_latent',
        }
        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked or fullname.split('.')[0] in blocked:
                    raise AssertionError('forbidden import: ' + fullname)
                return None
        sys.meta_path.insert(0, Blocker())
        from evaluation.io import parse_config, read_inputs
        from evaluation.report import evaluate
        root, manifest, predictions, config_path = map(Path, sys.argv[1:])
        config = parse_config(json.loads(config_path.read_text()), root)
        prepared = read_inputs(manifest, predictions, config, root)
        scores, summary = evaluate(prepared, config)
        assert scores[0].correctness is True
        assert summary['accuracy'] == 1.0
        assert not any(name in sys.modules for name in blocked)
        print('guarded CoMT score passed')
        """
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = "src:tests/evaluation:."
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            child,
            str(case["root"]),
            str(case["manifest"]),
            str(case["prediction_path"]),
            str(config_path),
        ],
        cwd=case["root"],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    log_path.write_text("stdout:\n" + proc.stdout + "\nstderr:\n" + proc.stderr)
    assert proc.returncode == 0, proc.stderr
    assert log_path.read_text().startswith("stdout:\nguarded CoMT score passed")
