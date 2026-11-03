from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import uuid

import pytest
import yaml

from evaluation.io import parse_config, read_inputs
from evaluation.report import evaluate
from evaluation.schema import canonical_json, run_fingerprint
from vsp_fixtures import make_vsp_case


def test_cli_module_exists():
    assert importlib.util.find_spec("scripts.run_eval") is not None


class _Serializable:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


class _FakeRuntime:
    def __init__(self, outputs=None, *, load_error=None):
        self.outputs = list(outputs or [])
        self.load_error = load_error
        self.calls = []
        loaded_config = SimpleNamespace(stage="stage2", latent_size=8)
        self.model = SimpleNamespace(
            config=loaded_config,
            generation_config=_Serializable({"max_new_tokens": 8}),
        )
        self.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template="active-tokenizer-template"),
            chat_template="active-processor-template",
            init_kwargs={"padding_side": "left"},
            to_dict=lambda: {"padding_side": "left"},
        )
        self.transformers = SimpleNamespace(
            __file__="/fake/transformers/__init__.py", __version__="0.test"
        )

    def load_evaluation_artifacts(self, *, model_dir, base_model_dir, cache_dir):
        self.calls.append(("load", model_dir, base_model_dir, cache_dir))
        if self.load_error is not None:
            raise self.load_error
        return self.model, self.processor

    def resolve_qwen2_5_vl_model(self, model):
        assert model is self.model
        return model

    def resolve_latent_token_ids(self, tokenizer, config):
        assert tokenizer is self.processor.tokenizer
        assert config is self.model.config
        return SimpleNamespace(pad_token_id=90, start_token_id=91, end_token_id=92)

    def run_one_example(self, model, processor, payload, generation_config):
        assert model is self.model
        assert processor is self.processor
        assert set(payload) == {"text_input", "image_input"}
        self.calls.append(("generate", payload, generation_config))
        value = self.outputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def _legacy_output(case, *, response="Action plan: R", stop_reason="eos"):
    return {
        "assistant_only": response,
        "generated_suffix_token_ids": [21, 22],
        "prompt_identity": "a" * 64,
        "prompt_token_length": 17,
        "raw_output": case["samples"][0]["model_input"]["text"]
        + "\nAction plan: L\n"
        + response,
        "extracted_final_answer": "R",
        "latent_start_count": 1,
        "latent_block_count": 1,
        "stop_reason": stop_reason,
        "inference_time": 0.25,
    }


def _write_config(case) -> Path:
    path = case["data_root"] / "config.yaml"
    path.write_text(yaml.safe_dump(case["config"]), encoding="utf-8")
    return path


def _fresh_output(case, suffix="generation") -> Path:
    return case["root"] / "results/evaluation" / (
        case["run_id"] + "-" + suffix + "-" + uuid.uuid4().hex
    )


def _args(case, output, *, config_file=None, **overrides):
    values = {
        "manifest": case["manifest"],
        "config": config_file or _write_config(case),
        "model-dir": case["model_dir"],
        "model-identity": "synthetic:model",
        "output-dir": output,
    }
    values.update(overrides)
    args = []
    for name, value in values.items():
        if value is not None:
            args.extend(("--" + name, str(value)))
    return args


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _cli():
    return importlib.import_module("scripts.run_eval")


def test_unknown_profile_never_imports_runtime(monkeypatch):
    case = make_vsp_case(1)
    case["config"]["profile"] = "unknown"
    config_file = _write_config(case)
    cli = _cli()
    monkeypatch.setattr(cli, "load_runtime", lambda: pytest.fail("runtime imported"))
    output = _fresh_output(case, "invalid-profile")

    assert cli.main(
        _args(case, output, config_file=config_file), repo_root=case["root"]
    ) == 2
    assert not output.exists()
    assert "input error" in (
        case["root"] / "logs/evaluation" / output.name / "run.log"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "kind",
    [
        "wrong_asset_hash",
        "missing_png",
        "bad_generation_kwargs",
        "identity_mismatch",
        "outside_model",
        "outside_base",
        "outside_base_symlink",
        "escaped_cache",
        "existing_result",
        "existing_log",
        "malformed_prompt",
        "malformed_yaml",
    ],
)
def test_preflight_rejects_each_invalid_input_before_runtime(kind, monkeypatch):
    case = make_vsp_case(1)
    output = _fresh_output(case, kind)
    overrides = {}
    config_file = None
    if kind == "wrong_asset_hash":
        case["samples"][0]["provenance"]["asset_sha256"][
            "maps/level3/img/0.png"
        ] = "0" * 64
        case["manifest"].write_bytes(
            canonical_json(case["samples"][0]) + b"\n"
        )
    elif kind == "missing_png":
        (case["data_root"] / "maps/level3/img/0.png").unlink()
    elif kind == "bad_generation_kwargs":
        case["config"]["run_metadata"]["generation_config"]["temperature"] = -1.0
    elif kind == "identity_mismatch":
        overrides["model-identity"] = "synthetic:other"
    elif kind == "outside_model":
        overrides["model-dir"] = case["data_root"]
    elif kind == "outside_base":
        (case["model_dir"] / "adapter_config.json").write_text("{}", encoding="utf-8")
        overrides["base-model-dir"] = case["data_root"]
    elif kind == "outside_base_symlink":
        (case["model_dir"] / "adapter_config.json").write_text("{}", encoding="utf-8")
        escaped = case["root"] / "checkpoints/evaluation-staging" / (
            case["run_id"] + "-escaped-base"
        )
        escaped.symlink_to(case["data_root"], target_is_directory=True)
        overrides["base-model-dir"] = escaped
    elif kind == "escaped_cache":
        overrides["cache-dir"] = case["model_dir"]
    elif kind == "existing_result":
        output.mkdir()
    elif kind == "existing_log":
        (case["root"] / "logs/evaluation" / output.name).mkdir(
            parents=True, exist_ok=False
        )
    elif kind == "malformed_prompt":
        (case["data_root"] / "prompt-text/ilvr-three-images-v1.txt").write_text(
            "altered prompt", encoding="utf-8"
        )
    elif kind == "malformed_yaml":
        config_file = case["data_root"] / "config.yaml"
        config_file.write_text("run_metadata: [", encoding="utf-8")
    cli = _cli()
    monkeypatch.setattr(cli, "load_runtime", lambda: pytest.fail("runtime imported"))

    assert cli.main(
        _args(case, output, config_file=config_file, **overrides),
        repo_root=case["root"],
    ) == 2
    assert not (output / "predictions.jsonl").exists()


def test_invalid_manifest_subprocess_blocks_heavy_runtime_imports():
    case = make_vsp_case(1)
    case["samples"][0]["provenance"]["asset_sha256"][
        "maps/level3/img/0.png"
    ] = "0" * 64
    case["manifest"].write_bytes(canonical_json(case["samples"][0]) + b"\n")
    output = _fresh_output(case, "blocked-import")
    config_file = _write_config(case)
    child = textwrap.dedent(
        """
        import importlib.abc
        import sys
        blocked = {'torch', 'transformers', 'eval'}
        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked or fullname.split('.')[0] in blocked:
                    raise AssertionError('heavy runtime import: ' + fullname)
                return None
        sys.meta_path.insert(0, Blocker())
        import scripts.run_eval as cli
        assert not any(name in sys.modules for name in blocked)
        raise SystemExit(cli.main())
        """
    )
    completed = subprocess.run(
        [
            str(case["root"] / "venv/ilvr-venv/bin/python"),
            "-c",
            child,
            *_args(case, output, config_file=config_file),
        ],
        cwd=case["root"],
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "src:tests/evaluation:.",
        },
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert completed.returncode == 2, completed.stderr
    assert "heavy runtime import" not in completed.stderr
    assert not output.exists()


def test_two_row_generation_is_consumable_by_real_p1_scorer(monkeypatch):
    case = make_vsp_case(2)
    output = _fresh_output(case, "success")
    runtime = _FakeRuntime([_legacy_output(case), _legacy_output(case)])
    cli = _cli()
    monkeypatch.setattr(cli, "load_runtime", lambda: runtime)

    assert cli.main(_args(case, output), repo_root=case["root"]) == 0

    predictions = _read_rows(output / "predictions.jsonl")
    details = _read_rows(output / "generation-details.jsonl")
    config = parse_config(case["config"], case["root"])
    manifest_hash = hashlib.sha256(case["manifest"].read_bytes()).hexdigest()
    fingerprint = run_fingerprint(config.run_metadata)
    assert len(predictions) == len(details) == 2
    assert [row["id"] for row in details] == [row["id"] for row in predictions]
    assert [row["id"] for row in predictions] == [
        "vsp/level3/0",
        "vsp/level3/1",
    ]
    assert {row["manifest_sha256"] for row in predictions + details} == {
        manifest_hash
    }
    assert {row["run_fingerprint"] for row in predictions + details} == {
        fingerprint
    }
    assert [row["response"] for row in predictions] == [
        "Action plan: R",
        "Action plan: R",
    ]
    assert all(case["samples"][0]["model_input"]["text"] not in row["response"]
               for row in predictions)
    assert [row["generated_suffix_token_ids"] for row in details] == [
        [21, 22],
        [21, 22],
    ]
    assert [Path(call[1]["image_input"][2]).name for call in runtime.calls[1:]] == [
        "0.png",
        "1.png",
    ]

    prepared = read_inputs(
        case["manifest"], output / "predictions.jsonl", config, case["root"]
    )
    _scores, summary = evaluate(prepared, config)
    assert summary["N_expected"] == 2
    assert summary["observed"] == 2
    assert summary["correct"] == 2
    assert summary["accuracy"] == 1.0

    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["schema_version"] == 1
    assert run["status"] == "complete"
    assert run["protocol_kind"] == "modified"
    assert run["manifest_sha256"] == manifest_hash
    assert run["run_fingerprint"] == fingerprint
    assert run["N_expected"] == run["processed"] == 2
    assert run["run_metadata"] == asdict(config.run_metadata)
    assert run["runtime_metadata"]["latent_token_ids"] == {
        "pad": 90,
        "start": 91,
        "end": 92,
    }
    assert run["paths"]["predictions"] == str(
        (output / "predictions.jsonl").resolve()
    )


def test_loader_preserves_lexical_base_alias_after_containment_validation(monkeypatch):
    case = make_vsp_case(1)
    (case["model_dir"] / "adapter_config.json").write_text("{}", encoding="utf-8")
    alias = case["root"] / "checkpoints" / "evaluation-staging" / (
        case["run_id"] + "-base-alias"
    )
    alias.symlink_to(case["base_model_dir"], target_is_directory=True)
    output = _fresh_output(case, "base-alias")
    expected_lexical = str(alias.absolute())
    runtime = _FakeRuntime([_legacy_output(case)])
    original_loader = runtime.load_evaluation_artifacts

    def strict_loader(*, model_dir, base_model_dir, cache_dir):
        assert base_model_dir == expected_lexical
        return original_loader(
            model_dir=model_dir,
            base_model_dir=base_model_dir,
            cache_dir=cache_dir,
        )

    runtime.load_evaluation_artifacts = strict_loader
    cli = _cli()
    monkeypatch.setattr(cli, "load_runtime", lambda: runtime)

    assert cli.main(
        _args(case, output, **{"base-model-dir": alias}), repo_root=case["root"]
    ) == 0
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["paths"]["base_model_dir"] == str(case["base_model_dir"].resolve())


@pytest.mark.parametrize(
    ("exception_message", "expected_error"),
    [
        ("", "RuntimeError"),
        ("   ", "RuntimeError"),
        ("synthetic inference failure", "synthetic inference failure"),
    ],
    ids=("empty", "whitespace", "meaningful"),
)
def test_first_inference_failure_records_pair_and_leaves_remaining_missing(
    exception_message, expected_error, monkeypatch
):
    case = make_vsp_case(2)
    output = _fresh_output(case, "inference-failure")
    runtime = _FakeRuntime([RuntimeError(exception_message)])
    cli = _cli()
    monkeypatch.setattr(cli, "load_runtime", lambda: runtime)

    assert cli.main(_args(case, output), repo_root=case["root"]) == 1

    predictions = _read_rows(output / "predictions.jsonl")
    details = _read_rows(output / "generation-details.jsonl")
    config = parse_config(case["config"], case["root"])
    prepared = read_inputs(
        case["manifest"], output / "predictions.jsonl", config, case["root"]
    )
    _scores, summary = evaluate(prepared, config)
    assert len(predictions) == len(details) == 1
    assert predictions[0]["id"] == details[0]["id"] == "vsp/level3/0"
    assert predictions[0]["status"] == details[0]["status"] == "infrastructure_error"
    assert predictions[0]["response"] is None
    assert predictions[0]["error"] == details[0]["error"] == expected_error
    for name in (
        "generated_suffix_token_ids",
        "prompt_identity",
        "prompt_token_length",
        "stop_reason",
        "latent_start_count",
        "latent_block_count",
        "inference_time",
        "raw_output",
    ):
        assert details[0][name] is None

    assert summary["status_counts"] == {"infrastructure_error": 1, "missing": 1}
    assert summary["accuracy"] is None
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert run["processed"] == 1
    assert run["error"] == expected_error


def test_model_load_failure_leaves_empty_outputs_and_scores_all_missing(monkeypatch):
    case = make_vsp_case(2)
    output = _fresh_output(case, "load-failure")
    runtime = _FakeRuntime(load_error=RuntimeError("synthetic load failure"))
    cli = _cli()
    monkeypatch.setattr(cli, "load_runtime", lambda: runtime)

    assert cli.main(_args(case, output), repo_root=case["root"]) == 1
    assert (output / "predictions.jsonl").read_bytes() == b""
    assert (output / "generation-details.jsonl").read_bytes() == b""

    config = parse_config(case["config"], case["root"])
    prepared = read_inputs(
        case["manifest"], output / "predictions.jsonl", config, case["root"]
    )
    _scores, summary = evaluate(prepared, config)
    assert summary["status_counts"] == {"missing": 2}
    assert summary["observed"] == 0
    assert summary["accuracy"] is None
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert run["processed"] == 0
    assert "synthetic load failure" in run["error"]


def test_blank_and_token_limit_outputs_remain_usable_observations(monkeypatch):
    case = make_vsp_case(2)
    output = _fresh_output(case, "usable-boundaries")
    blank = _legacy_output(case, response="", stop_reason="eos")
    blank["generated_suffix_token_ids"] = []
    blank["extracted_final_answer"] = ""
    token_limit = _legacy_output(
        case, response="unfinished prose", stop_reason="max_new_tokens"
    )
    token_limit["extracted_final_answer"] = ""
    runtime = _FakeRuntime([blank, token_limit])
    cli = _cli()
    monkeypatch.setattr(cli, "load_runtime", lambda: runtime)

    assert cli.main(_args(case, output), repo_root=case["root"]) == 0
    predictions = _read_rows(output / "predictions.jsonl")
    details = _read_rows(output / "generation-details.jsonl")
    assert [row["status"] for row in predictions] == ["ok", "ok"]
    assert [row["response"] for row in predictions] == ["", "unfinished prose"]
    assert [row["termination_reason"] for row in predictions] == [
        "eos",
        "max_new_tokens",
    ]
    assert [row["stop_reason"] for row in details] == ["eos", "max_new_tokens"]
    assert json.loads((output / "run.json").read_text())["status"] == "complete"


def test_detail_write_failure_never_marks_run_complete(monkeypatch):
    case = make_vsp_case(1)
    output = _fresh_output(case, "detail-write-failure")
    runtime = _FakeRuntime([_legacy_output(case)])
    cli = _cli()
    monkeypatch.setattr(cli, "load_runtime", lambda: runtime)
    real_append = cli._append_record

    def fail_detail(handle, record):
        if Path(handle.name).name == "generation-details.jsonl":
            raise OSError("synthetic detail write failure")
        return real_append(handle, record)

    monkeypatch.setattr(cli, "_append_record", fail_detail)

    assert cli.main(_args(case, output), repo_root=case["root"]) == 1
    assert (output / "predictions.jsonl").read_bytes() == b""
    assert (output / "generation-details.jsonl").read_bytes() == b""
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert run["processed"] == 0
    assert "synthetic detail write failure" in run["error"]
    assert "synthetic detail write failure" in (
        case["root"] / "logs/evaluation" / output.name / "run.log"
    ).read_text(encoding="utf-8")
