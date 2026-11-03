from io import BytesIO
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import uuid

from PIL import Image
import yaml

from evaluation.inference_bridge import validate_generation_config
from evaluation.io import parse_config
from evaluation.schema import canonical_json
from evaluation.vsp_prepare import LEVELS, build_prompt, prompt_identity
from vsp_fixtures import make_vsp_case


ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = ROOT / "configs/eval/vsp_ilvr_three_images_v1.yaml"
PIPELINE_FILES = (
    "scripts/__init__.py",
    "scripts/run_eval.py",
    "scripts/score_eval.py",
    "src/evaluation/__init__.py",
    "src/evaluation/benchmarks/__init__.py",
    "src/evaluation/benchmarks/vsp.py",
    "src/evaluation/inference_bridge.py",
    "src/evaluation/io.py",
    "src/evaluation/registry.py",
    "src/evaluation/report.py",
    "src/evaluation/schema.py",
    "src/evaluation/vsp_prepare.py",
)
EXPECTED_MODULE_ORIGINS = {
    "evaluation",
    "evaluation.benchmarks",
    "evaluation.benchmarks.vsp",
    "evaluation.inference_bridge",
    "evaluation.io",
    "evaluation.registry",
    "evaluation.report",
    "evaluation.schema",
    "evaluation.vsp_prepare",
    "scripts",
    "scripts.run_eval",
    "scripts.score_eval",
}
PROMPT_SOURCE = "Legend: <IMAGE-1>\nExample: <IMAGE-2>\nSolve: <TEST-IMAGE>\n"
GENERATION_CONFIG = {
    "max_new_tokens": 8,
    "max_sequence_length": 32768,
    "repetition_penalty": 1.05,
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
}


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (3, 3), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _prepare_copied_workspace() -> dict:
    run_id = "vsp-pipeline-" + uuid.uuid4().hex
    fixture_root = ROOT / "data/evaluation" / run_id
    copied_root = fixture_root / "copied-repository"
    copied_root.mkdir(parents=True, exist_ok=False)

    for storage in ("data", "results", "logs", "checkpoints"):
        (copied_root / storage).symlink_to(ROOT / storage, target_is_directory=True)
    for relative in PIPELINE_FILES:
        target = copied_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)

    dataset_relative = Path("data/evaluation") / run_id / "prepared-synthetic"
    dataset_root = copied_root / dataset_relative
    prompt = build_prompt(PROMPT_SOURCE)
    _write(
        dataset_root / "prompt-text/ilvr-three-images-v1.txt",
        prompt.encode("utf-8"),
    )
    assets = {
        "prompt-visual-images/system-figure-1.png": _png_bytes((255, 0, 0)),
        "prompt-visual-images/system-figure-2.png": _png_bytes((0, 255, 0)),
        "maps/level3/img/0.png": _png_bytes((0, 0, 255)),
        "maps/level3/img/1.png": _png_bytes((255, 255, 0)),
    }
    for relative, payload in assets.items():
        _write(dataset_root / relative, payload)
    hashes = {
        relative: hashlib.sha256(payload).hexdigest()
        for relative, payload in assets.items()
    }
    rows = []
    for index in range(2):
        images = [
            "prompt-visual-images/system-figure-1.png",
            "prompt-visual-images/system-figure-2.png",
            f"maps/level3/img/{index}.png",
        ]
        rows.append(
            {
                "schema_version": 1,
                "id": f"vsp/level3/{index}",
                "benchmark": "vsp",
                "subtask": "level3",
                "model_input": {"text": prompt, "images": images, "choices": []},
                "target": {"map_desc": [[1, 2], [0, 0]], "max_steps": 100},
                "provenance": {
                    "dataset_id": "synthetic:vsp-pipeline-fixture",
                    "revision": "fixture-v1",
                    "config": "level3",
                    "split": "repository-maps",
                    "source_id": str(index),
                    "asset_sha256": {name: hashes[name] for name in images},
                },
            }
        )
    manifest = dataset_root / "manifest.jsonl"
    manifest.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))

    config_relative = Path("data/evaluation") / run_id / "fixture-config.yaml"
    config = {
        "schema_version": 1,
        "benchmark": "vsp",
        "profile": "vsp_validated_v1",
        "categories": list(LEVELS),
        "dataset_root": str(dataset_relative),
        "max_steps": 100,
        "run_metadata": {
            "model_identity": "synthetic:pipeline-fake",
            "prompt_template_identity": prompt_identity(prompt),
            "generation_config": GENERATION_CONFIG,
        },
    }
    (copied_root / config_relative).write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    model_relative = (
        Path("checkpoints/evaluation-staging") / run_id / "metadata-only-fake-model"
    )
    model_dir = copied_root / model_relative
    model_dir.mkdir(parents=True, exist_ok=False)
    (model_dir / "config.json").write_text(
        '{"model_type":"test-only-fake"}\n', encoding="utf-8"
    )
    return {
        "run_id": run_id,
        "copied_root": copied_root,
        "dataset_relative": dataset_relative,
        "config_relative": config_relative,
        "model_relative": model_relative,
        "manifest_relative": dataset_relative / "manifest.jsonl",
    }


CHILD = textwrap.dedent(
    r"""
    from pathlib import Path
    from types import ModuleType, SimpleNamespace
    import json
    import sys

    root = Path(sys.argv[1]).resolve()
    run_id = sys.argv[2]
    manifest = sys.argv[3]
    config = sys.argv[4]
    model_dir = sys.argv[5]

    blocked = {"torch", "transformers"}
    trace = {"load_calls": [], "generation_calls": []}
    fake = ModuleType("eval")
    fake.__file__ = "<test-only-fake-eval>"
    fake.transformers = SimpleNamespace(
        __file__="<test-only-fake-transformers>", __version__="fixture-v1"
    )

    class Serializable:
        def __init__(self, payload):
            self.payload = payload
        def to_dict(self):
            return self.payload

    model = SimpleNamespace(
        config=SimpleNamespace(stage="fixture", latent_size=8),
        generation_config=Serializable({"max_new_tokens": 8}),
    )
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template="fixture-tokenizer-template"),
        chat_template="fixture-processor-template",
        init_kwargs={"padding_side": "left"},
        to_dict=lambda: {"padding_side": "left"},
    )

    def load_evaluation_artifacts(*, model_dir, base_model_dir, cache_dir):
        trace["load_calls"].append(
            {"model_dir": model_dir, "base_model_dir": base_model_dir,
             "cache_dir": cache_dir}
        )
        return model, processor

    def resolve_qwen2_5_vl_model(value):
        assert value is model
        return value

    def resolve_latent_token_ids(tokenizer, config):
        assert tokenizer is processor.tokenizer
        assert config is model.config
        return SimpleNamespace(pad_token_id=90, start_token_id=91, end_token_id=92)

    failure_mode = False
    def run_one_example(model_arg, processor_arg, sample, generation_config):
        assert model_arg is model
        assert processor_arg is processor
        call = {
            "sample_keys": sorted(sample),
            "ordered_images": sample["image_input"],
            "generation_config": generation_config,
        }
        trace["generation_calls"].append(call)
        if failure_mode:
            raise RuntimeError("controlled fixture generation failure")
        target_name = Path(sample["image_input"][2]).name
        suffixes = {"0.png": [101, 102], "1.png": [201, 202]}
        return {
            "assistant_only": "Action plan: R",
            "generated_suffix_token_ids": suffixes[target_name],
            "prompt_identity": "a" * 64,
            "prompt_token_length": 17,
            "raw_output": "fixture prompt boundary\\nAction plan: R",
            "extracted_final_answer": "R",
            "latent_start_count": 1,
            "latent_block_count": 1,
            "stop_reason": "eos",
            "inference_time": 0.01,
        }

    fake.load_evaluation_artifacts = load_evaluation_artifacts
    fake.resolve_qwen2_5_vl_model = resolve_qwen2_5_vl_model
    fake.resolve_latent_token_ids = resolve_latent_token_ids
    fake.run_one_example = run_one_example
    sys.modules["eval"] = fake

    import scripts.run_eval as run_cli
    import scripts.score_eval as score_cli

    def run_args(output):
        return [
            "--manifest", manifest,
            "--config", config,
            "--model-dir", model_dir,
            "--model-identity", "synthetic:pipeline-fake",
            "--output-dir", output,
        ]

    success_generation = f"results/evaluation/{run_id}-success-generation"
    success_score = f"results/evaluation/{run_id}-success-score"
    success_run_exit = run_cli.main(run_args(success_generation), repo_root=root)
    success_score_exit = score_cli.main(
        ["--manifest", manifest,
         "--predictions", success_generation + "/predictions.jsonl",
         "--config", config,
         "--output-dir", success_score],
        repo_root=root,
    )

    failure_mode = True
    failure_generation = f"results/evaluation/{run_id}-failure-generation"
    failure_score = f"results/evaluation/{run_id}-failure-score"
    failure_run_exit = run_cli.main(run_args(failure_generation), repo_root=root)
    failure_score_exit = score_cli.main(
        ["--manifest", manifest,
         "--predictions", failure_generation + "/predictions.jsonl",
         "--config", config,
         "--output-dir", failure_score],
        repo_root=root,
    )

    module_origins = {
        name: str(Path(module.__file__).resolve())
        for name, module in sorted(sys.modules.items())
        if (name == "evaluation" or name.startswith("evaluation.")
            or name in {"scripts", "scripts.run_eval", "scripts.score_eval"})
        and getattr(module, "__file__", None)
    }
    evidence = {
        "run_id": run_id,
        "module_origins": module_origins,
        "fake_eval_origin": fake.__file__,
        "heavy_modules_loaded": sorted(name for name in blocked if name in sys.modules),
        "trace": trace,
        "exit_codes": {
            "success_generation": success_run_exit,
            "success_score": success_score_exit,
            "failure_generation": failure_run_exit,
            "failure_score": failure_score_exit,
        },
        "success_generation": success_generation,
        "success_score": success_score,
        "failure_generation": failure_generation,
        "failure_score": failure_score,
    }
    evidence_dir = root / "results/evaluation" / f"{run_id}-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    evidence_path = evidence_dir / "pipeline-evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evidence_path": str(evidence_path)}))
    """
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_real_recipe_has_the_validated_generation_contract():
    raw = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    assert raw["dataset_root"] == "data/evaluation/vsp-pinned-6d74d7bb46ae-v1"
    case = make_vsp_case(1)
    test_input = copy.deepcopy(raw)
    test_input["dataset_root"] = str(case["data_root"])
    config = parse_config(test_input, ROOT)

    assert config.benchmark == "vsp"
    assert config.profile == "vsp_validated_v1"
    assert config.categories == ("level3", "level4", "level5", "level6", "level7", "level8")
    assert config.dataset_root == case["data_root"].resolve()
    assert config.max_steps == 100
    assert config.run_metadata.model_identity == "stage1-dora-continuous-v29"
    assert config.run_metadata.prompt_template_identity == (
        "vsp-ilvr-three-images-v1:"
        "8034d008ea5ca5e693c73b5bc8a079e2aa55a1270bbe7e5a4e78c706037af03d"
    )
    assert config.run_metadata.generation_config == {
        "max_new_tokens": 4096,
        "max_sequence_length": 32768,
        "repetition_penalty": 1.05,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
    }
    validate_generation_config(config.run_metadata.generation_config)


def test_copied_reference_free_run_and_score_pipeline_records_success_and_failure():
    case = _prepare_copied_workspace()
    copied_root = case["copied_root"]
    log_dir = ROOT / "logs/evaluation" / (case["run_id"] + "-evidence")
    log_dir.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD,
            str(copied_root),
            case["run_id"],
            str(case["manifest_relative"]),
            str(case["config_relative"]),
            str(case["model_relative"]),
        ],
        cwd=copied_root,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "src:.",
        },
        text=True,
        capture_output=True,
        timeout=60,
    )
    (log_dir / "copied-pipeline.log").write_text(
        "STDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    evidence_path = Path(json.loads(completed.stdout.splitlines()[-1])["evidence_path"])
    evidence = _read_json(evidence_path)

    assert evidence["exit_codes"] == {
        "success_generation": 0,
        "success_score": 0,
        "failure_generation": 1,
        "failure_score": 1,
    }
    assert set(evidence["module_origins"]) == EXPECTED_MODULE_ORIGINS
    assert all(
        Path(origin).is_relative_to(copied_root.resolve())
        for origin in evidence["module_origins"].values()
    )
    assert evidence["fake_eval_origin"] == "<test-only-fake-eval>"
    assert evidence["heavy_modules_loaded"] == []
    assert not (copied_root / "Reference").exists()
    assert not (copied_root / "eval.py").exists()
    assert all((copied_root / name).is_file() for name in PIPELINE_FILES)
    for storage in ("data", "results", "logs", "checkpoints"):
        assert (copied_root / storage).is_symlink()
        assert (copied_root / storage).resolve() == (ROOT / storage).resolve()

    calls = evidence["trace"]["generation_calls"]
    assert [call["sample_keys"] for call in calls] == [
        ["image_input", "text_input"],
        ["image_input", "text_input"],
        ["image_input", "text_input"],
    ]
    dataset_root = (copied_root / case["dataset_relative"]).resolve()
    common_images = [
        str(dataset_root / "prompt-visual-images/system-figure-1.png"),
        str(dataset_root / "prompt-visual-images/system-figure-2.png"),
    ]
    assert [call["ordered_images"] for call in calls] == [
        common_images + [str(dataset_root / "maps/level3/img/0.png")],
        common_images + [str(dataset_root / "maps/level3/img/1.png")],
        common_images + [str(dataset_root / "maps/level3/img/0.png")],
    ]
    assert all(call["generation_config"] == GENERATION_CONFIG for call in calls)

    success_generation = copied_root / evidence["success_generation"]
    success_score = copied_root / evidence["success_score"]
    predictions = _read_jsonl(success_generation / "predictions.jsonl")
    details = _read_jsonl(success_generation / "generation-details.jsonl")
    run = _read_json(success_generation / "run.json")
    summary = _read_json(success_score / "summary.json")
    assert [row["id"] for row in predictions] == ["vsp/level3/0", "vsp/level3/1"]
    assert [row["response"] for row in predictions] == [
        "Action plan: R",
        "Action plan: R",
    ]
    assert [row["generated_suffix_token_ids"] for row in details] == [
        [101, 102],
        [201, 202],
    ]
    assert run["status"] == "complete"
    assert run["N_expected"] == run["processed"] == 2
    assert summary["N_expected"] == summary["observed"] == summary["correct"] == 2
    assert summary["accuracy"] == 1.0
    assert summary["complete"] is True
    assert {row["manifest_sha256"] for row in predictions + details} == {
        run["manifest_sha256"]
    }
    assert {row["run_fingerprint"] for row in predictions + details} == {
        run["run_fingerprint"]
    }
    assert summary["manifest_sha256"] == run["manifest_sha256"]
    assert summary["run_fingerprint"] == run["run_fingerprint"]

    failure_generation = copied_root / evidence["failure_generation"]
    failure_score = copied_root / evidence["failure_score"]
    failed_run = _read_json(failure_generation / "run.json")
    failed_summary = _read_json(failure_score / "summary.json")
    failed_predictions = _read_jsonl(failure_generation / "predictions.jsonl")
    assert failed_run["status"] == "failed"
    assert failed_run["processed"] == 1
    assert failed_predictions[0]["status"] == "infrastructure_error"
    assert "controlled fixture generation failure" in failed_predictions[0]["error"]
    assert failed_summary["N_expected"] == 2
    assert failed_summary["observed"] == 0
    assert failed_summary["complete"] is False
    assert failed_summary["accuracy"] is None
    assert failed_summary["status_counts"] == {
        "infrastructure_error": 1,
        "missing": 1,
    }
    assert failed_summary["manifest_sha256"] == failed_run["manifest_sha256"]
    assert failed_summary["run_fingerprint"] == failed_run["run_fingerprint"]
    print(json.dumps({"fixture_pipeline_evidence": str(evidence_path)}))
