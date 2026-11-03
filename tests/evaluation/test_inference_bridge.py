from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import math
import struct
import subprocess
import zlib

import pytest

import evaluation.inference_bridge as bridge
from evaluation.io import parse_config, read_manifest
from evaluation.schema import ContractError, ModelInput
from evaluation.vsp_prepare import PROCESSED_PROMPT
from vsp_fixtures import make_vsp_case

from evaluation.inference_bridge import (
    generate_one,
    load_runtime,
    runtime_metadata,
    validate_generation_config,
    validate_runtime_paths,
    validate_vsp_inputs,
)


def _prepared_case(sample_count=1):
    case = make_vsp_case(sample_count)
    config = parse_config(case["config"], case["root"])
    prepared = read_manifest(case["manifest"], config, case["root"])
    return case, config, prepared


def _legacy_result(**updates):
    result = {
        "assistant_only": "Action plan: R",
        "generated_suffix_token_ids": [21, 22],
        "prompt_identity": "a" * 64,
        "prompt_token_length": 7,
        "raw_output": "prompt Action plan: L\nAction plan: R",
        "extracted_final_answer": "R",
        "latent_start_count": 1,
        "latent_block_count": 1,
        "stop_reason": "eos",
        "inference_time": 0.1,
    }
    result.update(updates)
    return result


def _png_with_valid_crcs_and_broken_idat() -> bytes:
    def chunk(kind, payload):
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(
            ">I", zlib.crc32(body) & 0xFFFFFFFF
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk("IHDR".encode(), struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk("IDAT".encode(), b"not-a-zlib-stream")
        + chunk("IEND".encode(), b"")
    )


def test_import_is_lightweight_and_load_runtime_is_explicit(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    code = """
import json, sys
import evaluation.inference_bridge as bridge
print(json.dumps({name: name in sys.modules for name in ('torch', 'transformers', 'eval')}))
"""
    completed = subprocess.run(
        [str(root / "venv/ilvr-venv/bin/python"), "-c", code],
        cwd=root,
        env={
            **dict(__import__("os").environ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "src:tests/evaluation:.",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "torch": False,
        "transformers": False,
        "eval": False,
    }
    imported = []
    expected = object()
    monkeypatch.setattr(
        bridge.importlib,
        "import_module",
        lambda name: imported.append(name) or expected,
    )
    assert load_runtime() is expected
    assert imported == ["eval"]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"extra": 1}, "unexpected or missing"),
        ({"max_new_tokens": True}, "max_new_tokens"),
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"max_sequence_length": 0}, "max_sequence_length"),
        ({"max_sequence_length": 2.0}, "max_sequence_length"),
        ({"repetition_penalty": True}, "repetition_penalty"),
        ({"repetition_penalty": 0}, "repetition_penalty"),
        ({"temperature": float("nan")}, "temperature"),
        ({"temperature": -0.1}, "temperature"),
        ({"top_p": float("inf")}, "top_p"),
        ({"top_p": 0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
        ({"do_sample": 1}, "do_sample"),
        ({"temperature": 0.1, "do_sample": False}, "do_sample"),
        ({"temperature": 0.0, "do_sample": True}, "do_sample"),
    ],
)
def test_generation_config_rejects_invalid_exact_contract(updates, message):
    raw = {
        "max_new_tokens": 0,
        "max_sequence_length": 32768,
        "repetition_penalty": 1.05,
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
    }
    raw.update(updates)
    with pytest.raises(ContractError, match=message):
        validate_generation_config(raw)


def test_generation_config_accepts_zero_tokens_and_finite_numeric_values():
    validate_generation_config(
        {
            "max_new_tokens": 0,
            "max_sequence_length": 1,
            "repetition_penalty": 1,
            "temperature": 0.25,
            "top_p": 0.5,
            "do_sample": True,
        }
    )


def test_vsp_inputs_accept_exact_processed_prompt_and_three_image_order():
    _case, config, prepared = _prepared_case(2)
    validate_vsp_inputs(prepared, config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("choices", ("R",), "choices"),
        (
            "images",
            (
                "prompt-visual-images/system-figure-2.png",
                "prompt-visual-images/system-figure-1.png",
                "maps/level3/img/0.png",
            ),
            "image order",
        ),
        ("images", ("/absolute.png",), "image order"),
        ("text", "changed prompt", "processed prompt"),
    ],
)
def test_vsp_inputs_reject_model_input_protocol_changes(field, value, message):
    _case, config, prepared = _prepared_case()
    sample = prepared.samples[0]
    changed_input = replace(sample.model_input, **{field: value})
    changed = replace(prepared, samples=(replace(sample, model_input=changed_input),))
    with pytest.raises(ContractError, match=message):
        validate_vsp_inputs(changed, config)


def test_vsp_inputs_reject_categories_source_identity_and_step_limit():
    _case, config, prepared = _prepared_case()
    with pytest.raises(ContractError, match="categories"):
        validate_vsp_inputs(prepared, replace(config, categories=("level3",)))

    sample = prepared.samples[0]
    wrong_source = replace(sample.provenance, source_id="9")
    with pytest.raises(ContractError, match="source identity"):
        validate_vsp_inputs(
            replace(prepared, samples=(replace(sample, provenance=wrong_source),)),
            config,
        )

    with pytest.raises(ContractError, match="max_steps"):
        validate_vsp_inputs(prepared, replace(config, max_steps=99))


def test_vsp_inputs_reject_prompt_symlink_escape_and_unreadable_png():
    case, config, prepared = _prepared_case()
    prompt_path = config.dataset_root / PROCESSED_PROMPT
    escaped = case["root"] / "data/evaluation" / (case["run_id"] + "-prompt.txt")
    escaped.write_text(prompt_path.read_text(encoding="utf-8"), encoding="utf-8")
    prompt_path.unlink()
    prompt_path.symlink_to(escaped)
    with pytest.raises(ContractError, match="processed prompt"):
        validate_vsp_inputs(prepared, config)

    prompt_path.unlink()
    prompt_path.write_text(prepared.samples[0].model_input.text, encoding="utf-8")
    image_path = config.dataset_root / prepared.samples[0].model_input.images[-1]
    image_path.write_bytes(b"not a PNG")
    with pytest.raises(ContractError, match="PNG"):
        validate_vsp_inputs(prepared, config)


def test_vsp_inputs_reject_png_whose_valid_chunks_hide_broken_pixel_stream():
    from PIL import Image

    _case, config, prepared = _prepared_case()
    image_path = config.dataset_root / prepared.samples[0].model_input.images[-1]
    image_path.write_bytes(_png_with_valid_crcs_and_broken_idat())

    with Image.open(image_path) as image:
        image.verify()
    with pytest.raises(ContractError, match="PNG"):
        validate_vsp_inputs(prepared, config)


def test_runtime_paths_enforce_storage_and_artifact_kind():
    case = make_vsp_case(1)
    cache = case["root"] / "data/cache/evaluation" / case["run_id"]
    model, base, resolved_cache = validate_runtime_paths(
        case["model_dir"], None, cache, case["root"]
    )
    assert (model, base, resolved_cache) == (
        case["model_dir"].resolve(),
        None,
        cache.resolve(),
    )

    with pytest.raises(ContractError, match="full artifact"):
        validate_runtime_paths(
            case["model_dir"], case["base_model_dir"], cache, case["root"]
        )

    (case["model_dir"] / "adapter_config.json").write_text("{}", encoding="utf-8")
    adapter_paths = validate_runtime_paths(
        case["model_dir"], case["base_model_dir"], cache, case["root"]
    )
    assert adapter_paths[1] == case["base_model_dir"].resolve()
    with pytest.raises(ContractError, match="base_model_dir is required"):
        validate_runtime_paths(case["model_dir"], None, cache, case["root"])


@pytest.mark.parametrize("kind", ["model", "base", "cache"])
def test_runtime_paths_reject_storage_escape(kind):
    case = make_vsp_case(1)
    model = case["model_dir"]
    base = None
    cache = case["root"] / "data/cache/evaluation" / case["run_id"]
    if kind == "model":
        model = case["data_root"]
    elif kind == "base":
        (model / "adapter_config.json").write_text("{}", encoding="utf-8")
        base = case["data_root"]
    else:
        cache = case["model_dir"]
    with pytest.raises(ContractError, match=f"{kind}.*inside"):
        validate_runtime_paths(model, base, cache, case["root"])


def test_bridge_passes_only_model_input_and_preserves_suffix():
    case, config, prepared = _prepared_case()
    sample = prepared.samples[0]
    returned = _legacy_result()
    calls = []

    def recorded(model, processor, payload, kwargs):
        calls.append((model, processor, payload, kwargs))
        assert set(payload) == {"text_input", "image_input"}
        assert payload["text_input"] == sample.model_input.text
        assert [Path(path).name for path in payload["image_input"]] == [
            "system-figure-1.png",
            "system-figure-2.png",
            "0.png",
        ]
        assert kwargs is config.run_metadata.generation_config
        return returned

    model, processor = object(), object()
    result = generate_one(
        model,
        processor,
        sample.model_input,
        config.dataset_root,
        config.run_metadata.generation_config,
        run_one_example=recorded,
    )
    assert len(calls) == 1
    assert result is returned
    assert result["assistant_only"] == "Action plan: R"
    assert result["generated_suffix_token_ids"] == [21, 22]


@pytest.mark.parametrize(
    ("model_input", "message"),
    [
        (ModelInput("prompt", ("/absolute.png",), ()), "inside dataset_root"),
        (ModelInput("prompt", ("../escape.png",), ()), "inside dataset_root"),
        (ModelInput("prompt", ("missing.png",), ()), "readable regular file"),
    ],
)
def test_bridge_rejects_unsafe_or_missing_images_before_call(model_input, message):
    case = make_vsp_case(1)
    with pytest.raises(ContractError, match=message):
        generate_one(
            object(),
            object(),
            model_input,
            case["data_root"],
            case["config"]["run_metadata"]["generation_config"],
            run_one_example=lambda *_args: pytest.fail("runtime called"),
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"assistant_only": None}, "assistant_only"),
        ({"generated_suffix_token_ids": [21, True]}, "suffix"),
        ({"generated_suffix_token_ids": [21, -1]}, "suffix"),
        ({"prompt_identity": "not-a-hash"}, "prompt_identity"),
        ({"prompt_token_length": -1}, "prompt_token_length"),
        ({"latent_start_count": 1.0}, "latent_start_count"),
        ({"latent_block_count": -1}, "latent_block_count"),
        ({"stop_reason": ""}, "stop_reason"),
        ({"inference_time": math.inf}, "inference_time"),
        ({"unexpected": "legacy"}, "unexpected or missing"),
    ],
)
def test_bridge_rejects_corrupt_legacy_results(updates, message):
    case, config, prepared = _prepared_case()
    returned = _legacy_result(**updates)
    with pytest.raises(ContractError, match=message):
        generate_one(
            object(),
            object(),
            prepared.samples[0].model_input,
            config.dataset_root,
            config.run_metadata.generation_config,
            run_one_example=lambda *_args: returned,
        )


def test_runtime_metadata_uses_loaded_objects_and_hashes_small_files():
    case = make_vsp_case(1)
    (case["model_dir"] / "tokenizer_config.json").write_text(
        '{"chat_template":"disk-template"}\n', encoding="utf-8"
    )

    class Serializable:
        def __init__(self, payload):
            self.payload = payload

        def to_dict(self):
            return self.payload

    loaded_config = SimpleNamespace(stage="stage2", latent_size=8)
    loaded_model = SimpleNamespace(
        config=loaded_config,
        generation_config=Serializable({"torch_dtype": object()}),
    )
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template="active-template"),
        chat_template="processor-template",
        to_dict=lambda: {"dtype": object(), "padding_side": "left"},
    )
    token_ids = SimpleNamespace(pad_token_id=90, start_token_id=91, end_token_id=92)
    transformers_module = SimpleNamespace(
        __file__="/runtime/transformers/__init__.py", __version__="9.9.9"
    )
    runtime = SimpleNamespace(
        resolve_qwen2_5_vl_model=lambda model: loaded_model,
        resolve_latent_token_ids=lambda tokenizer, config: token_ids,
        transformers=transformers_module,
    )
    generation = case["config"]["run_metadata"]["generation_config"]

    metadata = runtime_metadata(
        loaded_model, processor, runtime, case["model_dir"], None, generation
    )

    assert metadata["model_dir"] == str(case["model_dir"].resolve())
    assert metadata["base_model_dir"] is None
    assert metadata["qwen_config"] == {"stage": "stage2", "latent_size": 8}
    assert metadata["latent_token_ids"] == {"pad": 90, "start": 91, "end": 92}
    assert metadata["generation_kwargs"] is generation
    assert metadata["processor"]["chat_template"] == "processor-template"
    assert metadata["processor"]["tokenizer_chat_template"] == "active-template"
    assert metadata["model_generation_config"]["torch_dtype"].startswith("<object object")
    assert metadata["transformers"] == {
        "file": "/runtime/transformers/__init__.py",
        "version": "9.9.9",
    }
    assert set(metadata["metadata_sha256"]) == {
        "config.json",
        "tokenizer_config.json",
    }
