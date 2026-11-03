import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts import analyze_comt_visual as runner


class Qwen2_5_VLSdpaAttention(nn.Module):
    pass


class Qwen2_5_VLFlashAttention2(nn.Module):
    pass


class Qwen2_5_VLVisionFlashAttention2(nn.Module):
    pass


class _DecoderLayer(nn.Module):
    def __init__(self, attention_class=Qwen2_5_VLSdpaAttention):
        super().__init__()
        self.self_attn = attention_class()


class _VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Qwen2_5_VLVisionFlashAttention2()


class FakeModel(nn.Module):
    def __init__(self, *, attention_class=Qwen2_5_VLSdpaAttention):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.model = SimpleNamespace(
            layers=nn.ModuleList([_DecoderLayer(attention_class), _DecoderLayer(attention_class)])
        )
        self.visual = SimpleNamespace(blocks=nn.ModuleList([_VisionBlock()]))
        self.lm_head = nn.Identity()
        self.config = SimpleNamespace(
            stage="stage2",
            latent_size=2,
            latent_pad_token_id=90,
            latent_start_token_id=91,
            latent_end_token_id=92,
            image_token_id=80,
            use_sliding_window=False,
            sliding_window=4096,
            max_position_embeddings=128,
            torch_dtype=torch.bfloat16,
            _attn_implementation="sdpa",
            vision_config=SimpleNamespace(_attn_implementation="flash_attention_2"),
        )
        self.generation_config = SimpleNamespace(eos_token_id=3, repetition_penalty=1.05)
        self.device = torch.device("cpu")


class FakeTokenizer:
    eos_token_id = 3
    unk_token_id = 0
    unk_token = "<unk>"

    _ids = {
        "<|image_pad|>": 80,
        "<|latent_pad|>": 90,
        "<|latent_start|>": 91,
        "<|latent_end|>": 92,
    }

    def convert_tokens_to_ids(self, token):
        return self._ids.get(token, self.unk_token_id)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.messages = []
        self.processor_calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.messages.append(messages)
        content = messages[0]["content"]
        image_count = sum(item["type"] == "image" for item in content)
        text = next(item["text"] for item in content if item["type"] == "text")
        return "<image>" * image_count + text

    def __call__(self, *, text, images, return_tensors, padding):
        self.processor_calls.append({"text": text, "images": images})
        ids = [5]
        result = {}
        if images:
            ids.extend([80] * len(images))
            result["pixel_values"] = torch.ones(len(images), 2, dtype=torch.float32)
            result["image_grid_thw"] = torch.ones(len(images), 3, dtype=torch.long)
        ids.append(6)
        result["input_ids"] = torch.tensor([ids], dtype=torch.long)
        result["attention_mask"] = torch.ones(1, len(ids), dtype=torch.long)
        return result

    def batch_decode(self, token_ids, *, skip_special_tokens):
        ids = token_ids[0].tolist()
        answer = "11" if 11 in ids else "10"
        return [f"assistant: The final answer is: {answer}<|im_end|>"]


def _write_source(tmp_path, rows):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for image in row.get("image_input", []):
            image_path = tmp_path / image
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"image")
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _successful_generator(**context):
    suffix = torch.tensor([[91, 90, 92, 10]], dtype=torch.long)
    sequences = torch.cat((context["prepared"].tensors["input_ids"], suffix), dim=1)
    return SimpleNamespace(
        sequences=sequences,
        generated_suffix_ids=(91, 90, 92, 10),
        latent_start_count=1,
        latent_block_count=1,
        stop_reason="max_new_tokens",
    )


def _run(tmp_path, rows, **overrides):
    source = _write_source(tmp_path, rows)
    processor = FakeProcessor()
    arguments = dict(
        model_dir="checkpoints/fake",
        base_model_dir="checkpoints/base",
        test_data_path=source,
        source_data_path=source,
        indices=None,
        output_dir=tmp_path / "results" / "run",
        visual_ablation_mode="baseline",
        attention_backend="sdpa",
        replacement_map=None,
        max_new_tokens=8,
        max_sequence_length=128,
        repetition_penalty=1.05,
        temperature=0.0,
        top_p=1.0,
        limit=None,
        artifact_loader=lambda **_kwargs: (FakeModel(), processor),
        generation_runner=_successful_generator,
        image_loader=lambda paths: [f"opened:{path}" for path in paths],
    )
    arguments.update(overrides)
    result = runner.run_analysis(**arguments)
    return result, processor, arguments["output_dir"]


def test_run_uses_only_recipient_text_for_prompt_and_e1_has_no_visual_tensors(tmp_path):
    """Catches gold/helper leakage and an E1 path that still prepares image inputs."""
    rows = [{
        "text_input": "VISIBLE QUESTION",
        "image_input": ["images_comt/creation/a.png"],
        "sequence_plan": [{"helper_image": "SECRET-HELPER", "content": "SECRET-RATIONALE"}],
        "original_final_answer": "SECRET-GOLD-10",
    }]
    captured = {}

    def generate(**context):
        captured.update(context)
        return _successful_generator(**context)

    result, processor, output_dir = _run(
        tmp_path,
        rows,
        visual_ablation_mode="no_input_image",
        generation_runner=generate,
    )

    rendered = processor.processor_calls[0]["text"][0]
    assert rendered == "VISIBLE QUESTION"
    assert "SECRET" not in rendered
    assert processor.processor_calls[0]["images"] is None
    assert set(captured["prepared"].tensors) == {"input_ids", "attention_mask"}
    assert captured["allow_latent_generation"] is True
    assert captured["intervention"].mode == "no_input_image"
    row = result["rows"][0]
    assert row["latent_block_count"] == 1
    assert row["recipient_image_input"] == ["images_comt/creation/a.png"]
    assert row["effective_image_input"] == []
    assert row["image_identity"] == "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    assert json.loads((output_dir / "predictions.jsonl").read_text()) == row


def test_run_records_raw_token_positions_scoring_and_instantiated_runtime_metadata(tmp_path):
    """Catches evidence derived from cleaned text or requested rather than actual runtime state."""
    rows = [{
        "text_input": "question",
        "image_input": ["images_comt/deletion/a.png"],
        "original_final_answer": "10",
    }]
    result, _, output_dir = _run(tmp_path, rows)

    row = result["rows"][0]
    assert row["generated_suffix_token_ids"] == [91, 90, 92, 10]
    assert row["latent_start_positions"] == {"suffix": [0], "absolute": [3]}
    assert row["latent_end_positions"] == {"suffix": [2], "absolute": [5]}
    assert row["prediction_normalized"] == "10"
    assert row["gold_normalized"] == "10"
    assert row["match"] is True
    assert row["category"] == "deletion"
    metadata = result["metadata"]
    assert metadata["runtime"]["decoder_attention_classes"] == [
        "Qwen2_5_VLSdpaAttention", "Qwen2_5_VLSdpaAttention"
    ]
    assert metadata["runtime"]["vision_attention_classes"] == [
        "Qwen2_5_VLVisionFlashAttention2"
    ]
    assert metadata["runtime"]["parameter_dtypes"] == ["torch.float32"]
    assert metadata["runtime"]["configured_torch_dtype"] == "torch.bfloat16"
    assert metadata["runtime"]["input_tensor_dtypes"]["pixel_values"] == ["torch.float32"]
    assert metadata["generation"]["latent_size"] == 2
    assert json.loads((output_dir / "predictions.jsonl").read_text()) == row
    assert json.loads((output_dir / "metadata.json").read_text()) == metadata


def test_image_token_id_is_resolved_from_tokenizer_and_checked_against_model(tmp_path):
    """Catches a hard-coded image token ID or tokenizer/config disagreement."""
    rows = [{"text_input": "q", "image_input": [], "original_final_answer": "10"}]
    captured = {}

    def generate(**context):
        captured["image_token_id"] = context["intervention"].image_token_id
        return _successful_generator(**context)

    _run(tmp_path, rows, generation_runner=generate)
    assert captured["image_token_id"] == 80

    def bad_loader(**_kwargs):
        model = FakeModel()
        model.config.image_token_id = 81
        return model, FakeProcessor()

    result, _, _ = _run(tmp_path / "bad", rows, artifact_loader=bad_loader)
    assert result["metadata"]["status"] == "failed"
    assert "image token ID mismatch" in result["metadata"]["error"]


def test_masked_modes_reject_every_incompatible_decoder_layer(tmp_path):
    """Catches a silent masked-mode fallback onto a decoder lacking the 4-D SDPA contract."""
    rows = [{"text_input": "q", "image_input": [], "original_final_answer": "10"}]
    result, _, _ = _run(
        tmp_path,
        rows,
        visual_ablation_mode="global_image_attention_block",
        artifact_loader=lambda **_kwargs: (FakeModel(attention_class=Qwen2_5_VLFlashAttention2), FakeProcessor()),
    )
    assert result["metadata"]["status"] == "failed"
    assert "requires Qwen2_5_VLSdpaAttention at every decoder layer" in result["metadata"]["error"]


def test_neutral_control_runner_keeps_images_and_fresh_policies(tmp_path):
    """Catches neutral runs dropping visual inputs or sharing intervention state."""
    seen = []

    def generate(**context):
        policy = context["intervention"]
        assert policy.mode == "causal_mask_control"
        assert {"pixel_values", "image_grid_thw"} <= set(context["inputs"])
        assert context["allow_latent_generation"] is True
        seen.append(policy)
        return _successful_generator(**context)

    rows = [
        {
            "text_input": "question A",
            "image_input": ["images_comt/deletion/a.png"],
            "original_final_answer": "10",
        },
        {
            "text_input": "question B",
            "image_input": ["images_comt/deletion/b.png"],
            "original_final_answer": "10",
        },
    ]

    result, _, _ = _run(
        tmp_path,
        rows,
        visual_ablation_mode="causal_mask_control",
        generation_runner=generate,
    )

    assert result["metadata"]["status"] == "completed"
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert {row["condition"] for row in result["rows"]} == {
        "causal_mask_control"
    }


@pytest.mark.parametrize(
    ("backend", "attention"),
    [
        ("native", "Qwen2_5_VLSdpaAttention"),
        ("sdpa", "Qwen2_5_VLFlashAttention2"),
    ],
)
def test_neutral_control_rejects_incompatible_backend(backend, attention):
    """Catches the neutral 4-D mask running through an incompatible decoder path."""
    with pytest.raises(ValueError, match="requires"):
        runner._require_mask_compatible(
            {"decoder_attention_classes": [attention]},
            "causal_mask_control",
            backend,
        )


def test_cli_accepts_neutral_control():
    """Catches the implemented neutral mode being unreachable from the run CLI."""
    args = runner.build_parser().parse_args(
        [
            "run",
            "--model-dir",
            "checkpoints/fake",
            "--test-data-path",
            "data/TEST.jsonl",
            "--source-data-path",
            "data/TEST.jsonl",
            "--output-dir",
            "results/fake",
            "--visual-ablation-mode",
            "causal_mask_control",
            "--attention-backend",
            "sdpa",
        ]
    )

    assert args.visual_ablation_mode == "causal_mask_control"


def _scientific_rows(rows):
    ignored = {"inference_time_sec", "wall_time_sec"}
    return [
        {key: value for key, value in row.items() if key not in ignored}
        for row in rows
    ]


def test_interrupted_run_resumes_completed_samples_without_changing_results(tmp_path):
    """Catches resume discarding or regenerating a completed sample."""
    rows = [
        {"text_input": "question A", "image_input": [], "original_final_answer": "10"},
        {"text_input": "question B", "image_input": [], "original_final_answer": "10"},
    ]
    calls = []

    def interrupt_second(**context):
        calls.append(context["sample"]["text_input"])
        if len(calls) == 2:
            raise KeyboardInterrupt("controlled interruption")
        return _successful_generator(**context)

    with pytest.raises(KeyboardInterrupt, match="controlled interruption"):
        _run(tmp_path / "resumed", rows, generation_runner=interrupt_second)

    output = tmp_path / "resumed" / "results" / "run"
    partials = sorted((output / "partial").glob("*.json"))
    assert [path.name for path in partials] == ["000000-0.json"]

    resumed_calls = []

    def resume_generator(**context):
        resumed_calls.append(context["sample"]["text_input"])
        return _successful_generator(**context)

    resumed, _, _ = _run(
        tmp_path / "resumed",
        rows,
        resume=True,
        generation_runner=resume_generator,
    )
    uninterrupted, _, _ = _run(tmp_path / "uninterrupted", rows)

    assert resumed_calls == ["question B"]
    assert resumed["metadata"]["resume"]["reused_count"] == 1
    assert _scientific_rows(resumed["rows"]) == _scientific_rows(
        uninterrupted["rows"]
    )
    assert [json.loads(line)["index"] for line in (output / "predictions.jsonl").read_text().splitlines()] == [0, 1]


def test_resume_rejects_generation_identity_mismatch_before_inference(tmp_path):
    """Catches resume mixing rows produced with different generation settings."""
    rows = [{"text_input": "question", "image_input": [], "original_final_answer": "10"}]
    _run(tmp_path, rows)
    calls = []

    with pytest.raises(ValueError, match="resume identity mismatch"):
        _run(
            tmp_path,
            rows,
            resume=True,
            max_new_tokens=9,
            generation_runner=lambda **context: calls.append(context) or _successful_generator(**context),
        )

    assert calls == []


def test_resume_rejects_changed_model_artifact_before_inference(tmp_path):
    """Catches resume trusting a reused model path whose bytes changed."""
    rows = [{"text_input": "question", "image_input": [], "original_final_answer": "10"}]
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_bytes(b"model-v1")
    _run(tmp_path, rows, model_dir=model)
    weights.write_bytes(b"model-v2")
    calls = []

    with pytest.raises(ValueError, match="resume identity mismatch"):
        _run(
            tmp_path,
            rows,
            model_dir=model,
            resume=True,
            generation_runner=lambda **context: calls.append(context) or _successful_generator(**context),
        )

    assert calls == []


def test_resume_rejects_changed_effective_image_before_inference(tmp_path):
    """Catches resume mixing rows after an image changed behind a stable JSONL path."""
    rows = [{
        "text_input": "question",
        "image_input": ["images_comt/deletion/a.png"],
        "original_final_answer": "10",
    }]
    dataset = tmp_path / "dataset"
    source = _write_source(dataset, rows)
    output = tmp_path / "results" / "run"
    harness = tmp_path / "harness"
    _run(
        harness,
        rows,
        test_data_path=source,
        source_data_path=source,
        output_dir=output,
    )
    (dataset / "images_comt/deletion/a.png").write_bytes(b"changed-image")
    calls = []

    with pytest.raises(ValueError, match="resume identity mismatch"):
        _run(
            harness,
            rows,
            test_data_path=source,
            source_data_path=source,
            output_dir=output,
            resume=True,
            generation_runner=lambda **context: calls.append(context) or _successful_generator(**context),
        )

    assert calls == []


def test_resume_identity_scan_preserves_missing_image_error_row(tmp_path):
    """Catches pre-resume hashing turning one missing image into a run-level failure."""
    rows = [{
        "text_input": "question",
        "image_input": ["images_comt/deletion/missing.png"],
        "original_final_answer": "10",
    }]
    dataset = tmp_path / "dataset"
    source = _write_source(dataset, rows)
    (dataset / "images_comt/deletion/missing.png").unlink()

    result, _, _ = _run(
        tmp_path / "harness",
        rows,
        test_data_path=source,
        source_data_path=source,
        output_dir=tmp_path / "results" / "run",
    )

    assert len(result["rows"]) == 1
    assert result["rows"][0]["index"] == 0
    assert "Image not found" in result["rows"][0]["error"]


def test_resume_reruns_corrupt_partial_record(tmp_path):
    """Catches a malformed partial row being trusted or blocking recovery."""
    rows = [
        {"text_input": "question A", "image_input": [], "original_final_answer": "10"},
        {"text_input": "question B", "image_input": [], "original_final_answer": "10"},
    ]
    calls = []

    def interrupt_second(**context):
        calls.append(context["sample"]["text_input"])
        if len(calls) == 2:
            raise KeyboardInterrupt("controlled interruption")
        return _successful_generator(**context)

    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, rows, generation_runner=interrupt_second)

    partial = tmp_path / "results" / "run" / "partial" / "000000-0.json"
    partial.write_text("{truncated", encoding="utf-8")
    resumed_calls = []

    result, _, _ = _run(
        tmp_path,
        rows,
        resume=True,
        generation_runner=lambda **context: resumed_calls.append(context["sample"]["text_input"])
        or _successful_generator(**context),
    )

    assert resumed_calls == ["question A", "question B"]
    assert result["metadata"]["resume"]["reused_count"] == 0


def test_failed_resume_preserves_completed_canonical_outputs(tmp_path):
    """Catches a run-level resume failure erasing a previously completed result."""
    rows = [{"text_input": "question", "image_input": [], "original_final_answer": "10"}]
    _run(tmp_path, rows)
    output = tmp_path / "results" / "run"
    predictions_before = (output / "predictions.jsonl").read_bytes()
    metadata_before = (output / "metadata.json").read_bytes()

    def fail_loader(**_kwargs):
        raise RuntimeError("loader failed during resume")

    with pytest.raises(RuntimeError, match="loader failed during resume"):
        _run(tmp_path, rows, resume=True, artifact_loader=fail_loader)

    assert (output / "predictions.jsonl").read_bytes() == predictions_before
    assert (output / "metadata.json").read_bytes() == metadata_before


def test_resume_rejects_stochastic_generation_without_rng_boundary_state(tmp_path):
    """Catches stochastic resume claiming equivalence without restoring RNG state."""
    rows = [{"text_input": "question", "image_input": [], "original_final_answer": "10"}]
    _run(tmp_path, rows, temperature=0.7)

    with pytest.raises(ValueError, match="stochastic generation"):
        _run(tmp_path, rows, temperature=0.7, resume=True)


def test_resume_reruns_checksum_valid_but_structurally_incomplete_partial(tmp_path):
    """Catches a syntactically valid partial with missing scientific fields being reused."""
    rows = [{"text_input": "question", "image_input": [], "original_final_answer": "10"}]
    _run(tmp_path, rows)
    partial = tmp_path / "results" / "run" / "partial" / "000000-0.json"
    record = json.loads(partial.read_text(encoding="utf-8"))
    record["row"].pop("condition")
    record["row_sha256"] = runner._value_identity(record["row"])
    partial.write_text(json.dumps(record), encoding="utf-8")
    calls = []

    result, _, _ = _run(
        tmp_path,
        rows,
        resume=True,
        generation_runner=lambda **context: calls.append(context["sample"]["text_input"])
        or _successful_generator(**context),
    )

    assert calls == ["question"]
    assert result["metadata"]["resume"]["reused_count"] == 0
    assert result["rows"][0]["condition"] == "baseline"


def test_resume_reruns_partial_whose_row_bytes_changed(tmp_path):
    """Catches a valid-looking partial whose persisted row no longer matches its digest."""
    rows = [{"text_input": "question", "image_input": [], "original_final_answer": "10"}]
    _run(tmp_path, rows)
    partial = tmp_path / "results" / "run" / "partial" / "000000-0.json"
    record = json.loads(partial.read_text(encoding="utf-8"))
    record["row"]["generated_suffix_token_ids"] = [999]
    partial.write_text(json.dumps(record), encoding="utf-8")
    calls = []

    result, _, _ = _run(
        tmp_path,
        rows,
        resume=True,
        generation_runner=lambda **context: calls.append(context["sample"]["text_input"])
        or _successful_generator(**context),
    )

    assert calls == ["question"]
    assert result["metadata"]["resume"]["reused_count"] == 0
    assert result["rows"][0]["generated_suffix_token_ids"] != [999]


def test_resume_reruns_partial_copied_from_another_run_identity(tmp_path):
    """Catches a checksum-valid partial from a different run being reused."""
    rows = [{"text_input": "question", "image_input": [], "original_final_answer": "10"}]
    model_a = tmp_path / "model-a"
    model_b = tmp_path / "model-b"
    model_a.mkdir()
    model_b.mkdir()
    (model_a / "weights.bin").write_bytes(b"model-a")
    (model_b / "weights.bin").write_bytes(b"model-b")
    _run(tmp_path / "a", rows, model_dir=model_a)
    _run(tmp_path / "b", rows, model_dir=model_b)
    partial_a = tmp_path / "a" / "results" / "run" / "partial" / "000000-0.json"
    partial_b = tmp_path / "b" / "results" / "run" / "partial" / "000000-0.json"
    partial_b.write_bytes(partial_a.read_bytes())
    calls = []

    result, _, _ = _run(
        tmp_path / "b",
        rows,
        model_dir=model_b,
        resume=True,
        generation_runner=lambda **context: calls.append(context["sample"]["text_input"])
        or _successful_generator(**context),
    )

    assert calls == ["question"]
    assert result["metadata"]["resume"]["reused_count"] == 0


@pytest.mark.parametrize("corruption", ["invalid_utf8", "nonfinite_json"])
def test_resume_reruns_partial_with_unreadable_or_unhashable_content(
    tmp_path, corruption
):
    """Catches corrupt bytes or non-finite JSON aborting the whole resume."""
    rows = [{"text_input": "question", "image_input": [], "original_final_answer": "10"}]
    _run(tmp_path, rows)
    partial = tmp_path / "results" / "run" / "partial" / "000000-0.json"
    if corruption == "invalid_utf8":
        partial.write_bytes(b"\xff\xfe")
    else:
        record = json.loads(partial.read_text(encoding="utf-8"))
        record["row"]["inference_time_sec"] = float("nan")
        partial.write_text(json.dumps(record), encoding="utf-8")
    calls = []

    result, _, _ = _run(
        tmp_path,
        rows,
        resume=True,
        generation_runner=lambda **context: calls.append(context["sample"]["text_input"])
        or _successful_generator(**context),
    )

    assert calls == ["question"]
    assert result["metadata"]["resume"]["reused_count"] == 0


def test_jsonl_publication_interruption_preserves_existing_file(tmp_path, monkeypatch):
    """Catches final publication truncating the previous canonical predictions."""
    output = tmp_path / "predictions.jsonl"
    output.write_bytes(b"old-canonical-result\n")
    real_dumps = runner.json.dumps
    calls = 0

    def interrupt_second_row(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("publication interrupted")
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(runner.json, "dumps", interrupt_second_row)
    with pytest.raises(KeyboardInterrupt, match="publication interrupted"):
        runner._write_jsonl(output, [{"index": 0}, {"index": 1}])

    assert output.read_bytes() == b"old-canonical-result\n"


def test_metadata_publication_failure_rolls_back_both_canonical_files(
    tmp_path, monkeypatch
):
    """Catches a new predictions file being paired with old metadata."""
    rows = [{"text_input": "question", "image_input": [], "original_final_answer": "10"}]
    _run(tmp_path, rows)
    output = tmp_path / "results" / "run"
    predictions = output / "predictions.jsonl"
    metadata = output / "metadata.json"
    predictions.write_bytes(b"old-canonical-predictions\n")
    predictions_before = predictions.read_bytes()
    metadata_before = metadata.read_bytes()
    real_atomic_write_json = runner._atomic_write_json

    def interrupt_metadata(path, value):
        if path == metadata:
            raise KeyboardInterrupt("metadata publication interrupted")
        return real_atomic_write_json(path, value)

    monkeypatch.setattr(runner, "_atomic_write_json", interrupt_metadata)
    with pytest.raises(KeyboardInterrupt, match="metadata publication interrupted"):
        _run(tmp_path, rows, resume=True)

    assert predictions.read_bytes() == predictions_before
    assert metadata.read_bytes() == metadata_before


def test_git_dirty_identity_changes_for_any_tracked_source_edit(tmp_path):
    """Catches a hand-maintained dependency list missing behavior-affecting edits."""
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    dependency = tmp_path / "previously-unlisted-dependency.py"
    dependency.write_text("VERSION = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", dependency.name], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    identity = getattr(runner, "_git_dirty_identity", lambda _root: None)
    before = identity(tmp_path)
    dependency.write_text("VERSION = 2\n", encoding="utf-8")
    after = identity(tmp_path)

    assert before != after


def test_resume_rejects_changed_runtime_code_bytes(tmp_path, monkeypatch):
    """Catches dirty behavior code changing without a Git commit change."""
    rows = [{"text_input": "question", "image_input": [], "original_final_answer": "10"}]
    code_path = tmp_path / "runtime_behavior.py"
    code_path.write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "_runtime_code_identity",
        lambda: [runner._artifact_identity(code_path)],
        raising=False,
    )
    _run(tmp_path, rows)
    code_path.write_text("VERSION = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="resume identity mismatch"):
        _run(tmp_path, rows, resume=True)


def test_runtime_metadata_records_software_device_and_determinism():
    """Catches resume identity omitting result-affecting runtime characteristics."""
    model = FakeModel()
    runtime = runner._runtime_metadata(model, model, "sdpa")

    assert runtime["software"]["python"]
    assert runtime["software"]["torch"] == torch.__version__
    assert "peft" in runtime["software"]
    assert "flash_attn" in runtime["software"]
    assert runtime["device"]["type"] == "cpu"
    assert isinstance(runtime["determinism"]["algorithms_enabled"], bool)


def test_cli_accepts_resume():
    """Catches the durable resume behavior being unreachable from the CLI."""
    args = runner.build_parser().parse_args(
        [
            "run",
            "--model-dir",
            "checkpoints/fake",
            "--test-data-path",
            "data/TEST.jsonl",
            "--source-data-path",
            "data/TEST.jsonl",
            "--output-dir",
            "results/fake",
            "--visual-ablation-mode",
            "baseline",
            "--attention-backend",
            "sdpa",
            "--resume",
        ]
    )

    assert args.resume is True


def test_replacement_keeps_recipient_text_and_scores_only_valid_matched_donor_gold(tmp_path):
    """Catches image replacement swapping the question or accepting an unmatched donor."""
    rows = [
        {"text_input": "same   question", "image_input": ["images_comt/update/a.png"], "original_final_answer": "10"},
        {"text_input": "same question", "image_input": ["images_comt/update/b.png"], "original_final_answer": "11"},
        {"text_input": "different", "image_input": ["images_comt/update/c.png"], "original_final_answer": "12"},
    ]
    result, processor, _ = _run(
        tmp_path,
        rows,
        indices=(0, 2),
        visual_ablation_mode="image_replacement",
        replacement_map={0: 1, 2: 1},
    )

    valid, invalid = result["rows"]
    assert processor.processor_calls[0]["text"] == ["<image>same   question"]
    assert processor.processor_calls[0]["images"] == ["opened:" + str(tmp_path / "images_comt/update/b.png")]
    assert valid["replacement_donor_index"] == 1
    assert valid["gold_final_answer"] == "11"
    assert valid["gold_source"] == "replacement_donor"
    assert invalid["index"] == 2
    assert "normalized question mismatch" in invalid["error"]
    assert result["metadata"]["status"] == "failed"


def test_valid_replacement_inference_error_retains_donor_gold_lineage(tmp_path):
    """Catches a failed E5 attempt being mislabeled with the recipient's gold."""
    rows = [
        {"text_input": "same q", "image_input": ["images_comt/update/a.png"], "original_final_answer": "10"},
        {"text_input": "same q", "image_input": ["images_comt/update/b.png"], "original_final_answer": "11"},
    ]

    def fail(**_context):
        raise RuntimeError("generation exploded")

    result, _, _ = _run(
        tmp_path,
        rows,
        indices=(0,),
        visual_ablation_mode="image_replacement",
        replacement_map={0: 1},
        generation_runner=fail,
    )

    row = result["rows"][0]
    assert row["replacement_donor_index"] == 1
    assert row["gold_final_answer"] == "11"
    assert row["gold_normalized"] == "11"
    assert row["gold_source"] == "replacement_donor"


def test_requested_missing_index_and_generation_error_are_serialized_as_failed_rows(tmp_path):
    """Catches requested or attempted samples disappearing from the output denominator."""
    rows = [{"text_input": "q", "image_input": [], "original_final_answer": "10"}]

    def fail(**_context):
        raise RuntimeError("generation exploded")

    result, _, _ = _run(tmp_path, rows, indices=(0, 7), generation_runner=fail)
    assert [row["index"] for row in result["rows"]] == [0, 7]
    assert result["rows"][0]["prediction_parse_status"] == "inference_error"
    assert result["rows"][0]["error"] == "generation exploded"
    assert result["rows"][1]["error"] == "source index 7 is missing"
    assert result["rows"][1]["condition"] == "baseline"
    assert result["rows"][1]["checkpoint"] == "checkpoints/fake"
    assert result["metadata"]["status"] == "failed"
    assert result["metadata"]["completed_count"] == 0
    assert result["metadata"]["error_count"] == 2


def test_native_loader_selection_uses_stage_artifact_and_sdpa_uses_mixed_loader(tmp_path):
    """Catches Stage 1/2 native loaders being interchanged or SDPA bypassing eval.py."""
    stage1 = tmp_path / "stage1"
    stage2 = tmp_path / "stage2"
    stage1.mkdir()
    stage2.mkdir()
    (stage1 / "stage1_peft_metadata.json").write_text("{}")
    (stage2 / "stage2_peft_metadata.json").write_text("{}")

    assert runner.select_loader_module_name(stage1, "native") == "eval"
    assert runner.select_loader_module_name(stage2, "native") == "src.evaluate_deepseed"
    assert runner.select_loader_module_name(stage2, "sdpa") == "eval"


def test_baseline_equivalence_requires_raw_prompt_and_scoring_identity():
    """Catches a baseline bridge calling normalized text equality exact equivalence."""
    native = [{
        "index": 4, "generated_suffix_token_ids": [1, 2], "prompt_identity": "p",
        "prediction_normalized": "10", "gold_normalized": "10", "match": True,
        "prediction_parse_status": "parsed", "gold_parse_status": "parsed",
        "invalid_gold_reason": None,
    }]
    same = [dict(native[0])]
    changed = [dict(native[0], generated_suffix_token_ids=[1, 3])]

    exact = runner.baseline_equivalence(native, same)
    assert exact["exact"] is True
    assert exact["raw_token_id_match_count"] == 1
    mismatch = runner.baseline_equivalence(native, changed)
    assert mismatch["exact"] is False
    assert mismatch["mismatched_indices"] == {"raw_token_ids": [4], "prompt_identity": [], "scoring": []}


def test_baseline_equivalence_reports_identical_error_rows_as_unavailable():
    """Catches matching absent evidence satisfying the exact no-op gate."""
    failed = [{
        "index": 4,
        "generated_suffix_token_ids": [],
        "prompt_identity": None,
        "prediction_normalized": "",
        "prediction_parse_status": "inference_error",
        "gold_normalized": "10",
        "gold_parse_status": "parsed",
        "invalid_gold_reason": None,
        "match": False,
        "error": "generation exploded",
    }]

    result = runner.baseline_equivalence(failed, [dict(failed[0])])

    assert result["exact"] is False
    assert result["paired_count"] == 1
    assert result["comparable_count"] == 0
    assert result["unavailable_count"] == 1
    assert result["raw_token_id_match_count"] == 0
    assert result["unavailable_evidence"] == [{
        "index": 4,
        "baseline": ["inference_error", "missing_prompt_identity"],
        "analysis": ["inference_error", "missing_prompt_identity"],
    }]


def test_baseline_equivalence_keeps_successful_invalid_gold_row_comparable():
    """Catches invalid gold excluding otherwise valid raw no-op evidence."""
    row = {
        "index": 9,
        "generated_suffix_token_ids": [7, 8],
        "prompt_identity": "prompt-9",
        "prediction_normalized": "answer",
        "prediction_parse_status": "parsed",
        "gold_normalized": "",
        "gold_parse_status": "invalid_gold",
        "invalid_gold_reason": "missing",
        "match": False,
        "error": None,
    }

    result = runner.baseline_equivalence([row], [dict(row)])

    assert result["exact"] is True
    assert result["comparable_count"] == 1
    assert result["unavailable_count"] == 0
    assert result["raw_token_id_match_count"] == 1


def test_groups_and_compare_are_cpu_file_operations_with_complete_outputs(tmp_path, monkeypatch):
    """Catches incomplete group/compare artifacts or a comparison that omits exact evidence."""
    monkeypatch.chdir(tmp_path)
    source = _write_source(tmp_path, [
        {"text_input": "same q", "image_input": [], "original_final_answer": "10"},
        {"text_input": "same   q", "image_input": [], "original_final_answer": "11"},
    ])
    manifest = runner.create_group_manifest(source, Path("data/groups.json"))
    assert manifest["groups"] == [[0, 1]]
    assert manifest["indices"] == [0, 1]

    baseline = tmp_path / "baseline.jsonl"
    condition = tmp_path / "condition.jsonl"
    base_rows = [
        {"index": 0, "category": "creation", "generated_suffix_token_ids": [1], "prompt_identity": "p0", "prediction_normalized": "10", "prediction_parse_status": "parsed", "gold_normalized": "10", "gold_parse_status": "parsed", "invalid_gold_reason": None, "match": True, "latent_block_count": 0, "inference_time_sec": 1.0},
        {"index": 1, "category": "creation", "generated_suffix_token_ids": [2], "prompt_identity": "p1", "prediction_normalized": "10", "prediction_parse_status": "parsed", "gold_normalized": "11", "gold_parse_status": "parsed", "invalid_gold_reason": None, "match": False, "latent_block_count": 0, "inference_time_sec": 1.0},
    ]
    condition_rows = [dict(row) for row in base_rows]
    condition_rows[1].update(prediction_normalized="11", match=True, generated_suffix_token_ids=[3])
    baseline.write_text("".join(json.dumps(row) + "\n" for row in base_rows))
    condition.write_text("".join(json.dumps(row) + "\n" for row in condition_rows))

    summary = runner.compare_prediction_files(baseline, condition, tmp_path / "comparison")
    assert summary["paired"]["accuracy_delta_pp"] == 50.0
    assert summary["equivalence"]["raw_token_id_match_count"] == 1
    assert (tmp_path / "comparison/summary.json").is_file()
    assert (tmp_path / "comparison/summary.csv").is_file()
    assert (tmp_path / "comparison/analysis.md").is_file()
    assert "Unavailable equivalence evidence: 0" in (
        tmp_path / "comparison/analysis.md"
    ).read_text()
