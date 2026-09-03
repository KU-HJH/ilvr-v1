from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from step_preference.negatives import (
    CheckpointFile,
    StepNegative,
    greedy_generate_step_negative,
    score_step_completion,
    tokenize_step_completion,
)
from step_preference.spans import ReasoningStepSpan
from tests.stage2.helpers import tiny_stage2_qwen


def _full_step_inputs():
    return {
        "input_ids": torch.tensor([[1, 5, 91, 90, 90, 92, 6, 7, 8]]),
        "attention_mask": torch.ones(1, 9, dtype=torch.long),
    }


def _step_span():
    return ReasoningStepSpan(
        step_id=0,
        prev_text_start=1,
        prev_text_end=2,
        latent_start=3,
        latent_end=5,
        target_text_start=6,
        target_text_end=8,
        helper_image_idx=0,
    )


class _TokenDecoder:
    eos_token_id = 127
    pad_token_id = 10
    all_special_ids = (91, 90, 92, 126, 127)

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(97 + token_id) for token_id in token_ids)


def _zero_logits_model():
    model = tiny_stage2_qwen(attn_implementation="eager").eval()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    return model


def test_greedy_generation_is_no_cache_deterministic_and_stops_at_maximum():
    """Catches sampling, cache use, or an off-by-one maximum token count."""
    model = _zero_logits_model()
    outer_calls = []
    handle = model.register_forward_pre_hook(
        lambda _module, _args, kwargs: outer_calls.append(kwargs), with_kwargs=True
    )
    try:
        first = greedy_generate_step_negative(
            model,
            full_model_inputs=_full_step_inputs(),
            step_span=_step_span(),
            modality="visual",
            tokenizer=_TokenDecoder(),
            max_new_tokens=3,
            attn_implementation="eager",
            eos_token_ids=(127,),
            control_token_ids=(126,),
        )
        second = greedy_generate_step_negative(
            model,
            full_model_inputs=_full_step_inputs(),
            step_span=_step_span(),
            modality="visual",
            tokenizer=_TokenDecoder(),
            max_new_tokens=3,
            attn_implementation="eager",
            eos_token_ids=(127,),
            control_token_ids=(126,),
        )
    finally:
        handle.remove()

    assert first == second
    assert first.token_ids == (0, 0, 0)
    assert first.text == "aaa"
    assert first.stop_reason == "max_new_tokens"
    assert len(outer_calls) == 6
    assert all(call["use_cache"] is False for call in outer_calls)
    assert all(call["stage2_selective_logits_only"] is True for call in outer_calls)


def test_greedy_generation_stops_before_eos_or_chat_control_token():
    """Catches EOS/control markers leaking into generated completion text or IDs."""
    model = _zero_logits_model()
    eos = greedy_generate_step_negative(
        model,
        full_model_inputs=_full_step_inputs(),
        step_span=_step_span(),
        modality="visual",
        tokenizer=_TokenDecoder(),
        max_new_tokens=4,
        attn_implementation="eager",
        eos_token_ids=(0,),
        control_token_ids=(126,),
    )
    control = greedy_generate_step_negative(
        model,
        full_model_inputs=_full_step_inputs(),
        step_span=_step_span(),
        modality="textual",
        tokenizer=_TokenDecoder(),
        max_new_tokens=4,
        attn_implementation="eager",
        eos_token_ids=(127,),
        control_token_ids=(0,),
    )

    assert (eos.text, eos.token_ids, eos.stop_reason) == ("", (), "eos")
    assert (control.text, control.token_ids, control.stop_reason) == (
        "",
        (),
        "control",
    )


@pytest.mark.parametrize(
    ("modality", "blocked_columns"),
    [("visual", (3, 4)), ("textual", (1,))],
)
def test_greedy_generation_builds_one_shifted_modality_mask_per_token(
    modality, blocked_columns
):
    """Catches mixing V/T sources, unshifted queries, or merged decoding masks."""
    model = _zero_logits_model()
    final_masks = []

    def record_final_mask(_module, _args, kwargs):
        mask = kwargs["attention_mask"]
        if mask.ndim == 4:
            final_masks.append(mask.detach().clone())

    handle = model.model.register_forward_pre_hook(record_final_mask, with_kwargs=True)
    try:
        greedy_generate_step_negative(
            model,
            full_model_inputs=_full_step_inputs(),
            step_span=_step_span(),
            modality=modality,
            tokenizer=_TokenDecoder(),
            max_new_tokens=2,
            attn_implementation="eager",
            eos_token_ids=(127,),
            control_token_ids=(126,),
        )
    finally:
        handle.remove()

    assert [tuple(mask.shape) for mask in final_masks] == [
        (1, 1, 7, 7),
        (1, 1, 8, 8),
    ]
    minimum = torch.finfo(model.dtype).min
    first_target_query_row = _step_span().target_text_start - 1
    for generation_index, mask in enumerate(final_masks):
        current_query_row = first_target_query_row + generation_index
        for query_row in range(first_target_query_row, current_query_row + 1):
            assert all(
                mask[0, 0, query_row, column] == minimum
                for column in blocked_columns
            )
        other_source = 1 if modality == "visual" else 3
        assert mask[0, 0, current_query_row, other_source] == 0


def test_dummy_terminal_token_cannot_change_the_selected_preceding_logits():
    """Catches selecting the dummy row instead of its causal predecessor."""
    torch.manual_seed(47)
    model = tiny_stage2_qwen(attn_implementation="eager").eval()
    seen_logits = []

    def capture_selected(_module, _args, output):
        seen_logits.append(output.stage2_selected_logits.detach().clone())

    handle = model.register_forward_hook(capture_selected)
    try:
        for dummy_token_id in (10, 11):
            greedy_generate_step_negative(
                model,
                full_model_inputs=_full_step_inputs(),
                step_span=_step_span(),
                modality="visual",
                tokenizer=_TokenDecoder(),
                max_new_tokens=1,
                attn_implementation="eager",
                eos_token_ids=(127,),
                control_token_ids=(126,),
                dummy_token_id=dummy_token_id,
            )
    finally:
        handle.remove()

    assert len(seen_logits) == 2
    torch.testing.assert_close(seen_logits[0], seen_logits[1], rtol=0, atol=0)


class _BoundaryTokenizer:
    is_fast = True
    all_special_ids = (90, 91)
    eos_token_id = 91
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    @staticmethod
    def _encoding(text):
        # The first completion character merges with the final prefix character:
        # ``bc`` is one token. The terminal control has its own nonzero offset.
        assert text == "<chat>abcd<|im_end|>"
        return (
            torch.tensor([[11, 12, 13, 90]]),
            torch.tensor([[[0, 7], [7, 9], [9, 10], [10, 20]]]),
        )

    def __call__(self, text, **kwargs):
        value = text[0] if isinstance(text, list) else text
        self.calls.append((value, dict(kwargs)))
        ids, offsets = self._encoding(value)
        result = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        return result


class _BoundaryProcessor:
    def __init__(self):
        self.tokenizer = _BoundaryTokenizer()
        self.processor_texts = []
        self.template_calls = []

    def apply_chat_template(self, conversation, **kwargs):
        self.template_calls.append((deepcopy(conversation), dict(kwargs)))
        assistant = conversation[1]["content"][0]["text"]
        return f"<chat>{assistant}<|im_end|>"

    def __call__(self, *, text, images, **kwargs):
        self.processor_texts.append((list(text), images, dict(kwargs)))
        return self.tokenizer(text, **kwargs)


def test_completion_tokenization_renders_one_full_string_and_excludes_controls():
    """Catches prefix/completion token concatenation and scoring a terminal control."""
    processor = _BoundaryProcessor()
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "q"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "unused"}]},
    ]

    encoded = tokenize_step_completion(
        processor,
        conversation=conversation,
        images=[],
        assistant_prefix="ab",
        completion_text="cd",
    )

    assert processor.template_calls[0][1] == {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    assert processor.processor_texts[0][0] == ["<chat>abcd<|im_end|>"]
    assert processor.tokenizer.calls[-1][0] == "<chat>abcd<|im_end|>"
    assert encoded.completion_token_positions == (1, 2)
    assert encoded.completion_token_ids == (12, 13)
    assert 90 not in encoded.completion_token_ids


def test_reference_score_uses_clean_causal_rows_and_only_completion_targets():
    """Catches ablated reference context, unshifted labels, or terminator scoring."""
    torch.manual_seed(53)
    model = tiny_stage2_qwen(attn_implementation="eager").eval()
    input_ids = _full_step_inputs()["input_ids"]
    attention_mask = torch.ones_like(input_ids)
    clean_calls = []
    handle = model.register_forward_pre_hook(
        lambda _module, _args, kwargs: clean_calls.append(kwargs), with_kwargs=True
    )
    try:
        score = score_step_completion(
            model,
            model_inputs={"input_ids": input_ids, "attention_mask": attention_mask},
            completion_token_positions=(6, 7),
        )
    finally:
        handle.remove()

    with torch.no_grad():
        full_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    expected = torch.log_softmax(full_logits[:, (5, 6), :].float(), dim=-1)
    expected = expected.gather(-1, input_ids[:, (6, 7)].unsqueeze(-1)).mean()
    assert score == pytest.approx(float(expected), rel=1e-6, abs=1e-7)
    assert len(clean_calls) == 1
    assert clean_calls[0]["stage2_final_attention_mask"] is None
    assert torch.equal(clean_calls[0]["stage2_logit_positions"], torch.tensor([5, 6]))
    assert clean_calls[0]["use_cache"] is False


def test_reference_scoring_requires_a_frozen_eval_model():
    """Catches reference values being produced by a mutable training-mode snapshot."""
    model = tiny_stage2_qwen().train()

    with pytest.raises(ValueError, match="eval mode"):
        score_step_completion(
            model,
            model_inputs=_full_step_inputs(),
            completion_token_positions=(6,),
        )


def test_generator_parser_exposes_the_normative_names_and_defaults(tmp_path):
    """Catches a CLI that cannot be invoked by the approved Phase A command."""
    from scripts.generate_step_negatives import build_parser

    args = build_parser().parse_args(
        [
            "--data_path",
            str(tmp_path / "data.jsonl"),
            "--image_root",
            str(tmp_path / "images"),
            "--stage2a_checkpoint_path",
            str(tmp_path / "checkpoint"),
            "--output_path",
            str(tmp_path / "negatives.jsonl"),
            "--manifest_path",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert args.max_samples is None
    assert args.latent_size == 8
    assert args.min_tokens == 2
    assert args.max_new_tokens == 32
    assert args.attention_implementation == "sdpa"
    assert args.device == "cuda:0"


def test_full_snapshot_loader_uses_checkpoint_directly_and_hashes_only_consumed_files(
    tmp_path,
):
    """Catches treating the real full snapshot as PEFT or hashing trainer debris."""
    from scripts.generate_step_negatives import load_frozen_stage2a_snapshot
    from stage2.checkpointing import Stage2ArtifactInspection, Stage2ArtifactKind

    checkpoint = tmp_path / "stage2-full"
    checkpoint.mkdir()
    payloads = {
        "config.json": b"config",
        "generation_config.json": b"generation",
        "model.safetensors": b"weights",
        "preprocessor_config.json": b"processor",
        "tokenizer_config.json": b"tokenizer-config",
        "tokenizer.json": b"tokenizer",
        "chat_template.json": b"template",
        "training_args.bin": b"must-not-be-hashed",
        "trainer_state.json": b"must-not-be-hashed",
    }
    for name, raw in payloads.items():
        (checkpoint / name).write_bytes(raw)

    processor = SimpleNamespace(tokenizer=SimpleNamespace())

    class FrozenModel:
        training = True

        def eval(self):
            self.training = False
            return self

    model = FrozenModel()
    processor_loads = []
    model_loads = []
    snapshot = load_frozen_stage2a_snapshot(
        checkpoint_path=checkpoint,
        attention_implementation="eager",
        device="cpu",
        inspect_artifact=lambda _path: Stage2ArtifactInspection(
            Stage2ArtifactKind.FULL
        ),
        load_processor=lambda source: processor_loads.append(source) or processor,
        load_full_model=lambda source, **kwargs: model_loads.append((source, kwargs))
        or model,
    )

    assert processor_loads == [str(checkpoint.resolve())]
    assert model_loads[0][0] == str(checkpoint.resolve())
    assert model_loads[0][1]["attn_implementation"] == "eager"
    assert model_loads[0][1]["device_map"] == "cpu"
    assert snapshot.model is model
    assert snapshot.processor is processor
    assert model.training is False
    assert snapshot.resolved_device == "cpu"
    assert snapshot.torch_dtype == "float32"
    expected_names = (
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    assert tuple(entry.relative_path for entry in snapshot.checkpoint_files) == expected_names
    assert snapshot.checkpoint_files == tuple(
        CheckpointFile(
            relative_path=name,
            size=len(payloads[name]),
            sha256=hashlib.sha256(payloads[name]).hexdigest(),
            load_root=str(checkpoint.resolve()),
        )
        for name in expected_names
    )
    assert snapshot.tokenizer_config_identity == hashlib.sha256(
        json.dumps(
            [
                {
                    "relative_path": entry.relative_path,
                    "sha256": entry.sha256,
                    "size": entry.size,
                }
                for entry in snapshot.checkpoint_files
                if entry.relative_path
                in {
                    "chat_template.json",
                    "preprocessor_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                }
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_peft_snapshot_loader_keeps_source_selectors_out_of_model_options(
    monkeypatch, tmp_path
):
    """Catches violating the validated PEFT loader's provenance/options split."""
    import scripts.generate_step_negatives as generator
    from stage2.checkpointing import Stage2ArtifactInspection, Stage2ArtifactKind

    checkpoint = tmp_path / "stage2-peft"
    checkpoint.mkdir()
    for name in (
        "stage2_peft_metadata.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
    ):
        (checkpoint / name).write_bytes(name.encode("utf-8"))
    inspection = Stage2ArtifactInspection(
        Stage2ArtifactKind.PEFT,
        peft_metadata_snapshot=SimpleNamespace(),
    )
    processor = SimpleNamespace(tokenizer=SimpleNamespace())
    base = tmp_path / "stage2-base"
    base.mkdir()
    (base / "config.json").write_bytes(b"base-config")
    (base / "model.safetensors").write_bytes(b"base-weights")
    underlying = _zero_logits_model()
    loaded_wrapper = object()
    calls = []
    monkeypatch.setattr(
        generator,
        "decode_stage2_peft_metadata_snapshot",
        lambda _snapshot: SimpleNamespace(base_model_reference=str(base)),
    )
    monkeypatch.setattr(
        generator,
        "resolve_stage2_qwen_model",
        lambda loaded: underlying if loaded is loaded_wrapper else None,
    )

    snapshot = generator.load_frozen_stage2a_snapshot(
        checkpoint_path=checkpoint,
        attention_implementation="sdpa",
        device="cpu",
        inspect_artifact=lambda path: (
            inspection
            if Path(path).resolve() == checkpoint.resolve()
            else Stage2ArtifactInspection(Stage2ArtifactKind.FULL)
        ),
        load_processor=lambda _source: processor,
        load_peft_model=lambda *args, **kwargs: calls.append((args, kwargs))
        or loaded_wrapper,
    )

    assert snapshot.model is underlying
    assert calls[0][0] == (str(base.resolve()), str(checkpoint.resolve()), processor)
    assert calls[0][1]["source_provenance_kwargs"] == {"local_files_only": True}
    assert "local_files_only" not in calls[0][1]["model_load_kwargs"]
    assert calls[0][1]["artifact_inspection"] is inspection
    assert tuple(
        (entry.load_root, entry.relative_path) for entry in snapshot.checkpoint_files
    ) == (
        (str(checkpoint.resolve()), "adapter_config.json"),
        (str(checkpoint.resolve()), "adapter_model.safetensors"),
        (str(checkpoint.resolve()), "stage2_peft_metadata.json"),
        (str(checkpoint.resolve()), "tokenizer_config.json"),
        (str(base.resolve()), "config.json"),
        (str(base.resolve()), "model.safetensors"),
    )


@pytest.mark.parametrize("base_reference", ["Qwen/Qwen2.5-VL-7B-Instruct", "missing"])
def test_peft_snapshot_rejects_nonlocal_base_before_any_load(
    monkeypatch, tmp_path, base_reference
):
    """Catches publishing unverifiable PEFT provenance for a remote/missing base."""
    import scripts.generate_step_negatives as generator
    from stage2.checkpointing import Stage2ArtifactInspection, Stage2ArtifactKind

    checkpoint = tmp_path / "stage2-peft"
    checkpoint.mkdir()
    for name in (
        "stage2_peft_metadata.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
    ):
        (checkpoint / name).write_bytes(name.encode("utf-8"))
    inspection = Stage2ArtifactInspection(
        Stage2ArtifactKind.PEFT,
        peft_metadata_snapshot=SimpleNamespace(),
    )
    monkeypatch.setattr(
        generator,
        "decode_stage2_peft_metadata_snapshot",
        lambda _snapshot: SimpleNamespace(base_model_reference=base_reference),
    )
    processor_loads = []
    model_loads = []

    with pytest.raises(ValueError, match="local regular directory"):
        generator.load_frozen_stage2a_snapshot(
            checkpoint_path=checkpoint,
            attention_implementation="sdpa",
            device="cpu",
            inspect_artifact=lambda _path: inspection,
            load_processor=lambda source: processor_loads.append(source),
            load_peft_model=lambda *args, **kwargs: model_loads.append(
                (args, kwargs)
            ),
        )

    assert processor_loads == []
    assert model_loads == []


def test_cli_publishes_and_validates_once_before_zero_pair_exit(monkeypatch, tmp_path):
    """Catches skipping validation or suppressing the zero-survivor failure signal."""
    import scripts.generate_step_negatives as generator

    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"sample_id":"sample-a"}\n', encoding="utf-8")
    image_root = tmp_path / "images"
    image_root.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint_file = checkpoint / "config.json"
    checkpoint_file.write_bytes(b"config")
    checkpoint_record = CheckpointFile(
        "config.json",
        6,
        hashlib.sha256(b"config").hexdigest(),
        str(checkpoint.resolve()),
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    negative_path = output_dir / "negatives.jsonl"
    manifest_path = output_dir / "manifest.json"
    snapshot = SimpleNamespace(
        model=object(),
        processor=object(),
        checkpoint_files=(checkpoint_record,),
        tokenizer_config_identity="tokenizer-v1",
        resolved_device="cpu",
        torch_dtype="float32",
    )
    row = StepNegative(
        sample_id="sample-a",
        step_id=0,
        chosen_text="chosen",
        rejected_visual_text="",
        rejected_textual_text="",
        valid_visual_negative=False,
        valid_textual_negative=False,
        reference_chosen_score=-1.0,
        reference_visual_rejected_score=None,
        reference_textual_rejected_score=None,
    )
    monkeypatch.setattr(generator, "_selected_examples", lambda *_args: (object(),))
    monkeypatch.setattr(
        generator, "load_frozen_stage2a_snapshot", lambda **_kwargs: snapshot
    )
    monkeypatch.setattr(
        generator, "generate_negative_rows", lambda **_kwargs: (row,)
    )
    real_load = generator.load_negative_store
    validations = []

    def validate_once(**kwargs):
        validations.append(kwargs)
        return real_load(**kwargs)

    monkeypatch.setattr(generator, "load_negative_store", validate_once)

    status = generator.main(
        [
            "--data_path",
            str(data_path),
            "--image_root",
            str(image_root),
            "--stage2a_checkpoint_path",
            str(checkpoint),
            "--output_path",
            str(negative_path),
            "--manifest_path",
            str(manifest_path),
            "--device",
            "cpu",
        ]
    )

    assert status == 2
    assert len(validations) == 1
    assert validations[0]["expected_generation_settings"] == {
        "attention_implementation": "sdpa",
        "device": "cpu",
        "max_new_tokens": 32,
        "min_tokens": 2,
        "torch_dtype": "float32",
    }
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "manifest.json",
        "negatives.jsonl",
    ]
    assert json.loads(negative_path.read_text(encoding="utf-8"))["sample_id"] == "sample-a"
