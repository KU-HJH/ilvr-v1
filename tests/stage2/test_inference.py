import copy
import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from evaluation.continuous_latent import (
    ContinuousLatentGenerationConfig,
    ContinuousLatentGenerationResult,
    build_continuous_generation_config,
)
from .helpers import (
    ToyCausalLanguageModel,
    activate_stage2_adapter_deltas,
    tiny_stage2_qwen,
)
from peft import PeftModel
from stage2.checkpointing import (
    build_stage2_peft_metadata,
    save_stage2_peft_metadata,
)
from stage2.inference import (
    Stage2GenerationConfig,
    Stage2GenerationResult,
    Stage2InferenceController,
    build_stage2_generation_config,
)
from stage2.peft import (
    ALLOWED_STAGE2_TARGET_SUFFIXES,
    Stage2TuningConfig,
    build_stage2_tuned_model,
    load_stage2_model_for_inference,
    resolve_stage2_qwen_model,
    resolve_stage2_target_module_names,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentRolloutEngine,
    LatentTokenIds,
    parse_latent_spans,
)


TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)


def test_stage2_generation_exports_remain_neutral_contract_aliases():
    """Catches Stage 2 compatibility exports drifting from neutral generation."""
    assert Stage2GenerationConfig is ContinuousLatentGenerationConfig
    assert Stage2GenerationResult is ContinuousLatentGenerationResult
    assert build_stage2_generation_config is build_continuous_generation_config


def test_stage2_generation_result_retains_legacy_two_field_construction():
    """Catches required neutral evidence fields breaking Stage 2 result callers."""
    sequences = torch.tensor([[1, 2]])

    result = Stage2GenerationResult(sequences, ())

    assert result.sequences is sequences
    assert result.latent_states_by_block == ()


def test_stage2_controller_retains_its_original_generate_signature():
    """Catches neutral-only evidence arguments leaking into Stage 2's API."""
    assert tuple(inspect.signature(Stage2InferenceController.generate).parameters) == (
        "self",
        "model",
        "input_ids",
        "attention_mask",
        "token_ids",
        "generation_config",
        "pixel_values",
        "image_grid_thw",
        "return_latent_states",
    )


def settings(method: str) -> Stage2TuningConfig:
    return Stage2TuningConfig(
        method=method,
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
    )


class TinyLatentTokenizer:
    unk_token_id = 0
    unk_token = "<unk>"
    ids = {
        "<|latent_pad|>": 90,
        "<|latent_start|>": 91,
        "<|latent_end|>": 92,
    }

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, self.unk_token_id)


def inference_settings(method: str) -> Stage2TuningConfig:
    return Stage2TuningConfig(
        method=method,
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
    )


class ScriptedHead(nn.Module):
    def __init__(self, token_ids, vocab_size=128):
        super().__init__()
        self.token_ids = list(token_ids)
        self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, hidden):
        token_id = self.token_ids[min(self.calls, len(self.token_ids) - 1)]
        self.calls += 1
        logits = hidden.new_full((*hidden.shape[:-1], self.vocab_size), -1e9)
        logits[..., token_id] = 0
        return logits


class RankedHead(nn.Module):
    def __init__(self, *, highest_id, fallback_id, vocab_size=128):
        super().__init__()
        self.highest_id = highest_id
        self.fallback_id = fallback_id
        self.vocab_size = vocab_size

    def forward(self, hidden):
        logits = hidden.new_full((*hidden.shape[:-1], self.vocab_size), -torch.inf)
        logits[..., self.highest_id] = 100
        logits[..., self.fallback_id] = 0
        return logits


class ScriptedStage2Model(nn.Module):
    def __init__(self, *, emitted_ids, latent_size):
        super().__init__()
        self.config = SimpleNamespace(latent_size=latent_size)
        self.model = ToyCausalLanguageModel(4)
        self.lm_head = ScriptedHead(emitted_ids)
        self.stage2_rollout_engine = LatentRolloutEngine()
        self.visual_calls = 0
        self.rope_inputs = []

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def prepare_stage2_base_inputs_embeds(
        self, *, input_ids, pixel_values, image_grid_thw
    ):
        self.visual_calls += int(pixel_values is not None)
        return self.get_input_embeddings()(input_ids)

    def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
        self.rope_inputs.append(
            (
                input_ids.clone(),
                None if image_grid_thw is None else image_grid_thw.clone(),
                None if attention_mask is None else attention_mask.clone(),
            )
        )
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = positions.view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        return positions, torch.zeros(input_ids.shape[0], 1, device=input_ids.device)


def scripted_model(*, latent_size, emitted_ids):
    return ScriptedStage2Model(emitted_ids=emitted_ids, latent_size=latent_size)


def test_inference_forces_exactly_k_latents_and_supports_multiple_blocks():
    model = scripted_model(
        latent_size=2,
        emitted_ids=[4, TOKENS.start_token_id, 5, TOKENS.start_token_id, 3],
    )
    result = Stage2InferenceController().generate(
        model=model,
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=16, eos_token_id=3),
        pixel_values=torch.ones(1, 3),
        image_grid_thw=torch.tensor([[1, 1, 1]]),
        return_latent_states=True,
    )

    spans = parse_latent_spans(result.sequences[0], token_ids=TOKENS, latent_size=2)
    assert [len(span.latent_positions) for span in spans] == [2, 2]
    assert len(result.latent_states_by_block) == 2
    assert all(len(block) == 2 for block in result.latent_states_by_block)
    assert model.lm_head.calls == 5
    assert model.visual_calls == 1


def test_started_block_finishes_past_nominal_budget_without_vocabulary_sampling():
    model = scripted_model(latent_size=2, emitted_ids=[TOKENS.start_token_id])

    result = Stage2InferenceController().generate(
        model=model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=1),
    )

    assert result.sequences[0, -4:].tolist() == [91, 90, 90, 92]
    assert model.lm_head.calls == 1


def test_eos_is_checked_only_after_a_completed_latent_block():
    model = scripted_model(latent_size=2, emitted_ids=[TOKENS.start_token_id, 3])

    result = Stage2InferenceController().generate(
        model=model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=5, eos_token_id=3),
    )

    assert result.sequences[0, -5:].tolist() == [91, 90, 90, 92, 3]
    assert model.lm_head.calls == 2


@pytest.mark.parametrize(
    ("forbidden_id", "temperature", "top_p"),
    [
        (TOKENS.pad_token_id, 0.0, 1.0),
        (TOKENS.end_token_id, 0.0, 1.0),
        (TOKENS.pad_token_id, 1.0, 0.5),
        (TOKENS.end_token_id, 1.0, 0.5),
    ],
)
def test_normal_text_generation_masks_block_internal_latent_tokens(
    forbidden_id, temperature, top_p
):
    model = scripted_model(latent_size=2, emitted_ids=[])
    model.lm_head = RankedHead(highest_id=forbidden_id, fallback_id=4)

    result = Stage2InferenceController().generate(
        model=model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(
            max_new_tokens=1,
            temperature=temperature,
            top_p=top_p,
        ),
    )

    assert result.sequences[0, -1].item() == 4


def test_nucleus_sampling_can_enter_latent_mode_through_latent_start():
    model = scripted_model(latent_size=2, emitted_ids=[])
    model.lm_head = RankedHead(highest_id=TOKENS.start_token_id, fallback_id=4)

    result = Stage2InferenceController().generate(
        model=model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(
            max_new_tokens=1,
            temperature=1.0,
            top_p=0.5,
        ),
    )

    assert result.sequences[0, -4:].tolist() == [91, 90, 90, 92]


@pytest.mark.parametrize("emitted_eos", [3, 7])
def test_inference_stops_on_either_member_of_multiple_eos_ids(emitted_eos):
    """Comparing an emitted ID to an EOS list as a scalar must fail this test."""
    model = scripted_model(latent_size=2, emitted_ids=[emitted_eos, 4])

    result = Stage2InferenceController().generate(
        model=model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(
            max_new_tokens=2,
            eos_token_id=[3, 7],
        ),
    )

    assert result.sequences[0, -1].item() == emitted_eos
    assert model.lm_head.calls == 1


def test_latent_block_rejects_hard_sequence_cap_before_rollout():
    model = scripted_model(latent_size=2, emitted_ids=[TOKENS.start_token_id])

    with pytest.raises(ValueError, match="latent block would exceed max_sequence_length"):
        Stage2InferenceController().generate(
            model=model,
            input_ids=torch.tensor([[1]]),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            token_ids=TOKENS,
            generation_config=Stage2GenerationConfig(
                max_new_tokens=1,
                max_sequence_length=4,
            ),
        )

    assert len(model.model.calls) == 1


def test_ordinary_text_rejects_hard_sequence_cap():
    model = scripted_model(latent_size=2, emitted_ids=[4])

    with pytest.raises(ValueError, match="max_sequence_length"):
        Stage2InferenceController().generate(
            model=model,
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            token_ids=TOKENS,
            generation_config=Stage2GenerationConfig(
                max_new_tokens=1,
                max_sequence_length=2,
            ),
        )


def test_inference_rejects_batch_greater_than_one():
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        Stage2InferenceController().generate(
            model=scripted_model(latent_size=2, emitted_ids=[4]),
            input_ids=torch.ones(2, 2, dtype=torch.long),
            attention_mask=torch.ones(2, 2, dtype=torch.long),
            token_ids=TOKENS,
            generation_config=Stage2GenerationConfig(),
        )


def test_generation_config_defaults_to_official_evaluation_output_limit():
    config = Stage2GenerationConfig()

    assert config.max_new_tokens == 4096


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": -1},
        {"max_sequence_length": 0},
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": float("-inf")},
        {"top_p": 0.0},
        {"top_p": 1.1},
    ],
)
def test_generation_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        Stage2GenerationConfig(**kwargs)


@pytest.mark.parametrize(
    "eos_token_id",
    [[], [3, "7"], [3, True], [-1], "3", True, -1],
)
def test_generation_config_rejects_malformed_eos_ids(eos_token_id):
    with pytest.raises(ValueError, match="eos_token_id"):
        Stage2GenerationConfig(eos_token_id=eos_token_id)


@pytest.mark.parametrize("max_sequence_length", [True, 1.5, "128"])
def test_generation_config_rejects_non_integer_sequence_limits(max_sequence_length):
    with pytest.raises(ValueError, match="max_sequence_length"):
        Stage2GenerationConfig(max_sequence_length=max_sequence_length)


def test_inference_rejects_non_positive_latent_size():
    with pytest.raises(ValueError, match="latent_size must be positive"):
        Stage2InferenceController().generate(
            model=scripted_model(latent_size=0, emitted_ids=[4]),
            input_ids=torch.tensor([[1]]),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            token_ids=TOKENS,
            generation_config=Stage2GenerationConfig(max_new_tokens=1),
        )


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_saved_stage2_adapter_reload_preserves_logits_and_latent_states(
    method, tmp_path
):
    """Skipping adapter reload or loading zero deltas must break parity."""
    tuning = inference_settings(method)
    artifact_dir = tmp_path / method
    source_base = tiny_stage2_qwen()
    source_base.config.supervise_latent_boundaries = True
    fresh_base = copy.deepcopy(source_base)
    resolved_targets = resolve_stage2_target_module_names(
        source_base, tuning.target_modules
    )
    source = build_stage2_tuned_model(source_base, tuning).eval()
    changed = activate_stage2_adapter_deltas(source)
    assert changed
    source_parameters = dict(source.named_parameters())
    assert all(torch.count_nonzero(source_parameters[name]) for name in changed)
    source.save_pretrained(artifact_dir)
    metadata = build_stage2_peft_metadata(
        tuning=tuning,
        base_model_reference="tiny-stage1",
        model_config=resolve_stage2_qwen_model(source).config,
        token_ids=TOKENS,
        resolved_target_names=resolved_targets,
    )
    save_stage2_peft_metadata(artifact_dir, metadata)

    # Compatibility must see the truly unmodified Stage 1 boundary.
    fresh_base.config.stage = "stage1"
    expected_architectures = list(fresh_base.config.architectures or [])
    base_dir = tmp_path / f"base-{method}"
    fresh_base.save_pretrained(base_dir)
    config_path = base_dir / "config.json"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload["architectures"] = expected_architectures
    config_path.write_text(
        json.dumps(config_payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    reloaded = load_stage2_model_for_inference(
        base_dir,
        artifact_dir,
        SimpleNamespace(tokenizer=TinyLatentTokenizer()),
    )

    assert isinstance(reloaded, PeftModel)
    assert reloaded.training is False
    assert resolve_stage2_qwen_model(reloaded).config.stage == "stage2"
    assert not any(parameter.requires_grad for parameter in reloaded.parameters())

    scaffold = torch.tensor([[1, 91, 90, 90, 92, 2]])
    mask = torch.ones_like(scaffold)
    with torch.inference_mode():
        source_output = source(
            input_ids=scaffold,
            attention_mask=mask,
            return_stage2_latent_states=True,
        )
        reload_output = reloaded(
            input_ids=scaffold,
            attention_mask=mask,
            return_stage2_latent_states=True,
        )

    torch.testing.assert_close(source_output.logits, reload_output.logits)
    assert len(source_output.stage2_latent_states) == len(
        reload_output.stage2_latent_states
    ) == 2
    for source_state, reload_state in zip(
        source_output.stage2_latent_states,
        reload_output.stage2_latent_states,
        strict=True,
    ):
        torch.testing.assert_close(source_state, reload_state)


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_peft_training_and_inference_share_latent_states_and_full_mrope_positions(
    method, monkeypatch
):
    """Catches inference using an unadapted decoder instead of the PEFT Qwen base."""
    training_wrapper = build_stage2_tuned_model(
        tiny_stage2_qwen(), settings(method)
    ).eval()
    activate_stage2_adapter_deltas(training_wrapper)
    training_model = resolve_stage2_qwen_model(training_wrapper).eval()
    training_model.lm_head = ScriptedHead([TOKENS.start_token_id])
    inference_model = copy.deepcopy(training_model).eval()
    context = torch.tensor([[1, 81, 80, 80, 80, 80, 2]])
    scaffold = torch.tensor(
        [[1, 81, 80, 80, 80, 80, 2, 91, 90, 90, 92]]
    )
    mask = torch.ones_like(scaffold)
    image_grid_thw = torch.tensor([[1, 2, 2]])
    pixel_values = torch.ones((4, 12))
    full_positions, _ = training_model.get_rope_index(
        scaffold,
        image_grid_thw=image_grid_thw,
        attention_mask=mask,
    )
    training_rope_inputs = []
    training_decoder_inputs = []
    inference_rope_inputs = []
    original_training_rope_index = training_model.get_rope_index
    original_inference_rope_index = inference_model.get_rope_index

    def record_training_rope_index(
        input_ids, image_grid_thw=None, attention_mask=None
    ):
        training_rope_inputs.append(
            (input_ids.clone(), image_grid_thw.clone(), attention_mask.clone())
        )
        return original_training_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

    def record_inference_rope_index(
        input_ids, image_grid_thw=None, attention_mask=None
    ):
        inference_rope_inputs.append(
            (input_ids.clone(), image_grid_thw.clone(), attention_mask.clone())
        )
        return original_inference_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

    def record_training_decoder_inputs(_module, _args, kwargs):
        training_decoder_inputs.append(
            (kwargs["position_ids"].clone(), kwargs["attention_mask"].clone())
        )

    monkeypatch.setattr(
        training_model, "get_rope_index", record_training_rope_index
    )
    monkeypatch.setattr(
        inference_model, "get_rope_index", record_inference_rope_index
    )
    handle = training_model.model.register_forward_pre_hook(
        record_training_decoder_inputs, with_kwargs=True
    )
    training = training_wrapper(
        input_ids=scaffold,
        attention_mask=mask,
        image_grid_thw=image_grid_thw,
        pixel_values=pixel_values,
        return_stage2_latent_states=True,
    )
    handle.remove()

    inference = Stage2InferenceController().generate(
        model=inference_model,
        input_ids=context,
        attention_mask=torch.ones_like(context),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=1),
        image_grid_thw=image_grid_thw,
        pixel_values=pixel_values,
        return_latent_states=True,
    )

    assert len(training.stage2_latent_states) == len(
        inference.latent_states_by_block[0]
    ) == 2
    for train_state, inference_state in zip(
        training.stage2_latent_states,
        inference.latent_states_by_block[0],
        strict=True,
    ):
        torch.testing.assert_close(train_state, inference_state)
    assert len(training_rope_inputs) == 1
    assert torch.equal(training_rope_inputs[0][0], scaffold)
    assert torch.equal(training_rope_inputs[0][1], image_grid_thw)
    assert torch.equal(training_rope_inputs[0][2], mask)
    assert len(training_decoder_inputs) == 3
    for (position_ids, attention_mask), prefix_length in zip(
        training_decoder_inputs, (8, 9, 11), strict=True
    ):
        assert torch.equal(position_ids, full_positions[..., :prefix_length])
        assert torch.equal(attention_mask, mask[:, :prefix_length])
    assert len(inference_rope_inputs) == 2
    assert torch.equal(inference_rope_inputs[-1][0], scaffold)
    assert torch.equal(inference_rope_inputs[-1][1], image_grid_thw)
    assert torch.equal(inference_rope_inputs[-1][2], mask)


def test_latent_states_are_returned_only_when_requested():
    model = scripted_model(latent_size=2, emitted_ids=[TOKENS.start_token_id])

    result = Stage2InferenceController().generate(
        model=model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=1),
    )

    assert result.latent_states_by_block == ()


def test_returned_inference_latent_states_do_not_retain_gradients():
    model = scripted_model(latent_size=2, emitted_ids=[TOKENS.start_token_id])

    result = Stage2InferenceController().generate(
        model=model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        token_ids=TOKENS,
        generation_config=Stage2GenerationConfig(max_new_tokens=1),
        return_latent_states=True,
    )

    assert all(
        state.requires_grad is False
        for block in result.latent_states_by_block
        for state in block
    )
