"""Opt-in full-text preference targets, with the original path kept intact."""

from dataclasses import replace

import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

from stage2.data import Stage2DataCollator, prepare_stage2_example
from step_preference.config import StepPreferenceConfig
from step_preference.sidecar import build_context_sidecar_plan
from step_preference.spans import ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF
from tests.step_preference.test_spans import STAGE2A_CHECKPOINT
from tests.stage2.test_launcher import run_launcher, dry_run_tokens, option_value
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
from utils_deepseed import parse_args


@pytest.mark.parametrize("scope", ["immediate_text", "until_next_latent"])
def test_cli_target_scope_reaches_config_independently_of_boundary_mode(scope):
    """Catches losing the option in CLI-to-config conversion or coupling boundaries."""
    for boundary_args in ([], ["--step_visual_off_suppress_latent_boundaries"]):
        args = parse_args([
            "--stage", "stage2", "--step_preference_enabled",
            "--step_preference_mode", "stage2b",
            "--stage2a_checkpoint_path", "checkpoint",
            "--step_visual_only_context_preference",
            "--step_preference_target_scope", scope, *boundary_args,
        ])
        assert args.step_preference_config.target_scope == scope
        assert args.step_preference_config.visual_off_suppress_latent_boundaries == bool(boundary_args)
    assert parse_args(["--stage", "stage2"]).step_preference_config.target_scope == "immediate_text"


def test_target_scope_rejects_unknown_or_non_context_configuration():
    """Catches silent fallback and changing dormant/legacy objectives."""
    with pytest.raises(ValueError, match="target.scope"):
        StepPreferenceConfig(target_scope="typo")
    with pytest.raises(ValueError, match="context.preference Stage 2B"):
        StepPreferenceConfig(target_scope="until_next_latent")


def test_launcher_forwards_opt_in_scope_and_rejects_invalid_values():
    """Catches the standard training launcher silently ignoring an opt-in."""
    environment = {
        "STEP_PREFERENCE_ENABLED": "true", "STEP_PREFERENCE_MODE": "stage2b",
        "STAGE2A_CHECKPOINT_PATH": "/models/stage2a",
    }
    for scope in ("immediate_text", "until_next_latent"):
        tokens = dry_run_tokens(run_launcher(env_overrides={
            **environment, "STEP_PREFERENCE_TARGET_SCOPE": scope,
        }))
        assert option_value(tokens, "--step_preference_target_scope") == scope
    invalid = run_launcher(env_overrides={**environment, "STEP_PREFERENCE_TARGET_SCOPE": "typo"})
    assert invalid.returncode == 2
    wrong_mode = run_launcher(env_overrides={"STEP_PREFERENCE_TARGET_SCOPE": "until_next_latent"})
    assert wrong_mode.returncode == 2


def test_launcher_rejects_target_scope_override_outside_managed_environment():
    """Catches duplicate CLI arguments bypassing the launcher's validated run mode."""
    result = run_launcher("--step_preference_target_scope", "until_next_latent")
    assert result.returncode == 2
    assert "--step_preference_target_scope" in result.stderr


def test_opt_in_real_collator_expands_targets_without_changing_input_or_ce(tmp_path):
    """Catches excluded final answers, next-block leakage, and input/CE drift."""
    Image.new("RGB", (56, 56)).save(tmp_path / "input.png")
    example = prepare_stage2_example(
        {"image_input": "input.png", "text_input": "question", "sequence_plan": [
            {"type": "text", "content": "Before."},
            {"type": "latent", "helper_image": "first.png"},
            {"type": "text", "content": "Reason one."},
            {"type": "text", "content": "More one."},
            {"type": "latent", "helper_image": "second.png"},
            {"type": "text", "content": "Reason two."},
            {"type": "text", "content": "The final answer is: 70"},
        ]},
        image_root=tmp_path, latent_size=2, require_reasoning_units=True,
        assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    )
    processor = AutoProcessor.from_pretrained(
        STAGE2A_CHECKPOINT, local_files_only=True
    )
    tokenizer = processor.tokenizer
    tokens = LatentTokenIds(**{
        f"{name}_token_id": tokenizer.convert_tokens_to_ids(f"<|latent_{name}|>")
        for name in ("pad", "start", "end")
    })
    kwargs = dict(processor=processor, token_ids=tokens, latent_size=2,
                  supervise_latent_boundaries=True, step_preference_enabled=True)
    default = Stage2DataCollator(**kwargs)([example])
    immediate = Stage2DataCollator(**kwargs, preference_target_scope="immediate_text")([example])
    full = Stage2DataCollator(**kwargs, preference_target_scope="until_next_latent")([example])
    for key in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"):
        assert torch.equal(default[key], immediate[key]), key
        assert torch.equal(default[key], full[key]), key
    assert default["reasoning_step_spans"] == immediate["reasoning_step_spans"]
    for old, new in zip(default["reasoning_step_spans"], full["reasoning_step_spans"]):
        assert replace(new, target_text_end=old.target_text_end) == old
    for batch, expected in (
        (default, ["Reason one.", "Reason two."]),
        (full, ["Reason one. \nMore one.", "Reason two. \nThe final answer is: 70"]),
    ):
        for suppress in (False, True):
            plan = build_context_sidecar_plan(
                clean_input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                spans=batch["reasoning_step_spans"], visual_only=True,
                suppress_latent_boundaries=suppress,
            )
            assert [tokenizer.decode(b.target_token_ids) for b in plan.groups[0]] == expected
            clean_targets = batch["input_ids"][0, [p + 1 for p in plan.root_selected_positions]]
            assert clean_targets.tolist() == [t for b in plan.groups[0] for t in b.target_token_ids]
            assert all(
                token not in b.target_token_ids
                for b in plan.groups[0]
                for token in (tokens.pad_token_id, tokens.start_token_id, tokens.end_token_id,
                              tokenizer.convert_tokens_to_ids("<|im_end|>"))
            )


def test_collator_rejects_full_scope_with_disabled_preference_or_legacy_negatives():
    """Catches silently ignored options or inconsistent legacy chosen-text joins."""
    for kwargs in ({}, {"step_preference_enabled": True, "negative_store": {}}):
        with pytest.raises(ValueError, match="target.scope|context.preference"):
            Stage2DataCollator(None, None, 2, True,
                preference_target_scope="until_next_latent", **kwargs)


@pytest.mark.parametrize("scope", ["immediate_text", "until_next_latent"])
def test_training_assembly_passes_target_scope_to_real_collator(monkeypatch, scope):
    """Catches main dropping the configured scope during component assembly."""
    import main
    from stage2.checkpointing import Stage2LoadDecision
    from stage2.peft import Stage2TuningConfig
    from tests.stage2.helpers import tiny_stage2_qwen
    from tests.stage2.test_routing import CompositionProcessor, stage2_args

    model, processor = tiny_stage2_qwen(), CompositionProcessor()
    config = StepPreferenceConfig(enabled=True, mode="stage2b",
        stage2a_checkpoint_path="checkpoint", target_scope=scope)
    decision = Stage2LoadDecision(
        base_model_source="checkpoint", processor_source="checkpoint",
        adapter_source=None, trainer_resume_path=None,
        tuning=Stage2TuningConfig(method="full", r=4, alpha=8, dropout=0.0,
                                 target_modules=("q_proj",)),
        latent_size=2, supervise_latent_boundaries=True,
    )
    monkeypatch.setattr(main, "resolve_stage2b_initialization_decision", lambda **kwargs: decision)
    # This is a CPU assembly test, not a FlashAttention backend or artifact audit.
    monkeypatch.setattr(main, "_require_stage2b_attention_contract", lambda args: None)
    monkeypatch.setattr(main, "build_prepared_example_length_measure", lambda p: None)
    monkeypatch.setattr(main, "filter_stage2_examples", lambda **kwargs: ([], None))
    components = main.build_stage2_training_components(
        stage2_args(step_preference_config=config,
                    _stage2b_preflight=main.Stage2BPreflight((), None)),
        source_dependencies=main.Stage2SourceDependencies(
            find_adapter_config=lambda *args, **kwargs: None,
            load_config=lambda *args, **kwargs: model.config,
            load_processor=lambda *args, **kwargs: processor,
            load_model=lambda *args, **kwargs: model,
        ),
    )
    assert isinstance(components.data_collator, Stage2DataCollator)
    assert components.data_collator.preference_target_scope == scope
    assert components.data_collator.step_preference_enabled is True
