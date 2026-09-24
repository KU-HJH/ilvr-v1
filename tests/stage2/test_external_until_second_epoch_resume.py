"""Contracts for resuming the completed external-until GB8 run for epoch two."""

from types import SimpleNamespace

import pytest
import torch

from tests.stage2.test_launcher import dry_run_tokens, option_value, run_launcher


def resume_tokens():
    return dry_run_tokens(run_launcher('--max_steps', '846', '--save_steps', '846', '--no-save_final_model', env_overrides={
        'STAGE2_EXECUTION_PROFILE': 'latest_external_until',
        'STAGE2_EXTERNAL_GLOBAL_BATCH_SIZE': '8',
        'RESUME_CHECKPOINT': '/checkpoints/run/checkpoint-423',
        'STEP_PREFERENCE_ENABLED': 'true',
        'STEP_PREFERENCE_MODE': 'stage2b',
        'STEP_PREFERENCE_TARGET_SCOPE': 'until_next_latent',
        'STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES': 'false',
        'STEP_SHARED_PREFIX_ENABLED': 'true',
        'STEP_PREFERENCE_BETA': '0.5',
        'STEP_PREFERENCE_WEIGHT': '1',
        'STEP_ENTROPY_WEIGHT': '0',
        'STAGE2_TOTAL_STEPS': '846',
        'GRADIENT_ACCUMULATION_STEPS': '2',
    }))


def test_launcher_forwards_second_epoch_resume_without_fresh_initialization():
    tokens = resume_tokens()
    assert option_value(tokens, '--resume_from_checkpoint_path') == '/checkpoints/run/checkpoint-423'
    assert '--stage1_adapter_path' not in tokens
    assert '--stage2a_checkpoint_path' not in tokens
    assert '--stage2b_direct_from_stage1' not in tokens
    assert option_value(tokens, '--stage2_total_steps') == '846'
    assert option_value(tokens, '--max_steps') == '846'
    assert '--no-save_final_model' in tokens


def test_profile_resume_keeps_saved_objective_and_total_budget():
    from utils_deepseed import parse_args
    args = parse_args(resume_tokens()[resume_tokens().index('--stage'):])
    config = args.step_preference_config
    assert args.resume_from_checkpoint_path == '/checkpoints/run/checkpoint-423'
    assert (args.max_steps, args.stage2_total_steps, args.gradient_accumulation_steps) == (846, 846, 2)
    assert config.phase_steps == 846
    assert config.target_scope == 'until_next_latent'
    assert config.visual_only_context_preference and not config.visual_off_suppress_latent_boundaries
    assert config.beta == .5 and config.lambda_preference == 1. and config.lambda_entropy == 0.
    assert not config.direct_from_stage1 and config.stage2a_checkpoint_path is None
    assert args.save_final_model is False


def test_resume_rejects_fresh_stage1_adapter_combination():
    from utils_deepseed import parse_args
    with pytest.raises(ValueError, match='mutually exclusive'):
        parse_args(resume_tokens()[resume_tokens().index('--stage'):] +
                   ['--stage1_adapter_path', '/checkpoints/stage1/checkpoint-431'])


def test_stage2b_resume_preflight_routes_to_checkpoint_and_skips_stage2a(monkeypatch, tmp_path):
    import main
    from utils_deepseed import parse_args
    checkpoint = str(tmp_path / 'checkpoint-423')
    args = parse_args(['--stage', 'stage2', '--step_preference_enabled',
        '--step_preference_mode', 'stage2b', '--step_visual_only_context_preference',
        '--step_preference_target_scope', 'until_next_latent',
        '--resume_from_checkpoint_path', checkpoint,
        '--stage2_total_steps', '846', '--max_steps', '846',
        '--vision_attention_implementation', 'flash_attention_2'])
    source = tmp_path / 'data.jsonl'
    source.write_text('{"sample_id":"one"}\n')
    args.data_path = str(source)
    args.image_root = str(tmp_path)
    monkeypatch.setattr(main, 'load_jsonl_examples', lambda path: [object()])
    monkeypatch.setattr(main, 'prepare_stage2_example', lambda *a, **k: SimpleNamespace(sample_id='one', reasoning_steps=()))
    calls = []
    decision = SimpleNamespace(processor_source=checkpoint, trainer_resume_path=checkpoint)
    monkeypatch.setattr(main, 'resolve_stage2_load_decision', lambda **kw: calls.append(kw) or decision)
    monkeypatch.setattr(main, '_default_stage2_source_dependencies', lambda: main.Stage2SourceDependencies(
        find_adapter_config=lambda *a, **k: None, load_config=lambda *a, **k: None,
        load_processor=lambda source: SimpleNamespace(tokenizer=object()), load_model=lambda *a, **k: None))
    preflight = main.preflight_stage2b_artifacts(args)
    assert len(preflight.prepared_examples) == 1
    assert calls[0]['resume_from_checkpoint_path'] == checkpoint
    assert calls[0].get('stage1_adapter_path') is None
    assert decision.trainer_resume_path == checkpoint


def test_resume_training_arguments_preserve_old_checkpoint_and_root_adapter():
    from main import build_stage2_training_arguments
    from utils_deepseed import parse_args
    args = parse_args(resume_tokens()[resume_tokens().index('--stage'):])
    training = build_stage2_training_arguments(args)
    assert training.max_steps == 846
    assert training.save_steps == 846
    assert training.save_total_limit == 2
    assert not args.save_final_model


def test_transformers_resume_scheduler_state_uses_extended_total_without_new_warmup():
    from transformers import get_linear_schedule_with_warmup
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    old = get_linear_schedule_with_warmup(optimizer, 10, 423)
    for _ in range(423):
        optimizer.step(); old.step()
    saved_optimizer = optimizer.state_dict()
    saved_scheduler = old.state_dict()
    assert optimizer.param_groups[0]['lr'] == 0.
    resumed_optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    resumed_scheduler = get_linear_schedule_with_warmup(resumed_optimizer, 10, 846)
    resumed_optimizer.load_state_dict(saved_optimizer)
    resumed_scheduler.load_state_dict(saved_scheduler)
    assert resumed_scheduler.last_epoch == 423
    assert resumed_optimizer.param_groups[0]['lr'] == 0.
    resumed_optimizer.step(); resumed_scheduler.step()
    assert resumed_scheduler.last_epoch == 424
    assert resumed_optimizer.param_groups[0]['lr'] == pytest.approx(2.5239234449760766e-5)


def test_completed_423_resume_runs_one_more_epoch_to_846(tmp_path):
    from tests.stage2.test_reference_sampler_resume_runtime import run_resume
    trainer, backwards = run_resume(tmp_path, 2, 423, 846, policy='external_until_balanced')
    assert trainer.state.global_step == 846
    assert trainer.state.epoch == 2.
    assert len(backwards) == 845
    assert backwards[0]['index'] == 0 and backwards[0]['step'] == 423
    assert backwards[-1]['index'] == 844 and backwards[-1]['step'] == 845


def test_checkpoint_rotation_retains_first_epoch_and_root_adapter(tmp_path):
    from tests.stage2.test_reference_random_sampler import cpu_trainer
    root = tmp_path / 'run'
    trainer = cpu_trainer(root, 2, policy='external_until_balanced', max_steps=846)
    trainer.args.save_total_limit = 2
    first = root / 'checkpoint-423'
    second = root / 'checkpoint-846'
    first.mkdir(parents=True)
    second.mkdir()
    (first / 'adapter_model.safetensors').write_bytes(b'first epoch')
    (root / 'adapter_model.safetensors').write_bytes(b'root first epoch')
    trainer._rotate_checkpoints(output_dir=str(root))
    assert (first / 'adapter_model.safetensors').read_bytes() == b'first epoch'
    assert second.is_dir()
    assert (root / 'adapter_model.safetensors').read_bytes() == b'root first epoch'


def test_training_components_select_resume_decision_before_stage2a(monkeypatch, tmp_path):
    import main
    from utils_deepseed import parse_args
    checkpoint = str(tmp_path / 'checkpoint-423')
    args = parse_args(['--stage', 'stage2', '--step_preference_enabled',
        '--step_preference_mode', 'stage2b', '--step_visual_only_context_preference',
        '--step_preference_target_scope', 'until_next_latent',
        '--resume_from_checkpoint_path', checkpoint,
        '--vision_attention_implementation', 'flash_attention_2'])
    args._stage2b_preflight = main.Stage2BPreflight((), None)
    dependencies = main.Stage2SourceDependencies(
        find_adapter_config=lambda *a, **k: None,
        load_config=lambda *a, **k: None,
        load_processor=lambda *a, **k: None,
        load_model=lambda *a, **k: None,
    )
    class SelectedResume(Exception):
        pass
    def stop_at_decision(**kwargs):
        assert kwargs['resume_from_checkpoint_path'] == checkpoint
        raise SelectedResume
    monkeypatch.setattr(main, 'resolve_stage2_load_decision', stop_at_decision)
    monkeypatch.setattr(main, 'resolve_stage2b_initialization_decision',
                        lambda **kwargs: pytest.fail('Stage 2A initialization selected'))
    with pytest.raises(SelectedResume):
        main.build_stage2_training_components(args, source_dependencies=dependencies)


def test_stage2b_resume_rejects_objective_without_resume_preflight_support():
    from step_preference.config import StepPreferenceConfig
    with pytest.raises(ValueError, match='V-only context preference'):
        StepPreferenceConfig(enabled=True, mode='stage2b', objective='legacy_dpo',
            negative_path='negative.jsonl', negative_manifest='negative.manifest.json',
            resume_from_checkpoint_path='checkpoint-423', total_steps=846)


def test_existing_stage2a_resume_keeps_phase_budget():
    from step_preference.config import StepPreferenceConfig
    config = StepPreferenceConfig(enabled=True, mode='stage2a',
        resume_from_checkpoint_path='checkpoint-5', total_steps=10, stage2a_ratio=.5)
    assert config.phase_steps == 5
