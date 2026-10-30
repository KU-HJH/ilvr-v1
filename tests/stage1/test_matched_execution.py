"""Native standard-loss control under the recorded V5 execution profile."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Trainer

from stage1.composition import _load_stage1_model
from stage1.trainer import CustomTrainerStage1
from utils_deepseed import parse_args


def test_worker_accepts_explicit_matched_standard_execution():
    """Catches losing the opt-in before the standard composition entrypoint."""
    args = parse_args(['--stage', 'stage1', '--stage1_match_v5_execution',
                       '--ce_weight', '1', '--sim_weight', '1'])
    assert args.stage1_variant == 'standard'
    assert args.stage1_match_v5_execution is True
    assert args.ce_weight == args.sim_weight == 1


@pytest.mark.parametrize('extra', [['--stage', 'stage2'], ['--stage1_variant', 'v5']])
def test_matched_execution_cannot_select_a_different_objective_path(extra):
    with pytest.raises(ValueError, match='standard Stage1'):
        parse_args(['--stage1_match_v5_execution', *extra])


def test_matched_model_constructs_real_sdpa_language_and_flash_vision():
    """Catches a vision flag accepted by argparse but ignored by the loader."""
    config = Qwen2_5_VLConfig(vocab_size=128, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        vision_config=dict(depth=1, hidden_size=8, intermediate_size=16, num_heads=2,
            patch_size=2, spatial_merge_size=1, temporal_patch_size=1,
            window_size=2, out_hidden_size=8, fullatt_block_indexes=[0]),
        stage='stage1', latent_size=2, latent_token_id=126,
        rope_scaling={'type':'mrope','mrope_section':[1,1,0]})
    def construct(source, config, **kwargs):
        opts = {'torch_dtype': kwargs['torch_dtype']}
        if 'attn_implementation' in kwargs:
            opts['attn_implementation'] = kwargs['attn_implementation']
        with torch.device('meta'):
            return Qwen2_5_VLForConditionalGeneration._from_config(config, **opts)
    args = SimpleNamespace(stage1_variant='standard', stage1_match_v5_execution=True,
        attention_implementation='sdpa', vision_attention_implementation='flash_attention_2',
        cache_dir=None)
    model = _load_stage1_model('config-only', config, args,
                              SimpleNamespace(load_model=construct))
    assert type(model.model.layers[0].self_attn).__name__ == 'Qwen2_5_VLSdpaAttention'
    assert type(model.visual.blocks[0].attn).__name__ == 'Qwen2_5_VLVisionFlashAttention2'


def test_matched_schedule_accounts_for_the_three_microbatch_final_window():
    """Catches a flag that fails to activate actual prepared-loader budgeting."""
    from stage1.matched_execution import enable_matched_standard_execution
    class Loader:
        def __len__(self): return 863
    owner = SimpleNamespace(num_examples=lambda loader: 3450)
    enable_matched_standard_execution(owner)
    args = SimpleNamespace(max_steps=2160, gradient_accumulation_steps=4,
                           num_train_epochs=10)
    values = Trainer.set_initial_training_values(owner, args, Loader(), 16)
    assert values[0] == 10
    assert values[1] == 216


@pytest.mark.parametrize('field,value', [
    ('ce_weight', .1), ('sim_weight', .1), ('gradient_accumulation_steps', 2),
    ('per_device_train_batch_size', 2), ('gradient_checkpointing', False),
    ('attention_implementation', 'flash_attention_2'),
    ('vision_attention_implementation', 'sdpa'), ('ema_tau', -0.1),
])
def test_matched_profile_rejects_drift_before_model_loading(field, value):
    from stage1.matched_execution import validate_matched_standard_execution
    args = SimpleNamespace(ce_weight=1., sim_weight=1.,
        gradient_accumulation_steps=4, per_device_train_batch_size=1,
        gradient_checkpointing=True, attention_implementation='sdpa',
        vision_attention_implementation='flash_attention_2', ema_tau=1.)
    setattr(args, field, value)
    with pytest.raises(ValueError, match='matched standard'):
        validate_matched_standard_execution(args,
            SimpleNamespace(method='dora'),
            SimpleNamespace(enabled=True, zero_stage=1, world_size=4))


def test_actual_standard_parent_backward_has_context_and_only_native_loss(tmp_path):
    """Catches missing checkpoint/dropout ownership or accidental V5 branches."""
    from stage1.equivalence import stage1_checkpoint_context_fn
    from tests.stage1.v5.test_trainer import integrated_fixture
    from tests.stage1.v5.test_replay import digest_state
    trainer, batch, calls, _, _, _ = integrated_fixture(tmp_path)
    trainer.__class__ = CustomTrainerStage1
    trainer.stage1_match_v5_execution = True
    trainer.ce_weight = trainer.sim_weight = 1.
    trainer.collect_stage1_diagnostics = True
    trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
        'use_reentrant':False, 'context_fn':stage1_checkpoint_context_fn})
    trainer.accelerator.backward = Accelerator.backward.__get__(trainer.accelerator)
    batch.pop('stage1_v5_metadata')
    teacher_before = digest_state(trainer.ema_teacher.model)
    loss = trainer.training_step(trainer.model, batch)
    diag = trainer._latest_stage1_diagnostics
    torch.testing.assert_close(loss, (diag['ce_loss'] + diag['alignment_loss']) / 4)
    assert calls == ['helper', 'F']
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in trainer.model.parameters() if p.requires_grad)
    assert digest_state(trainer.ema_teacher.model) == teacher_before
    with pytest.raises(RuntimeError, match='missing Stage 1 stochastic context'):
        stage1_checkpoint_context_fn()


def test_matched_standard_uses_verified_native_deepspeed_lifecycle(
        native_deepspeed_lifecycle, monkeypatch):
    """Catches retaining the isinstance(V5) gate for the explicit standard mode."""
    trainer, engine, wrapper = native_deepspeed_lifecycle
    trainer.__class__ = CustomTrainerStage1
    del trainer.stage1_v5_debug  # Production standard has no inherited V5 state.
    from stage1.matched_execution import enable_matched_standard_execution
    enable_matched_standard_execution(trainer)
    monkeypatch.setattr(trainer, '_validate_deepspeed_gradient_norm',
        lambda engine: pytest.fail('reference profile disables the extra norm reduction'))
    trainer._install_deepspeed_step_adapter()
    wrapper.backward(engine.module(torch.ones(1, 1)).sum(),
                     sync_gradients=True, scale_wrt_gas=False)
    assert trainer._successful_optimizer_steps == 1
    assert engine.was_step_applied() is True


@pytest.mark.parametrize('matched', [False, True])
@pytest.mark.parametrize('policy', ['production', 'legacy'])
def test_production_constructor_scopes_target_checks_to_execution_profile(monkeypatch, matched, policy):
    """Construct the native trainer without inherited V5 fixture attributes."""
    from trl import SFTTrainer
    from deepspeed_runtime import DeepSpeedRuntimeProfile
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {'input_ids': torch.tensor([[1]])}
    def parent_init(self, **kwargs):
        self.model = torch.nn.Linear(1, 1)
        self.processing_class = Tokenizer()
    monkeypatch.setattr(SFTTrainer, '__init__', parent_init)
    monkeypatch.setattr(CustomTrainerStage1, 'add_callback', lambda *args: None)
    trainer = CustomTrainerStage1(
        deepspeed_runtime_profile=DeepSpeedRuntimeProfile(True, 1, None, None, 4),
        ema_teacher=object(), match_v5_execution=matched, stage1_check_policy=policy)
    expected = policy == 'legacy' and not matched
    assert trainer._stage1_checks_enabled() is expected
    assert trainer.target_builder.validate is expected
    assert trainer.target_builder.collect_diagnostics is expected


from tests.stage1.v5.test_trainer import native_deepspeed_lifecycle


def test_matched_checkpoint_marker_rejects_resume_even_without_flag(tmp_path):
    """Catches a matched checkpoint silently resuming with legacy epoch semantics."""
    from stage1.matched_execution import (
        MATCHED_EXECUTION_FILENAME, reject_matched_execution_resume,
        write_matched_execution_identity,
    )
    write_matched_execution_identity(tmp_path)
    assert json.loads((tmp_path / MATCHED_EXECUTION_FILENAME).read_text())['schema_version'] == 1
    with pytest.raises(ValueError, match='fresh'):
        reject_matched_execution_resume(tmp_path, enabled=False)
    other = tmp_path / 'legacy'
    other.mkdir()
    reject_matched_execution_resume(other, enabled=False)
    with pytest.raises(ValueError, match='fresh'):
        reject_matched_execution_resume(other, enabled=True)


def test_matched_identity_is_collected_as_training_metadata_only():
    from tests.multinode.test_evaluation_coordinator import _manifest
    from src.multinode_artifacts import (
        ArtifactEntry, select_common_metadata_entries, select_inference_entries,
    )
    manifest = _manifest()
    entry = ArtifactEntry(Path('stage1_matched_execution.json'), 100, 0)
    manifest = replace(manifest, entries=manifest.entries + (entry,))
    assert entry in select_common_metadata_entries(manifest)
    assert entry not in select_inference_entries(manifest)


def test_coordinator_opt_in_reaches_both_actual_worker_parsers():
    """Catches coordinator allowlist or worker shell silently dropping the flag."""
    import os
    import shlex
    import subprocess
    from tests.stage1.v5.test_launcher import ROOT, REMOTE, HOST
    profile = ['--stage1-variant','standard','--stage1-match-v5-execution',
        '--ce-weight','1','--sim-weight','1','--micro-batch','1',
        '--gradient-accumulation','4','--epochs','10','--max-steps','2160',
        '--attention-implementation','sdpa','--vision-attention-implementation','flash_attention_2']
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'CUDA_VISIBLE_DEVICES':''}
    run = subprocess.run([str(ROOT/'run_multinode_stage1_training.sh'),
        '--run-id','matched-standard-command-test','--zero-stage','1',
        '--main-process-port','29683','--local-repo-root',str(ROOT),
        '--remote-repo-root',str(REMOTE),'--remote-host-repo-root',str(HOST),
        '--dry-run','--method','dora','--model','/original/base',
        '--data-path',str(ROOT/'data/TRAIN.jsonl'),
        '--image-root',str(ROOT/'data/images_comt'), *profile],
        cwd=ROOT, env=env, text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    commands = json.loads(run.stdout)
    remote = shlex.split(shlex.split(commands['remote'][-1])[-1])
    for owned in (remote, commands['local']):
        cursor = owned.index('-i') + 2
        worker_env = {}
        while '=' in owned[cursor] and not owned[cursor].startswith('--'):
            name, value = owned[cursor].split('=',1)
            worker_env[name] = value
            cursor += 1
        stage = owned[max(i for i,v in enumerate(owned) if v=='--')+1:]
        dry = subprocess.run(['bash',str(ROOT/'run_stage1_training.sh'), *stage[2:], '--dry-run'],
            cwd=ROOT, env={**env, **worker_env}, text=True, capture_output=True)
        assert dry.returncode == 0, dry.stderr
        argv = shlex.split(dry.stdout)
        parsed = parse_args(argv[argv.index('src/main.py')+1:])
        assert parsed.stage1_match_v5_execution is True
        assert parsed.stage1_variant == 'standard'
        assert parsed.ce_weight == parsed.sim_weight == 1
        assert parsed.gradient_accumulation_steps == 4 and parsed.epochs == 10


@pytest.mark.parametrize('decay', [0., .999, 1.])
def test_matched_standard_accepts_native_ema_decay(decay):
    """Catches rejecting EMA before matched standard model construction."""
    from stage1.matched_execution import validate_matched_standard_execution
    args = SimpleNamespace(ce_weight=1., sim_weight=1., ema_tau=decay,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        gradient_checkpointing=True, attention_implementation='sdpa',
        vision_attention_implementation='flash_attention_2')
    validate_matched_standard_execution(args, SimpleNamespace(method='dora'),
        SimpleNamespace(enabled=True, zero_stage=1, world_size=4))


@pytest.mark.parametrize('decay', [-.1, 1.01, float('nan'), float('inf')])
def test_matched_standard_rejects_invalid_ema_decay(decay):
    with pytest.raises(ValueError):
        test_matched_standard_accepts_native_ema_decay(decay)


def test_matched_native_deepspeed_updates_ema_only_after_applied_boundary(native_deepspeed_lifecycle):
    """An admitted EMA profile must actually update once, after the student."""
    from stage1.matched_execution import enable_matched_standard_execution
    trainer, engine, wrapper = native_deepspeed_lifecycle
    trainer.__class__ = CustomTrainerStage1
    enable_matched_standard_execution(trainer)
    trainer.ema_teacher.decay = .999
    trainer._install_deepspeed_step_adapter()
    for _ in range(3):
        wrapper.backward(engine.module(torch.ones(1, 1)).sum()/4,
                         sync_gradients=False, scale_wrt_gas=False)
        torch.testing.assert_close(trainer.ema_teacher.model.weight, torch.tensor([[1.]]), rtol=0, atol=0)
        assert trainer._successful_optimizer_steps == 0
    wrapper.backward(engine.module(torch.ones(1, 1)).sum()/4,
                     sync_gradients=True, scale_wrt_gas=False)
    torch.testing.assert_close(engine.module.weight, torch.tensor([[.9]]))
    torch.testing.assert_close(trainer.ema_teacher.model.weight, torch.tensor([[.9999]]), rtol=0, atol=1e-7)
    assert trainer._successful_optimizer_steps == 1
    engine.optimizer.overflow = True
    wrapper.backward(engine.module(torch.ones(1, 1)).sum(),
                     sync_gradients=True, scale_wrt_gas=False)
    torch.testing.assert_close(trainer.ema_teacher.model.weight, torch.tensor([[.9999]]), rtol=0, atol=1e-7)
    assert trainer._successful_optimizer_steps == 1
