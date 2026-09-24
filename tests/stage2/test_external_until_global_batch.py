"""Global-batch selector and original external stream contracts."""

from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler

from stage2.sampler import (ExternalUntilBalancedSampler,
                            read_stage2_sampler_sidecar,
                            validate_stage2_sampler_resume_identity,
                            write_stage2_sampler_sidecar)


def test_launcher_parser_resolves_eight_and_preserves_sixteen_defaults():
    from tests.stage2.test_launcher import dry_run_tokens, option_value, run_launcher
    from utils_deepseed import parse_args

    eight = dry_run_tokens(run_launcher(env_overrides={
        'STAGE2_EXECUTION_PROFILE': 'latest_external_until',
        'STAGE2_EXTERNAL_GLOBAL_BATCH_SIZE': '8',
        'GRADIENT_ACCUMULATION_STEPS': '99',
    }))
    assert option_value(eight, '--stage2_external_global_batch_size') == '8'
    resolved = parse_args(eight[eight.index('--stage'):])
    assert (resolved.gradient_accumulation_steps, resolved.max_steps,
            resolved.stage2_total_steps) == (2, 423, 423)
    assert resolved.step_preference_config.beta == .5
    assert resolved.step_preference_config.lambda_preference == 1.

    sixteen = dry_run_tokens(run_launcher(env_overrides={
        'STAGE2_EXECUTION_PROFILE': 'latest_external_until',
    }))
    default = parse_args(sixteen[sixteen.index('--stage'):])
    assert (default.gradient_accumulation_steps, default.max_steps,
            default.stage2_total_steps) == (4, 212, 212)
    assert parse_args(['--stage', 'stage2', '--stage2_execution_profile',
                       'latest_external_until']).stage2_external_global_batch_size == 16


def test_external_stream_and_identity_keep_grouping_two_across_accumulations(tmp_path):
    data = [SimpleNamespace(latent_block_count=1)] * 3379
    old = ExternalUntilBalancedSampler(data, num_replicas=4, seed=42,
                                       gradient_accumulation_steps=4)
    new = ExternalUntilBalancedSampler(data, num_replicas=4, seed=42,
                                       gradient_accumulation_steps=2)
    assert list(new) == list(old)
    shards = [list(BatchSamplerShard(BatchSampler(new, 1, False),
              num_processes=4, process_index=rank, even_batches=False))
              for rank in range(4)]
    stream = [batch[0] for row in zip(*shards) for batch in row]
    assert len(stream) == 3380
    assert [len(shard) for shard in shards] == [845] * 4
    assert len(set(stream)) == 3379
    assert len([stream[i:i+8] for i in range(0, 3380, 8)]) == 423
    assert len(stream[-4:]) == 4
    assert old.identity.to_dict() == {
        'schema_version': 3, 'seed': 42, 'world_size': 4,
        'dataset_size': 3379, 'bucket_sizes': {'1': 3379},
        'padded_samples': 1, 'policy': 'external_until_balanced',
        'gradient_accumulation_steps': 4, 'update_accounting': 'ceil',
        'padding_weight': 1, 'reference_grouping': 2,
    }
    for name, sampler in (('old', old), ('new', new)):
        destination = tmp_path / name
        write_stage2_sampler_sidecar(destination, sampler.identity, global_step=1)
        assert read_stage2_sampler_sidecar(destination) == sampler.identity
        validate_stage2_sampler_resume_identity(destination, sampler.identity)
        with pytest.raises(ValueError, match='gradient_accumulation_steps'):
            validate_stage2_sampler_resume_identity(destination,
                new.identity if sampler is old else old.identity)


def test_external_cpu_trainer_last_four_retain_global_eight_denominator(tmp_path, monkeypatch):
    from deepspeed_runtime import DeepSpeedRuntimeProfile
    from stage2.trainer import Stage2Trainer
    from step_preference.config import StepPreferenceConfig
    from tests.stage2.test_reference_random_sampler import ScalarModel, cpu_trainer
    from transformers import TrainerCallback, TrainingArguments

    class BeforeStep(TrainerCallback):
        def __init__(self):
            self.gradients = []

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            self.gradients.append(float(kwargs['model'].weight.grad))

    monkeypatch.setattr(TrainingArguments, 'world_size', property(lambda self: 4))
    production_args = TrainingArguments(output_dir=str(tmp_path / 'constructor'),
        use_cpu=True, report_to=[], per_device_train_batch_size=1,
        gradient_accumulation_steps=2, disable_tqdm=True)
    production = Stage2Trainer(model=ScalarModel(), args=production_args,
        train_dataset=[{'input_ids': torch.tensor([1])}] * 845,
        stage2_sampler_policy='external_until_balanced',
        stage2_execution_profile='latest_external_until',
        step_preference_config=StepPreferenceConfig(),
        deepspeed_runtime_profile=DeepSpeedRuntimeProfile(True, 2, None, None, 4))
    assert production.model_accepts_loss_kwargs is False
    assert isinstance(production.train_dataset[0], dict)
    monkeypatch.undo()

    trainer = cpu_trainer(tmp_path / 'loop', 2, policy='external_until_balanced')
    values = trainer.set_initial_training_values(trainer.args, list(range(845)), 8)
    assert values[1] == values[6] == 423
    # The CPU Trainer has one process, fed one rank's 845 microbatches. Match
    # the verified four-rank constructor flag while leaving the loop real.
    trainer.model_accepts_loss_kwargs = production.model_accepts_loss_kwargs
    trainer.args.max_grad_norm = 0.
    trainer.args.learning_rate = 0.
    witness = BeforeStep()
    trainer.add_callback(witness)
    result = trainer.train()
    assert result.global_step == 423
    assert len(witness.gradients) == 423
    assert witness.gradients[-2] == pytest.approx(2.)
    # Final rank-local one-item accumulation keeps GAS2 divisor, yielding
    # half the full rank-local update; four-rank mean has global8 denominator.
    assert witness.gradients[-1] == pytest.approx(1.)


def test_multinode_environment_carries_explicit_external_batch():
    from src.multinode import build_launch_plan
    from src.multinode_runtime import build_owned_tmux_commands
    from src.multinode_runtime import validated_stage2_environment

    environment = validated_stage2_environment({
        'STAGE1_CHECKPOINT': '/checkpoint', 'DATA_PATH': '/data',
        'IMAGE_ROOT': '/images', 'STAGE2_EXECUTION_PROFILE': 'latest_external_until',
        'STAGE2_EXTERNAL_GLOBAL_BATCH_SIZE': '8',
    }, output_dir='/output', log_file='/log', deepspeed_config='/zero2')
    assert environment['STAGE2_EXTERNAL_GLOBAL_BATCH_SIZE'] == '8'
    plan = build_launch_plan(stage='stage2', tuning_method='dora', zero_stage=2,
        run_id='external-global-eight-test', main_process_port=29631)
    commands = build_owned_tmux_commands(plan, commit='a' * 40,
        stage_environment=environment)
    assert [dict(spec.environment)['STAGE2_EXTERNAL_GLOBAL_BATCH_SIZE']
            for spec in commands.node_specs] == ['8', '8']


def test_eight_selector_requires_latest_external_profile():
    from utils_deepseed import parse_args

    with pytest.raises(ValueError, match='latest_external_until'):
        parse_args(['--stage', 'stage2', '--stage2_external_global_batch_size', '8'])
