"""Reference global order, fixed-denominator padding, and real Trainer resume."""
import copy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard, SeedableRandomSampler
from torch.utils.data import BatchSampler, DataLoader
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerState

from stage2 import sampler as sampling
from stage2.trainer import Stage2Trainer
from step_preference.config import StepPreferenceConfig


def reference_sampler(dataset, **kwargs):
    assert hasattr(sampling, 'ReferenceRandomSampler'), 'explicit reference_random sampler is missing'
    return sampling.ReferenceRandomSampler(dataset, **kwargs)


def context_config(**kwargs):
    return StepPreferenceConfig(enabled=True, mode='stage2b', stage2a_checkpoint_path='tiny',
        sidecar_attention_backend='dense', beta=.5, lambda_preference=1., **kwargs)


@pytest.mark.parametrize('epoch', [0, 1])
@pytest.mark.parametrize('gas', [2, 4])
def test_actual_accelerate_shards_preserve_reference_order_and_update_membership(epoch, gas):
    # Catches latent bucketing, duplicate real exposure, and padded extra updates.
    data = list(range(3379))
    source = SeedableRandomSampler(data, data_seed=917)
    source.set_epoch(epoch)
    expected = list(source)
    sampler = reference_sampler(data, num_replicas=4, seed=917, gradient_accumulation_steps=gas)
    sampler.set_epoch(epoch)
    states = torch.get_rng_state().clone()
    shards = [list(BatchSamplerShard(BatchSampler(sampler, 1, False), num_processes=4,
        process_index=rank, even_batches=False)) for rank in range(4)]
    assert torch.equal(torch.get_rng_state(), states)
    assert [len(shard) for shard in shards] == [845] * 4
    actual = [batch[0] for microstep in zip(*shards) for batch in microstep]
    assert actual == expected + [-1]
    assert len(set(i for i in actual if i >= 0)) == 3379
    for start in range(0, 3380, 4 * gas):
        assert [i for i in actual[start:start+4*gas] if i >= 0] == expected[start:start+4*gas]


def test_reference_identity_roundtrip_rejects_policy_or_gas_changing_resume(tmp_path):
    # Catches silently reinterpreting historical floor/bucket checkpoint semantics.
    from tests.stage2.test_sampler import examples_with_counts
    data = examples_with_counts([1, 2, 1])
    legacy = sampling.LatentBlockBalancedSampler(data, num_replicas=4, seed=42).identity
    fresh = reference_sampler(data, num_replicas=4, seed=42, gradient_accumulation_steps=2).identity
    sampling.write_stage2_sampler_sidecar(tmp_path, legacy, global_step=1)
    with pytest.raises(ValueError, match='policy'):
        sampling.validate_stage2_sampler_resume_identity(tmp_path, fresh)
    dest = tmp_path / 'new'
    sampling.write_stage2_sampler_sidecar(dest, fresh, global_step=1)
    assert sampling.read_stage2_sampler_sidecar(dest) == fresh
    sampling.validate_stage2_sampler_resume_identity(dest, fresh)
    changed = reference_sampler(data, num_replicas=4, seed=42, gradient_accumulation_steps=4).identity
    with pytest.raises(ValueError, match='gradient_accumulation'):
        sampling.validate_stage2_sampler_resume_identity(dest, changed)
    assert legacy.to_dict()['schema_version'] == 1
    assert 'policy' not in legacy.to_dict()


def test_dataset_collator_padding_is_valid_example_with_explicit_zero_weight(tmp_path):
    # Catches treating Python index -1 as a real final item or leaking model metadata.
    from PIL import Image
    from stage2.data import Stage2DataCollator, prepare_stage2_example
    from tests.stage2.test_data import RecordingProcessor, TOKENS
    assert hasattr(sampling, 'ReferenceRandomDataset'), 'padding-aware dataset is missing'
    Image.new('RGB', (2, 2)).save(tmp_path/'image.png')
    example = prepare_stage2_example(dict(sample_id='real', image_input='image.png',
        text_input='question', sequence_plan=[dict(type='latent', helper_image='unused.png'),
        dict(type='text', content='answer')]), image_root=tmp_path, latent_size=2)
    dataset = sampling.ReferenceRandomDataset([example])
    collator = Stage2DataCollator(processor=RecordingProcessor(), token_ids=TOKENS,
        latent_size=2, supervise_latent_boundaries=True)
    real, padding = collator([dataset[0]]), collator([dataset[-1]])
    assert real['stage2_sample_weight'] == 1.
    assert padding['stage2_sample_weight'] == 0.
    assert padding['stage2_sample_index'] == -1
    assert padding['stage2_sample_latent_block_count'] == 1
    assert torch.equal(real['input_ids'], padding['input_ids'])
    assert (padding['labels'] != -100).any()  # valid CE forward, never empty labels/NaN


@pytest.mark.parametrize('gas', [2, 4])
def test_composed_ce_preference_padding_gradient_is_zero_at_fixed_global_denominator(gas):
    # Catches zeroing CE alone, renormalizing a 3-real tail, and metric contamination.
    from step_preference.equivalence import _trainer
    from step_preference.shared_prefix_measurement import tiny_measurement_case
    model, batch = tiny_measurement_case(steps=2, total_target_tokens=2, device='cpu')
    batch['step_preference']['steps'] = tuple({'span': span} for span in batch['reasoning_step_spans'])
    trainer = _trainer(context_config())
    trainer.stage2_sampler_policy = 'reference_random'
    parameters = [p for p in model.parameters() if p.requires_grad]
    real_loss = trainer.compute_loss(model, copy.deepcopy(batch))
    real_grad = torch.autograd.grad(real_loss, parameters, allow_unused=True)
    trainer._step_preference_metric_totals = {}
    trainer._step_preference_metric_batches = 0
    trainer._runtime_objective_observation_counts = {}
    padding_batch = dict(copy.deepcopy(batch), stage2_sample_weight=0., stage2_sample_index=-1,
        stage2_sample_latent_block_count=2)
    torch.manual_seed(917)
    before = torch.get_rng_state().clone()
    padding_loss, output = trainer.compute_loss(model, padding_batch, return_outputs=True)
    padding_grad = torch.autograd.grad(padding_loss, parameters, allow_unused=True)
    assert float(padding_loss) == 0.
    assert float(output.loss) == 0.
    assert torch.equal(before, torch.get_rng_state())
    assert trainer._step_preference_metric_batches == 0
    assert trainer._runtime_objective_observation_counts == {}
    for real, padded in zip(real_grad, padding_grad):
        if real is not None:
            assert padded is not None and torch.count_nonzero(padded) == 0
            # The independent mathematical denominator is 4 ranks * configured GAS.
            actual = (real + real + real + padded) / (4 * gas)
            assert torch.equal(actual, real * 3 / (8 if gas == 2 else 16))


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(loss=self.weight.square())


def cpu_trainer(tmp_path, gas, policy='reference_random', max_steps=-1):
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[],
        per_device_train_batch_size=1, gradient_accumulation_steps=gas,
        num_train_epochs=1, max_steps=max_steps, disable_tqdm=True)
    trainer = Stage2Trainer(model=ScalarModel(), args=args,
        train_dataset=[{'input_ids': torch.tensor([1])}] * 845)
    # Exercise the real Trainer control loop on CPU; distributed policy guards are
    # separately covered, because initializing four DeepSpeed ranks is the GPU gate.
    trainer.stage2_sampler_policy = policy
    return trainer


def actual_reference_trainer(tmp_path, model, dataset, config, monkeypatch):
    """Construct the real Trainer under its four-rank runtime contract."""
    from deepspeed_runtime import DeepSpeedRuntimeProfile
    monkeypatch.setattr(TrainingArguments, 'world_size', property(lambda self: 4))
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[],
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_train_epochs=1, disable_tqdm=True)
    return Stage2Trainer(model=model, args=args, train_dataset=dataset,
        stage2_sampler_policy='reference_random', step_preference_config=config,
        deepspeed_runtime_profile=DeepSpeedRuntimeProfile(True, 2, None, None, 4))


def test_ce_only_reference_trainer_preserves_tail_update_and_fixed_denominator(tmp_path, monkeypatch):
    # Catches rejecting CE-only, changing global order, or renormalizing the partial tail.
    trainer = actual_reference_trainer(tmp_path, ScalarModel(),
        [{'input_ids': torch.tensor([1])}] * 19, StepPreferenceConfig(), monkeypatch)
    assert trainer.step_preference_config.enabled is False
    assert trainer.model_accepts_loss_kwargs is False
    sampler = trainer._get_train_sampler()
    shards = [list(BatchSamplerShard(BatchSampler(sampler, 1, False), num_processes=4,
        process_index=rank, even_batches=False)) for rank in range(4)]
    global_order = [batch[0] for row in zip(*shards) for batch in row]
    assert [len(shard) for shard in shards] == [5] * 4
    update_groups = (global_order[:16], global_order[16:])
    assert [len(group) for group in update_groups] == [16, 4]
    assert sorted(index for index in global_order if index >= 0) == list(range(19))
    assert [trainer.train_dataset[index].weight
        for index in update_groups[-1]] == [1., 1., 1., 0.]
    rank_zero_loader = DataLoader(trainer.train_dataset,
        batch_sampler=BatchSamplerShard(BatchSampler(sampler, 1, False), num_processes=4,
            process_index=0, even_batches=False))
    values = trainer.set_initial_training_values(trainer.args, rank_zero_loader, 16)
    assert values[1] == 2
    assert values[6] == 2

    rank_gradients = []
    for weight in (1., 1., 1., 0.):
        trainer.model.zero_grad(set_to_none=True)
        loss = trainer.training_step(trainer.model, {
            'input_ids': torch.tensor([[1]]), 'stage2_sample_weight': weight,
            'stage2_sample_index': 0 if weight else -1,
            'stage2_sample_latent_block_count': 1,
        })
        assert float(loss) == pytest.approx(.25 if weight else 0.)
        rank_gradients.append(trainer.model.weight.grad.detach().clone())
    torch.testing.assert_close(
        torch.stack(rank_gradients).mean(), torch.tensor(3 * 2 / 16))


def test_ce_only_reference_trainer_matches_voff_clean_ce_without_sidecars(tmp_path, monkeypatch):
    # Catches a hidden preference/entropy route or sidecar work changing ordinary clean CE.
    from step_preference.shared_prefix_measurement import RuntimeCounters, tiny_measurement_case
    model, batch = tiny_measurement_case(steps=1, total_target_tokens=2, device='cpu')
    batch['step_preference']['steps'] = tuple(
        {'span': span} for span in batch['reasoning_step_spans'])
    voff_model = copy.deepcopy(model)
    ce = actual_reference_trainer(tmp_path/'ce', model, [batch] * 19,
        StepPreferenceConfig(), monkeypatch)
    voff_config = replace(context_config(), lambda_preference=0., lambda_entropy=0.,
        shared_prefix_enabled=True, visual_off_suppress_latent_boundaries=False)
    voff = actual_reference_trainer(tmp_path/'voff', voff_model, [batch] * 19,
        voff_config, monkeypatch)

    ce_batch = {name: copy.deepcopy(batch[name])
        for name in ('input_ids', 'attention_mask', 'labels')}
    with RuntimeCounters(model) as ce_counters:
        ce_loss, ce_outputs = ce.compute_loss(model, ce_batch, return_outputs=True)
        ce_counters.phase = 'backward'
        ce_loss.backward()
    with RuntimeCounters(voff_model) as voff_counters:
        voff_loss = voff.compute_loss(voff_model, copy.deepcopy(batch))
        voff_counters.phase = 'backward'
        voff_loss.backward()

    torch.testing.assert_close(ce_loss, voff_loss, atol=1e-6, rtol=1e-6)
    for (ce_name, ce_parameter), (voff_name, voff_parameter) in zip(
            model.named_parameters(), voff_model.named_parameters()):
        assert ce_name == voff_name
        if ce_parameter.grad is not None or voff_parameter.grad is not None:
            assert ce_parameter.grad is not None and voff_parameter.grad is not None
            torch.testing.assert_close(ce_parameter.grad, voff_parameter.grad, atol=1e-6, rtol=1e-6)
    assert ce_outputs.stage2_sidecar_results is None
    assert ce_counters.counts.get('sidecar_chunks_forward', 0) == 0
    assert ce._step_preference_metric_batches == 0
    assert ce._runtime_objective_observation_counts == {}
    assert voff_counters.counts['sidecar_chunks_forward'] > 0


@pytest.mark.parametrize('gas,updates', [(2, 423), (4, 212)])
def test_real_trainer_epoch_budget_and_linear_schedule_include_tail(tmp_path, gas, updates):
    # Catches schedule max_steps retaining the inherited floor after sampler repair.
    trainer = cpu_trainer(tmp_path, gas)
    loader = DataLoader(trainer.train_dataset, batch_size=1)
    values = trainer.set_initial_training_values(trainer.args, loader, 4 * gas)
    assert values[1] == updates
    assert values[6] == updates
    trainer.create_optimizer_and_scheduler(num_training_steps=values[6])
    assert trainer.lr_scheduler.lr_lambdas[0](updates - 1) == pytest.approx(1 / updates)
    trainer.stage2_sampler_policy = 'latent_balanced'
    assert trainer.set_initial_training_values(trainer.args, loader, 4 * gas)[1] == updates - 1


@pytest.mark.parametrize('gas,step,want_epoch,want_skip', [
    (2, 422, 0, 844), (2, 423, 1, 0), (4, 211, 0, 844), (4, 212, 1, 0)])
def test_inherited_resume_control_flow_preserves_pre_tail_microstep(tmp_path, gas, step, want_epoch, want_skip):
    # Catches correct helper arithmetic that the actual Trainer resume never uses.
    trainer = cpu_trainer(tmp_path/'output', gas, max_steps=1000)
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    TrainerState(global_step=step, train_batch_size=1).save_to_json(str(checkpoint/'trainer_state.json'))
    observed = {}
    class ReachedResume(Exception):
        pass
    def trace(frame, event, arg):
        if (event == 'line' and frame.f_code is Trainer._inner_training_loop.__code__
                and 'epochs_trained' in frame.f_locals
                and 'steps_trained_in_current_epoch' in frame.f_locals):
            # The callback references are updated immediately after resume arithmetic.
            if frame.f_lineno == Trainer._inner_training_loop.__code__.co_firstlineno:
                return trace
            import linecache
            if 'self.state.init_training_references(self, max_steps' in linecache.getline(frame.f_code.co_filename, frame.f_lineno):
                observed.update(epoch=frame.f_locals['epochs_trained'], skip=frame.f_locals['steps_trained_in_current_epoch'])
                raise ReachedResume
        return trace
    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(ReachedResume):
            trainer._inner_training_loop(batch_size=1, args=trainer.args, resume_from_checkpoint=str(checkpoint))
    finally:
        sys.settrace(previous)
    assert observed == dict(epoch=want_epoch, skip=want_skip)
    if want_skip:
        source = SeedableRandomSampler(range(3379), data_seed=42)
        tail = list(source)[3376:]
        assert len(tail) == 3
        sampler = reference_sampler(range(3379), num_replicas=4, seed=42, gradient_accumulation_steps=gas)
        shards = [list(BatchSamplerShard(BatchSampler(sampler, 1, False), num_processes=4,
            process_index=rank, even_batches=False))[observed['skip']:] for rank in range(4)]
        assert [batch[0] for row in zip(*shards) for batch in row] == tail + [-1]


def test_reference_cli_coordinator_and_launcher_preserve_explicit_opt_in():
    # Catches a sampler helper with no executable launch path or dropped remote opt-in.
    from utils_deepseed import parse_args
    from src.multinode_runtime import validated_stage2_environment
    from tests.stage2.test_launcher import run_launcher, dry_run_tokens, option_value
    args = parse_args(['--stage2_sampler_policy', 'reference_random', '--stage2_reference_sampler_probe_dir', 'results/probe'])
    assert args.stage2_sampler_policy == 'reference_random'
    env = validated_stage2_environment(dict(STAGE1_CHECKPOINT='a', DATA_PATH='b', IMAGE_ROOT='c',
        STAGE2_SAMPLER_POLICY='reference_random', STAGE2_REFERENCE_SAMPLER_PROBE_DIR='results/probe'),
        output_dir='checkpoints/test', log_file='logs/test', deepspeed_config='configs/deepspeed_zero2.json')
    assert env['STAGE2_SAMPLER_POLICY'] == 'reference_random'
    tokens = dry_run_tokens(run_launcher(env_overrides=env))
    assert option_value(tokens, '--stage2_sampler_policy') == 'reference_random'
    assert option_value(tokens, '--stage2_reference_sampler_probe_dir') == 'results/probe'


def test_fresh_voff_zero2_selects_reference_but_saved_legacy_is_preserved(tmp_path):
    from deepspeed_runtime import DeepSpeedRuntimeProfile
    assert hasattr(sampling, 'select_stage2_sampler_policy'), 'fresh reproduction route is not selected'
    profile = DeepSpeedRuntimeProfile(True, 2, None, None, 4)
    config = context_config()
    select = sampling.select_stage2_sampler_policy
    assert select('auto', profile=profile, config=config, resume=None) == 'reference_random'
    assert select('auto', profile=replace(profile, zero_stage=3), config=config, resume=None) == 'latent_balanced'
    assert select('auto', profile=profile, config=StepPreferenceConfig(), resume=None) == 'latent_balanced'
    from tests.stage2.test_sampler import examples_with_counts
    identity = sampling.LatentBlockBalancedSampler(examples_with_counts([1, 2]), num_replicas=4, seed=42).identity
    sampling.write_stage2_sampler_sidecar(tmp_path, identity, global_step=1)
    assert select('auto', profile=profile, config=config, resume=tmp_path) == 'latent_balanced'
    with pytest.raises(ValueError, match='policy'):
        select('reference_random', profile=profile, config=config, resume=tmp_path)


@pytest.mark.parametrize('zero,config', [
    (3, context_config()),
    (3, StepPreferenceConfig()),
    (2, StepPreferenceConfig(enabled=True, mode='stage2a')),
])
def test_explicit_reference_policy_rejects_outside_voff_zero2(tmp_path, zero, config):
    from deepspeed_runtime import DeepSpeedRuntimeProfile
    with pytest.raises(ValueError, match='reference_random requires'):
        Stage2Trainer(model=ScalarModel(), args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[], per_device_train_batch_size=1),
            train_dataset=[1], stage2_sampler_policy='reference_random', step_preference_config=config,
            deepspeed_runtime_profile=DeepSpeedRuntimeProfile(True, zero, None, None, 4))


def test_reference_constructor_rejects_single_actual_rank_despite_four_rank_profile(tmp_path):
    from deepspeed_runtime import DeepSpeedRuntimeProfile
    with pytest.raises(ValueError, match='four actual ranks'):
        Stage2Trainer(model=ScalarModel(), args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[], per_device_train_batch_size=1),
            train_dataset=[1], stage2_sampler_policy='reference_random', step_preference_config=context_config(),
            deepspeed_runtime_profile=DeepSpeedRuntimeProfile(True, 2, None, None, 4))


def test_effective_data_seed_and_real_accelerate_loader_keep_one_padding_slot():
    # Catches ignoring TrainingArguments.data_seed or Accelerate resampling/padding.
    from accelerate.data_loader import prepare_data_loader
    data = sampling.ReferenceRandomDataset(list(range(11)))
    trainer = object.__new__(Stage2Trainer)
    trainer.stage2_sampler_policy = 'reference_random'
    trainer.args = SimpleNamespace(world_size=4, seed=42, data_seed=917, gradient_accumulation_steps=2)
    trainer.train_dataset = data
    source = SeedableRandomSampler(range(11), data_seed=917)
    expected = list(source)
    def collate(rows):
        return {'index': torch.tensor([row.index for row in rows]),
                'weight': torch.tensor([row.weight for row in rows])}
    loaders = [prepare_data_loader(DataLoader(data, batch_size=1, sampler=trainer._get_train_sampler(),
        collate_fn=collate), num_processes=4, process_index=rank, even_batches=False,
        use_seedable_sampler=True, put_on_device=False) for rank in range(4)]
    observed = [[batch['index'].item() for batch in loader] for loader in loaders]
    assert [index for row in zip(*observed) for index in row] == expected + [-1]
    assert trainer.stage2_sampler_identity.seed == 917


def test_padding_training_step_restores_rng_drawn_in_forward_and_backward(tmp_path):
    # Catches a forward-only RNG fork that lets stochastic backward alter next draws.
    class RandomBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value
        @staticmethod
        def backward(ctx, grad):
            torch.rand(3)
            return grad
    class StochasticModel(ScalarModel):
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(loss=RandomBackward.apply(self.weight.square() * torch.rand(())))
    trainer = cpu_trainer(tmp_path, 2)
    model = StochasticModel()
    trainer.model_accepts_loss_kwargs = False
    torch.manual_seed(42)
    before = torch.get_rng_state().clone()
    trainer.training_step(model, {'input_ids': torch.tensor([[1]]), 'stage2_sample_weight': 0.,
        'stage2_sample_index': -1, 'stage2_sample_latent_block_count': 1})
    assert torch.equal(before, torch.get_rng_state())
    assert model.weight.grad.item() == 0.
    trainer.training_step(model, {'input_ids': torch.tensor([[1]])})
    assert not torch.equal(before, torch.get_rng_state())
    assert model.weight.grad.item() != 0.
