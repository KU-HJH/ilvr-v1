"""Matched checkpoints must restore the saved objective and execution profile."""
import json
from copy import deepcopy
from types import SimpleNamespace
import hashlib

import pytest

from stage1.matched_execution import (
    reject_matched_execution_resume, write_matched_execution_identity,
)


def checkpoint_with_resume_contract(tmp_path):
    checkpoint = tmp_path / 'source' / 'checkpoint-216'
    checkpoint.mkdir(parents=True)
    write_matched_execution_identity(checkpoint, ce_entropy_top_percent=20)
    contract = {
        'schema_version': 1,
        'kind': 'stage1_matched_resume',
        'source': {'run_id': 'source', 'checkpoint_step': 216,
                   'commit': '8' * 40},
        'profile': {
            'worker': {'stage1_variant': 'standard', 'stage1_match_v5_execution': True,
                       'stage1_ce_entropy_top_percent': 20., 'ce_weight': 1.,
                       'sim_weight': 1., 'ema_tau': 1.},
            'training': {'max_steps': 2160, 'num_train_epochs': 10.,
                         'per_device_train_batch_size': 1,
                         'gradient_accumulation_steps': 4,
                         'dataloader_num_workers': 0, 'learning_rate': 5e-5,
                         'warmup_steps': 10, 'seed': 42},
            'data_sha256': 'd' * 64,
            'base_sha256': {'config.json': 'b' * 64},
        },
    }
    (checkpoint / 'stage1_matched_resume.json').write_text(json.dumps(contract))
    return checkpoint, contract


def test_compatible_matched_checkpoint_is_not_blanket_rejected(tmp_path):
    """The historical fresh-only guard currently blocks an otherwise bound resume."""
    checkpoint, _ = checkpoint_with_resume_contract(tmp_path)
    before = (checkpoint / 'stage1_matched_execution.json').read_bytes()
    reject_matched_execution_resume(checkpoint, enabled=True)
    assert (checkpoint / 'stage1_matched_execution.json').read_bytes() == before


@pytest.mark.parametrize('section,key,value', [
    ('training', 'max_steps', 216),
    ('training', 'dataloader_num_workers', 4),
    ('training', 'learning_rate', 1e-4),
    ('training', 'seed', 43),
    ('worker', 'stage1_ce_entropy_top_percent', None),
    ('worker', 'sim_weight', .1),
])
def test_resume_refuses_live_profile_drift(tmp_path, section, key, value):
    """A saved globalstep cannot authorize a changed objective, LR or loader."""
    from stage1.matched_resume import validate_matched_resume_profile
    _, contract = checkpoint_with_resume_contract(tmp_path)
    requested = deepcopy(contract['profile'])
    requested[section][key] = value
    with pytest.raises(ValueError, match='profile'):
        validate_matched_resume_profile(contract['profile'], requested)


def test_resume_refuses_another_dataset_even_at_same_path(tmp_path):
    from stage1.matched_resume import validate_matched_resume_profile
    _, contract = checkpoint_with_resume_contract(tmp_path)
    requested = deepcopy(contract['profile'])
    requested['data_sha256'] = 'e' * 64
    with pytest.raises(ValueError, match='profile'):
        validate_matched_resume_profile(contract['profile'], requested)


def test_marker_and_resume_contract_must_agree_on_clean20(tmp_path):
    checkpoint, contract = checkpoint_with_resume_contract(tmp_path)
    contract['profile']['worker']['stage1_ce_entropy_top_percent'] = None
    (checkpoint / 'stage1_matched_resume.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='objective marker'):
        reject_matched_execution_resume(checkpoint, enabled=True)


def test_existing_marker_cannot_resume_as_ordinary_standard(tmp_path):
    checkpoint, _ = checkpoint_with_resume_contract(tmp_path)
    with pytest.raises(ValueError, match='execution flag'):
        reject_matched_execution_resume(checkpoint, enabled=False)


def test_legacy_checkpoint_without_imported_profile_still_fails_closed(tmp_path):
    checkpoint, _ = checkpoint_with_resume_contract(tmp_path)
    (checkpoint / 'stage1_matched_resume.json').unlink()
    with pytest.raises(ValueError, match='validated resume contract'):
        reject_matched_execution_resume(checkpoint, enabled=True)


def test_resume_contract_without_marker_cannot_fall_through_to_full_ce(tmp_path):
    checkpoint, _ = checkpoint_with_resume_contract(tmp_path)
    (checkpoint / 'stage1_matched_execution.json').unlink()
    with pytest.raises(ValueError, match='matched'):
        reject_matched_execution_resume(checkpoint, enabled=False)


def test_profile_captures_content_and_effective_settings(tmp_path):
    from stage1.matched_resume import capture_matched_resume_profile
    data = tmp_path / 'TRAIN.jsonl'
    data.write_bytes(b'training samples\n')
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'config.json').write_bytes(b'{"model_type":"qwen2_5_vl"}')
    (base / 'model.safetensors').write_bytes(b'original base weights')
    worker = SimpleNamespace(model=str(base), data_path=str(data),
        stage1_variant='standard', stage1_match_v5_execution=True,
        stage1_ce_entropy_top_percent=20., ce_weight=1., sim_weight=1., ema_tau=1.,
        lora_target_modules='up_proj,q_proj', latent_size=8)
    training = SimpleNamespace(max_steps=2160, num_train_epochs=10.,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        dataloader_num_workers=0, learning_rate=5e-5, warmup_steps=10, seed=42)
    profile = capture_matched_resume_profile(worker, training)
    assert profile['training']['max_steps'] == 2160
    assert profile['training']['dataloader_num_workers'] == 0
    assert profile['worker']['stage1_ce_entropy_top_percent'] == 20.
    assert profile['worker']['lora_target_modules'] == ['q_proj', 'up_proj']
    assert profile['data_sha256'] == hashlib.sha256(b'training samples\n').hexdigest()
    assert profile['base_sha256']['model.safetensors'] == hashlib.sha256(b'original base weights').hexdigest()
    data.write_bytes(b'changed samples\n')
    changed = capture_matched_resume_profile(worker, training)
    assert changed['data_sha256'] != profile['data_sha256']


def test_exported_contract_is_readable_without_rewriting_legacy_marker(tmp_path):
    from stage1.matched_resume import write_matched_resume_contract
    checkpoint, contract = checkpoint_with_resume_contract(tmp_path)
    (checkpoint / 'stage1_matched_resume.json').unlink()
    before = (checkpoint / 'stage1_matched_execution.json').read_bytes()
    write_matched_resume_contract(checkpoint, contract['profile'], source=contract['source'])
    restored = reject_matched_execution_resume(checkpoint, enabled=True)
    assert restored['source']['checkpoint_step'] == 216
    assert restored['profile']['training']['max_steps'] == 2160
    assert (checkpoint / 'stage1_matched_execution.json').read_bytes() == before


def test_exported_contract_rejects_colliding_profile(tmp_path):
    from stage1.matched_resume import write_matched_resume_contract
    checkpoint, contract = checkpoint_with_resume_contract(tmp_path)
    before = (checkpoint / 'stage1_matched_resume.json').read_bytes()
    contract['profile']['training']['max_steps'] = 432
    with pytest.raises(ValueError, match='exist|conflict'):
        write_matched_resume_contract(checkpoint, contract['profile'], source=contract['source'])
    assert (checkpoint / 'stage1_matched_resume.json').read_bytes() == before


def test_native_clean20_resume_matches_epoch_tail_and_all_state(tmp_path):
    """Real Stage1 Trainer/EMA/clean20: seven microbatches, GAS4, split at epoch1."""
    from dataclasses import replace
    import torch
    from transformers import TrainerCallback
    from accelerate.state import AcceleratorState, GradientState, PartialState
    from tests.stage1.test_integration import _stage1_args, _tiny_dependencies
    from tests.stage1.test_checkpointing import _assert_nested_state_equal, _capture_rng_state, _assert_rng_state_equal
    from stage1.composition import build_stage1_training_components
    from stage1.matched_execution import enable_matched_standard_execution
    from stage1.matched_resume import _values, TRAINING_FIELDS

    class Stop(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.global_step == 2:
                control.should_training_stop = True
    class Before(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer
            self.states = {}
        def on_step_begin(self, args, state, control, **kwargs):
            self.states[state.global_step] = deepcopy(self.trainer.model.state_dict())
    def reset():
        AcceleratorState._reset_state(reset_partial_state=True)
        PartialState._reset_state()
        GradientState._reset_state()
    def make(name):
        reset()
        args = _stage1_args(tmp_path, method='dora', output_dir=tmp_path/name,
            batch_size=1, gradient_accumulation_steps=4, max_steps=6, save_steps=2)
        args.ema_tau = 1.
        args.lora_dropout = .05
        args.sim_weight = 1.
        trainer = build_stage1_training_components(args, dependencies=replace(
            _tiny_dependencies(), prepare_dataset=lambda *_: list(range(7)))).trainer
        enable_matched_standard_execution(trainer)
        trainer.ce_weight = trainer.sim_weight = 1.
        trainer.ce_entropy_top_percent = 20.
        trainer.ce_entropy_special_token_ids = (125, 126, 127)
        trainer.matched_resume_profile = {
            'worker': {'stage1_variant': 'standard', 'stage1_match_v5_execution': True,
                       'stage1_ce_entropy_top_percent': 20., 'ce_weight': 1.,
                       'sim_weight': 1., 'ema_tau': 1.},
            'training': _values(trainer.args, TRAINING_FIELDS),
            'data_sha256': 'd'*64, 'base_sha256': {'config.json': 'b'*64}}
        order = []
        original = trainer.data_collator
        def collate(rows):
            order.extend(rows)
            return original(rows)
        trainer.data_collator = collate
        return trainer, order
    try:
        full, full_order = make('full')
        full_before = Before(full)
        full.add_callback(full_before)
        full.train()
        expected_model = deepcopy(full.model.state_dict())
        expected_optimizer = deepcopy(full.optimizer.state_dict())
        expected_scheduler = deepcopy(full.lr_scheduler.state_dict())
        expected_teacher = deepcopy(full.ema_teacher.state_dict())
        expected_rng = _capture_rng_state()
        interrupted, first_order = make('split')
        interrupted.add_callback(Stop())
        interrupted.train()
        checkpoint = tmp_path/'split/checkpoint-2'
        assert (checkpoint/'stage1_matched_resume.json').is_file()
        resumed, resumed_order = make('continued')
        resumed_before = Before(resumed)
        resumed.add_callback(resumed_before)
        resumed.train(resume_from_checkpoint=str(checkpoint))
        assert resumed.state.global_step == 6
        assert first_order + resumed_order == full_order
        for name, tensor in resumed_before.states[2].items():
            torch.testing.assert_close(tensor, full_before.states[2][name], rtol=0, atol=0, msg='restored '+name)
        for name, tensor in resumed.model.state_dict().items():
            torch.testing.assert_close(tensor, expected_model[name], rtol=0, atol=0, msg=name)
        _assert_nested_state_equal(resumed.optimizer.state_dict(), expected_optimizer)
        _assert_nested_state_equal(resumed.lr_scheduler.state_dict(), expected_scheduler)
        _assert_nested_state_equal(resumed.ema_teacher.state_dict(), expected_teacher)
        _assert_rng_state_equal(_capture_rng_state(), expected_rng)
        terminal, terminal_order = make('terminal')
        terminal.train(resume_from_checkpoint=str(tmp_path/'full/checkpoint-6'))
        assert terminal.state.global_step == 6
        assert terminal_order == []
        _assert_nested_state_equal(terminal.model.state_dict(), expected_model)
        _assert_nested_state_equal(terminal.optimizer.state_dict(), expected_optimizer)
        _assert_nested_state_equal(terminal.lr_scheduler.state_dict(), expected_scheduler)
        _assert_nested_state_equal(terminal.ema_teacher.state_dict(), expected_teacher)
    finally:
        reset()


def test_direct_trainer_rejects_objective_drift_before_runtime_setup(tmp_path):
    from stage1.trainer import CustomTrainerStage1
    checkpoint, contract = checkpoint_with_resume_contract(tmp_path)
    trainer = CustomTrainerStage1.__new__(CustomTrainerStage1)
    trainer.stage1_match_v5_execution = True
    trainer.matched_resume_profile = contract['profile']
    trainer.args = SimpleNamespace(**contract['profile']['training'])
    trainer.ce_entropy_top_percent = None
    trainer.ce_weight = trainer.sim_weight = 1.
    trainer.ema_teacher = SimpleNamespace(decay=1.)
    with pytest.raises(ValueError, match='profile'):
        trainer.train(resume_from_checkpoint=checkpoint)
