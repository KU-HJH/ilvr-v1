"""Initial reference persistence against actual DoRA and sparse token modules."""
import copy
import importlib
from pathlib import Path
import sys

import pytest
import torch
from stage1.ema import Stage1EmaTeacher
from stage1.peft import build_stage1_tuned_model

# Reuse the existing tiny model/processor fixture convention, never real weights.
sys.path.insert(0, str(Path(__file__).parents[1]))
from test_checkpointing import (tiny_original_qwen, stage1_tuning, TinyStage1Processor,
                                LATENT_TOKEN_IDS, RESIZED_VOCAB_SIZE)


def api():
    assert importlib.util.find_spec('stage1.v5_reference'), 'immutable reference implementation missing'
    return importlib.import_module('stage1.v5_reference')


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    import artifact_paths
    monkeypatch.setattr(artifact_paths, 'CANONICAL_CHECKPOINT_ROOT', tmp_path / 'checkpoints')
    base_path = tmp_path / 'base'
    base = tiny_original_qwen()
    base.save_pretrained(base_path)
    def fresh():
        model = copy.deepcopy(base)
        model.resize_token_embeddings(RESIZED_VOCAB_SIZE)
        model.config.latent_token_id, model.config.latent_start_id, model.config.latent_end_id = LATENT_TOKEN_IDS
        return build_stage1_tuned_model(model, stage1_tuning(), LATENT_TOKEN_IDS)
    student = fresh()
    processor = TinyStage1Processor()
    directory = tmp_path / 'checkpoints' / 'initial'
    return student, processor, base_path, directory, fresh


def test_restores_initial_state_from_original_base_not_resumed_student(bundle):
    ref = api()
    student, processor, base, directory, fresh = bundle
    teacher = Stage1EmaTeacher(student, 1., 'dora')
    count = []
    teacher.model.register_forward_pre_hook(lambda *args: count.append(1))
    identity = ref.capture_v5_reference(teacher, student, processor, base_path=base, directory=directory)
    initial = teacher.state_dict()['parameters']
    assert any('trainable_tokens_delta' in name for name in identity.selected_parameter_names)
    assert any('output_delta' in name for name in identity.selected_parameter_names)
    with torch.no_grad():
        for p in student.parameters():
            if p.requires_grad:
                p.add_(3.)
    resumed = Stage1EmaTeacher(student, 1., 'dora', initial_reference_model=fresh())
    resumed.model.register_forward_pre_hook(lambda *args: count.append(1))
    assert all(not torch.equal(dict(student.named_parameters())[name], value) for name, value in initial.items())
    ref.restore_v5_reference(resumed, student, processor, base_path=base, directory=directory, expected=identity)
    restored = resumed.state_dict()['parameters']
    assert all(torch.equal(restored[name], value) for name, value in initial.items())
    assert count == []
    assert all(p.grad is None and not p.requires_grad for p in resumed.model.parameters())
    assert all(not m.training for m in resumed.model.modules())
    assert ref.capture_v5_reference(resumed, student, processor, base_path=base,
                                    directory=directory.parent / 'copy') == identity
    changed = Stage1EmaTeacher(student, 1., 'dora')
    changed_identity = ref.capture_v5_reference(changed, student, processor, base_path=base,
                                               directory=directory.parent / 'resumed-policy')
    assert changed_identity.effective_state_sha256 != identity.effective_state_sha256


@pytest.mark.parametrize('tamper', ['base', 'processor', 'state', 'missing', 'identity'])
def test_rejects_content_conflicts_before_forward(bundle, tamper):
    ref = api()
    student, processor, base, directory, fresh = bundle
    teacher = Stage1EmaTeacher(student, 1., 'dora')
    identity = ref.capture_v5_reference(teacher, student, processor, base_path=base, directory=directory)
    if tamper == 'base':
        with (base / 'model.safetensors').open('ab') as stream:
            stream.write(b'changed')
    elif tamper == 'processor':
        processor.tokenizer._size += 1
    elif tamper == 'state':
        path = directory / ref.V5_REFERENCE_FILENAME
        state = torch.load(path, weights_only=True)
        next(iter(state['parameters'].values())).add_(1.)
        torch.save(state, path)
    elif tamper == 'missing':
        (directory / ref.V5_REFERENCE_FILENAME).unlink()
    else:
        from dataclasses import replace
        identity = replace(identity, effective_state_sha256='0' * 64)
    calls = []
    teacher.model.register_forward_pre_hook(lambda *args: calls.append(1))
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        ref.restore_v5_reference(teacher, student, processor, base_path=base, directory=directory, expected=identity)
    assert calls == []


@pytest.mark.parametrize('invalid', ['moving', 'alias', 'frozen_adapter', 'uncovered_base', 'full'])
def test_rejects_incomplete_or_mutable_reference(bundle, invalid):
    ref = api()
    student, processor, base, directory, _ = bundle
    teacher = Stage1EmaTeacher(student, 0.9 if invalid == 'moving' else 1., 'dora')
    if invalid == 'alias':
        name = teacher.parameter_names[0]
        owner, key = name.rsplit('.', 1)
        teacher.model.get_submodule(owner)._parameters[key] = dict(student.named_parameters())[name]
    elif invalid == 'frozen_adapter':
        dict(student.named_parameters())[teacher.parameter_names[0]].requires_grad_(False)
    elif invalid == 'uncovered_base':
        with torch.no_grad():
            teacher.model.base_model.model.model.norm.weight.add_(1.)
    elif invalid == 'full':
        teacher.method = 'full'
    with pytest.raises((ValueError, RuntimeError)):
        ref.capture_v5_reference(teacher, student, processor, base_path=base, directory=directory)


def test_reference_local_rows_change_logits_but_shared_latent_values_stay_fixed(bundle):
    ref = api()
    student, processor, base, directory, _ = bundle
    teacher = Stage1EmaTeacher(student, 1., 'dora')
    ref.capture_v5_reference(teacher, student, processor, base_path=base, directory=directory)
    from peft_common import resolve_qwen2_5_vl_model
    reference_model = resolve_qwen2_5_vl_model(teacher.model)
    shared = torch.tensor([[0.3] * 8]).detach()
    ids = torch.tensor([[1, 125, 125, 2]])
    kwargs = dict(input_ids=ids, latent_hidden_states=shared,
                  image_out_mask=torch.tensor([[False, True, False, False]]),
                  position_ids=torch.arange(4)[None, None].expand(3, 1, 4),
                  attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True,
                  stage1_selected_positions=((0, 0), (0, 1), (0, 2)))
    with torch.no_grad():
        before = reference_model(**kwargs)
        for name, p in teacher.model.named_parameters():
            if 'trainable_tokens_delta' in name:
                p.add_(0.5)
        after = reference_model(**kwargs)
    assert not torch.equal(before.inputs_embeds[0, 2], after.inputs_embeds[0, 2])
    assert not torch.equal(before.stage1_selected_logits, after.stage1_selected_logits)
    torch.testing.assert_close(before.inputs_embeds[0, 1], shared[0])
    torch.testing.assert_close(after.inputs_embeds[0, 1], shared[0])


def test_refuses_existing_initial_snapshot_overwrite(bundle):
    ref = api()
    student, processor, base, directory, _ = bundle
    teacher = Stage1EmaTeacher(student, 1., 'dora')
    ref.capture_v5_reference(teacher, student, processor, base_path=base, directory=directory)
    with torch.no_grad():
        dict(teacher.model.named_parameters())[teacher.parameter_names[0]].add_(2.)
    with pytest.raises(ValueError, match='immutable'):
        ref.capture_v5_reference(teacher, student, processor, base_path=base, directory=directory)


def test_rejects_reference_adapter_scaling_conflict(bundle):
    ref = api()
    student, processor, base, directory, _ = bundle
    teacher = Stage1EmaTeacher(student, 1., 'dora')
    teacher.model.peft_config['default'].lora_alpha *= 2
    with pytest.raises(ValueError, match='adapter'):
        ref.capture_v5_reference(teacher, student, processor, base_path=base, directory=directory)


def test_explicit_initial_model_never_freezes_aliased_student(bundle):
    student, _, _, _, _ = bundle
    trainable = [p for p in student.parameters() if p.requires_grad]
    with pytest.raises(ValueError, match='alias'):
        Stage1EmaTeacher(student, 1., 'dora', initial_reference_model=student)
    assert all(p.requires_grad for p in trainable)


@pytest.mark.parametrize('change', ['vocabulary', 'special_ids'])
def test_initial_processor_must_match_reference_token_layout(bundle, change):
    ref = api()
    student, processor, base, directory, _ = bundle
    teacher = Stage1EmaTeacher(student, 1., 'dora')
    if change == 'vocabulary':
        processor.tokenizer._size += 1
    else:
        processor.tokenizer._ids['<|latent_pad|>'] = 1
    with pytest.raises(ValueError, match='token|vocab'):
        ref.capture_v5_reference(teacher, student, processor, base_path=base, directory=directory)


def test_existing_reference_metadata_cannot_change_without_tensor_change(bundle):
    ref = api()
    student, processor, base, directory, _ = bundle
    teacher = Stage1EmaTeacher(student, 1., 'dora')
    ref.capture_v5_reference(teacher, student, processor, base_path=base, directory=directory)
    path = directory / ref.V5_REFERENCE_FILENAME
    state = torch.load(path, weights_only=True)
    state['decay'] = 0.9
    torch.save(state, path)
    with pytest.raises(ValueError, match='immutable'):
        ref.capture_v5_reference(teacher, student, processor, base_path=base, directory=directory)


def test_original_tied_safetensors_alias_is_complete_effective_base_coverage(bundle):
    ref = api()
    _, processor, base_path, directory, _ = bundle
    original = tiny_original_qwen()
    original.config.tie_word_embeddings = True
    original.tie_weights()
    base_path = base_path.parent / 'tied-original'
    original.save_pretrained(base_path)
    from safetensors import safe_open
    with safe_open(base_path / 'model.safetensors', framework='pt') as stream:
        assert 'lm_head.weight' not in stream.keys()
    original.resize_token_embeddings(RESIZED_VOCAB_SIZE)
    original.config.latent_token_id, original.config.latent_start_id, original.config.latent_end_id = LATENT_TOKEN_IDS
    student = build_stage1_tuned_model(original, stage1_tuning(), LATENT_TOKEN_IDS)
    teacher = Stage1EmaTeacher(student, 1., 'dora')
    identity = ref.capture_v5_reference(teacher, student, processor, base_path=base_path, directory=directory)
    ref.restore_v5_reference(teacher, student, processor, base_path=base_path, directory=directory, expected=identity)


@pytest.mark.parametrize('operation', ['capture', 'restore'])
@pytest.mark.parametrize('drift', ['scaling', 'active_adapter', 'variant', 'use_dora'])
def test_runtime_adapter_drift_rejects_before_forward(bundle, operation, drift):
    ref = api()
    student, processor, base, directory, _ = bundle
    teacher = Stage1EmaTeacher(student, 1., 'dora')
    module = next(m for m in teacher.model.modules() if hasattr(m, 'lora_B') and 'default' in m.lora_B)
    # Nonzero B makes the scale change alter the actual effective module policy.
    with torch.no_grad():
        module.lora_B['default'].weight.fill_(0.1)
    identity = ref.capture_v5_reference(teacher, student, processor, base_path=base, directory=directory)
    parameters = {name: p.detach().clone() for name, p in teacher.model.named_parameters()}
    declared_config = copy.deepcopy(teacher.model.peft_config['default'].to_dict())
    if drift == 'scaling':
        x = torch.arange(module.in_features, dtype=module.weight.dtype)[None, None] / module.in_features
        with torch.no_grad():
            before = module(x)
            module.set_scale('default', 2.)
            after = module(x)
        assert not torch.equal(before, after)
    elif drift == 'active_adapter':
        module._active_adapter = []
    elif drift == 'variant':
        module.lora_variant.clear()
    else:
        module.use_dora['default'] = False
    assert teacher.model.peft_config['default'].to_dict() == declared_config
    assert all(torch.equal(dict(teacher.model.named_parameters())[name], value) for name, value in parameters.items())
    calls = []
    handles = [m.register_forward_pre_hook(lambda *args: calls.append(1)) for m in teacher.model.modules()]
    try:
        with pytest.raises(ValueError, match='runtime adapter'):
            if operation == 'capture':
                ref.capture_v5_reference(teacher, student, processor, base_path=base, directory=directory)
            else:
                ref.restore_v5_reference(teacher, student, processor, base_path=base, directory=directory, expected=identity)
        assert calls == []
    finally:
        for handle in handles:
            handle.remove()
