"""Causal CPU contracts for the passive OFF/cache comparison diagnostic."""
import importlib

import pytest
import torch
import torch.nn.functional as F

from evaluation.continuous_latent import ContinuousLatentController, ContinuousLatentGenerationConfig
from tests.evaluation.test_continuous_latent import TOKENS
from tests.evaluation.test_off_kv_cache import image_prompt
from tests.stage2.helpers import tiny_stage2_qwen


def probe():
    return importlib.import_module('scripts.probe_stage1_off_kv')


def test_rejects_observer_interference_even_when_common_prefix_matches():
    # Removing full-suffix checking would wrongly admit this changed final token.
    p = probe()
    frozen = {'generated_suffix_token_ids': list(range(32)), 'prompt_identity': 'same'}
    actual = dict(frozen, generated_suffix_token_ids=list(range(31))+[99])
    with pytest.raises(ValueError, match='observer fidelity'):
        p.require_fidelity(actual, frozen)
    assert p.require_fidelity(frozen, frozen) is None


def test_alignment_includes_branch_prediction_but_never_compares_its_children():
    p = probe()
    assert p.comparison_steps([3, 4, 5], [3, 9, 5]) == 2
    assert p.comparison_steps([3, 4], [3, 4]) == 2
    # Shape differences must not conceal equal newest logical rows.
    d = p.tensor_difference(torch.tensor([[[1., 2.]]]), torch.tensor([[[1., 3.]]]))
    assert d['max_abs'] == 1 and d['unequal_count'] == 1
    assert p.last_query_visibility(None, 3, 3, True).tolist() == [True]*3
    assert p.last_query_visibility(None, 1, 3, False).tolist() == [True]*3
    mask = torch.tensor([[[[0., -float('inf'), 0.]]]])
    assert p.last_query_visibility(mask, 1, 3, False).tolist() == [True, False, True]


def generate(model, cached):
    return ContinuousLatentController().generate(model=model, token_ids=TOKENS,
        generation_config=ContinuousLatentGenerationConfig(max_new_tokens=3, repetition_penalty=1.05),
        disable_latent_generation=True, use_kv_cache=cached, **image_prompt())


def test_native_observer_collects_effective_operands_and_preserves_generation():
    p = probe()
    torch.manual_seed(12)
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    with torch.inference_mode():
        expected_off = generate(model, False)
        expected_on = generate(model, True)
        with p.PassiveTrace(model, max_steps=3) as off:
            actual_off = generate(model, False)
        with p.PassiveTrace(model, max_steps=3) as on:
            actual_on = generate(model, True)
    assert actual_off.generated_suffix_ids == expected_off.generated_suffix_ids
    assert actual_on.generated_suffix_ids == expected_on.generated_suffix_ids
    assert len(off.steps) == len(on.steps) == 3
    assert off.steps[1]['history'] == on.steps[1]['history']
    assert off.steps[1]['decoder']['query_length'] == 9
    assert on.steps[1]['decoder']['query_length'] == 1
    assert on.steps[1]['decoder']['past_length'] == 8
    for trace in (off, on):
        events = trace.steps[1]['events']
        assert events['layer.0.sdpa.k']['value'].shape == (1, 2, 9, 4)
        assert events['layer.0.sdpa.visibility']['value'].tolist() == [False]+[True]*8
        assert 'layer.0.q_proj' in events and 'logits.raw' in events and 'logits.post_penalty' in events
    compared = p.compare_traces(off, on)
    assert compared['compared_steps'] >= 1
    # A later generation sees no retained cache or observation scope.
    with torch.inference_mode():
        again = generate(model, True)
    assert again.generated_suffix_ids == expected_on.generated_suffix_ids


def test_restores_hooks_and_wrappers_on_exception_and_replays_real_linear():
    p = probe()
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
    original_sdpa = F.scaled_dot_product_attention
    original_penalty = RepetitionPenaltyLogitsProcessor.__call__
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    hook_counts = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    with pytest.raises(RuntimeError, match='deliberate'):
        with p.PassiveTrace(model, max_steps=2):
            raise RuntimeError('deliberate')
    assert F.scaled_dot_product_attention is original_sdpa
    assert RepetitionPenaltyLogitsProcessor.__call__ is original_penalty
    assert hook_counts == [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    linear = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
    replay = p.replay_row_operation(linear, torch.tensor([[[1., 2., 3.]]]).repeat(1, 7, 1))
    assert replay['difference']['max_abs'] == 0
    assert replay['full_shape'] == [1, 7, 3]


def test_published_sample_image_is_resolved_before_existing_runner_opens_it():
    # Omitting eval.main's image resolution makes direct run_one_example fail.
    sample = probe().selected_sample()
    from pathlib import Path
    assert Path(sample['image_input'][0]).is_absolute()
    assert Path(sample['image_input'][0]).is_file()
    assert sample['text_input'].startswith('In △ABC')


def test_output_reservation_rejects_overwrite_and_outside_results(tmp_path, monkeypatch):
    p = probe()
    monkeypatch.chdir(tmp_path)
    output = p.output_directory('results/probe')
    (output/'report.json').write_text('original')
    with pytest.raises(FileExistsError):
        p.output_directory('results/probe')
    assert (output/'report.json').read_text() == 'original'
    with pytest.raises(ValueError, match='results'):
        p.output_directory('logs/probe')


from tests.evaluation.test_numerics import restore_precision, state


def complete_trace(cached=False, count=3):
    """Literal event tensors, logical histories and native cache geometry."""
    trace = probe().PassiveTrace(None, max_steps=count)
    for i in range(count):
        trace.steps.append(dict(history=[7, 8]+[4]*i,
            decoder=dict(query_length=1 if cached and i else 2+i,
                past_length=1+i if cached and i else 0, use_cache=cached,
                returned_cache_length=2+i if cached else None),
            events={name: dict(value=torch.tensor([[1., 2., 3., 4., 5.]]), shape=[1, 5])
                    for name in ['layer.0.q_proj', 'logits.raw', 'logits.post_penalty']}))
    return trace


def result_tokens(count=3):
    return dict(generated_suffix_token_ids=[4]*count, prompt_identity='same',
                prompt_token_length=2, stop_reason='max_new_tokens')


def test_numerical_verdict_requires_complete_finite_equal_tensors_and_cache_geometry():
    p = probe()
    assert hasattr(p, 'alignment_verdict'), 'missing numerical alignment verdict'
    verdict = p.alignment_verdict(complete_trace(), complete_trace(True), result_tokens(), result_tokens(), expected_steps=3)
    assert verdict['status'] == 'pass'
    assert all(verdict[key] is True for key in ['q_equal','logits_equal','tokens_equal','coverage_complete','finite','history_equal','cache_geometry_expected'])


@pytest.mark.parametrize('event', ['layer.0.q_proj', 'logits.raw', 'logits.post_penalty'])
def test_numerical_verdict_rejects_token_only_coincidence(event):
    p = probe()
    assert hasattr(p, 'alignment_verdict'), 'missing numerical alignment verdict'
    off, on = complete_trace(), complete_trace(True)
    on.steps[1]['events'][event]['value'][0, 1] = 3.
    verdict = p.alignment_verdict(off, on, result_tokens(), result_tokens(), expected_steps=3)
    assert verdict['status'] == 'fail' and verdict['tokens_equal'] is True
    assert verdict['q_equal' if event.startswith('layer') else 'logits_equal'] is False


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_numerical_verdict_rejects_nonfinite_even_when_both_sides_match(value):
    p = probe()
    assert hasattr(p, 'alignment_verdict'), 'missing numerical alignment verdict'
    off, on = complete_trace(), complete_trace(True)
    for trace in [off, on]:
        trace.steps[0]['events']['logits.raw']['value'][0, 0] = value
    verdict = p.alignment_verdict(off, on, result_tokens(), result_tokens(), expected_steps=3)
    assert verdict['status'] == 'fail' and verdict['finite'] is False


@pytest.mark.parametrize('missing', ['event', 'step', 'result', 'history', 'geometry'])
def test_numerical_verdict_is_inconclusive_when_evidence_is_missing(missing):
    p = probe()
    assert hasattr(p, 'alignment_verdict'), 'missing numerical alignment verdict'
    off, on, result = complete_trace(), complete_trace(True), result_tokens()
    if missing == 'event':
        del on.steps[1]['events']['logits.raw']
    elif missing == 'step':
        on.steps.pop()
    elif missing == 'result':
        del result['generated_suffix_token_ids']
    elif missing == 'history':
        on.steps[1]['history'] = None
    else:
        del on.steps[1]['decoder']['past_length']
    verdict = p.alignment_verdict(off, on, result_tokens(), result, expected_steps=3)
    assert verdict['status'] == 'inconclusive' and verdict['coverage_complete'] is False


def test_unequal_history_child_is_excluded_from_comparison_and_coverage():
    p = probe()
    off, on = complete_trace(), complete_trace(True)
    on.steps[1]['history'] = [7, 8, 99]
    # A huge child difference cannot count as a matched-history numerical failure.
    on.steps[1]['events']['layer.0.q_proj']['value'][0, 0] = 1000.
    assert p.compare_traces(off, on)['compared_steps'] == 1
    assert hasattr(p, 'alignment_verdict'), 'missing numerical alignment verdict'
    verdict = p.alignment_verdict(off, on, result_tokens(), result_tokens(), expected_steps=3)
    assert verdict['status'] == 'inconclusive' and verdict['coverage_complete'] is False
    assert verdict['compared_steps'] == 1


def test_numerical_verdict_detects_token_difference_and_wrong_cache_geometry():
    p = probe()
    assert hasattr(p, 'alignment_verdict'), 'missing numerical alignment verdict'
    off, on, result = complete_trace(), complete_trace(True), result_tokens()
    result['generated_suffix_token_ids'][2] = 99
    assert p.alignment_verdict(off, on, result_tokens(), result, expected_steps=3)['status'] == 'fail'
    on.steps[1]['decoder']['query_length'] = 3  # caching must really decode one token
    verdict = p.alignment_verdict(off, on, result_tokens(), result_tokens(), expected_steps=3)
    assert verdict['status'] == 'fail' and verdict['cache_geometry_expected'] is False


def test_fresh_fidelity_preserves_normal_eos_and_rejects_changed_stop_semantics():
    p = probe()
    result = dict(result_tokens(2), stop_reason='eos')
    assert p.require_fidelity(result, result, expected_steps=None) is None
    with pytest.raises(ValueError, match='observer fidelity'):
        p.require_fidelity(dict(result, stop_reason='max_new_tokens'), result, expected_steps=None)


@pytest.mark.parametrize('index,image', [(0,'12035.png'), (3,'12044.png'), (12,'12069.png')])
def test_probe_selects_requested_source_and_resolves_image(index, image):
    from pathlib import Path
    sample = probe().selected_sample(index)
    assert Path(sample['image_input'][0]).name == image
    assert Path(sample['image_input'][0]).is_file()
    if index == 12:
        assert sample == probe().selected_sample()


@pytest.mark.parametrize('policy', ['tf32_off','full_accumulation','math_sdpa'])
def test_native_trace_policy_observes_real_operand_dtypes_and_restores(policy, monkeypatch):
    from tests.stage2.test_eval_integration import _import_entry_point, FakeProcessor
    module, p = _import_entry_point('eval', monkeypatch), probe()
    torch.manual_seed(12)
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    torch.set_float32_matmul_precision('medium')
    before = state()
    for cached in [False, True]:
        with p.PassiveTrace(model, max_steps=3) as trace:
            result = module.run_one_example(model, FakeProcessor(include_visual_inputs=False),
                {'text_input':'question'}, dict(max_new_tokens=3, repetition_penalty=1.05),
                disable_latent_generation=True, use_kv_cache=cached, numerical_policy=policy)
        assert state() == before
        event = trace.steps[0]['events']['layer.0.q_proj']
        assert event['operand_dtype'] == 'torch.float32'
        assert event['output_dtype'] == 'torch.float32'
        assert len(event['operand_stride']) == 3 and event['operand_stride'][-1] == 1
        assert result['numerical_observation']['policy'] == policy


def test_probe_fresh_controls_run_before_observers_with_same_real_evaluator(tmp_path, monkeypatch):
    from tests.stage2.test_eval_integration import _import_entry_point, FakeProcessor
    module, p = _import_entry_point('eval', monkeypatch), probe()
    torch.manual_seed(12)
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    monkeypatch.setattr(module, 'load_evaluation_artifacts', lambda **_kw: (model, FakeProcessor(include_visual_inputs=False)))
    monkeypatch.setattr(p, 'selected_sample', lambda index=12: {'text_input':'question'})
    starts = []
    def observe_start(_module, _args, kwargs):
        if kwargs['inputs_embeds'].shape[1] == 2 and kwargs.get('past_key_values') is None:
            starts.append(bool(model.model.layers[0].self_attn.q_proj._forward_hooks))
    hook = model.model.register_forward_pre_hook(observe_start, with_kwargs=True)
    monkeypatch.chdir(tmp_path)
    p.main(['--output-dir','results/fresh','--source-index','0', '--fresh-controls',
            '--numerical-policy','full_accumulation'])
    import json
    report = json.loads((tmp_path/'results/fresh/report.json').read_text())
    hook.remove()
    assert starts == [False, False, True, True]
    assert report['observer_fidelity'] == [True, True]
    assert len(report['fresh_controls']) == len(report['results']) == 2
    assert all(r['numerical_policy'] == 'full_accumulation' for r in report['fresh_controls']+report['results'])
    for result, control in zip(report['results'], report['fresh_controls']):
        assert result['generated_suffix_token_ids'] == control['generated_suffix_token_ids']
    assert report['source_index'] == 0
    assert report['max_new_tokens'] == 32


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_tensor_difference_remains_json_serializable(value):
    import json
    difference = probe().tensor_difference(torch.tensor([value]), torch.tensor([0.]))
    assert difference['finite'] is False
    json.dumps(difference, allow_nan=False)


def test_comparison_report_preserves_observed_outer_operand_dtype_and_layout():
    p = probe()
    torch.manual_seed(12)
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    with p.PassiveTrace(model, max_steps=3) as off:
        generate(model, False)
    with p.PassiveTrace(model, max_steps=3) as on:
        generate(model, True)
    report = p.compare_traces(off, on)
    row = next(r for r in report['comparisons'] if r['operation'] == 'layer.0.q_proj')
    assert row['outer_operands_off']['dtype'] == 'torch.float32'
    assert row['outer_operands_on']['dtype'] == 'torch.float32'
    assert row['outer_operands_off']['shape'] == [1, 8, 8]
    assert row['outer_operands_off']['stride'][-1] == 1


def test_nonfinite_comparison_report_is_json_safe_without_hiding_finite_failure():
    import json
    p = probe()
    off, on = complete_trace(), complete_trace(True)
    on.steps[0]['events']['logits.raw']['value'][0, 0] = float('inf')
    report = p.compare_traces(off, on)
    json.dumps(report, allow_nan=False)
    verdict = p.alignment_verdict(off, on, result_tokens(), result_tokens(), expected_steps=3)
    assert verdict['finite'] is False and verdict['status'] == 'fail'


def test_replay_uses_requested_policy_records_active_controls_and_preserves_layout_uncertainty():
    # A replay outside the evaluator scope must not silently revert to medium/TF32.
    p = probe()
    linear = torch.nn.Linear(2, 2, bias=False)
    observed = []
    handle = linear.register_forward_pre_hook(lambda _m, _a: observed.append(state()))
    torch.set_float32_matmul_precision('medium')
    before = state()
    off, on = complete_trace(), complete_trace(True)
    left = off.steps[0]['events']['layer.0.q_proj']
    right = on.steps[0]['events']['layer.0.q_proj']
    left.update(module=linear, operand=torch.ones(1, 2, 2))
    right.update(module=linear, operand=torch.ones(1, 1, 2))
    right['value'][0, 0] = 99.
    try:
        report = p.compare_traces(off, on, numerical_policy='full_accumulation')
    finally:
        handle.remove()
    assert state() == before
    assert observed and all(s[0] == 'highest' and s[1] is False and s[2] is False for s in observed)
    assert report['replay']['numerical_observation']['policy'] == 'full_accumulation'
    assert report['replay']['numerical_observation']['float32_matmul_precision'] == 'highest'
    assert 'layout' in report['replay_limitation']


def tiny_dora():
    from peft.tuners.lora.layer import Linear
    torch.manual_seed(17)
    module = Linear(torch.nn.Linear(4, 6, bias=False), 'default', r=2,
                    lora_alpha=4, use_dora=True).eval()
    with torch.no_grad():
        module.lora_B['default'].weight.fill_(.2)
    return module


def test_mlp_components_keep_real_calls_roles_layout_state_and_output_identity():
    # Dropping earlier A/B calls, casting operands, or replacing outputs loses evidence.
    p, module = probe(), tiny_dora()
    assert hasattr(p, 'MLPComponents'), 'missing actual component observer'
    x = torch.arange(24.).reshape(1, 3, 8)[..., 1::2]
    returned = []
    h = module.register_forward_hook(lambda _m, _a, out: returned.append(out))
    with torch.inference_mode():
        expected = module(x)
        with p.MLPComponents(module) as observer:
            actual = module(x)
    h.remove()
    assert actual is returned[-1] and torch.equal(actual, expected)
    assert observer.operand is x
    assert [(e['component'], e['role']) for e in observer.events] == [
        ('outer', 'token'), ('base', 'token'), ('dora', 'token'),
        ('lora_A', 'identity_reconstruction'), ('lora_B', 'identity_reconstruction'),
        ('lora_A', 'token'), ('lora_B', 'token')]
    assert [e['start_order'] for e in observer.events] == [0, 1, 3, 4, 6, 8, 10]
    outer = observer.events[0]
    assert outer['operand']['stride'] == [24, 8, 2]
    assert outer['operand']['storage_offset'] == 1
    assert outer['operand']['dtype'] == outer['output']['dtype'] == 'torch.float32'
    assert outer['state']['active_adapters'] == ['default']
    assert outer['state']['training'] is False
    assert outer['state']['autocast_enabled'] is False
    assert 'Linear.forward' in outer['state']['forward']['qualname']
    assert outer['state']['forward']['bound_to_module'] is True
    assert outer['state']['parameters']['base_layer.weight']['dtype'] == 'torch.float32'
    assert observer.events[2]['end_order'] > observer.events[-1]['end_order']
    assert not module._forward_pre_hooks and not module._forward_hooks


def test_mlp_component_hooks_restore_on_real_forward_exception():
    # An exception in an actual component must not leave observer hooks installed.
    p, module = probe(), tiny_dora()
    assert hasattr(p, 'MLPComponents'), 'missing actual component observer'
    before = [(len(m._forward_hooks), len(m._forward_pre_hooks), m.forward) for m in module.modules()]
    with pytest.raises(RuntimeError):
        with p.MLPComponents(module):
            module(torch.ones(1, 2, 3))  # actual incompatible base GEMM
    assert before == [(len(m._forward_hooks), len(m._forward_pre_hooks), m.forward) for m in module.modules()]


@pytest.mark.parametrize('damage', ['none', 'mutation', 'missing', 'ambiguous', 'unequal', 'replay'])
def test_mlp_component_attribution_requires_both_original_replays_and_complete_evidence(damage):
    # Mutation/missing roles/input differences/nonreproducing output must veto attribution.
    p, module = probe(), tiny_dora()
    assert hasattr(p, 'MLPComponents'), 'missing actual component observer'
    assert hasattr(p, 'compare_mlp_components'), 'missing evidence-gated component comparison'
    x = torch.arange(12.).reshape(1, 3, 4)
    with torch.inference_mode():
        with p.MLPComponents(module) as off:
            module(x)
        with p.MLPComponents(module) as on:
            module(x[..., -1:, :])
    if damage == 'mutation':
        x.add_(1)
    elif damage == 'missing':
        on.events.pop()
    elif damage == 'ambiguous':
        on.events[-1]['role'] = 'ambiguous'
    elif damage == 'unequal':
        with torch.inference_mode():
            with p.MLPComponents(module) as on:
                module(torch.zeros(1, 1, 4))
    elif damage == 'replay':
        with torch.inference_mode():
            on.output.add_(1)
    before_count = [len(off.events), len(on.events)]
    report = p.compare_mlp_components(off, on, numerical_policy='full_accumulation')
    assert [len(off.events), len(on.events)] == before_count  # replay after hooks removed
    assert report['status'] == ('observed' if damage == 'none' else 'inconclusive')
    if damage != 'none':
        assert report['invocation_geometry_layout_sensitivity'] is False
    assert report['internal_accumulator_dtype'] == report['cuda_kernel_identity'] == 'unobserved'
    assert report['internal_normalization_correction_add_cast'] == 'unobserved'
    if damage == 'none':
        assert all(report['replay'][side]['matches_observed'] for side in ('off', 'on'))
        assert all(report['replay'][side]['numerical_observation']['policy'] == 'full_accumulation'
                   for side in ('off', 'on'))
        assert report['observed']['newest_input_difference']['equal']
        assert report['replay']['off']['operand_unchanged']
        # Real CPU GEMMs may themselves differ by invocation geometry.
        differ = not torch.equal(off.output, on.output)
        assert report['invocation_geometry_layout_sensitivity'] == differ
        assert (report['observed']['earliest_differing_boundary'] is not None) == differ
    import json
    json.dumps(report, allow_nan=False)


def test_mlp_components_native_trace_targets_only_second_prediction_layer24():
    # Attaching to every step/layer would capture unrelated adapter calls.
    p = probe()
    import inspect
    assert 'mlp_components' in inspect.signature(p.PassiveTrace).parameters, 'missing opt-in native target gate'
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    from copy import deepcopy
    model.model.layers.extend([deepcopy(model.model.layers[0]) for _ in range(24)])
    for index, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = index
    model.model.layers[24].mlp.up_proj = tiny_dora_native = __import__(
        'peft.tuners.lora.layer', fromlist=['Linear']).Linear(
            model.model.layers[24].mlp.up_proj, 'default', r=2, lora_alpha=4, use_dora=True).eval()
    with p.PassiveTrace(model, max_steps=3, mlp_components=True) as trace:
        generate(model, False)
    assert trace.mlp_components.operand.shape == (1, 9, 8)
    assert len(trace.mlp_components.events) == 7
    assert trace.mlp_components.module is tiny_dora_native


@pytest.mark.parametrize('extra', [[], ['--fresh-controls'],
    ['--fresh-controls', '--numerical-policy', 'full_accumulation', '--source-index', '0']])
def test_mlp_components_cli_rejects_unapproved_profile_before_loading(extra, tmp_path, monkeypatch, capsys):
    # Accepting an unsupported flag combination must fail before selecting data or loading models.
    p = probe()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        p.main(['--output-dir', 'results/forbidden', '--mlp-components']+extra)
    assert error.value.code == 2
    assert '--mlp-components requires' in capsys.readouterr().err
    assert not (tmp_path/'results').exists()


def test_mlp_component_report_opt_in_preserves_controls_and_alignment(tmp_path, monkeypatch):
    # A missing report integration or implicit default diagnostic breaks the opt-in contract.
    from tests.stage2.test_eval_integration import _import_entry_point, FakeProcessor
    import json
    evaluation, p = _import_entry_point('eval', monkeypatch), probe()
    import inspect
    assert 'mlp_components' in inspect.signature(p.PassiveTrace).parameters, 'missing opt-in diagnostic integration'
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    monkeypatch.setattr(evaluation, 'load_evaluation_artifacts', lambda **_kw: (model, FakeProcessor(include_visual_inputs=False)))
    monkeypatch.setattr(p, 'selected_sample', lambda index=12: {'text_input': 'question'})
    monkeypatch.chdir(tmp_path)
    for flag, path in [([], 'default'), (['--mlp-components'], 'components')]:
        p.main(['--output-dir', 'results/'+path, '--fresh-controls',
                '--numerical-policy', 'full_accumulation']+flag)
    default = json.loads((tmp_path/'results/default/report.json').read_text())
    report = json.loads((tmp_path/'results/components/report.json').read_text())
    assert 'mlp_components' not in default
    assert report['mlp_components']['status'] == 'inconclusive'  # no layer24 in tiny native fixture
    assert report['alignment_verdict'] == default['alignment_verdict']
    for actual, reference in zip(report['fresh_controls'], default['fresh_controls']):
        p.require_fidelity(actual, reference, expected_steps=None)


def test_mlp_component_roles_publish_actual_operand_evidence():
    # Labels alone would hide an assumption that an A/B call used identity input.
    p, module = probe(), tiny_dora()
    with torch.inference_mode():
        with p.MLPComponents(module) as trace:
            module(torch.ones(1, 3, 4))
    identity_a, identity_b, token_a, token_b = trace.events[3:]
    assert identity_a.get('role_evidence') == dict(identity_operand=True, same_as_dora_operand=False)
    assert token_a.get('role_evidence') == dict(identity_operand=False, same_as_dora_operand=True)
    assert identity_b.get('role_evidence') == dict(same_as_preceding_a_output=True,
                                                 preceding_a_start_order=4)
    assert token_b.get('role_evidence') == dict(same_as_preceding_a_output=True,
                                              preceding_a_start_order=8)


def test_mlp_components_require_full_32_token_fidelity_before_attribution(tmp_path, monkeypatch):
    # Matching early stopping must not silently admit a shorter source12 attribution.
    from tests.stage2.test_eval_integration import _import_entry_point, FakeProcessor
    evaluation, p = _import_entry_point('eval', monkeypatch), probe()
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    monkeypatch.setattr(evaluation, 'load_evaluation_artifacts', lambda **_kw: (model, FakeProcessor(include_visual_inputs=False)))
    monkeypatch.setattr(p, 'selected_sample', lambda index=12: {'text_input': 'question'})
    original = evaluation.run_one_example
    def short_run(model, processor, sample, profile, **kwargs):
        return original(model, processor, sample, dict(profile, max_new_tokens=1), **kwargs)
    monkeypatch.setattr(evaluation, 'run_one_example', short_run)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='full 32-token suffix'):
        p.main(['--output-dir', 'results/short', '--fresh-controls', '--mlp-components',
                '--numerical-policy', 'full_accumulation'])


def retained_pair():
    """Actual installed DoRA calls with original full-row operands."""
    p, module = probe(), tiny_dora()
    # Integer/power-of-two weights make equality independent of CPU GEMM shapes.
    with torch.no_grad():
        module.lora_A['default'].weight.fill_(.25)
        module.lora_B['default'].weight.fill_(.5)
    x = torch.arange(12.).reshape(1, 3, 4)
    with torch.inference_mode():
        with p.MLPComponents(module, retain_adapter_operands=True) as off:
            module(x)
        with p.MLPComponents(module, retain_adapter_operands=True) as on:
            module(x[..., -1:, :])
    return off, on


def test_precheck_replays_actual_original_A_B_before_fp64(monkeypatch):
    # Catches shadow arithmetic before four original-operand positive controls.
    p = probe()
    assert hasattr(p, 'fp64_adapter_precheck'), 'missing operand precheck'
    off, on = retained_pair()
    calls = []
    native = F.linear
    def observe(x, weight, bias=None):
        calls.append(x.dtype)
        return native(x, weight, bias)
    monkeypatch.setattr(F, 'linear', observe)
    report = p._fp64_operand_precheck(off, on, baseline_fidelity=[True, True])
    assert report['status'] == 'pass'
    assert calls[:4] == [torch.float32]*4
    assert report['native_replays_match'] == {'A': [True, True], 'B': [True, True]}
    assert report['input_difference']['equal'] is True
    assert report['protected_A_difference']['equal'] is True
    assert report['A64_B64_difference']['equal'] is True
    assert report['temporary_operand_dtypes']['A']['input'] == 'torch.float64'
    import json
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('damage', ['mutation', 'missing', 'nan', 'unequal', 'baseline', 'unreproduced'])
def test_precheck_requires_equal_actual_A_inputs_and_finite_reproduced_baseline(damage):
    p = probe()
    assert hasattr(p, 'fp64_adapter_precheck'), 'missing operand precheck'
    off, on = retained_pair()
    fidelity = [True, True]
    if damage == 'mutation':
        with torch.no_grad():
            off.events[5]['_operand'].add_(1)
    elif damage == 'missing':
        on.events.pop()
    elif damage == 'nan':
        with torch.no_grad():
            on.events[5]['_operand'].fill_(float('nan'))
    elif damage == 'unequal':
        with torch.no_grad():
            on.events[5]['_operand'].add_(1)
        # Honest capture of an actually different original operand.
        on.events[5]['input_capture_sha256'] = p.tensor_record(on.events[5]['_operand'])['sha256']
    elif damage == 'baseline':
        fidelity[1] = False
    else:
        with torch.no_grad():
            off.module.lora_A['default'].weight.add_(1)
    report = p._fp64_operand_precheck(off, on, baseline_fidelity=fidelity)
    assert report['status'] == 'inconclusive'


def ordered_math_report(*, layer_count=1, target_layer=0):
    """Literal native boundary order with a real tensor difference at step1 SDPA."""
    from types import SimpleNamespace
    p = probe()
    off, on = complete_trace(count=2), complete_trace(True, count=2)
    for trace in (off, on):
        trace.model = SimpleNamespace(model=SimpleNamespace(layers=[None]*layer_count))
        for i, step in enumerate(trace.steps):
            values = {name: torch.ones(1, 1, 2) for name in
                      ('decoder.embedding', 'decoder.mask2d', 'decoder.mrope')}
            for layer in range(layer_count):
                # Hand-written actual hook order, independent of production gate helpers.
                for name in ('input', 'cache_position', 'input_norm', 'q_proj', 'k_proj', 'v_proj'):
                    values[f'layer.{layer}.{name}'] = torch.ones(1, 1, 2)
                values[f'layer.{layer}.sdpa.q'] = torch.ones(1, 1, 1, 2)
                values[f'layer.{layer}.sdpa.k'] = torch.ones(1, 1, 2+i, 2)
                values[f'layer.{layer}.sdpa.v'] = torch.ones(1, 1, 2+i, 2)
                values[f'layer.{layer}.sdpa.visibility'] = torch.ones(2+i, dtype=torch.bool)
                values[f'layer.{layer}.sdpa.output'] = torch.ones(1, 1, 1, 2)
                for name in ('o_proj', 'post_attention_norm.input_residual', 'post_attention_norm',
                             'mlp_gate', 'mlp_up', 'mlp_down', 'mlp', 'output'):
                    values[f'layer.{layer}.{name}'] = torch.ones(1, 1, 2)
            for name in ('decoder.final_norm', 'logits.raw', 'logits.post_penalty'):
                values[name] = torch.ones(1, 1, 5 if name.startswith('logits.') else 2)
            step['events'] = {name: dict(value=value, shape=list(value.shape)) for name, value in values.items()}
    on.steps[1]['events'][f'layer.{target_layer}.sdpa.output']['value'].add_(1)
    return p.compare_traces(off, on, numerical_policy='adapter_token_fp64')


@pytest.mark.parametrize('damage', ['earlier_step', 'earlier_layer', 'earlier_projection',
    'earlier_step_projection', 'reordered_projection', 'duplicate_projection', 'layer_counts'])
def test_math_gate_rejects_missing_or_unordered_earlier_numeric_coverage(damage):
    p, report = probe(), ordered_math_report(layer_count=2, target_layer=1)
    assert p.math_sdpa_eligible(report) is True
    if damage == 'earlier_step':
        report['comparisons'] = [row for row in report['comparisons'] if row['step'] != 0]
    elif damage == 'earlier_layer':
        report['comparisons'] = [row for row in report['comparisons']
            if not (row['step'] == 1 and row['operation'].startswith('layer.0.'))]
    elif damage in ('earlier_projection', 'earlier_step_projection'):
        damaged_step = 0 if damage == 'earlier_step_projection' else 1
        report['comparisons'] = [row for row in report['comparisons']
            if (row['step'], row['operation']) != (damaged_step, 'layer.0.q_proj')]
    elif damage in ('reordered_projection', 'duplicate_projection'):
        index = next(i for i, row in enumerate(report['comparisons'])
                     if (row['step'], row['operation']) == (1, 'layer.0.q_proj'))
        if damage == 'reordered_projection':
            report['comparisons'][index:index+2] = reversed(report['comparisons'][index:index+2])
        else:
            report['comparisons'].insert(index, report['comparisons'][index])
    else:
        report.pop('decoder_layer_counts', None)
    assert p.math_sdpa_eligible(report) is False


def test_math_gate_requires_first_sdpa_output_and_equal_operands():
    # Ordered real tensor comparisons, not a trusted first-difference label.
    p, report = probe(), ordered_math_report()
    assert p.math_sdpa_eligible(report) is True
    from copy import deepcopy
    for operation in ('layer.0.q_proj', 'layer.0.sdpa.k', 'layer.0.sdpa.v', 'layer.0.sdpa.visibility'):
        damaged = deepcopy(report)
        row = next(r for r in damaged['comparisons'] if r['step'] == 1 and r['operation'] == operation)
        row['difference']['equal'] = False
        assert p.math_sdpa_eligible(damaged) is False
    for damage in ('missing', 'nonfinite', 'geometry', 'declared_only'):
        damaged = deepcopy(report)
        if damage == 'missing':
            damaged['comparisons'] = [r for r in damaged['comparisons'] if r['operation'] != 'layer.0.sdpa.v']
        elif damage == 'nonfinite':
            damaged['comparisons'][0]['difference']['finite'] = False
        elif damage == 'geometry':
            damaged['decoder_on'][1]['query_length'] = 3
        else:
            damaged['comparisons'] = []
        assert p.math_sdpa_eligible(damaged) is False


def test_candidate_requires_full32_selected_A_B_Q_and_logits():
    p = probe()
    off, on = complete_trace(count=32), complete_trace(True, count=32)
    import inspect
    assert 'numerical_policy' in inspect.signature(p.alignment_verdict).parameters, 'missing candidate component gate'
    assert p.alignment_verdict(off, on, result_tokens(32), result_tokens(32), expected_steps=32,
        numerical_policy='adapter_token_fp64')['status'] == 'inconclusive'
    a, b = retained_pair()
    components = p.compare_mlp_components(a, b, numerical_policy='full_accumulation')
    assert p.alignment_verdict(off, on, result_tokens(32), result_tokens(32), expected_steps=32,
        numerical_policy='adapter_token_fp64', component_report=components)['status'] == 'pass'
    next(row for row in components['observed']['boundaries']
         if row['component'] == 'lora_B' and row['role'] == 'token')['difference']['equal'] = False
    assert p.alignment_verdict(off, on, result_tokens(32), result_tokens(32), expected_steps=32,
        numerical_policy='adapter_token_fp64', component_report=components)['status'] == 'fail'
    assert p.alignment_verdict(complete_trace(), complete_trace(True), result_tokens(), result_tokens(), expected_steps=32,
        numerical_policy='adapter_token_fp64', component_report=components)['status'] != 'pass'


def baseline_native_fixture(monkeypatch):
    from copy import deepcopy
    from peft.tuners.lora.layer import Linear
    from tests.stage2.test_eval_integration import _import_entry_point, FakeProcessor
    evaluation, p = _import_entry_point('eval', monkeypatch), probe()
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    model.model.layers.extend([deepcopy(model.model.layers[0]) for _ in range(24)])
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = i
    layer = model.model.layers[24].mlp
    layer.up_proj = Linear(layer.up_proj, 'default', r=2, lora_alpha=4, use_dora=True).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        layer.up_proj.base_layer.weight.fill_(.125)
        layer.up_proj.lora_A['default'].weight.fill_(.25)
        layer.up_proj.lora_B['default'].weight.fill_(.5)
        layer.up_proj.lora_magnitude_vector['default'].weight.fill_(1.)
    processor, sample = FakeProcessor(include_visual_inputs=False), {'text_input': 'question'}
    traces, results = [], []
    for cached in (False, True):
        with p.PassiveTrace(model, max_steps=32, mlp_components=True) as trace:
            results.append(evaluation.run_one_example(model, processor, sample, p.PROFILE,
                disable_latent_generation=True, use_kv_cache=cached, numerical_policy='full_accumulation'))
        traces.append(trace)
    baseline = dict(model=p.MODEL, base=p.BASE, source=p.SOURCE, source_index=12, profile=p.PROFILE,
        numerical_policy='full_accumulation', observer_fidelity=[True, True], results=results,
        mlp_components=p.compare_mlp_components(*(trace.mlp_components for trace in traces), numerical_policy='full_accumulation'))
    return evaluation, model, processor, sample, baseline


def test_precheck_native_two_token_baseline_reproduces_saved_full_witness(monkeypatch):
    p = probe()
    assert hasattr(p, 'fp64_adapter_precheck'), 'missing native local precheck'
    evaluation, model, processor, sample, baseline = baseline_native_fixture(monkeypatch)
    starts = []
    original = evaluation.run_one_example
    def observe(*args, **kwargs):
        starts.append((args[3]['max_new_tokens'], kwargs['numerical_policy']))
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluation, 'run_one_example', observe)
    report = p.fp64_adapter_precheck(model, processor, sample, baseline_report=baseline)
    assert report['status'] == 'pass'
    assert report['baseline_fidelity'] == [True, True]
    assert starts == [(2, 'full_accumulation')]*4
    assert report['profile'] == dict(p.PROFILE, max_new_tokens=2)
    assert report['native_replays_match'] == {'A': [True, True], 'B': [True, True]}
    assert not model.model.layers[24].mlp.up_proj.lora_A['default']._forward_hooks
    from copy import deepcopy
    for key in ('model', 'profile', 'results', 'mlp_components', 'observed'):
        damaged = deepcopy(baseline)
        if key == 'observed':
            damaged['mlp_components']['observed'] = None
        else:
            damaged[key] = None
        assert p.fp64_adapter_precheck(model, processor, sample, baseline_report=damaged)['status'] == 'inconclusive'


def test_precheck_failure_skips_full_candidate_without_faking_alignment(tmp_path, monkeypatch):
    import json
    from tests.stage2.test_eval_integration import _import_entry_point, FakeProcessor
    from tests.evaluation.test_off_kv_cache import adapter_model
    evaluation, p = _import_entry_point('eval', monkeypatch), probe()
    assert hasattr(p, 'FP64_BASELINE_REPORT'), 'missing baseline gate'
    model, processor = adapter_model(), FakeProcessor(include_visual_inputs=False)
    baseline = tmp_path/'damaged-baseline.json'
    baseline.write_text(json.dumps({'model': 'changed'}))
    monkeypatch.setattr(p, 'FP64_BASELINE_REPORT', baseline)
    monkeypatch.setattr(p, 'selected_sample', lambda index=12: {'text_input': 'question'})
    monkeypatch.setattr(evaluation, 'load_evaluation_artifacts', lambda **_kw: (model, processor))
    monkeypatch.chdir(tmp_path)
    p.main(['--output-dir', 'results/rejected', '--fresh-controls', '--source-index', '12',
            '--numerical-policy', 'adapter_token_fp64', '--mlp-components'])
    report = json.loads((tmp_path/'results/rejected/report.json').read_text())
    assert report['precheck']['status'] == 'inconclusive'
    assert report['status'] == 'inconclusive'
    assert report['fresh_controls'] == report['results'] == []
    assert report['alignment_verdict']['status'] == 'inconclusive'
    assert report['alignment_verdict']['executed'] is False


def test_precheck_reports_finite_protected_difference_as_failure():
    # Separate real modules supply a literal adversarial protected-output witness;
    # this exercises the operand comparator, never claims a GPU/model correction.
    p = probe()
    assert hasattr(p, '_fp64_operand_precheck'), 'missing protected-output decision'
    off, _ = retained_pair()
    other = tiny_dora()
    with torch.no_grad():
        other.lora_A['default'].weight.fill_(.5)
        other.lora_B['default'].weight.fill_(.5)
    with torch.inference_mode(), p.MLPComponents(other, retain_adapter_operands=True) as on:
        other(torch.arange(12.).reshape(1, 3, 4)[..., -1:, :])
    report = p._fp64_operand_precheck(off, on, baseline_fidelity=[True, True])
    assert report['status'] == 'fail'
    assert report['native_replays_match'] == {'A': [True, True], 'B': [True, True]}
    assert report['protected_A_difference']['equal'] is False


@pytest.mark.parametrize('policy', ['adapter_token_fp64', 'adapter_token_fp64_math_sdpa'])
def test_new_profile_real_probe_controls_observation_and_replay_use_same_scope(policy, tmp_path, monkeypatch):
    import json
    from tests.stage2.test_eval_integration import _import_entry_point, FakeProcessor
    from tests.evaluation.test_off_kv_cache import adapter_model
    from tests.evaluation.test_adapter_numerics import bindings, assert_bindings
    evaluation, p = _import_entry_point('eval', monkeypatch), probe()
    model, processor = adapter_model(), FakeProcessor(include_visual_inputs=False)
    saved = bindings(model)
    monkeypatch.setattr(evaluation, 'load_evaluation_artifacts', lambda **_kw: (model, processor))
    monkeypatch.setattr(p, 'selected_sample', lambda index=12: {'text_input': 'question'})
    monkeypatch.chdir(tmp_path)
    p.main(['--output-dir', 'results/new-real', '--fresh-controls', '--source-index', '0',
            '--numerical-policy', policy])
    report = json.loads((tmp_path/'results/new-real/report.json').read_text())
    assert report['observer_fidelity'] == [True, True]
    assert report['decoder_layer_counts'] == [1, 1]
    for row in report['fresh_controls']+report['results']:
        arithmetic = row['numerical_observation']['adapter_token_arithmetic']
        assert arithmetic['token_calls'] == 2*len(row['generated_suffix_token_ids'])
        assert arithmetic['actual_operand_dtypes']['input'] == 'torch.float64'
    assert report['decoder_off'][1]['query_length'] == 3
    assert report['decoder_off'][1]['use_cache'] is False
    assert report['decoder_off'][1]['returned_cache_length'] is None
    assert report['decoder_on'][1]['query_length'] == 1
    assert report['decoder_on'][1]['use_cache'] is True
    assert report['decoder_on'][1]['past_length'] == 2
    assert_bindings(saved)


def mixed_dtype_retained_pair():
    """Actual installed PEFT casts BF16 outer inputs to FP32 before DoRA."""
    from peft.tuners.lora.layer import Linear
    p = probe()
    module = Linear(torch.nn.Linear(4, 6, bias=False).bfloat16(), 'default',
        r=2, lora_alpha=4, use_dora=True).eval()
    for collection in (module.lora_A, module.lora_B, module.lora_magnitude_vector):
        collection['default'].float()
    with torch.no_grad():
        module.lora_A['default'].weight.fill_(.25)
        module.lora_B['default'].weight.fill_(.5)
    x = torch.arange(12., dtype=torch.bfloat16).reshape(1, 3, 4)
    actual_dora_inputs = []
    hook = module.lora_magnitude_vector['default'].register_forward_pre_hook(
        lambda _m, args: actual_dora_inputs.append(args[0]))
    try:
        with torch.inference_mode():
            with p.MLPComponents(module, retain_adapter_operands=True) as off:
                module(x)
            with p.MLPComponents(module, retain_adapter_operands=True) as on:
                module(x[..., -1:, :])
    finally:
        hook.remove()
    return off, on, actual_dora_inputs


def test_precheck_uses_actual_mixed_dtype_dora_operand(monkeypatch):
    p = probe()
    off, on, actual_dora_inputs = mixed_dtype_retained_pair()
    for trace, actual in zip((off, on), actual_dora_inputs):
        assert trace.operand.dtype == torch.bfloat16
        assert trace.events[1]['operand']['dtype'] == 'torch.bfloat16'
        assert actual.dtype == torch.float32
        assert trace.events[5]['_operand'] is actual
        assert actual is not trace.operand
        assert torch.equal(actual, trace.operand.float())
    before = [(parameter, parameter.detach().clone(), parameter.data_ptr(), parameter.dtype)
              for parameter in off.module.parameters()]
    delegated = []
    native = F.linear
    def observe(operand, weight, bias=None):
        output = native(operand, weight, bias)
        delegated.append((operand.dtype, weight.dtype, output.dtype))
        return output
    monkeypatch.setattr(F, 'linear', observe)
    report = p._fp64_operand_precheck(off, on, baseline_fidelity=[True, True])
    assert report['status'] == 'pass', report
    assert report['native_replays_match'] == {'A': [True, True], 'B': [True, True]}
    assert delegated[:4] == [(torch.float32, torch.float32, torch.float32)]*4
    assert report['input_difference']['equal'] is True
    assert report['protected_A_difference']['equal'] is True
    assert report['A64_B64_difference']['equal'] is True
    for trace, actual in zip((off, on), actual_dora_inputs):
        assert trace.events[2]['_operand'] is actual
        assert trace.operand.dtype == torch.bfloat16
    assert all(torch.equal(parameter, original) and parameter.data_ptr() == pointer and parameter.dtype == dtype
               for parameter, original, pointer, dtype in before)


@pytest.mark.parametrize('damage', ['missing_DoRA', 'replaced_DoRA', 'replaced_B'])
def test_mixed_dtype_precheck_rejects_missing_or_replaced_actual_linkage(damage):
    p = probe()
    off, on, actual_dora_inputs = mixed_dtype_retained_pair()
    if damage == 'missing_DoRA':
        off.events[2].pop('_operand', None)
    elif damage == 'replaced_DoRA':
        off.events[2]['_operand'] = actual_dora_inputs[0].clone()
    else:
        off.events[6]['_operand'] = off.events[6]['_operand'].clone()
    report = p._fp64_operand_precheck(off, on, baseline_fidelity=[True, True])
    assert report['status'] == 'inconclusive'
    assert report['native_replays_match'] == {'A': [False, False], 'B': [False, False]}
