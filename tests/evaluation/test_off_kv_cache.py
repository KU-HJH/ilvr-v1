"""Runtime witnesses for the opt-in Stage 1 OFF cache execution path."""
import json
import sys

import pytest
import torch
from torch import nn

from evaluation.continuous_latent import (
    ContinuousLatentController, ContinuousLatentGenerationConfig,
)
from evaluation.stage1_v5_causal import pair_generation
from tests.evaluation.test_continuous_latent import TOKENS, scripted_model
from tests.evaluation.test_stage1_v5_causal import row
from tests.multinode.test_evaluation import _write_fake_shards, _merge
from tests.stage2.helpers import tiny_stage2_qwen
from tests.stage2.test_eval_integration import (
    _import_entry_point, RecordingStage1Model, RecordingController,
    FakeProcessor, ModelWhoseGenerateRaises,
)


class ObservedHead(nn.Module):
    """Keep the real logits, force START to test suppression, avoid vision controls."""
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.logits = []

    def forward(self, hidden):
        logits = self.head(hidden)
        self.logits.append(logits.detach().clone())
        logits = logits.clone()
        logits[..., 80:83] = -100
        logits[..., TOKENS.start_token_id] = 100
        return logits


def image_prompt():
    # The leading masked slot separates physical cache offsets from MRoPE offsets.
    return dict(input_ids=torch.tensor([[0, 7, 81, 80, 80, 80, 80, 8]]),
                attention_mask=torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]]),
                pixel_values=torch.arange(48, dtype=torch.float32).reshape(4, 12)/48,
                image_grid_thw=torch.tensor([[1, 2, 2]]))


@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
def test_native_cached_decoder_matches_full_image_prompt_logits_and_resets(backend):
    """Catches replaying full prefixes, losing image MRoPE/masks, or stale caches."""
    torch.manual_seed(12)
    model = tiny_stage2_qwen(attn_implementation=backend).eval()
    model.config.stage = 'stage1'
    model.lm_head = ObservedHead(model.lm_head)
    controller = ContinuousLatentController()
    prompt = image_prompt()
    config = ContinuousLatentGenerationConfig(max_new_tokens=4, repetition_penalty=1.05)
    plain = controller.generate(model=model, token_ids=TOKENS,
        generation_config=config, disable_latent_generation=True, **prompt)
    reference_logits = model.lm_head.logits[:]
    model.lm_head.logits.clear()
    calls, lengths, caches = [], [], []

    def before(_module, _args, kwargs):
        calls.append(dict(length=kwargs['inputs_embeds'].shape[1],
            mask=kwargs['attention_mask'].clone(), positions=kwargs['position_ids'].clone(),
            cache_position=kwargs['cache_position'].clone(),
            past_length=(kwargs['past_key_values'].get_seq_length()
                         if kwargs['past_key_values'] is not None else 0),
            use_cache=kwargs['use_cache'], generate_mode=kwargs['generate_mode']))

    def after(_module, _args, output):
        lengths.append(output.past_key_values.get_seq_length())
        caches.append(output.past_key_values)

    pre = model.model.register_forward_pre_hook(before, with_kwargs=True)
    post = model.model.register_forward_hook(after)
    try:
        cached = controller.generate(model=model, token_ids=TOKENS,
            generation_config=config, disable_latent_generation=True, use_kv_cache=True, **prompt)
        again = controller.generate(model=model, token_ids=TOKENS,
            generation_config=config, disable_latent_generation=True, use_kv_cache=True, **prompt)
    finally:
        pre.remove()
        post.remove()
    assert cached.generated_suffix_ids == again.generated_suffix_ids == plain.generated_suffix_ids
    assert cached.stop_reason == plain.stop_reason == 'max_new_tokens'
    assert cached.latent_block_count == cached.latent_start_count == 0
    assert not set(cached.generated_suffix_ids) & {90, 91, 92}
    assert [c['length'] for c in calls] == [8, 1, 1, 1]*2
    assert [c['past_length'] for c in calls] == [0, 8, 9, 10]*2
    assert lengths == [8, 9, 10, 11]*2
    assert caches[0] is not caches[4]
    assert all(c['use_cache'] and c['generate_mode'] for c in calls)
    # Hand-derived Qwen image positions, including the ignored pad's position=1.
    assert calls[0]['positions'][:, 0].tolist() == [
        [1, 0, 1, 2, 2, 2, 2, 4],
        [1, 0, 1, 2, 2, 3, 3, 4],
        [1, 0, 1, 2, 3, 2, 3, 4]]
    assert calls[0]['cache_position'].tolist() == list(range(8))
    for i in range(1, 4):
        assert calls[i]['positions'][:, 0].tolist() == [[4+i]]*3
        assert calls[i]['cache_position'].tolist() == [7+i]
        assert calls[i]['mask'].tolist() == [[0]+[1]*(7+i)]
    for actual, expected in zip(model.lm_head.logits, reference_logits*2, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('limit,eos,reason,length', [
    (None, None, 'max_new_tokens', 4), (10, None, 'max_sequence_length', 2),
    (None, 'first', 'eos', 1),
])
def test_cached_off_preserves_eos_and_bounded_termination(limit, eos, reason, length):
    """Catches cache mode bypassing EOS or the original strict length policy."""
    torch.manual_seed(12)
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    model.lm_head = ObservedHead(model.lm_head)
    prompt = image_prompt()
    controller = ContinuousLatentController()
    if eos:
        first = controller.generate(model=model, token_ids=TOKENS,
            generation_config=ContinuousLatentGenerationConfig(max_new_tokens=1),
            disable_latent_generation=True, **prompt)
        eos = first.generated_suffix_ids[0]
    config = ContinuousLatentGenerationConfig(max_new_tokens=4,
        max_sequence_length=limit, eos_token_id=eos)
    plain = controller.generate(model=model, token_ids=TOKENS,
        generation_config=config, disable_latent_generation=True, **prompt)
    cached = controller.generate(model=model, token_ids=TOKENS,
        generation_config=config, disable_latent_generation=True, use_kv_cache=True, **prompt)
    assert cached.generated_suffix_ids == plain.generated_suffix_ids
    assert len(cached.generated_suffix_ids) == length
    assert cached.stop_reason == plain.stop_reason == reason


@pytest.mark.parametrize('value', [1, 'true', None])
def test_controller_cache_option_rejects_non_boolean(value):
    """Catches truthy coercion at the public cache boundary."""
    with pytest.raises(TypeError, match='use_kv_cache.*boolean'):
        ContinuousLatentController().generate(
            model=scripted_model(latent_size=2, emitted_ids=[3], stage='stage1'),
            input_ids=torch.tensor([[1]]), attention_mask=torch.ones(1, 1),
            token_ids=TOKENS, generation_config=ContinuousLatentGenerationConfig(),
            disable_latent_generation=True, use_kv_cache=value)


@pytest.mark.parametrize('stage,disable', [('stage1', False), ('stage2', True)])
def test_controller_rejects_unsupported_cache_before_forward(stage, disable):
    """Catches accidentally caching an enabled latent or Stage 2 execution."""
    with pytest.raises(ValueError, match='Stage 1.*disabled'):
        ContinuousLatentController().generate(
            model=scripted_model(latent_size=2, emitted_ids=[3], stage=stage),
            input_ids=torch.tensor([[1]]), attention_mask=torch.ones(1, 1),
            token_ids=TOKENS, generation_config=ContinuousLatentGenerationConfig(),
            disable_latent_generation=disable, use_kv_cache=True)


def test_requested_cache_without_returned_state_fails_explicitly():
    """A backend that drops the cache must not silently replay the prefix."""
    model = tiny_stage2_qwen().eval()
    model.config.stage = 'stage1'
    def drop_cache(_module, _args, output):
        output.past_key_values = None
        return output
    handle = model.model.register_forward_hook(drop_cache)
    try:
        with pytest.raises(RuntimeError, match='cache.*returned'):
            ContinuousLatentController().generate(model=model, token_ids=TOKENS,
                generation_config=ContinuousLatentGenerationConfig(max_new_tokens=2),
                disable_latent_generation=True, use_kv_cache=True, **image_prompt())
    finally:
        handle.remove()


@pytest.mark.parametrize('cache', [False, True])
def test_cli_parses_cache_and_forwards_it_to_stage1(cache, monkeypatch):
    """Catches an ignored CLI switch or dropped controller argument."""
    module = _import_entry_point('eval', monkeypatch)
    argv = ['eval', '--model_dir', 'model', '--test_data_path', 'data',
            '--task_name', 'zebra-cot', '--runtime_identity_path', 'runtime']
    monkeypatch.setattr(sys, 'argv', argv+(['--use-kv-cache'] if cache else []))
    assert module.get_eval_args().use_kv_cache is cache
    controller = RecordingController()
    monkeypatch.setattr(module, 'ContinuousLatentController', lambda: controller)
    module.run_one_example(RecordingStage1Model(), FakeProcessor(),
        {'text_input': 'question'}, {}, disable_latent_generation=True, use_kv_cache=cache)
    assert controller.calls[0]['use_kv_cache'] is cache


@pytest.mark.parametrize('model,disable', [(RecordingStage1Model, False),
                                         (ModelWhoseGenerateRaises, True)])
def test_eval_rejects_unsupported_cache_before_inference(model, disable, monkeypatch):
    module = _import_entry_point('eval', monkeypatch)
    with pytest.raises(ValueError, match='Stage 1.*disabled'):
        module.run_one_example(model(), FakeProcessor(), {'text_input': 'question'}, {},
            disable_latent_generation=disable, use_kv_cache=True)


def generation_config(cache):
    return dict(max_new_tokens=4, max_sequence_length=32, repetition_penalty=1.,
                temperature=0., top_p=1., use_kv_cache=cache,
                disable_latent_generation=True)


@pytest.mark.parametrize('cache', [False, True])
def test_manifest_roundtrips_cache_choice_and_merge_summary(cache, tmp_path):
    """Catches dropping cache provenance from normalized configs or final summary."""
    shards = _write_fake_shards(tmp_path, generation_configs=[generation_config(cache)]*4,
        row_overrides={i: dict(index=i, task_name='task', match=True, use_kv_cache=cache) for i in range(4)})
    assert json.loads(shards[0].read_text())['generation_config']['use_kv_cache'] is cache
    assert _merge(tmp_path, shards, 4)['generation_config']['use_kv_cache'] is cache


def test_merge_rejects_mixed_cache_modes(tmp_path):
    shards = _write_fake_shards(tmp_path,
        generation_configs=[generation_config(c) for c in [True, False, True, True]])
    with pytest.raises(ValueError, match='generation config mismatch'):
        _merge(tmp_path, shards, 4)


@pytest.mark.parametrize('value', [1, 'true', None])
def test_manifest_rejects_non_boolean_cache(value, tmp_path):
    with pytest.raises(ValueError, match='use_kv_cache'):
        _write_fake_shards(tmp_path, generation_configs=[generation_config(value)]*4)


def test_merge_rejects_row_cache_disagreeing_with_manifest(tmp_path):
    shards = _write_fake_shards(tmp_path, generation_configs=[generation_config(True)]*4,
        row_overrides={i: dict(index=i, task_name='task', match=True, use_kv_cache=(i != 2)) for i in range(4)})
    with pytest.raises(ValueError, match='cache.*mismatch'):
        _merge(tmp_path, shards, 4)


@pytest.mark.parametrize('cache', [False, True])
def test_paired_summary_preserves_off_cache_mode_without_altering_semantic_profile(cache):
    on = [row(0, True), row(1, True)]
    off = [dict(row(i, True, off=True), use_kv_cache=cache) for i in range(2)]
    result = pair_generation(on, off, [0, 1])
    assert result['off_use_kv_cache'] is cache
    assert result['effect_off_minus_on'] == 0


@pytest.mark.parametrize('modes', [[True, False], [True, 1], [True, None]])
def test_paired_summary_rejects_mixed_or_non_boolean_cache_rows(modes):
    on = [row(0, True), row(1, True)]
    off = [dict(row(i, True, off=True), use_kv_cache=modes[i]) for i in range(2)]
    with pytest.raises(ValueError, match='cache'):
        pair_generation(on, off, [0, 1])


@pytest.mark.parametrize('shards,cache', [(1, False), (1, True), (4, False), (4, True)])
def test_main_records_cache_execution_without_changing_semantic_profile(shards, cache, tmp_path, monkeypatch):
    """Catches dropping the main-loop switch or result/manifest provenance."""
    from types import SimpleNamespace
    module = _import_entry_point('eval', monkeypatch)
    dataset, output = tmp_path/'data.jsonl', tmp_path/'worker.jsonl'
    dataset.write_text(json.dumps(dict(text_input='question', image_input=[],
                                       original_final_answer='42'))+'\n')
    args = SimpleNamespace(model_dir='model', base_model_dir=None, cache_dir='cache',
        test_data_path=str(dataset), output_json_path=str(output), task_name='zebra-cot',
        max_new_tokens=2, max_sequence_length=32, repetition_penalty=1., temperature=0.,
        top_p=1., num_shards=shards, shard_index=0, model_identity='model-A',
        disable_latent_generation=True, use_kv_cache=cache)
    monkeypatch.setattr(module, 'get_eval_args', lambda: args)
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    monkeypatch.setattr(module, 'load_evaluation_artifacts',
        lambda **_kwargs: (model, FakeProcessor(include_visual_inputs=False)))
    assert module.main() == (0 if shards == 4 else None)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(r['use_kv_cache'] is cache for r in rows)
    assert all('use_kv_cache' not in r['generation_profile'] for r in rows)
    assert rows[0]['latent_start_count'] == rows[0]['latent_block_count'] == 0
    if shards == 4:
        manifest = json.loads((tmp_path/'worker.jsonl.manifest.json').read_text())
        assert manifest['generation_config']['use_kv_cache'] is cache


@pytest.mark.parametrize('explicit_baseline_cache', [False, True])
def test_summary_cli_keeps_execution_metadata_out_of_semantic_profile(explicit_baseline_cache, tmp_path, monkeypatch):
    """Catches treating new uncached baseline provenance as a decoding mismatch."""
    import hashlib
    from types import SimpleNamespace
    from scripts import analyze_stage1_v5_causal as cli
    source = tmp_path/'source.jsonl'
    source.write_text('{}\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    baseline = [row(3, True), row(12, True)]
    spot = [dict(r, index=i, model_artifact='artifact', use_kv_cache=False)
            for i, r in enumerate(baseline)]
    off = [dict(r, index=i, model_artifact='artifact', disable_latent_generation=True,
                source_sha256='subset', use_kv_cache=True) for i, r in enumerate(baseline)]
    config = dict(max_new_tokens=4096)
    if explicit_baseline_cache:
        config['use_kv_cache'] = False
    values = dict(on=baseline, spot=spot, off=off,
        manifest=dict(source_indices=[3, 12], source_sha256=digest, subset_sha256='subset'),
        baseline_manifest=dict(successful=True, model_identity='model-A',
            generation_config=config, dataset_identity={'path': str(source)}))
    paths = {}
    for key, value in values.items():
        path = tmp_path/(key+'.json')
        path.write_text(''.join(json.dumps(r)+'\n' for r in value) if isinstance(value, list)
                        else json.dumps(value))
        paths[key] = str(path)
    monkeypatch.setattr(cli, 'model_weights_identity', lambda _path: 'fixed-weight-identity')
    output = tmp_path/'out.json'
    cli.generation_summary(SimpleNamespace(**paths, seed=42, output=str(output)))
    summary = json.loads(output.read_text())
    assert summary['off_use_kv_cache'] is True
    assert summary['generation_profile'] == {'max_new_tokens': 4096}


from tests.evaluation.test_numerics import restore_precision, state


@pytest.mark.parametrize('cache', [False, True])
@pytest.mark.parametrize('policy', ['historical', 'tf32_off', 'full_accumulation', 'math_sdpa'])
def test_eval_policy_is_separate_scoped_and_records_active_controls(cache, policy, monkeypatch):
    # Losing the inference scope, leaking kwargs or metadata, or retaining precision fails here.
    module = _import_entry_point('eval', monkeypatch)
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    processor = FakeProcessor(include_visual_inputs=False)
    torch.set_float32_matmul_precision('medium')
    before = state()
    observed = []
    handle = model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(
        lambda _m, _a: observed.append(state()))
    kwargs = dict(max_new_tokens=2, repetition_penalty=1.05, temperature=0., top_p=1.)
    try:
        result = module.run_one_example(model, processor, {'text_input': 'question'}, kwargs,
            disable_latent_generation=True, use_kv_cache=cache, numerical_policy=policy)
        assert state() == before
        assert kwargs == dict(max_new_tokens=2, repetition_penalty=1.05, temperature=0., top_p=1.)
        if policy == 'historical':
            assert 'numerical_policy' not in result and 'numerical_observation' not in result
            assert all(s == before for s in observed)
        else:
            assert result['numerical_policy'] == policy
            assert result['numerical_observation']['float32_matmul_precision'] == 'highest'
            assert result['numerical_observation']['matmul_allow_tf32'] is False
            assert result['numerical_observation']['parameter_dtypes']['weight'] == 'torch.float32'
            assert all(s[0] == 'highest' and s[1] is False for s in observed)
        observed.clear()
        module.run_one_example(model, processor, {'text_input': 'question'}, kwargs,
                               disable_latent_generation=True, use_kv_cache=cache)
        assert all(s == before for s in observed)
    finally:
        handle.remove()


def test_eval_parser_accepts_explicit_policy_and_preserves_historical_default(monkeypatch):
    module = _import_entry_point('eval', monkeypatch)
    argv = ['eval', '--model_dir', 'model', '--test_data_path', 'data', '--task_name',
            'zebra-cot', '--runtime_identity_path', 'runtime']
    monkeypatch.setattr(sys, 'argv', argv)
    assert module.get_eval_args().numerical_policy == 'historical'
    monkeypatch.setattr(sys, 'argv', argv+['--numerical-policy', 'full_accumulation'])
    assert module.get_eval_args().numerical_policy == 'full_accumulation'
    monkeypatch.setattr(sys, 'argv', argv+['--numerical-policy', 'unknown'])
    with pytest.raises(SystemExit):
        module.get_eval_args()


def policy_main_args(tmp_path, *, shards=1, policy='tf32_off', protocol='continuous', disable=True):
    from types import SimpleNamespace
    dataset = tmp_path/'policy-data.jsonl'
    dataset.write_text(json.dumps(dict(text_input='question', image_input=[], original_final_answer='42'))+'\n')
    return SimpleNamespace(model_dir=str(tmp_path), base_model_dir=None, cache_dir='data/hf-cache',
        test_data_path=str(dataset), output_json_path=str(tmp_path/'policy-worker.jsonl'),
        task_name='zebra-cot', max_new_tokens=2, max_sequence_length=32, repetition_penalty=1.05,
        temperature=0., top_p=1., num_shards=shards, shard_index=0, model_identity='model-A',
        disable_latent_generation=disable, use_kv_cache=False, evaluation_protocol=protocol,
        numerical_policy=policy, runtime_identity_path=str(tmp_path/'runtime.json'))


@pytest.mark.parametrize('protocol,stage,disable', [('original','stage1',False),
    ('continuous','stage2',True), ('continuous','stage1',False)])
def test_main_rejects_unsupported_policy_globally_before_any_output(protocol, stage, disable, tmp_path, monkeypatch):
    module = _import_entry_point('eval', monkeypatch)
    from step_preference import capacity_evidence
    monkeypatch.setattr(capacity_evidence, 'git_identity', lambda: 'a'*40)
    args = policy_main_args(tmp_path, protocol=protocol, disable=disable)
    args.runtime_identity_path = None
    monkeypatch.setattr(module, 'get_eval_args', lambda: args)
    model = tiny_stage2_qwen().eval()
    model.config.stage = stage
    monkeypatch.setattr(module, 'load_evaluation_artifacts', lambda **_kw: (model, FakeProcessor()))
    with pytest.raises(ValueError, match='numerical policy'):
        module.main()
    from pathlib import Path
    assert not Path(args.output_json_path).exists()
    assert not (tmp_path/'runtime.json').exists()


@pytest.mark.parametrize('shards', [1, 4])
def test_main_records_new_policy_runtime_and_row_identity(shards, tmp_path, monkeypatch):
    module = _import_entry_point('eval', monkeypatch)
    from step_preference import capacity_evidence
    monkeypatch.setattr(capacity_evidence, 'git_identity', lambda: 'a'*40)
    args = policy_main_args(tmp_path, shards=shards)
    monkeypatch.setattr(module, 'get_eval_args', lambda: args)
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    monkeypatch.setattr(module, 'load_evaluation_artifacts',
                        lambda **_kw: (model, FakeProcessor(include_visual_inputs=False)))
    assert module.main() == (0 if shards == 4 else None)
    from pathlib import Path
    rows = [json.loads(line) for line in Path(args.output_json_path).read_text().splitlines()]
    assert all(r['numerical_policy'] == 'tf32_off' for r in rows)
    assert all(r['numerical_observation']['matmul_allow_tf32'] is False for r in rows)
    runtime = json.loads(Path(args.runtime_identity_path).read_text())
    assert runtime['kind'] == 'numerical-policy-evaluation-runtime-identity'
    assert runtime['numerical_policy'] == 'tf32_off'
    assert runtime['numerical_observation']['float32_matmul_precision'] == 'highest'
    assert runtime['execution_identity']['evaluation_protocol'] == 'continuous'
    if shards == 4:
        manifest = json.loads(Path(args.output_json_path+'.manifest.json').read_text())
        assert manifest['generation_config']['numerical_policy'] == 'tf32_off'


def adapter_model():
    from peft.tuners.lora.layer import Linear
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    attention = model.model.layers[0].self_attn
    attention.q_proj = Linear(attention.q_proj, 'default', r=2, lora_alpha=4, use_dora=True).eval()
    with torch.no_grad():
        attention.q_proj.lora_B['default'].weight.fill_(.1)
    return model


@pytest.mark.parametrize('cache', [False, True])
@pytest.mark.parametrize('policy', ['adapter_token_fp64', 'adapter_token_fp64_math_sdpa'])
def test_evaluator_observes_actual_fp64_after_generation_and_no_historical_leak(cache, policy, tmp_path, monkeypatch):
    module = _import_entry_point('eval', monkeypatch)
    from step_preference import capacity_evidence
    monkeypatch.setattr(capacity_evidence, 'git_identity', lambda: 'a'*40)
    model, processor = adapter_model(), FakeProcessor(include_visual_inputs=False)
    runtime_path = tmp_path/'adapter-runtime.json'
    module.publish_evaluation_runtime_identity(model, tmp_path, runtime_path, numerical_policy=policy)
    runtime = json.loads(runtime_path.read_text())
    assert runtime['numerical_observation']['adapter_token_arithmetic']['token_calls'] == 0
    before = state()
    kwargs = dict(max_new_tokens=2, repetition_penalty=1.05, temperature=0., top_p=1.)
    result = module.run_one_example(model, processor, {'text_input': 'question'}, kwargs,
        disable_latent_generation=True, use_kv_cache=cache, numerical_policy=policy)
    arithmetic = result['numerical_observation']['adapter_token_arithmetic']
    assert arithmetic['token_calls'] > 0
    assert arithmetic['actual_operand_dtypes']['input'] == 'torch.float64'
    assert result['numerical_policy'] == policy
    assert state() == before
    historical = module.run_one_example(model, processor, {'text_input': 'question'}, kwargs,
        disable_latent_generation=True, use_kv_cache=cache)
    assert 'numerical_observation' not in historical
    assert 'forward' not in model.model.layers[0].self_attn.q_proj.lora_A['default'].__dict__


@pytest.mark.parametrize('static', [False, True])
def test_new_policy_rejects_static_or_unsupported_model_before_outputs(static, tmp_path, monkeypatch):
    module = _import_entry_point('eval', monkeypatch)
    args = policy_main_args(tmp_path, policy='adapter_token_fp64')
    args.runtime_identity_path = None
    args.static_inference_cache = static
    args.use_kv_cache = static
    monkeypatch.setattr(module, 'get_eval_args', lambda: args)
    model = adapter_model() if static else tiny_stage2_qwen().eval()
    model.config.stage = 'stage1'
    monkeypatch.setattr(module, 'load_evaluation_artifacts', lambda **_kw: (model, FakeProcessor()))
    with pytest.raises(ValueError):
        module.main()
    from pathlib import Path
    assert not Path(args.output_json_path).exists()


def test_role_error_is_fatal_before_success_summary(tmp_path, monkeypatch):
    from evaluation.adapter_numerics import AdapterNumericsError
    module = _import_entry_point('eval', monkeypatch)
    args = policy_main_args(tmp_path, policy='adapter_token_fp64')
    args.runtime_identity_path = None
    monkeypatch.setattr(module, 'get_eval_args', lambda: args)
    model = adapter_model()
    a = model.model.layers[0].self_attn.q_proj.lora_A['default']
    handle = a.register_forward_pre_hook(lambda _m, args: (torch.zeros_like(args[0]),) if args[0].ndim == 2 else None)
    monkeypatch.setattr(module, 'load_evaluation_artifacts',
        lambda **_kw: (model, FakeProcessor(include_visual_inputs=False)))
    try:
        with pytest.raises(AdapterNumericsError):
            module.main()
    finally:
        handle.remove()
    from pathlib import Path
    path = Path(args.output_json_path)
    assert not path.exists() or 'summary' not in path.read_text()


@pytest.mark.parametrize('shards', [1, 4])
@pytest.mark.parametrize('policy', ['adapter_token_fp64', 'adapter_token_fp64_math_sdpa'])
def test_main_adapter_rows_keep_own_post_generation_interval(shards, policy, tmp_path, monkeypatch):
    # Main must not overwrite each real row with its zero-call preflight snapshot.
    from pathlib import Path
    from step_preference import capacity_evidence
    module = _import_entry_point('eval', monkeypatch)
    monkeypatch.setattr(capacity_evidence, 'git_identity', lambda: 'a'*40)
    args = policy_main_args(tmp_path, shards=shards, policy=policy)
    data = Path(args.test_data_path)
    data.write_text(data.read_text()*4)
    model = adapter_model()
    monkeypatch.setattr(module, 'get_eval_args', lambda: args)
    monkeypatch.setattr(module, 'load_evaluation_artifacts',
        lambda **_kw: (model, FakeProcessor(include_visual_inputs=False)))
    assert module.main() == (0 if shards == 4 else None)
    rows = [json.loads(line) for line in Path(args.output_json_path).read_text().splitlines()]
    rows = [row for row in rows if not row.get('summary')]
    assert len(rows) == (1 if shards == 4 else 4)
    assert [row['numerical_observation']['adapter_token_arithmetic']['token_calls'] for row in rows] == [4]*len(rows)
    assert all(row['numerical_observation']['adapter_token_arithmetic']['actual_operand_dtypes']['input'] == 'torch.float64' for row in rows)
    runtime = json.loads(Path(args.runtime_identity_path).read_text())
    assert runtime['numerical_observation']['adapter_token_arithmetic']['token_calls'] == 0
    if shards == 4:
        manifest = json.loads(Path(args.output_json_path+'.manifest.json').read_text())
        assert manifest['generation_config']['numerical_policy'] == policy
        assert 'adapter_token_arithmetic' not in manifest['generation_config']
