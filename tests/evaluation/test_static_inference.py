"""Causal witnesses for per-generation fixed inference state (CPU FP32)."""
import json
import sys
from contextlib import contextmanager

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.dora import DoraLinearLayer
from peft.tuners.trainable_tokens.layer import TrainableTokensLayer

from evaluation.continuous_latent import ContinuousLatentController, ContinuousLatentGenerationConfig
from evaluation.stage1_v5_causal import pair_generation
from stage1.token_rows import SparseOutputTokenRows
from tests.evaluation.test_continuous_latent import TOKENS
from tests.evaluation.test_off_kv_cache import ObservedHead, image_prompt, generation_config
from tests.evaluation.test_stage1_v5_causal import row
from tests.multinode.test_evaluation import _write_fake_shards, _merge
from tests.stage2.helpers import tiny_stage2_qwen
from tests.stage2.test_eval_integration import _import_entry_point, RecordingController, RecordingStage1Model, FakeProcessor


def adapted_model():
    torch.manual_seed(42)
    qwen = tiny_stage2_qwen(attn_implementation='sdpa')
    qwen.config.stage = 'stage1'
    get_peft_model(qwen, LoraConfig(r=2, lora_alpha=4, use_dora=True,
        target_modules=['q_proj', 'v_proj'], trainable_token_indices=[90, 91, 92]))
    qwen.lm_head = SparseOutputTokenRows(qwen.lm_head, [90, 91, 92])
    with torch.no_grad():
        for name, p in qwen.named_parameters():
            if 'lora_B' in name: p.fill_(.13)
            if 'trainable_tokens_delta' in name: p.add_(.25)
        qwen.lm_head.output_delta.fill_(.17)
    qwen.lm_head = ObservedHead(qwen.lm_head)
    return qwen.eval()


def generate(model, prompt, static=False, count=4):
    return ContinuousLatentController().generate(model=model, token_ids=TOKENS,
        generation_config=ContinuousLatentGenerationConfig(max_new_tokens=count),
        disable_latent_generation=True, use_kv_cache=True,
        static_inference_cache=static, **prompt)


@contextmanager
def count_constants(model):
    counts = {'norm': 0, 'embedding': 0, 'rope': 0, 'eye': 0}
    original_eye = torch.eye
    def eye(*args, **kwargs):
        counts['eye'] += 1
        return original_eye(*args, **kwargs)
    saved = [(torch, 'eye', original_eye, True)]
    torch.eye = eye
    for module in model.modules():
        method, key = (('get_weight_norm', 'norm') if type(module) is DoraLinearLayer
            else ('get_merged_weights', 'embedding') if isinstance(module, TrainableTokensLayer)
            else (None, None))
        if method:
            original = getattr(module, method)
            def observe(*a, _original=original, _key=key, **kw):
                counts[_key] += 1
                return _original(*a, **kw)
            saved.append((module, method, original, method in module.__dict__))
            setattr(module, method, observe)
    original = model.get_rope_index
    def rope(*a, **kw):
        counts['rope'] += 1
        return original(*a, **kw)
    saved.append((model, 'get_rope_index', original, 'get_rope_index' in model.__dict__))
    model.get_rope_index = rope
    try:
        yield counts
    finally:
        for obj, key, value, existed in saved:
            if existed: setattr(obj, key, value)
            else: delattr(obj, key)


@pytest.mark.parametrize('adapted', [False, True])
@pytest.mark.parametrize('kind', ['image-left', 'image-right', 'image-none', 'text-left', 'text-right', 'text-none', 'text-one'])
def test_static_matches_original_cached_positions_logits_and_constant_counts(kind, adapted):
    """Catches stale/miscomputed constants, missing prompt reuse, or wrong padded MRoPE."""
    model = adapted_model() if adapted else tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    if not adapted: model.lm_head = ObservedHead(model.lm_head)
    model.eval()
    if kind.startswith('image'):
        prompt = image_prompt()
        if kind.endswith('none'):
            prompt['input_ids'] = prompt['input_ids'][:, 1:]
            prompt['attention_mask'] = prompt['attention_mask'][:, 1:]
        elif kind.endswith('right'):
            prompt['input_ids'] = torch.cat((prompt['input_ids'][:, 1:], torch.tensor([[0]])), 1)
            prompt['attention_mask'] = torch.tensor([[1]*7+[0]])
    else:
        ids = {'text-left': [0, 7, 8], 'text-right': [7, 8, 0], 'text-none': [7, 8], 'text-one': [0, 0, 7]}[kind]
        prompt = dict(input_ids=torch.tensor([ids]), attention_mask=torch.tensor([[int(i != 0) for i in ids]]))
    positions = []
    hook = model.model.register_forward_pre_hook(lambda m, a, kw: positions.append((kw['position_ids'].clone(), kw['cache_position'].clone())), with_kwargs=True)
    try:
        with count_constants(model) as original_counts:
            original = generate(model, prompt)
        logits = model.lm_head.logits[:]
        model.lm_head.logits.clear()
        original_positions, positions[:] = positions[:], []
        with count_constants(model) as optimized_counts:
            optimized = generate(model, prompt, True)
    finally:
        hook.remove()
    assert optimized.generated_suffix_ids == original.generated_suffix_ids
    assert optimized.stop_reason == original.stop_reason
    for actual, expected in zip(model.lm_head.logits, logits, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual, expected in zip(positions, original_positions, strict=True):
        for a, e in zip(actual, expected): assert torch.equal(a, e)
    assert original_counts['rope'] == 4
    assert optimized_counts['rope'] == 1
    if adapted:
        assert original_counts['norm'] == 8 and optimized_counts['norm'] == 2
        assert original_counts['eye'] == 8 and optimized_counts['eye'] == 2
        assert original_counts['embedding'] == 5 and optimized_counts['embedding'] == 1


def test_scope_restores_exact_instance_state_on_exception_and_recomputes_after_change():
    """Catches leaked overrides on exceptions and stale learned rows across calls."""
    from evaluation.static_inference import static_inference_cache
    model = adapted_model()
    modules = list(model.modules())
    before = [{k: v for k, v in m.__dict__.items() if k in ('forward', 'get_merged_weights')} for m in modules]
    ids = torch.tensor([[90, 7, 92]])
    embedding = model.get_input_embeddings()
    with torch.no_grad():
        old = embedding(ids).clone()
        with pytest.raises(RuntimeError, match='injected'):
            with static_inference_cache(model):
                assert torch.equal(embedding(ids), old)
                raise RuntimeError('injected')
        assert before == [{k: v for k, v in m.__dict__.items() if k in ('forward', 'get_merged_weights')} for m in modules]
        embedding.token_adapter.trainable_tokens_delta['default'].add_(.4)
        for name, p in model.named_parameters():
            if 'lora_B' in name: p.add_(.07)
        expected = embedding(ids)
        with static_inference_cache(model): assert torch.equal(embedding(ids), expected)
        assert not torch.equal(old, expected)
    prompt = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
    a = generate(model, prompt)
    b = generate(model, prompt, True)
    assert a.generated_suffix_ids == b.generated_suffix_ids


@pytest.mark.parametrize('training,grad', [(True, False), (False, True)])
def test_static_context_rejects_mutable_execution(training, grad):
    from evaluation.static_inference import static_inference_cache
    model = adapted_model().train(training)
    with torch.set_grad_enabled(grad), pytest.raises(ValueError, match='eval|grad'):
        with static_inference_cache(model): pass


@pytest.mark.parametrize('value', [1, 'true', None])
def test_static_controller_requires_strict_boolean(value):
    with pytest.raises(TypeError, match='static_inference_cache.*boolean'):
        generate(adapted_model(), image_prompt(), value)


@pytest.mark.parametrize('stage,off,cache', [('stage1', False, True), ('stage2', True, True), ('stage1', True, False)])
def test_static_controller_rejects_unsupported_mode(stage, off, cache):
    model = adapted_model()
    model.config.stage = stage
    with pytest.raises(ValueError, match='Stage 1|static'):
        ContinuousLatentController().generate(model=model, token_ids=TOKENS, **image_prompt(),
            generation_config=ContinuousLatentGenerationConfig(), disable_latent_generation=off,
            use_kv_cache=cache, static_inference_cache=True)


@pytest.mark.parametrize('structural', [80, 81, 82])
def test_generated_vision_controls_fall_back_including_original_errors(structural):
    """Catches suppressing new tokens or skipping original malformed-grid errors."""
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    class ForcedHead(torch.nn.Module):
        def forward(self, hidden):
            result = torch.zeros(hidden.shape[0], 128)
            result[:, structural] = 100
            return result
    model.lm_head = ForcedHead().eval()
    outcomes = []
    for static in (False, True):
        with count_constants(model) as counts:
            try:
                result = generate(model, image_prompt(), static, count=3)
                outcomes.append(('result', result.generated_suffix_ids))
            except (IndexError, RuntimeError) as exc:
                outcomes.append((type(exc), str(exc)))
        assert counts['rope'] >= 2
    assert outcomes[0] == outcomes[1]


def static_config(value=True):
    return dict(generation_config(True), static_inference_cache=value)


@pytest.mark.parametrize('enabled', [False, True])
def test_cli_forwards_static_choice(enabled, monkeypatch):
    module = _import_entry_point('eval', monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['eval', '--model_dir', 'model', '--test_data_path', 'data', '--task_name', 'zebra-cot', '--runtime_identity_path', 'runtime']+(['--static-inference-cache'] if enabled else []))
    assert module.get_eval_args().static_inference_cache is enabled
    controller = RecordingController()
    monkeypatch.setattr(module, 'ContinuousLatentController', lambda: controller)
    module.run_one_example(RecordingStage1Model(), FakeProcessor(), {'text_input': 'question'}, {},
        disable_latent_generation=True, use_kv_cache=True, static_inference_cache=enabled)
    assert controller.calls[0]['static_inference_cache'] is enabled


@pytest.mark.parametrize('value', [True, False])
def test_metadata_roundtrip_and_pair_summary(value, tmp_path):
    shards = _write_fake_shards(tmp_path, generation_configs=[static_config(value)]*4,
        row_overrides={i: dict(index=i, task_name='task', match=True, use_kv_cache=True, static_inference_cache=value) for i in range(4)})
    assert _merge(tmp_path, shards, 4)['generation_config']['static_inference_cache'] is value
    summary = pair_generation([row(0, True)], [dict(row(0, True, off=True), use_kv_cache=True, static_inference_cache=value)], [0])
    assert summary['off_static_inference_cache'] is value


@pytest.mark.parametrize('value', [1, 'true', None])
def test_metadata_rejects_invalid_boolean(value, tmp_path):
    with pytest.raises(ValueError, match='static_inference_cache'):
        _write_fake_shards(tmp_path, generation_configs=[static_config(value)]*4)


def test_merge_and_pair_reject_mixed_static_modes(tmp_path):
    shards = _write_fake_shards(tmp_path, generation_configs=[static_config(v) for v in [True, False, True, True]])
    with pytest.raises(ValueError, match='generation config mismatch'): _merge(tmp_path, shards, 4)
    off = [dict(row(i, True, off=True), use_kv_cache=True, static_inference_cache=bool(i)) for i in range(2)]
    with pytest.raises(ValueError, match='static'): pair_generation([row(0, True), row(1, True)], off, [0, 1])


def test_zero_budget_does_not_parse_an_unused_malformed_vision_prompt():
    """Catches eager MRoPE setup introducing an error when no decode is requested."""
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    prompt = dict(input_ids=torch.tensor([[7, 81]]), attention_mask=torch.ones(1, 2),
                  image_grid_thw=torch.tensor([[1, 2, 2]]))
    for static in (False, True):
        with count_constants(model) as counts:
            result = generate(model, prompt, static, count=0)
        assert result.generated_suffix_ids == () and counts['rope'] == 0


def test_absent_static_and_explicit_false_are_the_same_merge_mode(tmp_path):
    """Catches rejecting a historical default shard as a different optimization mode."""
    configs = [generation_config(True), static_config(False)]*2
    shards = _write_fake_shards(tmp_path, generation_configs=configs,
        row_overrides={i: dict(index=i, task_name='task', match=True, use_kv_cache=True) for i in range(4)})
    assert _merge(tmp_path, shards, 4)['source_count'] == 4


def test_merge_rejects_row_static_disagreeing_with_manifest(tmp_path):
    shards = _write_fake_shards(tmp_path, generation_configs=[static_config()]*4,
        row_overrides={i: dict(index=i, task_name='task', match=True, use_kv_cache=True,
                              static_inference_cache=(i != 2)) for i in range(4)})
    with pytest.raises(ValueError, match='static.*mismatch'): _merge(tmp_path, shards, 4)


@pytest.mark.parametrize('value', [1, 'true', None])
def test_eval_boundary_rejects_nonboolean_static_before_controller(value, monkeypatch):
    module = _import_entry_point('eval', monkeypatch)
    with pytest.raises(TypeError, match='static_inference_cache.*boolean'):
        module.run_one_example(RecordingStage1Model(), FakeProcessor(), {'text_input': 'question'}, {},
            disable_latent_generation=True, use_kv_cache=True, static_inference_cache=value)


@pytest.mark.parametrize('shards,enabled', [(1, False), (1, True), (4, False), (4, True)])
def test_main_static_metadata_is_opt_in_and_forwarded(shards, enabled, tmp_path, monkeypatch):
    """Catches dropping the main-loop switch or altering ordinary default serialization."""
    from types import SimpleNamespace
    module = _import_entry_point('eval', monkeypatch)
    dataset, output = tmp_path/'data.jsonl', tmp_path/'worker.jsonl'
    dataset.write_text(json.dumps(dict(text_input='question', image_input=[], original_final_answer='42'))+'\n')
    args = SimpleNamespace(model_dir='model', base_model_dir=None, cache_dir='cache',
        test_data_path=str(dataset), output_json_path=str(output), task_name='zebra-cot',
        max_new_tokens=2, max_sequence_length=32, repetition_penalty=1., temperature=0.,
        top_p=1., num_shards=shards, shard_index=0, model_identity='model-A',
        disable_latent_generation=True, use_kv_cache=True, static_inference_cache=enabled)
    monkeypatch.setattr(module, 'get_eval_args', lambda: args)
    model = tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage = 'stage1'
    monkeypatch.setattr(module, 'load_evaluation_artifacts',
        lambda **kw: (model, FakeProcessor(include_visual_inputs=False)))
    observations = []
    real_generate = ContinuousLatentController.generate
    def observe(self, **kwargs):
        observations.append(kwargs['static_inference_cache'])
        return real_generate(self, **kwargs)
    monkeypatch.setattr(ContinuousLatentController, 'generate', observe)
    module.main()
    assert observations == [enabled]
    saved = json.loads(output.read_text().splitlines()[0])
    assert saved.get('static_inference_cache', False) is enabled
    assert ('static_inference_cache' in saved) is enabled
    assert 'static_inference_cache' not in saved['generation_profile']
    if shards == 4:
        config = json.loads((tmp_path/'worker.jsonl.manifest.json').read_text())['generation_config']
        assert config.get('static_inference_cache', False) is enabled
        assert ('static_inference_cache' in config) is enabled


def test_successive_generations_recompute_constants_after_weight_and_prompt_changes():
    """Catches a model-global constant cache surviving a sample or state change."""
    model = adapted_model()
    for step, ids in enumerate(([90, 7, 8], [92, 8, 7, 91])):
        if step:
            with torch.no_grad():
                model.model.embed_tokens.token_adapter.trainable_tokens_delta['default'].add_(.31)
                for name, p in model.named_parameters():
                    if 'lora_B' in name: p.add_(.23)
        prompt = dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids)))
        original = generate(model, prompt)
        expected_logits = model.lm_head.logits[-4:]
        with count_constants(model) as counts:
            optimized = generate(model, prompt, True)
        assert counts == dict(norm=2, embedding=1, rope=1, eye=2)
        assert original.generated_suffix_ids == optimized.generated_suffix_ids
        for a, e in zip(model.lm_head.logits[-4:], expected_logits, strict=True):
            assert torch.equal(a, e)


def test_nonstandard_dora_linear_keeps_original_computation():
    """Catches applying the installed-standard-linear formula to a custom layer."""
    from evaluation.static_inference import static_inference_cache
    model = adapted_model()
    class CustomLinear(torch.nn.Linear):
        def forward(self, x): return super().forward(x) + .01
    for layer in model.modules():
        if hasattr(layer, 'lora_A') and 'default' in layer.lora_A:
            original = layer.lora_A['default']
            custom = CustomLinear(original.in_features, original.out_features, bias=False)
            custom.load_state_dict(original.state_dict())
            layer.lora_A['default'] = custom.eval()
    prompt = dict(input_ids=torch.tensor([[90, 7, 8]]), attention_mask=torch.ones(1, 3))
    a = generate(model, prompt)
    logits = model.lm_head.logits[:]
    with count_constants(model) as counts:
        b = generate(model, prompt, True)
    assert a.generated_suffix_ids == b.generated_suffix_ids
    assert counts['norm'] == counts['eye'] == 8
    for actual, expected in zip(model.lm_head.logits[4:], logits, strict=True):
        assert torch.equal(actual, expected)


def test_decoder_exception_restores_populated_dora_and_embedding_overrides():
    """Catches constant tensors/instance overrides surviving an actual decoder failure."""
    model = adapted_model()
    layers = list(model.modules())
    before = [{k: v for k, v in m.__dict__.items() if k in ('forward', 'get_merged_weights')} for m in layers]
    def fail(_m, _a, _out): raise RuntimeError('decoder failed after DoRA')
    handle = model.model.register_forward_hook(fail)
    try:
        with count_constants(model) as counts, pytest.raises(RuntimeError, match='decoder failed'):
            generate(model, image_prompt(), True)
        assert counts['norm'] == 2 and counts['embedding'] == 1
    finally:
        handle.remove()
    assert before == [{k: v for k, v in m.__dict__.items() if k in ('forward', 'get_merged_weights')} for m in layers]
    # Reuse after failure must still behave as the untouched reference path.
    reference = generate(model, image_prompt())
    optimized = generate(model, image_prompt(), True)
    assert reference.generated_suffix_ids == optimized.generated_suffix_ids


def test_custom_dora_forward_is_preserved_as_an_unsupported_configuration():
    """Catches replacing an instance wrapper (e.g. an execution hook) with bare DoRA."""
    from evaluation.static_inference import static_inference_cache
    model = adapted_model()
    layer = next(m for m in model.modules() if hasattr(m, 'lora_magnitude_vector') and len(m.lora_magnitude_vector))
    dora = layer.lora_magnitude_vector['default']
    original = dora.forward
    def custom(*args, **kwargs): return original(*args, **kwargs) + .37
    dora.forward = custom
    with torch.no_grad():
        x = torch.arange(8, dtype=torch.float32).reshape(1, 8)/8
        expected = layer(x)
        with static_inference_cache(model): actual = layer(x)
        assert torch.equal(actual, expected)
        assert dora.forward is custom


def test_mixed_dtype_dora_releases_converted_weight_inside_active_cache_scope():
    """Catches retaining a full converted base matrix alongside the small factor."""
    import weakref
    from evaluation.static_inference import static_inference_cache

    model = adapted_model()
    layer = next(m for m in model.modules() if hasattr(m, 'lora_magnitude_vector') and len(m.lora_magnitude_vector))
    dora = layer.lora_magnitude_vector['default']
    a, b = layer.lora_A['default'].double(), layer.lora_B['default'].double()
    dora.double()
    base = layer.get_base_layer()
    x = torch.arange(8, dtype=torch.float64).reshape(1, 8)/8
    kwargs = dict(lora_A=a, lora_B=b, scaling=layer.scaling['default'], base_layer=base)
    converted = []
    original_norm = dora.get_weight_norm
    def observe_norm(weight, lora_weight, scaling):
        assert weight.dtype == torch.float64 and base.weight.dtype == torch.float32
        assert weight.data_ptr() != base.weight.data_ptr()
        converted.append(weakref.ref(weight))
        return original_norm(weight, lora_weight, scaling)

    with torch.no_grad():
        base_result = base(x.float())
        expected = dora(x, base_result=base_result, **kwargs)
        expected_fallback = dora(x, base_result=None, **kwargs)
        dora.get_weight_norm = observe_norm
        try:
            with static_inference_cache(model):
                actual = dora(x, base_result=base_result, **kwargs)
                assert torch.equal(actual, expected)
                assert len(converted) == 1
                assert converted[0]() is None, 'converted base matrix remains resident during generation'
                assert torch.equal(dora(x, base_result=base_result, **kwargs), expected)
                assert torch.equal(dora(x, base_result=None, **kwargs), expected_fallback)
                assert len(converted) == 1, 'factor was recomputed instead of reused'
        finally:
            del dora.get_weight_norm
