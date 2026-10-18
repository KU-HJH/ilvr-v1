"""Native original execution witnesses for immutable per-generation PEFT reuse."""
import dataclasses
import importlib
import json
import sys

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from evaluation import original_witness as witness
from stage1.token_rows import SparseOutputTokenRows
from tests.evaluation.test_original_ilvr import tiny_original, START, PAD, END
from tests.evaluation.test_original_entrypoints import NativeProcessor, identity, shards
from tests.evaluation.test_original_review_fixes import arguments, setup_restored, observed
from tests.evaluation.test_static_inference import count_constants
from tests.multinode.test_evaluation_coordinator import _plan, _manifest
from src.multinode_runtime import build_evaluation_commands
from src.multinode_evaluation import merge_shards
from src.multinode_artifacts import validate_evaluation_sidecars


class ObservedSparseHead(nn.Module):
    def __init__(self, head, tokens=None):
        super().__init__()
        self.head, self.tokens = head, tokens
        self.calls = 0
        self.hidden, self.logits = [], []

    def forward(self, hidden):
        logits = self.head(hidden)
        self.hidden.append(hidden.detach().clone())
        self.logits.append(logits.detach().clone())
        if self.tokens is not None:
            token = self.tokens[min(self.calls, len(self.tokens)-1)]
            logits = torch.full_like(logits, -100.)
            logits[..., token] = 100.
        self.calls += 1
        return logits


def adapted_original(dtype=torch.float32, *, tokens=None, use_cache=True):
    api, model, calls = tiny_original([3], use_cache=use_cache)
    model.lm_head = nn.Linear(8, END+1, bias=False)
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, use_dora=True,
        target_modules=['q_proj', 'v_proj'], trainable_token_indices=[START, PAD, END]))
    qwen = model.base_model.model
    qwen.lm_head = SparseOutputTokenRows(qwen.lm_head, [START, PAD, END])
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name: parameter.fill_(.13)
            if 'trainable_tokens_delta' in name: parameter.add_(.25)
        qwen.lm_head.base_head.weight.zero_()
        qwen.lm_head.output_delta.fill_(.17)
    model.to(dtype=dtype).eval()
    if tokens is None:
        with torch.inference_mode():
            ids = torch.tensor([[7,8]])
            hidden = qwen(input_ids=ids, attention_mask=torch.ones_like(ids),
                use_cache=False, generate_mode=True, return_dict=True).hidden_states[0,-1]
            qwen.lm_head.base_head.weight[START].copy_(hidden)
    qwen.lm_head = ObservedSparseHead(qwen.lm_head, tokens).eval()
    calls.clear()
    return api, model, qwen, calls


def native(api, model, static=False, cap=6):
    model.base_model.model.lm_head.calls = 0
    with torch.inference_mode():
        return api.generate_original(model, dict(input_ids=torch.tensor([[7,8]]),
            attention_mask=torch.ones(1,2,dtype=torch.long)), dict(max_new_tokens=cap, do_sample=False),
            static_inference_cache=static)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('tokens', [None, [START,5,5,5,9,START,5,5,5,3]])
def test_native_peft_static_preserves_exact_tokens_logits_hidden_and_native_positions(dtype, tokens):
    api, model, qwen, calls = adapted_original(dtype, tokens=tokens)
    cap = 6 if tokens is None else 20
    with count_constants(qwen) as baseline_counts:
        baseline = native(api, model, cap=cap)
    expected_calls, expected_logits, expected_hidden = calls[:], qwen.lm_head.logits[:], qwen.lm_head.hidden[:]
    calls.clear(); qwen.lm_head.logits.clear(); qwen.lm_head.hidden.clear()
    with count_constants(qwen) as optimized_counts:
        optimized = native(api, model, True, cap=cap)
    assert torch.equal(optimized.sequences, baseline.sequences)
    suffix = optimized.sequences[0,2:].tolist()
    assert suffix[:4] == [START,PAD,PAD,END] and len(suffix)>4
    if tokens is not None:
        assert suffix == [START,PAD,PAD,END,9,START,PAD,PAD,END,3]
        assert optimized.stop_reason == 'eos'
    for actual, expected in zip(qwen.lm_head.logits, expected_logits, strict=True):
        assert torch.equal(actual, expected)
    for actual, expected in zip(qwen.lm_head.hidden, expected_hidden, strict=True):
        assert torch.equal(actual, expected)
    for actual, expected in zip(calls, expected_calls, strict=True):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert torch.equal(actual[key], expected[key]) if torch.is_tensor(actual[key]) else actual[key] == expected[key]
    assert baseline_counts['norm'] == len(expected_logits)*2
    assert optimized_counts['norm'] == optimized_counts['eye'] == 2
    assert baseline_counts['embedding'] > optimized_counts['embedding'] == 1
    assert optimized_counts['rope'] == baseline_counts['rope'], 'native MRoPE must remain untouched'


@pytest.mark.parametrize('cap', [1,2,3,4,5])
def test_native_static_keeps_caps_and_recomputes_after_parameter_changes(cap):
    api, model, qwen, calls = adapted_original(tokens=[START,5,5,5,3])
    for step in range(2):
        if step:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if 'lora_B' in name or 'trainable_tokens_delta' in name: parameter.add_(.09)
        qwen.lm_head.logits.clear()
        baseline = native(api, model, cap=cap)
        expected = qwen.lm_head.logits[:]
        qwen.lm_head.logits.clear(); calls.clear()
        with count_constants(qwen) as counts:
            optimized = native(api, model, True, cap=cap)
        assert torch.equal(baseline.sequences, optimized.sequences)
        assert len(optimized.sequences[0,2:]) == cap
        assert counts['norm'] == 2 and counts['embedding'] == 1
        assert calls[0]['cache_position'].tolist() == [0,1]
        for actual, value in zip(qwen.lm_head.logits, expected, strict=True): assert torch.equal(actual,value)


def test_native_static_restores_after_populated_decoder_exception():
    api, model, qwen, _ = adapted_original(tokens=[START,5,5,5,3])
    modules = list(model.modules())
    before = [{k:v for k,v in m.__dict__.items() if k in ('forward','get_merged_weights')} for m in modules]
    def fail(*args): raise RuntimeError('causal native decoder failure')
    handle = qwen.model.register_forward_hook(fail)
    try:
        with count_constants(qwen) as counts, pytest.raises(RuntimeError,match='causal native'):
            native(api, model, True)
        assert counts['norm'] == 2 and counts['embedding'] == 1
    finally:
        handle.remove()
    assert before == [{k:v for k,v in m.__dict__.items() if k in ('forward','get_merged_weights')} for m in modules]
    assert torch.equal(native(api,model).sequences, native(api,model,True).sequences)


@pytest.mark.parametrize('value', [1, 'true', None])
def test_native_static_strict_boolean_at_api_and_eval_boundary(value):
    api, model, _, calls = adapted_original(tokens=[3])
    with pytest.raises(TypeError,match='static_inference_cache.*boolean'):
        native(api,model,value)
    with pytest.raises(TypeError,match='static_inference_cache.*boolean'):
        importlib.import_module('eval').run_one_example(model,NativeProcessor(),{},dict(max_new_tokens=2),
            evaluation_protocol='original',static_inference_cache=value)
    assert calls == []


@pytest.mark.parametrize('grad,training', [(True,False),(False,True)])
def test_native_static_guard_requires_eval_and_no_grad(grad,training):
    api, model, _, _ = adapted_original(tokens=[3])
    model.train(training)
    with torch.set_grad_enabled(grad), pytest.raises(ValueError,match='eval.*no grad'):
        api.generate_original(model,dict(input_ids=torch.tensor([[7,8]])),dict(max_new_tokens=1),static_inference_cache=True)


def test_native_static_does_not_force_source_cache_setting():
    api, model, qwen, _ = adapted_original(tokens=[START,5],use_cache=False)
    with pytest.raises(AssertionError): native(api,model,True)
    assert qwen.generation_config.use_cache is False and qwen.config.use_cache is False


@pytest.mark.parametrize('sample_errors', [False,True])
def test_actual_four_worker_main_manifests_sidecars_and_merger_support_native_static(tmp_path,monkeypatch,sample_errors):
    from step_preference import capacity_evidence
    module = importlib.import_module('eval')
    _, model, _ = tiny_original([3])
    model.eval()
    processor = NativeProcessor()
    if sample_errors:
        module = setup_restored(monkeypatch,model)
    else:
        monkeypatch.setattr(module,'load_evaluation_artifacts',lambda **kwargs:(model,processor))
        monkeypatch.setattr(module,'publish_evaluation_runtime_identity',lambda *args,**kwargs:None)
        monkeypatch.setattr(capacity_evidence,'git_identity',lambda:'a'*40)
    dataset = tmp_path/'test.jsonl'; dataset.write_text('{}\n'*4)
    outputs, manifests = [], []
    for shard in range(4):
        output = tmp_path/f'shard-{shard}.jsonl'
        monkeypatch.setattr(sys,'argv',arguments(dataset,output,tmp_path/f'runtime-{shard}.json',shard=shard)+['--static-inference-cache'])
        assert module.main() == 0
        outputs.append(output); manifests.append(tmp_path/f'shard-{shard}.jsonl.manifest.json')
    expected = dict(max_new_tokens=4096,max_sequence_length=32768,temperature=0.,top_p=1.,
        generation_limit_policy='native',vision_ablation='none',static_inference_cache=True)
    validate_evaluation_sidecars(manifests,outputs,expected_model_identity='same-model',expected_sample_count=4,
        expected_evaluation_protocol='original',expected_generation_config=expected)
    summary = merge_shards(manifests,expected_sample_count=4,output_jsonl=tmp_path/'merged.jsonl',summary_json=tmp_path/'summary.json')
    assert summary['num_samples'] == 4 and summary['success'] == (0 if sample_errors else 4)
    assert summary['generation_config']['static_inference_cache'] is True
    assert all(json.loads(output.read_text())['static_inference_cache'] is True for output in outputs)
    payload = json.loads(manifests[-1].read_text()); payload['generation_config'].pop('static_inference_cache')
    manifests[-1].write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='generation config mismatch'):
        merge_shards(manifests,expected_sample_count=4,output_jsonl=tmp_path/'bad.jsonl',summary_json=tmp_path/'bad-summary.json')


def test_real_coordinator_cli_plan_and_four_worker_static_propagation(capsys):
    from src.multinode import _main
    plan = dataclasses.replace(_plan(),static_inference_cache=True,evaluation_protocol='original')
    commands = build_evaluation_commands(plan,inference_manifest=_manifest(),commit='b'*40)
    assert len(commands.workers)==4
    assert all('--static-inference-cache' in worker.node_spec.command for worker in commands.workers)
    assert _main(['evaluate','--dry-run','--run-id',plan.run_id,'--model-dir',str(plan.model_dir),
        '--test-data-path',str(plan.test_data_path),'--task-name','zebra-cot','--output-dir',str(plan.output_dir),
        '--evaluation-protocol','original','--static-inference-cache']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['plan']['static_inference_cache'] is True
    with pytest.raises(ValueError,match='static.*original'):
        build_evaluation_commands(dataclasses.replace(plan,evaluation_protocol='continuous'),inference_manifest=_manifest(),commit='b'*40)
    with pytest.raises(ValueError,match='static_inference_cache.*boolean'):
        dataclasses.replace(plan,static_inference_cache=1)


def test_witness_optimized_comparison_mode_separates_timing_and_constant_counts(tmp_path,monkeypatch):
    monkeypatch.setattr(witness,'__file__',str(tmp_path/'src/evaluation/original_witness.py'))
    modes=[]
    def run(mode,**kwargs):
        modes.append(mode)
        value=observed([START,PAD,PAD,END,3])
        value['instrumentation']=[dict(index=7,generation_wall_seconds=2. if mode=='restored' else 1.,
            constant_calls=dict(dora_weight_norm=10 if mode=='restored' else 2,embedding_materialization=5 if mode=='restored' else 1))]
        return value
    monkeypatch.setattr(witness,'run_mode',run)
    dataset=tmp_path/'test.jsonl'; dataset.write_text('{}\n')
    output=tmp_path/'results/witness'
    assert witness.main(['--model-dir','unused','--test-data-path',str(dataset),'--indices','7',
        '--max-new-tokens','5','--output-dir',str(output),'--comparison-mode','restored-vs-optimized']) == 0
    assert modes==['restored','optimized']
    report=json.loads((output/'comparison.json').read_text())
    assert report['verdict']=='pass' and report['token_equality'] is True
    assert report['latent_transition_coverage']=={'restored':[7],'optimized':[7]}
    assert report['instrumentation']['restored'][0]['generation_wall_seconds']==2.
    assert report['instrumentation']['optimized'][0]['constant_calls']['dora_weight_norm']==2


def test_witness_real_native_modes_count_constants_and_time_actual_cpu_without_cuda_sync(tmp_path,monkeypatch):
    import time
    def load(*args):
        _, model, _, _ = adapted_original()
        return model,NativeProcessor()
    monkeypatch.setattr(witness,'load_for_mode',load)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    def fail_sync(*args): raise AssertionError('CPU witness must not synchronize CUDA')
    monkeypatch.setattr(torch.cuda,'synchronize',fail_sync)
    dataset=tmp_path/'test.jsonl'; dataset.write_text('{}\n')
    values={mode:witness.run_mode(mode,model_dir='model',base_model_dir=None,dataset_path=dataset,
        indices=[0],cap=6,cache_dir='data/cache',deadline=time.monotonic()+60) for mode in ('restored','optimized')}
    assert witness.compare(values['restored'],values['optimized'])==[]
    for mode in values:
        assert values[mode]['instrumentation'][0]['generation_wall_seconds']>0
    a,b=(values[mode]['instrumentation'][0]['constant_calls'] for mode in ('restored','optimized'))
    assert a['dora_weight_norm']==12 and b['dora_weight_norm']==2
    assert a['embedding_materialization']>b['embedding_materialization']==1


def test_original_sidecars_preserve_historical_absence_and_explicit_false_equivalence(tmp_path):
    manifests = shards(tmp_path,[identity()]*4)
    payload = json.loads(manifests[-1].read_text())
    payload['generation_config']['static_inference_cache'] = False
    manifests[-1].write_text(json.dumps(payload))
    expected = dict(payload['generation_config']); expected.pop('static_inference_cache')
    validate_evaluation_sidecars(manifests,[tmp_path/f'shard-{index}.jsonl' for index in range(4)],
        expected_model_identity='same-model',expected_sample_count=4,
        expected_evaluation_protocol='original',expected_generation_config=expected)
    assert merge_shards(manifests,expected_sample_count=4,output_jsonl=tmp_path/'merged.jsonl',
        summary_json=tmp_path/'summary.json')['success']==4


@pytest.mark.parametrize('protocol,flags', [
    ('original',['--static-inference-cache','--use-kv-cache']),
    ('original',['--static-inference-cache','--disable-latent-generation'])])
def test_global_invalid_static_combinations_reject_before_sample_loop(tmp_path,monkeypatch,protocol,flags):
    from step_preference import capacity_evidence
    module=importlib.import_module('eval')
    _, model, calls=tiny_original([3])
    processor=NativeProcessor()
    dataset=tmp_path/'test.jsonl'; dataset.write_text('{}\n')
    output=tmp_path/'worker.jsonl'
    monkeypatch.setattr(module,'load_evaluation_artifacts',lambda **kwargs:(model,processor))
    monkeypatch.setattr(capacity_evidence,'git_identity',lambda:'a'*40)
    argv=arguments(dataset,output,tmp_path/'runtime.json')
    argv[argv.index('--evaluation-protocol')+1]=protocol
    argv[argv.index('--generation-limit-policy')+1]='native' if protocol=='original' else 'bounded'
    monkeypatch.setattr(sys,'argv',argv+flags)
    with pytest.raises(ValueError,match='static inference cache|diagnostics require explicit continuous'):
        module.main()
    assert processor.conversations==[] and calls==[] and not output.exists()


def test_continuous_invalid_static_option_stays_a_sample_error_and_failed_worker(tmp_path,monkeypatch):
    from step_preference import capacity_evidence
    from tests.stage2.helpers import tiny_stage2_qwen
    module=importlib.import_module('eval')
    model=tiny_stage2_qwen(attn_implementation='sdpa').eval()
    model.config.stage='stage1'
    processor=NativeProcessor()
    dataset=tmp_path/'test.jsonl'; dataset.write_text('{}\n')
    output=tmp_path/'worker.jsonl'
    monkeypatch.setattr(module,'load_evaluation_artifacts',lambda **kwargs:(model,processor))
    monkeypatch.setattr(module,'publish_evaluation_runtime_identity',lambda *args,**kwargs:None)
    monkeypatch.setattr(capacity_evidence,'git_identity',lambda:'a'*40)
    argv=arguments(dataset,output,tmp_path/'runtime.json')
    argv[argv.index('--evaluation-protocol')+1]='continuous'
    argv[argv.index('--generation-limit-policy')+1]='bounded'
    monkeypatch.setattr(sys,'argv',argv+['--static-inference-cache'])
    assert module.main()==1
    row,summary=[json.loads(line) for line in output.read_text().splitlines()]
    assert row['index']==0 and row['match'] is False
    assert row['error']=='static inference cache requires Stage 1 OFF with KV cache'
    assert row['inference_time_sec'] is None
    assert summary['num_samples']==1 and summary['success']==0
    assert processor.conversations==[]


def test_coordinator_cli_rejects_static_continuous_with_usage_error():
    from src.multinode import _main
    plan=_plan()
    with pytest.raises(SystemExit) as exit_code:
        _main(['evaluate','--dry-run','--run-id',plan.run_id,'--model-dir',str(plan.model_dir),
            '--test-data-path',str(plan.test_data_path),'--task-name','zebra-cot','--output-dir',str(plan.output_dir),
            '--evaluation-protocol','continuous','--static-inference-cache'])
    assert exit_code.value.code==2
