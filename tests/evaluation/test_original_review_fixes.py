"""Independent-review regressions: original sample errors and witness coverage."""
import ast
import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.evaluation.test_original_entrypoints import NativeProcessor, identity, shards
from tests.evaluation.test_original_ilvr import tiny_original, START, PAD, END
from src.multinode_evaluation import merge_shards


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


class FailingProcessor(NativeProcessor):
    def apply_chat_template(self, *args, **kwargs):
        raise ValueError('causal invalid sample prompt')


def source_error_summary(tmp_path, dataset, model):
    """Execute the pinned original main, including its actual exception handler."""
    from evaluation.original_oracle import EVAL_SOURCE, source_evaluation_functions
    original = source_evaluation_functions()
    tree = ast.parse(EVAL_SOURCE.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main']
    output = tmp_path / 'source.jsonl'
    original.update(json=json, get_eval_args=lambda: SimpleNamespace(
        model_dir='model', cache_dir='cache', test_data_path=str(dataset),
        task_name='zebra-cot', output_json_path=str(output), max_new_tokens=4,
        temperature=0., top_p=1.), load_model=lambda *args: model,
        load_processor=lambda *args: FailingProcessor())
    exec(compile(tree, str(EVAL_SOURCE), 'exec'), original)
    assert original['main']() is None
    return [json.loads(line) for line in output.read_text().splitlines()]


def setup_restored(monkeypatch, model):
    from step_preference import capacity_evidence
    module = importlib.import_module('eval')
    monkeypatch.setattr(module, 'load_evaluation_artifacts', lambda **kwargs: (model, FailingProcessor()))
    monkeypatch.setattr(module, 'publish_evaluation_runtime_identity', lambda *args, **kwargs: None)
    monkeypatch.setattr(capacity_evidence, 'git_identity', lambda: 'a' * 40)
    return module


def arguments(dataset, output, runtime, *, shard=None):
    args = ['eval.py', '--model_dir', 'model', '--test_data_path', str(dataset),
        '--task_name', 'zebra-cot', '--output_json_path', str(output),
        '--runtime_identity_path', str(runtime), '--evaluation-protocol', 'original',
        '--generation-limit-policy', 'native', '--model-identity', 'same-model']
    if shard is not None:
        args += ['--num-shards', '4', '--shard-index', str(shard)]
    return args


def test_original_single_main_sample_errors_match_source_normal_completion_and_denominator(tmp_path, monkeypatch):
    _, model, _ = tiny_original([3])
    dataset = tmp_path / 'test.jsonl'
    dataset.write_text('{}\n' * 4)
    source = source_error_summary(tmp_path, dataset, model)
    module = setup_restored(monkeypatch, model)
    output = tmp_path / 'restored.jsonl'
    monkeypatch.setattr(sys, 'argv', arguments(dataset, output, tmp_path / 'runtime.json'))
    assert module.main() is None, 'caught original sample exceptions must normal-exit like source'
    restored = [json.loads(line) for line in output.read_text().splitlines()]
    for field in ('num_samples', 'success', 'accuracy'):
        assert restored[-1][field] == source[-1][field]
    assert source[-1]['num_samples'] == 4 and source[-1]['success'] == 0
    assert [(row['index'], row['error']) for row in restored[:-1]] == [
        (row['index'], row['error']) for row in source[:-1]]


def test_four_original_worker_sample_errors_normal_exit_validate_and_merge_like_source(tmp_path, monkeypatch):
    from src.multinode_artifacts import validate_evaluation_sidecars
    _, model, _ = tiny_original([3])
    dataset = tmp_path / 'test.jsonl'
    dataset.write_text('{}\n' * 4)
    source = source_error_summary(tmp_path, dataset, model)
    module = setup_restored(monkeypatch, model)
    outputs, manifests = [], []
    for shard in range(4):
        output = tmp_path / f'shard-{shard}.jsonl'
        monkeypatch.setattr(sys, 'argv', arguments(dataset, output, tmp_path / f'runtime-{shard}.json', shard=shard))
        assert module.main() == 0, 'caught sample errors must not become worker failures in original'
        outputs.append(output)
        manifests.append(Path(str(output) + '.manifest.json'))
    validate_evaluation_sidecars(manifests, outputs, expected_model_identity='same-model', expected_sample_count=4,
        expected_evaluation_protocol='original', expected_generation_config={
            'max_new_tokens':4096, 'max_sequence_length':32768, 'temperature':0., 'top_p':1.,
            'generation_limit_policy':'native', 'vision_ablation':'none'})
    merged = tmp_path / 'merged.jsonl'
    summary = merge_shards(manifests, expected_sample_count=4, output_jsonl=merged, summary_json=tmp_path/'summary.json')
    for field in ('num_samples', 'success', 'accuracy'):
        assert summary[field] == source[-1][field]
    assert len(merged.read_text().splitlines()) == 4
    assert all('error' in json.loads(line) for line in merged.read_text().splitlines())


def test_merger_accepts_valid_original_sample_errors_but_rejects_continuous_and_invalid_errors(tmp_path):
    manifests = shards(tmp_path, [identity()] * 4)
    first = tmp_path / 'shard-0.jsonl'
    first.write_text(json.dumps(dict(index=0, task_name='zebra-cot', error='sample failure',
        inference_time_sec=None, evaluation_protocol='original')) + '\n')
    summary = merge_shards(manifests, expected_sample_count=4, output_jsonl=tmp_path/'merged.jsonl', summary_json=tmp_path/'summary.json')
    assert summary['success'] == 3 and summary['accuracy'] == .75
    first.write_text(json.dumps(dict(index=0, task_name='zebra-cot', error='failure', match=True,
        inference_time_sec=None, evaluation_protocol='original')) + '\n')
    with pytest.raises(ValueError, match='error row'):
        merge_shards(manifests, expected_sample_count=4, output_jsonl=tmp_path/'bad.jsonl', summary_json=tmp_path/'bad-summary.json')
    first.write_text(json.dumps(dict(index=0, task_name='zebra-cot', error='failure', match=False,
        inference_time_sec=None, evaluation_protocol='continuous')) + '\n')
    for path in manifests:
        payload=json.loads(path.read_text())
        payload['execution_identity']=identity('continuous')
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='error row'):
        merge_shards(manifests, expected_sample_count=4, output_jsonl=tmp_path/'continuous.jsonl', summary_json=tmp_path/'continuous-summary.json')


def observed(tokens):
    return dict(metadata=dict(parameters={'sha256':'same'},execution_identity={
        'model_config':{'latent_size':2}}),observations=[dict(index=7, generated_suffix_token_ids=tokens)])


@pytest.mark.parametrize('tokens', [[9,3], [START,PAD], [START,PAD,PAD,END], [START,END,9]])
def test_equal_witness_without_complete_natural_block_and_post_end_is_inconclusive(tokens):
    from evaluation import original_witness as witness
    assert hasattr(witness, 'comparison_verdict'), 'token equality needs a separate coverage verdict'
    result = witness.comparison_verdict(observed(tokens), observed(tokens))
    assert result['token_equality'] is True and result['observed_trace_equality'] is True
    assert result['verdict'] == 'inconclusive' and result['parity'] is None
    assert result['first_nail_passed'] is False


def test_actual_native_argmax_block_with_post_end_satisfies_witness_coverage():
    from evaluation import original_witness as witness
    assert hasattr(witness, 'comparison_verdict'), 'covered first nail needs an explicit verdict'
    api, model, _ = tiny_original([3])
    ids = torch.tensor([[7,8]])
    with torch.inference_mode():
        hidden = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
            generate_mode=True, return_dict=True).hidden_states[0,-1]
        head = nn.Linear(8, END + 1, bias=False)
        head.weight.zero_()
        head.weight[START].copy_(hidden)
        model.lm_head = head
        # START is selected by actual model logits; no processor forces START.
        generated = api.generate_original(model,dict(input_ids=ids,attention_mask=torch.ones_like(ids)),
            dict(max_new_tokens=5,do_sample=False))
    suffix = generated.sequences[0,2:].tolist()
    assert suffix[:4] == [START,PAD,PAD,END] and len(suffix) == 5
    result = witness.comparison_verdict(observed(suffix), observed(suffix))
    assert result['verdict'] == 'pass' and result['parity'] is True
    assert result['first_nail_passed'] is True
    assert result['latent_transition_coverage'] == {'source':[7], 'restored':[7]}


def test_original_manifest_write_failure_still_propagates(tmp_path, monkeypatch):
    _, model, _ = tiny_original([3])
    dataset=tmp_path/'test.jsonl'; dataset.write_text('{}\n'*4)
    module=setup_restored(monkeypatch,model)
    def fail_manifest(*args, **kwargs):
        raise OSError('causal manifest write failure')
    monkeypatch.setattr(module,'write_shard_manifest',fail_manifest)
    monkeypatch.setattr(sys,'argv',arguments(dataset,tmp_path/'shard.jsonl',tmp_path/'runtime.json',shard=0))
    with pytest.raises(OSError,match='causal manifest write failure'):
        module.main()


def test_witness_cli_equal_ordinary_tokens_reports_inconclusive_and_nonzero(tmp_path, monkeypatch):
    from evaluation import original_witness as witness
    monkeypatch.setattr(witness,'__file__',str(tmp_path/'src/evaluation/original_witness.py'))
    monkeypatch.setattr(witness,'run_mode',lambda *args,**kwargs:observed([9,3]))
    dataset=tmp_path/'test.jsonl'; dataset.write_text('{}\n')
    output=tmp_path/'results/witness'
    code=witness.main(['--model-dir','unused','--test-data-path',str(dataset),
        '--indices','7','--max-new-tokens','5','--output-dir',str(output)])
    assert code==2, 'an uncovered first nail must not return the success exit code'
    report=json.loads((output/'comparison.json').read_text())
    assert report['token_equality'] is True and report['parity'] is None
    assert report['verdict']=='inconclusive' and report['first_nail_passed'] is False


@pytest.mark.parametrize('flag', ['--disable-latent-generation', '--use-kv-cache'])
def test_original_global_diagnostic_flags_fail_before_sample_processing_or_output(tmp_path, monkeypatch, flag):
    from step_preference import capacity_evidence
    module = importlib.import_module('eval')
    _, model, decoder_calls = tiny_original([3])
    processor = NativeProcessor()
    dataset = tmp_path/'test.jsonl'
    dataset.write_text('{}\n')
    output = tmp_path/'result.jsonl'
    publications = []
    monkeypatch.setattr(module, 'load_evaluation_artifacts', lambda **kwargs: (model, processor))
    monkeypatch.setattr(module, 'publish_evaluation_runtime_identity', lambda *args, **kwargs: publications.append(True))
    monkeypatch.setattr(capacity_evidence, 'git_identity', lambda: 'a'*40)
    monkeypatch.setattr(sys, 'argv', arguments(dataset, output, tmp_path/'runtime.json') + [flag])
    with pytest.raises(ValueError, match='diagnostics require explicit continuous'):
        module.main()
    assert processor.conversations == []
    assert decoder_calls == []
    assert publications == []
    assert not output.exists()
