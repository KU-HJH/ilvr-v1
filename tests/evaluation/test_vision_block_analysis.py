"""CLI orchestration tests; replace only GPU/model and processor work."""
import copy
from contextlib import contextmanager
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evaluation import vision_block_analysis as cli
from evaluation.vision_block_inputs import PreparedSample, Site


REAL_LOAD_RUNTIME = cli._load_runtime
MATH_POLICY = dict(sdpa_math=True, sdpa_flash=False, sdpa_mem_efficient=False,
                   sdpa_cudnn=False, tf32_matmul=False, tf32_cudnn=False)


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    for name in ('data', 'results', 'logs', 'checkpoints'):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(cli, '_repository_root', lambda: tmp_path)
    sites = (Site(4, 1, 0, 1, 9, 9, (10, 11), (20, 21)),
             Site(4, 3, 12, 13, 21, 21, (22,), (30,)))
    samples = [PreparedSample(4, {}, sites),
               PreparedSample(2, {}, (Site(2, 1, 0, 1, 9, 9, (10,), (10,)),))]
    manifest = {'schema_version': 1, 'S': 3, 'N': 4, 'exclusions': [],
                'source': {'sha256': '6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618'},
                'selection': {'sha256': 'fb1f15fef5339abde6ba74d140e5f9a1f8823cedafe8aef1d4bb245a8e0286aa',
                              'requested_indices': list(range(100)),
                              'category_counts': {'creation': 25, 'deletion': 30, 'update': 15, 'selection': 30}},
                'processor': {'sha256': 'processor'},
                'samples': [{'source_index': s.source_index,
                             'sites': [asdict(site) for site in s.sites]} for s in samples]}
    manifest_path = tmp_path / 'data/manifest.json'
    model = tmp_path / 'checkpoints/model'
    model.mkdir()
    monkeypatch.setattr(cli, '_load_processor', lambda args: object())
    def prepare(args, processor):
        args.manifest.write_text(json.dumps(manifest))
        return manifest
    monkeypatch.setattr(cli, '_prepare', prepare)
    runtime_identity = {'parameter_dtypes': {'torch.bfloat16': 1},
                        'numerical_policy': copy.deepcopy(MATH_POLICY),
                        'tf32_matmul': False, 'tf32_cudnn': False,
                        'execution_strategy': 'full_length_v2', 'decoder_layers': 2}
    monkeypatch.setattr(cli, '_load_runtime', lambda args: (object(), object(), copy.deepcopy(runtime_identity)))
    monkeypatch.setattr(cli, '_processor_identity', lambda processor: {'sha256': 'processor'})
    identity = {'model': 'model-v1', 'code': 'code-v1', 'software': 'software-v1',
                'runtime': copy.deepcopy(runtime_identity)}
    monkeypatch.setattr(cli, '_execution_identity', lambda *args: copy.deepcopy(identity))
    monkeypatch.setattr(cli, '_samples', lambda *args: iter(samples))
    def analyze(model, sample, repeat_clean=False):
        rows = []
        for condition in ('clean', 'body_mask', 'boundary_mask', 'full_mask'):
            for site in sample.sites:
                for pos, target in zip(site.target_positions, site.target_ids):
                    rows.append(dict(source_index=sample.source_index, plan_index=site.plan_index,
                        target_position=pos, target_id=target, condition=condition,
                        argmax_id=target + int(condition == 'full_mask'),
                        nll=1.0 + int(condition == 'full_mask')))
        comparison = dict(tokens=len(rows)//4, argmax_mismatches=0,
                          max_nll_abs_diff=0., mean_nll_abs_diff=0.)
        witness = {key: comparison for key in ('normal_vs_captured', 'normal_vs_replay',
                   'captured_vs_replay', 'repeat_clean', 'repeat_normal', 'repeat_captured')}
        records = [dict(plan_index=site.plan_index, condition=condition,
                        prefix_length=site.end, layers=2, kv_comparisons=4, exact=True)
                   for condition in ('clean', 'body_mask', 'boundary_mask', 'full_mask')
                   for site in sample.sites]
        witness.update(finite=True, visibility_positions=True, source_index=sample.source_index,
                       positions_shape=[3, 1, 24], embeds_shape=[1, 24, 8],
                       execution_strategy='full_length_v2', numerical_policy=copy.deepcopy(MATH_POLICY),
                       prefix_identity=records,
                       repeat_prefix_identity=[record for record in records if record['condition'] == 'clean'])
        return rows, witness
    monkeypatch.setattr(cli, '_analyze', analyze)
    common = ['--model-dir', str(model), '--base-model-dir', str(model),
              '--manifest', str(manifest_path), '--image-root', str(tmp_path/'data')]
    def invoke(command, output='main', extra=()):
        args = [command, '--manifest', str(manifest_path)] if command == 'summarize' else [command, *common]
        if command == 'prepare':
            args += ['--data', str(tmp_path/'data/TEST.jsonl'), '--selection', str(tmp_path/'data/selection.json')]
        else:
            args += ['--output-dir', str(tmp_path/'results'/output)]
        if command == 'run':
            args += ['--smoke-report', str(tmp_path/'results/smoke/smoke.json')]
        return cli.main([*args, *extra])
    return tmp_path, manifest, samples, identity, invoke, analyze


@pytest.mark.parametrize('mutation', [
    'missing_prefix', 'duplicate_prefix', 'wrong_comparisons', 'inexact_prefix',
    'missing_repeat_prefix', 'old_strategy', 'math_disabled', 'flash_enabled',
    'mem_efficient_enabled', 'cudnn_enabled', 'matmul_tf32_enabled', 'cudnn_tf32_enabled',
])
def test_witness_rejects_incomplete_or_wrong_v2_identity(cohort, mutation):
    # Catches accepting an old/partial smoke as authorization for the main run.
    _, _, samples, _, _, analyze = cohort
    _, witness = analyze(None, samples[0], repeat_clean=True)
    if mutation == 'missing_prefix':
        witness['prefix_identity'].pop()
    elif mutation == 'duplicate_prefix':
        witness['prefix_identity'][-1] = copy.deepcopy(witness['prefix_identity'][0])
    elif mutation == 'wrong_comparisons':
        witness['prefix_identity'][-1]['kv_comparisons'] = 3
    elif mutation == 'inexact_prefix':
        witness['prefix_identity'][-1]['exact'] = False
    elif mutation == 'missing_repeat_prefix':
        witness['repeat_prefix_identity'].pop()
    elif mutation == 'old_strategy':
        witness['execution_strategy'] = 'short_suffix_v1'
    else:
        flag, value = {
            'math_disabled': ('sdpa_math', False),
            'flash_enabled': ('sdpa_flash', True),
            'mem_efficient_enabled': ('sdpa_mem_efficient', True),
            'cudnn_enabled': ('sdpa_cudnn', True),
            'matmul_tf32_enabled': ('tf32_matmul', True),
            'cudnn_tf32_enabled': ('tf32_cudnn', True),
        }[mutation]
        witness['numerical_policy'][flag] = value
    with pytest.raises(ValueError):
        cli._validate_witness(witness, sites=samples[0].sites, layers=2, repeat=True)


def test_inference_policy_covers_loading_identity_and_analysis(cohort, monkeypatch):
    # Catches a policy entered only inside replay, after model loading/identity capture.
    _, _, _, _, invoke, _ = cohort
    assert invoke('prepare') == 0
    runtime = importlib.import_module('evaluation.vision_block_runtime')
    active = []

    @contextmanager
    def policy():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setattr(runtime, 'numerical_policy', policy)
    monkeypatch.setattr(runtime, 'numerical_policy_identity', lambda: copy.deepcopy(MATH_POLICY))
    for name in ('_load_runtime', '_execution_identity', '_analyze'):
        original = getattr(cli, name)
        def checked(*args, _original=original, **kwargs):
            assert active
            return _original(*args, **kwargs)
        monkeypatch.setattr(cli, name, checked)
    assert invoke('smoke', 'smoke') == 0
    assert active == []


def test_loader_tf32_side_effect_is_reenforced_before_identity(cohort, monkeypatch):
    # Catches eval.load_model enabling TF32 after the outer policy was entered.
    import torch
    import eval as eval_module
    from stage2 import attention_backend, peft

    root, _, _, identity, invoke, _ = cohort
    assert invoke('prepare') == 0
    weight = torch.zeros(1, dtype=torch.bfloat16)
    qwen = SimpleNamespace(
        config=SimpleNamespace(stage='stage2', latent_size=8, _attn_implementation='sdpa'),
        visual=SimpleNamespace(config=SimpleNamespace(_attn_implementation='flash_attention_2')),
        model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight), layers=[object(), object()]))

    class BoundaryModel:
        def eval(self):
            return self

        def parameters(self):
            return iter((weight,))

    model = BoundaryModel()
    def load_artifacts(**kwargs):
        torch.backends.cuda.matmul.allow_tf32 = True
        return model, object()

    monkeypatch.setattr(cli, '_load_runtime', REAL_LOAD_RUNTIME)
    monkeypatch.setattr(eval_module, 'load_evaluation_artifacts', load_artifacts)
    monkeypatch.setattr(peft, 'resolve_stage2_qwen_model', lambda loaded: qwen)
    monkeypatch.setattr(attention_backend, 'validate_qwen2_5_vl_attention_backends', lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'test-gpu')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(cli, '_execution_identity', lambda args, processor, runtime: {
        **copy.deepcopy(identity), 'runtime': copy.deepcopy(runtime)})
    ambient = torch.backends.cuda.matmul.allow_tf32
    try:
        assert invoke('smoke', 'smoke') == 0
        runtime_identity = read(root/'results/smoke/identity.json')['runtime']
        assert runtime_identity['numerical_policy'] == MATH_POLICY
        assert runtime_identity['tf32_matmul'] is False
        assert runtime_identity['tf32_cudnn'] is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = ambient
    assert torch.backends.cuda.matmul.allow_tf32 is ambient


def test_clean_case_witness_cannot_redeem_wrong_runtime_identity(cohort):
    # Catches smoke passing when runtime identity says TF32 despite clean case witnesses.
    root, _, _, identity, invoke, _ = cohort
    assert invoke('prepare') == 0
    identity['runtime']['tf32_matmul'] = True
    identity['runtime']['numerical_policy']['tf32_matmul'] = True
    assert invoke('smoke', 'smoke') == 1
    assert read(root/'results/smoke/smoke.json')['status'] == 'failed'


@pytest.mark.parametrize('two_sites', [False, True])
def test_actual_analyze_uses_full_length_v2_witnesses(two_sites):
    # Catches a correct runtime hidden behind stale CLI shape/visibility assertions.
    from tests.evaluation.test_vision_block_runtime import image_case

    model, sample = image_case(two_sites)
    rows, witness = cli._analyze(model, sample, repeat_clean=True)
    assert len(rows) == 4 * sum(len(site.target_ids) for site in sample.sites)
    cli._validate_witness(witness, sites=sample.sites, layers=2, repeat=True)
    assert witness['visibility_positions'] is True


def test_unknown_command_fails_before_loading_model():
    with pytest.raises(SystemExit) as error:
        cli.main(['unknown-analysis'])
    assert error.value.code == 2


def test_prepare_smoke_run_and_offline_metrics(cohort):
    root, _, _, _, invoke, _ = cohort
    assert invoke('prepare') == 0
    assert invoke('smoke', 'smoke') == 0
    gate = read(root/'results/smoke/smoke.json')
    assert gate['status'] == 'passed'
    assert gate['identity']['runtime']['numerical_policy'] == MATH_POLICY
    assert gate['identity']['runtime']['execution_strategy'] == 'full_length_v2'
    assert gate['identity']['runtime']['decoder_layers'] == 2
    assert [(c['source_index'], c['plan_indices']) for c in gate['cases']] == [(2, [1]), (4, [1, 3])]
    assert invoke('run') == 0
    assert invoke('summarize') == 0
    summary = read(root/'results/main/summary.json')
    assert summary['sites'] == 3 and summary['tokens'] == 4
    assert summary['conditions']['clean']['mean_nll'] == 1.
    assert summary['conditions']['full_mask'] == dict(first_changed=3, any_changed=3,
        tokens_changed=4, mean_nll=2., delta_nll=1.)
    assert read(root/'results/main/status.json')['status'] == 'completed'
    assert '3/3 (100.00%)' in (root/'results/main/report.md').read_text()
    assert (root/'logs/main/execution.log').is_file()
    assert len((root/'results/main/sites.jsonl').read_text().splitlines()) == 3


@pytest.mark.parametrize('change', [
    'model', 'code', 'policy', 'strategy', 'layers', 'manifest', 'failed',
])
def test_changed_identity_or_failed_gate_blocks_run(cohort, change):
    root, _, _, identity, invoke, _ = cohort
    invoke('prepare'); invoke('smoke', 'smoke')
    if change in identity:
        identity[change] = 'changed'
    elif change == 'policy':
        identity['runtime']['numerical_policy']['sdpa_flash'] = True
    elif change == 'strategy':
        identity['runtime']['execution_strategy'] = 'short_suffix_v1'
    elif change == 'layers':
        identity['runtime']['decoder_layers'] = 3
    elif change == 'manifest':
        manifest = read(root/'data/manifest.json')
        manifest['exclusions'] = [{'reason': 'changed'}]
        (root/'data/manifest.json').write_text(json.dumps(manifest))
    else:
        path = root/'results/smoke/smoke.json'
        gate = read(path); gate['status'] = 'failed'; path.write_text(json.dumps(gate))
    assert invoke('run') == 1
    assert not (root/'results/main/summary.json').exists()


def test_incomplete_conditions_remain_incomplete_offline(cohort, monkeypatch):
    root, _, _, _, invoke, analyze = cohort
    invoke('prepare'); invoke('smoke', 'smoke')
    monkeypatch.setattr(cli, '_analyze', lambda *a, **kw: (analyze(*a, **kw)[0][:-1], analyze(*a, **kw)[1]))
    assert invoke('run') == 1
    assert read(root/'results/main/status.json')['status'] == 'incomplete'
    assert invoke('summarize') == 1
    assert not (root/'results/main/summary.json').exists()


def test_offline_refuses_complete_rows_from_failed_run(cohort):
    root, _, _, _, invoke, _ = cohort
    invoke('prepare'); invoke('smoke', 'smoke'); invoke('run')
    path = root/'results/main/status.json'
    status = read(path); status['status'] = 'incomplete'; path.write_text(json.dumps(status))
    assert invoke('summarize') == 1


@pytest.mark.parametrize('bad_flag', ['--output-dir', '--log-dir'])
def test_storage_escape_rejected(cohort, bad_flag):
    root, _, _, _, invoke, _ = cohort
    invoke('prepare')
    assert invoke('smoke', 'smoke', [bad_flag, str(root/'outside')]) == 1
    assert not (root/'outside').exists()


def test_existing_output_is_not_overwritten(cohort):
    root, _, _, _, invoke, _ = cohort
    invoke('prepare'); invoke('smoke', 'smoke')
    before = (root/'results/smoke/smoke.json').read_bytes()
    assert invoke('smoke', 'smoke') == 1
    assert (root/'results/smoke/smoke.json').read_bytes() == before
    assert invoke('prepare') == 1


def test_smoke_fallback_and_missing_multisite(cohort):
    root, _, samples, _, invoke, _ = cohort
    invoke('prepare')
    samples.pop()  # no valid single-site sample: branch from full first multi-site prefix
    manifest = read(root/'data/manifest.json')
    manifest['samples'] = manifest['samples'][:1]
    manifest.update(S=2, N=3)
    (root/'data/manifest.json').write_text(json.dumps(manifest))
    assert invoke('smoke', 'smoke') == 0
    gate = read(root/'results/smoke/smoke.json')
    assert [(c['source_index'], c['plan_indices']) for c in gate['cases']] == [(4, [1]), (4, [1, 3])]
    samples[:] = [PreparedSample(4, {}, samples[0].sites[:1])]
    assert invoke('smoke', 'no-multi') == 1
    assert read(root/'results/no-multi/smoke.json')['status'] == 'failed'


def test_failed_fidelity_blocks_smoke(cohort, monkeypatch):
    root, _, _, _, invoke, analyze = cohort
    invoke('prepare')
    def bad(*args, **kwargs):
        rows, witness = analyze(*args, **kwargs)
        witness['repeat_clean'] = dict(tokens=1, argmax_mismatches=1,
            max_nll_abs_diff=0., mean_nll_abs_diff=0.)
        return rows, witness
    monkeypatch.setattr(cli, '_analyze', bad)
    assert invoke('smoke', 'smoke') == 1
    assert read(root/'results/smoke/smoke.json')['status'] == 'failed'


def test_summarize_imports_no_model_dependencies():
    code = '''
import sys
from evaluation.vision_block_analysis import main
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'eval' not in sys.modules
try:
    main(['summarize', '--help'])
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_smoke_cannot_authorize_different_manifest_sites(cohort):
    root, _, _, _, invoke, _ = cohort
    invoke('prepare'); invoke('smoke', 'smoke')
    path = root/'results/smoke/smoke.json'
    gate = read(path)
    gate['cases'][1]['source_index'] = 987
    path.write_text(json.dumps(gate))
    assert invoke('run') == 1
    assert read(root/'results/main/status.json')['status'] == 'incomplete'


def test_smoke_rejects_manifest_outside_fixed_selection(cohort):
    root, _, _, _, invoke, _ = cohort
    invoke('prepare')
    path = root/'data/manifest.json'
    manifest = read(path)
    manifest['source'] = {'path': 'TEST.jsonl', 'sha256': 'changed-data'}
    path.write_text(json.dumps(manifest))
    assert invoke('smoke', 'smoke') == 1


def test_image_root_must_use_repository_data(cohort):
    root, _, _, _, invoke, _ = cohort
    assert invoke('prepare', extra=['--image-root', str(root/'outside')]) == 1
    assert not (root/'data/manifest.json').exists()
