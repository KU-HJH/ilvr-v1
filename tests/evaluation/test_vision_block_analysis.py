"""CLI orchestration tests; replace only GPU/model and processor work."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from evaluation import vision_block_analysis as cli
from evaluation.vision_block_inputs import PreparedSample, Site


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
    monkeypatch.setattr(cli, '_load_runtime', lambda args: (object(), object(), {'dtype': 'bf16'}))
    monkeypatch.setattr(cli, '_processor_identity', lambda processor: {'sha256': 'processor'})
    identity = {'model': 'model-v1', 'code': 'code-v1', 'software': 'software-v1'}
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
        witness.update(finite=True, visibility_positions=True, source_index=sample.source_index,
                       positions_shape=[3, 1, 24], embeds_shape=[1, 24, 8])
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


@pytest.mark.parametrize('change', ['model', 'code', 'manifest', 'failed'])
def test_changed_identity_or_failed_gate_blocks_run(cohort, change):
    root, _, _, identity, invoke, _ = cohort
    invoke('prepare'); invoke('smoke', 'smoke')
    if change in identity:
        identity[change] = 'changed'
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
