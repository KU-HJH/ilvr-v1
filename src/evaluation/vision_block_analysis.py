"""Fixed CE vision-block analysis; offline reporting imports only the stdlib."""
import argparse
import csv
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from evaluation.vision_block_metrics import CONDITIONS, summarize


MATH_POLICY = dict(sdpa_math=True, sdpa_flash=False, sdpa_mem_efficient=False,
                   sdpa_cudnn=False, tf32_matmul=False, tf32_cudnn=False)


def _repository_root():
    return Path(__file__).resolve().parents[2]


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _within(path, tree):
    path, tree = Path(path).resolve(), Path(tree).resolve()
    if path == tree or not path.is_relative_to(tree):
        raise ValueError(f'path must belong to {tree}: {path}')
    return path


def _expected(manifest):
    expected = {}
    for sample in manifest['samples']:
        for site in sample['sites']:
            key = (sample['source_index'], site['plan_index'])
            if key in expected or site['source_index'] != key[0]:
                raise ValueError('duplicate or inconsistent manifest site')
            if len(site['target_positions']) != len(site['target_ids']):
                raise ValueError('manifest target identity lengths differ')
            expected[key] = tuple(zip(site['target_positions'], site['target_ids']))
    if manifest['S'] != len(expected) or manifest['N'] != sum(map(len, expected.values())):
        raise ValueError('manifest S/N denominators do not match sites/tokens')
    if not expected:
        raise ValueError('manifest has no valid sites')
    return expected


def _load_processor(args):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(str(args.model_dir), local_files_only=True,
                                        cache_dir=str(_repository_root()/'data/cache'))


def _prepare(args, processor):
    from evaluation.vision_block_inputs import prepare_manifest
    return prepare_manifest(args.data, args.selection, processor, args.image_root, args.manifest)


def _processor_identity(processor):
    from evaluation.vision_block_inputs import _processor_identity as identify
    return identify(processor)


def _load_runtime(args):
    import torch
    from eval import load_evaluation_artifacts
    from evaluation.vision_block_runtime import numerical_policy_identity
    from stage2.peft import resolve_stage2_qwen_model
    from stage2.attention_backend import validate_qwen2_5_vl_attention_backends

    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 1 or not visible[0].strip() or torch.cuda.device_count() != 1:
        raise ValueError('expose exactly one GPU using CUDA_VISIBLE_DEVICES')
    model, processor = load_evaluation_artifacts(model_dir=str(args.model_dir),
        base_model_dir=str(args.base_model_dir), cache_dir=str(_repository_root()/'data/cache'))
    qwen = resolve_stage2_qwen_model(model)
    if qwen.config.stage != 'stage2' or qwen.config.latent_size != 8:
        raise ValueError('requires Stage 2 artifact with latent size 8')
    validate_qwen2_5_vl_attention_backends(qwen, attention_implementation='sdpa',
                                         vision_attention_implementation='flash_attention_2')
    if qwen.model.embed_tokens.weight.dtype != torch.bfloat16:
        raise ValueError('requires BF16 language model')
    model.eval()
    dtypes = {}
    for parameter in model.parameters():
        dtype = str(parameter.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + parameter.numel()
    runtime = {'parameter_dtypes': dtypes, 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0),
        'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'language_backend': qwen.config._attn_implementation,
        'vision_backend': qwen.visual.config._attn_implementation,
        'stage': qwen.config.stage, 'latent_size': qwen.config.latent_size,
        'numerical_policy': numerical_policy_identity(),
        'execution_strategy': 'full_length_v2',
        'decoder_layers': len(qwen.model.layers)}
    return model, processor, runtime


def _directory_identity(path):
    path = Path(path).resolve()
    files = {str(file.relative_to(path)): _hash(file) for file in sorted(path.rglob('*')) if file.is_file()}
    if not files:
        raise ValueError(f'empty artifact directory: {path}')
    return {'path': str(path), 'files': files}


def _execution_identity(args, processor, runtime):
    root = _repository_root()
    names = subprocess.check_output(['git', 'ls-files', '-z', 'src', 'transformers/src', 'eval.py',
                                      'scripts/analyze_vision_block.py'], cwd=root).decode().split('\0')
    code = {name: _hash(root/name) for name in names if name.endswith('.py')}
    return {'model': _directory_identity(args.model_dir),
            'base_model': _directory_identity(args.base_model_dir),
            'processor': _processor_identity(processor), 'runtime': runtime,
            'code': {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
                     'files': code},
            'software': {name: importlib.metadata.version(name) for name in
                         ('torch', 'transformers', 'peft', 'flash-attn', 'numpy')},
            'python': sys.version}


def _samples(manifest, processor, image_root):
    from evaluation.vision_block_inputs import load_prepared_samples
    return load_prepared_samples(manifest, processor, image_root)


def _analyze(model, sample, repeat_clean=False):
    import torch
    from evaluation.vision_block_runtime import analyze_sample, capture_clean, build_replay_inputs
    with torch.inference_mode():
        if repeat_clean:
            # Verify masks and original position columns on real captured GPU tensors.
            state = capture_clean(model, sample)
            for site in sample.sites:
                for condition in CONDITIONS:
                    embeds, positions, mask = build_replay_inputs(state, site, condition)
                    length = state.embeds.shape[1]
                    if not torch.equal(positions, state.positions):
                        raise ValueError('replay changed original position columns')
                    blocked = set(range(site.body_start, site.body_end)) if condition in ('body_mask', 'full_mask') else set()
                    if condition in ('boundary_mask', 'full_mask'):
                        blocked.add(site.start)
                    expected = torch.tensor([
                        [key <= query and (query < site.end or key not in blocked)
                         for key in range(length)]
                        for query in range(length)], device=mask.device)
                    if not torch.equal(torch.isfinite(mask[0, 0]), expected):
                        raise ValueError('replay visibility differs from literal site contract')
                    expected_embeds = state.embeds.clone()
                    if condition in ('boundary_mask', 'full_mask'):
                        expected_embeds[:, site.end] = 0
                    if not torch.equal(embeds, expected_embeds):
                        raise ValueError('replayed END embedding differs from condition contract')
            del state
        rows, witness = analyze_sample(model, sample, repeat_clean=repeat_clean)
        if repeat_clean:
            witness['visibility_positions'] = True
        return rows, witness


def _site_field(site, name):
    return site[name] if isinstance(site, dict) else getattr(site, name)


def _validate_prefix_records(records, sites, layers, conditions):
    expected = {(_site_field(site, 'plan_index'), condition): _site_field(site, 'end')
                for condition in conditions for site in sites}
    if not isinstance(records, list):
        raise ValueError('incomplete prefix identity coverage')
    try:
        actual = {(record['plan_index'], record['condition']): record for record in records}
    except (KeyError, TypeError):
        raise ValueError('incomplete prefix identity coverage') from None
    if len(actual) != len(records) or actual.keys() != expected.keys():
        raise ValueError('incomplete prefix identity coverage')
    for key, end in expected.items():
        record = actual[key]
        if (record.get('prefix_length') != end or record.get('layers') != layers
                or record.get('kv_comparisons') != 2 * layers or record.get('exact') is not True):
            raise ValueError('failed prefix identity witness')


def _validate_witness(witness, *, sites, layers, repeat=False):
    if witness.get('finite') is not True:
        raise ValueError('nonfinite or missing runtime witness')
    if witness['positions_shape'] != [3, 1, witness['embeds_shape'][1]]:
        raise ValueError('invalid three-axis position witness')
    if witness.get('execution_strategy') != 'full_length_v2':
        raise ValueError('invalid execution strategy witness')
    if witness.get('numerical_policy') != MATH_POLICY:
        raise ValueError('invalid numerical policy witness')
    _validate_prefix_records(witness.get('prefix_identity'), sites, layers, CONDITIONS)
    names = ['normal_vs_captured', 'normal_vs_replay', 'captured_vs_replay']
    if repeat:
        names += ['repeat_clean', 'repeat_normal', 'repeat_captured']
        if witness.get('visibility_positions') is not True:
            raise ValueError('missing visibility/position assertions')
        _validate_prefix_records(witness.get('repeat_prefix_identity'), sites, layers, ('clean',))
    for name in names:
        comparison = witness.get(name)
        if not comparison or comparison['tokens'] <= 0 or comparison['argmax_mismatches'] != 0:
            raise ValueError(f'failed fidelity gate: {name}')
        for field, limit in [('max_nll_abs_diff', .01), ('mean_nll_abs_diff', .001)]:
            value = comparison[field]
            if not math.isfinite(value) or not 0 <= value <= limit:
                raise ValueError(f'failed fidelity gate: {name}/{field}')


def _case_expected(sample):
    return {(sample.source_index, s.plan_index): tuple(zip(s.target_positions, s.target_ids)) for s in sample.sites}


def _check_protocol(manifest):
    from evaluation.vision_block_inputs import EXPECTED_DATA_SHA256, EXPECTED_SELECTION_SHA256, EXPECTED_CATEGORY_COUNTS
    if (manifest['source']['sha256'] != EXPECTED_DATA_SHA256
            or manifest['selection']['sha256'] != EXPECTED_SELECTION_SHA256
            or manifest['selection']['category_counts'] != EXPECTED_CATEGORY_COUNTS):
        raise ValueError('manifest differs from fixed approved data/selection protocol')
    indices = manifest['selection']['requested_indices']
    if (len(indices) != 100 or any(type(i) is not int or i < 0 for i in indices)
            or len(set(indices)) != 100
            or any(s['source_index'] not in indices for s in manifest['samples'])):
        raise ValueError('manifest does not preserve fixed 100-sample selection')


def _smoke(args, manifest, model, processor, identity, layers):
    single = multi = None
    ordered = dict(manifest, samples=sorted(manifest['samples'], key=lambda s: s['source_index']))
    for sample in _samples(ordered, processor, args.image_root):
        if len(sample.sites) == 1 and (single is None or sample.source_index < single.source_index):
            single = sample
        if len(sample.sites) > 1 and (multi is None or sample.source_index < multi.source_index):
            multi = sample
    if multi is None:
        raise ValueError('smoke requires a valid multi-site sample')
    fallback = single is None
    if fallback:
        single = replace(multi, sites=multi.sites[:1])
    cases = []
    for label, sample in [('single', single), ('multi', multi)]:
        rows, witness = _analyze(model, sample, repeat_clean=True)
        _validate_witness(witness, sites=sample.sites, layers=layers, repeat=True)
        summarize(rows, _case_expected(sample))
        cases.append({'case': label, 'source_index': sample.source_index,
                      'plan_indices': [s.plan_index for s in sample.sites], 'witness': witness})
    _write(args.output_dir/'smoke.json', {'status': 'passed', 'identity': identity,
        'manifest_sha256': _hash(args.manifest), 'single_site_fallback': fallback, 'cases': cases})


def _validate_gate(gate, identity, manifest_hash, manifest):
    if gate.get('status') != 'passed':
        raise ValueError('smoke gate did not pass')
    if gate.get('identity') != identity or gate.get('manifest_sha256') != manifest_hash:
        raise ValueError('smoke execution/manifest identity changed')
    cases = gate.get('cases', [])
    if len(cases) != 2 or [case['case'] for case in cases] != ['single', 'multi']:
        raise ValueError('smoke lacks both required cases')
    if len(cases[0]['plan_indices']) != 1 or len(cases[1]['plan_indices']) < 2:
        raise ValueError('smoke case site counts are invalid')
    ordered = sorted(manifest['samples'], key=lambda s: s['source_index'])
    multi = next((s for s in ordered if len(s['sites']) > 1), None)
    single = next((s for s in ordered if len(s['sites']) == 1), multi)
    if multi is None:
        raise ValueError('manifest lacks required multi-site smoke case')
    expected_cases = [(single['source_index'], [single['sites'][0]['plan_index']]),
                      (multi['source_index'], [s['plan_index'] for s in multi['sites']])]
    if [(c['source_index'], c['plan_indices']) for c in cases] != expected_cases:
        raise ValueError('smoke tested different manifest sites')
    layers = identity.get('runtime', {}).get('decoder_layers')
    if type(layers) is not int or layers <= 0:
        raise ValueError('invalid decoder layer identity')
    expected_sites = ([single['sites'][0]], multi['sites'])
    for case, sites in zip(cases, expected_sites):
        _validate_witness(case['witness'], sites=sites, layers=layers, repeat=True)


def _report(args, manifest, expected):
    status = _read(args.output_dir/'status.json')
    if status.get('status') != 'completed' or status.get('manifest_sha256') != _hash(args.manifest):
        raise ValueError('offline summary requires completed inference for the same manifest')
    rows = [json.loads(line) for line in (args.output_dir/'tokens.jsonl').read_text().splitlines() if line.strip()]
    summary = summarize(rows, expected)
    summary['exclusions'] = manifest['exclusions']
    _write(args.output_dir/'summary.json', summary)
    with (args.output_dir/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['condition', 'S', 'N', 'first_changed', 'any_changed', 'tokens_changed', 'mean_nll', 'delta_nll'])
        for condition, metrics in summary['conditions'].items():
            writer.writerow([condition, summary['sites'], summary['tokens'], *metrics.values()])
    lines = ['# CE vision-block 분석', '',
             f"S={summary['sites']} 사이트, N={summary['tokens']} 토큰; 제외 항목={len(manifest['exclusions'])}.", '',
             '| 조건 | 첫 토큰 변화 | 사이트 내 하나 이상 변화 | 전체 토큰 변화 | 평균 NLL (nat) | ΔNLL (clean 대비) |',
             '|---|---:|---:|---:|---:|---:|']
    for condition, metrics in summary['conditions'].items():
        fractions = [f'{metrics[key]}/{denominator} ({100*metrics[key]/denominator:.2f}%)'
                     for key, denominator in [('first_changed', summary['sites']), ('any_changed', summary['sites']),
                                              ('tokens_changed', summary['tokens'])]]
        lines.append(f"| {condition} | {' | '.join(fractions)} | {metrics['mean_nll']:.6f} | {metrics['delta_nll']:+.6f} |")
    lines += ['', '각 사이트는 동일한 clean prefix와 정답 토큰을 사용하여 독립적으로 teacher forcing했습니다. '
              'NLL은 FP32 log-softmax의 자연로그 토큰 평균이며 ΔNLL은 조건 평균 − clean 평균입니다.', '',
              '원본 이미지 접근은 유지되며, 앞선 hidden/KV 및 다른 Body의 이미지 정보는 남아 있습니다. '
              '따라서 국소 attention 차단은 전체 이미지 정보 제거를 뜻하지 않습니다. '
              'Body는 clean 실행에서 한 번 생성한 이력을 재사용하며 조건별로 재생성하지 않습니다.', '']
    (args.output_dir/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'smoke', 'run', 'summarize'):
        command = commands.add_parser(name)
        command.add_argument('--manifest', type=Path, required=True)
        if name != 'summarize':
            command.add_argument('--model-dir', type=Path, required=True)
            command.add_argument('--base-model-dir', type=Path, required=True)
            command.add_argument('--image-root', type=Path, required=True)
            command.add_argument('--data', type=Path, required=name == 'prepare')
            command.add_argument('--selection', type=Path, required=name == 'prepare')
        if name != 'prepare':
            command.add_argument('--output-dir', type=Path, required=True)
            command.add_argument('--log-dir', type=Path)
        if name == 'run':
            command.add_argument('--smoke-report', type=Path, required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    root = _repository_root()
    log_path = None
    owned_output = False
    try:
        args.manifest = _within(args.manifest, root/'data')
        if args.command != 'summarize':
            args.model_dir = _within(args.model_dir, root/'checkpoints')
            args.base_model_dir = _within(args.base_model_dir, root/'checkpoints')
            args.image_root = args.image_root.resolve()
            if not args.image_root.is_relative_to((root/'data').resolve()):
                raise ValueError('image root must belong to repository data tree')
            if args.data is not None:
                args.data = _within(args.data, root/'data')
        if args.command == 'prepare':
            if args.manifest.exists():
                raise ValueError('manifest already exists; use a fresh run ID')
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest = _prepare(args, _load_processor(args))
            _expected(manifest)
            print(json.dumps({'S': manifest['S'], 'N': manifest['N'], 'exclusions': len(manifest['exclusions'])}))
            return 0
        args.output_dir = _within(args.output_dir, root/'results')
        relative = args.output_dir.relative_to((root/'results').resolve())
        args.log_dir = _within(args.log_dir or root/'logs'/relative, root/'logs')
        manifest = _read(args.manifest)
        expected = _expected(manifest)
        if args.command == 'summarize':
            _report(args, manifest, expected)
            return 0
        _check_protocol(manifest)
        for directory in (args.output_dir, args.log_dir):
            if directory.exists() and any(directory.iterdir()):
                raise ValueError(f'nonempty run directory; use a fresh run ID: {directory}')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir/'execution.log'
        owned_output = True
        log_path.write_text(f'{args.command} starting\n', encoding='utf-8')
        if args.command == 'run':
            _write(args.output_dir/'status.json', {'status': 'incomplete', 'manifest_sha256': _hash(args.manifest)})
            gate = _read(_within(args.smoke_report, root/'results'))
            if gate.get('status') != 'passed' or gate.get('manifest_sha256') != _hash(args.manifest):
                raise ValueError('smoke gate failed or manifest identity changed')
        from evaluation.vision_block_runtime import numerical_policy
        with numerical_policy():
            model, processor, runtime = _load_runtime(args)
            if _processor_identity(processor) != manifest['processor']:
                raise ValueError('manifest processor identity changed')
            identity = _execution_identity(args, processor, runtime)
            _write(args.output_dir/'identity.json', identity)
            layers = runtime.get('decoder_layers')
            if type(layers) is not int or layers <= 0:
                raise ValueError('invalid decoder layer identity')
            if args.command == 'smoke':
                _smoke(args, manifest, model, processor, identity, layers)
            else:
                _validate_gate(gate, identity, _hash(args.manifest), manifest)
                all_rows, completed_sites = [], 0
                with (args.output_dir/'tokens.jsonl').open('x', encoding='utf-8') as tokens, \
                     (args.output_dir/'sites.jsonl').open('x', encoding='utf-8') as sites:
                    for completed, sample in enumerate(_samples(manifest, processor, args.image_root), 1):
                        rows, witness = _analyze(model, sample)
                        _validate_witness(witness, sites=sample.sites, layers=layers)
                        for row in rows:
                            tokens.write(json.dumps(row, allow_nan=False) + '\n')
                        tokens.flush()
                        all_rows.extend(rows)
                        for site in sample.sites:
                            site_rows = [r for r in rows if r['plan_index'] == site.plan_index]
                            site_summary = summarize(site_rows, {(sample.source_index, site.plan_index):
                                                           tuple(zip(site.target_positions, site.target_ids))})
                            sites.write(json.dumps({'source_index': sample.source_index, 'plan_index': site.plan_index,
                                                    'metrics': site_summary, 'witness': witness}, allow_nan=False) + '\n')
                        sites.flush()
                        completed_sites += len(sample.sites)
                        progress = f'samples={completed}/{len(manifest["samples"])} sites={completed_sites}/{manifest["S"]}'
                        print(progress, flush=True)
                        with log_path.open('a') as log:
                            log.write(progress + '\n')
                summarize(all_rows, expected)
                _write(args.output_dir/'exclusions.json', manifest['exclusions'])
                _write(args.output_dir/'status.json', {'status': 'completed', 'manifest_sha256': _hash(args.manifest),
                    'samples': len(manifest['samples']), 'S': manifest['S'], 'N': manifest['N']})
        with log_path.open('a') as log:
            log.write(f'{args.command} completed\n')
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        message = f'{type(error).__name__}: {error}'
        print(message, file=sys.stderr, flush=True)
        if log_path is not None:
            with log_path.open('a') as log:
                log.write(message + '\n')
        if owned_output and args.command == 'smoke':
            _write(args.output_dir/'smoke.json', {'status': 'failed', 'error': message})
        return 1
