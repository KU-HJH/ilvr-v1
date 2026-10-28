#!/usr/bin/env python
"""Bounded cache acceptance, always through the existing two-node coordinator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'transformers/src'), str(REPO / 'src'), str(REPO)]

from stage1.cache_validation import compare_acceptance
from src.multinode import ClusterSpec
from src.multinode_runtime import _build_remote_stdin_exec, build_remote_exec


def common_arguments(profile, data):
    args = ['--method', 'dora', '--model', '/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct',
        '--data-path', str(data), '--image-root', str(REPO / 'data/images_comt'),
        '--ce-weight', '.1', '--sim-weight', '.1' if profile == 'v5' else '1',
        '--micro-batch', '1', '--gradient-accumulation', '4', '--epochs', '10', '--max-steps', '2160',
        '--learning-rate', '5e-5', '--seed', '42', '--warm-up-steps', '10',
        '--ema-tau', '1' if profile == 'v5' else '.999', '--latent-size', '8',
        '--attention-implementation', 'sdpa' if profile == 'v5' else 'flash_attention_2',
        '--vision-attention-implementation', 'flash_attention_2', '--max-seq-length-train', '32768',
        '--dataloader-num-workers', '4', '--save-strategy', 'steps', '--save-steps', '1',
        '--save-total-limit', '2', '--no-final-save', '--query_mode', 'legacy', '--textual_states', 'previous',
        '--stage1-check-policy', 'production']
    if profile == 'native-ema':
        return args + ['--stage1-runtime-profile', 'native-ema-0180']
    return args + ['--stage1-variant', 'v5', '--no-stage1-v5-debug', '--stage1-loss-type', 'dil-bce-orig',
        '--lambda-preference', '1', '--lambda-topk-entropy', '.1', '--lambda-topk-mass', '.05',
        '--lambda-topk-chosen-mass', '0', '--lambda-entropy', '0', '--preference-topk-k', '50',
        '--preference-topk-mode', 'independent', '--no-topk-include-gt', '--stage1-v5-selection', 'clean',
        '--preference-entropy-top-percent', '20', '--stage1-v5-text-kl-weight', '.01',
        '--loss-chunk-positions', '128', '--step-preference-beta', '1', '--step-preference-margin', '0',
        '--stage1-v5-voff-mode', 'attention_mask']


def command(args, *, arm, mode, cache, stop=2, resume=False):
    run = f'{args.run_id}-{args.profile}-{arm}'
    data = REPO / 'data/cache-validation' / args.run_id / 'TRAIN.jsonl'
    values = [str(REPO / 'run_multinode_stage1_training.sh'), '--run-id', run,
        '--zero-stage', '1', '--main-process-port', str(args.port),
        '--remote-repo-root', str(args.remote_repo_root), '--remote-host-repo-root', str(args.remote_host_repo_root),
        *common_arguments(args.profile, data), '--stage1-cache-validation', mode,
        '--stage1-cache-validation-stop', str(stop), '--stage1-cache', 'run' if cache else 'off']
    if resume:
        values += ['--attempt-id', 'resume-cache', '--resume', str(REPO / 'checkpoints' / run / 'checkpoint-1')]
    return values, run


def cluster(args):
    return ClusterSpec.default(local_repo_root=REPO, remote_repo_root=args.remote_repo_root,
                               remote_host_repo_root=args.remote_host_repo_root)


def prepare_data(args):
    source = REPO / 'data/TRAIN.jsonl'
    rows = [line for line in source.read_text().splitlines() if line.strip()][:16]
    if len(rows) != 16:
        raise ValueError('verification requires 16 source rows')
    raw = ('\n'.join(rows) + '\n').encode()
    destination = REPO / 'data/cache-validation' / args.run_id / 'TRAIN.jsonl'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != raw:
        raise ValueError('verification dataset identity conflict')
    destination.write_bytes(raw)
    remote = args.remote_repo_root / destination.relative_to(REPO)
    code = ('import pathlib,sys; p=pathlib.Path(sys.argv[1]); data=sys.stdin.buffer.read(); '
        'p.parent.mkdir(parents=True,exist_ok=True); '
        'assert not p.exists() or p.read_bytes()==data; p.write_bytes(data)')
    subprocess.run(_build_remote_stdin_exec([str(args.remote_repo_root / 'venv/ilvr-venv/bin/python'),
        '-c', code, str(remote)], working_directory=args.remote_repo_root, cluster=cluster(args)),
        input=raw, check=True, timeout=60)


def read_records(run, attempt='initial'):
    return [json.loads((REPO / 'results' / run / f'{attempt}-cache-rank-{rank}.json').read_text()) for rank in range(4)]


def run_arm(args, **kwargs):
    cmd, run = command(args, **kwargs)
    root = REPO / 'logs' / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    attempt = 'resume-cache' if kwargs.get('resume') else 'initial'
    (root / f'{run}-{attempt}-command.json').write_text(json.dumps(cmd, indent=2) + '\n')
    # All orchestration, preflight, ownership and checkpoint transfer remain
    # owned by the existing coordinator. No manual rank launch or retry loop.
    with (root / f'{run}-{attempt}.log').open('x') as log:
        subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    return run, read_records(run, attempt)


def remove_one_derived_record(args, run):
    code = """import json,pathlib,sys
root=pathlib.Path('/root/workspace/ILVR-stage1-cache')/sys.argv[1]
identity=json.loads((root/'identity.json').read_text())
assert identity['run_id']==sys.argv[1] and not root.is_symlink()
layer=identity['helper']
assert layer in ('targets','vision')
path=next(iter(sorted((root/layer).glob('*.pt'))))
assert path.is_file() and not path.is_symlink()
path.unlink()
print(path.name)
"""
    removed = {}
    for rank in (0, 1):
        repo = args.remote_repo_root if rank == 0 else REPO
        cmd = [str(repo / 'venv/ilvr-venv/bin/python'), '-c', code, run]
        if rank == 0:
            cmd = build_remote_exec(cmd, working_directory=repo, cluster=cluster(args))
        removed[str(rank)] = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    return removed


def merged_resume(first, second):
    combined = dict(second)
    combined['before'] = first['before']
    combined['microbatches'] = first['microbatches'] + second['microbatches']
    combined['optimizer_steps'] = first['optimizer_steps'] + second['optimizer_steps']
    combined['production_checks'] = {'optional_calls': first['production_checks']['optional_calls'] + second['production_checks']['optional_calls']}
    return combined


def cleanup_completed_cache(args, run, attempt='initial'):
    for rank in (0, 1):
        repo = args.remote_repo_root if rank == 0 else REPO
        cmd = ['/usr/bin/env', f'PYTHONPATH={repo / "transformers/src"}:{repo / "src"}:{repo}',
            str(repo / 'venv/ilvr-venv/bin/python'), '-m', 'src.stage1.cache_transport', 'remove',
            '/root/workspace/ILVR-stage1-cache/' + run,
            str(repo / 'logs/multinode' / run / attempt / 'cache')]
        if rank == 0:
            cmd = _build_remote_stdin_exec(cmd, working_directory=repo, cluster=cluster(args))
        subprocess.run(cmd, input=(json.dumps({'run_id': run, 'attempt': attempt}) + '\n').encode(),
                       capture_output=True, check=True, timeout=60)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--phase', choices=('dry-run', 'correctness', 'performance'), required=True)
    parser.add_argument('--profile', choices=('v5', 'native-ema'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--port', type=int, default=29952)
    parser.add_argument('--remote-repo-root', type=Path,
        default=Path('/root/workspace/ILVR/.worktrees/stage1-cache-probe-20261028'))
    parser.add_argument('--remote-host-repo-root', type=Path,
        default=Path('/home/korea/Workspace/hjh/workspace/ILVR/.worktrees/stage1-cache-probe-20261028'))
    args = parser.parse_args(argv)
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id) is None:
        parser.error('invalid run ID')
    if args.phase == 'dry-run':
        commands = [command(args, arm=arm, mode=mode, cache=cache, stop=stop, resume=resume)[0]
            for arm, mode, cache, stop, resume in (
                ('payload', 'payload', True, 2, False), ('a', 'correctness', False, 2, False),
                ('aa', 'correctness', False, 2, False), ('b', 'correctness', True, 1, False),
                ('b', 'correctness', True, 2, True))]
        print(json.dumps({'commands': commands, 'optimizer_updates': 6, 'samples': 16,
            'limit': '2 updates per correctness arm; full 2160-step schedule retained'}, indent=2))
        return
    output = REPO / 'results' / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    report = {'profile': args.profile, 'phase': args.phase, 'accepted': False}
    report_path = output / f'{args.profile}-{args.phase}.json'
    if report_path.exists():
        raise FileExistsError('use a fresh verification run ID')
    try:
        prepare_data(args)
        if args.phase == 'correctness':
            run, payload = run_arm(args, arm='payload', mode='payload', cache=True)
            report['payload_run'] = run
            _, a = run_arm(args, arm='a', mode='correctness', cache=False)
            _, repeated = run_arm(args, arm='aa', mode='correctness', cache=False)
            for rank in range(4):
                compare_acceptance(a[rank], repeated[rank], repeated[rank])
            run, b_first = run_arm(args, arm='b', mode='correctness', cache=True, stop=1)
            report['removed_for_resume'] = remove_one_derived_record(args, run)
            _, b_second = run_arm(args, arm='b', mode='correctness', cache=True, stop=2, resume=True)
            report['ranks'] = [compare_acceptance(a[rank], repeated[rank], merged_resume(b_first[rank], b_second[rank]))
                               for rank in range(4)]
            report['lifecycle'] = all(item.get('lifecycle') is True for item in payload)
            report['resume_missing_cache'] = True
            cleanup_completed_cache(args, report['payload_run'])
            cleanup_completed_cache(args, run, 'resume-cache')
        else:
            correctness = json.loads((output / f'{args.profile}-correctness.json').read_text())
            if correctness.get('accepted') is not True:
                raise ValueError('performance requires accepted correctness for this profile')
            _, live = run_arm(args, arm='perf-live', mode='performance', cache=False, stop=1)
            run, cached = run_arm(args, arm='perf-cache', mode='performance', cache=True, stop=1)
            report.update(live=live, cached=cached,
                limitation='One observed update per arm; checkpoint and startup included in total; no whole-run speedup claim.')
            cleanup_completed_cache(args, run)
        report['accepted'] = True
    except BaseException as error:
        report['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
