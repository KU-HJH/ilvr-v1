#!/usr/bin/env python3
"""Initial Stage1 full-vocabulary entropy diagnostic; no training or model writes."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

BASE = '/raid/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct'
INITIALIZER = 'dil-bce-original-stage1-initializer-20260929/checkpoint-431'
LENGTH_MANIFEST = 'results/dil-both-lower-ce01-b1-m0-20261002-01/stage2-length-manifest.json'
PROFILE = {'loss': 'dil_bce_rf_topk_both_lower', 'beta': 1., 'margin': 0., 'ce_weight': .1,
           'preference_weight': 1., 'entropy_weight': 0., 'mass_weight': 1., 'topk': 1516,
           'entropy_percent': 20., 'entropy_source': 'clean', 'target_scope': 'until_next_latent',
           'prefix': 'clean_prefix', 'visual_only': True, 'original_image_accessible': True,
           'boundaries_visible_supervised': True, 'rendering': 'concatenated-v1',
           'decoder': 'sdpa', 'vision': 'flash_attention_2', 'sidecar': 'flex', 'query_budget': 1024,
           'latent_size': 8, 'dora_r': 32, 'dora_alpha': 64, 'dora_dropout': .05,
           'max_sequence': 8192, 'native_checkpointing': False, 'latent_checkpointing': True,
           'microbatch': 1, 'accumulation': 4, 'global_batch': 16, 'data_workers': 0,
           'sampler': 'reference_random', 'zero_stage': 2, 'mode': 'eval', 'grad': False}


def sha256(path):
    from scripts.probe_dil_full_model import sha256 as digest
    return digest(path)


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, allow_nan=False, indent=2)
        f.write('\n')


def length_identity(value):
    """Host realpaths differ across mounts; all other manifest evidence is bound."""
    value = json.loads(json.dumps(value))
    for image in value.get('images', []):
        image.pop('path', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def shard_indices(count, rank, world):
    if type(count) is not int or count < 0 or world != 4 or rank not in range(world):
        raise ValueError('requires exact four-rank shards')
    return list(range(rank, count, world))


def smoke_indices(retained, lengths, rank):
    longest = max(range(len(retained)), key=lambda i: lengths[retained[i]])
    indices = shard_indices(len(retained), rank, 4)[:1]
    if longest % 4 == rank and longest not in indices:
        indices.append(longest)
    return indices


def validate_samples(samples, retained):
    seen, identities = set(), set()
    for s in samples:
        i = s['retained_index']
        if i in seen or s['sample_id'] in identities:
            raise ValueError('duplicate sample record')
        seen.add(i)
        identities.add(s['sample_id'])
        if type(i) is not int or not 0 <= i < len(retained) or s['original_index'] != retained[i]:
            raise ValueError('sample coverage/index mismatch')
        tokens = set()
        for t in s['tokens']:
            key = (t['step'], t['target_position'])
            if key in tokens:
                raise ValueError('duplicate token record')
            tokens.add(key)
            if t['predictor_position'] != t['target_position'] - 1:
                raise ValueError('target alignment mismatch')
            if any(not math.isfinite(t[k]) or t[k] < 0 for k in ('Hc', 'Hv', 'G')):
                raise ValueError('nonfinite or negative entropy')
            if t['G'] != abs(t['Hv'] - t['Hc']):
                raise ValueError('gap mismatch')
    if seen != set(range(len(retained))):
        raise ValueError('sample coverage missing')


def paired_tokens(input_ids, spans, clean_entropies, results):
    import torch
    if len({s.step_id for s in spans}) != len(spans):
        raise ValueError('duplicate eligible step')
    if len(spans) != len(clean_entropies) or set(results) != {(s.step_id, 'visual') for s in spans}:
        raise ValueError('entropy alignment layout mismatch')
    records = []
    for span, hc in zip(spans, clean_entropies):
        row = results[(span.step_id, 'visual')]
        targets = input_ids[0, span.target_text_start:span.target_text_end]
        positions = tuple(range(span.target_text_start-1, span.target_text_end-1))
        hv = row['entropies']  # Deliberately not dil_tokens' top-k-renormalized entropy.
        if (row.get('name') != f'step-{span.step_id}-visual-off'
                or row.get('predictor_positions') != positions
                or not torch.equal(row.get('target_ids'), targets)
                or hc.shape != targets.shape or hv.shape != targets.shape):
            raise ValueError('paired entropy target identity alignment mismatch')
        if not hc.is_cuda or not hv.is_cuda or not bool(torch.isfinite(hc).all() & torch.isfinite(hv).all()):
            raise ValueError('paired entropies must be finite CUDA tensors')
        for p, token, c, v in zip(positions, targets.tolist(), hc.detach().float().tolist(), hv.detach().float().tolist()):
            records.append({'step': int(span.step_id), 'target_position': p+1, 'predictor_position': p,
                            'token_id': token, 'Hc': c, 'Hv': v, 'G': abs(v-c)})
    return records


def native_arguments(manifest, repository, run_id):
    from scripts.probe_dil_full_model import runtime_arguments
    args = runtime_arguments(manifest, repository, run_id, 'portTorch')
    overrides = {'--step_preference_beta': '1', '--ce_weight': '.1', '--dataloader_num_workers': '0',
                 '--step_preference_loss_type': PROFILE['loss'],
                 '--log_file': str(Path(repository)/'logs/dil-entropy-gap-workflow'/run_id/'native.log')}
    for key, value in overrides.items():
        args[args.index(key)+1] = value
    args += ['--stage2_length_manifest', manifest['length_manifest'], '--stage2_length_manifest_sha256',
             manifest['length_manifest_sha256'], '--stage2_sampler_policy', 'reference_random']
    return args


def statistic(values):
    import torch
    if not values.numel():
        return {k: None for k in ('mean','std','min','p50','p80','p90','p95','p99','max')}
    q = torch.quantile(values, torch.tensor([.5,.8,.9,.95,.99], device=values.device, dtype=values.dtype)).tolist()
    return dict(mean=values.mean().item(), std=values.std(correction=0).item(), min=values.min().item(),
                **dict(zip(('p50','p80','p90','p95','p99'), q)), max=values.max().item())


def magnitude(values):
    n = len(values)
    if not n:
        return {'count': 0, **{k: None for k in ('mean_Hc','mean_Hv','mean_G','mean_signed_gap','relative_gap_percent',
                  'proportion_Hv_gt_Hc','proportion_Hv_eq_Hc','proportion_Hv_lt_Hc')},
                'statistics': {k: statistic(values[:,j]) for j,k in enumerate(('Hc','Hv','G'))}}
    hc,hv,g = values.unbind(1)
    c,v,a = hc.mean().item(),hv.mean().item(),g.mean().item()
    return {'count': n, 'mean_Hc': c, 'mean_Hv': v, 'mean_G': a, 'mean_signed_gap': (hv-hc).mean().item(),
            'relative_gap_percent': 100*a/c if c else None,
            'proportion_Hv_gt_Hc': (hv>hc).double().mean().item(),
            'proportion_Hv_eq_Hc': (hv==hc).double().mean().item(),
            'proportion_Hv_lt_Hc': (hv<hc).double().mean().item(),
            'statistics': {k: statistic(values[:,j]) for j,k in enumerate(('Hc','Hv','G'))}}


def analyze_samples(samples, device='cuda'):
    import torch
    from step_preference.entropy_selection import select_entropy_masks
    if not torch.device(device).type == 'cuda':
        raise ValueError('entropy/selection/aggregation requires CUDA')
    annotated, paired, blocks, cmasks, dmasks = [], [], [], [], []
    for s in sorted(samples, key=lambda x:x['retained_index']):
        vals = torch.tensor([[t['Hc'],t['Hv'],t['G']] for t in s['tokens']], device=device, dtype=torch.float64).reshape(-1,3)
        c = select_entropy_masks([vals[:,0]],20.)[0]
        d = select_entropy_masks([vals[:,2]],20.)[0]
        cs, ds = magnitude(vals[c]), magnitude(vals[d])
        pc = {'retained_index':s['retained_index'],'original_index':s['original_index'],'sample_id':s['sample_id'],
              'count':len(vals), 'selected_count':int(c.sum()), 'intersection_count':int((c&d).sum()),
              'overlap_fraction':float((c&d).sum()/c.sum()) if bool(c.any()) else None,
              'clean_cutoff':vals[c,0].min().item() if bool(c.any()) else None,
              'gap_cutoff':vals[d,2].min().item() if bool(d.any()) else None,
              'meanG_D_minus_C':ds['mean_G']-cs['mean_G'] if len(vals) else None,
              'meanHc_D_minus_C':ds['mean_Hc']-cs['mean_Hc'] if len(vals) else None,
              'ALL':magnitude(vals), 'C':cs, 'D':ds}
        paired.append(pc)
        annotated.append({**s,'tokens':[{**t,'C':a,'D':b} for t,a,b in zip(s['tokens'],c.tolist(),d.tolist())]})
        blocks.append(vals); cmasks.append(c); dmasks.append(d)
    vals = torch.cat(blocks) if blocks else torch.empty((0,3),device=device,dtype=torch.float64)
    c = torch.cat(cmasks) if cmasks else torch.empty(0,device=device,dtype=torch.bool)
    d = torch.cat(dmasks) if dmasks else torch.empty(0,device=device,dtype=torch.bool)
    masks = {'ALL':torch.ones(len(vals),device=device,dtype=torch.bool),'C':c,'D':d,'C&D':c&d,'C-D':c&~d,'D-C':d&~c}
    groups = {name:magnitude(vals[mask]) for name,mask in masks.items()}
    for group in groups.values():
        group['share_ALL'] = group['count']/len(vals) if len(vals) else None
        group['share_C'] = group['count']/groups['C']['count'] if groups['C']['count'] else None
        group['share_D'] = group['count']/groups['D']['count'] if groups['D']['count'] else None
        group['gap_enrichment_vs_ALL'] = (group['mean_G']/groups['ALL']['mean_G']
                                        if group['mean_G'] is not None and groups['ALL']['mean_G'] else None)
    def diff(key):
        a,b = groups['D'][key],groups['C'][key]
        return a-b if a is not None and b is not None else None
    def per_sample(key):
        return statistic(torch.tensor([p[key] for p in paired if p[key] is not None],device=device,dtype=torch.float64))
    sample_means = {name:{metric:statistic(torch.tensor([p[name][metric] for p in paired if p[name][metric] is not None],device=device,dtype=torch.float64))
                       for metric in ('mean_Hc','mean_Hv','mean_G')} for name in ('ALL','C','D')}
    return {'units':'nats', 'aggregation':'token-weighted', 'selection':'sample-local stable descending ceil(N*0.20)',
            'sample_count':len(samples),'eligible_empty_samples':sum(not s['tokens'] for s in samples),
            'groups':groups,'differences':{'meanG_D_minus_C':diff('mean_G'),'meanHc_D_minus_C':diff('mean_Hc')},
            'paired_sample_differences':{'G':per_sample('meanG_D_minus_C'),'Hc':per_sample('meanHc_D_minus_C')},
            'cutoffs':{'C':per_sample('clean_cutoff'),'D':per_sample('gap_cutoff')},
            'overlap_per_sample':per_sample('overlap_fraction'),'sample_average_metrics':sample_means,
            'interpretation':'Larger gap in D is selection-defined and does not establish better training.'}, annotated, paired


def load_manifest(path, repository):
    value = json.loads(Path(path).read_text())
    root = Path(repository)
    if value['schema'] != 1 or value['profile'] != PROFILE or value['base'] != BASE or Path(value['initializer']).resolve() != (root/'checkpoints'/INITIALIZER).resolve():
        raise ValueError('diagnostic profile/initial Stage1 source changed')
    length = root/LENGTH_MANIFEST
    raw = json.loads(length.read_text())
    if length_identity(raw) != value['length_identity'] or raw['retained_indices'] != value['retained_indices']:
        raise ValueError('length manifest semantic identity changed')
    if len(value['retained_indices']) != 3379 or len(set(value['retained_indices'])) != 3379:
        raise ValueError('requires exactly 3379 unique retained samples')
    if sha256(root/'data/TRAIN.jsonl') != value['data_sha256']:
        raise ValueError('TRAIN identity changed')
    value.update(data_path=str(root/'data/TRAIN.jsonl'), image_root=str(root/'data'),
                 length_manifest=str(length),length_manifest_sha256=sha256(length))
    return value


def create_manifest(path, repository):
    root = Path(repository)
    raw = json.loads((root/LENGTH_MANIFEST).read_text())
    value = {'schema':1,'profile':PROFILE,'base':BASE,'initializer':str(root/'checkpoints'/INITIALIZER),
             'retained_indices':raw['retained_indices'],'length_identity':length_identity(raw),
             'data_sha256':sha256(root/'data/TRAIN.jsonl')}
    write_json(path,value)
    load_manifest(path,root)


def worker(args):
    from scripts.probe_dil_full_model import bootstrap
    root = Path(args.repository)
    manifest = load_manifest(args.manifest,root)
    if sha256(args.manifest) != args.manifest_sha256:
        raise ValueError('diagnostic manifest changed')
    bootstrap(root,root)
    import torch
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    from utils_deepseed import parse_args
    from main import build_stage2_training_components, build_stage2_training_arguments
    from deepspeed_runtime import bind_deepspeed_runtime_profile
    native_args = parse_args(native_arguments(manifest,root,args.run_id))
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    components = build_stage2_training_components(native_args)
    capture = {}
    class ObservedTrainer(components.trainer_class):
        def _reduce_context_objective(self, clean_ce, clean_scores, clean_entropies, results, *, clean_token_logps=None, spans=None):
            if capture.get('tokens') is not None:
                raise ValueError('duplicate reducer observation')
            capture['tokens'] = paired_tokens(capture['input_ids'],spans,clean_entropies,results)
            return clean_ce, {}
    trainer = ObservedTrainer(model=components.model,args=build_stage2_training_arguments(native_args),
        train_dataset=components.train_dataset,data_collator=components.data_collator,processing_class=components.processor,
        stage2_peft_metadata=components.peft_metadata,step_preference_config=components.step_preference_config,
        deepspeed_runtime_profile=bind_deepspeed_runtime_profile(native_args,world_size=4))
    if trainer.optimizer is not None or getattr(trainer,'deepspeed',None) is not None:
        raise ValueError('diagnostic cannot prepare optimizer/engine')
    rank, world = trainer.accelerator.process_index, trainer.accelerator.num_processes
    if world != 4 or rank != int(os.environ['RANK']):
        raise ValueError('actual four-rank participation required')
    model = trainer.model.to(device=trainer.accelerator.device,dtype=torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    versions = [(p,p._version,p.data_ptr()) for p in model.parameters()]
    if len(components.train_dataset) != 3379:
        raise ValueError('builder retained coverage differs')
    from stage2.data import load_jsonl_examples
    from step_preference.spans import canonical_sample_id
    originals = load_jsonl_examples(Path(manifest['data_path']))
    expected_ids = [canonical_sample_id(originals[i]) for i in manifest['retained_indices']]
    if ([e.sample_id for e in components.train_dataset] != expected_ids
            or len(set(expected_ids)) != len(expected_ids)):
        raise ValueError('builder retained sample identity/order mismatch')
    output = root/'results'/args.run_id
    output.mkdir(parents=True,exist_ok=True)
    indices = shard_indices(3379,rank,world)
    if args.smoke:
        raw = json.loads(Path(manifest['length_manifest']).read_text())
        indices = smoke_indices(manifest['retained_indices'],raw['lengths'],rank)
    witness = {'schema':1,'rank':rank,'world_size':world,'host':socket.gethostname(),'machine_rank':int(os.environ['ILVR_MACHINE_RANK']),
               'local_rank':int(os.environ['LOCAL_RANK']),'device':torch.cuda.current_device(),'profile':PROFILE,
               'manifest_sha256':args.manifest_sha256,'native_cli':native_arguments(manifest,root,args.run_id),
               'samples':len(indices),'tokens':0,'eval_no_grad_frozen':True,'weights_version_unchanged':False,'smoke':args.smoke}
    # Check the real top-level forward mode, rather than only the caller's intent.
    def guard(module, _inputs):
        if torch.is_grad_enabled() or any(m.training for m in module.modules()) or any(p.requires_grad for p in module.parameters()):
            raise ValueError('forward must run eval/no_grad/frozen')
    hook = model.register_forward_pre_hook(guard)
    try:
        with (output/f'entropy-rank-{rank}.jsonl').open('x') as stream, torch.no_grad():
            for i in indices:
                example = components.train_dataset[i]
                batch = trainer._prepare_inputs(components.data_collator([example]))
                capture.clear(); capture['input_ids'] = batch['input_ids']
                loss = trainer.compute_loss(model,batch,return_outputs=False)
                if capture.get('tokens') is None or not bool(torch.isfinite(loss)):
                    raise ValueError('missing paired token observation/finite forward')
                item = {'retained_index':i,'original_index':manifest['retained_indices'][i],
                        'sample_id':example.sample_id,'tokens':capture.pop('tokens')}
                witness['tokens'] += len(item['tokens'])
                stream.write(json.dumps(item,allow_nan=False)+'\n'); stream.flush()
                del loss,batch,item
        if any(p._version != v or p.data_ptr() != ptr or p.grad is not None for p,v,ptr in versions):
            raise ValueError('weight mutation or gradients observed')
        witness['weights_version_unchanged'] = True
        write_json(output/f'entropy-witness-rank-{rank}.json',witness)
    finally:
        hook.remove()


def aggregate(args):
    root = Path(args.repository)
    manifest = load_manifest(args.manifest,root)
    output = root/'results'/args.run_id
    samples = []
    witnesses = []
    for rank in range(4):
        witness = json.loads((output/f'entropy-witness-rank-{rank}.json').read_text())
        if witness['rank'] != rank or witness['world_size'] != 4 or witness['smoke'] or not witness['eval_no_grad_frozen'] or not witness['weights_version_unchanged'] or witness['manifest_sha256'] != sha256(args.manifest):
            raise ValueError('invalid full-collection forward witness')
        witnesses.append(witness)
        rows = [json.loads(line) for line in (output/f'entropy-rank-{rank}.jsonl').read_text().splitlines()]
        if [s['retained_index'] for s in rows] != shard_indices(3379,rank,4) or len(rows) != witness['samples'] or sum(len(s['tokens']) for s in rows) != witness['tokens']:
            raise ValueError('rank shard coverage/record count mismatch')
        samples += rows
    validate_samples(samples,manifest['retained_indices'])
    summary,annotated,paired = analyze_samples(samples)
    summary['runtime_witnesses'] = witnesses
    write_analysis(summary,annotated,paired,output)
    print(json.dumps({'summary':str(output/'summary.json'),'samples':len(samples)}))


def write_analysis(summary, annotated, paired, output):
    write_json(output/'summary.json',summary)
    with (output/'paired-samples.jsonl').open('x') as f:
        for row in paired:
            f.write(json.dumps(row,allow_nan=False)+'\n')
    fields = ('retained_index','original_index','sample_id','step','target_position','predictor_position','token_id','Hc','Hv','G','C','D')
    with (output/'paired-tokens.csv').open('x',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for s in annotated:
            for t in s['tokens']:
                writer.writerow({**{k:s[k] for k in fields[:3]},**t})
    with (output/'magnitude-table.csv').open('x',newline='') as f:
        fields = ['group']+[k for k in summary['groups']['ALL'] if k != 'statistics']
        writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for name,row in summary['groups'].items():
            writer.writerow({'group':name,**{k:v for k,v in row.items() if k != 'statistics'}})
    plot(annotated,output)


def plot(samples, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    groups = {name:[] for name in ('ALL','C','D')}
    for s in samples:
        for t in s['tokens']:
            groups['ALL'].append(t)
            for name in ('C','D'):
                if t[name]: groups[name].append(t)
    for metric in ('Hc','G'):
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        for name,rows in groups.items():
            x = np.asarray([r[metric] for r in rows])
            if not len(x): continue
            axes[0].hist(x,bins=80,density=True,histtype='step',label=name)
            axes[1].plot(np.sort(x),np.arange(1,len(x)+1)/len(x),label=name)
        for ax in axes:
            ax.set_xlabel(('Clean entropy' if metric == 'Hc' else 'Absolute entropy gap')+' (nats)')
            if groups['ALL']: ax.legend()
        axes[0].set_ylabel('Density (1/nat)'); axes[1].set_ylabel('Cumulative proportion')
        fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(output/f'{metric}-histogram-cdf.{ext}')
        plt.close(fig)
    fig,ax = plt.subplots(figsize=(6,5))
    if groups['ALL']:
        density = ax.hexbin([r['Hc'] for r in groups['ALL']],[r['G'] for r in groups['ALL']],gridsize=65,mincnt=1)
        fig.colorbar(density,ax=ax,label='Token count')
    ax.set_xlabel('Clean entropy (nats)'); ax.set_ylabel('Absolute entropy gap (nats)'); fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(output/f'Hc-versus-G-density.{ext}')
    plt.close(fig)


def coordinate(args):
    from src.multinode import (build_launch_plan,build_preflight_commands,collect_node_snapshot,validate_preflight,
        _repository_git_stdout,start_rank0_then_rank1,monitor_owned_jobs,_json_ready)
    from src.multinode_runtime import build_dil_full_model_probe_commands,SubprocessCoordinatorRunner,build_remote_exec
    root = Path(args.repository)
    manifest = load_manifest(args.manifest,root)
    digest = sha256(args.manifest)
    plan = build_launch_plan(stage='stage2',tuning_method='dora',zero_stage=2,run_id=args.run_id,main_process_port=args.port)
    commit = _repository_git_stdout(('rev-parse','HEAD'))
    commands = build_dil_full_model_probe_commands(plan,commit=commit,manifest=args.manifest,manifest_sha256=digest,
                                                 diagnostic='entropy-distribution',smoke=args.smoke)
    inputs = (Path(args.manifest),root/'scripts/collect_entropy_distribution.py',root/'scripts/probe_dil_full_model.py',
              root/LENGTH_MANIFEST,root/'data/TRAIN.jsonl',root/'data',Path(manifest['base']),Path(manifest['initializer']),plan.deepspeed_config)
    roots = (root/'checkpoints',root/'results',root/'logs')
    paths = (root/'results'/args.run_id,root/'logs/multinode'/args.run_id/'initial')
    sessions = tuple(f'ilvr-{args.run_id}-initial-rank{r}' for r in (0,1))
    preflight = build_preflight_commands(plan,input_paths=inputs,artifact_roots=roots,remote_run_paths=paths,local_run_paths=paths,
                                        remote_tmux_sessions=(sessions[0],),local_tmux_sessions=(sessions[1],))
    identity = []
    for rank in (0,1):
        command = (str(root/'venv/ilvr-venv/bin/python'),str(root/'scripts/collect_entropy_distribution.py'),'preflight',
                   '--repository',str(root),'--manifest',str(args.manifest))
        identity.append(tuple(build_remote_exec(command)) if rank == 0 else command)
    if args.dry_run:
        print(json.dumps({'profile':PROFILE,'retained_count':3379,'manifest_sha256':digest,'plan':_json_ready(plan),
                          'local':commands.local,'remote':commands.remote,'resource_preflight':_json_ready(preflight),
                          'identity_preflight':identity,'smoke':args.smoke},sort_keys=True))
        return
    def run(command):
        return subprocess.run(command,capture_output=True,text=True,check=False,timeout=600)
    snapshots = [collect_node_snapshot(plan,node=node,runner=run,commands=preflight,input_paths=inputs,artifact_roots=roots,
                  run_paths=paths,tmux_sessions=(sessions[i],),repository=plan.cluster.remote_repo_root) for i,node in enumerate(('remote','local'))]
    if validate_preflight(*snapshots).commit != commit:
        raise ValueError('preflight commit changed')
    for command in identity:
        completed = run(command)
        if completed.returncode:
            raise RuntimeError('identity preflight failed: '+completed.stderr)
    runner = SubprocessCoordinatorRunner(plan,commands=commands,commit=commit,log_root=root/'logs/multinode')
    start_rank0_then_rank1(plan,runner,commands=commands,ready_timeout_seconds=600)
    exits = monitor_owned_jobs(plan,runner,expected_commit=commit,expected_command_digests=dict(enumerate(commands.command_digests)))
    if exits != {0:0,1:0}:
        raise RuntimeError(f'entropy workers failed: {exits}; preserve logs and use fresh ID')
    output = root/'results'/args.run_id
    for rank in (0,1):
        for kind in ('records','witness'):
            command = tuple(build_remote_exec((str(root/'venv/ilvr-venv/bin/python'),str(root/'scripts/collect_entropy_distribution.py'),
                        'record-show','--repository',str(root),'--run-id',args.run_id,'--rank',str(rank),'--kind',kind)))
            path = output/(f'entropy-rank-{rank}.jsonl' if kind == 'records' else f'entropy-witness-rank-{rank}.json')
            with path.open('xb') as stream:
                subprocess.run(command,stdout=stream,check=True,timeout=600)
    if not args.smoke:
        aggregate(args)
    else:
        print(json.dumps({'smoke_witnesses':[str(output/f'entropy-witness-rank-{r}.json') for r in range(4)]}))


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('command',choices=('create-manifest','preflight','run','launch','aggregate','record-show'))
    parser.add_argument('--repository',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--run-id')
    parser.add_argument('--port',type=int,default=29884)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--rank',type=int,choices=range(4))
    parser.add_argument('--kind',choices=('records','witness'))
    args = parser.parse_args(argv)
    # stdlib bootstrap permits directly running the script from any cwd.
    root = Path(args.repository)
    sys.path.insert(0,str(root))
    from scripts.probe_dil_full_model import contained
    contained(root/'results',root/'results')
    if args.command != 'record-show':
        if args.manifest is None: parser.error('--manifest required')
        contained(args.manifest,root/'results')
    if args.command in ('run','launch','aggregate','record-show'):
        if not args.run_id: parser.error('--run-id required')
        from scripts.probe_dil_full_model import tensor_directory
        tensor_directory(root,args.run_id)  # validation only; creates no model state.
    if args.command == 'create-manifest': create_manifest(args.manifest,root)
    elif args.command == 'preflight':
        manifest = load_manifest(args.manifest,root)
        print(json.dumps({'profile':PROFILE,'retained_count':len(manifest['retained_indices']),
                          'native_cli':native_arguments(manifest,root,'preflight-only')}))
    elif args.command == 'run': worker(args)
    elif args.command == 'launch': coordinate(args)
    elif args.command == 'aggregate':
        sys.path[:0] = [str(root/'src'),str(root/'transformers/src')]
        aggregate(args)
    else:
        if args.rank is None or args.kind is None: parser.error('--rank and --kind required')
        path = root/'results'/args.run_id/(f'entropy-rank-{args.rank}.jsonl' if args.kind == 'records' else f'entropy-witness-rank-{args.rank}.json')
        with path.open('rb') as stream:
            import shutil
            shutil.copyfileobj(stream,sys.stdout.buffer)


if __name__ == '__main__':
    main()
