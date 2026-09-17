#!/usr/bin/env python
"""Bounded CPU differential evidence; never authorizes or starts training.

The external tree is loaded only in a separate, read-only evidence process.
Exit 2 means the full-run gate is blocked (including missing GPU evidence).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace


def reference_objective(examples):
    """Actual-example mean of mean-token CE plus mean-valid-step preference.

    Preference entries are already beta=.1, weight=1 losses. No Trainer or
    configured accumulation divisor participates in this arithmetic oracle.
    """
    import torch
    if not examples:
        raise ValueError('objective requires at least one example')
    losses = []
    for ce_tokens, preference_steps in examples:
        if ce_tokens.numel() == 0:
            raise ValueError('example requires supervised CE tokens')
        losses.append(ce_tokens.mean() + (preference_steps.mean() if preference_steps.numel() else ce_tokens.sum()*0))
    return torch.stack(losses).mean()


def gradient_comparison(expected, observed, *, rtol=1e-5, atol=1e-7):
    import torch
    if observed is None or expected.shape != observed.shape:
        return {'pass': False, 'reason': 'missing gradient or shape mismatch'}
    if not bool(torch.isfinite(expected).all() and torch.isfinite(observed).all()):
        return {'pass': False, 'reason': 'nonfinite gradient'}
    return {'pass': bool(torch.allclose(expected, observed, rtol=rtol, atol=atol)),
            'max_abs_error': float((expected-observed).abs().max()), 'rtol': rtol, 'atol': atol}


def compare_records(current, external):
    """Report unequal fields without converting partial agreement into a pass."""
    if len(current) != len(external):
        return [{'field': 'record_count', 'current': len(current), 'external': len(external)}]
    mismatches = []
    for left, right in zip(current, external):
        for field in sorted(left.keys() | right.keys()):
            if left.get(field) != right.get(field):
                mismatches.append({'sample_id': left.get('sample_id'), 'field': field,
                                   'current': left.get(field), 'external': right.get(field)})
    return mismatches


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parent_scaling_characterization(source=None):
    """Execute actual imported Trainer.training_step, with a plain CPU backward.

    This exercises parent scaling only: it is NOT a DeepSpeed or DDP witness.
    The backward boundary intentionally has no additional accumulation scaling.
    """
    import torch
    from contextlib import nullcontext
    from transformers import Trainer
    from accelerate.utils import DistributedType
    from transformers.training_args import OptimizerNames

    class Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))

    class Backward:
        distributed_type = DistributedType.DEEPSPEED
        def __init__(self):
            self.calls = []
        def backward(self, loss, **kwargs):
            self.calls.append(kwargs)
            loss.backward()

    import inspect
    import ast
    import transformers.trainer as trainer_module
    source_file = Path(source)/'transformers/src/transformers/trainer.py' if source else Path(inspect.getfile(Trainer))
    step = Trainer.training_step
    if source:
        tree = ast.parse(source_file.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
        function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'training_step')
        namespace = dict(vars(trainer_module))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_file), 'exec'), namespace)
        step = namespace['training_step']
    records = []
    for count in (4, 1):
        model = Scalar()
        backwards = Backward()
        # Heterogeneous per-example scalar losses, independent arithmetic mean.
        coefficients = tuple(range(1, count+1))
        reference = torch.stack([c*model.p.square() for c in coefficients]).mean()
        expected = torch.autograd.grad(reference, model.p)[0]
        harness = SimpleNamespace(
            optimizer=None, args=SimpleNamespace(gradient_accumulation_steps=4,
                torch_empty_cache_steps=None, optim=OptimizerNames.ADAMW_TORCH, n_gpu=1),
            _prepare_inputs=lambda x: x, compute_loss_context_manager=nullcontext,
            compute_loss=lambda model, inputs, **kw: inputs['coefficient']*model.p.square(),
            use_apex=False, model_accepts_loss_kwargs=False, compute_loss_func=None,
            accelerator=backwards,
        )
        for coefficient in coefficients:
            step(harness, model, {'coefficient': coefficient}, num_items_in_batch=count)
        records.append({'actual_examples_in_local_group': count, 'configured_accumulation': 4,
                        'expected_gradient': expected.item(), 'observed_gradient': model.p.grad.item(),
                        'comparison': gradient_comparison(expected, model.p.grad),
                        'backward_kwargs': backwards.calls})
    return {'scope': 'CPU parent training_step only; plain backward; no distributed engine or optimizer',
            'assumption': 'model_accepts_loss_kwargs=False; compute_loss_func=None',
            'source': str(source_file), 'sha256': _sha(source_file), 'groups': records}


def _external_primitives(source):
    """Load supplied pure modules without executing unrelated package initializers.

    The snapshot lacks a complete Transformers distribution. Relative cache
    utilities are explicitly shared, while parser/sidecar function bodies are
    supplied-source code. Everything stays inside this evidence subprocess.
    """
    import importlib
    import types
    import transformers.cache_utils
    root = Path(source)/'transformers/src/transformers/models/qwen2_5_vl'
    for name, path in [('_voff_external',root.parent.parent),
                       ('_voff_external.models',root.parent),
                       ('_voff_external.models.qwen2_5_vl',root),
                       ('stage2',Path(source)/'src/stage2')]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    sys.modules['_voff_external.cache_utils'] = transformers.cache_utils
    rollout = importlib.import_module('_voff_external.models.qwen2_5_vl.stage2_latent_rollout')
    sidecar = importlib.import_module('_voff_external.models.qwen2_5_vl.stage2_sidecar')
    return rollout, sidecar


def _common_logit_probe(external):
    """Differentiate supplied scoring/loss functions on identical fixed logits.

    These derivatives are with respect to logits, not production parameters.
    Two independently normalized examples have two and three target tokens.
    """
    import torch
    from step_preference import dpo_loss
    clean = (torch.arange(40, dtype=torch.float32).reshape(2,4,5)/17).requires_grad_()
    ablated = (clean.detach().flip(-1)/2).requires_grad_()
    ids = torch.tensor([[0,1,2,3],[0,2,3,1]])
    mask = torch.tensor([[False,True,True,False],[False,True,True,True]])
    if external:
        clean_scores = dpo_loss.mean_token_logps(clean[:,:-1],ids[:,1:],mask[:,1:])
        ablated_scores = dpo_loss.mean_token_logps(ablated[:,:-1],ids[:,1:],mask[:,1:])
        preference = dpo_loss.context_preference_loss(clean_scores,ablated_scores,beta=.1,margin=0.).loss
    else:
        clean_scores = dpo_loss.mean_completion_logps(clean,ids,mask)
        ablated_scores = dpo_loss.mean_completion_logps(ablated,ids,mask)
        preference = dpo_loss.context_preference_loss(clean_scores,ablated_scores,beta=.1,margin=0.)
    loss = -clean_scores.mean() + preference
    gradients = torch.autograd.grad(loss,(clean,ablated))
    return {'scope': 'CPU FP32 common-logit CE plus preference loss primitives; no model or optimizer',
            'target_counts': [2,3], 'beta': .1, 'preference_weight': 1.,
            'loss': float(loss.detach()), 'preference_loss': float(preference.detach()),
            'clean_scores': clean_scores.detach().tolist(), 'ablated_scores': ablated_scores.detach().tolist(),
            'gradient_finite': all(bool(torch.isfinite(g).all()) for g in gradients),
            'clean_gradient_nonzero': bool(gradients[0].abs().max()>0),
            'ablated_gradient_nonzero': bool(gradients[1].abs().max()>0),
            'gradients': [g.tolist() for g in gradients]}


def _dataset_worker(args):
    import torch
    from transformers import AutoProcessor
    external = args.worker == 'external'
    scaling = _parent_scaling_characterization(args.source if external else None)
    if external:
        rollout, sidecar = _external_primitives(args.source)
    import stage2.data as data
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
    if external:
        data.parse_latent_spans = rollout.parse_latent_spans
        LatentTokenIds = rollout.LatentTokenIds
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True, use_fast=False)
    token_ids = LatentTokenIds(*[processor.tokenizer.convert_tokens_to_ids(t) for t in
                               ('<|latent_pad|>', '<|latent_start|>', '<|latent_end|>')])
    collator = data.Stage2DataCollator(processor, token_ids, 8, True,
        **({'context_preference_enabled': True} if external else
           {'step_preference_enabled': True, 'preference_target_scope': 'until_next_latent'}))
    samples = data.load_jsonl_examples(Path(args.data))
    counts = [sum(s['type']=='latent' for s in x['sequence_plan']) for x in samples]
    selected = []
    for predicate in (lambda i: counts[i] == 1, lambda i: counts[i] > 1,
                      lambda i: len(samples[i]['sequence_plan']) >= 2 and
                      all(s['type']=='text' for s in samples[i]['sequence_plan'][-2:])):
        match = next((i for i in range(len(samples)) if predicate(i) and i not in selected), None)
        if match is not None:
            selected.append(match)
    if len(selected) != 3:
        raise ValueError('TRAIN lacks three distinct representative cases')
    records = []
    for index in selected:
        sample = samples[index]
        prepared = data.prepare_stage2_example(sample, image_root=Path(args.image_root), latent_size=8,
            **({} if external else {'assistant_plan_rendering': 'concatenated-v1', 'require_reasoning_units': True}))
        batch = collator([prepared])
        ids, attention = batch['input_ids'], batch['attention_mask']
        if external:
            from step_preference.packing import build_context_sidecar_execution_plan
            pack_stage2_sidecar_plan = sidecar.pack_stage2_sidecar_plan
            build_dense_sidecar_mask = sidecar.build_dense_sidecar_mask
            plan = build_context_sidecar_execution_plan(clean_input_ids=ids,
                items=batch['step_preference_items'], modalities='visual')
            layout = pack_stage2_sidecar_plan(plan, clean_input_ids=ids,
                clean_attention_mask=attention,
                clean_position_ids=torch.arange(ids.shape[1])[None,None,:].expand(3,1,-1))[0]
            visibility = build_dense_sidecar_mask(layout, dtype=torch.float32, device=ids.device)[0,0] == 0
            branches = [{'targets': b.target_token_ids.tolist(), 'tokens': b.input_token_ids.tolist(),
                         'positions': (b.logical_position_offsets+b.replay_boundary).tolist(),
                         'prefix_end': b.clean_visible_end, 'blocked': list(b.blocked_clean_interval),
                         'selected': b.score_offsets.tolist()} for b in plan.visual]
        else:
            from step_preference.sidecar import build_context_sidecar_plan, sidecar_dense_visibility
            plan = build_context_sidecar_plan(clean_input_ids=ids, attention_mask=attention,
                spans=batch['reasoning_step_spans'], visual_only=True)
            visibility = sidecar_dense_visibility(plan.root_length, plan.groups[0])
            branches = [{'targets': list(b.target_token_ids), 'tokens': list(b.token_ids),
                         'positions': list(b.logical_positions), 'prefix_end': b.clean_ancestor_end,
                         'blocked': [b.blocked_start,b.blocked_end], 'selected': list(b.selected_positions)} for b in plan.groups[0]]
        records.append({'sample_id': prepared.sample_id, 'train_index': index,
                        'latent_blocks': counts[index], 'input_ids': ids[0].tolist(),
                        'labels': batch['labels'][0].tolist(), 'branches': branches,
                        'visibility_shape': list(visibility.shape),
                        'visibility_sha256': hashlib.sha256(visibility.numpy().tobytes()).hexdigest()})
    root = Path(args.source)
    source_paths = ['src/stage2/data.py', 'src/stage2/trainer.py', 'src/step_preference/spans.py', 'src/step_preference/dpo_loss.py',
                    'src/step_preference/packing.py' if external else 'src/step_preference/sidecar.py']
    common_logits = _common_logit_probe(external)
    shared = {name: _sha(module.__file__) for name,module in tuple(sys.modules.items())
              if name.startswith('transformers') and getattr(module,'__file__',None)
              and Path(module.__file__).is_file()}
    if external:
        source_paths += ['transformers/src/transformers/models/qwen2_5_vl/'+name+'.py'
                         for name in ('stage2_prefix_cache','stage2_latent_rollout','stage2_sidecar')]
    return {'samples': records, 'common_logits': common_logits, 'shared_dependency_hashes': shared,
            'source_scope': 'source-level differential with shared current Transformers scaffolding',  'source_hashes': {p: _sha(root/p) for p in source_paths},
            'data_sha256': _sha(args.data), 'processor_path': args.processor,
            'parent_scaling': scaling}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--external-source', type=Path)
    parser.add_argument('--data', default='data/TRAIN.jsonl')
    parser.add_argument('--image-root', default='data')
    parser.add_argument('--processor', default='checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431')
    parser.add_argument('--worker', choices=('current','external'), help=argparse.SUPPRESS)
    parser.add_argument('--source', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    # Local Python build marker is rejected by this pinned upstream package.
    original = platform.python_version
    if original().endswith('+'):
        platform.python_version = lambda: original().rstrip('+')
    if args.worker:
        print(json.dumps(_dataset_worker(args), sort_keys=True))
        return 0
    if args.output_dir is None or args.external_source is None:
        parser.error('--output-dir and --external-source are required')
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if not output.is_relative_to((repo/'results').resolve()):
        parser.error('--output-dir must be under this repository results/')
    output.mkdir(parents=True, exist_ok=False)
    logdir = repo/'logs'/output.name
    logdir.mkdir(parents=True, exist_ok=False)
    results = {}
    for flavor, source in [('current',repo),('external',args.external_source.resolve())]:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', PYTHONPATH=os.pathsep.join([str(source/'src'),str(repo/'transformers/src')]), TOKENIZERS_PARALLELISM='false')
        command = [sys.executable,str(Path(__file__).resolve()),'--worker',flavor,'--source',str(source),
                   '--data',str(Path(args.data).resolve()),'--image-root',str(Path(args.image_root).resolve()),
                   '--processor',str(Path(args.processor).resolve())]
        completed = subprocess.run(command, env=env, capture_output=True, text=True)
        (logdir/f'{flavor}.log').write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(f'{flavor} worker failed; see {logdir}/{flavor}.log')
        results[flavor] = json.loads(completed.stdout)
        (output/f'{flavor}.json').write_text(json.dumps(results[flavor], indent=2)+'\n')
    mismatches = compare_records(results['current']['samples'],results['external']['samples'])
    import torch
    left, right = (results[k]['common_logits'] for k in ('current','external'))
    common_comparison = {
        field: gradient_comparison(torch.tensor(left[field]),torch.tensor(right[field]))
        for field in ('loss','preference_loss','clean_scores','ablated_scores','gradients')}
    report = {'common_logit_comparison': common_comparison, 'full_training_authorized': False, 'status': 'blocked', 'data_mismatches': mismatches,
              'parent_scaling': {k:v['parent_scaling'] for k,v in results.items()},
              'missing_evidence': ['common-input model loss and parameter gradients against external snapshot',
                 'actual four-rank DeepSpeed effective gradients, preclip norm, nonzero-LR AdamW deltas and partial group',
                 'actual sampled IDs and retained/padding coverage at max clean sequence length 8192',
                 'production dropout/replay and longest-retained-input memory checks'],
              'scope': 'CPU real TRAIN collation, pure sidecar visibility/replay, parent training_step scalar characterization; no training launched'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report':str(output/'report.json'),'status':'blocked','data_mismatches':len(mismatches)}))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
