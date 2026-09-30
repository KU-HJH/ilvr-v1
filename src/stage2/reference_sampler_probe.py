"""Opt-in two-update sampler witness; same-forward arithmetic, not model equivalence."""
import argparse
import json
import math
from pathlib import Path
import socket

import torch

from .reproduction_probe import (ReproductionObserver, compare_vectors,
    describe_collective_slices, install_collective_observation)


def compare_fixed_gradient(expected, actual):
    """Previously approved BF16 global-L2 criterion; preserve strict diagnostics."""
    coordinate = compare_vectors(expected, actual)
    norm = coordinate.get('expected_l2')
    error = coordinate.get('l2_error')
    relative = (error / norm if norm else (0. if error == 0. else None)) if error is not None else None
    return {'pass': bool(coordinate.get('finite') and relative is not None and relative <= .02),
            'relative_l2': relative, 'relative_l2_limit': .02, 'coordinate': coordinate}


def check_records(records, expected_order):
    """Fail closed on missing or contradictory bounded runtime evidence."""
    failures = []
    def require(condition, message):
        if not condition:
            failures.append(message)
    require(len(expected_order) == 11 and len(set(expected_order)) == 11, 'expected order must contain 11 distinct IDs')
    require({r.get('rank') for r in records} == set(range(4)), 'four ranks not present')
    initial = [r for r in records if r.get('event') == 'init']
    require(len(initial) == 4 and {r['rank'] for r in initial} == set(range(4)), 'one init per rank required')
    hosts = [r.get('hostname') for r in initial]
    require(len(set(hosts)) == 2 and all(hosts.count(h) == 2 for h in set(hosts)), 'two nodes with two ranks each required')
    for r in initial:
        require((r.get('world_size'),r.get('gas'),r.get('microbatch'),r.get('fixed_denominator'),r.get('zero_stage')) == (4,2,1,8,2), 'wrong initialized geometry')
    examples = [r for r in records if r.get('event') == 'example']
    examples.sort(key=lambda r:(r['micro'],r['rank']))
    require([(r['micro'],r['rank']) for r in examples] == [(m,r) for m in range(3) for r in range(4)], 'expected exactly three microsteps on all ranks')
    require([r.get('sample_id') for r in examples if r.get('sample_index',-1)>=0] == expected_order, 'real ID order or membership differs')
    require(sorted(r['sample_index'] for r in examples if r.get('sample_index',-1)>=0) == list(range(11)), 'real index coverage differs')
    require(len({r.get('latent_blocks') for r in examples if r.get('weight') == 1}) > 1, 'mixed latent counts not observed')
    for r in examples:
        padding = r['micro'] == 2 and r['rank'] == 3
        require(r.get('weight') == (0. if padding else 1.), 'wrong example weight')
        require((r.get('sample_index') == -1) == padding, 'padding index is not final rank slot')
        require(r.get('update') == (1 if r['micro']<2 else 2), 'wrong example update assignment')
        require(r.get('reference_zero_hook_calls') == 0, 'oracle invoked reduction hooks')
        if padding:
            require(r.get('loss') == 0. and r.get('reference_gradient_max_abs') == 0., 'padding reference is nonzero')
    backwards = [r for r in records if r.get('event') == 'backward']
    require(sorted((r['micro'],r['rank']) for r in backwards) == [(m,r) for m in range(3) for r in range(4)], 'missing actual backward evidence')
    for r in backwards:
        require(r.get('finite') is True and r.get('parameter_count',0)>0, 'missing or nonfinite actual gradients')
        if r['micro'] == 2 and r['rank'] == 3:
            require(r.get('gradient_max_abs') == 0., 'actual padding gradient is nonzero')
    collective_fields = ('micro','sequence','layout','slices','tensor_shapes','communication_dtype','reduce_rank')
    layouts = []
    for rank in range(4):
        events = [r for r in records if r.get('event') == 'collective' and r['rank'] == rank]
        require({r['micro'] for r in events} == {0,1,2}, 'collectives missing for a microstep')
        require([r['sequence'] for r in events] == list(range(len(events))), 'collective sequence incomplete')
        require(all(r.get('layout') and r.get('slices') for r in events), 'empty collective layout')
        layouts.append([{k:r.get(k) for k in collective_fields} for r in events])
    require(all(layout == layouts[0] for layout in layouts[1:]), 'collective bucket or payload order differs across ranks')
    updates = [r for r in records if r.get('event') == 'update']
    require(sorted((r['update'],r['rank']) for r in updates) == [(u,r) for u in (1,2) for r in range(4)], 'two completed updates per rank required')
    for r in updates:
        require(r.get('completed') is True, 'optimizer step completion missing')
        require(r.get('fixed_denominator') == 8, 'fixed denominator changed')
        require((r.get('microsteps'),r.get('actual_count')) == ((2,8) if r['update']==1 else (1,3)), 'full or tail geometry differs')
        comparison=r.get('fixed_gradient',{})
        coordinate=comparison.get('coordinate',{})
        norm,error=coordinate.get('expected_l2'),coordinate.get('l2_error')
        numeric = (isinstance(norm,(int,float)) and isinstance(error,(int,float))
                   and math.isfinite(norm) and norm>=0 and math.isfinite(error)
                   and ((error == 0.) if norm == 0. else 0<=error/norm<=.02))
        require(comparison.get('pass') is True and coordinate.get('finite') is True and numeric
                and comparison.get('relative_l2_limit') == .02,
                'fixed-denominator distributed gradient differs')
    return {'pass':not failures,'failures':failures,'ranks':4,'real_samples':11,'padding_slots':1,
            'updates':2,'fixed_denominator':8,'arithmetic_scope':'same-forward per-example autograd sum /8 versus actual ZeRO gradient; BF16 global relative L2 <=2%'}


class ReferenceSamplerObserver(ReproductionObserver):
    """Reuse audited DeepSpeed output-hook isolation, without legacy GAS4 semantics."""
    def __init__(self, trainer, engine, output_dir):
        import torch.distributed as dist
        self.engine, self.rank, self.world = engine, dist.get_rank(), dist.get_world_size()
        if (self.world != 4 or trainer.deepspeed_runtime_profile.zero_stage != 2
                or trainer.args.per_device_train_batch_size != 1 or trainer.args.gradient_accumulation_steps != 2
                or trainer.state.global_step != 0 or trainer.stage2_sampler_policy != 'reference_random'):
            raise ValueError('sampler witness requires fresh reference_random four-rank ZeRO2 micro1/GAS2')
        if trainer.model_accepts_loss_kwargs or trainer.compute_loss_func is not None:
            raise ValueError('sampler witness requires fixed Trainer GAS scaling')
        self.named = [(name,p) for name,p in engine.module.named_parameters() if p.requires_grad]
        self.parameters = [p for _,p in self.named]
        if not 0 < sum(p.numel() for p in self.parameters) <= 100_000_000:
            raise ValueError('sampler witness requires 1..100M trainable elements')
        self.names = {id(p):name for name,p in self.named}
        path = Path(output_dir).resolve()
        if not path.is_relative_to((Path.cwd()/'results').resolve()):
            raise ValueError('probe output must be under current results tree')
        path.mkdir(parents=True,exist_ok=True)
        self.stream = (path/f'rank-{self.rank}.jsonl').open('x')
        self.total, self.pending = None, None
        self.update, self.micro, self.sequence = 0, -1, 0
        self.group_weights = []
        self.hook_calls = 0
        self.actual_gradients = {}
        self._reference_output_managers = []
        self._output_capture_hook = engine.register_forward_hook(self._capture_output_hooks)
        zero = engine.optimizer
        if not zero.reduce_scatter or not zero.contiguous_gradients or float(zero.loss_scale) != 1.:
            raise ValueError('sampler witness requires contiguous reduce-scatter BF16 ZeRO2')
        install_collective_observation(self, zero)
        self.write(dict(event='init',rank=self.rank,world_size=self.world,hostname=socket.gethostname(),
                        gas=2,microbatch=1,fixed_denominator=8,zero_stage=2,
                        reference='same production forward weighted numerator; independent summation/fixed divisor',
                        relative_l2_limit=.02,parameter_names=[name for name,_ in self.named]))

    def _finish_backward(self):
        if self.micro < 0:
            return
        values = list(self.actual_gradients.values())
        self.write(dict(event='backward',rank=self.rank,micro=self.micro,parameter_count=len(values),
                        finite=bool(values) and all(v[0] for v in values),
                        gradient_max_abs=max((v[1] for v in values),default=None)))
        self.actual_gradients.clear()

    def observe_loss(self, loss, inputs):
        if self.update >= 2 or self.micro >= 2:
            raise RuntimeError('sampler witness is bounded to 11 examples/two updates')
        self._finish_backward()
        self.micro += 1
        ids = inputs.get('sample_ids')
        if not isinstance(ids,(tuple,list)) or len(ids) != 1 or not isinstance(ids[0],str):
            raise ValueError('sampler witness requires an actual sample ID')
        weight = float(inputs['stage2_sample_weight'])
        index = int(inputs['stage2_sample_index'])
        previous = self.hook_calls
        gradients = self._reference_gradients(loss)
        if self.hook_calls != previous:
            raise RuntimeError('reference backward invoked ZeRO reduction hooks')
        if not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError('nonfinite reference numerator')
        maximum = max(float(g.abs().max()) for g in gradients)
        if self.total is None:
            self.total = gradients
        else:
            for total,gradient in zip(self.total,gradients):
                total.add_(gradient)
        self.group_weights.append(weight)
        self.write(dict(event='example',rank=self.rank,micro=self.micro,update=self.update+1,
                        sample_id=ids[0],sample_index=index,weight=weight,
                        latent_blocks=int(inputs['stage2_sample_latent_block_count']),
                        loss=float(loss.detach()),reference_gradient_max_abs=maximum,
                        reference_zero_hook_calls=self.hook_calls-previous))

    def before_step(self):
        import torch.distributed as dist
        from deepspeed.utils import safe_get_full_grad
        if self.total is None:
            raise RuntimeError('empty sampler witness optimizer group')
        weights = [None]*self.world
        dist.all_gather_object(weights,self.group_weights)
        expected, actual = [], []
        for parameter,total in zip(self.parameters,self.total):
            summed = total.to(parameter.device)
            dist.all_reduce(summed)
            expected.append(summed.cpu()/8.)
            gradient = safe_get_full_grad(parameter)
            if gradient is None:
                raise RuntimeError('full ZeRO gradient unavailable')
            actual.append(gradient.detach().float().cpu())
        self.pending = dict(event='update',rank=self.rank,update=self.update+1,
                            microsteps=len(self.group_weights),actual_count=sum(sum(w) for w in weights),
                            fixed_denominator=8,fixed_gradient=compare_fixed_gradient(expected,actual))

    def after_step(self):
        if self.pending is None:
            raise RuntimeError('step completion lacks preceding gradient witness')
        self.pending['completed'] = True
        self.write(self.pending)
        self.update += 1
        self.total,self.pending,self.group_weights = None,None,[]
        if self.update == 2:
            self._finish_backward()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',required=True)
    parser.add_argument('--expected-order',required=True,help='JSON list of 11 sample IDs in independently derived reference order')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    records=[]
    for rank in range(4):
        path=Path(args.input_dir)/f'rank-{rank}.jsonl'
        if path.exists():
            records.extend(json.loads(line) for line in path.read_text().splitlines())
    report=check_records(records,json.loads(Path(args.expected_order).read_text()))
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
