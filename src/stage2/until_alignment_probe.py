"""Bounded Trainer/ZeRO comparison evidence; never authorizes a full run itself."""
import itertools
import json
from pathlib import Path
import socket

import torch

from .reproduction_probe import ReproductionObserver

NUMERICAL_PROTOCOL = {
    'version': 1,
    'metrics': {'relative_l2': 'norm(a-b)/max(norm(a),norm(b))',
                'relative_linf': 'max(abs(a-b))/max(max(abs(a)),max(abs(b)))'},
    'controls': 3, 'control_maximum': .01, 'multiplier': 2., 'absolute_cap': .02,
    'zero_rule': 'zero denominator requires exact zero error; zero envelope requires exact equality',
    'structural': 'names, shapes, dtypes, connectivity, input IDs, RNG, weights exactly equal',
    'ordering': 'freeze external-only controls before opening aligned artifacts',
    'scope': 'bounded common-runtime comparison; not statistical or historical trajectory proof',
}


def tensor_metrics(left, right):
    if left is None or right is None:
        return {'structural': left is None and right is None, 'relative_l2': 0., 'relative_linf': 0.}
    structural = left.shape == right.shape and left.dtype == right.dtype
    if not structural or not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        return {'structural': False}
    left, right = left.double(), right.double()
    delta = (left-right).abs()
    error = float(delta.norm())
    maximum = float(delta.max()) if delta.numel() else 0.
    norm = max(float(left.norm()), float(right.norm()))
    scale = max(float(left.abs().max()), float(right.abs().max())) if left.numel() else 0.
    return {'structural': True, 'relative_l2': error/norm if norm else 0.,
            'relative_linf': maximum/scale if scale else 0., 'l2_error': error, 'max_abs_error': maximum}


def freeze_envelope(controls):
    if len(controls) != 3 or not controls[0] or any(c.keys() != controls[0].keys() for c in controls):
        raise ValueError('three identically named nonempty controls required')
    envelope = {}
    for name in controls[0]:
        pairs = [tensor_metrics(a[name], b[name]) for a,b in itertools.combinations(controls, 2)]
        if any(not p['structural'] for p in pairs):
            raise ValueError(f'control structure/nonfinite mismatch: {name}')
        maxima = {metric: max(p[metric] for p in pairs) for metric in ('relative_l2','relative_linf')}
        if any(v > .01 for v in maxima.values()):
            raise ValueError(f'unstable controls: {name}')
        envelope[name] = {k: min(.02, 2*v) for k,v in maxima.items()}
    return envelope


def compare_snapshot(reference, candidate, envelope):
    if reference.keys() != candidate.keys() or reference.keys() != envelope.keys():
        return {'pass': False, 'reason': 'named tensor inventory differs'}
    results = {}
    for name, value in reference.items():
        metrics = tensor_metrics(value, candidate[name])
        metrics['pass'] = metrics['structural'] and all(
            metrics[k] <= envelope[name][k] for k in ('relative_l2','relative_linf'))
        results[name] = metrics
    return {'pass': all(m['pass'] for m in results.values()), 'tensors': results}


def select_windows(order, lengths, latent_counts, *, source_updates=None):
    """Select first three groups, a manifest-cost challenge, and real tail."""
    if len(order) < 52 or len(order)%16 != 4:
        raise ValueError('bounded replay requires three full groups and a four-real-example tail')
    groups = [list(order[i:i+16]) for i in range(0,len(order),16)]
    # Attention/replay proxy, not a claim of global peak GPU allocation.
    costs = [sum(lengths[i]**2*(1+latent_counts[i]) for i in group) for group in groups[:-1]]
    challenge = max(range(len(costs)), key=lambda i:(costs[i],-i))
    selected = [0,1,2,challenge,len(groups)-1]
    if source_updates is not None:
        if (not isinstance(source_updates,list) or len(source_updates)!=2
                or any(type(i) is not int or not 1<=i<len(groups) for i in source_updates)
                or len(set(source_updates))!=2):
            raise ValueError('memory probe requires two distinct full source updates')
        selected = [i-1 for i in source_updates]
    indices = [j for i in selected for j in groups[i]]
    replay = {'source_updates': [i+1 for i in selected],
            'indices': indices,
            'coefficients': [1/16]*len(indices), 'challenge_source_update': challenge+1,
            'challenge_proxy': 'sum(length^2*(1+latent_count))',
            'historical_optimizer_state_reproduced': False}
    if source_updates is not None:
        replay.update(lengths=[lengths[i] for i in indices],
                      latent_counts=[latent_counts[i] for i in indices])
    return replay


def tensor_destination(path):
    path = Path(path).resolve()
    if not path.is_relative_to((Path.cwd()/'checkpoints').resolve()):
        raise ValueError('tensor artifacts must be under current checkpoints tree')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cpu(value):
    return None if value is None else value.detach().cpu().clone()


class AlignmentObserver(ReproductionObserver):
    """Observe actual Trainer backward/ZeRO steps, retaining named CPU witnesses."""
    def __init__(self, trainer, engine, spec, replay):
        import torch.distributed as dist
        self.trainer, self.engine = trainer, engine
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        if (self.world, trainer.args.per_device_train_batch_size, trainer.args.gradient_accumulation_steps,
                engine.zero_optimization_stage(), trainer.args.max_steps, trainer.args.warmup_steps) != (4,1,4,2,212,10):
            raise ValueError('gate requires four-rank ZeRO2 micro1/GAS4 schedule212/warmup10')
        if trainer.model_accepts_loss_kwargs or trainer.compute_loss_func is not None:
            raise ValueError('fixed Trainer GAS4 denominator required')
        self.named = [(n,p) for n,p in engine.module.named_parameters() if p.requires_grad]
        self.parameters = [p for _,p in self.named]
        self.tensor_dir = tensor_destination(spec['tensor_dir']) / f'rank-{self.rank}'
        self.tensor_dir.mkdir(exist_ok=False)
        result_dir = Path(spec['result_dir']).resolve()
        if not result_dir.is_relative_to((Path.cwd()/'results').resolve()):
            raise ValueError('gate metadata must be under current results tree')
        result_dir.mkdir(parents=True,exist_ok=True)
        self.stream = (result_dir/f'rank-{self.rank}.jsonl').open('x')
        self.replay, self.micro, self.update = replay, -1, 0
        self.totals = {'preference': None, 'total': None}
        self.pending = None
        self._reference_output_managers = []
        self._output_capture_hook = engine.register_forward_hook(self._capture_output_hooks)
        self.hook_calls = 0
        original = engine.optimizer.process_gradients
        def count_hooks(*args, **kwargs):
            self.hook_calls += 1
            return original(*args, **kwargs)
        engine.optimizer.process_gradients = count_hooks
        self.actual_hook_counts = []
        self.write(dict(event='init',rank=self.rank,hostname=socket.gethostname(),world_size=4,
            gas=4,microbatch=1,fixed_denominator=16,zero_stage=2,schedule_steps=212,warmup_steps=10,
            parameter_names=[n for n,_ in self.named],path=spec['path'],replay=replay))
        torch.cuda.reset_peak_memory_stats(engine.device)

    def _capture_output_hooks(self, engine, args, outputs):
        # The frozen external path embeds tensor-bearing sidecar dataclasses.
        from dataclasses import is_dataclass
        def containers(value):
            if isinstance(value, dict):
                return {k:containers(v) for k,v in value.items()}
            if isinstance(value, (list,tuple)):
                return [containers(v) for v in value]
            if is_dataclass(value):
                return None  # DeepSpeed OutputBackwardHookManager also treats dataclasses as opaque.
            return value
        return super()._capture_output_hooks(engine,args,containers(outputs))

    def save(self, label, values):
        torch.save(values, self.tensor_dir/f'{label}.pt')

    def gradients(self, loss):
        """Same audited output-hook isolation, retaining None versus connected zero."""
        from deepspeed.runtime.utils import OutputBackwardHookManager
        managers = self._reference_output_managers
        if not managers or self.engine._running_engine_backward or self.engine.optimizer._backward_active_depth != 0:
            raise RuntimeError('unsupported DeepSpeed output hook state')
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        leaf_state = [(p.grad, None if p.grad is None else p.grad._version) for p in self.parameters]
        previous = self.hook_calls
        saved = []
        try:
            for manager in managers:
                if (type(manager) is not OutputBackwardHookManager or manager.preprocess_done is not False
                        or manager.preprocess_once_fn != self.engine._backward_prologue
                        or manager.preprocess_per_tensor_fn != self.engine._backward_prologue_per_tensor
                        or not manager.hook_handles):
                    raise RuntimeError('unsupported DeepSpeed output hook manager')
                saved.append((manager, manager.preprocess_done, manager.preprocess_per_tensor_fn))
                manager.preprocess_done, manager.preprocess_per_tensor_fn = True, None
            gradients = torch.autograd.grad(loss, self.parameters, retain_graph=True, allow_unused=True)
        finally:
            for manager,done,per_tensor in saved:
                manager.preprocess_done,manager.preprocess_per_tensor_fn = done,per_tensor
        if (self.hook_calls != previous or not torch.equal(cpu_rng,torch.get_rng_state())
                or any(not torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
                or any(p.grad is not old or (old is not None and old._version != version)
                       for p,(old,version) in zip(self.parameters,leaf_state))):
            raise RuntimeError('oracle backward mutated Trainer RNG/leaf gradients/ZeRO state')
        return {n:_cpu(g) for (n,_),g in zip(self.named,gradients)}

    def observe(self, total, preference, inputs, scores):
        self.micro += 1
        if self.micro >= 17:
            raise RuntimeError('bounded gate exceeded 17 microsteps')
        index = self.micro*4+self.rank
        expected_id = self.replay['sample_ids'][index]
        if inputs['sample_ids'] != [expected_id] and inputs['sample_ids'] != (expected_id,):
            raise ValueError('consumed ID differs from frozen external sampler replay')
        self.save(f'micro-{self.micro}-scores', scores)
        self.save(f'micro-{self.micro}-rng', {'cpu':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state(self.engine.device)})
        for label, loss in (('preference',preference),('total',total)):
            values = self.gradients(loss)
            # These raw named gradients retain missing-versus-connected-zero evidence.
            self.save(f'micro-{self.micro}-{label}',values)
            if self.totals[label] is None:
                self.totals[label] = {n:None if g is None else g.float() for n,g in values.items()}
            else:
                for name,gradient in values.items():
                    if gradient is not None:
                        old = self.totals[label][name]
                        self.totals[label][name] = gradient.float() if old is None else old+gradient.float()
        self._reference_output_managers.clear()
        self.actual_hook_counts.append(self.hook_calls)
        self.write(dict(event='example',rank=self.rank,micro=self.micro,sample_id=expected_id,
            original_update_index=self.replay['source_updates'][min(self.micro//4,4)],
            probe_optimizer_step=self.update+1,coefficient=1/16,weight=1.,
            tokens=int(inputs['input_ids'].shape[-1]),loss=float(total.detach()),
            peak_allocated=torch.cuda.max_memory_allocated(self.engine.device)))

    def before_step(self):
        import torch.distributed as dist
        from deepspeed.utils import safe_get_full_grad,safe_get_full_fp32_param
        if self.update >= 5 or self.totals['total'] is None:
            raise RuntimeError('unexpected optimizer event')
        references = {}
        for label,values in self.totals.items():
            aggregate = {}
            for name,parameter in self.named:
                value = values[name]
                present = torch.tensor(int(value is not None),device=parameter.device)
                dist.all_reduce(present)
                tensor = torch.zeros_like(parameter,dtype=torch.float32) if value is None else value.to(parameter.device)
                dist.all_reduce(tensor)
                aggregate[name] = None if present.item()==0 else tensor.cpu()/16
            references[label] = aggregate
            self.save(f'update-{self.update+1}-{label}-oracle',aggregate)
        actual, master, bf16 = {},{},{}
        for name,parameter in self.named:
            actual[name] = _cpu(safe_get_full_grad(parameter))
            master[name] = _cpu(safe_get_full_fp32_param(parameter))
            bf16[name] = _cpu(parameter)
            if master[name] is None:
                raise RuntimeError('ZeRO FP32 master unavailable')
        self.save(f'update-{self.update+1}-total-actual',actual)
        self.save(f'update-{self.update+1}-master-before',master)
        self.save(f'update-{self.update+1}-bf16-before',bf16)
        self.preference_nonzero=any(v is not None and bool(v.abs().max()>0) for v in references['preference'].values())
        self.pending = (master,bf16)
        options = self.engine.optimizer.optimizer.param_groups
        self.lr = [float(g['lr']) for g in options]
        # Independent coefficient oracle, fixed before any cross-path comparison.
        comparable = {n: torch.zeros_like(actual[n]) if v is None and actual[n] is not None else v
                      for n,v in references['total'].items()}
        arithmetic = {n:tensor_metrics(v,actual[n]) for n,v in comparable.items()}
        # Arithmetic rounding policy is explicitly reported, never confused with the repeat envelope.
        self.arithmetic = arithmetic
        from .reference_sampler_probe import compare_fixed_gradient
        connected = [(comparable[n],actual[n]) for n in comparable if comparable[n] is not None and actual[n] is not None]
        self.fixed_gradient = compare_fixed_gradient([a for a,b in connected], [b for a,b in connected])
        if any((comparable[n] is None) != (actual[n] is None) for n in comparable):
            raise RuntimeError("oracle/actual total-gradient connectivity mismatch")

    def after_step(self):
        from deepspeed.utils import safe_get_full_fp32_param
        before,before_bf16 = self.pending
        master, bf16 = {},{}
        for name,parameter in self.named:
            master[name] = _cpu(safe_get_full_fp32_param(parameter))-before[name]
            bf16[name] = _cpu(parameter).float()-before_bf16[name].float()
        self.update += 1
        self.save(f'update-{self.update}-master-delta',master)
        self.save(f'update-{self.update}-bf16-delta',bf16)
        self.write(dict(event='update',rank=self.rank,probe_optimizer_step=self.update,
            original_update_index=self.replay['source_updates'][self.update-1],completed=True,
            actual_count=4 if self.update==5 else 16,fixed_denominator=16,lr=self.lr,
            master_delta_nonzero=any(bool(v.abs().max()>0) for v in master.values()),
            preference_gradient_nonzero=self.preference_nonzero,
            oracle_arithmetic=self.arithmetic,fixed_gradient=self.fixed_gradient,actual_backward_hooks=self.hook_calls,
            peak_allocated=torch.cuda.max_memory_allocated(self.engine.device)))
        self.pending = None
        self.totals = {'preference':None,'total':None}


class MemoryDiagnosticObserver(AlignmentObserver):
    """Ordinary Trainer backward owns gradients; optional detached boundary reads."""
    def __init__(self, trainer, engine, spec, replay):
        super().__init__(trainer,engine,spec,replay)
        self.capture_ordinary_gradients = spec['path']=='aligned'
        if self.capture_ordinary_gradients:
            from torch._functorch import config
            if spec.get('save_on_cpu',False) or engine.optimizer.cpu_offload:
                raise ValueError('aligned ordinary probe prohibits CPU offload')
            self.write(dict(event='ordinary_probe_config',rank=self.rank,
                off_activation_checkpointing=trainer.step_preference_config.off_activation_checkpointing,
                save_on_cpu=False,optimizer_cpu_offload=False,donated_buffer=config.donated_buffer))
    def memory(self, event, **fields):
        record = dict(event=event, rank=self.rank, micro=self.micro,
            allocated=torch.cuda.memory_allocated(self.engine.device),
            peak_allocated=torch.cuda.max_memory_allocated(self.engine.device))
        record.update(fields)
        self.write(record)

    def observe(self, total, preference, inputs, scores):
        self.micro += 1
        expected_id = self.replay['sample_ids'][self.micro*4+self.rank]
        if self.micro >= 8 or list(inputs['sample_ids']) != [expected_id]:
            raise ValueError('memory diagnostic replay differs')
        self.save(f'micro-{self.micro}-scores', scores)
        self.save(f'micro-{self.micro}-rng', {'cpu':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state(self.engine.device)})
        self._reference_output_managers.clear()
        self.memory('forward', sample_id=expected_id, tokens=int(inputs['input_ids'].shape[-1]),
                    loss=float(total.detach()),original_update_index=self.replay['source_updates'][self.micro//4],
                    latent_blocks=len(inputs.get('reasoning_step_spans', ())))

    def before_step(self):
        self.lr = [float(group['lr']) for group in self.engine.optimizer.optimizer.param_groups]
        if getattr(self,'capture_ordinary_gradients',False):
            from deepspeed.utils import safe_get_full_grad,safe_get_full_fp32_param
            actual={name:_cpu(safe_get_full_grad(p)) for name,p in self.named}
            self.save(f'update-{self.update+1}-total-actual',actual)
            # One connected nonzero parameter suffices to witness a real master
            # update without retaining another model-sized snapshot inventory.
            self.master_witness_name,self.master_parameter=next(
                (name,p) for name,p in self.named if actual[name] is not None and bool(actual[name].abs().max()>0))
            self.master_before=_cpu(safe_get_full_fp32_param(self.master_parameter))
            if self.master_before is None:
                raise RuntimeError('ordinary probe FP32 master witness unavailable')

    def after_step(self):
        self.update += 1
        fields={}
        if getattr(self,'capture_ordinary_gradients',False):
            from deepspeed.utils import safe_get_full_fp32_param
            after=_cpu(safe_get_full_fp32_param(self.master_parameter))
            fields=dict(master_witness_name=self.master_witness_name,
                        master_changed=not torch.equal(self.master_before,after))
            self.master_before=None
        self.memory('memory_update', probe_optimizer_step=self.update, lr=self.lr,
            original_update_index=self.replay['source_updates'][self.update-1],
            ordinary_gradient_snapshot=(f'update-{self.update}-total-actual.pt'
                if getattr(self,'capture_ordinary_gradients',False) else None),**fields)


def check_memory_runtime(records, replay):
    """Capacity witness only: no numerical-equivalence or full-training approval."""
    failures = []
    initial = [r for r in records if r['event']=='init']
    hosts = [r.get('hostname') for r in initial]
    if (sorted(r['rank'] for r in initial)!=list(range(4)) or len(set(hosts))!=2
            or any(hosts.count(h)!=2 for h in hosts)):
        failures.append('four ranks on two nodes required')
    for r in initial:
        if tuple(r.get(k) for k in ('world_size','microbatch','gas','zero_stage','schedule_steps','warmup_steps')) != (4,1,4,2,212,10):
            failures.append('runtime profile differs')
    for rank in range(4):
        local = [r for r in records if r['rank']==rank]
        aligned=any(r.get('path')=='aligned' for r in local if r['event']=='init')
        settings=[r for r in local if r['event']=='ordinary_probe_config']
        if aligned and (len(settings)!=1 or settings[0].get('save_on_cpu') is not False
                        or settings[0].get('optimizer_cpu_offload') is not False
                        or type(settings[0].get('off_activation_checkpointing')) is not bool
                        or type(settings[0].get('donated_buffer')) is not bool):
            failures.append('aligned ordinary runtime settings missing or offloaded')
        for event in ('micro_entry','forward','backward'):
            rows = [r for r in local if r['event']==event]
            if [r['micro'] for r in rows] != list(range(8)):
                failures.append(f'rank{rank} missing {event}')
            for row in rows:
                if (row.get('sample_id')!=replay['sample_ids'][row['micro']*4+rank]
                        or not 0 <= row.get('allocated',-1) <= row.get('peak_allocated',-1)):
                    failures.append('input identity or memory evidence differs')
                if event=='backward' and row.get('actual_backward_hooks',0)<=0:
                    failures.append('ordinary backward not witnessed')
                if aligned:
                    index=row['micro']*4+rank
                    if (row.get('original_update_index')!=replay['source_updates'][row['micro']//4]
                            or row.get('tokens')!=replay['lengths'][index]
                            or row.get('latent_blocks')!=replay['latent_counts'][index]):
                        failures.append('selected source geometry differs')
                    if event=='backward':
                        counts=row.get('off_decoder_calls') or {}
                        enabled=settings[0].get('off_activation_checkpointing') if settings else None
                        if (counts.get('forward')!=1 or type(counts.get('backward')) is not int
                                or (counts.get('backward',0)>0)!=enabled or row.get('dora_scope_cleaned') is not True):
                            failures.append('actual OFF replay or DoRA cleanup missing')
        updates = [r for r in local if r['event']=='memory_update']
        if [r['probe_optimizer_step'] for r in updates] != [1,2] or [r['lr'] for r in updates] != [[0.],[5e-6]]:
            failures.append('two scheduled optimizer events required')
        if aligned:
            for row in updates:
                step=row['probe_optimizer_step']
                if (row.get('original_update_index')!=replay['source_updates'][step-1]
                        or row.get('ordinary_gradient_snapshot')!=f'update-{step}-total-actual.pt'
                        or not row.get('master_witness_name') or row.get('master_changed') is not (step==2)):
                    failures.append('ordinary gradient or real master update witness missing')
        compiler = [r for r in local if r['event']=='compiler']
        calls = [c for r in compiler for c in r['calls'] if c['callable']=='torch.nn.attention.flex_attention.flex_attention']
        donation=settings[0].get('donated_buffer') if aligned and settings else False
        if len(compiler)!=1 or not calls or any(c.get('dynamic') is not True or c.get('donated_buffer') is not donation for c in calls):
            failures.append('compiler profile differs')
    return {'complete':not failures, 'comparison_pass':False, 'full212_authorized':False,
            'failures':failures, 'scope':'two selected replay groups ordinary-backward capacity and gradient observations only'}


def check_runtime(records, replay):
    """Recompute event/weight acceptance from evidence, never trust a pass flag."""
    import math
    failures=[]
    def require(ok,message):
        if not ok:
            failures.append(message)
    initial=[r for r in records if r.get('event')=='init']
    require(sorted(r['rank'] for r in initial)==list(range(4)), 'missing init rank')
    hosts=[r.get('hostname') for r in initial]
    require(len(set(hosts))==2 and all(hosts.count(h)==2 for h in set(hosts)), 'two nodes required')
    for r in initial:
        require(tuple(r.get(k) for k in ('world_size','gas','microbatch','fixed_denominator','zero_stage','schedule_steps','warmup_steps'))==(4,4,1,16,2,212,10),'wrong runtime geometry')
    compilers=[r for r in records if r.get('event')=='compiler']
    require(sorted(r['rank'] for r in compilers)==list(range(4)), 'missing actual compiler dispatch')
    for r in compilers:
        calls=[c for c in r.get('calls',[]) if c.get('callable')=='torch.nn.attention.flex_attention.flex_attention']
        require(bool(calls) and all(c.get('dynamic') is True for c in calls),'Flex dynamic=True dispatch differs')
    examples=sorted((r for r in records if r.get('event')=='example'),key=lambda r:(r['micro'],r['rank']))
    require([(r['micro'],r['rank']) for r in examples]==[(m,r) for m in range(17) for r in range(4)],'missing real microstep')
    require([r.get('sample_id') for r in examples]==replay['sample_ids'],'consumed IDs differ')
    for r in examples:
        require(r.get('coefficient')==1/16 and r.get('weight')==1.,'wrong independent example coefficient')
        require(r.get('peak_allocated',0)>0,'memory witness missing')
    backwards=[r for r in records if r.get('event')=='backward']
    require(sorted((r['micro'],r['rank']) for r in backwards)==[(m,r) for m in range(17) for r in range(4)],'missing backward including post-update microbatch')
    for r in backwards:
        require(r.get('dora_norm_calls',0)>0 and r.get('dora_scope_cleaned') is True,'DoRA norm scope not witnessed')
        require(r.get('actual_backward_hooks',0)>0,'actual ZeRO backward not witnessed')
    updates=[r for r in records if r.get('event')=='update']
    require(sorted((r['probe_optimizer_step'],r['rank']) for r in updates)==[(s,r) for s in range(1,6) for r in range(4)],'missing optimizer event including partial tail')
    require(all(any(r.get('preference_gradient_nonzero') is True for r in updates if r['rank']==rank) for rank in range(4)),
            'nonzero accumulated preference signal not witnessed')
    for r in updates:
        step=r['probe_optimizer_step']
        require(r.get('completed') is True and r.get('fixed_denominator')==16,'optimizer event or denominator invalid')
        require(r.get('actual_count')==(4 if step==5 else 16),'tail count invalid')
        require(r.get('original_update_index')==replay['source_updates'][step-1],'source update identity differs')
        require(r.get('lr') and all(abs(v-(step-1)*5e-6)<1e-12 for v in r['lr']),'warmup LR differs')
        require(r.get('master_delta_nonzero') is (step>1),'nonzero LR update not witnessed')
        coordinate=r.get('fixed_gradient',{}).get('coordinate',{})
        norm,error=coordinate.get('expected_l2'),coordinate.get('l2_error')
        require(coordinate.get('finite') is True and isinstance(norm,(int,float)) and isinstance(error,(int,float))
                and math.isfinite(norm) and math.isfinite(error) and norm>=0 and error>=0
                and (error==0 if norm==0 else error/norm<=.02),'independent fixed16 arithmetic failed')
    return {'pass':not failures,'failures':failures,'scope':'five actual optimizer events, projected original212 tail; no historical state212 claim'}


def expected_snapshots(ranks=range(4)):
    expected = {f'rank-{rank}/micro-{micro}-{kind}.pt' for rank in ranks for micro in range(17)
                for kind in ('scores','rng','preference','total','dora-norms')}
    expected |= {f'rank-{rank}/update-{step}-{kind}.pt' for rank in ranks for step in range(1,6)
                 for kind in ('preference-oracle','total-oracle','total-actual','master-before','bf16-before','master-delta','bf16-delta')}
    return expected


def required_snapshots(observed):
    expected = expected_snapshots()
    if observed != expected:
        raise ValueError(f'tensor inventory differs: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}')
