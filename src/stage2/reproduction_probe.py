"""Opt-in, bounded V-off reproduction arithmetic witness; never authorizes training."""
import json
import math
import socket
from pathlib import Path
import torch


def reference_gradients(loss, parameters):
    """Unscaled production numerator, without accumulating into leaf gradients."""
    cpu_rng = torch.get_rng_state().clone()
    devices = sorted({p.device.index for p in parameters if p.is_cuda})
    cuda_rng = {d: torch.cuda.get_rng_state(d).clone() for d in devices}
    versions = [(p.grad, None if p.grad is None else p.grad._version) for p in parameters]
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    if not torch.equal(cpu_rng, torch.get_rng_state()) or any(
        not torch.equal(s, torch.cuda.get_rng_state(d)) for d, s in cuda_rng.items()
    ):
        raise RuntimeError('reference backward changed RNG state')
    if any(p.grad is not old or (old is not None and old._version != version)
           for p, (old, version) in zip(parameters, versions)):
        raise RuntimeError('reference backward changed accumulated leaf gradients')
    return [torch.zeros_like(p, device='cpu', dtype=torch.float32) if g is None
            else g.detach().float().cpu() for p, g in zip(parameters, gradients)]


def compare_vectors(expected, actual):
    # Tolerances fixed before distributed observations; retain errors as evidence.
    if not expected or len(expected) != len(actual) or any(e.shape != a.shape for e,a in zip(expected,actual)):
        return {"pass":False,"reason":"missing vector or shape mismatch"}
    finite = all(bool(torch.isfinite(v).all()) for v in expected + actual)
    squared = sum(float((a.double()-e.double()).square().sum()) for e, a in zip(expected, actual))
    norm = sum(float(e.double().square().sum()) for e in expected)**.5
    return {'pass': finite and all(torch.allclose(e, a, rtol=.02, atol=1e-7) for e, a in zip(expected, actual)),
            'finite': finite, 'l2_error': squared**.5 if finite else None,
            'expected_l2': norm if finite else None,
            'max_abs_error': max(float((a-e).abs().max()) for e, a in zip(expected, actual)) if finite else None,
            'rtol': .02, 'atol': 1e-7}


class AdamReference:
    """CPU FP32 moments only; no model/optimizer clone and no Trainer helpers."""
    def __init__(self):
        self.steps = 0
        self.moments = None

    def delta(self, weights, gradients, options, clip):
        self.steps += 1
        if self.moments is None:
            self.moments = [(torch.zeros_like(g), torch.zeros_like(g)) for g in gradients]
        norm = sum(float(g.double().square().sum()) for g in gradients)**.5
        factor = min(1., clip / (norm + 1e-6)) if clip > 0 else 1.
        result = []
        for p, g, (m, v), opt in zip(weights, gradients, self.moments, options):
            beta1, beta2 = opt['betas']
            g = g * factor
            m.mul_(beta1).add_(g, alpha=1-beta1)
            v.mul_(beta2).addcmul_(g, g, value=1-beta2)
            updated = p.clone()
            updated.mul_(1-opt['lr']*opt['weight_decay'])
            denominator = v.sqrt().div_((1-beta2**self.steps)**.5).add_(opt['eps'])
            updated.addcdiv_(m, denominator, value=-opt['lr']/(1-beta1**self.steps))
            result.append(updated.sub_(p))
        return result


def describe_collective_slices(flat, layout, tensors):
    """Map actual reduction views back to ordered parameter portions, before copying."""
    result = []
    if sum(item['numel'] for item in layout) != flat.numel():
        raise RuntimeError('IPG layout does not cover its actual tensor')
    for tensor in tensors:
        delta = tensor.data_ptr() - flat.data_ptr()
        if (tensor.dtype != flat.dtype or tensor.device != flat.device or not tensor.is_contiguous()
                or delta < 0 or delta % flat.element_size()):
            raise RuntimeError('collective tensor is outside observed IPG storage')
        start = delta // flat.element_size()
        end = start + tensor.numel()
        if end > flat.numel():
            raise RuntimeError('collective tensor is outside observed IPG storage')
        position = 0
        for item in layout:
            stop = position + item['numel']
            left, right = max(position, start), min(stop, end)
            if left < right:
                result.append(dict(name=item['name'], offset=left-position, numel=right-left))
            position = stop
    return result


def install_collective_observation(observer, zero):
    """Record existing ZeRO2 pre-reduction views, forwarding original hooks unchanged."""
    if not zero.reduce_scatter or not zero.contiguous_gradients or float(zero.loss_scale) != 1.:
        raise ValueError("layout witness requires contiguous reduce-scatter BF16 ZeRO2")
    observer.names = {id(p):name for name,p in observer.named}
    original_process = zero.process_gradients
    def process(parameter, group):
        observer.hook_calls += 1
        grad = zero.get_gradient_for_reduction(parameter)
        if grad is None:
            raise RuntimeError('actual backward hook has no gradient')
        # Inspect before ZeRO partitions/reduces/clears local contributions.
        observer.actual_gradients[observer.names[id(parameter)]] = (
            bool(torch.isfinite(grad).all()), float(grad.detach().abs().max()))
        if getattr(observer, 'capture_layout', False):
            observer._observe_selected_leaf(parameter)
        return original_process(parameter, group)
    zero.process_gradients = process
    original_average, original_allreduce = zero.average_tensor, zero.allreduce_bucket
    observer._active_ipg = None
    def average(tensor, communication_data_type):
        bucket = zero.ipg_buckets[communication_data_type]
        layout = []
        for group,index,_ in bucket.params:
            parameter = zero.bit16_groups[group][index]
            layout.append(dict(name=observer.names[id(parameter)],shape=list(parameter.shape),
                               dtype=str(parameter.dtype),numel=parameter.numel()))
        observer._active_ipg = (tensor,layout)
        try:
            return original_average(tensor,communication_data_type)
        finally:
            observer._active_ipg = None
    def allreduce(bucket, communication_data_type, rank=None, log=None, divide=True, process_group=None):
        if observer._active_ipg is None:
            raise RuntimeError('reduction lacks its actual IPG layout')
        flat,layout = observer._active_ipg
        slices = describe_collective_slices(flat,layout,bucket)
        observer.write(dict(event='collective',rank=observer.rank,micro=observer.micro,sequence=observer.sequence,
                        layout=layout,slices=slices,tensor_shapes=[list(t.shape) for t in bucket],
                        communication_dtype=str(communication_data_type),reduce_rank=rank))
        observer.sequence += 1
        return original_allreduce(bucket,communication_data_type,rank=rank,log=log,divide=divide,process_group=process_group)
    zero.average_tensor, zero.allreduce_bucket = average, allreduce

class ReproductionObserver:
    """Three-update observer; a failed comparison is evidence, never authorization."""
    def __init__(self, trainer, engine, output_dir):
        import torch.distributed as dist
        self.engine = engine
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        if self.world != 4 or trainer.deepspeed_runtime_profile.zero_stage != 2:
            raise ValueError('reproduction witness requires four ranks/ZeRO2')
        if trainer.args.per_device_train_batch_size != 1 or trainer.args.gradient_accumulation_steps != 4:
            raise ValueError('reproduction witness requires microbatch1/GAS4')
        if trainer.state.global_step != 0:
            raise ValueError('reproduction witness requires a fresh optimizer')
        self.named = [(name,p) for name,p in engine.module.named_parameters() if p.requires_grad]
        self.parameters = [p for _,p in self.named]
        count = sum(p.numel() for p in self.parameters)
        if not count or count > 100_000_000:
            raise ValueError('reference requires 1..100M trainable elements')
        path = Path(output_dir).resolve()
        if not path.is_relative_to((Path.cwd()/'results').resolve()):
            raise ValueError('probe output must be under current results tree')
        path.mkdir(parents=True,exist_ok=True)
        self.stream = (path/f'rank-{self.rank}.jsonl').open('x')
        self.total = None
        self.ids = []
        self.pending = None
        self.hook_calls = 0
        self.clip_calls = []
        self.clip_groups = []
        self.clip_factors = []
        self.mean_adam = AdamReference()
        self.fixed_adam = AdamReference()
        self.effective_adam = AdamReference()
        self.capture_layout = getattr(trainer, 'stage2_sampler_policy', None) == 'historical_balanced_2rank'
        self.micro, self.sequence = -1, 0
        self.actual_gradients = {}
        self._backward_recorded = True
        self.selected_leaf = next(((i,name,p) for i,(name,p) in enumerate(self.named)
                                   if p.numel() <= 4096), None) if self.capture_layout else None
        self.selected_reference = None
        self.selected_leaf_hook_count = 0
        zero = engine.optimizer
        original = zero.process_gradients
        def observe_reduction(*args,**kwargs):
            self.hook_calls += 1
            return original(*args,**kwargs)
        if self.capture_layout:
            if trainer.model_accepts_loss_kwargs or trainer.compute_loss_func is not None:
                raise ValueError('historical witness requires fixed Trainer GAS scaling')
            install_collective_observation(self, zero)
        else:
            zero.process_gradients = observe_reduction
        original_clip = zero.unscale_and_clip_grads
        def observe_clip(gradients,norm):
            value = float(norm.detach().cpu())
            matches = [index for index,master in enumerate(zero.single_partition_of_fp32_groups)
                       if len(gradients)==1 and master.grad is gradients[0]]
            if len(matches)!=1:
                raise RuntimeError('clipping input does not identify exactly one FP32 partition group')
            self.clip_calls.append(value)
            self.clip_groups.append(matches[0])
            self.clip_factors.append(max(1.,(value/float(zero.loss_scale)+1e-6)/float(zero.clip_grad))*float(zero.loss_scale))
            return original_clip(gradients,norm)
        zero.unscale_and_clip_grads = observe_clip
        self._reference_output_managers = []
        self._output_capture_hook = engine.register_forward_hook(self._capture_output_hooks)
        self.write(dict(event='init',rank=self.rank,world_size=self.world,
            hostname=socket.gethostname(),gas=4,microbatch=1,zero_stage=2,fixed_denominator=16,
            sampler_policy=getattr(trainer, 'stage2_sampler_policy', None),
            model_accepts_loss_kwargs=trainer.model_accepts_loss_kwargs,
            compute_loss_func_present=trainer.compute_loss_func is not None,
            parameter_names=[name for name,_ in self.named],parameter_elements=count,
            gradient_dtype=str(getattr(zero,'dtype',None)),
            gradient_accumulation_dtype=str(getattr(zero,'gradient_accumulation_dtype',None)),
            communication_data_type=str(getattr(zero,'communication_data_type',None)),
            reduce_scatter=getattr(zero,'reduce_scatter',None),
            selected_leaf_advisory=None if self.selected_leaf is None else self.selected_leaf[1],
            reference='same production forward numerator, not an external-model oracle',
            full_run_authorized=False))

    def _capture_output_hooks(self, engine, args, outputs):
        # DeepSpeed attaches each manager to its first output, which need not
        # contribute to the trainer's composed loss. Keep managers, not tensors.
        tensors, managers = [], []
        def visit(value):
            if isinstance(value, torch.Tensor):
                if value.grad_fn is not None:
                    tensors.append(value)
                for manager in getattr(value, '_backward_hook_managers', ()):
                    if not any(manager is old for old in managers):
                        managers.append(manager)
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    visit(child)
            elif value is not None and not isinstance(value, (bool, int, float, str)):
                raise RuntimeError('unsupported DeepSpeed output hook container')
        try:
            visit(outputs)
            # Validate this forward independently: a manager from an earlier
            # forward cannot cover a later output whose holder was removed.
            for tensor in tensors:
                hooks = tensor._backward_hooks
                if not hooks or not any(
                    handle.hooks_dict_ref() is hooks and handle.id in hooks
                    for manager in managers for handle in manager.hook_handles
                ):
                    raise RuntimeError('incomplete DeepSpeed output hook capture')
            for manager in managers:
                if not any(manager is old for old in self._reference_output_managers):
                    self._reference_output_managers.append(manager)
        except Exception:
            self._reference_output_managers.clear()
            raise

    def _reference_gradients(self, loss):
        from deepspeed.runtime.utils import OutputBackwardHookManager
        managers = self._reference_output_managers
        saved = []
        try:
            if (not managers or self.engine._running_engine_backward
                    or self.engine.optimizer._backward_active_depth != 0):
                raise RuntimeError('unsupported DeepSpeed output hook backward state')
            for manager in managers:
                if (type(manager) is not OutputBackwardHookManager
                        or manager.preprocess_done is not False
                        or manager.preprocess_once_fn != self.engine._backward_prologue
                        or manager.preprocess_per_tensor_fn != self.engine._backward_prologue_per_tensor
                        or not manager.hook_handles):
                    raise RuntimeError('unsupported DeepSpeed output hook manager')
                saved.append((manager, manager.preprocess_done, manager.preprocess_per_tensor_fn))
            # autograd.grad skips leaf accumulation, but still visits output
            # hooks. Bypass their prologue and GAS scaling only for this pass.
            for manager, _, _ in saved:
                manager.preprocess_done = True
                manager.preprocess_per_tensor_fn = None
            return reference_gradients(loss, self.parameters)
        finally:
            for manager, done, per_tensor in saved:
                manager.preprocess_done = done
                manager.preprocess_per_tensor_fn = per_tensor
            managers.clear()

    def write(self,record):
        self.stream.write(json.dumps(record,allow_nan=False)+'\n')
        self.stream.flush()

    def _finish_backward(self):
        if not self.capture_layout or self._backward_recorded:
            return
        values = list(self.actual_gradients.values())
        self.write(dict(event='backward',rank=self.rank,micro=self.micro,parameter_count=len(values),
            finite=bool(values) and all(v[0] for v in values),
            gradient_max_abs=max((v[1] for v in values), default=None),
            selected_leaf_hook_count=self.selected_leaf_hook_count))
        self.actual_gradients.clear()
        self._backward_recorded = True
        self.selected_reference = None

    def _observe_selected_leaf(self, parameter):
        if self.selected_leaf is None or parameter is not self.selected_leaf[2]:
            return
        self.selected_leaf_hook_count += 1
        gradient = parameter.grad
        if self.selected_reference is None or gradient is None:
            raise RuntimeError('selected leaf advisory lacks current raw reference or untouched leaf')
        self.write(dict(event='selected_leaf_advisory',rank=self.rank,micro=self.micro,
            sample_id=self.ids[-1],parameter=self.selected_leaf[1],shape=list(parameter.shape),
            reference_dtype=str(self.selected_reference.dtype),actual_dtype=str(gradient.dtype),
            hook_ordinal=self.selected_leaf_hook_count,
            comparison=compare_vectors([self.selected_reference/4.],[gradient.detach().float().cpu()]),
            scope='one selected tensor; raw-reference post-hoc /GAS4, not a scaled-backward oracle'))

    def observe_loss(self,loss,inputs):
        if self.effective_adam.steps >= 3:
            raise RuntimeError('reproduction witness is bounded to three updates')
        ids = inputs.get('sample_ids')
        if not isinstance(ids,(tuple,list)) or len(ids)!=1 or not isinstance(ids[0],str):
            raise ValueError('probe requires one actual sample ID')
        metadata = {}
        if self.capture_layout:
            self._finish_backward()
            self.micro += 1
            self._backward_recorded = False
            self.selected_leaf_hook_count = 0
            start_id = getattr(self.engine.module.config, 'latent_start_token_id', None)
            if type(start_id) is not int:
                raise ValueError('historical witness requires runtime latent START token identity')
            blocks = int((inputs['input_ids'] == start_id).sum().item())
            if not 1 <= blocks <= 6:
                raise ValueError('historical witness requires 1..6 actual latent blocks')
            metadata = dict(micro=self.micro,update=self.effective_adam.steps+1,weight=1.,latent_blocks=blocks)
        previous_calls = self.hook_calls
        gradients = self._reference_gradients(loss)
        if self.hook_calls != previous_calls:
            raise RuntimeError('reference backward invoked ZeRO reduction hooks')
        if not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError('nonfinite reference numerator')
        if self.capture_layout and self.selected_leaf is not None:
            # Clone only this small slice: the full reference list becomes the
            # mutable accumulated numerator below. No extra backward/collective.
            self.selected_reference = gradients[self.selected_leaf[0]].clone()
        if self.total is None:
            self.total = gradients
        else:
            for total,gradient in zip(self.total,gradients):
                total.add_(gradient)
        self.ids.append(ids[0])
        self.write(dict(event='example',rank=self.rank,sample_id=ids[0],
            tokens=int(inputs['input_ids'].shape[-1]),loss=float(loss.detach()),
            reference_zero_hook_calls=self.hook_calls-previous_calls,rng_preserved=True,**metadata))

    def before_step(self):
        import torch.distributed as dist
        from deepspeed.utils import safe_get_full_grad,safe_get_full_fp32_param
        ids = [None]*self.world
        dist.all_gather_object(ids,self.ids)
        count = sum(len(rank_ids) for rank_ids in ids)
        if count == 0 or self.total is None:
            raise RuntimeError('empty optimizer group')
        means,fixed,effective,weights = [],[],[],[]
        for parameter,total in zip(self.parameters,self.total):
            summed = total.to(parameter.device).clone()
            dist.all_reduce(summed)
            summed = summed.cpu()
            means.append(summed/count)
            fixed.append(summed/16)
            gradient = safe_get_full_grad(parameter)
            weight = safe_get_full_fp32_param(parameter)
            if gradient is None or weight is None:
                raise RuntimeError('full ZeRO gradient/master parameter is unavailable')
            effective.append(gradient.detach().float().cpu())
            weights.append(weight.detach().float().cpu())
        zero = self.engine.optimizer
        if float(zero.loss_scale) != 1. or float(zero.clip_grad) != 1.:
            raise RuntimeError('probe requires BF16 loss_scale1 and clipping1')
        if not isinstance(zero.optimizer,torch.optim.AdamW):
            raise RuntimeError('probe requires actual torch AdamW')
        by_parameter = {}
        for parameters,group in zip(zero.bit16_groups,zero.optimizer.param_groups,strict=True):
            if group.get('amsgrad') or group.get('maximize'):
                raise RuntimeError('unsupported AdamW variant')
            for parameter in parameters:
                by_parameter[id(parameter)] = {key:group[key] for key in ('lr','betas','eps','weight_decay')}
        options = [by_parameter[id(p)] for p in self.parameters]
        self.clip_calls = []
        self.clip_groups = []
        self.clip_factors = []
        self.pending = dict(weights=weights,
            mean_delta=self.mean_adam.delta(weights,means,options,clip=1.),
            fixed_delta=self.fixed_adam.delta(weights,fixed,options,clip=1.),
            effective_delta=self.effective_adam.delta(weights,effective,options,clip=1.),
            report=dict(event='update',rank=self.rank,update=self.effective_adam.steps,
                group_ids_by_rank=ids,actual_count=count,fixed_denominator=16,options=options,
                microsteps=len(self.ids),
                actual_group_gradient=compare_vectors(means,effective),
                historical_fixed_gradient=compare_vectors(fixed,effective),
                clip=1.,full_run_authorized=False))

    def after_step(self):
        from deepspeed.utils import safe_get_full_fp32_param
        after = [safe_get_full_fp32_param(p).detach().float().cpu() for p in self.parameters]
        actual = [new-old for new,old in zip(after,self.pending['weights'])]
        report = self.pending['report']
        report.update(actual_group_adam_delta=compare_vectors(self.pending['mean_delta'],actual),
            completed=True,
            historical_fixed_adam_delta=compare_vectors(self.pending['fixed_delta'],actual),
            effective_gradient_adam_delta=compare_vectors(self.pending['effective_delta'],actual),
            observed_delta_l2=sum(float(delta.double().square().sum()) for delta in actual)**.5,
            clip_calls_global_norm=list(self.clip_calls),clip_group_indices=list(self.clip_groups),
            clip_divisors=list(self.clip_factors),
            clip_shared_global_factor=bool(self.clip_factors) and len(set(self.clip_factors))==1,
            clip_once_per_partition_group=sorted(self.clip_groups)==list(range(len(self.engine.optimizer.bit16_groups))))
        self.write(report)
        self._finish_backward()
        self.total,self.ids,self.pending = None,[],None


def check_historical_records(records, expected_order, expected_counts):
    """Check only the approved mixed20/five-microstep/two-update protocol.

    Keep strict independent numerator diagnostics as failures. Effective-gradient
    Adam agreement or a zero-LR step cannot override a failed numerator check.
    """
    failures = []
    def require(condition, message):
        if not condition:
            failures.append(message)
    require(len(expected_order) == 20 and len(set(expected_order)) == 20,
            'expected order must contain 20 distinct IDs')
    require(set(expected_counts) == set(expected_order), 'expected count identity incomplete')
    require([expected_counts.get(sid) for sid in expected_order] ==
            [3,3,3,3,1,1,1,1,3,3,6,6,6,6,1,1,3,3,1,1], 'wrong mixed20 count pattern')
    require({r.get('rank') for r in records} == set(range(4)), 'four ranks not present')
    initial = [r for r in records if r.get('event') == 'init']
    require(len(initial) == 4 and {r.get('rank') for r in initial} == set(range(4)), 'one init per rank required')
    hosts = [r.get('hostname') for r in initial]
    require(all(type(h) is str and h for h in hosts) and len(set(hosts)) == 2
            and all(hosts.count(h) == 2 for h in set(hosts)), 'two hosts with two ranks each required')
    for r in initial:
        require((r.get('world_size'),r.get('gas'),r.get('microbatch'),r.get('fixed_denominator'),r.get('zero_stage'))
                == (4,4,1,16,2), 'wrong initialized geometry')
        require(r.get('sampler_policy') == 'historical_balanced_2rank'
                and r.get('model_accepts_loss_kwargs') is False
                and r.get('compute_loss_func_present') is False, 'wrong policy or loss scaling')
    grid = [(m,r) for m in range(5) for r in range(4)]
    examples = sorted((r for r in records if r.get('event') == 'example'),
                      key=lambda r:(r.get('micro',-1),r.get('rank',-1)))
    require([(r.get('micro'),r.get('rank')) for r in examples] == grid, 'five examples per rank required')
    require([r.get('sample_id') for r in examples] == expected_order, 'ordered sample membership differs')
    for r in examples:
        require(r.get('weight') == 1. and r.get('latent_blocks') == expected_counts.get(r.get('sample_id')),
                'ordinary weight or runtime latent block count differs')
        require(r.get('update') == (1 if r.get('micro',5)<4 else 2), 'wrong update assignment')
        require(r.get('reference_zero_hook_calls') == 0 and r.get('rng_preserved') is True,
                'reference mutated hooks or RNG')
    backwards = [r for r in records if r.get('event') == 'backward']
    require(sorted((r.get('micro',-1),r.get('rank',-1)) for r in backwards) == grid, 'actual backward completion missing')
    for r in backwards:
        maximum = r.get('gradient_max_abs')
        require(r.get('finite') is True and r.get('parameter_count',0)>0
                and isinstance(maximum,(int,float)) and math.isfinite(maximum) and maximum>0,
                'actual gradients missing, zero or nonfinite')
    fields = ('micro','sequence','layout','slices','tensor_shapes','communication_dtype','reduce_rank')
    layouts = []
    for rank in range(4):
        events = [r for r in records if r.get('event') == 'collective' and r.get('rank') == rank]
        require({r.get('micro') for r in events} == set(range(5)), 'collectives missing for a microstep')
        require([r.get('sequence') for r in events] == list(range(len(events))), 'collective sequence incomplete')
        require(all(r.get('layout') and r.get('slices') and r.get('tensor_shapes')
                    and r.get('communication_dtype') for r in events), 'collective layout incomplete')
        layouts.append([{k:r.get(k) for k in fields} for r in events])
    require(all(layout == layouts[0] for layout in layouts[1:]), 'collective ordered layout, slice, dtype or count differs')
    updates = [r for r in records if r.get('event') == 'update']
    require(sorted((r.get('update',-1),r.get('rank',-1)) for r in updates) ==
            [(u,r) for u in (1,2) for r in range(4)], 'two completed updates per rank required')
    for r in updates:
        require(r.get('completed') is True and r.get('fixed_denominator') == 16, 'completion or fixed16 denominator missing')
        require((r.get('microsteps'),r.get('actual_count')) == ((4,16) if r.get('update')==1 else (1,4)),
                'full or tail count differs')
        for key in ('historical_fixed_gradient','historical_fixed_adam_delta','effective_gradient_adam_delta'):
            c = r.get(key,{})
            numeric = all(isinstance(c.get(k),(int,float)) and math.isfinite(c[k]) and c[k]>=0
                          for k in ('expected_l2','l2_error','max_abs_error'))
            require(c.get('pass') is True and c.get('finite') is True and numeric
                    and c.get('rtol') == .02 and c.get('atol') == 1e-7,
                    f'strict {key} failed or missing')
        require(r.get('clip') == 1. and r.get('clip_shared_global_factor') is True
                and r.get('clip_once_per_partition_group') is True
                and bool(r.get('clip_calls_global_norm')) and bool(r.get('clip_group_indices'))
                and bool(r.get('clip_divisors')), 'shared once-per-partition clipping missing')
    return {'pass':not failures, 'failures':failures, 'ranks':4, 'real_samples':20,
            'updates':2,'fixed_denominator':16,'scope':'mixed20 runtime evidence only'}
