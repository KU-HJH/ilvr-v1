"""Opt-in, bounded V-off reproduction arithmetic witness; never authorizes training."""
import json
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
        zero = engine.optimizer
        original = zero.process_gradients
        def observe_reduction(*args,**kwargs):
            self.hook_calls += 1
            return original(*args,**kwargs)
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
        self.write(dict(event='init',rank=self.rank,world_size=self.world,
            model_accepts_loss_kwargs=trainer.model_accepts_loss_kwargs,
            compute_loss_func_present=trainer.compute_loss_func is not None,
            parameter_names=[name for name,_ in self.named],parameter_elements=count,
            reference='same production forward numerator, not an external-model oracle',
            full_run_authorized=False))

    def write(self,record):
        self.stream.write(json.dumps(record,allow_nan=False)+'\n')
        self.stream.flush()

    def observe_loss(self,loss,inputs):
        if self.effective_adam.steps >= 3:
            raise RuntimeError('reproduction witness is bounded to three updates')
        ids = inputs.get('sample_ids')
        if not isinstance(ids,(tuple,list)) or len(ids)!=1 or not isinstance(ids[0],str):
            raise ValueError('probe requires one actual sample ID')
        previous_calls = self.hook_calls
        gradients = reference_gradients(loss,self.parameters)
        if self.hook_calls != previous_calls:
            raise RuntimeError('reference backward invoked ZeRO reduction hooks')
        if not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError('nonfinite reference numerator')
        if self.total is None:
            self.total = gradients
        else:
            for total,gradient in zip(self.total,gradients):
                total.add_(gradient)
        self.ids.append(ids[0])
        self.write(dict(event='example',rank=self.rank,sample_id=ids[0],
            tokens=int(inputs['input_ids'].shape[-1]),loss=float(loss.detach()),
            reference_zero_hook_calls=self.hook_calls-previous_calls,rng_preserved=True))

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
                actual_group_gradient=compare_vectors(means,effective),
                historical_fixed_gradient=compare_vectors(fixed,effective),
                clip=1.,full_run_authorized=False))

    def after_step(self):
        from deepspeed.utils import safe_get_full_fp32_param
        after = [safe_get_full_fp32_param(p).detach().float().cpu() for p in self.parameters]
        actual = [new-old for new,old in zip(after,self.pending['weights'])]
        report = self.pending['report']
        report.update(actual_group_adam_delta=compare_vectors(self.pending['mean_delta'],actual),
            historical_fixed_adam_delta=compare_vectors(self.pending['fixed_delta'],actual),
            effective_gradient_adam_delta=compare_vectors(self.pending['effective_delta'],actual),
            observed_delta_l2=sum(float(delta.double().square().sum()) for delta in actual)**.5,
            clip_calls_global_norm=list(self.clip_calls),clip_group_indices=list(self.clip_groups),
            clip_divisors=list(self.clip_factors),
            clip_shared_global_factor=bool(self.clip_factors) and len(set(self.clip_factors))==1,
            clip_once_per_partition_group=sorted(self.clip_groups)==list(range(len(self.engine.optimizer.bit16_groups))))
        self.write(report)
        self.total,self.ids,self.pending = None,[],None
