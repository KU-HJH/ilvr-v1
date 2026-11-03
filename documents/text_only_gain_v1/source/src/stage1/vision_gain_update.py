"""Update-boundary score reductions and direct auxiliary displacement."""
import torch
from torch import nn
from .vision_gain_execution import Stage1GainScores


class GainGradientAccumulator:
    def __init__(self, parameters: tuple[nn.Parameter,...]):
        if not parameters or any(not p.requires_grad for p in parameters):
            raise ValueError('gain accumulator requires trainable parameters')
        self.parameters=parameters
        self.reset()

    def reset(self):
        self.clean_grads=None
        self.masked_grads=None
        self.block_count=0
        self.microbatch_count=0
        self.failed=False
        self.error=None
        self.clean_total=0.
        self.masked_total=0.
        self._global_metrics=None

    def fail(self,error):
        self.failed=True
        self.error=str(error)

    def _allocate(self):
        if self.clean_grads is None:
            self.clean_grads=[torch.zeros_like(p,dtype=torch.float32) for p in self.parameters]
            self.masked_grads=[torch.zeros_like(p,dtype=torch.float32) for p in self.parameters]

    def accumulate(self,scores: Stage1GainScores, *, optimizer_loss=None):
        if not scores.block_count:
            return
        if len(scores.score_parameters)!=len(self.parameters):
            raise ValueError('gain alias parameter order/length mismatch')
        self._allocate()
        self.block_count+=scores.block_count
        self.clean_total+=float(scores.clean_sum.detach())
        self.masked_total+=float(scores.masked_sum.detach())
        proxy = None
        if optimizer_loss is None:
            clean=torch.autograd.grad(scores.clean_sum,scores.score_parameters,allow_unused=True,retain_graph=True)
        else:
            from .text_kl import joint_positive_kl_gradients, optimizer_gradient_loss
            clean, kl_gradients = joint_positive_kl_gradients(
                scores.clean_sum, optimizer_loss, scores.score_parameters)
            proxy = optimizer_gradient_loss(optimizer_loss, kl_gradients, self.parameters)
        masked=torch.autograd.grad(scores.masked_sum,scores.score_parameters,allow_unused=True)
        for dst,src in ((self.clean_grads,clean),(self.masked_grads,masked)):
            for buffer,gradient in zip(dst,src):
                if gradient is not None:
                    buffer.add_(gradient.detach().float())
        return proxy

    def accumulate_branch(self, score, parameters, *, masked, block_count=0):
        """Consume one independent graph immediately into unscaled FP32 sums."""
        if len(parameters) != len(self.parameters):
            raise ValueError('gain alias parameter order/length mismatch')
        self._allocate()
        gradients = torch.autograd.grad(score, parameters, allow_unused=True)
        buffers = self.masked_grads if masked else self.clean_grads
        for buffer, gradient in zip(buffers, gradients):
            if gradient is not None:
                buffer.add_(gradient.detach().float())
        if masked:
            self.masked_total += float(score.detach())
        else:
            self.clean_total += float(score.detach())
            self.block_count += block_count

    def finalize(self,process_group=None):
        device=self.parameters[0].device
        distributed=torch.distributed.is_available() and torch.distributed.is_initialized()
        def reduce(t):
            if distributed:
                torch.distributed.all_reduce(t,op=torch.distributed.ReduceOp.SUM,group=process_group)
        bad=self.failed
        if self.clean_grads is not None:
            bad=bad or any(not bool(torch.isfinite(v).all()) for v in (*self.clean_grads,*self.masked_grads))
        totals=torch.tensor([self.clean_total,self.masked_total],dtype=torch.float64,device=device)
        bad=bad or not bool(torch.isfinite(totals).all())
        counts=torch.tensor([self.block_count,int(bad)],dtype=torch.long,device=device)
        reduce(counts)
        if int(counts[1]):
            raise FloatingPointError('vision gain failed on one or more ranks: '+(self.error or 'nonfinite score/gradient'))
        b=int(counts[0]);reduce(totals)
        self._global_metrics=(float(totals[0]),float(totals[1]),b)
        if not b:
            return (),(),0
        self._allocate()
        gs,hs=[],[]
        for f,m in zip(self.clean_grads,self.masked_grads):
            reduce(f);reduce(m)
            hs.append(m/b);gs.append((f-m)/b)
        return tuple(gs),tuple(hs),b

    def metric_totals(self):
        return self._global_metrics or (self.clean_total,self.masked_total,self.block_count)


def validate_gain_runtime(config, attention_implementation, profile, *, fp16=False):
    if not config.alpha:
        return
    if attention_implementation not in {'eager','sdpa'}:
        raise ValueError('enabled Stage 1 vision gain requires eager or sdpa attention')
    if fp16:
        raise ValueError('enabled Stage 1 vision gain does not support fp16')
    if profile.enabled:
        if profile.zero_stage!=1:
            raise ValueError('enabled Stage 1 vision gain supports only ZeRO-1')
        if profile.config_path is not None:
            import json
            payload=json.loads(profile.config_path.read_text())
            zero=payload.get('zero_optimization',{})
            if zero.get('offload_optimizer') or zero.get('offload_param'):
                raise ValueError('enabled Stage 1 vision gain does not support offload')
            if not payload.get('bf16',{}).get('enabled') or payload.get('fp16',{}).get('enabled'):
                raise ValueError('enabled ZeRO-1 vision gain requires bf16')


@torch.no_grad()
def apply_gain_displacement(model,parameters,direction,*,alpha,optimizer,runtime_profile):
    from .vision_gain import Stage1VisionGainConfig
    Stage1VisionGainConfig(alpha)
    if len(parameters)!=len(direction):
        raise ValueError('gain displacement parameter count mismatch')
    if not alpha:
        return
    if any(p.shape!=d.shape or not bool(torch.isfinite(d).all()) for p,d in zip(parameters,direction)):
        raise ValueError('gain displacement shape or finite-value mismatch')
    if runtime_profile.enabled:
        if runtime_profile.zero_stage!=1:
            raise ValueError('gain master-weight updates require ZeRO-1')
        from deepspeed.utils import safe_get_full_fp32_param, safe_set_full_fp32_param
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        if not isinstance(optimizer,DeepSpeedZeroOptimizer) or optimizer.partition_gradients or optimizer.cpu_offload:
            raise RuntimeError('unsupported vision gain ZeRO optimizer')
        for p,d in zip(parameters,direction):
            master=safe_get_full_fp32_param(p)
            if master is None:
                raise RuntimeError('vision gain FP32 master mapping missing')
            master.add_(d.to(master),alpha=alpha)
            safe_set_full_fp32_param(p,master)
        optimizer.update_lp_params()
    else:
        for p,d in zip(parameters,direction):
            p.add_(d.to(p),alpha=alpha)


class Stage1GainController:
    def __init__(self,parameters,config):
        self.config=config
        self.accumulator=GainGradientAccumulator(parameters)
        self.pending=None
        self.conditioned=False
        self.last_metrics={}
        self.completed_metrics={}
        self.completed_windows=0
        self.counts={'updates':0,'projected':0,'zero':0,'tiny':0}

    def prepare(self):
        from .vision_gain import project_gain_direction
        if self.pending is not None:
            raise RuntimeError('gain update prepared twice')
        g,h,b=self.accumulator.finalize()
        window_metrics = dict(accumulation_window_index=self.completed_windows + 1,
                              local_microbatch_count=self.accumulator.microbatch_count)
        if not b:
            self.last_metrics={'block_count':0, **window_metrics}
            return
        self.pending=project_gain_direction(g,h,min_h_norm_sq=self.config.min_h_norm_sq)
        f,m,_=self.accumulator.metric_totals()
        positive, negative, gain = ('S_R', 'S_RV', 'G_cond') if self.conditioned else ('S_F', 'S_M', 'G')
        self.last_metrics={positive:f/b,negative:m/b,gain:(f-m)/b,'loss_gain':(m-f)/b,'block_count':b,
                           **window_metrics}
        if self.conditioned:
            # g is the difference; the positive gradient is g+h. Reuse h's
            # squared norm and h.g from projection without retaining new vectors.
            g_norm_sq = sum(float(v.double().square().sum()) for v in g)
            positive_norm_sq = g_norm_sq + 2 * self.pending.h_dot_g + self.pending.h_norm_sq
            self.last_metrics.update(
                gain_positive_grad_norm=max(0., positive_norm_sq) ** .5,
                gain_negative_grad_norm=self.pending.h_norm_sq ** .5)

    def finish(self,model,optimizer,profile,*,successful):
        try:
            if successful and self.pending is not None:
                result=self.pending
                if not result.skipped and not result.zero_direction:
                    apply_gain_displacement(model,self.accumulator.parameters,result.direction,
                        alpha=self.config.alpha,optimizer=optimizer,runtime_profile=profile)
                self.counts['updates']+=1
                self.counts['projected']+=int(result.projected)
                self.counts['zero']+=int(result.zero_direction)
                self.counts['tiny']+=int(result.skipped)
                n=self.counts['updates']
                self.last_metrics.update(projection_fraction=self.counts['projected']/n,
                    zero_u_fraction=self.counts['zero']/n,tiny_h_skip_fraction=self.counts['tiny']/n,
                    auxiliary_norm=self.config.alpha*sum(float(v.double().square().sum()) for v in result.direction)**.5)
            self.completed_windows += 1
            self.completed_metrics = dict(self.last_metrics, optimizer_step_successful=bool(successful))
        finally:
            self.accumulator.reset()
            self.pending=None
