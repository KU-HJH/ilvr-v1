"""Unscaled auxiliary accumulation and post-Adam ZeRO-1 FP32 displacement."""
from __future__ import annotations
from importlib.metadata import version
import torch
import torch.distributed as dist
from torch import nn
from deepspeed.utils import safe_get_full_fp32_param,safe_set_full_fp32_param
from .v3_config import Stage1V3Config
from .v3_gradients import V3Gradients
from .v3_objective import project_gain


class V3GainController:
    def __init__(self,parameters: tuple[nn.Parameter,...],config: Stage1V3Config):
        if not parameters or len({id(p) for p in parameters})!=len(parameters) or any(not p.requires_grad for p in parameters):
            raise ValueError('Gain requires unique optimizer-owned trainable parameters')
        self.parameters,self.config=parameters,config
        self.state='IDLE';self.successful_steps=0;self.global_block_count=0
        self.direction=();self._engine=None
        self._optimizer_adapter=None;self.last_metrics={}
        self._clear()

    def _clear(self):
        # Primary lane: raw R for none, already-differenced Gain for weighted.
        self._r=tuple(torch.zeros_like(p,dtype=torch.float32) for p in self.parameters)
        self._rv=tuple(torch.zeros_like(p,dtype=torch.float32) for p in self.parameters)
        self._count=0;self._invalid=False;self.direction=()
        self.raw_gain=();self.raw_rv=()

    def accumulate(self,result: V3Gradients) -> None:
        if self.state not in ('IDLE','ACCUMULATING'):
            raise RuntimeError('Gain accumulation outside a valid update window')
        # DeepSpeed places parameters after trainer construction. Align only
        # before a new window; moving an open window must not erase its sums.
        if self.state=='IDLE' and any(buffer.device!=parameter.device for buffer,parameter in
                                     zip(self._r+self._rv,self.parameters+self.parameters)):
            self._clear()
        self.state='ACCUMULATING'
        weighted=self.config.gain_weighting=='token_entropy'
        primary=result.gain_sum_grads if weighted else result.r_sum_grads
        if (type(result.block_count) is not int or result.block_count<0
                or primary is None or len(primary)!=len(self.parameters)
                or (result.r_sum_grads if weighted else result.gain_sum_grads) is not None
                or len(result.rv_sum_grads)!=len(self.parameters)):
            self._invalid=True;return
        self._count+=result.block_count
        for destination,source in zip(self._r+self._rv,primary+result.rv_sum_grads):
            if source.shape!=destination.shape or source.device!=destination.device or source.dtype!=torch.float32:
                self._invalid=True;continue
            destination.add_(source.detach())
            self._invalid |= not bool(torch.isfinite(destination).all())

    def _sum(self,tensor,group):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor,op=dist.ReduceOp.SUM,group=group)
        return tensor

    def prepare(self,process_group: object|None=None) -> None:
        if self.state!='ACCUMULATING':raise RuntimeError('duplicate or unstarted Gain prepare')
        device=self.parameters[0].device
        try:
            invalid=self._sum(torch.tensor(int(self._invalid),device=device,dtype=torch.int32),process_group)
            if invalid.item():raise RuntimeError('invalid or nonfinite auxiliary gradient on a rank')
            count=self._sum(torch.tensor(self._count,device=device,dtype=torch.int64),process_group)
            r=tuple(self._sum(g.clone(),process_group) for g in self._r)
            rv=tuple(self._sum(g.clone(),process_group) for g in self._rv)
            self.global_block_count=int(count.item())
            error=None
            try:
                if self.global_block_count:
                    b=self.global_block_count
                    gain=(tuple(a/b for a in r) if self.config.gain_weighting=='token_entropy'
                          else tuple((a-h)/b for a,h in zip(r,rv)))
                    self.raw_gain=gain;self.raw_rv=tuple(h/b for h in rv)
                    if self.config.gain_update_mode=='post_adam':
                        self.direction=project_gain(gain,self.raw_rv,
                                                    min_h_norm_sq=self.config.min_h_norm_sq)
                else:
                    self.raw_gain=tuple(torch.zeros_like(g) for g in r)
                    self.raw_rv=tuple(torch.zeros_like(g) for g in rv)
                    self.direction=self.raw_gain
                if any(not torch.isfinite(g).all() for g in self.raw_gain+self.raw_rv):
                    raise ValueError('nonfinite global Gain/RV gradient')
            except (ValueError,RuntimeError) as failure:error=failure
            invalid=self._sum(torch.tensor(int(error is not None),device=device,dtype=torch.int32),process_group)
            if invalid.item():raise RuntimeError('invalid or nonfinite global Gain projection') from error
            self.state='PREPARED'
        except BaseException:
            self.state='FAILED';raise

    def validate_engine(self,engine: object) -> None:
        if self.config.gain_update_mode=='adamw_projected':
            # The new adapter performs collective profile/mapping validation after
            # raw-gradient reduction; do not run a rank-local legacy guard first.
            return
        if self._engine is engine:return
        optimizer=getattr(engine,'optimizer',None)
        if (version('deepspeed')!='0.18.7' or not callable(getattr(engine,'zero_optimization_stage',None))
                or engine.zero_optimization_stage()!=1 or getattr(optimizer,'partition_gradients',None) is not False
                or getattr(optimizer,'cpu_offload',None) is not False
                or not callable(getattr(optimizer,'update_lp_params',None))):
            raise RuntimeError('v3 requires validated DeepSpeed0.18.7 ZeRO-1 without offload and master synchronization')
        groups=getattr(optimizer,'bit16_groups',())
        if {id(p) for group in groups for p in group}!={id(p) for p in self.parameters}:
            raise RuntimeError('v3 optimizer/master parameter ownership mismatch')
        masters=getattr(optimizer,'single_partition_of_fp32_groups',())
        if not masters or any(p.dtype!=torch.float32 for p in masters):
            raise RuntimeError('v3 FP32 master groups unavailable')
        for p in self.parameters:
            if (not hasattr(p,'_hp_mapping') or not callable(getattr(p,'get_full_hp_param',None))
                    or not callable(getattr(p,'set_full_hp_param',None))):
                raise RuntimeError('v3 ZeRO-1 full master mapping unavailable')
        self._engine=engine

    def prepare_optimizer_step(self,engine: object) -> None:
        if self.config.gain_update_mode=='post_adam':return
        if self.state!='PREPARED':raise RuntimeError('optimizer Gain requires prepared raw gradients')
        from .v3_zero1_update import Zero1AdamWGainAdapter
        self._optimizer_adapter=Zero1AdamWGainAdapter(engine,self.parameters,self.config)
        self._optimizer_adapter.prepare(self.raw_gain,self.raw_rv,block_count=self.global_block_count)

    def abort_optimizer_step(self) -> None:
        if self._optimizer_adapter is not None:
            self._optimizer_adapter.abort()
            self._optimizer_adapter=None

    @torch.no_grad()
    def finish(self,engine: object,*,successful: bool) -> None:
        if self.state!='PREPARED' or type(successful) is not bool:
            raise RuntimeError('duplicate or unprepared Gain finish')
        try:
            if self.config.gain_update_mode=='adamw_projected':
                if self._optimizer_adapter is None:raise RuntimeError('optimizer Gain adapter was not prepared')
                self.last_metrics=self._optimizer_adapter.finish(successful=successful)
                self._optimizer_adapter=None
                if successful:self.successful_steps+=1
                self._clear();self.state='IDLE'
                return
            self.validate_engine(engine)
            if successful:
                if self.global_block_count and self.config.gain_alpha:
                    updated=[]
                    for param,direction in zip(self.parameters,self.direction):
                        master=safe_get_full_fp32_param(param)
                        if master is None or master.dtype!=torch.float32 or master.shape!=param.shape:
                            raise RuntimeError('v3 full FP32 master shape/dtype unavailable')
                        value=master+self.config.gain_alpha*direction
                        if not torch.isfinite(value).all():raise RuntimeError('nonfinite post-step Gain master')
                        updated.append(value)
                    for param,value in zip(self.parameters,updated):safe_set_full_fp32_param(param,value)
                    engine.optimizer.update_lp_params()
                self.successful_steps+=1
            self._clear();self.state='IDLE'
        except BaseException:
            self.abort_optimizer_step()
            self.state='FAILED';raise

    def assert_checkpointable(self) -> None:
        if self.state!='IDLE':raise RuntimeError('v3 checkpoint requires a complete, nonfailed update boundary')
