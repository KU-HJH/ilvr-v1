"""Observer contract against installed DS engine hooks; CPU stand-in reducer only.

Removing output-manager suspension must reproduce the actual empty-bucket error.
Real forward/backward, output hooks and ZeRO state/leaf hooks are exercised; no
GPU/distributed partition correctness is claimed by the small reducer below.
"""
import copy
from types import SimpleNamespace
import pytest
import torch
from torch.utils.checkpoint import checkpoint
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.base_optimizer import ZeROOptimizer
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from stage2.reproduction_probe import ReproductionObserver

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.p=torch.nn.Parameter(torch.arange(1.,5.))
    def forward(self, factor):
        x=checkpoint(lambda p: torch.nn.functional.dropout(p,.3,True).square(),self.p,use_reentrant=False)
        return {'unused_first': x.mean()*factor, 'logits': x*factor, 'nested': (x.sum()*factor,)}

class Zero(ZeROOptimizer):
    create_gradient_handling_hooks=DeepSpeedZeroOptimizer.create_gradient_handling_hooks
    def __init__(self,p,exact_failure=False):
        super().__init__(); self.bit16_groups=[[p]]; self._grad_acc_hooks=[]
        self.custom_loss_scaler=False; self.torch_autocast_gradscaler=None
        self.events=[]; self.micro_step_id=-1; self.ready_for_gradients=False
        self.is_gradient_accumulation_boundary=False
        self.bucket=SimpleNamespace(buffer=[],index=0,elements=0,params=[])
        self.ipg_buckets={torch.float32:self.bucket}; self.contiguous_gradients=True
        self.extra_large_param_to_reduce={}; self.exact_failure=exact_failure
        self.create_gradient_handling_hooks()
    def average_tensor(self,*args):raise AssertionError("unexpected averaging")
    def process_gradients(self,p,index):
        self.events.append('leaf'); self.micro_step_id+=1; self.ready_for_gradients=True
        self.bucket.buffer=[p.grad.clone()]; self.bucket.elements=p.numel()
    def overlapping_partition_gradients_reduce_epilogue(self):
        self.events.append('reduce')
        if not self.bucket.buffer:
            DeepSpeedZeroOptimizer.reduce_ipg_grads(self)
        else:
            assert self.bucket.buffer, 'reducer invoked without leaf hook'
            self.bucket.buffer=[]; self.bucket.elements=0; self.ready_for_gradients=False
    def snapshot(self):
        return (copy.deepcopy({k:v for k,v in vars(self._backward_hook_state).items() if k!='_grad_acc_post_hooks'}),
                copy.deepcopy((self.events,self.micro_step_id,self.ready_for_gradients,self.is_gradient_accumulation_boundary,
                               self.bucket.buffer,self.bucket.index,self.bucket.elements)))

class Engine(torch.nn.Module):
    forward=DeepSpeedEngine.forward
    backward=DeepSpeedEngine.backward
    _backward_prologue=DeepSpeedEngine._backward_prologue
    _backward_prologue_per_tensor=DeepSpeedEngine._backward_prologue_per_tensor
    _backward_post_hook=DeepSpeedEngine._backward_post_hook
    _backward_epilogue=DeepSpeedEngine._backward_epilogue
    allreduce_gradients=DeepSpeedEngine.allreduce_gradients
    def __init__(self,exact_failure=False):
        super().__init__(); self.module=Model(); self.optimizer=Zero(self.module.p,exact_failure)
        self._support_torch_style_backward=True; self._running_engine_backward=False; self._scale_wrt_gas=True
        self._manual_backward_expected=False; self.zenflow=False; self.enable_backward_allreduce=True
        self.inside_no_sync_ctxt=False; self._is_compiled_autograd_enabled=False; self._compile_kwargs={}
        self.engine_timers=SimpleNamespace(backward_timers=[],backward_inner_timers=[],backward_reduce_timers=[])
        self.optimizer.register_grad_acc_post_hook(self._backward_post_hook)
    def _start_timers(self,*a):pass
    def _stop_timers(self,*a):pass
    def memory_breakdown(self):return False
    def eigenvalue_enabled(self):return False
    def amp_enabled(self):return False
    def is_deepcompile_active(self):return False
    def is_deepcompile_enabled(self):return False
    def autotuning_profile_model_info(self):return False
    def torch_autocast_enabled(self):return False
    def zero_optimization(self):return True
    def zero_optimization_partition_gradients(self):return True
    def is_gradient_accumulation_boundary(self):return True
    def gradient_accumulation_steps(self):return 4


@pytest.fixture
def observer_factory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 4)
    streams=[]
    def make(engine, name):
        engine.optimizer.unscale_and_clip_grads=lambda gradients,norm: None
        trainer=SimpleNamespace(deepspeed_runtime_profile=SimpleNamespace(zero_stage=2),
            args=SimpleNamespace(per_device_train_batch_size=1,gradient_accumulation_steps=4),
            state=SimpleNamespace(global_step=0),model_accepts_loss_kwargs=False,compute_loss_func=None)
        observer=ReproductionObserver(trainer,engine,tmp_path/'results'/name)
        streams.append(observer.stream)
        return observer
    yield make
    for stream in streams:stream.close()


def inputs(index=0):
    return dict(sample_ids=[str(index)],input_ids=torch.tensor([[1,2]]))


def test_reference_with_engine_hooks_preserves_four_normal_backwards(observer_factory):
    def run(enabled):
        torch.manual_seed(123)
        engine=Engine(); parameter=engine.module.p
        parameter.grad=torch.full_like(parameter,7.)
        observer=observer_factory(engine,'on') if enabled else None
        states=[]
        for index in range(4):
            out=engine(float(index+1));loss=out['logits'].sum()+out['nested'][0]
            assert not hasattr(loss,'_backward_hook_managers')
            manager=out['unused_first']._backward_hook_managers[0]
            before_manager=(manager.preprocess_done,manager.preprocess_per_tensor_fn)
            if observer:
                before=engine.optimizer.snapshot();rng=torch.get_rng_state().clone()
                old_grad=parameter.grad;old_version=old_grad._version
                observer.observe_loss(loss,inputs(index))
                assert engine.optimizer.snapshot()==before
                assert parameter.grad is old_grad and parameter.grad._version==old_version
                assert torch.equal(rng,torch.get_rng_state())
                assert (manager.preprocess_done,manager.preprocess_per_tensor_fn)==before_manager
            engine.backward(loss,scale_wrt_gas=False)
            state=engine.optimizer.snapshot()
            assert isinstance(state[0]['post_backward_callback_graph_task_id'],int)
            state[0]['post_backward_callback_graph_task_id']='graph-local identity'
            states.append(state)
        return parameter.grad,torch.get_rng_state(),states,observer,engine
    on=run(True);off=run(False)
    assert torch.equal(on[0],off[0]) and torch.equal(on[1],off[1]) and on[2]==off[2]
    # Independent observer-off gradient, so a quarter-scaled reference cannot pass.
    torch.testing.assert_close(on[3].total[0],off[0]-7,rtol=1e-6,atol=1e-5)
    assert on[4].optimizer.events==['leaf','reduce']*4


def test_reference_failure_restores_hooks_before_retained_graph_backward(observer_factory):
    engine=Engine();observer=observer_factory(engine,'exception')
    out=engine(1.);loss=out['logits'].sum()+out['nested'][0]
    manager=out['unused_first']._backward_hook_managers[0]
    saved=(manager.preprocess_done,manager.preprocess_per_tensor_fn)
    def fail(grad):raise RuntimeError('injected autograd failure')
    handle=out['logits'].register_hook(fail)
    before=engine.optimizer.snapshot()
    with pytest.raises(RuntimeError,match='injected autograd failure'):
        observer.observe_loss(loss,inputs())
    assert saved==(manager.preprocess_done,manager.preprocess_per_tensor_fn)
    assert engine.optimizer.snapshot()==before
    handle.remove()
    engine.backward(loss,scale_wrt_gas=False)
    assert engine.optimizer.events==['leaf','reduce']


@pytest.mark.parametrize('broken',['no_manager','consumed','foreign_callback'])
def test_reference_rejects_unsupported_output_hook_state(observer_factory,broken):
    engine=Engine();observer=observer_factory(engine,broken)
    if broken=='no_manager':
        out=engine.module(1.)
    else:
        out=engine(1.);manager=out['unused_first']._backward_hook_managers[0]
        if broken=='consumed':manager.preprocess_done=True
        else:manager.preprocess_per_tensor_fn=lambda grad:grad
    with pytest.raises(RuntimeError,match='output hook'):
        observer.observe_loss(out['logits'].sum()+out['nested'][0],inputs())
    assert engine.optimizer.events==[]
