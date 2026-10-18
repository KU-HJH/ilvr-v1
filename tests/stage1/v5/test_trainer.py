"""Real Qwen/DoRA + real parent Trainer one-backward integration witnesses."""
from copy import deepcopy
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from transformers import TrainingArguments
from peft_common import resolve_qwen2_5_vl_model
from stage1.ema import Stage1EmaTeacher
from stage1.types import Stage1BatchMetadata, Stage1SampleMetadata, Stage1LatentSpan, Stage1TeacherTargets
from stage1.v5_config import Stage1V5Config
from stage1.v5_metadata import V5BatchMetadata, V5Block
from tests.stage1.v5.test_replay import fixture, IDS, BODY, POSITIONS, digest_state


def api():
    assert importlib.util.find_spec('stage1.v5_trainer'), 'one-backward trainer missing'
    from stage1.v5_trainer import CustomTrainerStage1V5
    return CustomTrainerStage1V5


@pytest.fixture
def native_deepspeed_lifecycle():
    """Native engine lifecycle; distributed optimizer leaves use local SGD."""
    from accelerate.utils import DeepSpeedEngineWrapper, DistributedType
    from deepspeed.runtime.config import DeepSpeedConfig
    from deepspeed.runtime.engine import DeepSpeedEngine
    from deepspeed.runtime.base_optimizer import BackwardHookStateManager
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    from deepspeed_runtime import DeepSpeedRuntimeProfile
    from torch import nn
    engine=DeepSpeedEngine.__new__(DeepSpeedEngine)
    nn.Module.__init__(engine)
    engine._set_client_model(nn.Linear(1,1,bias=False))
    engine.module.weight.data.fill_(1.)
    engine._config=DeepSpeedConfig({'train_batch_size':4,'train_micro_batch_size_per_gpu':1,
        'gradient_accumulation_steps':4,'bf16':{'enabled':True},'zero_optimization':{'stage':1}})
    optimizer=DeepSpeedZeroOptimizer.__new__(DeepSpeedZeroOptimizer)
    optimizer.partition_gradients=False
    optimizer.custom_loss_scaler=False; optimizer.torch_autocast_gradscaler=None
    optimizer.use_grad_accum_attribute=False
    optimizer._backward_hook_state=BackwardHookStateManager()
    optimizer.optimizer=torch.optim.SGD(engine.module.parameters(),lr=.1)
    optimizer.overflow=False
    # No distributed initialization/partitions: retain native engine step and
    # applied-step publication, replace only optimizer mutation/reduction leaves.
    optimizer.step=lambda: None if optimizer.overflow else optimizer.optimizer.step()
    optimizer.zero_grad=optimizer.optimizer.zero_grad
    engine.optimizer=optimizer
    engine._is_gradient_accumulation_boundary=None
    engine._is_compiled_autograd_enabled=False; engine._compile_kwargs={}
    engine.torch_autocast_z0_gradscaler=None
    engine.inside_no_sync_ctxt=False; engine.enable_backward_allreduce=False
    engine.zenflow=False; engine.quantizer=None; engine.progressive_layer_drop=False
    engine._support_torch_style_backward=False
    engine.micro_steps=0; engine.global_steps=0; engine.global_samples=0
    engine.gas_boundary_ctr=0; engine.skipped_steps=0; engine.global_rank=0
    engine.engine_timers=SimpleNamespace(step_timers=(),backward_inner_timers=(),
        backward_reduce_timers=(),backward_timers=())
    engine.tput_timer=SimpleNamespace(stop=lambda **kw:None)
    engine.monitor=SimpleNamespace(enabled=False)
    engine.checkpoint_engine=SimpleNamespace(is_decoupled=lambda:False)
    engine.compression_scheduler=SimpleNamespace(step=lambda:None)
    engine.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer,lambda _:1.)
    trainer=api().__new__(api())
    trainer.stage1_v5_debug=False
    trainer._deepspeed_step_adapter=None; trainer._successful_optimizer_steps=0
    trainer.model=engine.module; trainer.model_wrapped=engine
    trainer.ema_teacher=Stage1EmaTeacher(engine.module,decay=1.,method='full')
    trainer.deepspeed_runtime_profile=DeepSpeedRuntimeProfile(True,1,None,None,4)
    wrapper=DeepSpeedEngineWrapper(engine)
    trainer.accelerator=SimpleNamespace(distributed_type=DistributedType.DEEPSPEED,
        deepspeed_engine_wrapped=wrapper)
    return trainer,engine,wrapper


def test_installed_deepspeed_v5_zero1_prepare_installs_adapter(native_deepspeed_lifecycle):
    """Catches rejecting the installed, witnessed v5 ZeRO-1 lifecycle."""
    from importlib.metadata import version
    assert (version('accelerate'),version('deepspeed'))==('1.10.1','0.18.7')
    trainer,engine,_=native_deepspeed_lifecycle
    trainer._install_deepspeed_step_adapter()
    assert trainer._deepspeed_step_adapter.engine is engine


def test_native_deepspeed_v5_boundary_applied_step_and_reference(native_deepspeed_lifecycle):
    """Catches counting skipped/nonboundary native steps or mutating reference."""
    trainer,engine,wrapper=native_deepspeed_lifecycle
    initial=trainer.ema_teacher.model.weight.detach().clone()
    trainer._attach_deepspeed_step_adapter(engine)
    gradients=[]
    engine.module.weight.register_hook(lambda grad:gradients.append(grad.clone()))
    for _ in range(3):
        wrapper.backward(engine.module(torch.ones(1,1)).sum()/4,sync_gradients=False,scale_wrt_gas=False)
        assert engine.is_gradient_accumulation_boundary() is False
        assert engine.optimizer.is_gradient_accumulation_boundary is False
        assert trainer._successful_optimizer_steps==engine.global_steps==0
        torch.testing.assert_close(engine.module.weight,initial,rtol=0,atol=0)
    wrapper.backward(engine.module(torch.ones(1,1)).sum()/4,sync_gradients=True,scale_wrt_gas=False)
    assert all(torch.equal(grad,torch.tensor([[.25]])) for grad in gradients)
    assert engine.is_gradient_accumulation_boundary() is True
    assert engine.optimizer.is_gradient_accumulation_boundary is True
    assert engine.was_step_applied() is True
    assert trainer._successful_optimizer_steps==engine.global_steps==1
    assert engine.lr_scheduler.last_epoch==1
    torch.testing.assert_close(engine.module.weight,torch.tensor([[.9]]))
    engine.optimizer.overflow=True
    wrapper.backward(engine.module(torch.ones(1,1)).sum(),sync_gradients=True,scale_wrt_gas=False)
    assert engine.was_step_applied() is False and engine.skipped_steps==1
    assert trainer._successful_optimizer_steps==1 and engine.global_steps==2
    assert engine.lr_scheduler.last_epoch==1
    torch.testing.assert_close(engine.module.weight,torch.tensor([[.9]]))
    engine.optimizer.overflow=False
    wrapper.backward(engine.module(torch.ones(1,1)).sum(),sync_gradients=True,scale_wrt_gas=False)
    assert engine.was_step_applied() is True and trainer._successful_optimizer_steps==2
    assert engine.lr_scheduler.last_epoch==2
    torch.testing.assert_close(engine.module.weight,torch.tensor([[.8]]))
    torch.testing.assert_close(trainer.ema_teacher.model.weight,initial,rtol=0,atol=0)


@pytest.mark.parametrize('accelerate_version,deepspeed_version,legacy,stage',[
    ('1.10.1','0.18.8',False,1),('1.10.1','0.18.9',False,1),
    ('1.11.0','0.18.7',False,1),('1.10.1','0.18.7',True,1),
    ('1.10.1','0.18.7',False,2),('1.10.1','0.18.7',False,3)])
def test_new_deepspeed_support_rejects_unverified_version_or_scope(
        native_deepspeed_lifecycle,monkeypatch,accelerate_version,deepspeed_version,legacy,stage):
    from stage1 import trainer as module
    from deepspeed_runtime import DeepSpeedRuntimeProfile
    trainer,engine,_=native_deepspeed_lifecycle
    if legacy: trainer.__class__=module.CustomTrainerStage1
    trainer.deepspeed_runtime_profile=DeepSpeedRuntimeProfile(True,stage,None,None,4)
    monkeypatch.setattr(module.importlib_metadata,'version',lambda name:
        {'accelerate':accelerate_version,'deepspeed':deepspeed_version}[name])
    with pytest.raises(RuntimeError,match='unsupported DeepSpeed'):
        trainer._install_deepspeed_step_adapter()
    assert trainer._deepspeed_step_adapter is None
    assert engine.global_steps==trainer._successful_optimizer_steps==0


def integrated_fixture(tmp_path, *, empty=False, fail=False, real_helper=False, debug=True, history_config=None):
    Trainer = api()
    student, reference, *_ = fixture(.05)
    q = resolve_qwen2_5_vl_model(student)
    q.model.gradient_checkpointing = True
    reference_q = resolve_qwen2_5_vl_model(reference)
    mask = torch.zeros_like(IDS, dtype=torch.bool); mask[0, list(BODY)] = True
    values = torch.randn(5,8)
    spans = (Stage1LatentSpan(6,9,(7,8)), Stage1LatentSpan(15,19,(16,17,18)))
    metadata = Stage1BatchMetadata((Stage1SampleMetadata(7,'cpu',(0,0),(0,0),spans,(0,1),26),),1,26)
    blocks = tuple(V5Block(0,i,s.start,s.end,() if empty else targets,
        () if empty else tuple(IDS[0,list(targets)].tolist()),(3,4,5))
        for i,s,targets in zip((0,1),spans,((10,11,12),(20,21,22,23,24))))
    batch = dict(input_ids=IDS, labels=IDS.clone(), attention_mask=torch.ones_like(IDS),
        position_ids=POSITIONS, pixel_values_latent=torch.empty(0),image_grid_thw_latent=torch.empty(0),
        stage1_batch_metadata=metadata, stage1_v5_metadata=V5BatchMetadata(blocks))
    history_case = None
    if history_config is not None:
        from tests.stage1.history_fixtures import make_history_case
        history_case = make_history_case(layers=2)
        for model in (q, reference_q):
            for config in (model.config, model.model.config):
                config.latent_token_id, config.latent_start_id, config.latent_end_id = 125, 126, 127
        batch = dict(history_case.inputs, labels=history_case.inputs['input_ids'].clone(),
            stage1_batch_metadata=history_case.metadata,
            stage1_v5_metadata=V5BatchMetadata(tuple(
                V5Block(0, i, span.start, span.end, () if empty else targets,
                    () if empty else tuple(history_case.inputs['input_ids'][0, list(targets)].tolist()),
                    candidates)
                for i, span, targets, candidates in zip((0, 1),
                    history_case.metadata.samples[0].latent_spans, ((17, 18), (23,)),
                    ((11, 12), (11, 12, 17, 18))))))
        batch['position_ids'], _ = q.get_rope_index(input_ids=batch['input_ids'],
            image_grid_thw=batch['image_grid_thw'], attention_mask=batch['attention_mask'])
    calls, helper_active, forwards, backward_events = [], [False], [], []
    q.model.register_forward_pre_hook(lambda _m, _a, kw: calls.append(
        'packed' if kw.get('stage2_prefix_key_values') is not None else 'F'), with_kwargs=True)
    reference_q.model.register_forward_pre_hook(lambda *_: calls.append('helper' if helper_active[0] else 'reference'))
    def inspect(_m,_a,kw):
        assert not any(k.startswith('stage1_v5') or k=='stage1_batch_metadata' for k in kw)
        # Direct target-builder probes also enter the Teacher wrapper now.
        # F and DIL reference scoring still require the original input object.
        observation = kw.get('output_hidden_states') is True
        if not helper_active[0] and not observation:
            assert kw['input_ids'] is batch['input_ids']
        forwards.append(kw)
    student.register_forward_pre_hook(inspect, with_kwargs=True)
    reference.register_forward_pre_hook(inspect, with_kwargs=True)
    class Helper:
        def build(self, teacher, inputs, meta):
            assert teacher is reference
            helper_active[0] = True
            with torch.no_grad(): teacher(input_ids=inputs['input_ids'],position_ids=POSITIONS,use_cache=False)
            helper_active[0] = False
            if fail: raise ValueError('helper failure')
            inputs.pop('pixel_values_latent'); inputs.pop('image_grid_thw_latent')
            return Stage1TeacherTargets(values,mask,((2,3),),{})
    trainer = Trainer.__new__(Trainer)
    trainer.stage1_v5_debug=debug
    trainer.model = student; trainer.model_wrapped = student
    trainer.ema_teacher = Stage1EmaTeacher(student,decay=1.,method='dora',initial_reference_model=reference)
    trainer.target_builder=Helper(); trainer._ema_execution_device=None
    if real_helper or history_config is not None:
        from dataclasses import replace
        from stage1.targets import Stage1TeacherTargetBuilder
        class Tokenizer:
            def __call__(self,text,**kw):
                ids={'<|im_start|>assistant':23,'<|im_start|>assistant\n':23,
                     '<|latent_pad|>':126,'<|latent_start|>':125,'<|latent_end|>':127,
                     '<|vision_start|>':81,'<|vision_end|>':83}
                return {'input_ids':torch.tensor([[ids[text]]])}
        native=Stage1TeacherTargetBuilder(
            history_case.tokenizer if history_case else Tokenizer(),
            image_pool_k=2, helper_group_L=2,
            **({'query_config': history_config} if history_case else {}))
        native.validate=debug; native.collect_diagnostics=debug
        class RealHelper:
            def build(self,teacher,inputs,meta):
                helper_active[0]=True
                try:
                    self.targets=native.build(teacher,inputs,meta)
                    return self.targets
                finally: helper_active[0]=False
        trainer.target_builder=RealHelper()
        if history_case is None:
            ids=IDS.clone(); ids[0,:4]=80
            sample=replace(metadata.samples[0],original_grid_range=(0,1),helper_grid_range=(0,2),
                           original_pixel_range=(0,4),helper_pixel_range=(0,8))
            batch.update(input_ids=ids,pixel_values=torch.randn(4,12),image_grid_thw=torch.tensor([[1,2,2]]),
                pixel_values_latent=torch.randn(8,12),image_grid_thw_latent=torch.tensor([[1,2,2],[1,2,2]]),
                stage1_batch_metadata=Stage1BatchMetadata((sample,),1,26))
    if history_config is not None: trainer.query_config = history_config
    trainer.ce_weight=trainer.sim_weight=.1; trainer.collect_stage1_diagnostics=debug
    trainer.v5_config=Stage1V5Config(lambda_topk_entropy=.1,preference_topk_k=50)
    trainer._v5_contract_ready=True; trainer._v5_step_stack=None
    trainer.args=TrainingArguments(str(tmp_path),use_cpu=True,per_device_train_batch_size=1,gradient_accumulation_steps=4,seed=42,report_to=[])
    trainer.state=SimpleNamespace(global_step=0)
    trainer.optimizer=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=5e-5)
    trainer.model_accepts_loss_kwargs=False; trainer.compute_loss_func=None; trainer.use_apex=False; trainer.use_cpu_amp=False
    trainer.accelerator=Accelerator(cpu=True)
    original_backward=trainer.accelerator.backward
    def observed_backward(loss, **kw):
        from stage1.v5_execution import current_v5_execution
        owner=current_v5_execution()
        assert owner is not None and owner.state=='BACKWARD'
        backward_events.append(owner)
        return original_backward(loss, **kw)
    trainer.accelerator.backward=observed_backward
    trainer._prepare_inputs=lambda x:x
    return trainer,batch,calls,forwards,backward_events,values


def test_actual_four_calls_one_parent_backward_native_update_and_reference_owned_inputs(tmp_path):
    trainer,batch,calls,forwards,backs,values=integrated_fixture(tmp_path)
    before=digest_state(trainer.ema_teacher.model)
    identities=tuple(id(p) for p in trainer.model.parameters())
    batches,_=trainer.get_batch_samples(iter([batch]),4)
    assert batches[0]['stage1_v5_window'].global_original_steps==2
    assert batches[0]['stage1_v5_window'].actual_microbatches==1
    loss=trainer.training_step(trainer.model,batches[0])
    assert torch.isfinite(loss)
    assert calls==['helper','F','reference','packed']
    assert len(backs)==1 and backs[0].state=='CLOSED'
    assert forwards[1]['latent_hidden_states'].data_ptr()==forwards[2]['latent_hidden_states'].data_ptr()==values.data_ptr()
    assert forwards[1]['labels'] is batch['labels'] and 'labels' not in forwards[2]
    assert not torch.equal(resolve_qwen2_5_vl_model(trainer.model).model.embed_tokens(IDS),
        resolve_qwen2_5_vl_model(trainer.ema_teacher.model).model.embed_tokens(IDS))
    diagnostics=trainer._latest_stage1_diagnostics
    expected=(.1*diagnostics['ce_loss']+.1*diagnostics['alignment_loss']+
        diagnostics['dil_scale']*diagnostics['dil_raw_sum']+
        diagnostics['auxiliary_scale']*.1*diagnostics['topk_entropy_raw_sum']+
        .01*diagnostics['text_kl_raw'])/4
    torch.testing.assert_close(loss,expected)
    assert any(e['phase']=='recompute' for e in backs[0].events)
    updates=[]
    trainer.optimizer.register_step_post_hook(lambda *_:updates.append(1))
    trainer.optimizer.step()
    assert len(updates)==1 and tuple(id(p) for p in trainer.model.parameters())==identities
    assert digest_state(trainer.ema_teacher.model)==before
    assert all(p.grad is None for p in trainer.ema_teacher.model.parameters())
    assert trainer._v5_step_stack is None


def test_v5_exclusively_owns_dropout_through_parent_backward_and_checkpoint_replay(tmp_path, monkeypatch):
    """A real V5 backward must not acquire the ordinary Stage 1 dropout owner."""
    from stage1.equivalence import SampleStableStage1Dropout, _STAGE1_STOCHASTIC_CONTEXT
    from stage1.v5_execution import current_v5_execution

    trainer, batch, calls, _, backwards, _ = integrated_fixture(tmp_path)
    dropout = [module for module in trainer.model.modules()
               if isinstance(module, SampleStableStage1Dropout)]
    assert dropout and all(module.p == .05 for module in dropout)
    assert resolve_qwen2_5_vl_model(trainer.model).model.gradient_checkpointing
    sites = {module.site_name for module in dropout}
    observations = []
    real_backward = trainer.accelerator.backward

    def observed_backward(loss, **kwargs):
        owner = current_v5_execution()
        legacy_owner = _STAGE1_STOCHASTIC_CONTEXT.get()
        assert owner is not None and owner.state == 'BACKWARD'
        result = real_backward(loss, **kwargs)
        observations.append(owner)
        assert current_v5_execution() is owner and owner.state == 'BACKWARD'
        assert legacy_owner is None, 'V5 backward acquired the legacy Stage 1 dropout owner'
        assert _STAGE1_STOCHASTIC_CONTEXT.get() is None
        return result

    monkeypatch.setattr(trainer.accelerator, 'backward', observed_backward)
    batches, _ = trainer.get_batch_samples(iter([batch]), 4)
    try:
        loss = trainer.training_step(trainer.model, batches[0])
    finally:
        assert current_v5_execution() is None
        assert _STAGE1_STOCHASTIC_CONTEXT.get() is None
        assert trainer._v5_step_stack is None and trainer._v5_execution is None

    assert torch.isfinite(loss)
    assert calls == ['helper', 'F', 'reference', 'packed']
    assert len(observations) == len(backwards) == 1
    assert observations[0] is backwards[0] and backwards[0].state == 'CLOSED'
    for role in ('F', 'packed'):
        for phase in ('original', 'recompute'):
            observed_sites = [event['site'] for event in backwards[0].events
                              if event['role'] == role and event['phase'] == phase
                              and event['site'] is not None]
            assert len(observed_sites) == len(sites) and set(observed_sites) == sites
    gradients = [parameter.grad for parameter in trainer.model.parameters()
                 if parameter.requires_grad]
    assert gradients and all(gradient is not None and bool(torch.isfinite(gradient).all())
                             for gradient in gradients)
    assert any(bool(gradient.abs().sum() > 0) for gradient in gradients)


def test_empty_original_blocks_count_but_skip_reference_and_packed(tmp_path):
    trainer,batch,calls,_,backs,_=integrated_fixture(tmp_path,empty=True)
    batches,_=trainer.get_batch_samples(iter([batch]),4)
    assert batches[0]['stage1_v5_window'].global_original_steps==2
    trainer.training_step(trainer.model,batches[0])
    assert calls==['helper','F'] and len(backs)==1


def test_missing_window_fails_before_helper_and_failure_releases_owner(tmp_path):
    trainer,batch,calls,*_=integrated_fixture(tmp_path)
    with pytest.raises(ValueError,match='window'): trainer.training_step(trainer.model,batch)
    assert calls==[]
    trainer,batch,_,*_=integrated_fixture(tmp_path,fail=True)
    batches,_=trainer.get_batch_samples(iter([batch]),4)
    with pytest.raises(ValueError,match='helper failure'): trainer.training_step(trainer.model,batches[0])
    from stage1.v5_execution import current_v5_execution
    assert current_v5_execution() is None and trainer._v5_step_stack is None


def test_parser_inherits_saved_v5_before_eager_entropy_validation(tmp_path):
    from utils_deepseed import parse_args
    from tests.stage1.v5.test_resume import contract
    import json
    saved=contract()
    saved['schedule'].update(configured_max_steps=3,num_train_epochs=1.)
    saved['execution'].update(warmup_steps=10,lora_r=32,lora_alpha=64,lora_dropout=.05,lora_target_modules='q_proj')
    (tmp_path/'stage1_v5.json').write_text(json.dumps(saved))
    args=parse_args(['--resume_from_checkpoint_path',str(tmp_path),'--lambda_topk_entropy','.1'])
    assert args.stage1_variant=='v5'
    assert args.stage1_v5_config.preference_topk_k==50
    assert args.ce_weight==args.sim_weight==.1
    assert args.ema_tau==1.
    assert args.max_steps==3 and args.epochs==1. and args.warm_up_steps==10
    from stage1.types import resolve_stage1_tuning_config
    assert resolve_stage1_tuning_config(args).target_suffixes==('q_proj',)
    with pytest.raises(ValueError,match='conflict'):
        parse_args(['--resume_from_checkpoint_path',str(tmp_path),'--stage1_variant','standard'])
    with pytest.raises(ValueError,match='conflict'):
        parse_args(['--resume_from_checkpoint_path',str(tmp_path),'--ce_weight','1'])
    with pytest.raises(ValueError):
        parse_args(['--resume_from_checkpoint_path',str(tmp_path/'legacy'),'--stage1_variant','v5'])


def test_save_copies_validated_small_reference_and_contract_not_policy(tmp_path,monkeypatch):
    Trainer=api()
    from stage1.trainer import CustomTrainerStage1
    from tests.stage1.v5.test_resume import contract
    import artifact_paths
    monkeypatch.setattr(artifact_paths,'CANONICAL_CHECKPOINT_ROOT',tmp_path)
    source=tmp_path/'run'; source.mkdir()
    (source/'stage1_v5_reference.pt').write_bytes(b'validated immutable initial state')
    trainer=Trainer.__new__(Trainer)
    trainer.args=SimpleNamespace(output_dir=str(source),should_save=True)
    trainer._v5_contract=contract(); trainer._v5_contract_ready=True
    trainer.accelerator=SimpleNamespace(device=torch.device('cpu'),reduce=lambda t,**kw:t,wait_for_everyone=lambda:None)
    saves=[]
    monkeypatch.setattr(CustomTrainerStage1,'save_model',lambda self,*a,**k:saves.append(a))
    dest=tmp_path/'checkpoint-1'
    trainer.save_model(str(dest))
    from stage1.v5_resume import load_v5_contract
    assert load_v5_contract(dest)==trainer._v5_contract
    assert (dest/'stage1_v5_reference.pt').read_bytes()==(source/'stage1_v5_reference.pt').read_bytes()
    assert len(saves)==1


def test_composition_rejects_unsupported_v5_before_loading(tmp_path):
    from tests.stage1.test_composition import stage1_args
    from stage1.composition import build_stage1_training_components
    from stage1.composition import DEFAULT_STAGE1_DEPENDENCIES
    from dataclasses import replace
    def forbidden(*a,**k): raise AssertionError('model/processor loaded before v5 profile rejection')
    dependencies=replace(DEFAULT_STAGE1_DEPENDENCIES,load_processor=forbidden)
    args=stage1_args(tmp_path,'--stage1_variant','v5',method='full')
    with pytest.raises(ValueError,match='v5.*profile'):
        build_stage1_training_components(args,dependencies=dependencies)


def test_v5_composition_constructs_actual_mixed_attention_modules(monkeypatch):
    from stage1 import composition
    assert hasattr(composition,'_load_stage1_model'), 'v5 mixed-attention construction missing'
    from tests.stage2.test_attention_backend import _tiny_qwen_config
    from transformers import Qwen2_5_VLForConditionalGeneration
    from stage2.attention_backend import validate_qwen2_5_vl_attention_backends
    monkeypatch.setattr(Qwen2_5_VLForConditionalGeneration,'_check_and_enable_flash_attn_2',
                        classmethod(lambda cls,config,**kw:config))
    args=SimpleNamespace(stage1_variant='v5',attention_implementation='sdpa',
        vision_attention_implementation='flash_attention_2',cache_dir=None)
    def load(source,config,**kwargs):
        return Qwen2_5_VLForConditionalGeneration(config)
    model=composition._load_stage1_model('tiny',_tiny_qwen_config(),args,SimpleNamespace(load_model=load))
    validate_qwen2_5_vl_attention_backends(model,attention_implementation='sdpa',
                                         vision_attention_implementation='flash_attention_2')


def test_parent_ema_restore_cannot_replace_immutable_initial_reference(tmp_path):
    trainer,*_=integrated_fixture(tmp_path)
    from stage1.v5_reference import V5ReferenceIdentity, _state_identity
    trainer.v5_reference_identity=V5ReferenceIdentity(1,'a'*64,
        _state_identity(trainer.ema_teacher,trainer.ema_teacher.state_dict()),'c'*64,
        trainer.ema_teacher.parameter_names)
    trainer._save_stage1_ema_checkpoint(tmp_path)
    trainer._load_stage1_ema_checkpoint(tmp_path)
    path=tmp_path/'stage1_ema_teacher.pt'
    payload=torch.load(path,weights_only=True)
    next(iter(payload['state']['parameters'].values())).add_(1.)
    torch.save(payload,path)
    with pytest.raises(ValueError,match='immutable'):
        trainer._load_stage1_ema_checkpoint(tmp_path)


def test_get_batch_samples_reduces_original_blocks_once_even_empty_rank(tmp_path):
    from tests.stage1.v5.test_reduction import FOUR_RANK_WINDOWS
    from stage1.v5_metadata import V5BatchMetadata,V5Block
    Trainer=api()
    for counts in FOUR_RANK_WINDOWS:
        total=sum(map(sum,counts))
        for rank,slots in enumerate(counts):
            reductions=[]
            trainer=Trainer.__new__(Trainer)
            trainer.args=SimpleNamespace(average_tokens_across_devices=False,gradient_accumulation_steps=4)
            def reduce(value,**kw):
                reductions.append(value.tolist())
                return torch.tensor([total, 0])
            trainer.accelerator=SimpleNamespace(device=torch.device('cpu'),num_processes=4,reduce=reduce)
            original=[{'stage1_v5_metadata':V5BatchMetadata(tuple(
                V5Block(0,i,2*i,2*i+1,(),(),()) for i in range(n)))} for n in slots]
            batches,_=trainer.get_batch_samples(iter(original),4)
            assert reductions==[[sum(slots), 0]]
            assert all(b['stage1_v5_window'].global_original_steps==total for b in batches)
            assert all(b['stage1_v5_window'].actual_microbatches==len(slots) for b in batches)
            assert all('stage1_v5_window' not in b for b in original)


def test_actual_native_helper_builder_shares_snapshot_with_own_input_reference(tmp_path):
    trainer,batch,calls,forwards,backs,_=integrated_fixture(tmp_path,real_helper=True)
    batches,_=trainer.get_batch_samples(iter([batch]),4)
    loss=trainer.training_step(trainer.model,batches[0])
    assert torch.isfinite(loss) and calls==['helper','F','reference','packed']
    targets=trainer.target_builder.targets
    assert targets.latent_values.shape[0]>0 and not targets.latent_values.requires_grad
    assert forwards[1]['latent_hidden_states'].data_ptr()==targets.latent_values.data_ptr()
    assert forwards[2]['latent_hidden_states'].data_ptr()==targets.latent_values.data_ptr()
    assert forwards[1]['pixel_values'] is forwards[2]['pixel_values'] is batch['pixel_values']
    assert len(backs)==1 and backs[0].state=='CLOSED'


@pytest.mark.parametrize('drift', ['lr', 'old_sources', 'missing_profile', 'changed_source', 'changed_profile'])
def test_train_publishes_actual_loader_contract_before_parent_dispatch_and_rejects_drift(tmp_path,monkeypatch,drift):
    from torch.utils.data import DataLoader
    from stage1.trainer import CustomTrainerStage1
    from stage1.v5_reference import V5ReferenceIdentity,_state_identity
    from deepspeed_runtime import DeepSpeedRuntimeProfile
    from stage1.v5_resume import load_v5_contract
    import artifact_paths
    monkeypatch.setattr(artifact_paths,'CANONICAL_CHECKPOINT_ROOT',tmp_path.parent)
    trainer,batch,calls,*_=integrated_fixture(tmp_path)
    trainer._v5_contract_ready=False; trainer.v5_saved_contract=None
    trainer.v5_reference_identity=V5ReferenceIdentity(1,'a'*64,
        _state_identity(trainer.ema_teacher,trainer.ema_teacher.state_dict()),'c'*64,
        trainer.ema_teacher.parameter_names)
    from stage1.types import DEFAULT_STAGE1_TARGET_SUFFIXES
    trainer.stage1_peft_metadata=SimpleNamespace(base_model_reference='original/tiny',requested_target_suffixes=DEFAULT_STAGE1_TARGET_SUFFIXES)
    trainer.coverage_p=.9; trainer.image_pool_k=2; trainer.helper_group_L=2; trainer.textual_states='previous'
    resolve_qwen2_5_vl_model(trainer.model).config.compress_strategy='attention'
    # Exercise contract collection, not BF16 computation or a ZeRO engine.
    trainer.args.bf16=True; trainer.args.max_seq_length=32768; trainer.args.max_steps=3
    trainer.deepspeed_runtime_profile=DeepSpeedRuntimeProfile(True,1,None,'a'*64,4)
    trainer.train_dataset=list(range(10))
    loader=DataLoader(trainer.train_dataset,batch_size=1)
    trainer.get_train_dataloader=lambda:loader
    (tmp_path/'stage1_v5_reference.pt').write_bytes(b'initial selected state')
    parent_calls=[]
    def parent(self,**kw):
        assert calls==[] and self._v5_contract_ready
        saved=load_v5_contract(tmp_path)
        assert saved['schedule']['prepared_microbatches_per_rank']==10
        assert saved['schedule']['one_exposure_window_updates']==3
        assert saved['schedule']['parent_epoch_update_budget']==3
        assert saved['schedule']['final_window_microbatches']==2
        assert saved['schedule']['configured_max_steps']==3
        assert saved['execution']['packages']['accelerate']=='1.10.1'
        assert saved['execution']['lora_target_modules']==','.join(DEFAULT_STAGE1_TARGET_SUFFIXES)
        assert set(saved['execution']['source_sha256'])=={'v5_trainer','stage1_trainer','transformers_trainer',
            'query','targets','history_attention','history_teacher','history_metric',
            'v5_execution','v5_selection','v5_trace','v5_replay'}
        assert saved['execution']['operator_profile']=={'id':'stage1_v5_vectorized_ops_v1',
            'attention_query_chunk_max':32,'attention_score_probability_target_bytes':33554432}
        parent_calls.append(kw)
        return 'parent-result'
    monkeypatch.setattr(CustomTrainerStage1,'train',parent)
    assert trainer.train()=='parent-result'
    saved=load_v5_contract(tmp_path)
    if drift=='lr': trainer.args.learning_rate*=2
    elif drift=='old_sources':
        for key in ('v5_execution','v5_selection','v5_trace','v5_replay'):
            saved['execution']['source_sha256'].pop(key)
    elif drift=='missing_profile': saved['execution'].pop('operator_profile')
    elif drift=='changed_source': saved['execution']['source_sha256']['v5_replay']='0'*64
    else: saved['execution']['operator_profile']['attention_query_chunk_max']=16
    import json
    sidecar=tmp_path/'stage1_v5.json'
    sidecar.write_text(json.dumps(saved))
    original_bytes=sidecar.read_bytes()
    assert load_v5_contract(tmp_path)==saved  # Historical inspection remains read-only.
    trainer.v5_saved_contract=saved
    forbidden=[]
    trainer._publish_v5_files=lambda *a:forbidden.append('publish')
    trainer.create_optimizer=lambda *a:forbidden.append('optimizer')
    with pytest.raises(ValueError,match='conflicts.*original reviewed revision'):
        trainer.train(resume_from_checkpoint=str(tmp_path))
    assert len(parent_calls)==1 and calls==[] and forbidden==[]
    assert not trainer._v5_contract_ready and sidecar.read_bytes()==original_bytes
    print(f'RESUME_DISPATCH drift={drift} parent_calls=1 optimizer_calls=0 publication_calls=0 historical_bytes_unchanged=True')


@pytest.mark.parametrize('changed', ['objective','lr','packages','source','base','processor','deepspeed','query',
                                    'v5_execution','v5_selection','v5_trace','v5_replay','operator_profile'])
def test_every_rank_rejects_divergent_effective_contract_before_ready(tmp_path,monkeypatch,changed):
    from tests.stage1.v5.test_resume import contract
    from stage1.trainer import CustomTrainerStage1
    import hashlib,json
    contracts=[deepcopy(contract()) for _ in range(4)]
    for rank,value in enumerate(contracts):
        value['execution'].update(model=f'/node{rank//2}/models/base',learning_rate=5e-5,
            packages={'torch':'pinned'},source_sha256={'trainer':'a'*64,'v5_execution':'b'*64,
                'v5_selection':'c'*64,'v5_trace':'d'*64,'v5_replay':'e'*64},
            operator_profile={'id':'stage1_v5_vectorized_ops_v1','attention_query_chunk_max':32,
                              'attention_score_probability_target_bytes':33554432},
            deepspeed_profile={'config_path':f'/node{rank//2}/repo/configs/zero1.json','config_digest':'d'*64})
    bad=contracts[2]
    if changed == 'objective': bad['config']['lambda_topk_entropy']=.2
    elif changed == 'lr': bad['execution']['learning_rate']=1e-4
    elif changed == 'packages': bad['execution']['packages']['torch']='other'
    elif changed == 'source': bad['execution']['source_sha256']['trainer']='e'*64
    elif changed.startswith('v5_'): bad['execution']['source_sha256'][changed]='0'*64
    elif changed == 'operator_profile': bad['execution']['operator_profile']['attention_query_chunk_max']=16
    elif changed == 'base': bad['reference']['base_content_sha256']='e'*64
    elif changed == 'deepspeed': bad['execution']['deepspeed_profile']['config_digest']='e'*64
    elif changed == 'query':
        from tests.stage1.test_history_query import NEW_QUERY
        bad['execution']['query_config'] = dict(NEW_QUERY)
    else: bad['reference']['processor_sha256']='e'*64
    def independent_digest(value):
        mapped=deepcopy(value)
        # Historical sidecars predate query metadata and represent this exact
        # legacy schema; keep the oracle independent of product normalization.
        mapped['execution'].setdefault('query_config', {
            'schema_version': 1, 'mode': 'legacy', 'formula_id': 'stage1_legacy_query_v1',
        })
        mapped['execution']['model']=mapped['reference']['base_content_sha256']
        mapped['execution']['deepspeed_profile']['config_path']=mapped['execution']['deepspeed_profile']['config_digest']
        return torch.tensor(list(hashlib.sha256(json.dumps(mapped,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).digest()),dtype=torch.uint8)
    ranks=torch.cat([torch.cat((independent_digest(c),torch.zeros(33,dtype=torch.uint8))) for c in contracts])
    calls=[]
    monkeypatch.setattr(CustomTrainerStage1,'train',lambda *a,**kw:calls.append('parent/helper'))
    for rank,local in enumerate(contracts):
        trainer=api().__new__(api())
        trainer.args=SimpleNamespace(output_dir=str(tmp_path))
        trainer.v5_saved_contract=None
        trainer.get_train_dataloader=lambda:None
        trainer._build_v5_runtime_contract=lambda _loader,value=local:value
        trainer._publish_v5_files=lambda directory:calls.append('publish')
        def gather(value):
            torch.testing.assert_close(value[:32],independent_digest(local),rtol=0,atol=0)
            return ranks
        trainer.accelerator=SimpleNamespace(device=torch.device('cpu'),num_processes=4,gather=gather)
        with pytest.raises(ValueError,match='agreement'):
            trainer.train()
        assert trainer._v5_contract_ready is False
    assert calls == []


@pytest.mark.parametrize('explicit_legacy_ranks', [(), (1, 3)],
                         ids=['historical', 'mixed_historical_explicit_legacy'])
def test_effective_contract_allows_only_content_equal_node_local_paths(tmp_path,monkeypatch,explicit_legacy_ranks):
    from tests.stage1.v5.test_resume import contract
    from stage1.trainer import CustomTrainerStage1
    import hashlib,json
    mapped=contract()
    mapped['execution']['query_config'] = {
        'schema_version': 1, 'mode': 'legacy', 'formula_id': 'stage1_legacy_query_v1',
    }
    mapped['execution'].update(model=mapped['reference']['base_content_sha256'],
        deepspeed_profile={'config_path':'d'*64,'config_digest':'d'*64})
    expected=torch.tensor(list(hashlib.sha256(json.dumps(mapped,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).digest())+[0]*33,dtype=torch.uint8)
    calls=[]
    monkeypatch.setattr(CustomTrainerStage1,'train',lambda *a,**kw:calls.append('parent'))
    for rank in range(4):
        trainer=api().__new__(api())
        local=contract()
        if rank in explicit_legacy_ranks:
            local['execution']['query_config'] = {
                'schema_version': 1, 'mode': 'legacy', 'formula_id': 'stage1_legacy_query_v1',
            }
        local['execution'].update(model=f'/node{rank//2}/models/base',
            deepspeed_profile={'config_path':f'/node{rank//2}/repo/zero1.json','config_digest':'d'*64})
        trainer.args=SimpleNamespace(output_dir=str(tmp_path))
        trainer.v5_saved_contract=None
        trainer.get_train_dataloader=lambda:None
        trainer._build_v5_runtime_contract=lambda loader:local
        trainer._publish_v5_files=lambda directory:calls.append('publish')
        trainer.accelerator=SimpleNamespace(device=torch.device('cpu'),num_processes=4,
            gather=lambda value:expected.repeat(4))
        trainer.train()
        assert trainer._v5_contract_ready
    assert calls == ['publish','parent']*4


def test_resume_parser_keeps_explicit_local_paths_for_later_content_validation(tmp_path,monkeypatch):
    from tests.stage1.v5.test_resume import contract
    from utils_deepseed import parse_args
    from pathlib import Path
    import hashlib,json
    local_base=tmp_path/'original-base'; local_base.mkdir()
    local_config=tmp_path/'zero1.json'
    raw=(Path(__file__).resolve().parents[3]/'configs/deepspeed_zero1.json').read_bytes()
    local_config.write_bytes(raw)
    saved=contract()
    saved['execution'].update(model='/other-node/original-base',
        deepspeed_profile={'config_path':'/other-node/zero1.json','config_digest':hashlib.sha256(raw).hexdigest()})
    (tmp_path/'stage1_v5.json').write_text(json.dumps(saved))
    monkeypatch.setenv('WORLD_SIZE','4')
    args=parse_args(['--resume_from_checkpoint_path',str(tmp_path),'--model',str(local_base),
                     '--deepspeed_config',str(local_config)])
    assert args.model==str(local_base) and args.deepspeed_config==str(local_config.resolve())
    from deepspeed_runtime import bind_deepspeed_runtime_profile
    profile=bind_deepspeed_runtime_profile(args,world_size=4)
    assert profile.config_digest==saved['execution']['deepspeed_profile']['config_digest']
    # A different local base still has to pass composition's mandatory content
    # restore (covered by actual inspector/restore tests); no HF-name fallback.
    with pytest.raises(ValueError):
        parse_args(['--resume_from_checkpoint_path',str(tmp_path),'--model','Qwen/not-local'])
