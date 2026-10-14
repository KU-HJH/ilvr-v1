"""Reject wrong lifecycle ownership; preserve native preparation and restoration."""
from copy import deepcopy
import pytest
import torch
from tests.stage1.v5.test_replay import runtime_api, fixture, IDS, BODY, POSITIONS
from peft_common import resolve_qwen2_5_vl_model


def test_execution_source_coordinates_are_shared_and_single_backward_owned():
    Execution,Spec,*_ = runtime_api()
    spec = Spec(42,3,((7,5),(9,3)),('test',))
    with Execution(spec) as execution:
        with execution.graph('F'):
            with execution.checkpoint_contexts('F',('decoder',0))[0]:
                clean = execution.dropout_mask('test',.05,sample_indices=torch.tensor([0,0,1]),
                    token_positions=torch.tensor([2,4,1]),hidden_size=8,device=torch.device('cpu'))
        with execution.graph('packed'):
            with execution.checkpoint_contexts('packed',('decoder',0))[0]:
                packed = execution.dropout_mask('test',.05,sample_indices=torch.tensor([1,0,0,1]),
                    token_positions=torch.tensor([1,4,2,1]),hidden_size=8,device=torch.device('cpu'))
        torch.testing.assert_close(packed,clean[[2,1,0,2]],atol=0,rtol=0)
        with execution.backward():
            with execution.checkpoint_contexts('packed',('decoder',0))[1]: pass
        with pytest.raises(RuntimeError,match='backward'):
            with execution.backward(): pass


def test_exception_restores_execution_and_native_layer_modes():
    Execution,Spec,capture,bind,*_ = runtime_api()
    student,*_ = fixture(.05)
    q = resolve_qwen2_5_vl_model(student)
    q.model.gradient_checkpointing = True
    original = q.model.layers[0].forward
    before = torch.autograd.is_multithreading_enabled()
    with pytest.raises(ValueError,match='intentional'):
        with Execution(Spec(42,0,((7,26),),student.stage1_stochastic_sites)) as execution:
            with execution.graph('F'),bind(q.model,execution,role='F'):
                hook = q.model.layers[0].register_forward_pre_hook(lambda *_: (_ for _ in ()).throw(ValueError('intentional')))
                try: q.model(inputs_embeds=torch.randn(1,26,8),position_ids=POSITIONS,use_cache=False)
                finally: hook.remove()
    assert torch.autograd.is_multithreading_enabled() == before
    assert q.model.gradient_checkpointing
    assert q.model.layers[0].forward == original
    # Native default eval call works after failed scoped execution.
    q.eval()(input_ids=IDS,position_ids=POSITIONS,use_cache=False)


def test_selected_endpoint_uses_reference_owned_image_tokens_and_latents():
    runtime_api()
    student,*_ = fixture(.05)
    q = resolve_qwen2_5_vl_model(student).eval()
    reference = deepcopy(q).eval()
    with torch.no_grad(): reference.model.embed_tokens.token_adapter.trainable_tokens_delta['default'].add_(.04)
    ids = IDS.clone()
    ids[0,:4] = 80
    mask = torch.zeros_like(ids,dtype=torch.bool); mask[0,list(BODY)] = True
    helper = torch.randn(5,8)
    image = torch.randn(4,12)
    kwargs = dict(input_ids=ids,position_ids=POSITIONS,attention_mask=torch.ones_like(ids),
                  latent_hidden_states=helper,image_out_mask=mask,pixel_values=image,
                  image_grid_thw=torch.tensor([[1,2,2]]),use_cache=False,return_dict=True)
    calls = []
    visual_hook = reference.visual.register_forward_hook(lambda *_: calls.append('reference-vision'))
    head_hook = reference.lm_head.register_forward_pre_hook(lambda _m,a: calls.append(tuple(a[0].shape)))
    keys = ((0,9),(0,12),(0,19))
    try:
        with torch.no_grad():
            full = reference(**kwargs)
            selected = reference(**kwargs,stage1_selected_positions=keys)
        assert selected.logits is None
        torch.testing.assert_close(selected.stage1_selected_logits,full.logits[0,[9,12,19]])
        torch.testing.assert_close(selected.inputs_embeds[mask],helper)
        assert calls == ['reference-vision',(1,26,8),'reference-vision',(3,8)]
        assert not torch.equal(selected.inputs_embeds[0,6],q.model.embed_tokens(ids)[0,6])
    finally:
        visual_hook.remove(); head_hook.remove()
    with pytest.raises(ValueError,match='labels'):
        reference(**kwargs,stage1_selected_positions=keys,labels=ids)
    with pytest.raises(ValueError,match='Stage 1'):
        reference.config.stage='stage2'
        reference(**kwargs,stage1_selected_positions=keys)


@pytest.mark.parametrize('keys',[((0,26),),((1,9),),((0,9),(0,9)),((0,-1),)])
def test_selected_endpoint_rejects_invalid_queries(keys):
    runtime_api()
    student,*_ = fixture(.05)
    q = resolve_qwen2_5_vl_model(student).eval()
    with pytest.raises(ValueError,match='selected'):
        q(input_ids=IDS,position_ids=POSITIONS,use_cache=False,stage1_selected_positions=keys)


def test_native_vision_checkpoint_binding_preserves_parameters_and_restores_callable():
    Execution,Spec,_,bind,*_ = runtime_api()
    student,*_ = fixture(.05)
    q = resolve_qwen2_5_vl_model(student)
    q.visual.gradient_checkpointing = True
    from torch.utils.checkpoint import checkpoint
    q.visual._gradient_checkpointing_func = checkpoint
    old = q.visual._gradient_checkpointing_func
    identities = tuple(id(p) for p in q.visual.parameters())
    ids = IDS.clone(); ids[0,:4]=80
    image = torch.randn(4,12,requires_grad=True)
    with Execution(Spec(42,0,((7,26),),student.stage1_stochastic_sites)) as execution:
        with execution.graph('F'),bind(q.model,execution,role='F',visual=q.visual):
            output = q(input_ids=ids,position_ids=POSITIONS,pixel_values=image,
                       image_grid_thw=torch.tensor([[1,2,2]]),use_cache=False)
        assert q.visual._gradient_checkpointing_func is old
        with execution.backward(): output.logits.sum().backward()
    assert image.grad is not None and image.grad.abs().sum()>0
    assert identities == tuple(id(p) for p in q.visual.parameters())
    assert any(e['unit'][0]=='vision' and e['phase']=='recompute' for e in execution.events)


def test_empty_targets_do_not_invoke_packed_decoder():
    Execution,Spec,capture,bind,Block,replay = runtime_api()
    student,*_ = fixture(.05)
    q = resolve_qwen2_5_vl_model(student)
    calls = []
    handle = q.model.register_forward_pre_hook(lambda *_:calls.append('decoder'))
    try:
        with Execution(Spec(42,0,((7,26),),student.stage1_stochastic_sites)) as execution:
            with execution.graph('F'),bind(q.model,execution),capture(q,(),()) as trace:
                output = q(input_ids=IDS,position_ids=POSITIONS,use_cache=False)
            with execution.graph('packed'):
                off = replay(q.model,q.lm_head,trace,(Block(0,6,9,(),(),()),),
                             execution=execution,need_voff=True,need_textoff=True)
            assert off.query_keys == () and off.voff_logits is None and off.textoff_logits is None
            assert calls == ['decoder']
            with execution.backward(): output.logits.sum().backward()
    finally: handle.remove()


def test_trace_observations_match_native_full_causal_attention():
    Execution,Spec,capture,bind,*_ = runtime_api()
    student,*_ = fixture(0.,backend='eager')
    q = resolve_qwen2_5_vl_model(student)
    keys,candidates = ((0,9),(0,19)),((3,4,5),(3,10,14))
    with Execution(Spec(42,0,((7,26),),student.stage1_stochastic_sites)) as execution:
        with execution.graph('F'),bind(q.model,execution),capture(q,keys,candidates) as trace:
            output = q(input_ids=IDS,position_ids=POSITIONS,use_cache=False,output_attentions=True)
        for key,positions in zip(keys,candidates):
            row,query = key
            expected = torch.stack([a[row,:,query][:,list(positions)].mean(0) for a in output.attentions]).mean(0)
            torch.testing.assert_close(trace.attention_rows[key],expected)
            assert not trace.attention_rows[key].requires_grad
            assert trace.attention_rows[key].sum()<1
        assert trace.inputs_embeds is output.inputs_embeds
        assert all(value.requires_grad for pair in trace.key_values for value in pair)
        with execution.backward(): output.logits.sum().backward()
