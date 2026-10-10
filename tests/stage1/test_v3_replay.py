"""Packed product output/derivative versus separately masked full Qwen calls."""
import importlib
from contextlib import nullcontext
from pathlib import Path
import pytest
import torch
from torch.utils.checkpoint import set_checkpoint_early_stop
from stage1.equivalence import Stage1StochasticContext, activate_stage1_stochastic_context
from stage1.v3_lifecycle import V3Lifecycle, packed_dropout_mask, current_v3_lifecycle
from tests.stage1.v3_helpers import replay_qwen


def modules():
    return tuple(importlib.import_module('transformers.models.qwen2_5_vl.'+name)
                 for name in ('stage1_v3_trace','stage1_v3_replay'))


def full_mask(length, block, rv, dtype):
    # Independent full-sequence oracle; mask indices are original positions.
    visible = torch.ones(length,length,dtype=torch.bool).tril()
    for t,keys in zip(block.target_positions,block.blocked_text_keys):
        visible[t-1,list(keys)] = False
    if rv:
        visible[block.end:,block.start+1:block.end] = False
    return torch.zeros(1,1,length,length,dtype=dtype).masked_fill(~visible,float('-inf'))


@pytest.mark.parametrize('dtype', [torch.float64,torch.float32])
@pytest.mark.parametrize('p',[0.,.05])
@pytest.mark.parametrize('mode',['off','early','full'])
def test_packed_replay_matches_independent_qwen(dtype,p,mode, native_norm=False, fault=False):
    trace_api, replay_api = modules()
    decoder,head,sites = replay_qwen(dtype,p)
    if dtype == torch.float64 and not native_norm:
        # Isolate FP64 graph algebra: production RMSNorm always reduces in FP32.
        import types
        for module in decoder.modules():
            if module.__class__.__name__ == 'Qwen2RMSNorm':
                def fp64_norm(self,value):
                    return self.weight*value*torch.rsqrt(value.square().mean(-1,keepdim=True)+self.variance_epsilon)
                module.forward = types.MethodType(fp64_norm,module)
    for layer in decoder.layers:
        layer.register_buffer('v3_test_counter',torch.zeros((),dtype=dtype))
        original_forward = layer.forward
        def instrumented(*args,_layer=layer,_original=original_forward,**kwargs):
            _layer.v3_test_counter.add_(1)
            active = current_v3_lifecycle()
            if fault and active is not None and active.frame[2:] == ('recompute','rv'):
                raise RuntimeError('injected real Qwen RV recompute failure')
            return _original(*args,**kwargs)
        layer.forward = instrumented
    original_forwards = [layer.forward for layer in decoder.layers]
    decoder.gradient_checkpointing = mode != 'off'
    parameters = dict(decoder.named_parameters())
    aliases = {name:(value.detach().clone()+.017).requires_grad_() for name,value in parameters.items()}
    buffers = {name:value.detach().clone() for name,value in decoder.named_buffers()}
    saved = {name:value.detach().clone() for name,value in parameters.items()}
    x = torch.linspace(-.7,.8,2*13*8,dtype=dtype).reshape(2,13,8).requires_grad_()
    position_ids = torch.stack([torch.arange(13)[None,:].expand(2,-1)*m for m in (1,2,3)])
    padding = torch.ones(2,13,dtype=torch.long)
    blocks = (
        replay_api.V3ReplayBlock(0,3,6,(7,9),(1,2),((1,),(2,))),
        replay_api.V3ReplayBlock(0,9,11,(12,13),(3,4),((1,7),(2,8))),
        replay_api.V3ReplayBlock(1,4,7,(8,10),(1,2),((2,),(3,))),
    )
    keys = tuple((b.batch_index,t-1) for b in blocks for t in b.target_positions)
    candidates = tuple(tuple(range(b.start)) for b in blocks for _ in b.target_positions)
    spec = Stage1StochasticContext(42,3,((17,13),(18,13)),sites)
    inventory = {(g,('decoder',i)):(site,) for g in ('F','gain_clean','packed_replay') for i,site in enumerate(sites)}
    life = V3Lifecycle(spec,inventory,has_auxiliary=True)
    calls = []
    hook = decoder.register_forward_hook(lambda *args:calls.append('decoder'))
    contexts = lambda graph:lambda i:life.checkpoint_contexts(graph,('decoder',i))
    try:
        with set_checkpoint_early_stop(mode!='full'), life.activate():
            with life.graph('F'), trace_api.bind_v3_execution(decoder,contexts('F')):
                f = decoder(inputs_embeds=x,attention_mask=padding,position_ids=position_ids,use_cache=False).last_hidden_state
            with life.graph('gain_clean'), trace_api.capture_v3_trace(decoder,keys,candidates) as trace:
                with trace_api.bind_v3_execution(decoder,contexts('gain_clean')):
                    clean = torch.func.functional_call(decoder,(aliases,buffers),(),dict(
                        inputs_embeds=x,attention_mask=padding,position_ids=position_ids,use_cache=False))
            assert trace.inputs_embeds is x and torch.equal(trace.position_ids,position_ids)
            assert all(k.requires_grad and v.requires_grad for k,v in trace.key_values)
            assert all(not row.requires_grad and row.dtype==torch.float32 for row in trace.attention_rows.values())
            with life.graph('packed_replay'):
                result = replay_api.run_v3_packed_replay(decoder,head,trace,blocks,
                    checkpoint_contexts=contexts('packed_replay'),dropout_mask=packed_dropout_mask,
                    parameter_bindings=aliases,buffer_bindings=buffers)
            assert result.query_keys == keys
            assert calls == ['decoder']*3
            # Gradients through attached shared prefix must survive both auxiliary passes.
            losses = result.r_logits.square().sum(),result.r_logits.sin().sum()
            with life.backward('joint_r_kl'):
                grads = torch.vmap(lambda seed:torch.autograd.grad(losses,tuple(v for n,v in aliases.items() if n != "embed_tokens.weight"),
                    grad_outputs=tuple(seed),retain_graph=True,allow_unused=True),randomness='same',out_dims=0)(torch.eye(2,dtype=dtype))
            with life.backward('rv'):
                prefix_grad = torch.autograd.grad(result.rv_logits.sum(),x,retain_graph=True)[0]
                # This is one traversal even though it also requests all aliases.
            with life.backward('ordinary'):
                f.square().sum().backward()
    finally:
        hook.remove()
        assert current_v3_lifecycle() is None
        assert decoder.gradient_checkpointing == (mode != 'off')
        assert all(layer.forward is original and layer.v3_test_counter.item()==0
                   for layer,original in zip(decoder.layers,original_forwards))
        assert all(dict(decoder.named_parameters())[name] is value and torch.equal(value,saved[name])
                   for name,value in parameters.items())
    assert prefix_grad[0,:3].norm() > 0
    # Independent calls never invoke the packed product code.
    decoder.gradient_checkpointing = False
    expected = [[],[]]
    for branch,rv in enumerate((False,True)):
        for block in blocks:
            row = block.batch_index
            one_spec = Stage1StochasticContext(42,3,(spec.rows[row],),sites)
            with activate_stage1_stochastic_context(one_spec):
                out = torch.func.functional_call(decoder,(aliases,buffers),(),dict(
                    inputs_embeds=x[row:row+1],attention_mask=full_mask(13,block,rv,dtype),
                    position_ids=position_ids[:,row:row+1],use_cache=False)).last_hidden_state
            expected[branch].append(head(out[0,[t-1 for t in block.target_positions]]))
    atol,rtol = (1e-9,1e-7) if dtype==torch.float64 else (1e-6,1e-4)
    torch.testing.assert_close(result.r_logits,torch.cat(expected[0]),atol=atol,rtol=rtol)
    torch.testing.assert_close(result.rv_logits,torch.cat(expected[1]),atol=atol,rtol=rtol)
    oracle_prefix = torch.autograd.grad(torch.cat(expected[1]).sum(),x)[0]
    torch.testing.assert_close(prefix_grad,oracle_prefix,atol=1e-6 if native_norm else atol,rtol=1e-4 if native_norm else rtol)


def test_trace_does_not_change_off_path_and_captures_full_causal_attention():
    trace_api,_ = modules()
    decoder,_,_ = replay_qwen(torch.float32,0.)
    decoder.eval()
    x = torch.randn(1,6,8)
    ids = torch.arange(6).view(1,1,-1).expand(3,1,-1)
    before = decoder(inputs_embeds=x,position_ids=ids,use_cache=False,output_attentions=True)
    with trace_api.capture_v3_trace(decoder,((0,4),),((0,2),)) as trace:
        after = decoder(inputs_embeds=x,position_ids=ids,use_cache=False,output_attentions=True)
    assert torch.equal(before.last_hidden_state,after.last_hidden_state)
    expected = torch.stack([a[0,:,4,[0,2]].mean(0) for a in before.attentions]).mean(0)
    torch.testing.assert_close(trace.attention_rows[(0,4)],expected)
    assert trace.attention_rows[(0,4)].sum() < 1


@pytest.mark.parametrize('bad',['flash_attention_2','sliding'])
def test_unsupported_effective_backend_rejected_before_forward(bad):
    trace_api,_ = modules()
    decoder,_,_ = replay_qwen(torch.float32,0.)
    if bad=='sliding': decoder.config.use_sliding_window=True
    else: decoder.config._attn_implementation=bad
    with pytest.raises(ValueError,match='backend|sliding'):
        with trace_api.capture_v3_trace(decoder,(),()):
            pytest.fail('must reject before decoder')


def test_modular_and_generated_decoder_share_opt_in_hook():
    root = Path('transformers/src/transformers/models/qwen2_5_vl')
    for name in ('modeling_qwen2_5_vl.py','modular_qwen2_5_vl.py'):
        text = (root/name).read_text()
        section = text.split('class Qwen2_5_VLModel(')[1].split('\nclass ')[0]
        assert '@v3_decoder_forward' in section


@pytest.mark.parametrize('mode',['early','full'])
def test_native_fp32_norm_under_double_weights_and_exception_restoration(mode):
    test_packed_replay_matches_independent_qwen(torch.float64,.05,mode,native_norm=True)
    with pytest.raises(RuntimeError,match='injected real Qwen RV recompute failure'):
        test_packed_replay_matches_independent_qwen(torch.float64,.05,mode,fault=True)
