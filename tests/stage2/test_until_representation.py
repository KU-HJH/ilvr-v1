"""First-micro diagnostic contracts: actual masks, replay identities, and stop safety."""
from types import SimpleNamespace
import importlib.util
import json
import pytest
import torch


def module():
    assert importlib.util.find_spec('stage2.until_representation') is not None, 'representation observer missing'
    from stage2 import until_representation
    return until_representation


def test_sparse_effective_mask_includes_block_table_and_predicate():
    p = module()
    from torch.nn.attention.flex_attention import create_block_mask
    def mask(b, h, q, k):
        return (q >= k) & ((q + k) % 2 == 0)
    block = create_block_mask(mask, B=1, H=1, Q_LEN=4, KV_LEN=4, device='cpu', BLOCK_SIZE=128)
    actual = p.effective_block_mask(block, batch=0, head=0, queries=4, keys=4)
    assert actual.tolist() == [[True,False,False,False],[False,True,False,False],
                             [True,False,True,False],[False,True,False,True]]
    # Predicate alone is insufficient: an absent sparse block must stay masked.
    block.kv_num_blocks.zero_()
    assert not p.effective_block_mask(block, batch=0, head=0, queries=4, keys=4).any()


def test_dense_effective_mask_records_implicit_causal_and_additive_values():
    p = module()
    assert p.effective_dense_mask(None, True, queries=2, keys=3).tolist() == [
        [True,False,False], [True,True,False]]
    additive = torch.tensor([[[[0.,-2.,float('-inf')],[0.,0.,-4.]]]])
    actual = p.effective_dense_mask(additive, False, queries=2, keys=3)
    assert torch.equal(actual, additive)


def test_trace_hashes_do_not_retain_graph_and_comparison_reports_missing_and_first_difference(tmp_path):
    p = module()
    left, right = tmp_path/'left', tmp_path/'right'
    for folder, value in [(left, 1.), (right, 2.)]:
        trace = p.RepresentationTrace(folder, rank=0)
        original = torch.tensor([value], requires_grad=True)
        rng = torch.get_rng_state().clone()
        trace.record('clean/layer-0/output', original)
        assert torch.equal(rng, torch.get_rng_state()) and original.grad is None
        assert all(not isinstance(v, torch.Tensor) for v in trace.records[0].values())
        trace.close()
    report = p.compare_representations(left, right, ranks=(0,))
    assert not report['all_equal']
    assert report['first_mismatch']['boundary'] == 'clean/layer-0/output'
    assert report['missing_coverage']  # A single hash cannot imply model coverage.


def test_checkpoint_binding_reuses_exact_slot_identity_and_preserves_rng_gradients(tmp_path):
    p = module()
    from torch.utils.checkpoint import checkpoint
    trace = p.RepresentationTrace(tmp_path, rank=0)
    calls = []
    def decoder(x):
        calls.append(trace.invocation)
        y = torch.nn.functional.dropout(x.sin(), p=.25, training=True)
        trace.record('layer-0/output', y)
        return y
    bound = trace.bind_latent_decoder(decoder, slot='block-2/latent-3', position=17)
    x = torch.tensor([.3,.7], requires_grad=True)
    torch.manual_seed(42)
    with torch.utils.checkpoint.set_checkpoint_early_stop(False):
        output = checkpoint(bound, x, use_reentrant=False, preserve_rng_state=True)
    rng = torch.get_rng_state().clone()
    trace.phase = 'preference_backward'
    torch.autograd.grad(output.sum(), x, retain_graph=True)
    assert torch.equal(rng, torch.get_rng_state()) and x.grad is None
    assert calls == ['latent/block-2/latent-3/position-17']*2
    forward = [r for r in trace.records if r['phase']=='forward']
    replay = [r for r in trace.records if r['phase']=='preference_backward']
    assert forward[0]['invocation'] == replay[0]['invocation']
    trace.close()


def test_first_preference_stop_keeps_original_save_guard_and_never_reads_total(tmp_path):
    p = module()
    from stage2.until_alignment_probe import AlignmentObserver
    observer = p.RepresentationObserver.__new__(p.RepresentationObserver)
    observer.rank, observer.micro, observer.update = 0, -1, 0
    observer.engine = SimpleNamespace(device='cpu')
    observer.replay = {'sample_ids':['a']}
    observer.representation_trace = p.RepresentationTrace(tmp_path, rank=0)
    observer.gradients = lambda loss: {'w':loss.detach()}
    saved = []
    observer.save = lambda name,value:saved.append(name)
    observer._reference_output_managers = []
    # CPU has no CUDA RNG: patch only that external runtime boundary.
    from unittest.mock import patch
    with patch.object(torch.cuda,'get_rng_state',return_value=torch.tensor([1],dtype=torch.uint8)):
        with pytest.raises(p.RepresentationInspectionStop):
            observer.observe(torch.tensor(9.),torch.tensor(2.), {'sample_ids':['a']},{'ce':torch.tensor(7.)})
    assert saved == ['micro-0-scores','micro-0-rng','micro-0-preference']
    assert observer.update == 0
    error = ValueError('native prefix tensor differs: micro-0-preference/w')
    observer.micro = -1
    def guard(name,value):
        if name.endswith('preference'):
            raise error
    observer.save = guard
    with patch.object(torch.cuda,'get_rng_state',return_value=torch.tensor([1],dtype=torch.uint8)):
        with pytest.raises(ValueError) as caught:
            observer.observe(torch.tensor(9.),torch.tensor(2.), {'sample_ids':['a']},{'ce':torch.tensor(7.)})
    assert caught.value is error


def test_expected_stop_finalizes_before_peer_rendezvous_and_rethrows_same_error():
    p = module()
    sequence = []
    trace = SimpleNamespace(close=lambda:sequence.append('flush_restore'))
    accelerator = SimpleNamespace(wait_for_everyone=lambda:sequence.append('rendezvous'))
    error = ValueError('native prefix tensor differs: micro-0-preference/w')
    with pytest.raises(ValueError) as caught:
        p.finish_representation_inspection(trace, accelerator, error)
    assert caught.value is error
    assert sequence == ['flush_restore','rendezvous']
    sequence.clear()
    with pytest.raises(RuntimeError):
        p.finish_representation_inspection(trace, accelerator, RuntimeError('unexpected'))
    assert sequence == ['flush_restore']


def test_nonfinite_activations_are_explicit_but_additive_mask_infinity_is_valid(tmp_path):
    p = module()
    trace = p.RepresentationTrace(tmp_path, rank=0)
    trace.record('attention/effective_mask', torch.tensor([0.,float('-inf')]), category='effective_mask')
    trace.record('layer-0/output', torch.tensor([float('nan')]), category='layer_output')
    trace.close()
    assert trace.records[0]['nonfinite'] == {'nan':0,'positive_inf':0,'negative_inf':1}
    summary = json.loads((tmp_path/'representation-summary-rank-0.json').read_text())
    assert any('nonfinite activation' in item for item in summary['missing_coverage'])
    assert not any('nonfinite activation' in item and 'effective_mask' in item for item in summary['missing_coverage'])


def test_scoped_hooks_capture_actual_sdpa_kv_cache_and_restore_on_guard_error(tmp_path):
    p = module()
    assert hasattr(p.RepresentationTrace,'install'), 'actual model/attention boundary capture missing'
    from torch import nn
    from torch.nn import functional as F
    from transformers.cache_utils import DynamicCache
    from torch.utils.checkpoint import checkpoint
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
        def forward(self, hidden, cache=None):
            q = hidden[:,None]
            k,v = (q+1, q+2)
            if cache is not None:
                k,v = cache.update(k,v,0)
            return F.scaled_dot_product_attention(q,k,v,is_causal=False)[:,0]
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
        def forward(self, hidden, cache=None):
            return (hidden+self.self_attn(hidden,cache),)
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(8,4)
            self.layers = nn.ModuleList([Layer()])
        def forward(self, inputs_embeds, position_ids, attention_mask=None, cache=None):
            return SimpleNamespace(last_hidden_state=self.layers[0](inputs_embeds,cache)[0])
    class Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.merger = nn.Linear(4,4)
        def forward(self,x):
            return self.merger(x)
    base = SimpleNamespace(model=Decoder(),visual=Visual(),lm_head=nn.Linear(4,8),
                           stage2_rollout_engine=SimpleNamespace())
    def prefix(language_model,inputs,mask,positions,stats,checkpointing):
        def predict(x):
            return language_model(inputs_embeds=x,position_ids=positions).last_hidden_state[:,-1,:]
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            return checkpoint(predict,inputs,use_reentrant=False,preserve_rng_state=True)
    def rollout(**kw):
        return prefix(kw['language_model'],kw['base_inputs_embeds'][:,:2],None,
                      kw['full_position_ids'][...,:2],None,True)
    source = SimpleNamespace(rollout=SimpleNamespace(_run_latent_prefix=prefix),
        model=SimpleNamespace(ALL_ATTENTION_FUNCTIONS={'flex_attention':lambda *a,**k:None},
                              execute_stage2_sidecar_groups=lambda **kw:None))
    base.stage2_rollout_engine.rollout = rollout
    original_sdpa, original_cache = F.scaled_dot_product_attention, DynamicCache.update
    trace = p.RepresentationTrace(tmp_path,rank=0)
    trace.install(base,source)
    try:
        inputs = base.model.embed_tokens(torch.tensor([[1,2,3]]))
        base.visual(torch.ones(1,4))
        result = base.model(inputs_embeds=inputs,position_ids=torch.arange(3)[None,None],cache=DynamicCache())
        base.lm_head(result.last_hidden_state)
        assert F.scaled_dot_product_attention is not original_sdpa
        assert any(r['category']=='attention_k' for r in trace.records)
        assert any(r['category']=='cache_k' for r in trace.records)
        assert any(r['category']=='effective_mask' for r in trace.records)
        raise ValueError('native prefix tensor differs: micro-0-preference/w')
    except ValueError:
        trace.close()
    assert F.scaled_dot_product_attention is original_sdpa and DynamicCache.update is original_cache
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in base.model.modules())


def test_frozen_native_qwen_checkpoint_capture_preserves_losses_rng_and_gradients(tmp_path):
    p = module()
    from pathlib import Path
    from scripts.verify_until_trainer import TrainerExternalSource, external_modules, bind_external_flex
    from step_preference.equivalence import _tiny_model
    root = Path('/root/workspace/ILVR/.cache/until-temp2-alignment-20260923/reference-source')
    if not root.is_dir():
        pytest.skip('frozen source unavailable')
    with TrainerExternalSource(root) as source:
        external = external_modules(source)
        bind_external_flex(source,external)
        config = _tiny_model().config
        config._attn_implementation = 'sdpa'
        config.stage2_latent_activation_checkpointing = True
        model = external.model.Qwen2_5_VLForConditionalGeneration(config).train()
        ids = torch.tensor([[11,91,90,90,92,12,13]])
        request = dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids,
                       stage2_loss_only=True)
        parameter = model.model.embed_tokens.weight
        torch.manual_seed(42)
        expected = model(**request).loss
        expected_gradient = torch.autograd.grad(expected,parameter)[0]
        expected_rng = torch.get_rng_state().clone()
        trace = p.RepresentationTrace(tmp_path,rank=0)
        original = external.rollout._run_latent_prefix
        trace.install(model,external)
        try:
            torch.manual_seed(42)
            actual = model(**request).loss
            trace.phase = 'preference_backward'
            actual_gradient = torch.autograd.grad(actual,parameter)[0]
            assert torch.equal(expected,actual)
            assert torch.equal(expected_gradient,actual_gradient)
            assert torch.equal(expected_rng,torch.get_rng_state())
            assert parameter.grad is None
        finally:
            trace.close()
        assert external.rollout._run_latent_prefix is original
        summary = json.loads((tmp_path/'representation-summary-rank-0.json').read_text())
        assert summary['within_run_recompute']['compared'] > 0
        assert summary['within_run_recompute']['differences'] == []
        assert not any('unmatched replay' in x for x in summary['missing_coverage'])
        assert {r['invocation'] for r in trace.records if r['phase']=='preference_backward'} == {
            'latent/block-0/latent-0/position-2', 'latent/block-0/latent-1/position-3'}


def test_representation_mode_requires_native_no_offload_and_strict_prefix():
    import scripts.verify_until_trainer as runner
    assert hasattr(runner,'representation_inspection'), 'runner cannot select approved representation mode'
    assert not runner.representation_inspection({'path':'external'})
    valid = {'path':'external','representation_inspection':True,'save_on_cpu':False,'native_prefix_reference':{'ranks':{}}}
    assert runner.representation_inspection(valid)
    for change in ({'path':'aligned'}, {'save_on_cpu':True}, {'native_prefix_reference':None}):
        with pytest.raises(ValueError):
            runner.representation_inspection(dict(valid,**change))


def test_cross_run_agreement_cannot_hide_within_run_recompute_mismatch(tmp_path):
    p = module()
    for name in ('left','right'):
        directory = tmp_path/name
        directory.mkdir()
        (directory/'representations-rank-0.jsonl').write_text('')
        (directory/'representation-summary-rank-0.json').write_text(json.dumps({
            'missing_coverage':[], 'within_run_recompute':{'compared':1,
                'differences':[{'invocation':'latent/block-0/latent-0/position-2','boundary':'layer-0/output'}]}}))
    report = p.compare_representations(tmp_path/'left',tmp_path/'right',ranks=(0,))
    assert not report['all_equal']
    assert report['within_run_recompute_differences']
