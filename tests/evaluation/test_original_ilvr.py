"""Causal contracts from ILVR 13a67ed; no GPU or pretrained weights needed."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import Qwen2_5_VLConfig

ROOT = Path(__file__).resolve().parents[2]
START, PAD, END = 151666, 151665, 151667


@pytest.fixture(autouse=True)
def restore_tf32():
    previous=torch.backends.cuda.matmul.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32=previous


def restored():
    spec = importlib.util.find_spec('evaluation.original_ilvr')
    assert spec is not None, 'the original ILVR evaluation runtime is missing'
    return importlib.import_module('evaluation.original_ilvr')


class ScriptedHead(nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens
        self.calls = 0
        self.hidden = []

    def forward(self, hidden):
        self.hidden.append(hidden.detach().clone())
        token = self.tokens[min(self.calls, len(self.tokens)-1)]
        self.calls += 1
        logits = hidden.new_full((*hidden.shape[:-1], END+1), -100.)
        logits[..., token] = 100.
        return logits


def tiny_original(tokens, *, use_cache=True):
    api = restored()
    config = Qwen2_5_VLConfig(vocab_size=END+1, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        vision_config=dict(depth=1, hidden_size=8, intermediate_size=16,
            num_heads=2, patch_size=2, spatial_merge_size=1, temporal_patch_size=1,
            window_size=2, out_hidden_size=8, fullatt_block_indexes=[0]),
        image_token_id=80, vision_start_token_id=81, video_token_id=82,
        stage='stage1', latent_size=2, latent_token_id=PAD,
        latent_pad_token_id=PAD, latent_start_token_id=START, latent_end_token_id=END,
        attn_implementation='eager', use_cache=use_cache,
        eos_token_id=[3,4], pad_token_id=0,
        rope_scaling={'type':'mrope', 'mrope_section':[1,1,0]})
    torch.manual_seed(42)
    model = api.original_model_class()(config).eval()
    model.lm_head = ScriptedHead(tokens)
    calls = []
    def record(_module, _args, kwargs):
        calls.append({key: value.detach().clone() if torch.is_tensor(value) else value
            for key,value in kwargs.items() if key in
            ('inputs_embeds','position_ids','attention_mask','cache_position','use_cache')})
    model.model.register_forward_pre_hook(record, with_kwargs=True)
    return api, model, calls


def generate(api, model, *, cap=20, ids=(7,8), image=False):
    inputs = dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1,len(ids),dtype=torch.long))
    if image:
        inputs.update(pixel_values=torch.arange(12,dtype=torch.float).reshape(1,12)/12,
            image_grid_thw=torch.tensor([[1,1,1]]))
    with torch.inference_mode():
        return api.generate_original(model, inputs, dict(max_new_tokens=cap,
            temperature=0., top_p=1., do_sample=False))


def test_mechanical_snapshots_match_independent_original_git_hashes():
    api = restored()
    generation = (ROOT/api.GENERATION_SNAPSHOT).read_text()
    generation = generation.replace('from .utils import GenerationMixin as _CurrentGenerationMixin\n','')
    generation = generation.replace('class GenerationMixin(_CurrentGenerationMixin):','class GenerationMixin:')
    assert hashlib.sha256(generation.encode()).hexdigest() == 'acc203e30739d40abd36f96b93c8110369a9742717f4db050075387e3a241863'
    modeling = (ROOT/api.MODELING_SNAPSHOT).read_text().replace(
        'from ...generation._ilvr_original_utils import GenerationMixin',
        'from ...generation import GenerationMixin')
    assert hashlib.sha256(modeling.encode()).hexdigest() == '7961e7841bb5aae550e42ccb13e2de8e4802057a0e8878baf5fe224f97e37002'
    cls = api.original_model_class()
    for method in ('generate','_sample','_update_model_kwargs_for_generation','_get_initial_cache_position'):
        assert getattr(cls,method).__module__ == 'transformers.generation._ilvr_original_utils'
    assert cls.prepare_inputs_for_generation.__module__.endswith('._ilvr_original_modeling')


def test_native_hidden_handoff_cache_mask_positions_and_per_call_reset():
    api, model, calls = tiny_original([START,5,5,5,9,START,5,5,5,3])
    result = generate(api,model)
    assert result.sequences[0,2:].tolist() == [START,PAD,PAD,END,9,START,PAD,PAD,END,3]
    assert (result.latent_start_count,result.latent_block_count,result.stop_reason) == (2,2,'eos')
    assert [c['cache_position'].tolist() for c in calls] == [[0,1]]+[[n] for n in range(2,11)]
    assert [c['attention_mask'].shape[-1] for c in calls] == list(range(2,12))
    assert [c['position_ids'][0,0].tolist() for c in calls] == [[0,1]]+[[n] for n in range(2,11)]
    torch.testing.assert_close(calls[2]['inputs_embeds'],model.lm_head.hidden[1],rtol=0,atol=0)
    torch.testing.assert_close(calls[3]['inputs_embeds'],model.lm_head.hidden[2],rtol=0,atol=0)
    torch.testing.assert_close(calls[4]['inputs_embeds'],model.get_input_embeddings()(torch.tensor([[END]])),rtol=0,atol=0)
    model.lm_head = ScriptedHead([START,5,5,5,4])
    calls.clear()
    again = generate(api,model)
    assert again.sequences[0,2:].tolist() == [START,PAD,PAD,END,4]
    assert calls[0]['cache_position'].tolist() == [0,1]


@pytest.mark.parametrize('cap,expected',[(1,[START]),(2,[START,PAD]),(3,[START,PAD,PAD]),(4,[START,PAD,PAD,END])])
def test_native_per_token_cap_keeps_incomplete_blocks(cap,expected):
    api,model,_ = tiny_original([START,5])
    result = generate(api,model,cap=cap)
    assert result.sequences[0,2:].tolist() == expected
    assert result.stop_reason == 'max_new_tokens'


def test_native_eos_and_image_prefill_run_real_original_decoder():
    api,model,calls = tiny_original([9,4])
    result = generate(api,model,ids=(7,81,80,8),image=True)
    assert result.sequences[0,4:].tolist() == [9,4]
    assert calls[0]['position_ids'].tolist() == [[[0,1,2,3]]]*3
    assert calls[1]['cache_position'].tolist() == [4]
    assert result.stop_reason == 'eos'


def test_source_no_cache_latent_limitation_is_not_silently_repaired():
    api,model,calls = tiny_original([START,5],use_cache=False)
    with pytest.raises(AssertionError):
        generate(api,model)
    assert [c['cache_position'].tolist() for c in calls] == [[0,1],[0,1,2]]
    assert model.generation_config.use_cache is False


def test_original_backend_route_retries_whole_model_eager_and_preserves_config():
    api=restored()
    calls=[]
    class Loaded:
        config=type('Config',(),{'use_cache':False})()
        generation_config=object()
        def eval(self): return self
    loaded=Loaded()
    def loader(source,**kwargs):
        calls.append((source,kwargs))
        if len(calls)==1: raise RuntimeError('FA2 unavailable')
        return loaded
    assert api.load_original_model('same-artifact','cache',load_model=loader) is loaded
    assert [kw['attn_implementation'] for _,kw in calls] == ['flash_attention_2','eager']
    assert all('config' not in kw and 'use_cache' not in kw for _,kw in calls)
    assert loaded.config.use_cache is False


def test_original_scoring_keeps_first_assistant_full_decode_and_empty_gold_quirks():
    spec=importlib.util.find_spec('evaluation.original_scoring')
    assert spec is not None, 'original offline scoring is missing'
    score=importlib.import_module('evaluation.original_scoring')
    raw='<|im_start|>user\nQ<|im_end|><|im_start|>assistant\nfinal answer: **2**.<|im_end|>'
    assert score.extract_final_answer(score.extract_assistant_content(raw)) == '2'
    assert score.normalize_for_match('2.0') == '2'
    assert score.normalize_for_match('a') == 'Yes'
    assert score.score_original('', {})['match'] is True
    assert score.score_original('', {'original_final_answer':None})['match'] is True
    assert score.extract_final_answer('answer: no\nfinal answer is: yes') == 'yes'
    # Legacy regex intentionally takes the first ChatML assistant, including in a prompt.
    raw='<|im_start|>assistant\nanswer: no<|im_end|><|im_start|>assistant\nanswer: yes'
    assert score.extract_assistant_content(raw) == 'answer: no'


def test_original_generate_preserves_outer_peft_generate_dispatch():
    from types import SimpleNamespace
    api,inner,_=tiny_original([3])
    calls=[]
    def outer_generate(**kwargs):
        calls.append(True)
        return inner.generate(**kwargs)
    wrapper=SimpleNamespace(module=inner,generate=outer_generate)
    with torch.inference_mode():
        result=api.generate_original(wrapper,dict(input_ids=torch.tensor([[7,8]])),dict(max_new_tokens=2,do_sample=False))
    assert calls==[True]
    assert result.sequences[0].tolist()==[7,8,3]


def test_native_multiaxis_image_mrope_positions_and_cache_delta():
    api,model,calls=tiny_original([9,4])
    ids=torch.tensor([[7,81,80,80,80,80,8]])
    inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
        pixel_values=torch.arange(48,dtype=torch.float).reshape(4,12)/48,
        image_grid_thw=torch.tensor([[1,2,2]]))
    with torch.inference_mode():
        result=api.generate_original(model,inputs,dict(max_new_tokens=3,do_sample=False))
    assert result.sequences[0,7:].tolist()==[9,4]
    assert calls[0]['position_ids'][:,0].tolist()==[
        [0,1,2,2,2,2,4],[0,1,2,2,3,3,4],[0,1,2,3,2,3,4]]
    assert calls[1]['cache_position'].tolist()==[7]
    assert calls[1]['position_ids'][:,0].tolist()==[[5],[5],[5]]


def test_native_repetition_penalty_inherits_loaded_generation_config():
    api,model,_=tiny_original([3])
    class Head(nn.Module):
        def forward(self,hidden):
            logits=hidden.new_full((*hidden.shape[:-1],END+1),-100.)
            logits[...,7]=10.
            logits[...,9]=8.
            return logits
    model.lm_head=Head()
    model.generation_config.repetition_penalty=2.
    result=generate(api,model,cap=1)
    assert result.sequences[0,2:].tolist()==[9]
    assert model.generation_config.repetition_penalty==2.
