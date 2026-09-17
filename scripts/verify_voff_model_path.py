#!/usr/bin/env python
"""Read-only tiny Qwen source differential. Prints metadata only, never model state."""
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib
import json
from pathlib import Path
import types
import platform
if platform.python_version().endswith('+'):
    _version = platform.python_version()
    platform.python_version = lambda: _version.rstrip('+')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)):(p.stat().st_size,p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def comparison(a,b):
    import torch
    if a is None or b is None:
        return {'pass':a is None and b is None, 'connected':False}
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {'pass':finite and torch.allclose(a,b,atol=1e-6,rtol=1e-5),
            'connected':True,'max_abs_error':float((a-b).abs().max()) if finite else None,
            'relative_l2':float((a.double()-b.double()).norm()/(a.double().norm()+1e-12)) if finite else None}


def run():
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLConfig,Qwen2_5_VLForConditionalGeneration
    root = Path('/raid/workspace/ILVR/ILVR/.worktrees/ours1/source')
    before = inventory(root)
    folder = root/'transformers/src/transformers/models/qwen2_5_vl'
    # Same package depth preserves upstream relative imports; framework is shared.
    name = 'transformers.models.voff_supplied_qwen'
    package = types.ModuleType(name)
    package.__path__ = [str(folder)]
    package.__package__ = name
    sys.modules[name] = package
    supplied = importlib.import_module(name+'.modeling_qwen2_5_vl')
    step_name = 'voff_supplied_step'
    step_package = types.ModuleType(step_name)
    step_package.__path__ = [str(root/'src/step_preference')]
    step_package.__package__ = step_name
    sys.modules[step_name] = step_package
    packing = importlib.import_module(step_name+'.packing')
    external_loss = importlib.import_module(step_name+'.dpo_loss')
    from step_preference.spans import ReasoningStepSpan
    from step_preference.sidecar import build_context_sidecar_plan
    from step_preference.dpo_loss import context_preference_loss
    tokenizer = AutoProcessor.from_pretrained(
        'checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431',
        local_files_only=True).tokenizer
    pad,start,end = [tokenizer.convert_tokens_to_ids(t) for t in ('<|latent_pad|>','<|latent_start|>','<|latent_end|>')]
    image,vision_start,video = [tokenizer.convert_tokens_to_ids(t) for t in ('<|image_pad|>','<|vision_start|>','<|video_pad|>')]
    config = dict(vocab_size=len(tokenizer),hidden_size=8,intermediate_size=16,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
        vision_config=dict(depth=1,hidden_size=8,intermediate_size=16,num_heads=2,
            patch_size=2,spatial_merge_size=1,temporal_patch_size=1,window_size=2,
            out_hidden_size=8,fullatt_block_indexes=[0]),
        image_token_id=image,vision_start_token_id=vision_start,video_token_id=video,
        stage='stage2',latent_size=2,latent_pad_token_id=pad,latent_start_token_id=start,
        latent_end_token_id=end,supervise_latent_boundaries=True,use_cache=False,
        attn_implementation='eager',attention_dropout=0.,stage2_latent_activation_checkpointing=True,
        rope_scaling={'type':'mrope','mrope_section':[1,1,0]})
    torch.set_num_threads(2)
    torch.manual_seed(42)
    current = Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(**config)).float().train()
    external = supplied.Qwen2_5_VLForConditionalGeneration(supplied.Qwen2_5_VLConfig(**config)).float().train()
    external.load_state_dict(current.state_dict(),strict=True)
    cases = []
    for blocks in (1,2):
        tokens = [1,vision_start,image,image,image,image,2]
        for _ in range(blocks):
            tokens += [start,pad,pad,end,3,4]
        tokens += [5,6]
        ids=torch.tensor([tokens]);labels=ids.clone();labels[:,:7]=-100;labels[ids==pad]=-100
        pixels=torch.linspace(-1,1,48).reshape(4,12)
        spans=[]
        for index in range(blocks):
            boundary=7+6*index
            spans.append(ReasoningStepSpan(index,6 if index==0 else boundary-2,boundary,
                boundary+1,boundary+3,boundary+4,boundary+6 if index<blocks-1 else len(tokens),None))
        plan=build_context_sidecar_plan(clean_input_ids=ids,attention_mask=torch.ones_like(ids),spans=spans,visual_only=True)
        items=[types.SimpleNamespace(sample_id='tiny',span=span,
            chosen_predictor_positions=torch.arange(span.target_text_start-1,span.target_text_end-1),
            chosen_target_ids=ids[0,span.target_text_start:span.target_text_end]) for span in spans]
        external_plan=packing.build_context_sidecar_execution_plan(clean_input_ids=ids,items=items,modalities='visual')
        score_mask=torch.zeros_like(ids,dtype=torch.bool)
        for item in items:
            score_mask[0,item.chosen_predictor_positions]=True
        outputs=[];gradients=[];pixel_gradients=[];preferences=[];clean_values=[];off_values=[];preference_nonzero=[]
        for model in (current,external):
            model.zero_grad(set_to_none=True)
            x=pixels.clone().requires_grad_()
            options=(dict(stage2_loss_only=True,stage2_logit_positions=torch.tensor(plan.root_selected_positions),
                stage2_shared_prefix_inputs=dict(plan=plan,backend='dense',query_budget=256,prefix_mode='clean_prefix',
                                                branch_mode='independent',block_original_image=False))
                if model is current else dict(stage2_loss_only=True,stage2_score_token_mask=score_mask,stage2_sidecar_plan=external_plan))
            output=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,
                         pixel_values=x,image_grid_thw=torch.tensor([[1,2,2]]),**options)
            rows=output.stage2_selected_logits if model is current else output.stage2_scored_logits
            clean=[];off=[];offset=0
            for item in items:
                targets=item.chosen_target_ids
                count=targets.numel()
                clean.append(rows[0,offset:offset+count].float().log_softmax(-1).gather(-1,targets[:,None]).mean())
                if model is current:
                    off.append(next(r['mean_logp'] for r in output.stage2_sidecar_results if r['step_id']==item.span.step_id))
                else:
                    logits=output.stage2_sidecar_logits.by_branch_key[f'tiny:{item.span.step_id}:visual:chosen_entropy']
                    off.append(logits.reshape(-1,logits.shape[-1]).float().log_softmax(-1).gather(-1,targets[:,None]).mean())
                offset+=count
            clean=torch.stack(clean);off=torch.stack(off)
            preference=(context_preference_loss(clean,off,beta=.1,margin=0.) if model is current
                        else external_loss.context_preference_loss(clean,off,beta=.1,margin=0.).loss)
            pg=torch.autograd.grad(preference,[p for p in model.parameters() if p.requires_grad],retain_graph=True,allow_unused=True)
            preference_nonzero.append(any(g is not None and bool(g.abs().max()>0) for g in pg))
            total=output.loss+preference
            total.backward()
            outputs.append(total.detach());preferences.append(preference.detach())
            clean_values.append(clean.detach());off_values.append(off.detach())
            gradients.append({n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()})
            pixel_gradients.append(x.grad)
        if gradients[0].keys()!=gradients[1].keys():
            raise RuntimeError('supplied/current model parameter names differ')
        comparisons={n:comparison(gradients[0][n],gradients[1][n]) for n in gradients[0]}
        cases.append({'latent_blocks':blocks,'input_ids':tokens,'loss':comparison(*outputs),
            'parameter_gradients':comparisons,'pixel_gradient':comparison(*pixel_gradients),
            'preference_loss':comparison(*preferences),'clean_scores':comparison(*clean_values),
            'visual_off_scores':comparison(*off_values),'preference_gradient_nonzero':all(preference_nonzero),
            'connected_parameter_count':sum(g['connected'] for g in comparisons.values())})
    source_hashes={str(p.relative_to(root)):sha(p) for p in list(folder.glob('*.py'))+[root/'src/step_preference/packing.py',root/'src/step_preference/dpo_loss.py']}
    shared={n:sha(m.__file__) for n,m in tuple(sys.modules.items()) if n.startswith('transformers')
            and getattr(m,'__file__',None) and Path(m.__file__).is_file()
            and not Path(m.__file__).is_relative_to(root)}
    return {'scope':'tiny FP32 real Qwen image/latent/CE+V-off model sidecar forward and all parameter gradients; shared current framework; no DoRA/production-scale equivalence',
        'config':config,'cases':cases,'source_hashes':source_hashes,'shared_dependency_hashes':shared,
        'external_files_unchanged':before==inventory(root),'dropout':0.,'seed':42,
        'full_run_authorized':False}


if __name__=='__main__':
    print(json.dumps(run(),allow_nan=False))
