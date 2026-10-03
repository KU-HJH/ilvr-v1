"""Single-checkpoint Clean/Cross evaluation on held-out GT trajectories."""
import argparse
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
from threading import Event
from types import SimpleNamespace

import torch
from PIL import Image

from .latent_causal_intervention import _atomic_write_text
from .latent_causal_intervention import (
    Block, extract_blocks, make_replay, score_replay, select_donors,
    summarize, verify_identity, verify_pair,
)


def stable_sample_id(row):
    """Use the current canonical digest while retaining the source's explicit-ID validation."""
    from step_preference.spans import canonical_sample_id
    explicit=row.get('sample_id')
    if explicit is not None and (not isinstance(explicit,str) or not explicit.strip()):
        raise ValueError('sample_id must be a non-empty string')
    return canonical_sample_id(row)


def infer_task(row):
    if isinstance(row.get('task'),str) and row['task'].strip():
        return row['task'].strip()
    tasks = set()
    images = row.get('image_input',[])
    if isinstance(images,str):
        images = [images]
    if not isinstance(images,list) or not all(isinstance(p,str) for p in images):
        raise ValueError('task requires valid image_input paths')
    for image in images:
        parts = Path(image).parts
        if 'images_comt' not in parts:
            raise ValueError('task cannot be inferred from image_input')
        index=parts.index('images_comt')
        if index+2 >= len(parts):
            raise ValueError('task directory missing')
        tasks.add(parts[index+1])
    if len(tasks) != 1 or not tasks.issubset({'creation','deletion','update','selection'}):
        raise ValueError('unknown or inconsistent task')
    return tasks.pop()


def image_digest(path, cache):
    """Hash decoded original-image RGB content once per canonical path."""
    path=Path(path).resolve()
    if path not in cache:
        with Image.open(path) as image:
            rgb=image.convert('RGB')
            digest=hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode())
            digest.update(rgb.tobytes())
        cache[path]=digest.hexdigest()
    return cache[path]


def skip_row(sample_id, task, block_index, reason, K=None):
    return dict(sample_id=sample_id,task=task,block_index=block_index,
        donor_sample_id=None,K=K,target_token_count=0,status='skip',skip_reason=reason)


def prepare_sample(row, processor, token_ids, K, image_root, max_tokens, digest_cache):
    from .data import (prepare_stage2_example, _load_original_images,
                       find_unique_subsequence_end)
    sid=stable_sample_id(row)
    task=None
    try:
        task=infer_task(row)
        example=prepare_stage2_example(row,image_root=image_root,latent_size=K)
    except (ValueError,FileNotFoundError,KeyError) as error:
        return None,[skip_row(sid,task,None,f'invalid_gt_or_metadata:{type(error).__name__}')]
    images=_load_original_images(example.original_image_paths)
    try:
        batch=processor(text=[processor.apply_chat_template(example.conversation,
            tokenize=False,add_generation_prompt=False)],images=[images],
            padding=False,truncation=False,return_tensors='pt')
    finally:
        for image in images:
            image.close()
    if {'pixel_values_latent','image_grid_thw_latent'}.intersection(batch):
        raise ValueError('helper-image features entered evaluator')
    ids=batch['input_ids']
    if ids.shape[1] > max_tokens:
        return None,[skip_row(sid,task,m,'context_length_exceeded',K) for m in range(example.latent_block_count)] or [skip_row(sid,task,None,'context_length_exceeded',K)]
    try:
        def token_pattern(text):
            return processor.tokenizer(text,add_special_tokens=False,return_tensors='pt')['input_ids'][0]
        start=find_unique_subsequence_end(ids[0],token_pattern('<|im_start|>assistant'))
        end_pattern=token_pattern('<|im_end|>')
        end=start+find_unique_subsequence_end(ids[0,start:],end_pattern)-end_pattern.numel()
        blocks,skipped=extract_blocks(ids,start,end,token_ids,K)
        if len(blocks)+len(skipped) != example.latent_block_count:
            raise ValueError('processor output lost GT blocks')
        forbidden={token_ids.pad_token_id,token_ids.start_token_id,token_ids.end_token_id,
                   *end_pattern.tolist()}
        if processor.tokenizer.pad_token_id is not None:
            forbidden.add(processor.tokenizer.pad_token_id)
        kept=[]
        for block in blocks:
            target=ids[0,block.target_start:block.target_end]
            if any(int(t) in forbidden for t in target) or not batch['attention_mask'][0,block.target_start:block.target_end].all():
                skipped.append(dict(block_index=block.block_index,skip_reason='control_or_padding_in_target'))
            else:
                kept.append(block)
        blocks=kept
    except ValueError:
        return None,[skip_row(sid,task,None,'invalid_tokenized_structure',K)]
    excluded=[skip_row(sid,task,s['block_index'],s['skip_reason'],K) for s in skipped]
    if not blocks:
        return None,excluded or [skip_row(sid,task,None,'no_latent_blocks',K)]
    fingerprints=[image_digest(p,digest_cache) for p in example.original_image_paths]
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import parse_latent_spans
    spans=parse_latent_spans(ids[0],token_ids=token_ids,latent_size=K)
    latent_blocks=[Block(i,tuple(s.latent_positions),s.end_position+1,
        spans[i+1].start_position if i+1<len(spans) else end) for i,s in enumerate(spans)]
    # No training collator or item-wise retokenization: positions are processor output.
    inputs={k:v for k,v in batch.items() if k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
    return SimpleNamespace(sample_id=sid,task=task,image_digests=fingerprints,
        model_inputs=inputs,blocks=blocks,latent_blocks=latent_blocks),excluded


@torch.inference_mode()
def capture_sample(qwen, sample, *, include_top1=False):
    if qwen.training:
        raise ValueError('normal trajectory collection requires eval mode')
    device=next(qwen.parameters()).device
    inputs={k:v.to(device) for k,v in sample.model_inputs.items()}
    predictors=torch.tensor([p for block in sample.blocks
        for p in range(block.target_start-1,block.target_end-1)],device=device)
    # Native rollout calls the decoder on prefixes, then exactly once on the full
    # final inputs. Capture that actual call instead of reconstructing embeddings.
    observed=[]
    def capture(module,args,kwargs):
        if kwargs['inputs_embeds'].shape[1]==inputs['input_ids'].shape[1]:
            if kwargs.get('use_cache') is not False or kwargs.get('generate_mode') is not True:
                raise ValueError('native final decoder must be cacheless')
            observed.append({k:kwargs[k].detach().clone() for k in
                ('inputs_embeds','attention_mask','position_ids')})
    hook=qwen.model.register_forward_pre_hook(capture,with_kwargs=True)
    try:
        native=qwen(**inputs,use_cache=False,return_dict=True,
            stage2_logit_positions=predictors,stage2_selective_logits_only=True)
    finally:
        hook.remove()
    if len(observed)!=1:
        raise ValueError('native final decoder capture must run exactly once')
    final=observed[0]
    if not torch.equal(final['inputs_embeds'],native.inputs_embeds):
        raise ValueError('native returned embeddings differ from actual decoder input')
    if not torch.equal(final['attention_mask'],inputs['attention_mask']):
        raise ValueError('native decoder changed attention mask')
    positions=final['position_ids']
    embeddings=final['inputs_embeds'].cpu()
    all_blocks=getattr(sample,'latent_blocks',sample.blocks)
    normal_logps, normal_top1, cursor = {}, {}, 0
    bank={b.block_index:embeddings[0,list(b.latent_positions)].clone() for b in all_blocks}
    for block in sample.blocks:
        n=block.target_end-block.target_start
        targets=inputs['input_ids'][0,block.target_start:block.target_end]
        logits=native.stage2_selected_logits[0,cursor:cursor+n].float()
        logps=logits.log_softmax(-1).gather(-1,targets[:,None]).squeeze(-1)
        if not torch.isfinite(logps).all():
            raise ValueError('nonfinite normal trajectory scores')
        normal_logps[block.block_index]=logps.cpu()
        if include_top1:
            normal_top1[block.block_index]=logits.argmax(-1).cpu()
        cursor+=n
    if cursor != native.stage2_selected_logits.shape[1]:
        raise ValueError('unassigned native target logits')
    return SimpleNamespace(sample_id=sample.sample_id,task=sample.task,
        image_digests=sample.image_digests,blocks=sample.blocks,
        input_ids=inputs['input_ids'].cpu(),attention_mask=inputs['attention_mask'].cpu(),
        position_ids=positions.cpu(),embeddings=embeddings,bank=bank,native_logps=normal_logps,
        native_top1=normal_top1,latent_blocks=all_blocks)


@torch.inference_mode()
def evaluate_collected(qwen, samples, mapping, logp_atol, nll_atol):
    lookup={s.sample_id:s for s in samples}
    rows=[]
    witness={'passed':True,'identity_checked_blocks':0,'max_logp_error':0.,'max_nll_error':0.,
        'identity_logp_atol':logp_atol,'identity_nll_atol':nll_atol,
        'cache':'disabled; entire prefix through R recomputed per condition',
        'cross_input_invariants_checked':0}
    for sample in samples:
        for block in sample.blocks:
            donor_id=mapping[(sample.sample_id,block.block_index)]
            if donor_id is None:
                rows.append(skip_row(sample.sample_id,sample.task,block.block_index,'no_donor',block.K))
                continue
            donor_sample=lookup[donor_id]
            if donor_id==sample.sample_id or donor_sample.task!=sample.task or set(sample.image_digests).intersection(donor_sample.image_digests):
                raise ValueError('donor eligibility invariant failed')
            donor=donor_sample.bank[block.block_index]
            clean=make_replay(sample,block)
            cross=make_replay(sample,block,donor)
            verify_pair(clean,cross,block,donor)
            a=score_replay(qwen,clean)
            identity=verify_identity(sample.native_logps[block.block_index],a,logp_atol,nll_atol)
            b=score_replay(qwen,cross)
            witness['identity_checked_blocks']+=1
            witness['cross_input_invariants_checked']+=1
            witness['max_logp_error']=max(witness['max_logp_error'],identity['max_logp_error'])
            witness['max_nll_error']=max(witness['max_nll_error'],identity['nll_error'])
            nll_a,nll_b=-float(a.mean()),-float(b.mean())
            rows.append(dict(sample_id=sample.sample_id,task=sample.task,
                block_index=block.block_index,donor_sample_id=donor_id,K=block.K,
                target_token_count=a.numel(),nll_clean=nll_a,nll_cross=nll_b,
                delta_nll=nll_b-nll_a,status='ok',identity=identity))
        print(json.dumps({'phase':'replay','sample_id':sample.sample_id,'completed_blocks':len(rows)}),flush=True)
    return rows,witness


def publish(output, rows, metadata, witness):
    valid=sum(row['status']=='ok' for row in rows)
    if not witness.get('passed') or witness.get('identity_checked_blocks')!=valid or witness.get('cross_input_invariants_checked')!=valid:
        raise ValueError('validation witness incomplete; refusing to publish results')
    summary={**metadata,**summarize(rows,metadata['seed']),'validation':witness,'status':'complete'}
    summary['evaluated_sample_count']=len({r['sample_id'] for r in rows})
    summary['excluded_units']=sum(r['status']=='skip' for r in rows)
    output=Path(output)
    # Publication reuses validation evidence; never reruns model/checkpoint validators.
    _atomic_write_text(output/'per_block.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    _atomic_write_text(output/'summary.json',json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','base-model-dir','data','image-root','output-dir'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--data-split',choices=('test','validation'),required=True,
        help='Declare a held-out GT trajectory split; this does not audit training-set overlap')
    parser.add_argument('--max-samples',type=int,required=True)
    parser.add_argument('--max-input-tokens',type=int,default=32768)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--analysis-suite',choices=('clean_cross','full'),default='full')
    parser.add_argument('--dtype',choices=('float32',),default='float32')
    parser.add_argument('--vision-dtype',choices=('bfloat16',),default='bfloat16',
        help='Optional vision-only dtype; latent recurrence and text retain --dtype')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--devices',nargs=2,metavar=('FIRST','SECOND'),default=None,
        help='Use two distinct visible CUDA devices for parallel full-suite collection and replay')
    parser.add_argument('--identity-logp-atol',type=float,required=True,
        help='Predeclare maximum per-token absolute log-probability error (nats)')
    parser.add_argument('--identity-nll-atol',type=float,required=True,
        help='Predeclare maximum per-block absolute NLL error (nats/token)')
    return parser


def model_load_options(dtype, device, vision_dtype=None):
    return {'torch_dtype':getattr(torch,dtype),'device_map':device}


def load_analysis_qwen(source, **kwargs):
    """Inject the existing mixed-attention constructor into the frozen PEFT loader."""
    from .attention_backend import load_qwen2_5_vl_with_attention
    provenance={k:kwargs.pop(k) for k in tuple(kwargs)
        if k in ('local_files_only','revision','cache_dir','token')}
    return load_qwen2_5_vl_with_attention(source,
        attention_implementation='sdpa',vision_attention_implementation='flash_attention_2',
        config_load_kwargs=provenance,model_load_kwargs=dict(provenance,**kwargs))


def validate_analysis_runtime(qwen):
    from .attention_backend import validate_qwen2_5_vl_attention_backends
    validate_qwen2_5_vl_attention_backends(qwen,attention_implementation='sdpa',
        vision_attention_implementation='flash_attention_2')
    if any(p.dtype!=torch.float32 for module in (qwen.model,qwen.lm_head) for p in module.parameters()):
        raise ValueError('analysis decoder must be FP32')
    if any(p.dtype!=torch.bfloat16 for p in qwen.visual.parameters()):
        raise ValueError('analysis vision must be BF16')
    if next(qwen.parameters()).device.type!='cuda':
        raise ValueError('analysis requires CUDA')

def apply_vision_dtype(qwen, vision_dtype):
    if vision_dtype is not None:
        qwen.visual.to(dtype=getattr(torch,vision_dtype))


def validate_parallel_devices(devices):
    """Reject a malformed or unavailable two-GPU request before model loading."""
    if len(devices)!=2:
        raise ValueError('--devices requires exactly two CUDA devices')
    parsed=[]
    for device in devices:
        try:
            value=torch.device(device)
        except (TypeError,RuntimeError) as error:
            raise ValueError(f'invalid CUDA device: {device}') from error
        if value.type!='cuda' or value.index is None:
            raise ValueError('--devices requires explicit CUDA indices')
        parsed.append(value.index)
    if parsed[0]==parsed[1]:
        raise ValueError('--devices requires distinct CUDA devices')
    count=torch.cuda.device_count()
    if any(index>=count for index in parsed):
        raise ValueError(f'unavailable CUDA device in {tuple(devices)}; visible device count={count}')
    return tuple(f'cuda:{index}' for index in parsed)


def replicate_qwen_for_device(qwen, device):
    """Copy the already authenticated, frozen model without reopening weights."""
    if qwen.training or any(p.requires_grad for p in qwen.parameters()):
        raise ValueError('model replica requires frozen eval source')
    if any(hasattr(module,'_hf_hook') for module in qwen.modules()):
        raise ValueError('accelerate dispatch hooks prevent safe device replication')
    replica=copy.deepcopy(qwen).to(device)
    replica.eval().requires_grad_(False)
    target=torch.device(device)
    tensors=list(replica.parameters())+list(replica.buffers())
    if any(t.device!=target for t in tensors):
        raise ValueError('model replica has tensors on the wrong device')
    original=list(qwen.parameters())+list(qwen.buffers())
    if any(a.data_ptr()==b.data_ptr() for a,b in zip(original,tensors) if a.numel()):
        raise ValueError('model replica shares mutable storage with source')
    if hasattr(replica,'hf_device_map'):
        replica.hf_device_map={'':str(target)}
    return replica


def _merge_suite_witnesses(witnesses):
    if not witnesses:
        raise ValueError('no suite workers completed')
    merged=dict(witnesses[0])
    for witness in witnesses[1:]:
        for key in ('identity_logp_atol','identity_nll_atol','cache'):
            if witness[key]!=merged[key]:
                raise ValueError(f'inconsistent worker validation witness: {key}')
        merged['passed']=merged['passed'] and witness['passed']
        for key in ('identity_checked_blocks','condition_inputs_checked','native_replay_top1_flips'):
            merged[key]+=witness[key]
        for key in ('max_logp_error','max_nll_error'):
            merged[key]=max(merged[key],witness[key])
    return merged


def _run_pair(work):
    stop=Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(work,index,stop):index for index in range(2)}
        results=[None,None]
        for future in as_completed(futures):
            try:
                results[futures[future]]=future.result()
            except BaseException:
                stop.set()
                for other in futures:
                    other.cancel()
                raise
    return results


def run_parallel_suite(models, devices, prepared, mapping, permutations,
                       logp_atol, nll_atol, start_id, end_id):
    """Run collection and replay on separate model streams with one frozen donor bank."""
    from .latent_intervention_suite import CONDITIONS, evaluate_suite
    if len(models)!=2 or len(devices)!=2 or len(prepared)<2:
        raise ValueError('two model streams require at least two prepared samples')
    shards=[list(enumerate(prepared))[index::2] for index in range(2)]

    def collect(index,stop):
        device=devices[index]
        guard=torch.cuda.device(device) if torch.device(device).type=='cuda' else nullcontext()
        with guard:
            collected=[]
            for ordinal,sample in shards[index]:
                if stop.is_set():
                    raise CancelledError('sibling collection worker failed')
                collected.append((ordinal,capture_sample(models[index],sample,include_top1=True)))
                print(json.dumps(dict(phase='collect',worker_device=device,
                    completed_samples=len(collected),worker_total=len(shards[index]))),flush=True)
            return collected

    gathered=_run_pair(collect)
    collected=[sample for _,sample in sorted(gathered[0]+gathered[1])]

    def replay(index,stop):
        device=devices[index]
        guard=torch.cuda.device(device) if torch.device(device).type=='cuda' else nullcontext()
        with guard:
            local=[collected[ordinal] for ordinal,_ in shards[index]]
            return evaluate_suite(models[index],local,mapping,permutations,
                logp_atol,nll_atol,start_id,end_id,donor_samples=collected,
                progress_device=device,cancel_event=stop)

    results=_run_pair(replay)
    sample_order={sample.sample_id:index for index,sample in enumerate(collected)}
    condition_order={condition:index for index,condition in enumerate(CONDITIONS)}
    rows=sorted((row for worker_rows,_ in results for row in worker_rows),
        key=lambda row:(sample_order[row['sample_id']],row['block_index'],condition_order[row['condition']]))
    return rows,_merge_suite_witnesses([witness for _,witness in results])


def run(args):
    from transformers import AutoProcessor
    from .checkpointing import resolve_latent_token_ids
    from .peft import load_stage2_model_for_inference,resolve_stage2_qwen_model
    from .data import load_jsonl_examples
    from .latent_intervention_suite import CONDITIONS, shuffle_mapping, evaluate_suite, publish_suite
    full=args.analysis_suite=='full'
    requested_devices=getattr(args,'devices',None)
    devices=validate_parallel_devices(requested_devices) if requested_devices is not None else None
    if devices is not None and not full:
        raise ValueError('--devices requires --analysis-suite full')
    if devices is not None and args.device!=devices[0]:
        raise ValueError('--device must match the first --devices entry')
    load_device=devices[0] if devices is not None else args.device
    if args.max_samples<=0 or args.max_input_tokens<=0:
        raise ValueError('sample and token limits must be positive')
    if any(not math.isfinite(x) or x<0 for x in (args.identity_logp_atol,args.identity_nll_atol)):
        raise ValueError('identity tolerances must be finite and nonnegative')
    repo=Path(__file__).resolve().parents[2]
    output=Path(args.output_dir).resolve()
    if not output.is_relative_to((repo/'results').resolve()):
        raise ValueError('output-dir must be inside this worktree results directory')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('output directory is not empty; completed runs are not rerun')
    checkpoint=Path(args.checkpoint).resolve()
    if not checkpoint.is_relative_to((repo/'checkpoints').resolve()):
        raise ValueError('checkpoint must be inside this worktree checkpoints tree')
    data=Path(args.data).resolve()
    if not data.is_relative_to((repo/'data').resolve()):
        raise ValueError('data must be inside this worktree data tree')
    raw=load_jsonl_examples(data)
    order=list(range(len(raw))); random.Random(args.seed).shuffle(order)
    order=order[:args.max_samples]
    seen=set(); selected=[]; excluded=[]
    for index in order:
        sid=stable_sample_id(raw[index])
        if sid in seen:
            raise ValueError('selected data has duplicate sample IDs')
        seen.add(sid); selected.append(raw[index])
    output.mkdir(parents=True,exist_ok=True)
    metadata=dict(checkpoint=str(checkpoint),base_model_dir=str(Path(args.base_model_dir).resolve()),
        data=str(data),data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),data_split=args.data_split,seed=args.seed,selected_sample_indices=order,
        selected_samples=len(selected),dtype=args.dtype,attention_backend='sdpa',
        vision_dtype=args.vision_dtype or args.dtype,
        vision_attention_backend='flash_attention_2',
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        conditions=list(CONDITIONS) if full else ['clean','cross'],
        analysis_suite=args.analysis_suite,
        target_scope='all GT text until next latent START or assistant end',
        image_identity='SHA256 of decoded RGB dimensions and pixels; any shared original image excludes donor',
        task_rule='explicit task field, else images_comt/{creation,deletion,update,selection}',
        split_provenance='caller-declared held-out split; training-overlap audit not performed',
        identity_logp_atol=args.identity_logp_atol,identity_nll_atol=args.identity_nll_atol,
        optimizer_updates=0,negative_generation=False,
        source_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__),repo/'src/stage2/latent_causal_intervention.py',
            repo/'src/stage2/latent_intervention_suite.py',
            repo/'src/stage2/attention_backend.py',repo/'src/stage2/data.py',
            repo/'src/stage2/peft.py',repo/'src/stage2/checkpointing.py',
            repo/'transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py',
            repo/'transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py']})
    if devices is not None:
        metadata['execution_devices']=list(devices)
    _atomic_write_text(output/'run_status.json',json.dumps(dict(metadata,status='loading'),indent=2)+'\n')
    try:
        torch.manual_seed(args.seed)
        processor=AutoProcessor.from_pretrained(checkpoint,local_files_only=True)
        model=load_stage2_model_for_inference(Path(args.base_model_dir),checkpoint,processor,
            source_provenance_kwargs={'local_files_only':True},
            model_load_kwargs=model_load_options(args.dtype,load_device,args.vision_dtype),
            load_model=load_analysis_qwen)
        model.eval(); model.requires_grad_(False)
        qwen=resolve_stage2_qwen_model(model)
        apply_vision_dtype(qwen,args.vision_dtype)
        validate_analysis_runtime(qwen)
        token_ids=resolve_latent_token_ids(processor.tokenizer,qwen.config)
        K=qwen.config.latent_size
        if K!=8:
            raise ValueError('checkpoint latent size invalid')
        prepared=[]; digest_cache={}
        for row in selected:
            sample,skips=prepare_sample(row,processor,token_ids,K,Path(args.image_root),args.max_input_tokens,digest_cache)
            excluded.extend(skips)
            if sample is not None:
                prepared.append(sample)
        descriptors=[dict(sample_id=s.sample_id,task=s.task,image_digests=s.image_digests,
            blocks={b.block_index:b.K for b in s.blocks}) for s in prepared]
        mapping=select_donors(descriptors,args.seed)
        metadata['donor_mapping']=[dict(sample_id=s,block_index=m,donor_sample_id=d) for (s,m),d in sorted(mapping.items())]
        metadata['sample_metadata']=descriptors
        metadata['K']=K; metadata['latent_token_ids']=asdict(token_ids)
        permutations={s.sample_id:shuffle_mapping(s.sample_id,
            [b.block_index for b in s.latent_blocks],args.seed) for s in prepared} if full else {}
        if full:
            metadata['shuffle_permutations']=permutations
            metadata['intervention_policy']={
                'mask_zero':'current body embeddings zeroed; K/mask/positions/boundaries unchanged',
                'mask_attention':'current body keys masked; embeddings/K/positions/boundaries unchanged',
                'cross':'current block from different same-task/same-step/same-K sample; no shared original image',
                'step_next':'current block replaced by clean next block; skip last step',
                'step_prev':'current block replaced by clean previous block; skip first step',
                'shuffle':'one seeded derangement of all whole blocks per sample; replace all blocks in replay prefix; skip M<2',
                'future_source':'next-step and shuffle may use clean latents conditioned on later GT; offline interventions, not online decoding',
                'normal_latents':'all conditions reuse a single frozen normal trajectory; no recurrence after intervention'}
        # Map is frozen before the first normal trajectory or score is computed.
        _atomic_write_text(output/'run_status.json',json.dumps(dict(metadata,status='collecting'),indent=2)+'\n')
        if full:
            if devices is None:
                collected=[]
                for sample in prepared:
                    collected.append(capture_sample(qwen,sample,include_top1=True))
                    print(json.dumps({'phase':'collect','completed_samples':len(collected),'total':len(prepared)}),flush=True)
                rows,witness=evaluate_suite(qwen,collected,mapping,permutations,
                    args.identity_logp_atol,args.identity_nll_atol,token_ids.start_token_id,token_ids.end_token_id)
            else:
                if len(prepared)<2:
                    raise ValueError('two GPU streams require at least two prepared samples')
                replica=replicate_qwen_for_device(qwen,devices[1])
                validate_analysis_runtime(replica)
                rows,witness=run_parallel_suite((qwen,replica),devices,prepared,mapping,permutations,
                    args.identity_logp_atol,args.identity_nll_atol,token_ids.start_token_id,token_ids.end_token_id)
            expanded_skips=[dict(r,condition=c) for r in excluded for c in CONDITIONS]
            summary=publish_suite(output,expanded_skips+rows,metadata,witness)
            valid_blocks=summary['conditions']['clean']['valid_blocks']
        else:
            collected=[]
            for sample in prepared:
                collected.append(capture_sample(qwen,sample,include_top1=False))
                print(json.dumps({'phase':'collect','completed_samples':len(collected),'total':len(prepared)}),flush=True)
            rows,witness=evaluate_collected(qwen,collected,mapping,args.identity_logp_atol,args.identity_nll_atol)
            summary=publish(output,excluded+rows,metadata,witness)
            valid_blocks=summary['valid_blocks']
        _atomic_write_text(output/'run_status.json',json.dumps({'status':'complete','valid_blocks':valid_blocks})+'\n')
        return summary
    except BaseException as error:
        _atomic_write_text(output/'run_status.json',json.dumps(dict(metadata,status='failed',error=repr(error)),indent=2)+'\n')
        raise


def main():
    print(json.dumps(run(build_parser().parse_args()),indent=2))
