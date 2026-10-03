"""Fixed-embedding intervention suite: local masks/swaps and joint block shuffle."""
from collections import Counter, defaultdict
from concurrent.futures import CancelledError
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from .latent_causal_intervention import _atomic_write_text
from .latent_causal_intervention import make_replay, score_replay, summarize, verify_identity

CONDITIONS=('clean','mask_zero','mask_attention','cross','step_next','step_prev','shuffle')


def shuffle_mapping(sample_id, block_indices, seed):
    """One predeclared sample-level derangement; preserve vectors within each block."""
    indices=sorted(block_indices)
    if len(indices)!=len(set(indices)):
        raise ValueError('duplicate block index')
    if len(indices)<2:
        return None
    rng=random.Random(int.from_bytes(hashlib.sha256(f'shuffle:{seed}:{sample_id}'.encode()).digest(),'big'))
    for _ in range(256):
        perm=indices.copy(); rng.shuffle(perm)
        if all(i!=j for i,j in zip(indices,perm)):
            return dict(zip(indices,perm))
    return dict(zip(indices,indices[1:]+indices[:1]))


def intervention_replay(sample, block, condition, *, donor=None, permutation=None):
    if condition not in CONDITIONS:
        raise ValueError('unknown intervention condition')
    clean=make_replay(sample,block)
    altered={}; mask_positions=[]; info={}
    positions=list(block.latent_positions)
    if condition=='mask_zero':
        altered[block.block_index]=torch.zeros_like(sample.bank[block.block_index])
    elif condition=='mask_attention':
        mask_positions=positions
    elif condition=='cross':
        if donor is None:
            return None,dict(skip_reason='no_donor')
        altered[block.block_index]=donor
    elif condition in ('step_next','step_prev'):
        source=block.block_index+(1 if condition=='step_next' else -1)
        if source not in sample.bank:
            return None,dict(skip_reason='no_adjacent_step')
        altered[block.block_index]=sample.bank[source]
        info['source_block_index']=source
    elif condition=='shuffle':
        if permutation is None:
            return None,dict(skip_reason='fewer_than_two_blocks')
        indices=set(sample.bank)
        if (set(permutation)!=indices or set(permutation.values())!=indices
                or any(k==v for k,v in permutation.items())):
            raise ValueError('shuffle requires a complete block derangement permutation')
        info['permutation']=permutation
        altered={b.block_index:sample.bank[permutation[b.block_index]]
            for b in getattr(sample,'latent_blocks',sample.blocks) if b.target_start<=block.target_end}
    embeds=clean.embeddings.clone(); mask=clean.attention_mask.clone()
    blocks={b.block_index:b for b in getattr(sample,'latent_blocks',sample.blocks)}
    changed=[]
    for index,value in altered.items():
        dest=blocks[index]; pos=list(dest.latent_positions)
        if value.shape!=(dest.K,embeds.shape[-1]) or not torch.isfinite(value).all():
            raise ValueError('source latent shape or finiteness differs')
        if pos[-1]>=embeds.shape[1]:
            raise ValueError('intervention includes future slots beyond target window')
        embeds[0,pos]=value.to(embeds); changed.extend(pos)
    if mask_positions:
        mask[0,mask_positions]=0
    replay=replace(clean,embeddings=embeds,attention_mask=mask)
    # Audit every allowed change against an independently preserved clean replay.
    keep=torch.ones(embeds.shape[1],dtype=torch.bool); keep[changed]=False
    if not torch.equal(embeds[:,keep],clean.embeddings[:,keep]):
        raise ValueError('intervention changed non-target embeddings')
    for index,value in altered.items():
        if not torch.equal(embeds[0,list(blocks[index].latent_positions)],value.to(embeds)):
            raise ValueError('source latent injection failed')
    expected=clean.attention_mask.clone(); expected[0,mask_positions]=0
    if not torch.equal(mask,expected):
        raise ValueError('intervention changed unexpected attention entries')
    for name in ('position_ids','labels','predictor_positions','targets'):
        if not torch.equal(getattr(clean,name),getattr(replay,name)):
            raise ValueError(f'intervention changed {name}')
    info['changed_embedding_positions']=changed
    info['masked_positions']=mask_positions
    return replay,info


def token_metrics(clean, current, targets, start_id, end_id):
    if not (clean.logps.shape==current.logps.shape==clean.top1_ids.shape==current.top1_ids.shape==targets.shape) or targets.numel()==0:
        raise ValueError('token scores must align with nonempty targets')
    a,b=clean.top1_ids,current.top1_ids; targets=targets.to(b.device)
    n=targets.numel(); clean_nll=-float(clean.logps.mean()); nll=-float(current.logps.mean())
    correct=int((b==targets).sum()); original=int((a==targets).sum()); flips=int((a!=b).sum())
    return dict(target_token_count=n,nll_clean=clean_nll,nll=nll,delta_nll=nll-clean_nll,
        top1_correct=correct,clean_top1_correct=original,top1_accuracy=correct/n,
        clean_top1_accuracy=original/n,delta_top1_accuracy=(correct-original)/n,
        top1_flips=flips,top1_flip_rate=flips/n,
        first_top1_flip=int(a[0]!=b[0]),first_clean_correct=int(a[0]==targets[0]),
        first_top1_correct=int(b[0]==targets[0]),
        clean_correct_to_wrong=int(((a==targets)&(b!=targets)).sum()),
        clean_wrong_to_correct=int(((a!=targets)&(b==targets)).sum()),
        latent_end_top1=int((b==end_id).sum()),latent_start_top1=int((b==start_id).sum()),
        first_latent_end_top1=int(b[0]==end_id),first_latent_start_top1=int(b[0]==start_id),
        first_top1_id=int(b[0]),first_clean_top1_id=int(a[0]),first_target_id=int(targets[0]),
        top1_ids=b.tolist(),clean_top1_ids=a.tolist(),target_ids=targets.tolist(),
        target_logps=current.logps.tolist(),clean_target_logps=clean.logps.tolist())


@torch.inference_mode()
def evaluate_suite(qwen, samples, mapping, permutations, logp_atol, nll_atol, start_id, end_id,
                   *, donor_samples=None, progress_device=None, cancel_event=None):
    lookup={s.sample_id:s for s in (samples if donor_samples is None else donor_samples)}; rows=[]
    witness=dict(passed=True,identity_checked_blocks=0,condition_inputs_checked=0,
        max_logp_error=0.,max_nll_error=0.,native_replay_top1_flips=0,
        identity_logp_atol=logp_atol,identity_nll_atol=nll_atol,
        cache='disabled; prefix through target R recomputed in each condition')
    for sample in samples:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError('sibling replay worker failed')
        for block in sample.blocks:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError('sibling replay worker failed')
            base=dict(sample_id=sample.sample_id,task=sample.task,block_index=block.block_index,K=block.K)
            clean_replay,_=intervention_replay(sample,block,'clean')
            clean=score_replay(qwen,clean_replay,include_top1=True)
            identity=verify_identity(sample.native_logps[block.block_index],clean.logps,logp_atol,nll_atol)
            witness['identity_checked_blocks']+=1
            witness['max_logp_error']=max(witness['max_logp_error'],identity['max_logp_error'])
            witness['max_nll_error']=max(witness['max_nll_error'],identity['nll_error'])
            witness['native_replay_top1_flips']+=int((sample.native_top1[block.block_index].to(clean.top1_ids.device)!=clean.top1_ids).sum())
            for condition in CONDITIONS:
                donor_id=mapping[(sample.sample_id,block.block_index)] if condition=='cross' else None
                donor=None
                if donor_id is not None:
                    donor_sample=lookup[donor_id]
                    if (donor_id==sample.sample_id or donor_sample.task!=sample.task
                            or set(donor_sample.image_digests).intersection(sample.image_digests)):
                        raise ValueError('cross donor eligibility invariant failed')
                    donor=donor_sample.bank[block.block_index]
                replay,info=intervention_replay(sample,block,condition,donor=donor,
                    permutation=permutations[sample.sample_id])
                row=dict(base,condition=condition,donor_sample_id=donor_id,**info)
                if replay is None:
                    rows.append(dict(row,status='skip',target_token_count=0)); continue
                scored=clean if condition=='clean' else score_replay(qwen,replay,include_top1=True)
                witness['condition_inputs_checked']+=1
                rows.append(dict(row,status='ok',identity=identity,
                    **token_metrics(clean,scored,replay.targets,start_id,end_id)))
        progress=dict(phase='suite',sample_id=sample.sample_id,
            completed_blocks=witness['identity_checked_blocks'],completed_conditions=witness['condition_inputs_checked'])
        if progress_device is not None:
            progress['worker_device']=progress_device
        print(json.dumps(progress),flush=True)
    return rows,witness


def summarize_condition(rows, seed):
    result=summarize(rows,seed)
    valid=[r for r in rows if r['status']=='ok']
    if not valid:
        result['micro']=None; return result
    total=sum(r['target_token_count'] for r in valid); blocks=len(valid)
    counts={k:sum(r[k] for r in valid) for k in ('top1_correct','clean_top1_correct','top1_flips',
        'first_top1_flip','first_top1_correct','first_clean_correct','latent_end_top1','latent_start_top1',
        'first_latent_end_top1','first_latent_start_top1','clean_correct_to_wrong','clean_wrong_to_correct')}
    result['micro']=dict(counts,target_tokens=total,blocks=blocks,
        top1_accuracy=counts['top1_correct']/total,clean_top1_accuracy=counts['clean_top1_correct']/total,
        top1_flip_rate=counts['top1_flips']/total,first_top1_flip_rate=counts['first_top1_flip']/blocks,
        first_top1_accuracy=counts['first_top1_correct']/blocks,
        nll=sum(r['nll']*r['target_token_count'] for r in valid)/total,
        clean_nll=sum(r['nll_clean']*r['target_token_count'] for r in valid)/total)
    grouped=defaultdict(list)
    for row in valid:
        grouped[row['sample_id']].append(row)
    for field in ('nll','nll_clean','delta_top1_accuracy','top1_flip_rate'):
        values=np.array([np.mean([r[field] for r in grouped[s]]) for s in sorted(grouped)])
        result['mean_'+field]=float(values.mean())
    result['max_abs_block_delta_nll']=max(abs(r['delta_nll']) for r in valid)
    return result


def publish_suite(output, rows, metadata, witness):
    valid=[r for r in rows if r['status']=='ok']
    clean_count=sum(r['condition']=='clean' for r in valid)
    if (not witness.get('passed') or witness.get('identity_checked_blocks')!=clean_count
            or witness.get('condition_inputs_checked')!=len(valid)):
        raise ValueError('incomplete validation witness; refusing publication')
    by_condition={c:summarize_condition([r for r in rows if r['condition']==c],metadata['seed']) for c in CONDITIONS}
    summary=dict(metadata,status='complete',conditions=by_condition,validation=witness,
        aggregation='block token mean -> sample block mean -> equal sample mean; micro top-1 weighted by target tokens',
        bootstrap_replicates=2000,donors_and_shuffle_fixed_during_bootstrap=True)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    _atomic_write_text(output/'per_block.jsonl',''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    _atomic_write_text(output/'summary.json',json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary
