"""Throwaway shared-F premise adapter; never a Stage1-v5 product implementation.

Uses the existing patched native Qwen K/V API. Packing coordinate idea comes
from pinned e555a2b6 stage1_v3_replay.py; this deliberately has independent
V-off/Text-off rows rather than v3's R/RV interventions and no alias graph.
"""
import torch
import torch.nn.functional as F
from stage1.trainer import CustomTrainerStage1
from step_preference.dil_bce_orig_loss import dil_bce_orig_components
from step_preference.topk_entropy_loss import CandidateSupport, topk_entropy_components
from types import SimpleNamespace


def execute(qwen,reference,x,positions,blocks,runner,control=None):
    if control not in (None,'detach_prefix','contaminate_v'):
        raise ValueError('unknown probe control')
    clean = runner(qwen.model,x,positions,role='F',capture=True)
    clean_logits = qwen.lm_head(clean.last_hidden_state)
    with torch.no_grad():
        frozen = runner(reference.model,x.detach(),positions,role='reference')
        reference_logits = reference.lm_head(frozen.last_hidden_state)
    rows = tuple(('v',b) for b in blocks)+tuple(('text',b) for b in blocks)
    prefix_length = max(b[1] for b in blocks)
    length = max(max(b[2])-b[1] for b in blocks)
    source = torch.stack([b[1]+torch.arange(length) for _,b in rows])
    valid = torch.stack([source[i]<max(b[2]) for i,(_,b) in enumerate(rows)])
    source = source.clamp_max(x.shape[1]-1)
    embeds = x[0,source]*valid[:,:,None]
    position_ids = positions[:,0,source]
    mask = x.new_full((len(rows),1,length,prefix_length+length),-torch.inf)
    for row,(kind,(start,end,targets,selected)) in enumerate(rows):
        actual = max(targets)-end
        for query in range(actual):
            mask[row,0,query,:end] = 0
            mask[row,0,query,prefix_length:prefix_length+query+1] = 0
        mask[row,0,actual:,0] = 0  # ignored padded queries remain well-defined
        if kind == 'v':
            mask[row,0,:actual,start+1:end] = -torch.inf
        if kind == 'text' or control == 'contaminate_v':
            for target,keys in zip(targets,selected):
                mask[row,0,target-1-end,list(keys)] = -torch.inf
    pairs = clean.stage2_key_values
    if control == 'detach_prefix':
        pairs = tuple(tuple(value.detach() for value in pair) for pair in pairs)
    prefixes = tuple(tuple(torch.cat([
        F.pad(value[:1,:,:b[1]],(0,0,0,prefix_length-b[1]))
        for _,b in rows],dim=0) for value in pair) for pair in pairs)
    replay = runner(qwen.model,embeds,position_ids,source_positions=source,
                    mask=mask,prefix=prefixes,role='packed')
    row_indices,offsets = zip(*[(i,target-1-b[1])
                              for i,(_,b) in enumerate(rows) for target in b[2]])
    logits = qwen.lm_head(replay.last_hidden_state[row_indices,offsets])
    n = logits.shape[0]//2
    return dict(clean=clean_logits,reference=reference_logits,
                hidden=clean.last_hidden_state,v=logits[:n],text=logits[n:])


def losses(output,x,noise,labels,support,ids,body,predictors,blocks):
    def original_block_mean(values):
        parts = torch.split(values,[len(b[2]) for b in blocks])
        return torch.stack([part.mean() for part in parts]).mean()
    clean = output['clean'][0,list(predictors)]
    reference = output['reference'][0,list(predictors)]
    targets = torch.cat([ids[0,list(b[2])] for b in blocks])
    parts = dil_bce_orig_components(clean,reference,targets)
    dil = original_block_mean(parts.chosen+parts.reference)
    candidates = CandidateSupport(support,torch.ones_like(support,dtype=torch.bool),
                                  (support==targets[:,None]).any(-1))
    entropy = original_block_mean(topk_entropy_components(output['v'],candidates).kl)
    temperature = 2
    target = (clean.detach().float()/temperature).softmax(-1)
    text_kl = temperature**2*F.kl_div((output['text'].float()/temperature).log_softmax(-1),
                                     target,reduction='none').sum(-1).mean()
    ce = F.cross_entropy(output['clean'][:,:-1].contiguous().float().view(-1,128),
                         labels[:,1:].contiguous().view(-1))
    # Exercise Stage1's actual alignment method with the fixed perturbation.
    # A local randn_like binding supplies fixed noise, not normalization changes.
    latent_mask = torch.zeros_like(labels,dtype=torch.bool)
    latent_mask[:,list(body)] = True
    targets_object = SimpleNamespace(latent_mask=latent_mask)
    outputs_object = SimpleNamespace(loss=ce,hidden_states=output['hidden'],inputs_embeds=x)
    original_randn_like = torch.randn_like
    def fixed_noise(value,*args,**kwargs):
        assert value.shape == noise.shape
        return noise.to(device=value.device,dtype=value.dtype)
    torch.randn_like = fixed_noise
    try:
        alignment = CustomTrainerStage1._alignment_loss(SimpleNamespace(sim_weight=.1),outputs_object,targets_object)
    finally:
        torch.randn_like = original_randn_like
    terms = dict(ce=ce,alignment=alignment,dil=dil,entropy=entropy,text_kl=text_kl)
    terms['total'] = .1*ce+.1*alignment+dil+.1*entropy+.01*text_kl
    return terms
