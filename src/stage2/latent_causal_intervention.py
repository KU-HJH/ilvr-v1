"""Independent fixed-embedding Clean/Cross replay; no training or generation."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
import random
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


@dataclass(frozen=True)
class Block:
    block_index: int
    latent_positions: tuple[int, ...]
    target_start: int
    target_end: int

    @property
    def K(self):
        return len(self.latent_positions)


def extract_blocks(input_ids, assistant_start, assistant_end, token_ids, K):
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import parse_latent_spans
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError('expected one tokenized sample')
    if not 0 <= assistant_start < assistant_end <= input_ids.shape[1]:
        raise ValueError('invalid assistant content bounds')
    spans = parse_latent_spans(input_ids[0], token_ids=token_ids, latent_size=K)
    valid, skipped = [], []
    for i, span in enumerate(spans):
        end = spans[i+1].start_position if i+1 < len(spans) else assistant_end
        if span.start_position < assistant_start or span.end_position >= assistant_end:
            raise ValueError('latent block outside assistant content')
        start = span.end_position + 1
        if start == end:
            skipped.append({'block_index': i, 'skip_reason': 'empty_target'})
        elif start > end:
            raise ValueError('overlapping latent blocks')
        else:
            valid.append(Block(i, tuple(span.latent_positions), start, end))
    return valid, skipped


def block_labels(input_ids, block):
    labels = torch.full_like(input_ids, -100)
    labels[:, block.target_start:block.target_end] = input_ids[:, block.target_start:block.target_end]
    return labels


def select_donors(samples, seed):
    """Stable per-unit mapping, with strict step/K/task and shared-image exclusion."""
    samples = sorted(samples, key=lambda s: s['sample_id'])
    if len({s['sample_id'] for s in samples}) != len(samples):
        raise ValueError('duplicate sample IDs in donor population')
    result = {}
    for sample in samples:
        if not sample['task'] or not sample['image_digests']:
            raise ValueError('task and original-image identity are required')
        for m, K in sorted(sample['blocks'].items()):
            candidates = [s['sample_id'] for s in samples
                if s['sample_id'] != sample['sample_id'] and s['task'] == sample['task']
                and s['blocks'].get(m) == K
                and not set(s['image_digests']).intersection(sample['image_digests'])]
            digest = hashlib.sha256(f'{seed}:{sample["sample_id"]}:{m}'.encode()).digest()
            rng = random.Random(int.from_bytes(digest, 'big'))
            result[(sample['sample_id'], m)] = rng.choice(candidates) if candidates else None
    return result


@dataclass(frozen=True)
class Replay:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    predictor_positions: torch.Tensor
    targets: torch.Tensor


def make_replay(sample, block, donor=None):
    """Keep the entire prefix; truncate at R end; never reuse suffix hidden/KV."""
    stop = block.target_end
    embeddings = sample.embeddings[:, :stop].clone()
    if donor is not None:
        if donor.shape != (block.K, embeddings.shape[-1]) or not torch.isfinite(donor).all():
            raise ValueError('donor embedding shape or finiteness differs')
        embeddings[0, list(block.latent_positions)] = donor.to(embeddings)
    labels = block_labels(sample.input_ids[:, :stop], block)
    targets = labels[0, block.target_start:stop].clone()
    if targets.numel() == 0 or (targets < 0).any():
        raise ValueError('empty or invalid target')
    return Replay(embeddings, sample.attention_mask[:, :stop].clone(),
        sample.position_ids[..., :stop].clone(), labels,
        torch.arange(block.target_start-1, stop-1, device=embeddings.device), targets)


def verify_pair(clean, cross, block, donor):
    keep = torch.ones(clean.embeddings.shape[1], dtype=torch.bool, device=clean.embeddings.device)
    keep[list(block.latent_positions)] = False
    if not torch.equal(clean.embeddings[:, keep], cross.embeddings[:, keep]):
        raise ValueError('Cross changed non-target embeddings')
    if not torch.equal(cross.embeddings[0, list(block.latent_positions)], donor.to(cross.embeddings)):
        raise ValueError('Cross did not inject exact donor embeddings')
    for name in ('attention_mask', 'position_ids', 'labels', 'predictor_positions', 'targets'):
        if not torch.equal(getattr(clean,name), getattr(cross,name)):
            raise ValueError(f'Cross changed {name}')


@dataclass(frozen=True)
class TokenScores:
    logps: torch.Tensor
    top1_ids: torch.Tensor


def score_replay(qwen, replay, *, include_top1=False):
    """Raw logits, FP32 log-softmax, no cache and no latent regeneration."""
    device = next(qwen.parameters()).device
    embeds = replay.embeddings.to(device)
    mask, pos = replay.attention_mask.to(device), replay.position_ids.to(device)
    observed = []
    def audit(module, args, kwargs):
        if not (torch.equal(kwargs['inputs_embeds'], embeds)
                and torch.equal(kwargs['attention_mask'], mask)
                and torch.equal(kwargs['position_ids'], pos)
                and kwargs['use_cache'] is False and kwargs['generate_mode'] is True):
            raise ValueError('backbone replay input invariant failed')
        observed.append(True)
    hook = qwen.model.register_forward_pre_hook(audit, with_kwargs=True)
    try:
        hidden = qwen.model(inputs_embeds=embeds, attention_mask=mask,
            position_ids=pos, use_cache=False, output_attentions=False,
            output_hidden_states=False, return_dict=True, generate_mode=True).last_hidden_state
    finally:
        hook.remove()
    if len(observed) != 1:
        raise ValueError('backbone replay must run exactly once')
    logits = qwen.lm_head(hidden[0, replay.predictor_positions.to(device)]).float()
    logps = logits.log_softmax(-1).gather(-1, replay.targets.to(device)[:,None]).squeeze(-1)
    if not torch.isfinite(logps).all():
        raise ValueError('nonfinite target log probability')
    return TokenScores(logps, logits.argmax(-1)) if include_top1 else logps


def verify_identity(native_logps, replay_logps, logp_atol, nll_atol):
    if any(not math.isfinite(x) or x < 0 for x in (logp_atol,nll_atol)):
        raise ValueError('identity tolerances must be finite and nonnegative')
    b = replay_logps.detach().float()
    a = native_logps.detach().to(device=b.device, dtype=torch.float32)
    if a.shape != b.shape or a.numel() == 0 or not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('identity target shape/finiteness failed')
    maximum, mean = float((a-b).abs().max()), float((a.mean()-b.mean()).abs())
    if maximum > logp_atol or mean > nll_atol:
        raise ValueError(f'identity replay failed: max logp error={maximum}, NLL error={mean}')
    return {'max_logp_error': maximum, 'nll_error': mean}


def summarize(rows, seed):
    grouped = defaultdict(list)
    skips = Counter()
    identities = set()
    for row in rows:
        key = (row['sample_id'],row['block_index'])
        if key in identities:
            raise ValueError('duplicate evaluation unit')
        identities.add(key)
        if row['status'] == 'skip':
            skips[row['skip_reason']] += 1
        elif row['status'] == 'ok' and math.isfinite(row['delta_nll']):
            grouped[row['sample_id']].append(row['delta_nll'])
        else:
            raise ValueError('invalid block result')
    values = np.array([np.mean(grouped[s]) for s in sorted(grouped)],dtype=np.float64)
    mean, ci = None, None
    if len(values):
        rng = np.random.default_rng(seed)
        boot = np.array([rng.choice(values,size=len(values),replace=True).mean() for _ in range(2000)])
        mean, ci = float(values.mean()), np.quantile(boot,[.025,.975]).tolist()
    return {'valid_samples':len(values), 'valid_blocks':sum(map(len,grouped.values())),
        'excluded_by_reason':dict(sorted(skips.items())), 'mean_delta_nll':mean,
        'delta_nll_95ci':ci, 'bootstrap_replicates':2000, 'bootstrap_seed':seed,
        'aggregation':'mean of per-sample means of per-block token-normalized delta NLL',
        'donor_mapping_fixed_during_bootstrap':True}


def _atomic_write_text(path, text):
    """Publish one metadata file atomically, without unrelated analysis dependencies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
