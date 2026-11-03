"""Opt-in, differentiable shared-prefix replay of independent Stage 1 ablations.

Each packed query can see only its own unmodified prefix and its own replay.
Clean end-marker/downstream states are never used by that branch. Cache updates
are functional so non-reentrant checkpoint recomputation cannot grow a cache.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from ...cache_utils import DynamicCache
from .stage1_prefix_conditioning import (ConditioningRequest, ConditionedReplayScores,
    compile_conditioned_layout, materialize_conditioned_inputs)
from .stage1_prefix_trace import PrefixTraceObserver, bind_prefix_observer


@dataclass(frozen=True)
class Stage1PackedGainBlock:
    batch_index: int
    start_marker: int
    end_marker: int
    target_positions: tuple[int, ...]
    target_ids: tuple[int, ...]


@dataclass(frozen=True)
class Stage1PackedGainRequest:
    blocks: tuple[Stage1PackedGainBlock, ...]
    branch_mode: str = 'independent'
    conditioning: ConditioningRequest | None = None


class _GainCache(DynamicCache):
    """Record the first clean KVs, or join an immutable differentiable prefix."""
    def __init__(self, layer_count, prefix=None):
        super().__init__()
        self.captured = [None] * layer_count
        self.prefix = prefix

    def get_seq_length(self, layer_idx=0):
        return 0 if self.prefix is None else self.prefix[layer_idx][0].shape[-2]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.prefix is None:
            if self.captured[layer_idx] is None:
                self.captured[layer_idx] = (key_states, value_states)
            return key_states, value_states
        key, value = self.prefix[layer_idx]
        return torch.cat((key, key_states), dim=-2), torch.cat((value, value_states), dim=-2)


@contextmanager
def bind_dropout_forwards(bindings):
    """Also used by checkpoint closures to bind the correct clean/replay phase."""
    saved = [(module, module.__dict__.get('forward')) for module, _ in bindings]
    try:
        for module, forward in bindings:
            module.forward = forward
        yield
    finally:
        for module, forward in saved:
            if forward is None:
                module.__dict__.pop('forward', None)
            else:
                module.forward = forward


class _PairedDropout:
    """Reuse each source token's dropout realization without retaining big masks.

Only RNG states and shapes are retained. Replay regenerates the original mask
under fork_rng, then gathers the packed token rows. The caller's RNG is unchanged.
"""
    def __init__(self, module, *, native_randomness=False):
        self.original = module.forward
        self.p = module.p
        self.state = None
        self.native_randomness = native_randomness

    def _random_scope(self, x):
        if not self.native_randomness:
            return nullcontext()
        if torch._C._functorch.is_batchedtensor(x):
            raise RuntimeError('paired dropout recompute requires unbatched activations')
        # Only cotangents are vmapped: checkpoint recomputation uses ordinary
        # activations. Keep native CUDA dropout's bool saved mask and Philox
        # sequence, rather than vmap's different dropout decomposition. Do not
        # disable batching for the backward or the surrounding layer.
        return torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.FuncTorchVmapMode))

    def clean(self, x):
        if self.state is None:
            device = x.device
            cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
            self.state = (tuple(x.shape), x.dtype, device, torch.get_rng_state(), cuda_rng)
        with self._random_scope(x):
            return self.original(x)

    def replay(self, indices, x):
        shape, dtype, device, cpu_rng, cuda_rng = self.state
        if len(shape) != 3 or x.shape != (1, indices.numel(), shape[-1]):
            raise ValueError('packed gain requires tokenwise decoder dropout [B,T,H]')
        devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            with self._random_scope(x):
                mask = F.dropout(torch.ones(shape, dtype=dtype, device=device), p=self.p, training=True)
        return x * mask.reshape(-1, shape[-1]).index_select(0, indices).unsqueeze(0)


def validate_packed_gain_model(decoder, lm_head):
    if decoder.config._attn_implementation not in {'eager', 'sdpa'}:
        raise ValueError('packed gain requires eager or sdpa attention')
    if getattr(decoder.config, 'use_sliding_window', False):
        raise ValueError('packed gain does not support sliding-window attention')
    if getattr(decoder.rotary_emb, 'rope_type', 'default') not in {'default', 'mrope'}:
        raise ValueError('packed gain requires fixed rotary frequencies')
    if any(getattr(m, 'attention_dropout', 0.) for m in decoder.modules()):
        raise ValueError('packed gain requires attention_dropout=0; adapter dropout is supported')
    if any(isinstance(m, nn.Dropout) and m.training and m.p for m in lm_head.modules()):
        raise ValueError('packed gain does not support dropout in the output head')


def _pack(request, padding, embeds, positions):
    if not isinstance(request, Stage1PackedGainRequest) or not request.blocks:
        raise ValueError('packed gain requires nonempty block requests')
    if request.branch_mode not in {'independent', 'cumulative'}:
        raise ValueError('invalid packed gain branch mode')
    batch, length = padding.shape
    if positions.ndim == 2:
        positions = positions.unsqueeze(0).expand(3, -1, -1)
    for block in request.blocks:
        b, l, r = block.batch_index, block.start_marker, block.end_marker
        targets = block.target_positions
        if (not 0 <= b < batch or not 0 <= l < l + 1 < r < length or not targets
                or len(targets) != len(block.target_ids)
                or any(not r < i < length for i in targets)
                or tuple(sorted(set(targets))) != targets):
            raise ValueError('invalid packed gain block or GT window')
    if request.branch_mode == 'independent':
        groups = [[i] for i in range(len(request.blocks))]
    else:
        groups = [[i for i, b in enumerate(request.blocks) if b.batch_index == row]
                  for row in sorted({b.batch_index for b in request.blocks})]
    indices, owners, rows, ends = [], [], [], []
    score_rows = [None] * len(request.blocks)
    cursor = 0
    for owner, group in enumerate(groups):
        blocks = [request.blocks[i] for i in group]
        row = blocks[0].batch_index
        start = min(b.end_marker for b in blocks)
        stop = max(b.target_positions[-1] for b in blocks)
        indices.extend(row * length + i for i in range(start, stop))
        owners.extend([owner] * (stop - start))
        rows.extend([row] * (stop - start))
        ends.extend([start] * (stop - start))
        for i, b in zip(group, blocks):
            score_rows[i] = [cursor + pos - 1 - start for pos in b.target_positions]
        cursor += stop - start
    device = embeds.device
    indices, owner, rows, ends = (torch.tensor(x, device=device) for x in (indices, owners, rows, ends))
    keys = torch.arange(batch * length, device=device)
    key_rows, key_positions = keys // length, keys % length
    prefix = ((rows[:, None] == key_rows) & (key_positions < ends[:, None])
              & padding.reshape(-1).bool())
    replay = (owner[:, None] == owner[None, :]) & (indices[None, :] <= indices[:, None])
    replay &= padding.reshape(-1)[indices].bool()[None, :]
    allowed = torch.cat((prefix, replay), dim=1)
    # Both halves use original physical coordinates; progressive cumulative
    # masking begins at each END and preserves within-body causal attention.
    all_positions = torch.cat((key_positions, indices % length))
    all_rows = torch.cat((key_rows, rows))
    for i, block in enumerate(request.blocks):
        affected = ((rows == block.batch_index) & (indices % length >= block.end_marker))
        if request.branch_mode == 'independent':
            affected &= owner == i
        blocked = ((all_rows == block.batch_index) & (all_positions > block.start_marker)
                   & (all_positions < block.end_marker))
        allowed &= ~(affected[:, None] & blocked[None, :])
    mask = embeds.new_zeros((1, 1, cursor, batch * length + cursor)).masked_fill_(~allowed, -torch.inf)
    packed_embeds = embeds.reshape(-1, embeds.shape[-1])[indices].unsqueeze(0)
    packed_positions = positions.reshape(3, -1)[:, indices].unsqueeze(1)
    return (indices, packed_embeds, packed_positions, mask,
            [i for group in score_rows for i in group],
            [i for b in request.blocks for i in b.target_ids],
            [len(b.target_ids) for b in request.blocks])


def run_packed_stage1_gain(decoder, lm_head, inputs_embeds, position_ids, padding, request):
    """One clean decoder call + one isolated packed replay, with both graphs live."""
    validate_packed_gain_model(decoder, lm_head)
    # This fork's 2D-mask path applies a finite window even when the config's
    # use_sliding_window flag is false; the legacy 4D ablation mask bypasses it.
    # Such a clean prefix is not equivalent to the independent masked prefix.
    window = getattr(decoder.config, 'sliding_window', None)
    if window is not None and inputs_embeds.shape[1] > window:
        raise ValueError('packed gain cannot share an effective sliding-window prefix; use independent execution')
    indices, embeds, positions, mask, score_rows, targets, sizes = _pack(
        request, padding, inputs_embeds, position_ids)
    pairs = [(m, _PairedDropout(m)) for m in decoder.modules()
             if isinstance(m, nn.Dropout) and m.training and m.p]
    capture = _GainCache(len(decoder.layers))
    with bind_dropout_forwards([(m, pair.clean) for m, pair in pairs]):
        clean = decoder(inputs_embeds=inputs_embeds, position_ids=position_ids,
                        attention_mask=padding, past_key_values=capture, use_cache=False,
                        output_attentions=False, output_hidden_states=False, return_dict=True)
    # Flatten samples into a prefix bank. Visibility above selects the matching
    # source sample; no downstream clean state is visible to its own replay.
    prefix = tuple(tuple(t.permute(1, 0, 2, 3).reshape(1, t.shape[1], -1, t.shape[-1])
                         for t in pair) for pair in capture.captured)
    from functools import partial
    with bind_dropout_forwards([(m, partial(pair.replay, indices)) for m, pair in pairs]):
        masked = decoder(inputs_embeds=embeds, position_ids=positions, attention_mask=mask,
                         past_key_values=_GainCache(len(decoder.layers), prefix), use_cache=False,
                         output_attentions=False, output_hidden_states=False, return_dict=True)
    logits = lm_head(masked.last_hidden_state[0, score_rows]).float()
    target = torch.tensor(targets, dtype=torch.long, device=logits.device)
    token_scores = logits.log_softmax(-1).gather(1, target[:, None]).squeeze(1)
    scores = torch.stack([part.mean() for part in token_scores.split(sizes)])
    return clean, scores


def run_conditioned_stage1_gain(decoder, lm_head, inputs_embeds, position_ids, padding,
                                request) -> tuple[object, ConditionedReplayScores]:
    """One traced clean pass, one frozen selector plan, one paired decoder pass."""
    conditioning = request.conditioning
    if not isinstance(conditioning, ConditioningRequest) or not request.blocks:
        raise ValueError('conditioned gain requires conditioning and nonempty blocks')
    validate_packed_gain_model(decoder, lm_head)
    window = getattr(decoder.config, 'sliding_window', None)
    if window is not None and inputs_embeds.shape[1] > window:
        raise ValueError('conditioned gain cannot share an effective sliding-window prefix')
    pairs = [(m, _PairedDropout(m, native_randomness=conditioning.consume_positive_logits is not None))
             for m in decoder.modules()
             if isinstance(m, nn.Dropout) and m.training and m.p]
    capture = _GainCache(len(decoder.layers))
    observer = PrefixTraceObserver(conditioning.trace_request, layer_count=len(decoder.layers),
                                   chunk_rows=conditioning.trace_chunk_rows)
    with conditioning.role_scope('gain_clean') as ledger, bind_prefix_observer(decoder, observer), \
            bind_dropout_forwards([(m, pair.clean) for m, pair in pairs]):
        if ledger is not None and hasattr(observer, 'observe'):
            original_observe = observer.observe
            def timed_observe(*args, **kwargs):
                with ledger.time_region('observer', device=inputs_embeds.device):
                    return original_observe(*args, **kwargs)
            observer.observe = timed_observe
        clean = decoder(inputs_embeds=inputs_embeds, position_ids=position_ids,
                        attention_mask=padding, past_key_values=capture, use_cache=False,
                        output_attentions=False, output_hidden_states=False, return_dict=True)
        rows = observer.finish()
    plan = conditioning.build_plan(clean.last_hidden_state, lm_head, position_ids, rows)
    layout = compile_conditioned_layout(request, plan, padding,
        element_size=inputs_embeds.element_size(), max_packed_tokens=conditioning.max_packed_tokens,
        max_mask_bytes=conditioning.max_mask_bytes)
    embeds, positions, mask = materialize_conditioned_inputs(layout, inputs_embeds, position_ids)
    if getattr(ledger, 'report', None) is not None:
        ledger.report.update(R_pack=int(embeds.shape[1]), clean_bank_size=int(inputs_embeds.shape[0] * inputs_embeds.shape[1]),
                             mask_shape=list(mask.shape))
    if any(pair is None for pair in capture.captured):
        raise RuntimeError('conditioned gain did not capture every clean KV layer')
    prefix = tuple(tuple(t.permute(1, 0, 2, 3).reshape(1, t.shape[1], -1, t.shape[-1])
                         for t in pair) for pair in capture.captured)
    from functools import partial
    with conditioning.role_scope('packed_pair'), bind_dropout_forwards(
            [(m, partial(pair.replay, layout.indices)) for m, pair in pairs]):
        replay = decoder(inputs_embeds=embeds, position_ids=positions, attention_mask=mask,
                         past_key_values=_GainCache(len(decoder.layers), prefix), use_cache=False,
                         output_attentions=False, output_hidden_states=False, return_dict=True)
    means = [[], []]
    for block, paired_rows in zip(request.blocks, layout.score_rows):
        for condition, score_rows in enumerate(paired_rows):
            chunks = []
            for start in range(0, len(score_rows), conditioning.logit_chunk_rows):
                end = start + conditioning.logit_chunk_rows
                logits = lm_head(replay.last_hidden_state[0, list(score_rows[start:end])]).float()
                if condition == 0 and conditioning.consume_positive_logits is not None:
                    keys = tuple((block.batch_index, p - 1) for p in block.target_positions[start:end])
                    conditioning.consume_positive_logits(keys, logits)
                targets = torch.tensor(block.target_ids[start:end], device=logits.device)
                chunks.append(logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1))
            means[condition].append(torch.cat(chunks).mean())
    return clean, ConditionedReplayScores(*(torch.stack(values) for values in means))
