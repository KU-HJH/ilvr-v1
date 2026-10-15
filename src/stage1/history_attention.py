"""Reduce native effective-DoRA causal attention without retaining S-by-S rows."""
from functools import partial
import math

import torch
from torch import Tensor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
)

from .query import Stage1QueryConfig


class ReducedHistoryAttentionObserver:
    """Observe one frozen, batch-one, uncached full-prefix decoder forward.

    Hooks read the effective q/k outputs and the decoder's actual rotary inputs.
    Only the active layer owns projection tensors; key/query/head chunks bound
    score workspace. No attention backend or model state is changed.

    The patched native mask uses config.sliding_window independently of its
    enable flag. Full-causal observation requires sliding_window=None or a
    prefix length no greater than that window (last query index < window).
    """

    def __init__(self, backbone, *, layer_indices: tuple[int, ...],
                 query_positions: tuple[int, ...], latent_positions: tuple[int, ...],
                 attention_mask: Tensor, config: Stage1QueryConfig):
        if (attention_mask.ndim != 2 or attention_mask.shape[0] != 1
                or attention_mask.shape[1] == 0
                or not bool(((attention_mask == 0) | (attention_mask == 1)).all())):
            raise ValueError('observer requires a binary [1,S] padding mask')
        if (not isinstance(layer_indices, tuple) or not layer_indices
                or any(type(i) is not int or not 0 <= i < len(backbone.layers)
                       for i in layer_indices)
                or len(set(layer_indices)) != len(layer_indices)):
            raise ValueError('observer requires unique valid decoder layer indices')
        for name, positions in (('query', query_positions), ('latent', latent_positions)):
            if (not isinstance(positions, tuple)
                    or any(type(i) is not int or not 0 <= i < attention_mask.shape[1]
                           or not bool(attention_mask[0, i]) for i in positions)
                    or len(set(positions)) != len(positions)):
                raise ValueError(f'{name} positions must be unique valid unpadded prefix positions')
        self.backbone = backbone
        self.layer_indices = layer_indices
        self.query_positions = query_positions
        self.latent_positions = latent_positions
        self.config = config
        self._mask = attention_mask.detach().clone()
        self._handles = []
        self._active = None
        self._seen = set()
        self._captured = set()
        self._failed = False
        self._entered = False
        self._sum = torch.zeros(len(latent_positions), device=attention_mask.device,
                                dtype=torch.float32)

    def __enter__(self):
        if self._entered:
            raise RuntimeError('observer context may be entered only once')
        self._entered = True
        if not self.query_positions or not self.latent_positions:
            return self
        # Native eager/SDPA mask construction uses sliding_window even when
        # use_sliding_window=False. Once S > window it excludes causal keys;
        # changing that native execution or observing a windowed objective is
        # outside this observer's full-causal contract.
        window = self.backbone.config.sliding_window
        if window is not None and self._mask.shape[1] > window:
            raise ValueError(
                'observer requires full-causal attention: prefix length '
                f'{self._mask.shape[1]} exceeds native sliding_window={window}')
        try:
            for i in self.layer_indices:
                layer = self.backbone.layers[i]
                if (any(m.training for m in layer.modules())
                        or any(p.requires_grad or p.grad is not None for p in layer.parameters())
                        or layer.attn_type not in ('sdpa', 'eager')):
                    raise ValueError('observer requires frozen eval full-causal eager/SDPA layers')
                self._handles.append(layer.register_forward_pre_hook(
                    partial(self._before_layer, i), with_kwargs=True))
                for name in ('q', 'k'):
                    self._handles.append(getattr(layer.self_attn, name + '_proj').register_forward_hook(
                        partial(self._projection, i, name)))
                self._handles.append(layer.register_forward_hook(partial(self._after_layer, i)))
        except BaseException:
            self._cleanup()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._failed = exc_type is not None
        self._cleanup()
        return False

    def _cleanup(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = None

    @torch.no_grad()
    def _before_layer(self, index, module, args, kwargs):
        if index in self._seen:
            raise RuntimeError('duplicate layer capture: each layer must execute exactly once')
        if self._active is not None:
            raise RuntimeError('previous layer projection capture is incomplete')
        # Mirror the native decoder signature, including positional invocation.
        def argument(name, offset, default=None):
            return kwargs.get(name, args[offset] if len(args) > offset else default)
        hidden = argument('hidden_states', 0)
        if (hidden.shape[:2] != self._mask.shape or hidden.device != self._mask.device
                or argument('past_key_value', 3) is not None
                or argument('use_cache', 5, False)
                or kwargs.get('stage2_prefix_key_value') is not None):
            raise ValueError('observer supports only an uncached batch-one complete prefix')
        embeddings = argument('position_embeddings', 7)
        if embeddings is None:
            raise ValueError('observer requires actual native position embeddings')
        self._seen.add(index)
        self._active = {'index': index, 'embeddings': embeddings}

    @torch.no_grad()
    def _projection(self, index, name, module, args, output):
        if self._active is None or self._active['index'] != index:
            raise RuntimeError('projection capture outside its active decoder layer')
        if name in self._active:
            raise RuntimeError('duplicate effective projection capture')
        if output.shape[:2] != self._mask.shape:
            raise ValueError('effective projection must cover the complete batch-one prefix')
        self._active[name] = output.detach()

    @torch.no_grad()
    def _after_layer(self, index, module, args, output):
        try:
            if self._active is None or any(name not in self._active for name in ('q', 'k')):
                raise RuntimeError('missing effective projection capture')
            self._sum.add_(self._reduce(module.self_attn, self._active))
            self._captured.add(index)
        finally:
            self._active = None

    def _reduce(self, attention, captured):
        length, dim = self._mask.shape[1], attention.head_dim
        q = captured['q'].view(1, length, attention.num_heads, dim).transpose(1, 2)
        k = captured['k'].view(1, length, attention.num_key_value_heads, dim).transpose(1, 2)
        q, k = apply_multimodal_rotary_pos_emb(
            q, k, *captured['embeddings'], attention.rope_scaling['mrope_section'])
        q, k = q[0], k[0]
        device = q.device
        queries = torch.tensor(self.query_positions, device=device)
        latents = torch.tensor(self.latent_positions, device=device)
        scores = torch.zeros_like(self._sum)
        for hs in range(0, attention.num_heads, self.config.head_chunk_size):
            heads = torch.arange(hs, min(hs + self.config.head_chunk_size,
                                       attention.num_heads), device=device)
            kv_heads = heads // attention.num_key_value_groups
            for qs in range(0, len(queries), self.config.query_chunk_size):
                positions = queries[qs:qs + self.config.query_chunk_size]
                query = q[heads[:, None], positions[None, :]].float()
                normalizer = torch.full(query.shape[:2], -torch.inf, device=device)

                def logits(keys):
                    key = k[kv_heads[:, None], keys[None, :]].float()
                    value = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dim)
                    valid = (keys[None, :] <= positions[:, None]) & self._mask[0, keys].bool()
                    return value.masked_fill(~valid[None, :, :], -torch.inf)

                # Streaming log-sum-exp includes every valid causal key, not just latents.
                for ks in range(0, length, self.config.key_chunk_size):
                    keys = torch.arange(ks, min(ks + self.config.key_chunk_size, length),
                                        device=device)
                    normalizer = torch.logaddexp(normalizer, torch.logsumexp(logits(keys), dim=-1))
                for ls in range(0, len(latents), self.config.key_chunk_size):
                    keys = latents[ls:ls + self.config.key_chunk_size]
                    probability = torch.exp(logits(keys) - normalizer[:, :, None])
                    scores[ls:ls + len(keys)] += probability.sum((0, 1))
        scores /= attention.num_heads * len(queries)
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError('history attention scores must be finite')
        return scores

    def scores(self) -> Tensor:
        if self._failed:
            raise RuntimeError('failed forward cannot provide history attention captures')
        if self.query_positions and self.latent_positions and self._captured != set(self.layer_indices):
            raise RuntimeError('missing native history attention layer captures')
        return self._sum.clone() / len(self.layer_indices)
