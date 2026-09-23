"""Opt-in first-micro native representation inspection; never an acceptance gate."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading

import torch

from stage2.until_alignment_probe import AlignmentObserver


class RepresentationInspectionStop(RuntimeError):
    """The preference read finished; total/ordinary backward must not execute."""


def effective_dense_mask(mask, is_causal, *, queries, keys):
    if mask is not None:
        return mask.detach().cpu().clone()
    if is_causal:
        return torch.arange(queries)[:, None] >= torch.arange(keys)[None, :]
    return torch.ones((queries, keys), dtype=torch.bool)


def effective_block_mask(mask, *, batch, head, queries, keys):
    """Resolve sparse full/partial tiles AND the actual token predicate."""
    qblock, kblock = mask.BLOCK_SIZE
    nq, nk = (queries + qblock - 1)//qblock, (keys + kblock - 1)//kblock
    def tiles(counts, indices):
        dense = torch.zeros((nq, nk), dtype=torch.bool)
        if counts is None:
            return dense
        counts, indices = counts.detach().cpu(), indices.detach().cpu()
        b, h = batch % counts.shape[0], head % counts.shape[1]
        for q in range(nq):
            selected = indices[b, h, q, :int(counts[b, h, q])].long()
            selected = selected[selected < nk]
            dense[q, selected] = True
        return dense.repeat_interleave(qblock, 0).repeat_interleave(kblock, 1)[:queries, :keys]
    partial = tiles(mask.kv_num_blocks, mask.kv_indices)
    full = tiles(mask.full_kv_num_blocks, mask.full_kv_indices)
    device = mask.kv_indices.device
    with torch.no_grad():
        predicate = mask.mask_mod(torch.tensor(batch, device=device), torch.tensor(head, device=device),
            torch.arange(queries, device=device)[:, None], torch.arange(keys, device=device)[None, :])
    if predicate.dtype != torch.bool:
        raise ValueError('unsupported non-boolean BlockMask predicate')
    return full | (partial & predicate.detach().cpu().expand(queries, keys))


class RepresentationTrace:
    """Hashes only: no activation or autograd graph survives a boundary call."""
    def __init__(self, directory, *, rank):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank, self.phase = rank, 'forward'
        self.local = threading.local()
        self.records, self.restore, self.handles = [], [], []
        self.counts = Counter()
        self.expected_latents = {}
        self.layers = None
        self.errors = []
        self.stream = (self.directory/f'representations-rank-{rank}.jsonl').open('x')
        self.closed = False

    @property
    def invocation(self):
        return getattr(self.local, 'invocation', 'root')

    @contextmanager
    def scope(self, invocation):
        previous = self.invocation
        self.local.invocation = invocation
        try:
            yield
        finally:
            self.local.invocation = previous

    def record(self, boundary, tensor, *, category=None):
        if not isinstance(tensor, torch.Tensor):
            self.errors.append(f'unsupported tensor at {self.invocation}/{boundary}')
            return
        value = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        key = self.phase, self.invocation, boundary
        row = dict(phase=self.phase, invocation=self.invocation, boundary=boundary,
            occurrence=self.counts[key], category=category, shape=list(value.shape),
            dtype=str(value.dtype), sha256=digest)
        if value.is_floating_point():
            row['nonfinite'] = dict(nan=int(torch.isnan(value).sum()),
                positive_inf=int(torch.isposinf(value).sum()), negative_inf=int(torch.isneginf(value).sum()))
            if any(row['nonfinite'].values()) and category != 'effective_mask':
                self.errors.append(f'nonfinite activation {self.phase}/{self.invocation}/{boundary}')
        self.counts[key] += 1
        self.records.append(row)
        self.stream.write(json.dumps(row)+'\n')
        self.stream.flush()

    def bind_latent_decoder(self, decoder, *, slot, position):
        identity = f'latent/{slot}/position-{position}'
        def bound(*args, **kwargs):
            with self.scope(identity):
                result = decoder(*args, **kwargs)
                hidden = getattr(result, 'last_hidden_state', None)
                if hidden is not None:
                    self.record('latent_prediction', hidden.detach()[:, -1, :], category='generated_latent')
                return result
        return bound

    def install(self, base, source):
        """Observe the supplied no-cache/V-only path without modifying its outputs."""
        from torch.nn import functional as F
        from transformers.cache_utils import DynamicCache
        self.layers = len(base.model.layers)

        def patch(owner, name, replacement):
            previous, own = getattr(owner, name), name in vars(owner)
            setattr(owner, name, replacement)
            self.restore.append(lambda: setattr(owner, name, previous) if own else delattr(owner, name))

        def output_hook(boundary, category):
            def capture(module, args, result):
                if result is not None:
                    self.record(boundary, result, category=category)
            return capture
        for obj, boundary, category in (
            (base.visual, 'vision/output', 'vision'),
            (base.visual.merger, 'vision/merger/output', 'merger'),
            (base.model.embed_tokens, 'token_embeddings/output', 'embedding'),
            (base.lm_head, 'lm_head/full_output', 'lm_head'),
        ):
            self.handles.append(obj.register_forward_hook(output_hook(boundary, category)))

        def decoder_enter(module, args, kwargs):
            previous = self.invocation
            stack = getattr(self.local, 'decoder_stack', [])
            stack.append(previous)
            self.local.decoder_stack = stack
            if previous == 'root':
                self.local.invocation = 'clean'
            self.record('combined_embeddings', kwargs.get('inputs_embeds'), category='combined_embeddings')
            self.record('position_ids', kwargs.get('position_ids'))
        def decoder_exit(module, args, result):
            if result is not None:
                self.record('decoder/output', result.last_hidden_state)
            self.local.invocation = self.local.decoder_stack.pop()
        self.handles.append(base.model.register_forward_pre_hook(decoder_enter, with_kwargs=True))
        self.handles.append(base.model.register_forward_hook(decoder_exit, always_call=True))
        for index, layer in enumerate(base.model.layers):
            def layer_enter(module, args, kwargs, index=index):
                self.record(f'layer-{index}/input', args[0] if args else kwargs.get('hidden_states'))
            def layer_exit(module, args, result, index=index):
                if result is not None:
                    self.record(f'layer-{index}/output', result[0], category='layer_output')
            def attention_enter(module, args, kwargs, index=index):
                self.local.attention_layer = index
                if kwargs.get('position_ids') is not None:
                    self.record(f'layer-{index}/attention/position_ids', kwargs['position_ids'])
                if kwargs.get('cache_position') is not None:
                    self.record(f'layer-{index}/attention/cache_position', kwargs['cache_position'])
            def attention_exit(module, args, result):
                self.local.attention_layer = None
            self.handles += [layer.register_forward_pre_hook(layer_enter, with_kwargs=True),
                layer.register_forward_hook(layer_exit),
                layer.self_attn.register_forward_pre_hook(attention_enter, with_kwargs=True),
                layer.self_attn.register_forward_hook(attention_exit, always_call=True)]

        original_sdpa = F.scaled_dot_product_attention
        def sdpa(query, key, value, *args, **kwargs):
            mask = kwargs.get('attn_mask', args[0] if args else None)
            causal = kwargs.get('is_causal', args[2] if len(args)>2 else False)
            if getattr(self.local, 'attention_layer', None) is not None:
                self.capture_attention(query, key, value, mask, is_causal=causal)
            return original_sdpa(query, key, value, *args, **kwargs)
        patch(F, 'scaled_dot_product_attention', sdpa)
        registry = source.model.ALL_ATTENTION_FUNCTIONS
        original_flex = registry['flex_attention']
        def flex(module, query, key, value, attention_mask, *args, **kwargs):
            self.capture_attention(query, key, value, attention_mask, flex=True)
            return original_flex(module, query, key, value, attention_mask, *args, **kwargs)
        patch(source.model, 'ALL_ATTENTION_FUNCTIONS', dict(registry, flex_attention=flex))

        original_update = DynamicCache.update
        def update(cache, key, value, layer_idx, *args, **kwargs):
            result = original_update(cache, key, value, layer_idx, *args, **kwargs)
            where = 'attention' if getattr(self.local, 'attention_layer', None) is not None else 'restore'
            self.record(f'cache/{where}/layer-{layer_idx}/key', result[0], category='cache_k')
            self.record(f'cache/{where}/layer-{layer_idx}/value', result[1], category='cache_v')
            return result
        patch(DynamicCache, 'update', update)

        original_rollout = base.stage2_rollout_engine.rollout
        def rollout(**kwargs):
            for block, span in enumerate(kwargs['latent_spans']):
                for slot, position in enumerate(span.latent_positions):
                    if position in self.expected_latents:
                        raise ValueError('ambiguous native latent position identity')
                    self.expected_latents[position] = f'latent/block-{block}/latent-{slot}/position-{position}'
            return original_rollout(**kwargs)
        patch(base.stage2_rollout_engine, 'rollout', rollout)
        original_prefix = source.rollout._run_latent_prefix
        def prefix(language_model, inputs, mask, positions, stats, checkpointing):
            position = inputs.shape[1]
            identity = self.expected_latents.get(position)
            if identity is None:
                raise ValueError('native checkpoint has no parsed latent-slot identity')
            slot = identity.removeprefix('latent/').rsplit('/position-',1)[0]
            decoder = self.bind_latent_decoder(language_model, slot=slot, position=position)
            return original_prefix(decoder, inputs, mask, positions, stats, checkpointing)
        patch(source.rollout, '_run_latent_prefix', prefix)
        original_sidecar = source.model.execute_stage2_sidecar_groups
        def sidecar(**kwargs):
            layouts = kwargs['layouts']
            if len(layouts)!=1 or layouts[0].modality!='visual':
                raise ValueError('representation inspection requires one native visual sidecar group')
            with self.scope('sidecar/visual'):
                return original_sidecar(**kwargs)
        patch(source.model, 'execute_stage2_sidecar_groups', sidecar)

    def capture_attention(self, query, key, value, mask, *, is_causal=False, flex=False):
        layer = getattr(self.local, 'attention_layer', None)
        if layer is None:
            self.errors.append('attention backend called without decoder-layer identity')
            return
        prefix = f'layer-{layer}/attention'
        for name, tensor in (('query',query), ('key',key), ('value',value)):
            self.record(f'{prefix}/{name}', tensor, category={'key':'attention_k','value':'attention_v'}.get(name))
        q, k = query.shape[-2], key.shape[-2]
        try:
            if flex:
                from torch.nn.attention.flex_attention import BlockMask
                if not isinstance(mask, BlockMask):
                    raise ValueError('native sidecar Flex did not supply BlockMask')
                for b in range(query.shape[0]):
                    for h in range(query.shape[1]):
                        effective = effective_block_mask(mask, batch=b, head=h, queries=q, keys=k)
                        self.record(f'{prefix}/effective_mask', effective, category='effective_mask')
            else:
                self.record(f'{prefix}/effective_mask', effective_dense_mask(mask,is_causal,queries=q,keys=k),
                            category='effective_mask')
        except (ValueError, TypeError, IndexError, RuntimeError) as error:
            self.errors.append(f'unsupported effective mask {self.invocation}/{prefix}: {error}')

    def close(self):
        if self.closed:
            return
        for handle in reversed(self.handles):
            handle.remove()
        for restore in reversed(self.restore):
            restore()
        required = {'vision', 'merger', 'embedding', 'combined_embeddings', 'generated_latent',
                    'layer_output', 'attention_k', 'attention_v', 'effective_mask', 'cache_k', 'cache_v', 'lm_head'}
        observed = {r['category'] for r in self.records}
        missing = sorted(required - observed) + self.errors
        originals = {(r['invocation'],r['boundary'],r['occurrence']):r for r in self.records if r['phase']=='forward'}
        replay = [r for r in self.records if r['phase']=='preference_backward']
        compared, differences = 0, []
        for row in replay:
            key = row['invocation'],row['boundary'],row['occurrence']
            original = originals.get(key)
            if original is None:
                missing.append(f'unmatched replay {key}')
            else:
                compared += 1
                if any(original[k] != row[k] for k in ('shape','dtype','sha256')):
                    differences.append({'invocation':row['invocation'],'boundary':row['boundary']})
        if self.layers is None:
            missing.append('decoder coverage was not configured')
        else:
            for invocation in ['clean', 'sidecar/visual', *self.expected_latents.values()]:
                phases = ('forward','preference_backward') if invocation.startswith('latent/') else ('forward',)
                for phase in phases:
                    for layer in range(self.layers):
                        for boundary in ('output','attention/key','attention/value','attention/effective_mask'):
                            name = f'layer-{layer}/{boundary}'
                            if not any(r['phase']==phase and r['invocation']==invocation and r['boundary']==name for r in self.records):
                                missing.append(f'{phase}/{invocation}/{name}')
        summary = dict(rank=self.rank, record_count=len(self.records), missing_coverage=missing,
            coverage_complete=not missing, optimizer_updates=getattr(self, 'optimizer_updates', None),
            within_run_recompute=dict(compared=compared, differences=differences),
            scope='first native forward and preference gradient only; hashes cover observed tensor contents')
        self.stream.close()
        (self.directory/f'representation-summary-rank-{self.rank}.json').write_text(json.dumps(summary,indent=2)+'\n')
        self.closed = True


class RepresentationObserver(AlignmentObserver):
    def observe(self, total, preference, inputs, scores):
        self.micro += 1
        if self.micro != 0 or list(inputs['sample_ids']) != [self.replay['sample_ids'][self.rank]]:
            raise ValueError('representation inspection permits only first microbatch')
        self.save('micro-0-scores', scores)
        self.save('micro-0-rng', {'cpu':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state(self.engine.device)})
        self.representation_trace.phase = 'preference_backward'
        values = self.gradients(preference)
        self.save('micro-0-preference', values)  # Original strict prefix guard remains attached here.
        raise RepresentationInspectionStop('first preference read captured; no optimizer update permitted')


def finish_representation_inspection(trace, accelerator, error):
    trace.close()  # Flush and restore before an expected stop can terminate peers.
    expected = isinstance(error, RepresentationInspectionStop) or (
        isinstance(error, ValueError) and 'native prefix tensor differs: micro-0-preference/' in str(error))
    if expected:
        accelerator.wait_for_everyone()
    raise error


def compare_representations(left, right, *, ranks=range(4)):
    missing, first, comparisons, within = [], None, 0, []
    for rank in ranks:
        directories = (Path(left), Path(right))
        inventories = []
        for directory in directories:
            summary = json.loads((directory/f'representation-summary-rank-{rank}.json').read_text())
            missing.extend(f'{directory.name}/rank-{rank}/{m}' for m in summary['missing_coverage'])
            within.extend(dict(directory=str(directory), rank=rank, **difference)
                          for difference in summary['within_run_recompute']['differences'])
            rows = [json.loads(line) for line in (directory/f'representations-rank-{rank}.jsonl').read_text().splitlines()]
            inventories.append({(r['phase'],r['invocation'],r['boundary'],r['occurrence']):r for r in rows})
        a,b = inventories
        missing.extend(f'rank-{rank}/missing-boundary/{key}' for key in a.keys() ^ b.keys())
        for key in a.keys() & b.keys():
            comparisons += 1
        for key,row in a.items():
            if key in b and any(row[k]!=b[key][k] for k in ('shape','dtype','sha256')) and first is None:
                first = dict(rank=rank, phase=row['phase'], invocation=row['invocation'], boundary=row['boundary'])
    return dict(all_equal=not missing and not within and first is None, first_mismatch=first,
                missing_coverage=missing, compared_boundaries=comparisons,
                within_run_recompute_differences=within)
