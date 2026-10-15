"""Bounded deterministic Stage1-v5 causal diagnostics (never a training mode)."""
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import random

import torch


def select_rows(source, *, count=None, seed=42, indices_path=None, expected_sha256=None):
    source = Path(source)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('source sha256 mismatch')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    indices = (json.loads(Path(indices_path).read_text()) if indices_path else
               sorted(random.Random(seed).sample(range(len(rows)), count)))
    if (not indices or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('invalid source indices')
    return [rows[i] for i in indices], dict(source=str(source.resolve()), source_sha256=digest,
        source_count=len(rows), source_indices=indices, seed=seed,
        selection='fixed_manifest' if indices_path else 'uniform_without_replacement_sorted')


def bootstrap_mean(values, *, seed=42, repetitions=10000):
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values)))/len(values) for _ in range(repetitions))
    return [means[int(.025*repetitions)], means[min(int(.975*repetitions), repetitions-1)]]


def block_metrics(clean, off, target_ids, block_counts, *, chunk_size=128):
    if (clean.shape != off.shape or clean.ndim != 2 or sum(block_counts) != len(target_ids)
            or len(target_ids) != clean.shape[0] or chunk_size <= 0 or any(n < 0 for n in block_counts)):
        raise ValueError('unaligned diagnostic logits/targets/blocks')
    values = {k: [] for k in ('ce_clean', 'ce_off', 'delta_ce', 'kl_clean_off',
                              'entropy_clean_top50', 'entropy_off_top50')}
    for start in range(0, len(target_ids), chunk_size):
        c, v = clean[start:start+chunk_size].float(), off[start:start+chunk_size].float()
        lp, lq = c.log_softmax(-1), v.log_softmax(-1)
        targets = torch.tensor(target_ids[start:start+chunk_size], device=c.device)[:, None]
        ce_c, ce_v = -lp.gather(1, targets)[:, 0], -lq.gather(1, targets)[:, 0]
        parts = dict(ce_clean=ce_c, ce_off=ce_v, delta_ce=ce_v-ce_c,
                     kl_clean_off=(lp.exp()*(lp-lq)).sum(-1))
        for name, logits in (('clean', c), ('off', v)):
            top = logits.topk(min(50, logits.shape[-1]), dim=-1).values.log_softmax(-1)
            parts[f'entropy_{name}_top50'] = -(top.exp()*top).sum(-1)
        for key, tensor in parts.items():
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError('nonfinite causal metric')
            values[key].extend(tensor.detach().cpu().tolist())
    blocks, offset = [], 0
    for n in block_counts:
        blocks.append(dict(target_count=n, **{k: sum(v[offset:offset+n])/n if n else None
                                             for k, v in values.items()}))
        offset += n
    active = [b for b in blocks if b['target_count']]
    return dict(blocks=blocks, original_blocks=len(blocks), eligible_blocks=len(active),
        empty_blocks=len(blocks)-len(active), target_count=len(target_ids),
        exclusion=None if active else 'no_eligible_targets',
        sample={k: sum(b[k] for b in active)/len(active) for k in values} if active else None,
        token_weighted={k: sum(v)/len(v) for k, v in values.items()} if active else None,
        positive_delta_token_fraction=(sum(x > 0 for x in values['delta_ce'])/len(target_ids)
                                       if active else None),
        positive_delta_block_fraction=(sum(b['delta_ce'] > 0 for b in active)/len(active)
                                       if active else None))


class _AnalysisExecution:
    debug = True

    def checkpoint_contexts(self, role, unit):
        if torch.is_grad_enabled():
            raise RuntimeError('analysis requires no_grad')
        return nullcontext(), nullcontext()

    def dropout_mask(self, *args, **kwargs):
        raise RuntimeError('analysis must never execute dropout')


@contextmanager
def analysis_execution(model):
    if torch.is_grad_enabled():
        raise RuntimeError('analysis requires no_grad')
    if any(module.training for module in model.modules()):
        raise RuntimeError('analysis requires every module in eval mode')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('analysis rejects existing gradients')
    yield _AnalysisExecution()


def student_sdpa_policy(device):
    return 'efficient_only' if torch.device(device).type == 'cuda' else 'automatic'


def student_sdpa_context(device):
    if student_sdpa_policy(device) == 'efficient_only':
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    return nullcontext()


def replay_selected(model, model_inputs, metadata, *, check_unmasked=False,
                    atol=.02, rtol=.02, chunk_size=128):
    from peft_common import resolve_qwen2_5_vl_model
    from transformers.models.qwen2_5_vl.stage1_v5_trace import bind_v5_execution, capture_v5_trace
    from transformers.models.qwen2_5_vl.stage1_v5_replay import V5ReplayBlock, run_v5_packed_replay
    qwen = resolve_qwen2_5_vl_model(model)
    keys = metadata.query_keys
    inputs = {k: v for k, v in model_inputs.items() if k != 'labels'}
    inputs.update(use_cache=False, return_dict=True, stage1_selected_positions=keys,
                  stage1_selected_chunk_positions=chunk_size)
    student_device = next(qwen.parameters()).device
    with analysis_execution(model) as execution, student_sdpa_context(student_device):
        with bind_v5_execution(qwen.model, execution, visual=qwen.visual), capture_v5_trace(qwen, (), ()) as trace:
            output = model(**inputs)
        clean = output.stage1_selected_logits
        blocks = tuple(V5ReplayBlock(b.batch_index, b.start, b.end, b.target_positions,
                       b.target_ids, tuple(() for _ in b.target_positions)) for b in metadata.blocks)
        replay = run_v5_packed_replay(qwen.model, qwen.lm_head, trace, blocks, execution=execution,
            need_voff=True, need_textoff=check_unmasked, chunk_positions=chunk_size)
        if replay.query_keys != keys:
            raise ValueError('replay query ordering differs from native selected logits')
        drift = None
        if check_unmasked and keys:
            oracle = replay.textoff_logits
            # Fixed before real BF16 probes; report absolute and normalized drift.
            drift = dict(passed=bool(torch.allclose(clean, oracle, atol=atol, rtol=rtol)),
                student_sdpa_policy=student_sdpa_policy(student_device),
                atol=atol, rtol=rtol, max_abs=float((clean.float()-oracle.float()).abs().max()),
                mean_abs=float((clean.float()-oracle.float()).abs().mean()),
                query_keys=[list(k) for k in keys], dtype=str(clean.dtype))
            no_op = block_metrics(clean, oracle, metadata.target_ids,
                                  [len(b.target_ids) for b in metadata.blocks], chunk_size=chunk_size)
            drift['no_op_sample'] = no_op['sample']
            drift['no_op_token_weighted'] = no_op['token_weighted']
            if not drift['passed']:
                raise ValueError(f'unmasked replay differs from clean: {drift}')
    return clean, replay.voff_logits if keys else clean, drift


def _indexed(rows):
    rows = [r for r in rows if not r.get('summary')]
    indexed = {r['index']: r for r in rows}
    if len(rows) != len(indexed):
        raise ValueError('duplicate coverage indices')
    return indexed


def verify_baseline(saved, spot, source_indices):
    saved, spot = _indexed(saved), _indexed(spot)
    if set(spot) != set(range(len(source_indices))):
        raise ValueError('baseline spot coverage mismatch')
    fields = ('prompt_identity', 'generated_suffix_token_ids', 'prediction_normalized',
              'prediction_parse_status', 'gold_parse_status', 'gold_normalized', 'match')
    for local, source in enumerate(source_indices):
        if source not in saved or saved[source].get('error') or spot[local].get('error'):
            raise ValueError('baseline spot coverage/error')
        for key in fields:
            if key not in saved[source] or saved[source][key] != spot[local].get(key):
                raise ValueError(f'baseline mismatch source={source} field={key}')


def pair_generation(on_rows, off_rows, source_indices, *, seed=42):
    on, off = _indexed(on_rows), _indexed(off_rows)
    if set(on) != set(source_indices) or set(off) != set(source_indices):
        raise ValueError('generation coverage mismatch')
    off_cache_modes = [r.get('use_kv_cache', False) for r in off.values()]
    if (any(type(mode) is not bool for mode in off_cache_modes)
            or len(set(off_cache_modes)) != 1):
        raise ValueError('OFF cache execution mode must be a uniform boolean')
    if any(type(r.get('use_kv_cache', False)) is not bool or r.get('use_kv_cache', False)
           for r in on.values()):
        raise ValueError('ON cache execution is unsupported')
    invalid, errors, effects, pairs = [], [], [], []
    for i in source_indices:
        a, b = on[i], off[i]
        for key in ('model_identity', 'source_sha256', 'generation_profile'):
            if not a.get(key) or a[key] != b.get(key):
                raise ValueError(f'generation identity mismatch: {key} source={i}')
        if a.get('disable_latent_generation', False) or b.get('disable_latent_generation') is not True:
            raise ValueError('generation condition mismatch')
        if a.get('error') or b.get('error'):
            errors.append(i)
            continue
        if a.get('prompt_identity') is None or a['prompt_identity'] != b.get('prompt_identity'):
            raise ValueError(f'prompt identity mismatch source={i}')
        if (b.get('latent_start_count') != 0 or b.get('latent_block_count') != 0
                or any(t in (151665, 151666, 151667) for t in b['generated_suffix_token_ids'])):
            raise ValueError('OFF contains latent execution/control tokens')
        if (a.get('gold_parse_status'), a.get('gold_normalized')) != (b.get('gold_parse_status'), b.get('gold_normalized')):
            raise ValueError('gold identity mismatch')
        if a.get('gold_parse_status') != 'parsed':
            invalid.append(i)
            continue
        effects.append(int(b['match'])-int(a['match']))
        pairs.append(i)
    count = len(source_indices)
    valid = [i for i in source_indices if on[i].get('gold_parse_status') == 'parsed']
    correct = lambda rows, ids: sum(bool(rows[i].get('match')) and not rows[i].get('error')
                                   and rows[i].get('gold_parse_status') == 'parsed' for i in ids)
    return dict(source_indices=source_indices, source_count=count, valid_gold_count=len(valid),
        off_use_kv_cache=off_cache_modes[0],
        legacy_on_accuracy=correct(on, source_indices)/count,
        legacy_off_accuracy=correct(off, source_indices)/count,
        valid_gold_on_accuracy=correct(on, valid)/len(valid) if valid else None,
        valid_gold_off_accuracy=correct(off, valid)/len(valid) if valid else None,
        inferential_pairs=len(pairs), excluded_invalid_gold=invalid, excluded_errors=errors,
        on_correct_off_wrong=effects.count(-1), on_wrong_off_correct=effects.count(1),
        answer_changes=sum(on[i].get('prediction_normalized') != off[i].get('prediction_normalized') for i in pairs),
        effect_off_minus_on=sum(effects)/len(effects) if effects else None,
        paired_bootstrap_ci=bootstrap_mean(effects, seed=seed), bootstrap_seed=seed,
        paired_effects={str(i): e for i, e in zip(pairs, effects)},
        suffix_equal_count=sum(on[i]['generated_suffix_token_ids'] == off[i]['generated_suffix_token_ids'] for i in pairs),
        lengths={label: [len(rows[i]['generated_suffix_token_ids']) for i in source_indices]
                 for label, rows in (('on', on), ('off', off))},
        stops={label: {stop: sum(rows[i].get('stop_reason') == stop for i in source_indices)
                      for stop in sorted(set(str(rows[i].get('stop_reason')) for i in source_indices))}
               for label, rows in (('on', on), ('off', off))})


def summarize_voff(rows, source_indices, *, seed=42):
    indexed = {r['source_index']: r for r in rows}
    if len(rows) != len(indexed) or set(indexed) != set(source_indices):
        raise ValueError('V-off coverage mismatch')
    active = [r for r in rows if r.get('sample') is not None and not r.get('error')]
    names = tuple(active[0]['sample']) if active else ()
    total_tokens = sum(r['target_count'] for r in active)
    blocks = [b for r in active for b in r['blocks'] if b['target_count']]
    return dict(source_count=len(source_indices), eligible_samples=len(active),
        exclusions=[dict(source_index=r['source_index'], reason=r.get('error') or r.get('exclusion'))
                    for r in rows if r not in active],
        original_blocks=sum(r.get('original_blocks', 0) for r in rows),
        empty_blocks=sum(r.get('empty_blocks', 0) for r in rows), target_count=total_tokens,
        sample_mean={k: sum(r['sample'][k] for r in active)/len(active) for k in names},
        sample_bootstrap_ci={k: bootstrap_mean([r['sample'][k] for r in active], seed=seed) for k in names},
        token_weighted={k: sum(r['token_weighted'][k]*r['target_count'] for r in active)/total_tokens for k in names},
        positive_delta_sample_fraction=sum(r['sample']['delta_ce'] > 0 for r in active)/len(active) if active else None,
        positive_delta_token_fraction=(sum(r['positive_delta_token_fraction']*r['target_count'] for r in active)/total_tokens
                                       if total_tokens else None),
        positive_delta_block_fraction=sum(b['delta_ce'] > 0 for b in blocks)/len(blocks) if blocks else None,
        paired_sample_metrics={str(r['source_index']): r['sample'] for r in active},
        bootstrap_seed=seed)


def prepare_analysis_inputs(teacher, processor, batch, saved_execution, device):
    """Reuse the real helper target method without constructing a Trainer lifecycle."""
    from stage1.targets import Stage1TeacherTargetBuilder
    from stage1.trainer import CustomTrainerStage1
    with analysis_execution(teacher.model):
        carrier = CustomTrainerStage1.__new__(CustomTrainerStage1)
        carrier.ema_teacher = teacher
        carrier._ema_execution_device = torch.device(device)
        carrier.stage1_v5_debug = True
        carrier.target_builder = Stage1TeacherTargetBuilder(processor.tokenizer,
            coverage_p=saved_execution['coverage_p'], image_pool_k=saved_execution['image_pool_k'],
            helper_group_L=saved_execution['helper_group_L'], textual_states=saved_execution['textual_states'],
            validate=True, collect_diagnostics=False)
        metadata = batch['stage1_v5_metadata']
        working = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
                   if k not in ('stage1_v5_metadata', 'stage1_v5_window')}
        inputs, _, _ = carrier._prepare_stage1_loss_inputs(working)
    return inputs, metadata
