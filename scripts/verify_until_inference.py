#!/usr/bin/env python
"""Two fixed samples, final Stage2 weights, actual local/external inference paths."""
import argparse
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict
import importlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from scripts import verify_until_native as native

CE_ROOT = Path('/root/workspace/ILVR/.worktrees/ce-block-analysis-20260921')
EXTERNAL = '/raid/workspace/temp/ILVR/.worktrees/ours1'
MANIFEST = CE_ROOT/'data/ce-vision-block-analysis/until-b05-block-main100-20260922-141735/manifest.json'
ARTIFACT = '/root/workspace/ILVR/checkpoints/evaluation-staging/voff-until-mask-b05-gb16-20260919-094938/prepared-checkpoint-212'


def ce_source(root):
    source = native.ExternalSource(root)
    source.mapping = {'evaluation': source.prefix + '_evaluation'}
    source.folders = {source.mapping['evaluation']: Path(root)/'src/evaluation'}
    return source


def external_modules(source):
    return SimpleNamespace(model=source.module('transformers.models.qwen2_5_vl.modeling_qwen2_5_vl'),
        data=source.module('stage2.data'), spans=source.module('step_preference.spans'),
        rollout=source.module('transformers.models.qwen2_5_vl.stage2_latent_rollout'),
        factorial=source.module('stage2.context_factorial'),
        runner=source.module('stage2.context_factorial_runner'),
        ablation=source.module('stage2.context_ablation_runner'))


def checkpoint_tensors(restored, saved):
    """Compare actual loaded values to artifact values, showing every dtype cast."""
    import torch
    result = {}
    for name in sorted(restored.keys() | saved.keys()):
        actual, expected = restored.get(name), saved.get(name)
        match = (actual is not None and expected is not None and actual.shape == expected.shape
                 and torch.equal(actual.detach().cpu(), expected.to(dtype=actual.dtype, device='cpu')))
        result[name] = {'equal_after_explicit_cast': match,
            'checkpoint_dtype': None if expected is None else str(expected.dtype),
            'restored_dtype': None if actual is None else str(actual.dtype),
            'checkpoint': native.tensor_record(expected), 'restored': native.tensor_record(actual)}
    return {'all_equal_after_explicit_cast': bool(result) and all(v['equal_after_explicit_cast'] for v in result.values()),
            'tensors': result}


def artifact_witness(model, artifact):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from stage1.token_rows import find_sparse_output_token_rows
    adapter = checkpoint_tensors(get_peft_model_state_dict(model, save_embedding_layers=False),
                                 load_file(str(Path(artifact)/'adapter_model.safetensors')))
    rows = find_sparse_output_token_rows(model)
    saved = load_file(str(Path(artifact)/'ilvr_token_rows.safetensors'))
    import torch
    actual = {'output_delta': rows.output_delta,
              'token_ids': torch.tensor(tuple(rows.token_ids), dtype=torch.int64)}
    sparse = checkpoint_tensors(actual, saved)
    return {'adapter_including_input_rows': adapter, 'sparse_output_rows': sparse,
            'all_restored': adapter['all_equal_after_explicit_cast'] and sparse['all_equal_after_explicit_cast']}


def selected_indices(manifest):
    indices = [record['source_index'] for record in manifest['samples'][:2]]
    if len(indices) != 2 or len(set(indices)) != 2:
        raise ValueError('diagnostic requires first two distinct manifest samples')
    return indices


def five_metrics(clean, changed, targets):
    import torch
    if not clean or len(clean) != len(changed) or len(clean) != len(targets):
        raise ValueError('metrics require aligned nonempty sites')
    first = any_count = changed_tokens = total = 0
    nll = clean_nll = 0.
    for left, right, target in zip(clean, changed, targets):
        if left.shape != right.shape or left.shape[0] != target.numel() or not target.numel():
            raise ValueError('unaligned target/logit geometry')
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            raise ValueError('nonfinite logits')
        difference = left.argmax(-1) != right.argmax(-1)
        first += int(difference[0]); any_count += int(difference.any())
        changed_tokens += int(difference.sum()); total += target.numel()
        index = target.cpu().long()[:, None]
        clean_nll += float(-left.float().cpu().log_softmax(-1).gather(-1, index).sum())
        nll += float(-right.float().cpu().log_softmax(-1).gather(-1, index).sum())
    return dict(S=len(clean), N=total, first_changed=first, any_changed=any_count,
                tokens_changed=changed_tokens, mean_nll=nll/total, delta_nll=(nll-clean_nll)/total)


def preflight(args):
    if not args.device.startswith('cuda'):
        raise ValueError('production diagnostic requires CUDA; CPU guards run before loading')
    for path in (args.external_source, args.ce_root, args.artifact, args.base, args.manifest, args.data, args.image_root):
        if not Path(path).exists():
            raise ValueError(f'missing input: {path}')
    if not (Path(args.artifact)/'stage2_peft_metadata.json').is_file():
        raise ValueError('requires genuine final Stage2 PEFT metadata')
    selected_indices(json.loads(Path(args.manifest).read_text()))
    root = Path(__file__).resolve().parents[1]
    for path, tree in ((args.output_dir, root/'results'), (args.log_dir, root/'logs')):
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(tree.resolve()) or resolved == tree.resolve():
            raise ValueError(f'output must be below {tree}')
        if resolved.exists() and any(resolved.iterdir()):
            raise ValueError(f'use a fresh empty output directory: {resolved}')


def diagnostic_model_load_kwargs(*, external):
    import torch
    options = {'local_files_only': True, 'torch_dtype': torch.bfloat16}
    if external:
        options['stage2_split_attention'] = True
    return options


def load_models(args, processor, external):
    import torch
    from stage2.peft import load_stage2_model_for_inference
    from stage2.attention_backend import load_qwen2_5_vl_with_attention
    from transformers import Qwen2_5_VLForConditionalGeneration
    models = []
    classes = (
        (Qwen2_5_VLForConditionalGeneration, False),
        (external.model.Qwen2_5_VLForConditionalGeneration, True),
    )
    for cls, external_protocol in classes:
        torch.manual_seed(42)
        def load(source, **kwargs):
            return load_qwen2_5_vl_with_attention(source, attention_implementation='sdpa',
                vision_attention_implementation='flash_attention_2',
                config_load_kwargs={'local_files_only': True},
                model_load_kwargs=diagnostic_model_load_kwargs(
                    external=external_protocol),
                load_model=cls.from_pretrained)
        model = load_stage2_model_for_inference(args.base, args.artifact, processor,
            source_provenance_kwargs={'local_files_only': True}, load_model=load)
        model.eval().requires_grad_(False)
        models.append(model.to(args.device))
    return models


def qwen(model):
    from peft_common import resolve_qwen2_5_vl_model
    return resolve_qwen2_5_vl_model(model)


def native_rollout(model, sample, side, external, runtime):
    """Observe the native ordinary final decoder call without replacing embeddings."""
    import torch
    resolved = qwen(model)
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in sample.model_inputs.items()}
    length = batch['input_ids'].shape[1]
    captured = []
    token_embeddings, image_embeddings, decoder_calls = [], [], []
    latent_ordinal = 0
    def token_output(module, args, output):
        if args and args[0].shape == batch['input_ids'].shape:
            token_embeddings.append(output.detach().clone())
    def image_output(module, args, output):
        image_embeddings.append(output.detach().clone())
    def observe(module, args, kwargs):
        nonlocal latent_ordinal
        value = kwargs.get('inputs_embeds')
        if value is None:
            return
        final = value.shape[1] == length
        call = {'kind': 'final' if final else 'latent_prefix',
                'latent_ordinal': None if final else latent_ordinal,
                'inputs': value.detach().clone(),
                'positions': kwargs['position_ids'].detach().clone(),
                'capture_kv': kwargs.get('stage2_capture_kv', False)}
        latent_ordinal += int(not final)
        decoder_calls.append(call)
        if final:
            captured.append({'embeds': call['inputs'], 'positions': call['positions'],
                             'capture_kv': call['capture_kv']})
    def observe_output(module, args, kwargs, output):
        decoder_calls[-1]['output'] = output.last_hidden_state.detach().clone()
    handles = [
        resolved.model.embed_tokens.register_forward_hook(token_output),
        resolved.visual.register_forward_hook(image_output),
        resolved.model.register_forward_pre_hook(observe, with_kwargs=True),
        resolved.model.register_forward_hook(observe_output, with_kwargs=True),
    ]
    try:
        predictors = sorted({p-1 for site in sample.sites for p in site.target_positions})
        if side == 'local':
            model(**batch, labels=None, stage2_shared_prefix_inputs=None, stage2_selective_logits_only=True,
                  stage2_logit_positions=torch.tensor(predictors, device=device), use_cache=False, return_dict=True)
        else:
            base = resolved.prepare_stage2_base_inputs_embeds(input_ids=batch['input_ids'],
                pixel_values=batch.get('pixel_values'), image_grid_thw=batch.get('image_grid_thw'))
            score = torch.zeros_like(batch['input_ids'], dtype=torch.bool)
            score[0, predictors] = True
            external.ablation._forward_prepared(resolved, dict(batch, prepared_base_inputs_embeds=base), score)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != 1 or captured[0]['capture_kv']:
        raise ValueError('native ordinary final decoder capture was not unique/ordinary')
    if len(token_embeddings) != 1 or any('output' not in call for call in decoder_calls):
        raise ValueError('native trace did not observe one base embedding and complete decoder calls')
    image_expected = batch.get('pixel_values') is not None
    if len(image_embeddings) != int(image_expected):
        raise ValueError('native trace image observation does not match image inputs')
    observed_tensors = [*token_embeddings, *image_embeddings]
    observed_tensors.extend(
        call[key] for call in decoder_calls for key in ('inputs', 'positions', 'output'))
    if any(tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
           for tensor in observed_tensors):
        raise ValueError('native trace contains a nonfinite observation')
    base = token_embeddings[0]
    if image_embeddings:
        image_mask = (batch['input_ids'] == resolved.config.image_token_id).unsqueeze(-1).expand_as(base)
        base = base.masked_scatter(image_mask, image_embeddings[0].to(base.device, base.dtype))
    captured[0]['trace'] = {'token_embeddings': token_embeddings[0],
                            'reconstructed_base_inputs': base,
                            'reconstructed_base_provenance': 'actual token embedding output with actual image output scattered at image-token positions',
                            'image_expected': image_expected,
                            'image_embeddings': image_embeddings[0] if image_embeddings else None,
                            'decoder_calls': decoder_calls,
                            'input_ids': batch['input_ids'].detach().clone()}
    return captured[0]


def common_state(model, bank, sample, runtime):
    """One observed local latent bank becomes the explicit forced-common variable."""
    output = qwen(model).model(inputs_embeds=bank['embeds'], position_ids=bank['positions'],
        attention_mask=sample.model_inputs['attention_mask'].to(bank['embeds'].device),
        generate_mode=True, stage2_capture_kv=True, use_cache=False, return_dict=True)
    return runtime.CleanState(bank['embeds'], bank['positions'], output.stage2_key_values, {}, {})


class Observation:
    """Read-only decoder hooks plus Python attention-return K/V observation."""
    def __init__(self, model, site):
        self.model, self.site = qwen(model), site
        self.positions = sorted(set(range(site.start, site.end+1)) | {p-1 for p in site.target_positions})
        self.layers, self.kv, self.masks, self.dispatch, self.logits = {}, {}, {}, {}, []
        self.handles = []

    def __enter__(self):
        def decoder(module, args, kwargs):
            self.dispatch = {'capture_kv': kwargs.get('stage2_capture_kv', False),
                'prefix_kv': kwargs.get('stage2_prefix_key_values') is not None,
                'generate_mode': kwargs.get('generate_mode'),
                'position_ids': native.tensor_record(kwargs['position_ids']),
                'embeds': native.tensor_record(kwargs['inputs_embeds'])}
        self.handles.append(self.model.model.register_forward_pre_hook(decoder, with_kwargs=True))
        for index, layer in enumerate(self.model.model.layers):
            def hidden(module, args, output, index=index):
                self.layers[index] = output[0][:, self.positions].detach().cpu().clone()
            def mask(module, args, kwargs, index=index):
                value = kwargs.get('attention_mask')
                if value is None:
                    self.masks[index] = {'none': True}
                else:
                    matrix = value.detach()[0,0]
                    key = self.site.body_start
                    self.masks[index] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                        'visible_sha256': native.baseline.tensor_digest(value == 0),
                        'blocked_values': [str(float(v)) for v in matrix[matrix != 0].unique().cpu()],
                        'blocked_rows_for_body_key': (matrix[self.site.end:,key] != 0).nonzero().flatten().add(self.site.end).cpu().tolist()}
            self.handles.append(layer.register_forward_hook(hidden))
            self.handles.append(layer.self_attn.register_forward_pre_hook(mask, with_kwargs=True))
        self.previous = sys.getprofile()
        if self.previous is not None:
            raise RuntimeError('diagnostic needs no existing profiler')
        self.attention_ids = {id(layer.self_attn): i for i, layer in enumerate(self.model.model.layers)}
        self.calls = set()
        def profile(frame, event, arg):
            if event != 'return':
                return
            index = self.attention_ids.get(id(frame.f_locals.get('self')))
            if index is None:
                return
            self.calls.add((index, frame.f_code.co_filename, frame.f_code.co_name))
            key, value = frame.f_locals.get('key_states'), frame.f_locals.get('value_states')
            if key is not None and value is not None and key.ndim == 4 and key.shape[2] > max(self.positions):
                self.kv[index] = (key[:,:,self.positions].detach().cpu().clone(),
                                  value[:,:,self.positions].detach().cpu().clone())
        sys.setprofile(profile)
        return self

    def __exit__(self, *exc):
        sys.setprofile(self.previous)
        for handle in self.handles:
            handle.remove()
        self.dispatch['executed_attention'] = [dict(layer=i, file=f, function=n) for i,f,n in sorted(self.calls)]


def external_items(sample, external):
    import torch
    return tuple(external.data.StepPreferenceBatchItem(sample_id=str(sample.source_index),
        span=external.spans.ReasoningStepSpan(step_id=i, prev_text_start=0, prev_text_end=s.start,
            latent_start=s.body_start, latent_end=s.body_end, target_text_start=s.end+1,
            target_text_end=s.target_positions[-1]+1, helper_image_idx=None),
        chosen_predictor_positions=torch.tensor([p-1 for p in s.target_positions]),
        chosen_target_ids=torch.tensor(s.target_ids)) for i,s in enumerate(sample.sites))


def decode_local(model, state, site, runtime, condition, backend):
    """Use the actual replay body; unwrap its policy only to control the explicit factor."""
    import torch
    captured = []
    def projection(module, args, output):
        captured.append(output.detach().float().cpu())
    hook = qwen(model).lm_head.register_forward_hook(projection)
    observation = Observation(model, site)
    try:
        with observation:
            replay = runtime.replay_site if backend == 'math' else inspect.unwrap(runtime.replay_site)
            replay(model, state, site, condition)
    finally:
        hook.remove()
    return torch.cat(captured), observation


def decode_external(model, state, sample, items, external, condition):
    inputs = sample.model_inputs
    ids = inputs['input_ids'].to(state.embeds.device)
    mask = inputs['attention_mask'].to(state.embeds.device)
    if condition == 'clean':
        branches = external.runner._clean_branches(ids, mask, state.embeds, items)
    else:
        branches = external.factorial.build_length_preserving_branches(input_ids=ids, attention_mask=mask,
            clean_final_inputs_embeds=state.embeds, items=items, scope='body',
            neutral_embedding=state.embeds[0,0])  # unused for body-only masking
    output = external.runner._decode_branches(qwen(model), branches, state.positions.expand(-1,len(items),-1))
    return [value.detach().float().cpu() for value in output]


def score_local_bank(model, sample, state, runtime):
    """Score clean/body against one branch's independently regenerated clean bank."""
    import torch
    targets = [torch.tensor(site.target_ids) for site in sample.sites]
    values = {}
    for condition in ('clean', 'body_mask', 'clean_repeat'):
        intervention = 'clean' if condition == 'clean_repeat' else condition
        values[condition] = [
            decode_local(model, state, site, runtime, intervention, 'math')[0]
            for site in sample.sites]
    report = {}
    for condition, current in values.items():
        metrics = five_metrics(values['clean'], current, targets)
        report[condition] = {
            'five_metrics': metrics,
            'first_end_changed': metrics['first_changed'],
            'logits': comparison(torch.cat(values['clean']), torch.cat(current)),
        }
    return report, {name: torch.cat(values[name]) for name in ('clean', 'body_mask')}


def score_external_bank(model, sample, bank, external):
    """Score the external implementation against its independently regenerated bank."""
    import torch
    state = SimpleNamespace(embeds=bank['embeds'], positions=bank['positions'])
    items = external_items(sample, external)
    values = {}
    for condition in ('clean', 'body_mask', 'clean_repeat'):
        intervention = 'clean' if condition == 'clean_repeat' else condition
        values[condition] = decode_external(
            model, state, sample, items, external, intervention)
    targets = [torch.tensor(site.target_ids) for site in sample.sites]
    report = {}
    for condition, current in values.items():
        metrics = five_metrics(values['clean'], current, targets)
        report[condition] = {
            'five_metrics': metrics,
            'first_end_changed': metrics['first_changed'],
            'logits': comparison(torch.cat(values['clean']), torch.cat(current)),
        }
    return report, {name: torch.cat(values[name]) for name in ('clean', 'body_mask')}


def comparison(left, right):
    return native.compare_tensors(left, right)


@contextmanager
def temporary_vision_rotary(module, substitute):
    """Route one module's vision rotary through a counted temporary callable."""
    original = module.apply_rotary_pos_emb_vision
    witness = {'dispatch_count': 0}
    def observed(*args, **kwargs):
        witness['dispatch_count'] += 1
        return substitute(*args, **kwargs)
    module.apply_rotary_pos_emb_vision = observed
    try:
        yield witness
    finally:
        module.apply_rotary_pos_emb_vision = original


def rotary_first_nail(crossover_trace, external_trace, *, dispatch_count):
    image = comparison(
        crossover_trace.get('image_embeddings'),
        external_trace.get('image_embeddings'))
    passed = dispatch_count > 0 and image.get('exact_equal') is True
    return {'status': 'passed' if passed else 'failed',
            'causal_attribution_allowed': passed,
            'substitute_dispatch_count': dispatch_count,
            'crossover_vs_external_image_embeddings': image}


def require_rotary_success(result):
    if result.get('status') != 'completed':
        raise RuntimeError('rotary crossover first nail failed')


def native_factor(backend, effective_numerics, local, native_other, common_other):
    """Record one native rollout factor without conflating native/common inputs."""
    return {
        'backend': backend,
        'effective_numerics': dict(effective_numerics),
        'native_input_rollout': {
            'embeds': comparison(local['embeds'], native_other['embeds']),
            'positions': comparison(local['positions'], native_other['positions']),
        },
        'common_input_rollout': {
            'embeds': comparison(local['embeds'], common_other['embeds']),
            'positions': comparison(local['positions'], common_other['positions']),
        },
    }


def compare_native_traces(left, right, *, image_token_id, latent_pad_token_id):
    """Localize the first observed difference across actual native rollout boundaries."""
    def image_comparison():
        expected = [left['image_expected'], right['image_expected']]
        present = [left['image_embeddings'] is not None, right['image_embeddings'] is not None]
        if expected == [False, False] and present == [False, False]:
            return {'status': 'not_applicable', 'reason': 'image_free',
                    'observation_present': present}
        if expected != [True, True] or present != [True, True]:
            return {'status': 'invalid', 'reason': 'observation_missing',
                    'image_expected': expected, 'observation_present': present}
        return {'status': 'observed', **comparison(
            left['image_embeddings'], right['image_embeddings'])}
    for side, trace in (('left', left), ('right', right)):
        tensors = [('token_embeddings', trace['token_embeddings']),
                   ('reconstructed_base_inputs', trace['reconstructed_base_inputs'])]
        if trace['image_embeddings'] is not None:
            tensors.append(('image_embeddings', trace['image_embeddings']))
        tensors.extend((f'decoder_calls[{index}].{key}', call[key])
                       for index, call in enumerate(trace['decoder_calls'])
                       for key in ('inputs', 'positions', 'output'))
        for name, tensor in tensors:
            if tensor.is_floating_point() and not bool(tensor.isfinite().all()):
                raise ValueError(f'nonfinite native trace observation: {side}.{name}')
    result = {
        'image_embeddings': image_comparison(),
        'token_embeddings': comparison(left['token_embeddings'], right['token_embeddings']),
        'reconstructed_base_inputs': comparison(
            left['reconstructed_base_inputs'], right['reconstructed_base_inputs']),
        'reconstructed_base_provenance': left['reconstructed_base_provenance'],
        'decoder_calls': [],
    }
    earliest = None
    def differs(value):
        return (value.get('status') == 'invalid' or value.get('comparable') is False
                or value.get('connected') is False or value.get('finite') is False
                or ('exact_equal' in value and value.get('exact_equal') is not True))
    if differs(result['image_embeddings']):
        earliest = {'stage': 'image_embeddings',
                    'reason': result['image_embeddings'].get('reason', 'values_differ')}
    elif differs(result['token_embeddings']):
        earliest = {'stage': 'token_embeddings'}
    elif differs(result['reconstructed_base_inputs']):
        earliest = {'stage': 'reconstructed_base_inputs'}
    if len(left['decoder_calls']) != len(right['decoder_calls']):
        if earliest is None:
            earliest = {'stage': 'decoder_call_count'}
    for ordinal, (a, b) in enumerate(zip(left['decoder_calls'], right['decoder_calls'])):
        call = {'ordinal': ordinal, 'kind': a['kind'],
                'latent_ordinal': a['latent_ordinal'],
                'inputs': comparison(a['inputs'], b['inputs']),
                'positions': comparison(a['positions'], b['positions']),
                'output_last_hidden': comparison(a['output'], b['output'])}
        if (a['kind'], a['latent_ordinal']) != (b['kind'], b['latent_ordinal']):
            call['call_identity_equal'] = False
        else:
            call['call_identity_equal'] = True
        if earliest is None:
            for field in ('inputs', 'positions', 'output_last_hidden'):
                if not call['call_identity_equal'] or differs(call[field]):
                    earliest = {'stage': a['kind'], 'call_ordinal': ordinal,
                                'latent_ordinal': a['latent_ordinal'], 'field': field}
                    break
        result['decoder_calls'].append(call)
    ids = left['input_ids'][0]
    regions = {
        'image': ids == image_token_id,
        'latent': ids == latent_pad_token_id,
    }
    regions['nonlatent_nonimage'] = ~(regions['image'] | regions['latent'])
    result['final_embedding_regions'] = {
        name: comparison(left['decoder_calls'][-1]['inputs'][:, mask],
                         right['decoder_calls'][-1]['inputs'][:, mask])
        for name, mask in regions.items()
    }
    result['earliest_difference'] = earliest
    return result


def run_rotary_crossover(local, external_model, prepared, external, runtime, output, progress):
    """Run the approved same-model A/B vision-rotary crossover under MATH."""
    import torch
    local_module = importlib.import_module(type(qwen(local)).__module__)
    substitute = external.model.apply_rotary_pos_emb_flashatt
    result = {'status': 'incomplete', 'backend': 'math', 'samples': []}
    with torch.inference_mode(), runtime.numerical_policy():
        for ordinal, (sample, _ext_sample, _witness) in enumerate(prepared):
            progress(f'sample {sample.source_index}: rotary crossover A/B own-bank scoring')
            a_native = native_rollout(local, sample, 'local', external, runtime)
            with temporary_vision_rotary(local_module, substitute) as dispatch:
                b_native = native_rollout(local, sample, 'local', external, runtime)
            e_native = native_rollout(
                external_model, sample, 'external', external, runtime)
            restored_native = native_rollout(local, sample, 'local', external, runtime)
            restoration = {
                'final_embeddings': comparison(a_native['embeds'], restored_native['embeds']),
            }
            if restoration['final_embeddings'].get('exact_equal') is not True:
                raise ValueError('local vision rotary restoration changed clean embeddings')
            entry = {'source_index': sample.source_index,
                     'substitute_dispatch_count': dispatch['dispatch_count'],
                     'a_restoration': restoration}
            if ordinal == 0:
                gate = rotary_first_nail(
                    b_native['trace'], e_native['trace'],
                    dispatch_count=dispatch['dispatch_count'])
                result['first_nail'] = gate
                entry['first_nail'] = gate
                if not gate['causal_attribution_allowed']:
                    result['samples'].append(entry)
                    result['status'] = 'failed_first_nail'
                    return result
            a_state = common_state(local, a_native, sample, runtime)
            a_report, a_logits = score_local_bank(local, sample, a_state, runtime)
            b_state = common_state(local, b_native, sample, runtime)
            b_report, b_logits = score_local_bank(local, sample, b_state, runtime)
            e_report, e_logits = score_external_bank(
                external_model, sample, e_native, external)
            restored_state = common_state(local, restored_native, sample, runtime)
            restored_report, restored_logits = score_local_bank(
                local, sample, restored_state, runtime)
            restoration.update({
                'clean_logits': comparison(a_logits['clean'], restored_logits['clean']),
                'clean_repeat': restored_report['clean_repeat'],
            })
            if restoration['clean_logits'].get('exact_equal') is not True:
                raise ValueError('local vision rotary restoration changed clean replay')
            logits_path = output / f'rotary-crossover-sample-{sample.source_index}-logits.pt'
            torch.save({'source_index': sample.source_index,
                        'A': a_logits, 'B': b_logits, 'E': e_logits}, logits_path)
            a_body, b_body = a_report['body_mask']['five_metrics'], b_report['body_mask']['five_metrics']
            entry.update({
                'A': a_report,
                'B': b_report,
                'E': e_report,
                'A_vs_B_native_trace': compare_native_traces(
                    a_native['trace'], b_native['trace'],
                    image_token_id=qwen(local).config.image_token_id,
                    latent_pad_token_id=qwen(local).config.latent_pad_token_id),
                'B_vs_E_native_trace': compare_native_traces(
                    b_native['trace'], e_native['trace'],
                    image_token_id=qwen(local).config.image_token_id,
                    latent_pad_token_id=qwen(local).config.latent_pad_token_id),
                'body_effect_B_minus_A': {
                    key: b_body[key] - a_body[key]
                    for key in ('first_changed', 'any_changed', 'tokens_changed',
                                'mean_nll', 'delta_nll')},
                'selected_logits': {'path': logits_path.name,
                                    'sha256': native.baseline.sha(logits_path)},
            })
            result['samples'].append(entry)
            del a_state, b_state, restored_state
            del a_native, b_native, e_native, restored_native
    result['status'] = 'completed'
    return result


def compare_observations(left, right):
    layers = []
    first = None
    for index in sorted(left.layers.keys() | right.layers.keys()):
        item = {'layer': index, 'hidden': comparison(left.layers.get(index), right.layers.get(index))}
        for offset, name in enumerate(('key', 'value')):
            item[name] = comparison(left.kv[index][offset] if index in left.kv else None,
                                    right.kv[index][offset] if index in right.kv else None)
        if first is None and any(item[name].get('exact_equal') is False for name in ('hidden','key','value')):
            first = index
        layers.append(item)
    return {'layers': layers, 'first_observed_divergent_layer': first,
            'layer_coordinate_scope': left.positions,
            'kv_scope': 'actual attention-return locals, post-RoPE/repeat where exposed; shape mismatch is explicit',
            'local_dispatch': left.dispatch, 'external_dispatch': right.dispatch,
            'mask_geometry': {'local_blocked_rows_for_body_key': left.masks[0].get('blocked_rows_for_body_key'),
                'external_blocked_rows_for_body_key': right.masks[0].get('blocked_rows_for_body_key'),
                'local_per_layer': left.masks, 'external_per_layer': right.masks}}


def compare_paths(local, other, sample, state, runtime, external, backend):
    import torch
    result = {'factors': {'backend': backend, 'external_batching': ['per_site', 'packed_sites']},
              'local': {}, 'external': {}, 'cross': {}}
    targets = [torch.tensor(s.target_ids) for s in sample.sites]
    items = external_items(sample, external)
    values = {side: {} for side in ('local','external')}
    observations = {}
    local_error = None
    policy = runtime.numerical_policy() if backend == 'math' else nullcontext()
    with policy:
        result['effective_numerics'] = runtime.numerical_policy_identity()
        for condition in ('clean', 'body_mask', 'clean_repeat'):
            intervention = 'clean' if condition == 'clean_repeat' else condition
            a_values, b_values, site_observations = [], [], []
            for site, item in zip(sample.sites, items):
                if local_error is None:
                    try:
                        a, a_observation = decode_local(
                            local, state, site, runtime, intervention, backend)
                    except ValueError as error:
                        if (backend == 'default' and type(error) is ValueError
                                and str(error) == 'full-length prefix K/V changed'):
                            local_error = f'{type(error).__name__}: {error}'
                        else:
                            raise
                    else:
                        a_values.append(a)
                        site_observations.append(a_observation)
                with Observation(other, site) as b_observation:
                    b = decode_external(other, state, sample, (item,), external, intervention)[0]
                b_values.append(b)
                if local_error is None:
                    site_observations[-1] = (site, site_observations[-1], b_observation)
            values['local'][condition] = a_values
            values['external'][condition] = b_values
            observations[condition] = site_observations
        for condition in ('clean', 'body_mask', 'clean_repeat'):
            current = values['external'][condition]
            baseline = values['external']['clean']
            result['external'][condition] = {
                'five_metrics': five_metrics(baseline,current,targets),
                'logits': comparison(torch.cat(baseline),torch.cat(current))}
        if local_error is None:
            for condition in ('clean', 'body_mask', 'clean_repeat'):
                a_values, b_values = values['local'][condition], values['external'][condition]
                site_reports = []
                for (site, a_observation, b_observation), a, b in zip(
                        observations[condition], a_values, b_values):
                    report = compare_observations(a_observation, b_observation)
                    report.update(plan_index=site.plan_index, logits=comparison(a,b),
                                  metrics=five_metrics([a],[b],[torch.tensor(site.target_ids)]))
                    site_reports.append(report)
                result['cross'][condition] = {'sites': site_reports,
                    'five_metrics': five_metrics(a_values,b_values,targets)}
                baseline = values['local']['clean']
                result['local'][condition] = {
                    'five_metrics': five_metrics(baseline,a_values,targets),
                    'logits': comparison(torch.cat(baseline),torch.cat(a_values))}
        else:
            unsupported = {'status': 'unsupported', 'error': local_error}
            result['local'] = dict(unsupported)
            result['cross'] = dict(unsupported)
        result['external']['batching'] = {}
        for condition in ('clean','body_mask'):
            duplicate = len(items) == 1
            packed_items = items*2 if duplicate else items
            baseline = values['external'][condition]*2 if duplicate else values['external'][condition]
            packed = decode_external(other, state, sample, packed_items, external, condition)
            result['external']['batching'][condition] = {
                'batch_size': len(packed_items), 'duplicate_control_only': duplicate,
                'unique_sites': len(items),
                'logits': comparison(torch.cat(baseline),torch.cat(packed)),
                'five_metrics': five_metrics(baseline,packed,targets*2 if duplicate else targets)}
    result['status'] = ('unsupported_control' if local_error else
                        'supported_primary' if backend == 'math' else 'supported_control')
    return result


def prepare_native(row, index, processor, image_root, ce_inputs, external):
    import torch
    local, exclusions = ce_inputs.prepare_sample(row,index,processor,image_root)
    if local is None:
        raise ValueError(f'preselected sample has no local sites: {index}: {exclusions}')
    token_ids = external.rollout.LatentTokenIds(*native.baseline.token_ids(processor.tokenizer)[:3])
    example = external.data.prepare_stage2_example(row,image_root=Path(image_root),latent_size=8)
    batch = external.data.Stage2DataCollator(processor,token_ids,8,True,
        context_preference_enabled=True,target_scope='immediate_text')([example])
    model_inputs = {key: value for key,value in batch.items()
                    if isinstance(value,torch.Tensor) and key in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
    sites = tuple(ce_inputs.Site(index,i, item.span.prev_text_end,item.span.latent_start,item.span.latent_end,
                item.span.target_text_start-1,tuple(int(x)+1 for x in item.chosen_predictor_positions),
                tuple(int(x) for x in item.chosen_target_ids)) for i,item in enumerate(batch['step_preference_items']))
    other = ce_inputs.PreparedSample(index,model_inputs,sites)
    witness = {'local': {'input_ids':local.model_inputs['input_ids'].tolist(), 'sites':[asdict(s) for s in local.sites], 'exclusions':exclusions},
        'external': {'input_ids':model_inputs['input_ids'].tolist(), 'sites':[asdict(s) for s in sites]},
        'input_comparisons': {key:comparison(local.model_inputs.get(key),model_inputs.get(key))
                              for key in local.model_inputs.keys() | model_inputs.keys()},
        'target_positions_equal': [s.target_positions for s in local.sites] == [s.target_positions for s in sites],
        'target_ids_equal': [s.target_ids for s in local.sites] == [s.target_ids for s in sites]}
    return local,other,witness


def run(args):
    preflight(args)
    import torch
    from transformers import AutoProcessor
    output, logs = Path(args.output_dir), Path(args.log_dir)
    output.mkdir(parents=True); logs.mkdir(parents=True)
    def progress(message):
        print(message, flush=True)
        with (logs/'progress.log').open('a') as stream:
            stream.write(message+'\n')
    report = {'status':'incomplete','scope':'first two manifest samples; no optimizer/backward; shared local final weights, not historical external weights',
              'manifest_sha256':native.baseline.sha(args.manifest), 'samples':[], 'optimizer_steps':0,
              'selection':selected_indices(json.loads(Path(args.manifest).read_text()))}
    def save():
        (output/'evidence.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        with native.ExternalSource(args.external_source) as source, ce_source(args.ce_root) as ce:
            external = external_modules(source)
            runtime = ce.module('evaluation.vision_block_runtime')
            ce_inputs = ce.module('evaluation.vision_block_inputs')
            processor = AutoProcessor.from_pretrained(args.artifact,local_files_only=True)
            rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
            prepared = [prepare_native(rows[index],index,processor,args.image_root,ce_inputs,external) for index in report['selection']]
            report['native_preparation'] = [witness for _,_,witness in prepared]
            relative = 'transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py'
            report['local_model_source'] = {'target_sha256':native.baseline.sha(Path(__file__).resolve().parents[1]/relative),
                                          'ce_sha256':native.baseline.sha(Path(args.ce_root)/relative)}
            report['local_model_source']['byte_identical'] = len(set(report['local_model_source'].values())) == 1
            save(); progress('CPU native preparation recorded; loading two independently restored Stage2 models')
            models = load_models(args,processor,external)
            report['artifact_restoration'] = [artifact_witness(model,args.artifact) for model in models]
            report['model_identity'] = native.model_identity(*models)
            save()
            if not all(w['all_restored'] for w in report['artifact_restoration']):
                raise ValueError('artifact restoration witness failed')
            if not report['model_identity']['all_state_exact_equal']:
                raise ValueError('separately loaded model parameter/buffer values differ; inspect evidence')
            if args.rotary_crossover:
                report['mode'] = 'rotary_crossover_v3'
                report['rotary_crossover'] = run_rotary_crossover(
                    models[0], models[1], prepared, external, runtime, output, progress)
                report['status'] = report['rotary_crossover']['status']
                save()
                require_rotary_success(report['rotary_crossover'])
                report['completion'] = {'rotary_crossover_math': 'supported'}
                report['status'] = 'completed'
                save(); progress('two-sample rotary crossover completed')
                return report
            with torch.inference_mode():
                for sample,ext_sample,witness in prepared:
                    progress(f'sample {sample.source_index}: native rollout before common latent forcing')
                    local_native = native_rollout(models[0],sample,'local',external,runtime)
                    ext_native = native_rollout(models[1],ext_sample,'external',external,runtime)
                    common_external = native_rollout(models[1],sample,'external',external,runtime)
                    default_native = native_factor(
                        'default', runtime.numerical_policy_identity(),
                        local_native, ext_native, common_external)
                    default_native['common_input_trace'] = compare_native_traces(
                        local_native['trace'], common_external['trace'],
                        image_token_id=qwen(models[0]).config.image_token_id,
                        latent_pad_token_id=qwen(models[0]).config.latent_pad_token_id)
                    with runtime.numerical_policy():
                        math_local_native = native_rollout(
                            models[0],sample,'local',external,runtime)
                        math_ext_native = native_rollout(
                            models[1],ext_sample,'external',external,runtime)
                        math_common_external = native_rollout(
                            models[1],sample,'external',external,runtime)
                        math_native = native_factor(
                            'math', runtime.numerical_policy_identity(),
                            math_local_native, math_ext_native, math_common_external)
                        math_native['common_input_trace'] = compare_native_traces(
                            math_local_native['trace'], math_common_external['trace'],
                            image_token_id=qwen(models[0]).config.image_token_id,
                            latent_pad_token_id=qwen(models[0]).config.latent_pad_token_id)
                    entry = {'source_index':sample.source_index,
                        'native_input_rollout': default_native['native_input_rollout'],
                        'common_input_rollout':default_native['common_input_rollout'],
                        'native_rollout_factors': {
                            'default': default_native,
                            'math': math_native,
                        },
                        'forced_common':{'origin':'local ordinary clean rollout; same bank for both implementations and backend factors',
                                         'embeds':native.tensor_record(local_native['embeds']),
                                         'positions':local_native['positions'].cpu().tolist(),
                                         'input_ids':sample.model_inputs['input_ids'].tolist(),
                                         'sites':[asdict(s) for s in sample.sites]}, 'controls':{}}
                    report['samples'].append(entry); save()
                    # Capture and branch replay share each separately labelled kernel policy.
                    for backend in ('math','default'):
                        progress(f'sample {sample.source_index}: {backend} local/external clean/body/repeat/batching')
                        with runtime.numerical_policy() if backend == 'math' else nullcontext():
                            state = common_state(models[0],local_native,sample,runtime)
                            entry['controls'][backend] = compare_paths(*models,sample,state,runtime,external,backend)
                        del state
                        save()
                    del local_native,ext_native,common_external
                    del math_local_native,math_ext_native,math_common_external
            report['loaded_modules'] = {name:{'path':module.__file__, 'sha256':native.baseline.sha(module.__file__),
                'supplied':Path(module.__file__).resolve().is_relative_to(Path(args.external_source).resolve())}
                for name,module in tuple(sys.modules.items()) if getattr(module,'__file__',None)
                and str(module.__file__).endswith('.py') and Path(module.__file__).is_file()
                and (name.startswith((source.prefix,ce.prefix,'stage2','stage1','peft','transformers.models.qwen2_5_vl')))}
            report['redirects'] = sorted(source.redirects)
            report['limitations'] = ['No external historical weights or historical runtime authentication.',
                'Shared available framework, processor, genuine local Stage2 loader and token-row helpers; supplied model/data/factorial dependencies isolated.',
                'Local numerical decorators unwrapped to expose backend as a separate explicit factor; function bodies unchanged.',
                'Layer/KV comparisons cover reported selected coordinates, not every hidden element.',
                'Native tracing localizes observed runtime differences but does not assign causality to known source-level vision rotary helper differences.',
                'Inference-only evidence does not settle training Flex, checkpointing, gradient or optimizer equivalence.']
            unsupported_default = any(
                sample['controls']['default']['status'] == 'unsupported_control'
                for sample in report['samples'])
            report['completion'] = {
                'math_primary': 'supported',
                'default_control': 'unsupported' if unsupported_default else 'supported',
            }
            report['status'] = ('completed_with_unsupported_control'
                                if unsupported_default else 'completed')
            save(); progress('two-sample diagnostic completed')
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'; save(); progress(report['error'])
        raise
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name,default in (('external-source',EXTERNAL),('ce-root',str(CE_ROOT)),('artifact',ARTIFACT),
        ('base','/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct'),
        ('manifest',str(MANIFEST)),('data','/root/workspace/ILVR/data/TEST.jsonl'),
        ('image-root','/root/workspace/ILVR/data'),('device','cuda:0')):
        parser.add_argument('--'+name,default=default)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--log-dir',required=True)
    parser.add_argument('--rotary-crossover', action='store_true',
                        help='run the opt-in v3 MATH-only rotary crossover')
    return parser.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
