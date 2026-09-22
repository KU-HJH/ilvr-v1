#!/usr/bin/env python
"""Read-only supplied/current Qwen forward/backward differential; never saves weights."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import types

sys.dont_write_bytecode = True
if platform.python_version().endswith('+'):
    _version = platform.python_version()
    platform.python_version = lambda: _version.rstrip('+')

SOURCE = '/raid/workspace/ILVR/ILVR/.worktrees/ours1/source'
STAGE1 = 'checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431'
BASE = 'checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct'
MANIFEST = 'results/voff-reproduction-full212-20260918-043839/stage2-length-manifest.json'
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def comparison(a, b):
    """Tolerance is supplementary; disconnected and nonfinite are not numeric zero."""
    import torch
    if a is None or b is None:
        same = a is None and b is None
        return {'pass': same, 'connected': False, 'connection': 'both_disconnected' if same else 'mismatch',
                'exact_equal': None, 'finite': None, 'max_abs_error': None, 'relative_l2': None}
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('comparison tensor shape/dtype mismatch')
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta = a.double() - b.double() if finite else None
    return {'pass': finite and torch.allclose(a, b, atol=1e-6, rtol=1e-5),
            'connected': True, 'connection': 'both_connected', 'finite': finite,
            'exact_equal': bool(torch.equal(a, b)),
            'max_abs_error': float(delta.abs().max()) if finite and delta.numel() else (0. if finite else None),
            'relative_l2': float(delta.norm() / (a.double().norm() + 1e-12)) if finite else None}


def select_short_cases(manifest):
    """Select by retained manifest length, not raw TRAIN order or filtered examples."""
    rows = [(index, manifest['lengths'][index], manifest['latent_block_counts'][index])
            for index in manifest['retained_indices']]
    selected = []
    for predicate in (lambda n: n == 1, lambda n: n == 2):
        candidates = [row for row in rows if predicate(row[2])]
        if not candidates:
            raise ValueError('retained manifest must contain both 1 and 2+ block cases')
        selected.append(min(candidates, key=lambda row: (row[1], row[0])))
    return selected


def tensor_digest(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_model_identity(current, external):
    import torch
    left, right = dict(current.named_parameters()), dict(external.named_parameters())
    if left.keys() != right.keys():
        raise ValueError('model parameter names differ')
    trainable = [name for name, p in left.items() if p.requires_grad]
    if trainable != [name for name, p in right.items() if p.requires_grad]:
        raise ValueError('model trainable parameter names differ')
    if any(left[n] is right[n] for n in left):
        raise ValueError('models share parameter objects')
    a, b = current.state_dict(), external.state_dict()
    if a.keys() != b.keys():
        raise ValueError('model state keys differ')
    for name in a:
        if a[name].shape != b[name].shape or a[name].dtype != b[name].dtype or not torch.equal(a[name], b[name]):
            raise ValueError(f'model state differs: {name}')
    return {'all_state_exact_equal': True, 'state_tensor_count': len(a),
            'distinct_parameter_objects': True, 'trainable_names': trainable,
            'trainable_parameter_count': sum(left[n].numel() for n in trainable)}


def runtime_model_metadata(model, *, production):
    import torch
    from peft_common import resolve_qwen2_5_vl_model
    qwen = resolve_qwen2_5_vl_model(model)
    module = importlib.import_module(type(qwen).__module__)
    decoder_type = module.Qwen2_5_VLSdpaAttention
    vision_type = module.Qwen2_5_VLVisionFlashAttention2 if production else module.Qwen2_5_VLVisionSdpaAttention
    if not qwen.model.layers or any(type(layer.self_attn) is not decoder_type for layer in qwen.model.layers):
        raise ValueError('runtime decoder attention differs from SDPA')
    if not qwen.visual.blocks or any(type(block.attn) is not vision_type for block in qwen.visual.blocks):
        raise ValueError('runtime vision attention differs from requested profile')
    if production and any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise ValueError('runtime parameter dtype differs from production BF16')
    return {'model_class': f'{type(qwen).__module__}.{type(qwen).__name__}',
            'model_source': module.__file__, 'decoder_attention': decoder_type.__name__,
            'vision_attention': vision_type.__name__,
            'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}),
            'trainable_parameter_dtypes': {name: str(p.dtype) for name, p in model.named_parameters() if p.requires_grad}}


def supplied_modules(root):
    folder = root / 'transformers/src/transformers/models/qwen2_5_vl'
    # Same package depth preserves relative imports. The enclosing framework is shared.
    name = 'transformers.models.voff_supplied_qwen'
    package = types.ModuleType(name)
    package.__path__, package.__package__ = [str(folder)], name
    sys.modules[name] = package
    supplied = importlib.import_module(name + '.modeling_qwen2_5_vl')
    name = 'voff_supplied_step'
    package = types.ModuleType(name)
    package.__path__, package.__package__ = [str(root / 'src/step_preference')], name
    sys.modules[name] = package
    return supplied, importlib.import_module(name + '.packing'), importlib.import_module(name + '.dpo_loss')


def tiny_models(tokenizer, supplied):
    import torch
    from peft import LoraConfig, get_peft_model
    from stage1.token_rows import SparseOutputTokenRows
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    pad, start, end, image, vision_start, video = token_ids(tokenizer)
    config = dict(vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
        vision_config=dict(depth=1, hidden_size=8, intermediate_size=16, num_heads=2,
            patch_size=2, spatial_merge_size=1, temporal_patch_size=1, window_size=2,
            out_hidden_size=8, fullatt_block_indexes=[0]),
        image_token_id=image, vision_start_token_id=vision_start, video_token_id=video,
        stage='stage2', latent_size=2, latent_pad_token_id=pad, latent_start_token_id=start,
        latent_end_token_id=end, supervise_latent_boundaries=True, use_cache=False,
        attn_implementation='sdpa', attention_dropout=0., stage2_latent_activation_checkpointing=True,
        rope_scaling={'type': 'mrope', 'mrope_section': [1, 1, 0]})
    torch.manual_seed(42)
    bases = [Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(**config)).float().train(),
             supplied.Qwen2_5_VLForConditionalGeneration(supplied.Qwen2_5_VLConfig(**config)).float().train()]
    bases[1].load_state_dict(bases[0].state_dict(), strict=True)
    models = []
    for base in bases:
        model = get_peft_model(base, LoraConfig(r=32, lora_alpha=64, lora_dropout=.05,
            use_dora=True,
            target_modules=r'model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))',
            trainable_token_indices={'model.embed_tokens': [pad, start, end]}))
        base.lm_head = SparseOutputTokenRows(base.lm_head, (pad, start, end))
        models.append(model.to(dtype=torch.bfloat16).train())
    with torch.no_grad():
        for name, parameter in models[0].named_parameters():
            if 'lora_B' in name or 'output_delta' in name:
                parameter.normal_(mean=0., std=.02)
    models[1].load_state_dict(models[0].state_dict(), strict=True)
    return models, config


def token_ids(tokenizer):
    return tuple(tokenizer.convert_tokens_to_ids(t) for t in
                 ('<|latent_pad|>', '<|latent_start|>', '<|latent_end|>',
                  '<|image_pad|>', '<|vision_start|>', '<|video_pad|>'))


def tiny_cases(tokenizer):
    import torch
    from step_preference.spans import ReasoningStepSpan
    pad, start, end, image, vision_start, _ = token_ids(tokenizer)
    cases = []
    for blocks in (1, 2):
        tokens = [1, vision_start, image, image, image, image, 2]
        for _ in range(blocks):
            tokens += [start, pad, pad, end, 3, 4]
        tokens += [5, 6]
        ids = torch.tensor([tokens])
        labels = ids.clone()
        labels[:, :7] = -100
        labels[ids == pad] = -100
        spans = []
        for index in range(blocks):
            boundary = 7 + 6 * index
            spans.append(ReasoningStepSpan(index, 6 if index == 0 else boundary - 2, boundary,
                boundary + 1, boundary + 3, boundary + 4, boundary + 6 if index < blocks - 1 else len(tokens), None))
        cases.append(dict(sample_id='tiny', latent_blocks=blocks, spans=spans,
            inputs=dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                        pixel_values=torch.linspace(-1, 1, 48).reshape(4, 12), image_grid_thw=torch.tensor([[1, 2, 2]]))))
    return cases


def real_cases(args, processor):
    import torch
    from stage2.data import Stage2DataCollator, load_jsonl_examples, prepare_stage2_example
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
    manifest = json.loads(Path(args.manifest).read_text())
    if sha(args.data) != manifest['data_sha256']:
        raise ValueError('TRAIN bytes differ from retained manifest')
    if (manifest['latent_size'], manifest['supervise_latent_boundaries'], manifest['assistant_plan_rendering']) != (8, True, 'concatenated-v1'):
        raise ValueError('retained manifest preparation settings differ')
    samples = load_jsonl_examples(Path(args.data))
    collator = Stage2DataCollator(processor, LatentTokenIds(*token_ids(processor.tokenizer)[:3]), 8, True,
        step_preference_enabled=True, preference_target_scope='until_next_latent')
    cases = []
    for index, length, blocks in select_short_cases(manifest):
        prepared = prepare_stage2_example(samples[index], image_root=Path(args.image_root), latent_size=8,
            assistant_plan_rendering='concatenated-v1', require_reasoning_units=True)
        batch = collator([prepared])
        if batch['input_ids'].shape[1] != length or prepared.latent_block_count != blocks:
            raise ValueError('selected real input differs from retained manifest')
        inputs = {key: batch[key] for key in ('input_ids', 'attention_mask', 'labels', 'pixel_values', 'image_grid_thw')}
        assert all(isinstance(tensor, torch.Tensor) for tensor in inputs.values())
        cases.append(dict(sample_id=prepared.sample_id, train_index=index, latent_blocks=blocks,
                          spans=batch['reasoning_step_spans'], inputs=inputs))
    return cases


def real_models(args, processor, supplied, *, verify_initial_state=True):
    import torch
    from stage2.attention_backend import load_qwen2_5_vl_with_attention
    from stage2.checkpointing import configure_stage2_metadata
    from stage2.peft import Stage2TuningConfig, load_stage1_dora_for_fresh_stage2
    from transformers import Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
    models = []
    for external, cls in ((False, Qwen2_5_VLForConditionalGeneration),
                          (True, supplied.Qwen2_5_VLForConditionalGeneration)):
        # Reset resize initialization as well as loading the same frozen base and adapter.
        torch.manual_seed(42)
        kwargs = dict(torch_dtype=torch.bfloat16, local_files_only=True)
        if external:
            kwargs['stage2_split_attention'] = True
        base = load_qwen2_5_vl_with_attention(args.base, attention_implementation='sdpa',
            vision_attention_implementation='flash_attention_2', config_load_kwargs={'local_files_only': True},
            model_load_kwargs=kwargs, load_model=cls.from_pretrained)
        model = load_stage1_dora_for_fresh_stage2(base, processor, args.stage1,
            Stage2TuningConfig('dora', 32, 64, .05, TARGETS), requested_base_model_reference=args.base)
        configure_stage2_metadata(base.config, LatentTokenIds(*token_ids(processor.tokenizer)[:3]),
            latent_size=8, supervise_latent_boundaries=True, latent_activation_checkpointing=True)
        base.config.use_cache = False
        model.gradient_checkpointing_disable()
        models.append(model.train())
    # Fail rather than copy over a loading mismatch and conceal it.
    if verify_initial_state:
        verify_model_identity(*models)
    # Actual DeepSpeed BF16 initialization casts the whole module, including PEFT adapters.
    return [model.to(device=args.device, dtype=torch.bfloat16) for model in models]


def preference_gradient_nonzero(preference, model):
    """Witness preference-only signal independently of the total CE gradient."""
    import torch
    gradients = torch.autograd.grad(preference, [p for p in model.parameters() if p.requires_grad],
                                    retain_graph=True, allow_unused=True)
    connected = [gradient for gradient in gradients if gradient is not None]
    return bool(connected) and all(bool(torch.isfinite(g).all()) for g in connected) and any(
        bool(g.abs().max() > 0) for g in connected)


def compare_case(models, case, packing, external_loss, device, backend, *, check_preference_gradient=False):
    import torch
    from step_preference.sidecar import build_context_sidecar_plan
    from step_preference.dpo_loss import context_preference_loss
    identity = verify_model_identity(*models)
    inputs = {name: value.to(device) for name, value in case['inputs'].items()}
    ids = inputs['input_ids']
    plan = build_context_sidecar_plan(clean_input_ids=ids, attention_mask=inputs['attention_mask'],
                                     spans=case['spans'], visual_only=True)
    items = [types.SimpleNamespace(sample_id=case['sample_id'], span=span,
        chosen_predictor_positions=torch.arange(span.target_text_start - 1, span.target_text_end - 1, device=device),
        chosen_target_ids=ids[0, span.target_text_start:span.target_text_end]) for span in case['spans']]
    external_plan = packing.build_context_sidecar_execution_plan(clean_input_ids=ids, items=items, modalities='visual')
    # Authenticate the independently built plans' target and replay identities.
    for branch, supplied_branch in zip(plan.groups[0], external_plan.visual, strict=True):
        if list(branch.token_ids) != supplied_branch.input_token_ids.tolist() or list(branch.target_token_ids) != supplied_branch.target_token_ids.tolist():
            raise ValueError('supplied/current sidecar inputs or targets differ')
    score_mask = torch.zeros_like(ids, dtype=torch.bool)
    for item in items:
        score_mask[0, item.chosen_predictor_positions] = True
    fingerprints = {name: tensor_digest(value) for name, value in inputs.items()}
    torch.manual_seed(173)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(173)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device.startswith('cuda') else []
    values, gradients, pixel_gradients, rng_after = [], [], [], []
    preference_nonzero = []
    for index, model in enumerate(models):
        model.zero_grad(set_to_none=True)
        batch = {name: tensor.clone() for name, tensor in inputs.items()}
        if {name: tensor_digest(value) for name, value in batch.items()} != fingerprints:
            raise ValueError('forward input identity mismatch')
        batch['pixel_values'].requires_grad_()
        options = (dict(stage2_loss_only=True, stage2_logit_positions=torch.tensor(plan.root_selected_positions, device=device),
            stage2_shared_prefix_inputs=dict(plan=plan, backend=backend, query_budget=1024,
                prefix_mode='clean_prefix', branch_mode='independent', block_original_image=False))
            if index == 0 else dict(stage2_loss_only=True, stage2_score_token_mask=score_mask, stage2_sidecar_plan=external_plan))
        torch.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=False):
            output = model(**batch, **options)
            rows = output.stage2_selected_logits if index == 0 else output.stage2_scored_logits
            clean, off, offset = [], [], 0
            for item in items:
                targets = item.chosen_target_ids
                count = targets.numel()
                clean.append(rows[0, offset:offset + count].float().log_softmax(-1).gather(-1, targets[:, None]).mean())
                if index == 0:
                    off.append(next(r['mean_logp'] for r in output.stage2_sidecar_results if r['step_id'] == item.span.step_id))
                else:
                    logits = output.stage2_sidecar_logits.by_branch_key[f"{case['sample_id']}:{item.span.step_id}:visual:chosen_entropy"]
                    off.append(logits.reshape(-1, logits.shape[-1]).float().log_softmax(-1).gather(-1, targets[:, None]).mean())
                offset += count
            clean, off = torch.stack(clean), torch.stack(off)
            preference = (context_preference_loss(clean, off, beta=.1, margin=0.) if index == 0
                          else external_loss.context_preference_loss(clean, off, beta=.1, margin=0.).loss)
            total = output.loss + preference
            values.append({name: tensor.detach().cpu().clone() for name, tensor in
                           dict(loss=total, ce_loss=output.loss, preference_loss=preference,
                                clean_scores=clean, visual_off_scores=off, selected_clean_logits=rows).items()})
        if check_preference_gradient:
            preference_nonzero.append(preference_gradient_nonzero(preference, model))
        total.backward()
        gradients.append({n: None if p.grad is None else p.grad.detach().cpu().clone()
                          for n, p in model.named_parameters() if p.requires_grad})
        pixel_gradients.append(None if batch['pixel_values'].grad is None else batch['pixel_values'].grad.detach().cpu().clone())
        rng_after.append({'cpu': tensor_digest(torch.get_rng_state()),
                          'cuda': [tensor_digest(state) for state in torch.cuda.get_rng_state_all()] if cuda_rng else []})
        del output, total, preference, rows, clean, off, batch
    comparisons = {name: comparison(gradients[0][name], gradients[1][name]) for name in identity['trainable_names']}
    connected = [value for value in comparisons.values() if value['connected']]
    return {key: value for key, value in case.items() if key not in ('inputs', 'spans')} | {
        'sequence_length': ids.shape[1], 'input_hashes': fingerprints, 'identical_inputs': True,
        'model_identity': identity, 'matched_rng_before_forward': True,
        'rng_before_forward': {'cpu': tensor_digest(cpu_rng), 'cuda': [tensor_digest(state) for state in cuda_rng]},
        'rng_after_backward': rng_after, 'rng_after_backward_equal': rng_after[0] == rng_after[1],
        **{name: comparison(values[0][name], values[1][name]) for name in values[0]},
        'parameter_gradients': comparisons, 'pixel_gradient': comparison(*pixel_gradients),
        'connected_parameter_count': len(connected),
        'preference_gradient_nonzero': all(preference_nonzero) if check_preference_gradient else None,
        'preference_gradient_witness_scope': 'tiny_only' if check_preference_gradient else 'not_run',
        'all_trainable_gradients_exact_zero_error': bool(connected) and len(connected) == len(comparisons)
            and all(value['finite'] and value['exact_equal'] for value in connected),
        'maximum_trainable_gradient_abs_error': max((v['max_abs_error'] for v in connected if v['finite']), default=None),
        'disconnected_parameters': [name for name, value in comparisons.items() if not value['connected']],
        'nonfinite_parameters': [name for name, value in comparisons.items() if value['finite'] is False]}


def run(args=None):
    import torch
    from transformers import AutoProcessor
    args = args or parse_args([])
    root = Path(args.external_source).resolve(strict=True)
    before = inventory(root)
    supplied, packing, external_loss = supplied_modules(root)
    processor = AutoProcessor.from_pretrained(args.stage1, local_files_only=True)
    torch.set_num_threads(2)
    if args.mode == 'tiny':
        models, config = tiny_models(processor.tokenizer, supplied)
        cases, device, backend = tiny_cases(processor.tokenizer), 'cpu', 'dense'
    else:
        if not args.device.startswith('cuda') or not torch.cuda.is_available():
            raise ValueError('3b diagnostic requires an available CUDA device')
        cases = real_cases(args, processor)
        models = real_models(args, processor, supplied)
        config, device, backend = dict(latent_size=8, stage1=args.stage1, base=args.base), args.device, 'flex'
    model_metadata = [runtime_model_metadata(model, production=args.mode == '3b') for model in models]
    results = [compare_case(models, case, packing, external_loss, device, backend,
                            check_preference_gradient=args.mode == 'tiny') for case in cases]
    source_paths = list((root / 'transformers/src/transformers/models/qwen2_5_vl').glob('*.py'))
    source_paths += [root / 'src/step_preference/packing.py', root / 'src/step_preference/dpo_loss.py']
    shared = {n: sha(m.__file__) for n, m in tuple(sys.modules.items()) if n.startswith('transformers')
              and getattr(m, '__file__', None) and Path(m.__file__).is_file() and not Path(m.__file__).is_relative_to(root)}
    unchanged = before == inventory(root)
    if not unchanged:
        raise RuntimeError('external source inventory changed')
    return {'scope': f'{args.mode} supplied/current model diagnostic; shared current Transformers scaffolding, PEFT, '
                     'tokenizer and data preparation; no optimizer, training, distributed or historical-weight equivalence',
        'config': config, 'cases': results, 'runtime_models': model_metadata,
        'cuda_bf16_autocast': False, 'external_files_unchanged': unchanged,
        'source_hashes': {str(p.relative_to(root)): sha(p) for p in source_paths},
        'external_source_hashes': {str(p.relative_to(root)): sha(p) for p in source_paths},
        'shared_dependency_hashes': shared, 'comparator_sha256': sha(__file__),
        'hardware': torch.cuda.get_device_name(device) if device.startswith('cuda') else platform.processor(),
        'torch': torch.__version__, 'device': device, 'decoder_backend': 'sdpa',
        'vision_backend': 'flash_attention_2' if args.mode == '3b' else 'sdpa', 'sidecar_backend': backend,
        'dora': {'rank': 32, 'alpha': 64, 'dropout': .05}, 'sparse_input_and_output_rows': True,
        'selective_latent_checkpoint': True, 'native_decoder_checkpoint': False,
        'initialization_seed': 42, 'forward_seed': 173, 'beta': .1, 'preference_weight': 1.,
        'beta_scope': 'historical comparison parameter, not a new training run',
        'synthetic_nonzero_B_and_output_delta_std': .02 if args.mode == 'tiny' else None,
        'data_sha256': sha(args.data) if args.mode == '3b' else None,
        'manifest_sha256': sha(args.manifest) if args.mode == '3b' else None}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('tiny', '3b'), default='tiny')
    parser.add_argument('--external-source', default=SOURCE)
    parser.add_argument('--stage1', default=STAGE1)
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--data', default='data/TRAIN.jsonl')
    parser.add_argument('--image-root', default='data')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args(argv)


if __name__ == '__main__':
    print(json.dumps(run(parse_args()), allow_nan=False))
