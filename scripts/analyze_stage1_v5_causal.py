#!/usr/bin/env python
"""Select fixed rows, run native Stage1-v5 V-off, and summarize paired diagnostics."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'transformers/src'))
from evaluation.stage1_v5_causal import (
    select_rows, replay_selected, block_metrics, summarize_voff, prepare_analysis_inputs,
    pair_generation, verify_baseline, bootstrap_mean)


def read_rows(path):
    return [r for line in Path(path).read_text().splitlines() if line.strip()
            and not (r := json.loads(line)).get('summary')]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def model_weights_identity(artifact):
    """Bind the two final Stage1 policy-weight files shared by both artifacts."""
    digest = hashlib.sha256()
    for name in ('adapter_model.safetensors', 'ilvr_token_rows.safetensors'):
        path = Path(artifact)/name
        digest.update(name.encode())
        digest.update(path.stat().st_size.to_bytes(8, 'big'))
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def select(args):
    rows, manifest = select_rows(args.source, count=args.count, seed=args.seed,
        indices_path=args.indices, expected_sha256=args.expected_sha256)
    if args.limit:
        rows = rows[:args.limit]
        manifest['source_indices'] = manifest['source_indices'][:args.limit]
    if args.verify_subset:
        existing = read_rows(args.verify_subset)
        if existing != rows:
            raise ValueError('existing subset does not match fixed original source rows')
    if args.absolute_images:
        def absolute(path):
            return str((Path(args.source).resolve().parent/path).resolve()) if not Path(path).is_absolute() else path
        for row in rows:
            images = row.get('image_input', [])
            row['image_input'] = absolute(images) if isinstance(images, str) else [absolute(p) for p in images]
    content = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        stream.write(content)
    manifest.update(subset=str(destination.resolve()),
                    subset_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    write_json(args.manifest, manifest)


def load_analysis_processor(base):
    """Reproduce initial training serialization; saved-tokenizer reload loses AddedToken types."""
    from transformers import AutoProcessor
    from stage1.composition import _add_fresh_latent_tokens
    processor = AutoProcessor.from_pretrained(base, cache_dir=str(ROOT/'data/cache'), trust_remote_code=True)
    _add_fresh_latent_tokens(processor.tokenizer)
    return processor


def load_analysis_models(checkpoint, base, device):
    """CPU BF16 training-shaped base/resize/DoRA ordering, then authenticate helper."""
    import torch
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration, set_seed
    from stage2.attention_backend import load_qwen2_5_vl_with_attention
    from stage1.composition import _configure_latent_model
    from stage1.checkpointing import load_stage1_peft_artifact, load_stage1_peft_metadata
    from stage1.ema import Stage1EmaTeacher
    from stage1.peft import build_stage1_tuned_model
    from stage1.types import Stage1TuningConfig
    from stage1.v5_reference import restore_v5_reference, V5ReferenceIdentity
    from stage1.v5_resume import load_v5_contract
    from artifact_paths import require_canonical_artifact_path
    checkpoint = require_canonical_artifact_path(Path(checkpoint), kind='checkpoint')
    contract = load_v5_contract(checkpoint)
    saved = contract['execution']
    set_seed(saved['seed'])
    metadata = load_stage1_peft_metadata(checkpoint)
    processor = load_analysis_processor(base)
    def original():
        config = AutoConfig.from_pretrained(base, trust_remote_code=True)
        _configure_latent_model(config, metadata.latent_token_ids, latent_size=saved['latent_size'])
        config.compress_strategy = saved['compress_strategy']
        config.use_cache = False
        return load_qwen2_5_vl_with_attention(base,
            attention_implementation=saved['attention_implementation'],
            vision_attention_implementation=saved['vision_attention_implementation'],
            load_config=lambda *a, **kw: config,
            load_model=Qwen2_5_VLForConditionalGeneration.from_pretrained,
            model_load_kwargs=dict(torch_dtype=torch.bfloat16, cache_dir=str(ROOT/'data/cache'), trust_remote_code=True))
    student = load_stage1_peft_artifact(original(), processor, checkpoint,
        requested_base_model_reference=str(base), is_trainable=True, v5_resume=True)
    reference = original()
    reference.resize_token_embeddings(len(processor.tokenizer))
    tuning = Stage1TuningConfig(method=saved['tuning_method'], rank=saved['lora_r'],
        alpha=saved['lora_alpha'], dropout=saved['lora_dropout'],
        target_suffixes=tuple(saved['lora_target_modules'].split(',')))
    reference = build_stage1_tuned_model(reference, tuning, metadata.latent_token_ids, zero_stage=1)
    teacher = Stage1EmaTeacher(student, decay=1., method='dora', initial_reference_model=reference)
    identity = V5ReferenceIdentity(**contract['reference'])
    restore_v5_reference(teacher, student, processor, base_path=Path(base), directory=checkpoint, expected=identity)
    student.to(device).eval()
    teacher.model.to(device).eval()
    return student, teacher, processor, metadata, contract


def voff(args):
    import torch
    from stage1.data import prepare_stage1_comt_examples, Stage1DataCollator
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds
    manifest = json.loads(Path(args.manifest).read_text())
    raw = Path(manifest['subset']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['subset_sha256']:
        raise ValueError('selected TRAIN subset sha256 mismatch')
    rows = read_rows(manifest['subset'])
    indices = manifest['source_indices']
    if len(rows) != len(indices):
        raise ValueError('selected TRAIN coverage mismatch')
    if args.limit:
        rows, indices = rows[:args.limit], indices[:args.limit]
    # Prepare ONLY requested rows, including first-two probes. Keep input under data.
    input_path = Path(args.manifest).parent/(Path(args.output).stem+'-input.jsonl')
    content = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
    if input_path.exists():
        if input_path.read_text() != content:
            raise ValueError('diagnostic input already exists with different rows')
    else:
        input_path.write_text(content)
    examples = prepare_stage1_comt_examples(input_path, Path(args.image_root))
    examples = [replace(e, source_index=i) for e, i in zip(examples, indices)]
    model, teacher, processor, metadata, contract = load_analysis_models(args.checkpoint, args.base, args.device)
    saved = contract['execution']
    collator = Stage1DataCollator(processor, LatentTokenIds(*metadata.latent_token_ids),
                                 saved['latent_size'], stage1_variant='v5')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with output.open('x') as stream, torch.no_grad():
        for ordinal, example in enumerate(examples):
            batch = collator([example])
            prepared, v5 = prepare_analysis_inputs(teacher, processor, batch, saved, args.device)
            clean, off, drift = replay_selected(model, prepared, v5,
                check_unmasked=args.check_unmasked and ordinal < 2,
                atol=args.atol, rtol=args.rtol, chunk_size=128)
            result = block_metrics(clean, off, v5.target_ids, [len(b.target_ids) for b in v5.blocks])
            result.update(source_index=example.source_index, unmasked_oracle=drift,
                block_coordinates=[dict(start=b.start, end=b.end, target_positions=b.target_positions) for b in v5.blocks])
            results.append(result)
            stream.write(json.dumps(result, allow_nan=False)+'\n')
            stream.flush()
            print(f'V-off source={example.source_index} targets={result["target_count"]}', flush=True)
    summary = summarize_voff(results, indices)
    summary.update(model_checkpoint=str(Path(args.checkpoint).resolve()),
        model_weights_sha256=model_weights_identity(args.checkpoint),
        initial_reference=contract['reference'], saved_execution=saved,
        source_sha256=manifest['source_sha256'], source_indices=indices,
        condition='eval_no_grad_conditional_helper_latent_body_attention_T1',
        unmasked_tolerance=dict(atol=args.atol, rtol=args.rtol),
        complete_selected_subset=not args.limit or args.limit >= len(manifest['source_indices']),
        unmasked_checks=[dict(source_index=r['source_index'], **r['unmasked_oracle'])
                         for r in results if r.get('unmasked_oracle')])
    write_json(str(output)+'.summary.json', summary)


def generation_summary(args):
    manifest = json.loads(Path(args.manifest).read_text())
    baseline_metadata = json.loads(Path(args.baseline_manifest).read_text())
    if not baseline_metadata.get('successful'):
        raise ValueError('saved baseline manifest is not successful')
    saved_source = Path(baseline_metadata['dataset_identity']['path'])
    if hashlib.sha256(saved_source.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('saved baseline source identity mismatch')
    baseline = read_rows(args.on)
    spot, off = read_rows(args.spot), read_rows(args.off)
    indices = manifest['source_indices']
    # First two source rows, exact prompt/token/scorer identity before reuse.
    verify_baseline(baseline, spot, indices[:2])
    if any(r.get('disable_latent_generation', False) for r in spot):
        raise ValueError('baseline spot is not generation ON')
    profile = dict(baseline_metadata['generation_config'])
    if profile.pop('vision_ablation', 'none') != 'none':
        raise ValueError('baseline uses vision ablation')
    model_id = baseline_metadata['model_identity']
    for row in spot+off:
        if (row.get('model_identity') != model_id or row.get('generation_profile') != profile):
            raise ValueError('saved baseline model/profile identity mismatch')
    artifacts = {r.get('model_artifact') for r in spot+off}
    if not artifacts or None in artifacts:
        raise ValueError('generation model artifact identity missing')
    weight_identities = {model_weights_identity(path) for path in artifacts}
    if len(weight_identities) != 1:
        raise ValueError('generation model weight identity mismatch')
    # Actual local subset data hash is explicit; then map local indices to source.
    if any(r.get('source_sha256') != manifest['subset_sha256'] for r in off):
        raise ValueError('OFF subset identity mismatch')
    if len(off) != len(indices) or {r['index'] for r in off} != set(range(len(indices))):
        raise ValueError('OFF coverage mismatch')
    baseline_by_index = {r['index']: r for r in baseline}
    on = [dict(baseline_by_index[i], model_identity=model_id, generation_profile=profile,
               source_sha256=manifest['source_sha256'], disable_latent_generation=False) for i in indices]
    off = [dict(r, index=indices[r['index']], source_sha256=manifest['source_sha256']) for r in off]
    result = pair_generation(on, off, indices, seed=args.seed)
    result.update(model_identity=model_id, generation_profile=profile,
        model_weights_sha256=next(iter(weight_identities)), model_artifacts=sorted(artifacts),
        source_sha256=manifest['source_sha256'], baseline_spot_sources=indices[:2],
        baseline_reuse='exact_prompt_suffix_and_scoring_first_two')
    write_json(args.output, result)


def report(args):
    generations = [json.loads(Path(p).read_text()) for p in args.generation]
    voffs = [json.loads(Path(p).read_text()) for p in args.voff]
    for generation, voff_summary in zip(generations, voffs):
        identity = generation.get('model_weights_sha256')
        if not identity or identity != voff_summary.get('model_weights_sha256'):
            raise ValueError('generation/TRAIN model weight identity mismatch')
    if any(not v.get('complete_selected_subset') for v in voffs):
        raise ValueError('cannot report incomplete selected TRAIN coverage')
    lines = ['# Stage1-v5 latent causal analysis', '',
        'Generation suppression on fixed TEST100 and conditional helper-latent body-attention sensitivity on TRAIN64 answer different questions.', '']
    for name, generation, v in zip(args.names, generations, voffs):
        lines += [f'## {name}', '',
            f"TEST generation ON/OFF: {generation['legacy_on_accuracy']:.3%} / {generation['legacy_off_accuracy']:.3%} over {generation['source_count']} rows; valid gold {generation['valid_gold_count']}.",
            f"Paired OFF−ON accuracy effect {generation['effect_off_minus_on']:.6f}, 95% bootstrap CI {generation['paired_bootstrap_ci']}; ON-correct/OFF-wrong {generation['on_correct_off_wrong']}, reverse {generation['on_wrong_off_correct']}.",
            f"Answer changes {generation['answer_changes']}; excluded invalid gold {generation['excluded_invalid_gold']}; errors {generation['excluded_errors']}.",
            f"TRAIN paired sample means: {json.dumps(v['sample_mean'], sort_keys=True)}.",
            f"TRAIN sample bootstrap intervals: {json.dumps(v['sample_bootstrap_ci'], sort_keys=True)}.",
            f"TRAIN token-weighted means: {json.dumps(v['token_weighted'], sort_keys=True)}.",
            f"First-two unmasked numerical checks: {json.dumps(v['unmasked_checks'], sort_keys=True)}.",
            f"Eligible samples {v['eligible_samples']}/{v['source_count']}; targets {v['target_count']}; empty blocks {v['empty_blocks']}; exclusions {v['exclusions']}.", '']
    if len(generations) == 2:
        a, b = generations
        if a['source_sha256'] != b['source_sha256'] or set(a['paired_effects']) != set(b['paired_effects']):
            raise ValueError('model contrast requires identical inferential source pairs')
        contrast = [b['paired_effects'][i]-a['paired_effects'][i] for i in a['paired_effects']]
        lines += [f"Model B minus A paired generation intervention effect (OFF−ON): {sum(contrast)/len(contrast):.6f}, 95% bootstrap CI {bootstrap_mean(contrast)}.", '']
        a, b = voffs
        if a['source_sha256'] != b['source_sha256'] or set(a['paired_sample_metrics']) != set(b['paired_sample_metrics']):
            raise ValueError('TRAIN model contrast requires identical eligible source pairs')
        contrast = [b['paired_sample_metrics'][i]['delta_ce']-a['paired_sample_metrics'][i]['delta_ce']
                    for i in a['paired_sample_metrics']]
        lines += [f"Model B minus A paired TRAIN intervention effect (ΔCE): {sum(contrast)/len(contrast):.6f}, 95% bootstrap CI {bootstrap_mean(contrast)}.", '']
    lines += ['No replicated-seed or compute-matched utility claim. TRAIN metrics use dropout-disabled helper latents and temperature 1; they are not held-out accuracy or the full stochastic training loss.', '']
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('select')
    p.add_argument('--source', required=True)
    p.add_argument('--indices')
    p.add_argument('--count', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--expected-sha256')
    p.add_argument('--limit', type=int)
    p.add_argument('--verify-subset')
    p.add_argument('--absolute-images', action='store_true')
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', required=True)
    p.set_defaults(action=select)
    p = sub.add_parser('voff')
    for name in ('checkpoint', 'base', 'manifest', 'image-root', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int)
    p.add_argument('--check-unmasked', action='store_true')
    p.add_argument('--atol', type=float, default=.02)
    p.add_argument('--rtol', type=float, default=.02)
    p.set_defaults(action=voff)
    p = sub.add_parser('generation-summary')
    for name in ('on', 'off', 'spot', 'manifest', 'baseline-manifest', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(action=generation_summary)
    p = sub.add_parser('report')
    p.add_argument('--generation', nargs='+', required=True)
    p.add_argument('--voff', nargs='+', required=True)
    p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(action=report)
    args = parser.parse_args(argv)
    if getattr(args, 'limit', None) is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.command == 'report' and not len(args.generation) == len(args.voff) == len(args.names):
        parser.error('report needs one generation and V-off summary per name')
    args.action(args)


if __name__ == '__main__':
    main()
