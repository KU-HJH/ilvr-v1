#!/usr/bin/env python
"""Local final-PEFT, fixed-clean-bank body-key evaluation of an independent cohort.

Full-length SDPA MATH replay is deliberate, not a claim of historical external
kernel identity. Targets come from one full processor stream, never the collator.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


def extract_sites(ids, assistant_start, assistant_end, token_ids, latent_size):
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import parse_latent_spans
    if ids.ndim != 2 or ids.shape[0] != 1 or not 0 <= assistant_start < assistant_end <= ids.shape[1]:
        raise ValueError('invalid assistant bounds or input shape')
    spans = parse_latent_spans(ids[0], token_ids=token_ids, latent_size=latent_size)
    sites = []
    for index, span in enumerate(spans):
        start = span.end_position + 1
        end = spans[index + 1].start_position if index + 1 < len(spans) else assistant_end
        if span.start_position < assistant_start or span.end_position >= assistant_end or start >= end:
            raise ValueError('empty target or invalid latent block bounds')
        sites.append(dict(block_index=index, body_positions=list(span.latent_positions),
                          target_start=start, target_end=end, target_count=end-start,
                          target_ids=ids[0, start:end].tolist()))
    if not sites:
        raise ValueError('no eligible sites')
    return sites


def validate_sites(sites, expected):
    if len(sites) != len(expected):
        raise ValueError(f'independent site count differs: {len(sites)} != {len(expected)}')
    for actual, oracle in zip(sites, expected, strict=True):
        for key in ('block_index', 'body_positions', 'target_start', 'target_end', 'target_count', 'target_ids'):
            if actual[key] != oracle[key]:
                raise ValueError(f'independent {key} differs at block {actual["block_index"]}')


def summarize(records):
    if not records:
        raise ValueError('empty evaluation')
    seen = set()
    first = any_count = changed = total = 0
    nll = clean_nll = 0.
    for row in records:
        key = row['sample_id'], row['block_index']
        if key in seen:
            raise ValueError('duplicate evaluation site')
        seen.add(key)
        a, b = row['clean_target_logps'], row['target_logps']
        x, y = row['clean_top1_ids'], row['top1_ids']
        if not a or not len(a) == len(b) == len(x) == len(y):
            raise ValueError('unaligned evaluation scores')
        if not all(math.isfinite(v) for v in a + b):
            raise ValueError('nonfinite evaluation scores')
        diffs = [i != j for i, j in zip(x, y)]
        first += diffs[0]; any_count += any(diffs); changed += sum(diffs)
        total += len(a); clean_nll -= math.fsum(a); nll -= math.fsum(b)
    return dict(sites=len(records), tokens=total, first_changed=first, any_changed=any_count,
                tokens_changed=changed, first_changed_rate=first/len(records),
                any_changed_rate=any_count/len(records), tokens_changed_rate=changed/total,
                clean_mean_nll=clean_nll/total, mean_nll=nll/total, delta_nll=(nll-clean_nll)/total)


@contextmanager
def numerical_policy():
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


def _qwen(model):
    from peft_common import resolve_qwen2_5_vl_model
    return resolve_qwen2_5_vl_model(model)


def _scores(logits, targets):
    import torch
    logits = logits.float()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('nonfinite logits')
    return dict(logps=logits.log_softmax(-1).gather(-1, targets.to(logits.device)[:, None]).squeeze(-1).cpu(),
                top1=logits.argmax(-1).cpu())


def capture_bank(model, sample):
    """Capture exactly the actual native final decoder inputs, including image/latents."""
    import torch
    qwen = _qwen(model)
    inputs = {k: v.to(model.device) for k, v in sample.model_inputs.items()}
    positions = [p-1 for site in sample.sites for p in range(site['target_start'], site['target_end'])]
    targets = torch.tensor([t for site in sample.sites for t in site['target_ids']], device=model.device)
    captured = []
    def observe(module, args, kwargs):
        embeds = kwargs.get('inputs_embeds')
        if embeds is not None and embeds.shape[1] == inputs['input_ids'].shape[1]:
            if kwargs.get('stage2_capture_kv') or kwargs.get('stage2_prefix_key_values') is not None:
                raise ValueError('native final call unexpectedly used sidecar KV')
            captured.append(dict(embeds=embeds.detach().clone(), positions=kwargs['position_ids'].detach().clone()))
    hook = qwen.model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        output = model(**inputs, labels=None, stage2_shared_prefix_inputs=None,
                       stage2_selective_logits_only=True,
                       stage2_logit_positions=torch.tensor(positions, device=model.device),
                       use_cache=False, return_dict=True)
    finally:
        hook.remove()
    if len(captured) != 1 or not bool(torch.isfinite(captured[0]['embeds']).all()):
        raise ValueError('native final bank must be unique and finite')
    return captured[0], _scores(output.stage2_selected_logits[0], targets)


def replay_scores(model, bank, sample, site, body_positions=()):
    """Full sequence, fixed positions/bank; zero only selected body KEY mask entries."""
    import torch
    qwen = _qwen(model)
    mask = sample.model_inputs['attention_mask'].to(bank['embeds'].device).clone()
    body = list(body_positions)
    if body and body != site['body_positions']:
        raise ValueError('intervention must mask exactly current body keys')
    mask[0, body] = 0
    hidden = qwen.model(inputs_embeds=bank['embeds'], position_ids=bank['positions'],
                        attention_mask=mask, generate_mode=True, use_cache=False,
                        output_attentions=False, output_hidden_states=False,
                        return_dict=True).last_hidden_state
    predictors = torch.arange(site['target_start']-1, site['target_end']-1, device=hidden.device)
    logits = qwen.lm_head(hidden[0, predictors])
    return _scores(logits, torch.tensor(site['target_ids'], device=hidden.device))


def verify_identity(native, replay, *, exact=False):
    import torch
    a, b = native['logps'], replay['logps']
    if a.shape != b.shape or not a.numel() or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError('identity shape/finiteness failed')
    maximum = float((a-b).abs().max())
    mean = float((a.mean()-b.mean()).abs())
    if maximum > (0 if exact else 1e-3) or mean > (0 if exact else 1e-4):
        raise ValueError(f'identity failed: logp={maximum}, mean NLL={mean}')
    if exact and not torch.equal(native['top1'], replay['top1']):
        raise ValueError('no-op argmax identity failed')
    return dict(max_logp_error=maximum, nll_error=mean,
                argmax_equal=bool(torch.equal(native['top1'], replay['top1'])))


def prepare_sample(row, processor, token_ids, image_root, latent_size, max_tokens):
    from stage2.data import prepare_stage2_example, _load_original_images, find_unique_subsequence_end
    example = prepare_stage2_example(row, image_root=Path(image_root), latent_size=latent_size,
                                     assistant_plan_rendering='concatenated-v1')
    images = _load_original_images(example.original_image_paths)
    try:
        batch = processor(text=[processor.apply_chat_template(example.conversation,
                          tokenize=False, add_generation_prompt=False)], images=[images],
                          padding=False, truncation=False, return_tensors='pt')
    finally:
        for image in images:
            image.close()
    if {'pixel_values_latent', 'image_grid_thw_latent'}.intersection(batch):
        raise ValueError('helper-image features entered evaluation')
    ids = batch['input_ids']
    if ids.shape[1] > max_tokens:
        raise ValueError('sample exceeds maximum tokens; cohort cannot be silently reduced')
    def pattern(text):
        return processor.tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    start = find_unique_subsequence_end(ids[0], pattern('<|im_start|>assistant'))
    ending = pattern('<|im_end|>')
    end = start + find_unique_subsequence_end(ids[0, start:], ending) - ending.numel()
    sites = extract_sites(ids, start, end, token_ids, latent_size)
    if len(sites) != example.latent_block_count:
        raise ValueError('processor lost latent blocks')
    forbidden = {token_ids.pad_token_id, token_ids.start_token_id, token_ids.end_token_id,
                 processor.tokenizer.pad_token_id, *ending.tolist()}
    for site in sites:
        if forbidden.intersection(site['target_ids']) or not bool(batch['attention_mask'][0, site['target_start']:site['target_end']].all()):
            raise ValueError('control or padding in target')
    return SimpleNamespace(sample_id=example.sample_id, sites=sites,
                           model_inputs={k: v for k, v in batch.items() if k in
                                         ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw')})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    import torch
    from transformers import AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import validate_stage2_config
    from stage2.peft import load_stage2_model_for_inference
    from stage2.attention_backend import load_qwen2_5_vl_with_attention
    from scripts.verify_until_inference import artifact_witness
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to((root/'results').resolve()) or output == (root/'results').resolve():
        raise ValueError('output must be below current worktree results')
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a fresh output directory')
    for path in (args.artifact, args.base):
        if not Path(path).resolve().is_relative_to((root/'checkpoints').resolve()):
            raise ValueError('model artifacts must be under checkpoints')
    if not (Path(args.artifact)/'stage2_peft_metadata.json').is_file():
        raise ValueError('requires final Stage2 PEFT artifact')
    cohort = json.loads(Path(args.cohort).read_text())
    reference = json.loads(Path(args.target_manifest).read_text())
    indices = cohort['selected_indices']
    if len(indices) != 100 or len(set(indices)) != 100 or sha256(args.data) != cohort['test_sha256']:
        raise ValueError('fixed100 cohort or dataset hash mismatch')
    oracle = reference['records']
    if {r['test_index'] for r in oracle} != set(indices):
        raise ValueError('independent target cohort differs')
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    processor = AutoProcessor.from_pretrained(args.artifact, local_files_only=True)
    config = AutoConfig.from_pretrained(args.artifact, local_files_only=True)
    token_ids, latent_size = validate_stage2_config(config)
    if config.stage2_execution_profile != 'latest_external_until' or latent_size != 8:
        raise ValueError('artifact is not latest_external_until latent8')
    prepared = []
    for index in indices:
        sample = prepare_sample(rows[index], processor, token_ids, args.image_root, latent_size, args.max_tokens)
        expected = [r for r in oracle if r['test_index'] == index]
        if any(r['sample_id'] != sample.sample_id for r in expected):
            raise ValueError(f'independent sample identity differs at index {index}')
        validate_sites(sample.sites, expected)
        prepared.append((index, sample))
    counts = dict(samples=len(prepared), sites=sum(len(s.sites) for _, s in prepared),
                  tokens=sum(b['target_count'] for _, s in prepared for b in s.sites))
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(counts=counts, cohort_sha256=sha256(args.cohort),
                    target_manifest_sha256=sha256(args.target_manifest), data_sha256=sha256(args.data),
                    artifact=str(Path(args.artifact).resolve()), base=str(Path(args.base).resolve()),
                    protocol='full-length fixed native clean bank; current body KEY mask; all GT until next START/assistant end',
                    decoder_dtype='float32', vision_dtype='bfloat16', vision_attention='flash_attention_2',
                    decoder_attention='sdpa', sdpa_backend='MATH', tf32=False,
                    native_identity_logp_atol=1e-3, native_identity_site_nll_atol=1e-4,
                    clean_noop_identity='exact logps and argmax',
                    historical_external_kernel_equivalence=False)
    (output/'preflight.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if args.preflight_only:
        return metadata
    if not args.device.startswith('cuda'):
        raise ValueError('full evaluation requires CUDA')
    def load(source, **kwargs):
        return load_qwen2_5_vl_with_attention(source, attention_implementation='sdpa',
            vision_attention_implementation='flash_attention_2', config_load_kwargs={'local_files_only':True},
            model_load_kwargs={'local_files_only':True, 'torch_dtype':torch.float32},
            load_model=Qwen2_5_VLForConditionalGeneration.from_pretrained)
    model = load_stage2_model_for_inference(args.base, args.artifact, processor,
        source_provenance_kwargs={'local_files_only':True}, load_model=load)
    model.eval().requires_grad_(False).to(device=args.device, dtype=torch.float32)
    qwen = _qwen(model)
    qwen.visual.to(dtype=torch.bfloat16)
    if (qwen.config.stage2_execution_profile != 'latest_external_until'
        or qwen.config.vision_config.stage2_execution_profile != 'latest_external_until'
        or any(b.attn.stage2_execution_profile != 'latest_external_until' for b in qwen.visual.blocks)):
        raise ValueError('saved execution profile was not restored')
    if any(p.dtype != torch.float32 for p in qwen.model.parameters() if p.is_floating_point()):
        raise ValueError('decoder must be FP32')
    witness = artifact_witness(model, args.artifact)
    (output/'artifact-witness.json').write_text(json.dumps(witness, indent=2)+'\n')
    if not witness['all_restored']:
        raise ValueError('PEFT artifact restoration mismatch')
    records = []
    with torch.inference_mode(), numerical_policy(), (output/'per_block.jsonl').open('w') as handle:
        for index, sample in prepared:
            bank, native = capture_bank(model, sample)
            cursor = 0
            for site in sample.sites:
                clean = replay_scores(model, bank, sample, site)
                noop = replay_scores(model, bank, sample, site, body_positions=[])
                n = site['target_count']
                normal = {k:v[cursor:cursor+n] for k,v in native.items()}
                identity = verify_identity(normal, clean)
                noop_identity = verify_identity(clean, noop, exact=True)
                masked = replay_scores(model, bank, sample, site, body_positions=site['body_positions'])
                row = dict(test_index=index, sample_id=sample.sample_id, **site, condition='mask_attention',
                           identity=identity, noop_identity=noop_identity,
                           clean_target_logps=clean['logps'].tolist(), target_logps=masked['logps'].tolist(),
                           clean_top1_ids=clean['top1'].tolist(), top1_ids=masked['top1'].tolist())
                records.append(row); handle.write(json.dumps(row)+'\n'); handle.flush()
                cursor += n
            del bank, native
            print(json.dumps(dict(completed_samples=indices.index(index)+1, test_index=index)), flush=True)
    summary = dict(metadata, status='complete', mask_attention=summarize(records),
                   clean=summarize([dict(r, target_logps=r['clean_target_logps'], top1_ids=r['clean_top1_ids']) for r in records]))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('artifact', 'base', 'cohort', 'target-manifest', 'data', 'image-root', 'output-dir'):
        parser.add_argument('--'+flag, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-tokens', type=int, default=128000)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
