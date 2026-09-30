#!/usr/bin/env python3
"""CPU integration witness for two isolated real DIL TRAIN input pipelines.

No model is instantiated. Each worker imports exactly one runtime tree and emits
compact hashes; masks are evaluated exhaustively in bounded query chunks.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def tensor_record(tensor):
    value = tensor.detach().cpu().contiguous()
    return {'dtype': str(value.dtype), 'shape': list(value.shape),
            'sha256': digest_bytes(value.numpy().tobytes())}


def differing_fields(left, right, prefix=''):
    """Return every unequal leaf, including missing fields and changed lengths."""
    if isinstance(left, dict) and isinstance(right, dict):
        fields = []
        for key in sorted(left.keys() | right.keys()):
            path = f'{prefix}.{key}' if prefix else key
            if key not in left or key not in right:
                fields.append(path)
            else:
                fields.extend(differing_fields(left[key], right[key], path))
        return fields
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [prefix + '.length']
        return [field for i, (a, b) in enumerate(zip(left, right))
                for field in differing_fields(a, b, f'{prefix}[{i}]')]
    return [] if left == right else [prefix]


def visibility_record(callback, query_count, key_count, chunk=64):
    """Hash every logical edge in canonical root/step/local order, plus padding."""
    import torch
    sha = hashlib.sha256()
    visible = edges = 0
    keys = torch.arange(key_count)[None, :]
    for start in range(0, query_count, chunk):
        queries = torch.arange(start, min(start + chunk, query_count))[:, None]
        values = callback(0, 0, queries, keys)
        if values.shape != (queries.numel(), key_count) or values.dtype != torch.bool:
            raise ValueError('visibility callback must emit every requested boolean edge')
        sha.update(values.contiguous().numpy().tobytes())
        edges += values.numel()
        visible += int(values.sum())
    # Flex probes padded lanes up to whole 128-sized blocks. Examine all of them.
    padded_q = ((query_count + 127) // 128) * 128
    padded_k = ((key_count + 127) // 128) * 128
    rejected = padded_edges = 0
    for start in range(0, padded_q, chunk):
        queries = torch.arange(start, min(start + chunk, padded_q))[:, None]
        keys = torch.arange(padded_k)[None, :]
        padding = (queries >= query_count) | (keys >= key_count)
        values = callback(0, 0, queries, keys)
        rejected += int((values & padding).sum())
        padded_edges += int(padding.sum())
    return {'sha256': sha.hexdigest(), 'shape': [query_count, key_count],
            'edges_examined': edges, 'visible_edges': visible,
            'padded_edges_examined': padded_edges, 'padded_visible_edges': rejected}


class CaptureTokenizer:
    """Observe the actual processor-expanded tokenizer input without altering it."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.text = None

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, text, *args, **kwargs):
        self.text = text
        return self.tokenizer(text, *args, **kwargs)


def file_hash(path):
    return digest_bytes(Path(path).read_bytes())


def worker(args):
    import torch
    import transformers
    from transformers import AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration
    from stage2.checkpointing import resolve_latent_token_ids
    from stage2.data import prepare_stage2_example, Stage2DataCollator
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import parse_latent_spans
    torch.set_num_threads(1)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('diagnostic requires CUDA hidden')
    root = Path(args.tree).resolve()
    port = args.kind == 'port'
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    config = AutoConfig.from_pretrained(args.config, local_files_only=True)
    token_ids = resolve_latent_token_ids(processor.tokenizer, config)
    capture = CaptureTokenizer(processor.tokenizer)
    processor.tokenizer = capture
    kwargs = ({'step_preference_enabled': True, 'preference_target_scope': 'until_next_latent'}
              if port else {'context_preference_enabled': True, 'target_scope': 'until_next_latent'})
    collator = Stage2DataCollator(processor, token_ids, 8, True, **kwargs)
    manifest = json.loads(Path(args.manifest).read_text())
    retained = manifest['retained_indices']
    if args.limit:
        retained = retained[:args.limit]
    raw = Path(args.data).read_bytes()
    if digest_bytes(raw) != manifest['data_sha256']:
        raise ValueError('TRAIN data differs from manifest')
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    modules = [inspect.getmodule(prepare_stage2_example), inspect.getmodule(parse_latent_spans),
               inspect.getmodule(Qwen2_5_VLForConditionalGeneration), inspect.getmodule(type(processor)),
               inspect.getmodule(type(capture.tokenizer)), inspect.getmodule(type(processor.image_processor))]
    paths = [Path(module.__file__) for module in modules]
    paths += [root / 'src/main.py', root / 'src/stage2/trainer.py', root / 'src/step_preference/spans.py',
              root / ('src/step_preference/sidecar.py' if port else 'src/step_preference/packing.py'),
              root / ('transformers/src/transformers/models/qwen2_5_vl/stage2_shared_prefix.py' if port else
                      'transformers/src/transformers/models/qwen2_5_vl/stage2_sidecar.py')]
    if any(not path.resolve().is_relative_to(root) for path in paths):
        raise ValueError('runtime imports escaped the selected source/port tree')
    seal = {str(path): file_hash(path) for path in paths}
    provenance = {'tree': str(root), 'kind': args.kind, 'imports': seal,
                  'transformers_path': transformers.__file__, 'data_sha256': digest_bytes(raw),
                  'manifest_sha256': file_hash(args.manifest),
                  'processor_files': {p.name: file_hash(p) for p in sorted(Path(args.processor).iterdir())
                                      if p.suffix in {'.json', '.txt'}},
                  'config_sha256': file_hash(Path(args.config) / 'config.json'),
                  'settings': {'latent_size': 8, 'supervised_boundaries': True, 'scope': 'until_next_latent',
                               'rendering': 'concatenated-v1', 'V_only': True, 'history': 'clean_prefix',
                               'tokenizer_is_fast': capture.tokenizer.is_fast, 'image_processor_class': type(processor.image_processor).__name__, 'query_chunk': 64, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
                  'special_token_ids': dataclasses.asdict(token_ids),
                  'image_processor': processor.image_processor.to_dict(),
                  'chat_template': processor.chat_template,
                  'model_weights_loaded': False}
    provenance['processor_files_sha256'] = digest_bytes(json.dumps(provenance['processor_files'], sort_keys=True).encode())
    (output / f'{args.kind}-provenance.json').write_text(json.dumps(provenance, indent=2, default=str) + '\n')
    started = time.perf_counter()
    with (output / f'{args.kind}-rows.jsonl').open('w') as handle:
        for ordinal, index in enumerate(retained):
            sample = rows[index]
            kwargs = {'require_reasoning_units': True, 'assistant_plan_rendering': 'concatenated-v1'} if port else {}
            example = prepare_stage2_example(sample, image_root=Path(args.image_root), latent_size=8, **kwargs)
            batch = collator([example])
            ids, attention = batch['input_ids'], batch['attention_mask']
            if ids.shape[1] != manifest['lengths'][index]:
                raise ValueError(f'row {index} length disagrees with manifest')
            position, delta = Qwen2_5_VLForConditionalGeneration.get_rope_index(
                SimpleNamespace(config=config), input_ids=ids, image_grid_thw=batch.get('image_grid_thw'),
                attention_mask=attention)
            latent_spans = parse_latent_spans(ids[0], token_ids=token_ids, latent_size=8)
            spans = batch['reasoning_step_spans'] if port else batch['reasoning_step_spans'][0]
            record = {'index': index, 'sample_id': example.sample_id,
                      'tensors': {key: tensor_record(value) for key, value in batch.items() if isinstance(value, torch.Tensor)},
                      'positions': tensor_record(position), 'rope_delta': tensor_record(delta),
                      'latent_spans': [dataclasses.asdict(span) for span in latent_spans],
                      'reasoning_spans': [dataclasses.asdict(span) for span in spans],
                      'assistant_text': digest_bytes(example.conversation[1]['content'][0]['text'].encode()),
                      'image_bytes': [file_hash(path) for path in example.original_image_paths]}
            # Preserve processor-produced token IDs/offsets and actual crossing-token membership.
            expanded = capture.text
            if isinstance(expanded, (tuple, list)):
                expanded = expanded[0]
            offsets = capture.tokenizer([expanded], return_tensors='pt', return_offsets_mapping=True,
                                        padding=True, truncation=False)
            if not torch.equal(offsets['input_ids'], ids):
                raise ValueError('actual processor-expanded offsets disagree with IDs')
            record['expanded_text'] = digest_bytes(expanded.encode())
            record['offsets'] = tensor_record(offsets['offset_mapping'])
            assistant = example.conversation[1]['content'][0]['text']
            char_start = expanded.rfind(assistant)
            if char_start < 0:
                raise ValueError('assistant absent from actual expanded text')
            joins = []
            cursor = char_start
            for left, right in zip(sample['sequence_plan'], sample['sequence_plan'][1:]):
                cursor += len(left['content']) if left['type'] == 'text' else len('<|latent_start|>' + '<|latent_pad|>' * 8 + '<|latent_end|>')
                if left['type'] == right['type'] == 'text':
                    crossing = [i for i, (a, b) in enumerate(offsets['offset_mapping'][0].tolist()) if a < cursor < b]
                    joins.append({'char': cursor, 'crossing': [{'position': i, 'token_id': int(ids[0, i]),
                                  'offset': offsets['offset_mapping'][0, i].tolist(), 'ce_label': int(batch['labels'][0, i]),
                                  'target_steps': [s.step_id for s in spans if s.target_text_start <= i < s.target_text_end],
                                  'predictor_steps': [s.step_id for s in spans if s.target_text_start - 1 <= i < s.target_text_end - 1]}
                                  for i in crossing]})
            record['text_joins'] = joins
            if port:
                from step_preference.sidecar import build_context_sidecar_plan, build_sidecar_flex_block_mask
                import torch.nn.attention.flex_attention as flex
                plan = build_context_sidecar_plan(clean_input_ids=ids, attention_mask=attention, spans=spans,
                                                 visual_only=True, suppress_latent_boundaries=False)
                branches = plan.groups[0]
                original = flex.create_block_mask
                flex.create_block_mask = lambda fn, **kw: (fn, kw)
                try:
                    callback, builder = build_sidecar_flex_block_mask(plan.root_length, branches, 'cpu')
                finally:
                    flex.create_block_mask = original
                normalized = [{'step_id': b.step_id, 'replay_boundary': b.clean_ancestor_end,
                               'blocked': [b.blocked_start, b.blocked_end],
                               'input': tensor_record(torch.tensor(b.token_ids, dtype=torch.long)),
                               'targets': tensor_record(torch.tensor(b.target_token_ids, dtype=torch.long)),
                               'predictors': list(b.logical_positions), 'selected': list(b.selected_positions),
                               'position': tensor_record(position[..., list(b.logical_positions)])} for b in branches]
                root_selected = list(plan.root_selected_positions)
                names = [b.name for b in branches]
            else:
                from step_preference.packing import build_context_sidecar_execution_plan
                from transformers.models.qwen2_5_vl.stage2_sidecar import pack_stage2_sidecar_plan, _sidecar_visibility
                items = batch['step_preference_items']
                plan = build_context_sidecar_execution_plan(clean_input_ids=ids, items=items, ablation='attention_mask',
                                                           modalities='visual', stage2_debugging=True)
                layout, = pack_stage2_sidecar_plan(plan, clean_input_ids=ids, clean_position_ids=position,
                                                   clean_attention_mask=attention)
                callback = _sidecar_visibility(layout, torch.device('cpu'))
                builder = {'Q_LEN': layout.input_token_ids.shape[1], 'KV_LEN': ids.shape[1] + layout.input_token_ids.shape[1]}
                normalized = [{'step_id': b.step_id, 'replay_boundary': b.replay_boundary,
                               'blocked': list(b.blocked_clean_interval),
                               'input': tensor_record(layout.input_token_ids[0, b.query_start:b.query_end]),
                               'targets': tensor_record(b.target_token_ids),
                               'predictors': items[n].chosen_predictor_positions.tolist(),
                               'selected': b.score_offsets.tolist(),
                               'position': tensor_record(layout.position_ids[..., b.query_start:b.query_end])}
                              for n, b in enumerate(layout.branches)]
                root_selected = [p for item in items for p in item.chosen_predictor_positions.tolist()]
                names = [b.branch_key for b in layout.branches]
            record['active_plan'] = {'branches': normalized, 'root_selected': root_selected}
            record['layout_metadata'] = {'branch_names': names, 'mask_callback_intercepted': port}
            record['visibility'] = visibility_record(callback, builder['Q_LEN'], builder['KV_LEN'])
            if record['visibility']['padded_visible_edges']:
                raise ValueError(f'row {index} exposes physical padded mask lanes')
            handle.write(json.dumps(record, separators=(',', ':')) + '\n')
            if ordinal % 100 == 0:
                handle.flush()
                print(json.dumps({'kind': args.kind, 'completed': ordinal + 1, 'index': index,
                                  'seconds': time.perf_counter() - started}), flush=True)
    if any(file_hash(path) != sealed for path, sealed in seal.items()):
        raise ValueError('runtime files changed during witness')
    print(json.dumps({'kind': args.kind, 'completed': len(retained), 'seconds': time.perf_counter() - started}), flush=True)


def compare(args):
    output = Path(args.output)
    manifest = json.loads(Path(args.manifest).read_text())
    expected = manifest['retained_indices'][:args.limit] if args.limit else manifest['retained_indices']
    source = [json.loads(line) for line in (output / 'source-rows.jsonl').read_text().splitlines()]
    port = [json.loads(line) for line in (output / 'port-rows.jsonl').read_text().splitlines()]
    if [r['index'] for r in source] != expected or [r['index'] for r in port] != expected:
        raise ValueError('coverage is incomplete, duplicated, reordered, or skipped')
    counts, examples = {}, {}
    active = metadata = edges = 0
    for left, right in zip(source, port):
        fields = differing_fields(left, right)
        for field in fields:
            category = field.split('.')[0].split('[')[0]
            counts[category] = counts.get(category, 0) + 1
            examples.setdefault(category, [])
            if len(examples[category]) < 4:
                examples[category].append({'index': left['index'], 'field': field})
            if category in {'reasoning_spans', 'layout_metadata'}:
                metadata += 1
            else:
                active += 1
        edges += left['visibility']['edges_examined']
    summary = {'rows': len(source), 'expected_rows': len(expected), 'full_cohort': not bool(args.limit),
               'active_mismatching_fields': active, 'metadata_mismatching_fields': metadata,
               'mismatch_counts': counts, 'first_examples': examples, 'logical_edges_per_tree': edges,
               'row1078': next((r for r in source if r['index'] == 1078), None),
               'limitations': ['CPU logical inputs/masks only; no GPU, backward, optimizer or historical binary equivalence.',
                               'Copied Git HEAD1f95f6c5 is recoverable via explicit copied gitdir; September27 full imported runtime hashes remain unverified.']}
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'row1078'}, indent=2))
    return int(active != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['worker', 'compare', 'run'], default='run')
    parser.add_argument('--kind', choices=['source', 'port'])
    parser.add_argument('--tree')
    parser.add_argument('--source')
    parser.add_argument('--port')
    parser.add_argument('--processor', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--logs')
    parser.add_argument('--limit', type=int, default=0, help='Explicit partial development probe; zero means all retained rows.')
    args = parser.parse_args()
    if args.mode == 'worker':
        worker(args)
        return 0
    if args.mode == 'run':
        for kind, tree in [('source', args.source), ('port', args.port)]:
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1',
                               TRANSFORMERS_OFFLINE='1', PYTHONPATH=f'{tree}/transformers/src:{tree}/src:{tree}')
            command = [sys.executable, str(Path(__file__).resolve()), '--mode', 'worker', '--kind', kind,
                       '--tree', tree]
            for key in ['processor', 'config', 'data', 'image_root', 'manifest', 'output', 'limit']:
                command += ['--' + key.replace('_', '-'), str(getattr(args, key))]
            Path(args.logs).mkdir(parents=True, exist_ok=True)
            with (Path(args.logs) / f'train-inputs-{kind}.log').open('w') as log:
                subprocess.run(command, env=environment, cwd=tree, stdout=log, stderr=subprocess.STDOUT, check=True)
    return compare(args)


if __name__ == '__main__':
    raise SystemExit(main())
