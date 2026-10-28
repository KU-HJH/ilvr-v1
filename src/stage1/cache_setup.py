"""Launch/resume/completion policy, outside the Stage1 training hot path."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import shutil

import numpy as np
import torch

from .cache_lifecycle import publish_json
from .run_cache import Stage1CachePlan, Stage1RunCache, file_sha256


CACHE_CONTRACT = 'stage1_run_cache.json'
INITIAL_TEACHER = 'stage1_cache_initial_teacher.pt'
SSD_ROOT = Path('/root/workspace/ILVR-stage1-cache')


def load_stage1_cache_contract(checkpoint, *, requested):
    if checkpoint is None:
        return None
    path = Path(checkpoint) / CACHE_CONTRACT
    if not path.exists():
        if requested != 'off':
            raise ValueError('old resume has no reconstructible initial cache contract')
        return {'check_policy': 'legacy', 'mode': 'off'}
    value = json.loads(path.read_text())
    if value.get('schema') != 1 or value.get('mode') != requested:
        raise ValueError('resume must preserve its saved cache mode/contract')
    return value


def write_stage1_cache_contract(checkpoint, plan, *, phase, check_policy='production', completion='train'):
    payload = {'schema': 1, 'mode': 'off' if plan is None else 'run',
               'check_policy': check_policy, 'completion': completion, 'phase': phase}
    if plan is not None:
        payload['plan'] = {**asdict(plan), 'root': str(plan.root)}
    publish_json(Path(checkpoint) / CACHE_CONTRACT, payload)


def expose_cache_on_ssd(cache, *, logical_root):
    logical_root = Path(logical_root)
    logical_root.mkdir(parents=True, exist_ok=True)
    link = logical_root / cache.plan.run_id
    try:
        link.symlink_to(cache.plan.root)
    except FileExistsError:
        if not link.is_symlink() or link.resolve() != cache.plan.root.resolve():
            raise ValueError('cache logical namespace belongs to another object')


def remove_owned_cache(cache, *, logical_root):
    if not cache._closed:
        raise RuntimeError('cache readers must be closed before deletion')
    root = cache.plan.root
    link = Path(logical_root) / cache.plan.run_id
    if root.is_symlink() or (root / 'identity.json').read_text() != cache.identity_json:
        raise ValueError('cache deletion identity mismatch')
    if not link.is_symlink() or link.resolve() != root.resolve():
        raise ValueError('cache deletion requires its exact logical namespace')
    # rmtree uses descriptor-relative traversal on this supported Linux runtime;
    # internal symlinks are unlinked, never followed to foreign contents.
    shutil.rmtree(root)
    link.unlink()


def finish_stage1_run_cache(*, run_id, completion, runner):
    if completion != 'complete':
        return
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id) is None:
        raise ValueError('invalid cache completion run ID')
    for rank in (0, 1):
        runner.remove_cache(rank, run_id)


def close_stage1_cache_readers(trainer):
    cache = getattr(trainer, 'run_cache', None)
    if cache is None:
        return
    for loader in getattr(trainer.accelerator, '_dataloaders', ()):
        iterator = getattr(loader, '_iterator', None)
        if iterator is not None and hasattr(iterator, '_shutdown_workers'):
            iterator._shutdown_workers()
    cache.close()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if trainer.args.local_process_index == 0:
        publish_json(cache.plan.root / 'closed.json', {'identity': cache.identity_sha256,
            'run_id': cache.plan.run_id, 'attempt': os.environ['ILVR_STAGE1_CACHE_ATTEMPT']})


@contextmanager
def preserve_rng():
    cpu, python, numpy = torch.random.get_rng_state(), random.getstate(), np.random.get_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu)
        random.setstate(python)
        np.random.set_state(numpy)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def resolve_expected(cache, expected):
    result = dict(expected)
    if result.get('reference', ()) is None:
        eligible = []
        for index in result['preprocess']:
            if cache._path('preprocess', index).is_file():
                row = cache._load('preprocess', index)
                if any(block['target_positions'] for block in row['v5_blocks']):
                    eligible.append(index)
        result['reference'] = tuple(eligible)
    return result


def prepare_stage1_cache_resume(*, cache, checkpoint, initial_teacher, expected, rebuild,
                                finish_boundary=None, rank=0, world_size=1):
    saved = load_stage1_cache_contract(checkpoint, requested='run')
    expected_plan = {**asdict(cache.plan), 'root': str(cache.plan.root)}
    if saved['plan'] != json.loads(json.dumps(expected_plan)):
        raise ValueError('cache resume identity differs from checkpoint')
    state = json.loads((Path(checkpoint) / 'trainer_state.json').read_text())
    if float(state['epoch'] or 0) < 1:
        return
    needed = resolve_expected(cache, expected)
    missing = sorted({index for layer, indices in needed.items() for index in indices
                      if not cache._path(layer, index).is_file()})
    current = initial_teacher.state_dict()
    with preserve_rng():
        try:
            if cache.plan.helper == 'targets' and missing:
                initial_path = Path(checkpoint) / INITIAL_TEACHER
                if initial_path.exists():
                    initial_teacher.load_state_dict(torch.load(initial_path, map_location='cpu', weights_only=True))
                elif not (Path(checkpoint) / 'stage1_v5_reference.pt').is_file():
                    raise FileNotFoundError('missing initial teacher for cache rebuild')
                # V5 composition restores its original reference before here.
            # Every rank sees the same missing set after a lost cache. With
            # partial node-local loss each node rebuilds its own missing set;
            # rank modulo is local within that node to avoid uncovered rows.
            if world_size == 4:
                owned = tuple(index for position, index in enumerate(missing) if position % 2 == rank % 2)
            else:
                owned = tuple(missing)
            if owned:
                rebuild(owned)
        finally:
            initial_teacher.load_state_dict(current)
    if finish_boundary is not None:
        finish_boundary()
    else:
        cache.validate_coverage(resolve_expected(cache, expected))
        cache.phase.value = 1


def publish_initial_teacher(trainer, output, checkpoint):
    """Additional reconstructible model state stays under checkpoints only."""
    destination = Path(output) / INITIAL_TEACHER
    if checkpoint is not None:
        source = Path(checkpoint) / INITIAL_TEACHER
        state = torch.load(source, map_location='cpu', weights_only=True)
    else:
        state = trainer.ema_teacher.state_dict()
    if trainer.args.local_process_index == 0:
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                torch.save(state, stream)
    # All workers use tensor content identity, independent of torch ZIP names.
    digest = hashlib.sha256()
    for name, value in state['parameters'].items():
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_cache_identity(args, trainer, examples, processor, data_path, checkpoint):
    from .v5_reference import _base_identity, _processor_identity
    from .equivalence import read_stage1_backend_settings
    from peft_common import resolve_qwen2_5_vl_model
    from importlib.metadata import version
    v5 = getattr(args, 'stage1_variant', 'standard') == 'v5'
    if v5:
        reference = asdict(trainer.v5_reference_identity)
    else:
        reference = {'base_content_sha256': _base_identity(Path(args.model))[0],
                     'processor_sha256': _processor_identity(processor)}
        if trainer.ema_teacher.decay == 1.:
            reference['initial_teacher_sha256'] = publish_initial_teacher(trainer, args.save_model_path, checkpoint)
    images = hashlib.sha256()
    known = {}
    for example in examples:
        images.update(str(example.source_index).encode())
        for path in (*example.original_image_paths, *example.helper_image_paths):
            if not isinstance(path, (str, Path)):
                raise ValueError('run cache requires content-addressable image files')
            if path not in known:
                known[path] = file_sha256(Path(path))
            images.update(known[path].encode())
    config_names = ('latent_size', 'ema_tau', 'coverage_p', 'image_pool_k', 'helper_group_L',
                    'textual_states', 'attention_implementation', 'vision_attention_implementation',
                    'stage1_runtime_profile')
    identity = {'reference': reference, 'data_sha256': file_sha256(Path(data_path)),
                'images_sha256': images.hexdigest(), 'sources': [x.source_index for x in examples],
                'config': {name: getattr(args, name, None) for name in config_names},
                'query': asdict(trainer.query_config), 'dtype': 'native',
                'checks': trainer.stage1_check_policy}
    owner = resolve_qwen2_5_vl_model(trainer.ema_teacher.model)
    identity['runtime'] = {'backend': read_stage1_backend_settings(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'teacher_dtype': str(next(owner.parameters()).dtype),
        'vision_dtype': str(next(owner.visual.parameters()).dtype),
        'bf16_autocast': trainer.args.bf16, 'tf32': trainer.args.tf32,
        'versions': {name: version(name) for name in ('torch', 'transformers', 'flash-attn', 'accelerate', 'deepspeed')}}
    if v5:
        identity['v5'] = asdict(trainer.v5_config)
    return json.loads(json.dumps(identity, sort_keys=True))


def configure_stage1_run_cache(args, *, trainer, examples, collator, processor, data_path,
                               checkpoint, saved):
    from .cache_lifecycle import Stage1CacheEpochCallback
    from .cache_payloads import capture_teacher_forward_state, restore_teacher_forward_state
    from peft_common import resolve_qwen2_5_vl_model
    from types import SimpleNamespace
    if (trainer.args.world_size != 4 or trainer.args.per_device_train_batch_size != 1
            or trainer.args.dataloader_num_workers != 4):
        raise ValueError('Stage1 run cache requires four ranks, microbatch 1 and workers 4')
    if not torch.distributed.is_initialized():
        raise ValueError('Stage1 run cache requires the coordinator four-rank process group')
    attempt = os.environ.get('ILVR_STAGE1_CACHE_ATTEMPT', '')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', attempt):
        raise ValueError('Stage1 run cache requires coordinator attempt identity')
    output = Path(args.save_model_path)
    repo = Path(__file__).resolve().parents[2]
    if output.parent.resolve() != (repo / 'checkpoints').resolve():
        raise ValueError('cache initial teacher must live under run checkpoints')
    if saved is not None and saved.get('completion') != args.stage1_cache_completion:
        raise ValueError('resume must preserve its scheduled cache completion boundary')
    owner = resolve_qwen2_5_vl_model(trainer.ema_teacher.model)
    capture_teacher_forward_state(trainer.ema_teacher.model)  # Reject unsupported mutable RoPE at launch.
    if owner.training or any(parameter.requires_grad for parameter in owner.visual.parameters()):
        raise ValueError('cache producer must have a frozen eval visual encoder')
    fixed = trainer.ema_teacher.decay == 1.
    v5 = hasattr(trainer, 'v5_config')
    if v5 and not fixed:
        raise ValueError('reference cache requires invariant fixed helper inputs')
    # Bind each rank's effective frozen teacher and node-local files once before
    # training. This is deliberately outside the per-step lookup path.
    identity = build_cache_identity(args, trainer, examples, processor, data_path, checkpoint)
    identities = [None] * 4
    torch.distributed.all_gather_object(identities, identity)
    if any(value != identity for value in identities):
        raise ValueError('cache data/processor/frozen state differs between ranks')
    root = SSD_ROOT / output.name
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.parent.stat().st_dev != Path('/root').stat().st_dev:
        raise ValueError('Stage1 cache root must use the verified root SSD filesystem')
    required = int((60 if fixed else 170) * 1024 ** 3 * len(examples) / 3450) + 1024 ** 3
    if checkpoint is None and shutil.disk_usage(root.parent).free < required:
        raise ValueError('insufficient SSD capacity for the complete node cache')
    plan = Stage1CachePlan(output.name, root, 'targets' if fixed else 'vision', v5, identity)
    cache = Stage1RunCache(plan, multiprocessing.Value('i', 0))
    expose_cache_on_ssd(cache, logical_root=repo / 'data/cache')
    if trainer.args.local_process_index == 0:
        (root / 'closed.json').unlink(missing_ok=True)
    trainer.run_cache = collator.run_cache = trainer.target_builder.run_cache = cache
    sources = tuple(example.source_index for example in examples)
    expected = {'preprocess': sources}
    expected[plan.helper] = sources if fixed else tuple(x.source_index for x in examples if x.helper_image_paths)
    if v5:
        expected['reference'] = None  # Resolved from actual first-epoch query metadata.
    callback = Stage1CacheEpochCallback(cache, expected=expected,
        control_dir=repo / 'logs/multinode' / plan.run_id / attempt / 'cache')
    trainer.add_callback(callback)
    if checkpoint is not None:
        old_state = capture_teacher_forward_state(trainer.ema_teacher.model)
        def rebuild(indices):
            # Preparation precedes Trainer's student/DeepSpeed placement. The
            # independent replicated teacher uses the rank's accelerator device.
            trainer.ema_teacher.model.to(device=trainer.accelerator.device).eval()
            trainer._ema_execution_device = trainer.accelerator.device
            for example in examples:
                if example.source_index not in indices:
                    continue
                inputs = trainer._prepare_inputs(collator([example]))
                v5_metadata = inputs.pop('stage1_v5_metadata', None)
                with torch.no_grad(), trainer.compute_loss_context_manager():
                    needs_reverse = (v5 and v5_metadata is not None and bool(v5_metadata.query_keys)
                        and (trainer.v5_config.lambda_topk_entropy > 0
                             or trainer.v5_config.selection in ('voff', 'gap', 'intersection')))
                    mode = trainer.v5_config.voff_mode if needs_reverse else 'attention_mask'
                    model_inputs, metadata, targets = trainer._prepare_stage1_loss_inputs(inputs, voff_mode=mode)
                    if v5_metadata is not None and v5_metadata.query_keys:
                        trainer._stage1_reference_logits(model_inputs, v5_metadata.query_keys, metadata)
        try:
            prepare_stage1_cache_resume(cache=cache, checkpoint=checkpoint,
                initial_teacher=trainer.ema_teacher, expected=expected, rebuild=rebuild,
                rank=trainer.args.process_index, world_size=4,
                finish_boundary=lambda: callback.on_epoch_end(trainer.args, SimpleNamespace(epoch=1.), None))
        finally:
            restore_teacher_forward_state(trainer.ema_teacher.model, old_state)
