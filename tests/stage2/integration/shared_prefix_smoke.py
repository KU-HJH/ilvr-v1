"""Two-rank shared Flex smoke through the actual Stage2Trainer training loop."""
import copy
from dataclasses import replace
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from artifact_paths import require_canonical_artifact_path, resolve_run_artifact_paths
from deepspeed_runtime import load_deepspeed_runtime_profile
from main import build_stage2_training_arguments
from stage2.peft import ALLOWED_STAGE2_TARGET_SUFFIXES, Stage2TuningConfig, build_stage2_tuned_model
from stage2.trainer import Stage2Trainer
from step_preference.equivalence import _config, write_evidence_exclusive
from step_preference.shared_prefix_measurement import RuntimeCounters, tiny_measurement_case


class _Samples(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return SimpleNamespace(latent_block_count=2, sample_id=f"tiny-shared-{index}")


def run_shared_prefix_smoke(options):
    from tiny_stage2_train import TinyStage2Processor
    if options.tuning_method != "dora" or options.zero_stage != 2 or int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise ValueError("shared smoke requires exactly two ranks, DoRA, and ZeRO2")
    profile = load_deepspeed_runtime_profile(options.deepspeed, world_size=2)
    if profile.zero_stage != 2:
        raise ValueError("shared smoke requires a ZeRO2 DeepSpeed configuration")
    paths = resolve_run_artifact_paths(options.experiment_id)
    checkpoint = require_canonical_artifact_path(paths.checkpoint_dir, kind="checkpoint")
    result = require_canonical_artifact_path(paths.result_dir / "ddp-witness.json", kind="result")
    log = require_canonical_artifact_path(Path.cwd() / "logs" / options.experiment_id, kind="log")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if rank == 0:
        checkpoint.mkdir(parents=True, exist_ok=False)
        result.parent.mkdir(parents=True, exist_ok=False)
        log.mkdir(parents=True, exist_ok=True)
    config = replace(_config(packing=False), attention_implementation="sdpa",
        shared_prefix_enabled=True, sidecar_attention_backend="flex", sidecar_query_budget=1024)
    values = SimpleNamespace(save_model_path=str(checkpoint), epochs=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=2, warm_up_steps=0,
        learning_rate=1e-3, save_steps=100, max_seq_length_train=128, max_steps=1,
        gradient_checkpointing=True, tuning_method="dora", deepspeed_config=options.deepspeed,
        step_preference_config=config)
    args = build_stage2_training_arguments(values)
    args.save_strategy = "no"
    args.logging_steps = 1
    args.disable_tqdm = True
    base, batch = tiny_measurement_case(steps=2, total_target_tokens=8, device="cpu")
    tuning = Stage2TuningConfig(method="dora", r=32, alpha=64, dropout=0.05,
                               target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES)
    model = build_stage2_tuned_model(base, tuning, is_trainable=True)
    pre_engine_parameter_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    before = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    gradients = []
    handles = [p.register_hook(lambda grad: gradients.append(bool(torch.isfinite(grad).all())) or grad)
               for p in model.parameters() if p.requires_grad]
    trainer = Stage2Trainer(model=model, args=args, train_dataset=_Samples(),
        data_collator=lambda samples: copy.deepcopy(batch), processing_class=TinyStage2Processor(),
        deepspeed_runtime_profile=profile, step_preference_config=config)
    with RuntimeCounters(model) as counters:
        original_backward = trainer.accelerator.backward
        def observed_backward(*args, **kwargs):
            counters.phase = "backward"
            try:
                return original_backward(*args, **kwargs)
            finally:
                counters.phase = "forward"
        trainer.accelerator.backward = observed_backward
        try:
            trained = trainer.train()
        finally:
            trainer.accelerator.backward = original_backward
        counts = counters.snapshot()
    for handle in handles:
        handle.remove()
    after = {n: p.detach().float().cpu() for n, p in model.named_parameters() if p.requires_grad}
    gradient_norm = trainer._representative_gradient_norm
    record = dict(rank=rank, global_step=int(trainer.state.global_step),
        pre_engine_parameter_dtypes=pre_engine_parameter_dtypes,
        trained_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        optimizer_step=trainer._observed_optimizer_step,
        # Metric accumulation is reset by Trainer.log; this monotonic index
        # counts actual compute_loss entries across logging boundaries.
        microbatches=trainer._stage2_microbatch_index,
        loss=float(trained.training_loss), finite_loss=math.isfinite(trained.training_loss),
        gradient_norm=gradient_norm,
        finite_gradients=bool(gradients) and all(gradients) and gradient_norm is not None and math.isfinite(gradient_norm),
        gradient_hook_calls=len(gradients), changed_trainable_tensors=sum(not torch.equal(before[n], p) for n,p in after.items()),
        rollout_calls=counts.get("rollout_calls_forward", 0), runtime_counters=counts)
    records = [None, None]
    dist.all_gather_object(records, record)
    for witness in records:
        assert witness["global_step"] == witness["optimizer_step"] == 1, witness
        assert witness["microbatches"] == witness["rollout_calls"] == 2, witness
        assert witness["finite_loss"] and witness["finite_gradients"] and witness["changed_trainable_tensors"] > 0, witness
    if rank == 0:
        write_evidence_exclusive(result, dict(status="pass", world_size=2, ranks=records,
            adapter=dict(method="dora", rank=32, alpha=64, dropout=0.05),
            zero_stage=int(trainer.model_wrapped.zero_optimization_stage()),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            backend="flex", dtype="bfloat16", gradient_checkpointing=True,
            checkpoint_dir=str(checkpoint), model_saved=False))
        print(f"SHARED_PREFIX_DDP_PASS {result}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
