# ZeRO stage training and resume

This guide is the operator contract for the canonical DeepSpeed configurations.
Stage 1 defaults to ZeRO-3 when its configuration is omitted; select ZeRO-1 or
ZeRO-2 explicitly with the matching canonical configuration. Stage 2 keeps
DeepSpeed optional for direct single-process operation, so set
`DEEPSPEED_CONFIG` explicitly to enable or select ZeRO. New runs use
`configs/deepspeed_zero1.json`, `configs/deepspeed_zero2.json`, or
`configs/deepspeed_zero3.json`; do not use the older `configs/config_stage*.json`
files for new compatibility evidence.

## Supported configurations

| Stage | Full | LoRA | DoRA | Official 2-GPU batch |
|---|---|---|---|---|
| Stage 1 | ZeRO-1 / ZeRO-2 / ZeRO-3 | ZeRO-1 / ZeRO-2 / ZeRO-3 | ZeRO-1 / ZeRO-2 / ZeRO-3 | matrix-selected, EGB 16 |
| Stage 2 | ZeRO-1 / ZeRO-2 / ZeRO-3 | ZeRO-1 / ZeRO-2 / ZeRO-3 | ZeRO-1 / ZeRO-2 / ZeRO-3 | micro 1 / accumulation 8 |

The final compatibility lifecycle evidence was collected on exactly two NVIDIA
H100 NVL GPUs at commit `431c519`. Ordinary fresh training may use another
positive world size, but its result is not an official two-rank compatibility
row. The Stage 1 and Stage 2 settings below describe compatibility contracts;
they are not throughput, capacity, or peak-memory measurements.

## Artifact ownership and experiment identity

Every newly generated artifact belongs beneath one of these exact roots:

```text
/root/workspace/ILVR/checkpoints/zero-stage-compatibility
/root/workspace/ILVR/results/zero-stage-compatibility
/root/workspace/ILVR/logs/zero-stage-compatibility
```

Use one lowercase, collision-resistant experiment ID beneath all three roots
for one independent attempt or standalone probe. Checkpoint inputs may come
from other validated locations, but every newly generated checkpoint, result,
and log must be below its canonical root, including when the command is run
from a linked worktree. The pre-policy `outputs/` tree is read-only legacy
provenance: do not move, rename, overwrite, revalidate, or add files there.

A checkpoint-to-resume lifecycle that is one logical experiment retains its
experiment ID and writes a distinct resume log filename in that ID's log
directory. Independently selected save/resume producers are separate attempts
and therefore use separate experiment IDs.

## Runtime and resume identity

Before accelerator or DeepSpeed initialization, the runtime captures the ZeRO
stage, exact canonical configuration digest, and world size. A resume whose
stage, digest, or world size differs from its numbered checkpoint is rejected
before DeepSpeed initialization; cross-stage and cross-world-size optimizer
conversion are unsupported. Optimizer resume requires the same ZeRO stage,
configuration digest, and world size; optimizer state cannot be shared across
different ZeRO identities. Keep the complete numbered checkpoint, its runtime
sidecar, and the same canonical configuration when resuming.

## Fresh and same-identity resume examples

The following Stage 1 and Stage 2 examples deliberately use ZeRO-2. Change
only `deepspeed_zero2.json` to `deepspeed_zero1.json` or
`deepspeed_zero3.json` to select that canonical stage. The generated suffix
prevents a fresh attempt from colliding with another independent experiment.

### Stage 1 LoRA

```bash
zero_checkpoint_root=/root/workspace/ILVR/checkpoints/zero-stage-compatibility
zero_result_root=/root/workspace/ILVR/results/zero-stage-compatibility
zero_log_root=/root/workspace/ILVR/logs/zero-stage-compatibility
run_suffix="$(date -u +%Y%m%dt%H%M%S)-$$-${RANDOM}"
experiment_id="stage1-zero2-lora-fresh-example-$run_suffix"

./run_stage1_training.sh \
  --method lora \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data-path "$PWD/data/TRAIN.jsonl" \
  --image-root "$PWD/data" \
  --output-dir "$zero_checkpoint_root/$experiment_id" \
  --log-file "$zero_log_root/$experiment_id/train.log" \
  --micro-batch 1 \
  --gradient-accumulation 8 \
  --epochs 3 \
  --learning-rate 1e-5 \
  --seed 42 \
  --deepspeed-config configs/deepspeed_zero2.json \
  --save-strategy steps \
  --save-steps 100

./run_stage1_training.sh \
  --method lora \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data-path "$PWD/data/TRAIN.jsonl" \
  --image-root "$PWD/data" \
  --output-dir "$zero_checkpoint_root/$experiment_id" \
  --log-file "$zero_log_root/$experiment_id/resume.log" \
  --micro-batch 1 \
  --gradient-accumulation 8 \
  --epochs 3 \
  --learning-rate 1e-5 \
  --seed 42 \
  --deepspeed-config configs/deepspeed_zero2.json \
  --resume "$zero_checkpoint_root/$experiment_id/checkpoint-100"
```

Stage 1 supports Full, LoRA, and DoRA for all three canonical ZeRO stages.

### Stage 2 DoRA

```bash
zero_checkpoint_root=/root/workspace/ILVR/checkpoints/zero-stage-compatibility
zero_result_root=/root/workspace/ILVR/results/zero-stage-compatibility
zero_log_root=/root/workspace/ILVR/logs/zero-stage-compatibility
run_suffix="$(date -u +%Y%m%dt%H%M%S)-$$-${RANDOM}"
experiment_id="stage2-zero2-dora-fresh-example-$run_suffix"

STAGE1_CHECKPOINT="$zero_checkpoint_root/stage1-dora-final" \
DATA_PATH="$PWD/data/TRAIN.jsonl" \
IMAGE_ROOT="$PWD/data" \
OUTPUT_DIR="$zero_checkpoint_root/$experiment_id" \
LOG_FILE="$zero_log_root/$experiment_id/train.log" \
TUNING_METHOD=dora \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero2.json" \
GRADIENT_ACCUMULATION_STEPS=8 \
ATTENTION_IMPLEMENTATION=flash_attention_2 \
./run_stage2_training.sh

STAGE1_CHECKPOINT="$zero_checkpoint_root/stage1-dora-final" \
DATA_PATH="$PWD/data/TRAIN.jsonl" \
IMAGE_ROOT="$PWD/data" \
OUTPUT_DIR="$zero_checkpoint_root/$experiment_id" \
LOG_FILE="$zero_log_root/$experiment_id/resume.log" \
RESUME_CHECKPOINT="$zero_checkpoint_root/$experiment_id/checkpoint-100" \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero2.json" \
GRADIENT_ACCUMULATION_STEPS=8 \
ATTENTION_IMPLEMENTATION=flash_attention_2 \
./run_stage2_training.sh
```

Stage 2 supports Full, LoRA, and DoRA for all three canonical ZeRO stages.
When resuming an adapter run with an explicit method, retain the compatible
Stage 1 base and the matching adapter settings; an omitted method inherits the
checkpoint metadata.

## Official matrix procedure and evidence

The Stage 1 controller tests the fixed EGB-16 candidates in this order:
`(2,4)`, `(4,2)`, `(8,1)`, then `(1,8)`, in ZeRO stage order 3, 2, 1. This
fixed matrix is separate from capacity probing: a capacity probe begins at
per-GPU micro-batch 2, and it may try micro-batch 1 only after a confirmed CUDA
capacity OOM at 2.
Each successful `(stage, method)` has one selected fastest row and only that
selected row receives the same-identity save-to-resume lifecycle. CUDA allocator
out-of-memory failures are recorded as `capacity_oom`. NCCL, timeout,
non-finite, contract, missing-evidence, and other nonzero failures are
`integration_failure`, not capacity results.

The Stage 1 performance comparison matrix remains scoped to Full and DoRA.
LoRA is included in the canonical ZeRO compatibility matrix, not the
Full/DoRA performance comparison.

The final aggregate is the compatibility lifecycle evidence linked below; it
does not report throughput, capacity, or peak-memory performance.

Each Stage 2 rank-skew witness is written below the same canonical result root
in its experiment-ID directory. It records the one-step/eight-micro-step
two-rank compatibility lifecycle only; it is not a peak-memory measurement.

## Final compatibility lifecycle evidence

All 18 Stage 1/Stage 2 × Full/LoRA/DoRA × ZeRO-1/2/3 two-rank save/resume
checks passed on two NVIDIA H100 NVL GPUs at commit `431c519`. The exact final
aggregate and manifest are
[aggregate_validation.json](/root/workspace/ILVR/results/zero-stage-compatibility/sdd-task7-r4-20260828-aggregate-20260828t123316-1956316/aggregate_validation.json)
and
[manifest.md](/root/workspace/ILVR/results/zero-stage-compatibility/sdd-task7-r4-20260828-aggregate-20260828t123316-1956316/manifest.md).

Throughput, capacity, and peak-memory performance measurements were not run.
The tables below remain schemas, not measurements; compatibility save/resume
success must not be read as a performance result. The Stage 1 performance
comparison scope remains Full and DoRA only; LoRA appears only in the
compatibility matrix.

| ZeRO | Method | Micro/GA | Peak reserved GiB | samples/s | Resume |
|---|---|---|---:|---:|---|
| 3 | Full | schema only | not measured | not measured | compatibility passed; no performance measurement |
| 3 | DoRA | schema only | not measured | not measured | compatibility passed; no performance measurement |
| 2 | Full | schema only | not measured | not measured | compatibility passed; no performance measurement |
| 2 | DoRA | schema only | not measured | not measured | compatibility passed; no performance measurement |
| 1 | Full | schema only | not measured | not measured | compatibility passed; no performance measurement |
| 1 | DoRA | schema only | not measured | not measured | compatibility passed; no performance measurement |

| ZeRO | Method | Micro/GA | Peak reserved GiB | samples/s | Resume |
|---|---|---|---:|---:|---|
| 3 | Full/LoRA/DoRA | schema only | not measured | not measured | compatibility passed; no performance measurement |
| 2 | Full/LoRA/DoRA | schema only | not measured | not measured | compatibility passed; no performance measurement |
| 1 | Full/LoRA/DoRA | schema only | not measured | not measured | compatibility passed; no performance measurement |
