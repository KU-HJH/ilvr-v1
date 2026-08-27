# ZeRO stage training and resume

This guide is the operator contract for the canonical DeepSpeed configurations.
ZeRO-3 is the launcher default; select ZeRO-1 or ZeRO-2 explicitly with the
matching canonical configuration.  New runs use
`configs/deepspeed_zero1.json`, `configs/deepspeed_zero2.json`, or
`configs/deepspeed_zero3.json`; do not use the older `configs/config_stage*.json`
files for new compatibility evidence.

## Supported configurations

| Stage | Full | LoRA | DoRA | Official 2-GPU batch |
|---|---|---|---|---|
| Stage 1 | ZeRO-1/2/3 | existing legacy path only | ZeRO-1/2/3 | matrix-selected, EGB 16 |
| Stage 2 | ZeRO-1/2/3 | ZeRO-1/2/3 | ZeRO-1/2/3 | micro 1 / accumulation 8 |

The official evidence environment is exactly two NVIDIA B200 GPUs. Ordinary
fresh training may use another positive world size, but its result is not an
official two-GPU matrix row. Stage 1's official effective global batch is 16;
until the matrix has measured a preferred row, its universally valid fallback
is `(micro_batch, gradient_accumulation) = (1, 8)`. Stage 2 fixes the local
micro batch to 1 and uses gradient accumulation 8 for the official setting;
there is no Stage 2 batch sweep.

## Artifact ownership and experiment identity

Every newly generated artifact belongs beneath one of these exact roots:

```text
/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/checkpoints/zero-stage-compatibility
/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/results/zero-stage-compatibility
/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/logs/zero-stage-compatibility
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
conversion are unsupported. Keep the complete numbered checkpoint, its runtime
sidecar, and the same canonical configuration when resuming.

## Fresh and same-identity resume examples

The following Stage 1 and Stage 2 examples deliberately use ZeRO-2. Change
only `deepspeed_zero2.json` to `deepspeed_zero1.json` or
`deepspeed_zero3.json` to select that canonical stage. The generated suffix
prevents a fresh attempt from colliding with another independent experiment.

### Stage 1 DoRA

```bash
zero_checkpoint_root=/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/checkpoints/zero-stage-compatibility
zero_log_root=/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/logs/zero-stage-compatibility
run_suffix="$(date -u +%Y%m%dt%H%M%S)-$$-${RANDOM}"
experiment_id="stage1-zero2-dora-fresh-example-$run_suffix"

./run_stage1_training.sh \
  --method dora \
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
  --deepspeed-config "$PWD/configs/deepspeed_zero2.json" \
  --save-strategy steps \
  --save-steps 100

./run_stage1_training.sh \
  --method dora \
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
  --deepspeed-config "$PWD/configs/deepspeed_zero2.json" \
  --resume "$zero_checkpoint_root/$experiment_id/checkpoint-100"
```

Stage 1 Full and DoRA are the canonical ZeRO methods. Stage 1 LoRA remains
only on the existing legacy path and is not a new ZeRO compatibility method.

### Stage 2 DoRA

```bash
zero_checkpoint_root=/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/checkpoints/zero-stage-compatibility
zero_log_root=/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/logs/zero-stage-compatibility
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

The Stage 1 controller tests the fixed EGB-16 candidates in reverse-size order:
`(8,1)`, `(4,2)`, `(2,4)`, then `(1,8)`, in ZeRO stage order 3, 2, 1.
Each successful `(stage, method)` has one selected fastest row and only that
selected row receives the same-identity save-to-resume lifecycle. CUDA allocator
out-of-memory failures are recorded as `capacity_oom`. NCCL, timeout,
non-finite, contract, missing-evidence, and other nonzero failures are
`integration_failure`, not capacity results.

The aggregate Stage 1 matrix report is:

```text
/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/results/zero-stage-compatibility/stage1-zero-matrix/zero-matrix-report.json
```

Each Stage 2 rank-skew witness is written below the same canonical result root
in its experiment-ID directory. It records the one-step/eight-micro-step
two-rank compatibility and peak-memory witness; no second Stage 2 memory run is
required.

## Result status: pending GPU evidence

The following is a schema, not a measurement. Tasks 9 and 10 have not yet
recorded hardware results here; this guide does not claim compatibility,
throughput, or memory success for them.

| ZeRO | Method | Micro/GA | Peak reserved GiB | samples/s | Resume |
|---|---|---|---:|---:|---|
| 3 | Full | recorded by Task 10 | not recorded in this document yet | not recorded in this document yet | pending Task 10 |
| 3 | DoRA | recorded by Task 10 | not recorded in this document yet | not recorded in this document yet | pending Task 10 |
| 2 | Full | recorded by Task 10 | not recorded in this document yet | not recorded in this document yet | pending Task 10 |
| 2 | DoRA | recorded by Task 10 | not recorded in this document yet | not recorded in this document yet | pending Task 10 |
| 1 | Full | recorded by Task 10 | not recorded in this document yet | not recorded in this document yet | pending Task 10 |
| 1 | DoRA | recorded by Task 10 | not recorded in this document yet | not recorded in this document yet | pending Task 10 |

| ZeRO | Method | Micro/GA | Peak reserved GiB | samples/s | Resume |
|---|---|---|---:|---:|---|
| 3 | Full/LoRA/DoRA | micro 1 / accumulation 8 | not recorded in this document yet | not recorded in this document yet | pending Task 9 |
| 2 | Full/LoRA/DoRA | micro 1 / accumulation 8 | not recorded in this document yet | not recorded in this document yet | pending Task 9 |
| 1 | Full/LoRA/DoRA | micro 1 / accumulation 8 | not recorded in this document yet | not recorded in this document yet | pending Task 9 |
