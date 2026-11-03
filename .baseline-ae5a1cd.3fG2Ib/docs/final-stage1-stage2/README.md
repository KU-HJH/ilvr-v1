# Final Stage 1 → Stage 2 Training Runbook

## Scope and confidence

This is the canonical operator entry point for the consolidated Stage 1 → Stage
2 training and evaluation workflow. The commands use only files in the active
checkout and put model state under `./checkpoints`, results under `./results`,
and logs under `./logs`.

The source and focused runtime contracts were verified during integration. The
18 routing combinations below are supported contracts, but they were not all
rerun as full training jobs. Historical training and evaluation numbers are not
revalidated here; their evidence grades and exact boundaries are recorded in
[PROVENANCE.md](PROVENANCE.md).

## Repository/runtime layout

Run every command from the active checkout returned by
`git rev-parse --show-toplevel`. The expected local layout is:

```text
data/                         dataset link or directory
venv/ilvr-venv/               required repository virtual environment
checkpoints/                  model-bearing artifacts only
checkpoints/evaluation-staging/<evaluation-id>/
results/                      metrics, manifests, predictions, and plots only
logs/                         training, evaluation, and execution logs only
configs/                      checked-in DeepSpeed profiles
```

Do not put model weights, adapters, optimizer/scheduler/RNG state, DeepSpeed
shards, prepared models, or links to them under `results/`. Each run should use
one distinct run ID across `checkpoints/`, `results/`, and `logs/`.

Use the repository environment directly:

```bash
cd "$(git rev-parse --show-toplevel)"
./venv/ilvr-venv/bin/python --version
```

## Supported Full/LoRA/DoRA and ZeRO routing

The supported Cartesian product is all 18 Stage × method × ZeRO combinations:

| Stage | Method | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| Stage 1 | Full | `stage1/full/zero1` | `stage1/full/zero2` | `stage1/full/zero3` |
| Stage 1 | LoRA | `stage1/lora/zero1` | `stage1/lora/zero2` | `stage1/lora/zero3` |
| Stage 1 | DoRA | `stage1/dora/zero1` | `stage1/dora/zero2` | `stage1/dora/zero3` |
| Stage 2 | Full | `stage2/full/zero1` | `stage2/full/zero2` | `stage2/full/zero3` |
| Stage 2 | LoRA | `stage2/lora/zero1` | `stage2/lora/zero2` | `stage2/lora/zero3` |
| Stage 2 | DoRA | `stage2/dora/zero1` | `stage2/dora/zero2` | `stage2/dora/zero3` |

ZeRO-1, ZeRO-2, and ZeRO-3 map respectively to
`configs/deepspeed_zero1.json`, `configs/deepspeed_zero2.json`, and
`configs/deepspeed_zero3.json`. The lifecycle routing code selects the existing
`configs/deepspeed_zero3_stage2_peft.json` specialization for Stage 2 LoRA/DoRA
with ZeRO-3; direct `run_stage2_training.sh` calls use the operator-supplied
`DEEPSPEED_CONFIG`. Stage 2 PEFT ZeRO-3 also requires gradient checkpointing to
be disabled.

“Supported” means the routing and checkpoint contract are preserved and
covered by integration tests. It does not mean 18 new full training runs were
performed for this consolidation.

## Stage 1 role and outputs

Stage 1 trains the model to emit selected latent visual states using the
momentum-teacher target path. It supports Full, LoRA, and DoRA tuning and the
`previous` or `all` textual-history mode (`previous` is the default). Its run
directory contains numbered training checkpoints and, when enabled, a final
artifact. Complete checkpoints retain optimizer, scheduler, global-step,
rank-local RNG, DeepSpeed, and Stage 1 sidecar state needed for strict resume.

Stage 1 Full checkpoints require `stage1_ema_teacher.pt`. Stage 1 LoRA and
DoRA checkpoints require both `stage1_ema_teacher.pt` and
`stage1_peft_metadata.json`.

## Stage 1 single-node launch

Set `STAGE1_NUM_PROCESSES` to the number of visible local GPUs. A two-GPU DoRA
ZeRO-1 example is:

```bash
cd "$(git rev-parse --show-toplevel)"
run_id="stage1-dora-z1-$(date +%Y%m%d-%H%M%S)"
STAGE1_NUM_PROCESSES=2 ./run_stage1_training.sh \
  --method dora \
  --model "$PWD/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct" \
  --data-path "$PWD/data/TRAIN.jsonl" \
  --image-root "$PWD/data" \
  --output-dir "$PWD/checkpoints/$run_id" \
  --learning-rate 2.5e-5 \
  --deepspeed-config "$PWD/configs/deepspeed_zero1.json" \
  --log-file "$PWD/logs/zero-stage-compatibility/$run_id/train.log"
```

Do not set any `ILVR_*` topology variable for a single-node launch.

## Stage 1 two-node/four-rank launch

Both operators must use the same final commit and the same logical container
checkout path, `/root/workspace/ILVR/.worktrees/stage1-stage2-final`. Choose one
shared run ID and port, then put the identical values in both blocks. Before
launching, each host must run this preflight and compare output; the HEAD values
must match and both tracked statuses must be empty:

```bash
cd /root/workspace/ILVR/.worktrees/stage1-stage2-final
git rev-parse HEAD
git status --short --untracked-files=no
```

Rank 0 — `korea@10.1.102.41`, container `llm_kd`:

```bash
cd /root/workspace/ILVR/.worktrees/stage1-stage2-final
run_id="stage1-dora-z1-REPLACE_WITH_SHARED_RUN_ID"
ILVR_NUM_MACHINES=2 \
ILVR_NUM_PROCESSES=4 \
ILVR_MACHINE_RANK=0 \
ILVR_MAIN_PROCESS_IP=10.1.102.41 \
ILVR_MAIN_PROCESS_PORT=29520 \
ILVR_SOCKET_IFNAME=bond1.102 \
STAGE1_NUM_PROCESSES=4 ./run_stage1_training.sh \
  --method dora \
  --model "$PWD/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct" \
  --data-path "$PWD/data/TRAIN.jsonl" \
  --image-root "$PWD/data" \
  --output-dir "$PWD/checkpoints/$run_id" \
  --learning-rate 2.5e-5 \
  --deepspeed-config "$PWD/configs/deepspeed_zero1.json" \
  --log-file "$PWD/logs/zero-stage-compatibility/$run_id/train.log"
```

Rank 1 — this host:

```bash
cd /root/workspace/ILVR/.worktrees/stage1-stage2-final
run_id="stage1-dora-z1-REPLACE_WITH_SHARED_RUN_ID"
ILVR_NUM_MACHINES=2 \
ILVR_NUM_PROCESSES=4 \
ILVR_MACHINE_RANK=1 \
ILVR_MAIN_PROCESS_IP=10.1.102.41 \
ILVR_MAIN_PROCESS_PORT=29520 \
ILVR_SOCKET_IFNAME=bond1.102 \
STAGE1_NUM_PROCESSES=4 ./run_stage1_training.sh \
  --method dora \
  --model "$PWD/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct" \
  --data-path "$PWD/data/TRAIN.jsonl" \
  --image-root "$PWD/data" \
  --output-dir "$PWD/checkpoints/$run_id" \
  --learning-rate 2.5e-5 \
  --deepspeed-config "$PWD/configs/deepspeed_zero1.json" \
  --log-file "$PWD/logs/zero-stage-compatibility/$run_id/train.log"
```

This runbook does not deploy or synchronize the remote checkout.

## Stage 1 checkpoint handoff

A complete numbered Stage 1 checkpoint (`checkpoint-N`) is a training-state
artifact used for same-contract resume. An evaluation-prepared copy under
`checkpoints/evaluation-staging/<evaluation-id>/` is a separately validated,
inference-ready artifact and is not assumed to retain resumable optimizer,
scheduler, DeepSpeed, or rank-local RNG state.

Before either use, validate the artifact identity and all Stage 1 sidecars
against its declared tuning method and base-model reference. Select exactly one
validated Stage 1 artifact and pass it to Stage 2 only through
`STAGE1_CHECKPOINT`; do not copy it into `results/` or substitute a partial
checkpoint inventory.

## Stage 2 role and outputs

Stage 2 consumes a validated Stage 1 artifact and trains the interleaved latent
rollout path. The launcher creates an immutable run-local
`results/<run-id>/stage2-length-manifest.json`, passes its SHA-256 into training,
and stores model/training state below `checkpoints/<run-id>/`. Stage 2 Full
checkpoints require `stage2-sampler.json`. Stage 2 LoRA and DoRA checkpoints
require both `stage2-sampler.json` and `stage2_peft_metadata.json`.

## Stage 2 single-node launch

This two-GPU DoRA ZeRO-2 example uses the same checked-in launcher without any
`ILVR_*` variables:

```bash
cd "$(git rev-parse --show-toplevel)"
run_id="stage2-dora-z2-b1-ga2-e2-lr5e5-egb4-$(date +%Y%m%d-%H%M%S)"
STAGE1_CHECKPOINT="$PWD/checkpoints/evaluation-staging/stage1-dora-z1-ema-frozen-b1-ga2-e1-lr2p5e5-egb8-20260902-054900/prepared-checkpoint-431" \
DATA_PATH="$PWD/data/TRAIN.jsonl" \
IMAGE_ROOT="$PWD/data" \
OUTPUT_DIR="$PWD/checkpoints/$run_id" \
LOG_FILE="$PWD/logs/zero-stage-compatibility/$run_id/train.log" \
TUNING_METHOD=dora \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero2.json" \
LEARNING_RATE=5e-5 \
GRADIENT_ACCUMULATION_STEPS=2 \
GRADIENT_CHECKPOINTING=true \
ATTENTION_IMPLEMENTATION=flash_attention_2 \
STAGE2_NUM_PROCESSES=2 \
./run_stage2_training.sh
```

## Stage 2 two-node/four-rank ZeRO-2 launch at lr=5e-5

Choose one fresh run ID and use it verbatim on both hosts. Both hosts must check
out the same final commit at
`/root/workspace/ILVR/.worktrees/stage1-stage2-final`; this task does not deploy
the remote checkout. Run and compare this preflight on both hosts before launch:

```bash
cd /root/workspace/ILVR/.worktrees/stage1-stage2-final
git rev-parse HEAD
git status --short --untracked-files=no
```

Rank 0 — `korea@10.1.102.41`, container `llm_kd`:

```bash
cd /root/workspace/ILVR/.worktrees/stage1-stage2-final
run_id="stage2-dora-z2-b1-ga2-e2-lr5e5-egb8-REPLACE_WITH_SHARED_RUN_ID"
STAGE1_CHECKPOINT="$PWD/checkpoints/evaluation-staging/stage1-dora-z1-ema-frozen-b1-ga2-e1-lr2p5e5-egb8-20260902-054900/prepared-checkpoint-431" \
DATA_PATH="$PWD/data/TRAIN.jsonl" \
IMAGE_ROOT="$PWD/data" \
OUTPUT_DIR="$PWD/checkpoints/$run_id" \
LOG_FILE="$PWD/logs/zero-stage-compatibility/$run_id/train.log" \
TUNING_METHOD=dora \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero2.json" \
LEARNING_RATE=5e-5 \
GRADIENT_ACCUMULATION_STEPS=2 \
GRADIENT_CHECKPOINTING=true \
ATTENTION_IMPLEMENTATION=flash_attention_2 \
ILVR_NUM_MACHINES=2 \
ILVR_NUM_PROCESSES=4 \
ILVR_MACHINE_RANK=0 \
ILVR_MAIN_PROCESS_IP=10.1.102.41 \
ILVR_MAIN_PROCESS_PORT=29520 \
ILVR_SOCKET_IFNAME=bond1.102 \
./run_stage2_training.sh
```

Rank 1 — this host:

```bash
cd /root/workspace/ILVR/.worktrees/stage1-stage2-final
run_id="stage2-dora-z2-b1-ga2-e2-lr5e5-egb8-REPLACE_WITH_SHARED_RUN_ID"
STAGE1_CHECKPOINT="$PWD/checkpoints/evaluation-staging/stage1-dora-z1-ema-frozen-b1-ga2-e1-lr2p5e5-egb8-20260902-054900/prepared-checkpoint-431" \
DATA_PATH="$PWD/data/TRAIN.jsonl" \
IMAGE_ROOT="$PWD/data" \
OUTPUT_DIR="$PWD/checkpoints/$run_id" \
LOG_FILE="$PWD/logs/zero-stage-compatibility/$run_id/train.log" \
TUNING_METHOD=dora \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero2.json" \
LEARNING_RATE=5e-5 \
GRADIENT_ACCUMULATION_STEPS=2 \
GRADIENT_CHECKPOINTING=true \
ATTENTION_IMPLEMENTATION=flash_attention_2 \
ILVR_NUM_MACHINES=2 \
ILVR_NUM_PROCESSES=4 \
ILVR_MACHINE_RANK=1 \
ILVR_MAIN_PROCESS_IP=10.1.102.41 \
ILVR_MAIN_PROCESS_PORT=29520 \
ILVR_SOCKET_IFNAME=bond1.102 \
./run_stage2_training.sh
```

## Resume

Resume never converts a checkpoint across ZeRO stages. The ZeRO stage, world
size, optimizer/scheduler/global step, rank-local RNG state, tuning method, base
identity, and Stage-specific sidecars must match the original run.

Stage 1 single-node resume repeats the original launcher arguments and adds
`STAGE1_RESUME_PATH`:

```bash
cd "$(git rev-parse --show-toplevel)"
run_id="REPLACE_WITH_EXISTING_STAGE1_RUN_ID"
STAGE1_RESUME_PATH="$PWD/checkpoints/$run_id/checkpoint-N" \
STAGE1_NUM_PROCESSES=2 ./run_stage1_training.sh \
  --method dora \
  --model "$PWD/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct" \
  --data-path "$PWD/data/TRAIN.jsonl" \
  --image-root "$PWD/data" \
  --output-dir "$PWD/checkpoints/$run_id" \
  --learning-rate 2.5e-5 \
  --deepspeed-config "$PWD/configs/deepspeed_zero1.json"
```

Stage 2 single-node resume repeats the original launcher environment and adds
`RESUME_CHECKPOINT`:

```bash
cd "$(git rev-parse --show-toplevel)"
run_id="REPLACE_WITH_EXISTING_STAGE2_RUN_ID"
STAGE1_CHECKPOINT="$PWD/checkpoints/evaluation-staging/stage1-dora-z1-ema-frozen-b1-ga2-e1-lr2p5e5-egb8-20260902-054900/prepared-checkpoint-431" \
RESUME_CHECKPOINT="$PWD/checkpoints/$run_id/checkpoint-N" \
DATA_PATH="$PWD/data/TRAIN.jsonl" \
IMAGE_ROOT="$PWD/data" \
OUTPUT_DIR="$PWD/checkpoints/$run_id" \
TUNING_METHOD=dora \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero2.json" \
LEARNING_RATE=5e-5 \
GRADIENT_ACCUMULATION_STEPS=2 \
GRADIENT_CHECKPOINTING=true \
ATTENTION_IMPLEMENTATION=flash_attention_2 \
STAGE2_NUM_PROCESSES=2 \
./run_stage2_training.sh
```

When `--log-file`/`STAGE1_LOG_FILE` or `LOG_FILE` is omitted during resume, each
launcher creates a distinct `resume-<timestamp>-<pid>.log` below the existing
run's log directory. Do not reuse its original `train.log`.

## Evaluation

Populate `model_dir` first using the repository's validated evaluation-
preparation path. It must be an evaluation-ready model below
`./checkpoints/evaluation-staging/<evaluation-id>/`. The locally partial
historical `checkpoint-846` inventory is not evaluation-ready and is not
advertised as one.

```bash
cd "$(git rev-parse --show-toplevel)"
evaluation_id="stage2-evaluation-$(date +%Y%m%d-%H%M%S)"
model_dir="$PWD/checkpoints/evaluation-staging/$evaluation_id/local-model"
base_model_dir="$PWD/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct"
mkdir -p "$PWD/results/$evaluation_id" "$PWD/logs/$evaluation_id"
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python eval.py \
    --model_dir "$model_dir" \
    --base_model_dir "$base_model_dir" \
    --test_data_path "$PWD/data/TEST.jsonl" \
    --task_name comt \
    --output_json_path "$PWD/results/$evaluation_id/predictions.jsonl" \
    >"$PWD/logs/$evaluation_id/eval.log" 2>&1
```

Never copy an evaluation model into `results/`.

## Checkpoint contract

- Model-bearing artifacts live only below `./checkpoints/`; evaluation-ready
  copies live only below `./checkpoints/evaluation-staging/<run-id>/`.
- Results and logs share the run ID but contain no model state.
- A resume source must be a complete numbered checkpoint with the original ZeRO
  stage and world size, optimizer/scheduler/global step, rank-local RNG state,
  and required Stage 1 or Stage 2 sidecars.
- The exact Stage/method sidecar matrix is:

  | Stage | Method | Required Stage-specific sidecars |
  |---|---|---|
  | Stage 1 | Full | `stage1_ema_teacher.pt` |
  | Stage 1 | LoRA | `stage1_ema_teacher.pt`, `stage1_peft_metadata.json` |
  | Stage 1 | DoRA | `stage1_ema_teacher.pt`, `stage1_peft_metadata.json` |
  | Stage 2 | Full | `stage2-sampler.json` |
  | Stage 2 | LoRA | `stage2-sampler.json`, `stage2_peft_metadata.json` |
  | Stage 2 | DoRA | `stage2-sampler.json`, `stage2_peft_metadata.json` |
- An evaluation-prepared copy is for inference/evaluation, not resume, unless it
  independently satisfies the complete numbered-checkpoint contract.
- Stage 1 reaches Stage 2 only through `STAGE1_CHECKPOINT`; Stage 2 resume uses
  only `RESUME_CHECKPOINT`. Neither variable converts state between ZeRO stages.

## Selected design and implementation documents

The auditable selected and excluded lineage index is in
[the provenance manifest](PROVENANCE.md#selected-implemented-documentation).
It includes only documents whose behavior is present in the final runtime.

## Known limits

- Integration tests establish routing and checkpoint-contract preservation, not
  numerical equivalence for 18 fresh full training jobs.
- The remote checkout is not deployed or synchronized by this runbook.
- Historical loss and accuracy figures were not revalidated; see their
  `historical-report` grades in the provenance manifest.
- The captured `checkpoint-846` is only a partial local inventory and cannot be
  resumed or evaluated as a complete model.
- Stage 2 PEFT ZeRO-3 requires its specialized config and disabled gradient
  checkpointing.
- Operators must assign collision-free run IDs and verify identical tracked
  commits on both hosts before a multi-node launch.
