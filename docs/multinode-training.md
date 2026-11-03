# Four-GPU multi-node operation

This is the operator contract for ILVR's fixed two-node, four-process
training and evaluation launchers. Run every coordinator command from
`/root/workspace/ILVR` with the repository virtual environment. Do not adapt
these commands to a discovered host, interface, container, rank order, or
Python installation.

## Fixed topology and one-time provisioning

| Item | Required value |
| --- | --- |
| Remote SSH target | `korea@172.16.11.41` |
| Remote container | `llm_kd` |
| Repository inside both containers | `/root/workspace/ILVR` |
| Remote host repository | `/home/korea/Workspace/hjh/workspace/ILVR` |
| Python on both nodes | `/root/workspace/ILVR/venv/ilvr-venv/bin/python` |
| Rendezvous/master | `172.16.11.41`, remote machine rank 0 |
| Local machine rank | 1 |
| Processes | 2 GPUs per node, 4 processes total |
| Distributed interface | `bond0.11` for Gloo and NCCL |

Provisioning is a one-time administrator action, not a launcher-time sync.
Before the first run, the `llm_kd` container must be running; the same clean,
committed checkout and repository venv must exist at the canonical repository
path on both nodes; `tmux`, OpenSSH, Docker, and `rsync` must be installed; and
the required model, data, and image inputs must exist at identical absolute
paths. The canonical `checkpoints` symlink and the `results` and `logs`
directories must already be writable on both nodes with at least 100 GiB free.

The commands below use existing inputs:

```text
/root/workspace/ILVR/checkpoints/comt_ckpt
/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct
/root/workspace/ILVR/data/TRAIN.jsonl
/root/workspace/ILVR/data/TEST.jsonl
/root/workspace/ILVR/data/images_comt
```

Every other checkpoint, result, log, prepared inference directory, subset, and
validation bundle named below is a one-time-created runtime output. A launcher
creates its own run directories exclusively. Never pre-create a run's
checkpoint directory, reuse a run/attempt identity, or overwrite an artifact.

The coordinator selects `configs/deepspeed_zero1.json`,
`configs/deepspeed_zero2.json`, or `configs/deepspeed_zero3.json` from
`--zero-stage`. The configuration, four-process world size, and tuning method
are part of resume identity and cannot change on resume.

Every launch requires an explicit free TCP port. The coordinator starts remote
rank 0 first, waits for its exact owned rendezvous listener, and only then
starts local rank 1. Do not start the ranks or tmux workers manually.

## Exact preflight

The training launchers repeat preflight with their complete input and output
set. This standalone command is useful before assigning a run ID and port to
a longer operation. Remove `--dry-run` for the real SSH/container/GPU
preflight; `--dry-run` only prints the immutable commands.

```bash
cd /root/workspace/ILVR
preflight_id="preflight-stage1-lora-zero3-$(date -u +%Y%m%d-%H%M%S)-$$"
./venv/ilvr-venv/bin/python -m src.multinode preflight \
  --stage stage1 \
  --tuning-method lora \
  --zero-stage 3 \
  --run-id "$preflight_id" \
  --main-process-port 29800 \
  --input-path /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --input-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --input-path /root/workspace/ILVR/data/images_comt \
  --artifact-root /root/workspace/ILVR/checkpoints \
  --artifact-root /root/workspace/ILVR/results \
  --artifact-root /root/workspace/ILVR/logs \
  --dry-run
```

A live preflight must report matching commit, Python, PyTorch, Accelerate,
DeepSpeed, two GPUs per node, `bond0.11`, fresh paths/sessions, readable
inputs, writable roots, and the required free space. The subsequent launcher
is still the authoritative final preflight.

## Stage 1: Full, LoRA, and DoRA on ZeRO 1, 2, and 3

This exact sequential block covers all nine cells. Each call creates
`/root/workspace/ILVR/checkpoints/$run_id`, its result witness below
`/root/workspace/ILVR/results/$run_id`, and owned logs. `--method` is the
Stage 1 training argument; `--tuning-method` belongs to preflight and the
capacity controller.

```bash
cd /root/workspace/ILVR
run_stamp="$(date -u +%Y%m%d-%H%M%S)-$$"

run_stage1_cell() {
  local run_id=$1 method=$2 zero_stage=$3 port=$4
  ./run_multinode_stage1_training.sh \
    --run-id "$run_id" \
    --zero-stage "$zero_stage" \
    --main-process-port "$port" \
    --method "$method" \
    --model /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
    --data-path /root/workspace/ILVR/data/TRAIN.jsonl \
    --image-root /root/workspace/ILVR/data/images_comt \
    --micro-batch 2 \
    --save-strategy steps \
    --save-steps 10 \
    --max-steps 10
}

run_stage1_cell "stage1-full-zero1-$run_stamp" full 1 29801
run_stage1_cell "stage1-full-zero2-$run_stamp" full 2 29802
run_stage1_cell "stage1-full-zero3-$run_stamp" full 3 29803
run_stage1_cell "stage1-lora-zero1-$run_stamp" lora 1 29804
run_stage1_cell "stage1-lora-zero2-$run_stamp" lora 2 29805
run_stage1_cell "stage1-lora-zero3-$run_stamp" lora 3 29806
run_stage1_cell "stage1-dora-zero1-$run_stamp" dora 1 29807
run_stage1_cell "stage1-dora-zero2-$run_stamp" dora 2 29808
run_stage1_cell "stage1-dora-zero3-$run_stamp" dora 3 29809
```

Runs are sequential, so a port is released before the next call. If these are
scheduled concurrently, allocate a disjoint port to every live attempt.

### Stage 1 same-identity resume

Reuse the original `run_id`, method, ZeRO stage, model/data settings, and
complete numbered checkpoint. Use a fresh attempt ID and port. This example
continues an earlier LoRA ZeRO-3 step-10 run to step 11:

```bash
run_id="stage1-lora-zero3-$run_stamp"
./run_multinode_stage1_training.sh \
  --run-id "$run_id" \
  --attempt-id resume-step10-to11 \
  --zero-stage 3 \
  --main-process-port 29810 \
  --method lora \
  --model /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
  --data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /root/workspace/ILVR/data/images_comt \
  --micro-batch 2 \
  --save-strategy steps \
  --save-steps 11 \
  --max-steps 11 \
  --resume "/root/workspace/ILVR/checkpoints/$run_id/checkpoint-10"
```

## Stage 2: Full, LoRA, and DoRA on ZeRO 1, 2, and 3

Stage 2 fixes rank-local batch size at 1. The effective global batch is
`4 × 1 × GRADIENT_ACCUMULATION_STEPS`. `--save_steps` must equal the desired
terminal `--max_steps`, because the coordinator validates and collects that
exact numbered checkpoint.

```bash
cd /root/workspace/ILVR
run_stamp="$(date -u +%Y%m%d-%H%M%S)-$$"

run_stage2_cell() {
  local run_id=$1 method=$2 zero_stage=$3 port=$4
  STAGE1_CHECKPOINT=/root/workspace/ILVR/checkpoints/comt_ckpt \
  DATA_PATH=/root/workspace/ILVR/data/TRAIN.jsonl \
  IMAGE_ROOT=/root/workspace/ILVR/data/images_comt \
  TUNING_METHOD="$method" \
  ./run_multinode_stage2_training.sh \
    --run-id "$run_id" \
    --zero-stage "$zero_stage" \
    --main-process-port "$port" \
    --max_steps 10 \
    --save_steps 10
}

run_stage2_cell "stage2-full-zero1-$run_stamp" full 1 29821
run_stage2_cell "stage2-full-zero2-$run_stamp" full 2 29822
run_stage2_cell "stage2-full-zero3-$run_stamp" full 3 29823
run_stage2_cell "stage2-lora-zero1-$run_stamp" lora 1 29824
run_stage2_cell "stage2-lora-zero2-$run_stamp" lora 2 29825
run_stage2_cell "stage2-lora-zero3-$run_stamp" lora 3 29826
run_stage2_cell "stage2-dora-zero1-$run_stamp" dora 1 29827
run_stage2_cell "stage2-dora-zero2-$run_stamp" dora 2 29828
run_stage2_cell "stage2-dora-zero3-$run_stamp" dora 3 29829
```

### Stage 2 same-identity resume

```bash
run_id="stage2-dora-zero2-$run_stamp"
STAGE1_CHECKPOINT=/root/workspace/ILVR/checkpoints/comt_ckpt \
DATA_PATH=/root/workspace/ILVR/data/TRAIN.jsonl \
IMAGE_ROOT=/root/workspace/ILVR/data/images_comt \
RESUME_CHECKPOINT="/root/workspace/ILVR/checkpoints/$run_id/checkpoint-10" \
TUNING_METHOD=dora \
./run_multinode_stage2_training.sh \
  --run-id "$run_id" \
  --attempt-id resume-step10-to11 \
  --zero-stage 2 \
  --main-process-port 29830 \
  --max_steps 11 \
  --save_steps 11
```

## Four-worker evaluation

Evaluation accepts only a Task 6 prepared inference artifact below the
canonical result root, not a raw checkpoint directory. `prepare-inference`
filters a completed canonical checkpoint to the inference allowlist, creates
the same fresh result path locally and remotely, transfers that exact list plus
its identity manifest without overwrite, and revalidates the remote artifact.
It starts no evaluator worker.

```bash
source_run_id="stage2-full-zero1-$run_stamp"
prepared_model_dir="/root/workspace/ILVR/results/$source_run_id/prepared-checkpoint-10"
./venv/ilvr-venv/bin/python -m src.multinode prepare-inference \
  --stage stage2 \
  --tuning-method full \
  --zero-stage 1 \
  --run-id "$source_run_id" \
  --main-process-port 29821 \
  --checkpoint-step 10 \
  --destination "$prepared_model_dir"

./venv/ilvr-venv/bin/python -m src.multinode prepared-inference-show \
  --model-dir "$prepared_model_dir"
ssh -o BatchMode=yes -o ConnectTimeout=8 -- korea@172.16.11.41 \
  "/usr/bin/docker exec --workdir /root/workspace/ILVR llm_kd /usr/bin/env PYTHONPATH=/root/workspace/ILVR /root/workspace/ILVR/venv/ilvr-venv/bin/python -m src.multinode prepared-inference-show --model-dir '$prepared_model_dir'"
```

Both checks must identify the same artifact. Then run:

```bash
eval_id="eval-four-worker-$(date -u +%Y%m%d-%H%M%S)-$$"
./run_multinode_evaluation.sh \
  --run-id "$eval_id" \
  --model-dir "$prepared_model_dir" \
  --test-data-path /root/workspace/ILVR/data/TEST.jsonl \
  --task-name zebra-cot \
  --output-dir "/root/workspace/ILVR/results/$eval_id" \
  --max-new-tokens 64
```

The wrapper owns four independent workers: remote GPUs produce shards 0 and
1, local GPUs produce shards 2 and 3, and the coordinator pulls only the two
remote shard JSONLs and sidecars without overwriting local data. Success
requires all sidecars, complete source-index coverage, no worker errors, a
merged JSONL, and the summary JSON. The fresh local inference copy is retained
outside that strict result tree at
`/root/workspace/ILVR/results/${eval_id}-inference-model`, with its sibling
identity manifest; do not remove it as part of evaluation cleanup.

## Stage 1 capacity probes and Stage 2 batch-1 confirmations

A Stage 1 probe always tries batches `2 → 4 → 8` in that order. It stops on
the first CUDA capacity OOM and never retries an integration failure. Batch 1
is attempted exactly once only when the real batch-2 attempt is classified as
CUDA capacity OOM. Do not begin a probe at batch 1 and do not reinterpret
NCCL, rendezvous, checkpoint, nonfinite, SSH, Docker, or other integration
failures as capacity.

The nine calls below are sequential. Each call visibly passes exactly four
repeated `--main-process-port` values: batches 2, 4, 8, then the conditional
batch-1 fallback.

```bash
capacity_suite="capacity-four-gpu-$(date -u +%Y%m%d-%H%M%S)-$$"
capacity_root="/root/workspace/ILVR/results/$capacity_suite"
mkdir -m 700 "$capacity_root" "$capacity_root/capacity" "$capacity_root/stage2-batch1"

run_stage1_capacity_cell() {
  local run_id=$1 method=$2 zero_stage=$3
  local batch2_port=$4 batch4_port=$5 batch8_port=$6 batch1_port=$7
  ./run_multinode_stage1_batch_probe.sh \
    --run-id "$run_id" \
    --result-root "$capacity_root/capacity" \
    --zero-stage "$zero_stage" \
    --tuning-method "$method" \
    --main-process-port "$batch2_port" \
    --main-process-port "$batch4_port" \
    --main-process-port "$batch8_port" \
    --main-process-port "$batch1_port" \
    --model /root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct \
    --data-path /root/workspace/ILVR/data/TRAIN.jsonl \
    --image-root /root/workspace/ILVR/data/images_comt \
    --save-strategy steps \
    --save-steps 1 \
    --max-steps 1
}

run_stage1_capacity_cell "$capacity_suite-stage1-full-zero1" full 1 30000 30001 30002 30003
run_stage1_capacity_cell "$capacity_suite-stage1-full-zero2" full 2 30004 30005 30006 30007
run_stage1_capacity_cell "$capacity_suite-stage1-full-zero3" full 3 30008 30009 30010 30011
run_stage1_capacity_cell "$capacity_suite-stage1-lora-zero1" lora 1 30012 30013 30014 30015
run_stage1_capacity_cell "$capacity_suite-stage1-lora-zero2" lora 2 30016 30017 30018 30019
run_stage1_capacity_cell "$capacity_suite-stage1-lora-zero3" lora 3 30020 30021 30022 30023
run_stage1_capacity_cell "$capacity_suite-stage1-dora-zero1" dora 1 30024 30025 30026 30027
run_stage1_capacity_cell "$capacity_suite-stage1-dora-zero2" dora 2 30028 30029 30030 30031
run_stage1_capacity_cell "$capacity_suite-stage1-dora-zero3" dora 3 30032 30033 30034 30035
```

Stage 2 has no capacity search: its contract fixes batch/GPU at 1. These nine
one-step calls populate the sibling validation tree:

```bash
run_stage2_batch1_cell() {
  local run_id=$1 method=$2 zero_stage=$3 port=$4
  STAGE1_CHECKPOINT=/root/workspace/ILVR/checkpoints/comt_ckpt \
  DATA_PATH=/root/workspace/ILVR/data/TRAIN.jsonl \
  IMAGE_ROOT=/root/workspace/ILVR/data/images_comt \
  TUNING_METHOD="$method" \
  ./run_multinode_stage2_training.sh \
    --run-id "$run_id" \
    --result-root "$capacity_root/stage2-batch1" \
    --zero-stage "$zero_stage" \
    --main-process-port "$port" \
    --max_steps 1 \
    --save_steps 1
}

run_stage2_batch1_cell "$capacity_suite-stage2-full-zero1" full 1 30100
run_stage2_batch1_cell "$capacity_suite-stage2-full-zero2" full 2 30101
run_stage2_batch1_cell "$capacity_suite-stage2-full-zero3" full 3 30102
run_stage2_batch1_cell "$capacity_suite-stage2-lora-zero1" lora 1 30103
run_stage2_batch1_cell "$capacity_suite-stage2-lora-zero2" lora 2 30104
run_stage2_batch1_cell "$capacity_suite-stage2-lora-zero3" lora 3 30105
run_stage2_batch1_cell "$capacity_suite-stage2-dora-zero1" dora 1 30106
run_stage2_batch1_cell "$capacity_suite-stage2-dora-zero2" dora 2 30107
run_stage2_batch1_cell "$capacity_suite-stage2-dora-zero3" dora 3 30108
```

## Eighteen-cell save/resume lifecycle matrix

The matrix order is Stage 1 then Stage 2, Full/LoRA/DoRA, then ZeRO 1/2/3.
It runs all 18 cells sequentially. Every cell proves initial step 0→2,
non-shared checkpoint redistribution, and resume step 2→3 on all four ranks.

```bash
matrix_id="tiny-matrix-$(date -u +%Y%m%d-%H%M%S)-$$"
./run_multinode_lifecycle_matrix.sh \
  --run-id "$matrix_id" \
  --first-port 30200
```

This run owns all 18 consecutive ports `30200` through `30217`, inclusive,
until it exits. None of those ports—not merely the first two—may be assigned
to a concurrent launch. The completion marker is
`/root/workspace/ILVR/results/$matrix_id/matrix-complete.json`; its absence
means the matrix is incomplete.

## Validate the result trees once

The validator requires one complete matrix root, evaluation root, and capacity
root. Its output is a new atomic bundle directory. Pass
`--output-root <absent-parent>/validation-report`; do not pass independent JSON
or Markdown filenames and do not create the bundle in advance.

```bash
validation_id="validation-four-gpu-$(date -u +%Y%m%d-%H%M%S)-$$"
validation_parent="/root/workspace/ILVR/results/$validation_id"
test ! -e "$validation_parent"
./venv/ilvr-venv/bin/python -m src.multinode validate-results \
  --matrix-root "/root/workspace/ILVR/results/$matrix_id" \
  --evaluation-root "/root/workspace/ILVR/results/$eval_id" \
  --capacity-root "$capacity_root" \
  --output-root "$validation_parent/validation-report"
```

Success publishes exactly:

```text
/root/workspace/ILVR/results/<validation-id>/validation-report/validation-manifest.json
/root/workspace/ILVR/results/<validation-id>/validation-report/validation-summary.md
```

If the no-replace rename raises for any reason, publication is ambiguous. The
diagnostic names the exact output root and staging/orphan path. The program
does not inspect, delete, empty, or relocate either namespace after that
exception. Preserve both for operator inspection; never replace foreign data.

## Inspect and stop only exact-owned work

The coordinator normally stops exact-owned peers automatically on launch,
readiness, monitoring, or worker failure. For inspection, list sessions on
each node and decode ownership records through the bounded reader:

```bash
tmux list-sessions -F '#{session_name} #{session_windows} #{session_attached}'
ssh -o BatchMode=yes -o ConnectTimeout=8 -- korea@172.16.11.41 \
  "/usr/bin/docker exec llm_kd /usr/bin/tmux list-sessions -F '#{session_name} #{session_windows} #{session_attached}'"

run_id=RUN_ID
attempt_id=initial
./venv/ilvr-venv/bin/python -m src.multinode ownership-show \
  --ownership-path "/root/workspace/ILVR/logs/multinode/$run_id/$attempt_id/ownership-rank1.json"
ssh -o BatchMode=yes -o ConnectTimeout=8 -- korea@172.16.11.41 \
  "/usr/bin/docker exec --workdir /root/workspace/ILVR llm_kd /usr/bin/env PYTHONPATH=/root/workspace/ILVR /root/workspace/ILVR/venv/ilvr-venv/bin/python -m src.multinode ownership-show --ownership-path '/root/workspace/ILVR/logs/multinode/$run_id/$attempt_id/ownership-rank0.json'"
```

When exceptional operator cleanup is required, stop rank 0 on the remote node
and rank 1 locally. Each command enumerates every canonical attempt for that
run/rank, tolerates exact records whose process already exited, revalidates PID
start time, argv/environment digest, commit, run, attempt, and rank, then sends
SIGTERM through a pidfd only:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -- korea@172.16.11.41 \
  "/usr/bin/docker exec --workdir /root/workspace/ILVR llm_kd /usr/bin/env PYTHONPATH=/root/workspace/ILVR /root/workspace/ILVR/venv/ilvr-venv/bin/python -m src.multinode stop-owned --run-id '$run_id' --machine-rank 0"
./venv/ilvr-venv/bin/python -m src.multinode stop-owned \
  --run-id "$run_id" --machine-rank 1
```

Manual `kill`, `pkill`, process-pattern signaling, `tmux kill-session`, and any
session-name-only termination are forbidden. A session name is inspection
evidence, never sufficient ownership proof.
