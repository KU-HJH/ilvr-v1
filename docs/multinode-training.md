# Four-GPU multi-node operation

This is the operator contract for ILVR's fixed two-node, four-process
training and evaluation launchers. The canonical examples below run from
`/root/workspace/ILVR` with the repository virtual environment. For Stage 2
InfiniBand execution, use the explicitly validated runtime in the next section.
Keep each path's specified host, interface, container, rank order and Python.

## Verified Stage 2 InfiniBand execution

This records the approved, observed Stage 2 IB path as of 2026-10-22. Reuse
the existing runtime for Stage 2 IB work; this section takes precedence over
the canonical `llm_kd`/repository examples below for that path. It does not
establish IB support for Stage 1, evaluation, or other projects.

| Item | Verified Stage 2 IB value |
| --- | --- |
| Remote node / machine rank / global ranks | `korea@172.16.11.41` / 0 / 0,1 |
| Local node / machine rank / global ranks | `korea@172.16.11.40` / 1 / 2,3 |
| Container on both nodes | `llm_kd_rdma`; preserve original `llm_kd` |
| Runtime on both nodes | `/root/workspace/ILVR/.worktrees/rdma-stage2-20261022` |
| Validated revision | `a39fe86774c965022c058f2cec4cc8dd311bab7a`, or a reviewed compatible descendant |
| Coordinator | Existing `run_multinode_stage2_training.sh`, inside local `llm_kd_rdma` |
| Rendezvous / socket interface | `172.16.11.41` / `bond0.11` |
| GPU / HCA | Two H100 NVL per node; active `mlx5_0`, port 1 |

The validated runtime descends from corrected-loss baseline
`2f48c51985c395c16bea80a89847d7586728f9cd`. Its remote exec, stdin and rsync
boundaries consistently target `llm_kd_rdma`; its sanitized Stage 2 worker
environment explicitly carries the IB settings. An unmodified canonical
coordinator still targets `llm_kd`. Changing a shell variable or copying only
the launcher is insufficient. Verify a compatible descendant on both nodes
before using another checkout; matching headline hyperparameters alone does
not prove algorithm or runtime compatibility.

The existing RDMA containers were provisioned additively from
`llm/cu128:codex_3` with host networking, both GPUs,
`--device /dev/infiniband/uverbs0:/dev/infiniband/uverbs0:rw`,
`--ulimit memlock=-1:-1`, `--cap-add IPC_LOCK`, `--shm-size 1g`, and the
`ibverbs-utils` / `ibverbs-providers` userspace packages. Their `/root` bind
sources are `/home/korea/Workspace/hjh2` on local .40 and
`/home/korea/Workspace/hjh` on remote .41; `/raid` is also bound at `/raid`.
Preserve the existing data/venv/checkpoints symlinks, which depend on these
mounts. Reuse the containers after checking their actual state and device
access; creating a new container for every run is unnecessary. Keep original
session containers and host drivers/modules intact. Earlier verify-only
`rdma-verify-*` containers are not this training environment.

From inside local `llm_kd_rdma`, start with:

```bash
cd /root/workspace/ILVR/.worktrees/rdma-stage2-20261022
source ./venv/ilvr-venv/bin/activate
```

Activation must put this venv's `bin` directory on the coordinator's `PATH`
so preflight can find `torchrun`; invoking the venv Python alone does not
guarantee this. Source any experiment settings, then explicitly enforce the
whole applicable profile in [training-defaults.md](../documents/training-defaults.md).
Use original Qwen2.5-VL-3B plus reproduced Stage 1 checkpoint-431, the existing
four-rank/global-16 profile, CE0.1 and ordinary beta10/margin0.5 outside beta
(DIL beta1/margin0), and preserve stage/resume semantics. Do not use the
historical 54% Stage 2 weights as a fresh-run initializer.

Both actual worker environments must resolve:

```text
MASTER_ADDR=172.16.11.41
NCCL_NET=IB
NCCL_IB_DISABLE=0
NCCL_IB_HCA=mlx5_0:1
NCCL_SOCKET_IFNAME=bond0.11
GLOO_SOCKET_IFNAME=bond0.11
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET
```

The coordinator sanitizes the worker environment with `env -i`. Confirm the
above values in its resolved remote/local commands, not just the invoking
shell. When IB is requested, resolve an IB failure before launch rather than
silently falling back to socket transport. Run the existing dry-run and live
preflight, checking matching code, compatible venvs, readable model/data/images,
active HCA/device/verbs access, both GPUs per node, a free explicit port and
fresh run destinations. Do not start ranks manually. Launch the existing
coordinator detached and retain its normal per-run logs/results.

For the approved same-budget fresh Stage 2B smoke, preserve the full
`STAGE2_TOTAL_STEPS=212` / `--max_steps 212`, `--save_steps 212`, warmup 10,
ZeRO-2 and all other standing defaults, and add
`--stage2_stop_after_checkpoint 2`. The final validated revision admits
ordinary `dpo` context-preference Stage 2B through the existing full-save-then-stop
callback. Older revisions may admit only DIL. This stop option is fresh-run
only and must be below the full budget; omit it for full training or resume.
Do not set `max_steps=2`, which would change the schedule. Recalculate the
full budget if retained data or stage protocol changes.

Accept IB/training success only with runtime evidence: all four initialized
global ranks, inter-node `NET/IB` channels on both nodes, finite loss and
gradients, the expected saved optimizer step, complete checkpoint contents,
and successful node/coordinator exits. The verified run completed two updates
with `global_samples=32`, `world_size=4` and all four GPUs idle afterward.
Shutdown process-group warnings appeared despite exit 0; this was not a
warning-free run.

The existing coordinator assembles the complete checkpoint on **remote
machine rank 0**, in `./checkpoints/<run-id>/checkpoint-<step>` on .41.
For the verified run, this resolved to
`/raid/workspace/ILVR/checkpoints/rdma-stage2-smoke-20261022-062054/checkpoint-2`.
Local rank-2/3 optimizer and RNG shards alone are incomplete. Before resume,
verify remote trainer/model/adapter state, all four optimizer/RNG shards,
sampler/runtime sidecars and the separate `scheduler.pt`; the DeepSpeed model
state's `lr_scheduler` field was `None`. Preserve existing checkpoint symlinks
and store weights only under `checkpoints`, metadata under `results`, logs
under `logs`.

**GPUDirect RDMA was disabled in the observed NCCL logs.** IB worked with
host staging; an installed `nvidia-peermem` module file does not mean it is
loaded or GDR is usable. Do not claim GPU-memory RDMA or enable host modules
as part of ordinary reuse. The earlier CPU perftest measured 373.48 Gb/s;
it is not a GPU-memory or model-training throughput measurement. The
two-update smoke establishes transport/training/save operation, not full-run
convergence, accuracy or a reproduction of the historical 54% result.

Controller-side evidence is retained at:

- `/root/workspace/ILVR/.worktrees/rdma-stage2-20261022/results/rdma-stage2-smoke-20261022-062054/summary.json`
- `/root/workspace/docs/superpowers/records/2026-10-22-ilvr-rdma-stage2-runtime.json`
  (exact effective profile and container provisioning argv).
- `/root/workspace/docs/superpowers/records/2026-10-22-ilvr-rdma-stage2-intake.json`
  (approvals, independent review finding, native fix and verification).

## Canonical topology and one-time provisioning

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

## Stage 2 native DDP objective comparisons

The opt-in entropy/Vision auxiliaries use the existing coordinator with
explicit `--backend ddp`; they do not run through DeepSpeed/ZeRO. The fixed
topology remains remote rank 0 at `172.16.11.41`, local rank 1, two GPUs per
node, and four ranks. Native workers use Accelerate `--multi_gpu` and reject an
active DeepSpeed, FSDP, Megatron, CPU, or Dynamo mode inherited from the
environment. Do not start ranks manually or replace the coordinator.

Use `run_stage2_objective_examples.sh` for the controlled CE-only, entropy, and
combined comparison commands. It requires an original Stage 1 source, retained
data and image roots, a fresh run ID, a verified positive Stage 2 update budget,
and a free port. Entropy additionally requires a positive `--lambda-dyn`;
combined also requires a positive `--lambda-vis`. It explicitly fixes the
current four-rank/global-batch-8/learning-rate/seed/DoRA/SDPA/checkpointing
baseline and sets `save_steps` equal to the terminal budget. All three modes
enable the new objective identity, including the zero-weight CE-only control.

The wrapper defaults to `--dry-run`. This invokes the actual coordinator's
dry-run path and prints its complete remote/local environment and argv without
starting SSH, tmux, or ranks and without creating the run's checkpoint, result,
or log directories. Dry-run output is not a live preflight or participation
witness. Review the formulas, exact three-command template, branch costs,
backend limits, and evidence boundary in
[stage2-objective.md](stage2-objective.md).

Before an authorized `--launch`, complete the live preflight from this guide on
both hosts. First integrate the feature delivery so both canonical repositories
contain the same committed implementation; a worktree dry-run does not deploy
it. Confirm identical committed code and environments, readable
Stage1/data/image inputs at the same absolute paths, all four required GPUs and
memory, `bond0.11`, the selected free port, and fresh destinations. The default
fresh Stage 2 DoRA source is the reproduced
`checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431`,
not `comt_ckpt` or checkpoint-215. Derive the Stage 2 step budget from retained
samples and actual Trainer remainder behavior; never substitute Stage 1's 431.

Native checkpoint collection expects Trainer-format common model/adapter,
optimizer, scheduler, Trainer, sampler, runtime, and conditional objective state
from remote ranks 0/1 plus rank RNG files 2/3 from the local node. The existing
collector validates both node identities, moves only locally owned RNG files to
the canonical checkpoint, and re-observes all four ranks. Resume redistributes
the complete canonical checkpoint without overwrite and requires the same DDP
backend, method, run, objective/reference/vocabulary identity, optimizer/
scheduler, sampler, and RNG contract. Use a fresh attempt ID and port for a
resume. A native DDP checkpoint is not an entry in the historical ZeRO matrix,
and native DDP inference preparation remains unsupported.

Current CPU evidence does not establish target-model GPU memory fit, NCCL,
real-host save/transfer/resume, or successful training. The required first GPU
acceptance is a separately authorized complete combined-objective optimizer
update through this coordinator with actual four-rank participation and per-rank
peak-memory evidence, followed by the requested save/resume protocol. Do not
silently change the objective, batch, sequence limit, or topology after an OOM.

## Four-worker evaluation

Evaluation accepts only a Task 6 prepared inference artifact below the
canonical checkpoint evaluation-staging root, not a raw checkpoint directory.
`prepare-inference`
filters a completed canonical checkpoint to the inference allowlist, creates
the same fresh checkpoint staging path locally and remotely, transfers that
exact list plus its identity manifest without overwrite, and revalidates the
remote artifact.
It starts no evaluator worker.

```bash
source_run_id="stage2-full-zero1-$run_stamp"
prepared_model_dir="/root/workspace/ILVR/checkpoints/evaluation-staging/$source_run_id/prepared-checkpoint-10"
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
  --max-new-tokens 4096
```

The wrapper owns four independent workers: remote GPUs produce shards 0 and
1, local GPUs produce shards 2 and 3, and the coordinator pulls only the two
remote shard JSONLs and sidecars without overwriting local data. Success
requires all sidecars, complete source-index coverage, no worker errors, a
merged JSONL, and the summary JSON. The fresh local inference copy and its
sibling identity manifest are retained at
`/root/workspace/ILVR/checkpoints/evaluation-staging/${eval_id}/local-model`;
do not place either artifact under `results/`.

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
