# Stage1-v5 managed training and automatic TEST400 evaluation

The reviewed v5 path uses original Qwen2.5-VL-3B initialization and a frozen
EMA1 reference. The selected objective is
`0.1 CE + 0.1 alignment + 1 D + 0.1 U + 0.01 K`, with independent k50
supports, no GT union, mass0, all eligible targets, beta1/margin0, Text T2,
no KL warmup and exact chunk128 reductions. Standard Stage1 and Stage2 loss
selectors retain their existing behavior.

Read canonical `/root/workspace/ILVR/AGENTS.md`,
`/root/workspace/ILVR/documents/training-defaults.md` and
[the multi-node guide](multinode-training.md) before any training. This selected
approved Stage1-v5 run uses its explicit stage-specific profile and fresh
original base, rather than the Stage2 initialization rule.

## Prepared execution

Root must provision a distinct compatible remote checkout at the reviewed
committed revision. Keep this checkout's `data/`, `checkpoints/` and `venv/`
symlinks; never overwrite canonical or another experiment's checkout.
Synchronize the whole committed checkout, including `src/stage1/v5_*.py`,
composition/checkpointing/parser changes and the opt-in hooks in vendored
`transformers/src/transformers/trainer.py`. The coordinator does not synchronize
source code. Verify the actual imported files and package versions on both
nodes. A matching CLI alone does not establish compatible execution.

Derive `max_steps` from the actual prepared loader using the Task7 audit. One
exposure uses ceil(prepared per-rank microbatches / configured GAS4), including
the final partial window. Do not substitute historical 431/212 or a guessed
sample-count budget. Ordinary losses retain the configured global16 denominator;
D/U/mass average over the actual window's original blocks, including empty ones.
The runtime sidecar records and checks the actual schedule and execution profile.

The recipe is fresh-run only. `--settings FILE` optionally sources existing
settings; all selected profile values are enforced afterwards. The original
FULL safetensors base is explicitly
`/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct`,
which must be readable with the same original content on both nodes. No trained
Stage1 initializer or HF-name fallback is selected.

```bash
./scripts/stage1_v5_train_eval.sh \
  --run-id stage1-v5-<distinct-id> --main-process-port <free-port> \
  --remote-repo-root /root/workspace/ILVR/.worktrees/<distinct-reviewed-copy> \
  --remote-host-repo-root /home/korea/Workspace/hjh/workspace/ILVR/.worktrees/<distinct-reviewed-copy> \
  --max-steps <Task7-prepared-loader-budget> --dry-run
```

Dry-run prints three quoted commands and performs no launch or model loading.
Use the existing training coordinator's `--dry-run` on the first printed command
and the evaluation command's `--dry-run` to inspect exact node commands.
Training preflight requires clean matching commits, compatible environments,
readable inputs, fresh namespaces, free resources and the fixed four-rank
profile: remote rank0 `korea@172.16.11.41`/`llm_kd`, local rank1, two GPUs/node,
rendezvous `172.16.11.41`, Gloo/NCCL `bond0.11`.

The selected training profile is micro1/GAS4/global16, LR5e-5, seed42, warmup10,
BF16, existing AdamW/linear, decay0.01/clip1, DoRA r32/alpha64/dropout0.05 on
validated decoder modules, latent8, decoder SDPA, vision FlashAttention2,
non-reentrant checkpointing, maximum training sequence32768, workers0 and ZeRO1.
These are explicit recipe values or immutable Stage1 Trainer settings, rather
than a claim that all historical launcher defaults changed.

After all development/runtime gates and both-node preflight pass, launch the
same command without `--dry-run`, detached using the existing operator practice:

```bash
mkdir -p logs/<distinct-run-id>
nohup ./scripts/stage1_v5_train_eval.sh \
  --run-id <distinct-run-id> --main-process-port <free-port> \
  --remote-repo-root <reviewed-remote-container-root> \
  --remote-host-repo-root <matching-remote-host-root> \
  --max-steps <audited-budget> \
  > logs/<distinct-run-id>/train-eval.log 2>&1 < /dev/null &
```

This recipe synchronously waits for the existing training coordinator, prepares
only its exact `checkpoint-<audited-budget>` as an inference artifact, then runs
TEST400. A failed training, preparation, or evaluation command exits immediately.
There is no search for another run's checkpoint. The user-facing monitor may
stop after runtime evidence shows both nodes, all four ranks, finite loss and
real optimizer progress; the detached recipe continues through evaluation.

## Artifacts and resume

Training state stays in `checkpoints/<run-id>/`, including numbered checkpoints,
`stage1_v5.json`, immutable `stage1_v5_reference.pt`, EMA, PEFT and ordinary
Trainer/DeepSpeed state. Existing collection and redistribution retain v5's
contract and initial reference. After a fresh explicit v5 final collection,
the coordinator uses the existing validated delta redistribution to retain
both files physically in the local final checkpoint before publishing success.
Only missing approved resume entries transfer; the original full base does not.
Training logs are under `logs/multinode/` and
`logs/zero-stage-compatibility/`; producer evidence and witnesses are under
`results/<run-id>/`.

Inference preparation selects only the existing loader allowlist from the
validated final PEFT artifact. It excludes v5 reference, EMA, optimizer,
scheduler and RNG state. The original base remains a separate evaluation input.
Prepared models stay in
`checkpoints/evaluation-staging/<run-id>/prepared-model` on both managed roots;
evaluation's local worker copy stays in
`checkpoints/evaluation-staging/<run-id>-test400/local-model`.

TEST400 is the established `data/TEST.jsonl` with exactly 400 nonempty records.
Evaluation uses temperature0/top_p1, max new4096, max sequence128000,
repetition penalty1, no vision ablation and `external_full_latent_block`.
For unequal managed roots, the coordinator hashes the exact local TEST file
and binds its canonical path/size/mtime plus SHA256 into every Stage1 worker
command. Workers validate local mapped bytes before model loading and again
before sidecar publication; legacy calls without this optional pair retain
physical identity semantics. The existing sidecar schema and strict merger
identity equality remain unchanged. The SHA binding is recorded before the
first worker launch in `logs/multinode/<eval-id>/<attempt>/dataset-identity.json`.
Hashing costs one coordinator read and two reads per worker of the small TEST
manifest; images/model weights are not hashed by this route.

Four independent workers write shards and manifests to
`results/<run-id>-test400/`, followed by complete-coverage validation, merged
JSONL and summary JSON. Stage1 workers also publish distinct
`shard-N.runtime-identity.json` backend witnesses in that results directory;
remote shards0/1 and their runtime witnesses are collected through the fixed
rsync list. Evaluation logs and ownership records stay under logs.

For resume, call `run_multinode_stage1_training.sh` with the same run ID, a new
`--attempt-id`, both explicit remote roots, and `--resume` pointing to that run's
numbered checkpoint. Omitted objective and execution options inherit the saved
v5 contract; explicit conflicts reject. Do not use the fresh recipe to change
or reinterpret a resumed objective. A content-equal original base may move to a
different local path only through v5's mandatory saved base-content check in
composition/preflight/load, followed by immutable reference restore. Standard
Stage1 keeps strict source-string checks. Saved sidecars are not rewritten.

## Evidence boundaries

CPU tests cover real command construction, fail-fast recipe sequencing,
managed training/preparation/evaluation and ownership paths, allowlist retention,
producer path binding, and tiny actual PEFT composition/loading with equal
relocated bytes and changed-content rejection. They do not establish live
SSH/rsync behavior, FA2 GPU computation, four-rank ZeRO scaling/resume,
full-size capacity, training completion or a TEST400 score.

Task7 must audit the actual prepared loader and update budget, real four-rank
engine gradients/continuation, BF16 and maximum geometry memory. Independent
review, final fresh project verification, compatible remote provisioning and
runtime participation/progress remain required before claiming production
training. No full-model training or evaluation result is claimed by this recipe.
